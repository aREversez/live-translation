import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from collections import deque
import time
import threading
import queue
import torch
from datetime import datetime, timedelta
import os

class RealtimeMeetingTranscriber:
    def __init__(self, device_index=None, model_size="medium", local_model_path=None):
        """
        实时会议转录器（支持中英文混合）
        
        参数:
        - device_index: 音频设备索引
        - model_size: Whisper模型大小 (base/small/medium/large-v3)
        - local_model_path: 本地NLLB模型路径（如果已下载）
        """
        self.sample_rate = 16000
        self.device_index = device_index
        self.model_size = model_size
        self.local_model_path = local_model_path
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.txt_file = f"meeting_{timestamp}.txt"
        self.srt_file = f"meeting_{timestamp}.srt"
        
        # SRT字幕计数器
        self.srt_counter = 1
        self.session_start_time = None
        
        # 初始化Silero VAD
        print("正在加载VAD模型...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        (self.get_speech_timestamps, _, self.read_audio, *_) = utils
        print("✅ VAD模型加载完成")
        
        # 初始化Whisper模型
        print(f"正在加载Whisper模型: {model_size}...")
        try:
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
            print("✅ Whisper使用GPU加速")
        except:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("✅ Whisper使用CPU模式")
        
        # 初始化翻译模型
        self.init_translator()
        
        # 音频缓冲
        self.audio_buffer = deque()
        self.speech_buffer = []
        
        # 控制变量
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
        # VAD参数
        self.vad_chunk_size = 512
        self.min_silence_duration = 0.5
        self.buffer_duration = 10.0
        self.max_buffer_samples = int(self.sample_rate * self.buffer_duration)
        
        self.last_speech_time = 0
        self.is_speaking = False
        
        # 记录累计时间（用于SRT）
        self.cumulative_audio_duration = 0.0
    
    def init_translator(self):
        """初始化NLLB翻译模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            print("正在加载NLLB-200翻译模型...")
            
            # 使用本地模型或从HuggingFace下载
            if self.local_model_path and os.path.exists(self.local_model_path):
                print(f"从本地加载: {self.local_model_path}")
                model_name = self.local_model_path
            else:
                print("从HuggingFace下载（首次使用需要下载约600MB）...")
                model_name = "facebook/nllb-200-distilled-600M"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.target_lang = "zho_Hans"  # 简体中文
            
            # 如果有GPU，将模型移到GPU
            if torch.cuda.is_available():
                self.translation_model = self.translation_model.cuda()
                print("✅ 翻译模型使用GPU加速")
            else:
                print("✅ 翻译模型使用CPU")
            
            print("✅ NLLB翻译模型加载完成")
            
        except Exception as e:
            print(f"❌ 初始化翻译模型失败: {e}")
            print("\n可能的原因:")
            print("1. 首次使用需要联网下载模型（约600MB）")
            print("2. 磁盘空间不足")
            print("3. 缺少依赖: pip install transformers sentencepiece sacremoses")
            import traceback
            traceback.print_exc()
            raise
    
    def translate_text(self, text):
        """使用NLLB翻译文本（英文到中文）"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, max_length=512, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            translated_tokens = self.translation_model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            return translated_text
            
        except Exception as e:
            print(f"   翻译出错: {e}")
            return None
    
    def format_srt_time(self, seconds):
        """格式化SRT时间戳"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def write_to_files(self, original_text, translated_text, detected_language, start_time, end_time):
        """写入TXT和SRT文件"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 写入TXT文件
        with open(self.txt_file, 'a', encoding='utf-8') as f:
            if detected_language == "zh":
                f.write(f"[{timestamp}] 🇨🇳 {original_text}\n")
            else:
                lang_flag = "🇬🇧" if detected_language == "en" else "🌐"
                f.write(f"[{timestamp}] {lang_flag} {original_text}\n")
                if translated_text:
                    f.write(f"           ➜ {translated_text}\n")
            f.write("\n")
        
        # 写入SRT文件
        with open(self.srt_file, 'a', encoding='utf-8') as f:
            f.write(f"{self.srt_counter}\n")
            f.write(f"{self.format_srt_time(start_time)} --> {self.format_srt_time(end_time)}\n")
            
            if detected_language == "zh":
                f.write(f"{original_text}\n")
            else:
                f.write(f"{original_text}\n")
                if translated_text:
                    f.write(f"{translated_text}\n")
            
            f.write("\n")
            self.srt_counter += 1
        
    def list_devices(self):
        """列出所有可用的音频输入设备"""
        print("\n=== 可用的音频设备 ===")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"[{i}] {device['name']}")
                print(f"    输入通道: {device['max_input_channels']}")
                print(f"    默认采样率: {device['default_samplerate']:.0f} Hz")
                print()
        return devices
    
    def audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数"""
        if status:
            print(f"音频状态: {status}")
        
        audio_data = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
        self.audio_queue.put(audio_data.copy())
    
    def detect_speech(self, audio_chunk):
        """使用Silero VAD检测语音"""
        try:
            if len(audio_chunk) != self.vad_chunk_size:
                return False
            
            audio_tensor = torch.from_numpy(audio_chunk).float()
            audio_tensor = audio_tensor.unsqueeze(0)
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
            
            return speech_prob > 0.5
        except Exception as e:
            print(f"VAD检测出错: {e}")
            return False
    
    def process_audio(self):
        """处理音频数据，使用VAD检测语音片段"""
        while self.is_running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                self.audio_buffer.extend(audio_chunk)
                
                while len(self.audio_buffer) >= self.vad_chunk_size:
                    chunk = np.array([self.audio_buffer.popleft() for _ in range(self.vad_chunk_size)])
                    
                    is_speech = self.detect_speech(chunk)
                    current_time = time.time()
                    
                    if is_speech:
                        if not self.is_speaking:
                            print("🎤 检测到语音...")
                            self.is_speaking = True
                            self.speech_start_time = self.cumulative_audio_duration
                        
                        self.speech_buffer.append(chunk)
                        self.last_speech_time = current_time
                        
                    else:
                        if self.is_speaking:
                            self.speech_buffer.append(chunk)
                            
                            silence_duration = current_time - self.last_speech_time
                            if silence_duration >= self.min_silence_duration:
                                if len(self.speech_buffer) > 5:
                                    audio_segment = np.concatenate(self.speech_buffer)
                                    segment_duration = len(audio_segment) / self.sample_rate
                                    
                                    threading.Thread(
                                        target=self.transcribe_audio, 
                                        args=(audio_segment, self.speech_start_time, 
                                              self.speech_start_time + segment_duration), 
                                        daemon=True
                                    ).start()
                                
                                self.speech_buffer = []
                                self.is_speaking = False
                        
                        if len(self.speech_buffer) * self.vad_chunk_size > self.max_buffer_samples:
                            if self.speech_buffer:
                                audio_segment = np.concatenate(self.speech_buffer)
                                segment_duration = len(audio_segment) / self.sample_rate
                                
                                threading.Thread(
                                    target=self.transcribe_audio, 
                                    args=(audio_segment, self.speech_start_time,
                                          self.speech_start_time + segment_duration), 
                                    daemon=True
                                ).start()
                                self.speech_buffer = []
                                self.is_speaking = False
                    
                    # 更新累计时间
                    self.cumulative_audio_duration += len(chunk) / self.sample_rate
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理音频出错: {e}")
    
    def is_chinese(self, text):
        """检测文本是否主要是中文"""
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        total_chars = len([c for c in text if c.strip()])
        
        if total_chars == 0:
            return False
        
        return (chinese_chars / total_chars) > 0.3
    
    def transcribe_audio(self, audio_data, start_time, end_time):
        """使用Whisper转录音频，自动检测语言并翻译"""
        try:
            audio_float32 = audio_data.astype(np.float32)
            
            duration = len(audio_float32) / self.sample_rate
            if duration < 0.5:
                return
            
            print(f"📝 正在识别语音（时长: {duration:.1f}秒）...")
            
            # 转录原文
            segments, info = self.model.transcribe(
                audio_float32,
                language=None,
                task="transcribe",
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False
            )
            
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
            
            if not text_parts:
                print("   (未检测到有效语音)")
                return
            
            original_text = "".join(text_parts).strip()
            if not original_text:
                return
            
            detected_language = info.language
            language_probability = info.language_probability
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # 判断是否需要翻译
            if detected_language == "zh" or self.is_chinese(original_text):
                # 中文直接输出
                result = f"[{timestamp}] 🇨🇳 {original_text}"
                print(f"\n✅ {result}\n")
                
                # 写入文件
                self.write_to_files(original_text, None, "zh", start_time, end_time)
                
            else:
                # 英文需要翻译
                lang_flag = "🇬🇧" if detected_language == "en" else "🌐"
                print(f"   检测到语言: {detected_language} (置信度: {language_probability:.2f})，正在翻译...")
                
                translated_text = self.translate_text(original_text)
                
                if translated_text:
                    result = f"[{timestamp}] {lang_flag} {original_text}\n           ➜ {translated_text}"
                    print(f"\n✅ {result}\n")
                    
                    # 写入文件
                    self.write_to_files(original_text, translated_text, detected_language, start_time, end_time)
                else:
                    result = f"[{timestamp}] {lang_flag} {original_text}\n           ➜ (翻译失败)"
                    print(f"\n✅ {result}\n")
                    
                    # 写入文件（无翻译）
                    self.write_to_files(original_text, None, detected_language, start_time, end_time)
        
        except Exception as e:
            print(f"转录出错: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """启动实时转录"""
        self.is_running = True
        self.session_start_time = time.time()
        self.cumulative_audio_duration = 0.0
        
        # 创建输出文件
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write(f"会议记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
        
        with open(self.srt_file, 'w', encoding='utf-8') as f:
            pass  # 创建空文件
        
        self.stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.sample_rate,
            callback=self.audio_callback,
            blocksize=self.vad_chunk_size
        )
        
        self.process_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.process_thread.start()
        
        self.stream.start()
        print("\n" + "=" * 60)
        print("✅ 开始实时转录会议音频")
        print("=" * 60)
        print(f"📝 文本记录: {self.txt_file}")
        print(f"📺 字幕文件: {self.srt_file}")
        print(f"🤖 Whisper模型: {self.model_size}")
        print(f"🌐 翻译模型: NLLB-200")
        print("=" * 60)
        print("按 Ctrl+C 停止\n")
    
    def stop(self):
        """停止转录"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2)
        
        print("\n" + "=" * 60)
        print("✅ 转录已停止")
        print(f"📝 文本记录已保存: {self.txt_file}")
        print(f"📺 字幕文件已保存: {self.srt_file}")
        print("=" * 60)


def main():
    print("=" * 60)
    print("实时会议转录系统 - 完全离线版")
    print("功能: 中英文识别 + 自动翻译 + TXT/SRT输出")
    print("=" * 60)
    
    # 检查是否有本地模型
    local_model = None
    if os.path.exists("./nllb-model"):
        print("\n✅ 检测到本地NLLB模型")
        local_model = "./nllb-model"
    
    # 创建转录器
    transcriber = RealtimeMeetingTranscriber(
        model_size="medium",
        local_model_path=local_model
    )
    
    # 列出设备
    transcriber.list_devices()
    
    # 选择设备
    try:
        device_input = input("\n请输入要使用的设备编号（直接回车使用默认设备）: ").strip()
        device_id = int(device_input) if device_input else None
        transcriber.device_index = device_id
    except ValueError:
        print("使用默认设备")
        transcriber.device_index = None
    
    # 开始转录
    transcriber.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n正在停止...")
    finally:
        transcriber.stop()


if __name__ == "__main__":
    main()
