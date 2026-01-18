import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from collections import deque
import time
import threading
import queue
import torch

class RealtimeMeetingTranscriber:
    def __init__(self, device_index=None, model_size="base", enable_translation=True):
        """
        实时会议转录器（支持中英文混合）
        
        参数:
        - device_index: 音频设备索引（None 则使用默认）
        - model_size: Whisper 模型大小 (tiny/base/small/medium/large-v3)
        - enable_translation: 是否启用英译中翻译
        """
        self.sample_rate = 16000
        self.device_index = device_index
        self.enable_translation = enable_translation
        
        # 初始化 Silero VAD
        print("正在加载 VAD 模型...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        (self.get_speech_timestamps, _, self.read_audio, *_) = utils
        
        # 初始化 Whisper 模型
        print(f"正在加载 Whisper 模型：{model_size}...")
        try:
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
            print("使用 GPU 加速")
        except:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("使用 CPU 模式")
        
        # 初始化翻译模型
        self.translator = None
        if enable_translation:
            self.init_translator()
        
        # 音频缓冲
        self.audio_buffer = deque()
        self.speech_buffer = []
        
        # 控制变量
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
        # VAD 参数
        self.vad_chunk_size = 512
        self.min_silence_duration = 0.5
        self.buffer_duration = 10.0
        self.max_buffer_samples = int(self.sample_rate * self.buffer_duration)
        
        self.last_speech_time = 0
        self.is_speaking = False
    
    def init_translator(self):
        """初始化 Helsinki-NLP 翻译模型"""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            print("正在加载翻译模型（首次使用需要下载，约 300MB）...")
            model_name = "Helsinki-NLP/opus-mt-en-zh"
            
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translation_model = MarianMTModel.from_pretrained(model_name)
            
            print("翻译模型加载完成")
        
        except Exception as e:
            print(f"⚠️  初始化翻译模型失败：{e}")
            self.enable_translation = False

    def translate_text(self, text):
        """使用 Helsinki-NLP 翻译"""
        if not self.enable_translation:
            return None
        
        try:
            # 分句翻译（长文本效果更好）
            sentences = text.split('. ')
            translated_sentences = []
            
            for sentence in sentences:
                if sentence.strip():
                    inputs = self.tokenizer(sentence, return_tensors="pt", padding=True)
                    translated = self.translation_model.generate(**inputs)
                    translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
                    translated_sentences.append(translated_text)
            
            return ''.join(translated_sentences)
            
        except Exception as e:
            print(f"   翻译出错：{e}")
            return None
        
    def list_devices(self):
        """列出所有可用的音频输入设备"""
        print("\n=== 可用的音频设备 ===")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"[{i}] {device['name']}")
                print(f"    输入通道：{device['max_input_channels']}")
                print(f"    默认采样率：{device['default_samplerate']:.0f} Hz")
                print()
        return devices
    
    def audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数"""
        if status:
            print(f"音频状态：{status}")
        
        audio_data = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
        self.audio_queue.put(audio_data.copy())
    
    def detect_speech(self, audio_chunk):
        """使用 Silero VAD 检测语音"""
        try:
            if len(audio_chunk) != self.vad_chunk_size:
                return False
            
            audio_tensor = torch.from_numpy(audio_chunk).float()
            audio_tensor = audio_tensor.unsqueeze(0)
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
            
            return speech_prob > 0.5
        except Exception as e:
            print(f"VAD 检测出错：{e}")
            return False
    
    def process_audio(self):
        """处理音频数据，使用 VAD 检测语音片段"""
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
                        
                        self.speech_buffer.append(chunk)
                        self.last_speech_time = current_time
                        
                    else:
                        if self.is_speaking:
                            self.speech_buffer.append(chunk)
                            
                            silence_duration = current_time - self.last_speech_time
                            if silence_duration >= self.min_silence_duration:
                                if len(self.speech_buffer) > 5:
                                    audio_segment = np.concatenate(self.speech_buffer)
                                    threading.Thread(
                                        target=self.transcribe_audio, 
                                        args=(audio_segment,), 
                                        daemon=True
                                    ).start()
                                
                                self.speech_buffer = []
                                self.is_speaking = False
                        
                        if len(self.speech_buffer) * self.vad_chunk_size > self.max_buffer_samples:
                            if self.speech_buffer:
                                audio_segment = np.concatenate(self.speech_buffer)
                                threading.Thread(
                                    target=self.transcribe_audio, 
                                    args=(audio_segment,), 
                                    daemon=True
                                ).start()
                                self.speech_buffer = []
                                self.is_speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理音频出错：{e}")
    
    def is_chinese(self, text):
        """检测文本是否主要是中文"""
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        total_chars = len([c for c in text if c.strip()])
        
        if total_chars == 0:
            return False
        
        return (chinese_chars / total_chars) > 0.3
    
    def transcribe_audio(self, audio_data):
        """使用 Whisper 转录音频，自动检测语言并翻译"""
        try:
            audio_float32 = audio_data.astype(np.float32)
            
            duration = len(audio_float32) / self.sample_rate
            if duration < 0.5:
                return
            
            print(f"📝 正在识别语音（时长：{duration:.1f}秒）...")
            
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
            timestamp = time.strftime("%H:%M:%S")
            
            # 判断是否需要翻译
            if detected_language == "zh" or self.is_chinese(original_text):
                # 中文直接输出
                result = f"[{timestamp}] 🇨🇳 {original_text}"
                print(f"\n✅ {result}\n")
                self.text_queue.put(result)
            else:
                # 英文需要翻译
                lang_flag = "🇬🇧" if detected_language == "en" else "🌐"
                
                if self.enable_translation:
                    print(f"   检测到语言：{detected_language} (置信度：{language_probability:.2f})，正在翻译...")
                    translated_text = self.translate_text(original_text)
                    
                    if translated_text:
                        result = f"[{timestamp}] {lang_flag} {original_text}\n           ➜ {translated_text}"
                        print(f"\n✅ {result}\n")
                        self.text_queue.put(result)
                    else:
                        result = f"[{timestamp}] {lang_flag} {original_text}\n           ➜ (翻译失败)"
                        print(f"\n✅ {result}\n")
                        self.text_queue.put(result)
                else:
                    # 不翻译，仅显示原文
                    result = f"[{timestamp}] {lang_flag} {original_text}"
                    print(f"\n✅ {result}\n")
                    self.text_queue.put(result)
        
        except Exception as e:
            print(f"转录出错：{e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """启动实时转录"""
        self.is_running = True
        
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
        print("\n✅ 开始实时转录会议音频...")
        print("📝 中文将直接显示，英文将自动翻译成中文")
        if self.enable_translation:
            print("🌐 翻译引擎：Argos Translate (本地)")
        else:
            print("⚠️  翻译已禁用，仅显示英文原文")
        print("按 Ctrl+C 停止\n")
    
    def stop(self):
        """停止转录"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2)
        print("\n已停止转录")
    
    def get_transcriptions(self):
        """获取所有转录结果"""
        results = []
        while not self.text_queue.empty():
            results.append(self.text_queue.get())
        return results


def main():
    print("=" * 60)
    print("实时会议转录系统 - 中英文混合支持（完全离线）")
    print("=" * 60)
    
    # 询问是否启用翻译
    enable_trans = input("\n是否启用英译中翻译？(y/n, 默认 y): ").strip().lower()
    enable_translation = enable_trans != 'n'
    
    # 创建转录器
    transcriber = RealtimeMeetingTranscriber(
        model_size="base",
        enable_translation=enable_translation
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
        
        print("\n" + "=" * 60)
        print("会议记录")
        print("=" * 60)
        results = transcriber.get_transcriptions()
        for result in results:
            print(result)


if __name__ == "__main__":
    main()
