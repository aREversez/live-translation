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
import warnings

# 忽略无关警告
warnings.filterwarnings("ignore")

class RealtimeMeetingTranscriber:
    def __init__(self, device_index=None, model_size="small", local_model_path=None):
        self.sample_rate = 16000
        self.device_index = device_index
        self.model_size = model_size
        self.local_model_path = local_model_path
        
        # 文件命名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.txt_file = f"meeting_{timestamp}.txt"
        self.srt_file = f"meeting_{timestamp}.srt"
        self.srt_counter = 1
        
        # 1. 初始化 VAD (使用 CPU 即可，GPU 加速收益极小)
        print("正在加载 VAD 模型...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        (self.get_speech_timestamps, _, self.read_audio, *_) = utils
        print("✅ VAD 模型加载完成")
        
        # 2. 初始化 Whisper (Faster-Whisper 原生支持 GPU FP16)
        print(f"正在加载 Whisper 模型 ({model_size})...")
        try:
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
            print("✅ Whisper: GPU (FP16) 就绪")
        except Exception as e:
            print(f"❌ Whisper GPU 加载失败：{e}，将使用 CPU")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        # 3. 初始化 NLLB 翻译模型
        self.init_translator()
        
        # 缓冲与队列
        self.audio_buffer = deque()
        self.speech_buffer = []
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.output_queue = queue.PriorityQueue()
        self.transcribe_counter = 0
        self.transcribe_lock = threading.Lock()
        
        # VAD 参数 (降低静音阈值以减少断句等待时间)
        self.vad_chunk_size = 512
        self.min_silence_duration = 0.5  # 从 0.6 改为 0.5 略微降低延迟
        self.buffer_duration = 15.0      # 允许更长的单句缓冲
        self.max_buffer_samples = int(self.sample_rate * self.buffer_duration)
        self.last_speech_time = 0
        self.is_speaking = False
        self.cumulative_audio_duration = 0.0

    def init_translator(self):
        """强制 GPU 加载 NLLB"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            print("正在加载 NLLB-200 翻译模型...")
            
            # 显式指定 CUDA
            self.device = torch.device("cuda")
            
            model_name = self.local_model_path if (self.local_model_path and os.path.exists(self.local_model_path)) else "facebook/nllb-200-distilled-600M"
            
            # 加载 Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
            
            # 【关键修改】直接加载到 cuda，不使用 device_map 防止误判
            # 12GB 显存足够直接加载 fp16 模型
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to(self.device)
            
            self.translation_model.eval()
            
            print(f"✅ NLLB: GPU (FP16) 就绪 | 设备：{self.translation_model.device}")
            
            # 预热一次模型 (消除第一次推理的延迟)
            print("🔄 正在预热翻译引擎...")
            self.translate_text("Hello world.")
            print("⚡ 系统准备就绪")

        except Exception as e:
            print(f"❌ NLLB 加载失败：{e}")
            raise

    def translate_text(self, text):
        """极速翻译模式"""
        try:
            if not text or not text.strip():
                return ""

            # 1. 设置源语言
            self.tokenizer.src_lang = "eng_Latn"
            
            # 2. 编码 (增加 max_length 防止输入过长被截断)
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids("zho_Hans")
            
            # 3. 生成 (关键优化参数)
            with torch.no_grad():
                translated_tokens = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=512,  # 【修复截断】使用 new_tokens 保证输出完整
                    num_beams=1,         # 【降低延迟】使用 Greedy Search (速度快 5 倍)
                    do_sample=False,     # 确定性输出，减少计算量
                    repetition_penalty=1.1 # 防止偶尔的复读机现象
                )
            
            # 4. 解码
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            return self.convert_punctuation_to_chinese(translated_text)
            
        except Exception as e:
            print(f"⚠️ 翻译异常：{e}")
            return None

    def convert_punctuation_to_chinese(self, text):
        if not text: return text
        mapping = {',': '，', '.': '。', '!': '！', '?': '？', ':': '：', ';': '；'}
        return "".join([mapping.get(c, c) for c in text])

    def format_srt_time(self, seconds):
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        millis = int((seconds % 1) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{millis:03d}"

    def write_to_files(self, original_text, translated_text, detected_language, start_time, end_time):
        timestamp = datetime.now().strftime("%H:%M:%S")
        # 写入 TXT
        with open(self.txt_file, 'a', encoding='utf-8') as f:
            flag = "🇨🇳" if detected_language == "zh" else "🇬🇧"
            f.write(f"[{timestamp}] {flag} {original_text}\n")
            if translated_text:
                f.write(f"           ➜ {translated_text}\n")
            f.write("\n")
        
        # 写入 SRT
        with open(self.srt_file, 'a', encoding='utf-8') as f:
            f.write(f"{self.srt_counter}\n")
            f.write(f"{self.format_srt_time(start_time)} --> {self.format_srt_time(end_time)}\n")
            f.write(f"{original_text}\n")
            if translated_text:
                f.write(f"{translated_text}\n")
            f.write("\n")
            self.srt_counter += 1

    def list_devices(self):
        print("\n=== 可用的音频设备 ===")
        print(sd.query_devices())

    def audio_callback(self, indata, frames, time_info, status):
        if status: print(status)
        self.audio_queue.put(indata[:, 0].copy())

    def process_audio(self):
        """音频处理主循环"""
        while self.is_running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                self.audio_buffer.extend(audio_chunk)
                
                # VAD 处理
                while len(self.audio_buffer) >= self.vad_chunk_size:
                    chunk = np.array([self.audio_buffer.popleft() for _ in range(self.vad_chunk_size)])
                    
                    # 转换张量
                    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
                    speech_prob = self.vad_model(chunk_tensor, self.sample_rate).item()
                    is_speech = speech_prob > 0.5
                    
                    current_time = time.time()
                    
                    if is_speech:
                        if not self.is_speaking:
                            self.is_speaking = True
                            self.speech_start_time = self.cumulative_audio_duration
                        self.speech_buffer.append(chunk)
                        self.last_speech_time = current_time
                    else:
                        if self.is_speaking:
                            # 静音超过阈值，触发识别
                            if (current_time - self.last_speech_time) >= self.min_silence_duration:
                                self.trigger_transcription()
                    
                    # 缓冲区过大强制截断
                    if len(self.speech_buffer) * self.vad_chunk_size > self.max_buffer_samples:
                        self.trigger_transcription()
                        
                    self.cumulative_audio_duration += len(chunk) / self.sample_rate
                    
            except queue.Empty:
                continue

    def trigger_transcription(self):
        if len(self.speech_buffer) > 5: # 忽略太短的杂音
            audio_segment = np.concatenate(self.speech_buffer)
            duration = len(audio_segment) / self.sample_rate
            
            # 启动线程进行识别，不阻塞录音
            threading.Thread(
                target=self.transcribe_task, 
                args=(audio_segment, self.speech_start_time, self.speech_start_time + duration), 
                daemon=True
            ).start()
            
        self.speech_buffer = []
        self.is_speaking = False

    def transcribe_task(self, audio_data, start_time, end_time):
        """识别 + 翻译任务"""
        try:
            audio_float32 = audio_data.astype(np.float32)
            
            # Whisper 识别
            segments, info = self.model.transcribe(
                audio_float32, 
                beam_size=1,        # 实时场景 beam_size=1 更快
                best_of=1,          # 减少候选采样
                vad_filter=True
            )
            
            original_text = "".join([s.text for s in segments]).strip()
            if not original_text: return

            # 获取 ID 锁
            with self.transcribe_lock:
                transcribe_id = self.transcribe_counter
                self.transcribe_counter += 1

            translated_text = None
            if info.language != "zh":
                # 调用翻译
                translated_text = self.translate_text(original_text)
            
            self.output_queue.put((transcribe_id, original_text, translated_text, info.language, start_time, end_time))
            
        except Exception as e:
            print(f"❌ 任务出错：{e}")

    def process_output(self):
        """输出线程：确保按顺序打印"""
        next_id = 0
        pending_results = {} # 暂存乱序到达的结果

        while not self.output_thread_stop:
            try:
                # 尝试获取结果
                item = self.output_queue.get(timeout=0.1)
                tid, orig, trans, lang, start, end = item
                
                pending_results[tid] = (orig, trans, lang, start, end)
                
                # 按顺序处理
                while next_id in pending_results:
                    orig, trans, lang, start, end = pending_results.pop(next_id)
                    
                    print(f"\n💬 [{datetime.now().strftime('%H:%M:%S')}] {orig}")
                    if trans:
                        print(f"   ➜ {trans}")
                    
                    self.write_to_files(orig, trans, lang, start, end)
                    next_id += 1
                    
            except queue.Empty:
                continue

    def start(self):
        self.is_running = True
        self.cumulative_audio_duration = 0.0
        
        # 清空文件
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write(f"会议记录 {datetime.now()}\n\n")
        open(self.srt_file, 'w').close()
        
        self.stream = sd.InputStream(
            device=self.device_index, channels=1, samplerate=self.sample_rate, 
            callback=self.audio_callback, blocksize=self.vad_chunk_size
        )
        self.stream.start()
        
        threading.Thread(target=self.process_audio, daemon=True).start()
        
        self.output_thread_stop = False
        self.output_thread = threading.Thread(target=self.process_output, daemon=True)
        self.output_thread.start()
        
        print(f"\n🚀 系统已启动！(设备 ID: {self.device_index if self.device_index else 'Default'})")
        print("请说话...")

    def stop(self):
        self.is_running = False
        self.output_thread_stop = True
        if hasattr(self, 'stream'): self.stream.stop()
        print("\n🛑 已停止")

def main():
    app = RealtimeMeetingTranscriber()
    app.list_devices()
    try:
        idx = input("\n输入设备 ID (回车默认): ").strip()
        if idx: app.device_index = int(idx)
        app.start()
        while True: time.sleep(1)
    except KeyboardInterrupt:
        app.stop()

if __name__ == "__main__":
    main()
