import os
import threading
import queue
import tempfile

import sounddevice as sd
import soundfile as sf
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pyttsx3
import gradio as gr
import keyboard

# ------------------ Configuration ------------------
# Path to a Hugging Face compatible TinyLlama model directory (full-precision)
MODEL_PATH = "C:/Users/Lenovo/tinyllama"
WHISPER_MODEL = "tiny"               # Whisper model size: tiny, base, small, etc.
CHUNK_DURATION = 1.0  # seconds per record chunk
RATE = 16000
HOTKEY = "space"    # Press SPACE to interrupt playback

# ------------------ Global Queues & Events ------------------
transcript_queue = queue.Queue()
response_queue = queue.Queue()
interrupt_event = threading.Event()

# ------------------ Load Models Once ------------------
# Whisper for STT
whisper_model = whisper.load_model(WHISPER_MODEL)

# Hugging Face for LLM (CPU only, full-precision)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True
)
# Create a CPU-based text-generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # run on CPU
    max_new_tokens=128,
    temperature=0.7
)

# pyttsx3 for TTS
tts_engine = pyttsx3.init()

# ------------------ Speech-to-Text Thread (Whisper) ------------------
class STTThread(threading.Thread):
    def __init__(self, q, interrupt):
        super().__init__(daemon=True)
        self.q = q
        self.interrupt = interrupt

    def run(self):
        while True:
            # Record audio chunk
            audio = sd.rec(int(CHUNK_DURATION * RATE), samplerate=RATE, channels=1)
            sd.wait()
            # Create temp file and close immediately to allow writing on Windows
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tmp_path = tmp.name
            tmp.close()
            # Write audio data
            sf.write(tmp_path, audio, RATE)
            # Transcribe
            result = whisper_model.transcribe(tmp_path)
            text = result.get("text", "").strip()
            # Remove temp file
            os.remove(tmp_path)
            if text:
                self.q.put(text)

# ------------------ LLM Inference Thread ------------------
class LLMThread(threading.Thread):
    def __init__(self, in_q, out_q):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.out_q = out_q

    def run(self):
        while True:
            user_text = self.in_q.get()
            prompt = f"User: {user_text}\nAssistant:"
            outputs = llm_pipeline(prompt)
            response = outputs[0]["generated_text"]
            # Remove prompt echo if present
            answer = response.replace(prompt, "").strip()
            self.out_q.put(answer)

# ------------------ Text-to-Speech Thread ------------------
class TTSThread(threading.Thread):
    def __init__(self, q, interrupt):
        super().__init__(daemon=True)
        self.q = q
        self.interrupt = interrupt
        self.engine = tts_engine

    def run(self):
        while True:
            text = self.q.get()
            if self.interrupt.is_set():
                self.engine.stop()
                self.interrupt.clear()
            self.engine.say(text)
            self.engine.runAndWait()

# ------------------ Hotkey Listener ------------------

def hotkey_listener(interrupt):
    keyboard.add_hotkey(HOTKEY, lambda: interrupt.set())
    keyboard.wait()

# ------------------ Gradio UI ------------------
def gradio_chat(audio):
    sr, wav = audio
    # Save incoming audio
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, wav, sr)
    # Transcribe
    result = whisper_model.transcribe(tmp_path)
    user_text = result.get("text", "").strip()
    os.remove(tmp_path)

    # Send for LLM inference
    transcript_queue.put(user_text)
    response = response_queue.get()

    # TTS output file
    tts_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tts_path = tts_tmp.name
    tts_tmp.close()
    tts_engine.save_to_file(response, tts_path)
    tts_engine.runAndWait()
    data, fs = sf.read(tts_path)
    os.remove(tts_path)

    return response, (fs, data)

iface = gr.Interface(
    fn=gradio_chat,
    inputs=gr.Audio(source='microphone', type='numpy', label="Speak"),
    outputs=[gr.Textbox(label="Assistant"), gr.Audio(type='numpy', label="Response")],
    live=False,
    title="TinyLlama Voice Assistant (Whisper + CPU HF)"
)

# ------------------ Main ------------------
if __name__ == "__main__":
    stt = STTThread(transcript_queue, interrupt_event)
    llm = LLMThread(transcript_queue, response_queue)
    tts = TTSThread(response_queue, interrupt_event)
    stt.start()
    llm.start()
    tts.start()
    threading.Thread(target=hotkey_listener, args=(interrupt_event,), daemon=True).start()
    iface.launch(share=False)
