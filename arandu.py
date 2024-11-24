import torch
import pyaudio
import numpy as np
import speech_recognition as sr
import time
import os
import simpleaudio as sa
from transformers import pipeline
from TTS.api import TTS
from ollama import Client

current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    chunk_length_s=30,
    device=device,
    generate_kwargs={"task": "transcribe", "language": "<|pt|>"},
)

# Load the TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to(device)

FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono audio
RATE = 16000  # Sampling rate
FRAME_DURATION_MS = 30

CHUNK = int(RATE * FRAME_DURATION_MS / 1000)

# Voice detection and recording
def voice_recorder():
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=RATE) as source:
        print("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=10)

    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
    return audio_data

# Get response from ollama
def ollama_model(text):
    try:
        client = Client()
        
        model = 'qwen2.5-coder:0.5b'
        max_words = 20

        messages = [
            {'role': 'system', 'content': f'Seu nome é Arandu. Você é um assistente que fornece respostas objetivas, claras e diretas, com no máximo {max_words} palavras.'},
            {'role': 'user', 'content': text}
        ]
        
        response = client.chat(model=model, messages=messages)
        
        return response['message']['content'][:120] # return max 120 characters
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return ""

# Voice syntization and reproduction
def voice_synthesizer(response):
    if response:
        file_path = "response.wav"
        speaker_wav = f"{current_folder}/speech.wav"
        language = "pt"
        
        tts.tts_to_file(text=response, speaker_wav=speaker_wav, language=language, file_path=file_path)
        print(f"Response synthesized to: {file_path}")

        # Play the synthesized audio
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()

def main():
    while True:
        audio = voice_recorder()

        if len(audio) == 0:
            continue

        # sample to get text
        sample = {"array": audio.astype(np.float32) / 32768.0, "sampling_rate": RATE}
        text = pipe(sample)["text"]
        print(text)

        response = ollama_model(text)
        
        print("Transcription:", text)
        if response:
            print("Response:", response)
            voice_synthesizer(response)
            
        time.sleep(0.5)

if __name__ == "__main__":
    main()
