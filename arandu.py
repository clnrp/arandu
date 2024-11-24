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
from langchain.agents import initialize_agent, load_tools
from langchain.agents import AgentType
from langchain_community.llms import Ollama
from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper

current_folder = os.path.dirname(os.path.abspath(__file__))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Voice detection and recording
class VoiceRecorder:
    def __init__(self, rate=16000, channels=1, frame_duration_ms=30):
        self.rate = rate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.chunk = int(rate * frame_duration_ms / 1000)
        self.recognizer = sr.Recognizer()
        #self.recognizer.energy_threshold = 400
        self.recognizer.dynamic_energy_threshold = True

    def record_voice(self):
        with sr.Microphone(sample_rate=self.rate) as source:
            print("Listening...")
            audio = self.recognizer.listen(source, phrase_time_limit=10)
        audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
        return audio_data

# Automatic speech recognition
class WhisperASR:
    def __init__(self, model_name="openai/whisper-tiny", chunk_length_s=30, device=device):
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=chunk_length_s,
            device=device,
            generate_kwargs={"task": "transcribe", "language": "<|pt|>"},
        )

    def transcribe(self, audio, rate):
        sample = {"array": audio.astype(np.float32) / 32768.0, "sampling_rate": rate}
        return self.pipe(sample)["text"]

# Get response from LLM using Langchain Agent
class LLMModel:
    def __init__(self, model_name='qwen2.5-coder:0.5b', max_words=20):
        self.model_name = model_name
        self.max_words = max_words
        #self.client = Client()
        self.llm = Ollama(model=self.model_name)
        self.search_tool = WikipediaAPIWrapper()
        self.tools = load_tools(["llm-math"], llm=self.llm)
        self.tools.append(
            Tool(
                name="Search",
                func=self.search_tool.run,
                description="Use esta ferramenta para pesquisar informações na Wikipedia, e de uma resposta curta."
            )
        )
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations = 2
        )

    def get_response(self, text):
        try:
            query = f"Seu nome é Arandu. Você é um assistente que fornece respostas objetivas, claras e diretas, com no máximo {self.max_words} palavras. {text}"
            response = self.agent.run(query)
            return response[:120]  # return max 120 characters
        
            #messages = [
            #        {'role': 'system', 'content': f'Seu nome é Arandu. Você é um assistente que fornece respostas objetivas, claras e diretas, com no máximo {self.max_words} palavras.'},
            #        {'role': 'user', 'content': text}
            #    ]
            #response = self.client.chat(model=self.model_name, messages=messages)
            #return response['message']['content'][:120]  # return max 120 characters
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return ""

# Voice syntization and reproduction
class VoiceSynthesizer:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        self.tts = TTS(model_name)
        self.tts.to(device)
        self.current_folder = current_folder

    def synthesize_voice(self, text, file_path="response.wav"):
        if text:
            speaker_wav = f"{self.current_folder}/speech.wav"
            language = "pt"
            self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)
            print(f"Response synthesized to: {file_path}")

            # Play the synthesized audio
            wave_obj = sa.WaveObject.from_wave_file(file_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()

def main():
    recorder = VoiceRecorder()
    whisper_asr = WhisperASR()
    llm_model = LLMModel()
    synthesizer = VoiceSynthesizer()

    while True:
        audio = recorder.record_voice()

        if len(audio) == 0:
            continue

        # Sample to get text
        text = whisper_asr.transcribe(audio, recorder.rate)
        print(text)

        response = llm_model.get_response(text)

        print("Transcription:", text)
        if response:
            print("Response:", response)
            synthesizer.synthesize_voice(response)

        time.sleep(0.5)

if __name__ == "__main__":
    main()
