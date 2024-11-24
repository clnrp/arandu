from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.api import TTS
import os

current_folder = os.path.dirname(os.path.abspath(__file__))

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to('cuda')

speaker_wav = f"{current_folder}/speech.wav"

text = """
Nos primeiros momentos, o universo era uma sopa quente e densa de partículas. 
Em questão de segundos, forças fundamentais como a gravidade e o eletromagnetismo começaram a moldar o jovem universo. 
Prótons, nêutrons e elétrons surgiram, e, após algumas centenas de milhares de anos, o universo esfriou o suficiente para 
que essas partículas se combinassem em átomos simples, como hidrogênio e hélio, dissipando a névoa cósmica inicial.
"""

language = "pt"

tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path="output.wav")
