from whisper_mic import WhisperMic

mic = WhisperMic(model="tiny")

while True:
    result = mic.listen()
    print(result)