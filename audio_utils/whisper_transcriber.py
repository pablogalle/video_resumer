import whisper
import torch
import time

def transcribir_audio_simple(audio_path, model_size="small"):
    print("Cargando modelo Whisper...")
    model = whisper.load_model(model_size)
    print("Transcribiendo...")
    start_time = time.time()
    result = model.transcribe(audio_path)
    print(f"⏱️ Transcripción completada en {time.time() - start_time:.2f} segundos.")
    return result["text"]
