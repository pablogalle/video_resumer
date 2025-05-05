import whisper
import subprocess
import os
import tempfile
import torch
import time

VIDEO_PATH = "testVideo1.webm"  # cambia este path

print("""🔹 CUDA disponible:""", torch.cuda.is_available())
if torch.cuda.is_available(): print("""🔸 Dispositivo CUDA:""", torch.cuda.get_device_name(0))

def extraer_audio(video_path, output_audio_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_audio_path
    ], check=True)

def transcribir_audio(audio_path, model_size="small"):
    print("Cargando modelo Whisper...")
    model = whisper.load_model(model_size)
    print("Transcribiendo...")
    start_time = time.time()
    result = model.transcribe(audio_path)
    end_time = time.time()
    print(f"⏱️ Transcripción completada en {end_time - start_time:.2f} segundos.")
    return result["text"]

def resumir_texto_con_ollama(texto, modelo="mistral"):
    prompt = f"Resume el siguiente texto en puntos clave, en español:\n\n{texto}\n\nResumen:"
    start_time = time.time()

    result = subprocess.run(
        ["ollama", "run", modelo],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    end_time = time.time()
    print(f"⏱️ Resumen generado en {end_time - start_time:.2f} segundos.")
    return result.stdout.decode("utf-8")

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        extraer_audio(VIDEO_PATH, audio_path)
        
        texto = transcribir_audio(audio_path, model_size="small")
        print("\n🔸 TRANSCRIPCIÓN COMPLETA:\n")
        print(texto[:1000] + "...\n")  # Muestra parte
        
        resumen = resumir_texto_con_ollama(texto)
        print("\n🔹 RESUMEN GENERADO:\n")
        print(resumen)

if __name__ == "__main__":
    main()
