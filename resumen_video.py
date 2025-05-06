from flask import Flask, request, jsonify
import whisper
import subprocess
import tempfile
import os
import torch
import time

app = Flask(__name__)

print("🔹 CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🔸 Dispositivo CUDA:", torch.cuda.get_device_name(0))


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

@app.route('/transcribe', methods=['POST'])
def transcribir():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo'}), 400

    file = request.files['file']

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, file.filename)
        file.save(input_path)

        # Detectar si es video o audio
        if file.filename.lower().endswith(('.mp4', '.webm', '.mkv', '.avi')):
            audio_path = os.path.join(tmpdir, "audio.wav")
            extraer_audio(input_path, audio_path)
        elif file.filename.lower().endswith(('.mp3', '.wav')):
            audio_path = input_path
        else:
            audio_path = input_path

        try:
            texto = transcribir_audio(audio_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        return jsonify({'transcripcion': texto})

if __name__ == '__main__':
    app.run(debug=True)
