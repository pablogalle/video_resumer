from flask import Flask, request, jsonify
import os
import tempfile

from audio_utils.audio_processing import extraer_audio
from audio_utils.whisper_transcriber import transcribir_audio_simple
from audio_utils.diarization import transcribir_con_diarizacion

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    return procesar_peticion(transcribir_audio_simple)

@app.route('/transcribe_diarized', methods=['POST'])
def transcribe_diarized():
    return procesar_peticion(transcribir_con_diarizacion)

def procesar_peticion(transcribir_funcion):
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo'}), 400

    file = request.files['file']
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, file.filename)
        file.save(input_path)

        if file.filename.lower().endswith(('.mp4', '.webm', '.mkv', '.avi')):
            audio_path = os.path.join(tmpdir, "audio.wav")
            extraer_audio(input_path, audio_path)
        elif file.filename.lower().endswith(('.mp3', '.wav')):
            audio_path = input_path
        else:
            audio_path = input_path

        try:
            texto = transcribir_funcion(audio_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        return jsonify({'transcripcion': texto})

if __name__ == '__main__':
    app.run(debug=True)
