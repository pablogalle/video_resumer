from pyannote.audio import Pipeline
from datetime import timedelta
import whisper
import torch
import os

def cargar_pipeline(token_path="models/hug_token.txt"):
    with open(token_path, "r") as f:
        token = f.read().strip()
    return Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=token)

def transcribir_con_diarizacion(audio_path, model_size="small"):
    pipeline = cargar_pipeline()
    diarization = pipeline(audio_path)
    model = whisper.load_model(model_size)

    resultados = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result = model.transcribe(audio_path, verbose=False, fp16=torch.cuda.is_available(),
                                  language="es", segment_start=turn.start, segment_end=turn.end)
        texto = result["text"].strip()
        if texto:
            tiempo = str(timedelta(seconds=int(turn.start)))
            resultados.append(f"[{tiempo}] {speaker}: {texto}")
    return "\n".join(resultados)
