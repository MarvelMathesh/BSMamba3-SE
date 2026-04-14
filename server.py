"""
BSMamba3-SE FastAPI Server
══════════════════════════
REST API for real-time speech enhancement.
POST /enhance with audio file → returns enhanced audio.
"""

import io
import os
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from inference import BSMamba3Enhancer


# Global enhancer instance
enhancer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global enhancer
    ckpt_path = os.environ.get(
        'BSMAMBA3_CHECKPOINT',
        './checkpoints/bsmamba3/checkpoint_best.pt'
    )
    device = os.environ.get('BSMAMBA3_DEVICE', 'cuda')
    print(f"[Server] Loading model from {ckpt_path}")
    enhancer = BSMamba3Enhancer(ckpt_path, device)
    print("[Server] Model loaded and ready")
    yield
    print("[Server] Shutting down")


app = FastAPI(
    title="BSMamba3-SE Speech Enhancement API",
    description="Upload noisy audio, get enhanced audio back.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/enhance")
async def enhance_audio(file: UploadFile = File(...)):
    """
    Enhance a noisy audio file.
    
    Accepts: WAV, FLAC, or any soundfile-supported format.
    Returns: Enhanced WAV audio (16kHz, mono, float32).
    """
    # Read uploaded audio
    audio_bytes = await file.read()
    audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')

    # Resample if needed
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

    # Mono conversion
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Enhance
    enhanced = enhancer.enhance(audio_data)

    # Write to bytes
    output_buffer = io.BytesIO()
    sf.write(output_buffer, enhanced, 16000, format='WAV', subtype='FLOAT')
    output_buffer.seek(0)

    return Response(
        content=output_buffer.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename=enhanced_{file.filename}"},
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": enhancer is not None,
        "params": enhancer.model.count_parameters() if enhancer else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
