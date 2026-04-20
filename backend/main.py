"""
SignFlow AI — FastAPI Backend
================================
Serves real-time predictions using an ONNX model with GPU via DirectML.
GPU is automatically used if a DirectX 12 compatible device is available;
falls back to CPU otherwise — no configuration needed.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from config import MODEL_ONNX, ACTIONS_JSON, ACTIONS, SEQUENCE_LENGTH


# ── Load action labels (from model's saved actions.json) ─────────────────────
def _load_actions() -> list:
    if os.path.exists(ACTIONS_JSON):
        with open(ACTIONS_JSON) as f:
            return json.load(f)
    return list(ACTIONS)   # fallback to config


# ── Load ONNX model — GPU (DirectML) if available, else CPU ──────────────────
def _load_session():
    try:
        import onnxruntime as ort
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        sess      = ort.InferenceSession(MODEL_ONNX, providers=providers)
        provider  = sess.get_providers()[0]
        return sess, provider, None
    except Exception as e:
        return None, None, str(e)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="SignFlow AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

session, gpu_provider, load_error = _load_session()
current_actions = _load_actions()

print(f"[SignFlow] Actions : {current_actions}")
print(f"[SignFlow] Provider: {gpu_provider or 'NONE — ' + str(load_error)}")


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    sequence: List[List[float]]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
async def status():
    return {
        "status":       "online",
        "model_ready":  session is not None,
        "gpu_provider": gpu_provider,
        "actions":      current_actions,
        "error":        load_error,
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    if session is None:
        raise HTTPException(
            503,
            detail=f"Model not loaded: {load_error}. "
                   f"Did you run convert_model.py? (manage.ps1 setup does this automatically)"
        )

    seq = np.array(request.sequence, dtype=np.float32)

    if seq.shape != (SEQUENCE_LENGTH, 1662):
        raise HTTPException(
            400,
            detail=f"Expected sequence shape ({SEQUENCE_LENGTH}, 1662), got {seq.shape}"
        )

    inp    = np.expand_dims(seq, axis=0)                          # (1, 30, 1662)
    result = session.run(None, {session.get_inputs()[0].name: inp})[0][0]  # (n_classes,)

    idx  = int(np.argmax(result))
    return {
        "action":        current_actions[idx],
        "confidence":    float(result[idx]),
        "probabilities": {current_actions[i]: float(result[i]) for i in range(len(current_actions))},
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
