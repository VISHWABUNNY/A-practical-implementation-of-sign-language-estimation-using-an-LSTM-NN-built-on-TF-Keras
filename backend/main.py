import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Suppress TF warnings if desired
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global variable for model
model = None
tf_available = False

try:
    import tensorflow as tf
    tf_available = True
except ImportError:
    print("WARNING: TensorFlow not found. AI Predictions will be disabled.")

app = FastAPI(title="Sign Language Recognition API")

# Enable CORS for the React/Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model if TF is available
if tf_available:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "action.h5")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Actions list as defined in the training notebook
ACTIONS = np.array(['hello', 'thanks', 'iloveyou'])

class PredictionRequest(BaseModel):
    # Expecting 30 frames of landmarks
    # Each frame has pose (33), face (468), left_hand (21), right_hand (21)
    # Total features = 1662
    sequence: List[List[float]]

@app.get("/")
async def root():
    return {
        "message": "Sign Language Recognition API is running",
        "tensorflow_ready": tf_available,
        "model_ready": model is not None
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    if not tf_available:
        raise HTTPException(status_code=500, detail="TensorFlow is not installed. Continuous Integration/Environment issue detected.")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found or corrupted.")
    
    # Expected input shape: (1, 30, 1662)
    sequence = np.array(request.sequence)
    
    if sequence.shape != (30, 1662):
        raise HTTPException(status_code=400, detail=f"Invalid sequence shape. Expected (30, 1662), got {sequence.shape}")
    
    try:
        # Add batch dimension
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        
        prediction_idx = np.argmax(res)
        confidence = float(res[prediction_idx])
        action = ACTIONS[prediction_idx]
        
        return {
            "action": action,
            "confidence": confidence,
            "probabilities": {ACTIONS[i]: float(res[i]) for i in range(len(ACTIONS))}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
