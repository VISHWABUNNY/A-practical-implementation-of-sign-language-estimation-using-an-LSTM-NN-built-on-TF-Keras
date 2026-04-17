# SignFlow AI Desktop

A premium, real-time sign language recognition **Desktop Application** using an LSTM Neural Network built on TensorFlow/Keras and MediaPipe Holistic landmarks.

## Project Structure

- `backend/`: FastAPI server handling LSTM model inference.
- `frontend/`: Electron + React + Tailwind CSS desktop application for real-time tracking.
- `backend/research/`: Original training notebooks and datasets.

## Quick Start

### 1. Setup Environment
Ensure you have **Python 3.12** and **Node.js** installed.
Run the management script to initialize all environments:
```powershell
.\manage.ps1 setup
```

### 2. Run the Application
Launch the standalone desktop application and the prediction backend:
```powershell
.\manage.ps1 run
```
The **SignFlow AI** desktop window will appear automatically.

## How it Works
1. **Frontend**: Captures webcam feed and uses `@mediapipe/holistic` to detect 1662 landmarks (Pose, Face, Hands).
2. **Buffer**: Landmarks are collected into a 30-frame sequence.
3. **Backend**: The sequence is sent via POST request to the FastAPI server.
4. **Inference**: The LSTM model processes the sequence and returns the predicted action (e.g., 'hello', 'thanks', 'iloveyou') with a confidence score.
5. **UI**: Predictions are displayed in a clean, glassmorphic interface.
