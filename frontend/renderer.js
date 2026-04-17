const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const gestureResult = document.getElementById('gesture_result');
const confidenceBar = document.getElementById('confidence_bar');
const confidenceVal = document.getElementById('confidence_val');
const backendStatus = document.getElementById('backend_status');
const statusText = document.getElementById('status_text');
const inferenceLog = document.getElementById('inference_log');
const loadingOverlay = document.getElementById('loading_overlay');
const fpsVal = document.getElementById('fps_val');
const bufferVal = document.getElementById('buffer_val');

let sequence = [];
let lastInferenceTime = 0;
let frameCount = 0;
let lastFpsUpdate = Date.now();

const ACTIONS = ['hello', 'thanks', 'iloveyou'];
const BACKEND_URL = 'http://localhost:8000/predict';

// Safer global mappings for MediaPipe in Electron
const Holistic = window.Holistic;
const Camera = window.Camera;
const drawConnectors = window.drawConnectors;
const drawLandmarks = window.drawLandmarks;
const POSE_CONNECTIONS = window.POSE_CONNECTIONS;
const HAND_CONNECTIONS = window.HAND_CONNECTIONS;
const FACEMESH_TESSELATION = window.FACEMESH_TESSELATION;

// Verify dependencies first
if (!Holistic || !Camera) {
    statusText.innerText = "Error: MediaPipe libs not found!";
    backendStatus.className = "dot red";
    console.error("Critical: MediaPipe dependencies are missing from window scope.");
}

// Initialize MediaPipe Holistic
const holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

function extractLandmarks(results) {
    const pose = results.poseLandmarks ? results.poseLandmarks.flatMap(res => [res.x, res.y, res.z, res.visibility]) : new Array(132).fill(0);
    const face = results.faceLandmarks ? results.faceLandmarks.flatMap(res => [res.x, res.y, res.z]) : new Array(1404).fill(0);
    const lh = results.leftHandLandmarks ? results.leftHandLandmarks.flatMap(res => [res.x, res.y, res.z]) : new Array(63).fill(0);
    const rh = results.rightHandLandmarks ? results.rightHandLandmarks.flatMap(res => [res.x, res.y, res.z]) : new Array(63).fill(0);
    return [...pose, ...face, ...lh, ...rh];
}

async function runInference() {
    if (sequence.length < 30) return;

    try {
        const response = await fetch(BACKEND_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence: sequence })
        });

        if (response.ok) {
            const data = await response.json();
            updateUI(data);
            updateBackendStatus(true);
        } else {
            console.error('Backend Error:', response.statusText);
            updateBackendStatus(false, "API Error");
        }
    } catch (error) {
        console.error('Fetch Error:', error);
        updateBackendStatus(false, "Not Connected");
    }
}

function updateUI(data) {
    const { action, confidence } = data;
    const confPercent = Math.round(confidence * 100);

    gestureResult.innerText = action;
    confidenceBar.style.width = `${confPercent}%`;
    confidenceVal.innerText = `${confPercent}% Confidence`;

    // Add to log if it's a strong prediction and different from last
    if (confidence > 0.7) {
        addToLog(action, confPercent);
    }
}

function addToLog(action, confidence) {
    const lastItem = inferenceLog.firstChild;
    if (lastItem && lastItem.dataset.action === action) return;

    const li = document.createElement('li');
    li.className = 'log-item';
    li.dataset.action = action;
    li.innerHTML = `<span>${action}</span> <span>${confidence}%</span>`;
    inferenceLog.prepend(li);

    // Keep only last 10
    if (inferenceLog.children.length > 10) {
        inferenceLog.removeChild(inferenceLog.lastChild);
    }
}

function updateBackendStatus(online, msg = "Vitals Stable") {
    backendStatus.className = `dot ${online ? 'green' : 'red'}`;
    statusText.innerText = online ? msg : "API Offline";
}

function onResults(results) {
    // Hide loading overlay on first results
    if (loadingOverlay.style.opacity !== '0') {
        loadingOverlay.style.opacity = '0';
        setTimeout(() => loadingOverlay.style.display = 'none', 500);
    }

    // Draw silhouettes
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    // Draw landmarks
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#ffffff33', lineWidth: 2});
    drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#38bdf8', lineWidth: 1, radius: 2});
    drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {color: '#38bdf8', lineWidth: 2});
    drawLandmarks(canvasCtx, results.leftHandLandmarks, {color: '#ffffff', lineWidth: 1, radius: 2});
    drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {color: '#38bdf8', lineWidth: 2});
    drawLandmarks(canvasCtx, results.rightHandLandmarks, {color: '#ffffff', lineWidth: 1, radius: 2});
    canvasCtx.restore();

    // Data Processing
    const landmarks = extractLandmarks(results);
    sequence.push(landmarks);

    // Maintain sliding window of 30 frames
    if (sequence.length > 30) {
        sequence.shift();
    }

    bufferVal.innerText = sequence.length;

    // Throttle inference to avoid overwhelming backend (every 5 frames when buffer full)
    if (sequence.length === 30 && frameCount % 5 === 0) {
        runInference();
    }

    // Performance Stats
    frameCount++;
    const now = Date.now();
    if (now - lastFpsUpdate > 1000) {
        fpsVal.innerText = frameCount;
        frameCount = 0;
        lastFpsUpdate = now;
    }
}

holistic.onResults(onResults);

try {
    const camera = new Camera(videoElement, {
        onFrame: async () => {
            try {
                await holistic.send({image: videoElement});
            } catch (err) {
                console.error("Holistic Send Error:", err);
            }
        },
        width: 640,
        height: 480
    });

    camera.start().catch(err => {
        console.error("Camera Start Error:", err);
        statusText.innerText = "Camera Access Denied";
        backendStatus.className = "dot red";
    });
} catch (err) {
    console.error("Camera Init Error:", err);
    statusText.innerText = "Hardware Failure";
}

// Initial Backend Check
fetch('http://localhost:8000/').then(r => r.ok ? updateBackendStatus(true) : updateBackendStatus(false))
.catch(() => updateBackendStatus(false));
