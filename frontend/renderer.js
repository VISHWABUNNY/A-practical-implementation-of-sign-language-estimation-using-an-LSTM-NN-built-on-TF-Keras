// ── DOM REFERENCES ────────────────────────────────────────────────────────────
const videoElement   = document.getElementById('input_video');
const canvasElement  = document.getElementById('output_canvas');
const canvasCtx      = canvasElement.getContext('2d');
const gestureResult  = document.getElementById('gesture_result');
const confidenceBar  = document.getElementById('confidence_bar');
const confidenceVal  = document.getElementById('confidence_val');
const backendStatus  = document.getElementById('backend_status');
const statusText     = document.getElementById('status_text');
const inferenceLog   = document.getElementById('inference_log');
const loadingOverlay = document.getElementById('loading_overlay');
const fpsVal         = document.getElementById('fps_val');
const bufferVal      = document.getElementById('buffer_val');
const signMeaning    = document.getElementById('sign_meaning');
const signEmoji      = document.getElementById('sign_emoji');
const overlaySign    = document.getElementById('sign_overlay');
const translationBox = document.getElementById('translation_box');
const wordChips      = document.getElementById('word_chips');
const clearBtn       = document.getElementById('clear_btn');
const logEmpty       = document.getElementById('log_empty');
const gestureList    = document.getElementById('gesture_list');
const gpuLabel       = document.getElementById('gpu_label');
const gpuBadge       = document.getElementById('gpu_badge');

// ── FULL SIGN DICTIONARY (all 15 signs + extensible) ─────────────────────────
// Frontend knows about all possible signs. Bars are built from what the current
// model actually supports (fetched from backend /status endpoint).
const COLOR_PALETTE = [
    '#38bdf8', '#a78bfa', '#f472b6', '#4ade80',
    '#f87171', '#fbbf24', '#fb923c', '#34d399',
    '#60a5fa', '#818cf8', '#f9a8d4', '#86efac',
    '#fca5a5', '#fcd34d', '#6ee7b7', '#c4b5fd',
];

const SIGN_DICT = {
    // ── Greetings ──────────────────────────────────────────────
    hello: {
        display: 'Hello',
        emoji:   '👋',
        meaning: 'A welcoming greeting. Wave your open hand near your forehead.',
    },
    // ── Politeness ─────────────────────────────────────────────
    thanks: {
        display: 'Thank You',
        emoji:   '🙏',
        meaning: 'Gratitude. Flat hand from lips extends forward.',
    },
    // ── Emotions ───────────────────────────────────────────────
    iloveyou: {
        display: 'I Love You',
        emoji:   '🤟',
        meaning: 'The ILY hand-shape — combines I, L, and Y fingers.',
    },
    // ── Responses ──────────────────────────────────────────────
    yes: {
        display: 'Yes',
        emoji:   '✅',
        meaning: 'Agreement. Fist nods up and down like a head nodding.',
    },
    no: {
        display: 'No',
        emoji:   '❌',
        meaning: 'Disagreement. Index and middle finger snap against thumb.',
    },
    // ── Politeness ─────────────────────────────────────────────
    please: {
        display: 'Please',
        emoji:   '🤲',
        meaning: 'Request. Flat hand rubs circular motion on chest.',
    },
    sorry: {
        display: 'Sorry',
        emoji:   '😔',
        meaning: 'Apology. Fist rubs circular motion on chest.',
    },
    // ── Commands ───────────────────────────────────────────────
    help: {
        display: 'Help',
        emoji:   '🆘',
        meaning: 'Assistance needed. Thumb-up hand lifts on open palm.',
    },
    stop: {
        display: 'Stop',
        emoji:   '✋',
        meaning: 'Halt. Edge of flat hand chops down onto open palm.',
    },
    // ── States ─────────────────────────────────────────────────
    good: {
        display: 'Good',
        emoji:   '👍',
        meaning: 'Positive feedback. Flat hand from chin moves outward.',
    },
    bad: {
        display: 'Bad',
        emoji:   '👎',
        meaning: 'Negative feedback. Flat hand from chin flips downward.',
    },
    // ── Conversational ─────────────────────────────────────────
    more: {
        display: 'More',
        emoji:   '➕',
        meaning: 'Quantity. Fingertips of both hands tap together repeatedly.',
    },
    again: {
        display: 'Again',
        emoji:   '🔄',
        meaning: 'Repeat. Bent hand arcs and lands on open palm.',
    },
    fine: {
        display: 'Fine',
        emoji:   '😊',
        meaning: 'Doing well. Thumb touches chest and moves outward.',
    },
    name: {
        display: 'Name',
        emoji:   '🏷️',
        meaning: 'Identity. Crossed index fingers tap H-handshapes together.',
    },
};

// ── STATE ─────────────────────────────────────────────────────────────────────
const BACKEND_URL  = 'http://localhost:8000';
let sequence       = [];
let frameCount     = 0;
let lastFpsUpdate  = Date.now();
let lastAddedWord  = null;
let currentActions = [];   // populated from backend

// ── MEDIAPIPE GLOBALS ─────────────────────────────────────────────────────────
const Holistic         = window.Holistic;
const Camera           = window.Camera;
const drawConnectors   = window.drawConnectors;
const drawLandmarks    = window.drawLandmarks;
const POSE_CONNECTIONS = window.POSE_CONNECTIONS;
const HAND_CONNECTIONS = window.HAND_CONNECTIONS;

if (!Holistic || !Camera) {
    statusText.innerText = 'Error: MediaPipe not loaded!';
    backendStatus.className = 'dot red';
    console.error('Critical: MediaPipe globals missing.');
}

// ── MEDIAPIPE INIT ────────────────────────────────────────────────────────────
const holistic = new Holistic({
    locateFile: (file) => `./node_modules/@mediapipe/holistic/${file}`,
});
holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true,
                      minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });

// ── LANDMARK EXTRACTION ───────────────────────────────────────────────────────
// Must match collect_data.py extract_keypoints() exactly → 1662 floats
function extractLandmarks(results) {
    const pose = results.poseLandmarks
        ? results.poseLandmarks.flatMap(r => [r.x, r.y, r.z, r.visibility])
        : new Array(132).fill(0);
    const face = results.faceLandmarks
        ? results.faceLandmarks.flatMap(r => [r.x, r.y, r.z])
        : new Array(1404).fill(0);
    const lh = results.leftHandLandmarks
        ? results.leftHandLandmarks.flatMap(r => [r.x, r.y, r.z])
        : new Array(63).fill(0);
    const rh = results.rightHandLandmarks
        ? results.rightHandLandmarks.flatMap(r => [r.x, r.y, r.z])
        : new Array(63).fill(0);
    return [...pose, ...face, ...lh, ...rh];
}

// ── DYNAMIC GESTURE BAR BUILDER ───────────────────────────────────────────────
// Called once after fetching /status. Rebuilds bars to match actual model output.
function buildGestureBars(actions) {
    gestureList.innerHTML = '';
    actions.forEach((action, i) => {
        const info  = SIGN_DICT[action] || { display: action, emoji: '❓' };
        const color = COLOR_PALETTE[i % COLOR_PALETTE.length];

        const row = document.createElement('div');
        row.className = 'gesture-row';
        row.id        = `prob_${action}`;
        row.dataset.color = color;
        row.innerHTML = `
            <div class="g-name">${info.emoji} ${info.display}</div>
            <div class="g-bar-wrap"><div class="g-bar" id="bar_${action}" style="background:linear-gradient(90deg,${color},${color}99)"></div></div>
            <div class="g-pct" id="pct_${action}">0%</div>
        `;
        gestureList.appendChild(row);
    });
}

// ── INFERENCE ─────────────────────────────────────────────────────────────────
async function runInference() {
    if (sequence.length < 30 || currentActions.length === 0) return;
    try {
        const res = await fetch(`${BACKEND_URL}/predict`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ sequence }),
        });
        if (res.ok) {
            updateUI(await res.json());
            setBackendStatus(true);
        } else {
            setBackendStatus(false, 'API Error');
        }
    } catch {
        setBackendStatus(false, 'Backend Offline');
    }
}

// ── UI UPDATE ─────────────────────────────────────────────────────────────────
function updateUI(data) {
    const { action, confidence, probabilities } = data;
    const pct  = Math.round(confidence * 100);
    const idx  = currentActions.indexOf(action);
    const color = COLOR_PALETTE[idx >= 0 ? idx % COLOR_PALETTE.length : 0];
    const info = SIGN_DICT[action] || { display: action, emoji: '❓', meaning: 'Unknown sign' };

    // --- Detection card ---
    gestureResult.innerText = info.display;
    signEmoji.innerText     = info.emoji;
    signMeaning.innerText   = info.meaning;

    // --- Video overlay ---
    overlaySign.innerText           = `${info.emoji}  ${info.display}`;
    overlaySign.style.color         = color;
    overlaySign.style.borderColor   = color + '55';

    // --- Confidence bar ---
    confidenceBar.style.width          = `${pct}%`;
    confidenceBar.style.background     = `linear-gradient(90deg, ${color}, ${color}aa)`;
    confidenceBar.style.boxShadow      = `0 0 10px ${color}66`;
    confidenceVal.innerText            = `${pct}%`;
    confidenceVal.style.color          = color;

    // --- Per-gesture bars ---
    if (probabilities) updateProbBars(probabilities, action);

    // --- Translation + History (high-confidence only) ---
    if (confidence > 0.7) {
        addToLog(action, pct, info, color);
        addToTranslation(action, info, color);
    }
}

function updateProbBars(probs, topAction) {
    currentActions.forEach(act => {
        const pct   = Math.round((probs[act] || 0) * 100);
        const bar   = document.getElementById(`bar_${act}`);
        const pctEl = document.getElementById(`pct_${act}`);
        const row   = document.getElementById(`prob_${act}`);
        if (bar)   bar.style.width    = `${pct}%`;
        if (pctEl) pctEl.innerText    = `${pct}%`;
        if (row) {
            const active = act === topAction;
            row.style.borderColor     = active ? (row.dataset.color + '44') : 'transparent';
            row.style.background      = active ? (row.dataset.color + '11') : 'rgba(255,255,255,0.02)';
            if (pctEl) pctEl.style.color = active ? row.dataset.color : '#475569';
        }
    });
}

// ── TRANSLATION ───────────────────────────────────────────────────────────────
function addToTranslation(action, info, color) {
    if (action === lastAddedWord) return;
    lastAddedWord = action;

    // Append word to text box
    const current = translationBox.innerText === 'Your sentence will appear here...'
        ? '' : (translationBox.innerText + ' ');
    translationBox.innerText = current + info.display;

    // Coloured chip
    const chip = document.createElement('span');
    chip.className  = 'word-chip';
    chip.innerText  = `${info.emoji} ${info.display}`;
    chip.style.cssText = `background:${color}18; color:${color}; border:1px solid ${color}44;`;
    wordChips.appendChild(chip);
    chip.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── HISTORY LOG ───────────────────────────────────────────────────────────────
function addToLog(action, pct, info, color) {
    if (inferenceLog.firstChild && inferenceLog.firstChild.dataset.action === action) return;

    logEmpty.style.display = 'none';
    const time = new Date().toLocaleTimeString('en', { hour12: false });

    const li = document.createElement('li');
    li.className  = 'log-item';
    li.dataset.action = action;
    li.style.borderLeftColor = color;
    li.innerHTML = `
        <span class="log-sign">${info.emoji} ${info.display}</span>
        <span class="log-conf" style="color:${color}">${pct}%</span>
        <span class="log-time">${time}</span>
    `;
    inferenceLog.prepend(li);
    while (inferenceLog.children.length > 10) inferenceLog.removeChild(inferenceLog.lastChild);
}

// ── BACKEND STATUS ────────────────────────────────────────────────────────────
function setBackendStatus(online, msg = 'Backend Online') {
    backendStatus.className = `dot ${online ? 'green' : 'red'}`;
    statusText.innerText    = msg;
}

function updateGpuBadge(provider) {
    if (!provider) {
        gpuLabel.innerText = 'CPU Mode';
        gpuBadge.classList.add('gpu-cpu');
        return;
    }
    if (provider.includes('Dml')) {
        gpuLabel.innerText = 'GPU (DirectML)';
        gpuBadge.classList.add('gpu-active');
    } else if (provider.includes('CUDA')) {
        gpuLabel.innerText = 'GPU (CUDA)';
        gpuBadge.classList.add('gpu-active');
    } else {
        gpuLabel.innerText = 'CPU Mode';
        gpuBadge.classList.add('gpu-cpu');
    }
}

// ── CLEAR ─────────────────────────────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
    lastAddedWord = null;
    translationBox.innerHTML = '<span class="translation-placeholder">Your sentence will appear here...</span>';
    wordChips.innerHTML = '';
});

// ── ON RESULTS (MediaPipe callback) ───────────────────────────────────────────
holistic.onResults(function onResults(results) {
    if (loadingOverlay.style.opacity !== '0') {
        loadingOverlay.style.opacity = '0';
        setTimeout(() => { loadingOverlay.style.display = 'none'; }, 500);
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (drawConnectors && drawLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
            { color: 'rgba(255,255,255,0.12)', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.poseLandmarks,
            { color: '#38bdf8', lineWidth: 1, radius: 2 });
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
            { color: 'rgba(56,189,248,0.7)', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.leftHandLandmarks,
            { color: '#ffffff', lineWidth: 1, radius: 3 });
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
            { color: 'rgba(244,114,182,0.7)', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.rightHandLandmarks,
            { color: '#f472b6', lineWidth: 1, radius: 3 });
    }
    canvasCtx.restore();

    sequence.push(extractLandmarks(results));
    if (sequence.length > 30) sequence.shift();
    bufferVal.innerText = sequence.length;

    if (sequence.length === 30 && frameCount % 5 === 0) runInference();

    frameCount++;
    const now = Date.now();
    if (now - lastFpsUpdate > 1000) {
        fpsVal.innerText = frameCount;
        frameCount       = 0;
        lastFpsUpdate    = now;
    }
});

// ── CAMERA ────────────────────────────────────────────────────────────────────
try {
    const camera = new Camera(videoElement, {
        onFrame: async () => {
            try { await holistic.send({ image: videoElement }); }
            catch (err) { console.error('Holistic send error:', err); }
        },
        width: 640, height: 480,
    });
    camera.start().catch(err => {
        statusText.innerText = 'Camera Access Denied';
        backendStatus.className = 'dot red';
        console.error('Camera error:', err);
    });
} catch (err) {
    statusText.innerText = 'Camera Hardware Failure';
    console.error('Camera init error:', err);
}

// ── BACKEND HEALTH + DYNAMIC ACTION LOAD ──────────────────────────────────────
async function checkBackend() {
    try {
        const res  = await fetch(`${BACKEND_URL}/`);
        const data = await res.json();

        setBackendStatus(data.status === 'online',
            data.model_ready ? 'Backend Online' : 'Model Not Ready — Run convert_model.py');
        updateGpuBadge(data.gpu_provider);

        if (data.actions && data.actions.length > 0 && currentActions.length === 0) {
            currentActions = data.actions;
            buildGestureBars(currentActions);
            console.log(`[SignFlow] Loaded ${currentActions.length} actions:`, currentActions);
        }
    } catch {
        setBackendStatus(false, 'Backend Offline');
        gpuLabel.innerText = 'Offline';
    }
}

checkBackend();
setInterval(checkBackend, 10000);  // re-check every 10s
