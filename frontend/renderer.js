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

// Studio UI
const studioBtn         = document.getElementById('studio_btn');
const studioOverlay     = document.getElementById('studio_overlay');
const studioSignSelect  = document.getElementById('studio_sign_select');
const recordBtn         = document.getElementById('record_btn');
const closeStudioBtn    = document.getElementById('close_studio_btn');
const recordProgress    = document.getElementById('record_progress');
const recordStatus      = document.getElementById('record_status');

// ── COLOR PALETTE ─────────────────────────────────────────────────────────────
const COLOR_PALETTE = [
    '#38bdf8', '#a78bfa', '#f472b6', '#4ade80',
    '#f87171', '#fbbf24', '#fb923c', '#34d399',
    '#60a5fa', '#818cf8', '#f9a8d4', '#86efac',
    '#fca5a5', '#fcd34d', '#6ee7b7', '#c4b5fd',
];

// ── CUSTOM SIGN DICTIONARY ────────────────────────────────────────────────────
// Specific definitions for emojis/meanings. For missing words, it auto-generates.
const SPECIFIC_SIGNS = {
    hello:      { display: 'Hello',       emoji: '👋', meaning: 'A welcoming greeting.' },
    thanks:     { display: 'Thank You',   emoji: '🙏', meaning: 'Gratitude.' },
    iloveyou:   { display: 'I Love You',  emoji: '🤟', meaning: 'The ILY hand-shape.' },
    yes:        { display: 'Yes',         emoji: '✅', meaning: 'Agreement.' },
    no:         { display: 'No',          emoji: '❌', meaning: 'Disagreement.' },
    please:     { display: 'Please',      emoji: '🤲', meaning: 'Request.' },
    sorry:      { display: 'Sorry',       emoji: '😔', meaning: 'Apology.' },
    help:       { display: 'Help',        emoji: '🆘', meaning: 'Assistance needed.' },
    stop:       { display: 'Stop',        emoji: '✋', meaning: 'Halt.' },
    good:       { display: 'Good',        emoji: '👍', meaning: 'Positive feedback.' },
    bad:        { display: 'Bad',         emoji: '👎', meaning: 'Negative feedback.' },
    home:       { display: 'Home',        emoji: '🏠', meaning: 'House or home.' },
    eat:        { display: 'Eat / Food',  emoji: '🍔', meaning: 'Eating or food.' },
    drink:      { display: 'Drink',       emoji: '🥤', meaning: 'Drinking or water.' },
    bathroom:   { display: 'Bathroom',    emoji: '🚽', meaning: 'Toilet or bathroom.' },
    time:       { display: 'Time',        emoji: '⌚', meaning: 'Clock or time.' },
    space:      { display: 'Space',       emoji: '␣',  meaning: 'Add a space between words.' },
    backspace:  { display: 'Backspace',   emoji: '⌫',  meaning: 'Delete last letter/word.' },
};

function getSignInfo(action) {
    if (SPECIFIC_SIGNS[action]) return SPECIFIC_SIGNS[action];
    
    // Auto-generate for A-Z
    if (action.length === 1 && action.match(/[a-z]/)) {
        return { display: action.toUpperCase(), emoji: '🔠', meaning: 'Fingerspelling letter' };
    }
    
    // Auto-generate for normal words (capitalise first letter)
    return {
        display: action.charAt(0).toUpperCase() + action.slice(1),
        emoji: '💬',
        meaning: `The ASL sign for '${action}'.`
    };
}

// ── STATE ─────────────────────────────────────────────────────────────────────
const BACKEND_URL  = 'http://localhost:8000';
let sequence       = [];
let frameCount     = 0;
let lastFpsUpdate  = Date.now();
let lastAddedAction= null;
let currentActions = [];   // populated from backend
let isSpellingWord = false;// tracks if we are currently building a fingerspelled word

// Studio State
let isRecordingStudio = false;
let recordingSign     = '';
let collectedClips    = [];
const TARGET_CLIPS    = 30;

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
}

const holistic = new Holistic({
    locateFile: (file) => `./node_modules/@mediapipe/holistic/${file}`,
});
holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true,
                      minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });

// Must match collect_data.py exactly
function extractLandmarks(results) {
    const pose = results.poseLandmarks ? results.poseLandmarks.flatMap(r => [r.x, r.y, r.z, r.visibility]) : new Array(132).fill(0);
    const face = results.faceLandmarks ? results.faceLandmarks.flatMap(r => [r.x, r.y, r.z])       : new Array(1404).fill(0);
    const lh   = results.leftHandLandmarks ? results.leftHandLandmarks.flatMap(r => [r.x, r.y, r.z]) : new Array(63).fill(0);
    const rh   = results.rightHandLandmarks ? results.rightHandLandmarks.flatMap(r => [r.x, r.y, r.z]) : new Array(63).fill(0);
    return [...pose, ...face, ...lh, ...rh];
}

function buildGestureBars(actions) {
    gestureList.innerHTML = '';
    actions.forEach((action, i) => {
        const info  = getSignInfo(action);
        const color = COLOR_PALETTE[i % COLOR_PALETTE.length];
        const row = document.createElement('div');
        row.className = 'gesture-row'; row.id = `prob_${action}`; row.dataset.color = color;
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
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence }),
        });
        if (res.ok) { updateUI(await res.json()); setBackendStatus(true); }
        else { setBackendStatus(false, 'API Error'); }
    } catch { setBackendStatus(false, 'Backend Offline'); }
}

// ── UI UPDATE ─────────────────────────────────────────────────────────────────
function updateUI(data) {
    const { action, confidence, probabilities } = data;
    const pct   = Math.round(confidence * 100);
    const idx   = currentActions.indexOf(action);
    const color = COLOR_PALETTE[idx >= 0 ? idx % COLOR_PALETTE.length : 0];
    const info  = getSignInfo(action);

    gestureResult.innerText = info.display;
    signEmoji.innerText     = info.emoji;
    signMeaning.innerText   = info.meaning;

    overlaySign.innerText         = `${info.emoji}  ${info.display}`;
    overlaySign.style.color       = color;
    overlaySign.style.borderColor = color + '55';

    confidenceBar.style.width      = `${pct}%`;
    confidenceBar.style.background = `linear-gradient(90deg, ${color}, ${color}aa)`;
    confidenceBar.style.boxShadow  = `0 0 10px ${color}66`;
    confidenceVal.innerText        = `${pct}%`;
    confidenceVal.style.color      = color;

    if (probabilities) {
        currentActions.forEach(act => {
            const pctAct = Math.round((probabilities[act] || 0) * 100);
            const bar = document.getElementById(`bar_${act}`);
            const pctEl = document.getElementById(`pct_${act}`);
            const row = document.getElementById(`prob_${act}`);
            if (bar) bar.style.width = `${pctAct}%`;
            if (pctEl) pctEl.innerText = `${pctAct}%`;
            if (row) {
                const active = act === action;
                row.style.borderColor = active ? (row.dataset.color + '44') : 'transparent';
                row.style.background  = active ? (row.dataset.color + '11') : 'rgba(255,255,255,0.02)';
                if (pctEl) pctEl.style.color = active ? row.dataset.color : '#475569';
            }
        });
    }

    if (confidence > 0.7 && action !== lastAddedAction) {
        lastAddedAction = action;
        addToLog(action, pct, info, color);
        handleTranslation(action, info, color);
    }
}

// ── TRANSLATION LOGIC (Handles Words + Letters A-Z) ───────────────────────────
function handleTranslation(action, info, color) {
    const isLetter = action.length === 1 && action.match(/[a-z]/i);

    // Initial placeholder clear
    if (translationBox.innerText === 'Your sentence will appear here...') {
        translationBox.innerText = '';
        wordChips.innerHTML = '';
    }

    if (action === 'space') {
        translationBox.innerText += ' ';
        isSpellingWord = false;
        return;
    }

    if (action === 'backspace') {
        const currentText = translationBox.innerText;
        translationBox.innerText = currentText.slice(0, -1);
        // Clean up last chip if spelling word gets fully deleted
        const lastChip = wordChips.lastChild;
        if (lastChip && lastChip.innerText.includes('🔠')) {
            lastChip.innerText = lastChip.innerText.slice(0, -1);
            if (lastChip.innerText === '🔠 ') wordChips.removeChild(lastChip);
        }
        return;
    }

    if (isLetter) {
        translationBox.innerText += info.display.toUpperCase();
        
        if (!isSpellingWord || wordChips.children.length === 0) {
            // Start a new spelt word chip
            const chip = document.createElement('span');
            chip.className = 'word-chip';
            chip.innerText = `🔠 ${info.display.toUpperCase()}`;
            chip.style.cssText = `background:${color}18; color:${color}; border:1px solid ${color}44;`;
            wordChips.appendChild(chip);
            isSpellingWord = true;
        } else {
            // Append to existing spelt word chip
            const lastChip = wordChips.lastChild;
            lastChip.innerText += info.display.toUpperCase();
        }
    } else {
        // Normal ASL Word
        if (translationBox.innerText.length > 0 && !translationBox.innerText.endsWith(' ')) {
            translationBox.innerText += ' ';
        }
        translationBox.innerText += info.display;
        
        const chip = document.createElement('span');
        chip.className = 'word-chip';
        chip.innerText = `${info.emoji} ${info.display}`;
        chip.style.cssText = `background:${color}18; color:${color}; border:1px solid ${color}44;`;
        wordChips.appendChild(chip);
        
        isSpellingWord = false;
    }

    // Scroll newest chips into view
    const last = wordChips.lastChild;
    if (last) last.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function addToLog(action, pct, info, color) {
    logEmpty.style.display = 'none';
    const time = new Date().toLocaleTimeString('en', { hour12: false });
    const li = document.createElement('li');
    li.className  = 'log-item';
    li.style.borderLeftColor = color;
    li.innerHTML = `
        <span class="log-sign">${info.emoji} ${info.display}</span>
        <span class="log-conf" style="color:${color}">${pct}%</span>
        <span class="log-time">${time}</span>
    `;
    inferenceLog.prepend(li);
    while (inferenceLog.children.length > 10) inferenceLog.removeChild(inferenceLog.lastChild);
}

function setBackendStatus(online, msg = 'Backend Online') {
    backendStatus.className = `dot ${online ? 'green' : 'red'}`;
    statusText.innerText    = msg;
}

function updateGpuBadge(provider) {
    if (!provider) { gpuLabel.innerText = 'CPU Mode'; gpuBadge.className = 'gpu-badge gpu-cpu'; return; }
    if (provider.includes('Dml') || provider.includes('CUDA')) {
        gpuLabel.innerText = provider.includes('Dml') ? 'GPU (DirectML)' : 'GPU (CUDA)';
        gpuBadge.className = 'gpu-badge gpu-active';
    } else {
        gpuLabel.innerText = 'CPU Mode';
        gpuBadge.className = 'gpu-badge gpu-cpu';
    }
}

clearBtn.addEventListener('click', () => {
    lastAddedAction = null;
    isSpellingWord  = false;
    translationBox.innerHTML = '<span class="translation-placeholder">Your sentence will appear here...</span>';
    wordChips.innerHTML = '';
});

holistic.onResults(function onResults(results) {
    if (loadingOverlay.style.opacity !== '0') {
        loadingOverlay.style.opacity = '0';
        setTimeout(() => { loadingOverlay.style.display = 'none'; }, 500);
    }
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    if (drawConnectors && drawLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: 'rgba(255,255,255,0.12)', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#38bdf8', lineWidth: 1, radius: 2 });
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: 'rgba(56,189,248,0.7)', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.leftHandLandmarks, { color: '#ffffff', lineWidth: 1, radius: 3 });
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: 'rgba(244,114,182,0.7)', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.rightHandLandmarks, { color: '#f472b6', lineWidth: 1, radius: 3 });
    }
    canvasCtx.restore();

    sequence.push(extractLandmarks(results));
    
    // --- STUDIO MODE ---
    if (isRecordingStudio) {
        if (sequence.length > 30) sequence.shift();
        bufferVal.innerText = sequence.length;
        
        if (sequence.length === 30) {
            collectedClips.push([...sequence]);
            sequence = []; // empty the buffer to start fresh for the next clip
            
            // Visual feedback
            const pct = Math.round((collectedClips.length / TARGET_CLIPS) * 100);
            recordProgress.style.width = `${pct}%`;
            recordStatus.innerText = `Captured ${collectedClips.length} / ${TARGET_CLIPS} clips...`;
            videoElement.style.border = '5px solid var(--green)';
            setTimeout(() => { videoElement.style.border = 'none'; }, 150);

            if (collectedClips.length >= TARGET_CLIPS) {
                finishRecording();
            }
        }
        return; // skip inference while recording
    }
    
    // --- INFERENCE MODE ---
    if (sequence.length > 30) sequence.shift();
    bufferVal.innerText = sequence.length;
    if (sequence.length === 30 && frameCount % 5 === 0) runInference();

    frameCount++;
    const now = Date.now();
    if (now - lastFpsUpdate > 1000) { fpsVal.innerText = frameCount; frameCount = 0; lastFpsUpdate = now; }
});

try {
    const camera = new Camera(videoElement, {
        onFrame: async () => { try { await holistic.send({ image: videoElement }); } catch(err){} },
        width: 640, height: 480,
    });
    camera.start().catch(() => { statusText.innerText = 'Camera Access Denied'; backendStatus.className = 'dot red'; });
} catch (err) { statusText.innerText = 'Camera Hardware Failure'; }

// ── STUDIO RECORDING LOGIC ────────────────────────────────────────────────────
studioBtn.addEventListener('click', () => {
    studioOverlay.style.display = 'flex';
});
closeStudioBtn.addEventListener('click', () => {
    if (isRecordingStudio) return alert('Please wait for recording to finish!');
    studioOverlay.style.display = 'none';
});
recordBtn.addEventListener('click', () => {
    recordingSign = studioSignSelect.value;
    if (!recordingSign) return alert('Select a sign first!');
    
    isRecordingStudio = true;
    collectedClips = [];
    sequence = [];
    recordBtn.innerText = '⏺ Recording...';
    recordBtn.style.background = 'var(--red)';
    recordProgress.style.width = '0%';
    recordStatus.innerText = `Recording 0 / 30... Keep signing!`;
});

async function finishRecording() {
    isRecordingStudio = false;
    recordBtn.innerText = '⌛ Saving...';
    recordBtn.style.background = 'var(--amber)';
    recordStatus.innerText = 'Uploading to backend...';

    try {
        const res = await fetch(`${BACKEND_URL}/collect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: recordingSign, sequences: collectedClips })
        });
        const data = await res.json();
        if (data.status === 'success') {
            recordStatus.innerText = `Saved successfully! Total clips for this sign: ${data.total_for_action}`;
        }
    } catch (err) {
        recordStatus.innerText = 'Error saving data.';
    }

    recordBtn.innerText = '⏺ Start Recording';
    recordBtn.style.background = 'var(--green)';
}

async function checkBackend() {
    try {
        const res  = await fetch(`${BACKEND_URL}/`);
        const data = await res.json();
        setBackendStatus(data.status === 'online', data.model_ready ? 'Backend Online' : 'Model Not Ready');
        updateGpuBadge(data.gpu_provider);
        
        // Populate gesture bars for trained actions
        if (data.actions && data.actions.length > 0 && currentActions.length === 0) {
            currentActions = data.actions;
            buildGestureBars(currentActions);
        }

        // Populate dropdown in Studio for all possible actions
        if (data.all_actions && studioSignSelect.options.length <= 1) {
            studioSignSelect.innerHTML = '<option value="">-- Choose Sign --</option>';
            data.all_actions.forEach(a => {
                const opt = document.createElement('option');
                opt.value = a; opt.innerText = getSignInfo(a).display;
                studioSignSelect.appendChild(opt);
            });
        }

    } catch { setBackendStatus(false, 'Backend Offline'); gpuLabel.innerText = 'Offline'; }
}

checkBackend();
setInterval(checkBackend, 10000);
