"""
SignFlow AI — Central Configuration
====================================
ONE place to control everything. Change ACTIONS here, then:
  1.  python collect_data.py   ← collect webcam samples for new signs
  2.  python train_model.py    ← retrain LSTM + convert to ONNX
  3.  Restart backend          ← picks up new model automatically
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Signs the system will recognise ──────────────────────────────────────────
# Pre-trained model supports the first 3. Add more, collect data, then retrain.
ACTIONS = [
    # ── Original (pre-trained) ──────────────────────
    'hello',     'thanks',   'iloveyou',
    # ── Basic responses ─────────────────────────────
    'yes',       'no',
    # ── Politeness ──────────────────────────────────
    'please',    'sorry',
    # ── Key commands ────────────────────────────────
    'help',      'stop',
    # ── States ──────────────────────────────────────
    'good',      'bad',
    # ── Conversational ──────────────────────────────
    'more',      'again',    'fine',      'name',
]

# ── Sequence parameters (MUST match whatever the model was trained on) ────────
SEQUENCE_LENGTH = 30   # frames per prediction window
N_SEQUENCES     = 30   # sequences to collect per action

# ── File paths ────────────────────────────────────────────────────────────────
MODEL_DIR    = os.path.join(BASE_DIR, 'model')
MODEL_H5     = os.path.join(MODEL_DIR, 'action.h5')
MODEL_ONNX   = os.path.join(MODEL_DIR, 'action.onnx')
ACTIONS_JSON = os.path.join(MODEL_DIR, 'actions.json')
DATA_PATH    = os.path.join(BASE_DIR,  'data')
