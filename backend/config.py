"""
SignFlow AI — Central Configuration
====================================
ONE place to control everything.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. The ASL Alphabet (Fingerspelling) ─────────────────────────────────────
ALPHABET = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'space', 'backspace' # Controls for spelling
]

# ── 2. Common ASL Dictionary (100+ Words) ────────────────────────────────────
COMMON_WORDS = [
    # ── Provided pre-trained originally
    'hello', 'thanks', 'iloveyou',
    # ── Questions
    'who', 'what', 'where', 'when', 'why', 'how',
    # ── People & Family
    'mother', 'father', 'brother', 'sister', 'baby', 'friend', 'man', 'woman',
    # ── Pronouns
    'me', 'you', 'he', 'she', 'we', 'they', 'my', 'your',
    # ── Time
    'now', 'later', 'today', 'tomorrow', 'yesterday', 'morning', 'night', 'day',
    # ── Feelings & States
    'happy', 'sad', 'angry', 'tired', 'hungry', 'thirsty', 'sick', 'excited',
    'good', 'bad', 'fine',
    # ── Actions (Verbs)
    'eat', 'drink', 'sleep', 'walk', 'run', 'read', 'write', 'learn', 'play',
    'work', 'help', 'stop', 'go', 'want', 'need', 'more', 'again',
    # ── Objects & Places
    'food', 'water', 'bathroom', 'home', 'school', 'store', 'car', 'book', 'phone',
    # ── Adjectives
    'big', 'small', 'hot', 'cold', 'clean', 'dirty',
    # ── Politeness / Responses
    'yes', 'no', 'please', 'sorry', 'name'
]

# ── COMBINED ACTIONS LIST ────────────────────────────────────────────────────
# This is the master list of all signs the system is programmed to handle.
ACTIONS = ALPHABET + COMMON_WORDS

# ── Sequence parameters (MUST match whatever the model was trained on) ────────
SEQUENCE_LENGTH = 30   # frames per prediction window
N_SEQUENCES     = 30   # sequences to collect per action

# ── File paths ────────────────────────────────────────────────────────────────
MODEL_DIR    = os.path.join(BASE_DIR, 'model')
MODEL_H5     = os.path.join(MODEL_DIR, 'action.h5')
MODEL_ONNX   = os.path.join(MODEL_DIR, 'action.onnx')
ACTIONS_JSON = os.path.join(MODEL_DIR, 'actions.json')
DATA_PATH    = os.path.join(BASE_DIR,  'data')
