"""
SignFlow AI — LSTM Training Script
=====================================
Loads collected .npy data, trains the LSTM, saves action.h5,
then auto-converts to ONNX so the backend can serve with GPU.

Usage:
    python train_model.py
"""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(__file__))

from config import ACTIONS, DATA_PATH, MODEL_H5, SEQUENCE_LENGTH, MODEL_DIR


def get_trainable_actions():
    """Return only actions that have at least 1 collected sample."""
    available = []
    for action in ACTIONS:
        path = os.path.join(DATA_PATH, action)
        if os.path.isdir(path) and any(f.endswith('.npy') for f in os.listdir(path)):
            count = len([f for f in os.listdir(path) if f.endswith('.npy')])
            available.append((action, count))
    return available


def load_dataset(actions):
    X, y = [], []
    for label, (action, _) in enumerate(actions):
        action_path = os.path.join(DATA_PATH, action)
        for fname in sorted(os.listdir(action_path)):
            if fname.endswith('.npy'):
                seq = np.load(os.path.join(action_path, fname))
                if seq.shape == (SEQUENCE_LENGTH, 1662):
                    X.append(seq)
                    y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


def train():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    # Ensure load_dataset can see numpy
    global np

    print("=" * 55)
    print("  SignFlow AI — LSTM Trainer")
    print("=" * 55)

    # ── GPU setup ─────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n[GPU] Training on: {gpus[0].name}")
    else:
        print("\n[CPU] No GPU detected — training on CPU (works fine for this model)")

    # ── Find available data ───────────────────────────────────
    trainable = get_trainable_actions()
    if not trainable:
        print("\n[ERROR] No training data found in:", DATA_PATH)
        print("        Run:  python collect_data.py")
        sys.exit(1)

    action_list = [a for a, _ in trainable]
    print(f"\n[DATA] Training on {len(action_list)} signs: {action_list}")
    for action, count in trainable:
        print(f"       {action:<12}  {count} sequences")

    # ── Load ──────────────────────────────────────────────────
    X, y = load_dataset(trainable)
    y_cat = to_categorical(y, num_classes=len(action_list))
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.1, random_state=42, stratify=y)
    print(f"\n[DATA] {X_train.shape[0]} train  /  {X_test.shape[0]} test  |  shape {X.shape[1:]}")

    # ── Model (clean 3-layer LSTM) ────────────────────────────
    model = Sequential([
        LSTM(64,  return_sequences=True,  activation='relu', input_shape=(SEQUENCE_LENGTH, 1662)),
        Dropout(0.2),
        LSTM(128, return_sequences=True,  activation='relu'),
        Dropout(0.2),
        LSTM(64,  return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(action_list), activation='softmax'),
    ], name="SignFlow_LSTM")

    model.summary()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────
    print(f"\n[TRAIN] Starting — up to 200 epochs with early stopping ...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200, batch_size=32,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[RESULT] Accuracy: {acc*100:.1f}%  |  Loss: {loss:.4f}")

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_H5)
    print(f"\n[SAVED] {MODEL_H5}")

    # ── Convert to ONNX ───────────────────────────────────────
    print("\n[CONVERT] Converting to ONNX for GPU inference ...")
    result = subprocess.run([sys.executable, "convert_model.py"], capture_output=False)
    if result.returncode == 0:
        print("\n✓ Training complete! Restart the backend to use the new model.")
    else:
        print("\n[WARN] ONNX conversion failed. Backend will still work via Keras fallback.")

    print("=" * 55)


if __name__ == "__main__":
    import numpy as np   # available for load_dataset closure
    train()
