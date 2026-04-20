"""
SignFlow AI — Convert Keras model → ONNX for GPU inference.
Run once after training (or whenever action.h5 is replaced).

Usage:
    python convert_model.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # silence TF startup noise

from config import MODEL_H5, MODEL_ONNX, MODEL_DIR, ACTIONS_JSON, SEQUENCE_LENGTH


def convert():
    print("=" * 55)
    print("  SignFlow AI -- Model Converter  (h5 -> ONNX + DirectML)")
    print("=" * 55)

    if not os.path.exists(MODEL_H5):
        print(f"[ERROR] Model not found: {MODEL_H5}")
        print("        Run train_model.py first (or keep original action.h5).")
        sys.exit(1)

    print(f"\n[1/3] Loading Keras model from {MODEL_H5} ...")
    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(MODEL_H5)
    n_classes = model.output_shape[-1]
    print(f"      Input  shape : {model.input_shape}")
    print(f"      Output shape : {model.output_shape}  ({n_classes} classes)")

    # Save which actions this model knows (derived from its output size)
    from config import ACTIONS
    saved_actions = list(ACTIONS[:n_classes])
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(ACTIONS_JSON, 'w') as f:
        json.dump(saved_actions, f, indent=2)
    print(f"      Actions saved: {saved_actions}")

    print(f"\n[2/3] Converting to ONNX ...")
    # Use from_function for better LSTM compatibility
    @tf.function(input_signature=[tf.TensorSpec((None, SEQUENCE_LENGTH, 1662), tf.float32, name="input")])
    def model_fn(x):
        return model(x)

    tf2onnx.convert.from_function(
        model_fn,
        input_signature=[tf.TensorSpec((None, SEQUENCE_LENGTH, 1662), tf.float32, name="input")],
        output_path=MODEL_ONNX
    )
    size_mb = os.path.getsize(MODEL_ONNX) / 1e6
    print(f"      Saved -> {MODEL_ONNX}  ({size_mb:.1f} MB)")

    print(f"\n[3/3] Verifying ONNX model with DirectML ...")
    import onnxruntime as ort
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session   = ort.InferenceSession(MODEL_ONNX, providers=providers)
    active    = session.get_providers()[0]
    print(f"      Active provider : {active}")
    print(f"\n✓ Conversion complete! Restart the backend to load new model.")
    print("=" * 55)


if __name__ == "__main__":
    convert()
