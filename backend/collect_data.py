"""
SignFlow AI — Data Collector
==============================
Uses your webcam + MediaPipe to record training samples for each sign.
Each sample = 30 frames of 1662 landmarks → saved as .npy file.

Usage:
    python collect_data.py                 # collect ALL actions in config.py
    python collect_data.py hello yes no    # collect specific signs only
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import ACTIONS, DATA_PATH, SEQUENCE_LENGTH, N_SEQUENCES
import cv2
import numpy as np
import mediapipe as mp


# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
DRAW_SPEC   = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)


def extract_keypoints(results):
    """Flatten all landmark groups into a 1662-element vector (matches frontend)."""
    pose = (np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    face = (np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(468 * 3))
    lh   = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3))
    rh   = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([pose, face, lh, rh])   # 132+1404+63+63 = 1662


def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(frame, results.pose_landmarks,       mp_holistic.POSE_CONNECTIONS,  DRAW_SPEC, DRAW_SPEC)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS,  DRAW_SPEC, DRAW_SPEC)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,  DRAW_SPEC, DRAW_SPEC)


def collect(targets: list[str]):
    # Create output dirs
    os.makedirs(DATA_PATH, exist_ok=True)
    for action in targets:
        os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    print(f"\nCollecting data for: {targets}")
    print("─" * 55)

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        for action in targets:
            action_path = os.path.join(DATA_PATH, action)
            existing    = len([f for f in os.listdir(action_path) if f.endswith('.npy')])

            print(f"\n  ACTION  : {action.upper()}")
            print(f"  Targets : {N_SEQUENCES} sequences  ({existing} already collected)")
            print(f"  Frames  : {SEQUENCE_LENGTH} per sequence")
            input(f"  ► Press ENTER to start recording '{action}'...")

            for seq in range(N_SEQUENCES):
                seq_path = os.path.join(action_path, f"{seq:04d}.npy")
                if os.path.exists(seq_path):
                    print(f"    Sequence {seq+1:02d}/{N_SEQUENCES} — already exists, skipping.")
                    continue

                window = []
                print(f"    Recording {seq+1:02d}/{N_SEQUENCES} ...", end=" ", flush=True)

                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        print("[WARN] Dropped frame")
                        window.append(np.zeros(1662))
                        continue

                    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb)
                    bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    draw_landmarks(bgr, results)

                    # HUD overlay
                    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 50), (0, 0, 0), -1)
                    cv2.putText(bgr, f"Action: {action.upper()}",
                                (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)
                    cv2.putText(bgr, f"Seq {seq+1}/{N_SEQUENCES}  Frame {frame_num+1}/{SEQUENCE_LENGTH}",
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Progress bar
                    progress = int((frame_num + 1) / SEQUENCE_LENGTH * bgr.shape[1])
                    cv2.rectangle(bgr, (0, bgr.shape[0] - 8), (progress, bgr.shape[0]), (0, 200, 100), -1)

                    cv2.imshow("SignFlow AI — Data Collection  (Q to quit)", bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[Quit by user]")
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit(0)

                    window.append(extract_keypoints(results))

                np.save(seq_path, np.array(window, dtype=np.float32))
                print("saved ✓")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Data collection complete for: {targets}")
    print("  Now run:  python train_model.py")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ACTIONS
    invalid = [a for a in targets if a not in ACTIONS]
    if invalid:
        print(f"[ERROR] Unknown actions: {invalid}")
        print(f"        Available in config.py: {ACTIONS}")
        sys.exit(1)
    collect(targets)
