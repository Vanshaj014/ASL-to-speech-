"""
collect_custom_data.py
-----------------------
Guided webcam-based data collector for dynamic ASL signs (common words).
Streams 30 frames per sequence and saves them as .npy keypoint arrays.

USAGE:
  python ml/collect_custom_data.py

CONTROLS:
  Q      - Quit
  SPACE  - Start recording a sequence
  S      - Skip current sign

OUTPUT:
  ml/data/raw/dynamic/<sign_name>/seq_0001.npy  (shape: 30, 258)
  ...
"""

import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mediapipe_utils import get_holistic_model, extract_keypoints
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic_sol = mp.solutions.holistic

# Configuration
DYNAMIC_SIGNS = [
    "hello", "thanks", "yes", "no", "please",
    "sorry", "iloveyou", "help", "good", "bad",
    "more", "stop", "eat", "drink", "where"
]
SEQUENCE_LENGTH = 30       # Frames per sequence
SEQUENCES_PER_SIGN = 30    # How many sequences to collect per sign
OUT_DIR = Path(__file__).parent / "data" / "raw" / "dynamic"

# Colours
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
DARK = (20, 20, 20)


def draw_landmarks(image, results):
    """Draw MediaPipe skeleton overlays."""
    # Draw pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic_sol.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2))
    # Draw right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic_sol.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2))
    # Draw left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic_sol.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2))


def overlay_text(image, text, pos, scale=1.0, color=WHITE, thickness=2):
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def collect_sign(sign_name, holistic):
    """Collect SEQUENCES_PER_SIGN sequences for one sign."""
    sign_dir = OUT_DIR / sign_name
    sign_dir.mkdir(parents=True, exist_ok=True)

    # Count existing sequences  
    existing = len(list(sign_dir.glob("*.npy")))
    start_seq = existing
    end_seq = existing + SEQUENCES_PER_SIGN

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print(f"\n{'='*50}")
    print(f"  Collecting: '{sign_name.upper()}'")
    print(f"  Sequences needed: {SEQUENCES_PER_SIGN}")
    print(f"  Already collected: {existing}")
    print(f"  Press SPACE to record, Q to quit this sign")
    print(f"{'='*50}")

    seq_num = start_seq

    while seq_num < end_seq:
        # Waiting screen
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw_landmarks(frame, results)

            # UI overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 100), DARK, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            overlay_text(frame, f"Sign: {sign_name.upper()}", (10, 35), 1.2, YELLOW, 2)
            overlay_text(frame, f"Sequence {seq_num - start_seq + 1}/{SEQUENCES_PER_SIGN}", (10, 70), 0.8, WHITE)
            overlay_text(frame, "SPACE = Record  |  Q = Next Sign", (10, 95), 0.6, (180, 180, 180))

            cv2.imshow("ASL Data Collector", frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                cap.release()
                return
            elif key == ord(' '):
                break

        # 3-second countdown before recording
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw_landmarks(frame, results)
            overlay_text(frame, f"Starting in {countdown}...", (200, 240), 2.0, GREEN, 3)
            cv2.imshow("ASL Data Collector", frame)
            cv2.waitKey(1000)

        # Record sequence
        sequence = []
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = holistic.process(img_rgb)
            img_rgb.flags.writeable = True

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            draw_landmarks(frame, results)

            # Progress bar
            progress = int((frame_num + 1) / SEQUENCE_LENGTH * 640)
            cv2.rectangle(frame, (0, 460), (progress, 480), GREEN, -1)
            overlay_text(frame, f"RECORDING... Frame {frame_num+1}/{SEQUENCE_LENGTH}", (10, 440), 0.8, RED, 2)
            cv2.imshow("ASL Data Collector", frame)
            cv2.waitKey(1)

        # Save sequence
        seq_path = sign_dir / f"seq_{seq_num:04d}.npy"
        np.save(str(seq_path), np.array(sequence))
        print(f"  ✓ Saved: {seq_path.name}  ({len(sequence)} frames)")

        seq_num += 1

        # Brief pause
        time.sleep(0.5)

    cap.release()
    print(f"\n[DONE] Finished collecting '{sign_name}'")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*50)
    print("  ASL Dynamic Sign Data Collector")
    print("  Signs:", ", ".join(DYNAMIC_SIGNS))
    print("="*50)
    print("\nPress ENTER to start, or type sign name to start from specific sign:")
    user_input = input(">> ").strip().lower()

    start_idx = 0
    if user_input in DYNAMIC_SIGNS:
        start_idx = DYNAMIC_SIGNS.index(user_input)
    elif user_input == "":
        start_idx = 0

    holistic = get_holistic_model(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    for sign in DYNAMIC_SIGNS[start_idx:]:
        collect_sign(sign, holistic)

    holistic.close()
    cv2.destroyAllWindows()
    print("\n[COMPLETE] All signs collected!")


if __name__ == "__main__":
    main()
