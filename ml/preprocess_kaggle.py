"""
preprocess_kaggle.py
---------------------
Processes the Kaggle ASL Alphabet dataset into MediaPipe landmark keypoints.

Dataset: "grassknoted/asl-alphabet" on Kaggle
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
  - 87,000 images, 200x200px
  - 29 classes: A-Z + del, nothing, space

USAGE:
  1. Download and unzip the dataset into: ml/data/raw/kaggle_asl/
     Structure: ml/data/raw/kaggle_asl/asl_alphabet_train/A/A1.jpg ...
  2. Run: python ml/preprocess_kaggle.py

OUTPUT:
  ml/data/processed/X_static.npy  — shape (N, 63)
  ml/data/processed/y_static.npy  — shape (N,) integer labels
  ml/data/processed/label_map.npy — dict mapping index -> letter
"""

import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

# Add ml/ dir to path so we can import mediapipe_utils
sys.path.insert(0, str(Path(__file__).parent))
from mediapipe_utils import get_holistic_model, extract_static_keypoints

# Paths
RAW_DIR = Path(__file__).parent / "data" / "raw" / "kaggle_asl" / "asl_alphabet_train"
OUT_DIR = Path(__file__).parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Only include A-Z (skip del, nothing, space for initial training)
VALID_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
MAX_IMAGES_PER_CLASS = 500   # Limit for speed; increase for better accuracy


def process_dataset():
    if not RAW_DIR.exists():
        print(f"[ERROR] Dataset not found at: {RAW_DIR}")
        print("Please download from Kaggle: grassknoted/asl-alphabet")
        print("and unzip to ml/data/raw/kaggle_asl/")
        sys.exit(1)

    print(f"[INFO] Processing dataset from: {RAW_DIR}")
    print(f"[INFO] Classes: {VALID_CLASSES}")

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.5
    )

    X, y = [], []
    label_map = {i: letter for i, letter in enumerate(VALID_CLASSES)}
    class_to_idx = {letter: i for i, letter in enumerate(VALID_CLASSES)}

    for letter in VALID_CLASSES:
        class_dir = RAW_DIR / letter
        if not class_dir.exists():
            print(f"  [WARN] Missing class dir: {class_dir}")
            continue

        image_files = list(class_dir.glob("*.jpg"))[:MAX_IMAGES_PER_CLASS]
        processed = 0
        skipped = 0

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            # Convert BGR to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(img_rgb)

            # Check if any hands were detected
            if not results.multi_hand_landmarks:
                skipped += 1
                continue

            # Extract the first hand
            landmarks = results.multi_hand_landmarks[0].landmark
            wrist = landmarks[0]
            
            # Normalize relative to wrist and scale
            coords = np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in landmarks])
            scale = np.max(np.abs(coords)) + 1e-6
            keypoints = (coords / scale).flatten()

            X.append(keypoints)
            y.append(class_to_idx[letter])
            processed += 1

        print(f"  [{letter}] Processed: {processed}, Skipped (no hand): {skipped}")

    mp_hands.close()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    np.save(OUT_DIR / "X_static.npy", X)
    np.save(OUT_DIR / "y_static.npy", y)
    np.save(OUT_DIR / "label_map.npy", label_map)

    print(f"\n[DONE] Saved {len(X)} samples → {OUT_DIR}")
    print(f"  X_static.npy: {X.shape}")
    print(f"  y_static.npy: {y.shape}")
    print(f"  Classes: {len(label_map)}")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Class distribution:")
    for idx, count in zip(unique, counts):
        print(f"    {label_map[idx]}: {count} samples")


if __name__ == "__main__":
    process_dataset()
