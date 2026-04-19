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
  ml/data/processed/X_static.npy  — shape (N, 87)  [enhanced features]
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
from mediapipe_utils import get_holistic_model, extract_static_keypoints_enhanced

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

    # Use a very low detection confidence for preprocessing:
    # M and N signs look like closed fists with no finger extension,
    # making them genuinely hard for MediaPipe to detect at high confidence thresholds.
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3  # Lowered from 0.5 to capture M, N, S, A shapes
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

            # Attempt 1: Standard detection at original size
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(img_rgb)

            # Attempt 2: Multi-scale retry (M, N need bigger context to detect)
            if not results.multi_hand_landmarks:
                for scale in [1.5, 2.0, 0.75]:
                    h, w = img.shape[:2]
                    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
                    img_rgb_scaled = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    results = mp_hands.process(img_rgb_scaled)
                    if results.multi_hand_landmarks:
                        img = resized  # Use the resized image for feature extraction
                        img_rgb = img_rgb_scaled
                        break

            # Attempt 3: Brightness boost (some M/N images are slightly darker)
            if not results.multi_hand_landmarks:
                bright = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
                img_rgb_bright = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)
                results = mp_hands.process(img_rgb_bright)
                if results.multi_hand_landmarks:
                    img = bright
                    img_rgb = img_rgb_bright

            # Skip if hand still not detected after all attempts
            if not results.multi_hand_landmarks:
                skipped += 1
                continue

            # Extract enhanced 87-feature vector (joints, angles, extension, spread)
            keypoints = extract_static_keypoints_enhanced(results, image_shape=img.shape)
            if np.all(keypoints == 0):
                skipped += 1
                continue

            X.append(keypoints)
            y.append(class_to_idx[letter])
            processed += 1

        if skipped > 100:
            print(f"  [{letter}] Processed: {processed}, Skipped (no hand): {skipped}  ⚠️  High skip rate")
        else:
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
