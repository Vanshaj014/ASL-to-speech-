"""
train_dynamic.py
-----------------
Trains a hybrid 1D-CNN + LSTM classifier for dynamic ASL signs (common words).
The CNN layers extract local motion patterns (velocity spikes, finger flicks)
across short frame windows, while the LSTM interprets their temporal sequence.

Input: sequences of shape (30, 258) — 30 frames × 258 keypoint features
Output: N classes (dynamic signs collected via collect_custom_data.py)

USAGE:
  python ml/train_dynamic.py

Prerequisites:
  - Run collect_custom_data.py first to gather dynamic sign sequences
"""

import numpy as np
import os
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Paths
BASE_DIR = Path(__file__).parent
DYNAMIC_DIR = BASE_DIR / "data" / "raw" / "dynamic"
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BACKEND_MODEL_DIR = BASE_DIR.parent / "backend" / "translator" / "ml" / "models"
BACKEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30
FRAME_FEATURES = 258  # DYNAMIC_FRAME_SIZE from mediapipe_utils


def load_dynamic_data():
    """Load all collected .npy sequences from ml/data/raw/dynamic/"""
    if not DYNAMIC_DIR.exists():
        print(f"[ERROR] Dynamic data directory not found: {DYNAMIC_DIR}")
        print("Run collect_custom_data.py first.")
        sys.exit(1)

    sign_dirs = [d for d in DYNAMIC_DIR.iterdir() if d.is_dir()]
    if not sign_dirs:
        print("[ERROR] No sign directories found in", DYNAMIC_DIR)
        sys.exit(1)

    classes = sorted([d.name for d in sign_dirs])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    label_map = {i: c for i, c in enumerate(classes)}

    print(f"[INFO] Found {len(classes)} dynamic signs: {classes}")

    X, y = [], []
    for sign_name in classes:
        sign_dir = DYNAMIC_DIR / sign_name
        seq_files = list(sign_dir.glob("*.npy"))
        for seq_file in seq_files:
            seq = np.load(str(seq_file))
            # Pad or truncate to SEQUENCE_LENGTH
            if len(seq) < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - len(seq), FRAME_FEATURES))
                seq = np.vstack([seq, pad])
            elif len(seq) > SEQUENCE_LENGTH:
                seq = seq[:SEQUENCE_LENGTH]
            X.append(seq)
            y.append(class_to_idx[sign_name])
        print(f"  [{sign_name}]: {len(seq_files)} sequences")

    X = np.array(X, dtype=np.float32)  # (N, 30, 258)
    y = np.array(y, dtype=np.int32)    # (N,)

    print(f"\n[INFO] Total sequences: {len(X)}, Shape: {X.shape}")
    return X, y, label_map


def augment_sequences(X, y):
    """
    Temporal data augmentation for dynamic gesture sequences.
    Generates synthetic variations to improve generalization:
      1. Speed jitter:  randomly stretch/compress time
      2. Gaussian noise: simulate sensor noise on landmarks
      3. Time shift:    shift the gesture forward/backward in the window
    """
    print("[INFO] Augmenting dynamic sequences...")
    X_aug, y_aug = list(X), list(y)

    for i in range(len(X)):
        seq = X[i]  # (30, 258)

        # 1. Speed jitter — resample the sequence at a slightly different rate
        speed_factor = np.random.uniform(0.85, 1.15)
        new_len = int(SEQUENCE_LENGTH * speed_factor)
        indices = np.linspace(0, SEQUENCE_LENGTH - 1, new_len).astype(int)
        stretched = seq[indices]
        # Pad or truncate back to SEQUENCE_LENGTH
        if len(stretched) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(stretched), FRAME_FEATURES))
            stretched = np.vstack([stretched, pad])
        else:
            stretched = stretched[:SEQUENCE_LENGTH]
        X_aug.append(stretched)
        y_aug.append(y[i])

        # 2. Gaussian noise on landmark coordinates
        noise = np.random.normal(0, 0.01, size=seq.shape)
        X_aug.append(seq + noise)
        y_aug.append(y[i])

        # 3. Time shift — shift gesture window by a few frames
        shift = np.random.randint(-3, 4)  # -3 to +3 frames
        shifted = np.roll(seq, shift, axis=0)
        if shift > 0:
            shifted[:shift] = 0  # Zero out wrapped frames
        elif shift < 0:
            shifted[shift:] = 0
        X_aug.append(shifted)
        y_aug.append(y[i])

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int32)
    print(f"[INFO] Augmented: {len(X)} -> {len(X_aug)} sequences (4x)")
    return X_aug, y_aug


def build_cnn_lstm_model(num_classes: int, sequence_length: int = 30,
                         frame_features: int = 258) -> keras.Model:
    """
    Hybrid 1D-CNN + LSTM architecture for dynamic gesture recognition.

    Pipeline:
      Input (30, 258)
        -> Conv1D(64, kernel=3)  -- detect local motion patterns across 3-frame windows
        -> BatchNorm -> ReLU -> MaxPool(2) -- compress to (15, 64)
        -> Conv1D(128, kernel=3) -- detect higher-level motion features
        -> BatchNorm -> ReLU -> MaxPool(2) -- compress to (7, 128)
        -> LSTM(128)             -- understand temporal sequence meaning
        -> Dropout(0.4)
        -> Dense(64) -> ReLU
        -> Dropout(0.3)
        -> Dense(num_classes) -> Softmax
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, frame_features)),

        # --- 1D CNN Block 1: Local motion pattern extraction ---
        layers.Conv1D(64, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool1D(pool_size=2),        # (30, 258) -> (15, 64)

        # --- 1D CNN Block 2: Higher-level motion features ---
        layers.Conv1D(128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool1D(pool_size=2),        # (15, 64) -> (7, 128)

        # --- LSTM: Temporal sequence understanding ---
        layers.LSTM(128, return_sequences=False, activation='tanh'),
        layers.Dropout(0.4),

        # --- Classification head ---
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name="asl_dynamic_cnn_lstm")

    return model


def train():
    X, y, label_map = load_dynamic_data()
    num_classes = len(label_map)

    # Temporal data augmentation
    X, y = augment_sequences(X, y)

    # One-hot encode
    y_onehot = keras.utils.to_categorical(y, num_classes)

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_onehot, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    model = build_cnn_lstm_model(num_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "dynamic_cnn_lstm_best.keras"),
            save_best_only=True, monitor='val_accuracy', verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    print("\n[TRAIN] Starting CNN-LSTM training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[EVAL] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Save models (keep filename as dynamic_lstm.h5 for backend compatibility)
    model.save(str(MODEL_DIR / "dynamic_lstm.h5"))
    model.save(str(BACKEND_MODEL_DIR / "dynamic_lstm.h5"))
    np.save(str(MODEL_DIR / "label_map_dynamic.npy"), label_map)
    np.save(str(BACKEND_MODEL_DIR / "label_map_dynamic.npy"), label_map)
    print(f"[SAVED] Models saved")

    # Evaluation plots
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    class_names = [label_map[i] for i in range(num_classes)]
    print("\n[REPORT]")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    plot_training(history, test_acc)
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)


def plot_training(history, test_acc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Dynamic CNN-LSTM Training (Test Acc: {test_acc:.2%})", fontsize=14, fontweight='bold')

    axes[0].plot(history.history['accuracy'], label='Train Acc', color='steelblue')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc', color='darkorange')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train Loss', color='steelblue')
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='darkorange')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(MODEL_DIR / "dynamic_training_curves.png"), dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix -- Dynamic ASL Signs (CNN-LSTM)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(str(MODEL_DIR / "dynamic_confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    train()
