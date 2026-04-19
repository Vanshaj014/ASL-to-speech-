"""
train_static.py
---------------
Trains an MLP classifier for static ASL signs (A-Z fingerspelling).
Input: 87 features (63 raw landmarks + 15 joint angles + 5 extension states + 4 spread distances)
Output: 26 classes (A-Z)

USAGE:
  python ml/train_static.py

Prerequisites:
  - Run preprocess_kaggle.py first to generate X_static.npy and y_static.npy
"""

import numpy as np
import os
import sys
from pathlib import Path

# Ensure ml/ is on the path so mediapipe_utils is importable
sys.path.insert(0, str(Path(__file__).parent))
from mediapipe_utils import FINGER_INDICES, FINGERTIP_INDICES, _angle_between

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BACKEND_MODEL_DIR = BASE_DIR.parent / "backend" / "translator" / "ml" / "models"
BACKEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X = np.load(DATA_DIR / "X_static.npy")
    y = np.load(DATA_DIR / "y_static.npy")
    label_map = np.load(DATA_DIR / "label_map.npy", allow_pickle=True).item()
    print(f"[INFO] Loaded {len(X)} samples, {len(label_map)} classes")
    print(f"[INFO] Feature shape: {X.shape}")
    return X, y, label_map


def build_model(num_classes: int, input_size: int = 87) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ], name="asl_static_mlp")
    return model


def augment_landmarks(X, y):
    """
    Apply geometric noise (rotation and jitter) to the raw coordinate portion of the
    87-feature vector to simulate different webcam angles and distances.
    The derived features (angles, extension, spread) are recomputed from augmented coords.
    Multiplies training data size 3x.
    """
    print("[INFO] Augmenting data to improve real-world robustness...")
    X_aug = []
    y_aug = []

    for i in range(len(X)):
        # Original sample
        X_aug.append(X[i])
        y_aug.append(y[i])

        # Extract raw 63-feature coordinate portion only
        pts = X[i][:63].reshape(21, 3)

        for pts_aug in _generate_augmented_pts(pts):
            derived = _compute_derived_features(pts_aug)
            X_aug.append(np.concatenate([pts_aug.flatten(), derived]))
            y_aug.append(y[i])

    print(f"[INFO] Augmented data: {len(X)} -> {len(X_aug)} samples")
    return np.array(X_aug), np.array(y_aug)


def _generate_augmented_pts(pts):
    """Generate augmented versions of a (21, 3) landmark array."""
    # 1. Jitter + Scale
    scale = np.random.uniform(0.9, 1.1)
    noise = np.random.normal(0, 0.015, size=pts.shape)
    yield (pts * scale) + noise

    # 2. Z-axis rotation
    theta = np.radians(np.random.uniform(-20, 20))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    yield np.dot(pts, rot.T)


def _compute_derived_features(coords_norm):
    """Compute the 24 derived features (angles + extension + spread) from (21, 3) coords."""
    joint_angles = []
    for finger in FINGER_INDICES:
        mcp, pip, dip, tip = finger
        v1 = coords_norm[mcp] - coords_norm[0]
        v2 = coords_norm[pip] - coords_norm[mcp]
        joint_angles.append(_angle_between(v1, v2))
        v1 = coords_norm[mcp] - coords_norm[pip]
        v2 = coords_norm[dip] - coords_norm[pip]
        joint_angles.append(_angle_between(v1, v2))
        v1 = coords_norm[pip] - coords_norm[dip]
        v2 = coords_norm[tip] - coords_norm[dip]
        joint_angles.append(_angle_between(v1, v2))

    extension_states = []
    for finger in FINGER_INDICES:
        mcp, pip, dip, tip = finger
        extension_states.append(1.0 if np.linalg.norm(coords_norm[tip]) > np.linalg.norm(coords_norm[pip]) else 0.0)

    spread_distances = []
    for i in range(len(FINGERTIP_INDICES) - 1):
        t1, t2 = FINGERTIP_INDICES[i], FINGERTIP_INDICES[i + 1]
        spread_distances.append(float(np.linalg.norm(coords_norm[t1] - coords_norm[t2])))

    return np.array(joint_angles + extension_states + spread_distances, dtype=np.float32)


def train():
    # Load data
    X, y, label_map = load_data()
    num_classes = len(label_map)

    # Augment mathematically to increase generic real-world accuracy
    X, y = augment_landmarks(X, y)

    # One-hot encode labels
    lb = LabelBinarizer()
    y_onehot = lb.fit_transform(y)

    # Train/val/test split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_onehot, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Compute class weights to fix M/N imbalance
    # Underrepresented letters (like M=282, N=166) will have higher weights
    # so the model is penalised more heavily for getting them wrong.
    y_train_int = np.argmax(y_train, axis=1)
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=y_train_int
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}
    print(f"[INFO] Class weights computed (higher = more penalised):")
    for i, w in class_weight_dict.items():
        if w > 1.3:
            print(f"  {label_map[i]}: {w:.2f}  <-- underrepresented")

    # Build model
    model = build_model(num_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "static_mlp_best.keras"),
            save_best_only=True, monitor='val_accuracy', verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.TensorBoard(log_dir=str(BASE_DIR / "logs" / "static"), histogram_freq=1)
    ]

    # Train
    print("\n[TRAIN] Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Compensate for M/N imbalance
        verbose=1
    )

    # Evaluate on test set
    print("\n[EVAL] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss:     {test_loss:.4f}")

    # Save final model
    model.save(str(MODEL_DIR / "static_mlp.h5"))
    model.save(str(BACKEND_MODEL_DIR / "static_mlp.h5"))  # Copy to backend
    print(f"[SAVED] Models saved to {MODEL_DIR} and {BACKEND_MODEL_DIR}")

    # Save label map alongside model for backend use
    np.save(str(BACKEND_MODEL_DIR / "label_map_static.npy"), label_map)
    np.save(str(MODEL_DIR / "label_map_static.npy"), label_map)

    # Confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\n[REPORT]")
    class_names = [label_map[i] for i in range(num_classes)]
    print(classification_report(y_true_classes, y_pred_classes, labels=np.arange(num_classes), target_names=class_names, zero_division=0))

    plot_training(history, test_acc)
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)


def plot_training(history, test_acc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Static MLP Training (Test Acc: {test_acc:.2%})", fontsize=14, fontweight='bold')

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
    plot_path = MODEL_DIR / "static_training_curves.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    print(f"[PLOT] Training curves saved: {plot_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix — Static ASL Signs (A-Z)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plot_path = MODEL_DIR / "static_confusion_matrix.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    print(f"[PLOT] Confusion matrix saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    train()
