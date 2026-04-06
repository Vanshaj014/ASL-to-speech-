"""
evaluate.py
-----------
Generates evaluation metrics and plots for both trained models.
Run this after training to validate model performance.

USAGE:
  python ml/evaluate.py --model static   # Evaluate static MLP
  python ml/evaluate.py --model dynamic  # Evaluate dynamic LSTM
  python ml/evaluate.py --model all      # Evaluate both
"""

import argparse
import numpy as np
import sys
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, top_k_accuracy_score
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
DYNAMIC_DIR = BASE_DIR / "data" / "raw" / "dynamic"
MODEL_DIR = BASE_DIR / "models"

SEQUENCE_LENGTH = 30
FRAME_FEATURES = 258


# ── Static Model Evaluation ────────────────────────────────────────────────────

def evaluate_static():
    print("\n" + "="*60)
    print("  STATIC MODEL EVALUATION (A–Z MLP)")
    print("="*60)

    X_path = DATA_DIR / "X_static.npy"
    y_path = DATA_DIR / "y_static.npy"
    labels_path = MODEL_DIR / "label_map_static.npy"
    model_path = MODEL_DIR / "static_mlp.h5"

    for p in [X_path, y_path, labels_path, model_path]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            print("Run preprocess_kaggle.py and train_static.py first.")
            return

    X = np.load(X_path)
    y = np.load(y_path)
    label_map = np.load(labels_path, allow_pickle=True).item()
    model = tf.keras.models.load_model(str(model_path))
    class_names = [label_map[i] for i in range(len(label_map))]
    num_classes = len(class_names)

    # Use same 15% test split as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    y_onehot = tf.keras.utils.to_categorical(y_test, num_classes)
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    top3_acc = top_k_accuracy_score(y_test, probs, k=3)

    print(f"\n  Top-1 Accuracy : {acc:.4f}  ({acc:.1%})")
    print(f"  Top-3 Accuracy : {top3_acc:.4f}  ({top3_acc:.1%})")
    print(f"  Test samples   : {len(X_test)}")
    print(f"\n  Per-class report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    _plot_confusion_matrix(y_test, y_pred, class_names, "Static MLP (A-Z)",
                           MODEL_DIR / "eval_static_confusion.png", figsize=(16, 14))
    _plot_per_class_accuracy(y_test, y_pred, class_names, "Static MLP",
                              MODEL_DIR / "eval_static_per_class.png")


# ── Dynamic Model Evaluation ───────────────────────────────────────────────────

def evaluate_dynamic():
    print("\n" + "="*60)
    print("  DYNAMIC MODEL EVALUATION (Word LSTM)")
    print("="*60)

    labels_path = MODEL_DIR / "label_map_dynamic.npy"
    model_path = MODEL_DIR / "dynamic_lstm.h5"

    for p in [labels_path, model_path, DYNAMIC_DIR]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            print("Run collect_custom_data.py and train_dynamic.py first.")
            return

    label_map = np.load(labels_path, allow_pickle=True).item()
    model = tf.keras.models.load_model(str(model_path))
    class_names = [label_map[i] for i in range(len(label_map))]
    num_classes = len(class_names)
    class_to_idx = {v: k for k, v in label_map.items()}

    # Load all sequences
    X, y = [], []
    for sign_name in class_names:
        sign_dir = DYNAMIC_DIR / sign_name
        if not sign_dir.exists():
            continue
        for seq_file in sign_dir.glob("*.npy"):
            seq = np.load(str(seq_file))
            if len(seq) < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - len(seq), FRAME_FEATURES))
                seq = np.vstack([seq, pad])
            elif len(seq) > SEQUENCE_LENGTH:
                seq = seq[:SEQUENCE_LENGTH]
            X.append(seq)
            y.append(class_to_idx[sign_name])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy    : {acc:.4f}  ({acc:.1%})")
    print(f"  Test samples: {len(X_test)}")
    print(f"\n  Per-class report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    _plot_confusion_matrix(y_test, y_pred, class_names, "Dynamic LSTM (Words)",
                           MODEL_DIR / "eval_dynamic_confusion.png", figsize=(12, 10))
    _plot_per_class_accuracy(y_test, y_pred, class_names, "Dynamic LSTM",
                              MODEL_DIR / "eval_dynamic_per_class.png")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, class_names, title, save_path, figsize=(12, 10)):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, ax=ax)
    ax.set_title(f"Confusion Matrix — {title}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"  [SAVED] {save_path.name}")
    plt.show()


def _plot_per_class_accuracy(y_true, y_pred, class_names, title, save_path):
    accs = []
    for i, name in enumerate(class_names):
        mask = np.array(y_true) == i
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append(accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask]))

    colors = ["#34d399" if a >= 0.9 else "#fbbf24" if a >= 0.7 else "#f87171" for a in accs]

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.6), 5))
    bars = ax.bar(class_names, [a * 100 for a in accs], color=colors, edgecolor="none", width=0.6)
    ax.axhline(90, color="#34d399", linestyle="--", linewidth=1.2, alpha=0.7, label="90% threshold")
    ax.set_title(f"Per-Class Accuracy — {title}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Sign")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45 if len(class_names) > 10 else 0)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"  [SAVED] {save_path.name}")
    plt.show()


# ── CLI Entry ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ASL Sign Language models")
    parser.add_argument("--model", choices=["static", "dynamic", "all"], default="all",
                        help="Which model to evaluate")
    args = parser.parse_args()

    if args.model in ("static", "all"):
        evaluate_static()
    if args.model in ("dynamic", "all"):
        evaluate_dynamic()

    print("\n[DONE] Evaluation complete. Plots saved to ml/models/")


if __name__ == "__main__":
    main()
