"""
predictor.py
-------------
Singleton inference engine for the Django backend.
Loads both the static MLP (A-Z) and dynamic LSTM models once at startup.
Provides thread-safe predict() methods for both model types.

The WebSocket consumer calls predict_static() or predict_dynamic()
depending on the current mode selected by the user.
"""

import numpy as np
import threading
import logging
from pathlib import Path
from collections import deque

import tensorflow as tf

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"

# Prediction confidence thresholds
STATIC_THRESHOLD = 0.70       # Minimum raw model confidence to consider
DYNAMIC_THRESHOLD = 0.75      # Minimum raw model confidence for dynamic

# Majority voting: collect N frames and only emit the dominant prediction
SMOOTHING_WINDOW = 7          # Number of recent predictions to track
MIN_AGREEMENT = 3             # Minimum votes for the winner to be emitted
MIN_AGREEMENT_RATIO = 0.42    # Minimum fraction of window that must agree


class ASLPredictor:
    """
    Thread-safe singleton predictor that manages both static and dynamic models.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._model_lock = threading.Lock()

        # Static model (A-Z)
        self.static_model = None
        self.static_label_map = None

        # Dynamic model (words)
        self.dynamic_model = None
        self.dynamic_label_map = None

        # Prediction smoothing buffers
        self.static_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.dynamic_buffer = deque(maxlen=SMOOTHING_WINDOW)

        # Load models
        self._load_models()

    def _load_models(self):
        """Load TF models from the models directory, preferring .keras over .h5."""
        # Static model — prefer .keras (modern), fall back to .h5 (legacy)
        static_path = MODEL_DIR / "static_mlp_best.keras"
        if not static_path.exists():
            static_path = MODEL_DIR / "static_mlp.h5"  # legacy fallback
        static_labels = MODEL_DIR / "label_map_static.npy"

        if static_path.exists() and static_labels.exists():
            try:
                self.static_model = tf.keras.models.load_model(str(static_path))
                self.static_label_map = np.load(str(static_labels), allow_pickle=True).item()
                logger.info(f"[PREDICTOR] Static model loaded ({static_path.suffix}): {len(self.static_label_map)} classes")
            except Exception as e:
                logger.error(f"[PREDICTOR] Failed to load static model: {e}")
        else:
            logger.warning(f"[PREDICTOR] Static model not found at {static_path}. Train it first.")

        # Dynamic model — prefer .keras (modern), fall back to .h5 (legacy)
        dynamic_path = MODEL_DIR / "dynamic_cnn_lstm_best.keras"
        if not dynamic_path.exists():
            dynamic_path = MODEL_DIR / "dynamic_lstm.h5"  # legacy fallback
        dynamic_labels = MODEL_DIR / "label_map_dynamic.npy"

        if dynamic_path.exists() and dynamic_labels.exists():
            try:
                self.dynamic_model = tf.keras.models.load_model(str(dynamic_path))
                self.dynamic_label_map = np.load(str(dynamic_labels), allow_pickle=True).item()
                logger.info(f"[PREDICTOR] Dynamic model loaded ({dynamic_path.suffix}): {len(self.dynamic_label_map)} classes")
            except Exception as e:
                logger.error(f"[PREDICTOR] Failed to load dynamic model: {e}")
        else:
            logger.warning(f"[PREDICTOR] Dynamic model not found at {dynamic_path}. Train it first.")

    def is_ready(self):
        return self.static_model is not None or self.dynamic_model is not None

    def predict_static(self, keypoints: np.ndarray) -> dict:
        """
        Predict a static ASL letter from a 63-feature landmark vector.
        Uses majority voting across a sliding window to eliminate flickering.
        Returns: { 'sign': 'A', 'confidence': 0.97, 'top3': [...] }
        """
        if self.static_model is None:
            return {"sign": "?", "confidence": 0.0, "top3": [], "error": "Static model not loaded"}

        if keypoints.shape != (87,):
            return {"sign": "?", "confidence": 0.0, "top3": [], "error": f"Bad shape: {keypoints.shape}, expected (87,)"}

        # Check if hand is detected (non-zero keypoints)
        if np.all(keypoints == 0):
            self.static_buffer.clear()  # Reset voting buffer when hand disappears
            return {"sign": None, "confidence": 0.0, "top3": [], "no_hand": True}

        with self._model_lock:
            x = keypoints.reshape(1, -1)
            probs = self.static_model.predict(x, verbose=0)[0]

        top_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {"sign": self.static_label_map.get(i, "?"), "confidence": float(probs[i])}
            for i in top_idx
        ]

        best_idx = top_idx[0]
        raw_conf = float(probs[best_idx])
        raw_sign = self.static_label_map.get(best_idx, "?")

        # --- Gate 1: Raw confidence must clear the minimum threshold ---
        if raw_conf < STATIC_THRESHOLD:
            self.static_buffer.append(None)  # Track low-confidence frames too
            return {"sign": None, "confidence": raw_conf, "top3": top3, "below_threshold": True}

        # --- Gate 2: Majority voting across the sliding window ---
        self.static_buffer.append(raw_sign)

        # Count votes for each sign in the buffer
        buffer_list = [s for s in self.static_buffer if s is not None]
        if len(buffer_list) < MIN_AGREEMENT:
            return {"sign": None, "confidence": raw_conf, "top3": top3, "stabilizing": True}

        vote_counts = {}
        for s in buffer_list:
            vote_counts[s] = vote_counts.get(s, 0) + 1

        winner = max(vote_counts, key=vote_counts.get)
        winner_votes = vote_counts[winner]
        agreement_ratio = winner_votes / len(self.static_buffer)

        # Only emit if the winner has enough votes AND enough ratio
        if winner_votes < MIN_AGREEMENT or agreement_ratio < MIN_AGREEMENT_RATIO:
            return {"sign": None, "confidence": raw_conf, "top3": top3, "below_threshold": True}

        # Blend raw model confidence with voting agreement for a robust score
        blended_conf = (raw_conf * 0.6) + (agreement_ratio * 0.4)

        return {
            "sign": winner,
            "confidence": round(blended_conf, 4),
            "raw_confidence": round(raw_conf, 4),
            "agreement": round(agreement_ratio, 4),
            "top3": top3,
            "model": "static"
        }

    def predict_dynamic(self, sequence: np.ndarray) -> dict:
        """
        Predict a dynamic ASL word from a (30, 258) sequence of frames.
        Returns: { 'sign': 'hello', 'confidence': 0.92, 'top3': [...] }
        """
        if self.dynamic_model is None:
            return {"sign": "?", "confidence": 0.0, "top3": [], "error": "Dynamic model not loaded"}

        if sequence.shape != (30, 258):
            return {"sign": "?", "confidence": 0.0, "top3": [],
                    "error": f"Bad shape: {sequence.shape}, expected (30, 258)"}

        with self._model_lock:
            x = sequence.reshape(1, 30, 258)
            probs = self.dynamic_model.predict(x, verbose=0)[0]

        top_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {"sign": self.dynamic_label_map.get(i, "?"), "confidence": float(probs[i])}
            for i in top_idx
        ]

        best_idx = top_idx[0]
        best_conf = float(probs[best_idx])
        best_sign = self.dynamic_label_map.get(best_idx, "?")

        if best_conf < DYNAMIC_THRESHOLD:
            return {"sign": None, "confidence": best_conf, "top3": top3, "below_threshold": True}

        return {
            "sign": best_sign,
            "confidence": best_conf,
            "top3": top3,
            "model": "dynamic"
        }

    def get_vocabulary(self) -> dict:
        """Return all supported signs by model type."""
        return {
            "static": list(self.static_label_map.values()) if self.static_label_map else [],
            "dynamic": list(self.dynamic_label_map.values()) if self.dynamic_label_map else [],
        }


# Global singleton instance
predictor = ASLPredictor()
