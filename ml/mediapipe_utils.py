"""
mediapipe_utils.py
------------------
Shared utilities for MediaPipe landmark extraction and normalization.
Used by both the ML training pipeline (ml/) and the Django backend.
"""

import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

# Number of landmarks per type
NUM_HAND_LANDMARKS = 21      # 21 keypoints per hand
NUM_POSE_LANDMARKS = 33      # 33 body keypoints
NUM_FACE_LANDMARKS = 468     # 468 face keypoints

# Feature vector sizes (x, y, z per landmark)
HAND_FEATURE_SIZE = NUM_HAND_LANDMARKS * 3   # 63
POSE_FEATURE_SIZE = NUM_POSE_LANDMARKS * 4   # 132 (includes visibility)
FACE_FEATURE_SIZE = NUM_FACE_LANDMARKS * 3   # 1404

# For the translator we only use HANDS + POSE (skip face for speed/simplicity)
# Static model input: 63 (right hand) = 63 features
# Dynamic model input per frame: 63 (right) + 63 (left) + 132 (pose) = 258 features
STATIC_FEATURE_SIZE = HAND_FEATURE_SIZE           # 63
DYNAMIC_FRAME_SIZE = HAND_FEATURE_SIZE * 2 + POSE_FEATURE_SIZE  # 258


def extract_keypoints(results):
    """
    Extract and flatten MediaPipe Holistic landmarks into a numpy array.
    Returns a 1D array of shape (DYNAMIC_FRAME_SIZE,) = (258,).
    Falls back to zeros if a landmark type is not detected.
    """
    # Right hand (primary hand for ASL)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(HAND_FEATURE_SIZE)

    # Left hand (some signs involve both hands)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(HAND_FEATURE_SIZE)

    # Pose (upper body context)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(POSE_FEATURE_SIZE)

    return np.concatenate([rh, lh, pose])


def extract_static_keypoints(results):
    """
    Extract only right-hand landmarks for static sign classification (A-Z).
    Returns a 1D array of shape (63,).
    Normalizes relative to the wrist landmark.
    """
    if results.right_hand_landmarks:
        landmarks = results.right_hand_landmarks.landmark
        wrist = landmarks[0]  # Wrist is landmark 0
        # Normalize relative to wrist and scale by bounding box
        coords = np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
                           for lm in landmarks])
        # Scale normalization: divide by the max absolute value
        scale = np.max(np.abs(coords)) + 1e-6
        coords = coords / scale
        return coords.flatten()
    else:
        return np.zeros(STATIC_FEATURE_SIZE)


def normalize_keypoints(keypoints):
    """
    Normalize a full keypoint array (258,) for dynamic model input.
    Applies min-max normalization per feature.
    """
    min_val = keypoints.min()
    max_val = keypoints.max()
    if max_val - min_val > 1e-6:
        return (keypoints - min_val) / (max_val - min_val)
    return keypoints


def get_holistic_model(static_image_mode=False, min_detection_confidence=0.7,
                       min_tracking_confidence=0.5):
    """Return a configured MediaPipe Holistic instance."""
    return mp_holistic.Holistic(
        static_image_mode=static_image_mode,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
