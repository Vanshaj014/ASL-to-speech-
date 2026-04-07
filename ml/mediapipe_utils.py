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


def extract_static_keypoints(results, image_shape=None):
    """
    Extract landmarks for static sign classification (A-Z).
    Normalizes relative to the wrist and corrects for webcam aspect ratio.
    If the right hand is missing, mirrors the left hand so the model works for both!
    Returns a 1D array of shape (63,).
    """
    landmarks = None
    is_left = False

    # 1. Try Holistic right hand
    if getattr(results, "right_hand_landmarks", None):
        landmarks = results.right_hand_landmarks.landmark
    # 2. Try Holistic left hand (and we'll mirror it)
    elif getattr(results, "left_hand_landmarks", None):
        landmarks = results.left_hand_landmarks.landmark
        is_left = True
    # 3. Fallback for the Hands() model used in kaggle preprocessing
    elif getattr(results, "multi_hand_landmarks", None):
        landmarks = results.multi_hand_landmarks[0].landmark

    if landmarks:
        wrist = landmarks[0]  # Wrist is landmark 0
        
        # Correct for aspect ratio distortion (Kaggle was 1:1, webcam is 4:3)
        x_scale = 1.0
        y_scale = 1.0
        if image_shape is not None:
            h, w = image_shape[:2]
            x_scale = w / h  # e.g., 640/480 = ~1.33

        coords = []
        for lm in landmarks:
            # Calculate distance from wrist
            x_rel = (lm.x - wrist.x) * x_scale
            y_rel = (lm.y - wrist.y) * y_scale
            z_rel = (lm.z - wrist.z) * x_scale
            
            # Mirror left hand coordinates to look like a right hand to the model
            if is_left:
                x_rel = -x_rel 
                
            coords.append([x_rel, y_rel, z_rel])

        coords = np.array(coords)
        # Scale normalization: divide by the max absolute value
        scale = np.max(np.abs(coords)) + 1e-6
        coords = coords / scale
        return coords.flatten()
        
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
