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
STATIC_FEATURE_SIZE = HAND_FEATURE_SIZE           # 63 — raw landmarks only
DYNAMIC_FRAME_SIZE = HAND_FEATURE_SIZE * 2 + POSE_FEATURE_SIZE  # 258

# Enhanced static features: 63 raw + 15 joint angles + 5 extension states + 4 spread = 87
STATIC_ENHANCED_FEATURE_SIZE = STATIC_FEATURE_SIZE + 15 + 5 + 4  # 87

# MediaPipe hand landmark indices for each finger segment
# Each finger: [MCP, PIP, DIP, TIP]
FINGER_INDICES = [
    [1, 2, 3, 4],    # Thumb
    [5, 6, 7, 8],    # Index
    [9, 10, 11, 12], # Middle
    [13, 14, 15, 16],# Ring
    [17, 18, 19, 20] # Pinky
]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # Tips of all 5 fingers


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


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the angle in radians between two 3D vectors.
    Returns a value in [0, pi].
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def extract_static_keypoints_enhanced(results, image_shape=None) -> np.ndarray:
    """
    Enhanced feature extraction for static sign classification.
    Returns a 1D array of shape (87,) consisting of:
      - 63: Raw wrist-relative, normalized landmark coordinates (x, y, z)
      - 15: Joint bend angles at MCP, PIP, DIP for all 5 fingers (in radians)
      - 5:  Finger extension states (1 = extended, 0 = curled) per finger
      - 4:  Adjacent fingertip spread distances (index-middle, middle-ring, etc.)

    These derived geometry features are scale-invariant and position-invariant,
    making signs like A, S, and E much easier for the model to discriminate.
    """
    landmarks = None
    is_left = False

    if getattr(results, "right_hand_landmarks", None):
        landmarks = results.right_hand_landmarks.landmark
    elif getattr(results, "left_hand_landmarks", None):
        landmarks = results.left_hand_landmarks.landmark
        is_left = True
    elif getattr(results, "multi_hand_landmarks", None):
        landmarks = results.multi_hand_landmarks[0].landmark

    if landmarks is None:
        return np.zeros(STATIC_ENHANCED_FEATURE_SIZE)

    # --- Build coordinate array ---
    wrist = landmarks[0]
    x_scale = 1.0
    if image_shape is not None:
        h, w = image_shape[:2]
        x_scale = w / h

    coords = []
    for lm in landmarks:
        x_rel = (lm.x - wrist.x) * x_scale
        y_rel = (lm.y - wrist.y)
        z_rel = (lm.z - wrist.z) * x_scale
        if is_left:
            x_rel = -x_rel
        coords.append([x_rel, y_rel, z_rel])

    coords = np.array(coords, dtype=np.float32)  # (21, 3)
    scale = np.max(np.abs(coords)) + 1e-6
    coords_norm = coords / scale  # Normalized raw features (base 63)

    # --- Feature 1: Joint bend angles (15 features) ---
    # For each of the 5 fingers, compute the bend angle at MCP, PIP, DIP joints
    # angle at PIP = angle between vector(MCP->PIP) and vector(PIP->DIP)
    joint_angles = []
    for finger in FINGER_INDICES:
        mcp, pip, dip, tip = finger
        # MCP angle: wrist -> MCP -> PIP
        v1 = coords_norm[mcp] - coords_norm[0]   # wrist to MCP
        v2 = coords_norm[pip] - coords_norm[mcp]  # MCP to PIP
        joint_angles.append(_angle_between(v1, v2))
        # PIP angle: MCP -> PIP -> DIP
        v1 = coords_norm[mcp] - coords_norm[pip]
        v2 = coords_norm[dip] - coords_norm[pip]
        joint_angles.append(_angle_between(v1, v2))
        # DIP angle: PIP -> DIP -> TIP
        v1 = coords_norm[pip] - coords_norm[dip]
        v2 = coords_norm[tip] - coords_norm[dip]
        joint_angles.append(_angle_between(v1, v2))

    joint_angles = np.array(joint_angles, dtype=np.float32)  # (15,)

    # --- Feature 2: Finger extension states (5 features) ---
    # A finger is "extended" if its tip is farther from the wrist than its PIP joint.
    # Measured using Euclidean distance from the wrist landmark.
    extension_states = []
    for finger in FINGER_INDICES:
        mcp, pip, dip, tip = finger
        dist_tip = np.linalg.norm(coords_norm[tip])
        dist_pip = np.linalg.norm(coords_norm[pip])
        extension_states.append(1.0 if dist_tip > dist_pip else 0.0)

    extension_states = np.array(extension_states, dtype=np.float32)  # (5,)

    # --- Feature 3: Adjacent fingertip spread distances (4 features) ---
    # Captures how spread apart adjacent fingertips are (important for signs like V, 3, etc.)
    spread_distances = []
    for i in range(len(FINGERTIP_INDICES) - 1):
        t1 = FINGERTIP_INDICES[i]
        t2 = FINGERTIP_INDICES[i + 1]
        dist = np.linalg.norm(coords_norm[t1] - coords_norm[t2])
        spread_distances.append(dist)

    spread_distances = np.array(spread_distances, dtype=np.float32)  # (4,)

    # Concatenate all features: 63 + 15 + 5 + 4 = 87
    return np.concatenate([coords_norm.flatten(), joint_angles, extension_states, spread_distances])


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


def serialize_hand_landmarks(results):
    """
    Extract the visible hand's landmarks as a compact list of {x, y} dicts
    for transmission to the frontend overlay canvas.

    Returns a list of 21 points, each with normalized [0-1] coordinates,
    or None if no hand is detected.

    Note: x coordinates are in the un-mirrored frame space. The frontend
    must mirror them (x = 1 - x) before drawing, because the <video>
    element is rendered with transform: scaleX(-1).
    """
    lm_list = None
    if getattr(results, "right_hand_landmarks", None):
        lm_list = results.right_hand_landmarks.landmark
    elif getattr(results, "left_hand_landmarks", None):
        lm_list = results.left_hand_landmarks.landmark

    if lm_list is None:
        return None

    return [{"x": round(lm.x, 4), "y": round(lm.y, 4)} for lm in lm_list]
