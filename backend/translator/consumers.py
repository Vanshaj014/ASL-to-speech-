"""
consumers.py
------------
Django Channels WebSocket consumer for real-time ASL translation.

Each browser client connects to ws://<host>/ws/translate/
The consumer:
  1. Receives a video frame (base64 JPEG) from the React frontend
  2. Runs it through MediaPipe to extract landmarks
  3. Feeds landmarks to the appropriate model (static/dynamic)
  4. Sends the prediction result back to the same client

Architecture (deployment-friendly):
  - Webcam is captured in the BROWSER via getUserMedia
  - Frames are sent as base64 strings over WebSocket
  - No server-side camera access required → works on any cloud server
"""

import json
import base64
import logging
import time
import numpy as np
import cv2
import asyncio
from collections import deque

from channels.generic.websocket import AsyncWebsocketConsumer

from .ml.mediapipe_utils import get_holistic_model, extract_static_keypoints_enhanced, extract_keypoints, serialize_hand_landmarks

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 60       # Target frames for dynamic model (must match training)
FRAME_FEATURES = 258       # Feature size per frame
MAX_FRAME_BYTES = 500_000  # 500 KB max per frame — DoS protection
ALLOWED_MODES = {"static", "dynamic"}  # Whitelist for set_mode
CAPTURE_WINDOW_SEC = 2.0   # Time window matching training data (~2s at 30 FPS)
MIN_FRAMES_FOR_PREDICT = 12  # Minimum frames needed in the window before predicting


class TranslatorConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer: receives frames from React frontend, returns predictions.
    """

    async def connect(self):
        self.client_id = self.scope.get("client", ("unknown", 0))[0]
        self.mode = "static"  # 'static' or 'dynamic'
        # Time-based frame buffer: list of (timestamp, keypoints) tuples
        # Instead of counting frames, we track time so the 2-second window
        # always matches training data regardless of actual browser FPS.
        self.frame_buffer = []
        self.holistic = None
        self._init_mediapipe()
        await self.accept()
        logger.info(f"[WS] Client connected")
        await self.send(json.dumps({
            "type": "connected",
            "message": "ASL Translator WebSocket connected",
            "mode": self.mode
        }))

    def _init_mediapipe(self):
        """Initialize MediaPipe Holistic (runs in sync context).
        
        IMPORTANT: These settings MUST match collect_custom_data.py exactly.
        Any mismatch in model_complexity or confidence thresholds will produce
        different landmark positions, causing the trained model to fail.
        """
        try:
            self.holistic = get_holistic_model(
                static_image_mode=False,
                model_complexity=1,            # Must match training (1=Full)
                min_detection_confidence=0.7,   # Must match training
                min_tracking_confidence=0.5     # Must match training
            )
        except Exception as e:
            logger.error(f"[WS] MediaPipe init failed: {e}")

    async def disconnect(self, close_code):
        if self.holistic:
            self.holistic.close()
        logger.info(f"[WS] Client disconnected (code: {close_code})")

    async def receive(self, text_data):
        """Handle incoming message from React frontend."""
        try:
            data = json.loads(text_data)
            msg_type = data.get("type")

            if msg_type == "frame":
                await self._process_frame(data)

            elif msg_type == "set_mode":
                requested = data.get("mode", "static")
                if requested not in ALLOWED_MODES:
                    await self.send(json.dumps({
                        "type": "error",
                        "message": f"Invalid mode. Allowed: {sorted(ALLOWED_MODES)}"
                    }))
                    return
                self.mode = requested
                self.frame_buffer = []  # Clear on mode switch
                await self.send(json.dumps({
                    "type": "mode_changed",
                    "mode": self.mode
                }))

            elif msg_type == "ping":
                await self.send(json.dumps({"type": "pong"}))

        except json.JSONDecodeError:
            await self.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
        except Exception as e:
            logger.error(f"[WS] Error processing message: {e}")
            # Never leak internal error details to the client
            await self.send(json.dumps({"type": "error", "message": "Internal server error"}))

    async def _process_frame(self, data):
        """Decode frame, extract landmarks, run inference, send result."""
        # Decode base64 JPEG frame from frontend
        frame_b64 = data.get("frame", "")
        if not frame_b64:
            return

        # DoS protection: reject oversized frames before any processing
        if len(frame_b64) > MAX_FRAME_BYTES:
            logger.warning("[WS] Oversized frame rejected")
            return

        # Remove data URL prefix if present
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",")[1]

        # Run CPU-bound decoding and MediaPipe processing in thread pool
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._run_mediapipe_and_predict, frame_b64
        )

        if result:
            await self.send(json.dumps(result))

    def _run_mediapipe_and_predict(self, frame_b64):
        """Synchronous (Worker Thread): Decode base64, extract landmarks, model inference."""
        if self.holistic is None:
            return None

        # 1. Decode base64 to image (moved here to free the ASGI event loop)
        try:
            img_bytes = base64.b64decode(frame_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return None
        except Exception as e:
            logger.error(f"[WS] worker thread decode error: {e}")
            return None

        from .ml.predictor import predictor

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.holistic.process(img_rgb)
        img_rgb.flags.writeable = True

        has_hand = results.right_hand_landmarks is not None or results.left_hand_landmarks is not None

        if self.mode == "static":
            keypoints = extract_static_keypoints_enhanced(results, image_shape=frame.shape)
            prediction = predictor.predict_static(keypoints)
            landmarks = serialize_hand_landmarks(results)
            return {
                "type": "prediction",
                "mode": "static",
                "has_hand": has_hand,
                "landmarks": landmarks,
                **prediction
            }

        elif self.mode == "dynamic":
            frame_kp = extract_keypoints(results)
            now = time.time()
            self.frame_buffer.append((now, frame_kp))
            landmarks = serialize_hand_landmarks(results)

            # Evict frames older than the capture window
            cutoff = now - CAPTURE_WINDOW_SEC
            self.frame_buffer = [(t, kp) for t, kp in self.frame_buffer if t >= cutoff]

            buffer_duration = self.frame_buffer[-1][0] - self.frame_buffer[0][0] if len(self.frame_buffer) > 1 else 0
            have_enough = len(self.frame_buffer) >= MIN_FRAMES_FOR_PREDICT and buffer_duration >= (CAPTURE_WINDOW_SEC * 0.8)

            if have_enough:
                # Resample to exactly SEQUENCE_LENGTH frames via linear interpolation.
                # This normalizes the temporal density: whether the browser sent
                # 12 frames or 20 frames over 2 seconds, the model always sees
                # a 60-frame sequence with the same temporal structure as training.
                sequence = self._resample_to_length(
                    [kp for _, kp in self.frame_buffer], SEQUENCE_LENGTH
                )
                prediction = predictor.predict_dynamic(sequence)
                return {
                    "type": "prediction",
                    "mode": "dynamic",
                    "has_hand": has_hand,
                    "landmarks": landmarks,
                    "buffer_fill": SEQUENCE_LENGTH,
                    **prediction
                }
            else:
                return {
                    "type": "buffering",
                    "mode": "dynamic",
                    "has_hand": has_hand,
                    "landmarks": landmarks,
                    "buffer_fill": int((buffer_duration / CAPTURE_WINDOW_SEC) * SEQUENCE_LENGTH),
                    "buffer_total": SEQUENCE_LENGTH
                }

        return None

    @staticmethod
    def _resample_to_length(frames, target_len):
        """
        Resample a variable-length list of frames to exactly target_len
        using linear interpolation across all 258 feature channels.

        This is the key fix: training data was captured at 30 FPS (60 frames = 2s),
        but the browser sends at 6-10 FPS (12-20 frames = 2s). By resampling,
        the model always sees the same temporal density it was trained on.
        """
        current = np.array(frames, dtype=np.float32)  # (N, 258)
        if len(current) == target_len:
            return current

        # Generate evenly spaced indices into the original sequence
        src_indices = np.arange(len(current))
        dst_indices = np.linspace(0, len(current) - 1, target_len)

        # Interpolate each feature channel independently
        resampled = np.zeros((target_len, current.shape[1]), dtype=np.float32)
        for feat in range(current.shape[1]):
            resampled[:, feat] = np.interp(dst_indices, src_indices, current[:, feat])

        return resampled
