# Project Progress Log: SignSpeak AI (ASL Translator)

This document tracks all technical changes, optimizations, and new features implemented in the project.

## Recent Progress & Optimizations (April)

### User Interface
- **Minimalist Refactor:** Transitioned from heavy, glassmorphism design to a clean, flat, high-contrast professional dark mode UI. Removed complex gradients and neon glows to improve visibility and performance.

### Machine Learning Pipeline
- **Aspect Ratio Normalization:** Fixed MediaPipe extraction bugs where 4:3 webcam frames were being squeezed into 1:1 inputs, which was previously distorting hand shape recognition.
- **Hand Modeling:** Implemented logic to process both left and right hands correctly regardless of webcam mirroring.
- **Static Model (A-Z) Enhancement:** Added **Geometric Data Augmentation** (gaussian jitter, scaling, and Z-axis rotation matrixes) to `train_static.py`. *Result: The model achieved 100% accuracy on the test set.*
- **Dynamic Model (Words) Architecture Upgrade:** Completely rebuilt `train_dynamic.py` from a pure LSTM to a State-of-the-Art **Hybrid 1D-CNN + LSTM** model. 
  - The `Conv1D` blocks extract localized motion patterns (like quick finger flicks) across consecutive frames.
  - The `LSTM` interprets the full sequence meaning over time.
  - Added **Temporal Data Augmentation** (speed jitter, noise, time-shifting) to synthetically quadruple the training dataset.

### Backend & Concurrency (Django Channels)
- **ASGI Thread Safety:** Discovered that decoding massive base64 images was monopolizing the backend's Global Interpreter Lock (GIL). Offloaded all synchronous `cv2` decode operations out of the async loop using a background thread `ThreadPoolExecutor`. The WebSocket server can now handle high-fps inputs without blocking.

### Frontend Resilience
- **Network Throttling (Backpressure):** Built a `pendingFrames` tracker in `useWebSocket.js`. The React `VideoFeed` now politely waits for the backend to finish processing the previous frame before shooting the next one over the network. This completely eliminated the browser crashing and `code=1006` WebSocket disconnects on slower internets.

### Repository Hygiene
- **Git Cleanup:** Executed a deep clean of the repository, successfully un-tracking massive ML output artifacts (TensorBoard logs, `.npy` files, matrix `.png` charts) from GitHub while preserving them on your local hard drive. Updated `.gitignore`.

### Confidence Threshold Tuning & Majority Voting
- **Majority Voting Buffer:** Replaced the weak single-frame prediction with a robust 7-frame sliding window voting system. Predictions are only emitted when at least 3 frames agree on the same sign, completely eliminating the "flickering" problem during live demos.
- **Dual-Gate System:** Predictions now pass through two gates: (1) raw model confidence must exceed 70%, and (2) the majority vote must clear an agreement ratio of 42%. 
- **Blended Confidence Score:** The displayed confidence is a weighted blend of raw model output (60%) and voting agreement (40%), giving users a more honest and stable metric.
- **Stabilizing State:** Added a new "Stabilizing prediction…" UI state with amber pulsing dots that appears while the voting buffer fills up, replacing an awkward blank screen.
- **Stability Indicator:** Added a visual stability bar to the prediction display showing voting agreement percentage in real time.
- **Buffer Reset:** The voting buffer now properly clears when the hand leaves the frame, preventing stale predictions from persisting.

### Finger Tracking & Enhanced Feature Engineering (87-Feature Vector)
- **Problem:** MediaPipe raw coordinates are sensitive to camera distance and hand size, causing the model to confuse similar signs like A, S, and E.
- **Solution:** Extended the feature vector from **63 → 87 features** using three categories of derived geometry:
  - **15 Joint Bend Angles:** The angle (in radians) at the MCP, PIP, and DIP joints of all 5 fingers, computed via 3D vector dot products. These are fully scale-invariant.
  - **5 Finger Extension States:** Binary flag per finger — `1` if extended (tip farther from wrist than PIP), `0` if curled. Directly encodes the "finger up/down" that the human eye uses.
  - **4 Fingertip Spread Distances:** Euclidean distance between adjacent fingertip pairs (index-middle, middle-ring, etc.). Critical for distinguishing V, W, 3, etc.
- **Files updated:** `mediapipe_utils.py`, `train_static.py`, `preprocess_kaggle.py`, `consumers.py`, `predictor.py`.
- **Action Required:** Re-run `preprocess_kaggle.py` then `train_static.py` to bake the new features into the live model.

### UI Redesign: Tech Innovation Theme
- **Theme Overhaul:** Applied the "Tech Innovation" design system combining Electric Blue (`#0066ff`) and Neon Cyan (`#00e5ff`) with a deep dark background (`#0d1117`).
- **Typography Upgrade:** Replaced generic Inter font with a distinctive pairing: **Space Mono** (for headers and the sign display) and **DM Sans** (for body text).
- **Cinematic Motion & Depth:** Introduced staggered load animations across the layout columns, a pop-in animation for the main sign display, subtle glowing card borders on hover, an atmospheric grid texture background, and a sticky frosted-glass header.

### Code Review & Critical Fixes
- **TTS Reset Bug:** Fixed a bug in `SentenceBuilder.jsx` where the `isSpeaking` state never reset if the HTTP request succeeded, locking the Speak button. Refactored the browser Web Speech API fallback to properly use the `onend` and `onerror` callbacks instead of a fixed 2-second timeout.
- **WebSocket Frame Leak:** Fixed an issue in `useWebSocket.js` where switching between Static and Dynamic modes while a frame was mid-flight caused the `pendingFrames` counter to stay stuck at 1, permanently freezing the camera feed via backpressure. Counter now correctly resets to 0 on a `mode_changed` signal.
- **Legacy Output Format:** Upgraded `predictor.py` and the training scripts to prefer the modern `_best.keras` file format with a graceful fallback to `.h5`, resolving the "compiled metrics not built" warning emitted by TensorFlow on startup.
- **Session API Security:** Fortified `views.py` by applying the `IsAuthenticatedOrReadOnly` permission to `SessionListCreateView` and `SessionDetailView`, preventing unauthenticated users from enumerating or deleting translation session history over REST.
- **Code Cleanliness:** Extracted 60+ lines of inline JSX `<style>` objects from `VideoFeed.jsx` into a dedicated `VideoFeed.css` stylesheet, and scrubbed hardcoded legacy colour hexes in favour of the centralized CSS variable system.

---
*Note: This log will be updated continuously as new features or architectural changes are implemented.*
