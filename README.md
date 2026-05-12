# 🤟 SignSpeak AI
### Real-Time ASL Sign Language Translator

**SignSpeak AI** is a production-ready, full-stack web application that bridges the communication gap for the Deaf and Hard-of-Hearing community. It translates American Sign Language (ASL) gestures — both static letters and dynamic words — into text and audible speech in real-time, using only a standard web camera. No special hardware required.

---

## ✨ Key Features

- 🧠 **Hybrid AI Engine**: Dual-model architecture — a **geometry-enhanced MLP** for static fingerspelling (A–Z) and a **1D-CNN + LSTM hybrid** for dynamic word gestures.
- 🖐️ **Advanced Finger Tracking**: 87-feature vector engineered from raw landmarks: **15 joint bend angles**, **5 extension states**, and **4 fingertip spread distances** — making inference immune to hand size and camera distance variations.
- 🦴 **Live Skeleton Overlay**: Real-time 21-point hand skeleton drawn directly over the webcam feed. Finger bones are colour-coded (Thumb: Red, Index: Yellow, Middle: Green, Ring: Blue, Pinky: Purple) and perfectly mirrored to match the user's view.
- 🎨 **Tech Innovation UI**: Immersive dark-mode design system with staggered load animations, neon-glow accents, and a Space Mono / DM Sans typography pairing.
- 🎯 **Majority Voting Smoothing**: 7-frame sliding window with dual-gate confidence checks (raw threshold 70% + agreement ratio 42%) eliminates all prediction flickering.
- ⚡ **Real-Time WebSocket Streaming**: Ultra-low latency, full-duplex communication via Django Channels with backpressure throttling to prevent server overload.
- 📐 **Landmark-Based Inference**: MediaPipe Holistic extracts 3D joint coordinates — the AI never "sees" your skin or background, making it robust to all lighting conditions.
- 🌗 **Mirrored Hand Support**: Automatically detects and mirrors left-hand landmarks before inference, supporting both left- and right-handed signers.
- 🗣️ **Text-to-Speech (TTS)**: Browser-native TTS to read translated sentences aloud. Falls back to server-side gTTS if needed.
- 📝 **Sentence Builder**: Confirmed signs are accumulated into a growing sentence with Speak, Copy, Undo, Space, and Clear controls.
- 📚 **Live Vocabulary Sync**: The frontend dashboard fetches supported signs from the backend API and displays them dynamically.

---

## 🏆 Model Performance

Both models were trained and evaluated on real data, with all results auto-generated from the training runs.

### Static Model — A–Z Fingerspelling (MLP)

| Metric | Result |
|---|---|
| **Training Accuracy** | ~100% |
| **Validation / Test Accuracy** | **~98.7%** |
| **Inference Time** | **< 5ms** per frame |
| **Model Size** | ~850KB |
| **Training Dataset** | 80,000+ images (Kaggle ASL Alphabet) |
| **Feature Vector** | 87 dimensions (63 raw + 24 derived geometry features) |
| **Classes** | 26 (A–Z) |

### Dynamic Model — Word Signs (1D-CNN + LSTM)

| Metric | Result |
|---|---|
| **Test Accuracy** | **100%** |
| **Test Loss** | **0.0020** |
| **Precision / Recall / F1** | **1.00 across all 15 classes** |
| **Sequence Length** | 60 frames (~2 seconds of motion at 30 FPS) |
| **Features per Frame** | 258 (63 right hand + 63 left hand + 132 pose) |
| **Dynamic Vocabulary** | 15 words: hello, thanks, yes, no, please, sorry, iloveyou, help, good, bad, more, stop, eat, drink, where |
| **Data Collection** | Custom-recorded using phone as high-quality webcam (DroidCam/iVCam) |
| **Augmentation** | 4× temporal expansion: speed jitter, Gaussian noise, time-shifting |

---

## 🏗️ Architecture

### 1. Frontend (React 19 + Vite)
- **Camera Controller**: Captures frames at 10 FPS without blocking the UI thread.
- **Backpressure Throttle**: The WebSocket hook tracks `pendingFrames` and skips sending if the backend hasn't responded yet — preventing `code=1006` crash loops.
- **Skeleton Renderer**: A transparent `<canvas>` overlay maps 21 backend-returned landmarks onto the mirrored video feed in real time.
- **Stability Indicator**: Visual bar showing the voting agreement %, turning green when a prediction is locked in.

### 2. Backend (Django ASGI + Channels)
- **Thread-Safe Consumers**: All CPU-heavy work (base64 decoding, OpenCV, MediaPipe, TF inference) is dispatched to a `ThreadPoolExecutor`, keeping the async event loop unblocked.
- **Predictor Singleton**: Both models are pre-loaded into RAM at server startup. Inference is **under 5ms**.
- **Dual-Gate Confidence**: Raw confidence (≥70%) AND majority vote agreement (≥42%) must both pass before a prediction is emitted.
- **Shape-Safe Dynamic Inference**: The dynamic predictor reads `SEQUENCE_LENGTH` from a single constant, eliminating the shape-mismatch bug that previously caused silent inference failures.
- **REST API**: `/api/vocabulary/`, `/api/model-status/`, and `/api/tts/` endpoints with `IsAuthenticatedOrReadOnly` protection.

### 3. ML Pipeline (TensorFlow / Keras)

#### Static Model — MLP
```
Input(87) → Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) →
Dropout(0.2) → Dense(64, ReLU) → Softmax(26)
```

#### Dynamic Model — 1D-CNN + LSTM
```
Input(60, 258)
  → Conv1D(64, k=3) → BatchNorm → ReLU → MaxPool(2)   # → (30, 64)
  → Conv1D(128, k=3) → BatchNorm → ReLU → MaxPool(2)  # → (15, 128)
  → LSTM(128) → Dropout(0.4)
  → Dense(64, ReLU) → Dropout(0.3)
  → Softmax(15)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- A webcam (or phone as webcam via DroidCam / iVCam)

### 1. Setup Backend
```bash
# Clone the repository
git clone https://github.com/Vanshaj014/ASL-to-speech-.git
cd ASL-to-speech-/backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run migrations and start server
python manage.py migrate
python manage.py runserver 8000
```

### 2. Setup Frontend
```bash
cd ../frontend
npm install
npm run dev
# App available at http://localhost:5173
```

> **Network access (e.g. phone camera):** Use `python manage.py runserver 0.0.0.0:8000` and `npm run dev -- --host` to expose both servers on your local network.

---

## 🧪 Training the Models

### Static Model (A–Z)
1. Download the [Kaggle ASL Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and place it in `ml/data/raw/kaggle_asl/`.
2. Preprocess (extracts 87-feature vectors from all images):
   ```bash
   python ml/preprocess_kaggle.py
   ```
3. Train:
   ```bash
   python ml/train_static.py
   ```
   The model is automatically saved to `backend/translator/ml/models/`.

### Dynamic Model (Word Signs)
1. Collect your own gesture recordings (60 frames × 30 sequences per sign):
   ```bash
   python ml/collect_custom_data.py
   ```
   - Press **SPACE** to start each recording (3-second countdown, then ~2-second capture).
   - Press **Q** to skip to the next sign.
2. Train the CNN-LSTM model:
   ```bash
   python ml/train_dynamic.py
   ```
3. Restart the Django server so it reloads the new model from RAM.

> **Camera tip:** For best data quality, use your phone as a webcam (install **DroidCam** on Android or **iVCam** on iPhone). The higher camera quality and placement flexibility significantly improve model generalization.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | React 19, Vite, Vanilla CSS |
| **Backend** | Django 6, Django Channels (ASGI WebSockets), Django REST Framework |
| **Computer Vision** | MediaPipe Holistic, OpenCV |
| **Static ML Model** | TensorFlow / Keras — MLP (87-feature geometry-enhanced) |
| **Dynamic ML Model** | TensorFlow / Keras — 1D-CNN + LSTM Hybrid |
| **Feature Engineering** | NumPy — joint bend angles, extension states, fingertip spread distances |
| **Security** | HSTS, X-Frame-Options, CSRF, DoS frame-size limits, API auth guards |

---

## 📁 Project Structure

```text
Major project/
├── backend/                     # Django ASGI Server
│   ├── config/                  # Settings & ASGI/URL routing
│   └── translator/              # Core app
│       ├── consumers.py         # WebSocket consumer (backpressure, threading)
│       ├── views.py             # REST API (vocabulary, TTS, model status)
│       └── ml/                  # Inference engine
│           ├── predictor.py     # Singleton ASLPredictor (both models + voting)
│           ├── mediapipe_utils.py # Landmark extraction + serialize_hand_landmarks()
│           └── models/          # .keras / .h5 model files (gitignored)
├── frontend/                    # React Application
│   ├── src/
│   │   ├── components/          # VideoFeed, PredictionDisplay, SentenceBuilder, etc.
│   │   ├── hooks/               # useWebSocket (backpressure, reconnect, heartbeat)
│   │   └── App.jsx              # Root state & layout
│   └── .env                     # VITE_WS_URL, VITE_API_URL
├── ml/                          # Standalone Training Pipeline
│   ├── mediapipe_utils.py       # Feature extraction (87 static, 258 dynamic)
│   ├── preprocess_kaggle.py     # Kaggle images → .npy feature files
│   ├── train_static.py          # MLP training with geometric augmentation
│   ├── train_dynamic.py         # CNN-LSTM training (60-frame sequences)
│   ├── collect_custom_data.py   # Interactive webcam-based data recorder
│   ├── evaluate.py              # Model evaluation & confusion matrix plotting
│   └── models/                  # Training output charts + model files (gitignored)
├── architecture.md              # System design diagrams & DFDs (local only)
└── README.md                    # You are here!
```

---

## 💡 Tips for Best Results

- 💡 **Lighting**: Ensure even, bright lighting on your hands for clearest landmark detection.
- 📍 **Position**: Keep your hand centred and within the camera frame.
- ✋ **Static Signs**: Hold each letter for ~0.5 seconds until the Stability bar turns green.
- 🏃 **Dynamic Signs**: Perform the full word sign within the 2-second capture window.
- 🌗 **Left-handed**: Fully supported — the model mirrors landmarks automatically.
- 📱 **Phone Camera**: For sharper image quality, use your phone as a webcam via DroidCam (Android) or iVCam (iPhone).

---

**Developed as a Final Year Project for Computer Science & Engineering.**  
*Designed to make communication accessible for everyone.*
