# 🤟 SignSpeak AI
### Real-Time ASL Sign Language Translator

**SignSpeak AI** is a professional-grade, full-stack application designed to bridge the communication gap for the Deaf and Hard-of-Hearing community. By leveraging computer vision and deep learning, it translates American Sign Language (ASL) gestures into text and audible speech in real-time.

---

## ✨ Key Features

- 🧠 **Hybrid AI Engine**: Dual-model architecture using a **MLP** (87-feature, finger-tracking enhanced) for static fingerspelling (A–Z) and a **1D-CNN + LSTM hybrid** for dynamic word gestures.
- 🖐️ **Advanced Finger Tracking**: Feature vector engineered from raw landmarks to include **15 joint bend angles**, **5 finger extension states**, and **4 fingertip spread distances** — making the model immune to hand size and camera distance variations.
- 🎯 **Majority Voting Smoothing**: A 7-frame sliding window voting system with a dual-gate confidence check (raw model threshold + agreement ratio) completely eliminates prediction flickering.
- ⚡ **Real-Time Streaming**: Ultra-low latency communication via **WebSockets** (Django Channels) with thread-safe background frame decoding.
- 📐 **Landmark-Based Inference**: Uses **MediaPipe Holistic** to extract 3D joint coordinates, making the system immune to lighting conditions and complex backgrounds.
- 🌗 **Mirrored Hand Support**: Automatically handles both left- and right-hand signers by mirroring left-hand landmarks before inference.
- 🗣️ **Text-to-Speech (TTS)**: Integrated browser-based TTS to convert translated text into clear audio.
- 📝 **Sentence Builder**: Intelligent buffer system that allows users to lock in signs to form complete sentences.
- 📚 **Dynamic Vocabulary**: Automatically synchronizes supported signs from the backend to the frontend dashboard.

---

## 🏗️ Architecture

### 1. Frontend (React + Vite)
- **Camera Controller**: Efficiently captures 10 FPS frames without blocking the UI thread.
- **Backpressure Throttle**: The WebSocket hook skips sending new frames if the backend hasn't responded yet, preventing network congestion and buffer overflows.
- **Stability Indicator**: Real-time visual UI showing the voting agreement percentage so users know when a prediction is locked-in vs. still stabilizing.

### 2. Backend (Django ASGI)
- **Thread-Safe Consumers**: CPU-heavy `base64` decoding and `cv2` image processing are fully offloaded to a background thread pool, leaving the ASGI event loop unblocked.
- **Predictor Singleton**: Pre-loads TensorFlow models into RAM for sub-5ms inference.
- **Dual Confidence Gates**: Predictions pass through a minimum raw confidence check (70%) AND a majority vote agreement check before being emitted.
- **REST API**: Provides endpoints for vocabulary lists and system health checks.

### 3. ML Pipeline (TensorFlow/Keras)

#### Static Model — MLP (A–Z Fingerspelling)
- **Input:** 87 features per frame (63 raw landmarks + 24 derived geometry features)
- **Architecture:** 4-layer MLP (Dense 256 → 128 → 64 → Softmax 26)
- **Accuracy:** ~99% on validation set after geometric data augmentation
- **Training Data:** 80,000+ Kaggle ASL images processed through MediaPipe

#### Dynamic Model — 1D-CNN + LSTM (Word Signs)
- **Input:** Sequence of 30 frames × 258 features each (~1 second of motion)
- **Architecture:**  
  `Conv1D(64)` → `MaxPool` → `Conv1D(128)` → `MaxPool` → `LSTM(128)` → `Dense(64)` → Softmax
- **Augmentation:** Speed jitter, Gaussian noise, and time-shifting (4× dataset expansion)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Webcam

### 1. Setup Backend
```bash
# Clone the repository
git clone https://github.com/Vanshaj014/ASL-to-speech-.git
cd ASL-to-speech-/backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

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
# Dashboard available at http://localhost:5173
```

---

## 🧪 Training the Models

### Static Model (A–Z)

1. Download the [Kaggle ASL Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and place it in `ml/data/raw/kaggle_asl/`.
2. Run preprocessing (extracts 87-feature vectors from all images):
   ```bash
   python ml/preprocess_kaggle.py
   ```
3. Train:
   ```bash
   python ml/train_static.py
   ```
   The model will automatically be saved to `backend/translator/ml/models/`.

### Dynamic Model (Word Signs)

1. Collect your own gesture recordings:
   ```bash
   python ml/collect_custom_data.py
   ```
2. Train the 1D-CNN + LSTM model:
   ```bash
   python ml/train_dynamic.py
   ```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | React 19, Vite, Vanilla CSS |
| **Backend** | Django 6, Django Channels (ASGI WebSockets), DRF |
| **Computer Vision** | MediaPipe Holistic, OpenCV |
| **Static ML Model** | TensorFlow / Keras MLP (87-feature, geometry-enhanced) |
| **Dynamic ML Model** | TensorFlow / Keras 1D-CNN + LSTM Hybrid |
| **Feature Engineering** | NumPy — joint angles, extension states, fingertip spread |

---

## 📁 Project Structure

```text
Major project/
├── backend/                    # Django ASGI Server
│   ├── config/                 # Settings & Routing
│   └── translator/             # ML Logic, WebSocket Consumer & API Views
│       └── ml/                 # Backend-side models & inference engine
├── frontend/                   # React Application
│   ├── src/components/         # PredictionDisplay, VideoFeed, SentenceBuilder, etc.
│   └── src/hooks/              # useWebSocket (with backpressure throttle)
├── ml/                         # Standalone ML Training Pipeline
│   ├── mediapipe_utils.py      # Feature extraction (63 raw + 24 derived = 87 features)
│   ├── preprocess_kaggle.py    # Kaggle dataset → 87-feature .npy files
│   ├── train_static.py         # MLP training with geometric augmentation
│   ├── train_dynamic.py        # CNN-LSTM training with temporal augmentation
│   ├── collect_custom_data.py  # Dynamic sign data recorder
│   └── models/                 # Exported .h5 files (gitignored — retrain locally)
├── progress.md                 # Full engineering change-log
└── README.md                   # You are here!
```

---

## 💡 Tips for Best Results
- 💡 **Lighting**: Ensure even, bright lighting on your hands.
- 📍 **Position**: Keep your hand centred in the camera frame.
- ✋ **Signing**: Hold static signs for ~0.5s until the stability indicator turns green.
- 🌗 **Both Hands**: Left-handed signers are fully supported — the model mirrors your hand automatically.

---

**Developed as a Final Year Project for Computer Science & Engineering.**
*Designed to make communication accessible for everyone.*
