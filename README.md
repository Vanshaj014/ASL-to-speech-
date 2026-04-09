# 🤟 SignSpeak AI
### Real-Time ASL Sign Language Translator

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-6.0-092E20?style=for-the-badge&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-007FFF?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)

**SignSpeak AI** is a professional-grade, full-stack application designed to bridge the communication gap for the Deaf and Hard-of-Hearing community. By leveraging computer vision and deep learning, it translates American Sign Language (ASL) gestures into text and audible speech in real-time.

---

## 📽️ Preview
> **[Add Demo Video/GIF Link Here]**
*(The interface features a real-time webcam feed, a dynamic sentence builder, and a history log of translated phrases.)*

---

## ✨ Key Features

- 🧠 **Hybrid AI Engine**: Dual-model architecture using **MLP** for static fingerspelling (A-Z) and **LSTM** for dynamic word gestures.
- ⚡ **Real-Time Streaming**: Ultra-low latency communication via **WebSockets** (Django Channels) and `base64` frame streaming.
- 📐 **Landmark-Based Inference**: Uses **MediaPipe Holistic** to extract 3D coordinates, making the system immune to lighting conditions and complex backgrounds.
- 🗣️ **Text-to-Speech (TTS)**: Integrated browser-based and server-side TTS to convert translated text into clear audio.
- 📝 **Sentence Builder**: Intelligent buffer system that allows users to "lock in" signs to form complete sentences.
- 📚 **Dynamic Vocabulary**: Automatically synchronizes supported signs from the backend to the frontend dashboard.

---

## 🏗️ Architecture

### 1. Frontend (React + Vite)
- **Camera Controller**: Efficiently captures frames without blocking the UI thread.
- **WebSocket Hook**: Manages persistent two-way communication with the Django server.
- **State Management**: Handles real-time predictions, sentence buffering, and history tracking.

### 2. Backend (Django ASGI)
- **WebSocket Consumers**: Decodes image frames and pipelines them into the ML predictor.
- **Predictor Singleton**: Pre-loads TensorFlow models into RAM for sub-millisecond inference.
- **REST API**: Provides endpoints for session management, vocabulary lists, and system health.

### 3. ML Pipeline (TensorFlow/Keras)
- **Static Model**: 4-layer MLP trained on 80,000+ ASL alphabet images.
- **Dynamic Model**: LSTM network analyzing 30-frame temporal sequences (approx. 1 second of movement).

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Webcam

### 1. Setup Backend
```bash
# Clone the repository
git clone https://github.com/yourusername/signspeak-ai.git
cd signspeak-ai/backend

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

If you wish to retrain the models with custom data:

1. **Static Data**: Download the [Kaggle ASL Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and place it in `ml/data/raw/kaggle_asl/`.
2. **Dynamic Data**: Run the collection script to record your own signs:
   ```bash
   python ml/collect_custom_data.py
   ```
3. **Train**:
   ```bash
   python ml/train_static.py
   python ml/train_dynamic.py
   ```
   Trained models will be automatically moved to `backend/translator/ml/models/`.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | React 19, Vite, CSS Modules |
| **Backend** | Django 6, Django Channels (WebSockets), DRF |
| **Computer Vision** | MediaPipe Holistic, OpenCV |
| **Deep Learning** | TensorFlow 2.16, Keras, NumPy, Scikit-Learn |
| **Communication** | ASGI, Daphne, JSON/Base64 |

---

## 📁 Project Structure

```text
Major project/
├── backend/            # Django ASGI Server
│   ├── config/         # Settings & Routing
│   └── translator/     # ML Logic & API Views
├── frontend/           # React Application
│   ├── src/components/ # Modular UI Components
│   └── src/hooks/      # WebSocket & Camera Hooks
├── ml/                 # Standalone ML Pipeline
│   ├── data/           # Raw & Processed Landmarks
│   └── models/         # Exported .h5 Files
└── README.md           # You are here!
```

---

## 💡 Tips for Best Results
- **Lighting**: Ensure even lighting on your hands and face.
- **Stability**: Keep your hand within the center of the camera frame.
- **Signing**: Hold static signs for ~0.5s; perform dynamic signs within a 1s window.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---

**Developed as a Final Year Project for Computer Science & Engineering.**
*Designed to make communication accessible for everyone.*
