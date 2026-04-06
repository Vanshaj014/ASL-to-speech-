# SignSpeak AI — Real-Time ASL Sign Language Translator

A full-stack AI application that translates American Sign Language (ASL) gestures to text and speech in real-time using a webcam, deep learning, and a modern web dashboard.

## Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | MediaPipe Holistic |
| Deep Learning | TensorFlow / Keras (MLP + LSTM) |
| Backend | Django 6 + Django Channels (ASGI/WebSocket) |
| Frontend | React (Vite) |
| Database | SQLite (dev) |

## Project Structure

```
Major project/
├── backend/          Django backend (API + WebSocket + ML inference)
├── frontend/         React dashboard (Vite)
├── ml/               ML training pipeline (standalone scripts)
└── venv/             Python virtual environment
```

## Quick Start

### 1. Backend
```bash
# Activate virtual environment
.\venv\Scripts\activate   # Windows

# Start Django server
cd backend
python manage.py runserver 8000
```

### 2. Frontend
```bash
cd frontend
npm run dev
# Opens at http://localhost:5173
```

## ML Pipeline

### Step 1 — Download Dataset
Download the Kaggle ASL Alphabet dataset:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Unzip to: `ml/data/raw/kaggle_asl/asl_alphabet_train/`

### Step 2 — Preprocess (Static A–Z)
```bash
python ml/preprocess_kaggle.py
```

### Step 3 — Collect Dynamic Signs (Words)
```bash
python ml/collect_custom_data.py
```
Follow the on-screen prompts. Signs: hello, thanks, yes, no, please, sorry, iloveyou, help, good, bad, more, stop, eat, drink, where

### Step 4 — Train Models
```bash
python ml/train_static.py    # A-Z MLP classifier
python ml/train_dynamic.py   # Word LSTM classifier
```

### Step 5 — Evaluate
```bash
python ml/evaluate.py --model all
```

Trained models are automatically copied to `backend/translator/ml/models/`

## WebSocket API

Connect to `ws://localhost:8000/ws/translate/`

**Send frames:**
```json
{ "type": "frame", "frame": "<base64 JPEG>" }
```

**Switch mode:**
```json
{ "type": "set_mode", "mode": "static" }
```

**Receive predictions:**
```json
{ "type": "prediction", "sign": "A", "confidence": 0.97, "top3": [...] }
```

## REST API

| Endpoint | Method | Description |
|---|---|---|
| `/api/vocabulary/` | GET | All supported signs |
| `/api/status/` | GET | Model health status |
| `/api/sessions/` | GET/POST | Translation sessions |
| `/api/tts/` | POST | Text-to-speech trigger |

## Features

- Real-time webcam capture in browser (no server-side camera)
- Static A–Z fingerspelling recognition
- Dynamic common word recognition (15+ words)
- Sentence builder with word chips
- Text-to-Speech output
- Translation history
- Deployment-ready architecture

## Deployment Notes

- Set `DEBUG=False` and a strong `DJANGO_SECRET_KEY` in `backend/.env`
- Run `npm run build` in `frontend/` then `python manage.py collectstatic`
- Use Redis channel layer for multi-process production (`channels_redis`)
- Serve with Gunicorn + Nginx

---

Made as a Final Year Project — CSE
