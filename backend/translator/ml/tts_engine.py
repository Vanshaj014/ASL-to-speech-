"""
tts_engine.py
--------------
Text-to-speech engine for the Django backend.
Uses pyttsx3 (offline, cross-platform) with gTTS as fallback.
For deployment: gTTS returns audio bytes that can be sent to the client.
"""

import logging
import io
import base64
import threading

logger = logging.getLogger(__name__)

# Lock to prevent simultaneous TTS from multiple threads
_tts_lock = threading.Lock()


def speak_text(text: str, use_gtts: bool = False) -> bool:
    """
    Speak the given text.
    - use_gtts=False: uses pyttsx3 (works locally)
    - use_gtts=True: uses gTTS (better for deployment; returns audio data)
    Returns True on success, False on failure.
    """
    if not text or not text.strip():
        return False

    if use_gtts:
        return _speak_gtts(text)
    else:
        return _speak_pyttsx3(text)


def _speak_pyttsx3(text: str) -> bool:
    """Offline TTS using pyttsx3."""
    try:
        import pyttsx3
        with _tts_lock:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)   # Words per minute
            engine.setProperty("volume", 0.9)
            engine.say(text)
            engine.runAndWait()
        return True
    except Exception as e:
        logger.error(f"[TTS] pyttsx3 error: {e}")
        return False


def _speak_gtts(text: str) -> bool:
    """Online TTS using gTTS (Google Text-to-Speech)."""
    try:
        from gtts import gTTS
        import tempfile
        import os

        tts = gTTS(text=text, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            tmp_path = f.name

        # Play the file (platform-dependent)
        import subprocess
        import sys
        if sys.platform == "win32":
            os.startfile(tmp_path)
        elif sys.platform == "darwin":
            subprocess.run(["afplay", tmp_path])
        else:
            subprocess.run(["mpg321", tmp_path], capture_output=True)

        return True
    except Exception as e:
        logger.error(f"[TTS] gTTS error: {e}")
        return False


def text_to_audio_base64(text: str) -> str | None:
    """
    Convert text to speech and return as base64-encoded MP3.
    Useful for sending audio directly to the browser via API.
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return audio_b64
    except Exception as e:
        logger.error(f"[TTS] Audio base64 error: {e}")
        return None
