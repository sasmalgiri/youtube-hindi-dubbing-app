"""Google Cloud TTS over REST with an API key (Chirp 3 HD by default).

Owner decision OQ5 (2026-11-20): Chirp 3 HD is the Standard voice, Sarvam the
fallback, Edge only for local use. Chirp 3 HD voices take no pitch/SSML, so
the request carries plain text and the voice name only.

Env:
  GOOGLE_TTS_API_KEY     API key with the Text-to-Speech API enabled
  GOOGLE_TTS_VOICE_HI    voice name (default hi-IN-Chirp3-HD-Kore)
"""
from __future__ import annotations

import base64
import os
from typing import Callable, Optional

DEFAULT_VOICE = {"hi": "hi-IN-Chirp3-HD-Kore"}
CHIRP_HINDI_VOICES = (
    # female
    "hi-IN-Chirp3-HD-Kore", "hi-IN-Chirp3-HD-Aoede", "hi-IN-Chirp3-HD-Leda", "hi-IN-Chirp3-HD-Zephyr",
    # male
    "hi-IN-Chirp3-HD-Charon", "hi-IN-Chirp3-HD-Puck", "hi-IN-Chirp3-HD-Fenrir", "hi-IN-Chirp3-HD-Orus",
)
URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


class ChirpError(RuntimeError):
    pass


def api_key() -> str:
    return os.environ.get("GOOGLE_TTS_API_KEY", "").strip()


def voice_for(lang: str) -> str:
    v = os.environ.get(f"GOOGLE_TTS_VOICE_{lang.upper()}", "").strip()
    return v or DEFAULT_VOICE.get(lang, f"{lang}-IN-Chirp3-HD-Kore")


def is_chirp(voice: str) -> bool:
    return "-Chirp3-HD-" in voice or "-Chirp-HD-" in voice


def request_body(text: str, voice: str, sample_rate: int, speaking_rate: Optional[float] = None) -> dict:
    lang = "-".join(voice.split("-")[:2])
    audio = {"audioEncoding": "LINEAR16", "sampleRateHertz": int(sample_rate)}
    if speaking_rate and abs(speaking_rate - 1.0) > 1e-3:
        audio["speakingRate"] = max(0.25, min(2.0, float(speaking_rate)))
    return {"input": {"text": text}, "voice": {"languageCode": lang, "name": voice}, "audioConfig": audio}


def synthesize(text: str, voice: str, sample_rate: int = 24000, *, key: Optional[str] = None,
               post: Optional[Callable] = None, timeout: int = 60) -> bytes:
    """Returns WAV bytes (LINEAR16 with header). Raises ChirpError with a short reason."""
    key = key if key is not None else api_key()
    if not key:
        raise ChirpError("GOOGLE_TTS_API_KEY not set")
    if not text.strip():
        raise ChirpError("empty text")
    if post is None:
        import requests
        post = requests.post
    r = post(URL, params={"key": key}, json=request_body(text, voice, sample_rate), timeout=timeout)
    if r.status_code != 200:
        # never echo the key; the body names the problem (quota, disabled API, bad voice)
        raise ChirpError(f"Google TTS HTTP {r.status_code}: {r.text[:160]}")
    try:
        return base64.b64decode(r.json()["audioContent"])
    except Exception as e:
        raise ChirpError(f"Google TTS returned no audio ({type(e).__name__})")
