"""Doctor / preflight for the Hindi dialogue profile.

Checks capabilities without printing secrets: only *whether* a credential is
set is reported, never its value.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

from . import audio

REQUIRED_FILTERS = ["loudnorm", "sidechaincompress", "amix", "atempo", "alimiter", "apad", "atrim"]


def _module_version(name: str):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except Exception:
        return None


def _major(version) -> int:
    try:
        return int(str(version).split(".")[0])
    except ValueError:
        return 0


def run_preflight(work_root: Path = None, providers: List[str] = ("edge",),
                  engines: List[str] = ("gemini", "groq", "cerebras")) -> Dict:
    checks: List[Dict] = []

    def add(name, ok, detail, level="blocking", ok_detail=None):
        """`detail` explains a failure; `ok_detail` (default: the value) a pass.
        Credential checks pass the secret itself as `ok` — never echo it."""
        if ok and ok_detail is None:
            secret = any(t in name.upper() for t in ("KEY", "TOKEN", "SECRET"))
            ok_detail = "set" if secret else (ok if isinstance(ok, str) else "yes")
        checks.append({"check": name, "ok": bool(ok), "detail": ok_detail if ok else detail,
                       "level": "ok" if ok else level})

    # FFmpeg
    try:
        ff = audio.find_ffmpeg()
        missing = [f for f in REQUIRED_FILTERS if not audio.has_filter(f)]
        add("ffmpeg", not missing, f"{ff}; missing filters: {missing}", ok_detail=ff)
        add("ffmpeg rubberband (better time-stretch)", audio.has_filter("rubberband"),
            "not available: atempo fallback will be used", level="warning",
            ok_detail="available")
    except RuntimeError as e:
        add("ffmpeg", False, str(e))

    add("numpy", _module_version("numpy"), "pip install numpy")
    add("python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0], level="warning",
        ok_detail=sys.version.split()[0])

    # ASR
    fw = _module_version("faster_whisper")
    add("faster-whisper (local ASR + Hindi re-ASR verification)", fw,
        f"version {fw}" if fw else "pip install faster-whisper; without it only Groq ASR "
        "works and spoken Hindi is NOT verified", level="warning")
    add("GROQ_API_KEY (cloud ASR, optional)", os.environ.get("GROQ_API_KEY"),
        "set" if os.environ.get("GROQ_API_KEY") else "not set", level="info")

    # Diarization
    pa = _module_version("pyannote.audio")
    add("pyannote.audio (speaker diarization)", pa,
        f"version {pa} ({'community-1 capable' if _major(pa) >= 4 else '3.1 only'})"
        if pa else "pip install 'pyannote.audio>=4'; without it all dialogue is one voice",
        level="warning")
    add("HF_TOKEN (gated pyannote models)", os.environ.get("HF_TOKEN"),
        "set" if os.environ.get("HF_TOKEN") else
        "not set: accept conditions at https://huggingface.co/pyannote/speaker-diarization-community-1",
        level="warning")

    # GPU
    try:
        import torch
        cuda = torch.cuda.is_available()
        detail = f"torch {torch.__version__}; CUDA {'yes' if cuda else 'no'}"
        if cuda:
            p = torch.cuda.get_device_properties(0)
            detail += f"; {p.name} {p.total_memory / 1e9:.1f} GB"
        add("torch / CUDA", cuda, detail + ("" if cuda else " (CPU works, much slower)"), level="warning")
    except Exception:
        add("torch", False, "not installed (needed by pyannote/demucs)", level="warning")

    add("audio-separator (background separation, preferred)", _module_version("audio_separator"),
        'not installed: pip install "audio-separator[gpu]" (Demucs is used if present)',
        level="warning")
    add("demucs (background separation fallback)", _module_version("demucs"),
        "not installed" + ("" if _module_version("audio_separator")
                           else ": output will be Hindi dialogue without background"),
        level="warning")

    # Translation
    from .translation import OpenAICompatClient
    have = [e for e in engines if e in OpenAICompatClient.ENDPOINTS and OpenAICompatClient(e).available]
    add("LLM translation engine", have, f"available: {have or 'none'} (keys: "
        + ", ".join(f"{OpenAICompatClient.ENDPOINTS[e][1]}" for e in engines
                    if OpenAICompatClient.ENDPOINTS.get(e, ('', ''))[1]) + ")", level="warning")
    add("deep-translator (last-resort basic translation)", _module_version("deep_translator"),
        "pip install deep-translator", level="info")

    # TTS
    for p in providers:
        if p == "edge":
            add("edge-tts", _module_version("edge_tts"), "pip install edge-tts (needs internet)")
        elif p == "sarvam":
            add("SARVAM_API_KEY", os.environ.get("SARVAM_API_KEY"), "paid provider key")
        elif p == "elevenlabs":
            add("ELEVENLABS_API_KEY + ELEVENLABS_VOICES_MALE/FEMALE",
                os.environ.get("ELEVENLABS_API_KEY") and (os.environ.get("ELEVENLABS_VOICES_MALE")
                                                          or os.environ.get("ELEVENLABS_VOICES_FEMALE")),
                "paid provider; voice IDs must be listed per category")
        elif p == "google":
            add("GOOGLE_TTS_API_KEY", os.environ.get("GOOGLE_TTS_API_KEY"), "paid provider key")

    # Optional, experimental: "Sound like the original speaker" (OpenVoice in
    # its own venv; never installed into the app's Python).
    from .local_workers import runtime_status
    ov_ok, ov_why = runtime_status("openvoice")
    set_up = (os.environ.get("OPENVOICE_PYTHON", "").strip()
              or not ov_why.startswith("not installed in this Python"))
    add("OpenVoice (optional, experimental: sound like the original speaker)", ov_ok,
        f"{ov_why} (only needed for that option; GPU recommended)" if set_up else
        "not set up (only needed for that option): run setup_local_ai.bat, which sets "
        "OPENVOICE_PYTHON", level="warning", ok_detail=ov_why)

    # Links
    add("yt-dlp (links)", _module_version("yt_dlp") or shutil.which("yt-dlp"),
        "pip install yt-dlp; local files work without it", level="warning")
    cookies = Path(__file__).resolve().parents[2] / "cookies.txt"
    add("cookies.txt (private/age-restricted links)", cookies.exists(),
        "present" if cookies.exists() else "absent (public videos only)", level="info")

    # Disk
    root = Path(work_root or Path(__file__).resolve().parents[2] / "work")
    try:
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free / 1e9
        add("free disk", free >= 10, f"{free:.1f} GB free at {root} (>=10 GB recommended)", level="warning")
    except Exception as e:
        add("free disk", False, str(e), level="warning")

    blocking = [c for c in checks if c["level"] == "blocking"]
    return {"ready": not blocking, "checks": checks}


def format_preflight(res: Dict) -> str:
    icon = {"ok": "OK  ", "warning": "WARN", "blocking": "FAIL", "info": "info"}
    lines = [f"{icon[c['level']]}  {c['check']}: {c['detail']}" for c in res["checks"]]
    lines.append("")
    lines.append("READY" if res["ready"] else "NOT READY (fix FAIL items)")
    return "\n".join(lines)
