"""
Dubbing Pipeline with Translation Support
==========================================
Refactored from pipeline_v2 with:
- Callback-based progress reporting (for SSE)
- Translation step (deep-translator)
- Hindi TTS by default
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import subprocess
import sys
import time
import array as _array
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from srt_utils import write_srt
import cache as _cache


# ── Hardened CUDA cleanup ───────────────────────────────────────────────────
# Prevents `Fatal Python error: Aborted` when releasing CUDA models on Windows.
# Root cause (PID 20328 @ 13:02 and PID 17704 @ 15:25 on 2026-04-09): calling
# `torch.cuda.empty_cache()` immediately after `del model` while CUDA still
# has in-flight kernels triggers a C-level abort that bypasses Python try/except.
# The fix is *order*: gc.collect → synchronize → empty_cache → ipc_collect → sleep.
# A C abort cannot be caught — only prevented — so each guard is best-effort.
def _hardened_cuda_cleanup():
    try:
        import gc as _gc
        _gc.collect()
    except Exception:
        pass
    try:
        import torch as _t
        if _t.cuda.is_available():
            try:
                _t.cuda.synchronize()
            except Exception:
                pass
            try:
                _t.cuda.empty_cache()
            except Exception:
                pass
            try:
                _t.cuda.ipc_collect()
            except Exception:
                pass
            try:
                import time as _ti
                _ti.sleep(0.3)
            except Exception:
                pass
    except Exception:
        pass


# ── Subprocess worker for local Whisper transcription ────────────────────────
# Runs in a child process so that C-level crashes (SIGABRT, CUDA OOM that
# bypasses Python try/except) only kill the child — the server stays alive.
def _whisper_child_worker(wav_path_str: str, model_name: str, device: str,
                          compute: str, source_lang, result_path: str):
    """Top-level function for multiprocessing — must be picklable."""
    import json as _json
    try:
        from faster_whisper import WhisperModel
        import wave

        # Detect long audio
        long_audio = False
        try:
            with wave.open(wav_path_str, "rb") as w:
                duration_sec = w.getnframes() / float(w.getframerate())
                long_audio = duration_sec > 1200
        except Exception:
            pass

        # Pre-clean GPU
        if device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        _model = WhisperModel(model_name, device=device, compute_type=compute)
        try:
            kwargs = {
                "vad_filter": True,
                "word_timestamps": True,
                "beam_size": 1,
                "best_of": 1,
                "condition_on_previous_text": (not long_audio),
                "no_speech_threshold": 0.5,
                "compression_ratio_threshold": 2.4,
                "vad_parameters": {"min_silence_duration_ms": 300},
            }
            if source_lang:
                kwargs["language"] = source_lang

            seg_iter, _info = _model.transcribe(wav_path_str, **kwargs)
            segments = []
            for seg in seg_iter:
                entry = {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                }
                if hasattr(seg, "words") and seg.words:
                    entry["words"] = [
                        {"word": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                        for w in seg.words
                    ]
                segments.append(entry)

            with open(result_path, "w", encoding="utf-8") as f:
                _json.dump({"segments": segments, "error": None}, f, ensure_ascii=False)
        finally:
            try:
                _model = None
            except Exception:
                pass
            _hardened_cuda_cleanup()
    except Exception as exc:
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                _json.dump({"segments": [], "error": str(exc)}, f, ensure_ascii=False)
        except Exception:
            pass
        import sys
        sys.exit(1)


# ── Groq API Key Rotator ────────────────────────────────────────────────────
# Rotates between multiple GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, etc.
# Auto-skips keys that hit rate limits. Thread-safe.
class _GroqKeyRotator:
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._keys = []
        self._index = 0
        self._failures = {}  # key_index → last_failure_time
        self._loaded = False

    def _load_keys(self):
        if self._loaded:
            return
        self._loaded = True
        # Load all GROQ_API_KEY variants
        main = os.environ.get("GROQ_API_KEY", "").strip()
        if main:
            self._keys.append(main)
        for i in range(2, 20):
            k = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
            if k:
                self._keys.append(k)

    def get_key(self) -> str:
        """Get the next available Groq API key (round-robin, skip rate-limited)."""
        with self._lock:
            self._load_keys()
            if not self._keys:
                return ""
            now = time.time()
            # Try each key, skip recently failed ones (cooldown 30s)
            for _ in range(len(self._keys)):
                idx = self._index % len(self._keys)
                self._index += 1
                last_fail = self._failures.get(idx, 0)
                if now - last_fail > 30:  # 30s cooldown after rate limit
                    return self._keys[idx]
            # All keys rate-limited — return first anyway (will retry)
            return self._keys[0]

    def report_rate_limit(self, key: str):
        """Mark a key as rate-limited."""
        with self._lock:
            for i, k in enumerate(self._keys):
                if k == key:
                    self._failures[i] = time.time()
                    break

    def count(self) -> int:
        """Number of available keys."""
        self._load_keys()
        return len(self._keys)

_groq_keys = _GroqKeyRotator()


def get_groq_key() -> str:
    """Get the next available Groq API key (rotated)."""
    return _groq_keys.get_key()


# ── Sarvam AI API Key Rotator ────────────────────────────────────────────────
# Rotates between SARVAM_API_KEY, SARVAM_API_KEY_2, ... SARVAM_API_KEY_5
# Auto-skips keys with exhausted quota. Thread-safe.
class _SarvamKeyRotator:
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._keys = []
        self._index = 0
        self._failures = {}  # key_index → last_failure_time
        self._loaded = False

    def _load_keys(self):
        if self._loaded:
            return
        self._loaded = True
        main = os.environ.get("SARVAM_API_KEY", "").strip()
        if main:
            self._keys.append(main)
        for i in range(2, 20):
            k = os.environ.get(f"SARVAM_API_KEY_{i}", "").strip()
            if k:
                self._keys.append(k)

    def get_key(self) -> str:
        """Get the next available Sarvam API key (round-robin, skip quota-exhausted)."""
        with self._lock:
            self._load_keys()
            if not self._keys:
                return ""
            now = time.time()
            for _ in range(len(self._keys)):
                idx = self._index % len(self._keys)
                self._index += 1
                last_fail = self._failures.get(idx, 0)
                if now - last_fail > 60:  # 60s cooldown after quota error
                    return self._keys[idx]
            return self._keys[0]

    def report_quota_error(self, key: str):
        """Mark a key as quota-exhausted."""
        with self._lock:
            for i, k in enumerate(self._keys):
                if k == key:
                    self._failures[i] = time.time()
                    break

    def count(self) -> int:
        self._load_keys()
        return len(self._keys)

_sarvam_keys = _SarvamKeyRotator()


def get_sarvam_key() -> str:
    """Get the next available Sarvam API key (rotated)."""
    return _sarvam_keys.get_key()


# ── Gemini API Key Rotator ──────────────────────────────────────────────────
# Rotates between GEMINI_API_KEY, GEMINI_API_KEY_2, ... GEMINI_API_KEY_10
# Enables parallel Gemma 4 translation with multiple accounts. Thread-safe.
class _GeminiKeyRotator:
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._keys = []
        self._index = 0
        self._failures = {}  # key_index → last_failure_time
        self._loaded = False

    def _load_keys(self):
        if self._loaded:
            return
        self._loaded = True
        main = os.environ.get("GEMINI_API_KEY", "").strip()
        if main:
            self._keys.append(main)
        for i in range(2, 20):
            k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
            if k:
                self._keys.append(k)

    def get_key(self) -> str:
        """Get the next available Gemini API key (round-robin, skip rate-limited)."""
        with self._lock:
            self._load_keys()
            if not self._keys:
                return ""
            now = time.time()
            for _ in range(len(self._keys)):
                idx = self._index % len(self._keys)
                self._index += 1
                last_fail = self._failures.get(idx, 0)
                if now - last_fail > 30:  # 30s cooldown after rate limit
                    return self._keys[idx]
            return self._keys[0]

    def report_rate_limit(self, key: str):
        """Mark a key as rate-limited."""
        with self._lock:
            for i, k in enumerate(self._keys):
                if k == key:
                    self._failures[i] = time.time()
                    break

    def count(self) -> int:
        self._load_keys()
        return len(self._keys)

_gemini_keys = _GeminiKeyRotator()


def get_gemini_key() -> str:
    """Get the next available Gemini API key (rotated)."""
    return _gemini_keys.get_key()


# ── Cerebras API Key Rotator ────────────────────────────────────────────────
# Cerebras: fastest LLM inference (~2400 tokens/sec) — Llama 3.3 70B
class _CerebrasKeyRotator:
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._keys = []
        self._index = 0
        self._failures = {}
        self._loaded = False

    def _load_keys(self):
        if self._loaded:
            return
        self._loaded = True
        main = os.environ.get("CEREBRAS_API_KEY", "").strip()
        if main:
            self._keys.append(main)
        for i in range(2, 10):
            k = os.environ.get(f"CEREBRAS_API_KEY_{i}", "").strip()
            if k:
                self._keys.append(k)

    def get_key(self) -> str:
        with self._lock:
            self._load_keys()
            if not self._keys:
                return ""
            now = time.time()
            for _ in range(len(self._keys)):
                idx = self._index % len(self._keys)
                self._index += 1
                last_fail = self._failures.get(idx, 0)
                if now - last_fail > 60:
                    return self._keys[idx]
            return self._keys[0]

    def report_rate_limit(self, key: str):
        with self._lock:
            for i, k in enumerate(self._keys):
                if k == key:
                    self._failures[i] = time.time()
                    break

    def count(self) -> int:
        self._load_keys()
        return len(self._keys)

_cerebras_keys = _CerebrasKeyRotator()


def get_cerebras_key() -> str:
    """Get the next available Cerebras API key (rotated)."""
    return _cerebras_keys.get_key()


# ── IndicTrans2 / NLLB Singleton (Stage A: Meaning Model) ────────────────────
_meaning_model = None
_meaning_tokenizer = None
_meaning_processor = None  # IndicProcessor for IndicTrans2
_meaning_engine = None     # "indictrans2" or "nllb"
_meaning_lock = __import__("threading").Lock()


def _get_meaning_model():
    """Lazy-load the meaning model on GPU (singleton).
    Tries IndicTrans2 first (better quality), falls back to NLLB-1.3B (ungated).
    """
    global _meaning_model, _meaning_tokenizer, _meaning_processor, _meaning_engine
    if _meaning_model is not None:
        return _meaning_model, _meaning_tokenizer, _meaning_processor, _meaning_engine
    with _meaning_lock:
        if _meaning_model is not None:
            return _meaning_model, _meaning_tokenizer, _meaning_processor, _meaning_engine
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try IndicTrans2 first
        try:
            from IndicTransToolkit.processor import IndicProcessor
            model_name = "ai4bharat/indictrans2-en-indic-1B"
            print(f"[MeaningModel] Loading {model_name}...", flush=True)
            _meaning_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            _meaning_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float16
            ).to(device)
            _meaning_model.eval()
            _meaning_processor = IndicProcessor(inference=True)
            _meaning_engine = "indictrans2"
            vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            print(f"[MeaningModel] IndicTrans2 loaded. VRAM: {vram:.0f} MB", flush=True)
        except Exception as e:
            # Fallback to NLLB
            print(f"[MeaningModel] IndicTrans2 unavailable ({e}), falling back to NLLB...", flush=True)
            model_name = "facebook/nllb-200-1.3B"
            _meaning_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _meaning_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(device)
            _meaning_model.eval()
            _meaning_processor = None
            _meaning_engine = "nllb"
            vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            print(f"[MeaningModel] NLLB-1.3B loaded. VRAM: {vram:.0f} MB", flush=True)

        return _meaning_model, _meaning_tokenizer, _meaning_processor, _meaning_engine


# Language code mapping (ISO 639-1 → FLORES-200 / IndicTrans2)
_LANG_MAP = {
    "hi": "hin_Deva", "hi-IN": "hin_Deva",
    "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda",
    "ml": "mal_Mlym", "pa": "pan_Guru", "ur": "urd_Arab",
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "zh": "zho_Hans", "ar": "arb_Arab", "pt": "por_Latn",
    "ru": "rus_Cyrl", "it": "ita_Latn",
}


# ── Hindi Dubbing Rule Engine v1 ────────────────────────────────────────────
class HindiRuleEngine:
    """Post-processing rules for Hindi dubbing output."""

    # Formal/bookish words to flag or replace
    BANNED_FORMAL = {
        "अतः", "किन्तु", "तथापि", "उक्त", "यद्यपि", "एवं",
        "आवश्यकता", "विक्षिप्त", "अथवा", "तत्पश्चात", "अतएव",
        "निश्चितरूपेण", "सर्वप्रथम", "तदुपरांत", "यथा", "किंचित",
        "प्रतीत", "उपरोक्त", "निम्नलिखित", "संदर्भ",
    }

    # Replacements: formal → spoken
    FORMAL_TO_SPOKEN = {
        "किन्तु": "लेकिन",
        "परन्तु": "लेकिन",
        "अतः": "इसलिए",
        "एवं": "और",
        "अथवा": "या",
        "तथा": "और",
        "यद्यपि": "भले ही",
        "तथापि": "फिर भी",
        "आवश्यकता": "ज़रूरत",
        "सम्पूर्ण": "पूरा",
        "अत्यंत": "बहुत",
        "अत्यधिक": "बहुत ज़्यादा",
        "प्रारम्भ": "शुरू",
        "समाप्त": "खत्म",
        "उत्तर": "जवाब",
        "प्रश्न": "सवाल",
        "विद्यमान": "मौजूद",
        "कदापि": "कभी",
        "सर्वदा": "हमेशा",
    }

    # Protected terms (names, brands, powers) — user can extend
    _glossary: dict  # english_term -> hindi_term (or keep as-is)

    def __init__(self, glossary: dict = None):
        self._glossary = glossary or {}

    def apply(self, text: str, max_chars: int = 0) -> str:
        """Apply all rules to a translated Hindi line."""
        text = self._replace_formal(text)
        text = self._apply_glossary(text)
        text = self._normalize_punctuation(text)
        if max_chars > 0:
            text = self._compress(text, max_chars)
        return text.strip()

    def _replace_formal(self, text: str) -> str:
        for formal, spoken in self.FORMAL_TO_SPOKEN.items():
            text = text.replace(formal, spoken)
        return text

    def _apply_glossary(self, text: str) -> str:
        for eng, hi in self._glossary.items():
            text = text.replace(eng, hi)
        return text

    def _normalize_punctuation(self, text: str) -> str:
        # Normalize Hindi punctuation
        text = text.replace("।।", "।")
        text = text.replace("  ", " ")
        # Remove trailing spaces before punctuation
        text = re.sub(r'\s+([।!?,])', r'\1', text)
        return text

    def _compress(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        # Simple compression: remove filler words
        fillers = ["बस ", "तो ", "ना ", "जो कि ", "असल में ", "वास्तव में "]
        for filler in fillers:
            if len(text) <= max_chars:
                break
            text = text.replace(filler, "", 1)
        return text

    def count_formal_words(self, text: str) -> list:
        """Return list of banned formal words found in text."""
        return [w for w in self.BANNED_FORMAL if w in text]

    def score_naturalness(self, text: str) -> float:
        """Score 0-1 how natural/spoken the Hindi is (1=very natural)."""
        score = 1.0
        formal_count = len(self.count_formal_words(text))
        score -= formal_count * 0.15
        # Long sentences are harder to speak
        if len(text) > 200:
            score -= 0.1
        # Very long words suggest Sanskrit compounds
        words = text.split()
        long_words = [w for w in words if len(w) > 15]
        score -= len(long_words) * 0.05
        return max(0.0, min(1.0, score))


# Global rule engine instance
_hindi_rules = HindiRuleEngine()


# ── Types ────────────────────────────────────────────────────────────────────
ProgressCallback = Callable[[str, float, str], None]

STEPS = ["download", "extract", "transcribe", "translate", "synthesize", "assemble"]
STEP_WEIGHTS = {
    "download": 0.15,
    "extract": 0.05,
    "transcribe": 0.25,
    "translate": 0.15,
    "synthesize": 0.30,
    "assemble": 0.10,
}


LANGUAGE_NAMES = {
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "ur": "Urdu", "en": "English", "es": "Spanish",
    "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "it": "Italian", "tr": "Turkish",
}

# Average spoken words-per-minute by language for TTS duration estimation.
# Used to compute target word counts so translated segments fit original timing.
LANGUAGE_WPM = {
    "en": 150, "hi": 120, "bn": 120, "ta": 110, "te": 115,
    "mr": 120, "gu": 120, "kn": 110, "ml": 105, "pa": 120,
    "ur": 120, "es": 160, "fr": 155, "de": 130, "ja": 200,
    "ko": 140, "zh": 160, "pt": 155, "ru": 130, "ar": 125,
    "it": 155, "tr": 130,
}

DEFAULT_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "it": "it-IT-ElsaNeural",
    "tr": "tr-TR-EmelNeural",
    "bn": "bn-IN-TanishaaNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "pa-IN-GurpreetNeural",
    "ur": "ur-PK-UzmaNeural",
}

MALE_VOICES = {
    "hi": "hi-IN-MadhurNeural",
    "en": "en-US-GuyNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "zh": "zh-CN-YunxiNeural",
    "pt": "pt-BR-AntonioNeural",
    "ru": "ru-RU-DmitryNeural",
    "ar": "ar-SA-HamedNeural",
    "it": "it-IT-DiegoNeural",
    "tr": "tr-TR-AhmetNeural",
    "bn": "bn-IN-BashkarNeural",
    "ta": "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural",
    "mr": "mr-IN-ManoharNeural",
    "gu": "gu-IN-NiranjanNeural",
    "kn": "kn-IN-GaganNeural",
    "ml": "ml-IN-MidhunNeural",
    "pa": "pa-IN-GurpreetNeural",
    "ur": "ur-PK-AsadNeural",
}

# Pool of distinct voices per gender per language for multi-speaker
VOICE_POOL = {
    "en": {
        "female": ["en-US-JennyNeural", "en-US-AriaNeural", "en-US-SaraNeural"],
        "male":   ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-EricNeural"],
    },
    "hi": {
        "female": ["hi-IN-SwaraNeural"],
        "male":   ["hi-IN-MadhurNeural"],
    },
    "es": {
        "female": ["es-ES-ElviraNeural", "es-MX-DaliaNeural"],
        "male":   ["es-ES-AlvaroNeural", "es-MX-JorgeNeural"],
    },
    "fr": {
        "female": ["fr-FR-DeniseNeural", "fr-FR-EloiseNeural"],
        "male":   ["fr-FR-HenriNeural"],
    },
    "de": {
        "female": ["de-DE-KatjaNeural", "de-DE-AmalaNeural"],
        "male":   ["de-DE-ConradNeural", "de-DE-KillianNeural"],
    },
    "ja": {
        "female": ["ja-JP-NanamiNeural"],
        "male":   ["ja-JP-KeitaNeural"],
    },
    "ko": {
        "female": ["ko-KR-SunHiNeural"],
        "male":   ["ko-KR-InJoonNeural"],
    },
    "zh": {
        "female": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaohanNeural"],
        "male":   ["zh-CN-YunxiNeural", "zh-CN-YunjianNeural"],
    },
    "pt": {
        "female": ["pt-BR-FranciscaNeural"],
        "male":   ["pt-BR-AntonioNeural"],
    },
    "ru": {
        "female": ["ru-RU-SvetlanaNeural", "ru-RU-DariyaNeural"],
        "male":   ["ru-RU-DmitryNeural"],
    },
    "ar": {
        "female": ["ar-SA-ZariyahNeural"],
        "male":   ["ar-SA-HamedNeural"],
    },
    "it": {
        "female": ["it-IT-ElsaNeural", "it-IT-IsabellaNeural"],
        "male":   ["it-IT-DiegoNeural"],
    },
    "tr": {
        "female": ["tr-TR-EmelNeural"],
        "male":   ["tr-TR-AhmetNeural"],
    },
    "bn": {
        "female": ["bn-IN-TanishaaNeural"],
        "male":   ["bn-IN-BashkarNeural"],
    },
    "ta": {
        "female": ["ta-IN-PallaviNeural"],
        "male":   ["ta-IN-ValluvarNeural"],
    },
    "te": {
        "female": ["te-IN-ShrutiNeural"],
        "male":   ["te-IN-MohanNeural"],
    },
    "mr": {
        "female": ["mr-IN-AarohiNeural"],
        "male":   ["mr-IN-ManoharNeural"],
    },
    "gu": {
        "female": ["gu-IN-DhwaniNeural"],
        "male":   ["gu-IN-NiranjanNeural"],
    },
    "kn": {
        "female": ["kn-IN-SapnaNeural"],
        "male":   ["kn-IN-GaganNeural"],
    },
    "ml": {
        "female": ["ml-IN-SobhanaNeural"],
        "male":   ["ml-IN-MidhunNeural"],
    },
    "pa": {
        "female": ["pa-IN-GurpreetNeural"],
        "male":   ["pa-IN-GurpreetNeural"],
    },
    "ur": {
        "female": ["ur-PK-UzmaNeural"],
        "male":   ["ur-PK-AsadNeural"],
    },
}


@dataclass
class PipelineConfig:
    source: str
    work_dir: Path
    output_path: Path
    source_language: str = "en"   # English source by default — pipeline is English→Hindi
    target_language: str = "hi"
    asr_model: str = "groq-whisper"     # Groq Whisper (cloud) → falls back to local large-v3
    translation_engine: str = "google"  # Google Translate (parallel x20, fastest free)
    tts_voice: str = "hi-IN-SwaraNeural"
    tts_rate: str = "+0%"
    mix_original: bool = False   # SUSPENDED — always False until explicitly reactivated
    original_volume: float = 0.10
    use_cosyvoice: bool = False          # OFF: slow GPU, not needed for SRT upload
    use_chatterbox: bool = False
    use_fish_speech: bool = False
    use_indic_parler: bool = False
    use_sarvam_bulbul: bool = False     # OFF: Edge-TTS is fast enough for SRT upload
    use_elevenlabs: bool = False
    use_google_tts: bool = False
    use_coqui_xtts: bool = False        # OFF: slow GPU, not needed for SRT upload
    use_edge_tts: bool = True
    prefer_youtube_subs: bool = True     # ON: skip Whisper if YouTube has subs
    # ON by default (2026-04-12): YouTube's auto-translated Hindi is higher
    # quality than Google Translate API for narrative content. If 429 or
    # unavailable, cascade falls back to English subs + Google Translate.
    use_yt_translate: bool = True
    multi_speaker: bool = False
    transcribe_only: bool = False
    simplify_english: bool = False
    step_by_step: bool = False
    srt_needs_translation: bool = False
    # Audio priority: let TTS speak at natural pace, video slows to match
    audio_priority: bool = True
    audio_untouchable: bool = False
    # Video slow to match: if dubbed audio is longer, slow video uniformly to sync
    video_slow_to_match: bool = True
    post_tts_level: str = "minimal"       # minimal: speechnorm + fade. full: + compressor
    audio_quality_mode: str = "fast"     # "fast" (ffmpeg) or "quality" (librosa)
    enable_sentence_gap: bool = False    # OFF: no artificial gaps, continuous audio
    enable_duration_fit: bool = False    # OFF: assembly split-the-diff handles timing
    dub_duration: int = 0
    # Audio quality: bitrate for final output (128k, 192k, 256k, 320k)
    audio_bitrate: str = "192k"
    # Video encode speed: NVENC GPU encoding
    encode_preset: str = "fast"
    # Download mode for yt-dlp:
    #   "remux"  — current default. Uses --remux-video mp4. Instant container
    #              swap (no re-encode). ~2x faster but fails if YouTube serves
    #              codecs that aren't MP4-compatible (rare, but happens with
    #              some VP9/Opus combos on certain videos).
    #   "encode" — old behavior. Uses --merge-output-format mp4. Lets ffmpeg
    #              re-encode video+audio into MP4 if needed. Slower but ALWAYS
    #              works regardless of source codec compatibility. Use this if
    #              "remux" downloads keep failing on a specific video.
    download_mode: str = "remux"
    # Split long videos into parts (0 = no split, 30/40 = split every N minutes)
    split_duration: int = 30             # 30 min chunks for long videos (1h+)
    # Fast assemble: use in-memory bytearray (instant) vs ffmpeg adelay+amix (slower, preserves overlaps)
    fast_assemble: bool = False
    # Pronunciation dictionary: JSON file mapping source terms → target phonetic spellings
    # Example: {"Pikachu": "Pikachu", "GPT": "जी-पी-टी"}
    pronunciation_path: str = ""
    # Manual review queue: save JSON of segments that failed QC after all retries
    enable_manual_review: bool = False
    # WhisperX forced alignment: refine word-level timestamps after transcription
    use_whisperx: bool = True
    # TTS verify retry loop: re-generates segments that fail duration/energy check.
    # OFF by default — Hindi WPM variance causes 70%+ false positives, making the
    # loop take hours on long videos. Turn ON for short videos where you want the
    # extra safety net (catches genuine silent/truncated TTS outputs).
    enable_tts_verify_retry: bool = False
    # ── INLINE TTS truncation guard ──
    # Edge-TTS occasionally drops the WebSocket stream mid-response and saves
    # only the partial audio (first sentence only on Hindi text containing `।`).
    # After each save, we probe the actual MP3 duration and compare to a fast
    # lower-bound estimate (word_count * 60 / max_wpm). If actual < lower * threshold,
    # we treat it as a truncation and retry the whole call.
    #
    # Threshold semantics (0.0 - 1.0):
    #   0.00 = OFF, never retry (old behavior, lets truncations through)
    #   0.30 = default, catches catastrophic truncation only (very few false positives)
    #   0.50 = stricter, catches partial truncation (some false positives on slow speech)
    #   0.70 = aggressive, may false-positive on naturally fast Hindi speakers
    #   0.90 = paranoid, mostly false-positives, useful only for verification testing
    tts_truncation_threshold: float = 0.30
    # ── Keep noun subjects in source language ──
    # When True: every sentence's main NOUN subject (proper nouns, common nouns,
    # noun phrases like "the young warrior") is masked before translation and
    # restored verbatim afterward. Pronoun subjects (he, she, it, they, we, I,
    # you) are LEFT TO TRANSLATE NORMALLY so the text doesn't sound jarring.
    # Requires spaCy + en_core_web_sm — silently no-ops if not installed.
    keep_subject_english: bool = False
    # ── Post-TTS exact word-count verification (Whisper-based) ──
    # When True: after every segment is synthesized, run Whisper on the WAV
    # to transcribe it and count actual words. If actual is outside the
    # tolerance window (expected_words ± tts_word_match_tolerance), re-call
    # Edge-TTS for that segment up to 3 times.
    #
    # Cost on GPU (turbo via auto): ~80 ms per segment scan + retries.
    #   487-segment job  ≈ +1 minute
    #   5000-segment job ≈ +7-8 minutes
    # Cost on CPU (tiny via auto): ~150-300 ms per segment.
    #   487-segment job  ≈ +1.5 minutes
    #   5000-segment job ≈ +15-25 minutes
    #
    # ON by default — user explicitly requested exact word matching for every
    # segment. Set to False if you need maximum throughput on long videos.
    tts_word_match_verify: bool = True
    # Tolerance for the word-count match. 0.15 = ±15% wiggle room (default).
    # Lower = stricter, higher = more permissive. Even Whisper has ~10%
    # natural variance on Hindi (contractions, quiet syllables, number forms),
    # so 0.0 will cause false retries on segments that are actually fine.
    tts_word_match_tolerance: float = 0.15
    # Whisper model for the post-TTS verifier:
    #   "auto"  — large-v3-turbo on GPU (best accuracy + reasonable speed),
    #             tiny on CPU (only fast option without a GPU). DEFAULT.
    #   "tiny"  — Whisper-tiny everywhere. ~150 ms/seg on CPU, 30 ms on GPU.
    #             Lower accuracy (85-90% on Hindi) — faster on CPU.
    #   "turbo" — Whisper large-v3-turbo everywhere. ~80 ms/seg on GPU,
    #             ~1500 ms/seg on CPU. Higher accuracy (96-97% on Hindi).
    #             Use this only if you have a GPU; on CPU it's 5-10x slower
    #             than tiny with marginal accuracy gain for word counting.
    tts_word_match_model: str = "auto"
    # ── Long-segment trace watchdog ──
    # When True: every segment with > long_segment_threshold_words words has
    # its full pipeline lifecycle recorded (text characteristics, slot duration,
    # post-save file size, post-WAV duration, truncation guard verdict, word
    # verifier verdict, post-speed-fit duration, final assembly duration).
    # The report is written to backend/logs/long_segment_trace_<job_id>.json
    # at the end of each run so you can see EXACTLY where each long segment
    # was modified or dropped. Cheap (~microseconds per event), so always-on
    # by default — only the long segments are traced, not every segment.
    long_segment_trace: bool = True
    long_segment_threshold_words: int = 15
    # ── No time pressure on TTS ──
    # When True (default per user request): TTS produces full natural-pace
    # audio for every segment with NO slot/duration constraint. ALL of these
    # downstream pressure points are bypassed:
    #   1. _apply_base_speedup (the 1.15x global pre-speedup)
    #   2. _speed_fit_segments (the time-stretch clamp that was cutting words)
    #   3. _qc_check_wav duration_ratio failures (TTS won't be re-rendered just
    #      because the audio is "too long" for its slot)
    #   4. tts_rate is forced to "+0%" (no rate manipulation)
    #   5. audio_priority is forced True (assembly adapts video to audio)
    # The slot timing is then handled ENTIRELY in assembly via the
    # _assemble_video_adapts_to_audio path (which is the audio_priority path).
    tts_no_time_pressure: bool = True
    # Dynamic worker scaling for Edge-TTS — adjusts concurrency based on
    # observed failure rate. Starts at tts_dynamic_min, grows to tts_dynamic_max
    # when no failures, halves on rate-limit / WebSocket errors.
    tts_dynamic_workers: bool = True
    tts_dynamic_min: int = 10
    tts_dynamic_max: int = 120
    tts_dynamic_start: int = 30
    # ── Auto TTS rate mode ──
    # "manual"  — tts_rate from the UI slider is used as-is (old behavior)
    # "auto"    — after translation, compute the optimal tts_rate from the
    #             total word count and the source video duration so the
    #             dubbed output matches the original runtime. Capped at
    #             tts_rate_ceiling (default +50%). Above the cap, the cap
    #             is used and the remaining overflow is absorbed by video
    #             stretching in assembly (audio_priority mode).
    # Default: "auto" — per user request 2026-04-12.
    tts_rate_mode: str = "auto"
    # Maximum rate the auto mode is allowed to produce. User selected +50%
    # (1.5x) as the ceiling — beyond this, audio quality degrades and the
    # extra time is absorbed by video stretching instead of faster speech.
    tts_rate_ceiling: str = "+50%"
    # Natural speaking rate (WPM) used as the "1.0x" baseline for auto rate
    # math. Hindi conversational pace is ~130 WPM, narration is ~120, fast
    # banter is ~180. 130 is a safe default. Can be tuned per project.
    tts_rate_target_wpm: int = 130
    # ── AV Sync Modules ──
    av_sync_mode: str = "original"       # "original" | "capped" | "audio_first"
    max_audio_speedup: float = 1.30      # cap for "capped" mode
    min_video_speed: float = 0.70        # floor before flagging
    slot_verify: str = "off"             # "off" | "dry_run" | "auto_fix"
    use_global_stretch: bool = False     # uniform video slowdown
    global_stretch_speedup: float = 1.25 # TTS speedup for global stretch
    segmenter: str = "dp"                # "dp" | "sentence"
    segmenter_buffer_pct: float = 0.20   # Hindi expansion buffer
    max_sentences_per_cue: int = 2       # max sentences per segment
    # YouTube transcript structuring mode:
    #   "yt_timeline"      — YouTube text + YouTube timelines (fast, no Whisper)
    #   "whisper_timeline" — YouTube text + Whisper timestamps (precise, slower)
    yt_transcript_mode: str = "yt_timeline"
    yt_segment_mode: str = "sentence"    # "sentence" | "wordcount"
    yt_text_correction: bool = True      # correct Whisper text using YouTube subs
    yt_replace_mode: str = "diff"        # "full" (total replace) | "diff" (word-level fix)
    tts_chunk_words: int = 0             # 0=off, 4/8/12=chunk translated text before TTS
    gap_mode: str = "micro"              # "none" | "micro" | "full"


# ── Module-level spaCy loader (cached, lazy) ──
# Used by Pipeline._mask_noun_subjects_in_segments. Loaded once per process,
# cached in the global so we don't reload the model on every translation call.
_SPACY_NLP = None
_SPACY_LOAD_FAILED = False

def _get_spacy_nlp():
    """Return a cached spaCy English NLP pipeline, or None if not available."""
    global _SPACY_NLP, _SPACY_LOAD_FAILED
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if _SPACY_LOAD_FAILED:
        return None
    try:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm")
        print("[KEEP-SUBJ] spaCy en_core_web_sm loaded for noun-subject masking", flush=True)
        return _SPACY_NLP
    except Exception as e:
        print(f"[KEEP-SUBJ] spaCy not available: {e} — keep_subject_english "
              f"feature disabled. Install with: pip install spacy && "
              f"python -m spacy download en_core_web_sm", flush=True)
        _SPACY_LOAD_FAILED = True
        return None


class Pipeline:
    """Dubbing pipeline with translation and callback-based progress."""

    SAMPLE_RATE = 48000
    N_CHANNELS = 2

    def __init__(self, cfg: PipelineConfig, on_progress: Optional[ProgressCallback] = None,
                 cancel_check: Optional[Callable[[], bool]] = None,
                 pause_event=None):
        self.cfg = cfg
        # SUSPENDED: mix_original is permanently disabled until explicitly reactivated
        self.cfg.mix_original = False
        self._on_progress = on_progress or (lambda *_: None)
        self._cancel_check = cancel_check or (lambda: False)
        self._pause_event = pause_event
        self.paused_at: Optional[str] = None
        self.segments: List[Dict] = []
        self.video_title: str = ""
        self.qa_score: Optional[float] = None
        self._voice_map = None
        self._whisper_audio = None  # Lightweight 16kHz mono audio for transcription
        self._has_nvenc: Optional[bool] = None  # Cached NVENC availability
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)

        # ── Subprocess tracking for fast cancellation ─────────────────────
        # Every subprocess (yt-dlp, ffmpeg, aria2c, ...) is registered here
        # so cancel can immediately kill in-flight children instead of waiting
        # for a multi-minute encode to finish on its own.
        import threading as _th
        self._active_procs: set = set()
        self._procs_lock = _th.Lock()
        self._cancelled_killed = False  # set true after kill_all_procs runs

        # Load pronunciation dictionary (maps source terms → target phonetic spellings)
        self._pronunciation: Dict[str, str] = {}
        pron_path = cfg.pronunciation_path or str(Path(__file__).resolve().parent / "pronunciation.json")
        try:
            import json as _json
            p = Path(pron_path)
            if p.exists():
                self._pronunciation = _json.loads(p.read_text(encoding="utf-8"))
                print(f"[Pipeline] Loaded {len(self._pronunciation)} pronunciation entries from {p}", flush=True)
        except Exception as e:
            print(f"[Pipeline] pronunciation.json load error: {e}", flush=True)

        # Load translation glossary (English words to preserve/transliterate)
        self._glossary: Dict[str, str] = {}
        glossary_path = Path(__file__).resolve().parent / "translation_glossary.json"
        try:
            import json as _json_g
            if glossary_path.exists():
                gdata = _json_g.loads(glossary_path.read_text(encoding="utf-8"))
                self._glossary = {k.lower(): v for k, v in gdata.items() if not k.startswith("_")}
                if self._glossary:
                    print(f"[Pipeline] Loaded {len(self._glossary)} glossary entries "
                          f"(e.g. {list(self._glossary.items())[:3]})", flush=True)
        except Exception as e:
            print(f"[Pipeline] translation_glossary.json load error: {e}", flush=True)

        # Resolve executable paths
        self._ytdlp = self._find_executable("yt-dlp")
        self._ffmpeg = "ffmpeg"  # resolved in _ensure_ffmpeg

    def _check_cancelled(self):
        """Raise if the job has been cancelled. Also kills any in-flight
        subprocesses so the worker thread can exit promptly."""
        if self._cancel_check():
            self._kill_all_procs()
            raise RuntimeError("Job cancelled by user")

    def _progress_watchdog(self, step: str, watch_dir: "Path",
                           label: str = "Working",
                           total_bytes_hint: Optional[int] = None,
                           start_pct: float = 0.05,
                           end_pct: float = 0.90,
                           file_glob: str = "*"):
        """Context manager: spawns a daemon thread that polls watch_dir for
        file growth and emits live progress callbacks every ~1 second.

        Used to give the UI live percentages during long blocking subprocess
        calls (yt-dlp, aria2c, ffmpeg) which would otherwise show 5% frozen.

        Usage:
            with self._progress_watchdog("download", work_dir, label="Downloading"):
                self._run_proc([...yt-dlp...], capture_output=True)

        - If total_bytes_hint is given, progress is calculated as bytes/total.
        - Otherwise it just reports MB downloaded so far + speed (no %).
        - Stops automatically when the with-block exits.
        """
        import threading as _th
        import time as _time
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            stop_event = _th.Event()
            state = {"last_bytes": 0, "last_time": _time.time(), "speed_mb": 0.0}

            def _scan_size():
                total = 0
                try:
                    for f in watch_dir.glob(file_glob):
                        try:
                            if f.is_file():
                                total += f.stat().st_size
                        except OSError:
                            pass
                except OSError:
                    pass
                return total

            def _watch_loop():
                # Initial baseline (excludes pre-existing files like cached subs)
                baseline = _scan_size()
                state["last_bytes"] = baseline
                state["last_time"] = _time.time()
                while not stop_event.is_set():
                    if stop_event.wait(timeout=1.0):
                        break
                    try:
                        cur = _scan_size()
                        delta_bytes = cur - state["last_bytes"]
                        delta_time = _time.time() - state["last_time"]
                        if delta_time > 0.5 and delta_bytes >= 0:
                            speed = delta_bytes / delta_time / (1024 * 1024)
                            # Smooth speed (EMA-ish)
                            state["speed_mb"] = (
                                0.7 * state["speed_mb"] + 0.3 * speed
                                if state["speed_mb"] > 0 else speed
                            )
                            state["last_bytes"] = cur
                            state["last_time"] = _time.time()

                        downloaded_mb = (cur - baseline) / (1024 * 1024)
                        if total_bytes_hint and total_bytes_hint > 0:
                            frac = min(max((cur - baseline) / total_bytes_hint, 0.0), 1.0)
                            # Use the FULL 0.0–1.0 range for step_progress so
                            # _calc_overall computes the correct overall %.
                            # start_pct is now only a floor: while bytes are
                            # ramping up the bar shows ≥ start_pct so it doesn't
                            # look frozen at 0%.
                            pct = max(start_pct, frac)
                            total_mb = total_bytes_hint / (1024 * 1024)
                            remaining_mb = max(0.0, total_mb - downloaded_mb)
                            # ETA from smoothed speed (only if non-zero)
                            if state["speed_mb"] > 0.05 and remaining_mb > 0:
                                eta_sec = int(remaining_mb / state["speed_mb"])
                                if eta_sec >= 3600:
                                    eta_str = f"{eta_sec // 3600}h {(eta_sec % 3600) // 60}m"
                                elif eta_sec >= 60:
                                    eta_str = f"{eta_sec // 60}m {eta_sec % 60}s"
                                else:
                                    eta_str = f"{eta_sec}s"
                            else:
                                eta_str = "—"
                            msg = (f"{label}: {downloaded_mb:.0f} / {total_mb:.0f} MB "
                                   f"({frac*100:.0f}%) @ {state['speed_mb']:.1f} MB/s · "
                                   f"{remaining_mb:.0f} MB left · ETA {eta_str}")
                        else:
                            # No total known — show MB and speed only.
                            # step_progress floor at start_pct so the bar
                            # isn't pinned at 0%.
                            pct = start_pct
                            msg = (f"{label}: {downloaded_mb:.0f} MB "
                                   f"@ {state['speed_mb']:.1f} MB/s")
                        try:
                            self._report(step, pct, msg)
                        except Exception:
                            pass
                    except Exception:
                        pass

            thread = _th.Thread(target=_watch_loop, daemon=True,
                                name=f"watchdog-{step}")
            thread.start()
            try:
                yield
            finally:
                stop_event.set()
                try:
                    thread.join(timeout=2)
                except Exception:
                    pass

        return _ctx()

    def _kill_all_procs(self):
        """Forcefully kill every subprocess registered in self._active_procs.
        Safe to call multiple times. Used by both _check_cancelled() and
        external cancel handlers (cancel button in UI)."""
        with self._procs_lock:
            procs = list(self._active_procs)
            self._active_procs.clear()
        for p in procs:
            try:
                if p.poll() is None:  # still running
                    # On Windows, kill() sends TerminateProcess which is hard.
                    # On POSIX, this sends SIGKILL via os.kill.
                    p.kill()
            except Exception:
                pass
        # Give them a moment to die, then wait briefly so we don't leak
        # zombies. We don't block forever because the cancel must be fast.
        for p in procs:
            try:
                p.wait(timeout=2)
            except Exception:
                pass
        self._cancelled_killed = True

    def _run_proc(self, cmd, *, check=False, capture_output=False, text=False,
                  timeout=None, encoding=None, errors=None, input=None,
                  cwd=None, env=None):
        """Drop-in replacement for self._run_proc() that:
          1. Registers the child in self._active_procs so cancel can kill it
          2. Polls the cancel flag while waiting (instant cancel during long
             yt-dlp / ffmpeg runs instead of waiting for the next checkpoint)
          3. Returns a CompletedProcess identical to self._run_proc()
        """
        # Fast-cancel before even spawning
        if self._cancel_check():
            raise RuntimeError("Job cancelled by user")

        stdout_dest = subprocess.PIPE if capture_output else None
        stderr_dest = subprocess.PIPE if capture_output else None
        stdin_dest = subprocess.PIPE if input is not None else None

        # Hide the console window on Windows so subprocesses don't flash
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        proc = subprocess.Popen(
            cmd,
            stdin=stdin_dest,
            stdout=stdout_dest,
            stderr=stderr_dest,
            text=text,
            encoding=encoding,
            errors=errors,
            cwd=cwd,
            env=env,
            creationflags=creationflags,
        )
        with self._procs_lock:
            self._active_procs.add(proc)

        try:
            # Poll-based wait so we can react to cancel within ~250ms
            # instead of blocking for the full subprocess duration.
            stdout_data, stderr_data = None, None
            poll_interval = 0.25
            elapsed = 0.0

            if input is not None:
                # communicate() blocks once. Acceptable for short stdin uses.
                try:
                    stdout_data, stderr_data = proc.communicate(
                        input=input, timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise
            else:
                # Drain pipes in background threads to prevent deadlock when
                # capture_output=True — the OS pipe buffer is small (~4-64 KB)
                # and ffmpeg/yt-dlp can fill it, blocking the child forever.
                import threading, io
                _out_chunks: list = []
                _err_chunks: list = []

                def _drain(stream, sink):
                    try:
                        while True:
                            chunk = stream.read(8192)
                            if not chunk:
                                break
                            sink.append(chunk)
                    except Exception:
                        pass

                drain_threads = []
                if proc.stdout:
                    t = threading.Thread(target=_drain, args=(proc.stdout, _out_chunks), daemon=True)
                    t.start()
                    drain_threads.append(t)
                if proc.stderr:
                    t = threading.Thread(target=_drain, args=(proc.stderr, _err_chunks), daemon=True)
                    t.start()
                    drain_threads.append(t)

                while True:
                    try:
                        rc = proc.wait(timeout=poll_interval)
                        break  # exited normally
                    except subprocess.TimeoutExpired:
                        elapsed += poll_interval
                        if self._cancel_check():
                            try:
                                proc.kill()
                            except Exception:
                                pass
                            try:
                                proc.wait(timeout=2)
                            except Exception:
                                pass
                            raise RuntimeError("Job cancelled by user")
                        if timeout is not None and elapsed >= timeout:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                            raise subprocess.TimeoutExpired(cmd, timeout)

                # Wait for drain threads to finish reading remaining data
                for t in drain_threads:
                    t.join(timeout=5)

                if capture_output:
                    # When text=True or encoding is set, Popen opens pipes in
                    # text mode → chunks are str.  Otherwise they are bytes.
                    is_text_mode = bool(text or encoding)
                    joiner = "" if is_text_mode else b""
                    stdout_data = joiner.join(_out_chunks) if _out_chunks else None
                    stderr_data = joiner.join(_err_chunks) if _err_chunks else None
        finally:
            with self._procs_lock:
                self._active_procs.discard(proc)

        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout_data,
            stderr=stderr_data,
        )
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd,
                output=stdout_data, stderr=stderr_data,
            )
        return result

    def _apply_pronunciation(self, text: str) -> str:
        """Apply pronunciation dictionary to translated text before TTS.

        Replaces proper nouns, abbreviations, and brand names with
        phonetically correct spellings for the target language.
        Longest match wins (sorted by length descending to avoid partial replacements).
        Keys starting with '_' are treated as comments and skipped.
        """
        if not self._pronunciation:
            return text
        entries = {k: v for k, v in self._pronunciation.items()
                   if not k.startswith("_") and isinstance(v, str)}
        for src in sorted(entries, key=len, reverse=True):
            if src in text:
                text = text.replace(src, entries[src])
        return text

    @staticmethod
    def _find_executable(name: str) -> str:
        """Find an executable by checking venv, PATH, WinGet packages, and system PATH."""
        ext = ".exe" if sys.platform == "win32" else ""
        full_name = name + ext

        def _works(path: str) -> bool:
            """Verify executable actually runs (catches broken venv shims)."""
            try:
                r = self._run_proc([path, "--version"], capture_output=True, timeout=5)
                return r.returncode == 0 and (r.stdout or r.stderr)
            except Exception:
                return False

        # 1. Check venv Scripts dir (where python.exe lives)
        venv_path = Path(sys.executable).parent / full_name
        if venv_path.exists() and _works(str(venv_path)):
            return str(venv_path)

        # 2. Check current PATH (verify it works to skip broken shims)
        found = shutil.which(name)
        if found and _works(found):
            return found

        # 2b. Check Python's own Scripts directory (pip-installed tools)
        scripts_path = Path(sys.executable).parent / "Scripts" / full_name
        if scripts_path.exists() and _works(str(scripts_path)):
            os.environ["PATH"] = str(scripts_path.parent) + os.pathsep + os.environ.get("PATH", "")
            return str(scripts_path)

        if sys.platform == "win32":
            # 3. Scan WinGet packages directory
            localappdata = os.environ.get("LOCALAPPDATA", "")
            if not localappdata:
                userprofile = os.environ.get("USERPROFILE", str(Path.home()))
                localappdata = str(Path(userprofile) / "AppData" / "Local")
            if localappdata:
                winget_pkgs = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
                if winget_pkgs.exists():
                    for exe in winget_pkgs.rglob(full_name):
                        os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
                        return str(exe)

            # 4. Check well-known Windows installation directories
            well_known_dirs = [
                Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")),
                Path(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)")),
                Path(os.environ.get("LOCALAPPDATA", "")),
            ]
            # Map executable names to their typical install subdirectories
            known_subdirs = {
                "node": ["nodejs"],
                "ffmpeg": ["ffmpeg", "ffmpeg\\bin"],
            }
            for base_dir in well_known_dirs:
                if not base_dir or not base_dir.exists():
                    continue
                for subdir in known_subdirs.get(name, []):
                    candidate = base_dir / subdir / full_name
                    if candidate.exists() and _works(str(candidate)):
                        os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
                        return str(candidate)

            # 5. Refresh PATH from system registry and try again
            try:
                result = self._run_proc(
                    ["powershell.exe", "-NoProfile", "-Command",
                     "[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    os.environ["PATH"] = result.stdout.strip() + os.pathsep + os.environ.get("PATH", "")
                    found = shutil.which(name)
                    if found and _works(found):
                        return found
            except Exception:
                pass

        return name  # fallback to bare name

    def _report(self, step: str, progress: float, message: str):
        """Report progress to callback.

        Auto-prepends a numeric percentage to the message so the user can see
        exact progress at a glance (e.g. '[34%] Synthesized 600/5433 segments').
        If the message already contains a percent token, we leave it alone.
        """
        clamped = min(max(progress, 0.0), 1.0)
        try:
            # Skip the prefix if the message already shows a percentage,
            # so we don't end up with "[34%] ... (34%)".
            if message and "%" not in message:
                message = f"[{int(round(clamped * 100))}%] {message}"
        except Exception:
            pass
        self._on_progress(step, clamped, message)

    # ── Emotion detection ────────────────────────────────────────────────
    # Keyword lists for heuristic emotion tagging
    _EMOTION_PUNCHY_WORDS = frozenset([
        "fight", "attack", "kill", "war", "battle", "danger", "power", "win", "lose",
        "destroy", "epic", "finally", "reveal", "shock", "now", "stop", "never",
        "defeat", "enemy", "blood", "scream", "boss", "rank", "unlock", "level",
        # Hindi equivalents
        "लड़ाई", "हमला", "शक्ति", "जीत", "हार", "ताकत", "खतरा", "रहस्य", "अब",
    ])
    _EMOTION_EMOTIONAL_WORDS = frozenset([
        "love", "miss", "cry", "sad", "heart", "please", "sorry", "alone", "die",
        "remember", "dream", "hope", "fear", "truth", "promise", "mother", "father",
        "friend", "lost", "pain", "hurt", "tears", "farewell", "goodbye",
        # Hindi equivalents
        "प्यार", "याद", "रोना", "दुख", "दिल", "माफ", "सच", "वादा", "अकेला", "दर्द",
    ])
    _EMOTION_COMEDIC_WORDS = frozenset([
        "funny", "joke", "laugh", "silly", "ridiculous", "seriously", "really",
        "unbelievable", "wait", "what", "huh", "dude", "bro", "man",
        # Hindi equivalents
        "मज़ाक", "हँसी", "यार", "भाई", "सच में", "अरे",
    ])

    def _detect_segment_emotion(self, seg: Dict) -> str:
        """Heuristic emotion tagger. Returns 'neutral' | 'punchy' | 'emotional' | 'comedic'.

        Uses punctuation, duration, and keyword signals. No extra API call.
        Order of precedence: comedic > emotional > punchy > neutral.
        """
        text = (seg.get("text", "") + " " + seg.get("text_translated", "")).lower()
        dur = seg.get("end", 0) - seg.get("start", 0)

        # Comedic: questions + common comedy markers
        q_count = text.count("?")
        has_comedic_kw = any(w in text for w in self._EMOTION_COMEDIC_WORDS)
        if q_count >= 2 or (q_count >= 1 and has_comedic_kw):
            return "comedic"

        # Emotional: ellipses / slow pace / emotional keywords
        has_ellipsis = "..." in text or "…" in text
        has_emotional_kw = any(w in text for w in self._EMOTION_EMOTIONAL_WORDS)
        if has_emotional_kw or (has_ellipsis and dur > 3.0):
            return "emotional"

        # Punchy: exclamations / short duration / action keywords
        excl = text.count("!")
        has_punchy_kw = any(w in text for w in self._EMOTION_PUNCHY_WORDS)
        if excl >= 1 or has_punchy_kw or dur < 2.0:
            return "punchy"

        return "neutral"

    # ── Speaker Diarization ───────────────────────────────────────────────

    def _diarize(self, wav_path: Path) -> tuple:
        """Run pyannote speaker diarization.
        Returns (speaker_genders, speaker_ranges) or ({}, {}) on failure.
        """
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not hf_token:
            self._report("transcribe", 0.85, "HF_TOKEN not set — skipping speaker diarization")
            return {}, {}

        try:
            from pyannote.audio import Pipeline as PyannotePipeline
        except ImportError:
            self._report("transcribe", 0.85, "pyannote-audio not installed — skipping diarization")
            return {}, {}

        try:
            self._report("transcribe", 0.82, "Loading speaker diarization model...")
            diarize_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )

            # Move to GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    diarize_pipeline.to(torch.device("cuda"))
            except Exception:
                pass

            self._report("transcribe", 0.86, "Running speaker diarization...")
            # Heartbeat: pyannote processes the entire audio in one blocking
            # call with no progress hooks. On long videos this looks like a
            # 5-15 minute freeze. Spawn a daemon thread that emits a _report
            # tick every 10 seconds so the UI knows we're still alive.
            import threading as _th
            import time as _time
            _heartbeat_stop = _th.Event()
            def _heartbeat():
                t0 = _time.time()
                tick = 0
                while not _heartbeat_stop.is_set():
                    if _heartbeat_stop.wait(timeout=10.0):
                        break
                    tick += 1
                    elapsed = int(_time.time() - t0)
                    try:
                        self._report("transcribe", 0.86,
                                     f"Diarizing speakers... ({elapsed}s elapsed)")
                    except Exception:
                        pass
            _hb_thread = _th.Thread(target=_heartbeat, daemon=True,
                                    name="diarize-heartbeat")
            _hb_thread.start()
            try:
                diarization = diarize_pipeline(str(wav_path))
            finally:
                _heartbeat_stop.set()

            # Extract unique speakers and their time ranges
            speaker_ranges: Dict[str, List[tuple]] = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_ranges:
                    speaker_ranges[speaker] = []
                speaker_ranges[speaker].append((turn.start, turn.end))

            if not speaker_ranges:
                return {}, {}

            self._report("transcribe", 0.92,
                         f"Found {len(speaker_ranges)} speakers, detecting genders...")

            # Detect gender via pitch analysis
            speaker_genders = self._detect_speaker_genders(wav_path, speaker_ranges)
            self._report("transcribe", 0.98,
                         f"Speakers: {', '.join(f'{k}={v}' for k, v in speaker_genders.items())}")
            return speaker_genders, speaker_ranges

        except Exception as e:
            self._report("transcribe", 0.85,
                         f"Diarization failed ({e}) — using single voice")
            return {}, {}

    def _detect_speaker_genders(self, wav_path: Path, speakers: Dict[str, List[tuple]]) -> Dict[str, str]:
        """Detect gender per speaker using pitch (F0) analysis. Male < 165Hz, Female >= 165Hz.
        Reads only needed time ranges to avoid OOM on long videos."""
        import struct

        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            max_val = float(2 ** (8 * sample_width - 1))
            fmt_char = "h" if sample_width == 2 else "i"

            result = {}
            for speaker, time_ranges in speakers.items():
                speaker_samples: list = []
                for t_start, t_end in time_ranges[:10]:
                    s_start = max(0, min(int(t_start * sample_rate), n_frames - 1))
                    s_end = max(0, min(int(t_end * sample_rate), n_frames))
                    count = s_end - s_start
                    if count < 1:
                        continue
                    wf.setpos(s_start)
                    raw = wf.readframes(count)
                    try:
                        chunk = struct.unpack(f"<{count * n_channels}{fmt_char}", raw)
                    except struct.error:
                        continue
                    if n_channels > 1:
                        chunk = chunk[::n_channels]
                    speaker_samples.extend(s / max_val for s in chunk)

                if len(speaker_samples) < sample_rate * 0.5:
                    result[speaker] = "female"
                    continue

                pitch = self._estimate_pitch_autocorrelation(speaker_samples, sample_rate)
                result[speaker] = "male" if pitch < 165 else "female"

        return result

    def _estimate_pitch_autocorrelation(self, samples: list, sample_rate: int) -> float:
        """Lightweight autocorrelation pitch estimator. Returns average F0 in Hz."""
        window_size = int(0.03 * sample_rate)  # 30ms windows
        hop = window_size // 2
        min_lag = int(sample_rate / 350)  # Max 350Hz
        max_lag = int(sample_rate / 60)   # Min 60Hz

        pitches = []
        for start in range(0, len(samples) - window_size, hop * 4):  # Skip windows for speed
            window = samples[start:start + window_size]
            # Simple energy check — skip silence
            energy = sum(s * s for s in window) / len(window)
            if energy < 0.001:
                continue

            # Autocorrelation for pitch detection
            best_lag = min_lag
            best_corr = -1.0
            for lag in range(min_lag, min(max_lag, len(window))):
                corr = 0.0
                for j in range(len(window) - lag):
                    corr += window[j] * window[j + lag]
                corr /= (len(window) - lag)
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

            if best_corr > energy * 0.3:  # Confidence threshold
                pitches.append(sample_rate / best_lag)

        if not pitches:
            return 200.0  # Default to ambiguous range

        # Return median pitch
        pitches.sort()
        return pitches[len(pitches) // 2]

    def _assign_speaker_to_segments(self, segments: List[Dict], diarization_speakers: Dict[str, List[tuple]]):
        """Assign speaker labels to transcription segments by max temporal overlap."""
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            best_speaker = None
            best_overlap = 0.0

            for speaker, time_ranges in diarization_speakers.items():
                overlap = 0.0
                for t_start, t_end in time_ranges:
                    ov_start = max(seg_start, t_start)
                    ov_end = min(seg_end, t_end)
                    if ov_end > ov_start:
                        overlap += ov_end - ov_start

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            seg["speaker_id"] = best_speaker or "SPEAKER_00"

    def _assign_voices_to_speakers(self, speaker_genders: Dict[str, str]) -> Dict[str, str]:
        """Map each speaker to a distinct Edge-TTS voice from VOICE_POOL."""
        lang = self.cfg.target_language
        pool = VOICE_POOL.get(lang, {})
        female_voices = list(pool.get("female", [DEFAULT_VOICES.get(lang, "en-US-JennyNeural")]))
        male_voices = list(pool.get("male", [MALE_VOICES.get(lang, "en-US-GuyNeural")]))

        voice_map = {}
        female_idx = 0
        male_idx = 0

        for speaker, gender in sorted(speaker_genders.items()):
            if gender == "male":
                voice_map[speaker] = male_voices[male_idx % len(male_voices)]
                male_idx += 1
            else:
                voice_map[speaker] = female_voices[female_idx % len(female_voices)]
                female_idx += 1

        return voice_map

    def _save_speaker_refs(self, audio_raw: Path, segments: List[Dict]):
        """Extract and save per-speaker voice reference clips for Coqui XTTS.

        Combines multiple short segments from each speaker into a single
        10-15 second reference clip for voice cloning.
        """
        refs_dir = self.cfg.work_dir / "speaker_refs"
        refs_dir.mkdir(exist_ok=True)

        # Group segments by speaker
        speaker_segs: Dict[str, List[Dict]] = {}
        for seg in segments:
            spk = seg.get("speaker_id")
            if spk:
                speaker_segs.setdefault(spk, []).append(seg)

        if not speaker_segs:
            return

        self._report("transcribe", 0.92,
                     f"Saving voice references for {len(speaker_segs)} speakers...")

        for spk, segs in speaker_segs.items():
            # Sort by start time and collect up to ~15 seconds of audio
            segs_sorted = sorted(segs, key=lambda s: s["start"])
            total_dur = 0.0
            clip_parts = []
            concat_txt = refs_dir / f"{spk}_concat.txt"

            for seg in segs_sorted:
                if total_dur >= 15.0:
                    break
                dur = seg["end"] - seg["start"]
                if dur < 0.5:
                    continue
                # Extract this segment's audio
                clip_path = refs_dir / f"{spk}_clip_{len(clip_parts):03d}.wav"
                try:
                    self._run_proc(
                        [self._ffmpeg, "-y",
                         "-i", str(audio_raw),
                         "-ss", f"{seg['start']:.3f}",
                         "-t", f"{dur:.3f}",
                         "-ar", "22050", "-ac", "1",
                         str(clip_path)],
                        check=True, capture_output=True,
                    )
                    clip_parts.append(clip_path)
                    total_dur += dur
                except Exception:
                    continue

            if not clip_parts:
                continue

            # Concatenate clips into one reference file
            ref_path = refs_dir / f"{spk}.wav"
            if len(clip_parts) == 1:
                shutil.copy2(clip_parts[0], ref_path)
            else:
                concat_txt.write_text(
                    "\n".join(f"file '{str(c).replace(chr(92), '/')}'" for c in clip_parts),
                    encoding="utf-8",
                )
                try:
                    self._run_proc(
                        [self._ffmpeg, "-y",
                         "-f", "concat", "-safe", "0",
                         "-i", str(concat_txt),
                         "-ar", "22050", "-ac", "1",
                         str(ref_path)],
                        check=True, capture_output=True,
                    )
                except Exception:
                    # Fall back to just the first clip
                    shutil.copy2(clip_parts[0], ref_path)

            # Clean up individual clips
            for c in clip_parts:
                c.unlink(missing_ok=True)
            concat_txt.unlink(missing_ok=True)

        saved = list(refs_dir.glob("SPEAKER_*.wav"))
        self._report("transcribe", 0.95,
                     f"Saved {len(saved)} speaker voice references for XTTS voice cloning")

    # ── Caching helpers ─────────────────────────────────────────────────
    def _save_segments_cache(self, segments: List[Dict], name: str):
        """Save segments to a JSON cache file in work_dir."""
        import json
        cache_path = self.cfg.work_dir / f"_cache_{name}.json"
        # Only serialize JSON-safe keys (skip Path objects etc.)
        safe = []
        for seg in segments:
            s = {}
            for k, v in seg.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    s[k] = v
                elif isinstance(v, list):
                    s[k] = v
            safe.append(s)
        cache_path.write_text(json.dumps(safe, ensure_ascii=False), encoding="utf-8")

    def _load_segments_cache(self, name: str) -> Optional[List[Dict]]:
        """DISABLED — user requested no reuse. Always returns None so every job
        re-runs every step from scratch."""
        return None

    def _find_cached_video(self) -> Optional[Path]:
        """Check if a video file was already downloaded in work_dir."""
        for ext in ["mp4", "webm", "mkv"]:
            p = self.cfg.work_dir / f"source.{ext}"
            if p.exists() and p.stat().st_size > 0:
                return p
        sources = list(self.cfg.work_dir.glob("source.*"))
        for s in sources:
            if s.suffix.lower() in (".mp4", ".webm", ".mkv", ".avi", ".mov") and s.stat().st_size > 0:
                return s
        return None

    # ── Main entry ───────────────────────────────────────────────────────
    def run(self):
        """Execute the full dubbing pipeline with step caching for crash recovery."""
        self._ensure_ffmpeg()

        # Step 1: Download / ingest — cache: source.mp4 exists
        video_path = self._find_cached_video()
        if video_path:
            self._report("download", 1.0, f"[cached] Using existing: {video_path.name}")
            # Recover title if possible
            if not self.video_title:
                self.video_title = video_path.stem
        else:
            self._report("download", 0.0, "Downloading video...")
            video_path = self._ingest_source(self.cfg.source)
            self._report("download", 1.0, f"Downloaded: {video_path.name}")

        # ── Measure source duration for auto rate mode ──
        # Auto rate mode needs the original source duration to compute the
        # speedup factor that will make the dubbed output match the source.
        # Probe ONCE here (cheap ffprobe call) and stash on self for
        # _tts_edge to read later.
        try:
            self._source_video_duration = self._get_duration(video_path)
            if self._source_video_duration > 0:
                print(f"[RATE-AUTO] Source video duration: "
                      f"{self._source_video_duration:.1f}s "
                      f"({self._source_video_duration / 60:.2f} min)", flush=True)
        except Exception as e:
            print(f"[RATE-AUTO] Could not probe source duration: {e}", flush=True)
            self._source_video_duration = 0.0

        self._check_cancelled()

        # Step 2: Extract audio — cache: audio_raw.wav exists
        audio_raw = self.cfg.work_dir / "audio_raw.wav"
        wav_16k = self.cfg.work_dir / "audio_16k.wav"
        if audio_raw.exists() and audio_raw.stat().st_size > 0:
            self._report("extract", 1.0, "[cached] Audio already extracted")
            if wav_16k.exists():
                self._whisper_audio = wav_16k
        else:
            self._report("extract", 0.0, "Extracting audio...")
            audio_raw = self._extract_audio(video_path)
            self._report("extract", 1.0, "Audio extracted")

        self._check_cancelled()

        # Step 3: Transcribe — cache: _cache_transcribe.json exists
        sub_segments = None
        cached_segs = self._load_segments_cache("transcribe")
        if cached_segs:
            self.segments = cached_segs
            text_segments = [s for s in self.segments if s.get("text", "").strip()]
            self._report("transcribe", 1.0,
                         f"[cached] Loaded {len(text_segments)} transcribed segments")
        else:
            sub_segments = None

            # ── YT Auto-Translate fast-fast path: skip Whisper AND translation ──
            # If user enabled use_yt_translate, fetch YouTube's pre-translated
            # subs in the target language directly. Whisper never runs.
            if self.cfg.use_yt_translate and re.match(r"^https?://", self.cfg.source):
                self._report("transcribe", 0.0,
                             "YT Auto-Translate ON — fetching YouTube's pre-translated subs (skipping Whisper)...")
                yt_translated_subs = self._fetch_youtube_translated_subs(self.cfg.source)
                if yt_translated_subs:
                    # These segments already have text_translated populated.
                    # Mark them so the fast path also skips translation.
                    for seg in yt_translated_subs:
                        seg.setdefault("_qc_issues", [])
                        seg.setdefault("_protected_terms", [])
                        seg.setdefault("emotion", "neutral")
                        seg.setdefault("text_en_clean", seg.get("text", ""))
                        # Ensure 'text' field is also set so downstream filters work
                        if not seg.get("text") and seg.get("text_translated"):
                            seg["text"] = seg["text_translated"]
                    self.segments = yt_translated_subs
                    sub_segments = yt_translated_subs  # mark for fast path below
                    self._yt_subs_fast_path = True
                    self._yt_already_translated = True  # tells _run_from_step4 to skip translation
                    self._report("transcribe", 1.0,
                                 f"YT Auto-Translate: {len(yt_translated_subs)} pre-translated segments — skipping Whisper AND translation")
                else:
                    self._report("transcribe", 0.05,
                                 "No YT translated subs found -- falling back to Whisper + our translator")
                # Mark that we already attempted, so step 4 doesn't retry
                self._yt_translate_attempted = True

            # ── CASCADE FALLBACK: English subs ──
            # If Hindi auto-translate failed (or wasn't attempted), try English
            # subs. This fires when EITHER:
            #   - prefer_youtube_subs=True (user explicitly chose English subs)
            #   - use_yt_translate=True but Hindi download failed (automatic
            #     fallback — the UI makes these toggles mutually exclusive,
            #     but the backend cascade should still fall through)
            # This way the user only needs to turn ON "YT Auto-Translate"
            # and the cascade handles Hindi -> English -> Whisper automatically.
            _try_english_subs = (
                self.cfg.prefer_youtube_subs
                or (self.cfg.use_yt_translate and not sub_segments)
            )
            if not sub_segments and _try_english_subs:
                self._report("transcribe", 0.0,
                             "Checking for YouTube English subtitles (cascade fallback)...")
                sub_segments = self._fetch_youtube_subtitles(self.cfg.source)

            if sub_segments:
                self.segments = sub_segments
                # Set fields for downstream compatibility + skip flag
                for seg in sub_segments:
                    seg.setdefault("_qc_issues", [])
                    seg.setdefault("_protected_terms", [])
                    seg.setdefault("emotion", "neutral")
                    seg.setdefault("text_en_clean", seg.get("text", ""))
                self._yt_subs_fast_path = True  # Skip ALL English processing
                self._report("transcribe", 1.0,
                             f"YouTube subs: {len(sub_segments)} segments — skipping English processing")
            else:
                if self.cfg.prefer_youtube_subs:
                    self._report("transcribe", 0.05, "No subtitles found, using Whisper...")
                self._report("transcribe", 0.1, "Loading ASR model...")
                whisper_audio = self._whisper_audio or audio_raw
                self.segments = self._transcribe(whisper_audio)
                self._report("transcribe", 1.0, f"Transcribed {len(self.segments)} segments")

            text_segments = [s for s in self.segments if s.get("text", "").strip()]
            if not text_segments:
                raise RuntimeError("No speech detected in the video")

            # Cache transcription for crash recovery
            self._save_segments_cache(text_segments, "transcribe")

        text_segments = [s for s in self.segments if s.get("text", "").strip()]
        if not text_segments:
            raise RuntimeError("No speech detected in the video")

        # ── YouTube subs fast path: skip ALL English processing to Step 4 ──
        if getattr(self, '_yt_subs_fast_path', False):
            self._ref_english_subs = None
            self._voice_map = None
            self._keyterms = {}

            # ── CRITICAL: merge YouTube SRT fragments into complete sentences ──
            # YouTube's raw SRT has ~192 tiny 2-4 second chunks split mid-sentence
            # (e.g., "also a white tiger, but he came out as a"). Without merging,
            # these fragments go directly to TTS and produce choppy audio.
            # _merge_broken_sentences is the same step that runs on Whisper output
            # in the main flow — we just need to also run it here.
            pre_merge = len(text_segments)
            yt_sentences = self._merge_broken_sentences(text_segments)
            if len(yt_sentences) < pre_merge:
                self._report("transcribe", 0.80,
                             f"Merged YouTube SRT fragments: {pre_merge} -> {len(yt_sentences)} sentences")

            # ── YouTube Transcript Mode: structure into Whisper-style segments ──
            _yt_mode = getattr(self.cfg, 'yt_transcript_mode', 'yt_timeline')

            if _yt_mode == "whisper_timeline":
                # ═══ OPTION 2: YouTube text + Whisper timeline ═══
                # Run Whisper for precise speech timestamps, then replace its
                # text with YouTube's (better quality). Slower but exact timelines.
                self._report("transcribe", 0.82,
                             "Option 2: Running Whisper for precise timestamps...")
                whisper_audio = self._whisper_audio or audio_raw
                whisper_raw = self._transcribe(whisper_audio)
                whisper_merged = self._merge_broken_sentences(whisper_raw)
                self._report("transcribe", 0.90,
                             f"Whisper: {len(whisper_raw)} raw -> {len(whisper_merged)} "
                             f"merged. Aligning YouTube text ({len(yt_sentences)} sentences)...")

                # Align: YouTube text + Whisper timestamps
                text_segments = self._align_yt_text_to_whisper_timeline(
                    yt_sentences, whisper_merged)

                # Merge again (alignment may have created fragments)
                text_segments = self._merge_broken_sentences(text_segments)
                self._report("transcribe", 0.92,
                             f"Aligned: {len(text_segments)} segments (Whisper timeline + YT text)")
            else:
                # ═══ OPTION 1: YouTube text + YouTube timeline (default) ═══
                # Fast path: no Whisper needed. YouTube's own timelines are used.
                text_segments = yt_sentences

            # ── Segment split mode: user choice ──
            _seg_mode = getattr(self.cfg, 'yt_segment_mode', 'sentence')

            # Safety: if user chose "sentence" but captions have no punctuation
            # (avg segment > 8s after merge), auto-fallback to wordcount.
            if _seg_mode == "sentence" and text_segments:
                avg_dur = sum(
                    s.get("end", 0) - s.get("start", 0) for s in text_segments
                ) / len(text_segments)
                if avg_dur > 8.0:
                    _seg_mode = "wordcount"
                    self._report("transcribe", 0.91,
                                 f"No punctuation detected (avg seg {avg_dur:.1f}s) "
                                 f"-> auto-switching to word-count split")

            # ── Total word count (across all sentences/segments) ──
            total_words = sum(len(s.get("text", "").split()) for s in text_segments)
            total_sents = len(text_segments)
            _unit = "sentences" if _seg_mode == "sentence" else "chunks"
            self._report("transcribe", 0.93,
                         f"Inventory: {total_sents} {_unit}, "
                         f"{total_words} total words, "
                         f"avg {total_words / max(total_sents, 1):.1f} words/{_unit[:-1]}")

            if _seg_mode == "sentence":
                # ── SENTENCE SPLIT: group 2 sentences per segment ──
                # Sentences are atomic — never split. Orphan merges into previous.
                max_per_cue = getattr(self.cfg, 'max_sentences_per_cue', 2)
                text_segments = self._group_sentences_by_count(
                    text_segments, target_per_group=max_per_cue)
            else:
                # ── WORD COUNT SPLIT: ~20 words per segment (uniform) ──
                # Join all text, split into even segments by word count.
                # Gaps removed anyway -> uniform word density = uniform speed.
                target_words_per_seg = 20  # ~2 sentences worth
                text_segments = self._split_by_even_wordcount(
                    text_segments, target_words_per_seg)

            # ── Redistribute slot timelines by word count ──
            # Each segment gets time proportional to its word count. Segments
            # with more words get more time -> matches what TTS will produce.
            text_segments = self._redistribute_slots_by_wordcount(text_segments)

            self.segments = text_segments

            from srt_utils import write_srt as _fast_ws
            src_srt = self.cfg.work_dir / "transcript_source.srt"
            if not src_srt.exists():
                _fast_ws(text_segments, src_srt, text_key="text")

            _mode_label = "YT text + Whisper timeline" if _yt_mode == "whisper_timeline" \
                          else "YT text + YT timeline"
            _split_label = "sentence-split" if _seg_mode == "sentence" else "word-split"
            self._report("transcribe", 1.0,
                         f"[{_mode_label}, {_split_label}] {len(text_segments)} segments, "
                         f"{total_words} words -> translation")
            # SKIP to Step 4 — run _run_from_step4 which contains translate -> TTS -> assemble
            self._run_from_step4(text_segments, video_path, audio_raw)
            return

        # Fetch English reference subs for QA (save for post-translation comparison)
        self._ref_english_subs = None
        if re.match(r"^https?://", self.cfg.source):
            self._report("transcribe", 0.85, "Fetching reference English subs for QA...")
            self._ref_english_subs = self._fetch_reference_subs(self.cfg.source)
            if self._ref_english_subs:
                self._report("transcribe", 0.90,
                             f"Found {len(self._ref_english_subs)} reference English sub segments for QA")
            else:
                self._report("transcribe", 0.90, "No English reference subs found")

        # Multi-speaker diarization (runs within "transcribe" step progress 82-98%)
        self._voice_map = None
        if self.cfg.multi_speaker:
            speaker_genders, speaker_ranges = self._diarize(audio_raw)
            if speaker_genders and speaker_ranges:
                self._assign_speaker_to_segments(text_segments, speaker_ranges)
                self._voice_map = self._assign_voices_to_speakers(speaker_genders)
                if self.cfg.use_coqui_xtts:
                    self._save_speaker_refs(audio_raw, text_segments)
                self._report("transcribe", 0.99,
                             f"Assigned {len(self._voice_map)} distinct voices")

        # Transcribe-only mode: save source SRT, extract per-speaker refs, and stop
        if self.cfg.transcribe_only:
            from srt_utils import write_srt as _write_srt
            source_srt = self.cfg.work_dir / "transcript_source.srt"
            has_speakers = any("speaker_id" in s for s in text_segments)
            _write_srt(text_segments, source_srt, text_key="text",
                       include_speaker=has_speakers)

            if has_speakers:
                self._save_speaker_refs(audio_raw, text_segments)

            speaker_note = f" ({len(set(s.get('speaker_id','') for s in text_segments if s.get('speaker_id')))} speakers detected)" if has_speakers else ""
            self._report("transcribe", 1.0,
                         f"Transcription complete — {len(text_segments)} segments{speaker_note}. Download SRT to translate.")
            return

        # Save source SRT for training data collection
        from srt_utils import write_srt as _write_srt_src
        source_srt = self.cfg.work_dir / "transcript_source.srt"
        if not source_srt.exists():
            has_speakers = any("speaker_id" in s for s in text_segments)
            _write_srt_src(text_segments, source_srt, text_key="text",
                           include_speaker=has_speakers)

        # ── YouTube text correction: fix Whisper text using YouTube subs ──
        # Whisper keeps its precise timestamps (proven to produce 5:34 output).
        # Only the TEXT is replaced with YouTube's (fewer hallucinations, better
        # punctuation, correct proper nouns). If YouTube subs aren't available,
        # Whisper's own text is used unchanged.
        if getattr(self.cfg, 'yt_text_correction', False) and re.match(r"^https?://", self.cfg.source):
            try:
                self._report("transcribe", 0.85,
                             "Fetching YouTube subs for text correction...")
                yt_subs = self._fetch_youtube_subtitles(self.cfg.source)
                if yt_subs:
                    # Merge YouTube fragments into complete sentences
                    yt_merged = self._merge_broken_sentences(yt_subs)
                    self._report("transcribe", 0.88,
                                 f"YouTube subs: {len(yt_subs)} fragments -> {len(yt_merged)} sentences. "
                                 f"Correcting Whisper text...")
                    # Replace Whisper text with YouTube text, keep Whisper timestamps
                    corrected = self._correct_whisper_with_yt(text_segments, yt_merged)
                    if corrected:
                        corrected_count = sum(
                            1 for i, s in enumerate(text_segments)
                            if i < len(corrected) and s.get("text") != corrected[i].get("text")
                        )
                        text_segments = corrected
                        self.segments = text_segments
                        self._report("transcribe", 0.90,
                                     f"Corrected {corrected_count}/{len(text_segments)} segments "
                                     f"using YouTube subs")
                    else:
                        self._report("transcribe", 0.90,
                                     "YT text correction returned empty -> using Whisper text as-is")
                else:
                    self._report("transcribe", 0.90,
                                 "No YouTube subs found -> using Whisper text as-is")
            except Exception as _ytc_err:
                print(f"[YT-text-correct] Failed: {_ytc_err} -> continuing with Whisper text",
                      flush=True)
                self._report("transcribe", 0.90,
                             f"YT text correction failed ({str(_ytc_err)[:60]}) -> using Whisper text")

        # ── Merge broken sentences: fix Whisper mid-sentence splits ──────
        text_segments = self._merge_broken_sentences(text_segments)

        # ── Close all inter-segment gaps ──────────────────────────────────
        # Segments must be perfectly contiguous (seg[i].end == seg[i+1].start).
        # Assembly's gap_mode is the ONLY source of gaps in the output.
        text_segments = self._close_segment_gaps(text_segments)
        self.segments = text_segments

        self._check_cancelled()

        # ── Simplify English: rewrite complex text for better translation ──
        if self.cfg.simplify_english:
            self._report("transcribe", 0.92, "Simplifying complex English for better Hindi...")
            text_segments = self._simplify_english_segments(text_segments)
            # Recalculate timeline: simplified text has fewer words → tighter slots
            self._adjust_timeline_after_simplify(text_segments)
            self.segments = text_segments
            self._report("transcribe", 0.98,
                         f"Simplified {len(text_segments)} segments")

        self._check_cancelled()

        # Step 4: Translate — cache: _cache_translate.json exists
        yt_translated = None
        # Only try YouTube translated subs if we haven't already attempted
        # them in the transcription step (line 1817). If the first attempt
        # failed (429/unavailable), retrying here wastes 10-30 seconds on
        # yt-dlp for the same result.
        _yt_already_attempted = getattr(self, '_yt_translate_attempted', False)
        if self.cfg.use_yt_translate and re.match(r"^https?://", self.cfg.source) and not _yt_already_attempted:
            self._report("translate", 0.0, "Fetching YouTube auto-translated subtitles...")
            yt_translated = self._fetch_youtube_translated_subs(self.cfg.source)
            if yt_translated:
                self.segments = yt_translated
                text_segments = [s for s in self.segments if s.get("text_translated", "").strip()]
                self._report("translate", 1.0,
                             f"Using YouTube translated subs — {len(text_segments)} segments (skipped Whisper translation)")
                self._save_segments_cache(text_segments, "translate")
            else:
                self._report("translate", 0.1, "No YouTube translated subs found, using normal translation...")

        if not yt_translated:
            cached_translated = self._load_segments_cache("translate")
            if cached_translated:
                # Verify the cached translation matches current target language
                has_translation = any(s.get("text_translated") for s in cached_translated)
                if has_translation:
                    self.segments = cached_translated
                    text_segments = [s for s in self.segments if s.get("text", "").strip()]
                    self._report("translate", 1.0,
                                 f"[cached] Loaded {len(text_segments)} translated segments")
                else:
                    cached_translated = None

            if not cached_translated:
                target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)

                # ── Try YouTube Hindi translation first (better than Google) ──
                _yt_hindi_used = False
                if (getattr(self.cfg, 'yt_text_correction', False)
                    and re.match(r"^https?://", self.cfg.source)):
                    try:
                        import time as _yth_time
                        yt_hindi = None
                        for _yth_attempt in range(3):
                            self._report("translate", 0.0,
                                         f"[YT Hindi] Downloading YouTube Hindi auto-translate "
                                         f"(attempt {_yth_attempt + 1}/3)...")
                            print(f"[YT-Hindi] Attempt {_yth_attempt + 1}/3...", flush=True)
                            yt_hindi = self._fetch_youtube_translated_subs(self.cfg.source)
                            if yt_hindi:
                                break
                            if _yth_attempt < 2:
                                _delay = 3 * (_yth_attempt + 1)
                                print(f"[YT-Hindi] Attempt {_yth_attempt + 1} returned None, "
                                      f"retrying in {_delay}s...", flush=True)
                                _yth_time.sleep(_delay)
                        if yt_hindi:
                            print(f"[YT-Hindi] SUCCESS: Got {len(yt_hindi)} Hindi segments from YouTube",
                                  flush=True)
                            # Show first segment as preview
                            _preview = ""
                            for _ys in yt_hindi[:1]:
                                _preview = _ys.get("text_translated", _ys.get("text", ""))[:80]
                            self._report("translate", 0.1,
                                         f"[YT Hindi] Downloaded {len(yt_hindi)} segments. "
                                         f"Preview: {_preview}...")

                            # Map YouTube Hindi text onto our Whisper-timed segments
                            yt_hindi_words: List[str] = []
                            for ys in yt_hindi:
                                tr = ys.get("text_translated", ys.get("text", ""))
                                yt_hindi_words.extend(tr.split())

                            if yt_hindi_words:
                                w_total = sum(
                                    max(len(s.get("text", "").split()), 1)
                                    for s in text_segments
                                )
                                yt_h_total = len(yt_hindi_words)
                                cursor = 0

                                for si, seg in enumerate(text_segments):
                                    seg_wc = max(len(seg.get("text", "").split()), 1)
                                    n = round((seg_wc / w_total) * yt_h_total)
                                    n = max(1, min(n, yt_h_total - cursor))
                                    if si == len(text_segments) - 1:
                                        n = max(0, yt_h_total - cursor)
                                    if n > 0 and cursor < yt_h_total:
                                        seg["text_translated"] = " ".join(
                                            yt_hindi_words[cursor:cursor + n])
                                        cursor += n
                                    elif not seg.get("text_translated"):
                                        # Safety: if cursor exhausted, use English as fallback
                                        seg["text_translated"] = seg.get("text", "")

                                _yt_hindi_used = True
                                self._report("translate", 0.9,
                                             f"[YT Hindi] USED — {yt_h_total} Hindi words -> "
                                             f"{len(text_segments)} segments (Google Translate SKIPPED)")
                                print(f"[YT-Hindi] Mapped {yt_h_total} Hindi words to "
                                      f"{len(text_segments)} segments. Google Translate skipped.",
                                      flush=True)
                                # Post-replace glossary words in YouTube Hindi
                                self._glossary_post_replace(text_segments)
                            else:
                                self._report("translate", 0.05,
                                             "[YT Hindi] Downloaded but empty text -> using Google Translate")
                                print("[YT-Hindi] Downloaded but segments had no text", flush=True)
                        else:
                            self._report("translate", 0.05,
                                         "[YT Hindi] NOT AVAILABLE -> using Google Translate instead")
                            print("[YT-Hindi] Not available for this video. "
                                  "Falling back to Google Translate.", flush=True)
                    except Exception as _yth_err:
                        self._report("translate", 0.05,
                                     f"[YT Hindi] FAILED ({str(_yth_err)[:40]}) -> using Google Translate")
                        print(f"[YT-Hindi] Failed: {_yth_err} -> falling back to Google Translate",
                              flush=True)

                if not _yt_hindi_used:
                    # Mask glossary words before Google Translate
                    self._glossary_mask(text_segments)
                    self._report("translate", 0.0,
                                 f"[Google Translate] Translating {len(text_segments)} segments to {target_name}...")
                    self._translate_segments(text_segments)
                    # Unmask glossary placeholders in translated text
                    self._glossary_unmask(text_segments)

                # Recalculate timeline for target language word count
                self._recalculate_timeline(text_segments)
                self._close_segment_gaps(text_segments)
                self.segments = text_segments
                # Tag emotion on each segment after translation
                for _seg in text_segments:
                    _seg["emotion"] = self._detect_segment_emotion(_seg)
                self._report("translate", 1.0, "Translation complete")
                # Cache translation for crash recovery
                self._save_segments_cache(text_segments, "translate")

        # ── QA Check: Compare our translation against reference English subs ──
        if getattr(self, '_ref_english_subs', None):
            self._report("translate", 0.95, "Running QA check against reference subs...")
            qa_result = self._qa_post_translation(text_segments, self._ref_english_subs)
            self.qa_score = qa_result["score"]
            qa_path = self.cfg.work_dir / "qa_report.txt"
            qa_path.write_text(qa_result["report"], encoding="utf-8")
            score_pct = f"{qa_result['score']:.0%}"
            self._report("translate", 0.98,
                         f"QA: {score_pct} match — {qa_result['matched']}/{qa_result['total']} segments verified")
            # If QA is poor and English subs are good, auto-switch to using them directly
            if qa_result["score"] < 0.4 and not sub_segments:
                print(f"[QA] Low match ({score_pct}), switching to English subs as source", flush=True)
                # Re-translate from English subs instead
                self._report("translate", 0.0, "QA failed — re-translating from English reference subs...")
                for ref_seg in self._ref_english_subs:
                    ref_seg["text"] = ref_seg.get("text", "")
                ref_copy = [dict(s) for s in self._ref_english_subs]
                self._glossary_mask(ref_copy)
                self._translate_segments(ref_copy)
                self._glossary_unmask(ref_copy)
                self.segments = ref_copy
                text_segments = ref_copy
                # Re-run QA against original English subs
                qa2 = self._qa_post_translation(text_segments, self._ref_english_subs)
                self.qa_score = qa2["score"]
                qa_path.write_text(qa2["report"], encoding="utf-8")
                self._report("translate", 1.0,
                             f"QA: Re-translated from English subs — {qa2['score']:.0%} match")

        # Write translated SRT (per-segment subtitles with proper timestamps)
        srt_translated = self.cfg.work_dir / f"transcript_{self.cfg.target_language}.srt"
        write_srt(self.segments, srt_translated, text_key="text_translated")

        self._check_cancelled()

        # ── Dub-duration cutoff: keep only segments within the limit ──
        dub_dur_min = getattr(self.cfg, 'dub_duration', 0) or 0
        if dub_dur_min > 0:
            max_sec = dub_dur_min * 60
            before = len(text_segments)
            text_segments = [s for s in text_segments if s.get("start", 0) < max_sec]
            if text_segments and text_segments[-1].get("end", 0) > max_sec:
                text_segments[-1]["end"] = max_sec
            self.segments = text_segments
            print(f"[DUB-DURATION] Trimmed segments to first {dub_dur_min}m "
                  f"({before} -> {len(text_segments)} segments, cutoff={max_sec}s)",
                  flush=True)

        # ── Optional: chunk segments for TTS (anti-truncation) ──
        _chunk_size = int(getattr(self.cfg, 'tts_chunk_words', 0) or 0)
        if _chunk_size > 0:
            text_segments = self._chunk_segments_for_tts(text_segments, _chunk_size)
            self._close_segment_gaps(text_segments)
            self.segments = text_segments

        # Step 5: Generate TTS (Edge-TTS already includes 1.25x speedup in one pass)
        self._report("synthesize", 0.0,
                     f"Generating speech ({self.cfg.tts_voice})...")
        tts_data = self._generate_tts_natural(text_segments)
        # Sync text_segments with the split segments from _generate_tts_natural
        text_segments = getattr(self, '_split_tts_segments', text_segments)
        self.segments = text_segments
        if not tts_data:
            raise RuntimeError("TTS synthesis produced no audio segments — check TTS engine settings")

        # ── TTS Manager: validate, retry, sequence, normalize, gap ──
        tts_data = self._tts_manager(tts_data, text_segments)

        # ── Completeness check: verify all segments have audio ──
        self._verify_tts_completeness(tts_data, text_segments)

        # ── Optional: post-TTS Whisper word-count verification + retry ──
        # Runs only if cfg.tts_word_match_verify is True. SLOW (~150-300ms
        # per segment on CPU) but exact: transcribes each WAV with Whisper-tiny
        # in the target language, compares actual word count to expected, and
        # re-runs Edge-TTS for any segment outside the tolerance window.
        try:
            self._post_tts_word_match_verify(tts_data, text_segments)
        except Exception as _e:
            print(f"[WORD-VERIFY] verification pass crashed: {_e} — "
                  f"continuing with unverified TTS", flush=True)

        self._check_cancelled()

        # NO separate speedup — audio plays at natural pace.
        # Video adapts per-segment in assembly (audio_priority mode).
        # The old 1.25x speedup was cutting final words in TTS output.
        for t in tts_data:
            t.pop("_already_sped", None)

        self._report("synthesize", 1.0,
                     f"All {len(tts_data)} segments ready (natural pace)")

        # Step 6: Assemble — AUDIO IS MASTER, video adapts
        self._report("assemble", 0.0, "Building dubbed output...")
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        video_duration = self._get_duration(video_path)
        if video_duration <= 0:
            raise RuntimeError(f"Could not determine video duration for {video_path.name}")

        # Audio-priority: video adapts per-segment (NVENC + parallel)
        self._report("assemble", 0.05, "Building per-segment video (NVENC parallel)...")
        self._assemble_video_adapts_to_audio(video_path, audio_raw, tts_data, video_duration)

        # Copy SRT to output
        out_srt = self.cfg.output_path.parent / f"subtitles_{self.cfg.target_language}.srt"
        shutil.copy2(srt_translated, out_srt)

        self._report("assemble", 1.0, "Done!")

        # Long-segment trace: write the JSON report at end of successful run
        try:
            _job_id = self.cfg.work_dir.parent.name if self.cfg.work_dir else "unknown"
            self._trace_write_report(_job_id)
        except Exception:
            pass

    def _run_from_step4(self, text_segments, video_path, audio_raw):
        """Run Step 4 (translate) through Step 6 (assembly) — used by YouTube subs fast path.

        Skips ALL English processing (keyterms, prosody, merge, cue rebuild, simplify).
        Goes straight: translate → TTS → assemble.
        """
        from srt_utils import write_srt
        import shutil

        # ── Dub-duration cutoff: keep only segments within the limit ──
        dub_dur_min = getattr(self.cfg, 'dub_duration', 0) or 0
        if dub_dur_min > 0:
            max_sec = dub_dur_min * 60
            before = len(text_segments)
            text_segments = [s for s in text_segments if s.get("start", 0) < max_sec]
            # Clamp the last segment's end time to the limit
            if text_segments and text_segments[-1].get("end", 0) > max_sec:
                text_segments[-1]["end"] = max_sec
            print(f"[DUB-DURATION] Trimmed segments to first {dub_dur_min}m "
                  f"({before} -> {len(text_segments)} segments, cutoff={max_sec}s)",
                  flush=True)

        self.segments = text_segments

        # ── If YT Auto-Translate already provided text_translated, skip ALL translation ──
        already_translated = getattr(self, '_yt_already_translated', False) and all(
            s.get("text_translated", "").strip() for s in text_segments
        )

        if already_translated:
            self._report("translate", 1.0,
                         f"YT Auto-Translate provided {len(text_segments)} pre-translated segments — skipping translation step")
        else:
            # Simplify English before translation (if enabled)
            if self.cfg.simplify_english:
                self._report("translate", 0.0, "Simplifying English for better translation...")
                text_segments = self._simplify_english_segments(text_segments)
                self._adjust_timeline_after_simplify(text_segments)
                self.segments = text_segments

            # Step 4: Translate
            self._glossary_mask(text_segments)
            self._report("translate", 0.05, "Translating...")
            self._translate_segments(text_segments)
            self._glossary_unmask(text_segments)
            # Recalculate timeline for target language word count
            self._recalculate_timeline(text_segments)
            # Close any gaps introduced by timeline recalculation
            self._close_segment_gaps(text_segments)
            self.segments = text_segments

            # Rule engine cleanup
            is_hindi = self.cfg.target_language in ("hi", "hi-IN")
            if is_hindi:
                for seg in text_segments:
                    seg["text_translated"] = _hindi_rules.apply(seg.get("text_translated", ""))

        # Write translated SRT
        srt_translated = self.cfg.work_dir / f"transcript_{self.cfg.target_language}.srt"
        write_srt(self.segments, srt_translated, text_key="text_translated")

        self._report("translate", 1.0, "Translation complete")

        # ── Optional: chunk segments for TTS (anti-truncation) ──
        _chunk_size = int(getattr(self.cfg, 'tts_chunk_words', 0) or 0)
        if _chunk_size > 0:
            text_segments = self._chunk_segments_for_tts(text_segments, _chunk_size)
            self._close_segment_gaps(text_segments)
            self.segments = text_segments

        # Step 5: TTS
        self._report("synthesize", 0.0, f"Generating speech ({self.cfg.tts_voice})...")
        tts_data = self._generate_tts_natural(text_segments)
        # Sync text_segments with the split segments from _generate_tts_natural
        text_segments = getattr(self, '_split_tts_segments', text_segments)
        self.segments = text_segments
        if not tts_data:
            raise RuntimeError("TTS produced no audio segments")

        # ── TTS Manager: validate, retry, sequence, normalize, gap ──
        tts_data = self._tts_manager(tts_data, text_segments)

        # ── Completeness check: verify all segments have audio ──
        self._verify_tts_completeness(tts_data, text_segments)

        # ── Optional: post-TTS Whisper word-count verification + retry ──
        # Runs only if cfg.tts_word_match_verify is True. SLOW (~150-300ms
        # per segment on CPU) but exact: transcribes each WAV with Whisper-tiny
        # in the target language, compares actual word count to expected, and
        # re-runs Edge-TTS for any segment outside the tolerance window.
        try:
            self._post_tts_word_match_verify(tts_data, text_segments)
        except Exception as _e:
            print(f"[WORD-VERIFY] verification pass crashed: {_e} — "
                  f"continuing with unverified TTS", flush=True)

        self._report("synthesize", 0.9, "TTS complete (managed)")
        for t in tts_data:
            t.pop("_already_sped", None)
        self._report("synthesize", 1.0, f"{len(tts_data)} segments ready")

        # Step 5.5: Global speed
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        video_duration = self._get_duration(video_path)

        total_tts = sum(t.get("duration", 0) for t in tts_data)
        total_slots = sum(max(0, t.get("end", 0) - t.get("start", 0)) for t in tts_data)
        if total_slots > 0 and total_tts > 0:
            ratio = total_tts / total_slots
            speed = min(max(ratio, 1.0), 1.25) if ratio > 0.95 else 1.0
            self._report("assemble", 0.02, f"Speed: {speed:.2f}x (ratio {ratio:.2f}x)")
            if abs(speed - 1.0) > 0.01:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                def _gs(it):
                    i, t = it
                    s = self.cfg.work_dir / f"tts_global_{i:04d}.wav"
                    self._time_stretch(t["wav"], speed, s)
                    t["wav"] = s
                    t["duration"] = t["duration"] / speed
                gs_total = len(tts_data)
                self._report("assemble", 0.03,
                             f"Applying global {speed:.2f}x speed to {gs_total} segments...")
                with ThreadPoolExecutor(max_workers=12) as pool:
                    futures = [pool.submit(_gs, item) for item in enumerate(tts_data)]
                    gs_done = 0
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception:
                            pass
                        gs_done += 1
                        if gs_done % 25 == 0 or gs_done == gs_total:
                            self._report("assemble", 0.03,
                                         f"[GlobalSpeed] {gs_done}/{gs_total} segments")

        # Step 6: Assembly — audio-priority, video adapts per-segment (NVENC + parallel)
        self._report("assemble", 0.1, "Assembling (per-segment NVENC parallel)...")
        self._assemble_video_adapts_to_audio(video_path, audio_raw, tts_data, video_duration)

        # Copy SRT
        out_srt = self.cfg.output_path.parent / f"subtitles_{self.cfg.target_language}.srt"
        if srt_translated.exists():
            shutil.copy2(srt_translated, out_srt)

        self._report("assemble", 1.0, "Done!")

    def run_from_source_srt(self, source_srt_path: Path):
        """Upload English SRT → translate → TTS → assemble.

        User provides English (source) SRT. Pipeline translates it to target
        language, then does TTS + assembly. NO transcription needed.

        Flow: Load SRT → Translate → Rule Engine → TTS → Assembly
        """
        from srt_utils import parse_srt

        self._ensure_ffmpeg()

        # Step 1-2: Download + extract (need video for final output)
        self.download_and_extract()
        video_path = self._find_cached_video()
        audio_raw = self.cfg.work_dir / "audio_raw.wav"
        if not video_path:
            raise RuntimeError("No video found for assembly")

        # Load source English SRT
        self._report("transcribe", 0.0, "Loading source English SRT...")
        segments = parse_srt(source_srt_path, text_key="text")
        if not segments:
            raise RuntimeError("Source SRT is empty or invalid")

        # Ensure text field is set
        for seg in segments:
            if not seg.get("text"):
                seg["text"] = seg.get("text_translated", "")
            seg.setdefault("_qc_issues", [])
            seg.setdefault("_protected_terms", [])
            seg.setdefault("emotion", "neutral")

        text_segments = [s for s in segments if s.get("text", "").strip()]
        self.segments = text_segments
        self._report("transcribe", 1.0,
                     f"Loaded {len(text_segments)} segments from SRT — translating...")

        # Step 4-6: Translate → TTS → Assembly
        self._run_from_step4(text_segments, video_path, audio_raw)

    def _run_transcription(self):
        """Run steps 1-3: download + extract + transcribe. Sets self.segments."""
        self._ensure_ffmpeg()
        video_path = self._find_cached_video()
        if video_path:
            self._report("download", 1.0, f"[cached] {video_path.name}")
            if not self.video_title:
                self.video_title = video_path.stem
        else:
            self._report("download", 0.0, "Downloading...")
            video_path = self._ingest_source(self.cfg.source)
            self._report("download", 1.0, f"Downloaded: {video_path.name}")
        audio_raw = self.cfg.work_dir / "audio_raw.wav"
        wav_16k = self.cfg.work_dir / "audio_16k.wav"
        if audio_raw.exists() and audio_raw.stat().st_size > 0:
            self._report("extract", 1.0, "[cached] Audio extracted")
            if wav_16k.exists():
                self._whisper_audio = wav_16k
        else:
            self._report("extract", 0.0, "Extracting audio...")
            audio_raw = self._extract_audio(video_path)
            self._report("extract", 1.0, "Audio extracted")
        cached_segs = self._load_segments_cache("transcribe")
        if cached_segs:
            self.segments = cached_segs
            self._report("transcribe", 1.0, f"[cached] {len(cached_segs)} segments")
        else:
            sub_segments = None
            if self.cfg.prefer_youtube_subs and re.match(r"^https?://", self.cfg.source):
                sub_segments = self._fetch_youtube_subtitles(self.cfg.source)
            if self.cfg.prefer_youtube_subs and sub_segments:
                self.segments = sub_segments
            else:
                whisper_audio = self._whisper_audio or audio_raw
                self.segments = self._transcribe(whisper_audio)
            text_segments = [s for s in self.segments if s.get("text", "").strip()]
            if text_segments:
                self._save_segments_cache(text_segments, "transcribe")

    def _run_tts_and_assembly(self, text_segments, audio_raw_path):
        """Run steps 5-6: TTS + assembly from pre-translated segments."""
        from srt_utils import write_srt
        import shutil
        video_path = self._find_cached_video()
        if not video_path:
            raise RuntimeError("No video found for assembly")
        audio_raw = Path(audio_raw_path)

        # ── Dub-duration cutoff: keep only segments within the limit ──
        dub_dur_min = getattr(self.cfg, 'dub_duration', 0) or 0
        if dub_dur_min > 0:
            max_sec = dub_dur_min * 60
            before = len(text_segments)
            text_segments = [s for s in text_segments if s.get("start", 0) < max_sec]
            if text_segments and text_segments[-1].get("end", 0) > max_sec:
                text_segments[-1]["end"] = max_sec
            self.segments = text_segments
            print(f"[DUB-DURATION] Trimmed segments to first {dub_dur_min}m "
                  f"({before} -> {len(text_segments)} segments, cutoff={max_sec}s)",
                  flush=True)

        srt_path = self.cfg.work_dir / f"transcript_{self.cfg.target_language}.srt"
        write_srt(text_segments, srt_path, text_key="text_translated")
        self._report("synthesize", 0.0, f"Generating speech ({self.cfg.tts_voice})...")
        tts_data = self._generate_tts_natural(text_segments)
        # Sync text_segments with the split segments from _generate_tts_natural
        text_segments = getattr(self, '_split_tts_segments', text_segments)
        self.segments = text_segments
        if not tts_data:
            raise RuntimeError("TTS produced no audio segments")
        for t in tts_data:
            t.pop("_already_sped", None)

        # ── TTS Manager: validate, retry, sequence, normalize, gap ──
        tts_data = self._tts_manager(tts_data, text_segments)

        # ── Completeness check: verify all segments have audio ──
        self._verify_tts_completeness(tts_data, text_segments)

        # ── Optional: post-TTS Whisper word-count verification + retry ──
        # Runs only if cfg.tts_word_match_verify is True. SLOW (~150-300ms
        # per segment on CPU) but exact: transcribes each WAV with Whisper-tiny
        # in the target language, compares actual word count to expected, and
        # re-runs Edge-TTS for any segment outside the tolerance window.
        try:
            self._post_tts_word_match_verify(tts_data, text_segments)
        except Exception as _e:
            print(f"[WORD-VERIFY] verification pass crashed: {_e} — "
                  f"continuing with unverified TTS", flush=True)

        # NO global speedup — split-the-diff handles timing per-segment in assembly
        for t in tts_data:
            t.pop("_already_sped", None)

        self._report("synthesize", 1.0, f"{len(tts_data)} segments ready (natural pace)")
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        video_duration = self._get_duration(video_path)
        self._report("assemble", 0.1, "Assembling (per-segment NVENC parallel)...")
        self._assemble_video_adapts_to_audio(video_path, audio_raw, tts_data, video_duration)
        out_srt = self.cfg.output_path.parent / f"subtitles_{self.cfg.target_language}.srt"
        if srt_path.exists():
            shutil.copy2(srt_path, out_srt)
        self._report("assemble", 1.0, "Done!")

    def download_and_extract(self):
        """Run only the download + audio extraction steps (steps 1-2).
        Sets self.video_title. Used when an external SRT is provided."""
        self._ensure_ffmpeg()

        # Step 1: Download / ingest
        video_path = self._find_cached_video()
        if video_path:
            self._report("download", 1.0, f"[cached] Using existing: {video_path.name}")
            if not self.video_title:
                self.video_title = video_path.stem
        else:
            self._report("download", 0.0, "Downloading video...")
            video_path = self._ingest_source(self.cfg.source)
            self._report("download", 1.0, f"Downloaded: {video_path.name}")

        self._check_cancelled()

        # Step 2: Extract audio
        audio_raw = self.cfg.work_dir / "audio_raw.wav"
        if audio_raw.exists() and audio_raw.stat().st_size > 0:
            self._report("extract", 1.0, "[cached] Audio already extracted")
        else:
            self._report("extract", 0.0, "Extracting audio...")
            self._extract_audio(video_path)
            self._report("extract", 1.0, "Audio extracted")

        self._check_cancelled()

    def run_from_srt(self, translated_srt_path: Path):
        """Resume pipeline from an uploaded translated SRT — runs TTS + assembly only."""
        from srt_utils import parse_srt

        self._ensure_ffmpeg()

        # Load translated segments
        self._report("translate", 0.0, "Loading uploaded translated SRT...")
        translated = parse_srt(translated_srt_path, text_key="text_translated")
        if not translated:
            raise RuntimeError("Uploaded SRT is empty or invalid")

        self.segments = translated

        # Check for speaker labels from SRT and rebuild voice map
        speakers_found = set(s.get("speaker_id") for s in translated if s.get("speaker_id"))
        if speakers_found and self.cfg.multi_speaker:
            self._report("translate", 0.5,
                         f"Found {len(speakers_found)} speakers in SRT, assigning voices...")
            # Re-run gender detection from original audio if available
            audio_raw = self.cfg.work_dir / "audio_raw.wav"
            if audio_raw.exists():
                speaker_ranges = {}
                for seg in translated:
                    spk = seg.get("speaker_id")
                    if spk:
                        if spk not in speaker_ranges:
                            speaker_ranges[spk] = []
                        speaker_ranges[spk].append((seg["start"], seg["end"]))
                speaker_genders = self._detect_speaker_genders(audio_raw, speaker_ranges)
                self._voice_map = self._assign_voices_to_speakers(speaker_genders)
                self._report("translate", 0.8,
                             f"Assigned voices: {', '.join(f'{k}={v}' for k, v in self._voice_map.items())}")
            else:
                # No audio for gender detection — alternate male/female
                self._voice_map = {}
                lang = self.cfg.target_language
                pool = VOICE_POOL.get(lang, {})
                female_voices = list(pool.get("female", [DEFAULT_VOICES.get(lang, "en-US-JennyNeural")]))
                male_voices = list(pool.get("male", [MALE_VOICES.get(lang, "en-US-GuyNeural")]))
                for i, spk in enumerate(sorted(speakers_found)):
                    if i % 2 == 0:
                        self._voice_map[spk] = female_voices[i // 2 % len(female_voices)]
                    else:
                        self._voice_map[spk] = male_voices[i // 2 % len(male_voices)]
        else:
            self._voice_map = None

        self._report("translate", 1.0, f"Loaded {len(translated)} translated segments")

        # Find existing video and audio from first run
        video_path = None
        for ext in ["mp4", "webm", "mkv"]:
            p = self.cfg.work_dir / f"source.{ext}"
            if p.exists():
                video_path = p
                break
        if not video_path:
            sources = list(self.cfg.work_dir.glob("source.*"))
            video_path = sources[0] if sources else None
        if not video_path:
            raise RuntimeError("Original video not found in work directory")

        audio_raw = self.cfg.work_dir / "audio_raw.wav"
        if not audio_raw.exists():
            self._report("extract", 0.0, "Re-extracting audio...")
            audio_raw = self._extract_audio(video_path)

        # Write the translated SRT to standard location
        srt_translated = self.cfg.work_dir / f"transcript_{self.cfg.target_language}.srt"
        shutil.copy2(translated_srt_path, srt_translated)

        # Recalculate timeline for Hindi word count (same as default path Stage 6)
        text_segments = translated
        self._recalculate_timeline(text_segments)
        self._close_segment_gaps(text_segments)

        # ── Dub-duration cutoff: keep only segments within the limit ──
        dub_dur_min = getattr(self.cfg, 'dub_duration', 0) or 0
        if dub_dur_min > 0:
            max_sec = dub_dur_min * 60
            before = len(text_segments)
            text_segments = [s for s in text_segments if s.get("start", 0) < max_sec]
            if text_segments and text_segments[-1].get("end", 0) > max_sec:
                text_segments[-1]["end"] = max_sec
            self.segments = text_segments
            print(f"[DUB-DURATION] Trimmed segments to first {dub_dur_min}m "
                  f"({before} -> {len(text_segments)} segments, cutoff={max_sec}s)",
                  flush=True)

        # Step 5: Generate TTS
        self._report("synthesize", 0.0,
                     f"Generating speech ({self.cfg.tts_voice})...")
        tts_data = self._generate_tts_natural(text_segments)
        # Sync text_segments with the split segments from _generate_tts_natural
        text_segments = getattr(self, '_split_tts_segments', text_segments)
        self.segments = text_segments
        if not tts_data:
            raise RuntimeError("TTS synthesis produced no audio segments — check TTS engine settings")

        # ── TTS Manager: validate, retry, sequence, normalize, gap ──
        tts_data = self._tts_manager(tts_data, text_segments)

        # ── Completeness check: verify all segments have audio ──
        self._verify_tts_completeness(tts_data, text_segments)

        # ── Optional: post-TTS Whisper word-count verification + retry ──
        # Runs only if cfg.tts_word_match_verify is True. SLOW (~150-300ms
        # per segment on CPU) but exact: transcribes each WAV with Whisper-tiny
        # in the target language, compares actual word count to expected, and
        # re-runs Edge-TTS for any segment outside the tolerance window.
        try:
            self._post_tts_word_match_verify(tts_data, text_segments)
        except Exception as _e:
            print(f"[WORD-VERIFY] verification pass crashed: {_e} — "
                  f"continuing with unverified TTS", flush=True)

        # NO separate speedup — audio plays at natural pace, video adapts.
        for t in tts_data:
            t.pop("_already_sped", None)

        self._report("synthesize", 1.0,
                     f"All {len(tts_data)} segments ready (natural pace)")

        # Step 6: Assemble
        self._report("assemble", 0.0, "Building dubbed output...")
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        video_duration = self._get_duration(video_path)
        if video_duration <= 0:
            raise RuntimeError(f"Could not determine video duration for {video_path.name}")

        # Audio-priority: video adapts per-segment (NVENC + parallel)
        self._report("assemble", 0.05, "Building per-segment video (NVENC parallel)...")
        self._assemble_video_adapts_to_audio(video_path, audio_raw, tts_data, video_duration)

        # Copy SRT to output
        out_srt = self.cfg.output_path.parent / f"subtitles_{self.cfg.target_language}.srt"
        shutil.copy2(srt_translated, out_srt)

        self._report("assemble", 1.0, "Done!")

    # ── FFmpeg check ─────────────────────────────────────────────────────
    def _ensure_ffmpeg(self):
        resolved = self._find_executable("ffmpeg")

        # Also scan WinGet install paths as last resort
        if resolved == "ffmpeg" and sys.platform == "win32":
            localappdata = os.environ.get("LOCALAPPDATA", "")
            # Fallback: derive LOCALAPPDATA from USERPROFILE if not set
            if not localappdata:
                userprofile = os.environ.get("USERPROFILE", str(Path.home()))
                localappdata = str(Path(userprofile) / "AppData" / "Local")
            winget_ffmpeg = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
            if winget_ffmpeg.exists():
                for exe in winget_ffmpeg.rglob("ffmpeg.exe"):
                    resolved = str(exe)
                    os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
                    break

        if resolved == "ffmpeg" and shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "FFmpeg not found! Install: winget install Gyan.FFmpeg"
            )
        self._ffmpeg = resolved

    def _check_nvenc(self) -> bool:
        """Check if NVIDIA NVENC hardware encoder is available."""
        if self._has_nvenc is not None:
            return self._has_nvenc
        try:
            r = self._run_proc(
                [self._ffmpeg, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            self._has_nvenc = "h264_nvenc" in (r.stdout or "")
        except Exception:
            self._has_nvenc = False
        if self._has_nvenc:
            print("[NVENC] GPU encoding available — using h264_nvenc", flush=True)
        return self._has_nvenc

    def _video_encode_args(self, crf: str = "18", force_cpu: bool = False) -> list:
        """Return FFmpeg video encoder arguments — NVENC if available, else libx264.

        force_cpu: use libx264 even if NVENC is available (for short clips where
        NVENC session overhead > encoding time, and rapid session creation crashes).
        """
        if not force_cpu and self._check_nvenc():
            # NVENC uses -qp (constant QP) instead of -crf; qp≈crf for visual quality
            # p1 = fastest preset, QP 23 = visually transparent for YouTube
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-qp", "23"]
        return ["-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", crf]

    # ── Step 1: Ingest ───────────────────────────────────────────────────
    def _find_cookies_file(self) -> Optional[str]:
        """Find a YouTube cookies file if available."""
        # Check common locations
        for path in [
            Path(__file__).resolve().parent / "cookies.txt",
            Path.home() / "cookies.txt",
            Path("/content/cookies.txt"),  # Colab
        ]:
            if path.exists():
                return str(path)
        return None

    def _get_cookies_args(self) -> list:
        """Get yt-dlp cookie arguments: use cookies file if available, otherwise none."""
        cookies_file = self._find_cookies_file()
        if cookies_file:
            return ["--cookies", cookies_file]
        # No cookies file found — skip cookies (browser cookie extraction fails
        # when the browser is running due to SQLite DB lock)
        return []

    def _ingest_source(self, src: str) -> Path:
        if re.match(r"^https?://", src):
            out_tpl = str(self.cfg.work_dir / "source.%(ext)s")

            # ── Source video cache: crash resilience only ──
            # Hash the URL → check backend/cache/sources/<hash>.mp4. If it
            # exists from a previous run, copy it into work_dir and skip the
            # download entirely. This is NOT general "reuse across runs" — it
            # exists ONLY so that a crash at a later step (transcribe, TTS,
            # assemble) doesn't waste a 3 GB / 1 hour download. Hits stay
            # 30 days then are evicted by age.
            import hashlib as _hl
            src_hash = _hl.sha256(src.encode("utf-8")).hexdigest()[:16]
            sources_cache_dir = Path(__file__).resolve().parent / "cache" / "sources"
            sources_cache_dir.mkdir(parents=True, exist_ok=True)
            cached_source = sources_cache_dir / f"{src_hash}.mp4"
            if cached_source.exists() and cached_source.stat().st_size > 1024 * 1024:
                # Cache hit — copy into work_dir and short-circuit
                target = self.cfg.work_dir / "source.mp4"
                self._report("download", 0.50,
                             f"Source cache hit ({cached_source.stat().st_size // (1024*1024)} MB) "
                             f"— skipping download")
                try:
                    shutil.copy2(cached_source, target)
                    # Try to recover title from a sidecar file
                    title_sidecar = sources_cache_dir / f"{src_hash}.title"
                    if title_sidecar.exists():
                        try:
                            self.video_title = title_sidecar.read_text(encoding="utf-8").strip()
                        except Exception:
                            pass
                    self._report("download", 0.95, f"Restored cached source: {target.name}")
                    return target
                except Exception as e:
                    print(f"[CACHE] Failed to restore cached source: {e} — re-downloading",
                          flush=True)
                    try:
                        cached_source.unlink(missing_ok=True)
                    except Exception:
                        pass

            # Get cookie args (file or browser fallback)
            cookies_args = self._get_cookies_args()

            # Enable Node.js runtime for YouTube extraction (required since yt-dlp 2025+)
            node_path = self._find_executable("node")
            # Only use --js-runtimes if we found a real path (not bare "node" fallback)
            js_args = ["--js-runtimes", f"node:{node_path}"] if (node_path and node_path != "node" and Path(node_path).exists()) else []

            # ── Pre-flight metadata fetch: title + duration + filesize ──
            # Single yt-dlp call (--print accepts a delimited template) so we
            # don't pay 2x the network round-trip. We surface duration and an
            # approximate file size to the UI BEFORE the multi-GB download
            # starts so the user can see "this is a 3h video, 3.4 GB" up front
            # instead of guessing from the download speed.
            self._report("download", 0.01, "Fetching video metadata...")
            video_duration_sec = 0
            video_filesize_bytes = 0
            try:
                meta_cmd = [
                    self._ytdlp,
                    "--print", "%(title)s|||%(duration)s|||%(filesize_approx)s",
                ] + cookies_args + js_args + [src]
                print(f"[YTDLP] meta cmd: {meta_cmd}", flush=True)
                meta_result = self._run_proc(
                    meta_cmd, capture_output=True, text=True,
                    timeout=60, encoding="utf-8", errors="replace",
                )
                print(f"[YTDLP] meta rc={meta_result.returncode} "
                      f"stdout={repr((meta_result.stdout or '')[:300])}", flush=True)
                if meta_result.returncode != 0:
                    print(f"[YTDLP] meta stderr: {(meta_result.stderr or '')[:300]}", flush=True)
                meta_line = (meta_result.stdout or "").strip().split("\n")[0]
                parts = meta_line.split("|||")
                if parts and parts[0].strip() and parts[0].strip() != "NA":
                    self.video_title = parts[0].strip()
                if len(parts) > 1 and parts[1].strip() and parts[1].strip() != "NA":
                    try:
                        video_duration_sec = int(float(parts[1].strip()))
                    except (ValueError, TypeError):
                        video_duration_sec = 0
                if len(parts) > 2 and parts[2].strip() and parts[2].strip() != "NA":
                    try:
                        video_filesize_bytes = int(float(parts[2].strip()))
                    except (ValueError, TypeError):
                        video_filesize_bytes = 0

                # Build a friendly pre-flight summary
                if video_duration_sec > 0:
                    h = video_duration_sec // 3600
                    m = (video_duration_sec % 3600) // 60
                    s = video_duration_sec % 60
                    dur_str = (f"{h}h {m}m" if h else
                               f"{m}m {s}s" if m else f"{s}s")
                else:
                    dur_str = "unknown"
                size_str = (f"{video_filesize_bytes / (1024*1024):.0f} MB"
                            if video_filesize_bytes > 0 else "unknown")
                print(f"[YTDLP] Got title='{self.video_title}' duration={dur_str} size={size_str}", flush=True)

                # Pre-flight refusal: extremely long videos. Cap at 8 hours.
                # Above this you almost certainly want split mode anyway, and
                # the UI sometimes shows people bricking their machines on
                # 12h streams. Soft refuse with a clear message.
                # Skip this check if dub_duration is set — user only wants
                # a portion of the video, so the full length doesn't matter.
                effective_dur = video_duration_sec
                dub_dur_min = getattr(self.cfg, 'dub_duration', 0) or 0
                if dub_dur_min > 0:
                    effective_dur = min(video_duration_sec, dub_dur_min * 60)
                if effective_dur > 8 * 3600:
                    raise RuntimeError(
                        f"Video is {dur_str} long (>8h). This is too long for a "
                        f"single dub job — enable split mode (split_duration=30) "
                        f"to process it in chunks, or pick a shorter video."
                    )

                self._report("download", 0.02,
                             f"Video: '{self.video_title}' — {dur_str}, ~{size_str}")
            except RuntimeError:
                raise  # propagate explicit refusal
            except Exception as e:
                print(f"[YTDLP] Metadata fetch failed: {e}", flush=True)
                self._report("download", 0.02, "Metadata fetch failed — continuing anyway")

            # Download video. Two modes:
            #   remux  → --remux-video mp4 (instant container swap, no re-encode)
            #   encode → --merge-output-format mp4 (re-encode via ffmpeg, slower
            #            but always works regardless of source codec)
            try:
                _dub_dur_min = getattr(self.cfg, 'dub_duration', 0) or 0
                # --download-sections requires --merge-output-format (encode)
                # because remux mode downloads the full file then trims.
                if _dub_dur_min > 0 or self.cfg.download_mode == "encode":
                    container_args = ["--merge-output-format", "mp4"]
                    mode_label = "encode (ffmpeg)" if _dub_dur_min == 0 else f"encode (first {_dub_dur_min}m)"
                else:
                    container_args = ["--remux-video", "mp4"]
                    mode_label = "remux (instant)"

                # 720p max — use chosen container handling above
                dl_cmd = [
                    self._ytdlp,
                    "-f", "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
                ] + container_args + [
                    "-o", out_tpl,
                ]

                # If dub_duration is set, only download the needed portion
                # to avoid fetching huge files for long videos.
                _dub_dur_min = getattr(self.cfg, 'dub_duration', 0) or 0
                if _dub_dur_min > 0:
                    # Add 2 minutes padding for safety
                    _cut_sec = (_dub_dur_min + 2) * 60
                    dl_cmd += ["--download-sections", f"*0-{_cut_sec}",
                               "--force-keyframes-at-cuts"]
                    print(f"[YTDLP] dub_duration={_dub_dur_min}m — "
                          f"downloading first {_cut_sec}s only", flush=True)

                self._report("download", 0.05, f"Downloading [{mode_label}]...")

                # Only add --ffmpeg-location if we have a real path (not bare "ffmpeg")
                ffmpeg_path = Path(self._ffmpeg)
                if ffmpeg_path.is_absolute():
                    dl_cmd += ["--ffmpeg-location", str(ffmpeg_path.parent)]
                dl_cmd += cookies_args + js_args + [src]
                print(f"[YTDLP] cmd: {dl_cmd}", flush=True)
                # ── Live progress: watchdog polls work_dir file growth every 1s ──
                # Without this the UI freezes at 5% for the entire download
                # because _run_proc blocks on yt-dlp without us parsing stdout.
                # We pass total_bytes_hint from the pre-flight metadata fetch
                # so the watchdog reports real % + remaining MB + ETA.
                with self._progress_watchdog(
                    "download", self.cfg.work_dir,
                    label="Downloading", start_pct=0.05, end_pct=1.0,
                    total_bytes_hint=video_filesize_bytes if video_filesize_bytes > 0 else None,
                    file_glob="source.*",
                ):
                    result = self._run_proc(dl_cmd, capture_output=True, text=True, timeout=14400)  # 4h max for very long videos
                print(f"[YTDLP] rc={result.returncode} stdout_len={len(result.stdout or '')} stderr_len={len(result.stderr or '')}", flush=True)

                if result.returncode != 0:
                    error_msg = (result.stderr or result.stdout or "Unknown error").strip()
                    print(f"[YTDLP] error: {error_msg[:500]}", flush=True)
                    raise RuntimeError(f"yt-dlp failed: {error_msg}")
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"yt-dlp failed: {e}") from e

            self._report("download", 0.9, f"Downloaded: {self.video_title}")

            # Find downloaded file
            mp4 = list(self.cfg.work_dir.glob("source.mp4"))
            chosen = None
            if mp4:
                chosen = mp4[0]
            else:
                all_sources = list(self.cfg.work_dir.glob("source.*"))
                if all_sources:
                    chosen = all_sources[0]
            if chosen is None:
                raise RuntimeError("Download completed but no video file found in work directory")

            # ── Source video cache: store for crash resilience ──
            # Copy the downloaded file into the source cache so a later step
            # crash (transcribe, TTS, assemble) doesn't waste this download.
            # Best-effort: if the copy fails (disk full, permissions), log and
            # continue — the job itself isn't blocked.
            try:
                if chosen.suffix.lower() == ".mp4" and chosen.stat().st_size > 1024 * 1024:
                    shutil.copy2(chosen, cached_source)
                    # Sidecar with title for next-run restoration
                    try:
                        (sources_cache_dir / f"{src_hash}.title").write_text(
                            self.video_title or "", encoding="utf-8"
                        )
                    except Exception:
                        pass
                    print(f"[CACHE] Stored source ({chosen.stat().st_size // (1024*1024)} MB) "
                          f"-> {cached_source.name}", flush=True)
                    # Evict source-cache entries older than 30 days
                    try:
                        import time as _time
                        cutoff = _time.time() - 30 * 86400
                        for old in sources_cache_dir.glob("*.mp4"):
                            try:
                                if old.stat().st_atime < cutoff:
                                    old.unlink(missing_ok=True)
                                    sidecar = sources_cache_dir / f"{old.stem}.title"
                                    sidecar.unlink(missing_ok=True)
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception as e:
                print(f"[CACHE] Failed to store source: {e}", flush=True)

            return chosen

        p = Path(src)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        self.video_title = p.stem
        return p

    # ── Step 2: Extract audio ────────────────────────────────────────────
    def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video. Creates both full-quality and lightweight versions.
        Full-quality (48kHz stereo) for final mix, lightweight (16kHz mono) for transcription."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        wav = self.cfg.work_dir / "audio_raw.wav"
        wav_16k = self.cfg.work_dir / "audio_16k.wav"

        def extract_full():
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(video_path),
                 "-vn", "-ac", str(self.N_CHANNELS), "-ar", str(self.SAMPLE_RATE),
                 "-acodec", "pcm_s16le", str(wav)],
                check=True, capture_output=True, timeout=7200,  # 2h max
            )
            return "full"

        def extract_16k():
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(video_path),
                 "-vn", "-ac", "1", "-ar", "16000",
                 "-acodec", "pcm_s16le", str(wav_16k)],
                check=True, capture_output=True, timeout=7200,  # 2h max
            )
            return "16k"

        # Extract both in parallel — saves ~50% extraction time.
        # Watchdog reports live MB extracted so the UI doesn't freeze on long videos.
        # end_pct=1.0 so step_progress reaches 1.0 and _calc_overall computes
        # the correct overall percentage (not 95% × weight).
        with self._progress_watchdog(
            "extract", self.cfg.work_dir,
            label="Extracting audio", start_pct=0.10, end_pct=1.0,
            file_glob="audio_*.wav",
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_full = pool.submit(extract_full)
                fut_16k = pool.submit(extract_16k)

                # Full-quality extraction is mandatory
                fut_full.result()

                # 16kHz is optional (for Groq API) — fall back to full if it fails
                try:
                    fut_16k.result()
                    self._whisper_audio = wav_16k
                except Exception:
                    self._whisper_audio = wav  # Fall back to full-quality for transcription

        return wav

    # ── Step 3a: Fetch YouTube subtitles (skip Whisper if available) ─────
    @staticmethod
    def _vtt_time_to_seconds(time_str: str) -> float:
        """Convert VTT/SRT timestamp (HH:MM:SS.mmm) to seconds."""
        parts = time_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].replace(",", ".").split(".")
        s = int(s_parts[0])
        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0

    _NOISE_RE = re.compile(r"^\[.*\]$|^\(.*\)$|^♪.*♪$")
    # Inline noise tags to strip from within segment text (not whole-segment)
    _NOISE_INLINE_RE = re.compile(
        r'\s*\[(?:music|applause|laughter|silence|noise|sound|cheering|singing)\]\s*',
        re.IGNORECASE,
    )

    def _parse_vtt(self, vtt_path: Path) -> List[Dict]:
        """Parse a WebVTT file into pipeline segment format."""
        content = vtt_path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        segments_raw: List[Dict] = []
        i = 0
        while i < len(lines):
            m = re.match(
                r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})",
                lines[i].strip(),
            )
            if m:
                start = self._vtt_time_to_seconds(m.group(1))
                end = self._vtt_time_to_seconds(m.group(2))
                text_lines = []
                i += 1
                # Collect text until we hit the NEXT timestamp line.
                # YouTube VTT may have empty lines BETWEEN text and next timestamp,
                # so we don't stop on empty lines — only on next timestamp or EOF.
                while i < len(lines):
                    line = lines[i].strip()
                    if re.match(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->", line):
                        break  # next cue
                    if line:  # skip empty lines but don't stop
                        text_lines.append(line)
                    i += 1
                raw_text = " ".join(text_lines)
                # Strip inline timing tags <00:00:19.039> and styling <c>
                clean_text = re.sub(r"<[^>]+>", "", raw_text).strip()
                # Strip inline noise tags like [music], [applause] from within text
                clean_text = self._NOISE_INLINE_RE.sub(" ", clean_text).strip()
                if clean_text and not self._NOISE_RE.match(clean_text):
                    segments_raw.append({"start": start, "end": end, "text": clean_text})
            else:
                i += 1

        if not segments_raw:
            return []

        # Deduplicate YouTube auto-gen rolling two-line format.
        # YouTube auto-captions use a rolling two-line display where the END
        # of cue N overlaps the START of cue N+1. We detect this partial
        # trailing/leading word overlap and trim it from the previous cue.
        deduped = [segments_raw[0]]
        for seg in segments_raw[1:]:
            prev_text = deduped[-1]["text"]
            curr_text = seg["text"]

            # Full substring containment (original logic)
            if prev_text in curr_text:
                deduped[-1] = seg
                continue
            if curr_text in prev_text:
                continue

            # Partial trailing/leading overlap: last N words of prev == first N words of curr
            prev_words_lower = prev_text.lower().split()
            curr_words_lower = curr_text.lower().split()
            max_check = min(len(prev_words_lower), len(curr_words_lower))
            overlap_n = 0
            for n in range(max_check, 1, -1):
                if prev_words_lower[-n:] == curr_words_lower[:n]:
                    overlap_n = n
                    break

            if overlap_n >= 2:
                # Trim overlapping tail from previous segment
                orig_prev_words = deduped[-1]["text"].split()
                trimmed = " ".join(orig_prev_words[:-overlap_n])
                if trimmed.strip():
                    prev_start = deduped[-1]["start"]
                    prev_end = deduped[-1]["end"]
                    kept_ratio = len(orig_prev_words[:-overlap_n]) / len(orig_prev_words)
                    deduped[-1]["text"] = trimmed.strip()
                    deduped[-1]["end"] = prev_start + (prev_end - prev_start) * kept_ratio
                else:
                    # Previous was entirely overlap — replace with current
                    deduped[-1] = seg
                    continue
                deduped.append(seg)
            else:
                deduped.append(seg)

        # Merge adjacent segments with identical text
        merged = [deduped[0]]
        for seg in deduped[1:]:
            if seg["text"] == merged[-1]["text"] and seg["start"] - merged[-1]["end"] < 0.5:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg)

        return merged

    def _parse_srt_file(self, srt_path: Path) -> List[Dict]:
        """Parse an SRT file into pipeline segment format.

        Applies the same rolling-overlap dedup and inline noise stripping
        as _parse_vtt, since YouTube auto-captions in SRT format have
        the same rolling two-line pattern.
        """
        content = srt_path.read_text(encoding="utf-8", errors="replace")
        segments_raw: List[Dict] = []
        pattern = re.compile(
            r"\d+\s*\n"
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n"
            r"((?:(?!\d+\s*\n\d{2}:\d{2}).+\n?)+)",
            re.MULTILINE,
        )
        for m in pattern.finditer(content):
            start_str = m.group(1).replace(",", ".")
            end_str = m.group(2).replace(",", ".")
            text = re.sub(r"<[^>]+>", "", m.group(3)).strip()
            text = " ".join(text.split())
            # Strip inline noise tags like [music], [applause] from within text
            text = self._NOISE_INLINE_RE.sub(" ", text).strip()
            if text and not self._NOISE_RE.match(text):
                segments_raw.append({
                    "start": self._vtt_time_to_seconds(start_str),
                    "end": self._vtt_time_to_seconds(end_str),
                    "text": text,
                })

        if not segments_raw:
            return []

        # Deduplicate rolling two-line format (same logic as _parse_vtt).
        # YouTube auto-captions repeat the tail of cue N at the head of cue N+1.
        deduped = [segments_raw[0]]
        for seg in segments_raw[1:]:
            prev_text = deduped[-1]["text"]
            curr_text = seg["text"]

            # Full substring containment
            if prev_text in curr_text:
                deduped[-1] = seg
                continue
            if curr_text in prev_text:
                continue

            # Partial trailing/leading overlap
            prev_words_lower = prev_text.lower().split()
            curr_words_lower = curr_text.lower().split()
            max_check = min(len(prev_words_lower), len(curr_words_lower))
            overlap_n = 0
            for n in range(max_check, 1, -1):
                if prev_words_lower[-n:] == curr_words_lower[:n]:
                    overlap_n = n
                    break

            if overlap_n >= 2:
                # Trim overlapping tail from previous segment
                orig_prev_words = deduped[-1]["text"].split()
                trimmed = " ".join(orig_prev_words[:-overlap_n])
                if trimmed.strip():
                    prev_start = deduped[-1]["start"]
                    prev_end = deduped[-1]["end"]
                    kept_ratio = len(orig_prev_words[:-overlap_n]) / len(orig_prev_words)
                    deduped[-1]["text"] = trimmed.strip()
                    deduped[-1]["end"] = prev_start + (prev_end - prev_start) * kept_ratio
                else:
                    deduped[-1] = seg
                    continue
                deduped.append(seg)
            else:
                deduped.append(seg)

        # Merge adjacent segments with identical text
        merged = [deduped[0]]
        for seg in deduped[1:]:
            if seg["text"] == merged[-1]["text"] and seg["start"] - merged[-1]["end"] < 0.5:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg)

        return merged

    def _fetch_youtube_subtitles(self, url: str) -> Optional[List[Dict]]:
        """Download and parse YouTube subtitles via yt-dlp. Returns segments or None.

        Single yt-dlp call with all common languages + auto-generated subs.
        ~5-10s per video. Battle-tested, updated daily, rarely breaks.

        Languages are tried in PRIORITY ORDER (not filesystem order) so we
        always prefer the source language → English → other common ones.
        """
        if not re.match(r"^https?://", url):
            return None

        # GUARD: NO cookies for subtitle downloads. Premium cookies trigger
        # YouTube's JS signature challenge which kills the entire yt-dlp process.
        # Subtitles are public — no authentication needed. See 2026-04-13 fix.
        # cookies_args = self._get_cookies_args()  # DO NOT USE for subs
        sub_dir = self.cfg.work_dir / "subs"
        sub_dir.mkdir(exist_ok=True)
        out_tpl = str(sub_dir / "sub.%(ext)s")

        # Build PRIORITY-ORDERED language list — explicit source first, English next,
        # then common fallbacks. Order matters: we pick the FIRST one that has subs.
        if self.cfg.source_language and self.cfg.source_language != "auto":
            priority_langs = [
                self.cfg.source_language,          # user-specified source
                f"{self.cfg.source_language}-US",  # variant
                f"{self.cfg.source_language}-GB",  # variant
                "en", "en-US", "en-GB",            # English fallback
            ]
        else:
            # auto-detect: English first (most YouTube content), then common ones
            priority_langs = [
                "en", "en-US", "en-GB",
                "hi", "hi-IN",
                "zh", "zh-Hans", "zh-Hant",
                "ja", "ko", "es", "ru", "fr", "de", "pt",
            ]
        # de-dup while preserving order
        seen = set()
        priority_langs = [l for l in priority_langs if not (l in seen or seen.add(l))]
        langs_csv = ",".join(priority_langs)

        # Single call: try BOTH manual and auto-generated subs in one shot.
        # Retry up to 3 times on failure (429 rate limits are common even with
        # Premium cookies — a 2-3 second delay usually resolves them).
        # Timeout scaled for long videos: 30s base + 1s per 10 minutes of source.
        source_dur = getattr(self, "_source_video_duration", 0.0) or 0.0
        sub_timeout = max(30, int(30 + source_dur / 600))
        cmd = [
            self._ytdlp,
            "--write-sub",          # manual subs (rare, but BEST quality)
            "--write-auto-sub",     # auto-generated (common)
            "--sub-lang", langs_csv,
            "--sub-format", "vtt/srt/best",
            "--skip-download",
            "--no-warnings",
            "-o", out_tpl,
            url,
        ]
        # NOTE: cookies intentionally omitted for subtitle-only downloads.
        # Premium cookies trigger YouTube's JS signature challenge which
        # aborts the process. Subtitles are public — no auth needed.

        import time as _sub_time
        for _attempt in range(3):
            try:
                self._run_proc(cmd, capture_output=True, text=True,
                               timeout=sub_timeout)
                break  # success
            except Exception as _sub_err:
                if _attempt < 2:
                    delay = 5 * (_attempt + 1)  # 5s, 10s (429 rate limits)
                    print(f"[YT Subs] Attempt {_attempt + 1}/3 failed: {_sub_err} "
                          f"— retrying in {delay}s", flush=True)
                    _sub_time.sleep(delay)
                else:
                    print(f"[YT Subs] All 3 attempts failed: {_sub_err}", flush=True)
                    return None

        # Walk languages in PRIORITY ORDER, not filesystem order.
        # For each language, prefer manual track (sub.en.vtt) over auto-caption.
        # yt-dlp file naming: sub.<lang>.<ext> for manual, sub.<lang>.<ext> for auto too,
        # so we just check each candidate language by name.
        for lang in priority_langs:
            # Manual + auto-cap end up with the same filename pattern in this output template,
            # so a single match per language is enough. Try .vtt first, then .srt.
            for ext in ("vtt", "srt"):
                candidate = sub_dir / f"sub.{lang}.{ext}"
                if not candidate.exists():
                    continue
                if ext == "vtt":
                    segments = self._parse_vtt(candidate)
                else:
                    segments = self._parse_srt_file(candidate)
                if segments:
                    print(f"[YT Subs] Picked {candidate.name} "
                          f"({len(segments)} segments, lang={lang})", flush=True)
                    return segments

        # Last-resort fallback: take ANY remaining file (rare — yt-dlp downloaded
        # something but in an unexpected language code).
        for vtt_file in sorted(sub_dir.glob("*.vtt")):
            segments = self._parse_vtt(vtt_file)
            if segments:
                print(f"[YT Subs] Fallback to {vtt_file.name} "
                      f"({len(segments)} segments) — language not in priority list", flush=True)
                return segments
        for srt_file in sorted(sub_dir.glob("*.srt")):
            segments = self._parse_srt_file(srt_file)
            if segments:
                print(f"[YT Subs] Fallback to {srt_file.name} "
                      f"({len(segments)} segments) — language not in priority list", flush=True)
                return segments

        return None

    def _fetch_youtube_translated_subs(self, url: str) -> Optional[List[Dict]]:
        """Download YouTube's auto-translated subtitles in the target language.

        YouTube can translate auto-captions to any language on-the-fly.
        yt-dlp format: --sub-lang {target}-{source} for translated subs.
        This skips both Whisper AND our translation step.
        """
        if not re.match(r"^https?://", url):
            return None

        # GUARD: NO cookies for subtitle downloads. Premium cookies trigger
        # YouTube's JS signature challenge which kills the entire yt-dlp process.
        # Subtitles are public — no authentication needed. See 2026-04-13 fix.
        # cookies_args = self._get_cookies_args()  # DO NOT USE for subs
        sub_dir = self.cfg.work_dir / "yt_translated"
        sub_dir.mkdir(exist_ok=True)

        target = self.cfg.target_language        # "hi"
        source = self.cfg.source_language or "en"
        if source == "auto":
            source = "en"

        # ONE attempt only: hi-en (Hindi auto-translated from English).
        # Pipeline is locked to English→Hindi so we never need to guess.
        sub_lang = f"{target}-{source}" if target != source else target

        # Clean previous attempts
        for f in sub_dir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass

        # Retry up to 3 times — YouTube's auto-translate endpoint is rate-limited
        # and returns 429 even with Premium cookies. A 3-second delay between
        # attempts usually resolves it. Timeout scaled for long videos.
        source_dur = getattr(self, "_source_video_duration", 0.0) or 0.0
        sub_timeout = max(30, int(30 + source_dur / 600))
        # NOTE: DO NOT pass cookies for subtitle-only downloads.
        # Premium cookies trigger YouTube's JS signature challenge which
        # causes a "Requested format is not available" error that aborts
        # the entire process — even though subtitles don't need auth.
        # Subtitles are public and download fine without cookies.
        cmd = [
            self._ytdlp,
            "--write-auto-sub",
            "--sub-lang", sub_lang,
            "--sub-format", "vtt/srt/best",
            "--skip-download",
            "--no-warnings",
            "-o", str(sub_dir / "ytsub.%(ext)s"),
            url,
        ]

        import time as _sub_time
        for _attempt in range(3):
            try:
                self._run_proc(cmd, capture_output=True, text=True,
                               timeout=sub_timeout,
                               encoding="utf-8", errors="replace")
                break  # success
            except Exception as e:
                if _attempt < 2:
                    delay = 5 * (_attempt + 1)  # 5s, 10s (429 rate limits)
                    print(f"[YT Translated] Attempt {_attempt + 1}/3 failed: {e} "
                          f"— retrying in {delay}s", flush=True)
                    _sub_time.sleep(delay)
                    # Clean partial files before retry
                    for f in sub_dir.glob("*"):
                        try: f.unlink()
                        except OSError: pass
                else:
                    print(f"[YT Translated] All 3 attempts failed: {e}", flush=True)
                    return None

        # Find the result (single language, single file)
        for vtt_file in sub_dir.glob("*.vtt"):
            segments = self._parse_vtt(vtt_file)
            if segments:
                self._report("translate", 0.5,
                             f"Got YouTube auto-translated {target} subs from {source} "
                             f"({len(segments)} segments)")
                for seg in segments:
                    seg["text_translated"] = seg.get("text", "")
                return segments
        for srt_file in sub_dir.glob("*.srt"):
            segments = self._parse_srt_file(srt_file)
            if segments:
                self._report("translate", 0.5,
                             f"Got YouTube auto-translated {target} subs from {source} "
                             f"({len(segments)} segments)")
                for seg in segments:
                    seg["text_translated"] = seg.get("text", "")
                return segments

        print(f"[YT Translated] No {sub_lang} subs available for this video", flush=True)
        return None

    def _fetch_reference_subs(self, url: str) -> Optional[List[Dict]]:
        """Fetch English reference subs: try YouTube subs first, then OCR burned-in subs."""
        # 1. Try YouTube subtitle download
        if re.match(r"^https?://", url):
            # GUARD: NO cookies for subtitle downloads. See 2026-04-13 fix.
            ref_dir = self.cfg.work_dir / "ref_subs"
            ref_dir.mkdir(exist_ok=True)
            out_tpl = str(ref_dir / "ref.%(ext)s")

            for write_flag in ["--write-sub", "--write-auto-sub"]:
                for f in ref_dir.glob("*"):
                    try:
                        f.unlink()
                    except OSError:
                        pass
                cmd = [
                    self._ytdlp, write_flag,
                    "--sub-lang", "en",
                    "--sub-format", "vtt/srt/best",
                    "--skip-download",
                    "-o", out_tpl,
                    url,
                ]
                try:
                    self._run_proc(cmd, capture_output=True, text=True, timeout=30)
                except Exception:
                    continue
                for vtt_file in ref_dir.glob("*.vtt"):
                    segments = self._parse_vtt(vtt_file)
                    if segments:
                        return segments
                for srt_file in ref_dir.glob("*.srt"):
                    segments = self._parse_srt_file(srt_file)
                    if segments:
                        return segments

        # 2. Fallback: OCR burned-in subtitles from video frames
        video_path = self._find_source_video()
        if video_path:
            ocr_segs = self._ocr_burned_subs(video_path)
            if ocr_segs:
                return ocr_segs

        return None

    def _find_source_video(self) -> Optional[Path]:
        """Find the downloaded source video in work directory."""
        for ext in ("mp4", "mkv", "webm", "avi"):
            for f in self.cfg.work_dir.glob(f"source.{ext}"):
                return f
        return None

    def _ocr_burned_subs(self, video_path: Path, sample_interval: float = 1.0) -> Optional[List[Dict]]:
        """Extract burned-in English subtitles from video using OCR on sampled frames."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            print("[OCR] pytesseract or Pillow not installed, skipping OCR", flush=True)
            return None

        print(f"[OCR] Extracting burned-in subs from {video_path.name}...", flush=True)
        frames_dir = self.cfg.work_dir / "ocr_frames"
        frames_dir.mkdir(exist_ok=True)

        # Get video duration
        try:
            dur_result = self._run_proc(
                [self._ffmpeg, "-i", str(video_path)],
                capture_output=True, text=True, timeout=10,
            )
            dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", dur_result.stderr)
            if dur_match:
                h, m, s, cs = [int(x) for x in dur_match.groups()]
                total_dur = h * 3600 + m * 60 + s + cs / 100
            else:
                total_dur = 600  # default 10 min
        except Exception:
            total_dur = 600

        # Sample frames (bottom 25% where subs usually are)
        # Extract 1 frame per second for first 5 min, then every 2s
        max_frames = min(int(total_dur / sample_interval), 600)
        self._run_proc(
            [self._ffmpeg, "-y", "-i", str(video_path),
             "-vf", f"fps=1/{sample_interval},crop=iw:ih/4:0:ih*3/4",
             "-frames:v", str(max_frames),
             str(frames_dir / "frame_%04d.png")],
            capture_output=True, timeout=300,
        )

        # OCR each frame
        raw_texts = []  # list of (timestamp, text)
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        print(f"[OCR] Processing {len(frame_files)} frames...", flush=True)

        for i, frame_file in enumerate(frame_files):
            timestamp = i * sample_interval
            try:
                with Image.open(frame_file) as img:
                    text = pytesseract.image_to_string(img, lang="eng").strip()
                # Clean up OCR noise
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 3:  # skip noise/empty
                    raw_texts.append((timestamp, text))
            except Exception:
                pass

        # Clean up frames
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)

        if not raw_texts:
            print("[OCR] No text found in video frames", flush=True)
            return None

        # Group consecutive identical texts into segments
        segments = []
        prev_text = None
        seg_start = 0.0
        for ts, text in raw_texts:
            if text != prev_text:
                if prev_text:
                    segments.append({
                        "start": seg_start,
                        "end": ts,
                        "text": prev_text,
                    })
                seg_start = ts
                prev_text = text
        # Last segment
        if prev_text:
            segments.append({
                "start": seg_start,
                "end": raw_texts[-1][0] + sample_interval,
                "text": prev_text,
            })

        print(f"[OCR] Extracted {len(segments)} subtitle segments from video", flush=True)
        return segments if segments else None

    def _qa_post_translation(self, our_segs: List[Dict], ref_english_segs: List[Dict]) -> Dict:
        """Compare our pipeline output against reference English subs.

        Strategy:
        - If target is English: compare our translated text directly against reference
        - If target is non-English: compare our SOURCE text (what Whisper heard → translated to English
          via text_translated if target=en, or the original English text from ref) against reference
        - Uses timestamp alignment + text similarity
        """
        from difflib import SequenceMatcher

        is_english_target = self.cfg.target_language == "en"
        matched = 0
        total = 0
        report_lines = [
            "=== Translation QA Report ===",
            f"Target language: {self.cfg.target_language}",
            f"Our segments: {len(our_segs)}, Reference English subs: {len(ref_english_segs)}",
            "",
        ]

        for our_seg in our_segs:
            # Use translated text if target is English, otherwise use source text
            if is_english_target:
                our_text = our_seg.get("text_translated", our_seg.get("text", "")).strip()
            else:
                # For non-English targets, we compare the original source text
                # This only works well if source is English; for Chinese→Hindi,
                # we compare timing/coverage instead of text
                our_text = our_seg.get("text", "").strip()

            if not our_text:
                continue
            total += 1
            our_start = our_seg.get("start", 0)
            our_end = our_seg.get("end", 0)

            # Find best time-aligned reference segment
            best_ratio = 0.0
            best_ref = ""
            best_time_overlap = 0.0
            for r_seg in ref_english_segs:
                r_start = r_seg.get("start", 0)
                r_end = r_seg.get("end", 0)
                overlap = min(our_end, r_end) - max(our_start, r_start)
                if overlap > -2.0:
                    r_text = r_seg.get("text", "").strip()
                    if is_english_target:
                        # Direct comparison: our English vs reference English
                        ratio = SequenceMatcher(None, our_text.lower(), r_text.lower()).ratio()
                    else:
                        # Non-English target: check if we have a segment covering this time
                        # Score based on time alignment quality
                        seg_dur = max(our_end - our_start, 0.1)
                        ratio = max(overlap / seg_dur, 0) if overlap > 0 else 0
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_ref = r_text
                        best_time_overlap = overlap

            if is_english_target:
                status = "MATCH" if best_ratio >= 0.5 else "MISMATCH" if best_ref else "NO_REF"
                if best_ratio >= 0.5:
                    matched += 1
            else:
                # For non-English: count as matched if there's good time coverage
                status = "COVERED" if best_time_overlap > 0.5 else "GAP" if best_ref else "NO_REF"
                if best_time_overlap > 0.5:
                    matched += 1

            report_lines.append(
                f"[{our_start:.1f}s - {our_end:.1f}s] {status} ({best_ratio:.0%})"
            )
            report_lines.append(f"  Ours:      {our_text[:120]}")
            report_lines.append(f"  Reference: {best_ref[:120]}")
            if not is_english_target:
                translated = our_seg.get("text_translated", "").strip()
                if translated:
                    report_lines.append(f"  Translated: {translated[:120]}")
            report_lines.append("")

        score = matched / total if total > 0 else 0.0
        summary = f"\nQA Score: {matched}/{total} segments verified ({score:.0%})\n"
        report_lines.insert(3, summary)

        return {
            "score": score,
            "matched": matched,
            "total": total,
            "report": "\n".join(report_lines),
        }

    # ── Step 3b: Transcribe speech from audio ─────────────────────────────
    def _transcribe(self, wav_path: Path) -> List[Dict]:
        """Transcribe speech from audio — tries Groq Whisper API first (30x faster),
        falls back to local faster-whisper if no API key or API fails.
        Results are cached by audio SHA-256 + model + language."""
        lang = self.cfg.source_language or "auto"
        model = self.cfg.asr_model

        # Check ASR cache first
        cached = _cache.get_asr(wav_path, model, lang)
        if cached is not None:
            self._report("transcribe", 0.95, f"ASR cache hit — {len(cached)} segments (skipping transcription)")
            return cached

        # Try Groq Whisper first (cloud, ~30x faster). On ANY failure or empty
        # result, fall straight through to local faster-whisper large-v3.
        groq_attempted = False
        if model == "groq-whisper" or model in ("groq", "auto"):
            groq_key = get_groq_key()
            if groq_key:
                groq_attempted = True
                try:
                    self._report("transcribe", 0.1, "Using Groq Whisper (cloud, fastest)...")
                    segments = self._transcribe_groq(wav_path, groq_key)
                    if not segments:
                        raise RuntimeError("Groq Whisper returned 0 segments")
                    if self.cfg.use_whisperx and segments:
                        segments = self._whisperx_align(wav_path, segments)
                    segments = self._dedup_segments(segments)
                    _cache.put_asr(wav_path, model, lang, segments)
                    return segments
                except Exception as e:
                    self._report("transcribe", 0.1,
                                 f"Groq Whisper failed ({e}) — falling back to local Whisper large-v3...")
            else:
                self._report("transcribe", 0.1, "No GROQ_API_KEY — using local Whisper large-v3...")

        segments = self._transcribe_local(wav_path)

        # Optional: refine word-level timestamps with WhisperX forced alignment
        if self.cfg.use_whisperx and segments:
            segments = self._whisperx_align(wav_path, segments)

        segments = self._dedup_segments(segments)
        _cache.put_asr(wav_path, model, lang, segments)
        return segments

    def _dedup_segments(self, segments: List[Dict]) -> List[Dict]:
        """Remove duplicate/overlapping segments from Whisper output.

        Whisper produces duplicates in three scenarios:
        1. Chunked transcription: boundary segments appear in both chunks
        2. Hallucination: same phrase repeated with slightly different timestamps
        3. Stuttered audio: Whisper outputs the same text twice

        Detection:
        - Time overlap > 50% between consecutive segments → duplicate
        - Same/very similar text with nearby timestamps → duplicate
        - Exact same text within 3 seconds → duplicate
        """
        if not segments or len(segments) < 2:
            return segments

        # Strip inline noise tags ([music], [applause], etc.) from all segments
        for seg in segments:
            text = seg.get("text", "")
            if text:
                cleaned = self._NOISE_INLINE_RE.sub(" ", text).strip()
                if cleaned != text:
                    seg["text"] = cleaned

        # Remove segments that became empty after noise stripping
        segments = [s for s in segments if s.get("text", "").strip()]
        if len(segments) < 2:
            return segments

        # Sort by start time first
        segments = sorted(segments, key=lambda s: s.get("start", 0))
        deduped = [segments[0]]

        for seg in segments[1:]:
            prev = deduped[-1]
            prev_start = prev.get("start", 0)
            prev_end = prev.get("end", 0)
            prev_text = prev.get("text", "").strip().lower()
            cur_start = seg.get("start", 0)
            cur_end = seg.get("end", 0)
            cur_text = seg.get("text", "").strip().lower()

            if not cur_text:
                continue

            prev_dur = max(prev_end - prev_start, 0.01)
            cur_dur = max(cur_end - cur_start, 0.01)

            # Check time overlap
            overlap = max(0, min(prev_end, cur_end) - max(prev_start, cur_start))
            overlap_ratio = overlap / min(prev_dur, cur_dur)

            # Case 1: High time overlap (>50%) — chunk boundary duplicate
            if overlap_ratio > 0.5:
                # Keep the longer segment (more complete)
                if cur_dur > prev_dur:
                    deduped[-1] = seg
                continue

            # Case 2: Same or nearly same text within 3 seconds
            if cur_start - prev_start < 3.0:
                # Exact match
                if cur_text == prev_text:
                    # Keep the one with longer duration
                    if cur_dur > prev_dur:
                        deduped[-1] = seg
                    continue

                # Substring match (one contains the other)
                if cur_text in prev_text or prev_text in cur_text:
                    # Keep the longer text
                    if len(cur_text) > len(prev_text):
                        deduped[-1] = seg
                    continue

                # High word overlap (>70% shared words)
                prev_words = set(prev_text.split())
                cur_words = set(cur_text.split())
                if prev_words and cur_words:
                    shared = len(prev_words & cur_words)
                    word_overlap = shared / min(len(prev_words), len(cur_words))
                    if word_overlap > 0.7:
                        if len(cur_text) > len(prev_text):
                            deduped[-1] = seg
                        continue

            # Case 3: Segment starts before previous ends (time regression)
            if cur_start < prev_end - 0.1:
                # Overlapping — keep both only if text is clearly different
                prev_words = set(prev_text.split())
                cur_words = set(cur_text.split())
                if prev_words and cur_words:
                    shared = len(prev_words & cur_words)
                    word_overlap = shared / min(len(prev_words), len(cur_words))
                    if word_overlap > 0.35:
                        # Too similar — skip duplicate
                        if cur_dur > prev_dur:
                            deduped[-1] = seg
                        continue

            # Case 4: Sequential word overlap (rolling/sliding window pattern)
            # Last N words of prev == first N words of curr → trim prev tail
            prev_wlist = prev_text.split()
            cur_wlist = cur_text.split()
            max_seq = min(len(prev_wlist), len(cur_wlist))
            seq_overlap = 0
            for n in range(max_seq, 1, -1):
                if prev_wlist[-n:] == cur_wlist[:n]:
                    seq_overlap = n
                    break
            if seq_overlap >= 2:
                # Trim overlap from previous, keep current as-is
                orig_prev_words = prev.get("text", "").strip().split()
                trimmed = " ".join(orig_prev_words[:-seq_overlap])
                if trimmed.strip():
                    p_start = prev.get("start", 0)
                    p_end = prev.get("end", 0)
                    ratio = len(orig_prev_words[:-seq_overlap]) / len(orig_prev_words)
                    deduped[-1]["text"] = trimmed.strip()
                    deduped[-1]["end"] = p_start + (p_end - p_start) * ratio
                else:
                    # Previous was entirely overlap — replace with current
                    deduped[-1] = seg
                    continue

            deduped.append(seg)

        if len(deduped) < len(segments):
            print(f"[Dedup] {len(segments)} -> {len(deduped)} segments "
                  f"(removed {len(segments) - len(deduped)} duplicates)", flush=True)

        return deduped

    # Sentence-ending punctuation (English + Hindi/Devanagari)
    _SENTENCE_ENDERS = re.compile(r'[.!?।॥]$')

    def _merge_broken_sentences(self, segments: List[Dict]) -> List[Dict]:
        """Merge segments that Whisper split mid-sentence.

        Whisper segments by silence gaps, not sentence boundaries. This causes:
        - Seg 1: "Today we will learn about machine le-"
        - Seg 2: "-arning and how it works"

        Fix: if a segment doesn't end with sentence-ending punctuation (.!?।),
        merge it with the next segment — as long as the gap is small (<1.5s)
        and combined duration stays reasonable (<15s for TTS).
        """
        if not segments or len(segments) < 2:
            return segments

        MAX_GAP = 1.5       # Max silence gap to merge across (seconds)
        MAX_MERGED_DUR = 15  # Don't create segments longer than 15s
        merged = []
        i = 0

        while i < len(segments):
            current = dict(segments[i])  # copy
            text = current.get("text", "").strip()

            # Keep merging while current segment doesn't end with sentence punctuation
            while (
                i + 1 < len(segments)
                and text
                and not self._SENTENCE_ENDERS.search(text)
            ):
                next_seg = segments[i + 1]
                next_text = next_seg.get("text", "").strip()
                if not next_text:
                    i += 1
                    continue

                # Check gap between segments
                gap = next_seg.get("start", 0) - current.get("end", 0)
                if gap > MAX_GAP:
                    break  # Too far apart — probably a real pause

                # Check combined duration
                combined_dur = next_seg.get("end", 0) - current.get("start", 0)
                if combined_dur > MAX_MERGED_DUR:
                    break  # Would be too long for TTS

                # Merge: extend current segment to include next
                current["end"] = next_seg["end"]
                current["text"] = text + " " + next_text
                # Also merge text_translated if EITHER segment has it (YT Auto-Translate path)
                if current.get("text_translated") or next_seg.get("text_translated"):
                    cur_tr = current.get("text_translated", "").strip()
                    nxt_tr = next_seg.get("text_translated", "").strip()
                    current["text_translated"] = (cur_tr + " " + nxt_tr).strip()
                # Preserve speaker from first segment
                text = current["text"].strip()
                i += 1

            merged.append(current)
            i += 1

        if len(merged) < len(segments):
            print(f"[Sentence-merge] {len(segments)} -> {len(merged)} segments "
                  f"(merged {len(segments) - len(merged)} broken sentences)", flush=True)

        return merged

    def _combine_sentence_group(self, group: List[Dict]) -> Dict:
        """Combine a list of sentences into a single segment dict."""
        combined_text = " ".join(
            s.get("text", "").strip() for s in group if s.get("text", "").strip()
        )
        combined = {
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": combined_text,
        }
        # Also concatenate text_translated if present (YT Auto-Translate path)
        if any(s.get("text_translated") for s in group):
            combined["text_translated"] = " ".join(
                s.get("text_translated", "").strip() for s in group
                if s.get("text_translated", "").strip()
            )
        # Preserve extra keys from first segment (emotion, speaker_id, etc.)
        for key in group[0]:
            if key not in ("start", "end", "text", "text_translated"):
                combined.setdefault(key, group[0][key])
        return combined

    def _group_sentences_by_count(self, sentences: List[Dict],
                                  target_per_group: int = 2,
                                  word_tolerance: float = 0.15) -> List[Dict]:
        """Group sentences into segments of exactly N sentences each.

        Sentences are the ATOMIC UNIT — never split. Grouping is purely
        by sentence count. Word count is NOT used for grouping decisions;
        it is only used LATER in _redistribute_slots_by_wordcount to
        assign fair timeline slots to each segment.

        Rules:
          - Each segment gets exactly target_per_group sentences
          - Last segment gets the remainder (1 to target_per_group)
          - If last segment is a single orphan sentence, merge it into
            the previous segment (so last segment gets target+1 instead)
          - Every sentence ends up in exactly one segment — none lost

        Example (target=2, 7 sentences):
          [S1, S2, S3, S4, S5, S6, S7]
          -> [Seg1(S1+S2), Seg2(S3+S4), Seg3(S5+S6+S7)]
             (S7 merged into Seg3 instead of orphan Seg4)
        """
        if not sentences:
            return []
        if len(sentences) <= target_per_group:
            return [self._combine_sentence_group(sentences)]

        # ── Simple fixed-count grouping ──
        raw_groups: List[List[Dict]] = []
        for i in range(0, len(sentences), target_per_group):
            raw_groups.append(sentences[i:i + target_per_group])

        # ── Merge orphan: if last group is a single sentence, fold into previous ──
        if len(raw_groups) >= 2 and len(raw_groups[-1]) == 1:
            raw_groups[-2].extend(raw_groups[-1])
            raw_groups.pop()

        # ── Combine into segment dicts ──
        grouped = [self._combine_sentence_group(g) for g in raw_groups]

        # ── Logging ──
        total_words = sum(len(g.get("text", "").split()) for g in grouped)
        seg_wcs = [len(g.get("text", "").split()) for g in grouped]
        seg_counts = [len(g) for g in raw_groups]
        min_wc, max_wc = min(seg_wcs), max(seg_wcs)
        avg_wc = total_words / len(grouped) if grouped else 0
        print(f"[Sentence-group] {len(sentences)} sentences -> {len(grouped)} segments "
              f"| {target_per_group} sent/seg (sentence-first, never split) "
              f"| words: total={total_words}, avg={avg_wc:.1f}, "
              f"min={min_wc}, max={max_wc} "
              f"| group sizes: {seg_counts}", flush=True)

        # Verify: every sentence accounted for
        total_in_groups = sum(seg_counts)
        if total_in_groups != len(sentences):
            print(f"[Sentence-group] WARNING: {total_in_groups} sentences in groups "
                  f"vs {len(sentences)} input — mismatch!", flush=True)

        return grouped

    def _glossary_mask(self, segments: List[Dict]) -> int:
        """Mask glossary words BEFORE translation with placeholders.

        Each glossary entry gets a FIXED placeholder based on its index in
        the glossary (e.g., first entry = __GL000__, second = __GL001__).
        Same word in different segments gets the same placeholder — unmask
        uses the same mapping so it always restores correctly.

        Returns count of masked occurrences.
        """
        if not self._glossary:
            return 0

        import re as _gre
        # Build fixed placeholder mapping: each glossary entry gets a stable index
        _gl_entries = list(self._glossary.items())
        _gl_patterns = []
        for idx, (eng, target) in enumerate(_gl_entries):
            pattern = _gre.compile(r'\b' + _gre.escape(eng) + r'\b', _gre.IGNORECASE)
            placeholder = f"__GL{idx:04d}__"
            _gl_patterns.append((pattern, placeholder, target))

        masked = 0
        for seg in segments:
            text = seg.get("text", "")
            if not text:
                continue
            replacements = {}
            for pattern, placeholder, target in _gl_patterns:
                if pattern.search(text):
                    text = pattern.sub(placeholder, text)
                    replacements[placeholder] = target
                    masked += 1
            if replacements:
                seg["text"] = text
                seg["_glossary_map"] = replacements
        if masked:
            print(f"[Glossary] Masked {masked} words before translation", flush=True)
        return masked

    def _glossary_unmask(self, segments: List[Dict]) -> int:
        """Restore glossary placeholders AFTER translation.

        Finds __GLxx__ placeholders in text_translated and replaces them
        with the user's desired word from the glossary.

        Two passes:
        1. Per-segment map (handles tokens that stayed in their original segment)
        2. Global sweep using the full glossary (catches tokens that migrated
           across segment boundaries due to overlapping VTT cues or dedup)
        """
        import re as _gre

        restored = 0

        # Build global placeholder→english and placeholder→target maps
        _gl_global_target = {}   # placeholder → glossary target (Hindi)
        _gl_global_english = {}  # placeholder → original English word
        if self._glossary:
            for idx, (eng, target) in enumerate(self._glossary.items()):
                ph = f"__GL{idx:04d}__"
                _gl_global_target[ph] = target
                _gl_global_english[ph] = eng

        # Pass 1: per-segment map
        for seg in segments:
            gmap = seg.pop("_glossary_map", None)
            if not gmap:
                continue
            # Restore translated text
            translated = seg.get("text_translated", "")
            if translated:
                for placeholder, target in gmap.items():
                    if placeholder in translated:
                        translated = translated.replace(placeholder, target)
                        restored += 1
                seg["text_translated"] = translated
            # Restore original English text (undo masking for display)
            orig = seg.get("text", "")
            if orig:
                for placeholder in gmap:
                    eng_word = _gl_global_english.get(placeholder, placeholder)
                    if placeholder in orig:
                        orig = orig.replace(placeholder, eng_word)
                seg["text"] = orig

        # Pass 2: global sweep — catch tokens that migrated across segment boundaries
        if _gl_global_target:
            _token_re = _gre.compile(r'__GL\d{4}__')
            for seg in segments:
                # Sweep text_translated
                translated = seg.get("text_translated", "")
                if translated and _token_re.search(translated):
                    for placeholder, target in _gl_global_target.items():
                        if placeholder in translated:
                            translated = translated.replace(placeholder, target)
                            restored += 1
                    seg["text_translated"] = translated
                # Sweep original text
                orig = seg.get("text", "")
                if orig and _token_re.search(orig):
                    for placeholder, eng_word in _gl_global_english.items():
                        if placeholder in orig:
                            orig = orig.replace(placeholder, eng_word)
                    seg["text"] = orig

        if restored:
            print(f"[Glossary] Restored {restored} words after translation", flush=True)
        return restored

    def _glossary_post_replace(self, segments: List[Dict]) -> int:
        """Post-translation glossary for YouTube Hindi path.

        YouTube Hindi auto-translate doesn't see our placeholders. Instead,
        we translate each glossary English word to Hindi (via Google Translate)
        to discover what YouTube likely used, then find-and-replace in the
        translated text with the glossary's desired output.

        The Hindi lookups are cached on self._glossary_hindi so they're only
        computed once per pipeline run.
        """
        if not self._glossary:
            return 0

        # Build Hindi lookup cache if not already done
        if not hasattr(self, '_glossary_hindi') or not self._glossary_hindi:
            self._glossary_hindi: Dict[str, List[str]] = {}
            # Hardcoded common translations (fast, no API call)
            _KNOWN = {
                "noble": ["कुलीन", "महान", "उत्कृष्ट", "शाही", "नेक"],
                "king": ["राजा", "किंग"],
                "queen": ["रानी", "क्वीन"],
                "princess": ["राजकुमारी", "प्रिंसेस"],
                "prince": ["राजकुमार", "युवराज", "प्रिंस"],
                "general": ["सेनापति", "जनरल"],
                "consort": ["पत्नी", "राजमहिषी", "संगिनी"],
                "immortal": ["अमर", "देवता", "अमरत्व"],
                "demon": ["राक्षस", "दानव", "असुर", "डीमन"],
                "dragon": ["ड्रैगन", "अजगर", "नाग"],
                "spirit": ["आत्मा", "भूत", "प्राण", "स्पिरिट"],
                "clan": ["कुल", "वंश", "कबीला", "क्लैन"],
                "contract": ["अनुबंध", "करार", "संविदा", "कॉन्ट्रैक्ट"],
                "cultivation": ["साधना", "तपस्या", "खेती"],
                "realm": ["लोक", "क्षेत्र", "राज्य", "दुनिया"],
                "warrior": ["योद्धा", "वारियर", "सैनिक"],
                "phoenix": ["फीनिक्स", "अग्निपक्षी"],
                "emperor": ["सम्राट", "एम्परर", "बादशाह"],
                "heavenly": ["स्वर्गीय", "दिव्य"],
                "divine": ["दिव्य", "देवी", "ईश्वरीय"],
                "beast": ["जानवर", "पशु", "बीस्ट"],
                "sparrow": ["गौरैया", "चिड़िया", "स्पैरो"],
                "fox": ["लोमड़ी", "फॉक्स"],
                "palace": ["महल", "राजमहल", "पैलेस"],
                "throne": ["सिंहासन", "गद्दी", "थ्रोन"],
            }
            for eng in self._glossary:
                forms = _KNOWN.get(eng.lower(), [])
                # Also try quick Google Translate for unknown words
                if not forms:
                    try:
                        tmp_segs = [{"text": eng}]
                        self._translate_segments(tmp_segs)
                        hindi_tr = tmp_segs[0].get("text_translated", "").strip()
                        if hindi_tr and hindi_tr != eng:
                            forms = [hindi_tr]
                            print(f"[Glossary] Looked up '{eng}' -> '{hindi_tr}' via Google", flush=True)
                    except Exception:
                        pass
                self._glossary_hindi[eng] = forms

        replaced = 0
        for seg in segments:
            translated = seg.get("text_translated", "")
            if not translated:
                continue
            changed = False
            for eng, target in self._glossary.items():
                hindi_forms = self._glossary_hindi.get(eng, [])
                for hindi_word in hindi_forms:
                    if hindi_word in translated:
                        translated = translated.replace(hindi_word, target)
                        changed = True
                        replaced += 1
            if changed:
                seg["text_translated"] = translated

        if replaced:
            print(f"[Glossary] Post-replaced {replaced} Hindi words with glossary entries", flush=True)
        return replaced

    def _chunk_segments_for_tts(self, segments: List[Dict],
                                chunk_words: int) -> List[Dict]:
        """Re-split segments into N-word chunks for TTS.

        Takes translated segments (which may be long sentences) and splits
        each into chunks of ~chunk_words. Each chunk gets a proportional
        slice of the original segment's timeline.

        This prevents Edge-TTS truncation on long segments by keeping each
        TTS call short. Tradeoff: smaller chunks = less natural prosody.

        The text_translated field is used for splitting (that's what TTS reads).
        The text (English) field is also chunked proportionally for reference.
        """
        if chunk_words <= 0 or not segments:
            return segments

        result: List[Dict] = []

        for seg in segments:
            hindi = seg.get("text_translated", "").split()
            english = seg.get("text", "").split()
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            seg_dur = seg_end - seg_start

            # If segment is already small enough, keep as-is
            if len(hindi) <= chunk_words:
                result.append(seg)
                continue

            # Split hindi words into chunks
            hindi_total = len(hindi)
            english_total = len(english) if english else 0
            cursor_time = seg_start

            for ci in range(0, hindi_total, chunk_words):
                chunk_hindi = hindi[ci:ci + chunk_words]
                chunk_len = len(chunk_hindi)

                # Proportional time slice
                proportion = chunk_len / hindi_total
                chunk_dur = seg_dur * proportion

                # Proportional English slice
                if english_total > 0:
                    en_start = int((ci / hindi_total) * english_total)
                    en_end = int(((ci + chunk_len) / hindi_total) * english_total)
                    chunk_english = " ".join(english[en_start:en_end])
                else:
                    chunk_english = ""

                chunk_seg = {
                    "start": round(cursor_time, 3),
                    "end": round(cursor_time + chunk_dur, 3),
                    "text": chunk_english,
                    "text_translated": " ".join(chunk_hindi),
                }
                # Preserve extra keys from original
                for key in seg:
                    if key not in ("start", "end", "text", "text_translated"):
                        chunk_seg.setdefault(key, seg[key])

                result.append(chunk_seg)
                cursor_time += chunk_dur

        orig_count = len(segments)
        new_count = len(result)
        if new_count != orig_count:
            print(f"[TTS-chunk] Split {orig_count} segments -> {new_count} chunks "
                  f"(~{chunk_words} words each)", flush=True)

        return result

    def _close_segment_gaps(self, segments: List[Dict]) -> List[Dict]:
        """Make segments perfectly contiguous — no gaps or overlaps.

        1. Sort by start time (handles out-of-order segments)
        2. Fix overlaps (seg[i].end > seg[i+1].start → clamp)
        3. Close gaps (seg[i].end < seg[i+1].start → extend)

        Result: seg[i].end == seg[i+1].start for all adjacent pairs.
        Assembly's gap_mode is the ONLY source of gaps in the output.
        """
        if not segments or len(segments) < 2:
            return segments

        # Sort by start time first
        segments.sort(key=lambda s: s.get("start", 0))

        total_gap = 0.0
        total_overlap = 0.0
        for i in range(len(segments) - 1):
            gap = segments[i + 1].get("start", 0) - segments[i].get("end", 0)
            if gap > 0.01:
                # Gap: extend current segment's end to meet next start
                total_gap += gap
                segments[i]["end"] = segments[i + 1]["start"]
            elif gap < -0.01:
                # Overlap: clamp current segment's end to next start
                total_overlap += abs(gap)
                segments[i]["end"] = segments[i + 1]["start"]

        if total_gap > 0.1 or total_overlap > 0.1:
            parts = []
            if total_gap > 0.1:
                parts.append(f"closed {total_gap:.1f}s gaps")
            if total_overlap > 0.1:
                parts.append(f"fixed {total_overlap:.1f}s overlaps")
            print(f"[Gap-close] {', '.join(parts)} across "
                  f"{len(segments)} segments (now contiguous)", flush=True)

        return segments

    def _redistribute_slots_by_wordcount(self, segments: List[Dict]) -> List[Dict]:
        """Redistribute segment timelines proportionally by word count.

        Keeps the total time span (first start -> last end) identical, but
        gives each segment a slot proportional to its word count. Segments
        with more words get more time -- matching TTS output duration behavior.

        Before: [Seg1(0-12s, 20 words), Seg2(12-22s, 5 words)]
                 Seg1 has 12s for 20w, Seg2 has 10s for 5w (unbalanced)
        After:  [Seg1(0-17.6s, 20 words), Seg2(17.6-22s, 5 words)]
                 Both get 0.88s per word (balanced)
        """
        if not segments or len(segments) < 2:
            return segments

        total_start = segments[0]["start"]
        total_end = segments[-1]["end"]
        total_duration = total_end - total_start

        if total_duration <= 0:
            return segments

        # Count words per segment (min 1 to avoid zero-division)
        word_counts = []
        for seg in segments:
            wc = len(seg.get("text", "").split())
            word_counts.append(max(wc, 1))

        total_words = sum(word_counts)

        # Redistribute: each segment gets time proportional to its word count
        cursor = total_start
        for i, seg in enumerate(segments):
            slot = (word_counts[i] / total_words) * total_duration
            seg["start"] = round(cursor, 3)
            seg["end"] = round(cursor + slot, 3)
            cursor += slot

        # Snap last segment's end to exact total_end (avoid float drift)
        segments[-1]["end"] = total_end

        avg_per_word = total_duration / total_words if total_words else 0
        print(f"[Slot-redistribute] {len(segments)} segments, {total_words} words, "
              f"{total_duration:.1f}s total, {avg_per_word:.3f}s/word avg", flush=True)

        return segments

    def _split_by_even_wordcount(self, segments: List[Dict],
                                 target_words: int = 20) -> List[Dict]:
        """Split segments into even-word-count chunks (no punctuation fallback).

        When YouTube captions have no punctuation, sentence boundaries are
        unreliable. Instead, join ALL text into one stream and split into
        segments of ~target_words each. Gaps are removed in assembly anyway,
        so uniform word density = uniform playback speed.

        Timeline: each output segment gets a proportional slice of the
        total time span based on its word count (same as redistribute).

        No word is lost — every word from input ends up in exactly one output.
        """
        if not segments:
            return []

        # Collect all words + total time span
        all_words: List[str] = []
        for seg in segments:
            words = seg.get("text", "").split()
            all_words.extend(words)

        if not all_words:
            return segments

        total_start = segments[0].get("start", 0)
        total_end = segments[-1].get("end", 0)
        total_duration = max(total_end - total_start, 0.1)
        total_word_count = len(all_words)

        # Also collect translated words if present
        all_translated: List[str] = []
        has_translated = any(s.get("text_translated") for s in segments)
        if has_translated:
            for seg in segments:
                tr_words = seg.get("text_translated", "").split()
                all_translated.extend(tr_words)

        # Split into chunks of ~target_words
        result: List[Dict] = []
        cursor = total_start
        i = 0
        while i < total_word_count:
            chunk_end = min(i + target_words, total_word_count)
            chunk_words = all_words[i:chunk_end]
            chunk_text = " ".join(chunk_words)

            # Proportional time slot
            slot = (len(chunk_words) / total_word_count) * total_duration
            seg_dict = {
                "start": round(cursor, 3),
                "end": round(cursor + slot, 3),
                "text": chunk_text,
            }

            # Proportional translated text if present
            if has_translated and all_translated:
                tr_ratio = len(all_translated) / max(total_word_count, 1)
                tr_start = int(i * tr_ratio)
                tr_end = int(chunk_end * tr_ratio)
                seg_dict["text_translated"] = " ".join(
                    all_translated[tr_start:tr_end])

            result.append(seg_dict)
            cursor += slot
            i = chunk_end

        # Snap last segment end
        if result:
            result[-1]["end"] = total_end

        print(f"[Word-split] {total_word_count} words -> {len(result)} segments "
              f"(~{target_words} words/seg, no punctuation path)", flush=True)

        return result

    def _align_yt_text_to_whisper_timeline(self, yt_sentences: List[Dict],
                                           whisper_segments: List[Dict]) -> List[Dict]:
        """Align YouTube text onto Whisper's precise timeline.

        YouTube gives better TEXT (human-curated captions vs Whisper guesses).
        Whisper gives better TIMESTAMPS (actual audio analysis vs display timing).

        Strategy: walk both lists in parallel (both are chronological). For each
        Whisper segment, find the YouTube sentence whose time overlaps most and
        take the YouTube text. If no good overlap, keep Whisper's own text.

        Returns segments with Whisper start/end but YouTube text.
        """
        if not yt_sentences or not whisper_segments:
            return whisper_segments or yt_sentences or []

        # Build a simple overlap matcher: for each Whisper segment, find the
        # best-matching YouTube sentence by time overlap
        result: List[Dict] = []
        yt_used = set()  # track which YT sentences we've consumed

        for wseg in whisper_segments:
            w_start = wseg.get("start", 0)
            w_end = wseg.get("end", 0)
            w_mid = (w_start + w_end) / 2

            best_idx = -1
            best_overlap = 0.0

            for j, yseg in enumerate(yt_sentences):
                if j in yt_used:
                    continue
                y_start = yseg.get("start", 0)
                y_end = yseg.get("end", 0)

                # Overlap = intersection of [w_start,w_end] and [y_start,y_end]
                overlap_start = max(w_start, y_start)
                overlap_end = min(w_end, y_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = j

            # Take YouTube text if we found a decent overlap (>20% of Whisper seg)
            w_dur = max(w_end - w_start, 0.1)
            if best_idx >= 0 and best_overlap > w_dur * 0.2:
                yt_used.add(best_idx)
                aligned = dict(wseg)  # keep Whisper timeline
                aligned["text"] = yt_sentences[best_idx].get("text", wseg.get("text", ""))
                result.append(aligned)
            else:
                # No good YouTube match -- keep Whisper's own text
                result.append(dict(wseg))

        # Skip unused YouTube sentences — appending them with their imprecise
        # YT timelines would create overlapping segments that break assembly.
        # The matched segments already carry YouTube's text quality where overlap
        # was found; unmatched ones are likely intros/outros or timing mismatches.
        unmatched = len(yt_sentences) - len(yt_used)

        matched = len(yt_used)
        total_yt = len(yt_sentences)
        print(f"[YT-Whisper-align] Matched {matched}/{total_yt} YouTube sentences "
              f"to {len(whisper_segments)} Whisper segments"
              f"{f' (skipped {unmatched} unmatched YT sentences)' if unmatched else ''}",
              flush=True)

        return result

    def _correct_whisper_with_yt(self, whisper_segments: List[Dict],
                                 yt_sentences: List[Dict]) -> List[Dict]:
        """Replace or correct Whisper text using YouTube subs.

        Both are English transcriptions of the same audio in the same order.
        Whisper has precise timestamps but garbled text (name variations,
        spelling). YouTube has clean text but imprecise display timelines.

        Two modes (controlled by cfg.yt_replace_mode):

        "full"  — Total replacement. Flatten both into word streams,
                  align by time, replace Whisper text entirely with YouTube.
                  Whisper timestamps kept. Best when YouTube text is much
                  cleaner overall.

        "diff"  — Word-level diff. Align both word streams using
                  SequenceMatcher, only swap words that DIFFER. Keeps
                  Whisper's punctuation and structure intact, fixes only
                  the misheard parts (names, nouns, spelling). Best when
                  Whisper's structure is good but specific words are wrong.
        """
        if not whisper_segments or not yt_sentences:
            return whisper_segments

        mode = getattr(self.cfg, 'yt_replace_mode', 'diff')

        # ── Flatten both into word streams ──
        # Whisper: word list per segment (we need to reconstruct back)
        w_seg_words: List[List[str]] = []
        for seg in whisper_segments:
            w_seg_words.append(seg.get("text", "").split())
        w_flat = []
        for words in w_seg_words:
            w_flat.extend(words)

        # YouTube: single word stream (order matches audio)
        yt_flat: List[str] = []
        for yt_seg in yt_sentences:
            yt_flat.extend(yt_seg.get("text", "").split())

        if not yt_flat or not w_flat:
            return whisper_segments

        w_total = len(w_flat)
        yt_total = len(yt_flat)

        if mode == "full":
            # ═══ FULL REPLACE: swap all Whisper words with YouTube words ═══
            # Proportional distribution: each Whisper segment gets its share
            # of YouTube words based on its word count relative to total.
            result = []
            yt_cursor = 0

            for i, wseg in enumerate(whisper_segments):
                seg = dict(wseg)
                seg_wc = len(w_seg_words[i])

                # Proportional share of YouTube words
                proportion = seg_wc / max(w_total, 1)
                n_words = round(proportion * yt_total)
                n_words = max(1, min(n_words, yt_total - yt_cursor))

                # Last segment gets everything remaining
                if i == len(whisper_segments) - 1:
                    n_words = max(0, yt_total - yt_cursor)

                if n_words > 0 and yt_cursor < yt_total:
                    yt_slice = yt_flat[yt_cursor:yt_cursor + n_words]
                    seg["_whisper_original"] = seg.get("text", "")
                    seg["text"] = " ".join(yt_slice)
                    yt_cursor += n_words

                result.append(seg)

            print(f"[YT-replace-full] All {len(result)} segments replaced | "
                  f"Whisper {w_total} words -> YouTube {yt_total} words | "
                  f"consumed {yt_cursor}/{yt_total}", flush=True)
            return result

        else:
            # ═══ DIFF REPLACE: only swap words that differ ═══
            # Use SequenceMatcher to align the two word streams. Where they
            # match, keep Whisper's word (with its punctuation). Where they
            # differ, use YouTube's word (correct spelling/names).
            from difflib import SequenceMatcher

            # Normalize for comparison (lowercase, strip punctuation)
            import string
            _punct = set(string.punctuation)

            def _normalize(word: str) -> str:
                return word.lower().strip("".join(_punct))

            w_norm = [_normalize(w) for w in w_flat]
            yt_norm = [_normalize(w) for w in yt_flat]

            # Align the two sequences
            sm = SequenceMatcher(None, w_norm, yt_norm, autojunk=False)
            opcodes = sm.get_opcodes()

            # Build the corrected flat word list
            corrected_flat: List[str] = []
            diff_count = 0

            for tag, i1, i2, j1, j2 in opcodes:
                if tag == "equal":
                    # Words match — keep Whisper's version (preserves punctuation)
                    corrected_flat.extend(w_flat[i1:i2])
                elif tag == "replace":
                    # Words differ — use YouTube's version (correct names/spelling)
                    corrected_flat.extend(yt_flat[j1:j2])
                    diff_count += (j2 - j1)
                elif tag == "insert":
                    # YouTube has extra words — insert them (Whisper dropped words)
                    corrected_flat.extend(yt_flat[j1:j2])
                    diff_count += (j2 - j1)
                elif tag == "delete":
                    # Whisper has extra words — skip them (hallucinations)
                    diff_count += (i2 - i1)

            # ── Reconstruct back into Whisper's segment structure ──
            # Each segment gets the same NUMBER of words it originally had
            # (from the corrected stream), preserving segment boundaries.
            result = []
            cursor = 0
            corrected_total = len(corrected_flat)

            for i, wseg in enumerate(whisper_segments):
                seg = dict(wseg)
                orig_wc = len(w_seg_words[i])

                # Proportional share from corrected stream
                if w_total > 0:
                    proportion = orig_wc / w_total
                    n_words = round(proportion * corrected_total)
                else:
                    n_words = orig_wc

                n_words = max(1, min(n_words, corrected_total - cursor))
                if i == len(whisper_segments) - 1:
                    n_words = max(0, corrected_total - cursor)

                if n_words > 0 and cursor < corrected_total:
                    new_text = " ".join(corrected_flat[cursor:cursor + n_words])
                    if new_text != seg.get("text", ""):
                        seg["_whisper_original"] = seg.get("text", "")
                    seg["text"] = new_text
                    cursor += n_words

                result.append(seg)

            match_pct = ((w_total - diff_count) / max(w_total, 1)) * 100
            print(f"[YT-replace-diff] {diff_count} words changed out of {w_total} "
                  f"({match_pct:.0f}% match) | "
                  f"Whisper {w_total} words, YouTube {yt_total} words, "
                  f"corrected {corrected_total} words", flush=True)
            return result

    def _whisperx_align(self, wav_path: Path, segments: List[Dict]) -> List[Dict]:
        """Refine word-level timestamps using WhisperX forced alignment.
        Falls back to original segments if whisperx is not installed."""
        try:
            import whisperx
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            lang = self.cfg.source_language
            if not lang or lang == "auto":
                lang = "en"

            self._report("transcribe", 0.97, "Running WhisperX forced alignment...")
            align_model, metadata = whisperx.load_align_model(
                language_code=lang, device=device
            )
            result = whisperx.align(
                segments, align_model, metadata, str(wav_path), device,
                return_char_alignments=False,
            )
            refined = result.get("segments", segments)
            # Normalise to our segment dict format (ensure start/end/text present)
            # Preserve all original fields (e.g. speaker_id for multi-speaker mode)
            out = []
            for seg in refined:
                entry = {**seg,  # Preserve original fields first
                    "start": float(seg.get("start", 0)),
                    "end":   float(seg.get("end", 0)),
                    "text":  seg.get("text", "").strip(),
                }
                words = seg.get("words")
                if words:
                    entry["words"] = [
                        {"word": w.get("word", "").strip(),
                         "start": float(w.get("start", entry["start"])),
                         "end":   float(w.get("end", entry["end"]))}
                        for w in words
                    ]
                out.append(entry)
            self._report("transcribe", 0.99, f"WhisperX alignment complete ({len(out)} segments)")
            return out
        except ImportError:
            print("[Pipeline] whisperx not installed — skipping forced alignment", flush=True)
            return segments
        except Exception as e:
            print(f"[Pipeline] WhisperX alignment failed ({e}) — using original timestamps", flush=True)
            return segments

    def _transcribe_groq(self, wav_path: Path, api_key: str) -> List[Dict]:
        """Transcribe using Groq Whisper API — ~25s per hour of audio.
        Handles files >25MB by chunking into smaller pieces."""
        import requests as _requests

        self._report("transcribe", 0.1, "Using Groq Whisper API (cloud, ultra-fast)...")

        # Groq has a 25MB file limit — check if we need to chunk
        file_size_mb = wav_path.stat().st_size / (1024 * 1024)

        if file_size_mb <= 24:
            # Single file upload
            return self._groq_whisper_single(wav_path, api_key)

        # Large file: split into chunks and transcribe each
        self._report("transcribe", 0.1,
                     f"Audio is {file_size_mb:.0f}MB — splitting into chunks for Groq API...")
        return self._groq_whisper_chunked(wav_path, api_key)

    def _groq_whisper_single(self, audio_path: Path, api_key: str) -> List[Dict]:
        """Send a single file to Groq Whisper API."""
        import requests as _requests

        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}

        data = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        if self.cfg.source_language and self.cfg.source_language != "auto":
            data["language"] = self.cfg.source_language

        with open(audio_path, "rb") as f:
            resp = _requests.post(
                url, headers=headers,
                data=data,
                files={"file": (audio_path.name, f, "audio/wav")},
                timeout=300,
            )
        resp.raise_for_status()
        result = resp.json()

        segments = []
        for seg in result.get("segments", []):
            entry = {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
            }
            segments.append(entry)

        self._report("transcribe", 0.95,
                     f"Groq Whisper: {len(segments)} segments transcribed")
        return segments

    def _groq_whisper_chunked(self, wav_path: Path, api_key: str) -> List[Dict]:
        """Split large audio into ~10min chunks, transcribe each via Groq API in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Split into 10-minute chunks (mono 16kHz = ~19MB per 10min)
        chunk_duration = 600  # 10 minutes in seconds
        total_duration = self._get_duration(wav_path)
        num_chunks = math.ceil(total_duration / chunk_duration)

        self._report("transcribe", 0.15,
                     f"Splitting into {num_chunks} chunks ({chunk_duration//60}min each)...")

        # Create chunks using ffmpeg
        chunk_paths = []
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = self.cfg.work_dir / f"whisper_chunk_{i:03d}.wav"
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(wav_path),
                 "-ss", str(start_time), "-t", str(chunk_duration),
                 "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                 str(chunk_path)],
                check=True, capture_output=True,
            )
            chunk_paths.append((i, start_time, chunk_path))

        # Transcribe chunks in parallel (up to 4 concurrent API calls)
        all_segments = []
        completed = 0

        def transcribe_chunk(args):
            idx, offset, path = args
            segs = self._groq_whisper_single(path, api_key)
            # Adjust timestamps by chunk offset
            for seg in segs:
                seg["start"] += offset
                seg["end"] += offset
            return idx, segs

        failed_chunks = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(transcribe_chunk, cp): cp[0] for cp in chunk_paths}
            for fut in as_completed(futures):
                try:
                    idx, segs = fut.result()
                    all_segments.append((idx, segs))
                except Exception as e:
                    failed_chunks += 1
                    self._report("transcribe", 0.15,
                                 f"Groq chunk failed ({e}) — {failed_chunks}/{num_chunks} bad")
                completed += 1
                self._report("transcribe", 0.15 + 0.8 * (completed / num_chunks),
                             f"Transcribed chunk {completed}/{num_chunks} via Groq API...")

        # If ANY chunk failed, raise so caller falls back to local Whisper
        # (otherwise we'd silently produce a video with missing transcription).
        if failed_chunks > 0:
            for _, _, path in chunk_paths:
                path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{failed_chunks}/{num_chunks} Groq Whisper chunks failed — "
                f"forcing fallback to local Whisper large-v3"
            )

        # Sort by chunk index and flatten
        all_segments.sort(key=lambda x: x[0])
        segments = []
        for _, segs in all_segments:
            segments.extend(segs)

        # Clean up chunk files
        for _, _, path in chunk_paths:
            path.unlink(missing_ok=True)

        # Deduplicate chunk boundary overlaps
        segments = self._dedup_segments(segments)

        return segments

    def _transcribe_local(self, wav_path: Path) -> List[Dict]:
        """Transcribe speech from audio using local faster-whisper (GPU/CPU).

        Runs Whisper in a **child process** so that C-level crashes
        (SIGABRT, CUDA OOM that bypasses Python try/except) only kill
        the child — the FastAPI server stays alive.

        Fallback chain: GPU child → CPU child → raise.
        """
        import multiprocessing as mp
        import json as _json
        import tempfile as _tmpmod

        local_model = self.cfg.asr_model
        if local_model in ("groq-whisper", "groq", "parakeet"):
            local_model = "medium"

        source_lang = self.cfg.source_language if self.cfg.source_language != "auto" else None

        def _run_in_child(device: str, compute: str) -> List[Dict]:
            """Spawn a child process to run Whisper. Returns segments or raises."""
            fd, result_path = _tmpmod.mkstemp(suffix=".json", prefix="whisper_result_")
            import os as _os
            _os.close(fd)
            try:
                self._report("transcribe", 0.1,
                             f"Loading Whisper ({local_model}) on {device.upper()} (isolated process)...")
                p = mp.Process(
                    target=_whisper_child_worker,
                    args=(str(wav_path), local_model, device, compute, source_lang, result_path),
                    daemon=True,
                )
                p.start()
                p.join(timeout=1800)  # 30 min max
                if p.is_alive():
                    p.kill()
                    p.join(5)
                    raise RuntimeError("Whisper transcription timed out (30 min limit)")
                if p.exitcode != 0:
                    err_msg = f"Whisper child process died with exit code {p.exitcode}"
                    try:
                        with open(result_path, "r", encoding="utf-8") as f:
                            data = _json.load(f)
                            if data.get("error"):
                                err_msg = data["error"]
                    except Exception:
                        pass
                    raise RuntimeError(err_msg)
                with open(result_path, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                segments = data.get("segments", [])
                self._report("transcribe", 0.95, f"Transcribed {len(segments)} segments")
                return segments
            finally:
                try:
                    Path(result_path).unlink(missing_ok=True)
                except Exception:
                    pass

        # Auto-detect GPU
        device, compute = "cpu", "int8"
        try:
            import torch as _torch_mod
            if _torch_mod.cuda.is_available():
                device, compute = "cuda", "float16"
                print(f"[Whisper] GPU detected: {_torch_mod.cuda.get_device_name(0)}", flush=True)
            else:
                print("[Whisper] torch.cuda.is_available() = False -> using CPU", flush=True)
        except ImportError:
            print("[Whisper] torch not installed -> using CPU", flush=True)
        except Exception as _gpu_err:
            print(f"[Whisper] GPU detection failed: {_gpu_err} -> using CPU", flush=True)

        try:
            return _run_in_child(device, compute)
        except RuntimeError as e:
            if device == "cuda":
                self._report("transcribe", 0.15,
                             f"GPU transcription failed ({str(e)[:80]}) — retrying on CPU...")
                return _run_in_child("cpu", "int8")
            raise

    # ── Step 4: Translate full narrative ─────────────────────────────────
    def _translate_full_narrative(self, text_segments: List[Dict], speech_duration: float = 0) -> tuple:
        """Join all speech into one narrative, translate as a whole."""
        # Combine all transcribed text into one continuous story
        full_text = " ".join(s.get("text", "").strip() for s in text_segments if s.get("text", "").strip())
        if not full_text.strip():
            self._report("translate", 1.0, "No text to translate (empty segments)")
            return "", ""
        self._report("translate", 0.2, f"Translating {len(full_text)} characters...")

        gemini_key = get_gemini_key()
        groq_key = get_groq_key()
        engine = self.cfg.translation_engine
        target_lang = self.cfg.target_language

        # Determine effective engine name for cache key (use actual engine chosen in auto mode)
        _eff_engine = engine
        if engine == "auto":
            if gemini_key:   _eff_engine = "gemini"
            elif groq_key:   _eff_engine = "groq"
            elif self._ollama_available(): _eff_engine = "ollama"
            else:            _eff_engine = "google"

        # Check translation cache
        cached_translation = _cache.get_translation(full_text, _eff_engine, target_lang)
        if cached_translation is not None:
            self._report("translate", 0.95, "Translation cache hit — skipping re-translation")
            return full_text, cached_translation

        # If a specific engine is selected, use it directly
        if engine == "hinglish" and self._ollama_available():
            self._report("translate", 0.25, "Using Hinglish AI (custom Ollama model)...")
            translated_text = self._translate_with_ollama(full_text, speech_duration, force_model="hinglish-translator")
        elif engine == "gemma4" and gemini_key:
            self._report("translate", 0.25, "Using Gemma 4 (31B) for translation...")
            translated_text = self._translate_with_gemma4(full_text, gemini_key, speech_duration)
        elif engine == "gemini" and gemini_key:
            self._report("translate", 0.25, "Using Gemini for translation...")
            translated_text = self._translate_with_gemini(full_text, gemini_key, speech_duration)
        elif engine == "groq" and groq_key:
            self._report("translate", 0.25, "Using Groq (Llama 3.3 70B) for translation...")
            translated_text = self._translate_with_groq(full_text, groq_key, speech_duration)
        elif engine == "ollama" and self._ollama_available():
            self._report("translate", 0.25, "Using Ollama (local LLM) for translation...")
            translated_text = self._translate_with_ollama(full_text, speech_duration)
        elif engine == "google":
            self._report("translate", 0.25, "Using Google Translate...")
            translated_text = self._translate_with_google(full_text)
        # Auto mode: try best available
        elif gemini_key:
            translated_text = self._translate_with_gemini(full_text, gemini_key, speech_duration)
        elif groq_key:
            self._report("translate", 0.25, "Using Groq (Llama 3.3 70B) for translation...")
            translated_text = self._translate_with_groq(full_text, groq_key, speech_duration)
        elif self._ollama_available():
            self._report("translate", 0.25, "Using Ollama (local LLM) for translation...")
            translated_text = self._translate_with_ollama(full_text, speech_duration)
        else:
            self._report("translate", 0.25, "No GEMINI/GROQ/Ollama found, using Google Translate...")
            translated_text = self._translate_with_google(full_text)

        _cache.put_translation(full_text, _eff_engine, target_lang, translated_text)
        return full_text, translated_text

    def _translate_with_gemini(self, full_text: str, api_key: str, speech_duration: float = 0) -> str:
        """Translate using Gemini LLM for natural, fluent output."""
        from google import genai

        client = genai.Client(api_key=api_key)

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the source language"

        # Calculate word count guidance for duration matching
        word_count = len(full_text.split())
        duration_hint = ""
        if speech_duration > 0:
            wpm = LANGUAGE_WPM.get(self.cfg.target_language, 135)
            target_words = int(speech_duration / 60 * wpm)
            duration_hint = (
                f"IMPORTANT TIMING CONSTRAINT: The original narration is {int(speech_duration)} seconds long "
                f"({word_count} words). Your {target_name} translation will be spoken by TTS and "
                f"MUST fit within this duration. Aim for approximately {target_words} {target_name} words. "
                f"Be concise — use shorter phrases where possible without losing meaning. "
                f"Avoid filler words and unnecessary elaboration. "
            )

        # Gemini free tier: 10 RPM, so split large texts into chunks
        chunks = self._split_text_for_translation(full_text, max_chars=8000)
        translated_parts = []
        chunk_duration = speech_duration / len(chunks) if speech_duration > 0 else 0

        for i, chunk in enumerate(chunks):
            chunk_words = len(chunk.split())
            chunk_target = int(chunk_duration / 60 * wpm) if chunk_duration > 0 else 0
            chunk_hint = ""
            if chunk_target > 0:
                chunk_hint = (
                    f"This chunk has {chunk_words} English words and must fit in ~{int(chunk_duration)} seconds. "
                    f"Aim for ~{chunk_target} {target_name} words. "
                )

            prompt = (
                self._get_translation_prompt("system") + "\n\n"
                f"This is a voiceover script for a dubbed video. "
                f"{duration_hint}{chunk_hint}"
                f"Do NOT translate literally word-by-word. Adapt idioms and phrasing naturally. "
                f"Output ONLY the translation, nothing else.\n\n"
                f"{chunk}"
            )

            retries = 3
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=prompt,
                    )
                    translated_parts.append((response.text or "").strip())
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.2 + 0.6 * (i / len(chunks)),
                                     f"Rate limited, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        # Only translate the failed chunk via fallback, keep already-translated parts
                        groq_key = get_groq_key()
                        if groq_key:
                            self._report("translate", 0.2, f"Gemini failed on chunk {i+1}: {e}, using Groq for this chunk...")
                            fallback = self._translate_with_groq(chunk, groq_key, chunk_duration)
                        else:
                            self._report("translate", 0.2, f"Gemini failed on chunk {i+1}: {e}, using Google for this chunk...")
                            fallback = self._translate_with_google(chunk)
                        translated_parts.append(fallback)

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)} (Gemini)")

        return " ".join(translated_parts)

    def _translate_with_gemma4(self, full_text: str, api_key: str, speech_duration: float = 0) -> str:
        """Translate using Gemma 4 31B — Google's open model, excellent Hindi quality.

        Uses the Gemini API to access gemma-4-31b-it.
        Better at natural Hindi than Gemini Pro for dubbing-style translation.
        Free tier: 30 RPM per key — multiple keys enable faster parallel translation.
        """
        from google import genai

        num_keys = _gemini_keys.count()
        client = genai.Client(api_key=api_key)

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the source language"

        word_count = len(full_text.split())
        duration_hint = ""
        if speech_duration > 0:
            wpm = LANGUAGE_WPM.get(self.cfg.target_language, 135)
            target_words = int(speech_duration / 60 * wpm)
            duration_hint = (
                f"TIMING: Original is {int(speech_duration)}s ({word_count} words). "
                f"Aim for ~{target_words} {target_name} words to fit the same duration. "
                f"Be concise — shorter phrases where possible without losing meaning. "
            )

        # Gemma 4 handles larger context well — use 6000 char chunks
        chunks = self._split_text_for_translation(full_text, max_chars=6000)
        chunk_duration = speech_duration / len(chunks) if speech_duration > 0 else 0
        wpm = LANGUAGE_WPM.get(self.cfg.target_language, 135)
        total_chunks = len(chunks)
        # Scale parallelism with number of keys: each key = 30 RPM
        max_parallel = min(2 * max(num_keys, 1), total_chunks)  # 2 per key — avoids rate limits

        # Results array — maintain order
        translated_parts = [""] * total_chunks
        completed = [0]

        def translate_chunk(i, chunk):
            # Each worker gets a rotated key for load distribution across accounts
            worker_key = get_gemini_key() or api_key
            worker_client = genai.Client(api_key=worker_key)

            chunk_words = len(chunk.split())
            chunk_target = int(chunk_duration / 60 * wpm) if chunk_duration > 0 else 0
            chunk_hint = ""
            if chunk_target > 0:
                chunk_hint = (
                    f"This chunk: {chunk_words} words, ~{int(chunk_duration)}s. "
                    f"Aim for ~{chunk_target} {target_name} words. "
                )

            prompt = (
                self._get_translation_prompt("system") + "\n\n"
                f"This is a voiceover script for a dubbed video. "
                f"{duration_hint}{chunk_hint}"
                f"Translate naturally — NOT word-by-word. Adapt idioms for {target_name} audience. "
                f"Use spoken/conversational style, not literary. "
                f"Output ONLY the translation, nothing else.\n\n"
                f"{chunk}"
            )

            retries = 3
            for attempt in range(retries):
                try:
                    response = worker_client.models.generate_content(
                        model="gemma-4-31b-it",
                        contents=prompt,
                    )
                    return (response.text or "").strip()
                except Exception as e:
                    _gemini_keys.report_rate_limit(worker_key)
                    if attempt < retries - 1:
                        # Get a different key for retry
                        worker_key = get_gemini_key() or api_key
                        worker_client = genai.Client(api_key=worker_key)
                        wait = 2 * (attempt + 1)
                        time.sleep(wait)
                    else:
                        # Fallback: try Groq, then Google
                        groq_key = get_groq_key()
                        if groq_key:
                            self._report("translate", 0.2, f"Gemma 4 failed chunk {i+1}: {e}, using Groq...")
                            return self._translate_with_groq(chunk, groq_key, chunk_duration)
                        else:
                            self._report("translate", 0.2, f"Gemma 4 failed chunk {i+1}: {e}, using Google...")
                            return self._translate_with_google(chunk)
            return ""

        if total_chunks == 1:
            # Single chunk — no need for thread pool
            translated_parts[0] = translate_chunk(0, chunks[0])
            self._report("translate", 0.9, "Translated 1/1 (Gemma 4)")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            keys_label = f", {num_keys} API keys" if num_keys > 1 else ""
            self._report("translate", 0.05,
                         f"PARALLEL: Gemma 4 × {max_parallel} workers ({total_chunks} chunks{keys_label})...")

            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {
                    executor.submit(translate_chunk, i, chunk): i
                    for i, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    translated_parts[idx] = future.result() or ""
                    completed[0] += 1
                    self._report("translate", 0.2 + 0.8 * (completed[0] / total_chunks),
                                 f"Translated chunk {completed[0]}/{total_chunks} (Gemma 4 parallel)")

        return " ".join(translated_parts)


    def _translate_with_groq(self, full_text: str, api_key: str, speech_duration: float = 0) -> str:
        """Translate using Groq (Llama 3.3 70B) for natural, fluent output."""
        import requests as _requests

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the source language"

        word_count = len(full_text.split())
        duration_hint = ""
        if speech_duration > 0:
            wpm = LANGUAGE_WPM.get(self.cfg.target_language, 135)
            target_words = int(speech_duration / 60 * wpm)
            duration_hint = (
                f"IMPORTANT TIMING CONSTRAINT: The original narration is {int(speech_duration)} seconds long "
                f"({word_count} words). Aim for approximately {target_words} {target_name} words. "
                f"Be concise — use shorter phrases where possible without losing meaning. "
            )

        chunks = self._split_text_for_translation(full_text, max_chars=8000)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            prompt = (
                self._get_translation_prompt("system") + "\n\n"
                f"This is a voiceover script for a dubbed video. "
                f"{duration_hint}"
                f"Do NOT translate literally word-by-word. Adapt idioms and phrasing naturally. "
                f"Output ONLY the translation, nothing else.\n\n"
                f"{chunk}"
            )

            retries = 3
            for attempt in range(retries):
                try:
                    resp = _requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}]},
                        timeout=60,
                    )
                    data = resp.json()
                    if "choices" in data:
                        translated_parts.append(data["choices"][0]["message"]["content"].strip())
                        break
                    else:
                        raise RuntimeError(data.get("error", {}).get("message", "Groq API error"))
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.2 + 0.6 * (i / len(chunks)),
                                     f"Groq rate limited, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        self._report("translate", 0.2, f"Groq failed on chunk {i+1}: {e}, falling back to Google for this chunk...")
                        fallback = self._translate_with_google(chunk)
                        translated_parts.append(fallback)
                        break

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)} (Groq)")

        return " ".join(translated_parts)

    def _ollama_available(self) -> bool:
        """Check if Ollama is running locally."""
        import requests as _requests
        try:
            resp = _requests.get("http://localhost:11434/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _translate_with_ollama(self, full_text: str, speech_duration: float = 0, force_model: str = "") -> str:
        """Translate using local Ollama LLM."""
        import requests as _requests

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)

        # Pick the best available model
        try:
            resp = _requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            models = []

        # If a specific model is requested, use it
        if force_model:
            model = force_model if any(force_model in m for m in models) else None
            if not model:
                self._report("translate", 0.25, f"Model '{force_model}' not found in Ollama, falling back...")
                # Try auto-pick instead
                force_model = ""

        if not force_model:
            # Prefer hindi-dubbing (optimized qwen2.5:14b), then raw qwen2.5, then others
            preferred = ["hindi-dubbing", "qwen2.5:14b", "qwen2.5:32b", "llama3.1:8b", "llama3:8b", "gemma2:9b", "mistral:7b"]
            model = None
            for pref in preferred:
                for m in models:
                    if pref.split(":")[0] in m:
                        model = m
                        break
                if model:
                    break
            if not model and models:
                model = models[0]
        if not model:
            self._report("translate", 0.25, "No Ollama models found, falling back to Google Translate...")
            return self._translate_with_google(full_text)

        self._report("translate", 0.25, f"Using Ollama model: {model}")

        word_count = len(full_text.split())
        duration_hint = ""
        if speech_duration > 0:
            wpm = LANGUAGE_WPM.get(self.cfg.target_language, 135)
            target_words = int(speech_duration / 60 * wpm)
            duration_hint = (
                f"IMPORTANT TIMING CONSTRAINT: The original narration is {int(speech_duration)} seconds long "
                f"({word_count} words). Aim for approximately {target_words} {target_name} words. "
                f"Be concise — use shorter phrases where possible without losing meaning. "
            )

        chunks = self._split_text_for_translation(full_text, max_chars=4000)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            prompt = (
                self._get_translation_prompt("system") + "\n\n"
                f"This is a voiceover script for a dubbed video. "
                f"{duration_hint}"
                f"Do NOT translate literally word-by-word. Adapt idioms and phrasing naturally. "
                f"Output ONLY the translation, nothing else.\n\n"
                f"{chunk}"
            )

            try:
                resp = _requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=120,
                )
                data = resp.json()
                translated_parts.append(data.get("response", "").strip())
            except Exception as e:
                self._report("translate", 0.2, f"Ollama failed: {e}, falling back to Google Translate...")
                return self._translate_with_google(full_text)

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)} (Ollama: {model})")

        return " ".join(translated_parts)

    def _translate_with_google(self, full_text: str) -> str:
        """Fallback: translate using free Google Translate via deep-translator."""
        from deep_translator import GoogleTranslator

        def _is_garbage(text: str) -> bool:
            if not text:
                return True
            low = text.lower()
            return ("error 500" in low or "server error" in low
                    or "that's an error" in low or "<html" in low
                    or "<!doctype" in low)

        src = self.cfg.source_language if self.cfg.source_language != "auto" else "auto"
        translator = GoogleTranslator(source=src, target=self.cfg.target_language)
        chunks = self._split_text_for_translation(full_text, max_chars=4500)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            retries = 3
            for attempt in range(retries):
                try:
                    result = translator.translate(chunk)
                    if _is_garbage(result):
                        if attempt < retries - 1:
                            time.sleep(1.5 * (attempt + 1))
                            continue
                        print(f"[Translate] Chunk {i} got garbage response after {retries} retries — using original", flush=True)
                        translated_parts.append(chunk)
                    else:
                        translated_parts.append(result)
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        print(f"[Translate] Chunk {i} failed after {retries} retries: {e} — using original text", flush=True)
                        translated_parts.append(chunk)

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)}")

        return " ".join(translated_parts)

    @staticmethod
    def _split_text_for_translation(text: str, max_chars: int = 4500) -> List[str]:
        """Split text into chunks at sentence boundaries for translation API limits."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        current = ""
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?।])\s+', text)

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = (current + " " + sentence).strip()

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

    @staticmethod
    def _compute_target_word_count(duration_seconds: float, target_language: str) -> int:
        """Compute target word count for translation based on language speaking rate."""
        wpm = LANGUAGE_WPM.get(target_language, 135)
        return max(1, round((duration_seconds / 60.0) * wpm))

    def _get_translation_prompt(self, mode: str = "system") -> str:
        """Build the translation system/user prompt based on target language.

        For Hindi: uses a detailed natural-spoken-Hindi prompt.
        For other languages: uses a generic colloquial prompt.
        mode: 'system' for system message, 'user_prefix' for user message intro.
        """
        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the detected language"
        is_hindi = self.cfg.target_language in ("hi", "hi-IN")

        if mode == "system":
            if is_hindi:
                return (
                    "You are a professional Hindi dubbing writer for YouTube story recap channels.\n"
                    "Your job: rewrite English lines as ORIGINAL Hindi narration — as if the script was "
                    "WRITTEN IN HINDI from the start. It must NEVER feel translated.\n\n"

                    "═══ CORE IDENTITY ═══\n"
                    "You are a Hindi narrator telling a gripping story to an audience. Think of channels "
                    "like 'Anime Jeetu', 'Recap Jeetu', 'Movie Jeetu' — dramatic, engaging, emotional Hindi "
                    "narration that hooks viewers.\n\n"

                    "═══ BANNED WORDS (NEVER USE) ═══\n"
                    "अतः, किन्तु, तथापि, उक्त, यद्यपि, अपितु, एवं, तदनुसार, अभिप्राय, कृपया (in narration), "
                    "आवश्यकता (use ज़रूरत), वास्तव में (use सच में/असल में), विक्षिप्त, घटित, "
                    "उत्साहित (use excited), सामान्य (use normal), तुरंत (use फौरन/झट से), "
                    "अचानक से (just अचानक). "
                    "If you catch yourself writing ANY of these, rewrite the line immediately.\n\n"

                    "═══ NATURAL HINDI RULES ═══\n"
                    "1. REWRITE, don't translate. Think: 'How would I SAY this in Hindi to a friend?'\n"
                    "2. Use तू/तुम for conversations, not आप (unless showing respect to elders intentionally)\n"
                    "3. Use contractions: नहीं→नी, कुछ नहीं→कुछ नी, मत करो→मत कर\n"
                    "4. Dramatic moments need PUNCH: 'और बस... यहीं से खेल शुरू हुआ' not 'और इसी स्थान से...'\n"
                    "5. Dialogue should sound REAL: 'अबे रुक! कहाँ जा रहा है?' not 'कृपया रुकिए'\n"
                    "6. Internal thoughts: 'मैंने सोचा, बस अब इसकी खैर नहीं' not 'मैंने विचार किया'\n"
                    "7. Suspense/tension: 'और तभी... जो हुआ वो किसी ने सोचा भी नहीं था' \n"
                    "8. Keep English words Indians actually use: truck, plan, phone, message, delete, "
                    "hospital, police, driver, accident, cargo, store, impact, vest, glass, bulletproof\n"
                    "9. Hindi words for impact: धड़ाम (crash sound), धम्म (thud), झन्नाटेदार (thrilling), "
                    "सनसनी (sensation), कंपकंपी (shiver), तबाही (destruction)\n"
                    "10. Emotional connectors: और भाई सुनो, अब यहाँ twist आता है, पर wait करो, "
                    "तो basically हुआ ये कि, और जो हुआ अगला वो तो...\n\n"

                    "═══ TRANSLATION FIXES (LEARN THESE) ═══\n"
                    "BAD: 'उसने ठंडी सांस ली' → GOOD: 'उसने फुंकार मारी'\n"
                    "BAD: 'कृपया हमें बचाओ' → GOOD: 'बचाओ! किसी को बुलाओ!'\n"
                    "BAD: 'बुजुर्ग आदमी उत्साहित होकर आया' → GOOD: 'बुड्ढा excited होके आया'\n"
                    "BAD: 'उसने अपना पेट पकड़ लिया और कराह उठा' → GOOD: 'उसने पेट पकड़ा और चिल्लाया'\n"
                    "BAD: 'मैंने उसे पहचान लिया' → GOOD: 'मैंने उसे फौरन पहचान लिया — ये वही था'\n"
                    "BAD: 'एक और शिकार' → GOOD: 'एक और शिकार अपने आप जाल में आ रहा था'\n"
                    "BAD: 'सबको तुरंत डर लग गया' → GOOD: 'सबकी हालत खराब हो गई'\n"
                    "BAD: 'वह चोटों से ढका हुआ था' → GOOD: 'पूरा लहूलुहान था'\n\n"

                    "═══ COMPLETENESS ═══\n"
                    "Translate EVERY word and idea. NOTHING may be skipped or summarized.\n"
                    "Each line must be COMPLETE — no truncation.\n"
                    "Output ONLY numbered translations, one per line, matching input numbering.\n"
                    "Do NOT add notes, comments, Chinese/Japanese characters, or metadata.\n"
                    "Do NOT output anything in any script other than Devanagari and English."
                )
            else:
                return (
                    f"You are a world-class dubbing translator. Make {target_name} sound like it was ORIGINALLY "
                    f"written by a native speaker — a real person talking, not a textbook.\n"
                    f"Sound like a YouTuber, podcast host, or friend explaining something. "
                    f"Use the EXACT way native speakers ACTUALLY talk in everyday life. "
                    f"Keep English words people naturally mix in: 'actually', 'basically', 'problem', 'phone', 'video', 'so', 'but'. "
                    f"NEVER use formal/literary/textbook language. "
                    f"Match the speaker's energy and emotion exactly. "
                    f"Keep proper nouns, brands, and technical terms unchanged.\n\n"
                    f"COMPLETENESS: Translate EVERY word and idea. NOTHING may be skipped. "
                    f"Each translation must be COMPLETE. Video adapts to audio — don't worry about length. "
                    f"Output ONLY numbered translations. No notes or metadata."
                )

        elif mode == "user_prefix":
            if is_hindi:
                return (
                    f"Rewrite each numbered line as ORIGINAL Hindi narration/dialogue. "
                    f"This is a story recap — make it dramatic, gripping, and natural. "
                    f"Every line must sound like it was WRITTEN in Hindi, not translated. "
                    f"Use natural spoken Hindi. Translate COMPLETELY — skip nothing.\n\n"
                )
            else:
                return (
                    f"Translate each line from {source_name} to {target_name}. "
                    f"Natural, conversational, the way real people actually talk — like a friend explaining something. "
                    f"Translate COMPLETELY — every idea, every nuance.\n\n"
                )

        return ""

    # ── Step 4b: Segment-level translation ────────────────────────────────
    def _get_turbo_engines(self):
        """Get available engines for turbo parallel translation."""
        engines = []
        groq_key = get_groq_key()
        sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "").strip()
        if groq_key:
            engines.append(("Groq", groq_key))
        if sambanova_key:
            engines.append(("SambaNova", sambanova_key))
        return engines

    # ── English Simplification (pre-translation) ──────────────────────
    def _simplify_english_segments(self, segments):
        """Rewrite complex English into simple, spoken-style English before translation.

        Complex English → bad Hindi. Simple English → great Hindi.
        Uses Gemma 4 (5 keys, parallel) for rewrites.

        Example:
        IN:  "The unprecedented implementation of sophisticated algorithmic trading
              strategies has fundamentally transformed the landscape of institutional
              investment management."
        OUT: "New computer trading methods have changed how big companies invest money."

        Rules:
        - Keep ALL meaning — never drop facts
        - Short sentences (max 15 words)
        - Simple vocabulary (avoid jargon)
        - Active voice, spoken style
        - Preserve names, numbers, technical terms
        - Output ONLY simplified text, nothing else
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        gemini_key = get_gemini_key()
        if not gemini_key:
            print("[Simplify] No GEMINI_API_KEY — skipping simplification", flush=True)
            return segments

        from google import genai

        total = len(segments)
        num_keys = _gemini_keys.count()
        max_parallel = min(2 * max(num_keys, 1), total)  # 2 per key — avoids rate limits
        batch_size = 20  # Process 20 segments per API call for context

        # Build batches
        batches = []
        for i in range(0, total, batch_size):
            batches.append(segments[i:i + batch_size])
        total_batches = len(batches)

        SIMPLIFY_PROMPT = (
            "You are an English text simplifier for video dubbing.\n"
            "Rewrite each numbered line into simple, spoken English.\n\n"
            "CRITICAL RULE: Keep the SAME word count as the original.\n"
            "Each line shows [Xw] = the target word count. Match it closely.\n\n"
            "RULES:\n"
            "- MATCH the original word count (±2 words max)\n"
            "- Keep ALL meaning and facts — never drop information\n"
            "- Use simple common words (no jargon, no academic language)\n"
            "- Use active voice, conversational tone\n"
            "- Preserve all names, numbers, and proper nouns exactly\n"
            "- Do NOT split into multiple sentences — keep as one sentence\n"
            "- Output ONLY the numbered simplified lines, nothing else\n\n"
        )

        completed = [0]

        def simplify_batch(batch_idx, batch):
            worker_key = get_gemini_key() or gemini_key
            worker_client = genai.Client(api_key=worker_key)

            # Include word count target so LLM matches original length
            lines = []
            for i, seg in enumerate(batch):
                text = seg.get('text', '')
                wc = len(text.split())
                lines.append(f"{i+1}. [{wc}w] {text}")
            prompt = SIMPLIFY_PROMPT + "\n".join(lines)

            for attempt in range(3):
                try:
                    # Timeout: 30s per batch to prevent hangs
                    import signal
                    import threading

                    result_holder = [None]
                    error_holder = [None]

                    def _call_api():
                        try:
                            result_holder[0] = worker_client.models.generate_content(
                                model="gemma-4-31b-it", contents=prompt)
                        except Exception as e:
                            error_holder[0] = e

                    t = threading.Thread(target=_call_api)
                    t.start()
                    t.join(timeout=30)  # 30 second timeout
                    if t.is_alive():
                        raise TimeoutError("Gemma 4 API hung (>30s)")

                    if error_holder[0]:
                        raise error_holder[0]

                    response = result_holder[0]
                    result_text = (response.text or "").strip()

                    # Parse numbered results
                    simplified = self._parse_numbered_translations(result_text, len(batch))
                    for i, seg in enumerate(batch):
                        if simplified[i] and simplified[i].strip():
                            seg["text_original_complex"] = seg.get("text", "")
                            seg["text"] = simplified[i].strip()
                    return True
                except Exception as e:
                    _gemini_keys.report_rate_limit(worker_key)
                    if attempt < 2:
                        worker_key = get_gemini_key() or gemini_key
                        worker_client = genai.Client(api_key=worker_key)
                        time.sleep(2)  # fixed 2s backoff (not escalating)
            return False

        keys_label = f", {num_keys} keys" if num_keys > 1 else ""
        self._report("transcribe", 0.93,
                     f"PARALLEL: Simplifying {total} segments "
                     f"(x{max_parallel} workers{keys_label})...")

        with ThreadPoolExecutor(max_workers=min(max_parallel, total_batches)) as executor:
            futures = {
                executor.submit(simplify_batch, idx, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                completed[0] += 1
                self._report("transcribe",
                             0.93 + 0.05 * (completed[0] / total_batches),
                             f"Simplified batch {completed[0]}/{total_batches}")

        # Count how many were actually simplified
        simplified_count = sum(1 for s in segments if s.get("text_original_complex"))
        print(f"[Simplify] {simplified_count}/{total} segments simplified", flush=True)

        return segments

    def _recalculate_timeline(self, segments):
        """Recalculate segment end times based on translated text word count.

        Problem: Translation keeps original English timestamps, but Hindi text
        has more words and takes longer to speak. This causes TTS audio to be
        longer than the video slot.

        Solution: Estimate how long the translated text WILL take to speak
        (using target language WPM), and extend the segment end time if needed.
        This gives assembly realistic timelines from the start.

        Only extends — never shrinks (shorter translation keeps original timing).
        """
        # Ensure sorted by start time before clamping to next segment
        segments.sort(key=lambda s: s.get("start", 0))

        wpm = LANGUAGE_WPM.get(self.cfg.target_language, 135)
        extended = 0

        for i, seg in enumerate(segments):
            text = seg.get("text_translated", "")
            if not text:
                continue

            original_start = seg.get("start", 0)
            original_end = seg.get("end", original_start)
            original_dur = original_end - original_start

            # Estimate spoken duration from word count
            word_count = len(text.split())
            estimated_dur = (word_count / wpm) * 60  # seconds

            # Only extend if translated text needs more time
            if estimated_dur > original_dur * 1.1:  # 10% tolerance
                new_end = original_start + estimated_dur

                # CRITICAL: Don't extend past next segment's start
                if i + 1 < len(segments):
                    next_start = segments[i + 1].get("start", 0)
                    if new_end > next_start - 0.05:
                        new_end = max(original_end, next_start - 0.05)

                seg["_original_end"] = original_end
                seg["end"] = new_end
                extended += 1

        if extended:
            print(f"[Timeline] Extended {extended}/{len(segments)} segment timelines "
                  f"for {self.cfg.target_language} word count "
                  f"(WPM={wpm})", flush=True)

    def _adjust_timeline_after_simplify(self, segments):
        """Tighten segment slots after English simplification reduces word count.

        Problem: Original slot was sized for complex English (10 words, 4.5s).
                 After simplification, text is shorter (6 words) but slot stays 4.5s.
                 This leaves dead air in TTS output.

        Solution: Shrink each segment's end time to match the simplified word count.
                  Only shrinks — never extends (that's done after translation).
                  Respects minimum slot of 0.5s.
                  Reclaimed time goes to the next segment's gap (natural pause).
        """
        segments.sort(key=lambda s: s.get("start", 0))
        en_wpm = LANGUAGE_WPM.get("en", 150)
        shrunk = 0

        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if not text or not seg.get("text_original_complex"):
                continue  # not simplified, skip

            original_start = seg.get("start", 0)
            original_end = seg.get("end", original_start)
            original_dur = original_end - original_start

            # Estimate how long simplified English takes to speak
            word_count = len(text.split())
            estimated_dur = max(0.5, (word_count / en_wpm) * 60)

            # Only shrink if new estimate is noticeably shorter
            if estimated_dur < original_dur * 0.85:  # 15% threshold
                new_end = original_start + estimated_dur

                # Don't shrink past next segment's start
                if i + 1 < len(segments):
                    next_start = segments[i + 1].get("start", 0)
                    new_end = min(new_end, next_start - 0.05)

                # Never go below minimum
                new_end = max(new_end, original_start + 0.5)

                seg["_pre_simplify_end"] = original_end
                seg["end"] = new_end
                shrunk += 1

        if shrunk:
            print(f"[Timeline] Shrunk {shrunk}/{len(segments)} slots after simplification "
                  f"(tighter English, less dead air)", flush=True)

    def _mask_noun_subjects_in_segments(self, segments) -> int:
        """Replace each sentence's main NOUN subject with a placeholder so the
        translator never sees it. Pronoun subjects are LEFT ALONE — they
        translate normally because keeping pronouns in English mid-Hindi sounds
        jarring (e.g., "She आ रही है").

        For each sentence in each segment:
        1. Find the main subject of the root verb (nsubj/nsubjpass)
        2. Skip it if it's a pronoun (PRON)
        3. Otherwise expand to the full subject phrase via the dependency
           subtree (so "The young warrior" not just "warrior")
        4. Replace with `__KEEP_SUBJ_<n>__` placeholder. Google Translate, LLMs,
           and most other translators preserve double-underscore + UPPERCASE
           tokens reliably.
        5. Stash the original subject text on the segment so the restore pass
           can put it back into the translated output.

        Returns the count of segments that had at least one subject masked.
        Silently no-ops (returns 0) if spaCy isn't available.
        """
        nlp = _get_spacy_nlp()
        if nlp is None:
            return 0

        masked_count = 0
        total_subjects = 0
        for seg in segments:
            text = seg.get("text", "") or ""
            if not text.strip():
                continue

            try:
                doc = nlp(text)
            except Exception as e:
                # Don't crash translation if a single segment fails to parse
                print(f"[KEEP-SUBJ] spaCy parse failed on a segment: {e}", flush=True)
                continue

            # Collect (start_char, end_char, original_text) for each NOUN subject
            ranges = []
            for sent in doc.sents:
                root = sent.root
                main_subj = None
                for child in root.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        main_subj = child
                        break
                if main_subj is None:
                    continue
                # Skip pronouns — they translate normally
                if main_subj.pos_ == "PRON":
                    continue
                # Expand to full subject phrase via dependency subtree
                subtree = sorted(main_subj.subtree, key=lambda t: t.i)
                if not subtree:
                    continue
                start_char = subtree[0].idx
                end_char = subtree[-1].idx + len(subtree[-1].text)
                if end_char <= start_char:
                    continue
                ranges.append((start_char, end_char, text[start_char:end_char]))

            if not ranges:
                continue

            # Sort by descending start so replacements don't shift earlier indices
            ranges.sort(key=lambda r: -r[0])

            kept = {}
            new_text = text
            # Number placeholders from leftmost-original (small numbers near the
            # front of the sentence — easier to read in logs).
            ranges_left_first = sorted(ranges, key=lambda r: r[0])
            number_for_range = {id(r): n for n, r in enumerate(ranges_left_first)}

            for r in ranges:  # descending order
                start, end, orig = r
                placeholder = f"__KEEP_SUBJ_{number_for_range[id(r)]}__"
                kept[placeholder] = orig
                new_text = new_text[:start] + placeholder + new_text[end:]

            seg["_text_pre_mask"] = text   # original source for restore
            seg["text"] = new_text          # translator sees the masked version
            seg["_kept_subjects"] = kept    # placeholder -> English subject
            masked_count += 1
            total_subjects += len(kept)

        if masked_count > 0:
            print(f"[KEEP-SUBJ] Masked {total_subjects} noun subjects across "
                  f"{masked_count}/{len(segments)} segments before translation",
                  flush=True)
        return masked_count

    def _restore_noun_subjects_in_segments(self, segments) -> int:
        """Undo _mask_noun_subjects_in_segments. For each segment that was
        masked, replace placeholders in the translated text with the original
        English subject, then restore the original source text.
        """
        restored_count = 0
        for seg in segments:
            kept = seg.pop("_kept_subjects", None)
            original_text = seg.pop("_text_pre_mask", None)

            if not kept:
                continue

            # Restore the original source text (so downstream code that reads
            # seg["text"] for SRT/QC sees the real English, not the placeholder)
            if original_text is not None:
                seg["text"] = original_text

            # Substitute placeholders in the translated text. If a placeholder
            # was somehow stripped by the translator, we leave that subject as
            # whatever the translator produced — better than crashing.
            translated = seg.get("text_translated", "") or ""
            for placeholder, english_subject in kept.items():
                if placeholder in translated:
                    translated = translated.replace(placeholder, english_subject)
                else:
                    print(f"[KEEP-SUBJ] Placeholder {placeholder} missing from "
                          f"translation — translator dropped it. Subject "
                          f"{english_subject!r} will not be inserted.", flush=True)
            seg["text_translated"] = translated
            restored_count += 1

        if restored_count > 0:
            print(f"[KEEP-SUBJ] Restored English noun subjects in "
                  f"{restored_count} translated segments", flush=True)
        return restored_count

    def _translate_segments(self, segments):
        """Translate each segment individually, preserving timestamps for sync.

        Respects self.cfg.translation_engine setting.
        Turbo mode: Groq + SambaNova in parallel (both Llama 3.3 70B).

        If self.cfg.keep_subject_english is True, NOUN subjects are masked
        before dispatch and restored after — see _mask_noun_subjects_in_segments
        and _restore_noun_subjects_in_segments.
        """
        # Same-language dubbing: skip translation, just copy text through
        src = (self.cfg.source_language or "").split("-")[0]
        tgt = (self.cfg.target_language or "").split("-")[0]
        if src and src != "auto" and src == tgt:
            self._report("translate", 0.5,
                         f"Same language ({src}→{tgt}) — skipping translation, re-voicing only...")
            for seg in segments:
                seg["text_translated"] = seg.get("text", "")
            self._report("translate", 1.0,
                         f"Copied {len(segments)} segments as-is (no translation needed)")
            return

        # ── Pre-translation: merge sentence fragments ──
        # Whisper sometimes splits mid-sentence, producing segments like:
        #   "filled forest This made him desperately want to gain the power to protect"
        #   "himself."
        # Translating these independently gives garbage ("जंगल भर गया।").
        # Fix: merge fragments into their neighbors before translation so
        # Google/Gemini/etc get complete sentences.
        #
        # Fragment detection (any of these → merge into previous):
        #   - Starts with lowercase letter (mid-sentence continuation)
        #   - Text has ≤ 3 words AND doesn't end with sentence terminator
        #   - Text starts with a conjunction (and, but, or, so, when, etc.)
        import re as _merge_re
        # Conservative conjunctions: only words that ALMOST NEVER start a
        # new sentence. "when", "while", "because", "if", "so" removed —
        # they frequently start new sentences in English narration.
        _MERGE_WORDS = {"and", "but", "or", "also", "yet", "still",
                        "himself", "herself", "itself", "themselves",
                        "too", "either", "neither", "nor"}
        merged_count = 0
        i = 1
        while i < len(segments):
            text = (segments[i].get("text", "") or "").strip()
            if not text:
                i += 1
                continue

            # ── Rule: NEVER merge into a predecessor that ends with
            # sentence-terminal punctuation. If the previous segment
            # already forms a complete sentence, the current segment
            # should start fresh regardless of its own characteristics.
            prev_text = (segments[i - 1].get("text", "") or "").strip()
            prev_ends_with_terminal = bool(_merge_re.search(r'[.!?।]$', prev_text))
            if prev_ends_with_terminal:
                i += 1
                continue

            first_word = text.split()[0] if text else ""
            word_count = len(text.split())
            starts_lower = text[0].islower() if text else False
            starts_merge_word = first_word.lower().rstrip(".,!?") in _MERGE_WORDS
            is_short_fragment = word_count <= 3 and not _merge_re.search(r'[.!?।]$', text)

            should_merge = starts_lower or starts_merge_word or is_short_fragment

            if should_merge and i > 0:
                # Merge this segment's text into the previous segment
                segments[i - 1]["text"] = prev_text + " " + text
                # Extend the previous segment's end time to cover this one
                segments[i - 1]["end"] = segments[i].get("end", segments[i - 1].get("end", 0))
                # Remove the merged segment
                segments.pop(i)
                merged_count += 1
                # Don't advance i — check the NEXT segment against the
                # now-extended previous segment (it might also be a fragment)
            else:
                i += 1

        if merged_count > 0:
            self._report("translate", 0.01,
                         f"Merged {merged_count} sentence fragments before translation "
                         f"({len(segments) + merged_count} -> {len(segments)} segments)")
            print(f"[FRAGMENT-MERGE] Merged {merged_count} fragments into neighbors "
                  f"for cleaner translation input", flush=True)

        # ── Keep noun subjects in source language: pre-translation mask ──
        # Wrapped in try/finally so the restore pass ALWAYS runs even if a
        # translator engine raises mid-dispatch — otherwise we'd leak placeholders
        # into the SRT, manual review files, and downstream pipeline state.
        masked_count = 0
        if getattr(self.cfg, "keep_subject_english", False):
            try:
                masked_count = self._mask_noun_subjects_in_segments(segments)
            except Exception as e:
                print(f"[KEEP-SUBJ] mask pass failed: {e} — continuing without masking",
                      flush=True)
                masked_count = 0

        try:
            self._dispatch_translation_engine(segments)
        finally:
            if masked_count > 0:
                try:
                    self._restore_noun_subjects_in_segments(segments)
                except Exception as e:
                    print(f"[KEEP-SUBJ] restore pass failed: {e} — translated text "
                          f"may contain placeholder tokens like __KEEP_SUBJ_0__",
                          flush=True)

    def _dispatch_translation_engine(self, segments):
        """Engine-routing helper extracted from _translate_segments.

        Reads self.cfg.translation_engine and calls the appropriate
        _translate_segments_<engine> method. Lives as its own method so the
        keep-subject-english wrapper can use a try/finally around the dispatch
        without rewriting all the early `return` statements inline.
        """
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        groq_key = get_groq_key()
        sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "").strip()
        gemini_key = get_gemini_key()
        engine = self.cfg.translation_engine
        print(f"[Translate] Engine selected: '{engine}' | GEMINI_API_KEY: {'SET' if gemini_key else 'EMPTY'}", flush=True)

        turbo_engines = self._get_turbo_engines()

        # Specific engine selected
        if engine == "turbo" and len(turbo_engines) >= 2:
            names = " + ".join(e[0] for e in turbo_engines)
            self._report("translate", 0.05, f"TURBO: {names} parallel translation ({len(turbo_engines)} engines)...")
            self._translate_segments_turbo(segments, turbo_engines)
            return
        elif engine == "turbo":
            self._report("translate", 0.05, "Turbo needs 2+ engine keys, using best available...")
        elif engine == "hinglish" and self._ollama_available():
            self._report("translate", 0.05, "Using Hinglish AI (custom Ollama model)...")
            self._translate_segments_ollama(segments, force_model="hinglish-translator")
            return
        elif engine == "sambanova" and sambanova_key:
            self._report("translate", 0.05, "Using SambaNova (Llama 3.3 70B) for translation...")
            self._translate_segments_sambanova(segments, sambanova_key)
            return
        elif engine == "groq" and groq_key:
            self._report("translate", 0.05, "Using Groq (Llama 3.3 70B) for translation...")
            self._translate_segments_groq(segments, groq_key)
            return
        elif engine == "gemma4" and gemini_key:
            self._report("translate", 0.05, "Using Gemma 4 (31B) for translation...")
            self._translate_segments_gemma4(segments, gemini_key)
            return
        elif engine == "gemini" and gemini_key:
            self._report("translate", 0.05, "Using Gemini for translation...")
            self._translate_segments_gemini(segments, gemini_key)
            return
        elif engine == "ollama" and self._ollama_available():
            self._report("translate", 0.05, "Using Ollama (local LLM) for translation...")
            self._translate_segments_ollama(segments)
            return
        elif engine == "chain_dub":
            self._report("translate", 0.01, "Chain Dub: Pass 1 — IndicTrans2 → LLM → Rules...")
            self._translate_segments_nllb_polish(segments)
            if len(turbo_engines) >= 1:
                names = " + ".join(e[0] for e in turbo_engines)
                self._report("translate", 0.50, f"Chain Dub: Pass 2 — {names} refinement...")
                self._translate_segments_turbo_refine(segments, turbo_engines)
                self._report("translate", 0.99, f"Chain Dub complete: NLLB+Polish → {names} refine!")
            else:
                self._report("translate", 0.99, "Chain Dub: Refine skipped (need Groq or SambaNova key)")
            return
        elif engine == "nllb_polish":
            self._report("translate", 0.02, "Using NLLB → LLM → Rules pipeline...")
            self._translate_segments_nllb_polish(segments)
            return
        elif engine == "nllb":
            self._report("translate", 0.02, "Using NLLB-200 (local meaning model)...")
            self._translate_segments_nllb(segments)
            return
        elif engine == "google_polish":
            self._report("translate", 0.02, "Using Google Translate + LLM Polish...")
            self._translate_segments_google_polish(segments)
            return
        elif engine == "google":
            self._report("translate", 0.1, "Using Google Translate (parallel x20)...")
            self._translate_segments_google(segments)
            return
        elif engine == "cerebras":
            self._report("translate", 0.05, "Using Cerebras (Llama 3.3 70B, fastest LLM)...")
            ok = self._translate_segments_cerebras(segments)
            if not ok:
                self._report("translate", 0.1, "Cerebras failed — falling back to Google Translate...")
                self._translate_segments_google(segments)
            return

        # Auto mode: NEW priority — Google → Cerebras → Gemma 4 → others
        # Google: fastest free, parallel x20
        # Cerebras: fastest LLM, decent Hindi
        # Gemma 4: best Hindi quality, slower
        cerebras_key = get_cerebras_key()
        is_hindi = self.cfg.target_language in ("hi", "hi-IN")

        # Try Google Translate first (FASTEST, FREE, parallel x20)
        try:
            self._report("translate", 0.05, "Auto: Google Translate (parallel x20, fastest)...")
            self._translate_segments_google(segments)
            # Check if all segments got translated
            missing = sum(1 for s in segments if not s.get("text_translated"))
            if missing > len(segments) * 0.05:  # >5% missing
                raise RuntimeError(f"Google Translate missed {missing} segments")
            return
        except Exception as e:
            self._report("translate", 0.1, f"Google failed ({e}), trying Cerebras...")

        # Fallback 1: Cerebras (fastest LLM)
        if cerebras_key:
            try:
                ok = self._translate_segments_cerebras(segments)
                if ok:
                    return
            except Exception as e:
                self._report("translate", 0.1, f"Cerebras failed ({e}), trying Gemma 4...")

        # Fallback 2: Original LLM cascade (Gemma 4 → Groq → SambaNova → Gemini)
        any_llm = groq_key or sambanova_key or gemini_key

        if any_llm:
            # Best quality for any language: IndicTrans2 meaning → LLM turbo rewrite → (Hindi: rules)
            turbo_label = (
                " + ".join([n for n, _ in [("Groq", groq_key), ("SambaNova", sambanova_key)] if _])
                + (" + Gemini" if gemini_key else "")
            )
            self._report("translate", 0.02,
                         f"Auto: IndicTrans2 → {turbo_label} rewrite"
                         + (" → Rules" if is_hindi else "") + " (best quality)...")
            self._translate_segments_nllb_polish(segments)
        elif len(turbo_engines) >= 2:
            names = " + ".join(e[0] for e in turbo_engines)
            self._report("translate", 0.05, f"TURBO: {names} parallel translation ({len(turbo_engines)} engines)...")
            self._translate_segments_turbo(segments, turbo_engines)
        elif openai_key:
            self._report("translate", 0.05, "Using GPT-4o for premium translation...")
            self._translate_segments_openai(segments, openai_key)
        elif groq_key:
            self._report("translate", 0.05, "Using Groq (Llama 3.3 70B) for translation...")
            self._translate_segments_groq(segments, groq_key)
        elif sambanova_key:
            self._report("translate", 0.05, "Using SambaNova (Llama 3.3 70B) for translation...")
            self._translate_segments_sambanova(segments, sambanova_key)
        elif gemini_key:
            self._report("translate", 0.05, "Using Gemini for colloquial translation...")
            self._translate_segments_gemini(segments, gemini_key)
        elif self._ollama_available():
            self._report("translate", 0.05, "Using Ollama (local LLM) for translation...")
            self._translate_segments_ollama(segments)
        else:
            # No API keys, no Ollama — try IndicTrans2 alone, fallback to Google
            try:
                _get_meaning_model()
                self._report("translate", 0.02, "Auto: Using IndicTrans2 (local, offline)...")
                self._translate_segments_nllb(segments)
            except Exception:
                self._report("translate", 0.1, "No engines available, using Google Translate...")
                self._translate_segments_google(segments)

    def _translate_segments_gemini(self, segments, api_key):
        """Translate segments in numbered batches using Gemini for context-aware output."""
        from google import genai
        client = genai.Client(api_key=api_key)

        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch)]
            prompt = (
                self._get_translation_prompt("system") + "\n\n"
                + self._get_translation_prompt("user_prefix")
                + "\n".join(lines)
            )

            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-pro", contents=prompt)
                    translations = self._parse_numbered_translations(response.text, len(batch))
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    success = True
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.1 + 0.8 * (batch_idx / total_batches),
                                     f"Rate limited, retrying in {wait}s...")
                        time.sleep(wait)

            if not success:
                self._report("translate", 0.1, "Gemini failed, using Google Translate for batch...")
                for seg in batch:
                    seg["text_translated"] = self._translate_single_fallback(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches}")

    def _translate_segments_gemma4(self, segments, api_key):
        """Translate segments using Gemma 4 + Groq DUAL PARALLEL.

        Gemma 4: 5 keys × 2 workers = 10 workers (best quality)
        Groq:    5 keys × 1 worker  =  5 workers (fast, good quality)
        Total:   15 parallel translation workers

        Batches distributed across both engines simultaneously.
        Gemma 4 gets priority batches (first 70%), Groq gets rest (30%).
        """
        from google import genai
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_gemini = _gemini_keys.count()
        num_groq = _groq_keys.count()
        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size

        # Build all batches
        batches = []
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batches.append((batch_idx, segments[start:end]))

        completed = [0]

        # Build the word-count-limited prompt for a batch
        def _build_prompt(batch):
            lines = []
            for i, seg in enumerate(batch):
                text = seg['text']
                en_words = len(text.split())
                max_words = max(2, int(en_words * 1.80))  # 80% increase cap
                lines.append(f"{i+1}. [{en_words}w, MAX {max_words}w] {text}")
            return (
                self._get_translation_prompt("system") + "\n\n"
                "You are dubbing a video. Use natural spoken Hindi, NOT literary.\n"
                "Adapt idioms for Hindi audience.\n\n"
                "STRICT WORD LIMIT: Each line shows [Xw, MAX Yw].\n"
                "Hindi MUST NOT exceed MAX word count. This is for voice-over dubbing.\n"
                "Use compact spoken Hindi. Drop filler words (है कि, तो, ही, भी).\n"
                "Shorter = better. If you can say it in fewer words, do it.\n\n"
                + self._get_translation_prompt("user_prefix")
                + "\n".join(lines)
            )

        # Gemma 4 worker
        def translate_gemma(batch_idx, batch):
            worker_key = get_gemini_key() or api_key
            worker_client = genai.Client(api_key=worker_key)
            prompt = _build_prompt(batch)

            for attempt in range(3):
                try:
                    import threading as _th
                    _result = [None]
                    _error = [None]
                    def _call():
                        try:
                            _result[0] = worker_client.models.generate_content(
                                model="gemma-4-31b-it", contents=prompt)
                        except Exception as e:
                            _error[0] = e
                    _t = _th.Thread(target=_call)
                    _t.start()
                    _t.join(timeout=30)
                    if _t.is_alive():
                        raise TimeoutError("Gemma 4 hung (>30s)")
                    if _error[0]:
                        raise _error[0]

                    translations = self._parse_numbered_translations(_result[0].text, len(batch))
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    return ("gemma4", True)
                except Exception:
                    _gemini_keys.report_rate_limit(worker_key)
                    if attempt < 2:
                        worker_key = get_gemini_key() or api_key
                        worker_client = genai.Client(api_key=worker_key)
                        time.sleep(2)
            # Final fallback: Google Translate
            for seg in batch:
                seg["text_translated"] = self._translate_single_fallback(seg["text"])
            return ("fallback", False)

        # Groq worker
        def translate_groq(batch_idx, batch):
            groq_key = get_groq_key()
            if not groq_key:
                for seg in batch:
                    seg["text_translated"] = self._translate_single_fallback(seg["text"])
                return ("fallback", False)

            prompt = _build_prompt(batch)
            try:
                translations = self._translate_batch_openai_compat(
                    batch, *self.TURBO_ENGINE_CONFIG.get("Groq", ("", "")), groq_key, "Groq")
                if translations:
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    return ("groq", True)
            except Exception:
                pass
            for seg in batch:
                seg["text_translated"] = self._translate_single_fallback(seg["text"])
            return ("fallback", False)

        # Split batches: 70% Gemma 4, 30% Groq
        gemma_count = int(total_batches * 0.70) if num_groq > 0 else total_batches
        groq_count = total_batches - gemma_count

        gemma_workers = min(2 * max(num_gemini, 1), gemma_count) if gemma_count > 0 else 0
        groq_workers = min(num_groq, groq_count) if groq_count > 0 else 0
        total_workers = gemma_workers + groq_workers

        self._report("translate", 0.05,
                     f"DUAL PARALLEL: Gemma4 ×{gemma_workers} + Groq ×{groq_workers} = "
                     f"{total_workers} workers ({total_batches} batches)...")

        with ThreadPoolExecutor(max_workers=max(total_workers, 1)) as executor:
            futures = {}
            for idx, batch in batches[:gemma_count]:
                futures[executor.submit(translate_gemma, idx, batch)] = idx
            for idx, batch in batches[gemma_count:]:
                futures[executor.submit(translate_groq, idx, batch)] = idx

            for future in as_completed(futures):
                completed[0] += 1
                self._report("translate", 0.1 + 0.9 * (completed[0] / total_batches),
                             f"Translated batch {completed[0]}/{total_batches} (Gemma 4 parallel)")

    def _translate_segments_groq(self, segments, api_key):
        """Translate segments using Groq (Llama 3.3 70B) — fast, free, context-aware."""
        from groq import Groq
        client = Groq(api_key=api_key)

        batch_size = 50  # Larger batches = fewer API calls = faster
        total_batches = (len(segments) + batch_size - 1) // batch_size
        system_msg = self._get_translation_prompt("system")

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch)]
            user_msg = self._get_translation_prompt("user_prefix") + "\n".join(lines)

            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.7,
                        max_tokens=8192,
                    )
                    result_text = response.choices[0].message.content
                    translations = self._parse_numbered_translations(result_text, len(batch))
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    success = True
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.1 + 0.8 * (batch_idx / total_batches),
                                     f"Groq rate limited, retrying in {wait}s...")
                        time.sleep(wait)

            if not success:
                self._report("translate", 0.1, "Groq failed, using Google Translate for batch...")
                for seg in batch:
                    seg["text_translated"] = self._translate_single_fallback(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches} (Groq)")

    def _translate_batch_openai_compat(self, batch, api_url, api_key, model, engine_name):
        """Translate a batch using any OpenAI-compatible API. Returns translations list or None."""
        import requests as _requests
        lines = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch)]
        system_msg = self._get_translation_prompt("system")
        user_msg = self._get_translation_prompt("user_prefix") + "\n".join(lines)

        for attempt in range(3):
            try:
                resp = _requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 8192,
                    },
                    timeout=60,
                )
                if resp.status_code == 429:
                    _groq_keys.report_rate_limit(api_key)
                    api_key = get_groq_key()  # Switch to next key
                resp.raise_for_status()
                return self._parse_numbered_translations(
                    resp.json()["choices"][0]["message"]["content"], len(batch))
            except Exception:
                if attempt < 2:
                    _groq_keys.report_rate_limit(api_key)
                    api_key = get_groq_key()  # Try next key on retry
                    time.sleep(1)
        return None

    # Engine configs for OpenAI-compatible APIs (all using Llama 3.3 70B)
    TURBO_ENGINE_CONFIG = {
        "Groq": ("https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
        "SambaNova": ("https://api.sambanova.ai/v1/chat/completions", "Meta-Llama-3.3-70B-Instruct"),
    }

    def _translate_segments_sambanova(self, segments, api_key):
        """Translate segments using SambaNova (Llama 3.3 70B) — fast, free."""
        url, model = self.TURBO_ENGINE_CONFIG["SambaNova"]
        batch_size = 50
        total_batches = (len(segments) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            translations = self._translate_batch_openai_compat(batch, url, api_key, model, "SambaNova")
            if translations:
                for i, seg in enumerate(batch):
                    seg["text_translated"] = translations[i] if translations[i] else seg["text"]
            else:
                self._report("translate", 0.1, "SambaNova failed, using Google Translate for batch...")
                for seg in batch:
                    seg["text_translated"] = self._translate_single_fallback(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches} (SambaNova)")

    def _translate_segments_turbo(self, segments, engines):
        """TURBO: Distribute batches across multiple engines in parallel.

        Round-robin batches across all available engines (Groq, SambaNova).
        All run simultaneously via threads = up to 2x faster translation.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        batch_size = 50
        total_batches = (len(segments) + batch_size - 1) // batch_size
        num_engines = len(engines)

        # Build batch list
        batches = []
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batches.append((batch_idx, segments[start:end]))

        # Round-robin batches across engines, run all in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=num_engines * 2) as pool:
            futures = {}
            for batch_idx, batch in batches:
                engine_name, api_key = engines[batch_idx % num_engines]
                url, model = self.TURBO_ENGINE_CONFIG[engine_name]
                fut = pool.submit(
                    self._translate_batch_openai_compat,
                    batch, url, api_key, model, engine_name)
                futures[fut] = (batch_idx, batch, engine_name)

            for fut in as_completed(futures):
                batch_idx, batch, engine_name = futures[fut]
                translations = fut.result()
                if translations:
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                else:
                    # Retry with another Turbo engine (Groq if SambaNova failed, vice versa)
                    retry_success = False
                    for retry_name, retry_key in engines:
                        if retry_name == engine_name:
                            continue  # Skip the engine that just failed
                        url, model = self.TURBO_ENGINE_CONFIG[retry_name]
                        self._report("translate", 0.1,
                                     f"{engine_name} failed batch {batch_idx+1}, retrying with {retry_name}...")
                        retry_result = self._translate_batch_openai_compat(
                            batch, url, retry_key, model, retry_name)
                        if retry_result:
                            for i, seg in enumerate(batch):
                                seg["text_translated"] = retry_result[i] if retry_result[i] else seg["text"]
                            retry_success = True
                            break
                    if not retry_success:
                        # All Turbo engines failed — last resort: Google Translate
                        self._report("translate", 0.1,
                                     f"All Turbo engines failed batch {batch_idx+1}, using Google Translate...")
                        for seg in batch:
                            seg["text_translated"] = self._translate_single_fallback(seg["text"])

                completed += 1
                self._report("translate", 0.1 + 0.9 * (completed / total_batches),
                             f"TURBO: {completed}/{total_batches} batches ({engine_name})")

    def _translate_segments_turbo_refine(self, segments, engines):
        """TURBO REFINE: Second pass — polish already-translated text using parallel engines.

        Takes text_translated from Pass 1 and asks LLMs to refine for natural,
        spoken dubbing quality. Round-robins batches across Groq + SambaNova.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        is_hindi = self.cfg.target_language in ("hi", "hi-IN")

        if is_hindi:
            refine_system = (
                "You are a Hindi dubbing quality reviewer. You receive Hindi dubbing lines "
                "that were already translated and polished.\n\n"
                "Your job: REVIEW and REFINE each line for maximum naturalness and flow.\n\n"
                "RULES:\n"
                "1. Fix any remaining formal/unnatural phrasing — make it sound 100% spoken\n"
                "2. Keep the EXACT SAME meaning — only improve how it sounds\n"
                "3. If a line is already perfect, return it unchanged\n"
                "4. Ensure rhythm and breath-pauses work for voice dubbing\n"
                "5. Keep English loanwords Indians commonly use\n"
                "6. Output ONLY numbered refined lines. No notes or explanations.\n"
            )
            refine_prefix = "REFINE these Hindi dubbing lines for maximum spoken naturalness:\n\n"
        else:
            refine_system = (
                f"You are a {target_name} dubbing quality reviewer. Review and refine these "
                f"already-translated {target_name} dubbing lines for maximum naturalness.\n"
                f"Fix awkward phrasing, keep meaning intact. If a line is perfect, keep it.\n"
                f"Output ONLY numbered refined lines."
            )
            refine_prefix = f"Refine these {target_name} dubbing lines:\n\n"

        batch_size = 50
        total_batches = (len(segments) + batch_size - 1) // batch_size
        num_engines = len(engines)

        batches = []
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batches.append((batch_idx, segments[start:end]))

        completed = 0
        with ThreadPoolExecutor(max_workers=num_engines * 2) as pool:
            futures = {}
            for batch_idx, batch in batches:
                engine_name, api_key = engines[batch_idx % num_engines]
                url, model = self.TURBO_ENGINE_CONFIG[engine_name]

                lines = [f"{i+1}. {seg['text_translated']}" for i, seg in enumerate(batch)]
                user_msg = refine_prefix + "\n".join(lines)

                fut = pool.submit(
                    self._turbo_refine_batch,
                    batch, url, api_key, model, engine_name,
                    refine_system, user_msg)
                futures[fut] = (batch_idx, batch, engine_name)

            for fut in as_completed(futures):
                batch_idx, batch, engine_name = futures[fut]
                refined = fut.result()
                if refined:
                    for i, seg in enumerate(batch):
                        if refined[i]:
                            seg["text_translated"] = refined[i]

                completed += 1
                self._report("translate", 0.50 + 0.48 * (completed / total_batches),
                             f"Turbo Refine: {completed}/{total_batches} batches ({engine_name})")

    def _turbo_refine_batch(self, batch, api_url, api_key, model, engine_name,
                            system_msg, user_msg):
        """Send a refinement batch to an OpenAI-compatible API. Returns list or None."""
        import requests as _requests
        retries = 3
        for attempt in range(retries):
            try:
                resp = _requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.5,
                        "max_tokens": 8192,
                    },
                    timeout=90,
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                return self._parse_numbered_translations(text, len(batch))
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
        return None

    def _translate_segments_ollama(self, segments, force_model: str = ""):
        """Translate segments using local Ollama LLM in batches."""
        import requests as _requests

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the detected language"

        # Pick model
        try:
            resp = _requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            models = []

        model = None
        if force_model:
            model = force_model if any(force_model in m for m in models) else None
        if not model:
            preferred = (
                ["hinglish-translator"] if self.cfg.target_language in ("hi", "hi-IN") else []
            ) + ["qwen2.5:14b", "qwen2.5:32b", "llama3.1:8b", "llama3:8b", "gemma2:9b", "mistral:7b"]
            for pref in preferred:
                for m in models:
                    if pref.split(":")[0] in m:
                        model = m
                        break
                if model:
                    break
            if not model and models:
                model = models[0]
        if not model:
            self._report("translate", 0.1, "No Ollama models found, using Google Translate...")
            self._translate_segments_google(segments)
            return

        self._report("translate", 0.05, f"Using Ollama model: {model}")

        batch_size = 15  # Smaller batches for local LLM
        total_batches = (len(segments) + batch_size - 1) // batch_size

        system_msg = self._get_translation_prompt("system")

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch)]
            user_msg = self._get_translation_prompt("user_prefix") + "\n".join(lines)

            try:
                resp = _requests.post("http://localhost:11434/api/chat", json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.7},
                }, timeout=120)
                result_text = resp.json().get("message", {}).get("content", "")
                translations = self._parse_numbered_translations(result_text, len(batch))
                for i, seg in enumerate(batch):
                    seg["text_translated"] = translations[i] if translations[i] else seg["text"]
            except Exception as e:
                self._report("translate", 0.1, f"Ollama failed for batch: {e}, using Google Translate...")
                for seg in batch:
                    seg["text_translated"] = self._translate_single_fallback(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches} (Ollama: {model})")

    def _translate_segments_openai(self, segments, api_key):
        """Translate segments using OpenAI GPT-4o for highest quality."""
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size
        system_msg = self._get_translation_prompt("system")

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch)]
            user_msg = self._get_translation_prompt("user_prefix") + "\n".join(lines)

            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.3,
                    )
                    text = response.choices[0].message.content or ""
                    translations = self._parse_numbered_translations(text, len(batch))
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    success = True
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.1 + 0.8 * (batch_idx / total_batches),
                                     f"GPT-4o rate limited, retrying in {wait}s...")
                        time.sleep(wait)

            if not success:
                # Fall back to Groq > Gemini > Google Translate
                groq_key = get_groq_key()
                gemini_key = get_gemini_key()
                if groq_key:
                    self._report("translate", 0.1, "GPT-4o failed, falling back to Groq...")
                    for seg in batch:
                        seg["text_translated"] = seg.get("text", "")
                    self._translate_segments_groq(batch, groq_key)
                elif gemini_key:
                    self._report("translate", 0.1, "GPT-4o failed, falling back to Gemini...")
                    for seg in batch:
                        seg["text_translated"] = seg.get("text", "")
                    self._translate_segments_gemini(batch, gemini_key)
                else:
                    for seg in batch:
                        seg["text_translated"] = self._translate_single_fallback(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches} (GPT-4o)")

    def _translate_segments_cerebras(self, segments):
        """Translate segments using Cerebras (Llama 3.3 70B) — FASTEST LLM inference.

        ~2400 tokens/sec — 3x faster than Groq.
        60 RPM free tier, 1M tokens/day.
        Hindi quality: 6-7/10 (Llama-based, occasional broken chars).
        Used as second-priority translator after Google Translate.
        """
        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError:
            print("[Cerebras] cerebras-cloud-sdk not installed", flush=True)
            return False

        from concurrent.futures import ThreadPoolExecutor, as_completed

        api_key = get_cerebras_key()
        if not api_key:
            print("[Cerebras] No CEREBRAS_API_KEY set", flush=True)
            return False

        num_keys = _cerebras_keys.count()
        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size
        # Cerebras: 60 RPM per key, use 2 workers per key
        max_parallel = min(2 * max(num_keys, 1), total_batches)

        batches = []
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batches.append((batch_idx, segments[start:end]))

        completed = [0]

        def translate_batch(batch_idx, batch):
            worker_key = get_cerebras_key() or api_key
            client = Cerebras(api_key=worker_key)

            lines = []
            for i, seg in enumerate(batch):
                text = seg['text']
                en_words = len(text.split())
                max_words = max(2, int(en_words * 1.80))  # 80% increase cap
                lines.append(f"{i+1}. [{en_words}w, MAX {max_words}w] {text}")

            prompt = (
                "You are dubbing a video to Hindi. Use natural spoken Hindi.\n"
                "STRICT WORD LIMIT: Each line shows [Xw, MAX Yw].\n"
                "Hindi MUST NOT exceed MAX word count. This is for voice-over dubbing.\n"
                "Output ONLY numbered Hindi translations, nothing else.\n\n"
                + "\n".join(lines)
            )

            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000,
                        temperature=0.3,
                    )
                    result_text = response.choices[0].message.content
                    translations = self._parse_numbered_translations(result_text, len(batch))
                    for i, seg in enumerate(batch):
                        if translations[i]:
                            seg["text_translated"] = translations[i]
                    return True
                except Exception as e:
                    _cerebras_keys.report_rate_limit(worker_key)
                    if attempt < 2:
                        worker_key = get_cerebras_key() or api_key
                        client = Cerebras(api_key=worker_key)
                        time.sleep(2)
            return False

        keys_label = f", {num_keys} keys" if num_keys > 1 else ""
        self._report("translate", 0.05,
                     f"Cerebras Llama 3.3 (x{max_parallel} workers, {total_batches} batches{keys_label})...")

        success_count = 0
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(translate_batch, idx, b): idx for idx, b in batches}
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                completed[0] += 1
                self._report("translate", 0.1 + 0.85 * (completed[0] / total_batches),
                             f"Cerebras: {completed[0]}/{total_batches} batches")

        # Check if we got translations for all segments
        missing = sum(1 for s in segments if not s.get("text_translated"))
        if missing > len(segments) * 0.1:  # >10% missing → failure
            print(f"[Cerebras] {missing}/{len(segments)} segments missing — fallback needed", flush=True)
            return False
        return True

    def _translate_segments_google(self, segments):
        """Translate segments using Google Translate — PARALLEL with 20 workers.

        Uses deep_translator (free public Google Translate website).
        No API key, no cost, 7/10 Hindi quality.
        20 parallel workers = ~3-5 seconds for 335 segments.
        """
        from deep_translator import GoogleTranslator
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(segments)
        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        src = self.cfg.source_language if self.cfg.source_language != "auto" else "auto"

        self._report("translate", 0.02,
                     f"Google Translate (parallel x20) → {target_name} ({total} segments)...")

        completed = [0]

        def _is_garbage(text: str) -> bool:
            """Detect Google Translate error pages returned as 'translations'."""
            if not text:
                return True
            low = text.lower()
            return ("error 500" in low or "server error" in low
                    or "that's an error" in low or "<html" in low
                    or "<!doctype" in low)

        def translate_one(idx_seg):
            idx, seg = idx_seg
            retries = 3
            for attempt in range(retries):
                try:
                    translator = GoogleTranslator(source=src, target=self.cfg.target_language)
                    result = translator.translate(seg["text"])
                    if _is_garbage(result):
                        if attempt < retries - 1:
                            import time; time.sleep(1.5 * (attempt + 1))
                            continue
                        # All retries returned garbage — keep original
                        seg["text_translated"] = seg["text"]
                    else:
                        seg["text_translated"] = result or seg["text"]
                    break
                except Exception:
                    if attempt < retries - 1:
                        import time; time.sleep(1.5 * (attempt + 1))
                    else:
                        seg["text_translated"] = seg["text"]
            return idx

        # 20 parallel workers — Google Translate handles this fine
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(translate_one, (i, s)): i for i, s in enumerate(segments)}
            for future in as_completed(futures):
                completed[0] += 1
                if completed[0] % 20 == 0 or completed[0] == total:
                    self._report("translate", 0.05 + 0.90 * (completed[0] / total),
                                 f"Google Translate: {completed[0]}/{total}")

        self._report("translate", 1.0, f"Google Translate complete: {total} segments")

    def _translate_segments_google_polish(self, segments):
        """Two-stage translation: Google Translate (fast) → LLM polish (natural).

        Stage 1: Google Translate for all segments (fast, free, reliable)
        Stage 2: LLM (Groq/Gemini/SambaNova) polishes into natural spoken Hindi
        """
        from deep_translator import GoogleTranslator

        total = len(segments)
        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)

        # ── Stage 1: Google Translate (fast bulk) ─────────────────────────
        self._report("translate", 0.02, f"Stage 1/2: Google Translate → {target_name} ({total} segments)...")
        src = self.cfg.source_language if self.cfg.source_language != "auto" else "auto"
        translator = GoogleTranslator(source=src, target=self.cfg.target_language)

        for i, seg in enumerate(segments):
            try:
                seg["text_translated"] = translator.translate(seg["text"]) or seg["text"]
            except Exception:
                seg["text_translated"] = seg["text"]
            if (i + 1) % 20 == 0 or i == total - 1:
                self._report("translate", 0.02 + 0.38 * ((i + 1) / total),
                             f"Stage 1/2: Google Translate {i + 1}/{total}")

        self._report("translate", 0.40, "Stage 1 complete. Starting LLM polish...")

        # ── Stage 2: LLM Polish ──────────────────────────────────────────
        groq_key = get_groq_key()
        sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "").strip()
        gemini_key = get_gemini_key()

        is_hindi = self.cfg.target_language in ("hi", "hi-IN")

        if is_hindi:
            polish_system = (
                "You are a Hindi dubbing polish expert. You receive Google-translated Hindi text that is "
                "grammatically correct but sounds robotic and formal.\n\n"
                "Your job: REWRITE each line to sound like NATURAL, daily-spoken, punchy Hindi narration.\n\n"
                "RULES:\n"
                "1. Keep the SAME meaning — don't add or remove information\n"
                "2. Make it sound like a Hindi YouTuber narrating a story — dramatic, gripping, engaging\n"
                "3. Replace formal/textbook Hindi with colloquial spoken Hindi\n"
                "4. Use contractions: नहीं→नी, कुछ नहीं→कुछ नी\n"
                "5. Use तू/तुम, not आप (unless showing respect to elders)\n"
                "6. Keep English words Indians naturally use: phone, plan, delete, hospital, police, driver\n"
                "7. Add punch: 'और बस... यहीं से खेल शुरू हुआ' not 'और इसी स्थान से'\n"
                "8. BANNED words: अतः, किन्तु, तथापि, उक्त, यद्यपि, एवं, आवश्यकता, विक्षिप्त, उत्साहित (use excited)\n"
                "9. Emotional connectors where natural: 'और भाई सुनो', 'अब यहाँ twist आता है'\n"
                "10. Output ONLY numbered polished lines. No notes, no explanations.\n"
            )
            polish_prefix = (
                "Polish these Google-translated Hindi lines into natural spoken Hindi narration. "
                "Each line must sound like it was WRITTEN in Hindi, not translated. "
                "Keep same meaning, just make it punchy and natural.\n\n"
                "GOOGLE TRANSLATED:\n"
            )
        else:
            polish_system = (
                f"You are a {target_name} language polishing expert. You receive Google-translated "
                f"{target_name} text that is correct but sounds unnatural.\n\n"
                f"Your job: REWRITE each line to sound like a NATIVE {target_name} speaker talking naturally.\n"
                f"Keep the SAME meaning. Make it conversational, not formal.\n"
                f"Output ONLY numbered polished lines. No notes."
            )
            polish_prefix = (
                f"Polish these Google-translated {target_name} lines into natural spoken {target_name}. "
                f"Same meaning, just more natural and conversational.\n\n"
                f"GOOGLE TRANSLATED:\n"
            )

        # Try engines in order: Groq (fastest) → SambaNova → Gemini
        engine_order = []
        if groq_key:
            engine_order.append(("Groq", groq_key))
        if sambanova_key:
            engine_order.append(("SambaNova", sambanova_key))
        if gemini_key:
            engine_order.append(("Gemini", gemini_key))

        if not engine_order:
            self._report("translate", 0.95,
                         "No LLM API keys for polish — using Google Translate as-is")
            return

        # Polish in batches
        batch_size = 40
        total_batches = (total + batch_size - 1) // batch_size
        engine_idx = 0

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch = segments[start:end]

            # Build numbered list with emotion mode hints
            _MODE_HINTS_GP = {
                "punchy": "[punchy]", "emotional": "[emotional]",
                "comedic": "[comedic]", "neutral": "",
            }
            lines = [
                f"{i+1}. {_MODE_HINTS_GP.get(seg.get('emotion', 'neutral'), '')} {seg['text_translated']}".strip()
                for i, seg in enumerate(batch)
            ]
            user_msg = polish_prefix + "\n".join(lines)

            polished = None
            # Try each engine until one succeeds
            for attempt in range(len(engine_order)):
                eidx = (engine_idx + attempt) % len(engine_order)
                name, key = engine_order[eidx]

                self._report("translate", 0.40 + 0.55 * (batch_idx / total_batches),
                             f"Stage 2/2: {name} polish — batch {batch_idx + 1}/{total_batches}")

                if name == "Gemini":
                    try:
                        from google import genai
                        client = genai.Client(api_key=key)
                        response = client.models.generate_content(
                            model="gemini-2.5-flash-preview-05-20",
                            contents=polish_system + "\n\n" + user_msg)
                        polished = self._parse_numbered_translations(response.text, len(batch))
                    except Exception:
                        continue
                else:
                    api_url, model = self.TURBO_ENGINE_CONFIG.get(name, (None, None))
                    if not api_url:
                        continue
                    try:
                        import requests as _requests
                        resp = _requests.post(
                            api_url,
                            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                            json={
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": polish_system},
                                    {"role": "user", "content": user_msg},
                                ],
                                "temperature": 0.7,
                                "max_tokens": 8192,
                            },
                            timeout=60,
                        )
                        resp.raise_for_status()
                        polished = self._parse_numbered_translations(
                            resp.json()["choices"][0]["message"]["content"], len(batch))
                    except Exception:
                        continue

                if polished:
                    engine_idx = eidx  # Stick with working engine
                    break

            if polished:
                for i, seg in enumerate(batch):
                    if polished[i]:
                        seg["text_translated"] = polished[i]
            # If all engines fail, keep Google Translate output (already set)

            self._report("translate", 0.40 + 0.55 * ((batch_idx + 1) / total_batches),
                         f"Stage 2/2: Polished batch {batch_idx + 1}/{total_batches}")

        self._report("translate", 0.98, "Google Translate + LLM polish complete!")

    def _translate_segments_nllb(self, segments):
        """Translate segments using local meaning model (IndicTrans2 or NLLB fallback)."""
        import torch
        model, tokenizer, processor, engine = _get_meaning_model()
        tgt_code = _LANG_MAP.get(self.cfg.target_language, "hin_Deva")

        total = len(segments)
        batch_size = 16

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = segments[batch_start:batch_end]
            texts = [seg["text"].strip() for seg in batch]

            if engine == "indictrans2" and processor is not None:
                # IndicTrans2: use IndicProcessor for pre/post processing
                processed = processor.preprocess_batch(texts, src_lang="eng_Latn", tgt_lang=tgt_code)
                inputs = tokenizer(processed, truncation=True, padding="longest",
                                   return_tensors="pt", max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_length=512, num_beams=5)
                raw = tokenizer.batch_decode(out, skip_special_tokens=True)
                translations = processor.postprocess_batch(raw, lang=tgt_code)
            else:
                # NLLB: direct tokenize with forced BOS token
                tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
                tokenizer.src_lang = "eng_Latn"
                inputs = tokenizer(texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, forced_bos_token_id=tgt_token_id, max_length=512)
                translations = tokenizer.batch_decode(out, skip_special_tokens=True)

            for seg, trans in zip(batch, translations):
                seg["text_translated"] = trans.strip()

            engine_label = "IndicTrans2" if engine == "indictrans2" else "NLLB"
            self._report("translate", 0.1 + 0.8 * (batch_end / total),
                         f"{engine_label} translated {batch_end}/{total} segments")

    def _translate_segments_nllb_polish(self, segments):
        """Two-stage pipeline: NLLB (meaning) → LLM (dubbing rewrite) → Rules.

        Stage A: NLLB-200-1.3B for faithful meaning transfer (local GPU)
        Stage B: LLM (Groq/SambaNova/Gemini) for dubbing-style rewrite
        Stage C: Rule engine for cleanup (formal→spoken, glossary, compression)
        """
        total = len(segments)

        # ── Stage A: NLLB meaning translation ────────────────────────────
        self._report("translate", 0.02, f"Stage 1/3: NLLB meaning transfer ({total} segments)...")
        self._translate_segments_nllb(segments)
        self._report("translate", 0.35, "Stage 1/3 complete. Starting dubbing rewrite...")

        # ── Stage B: LLM dubbing polish ──────────────────────────────────
        groq_key = get_groq_key()
        sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "").strip()
        gemini_key = get_gemini_key()

        is_hindi = self.cfg.target_language in ("hi", "hi-IN")

        if is_hindi:
            polish_system = (
                "You are a Hindi DUBBING REWRITER. You receive literal machine-translated Hindi.\n\n"
                "Your job: REWRITE each line into NATURAL, daily-spoken, PUNCHY Hindi dubbing lines.\n\n"
                "RULES:\n"
                "1. Keep the EXACT SAME meaning — don't add/remove information\n"
                "2. Make it sound like a Hindi YouTuber narrating — dramatic, gripping\n"
                "3. Replace ALL formal/bookish Hindi with spoken Hindi\n"
                "4. Use contractions: नहीं→नी, कुछ नहीं→कुछ नी, मुझे→मुझे/मुझको\n"
                "5. Use तू/तुम for peers/enemies, आप only for elders\n"
                "6. Keep English words Indians use: phone, plan, delete, hospital, police, power\n"
                "7. Add punch: 'और बस... यहीं से खेल शुरू हुआ' not 'और इसी स्थान से'\n"
                "8. BANNED: अतः, किन्तु, तथापि, उक्त, यद्यपि, एवं, आवश्यकता, विक्षिप्त\n"
                "9. Lines must be EASY TO SAY ALOUD — short breaths, natural rhythm\n"
                "10. Output ONLY numbered rewritten lines. No notes.\n"
            )
            polish_prefix = (
                "REWRITE these literal Hindi translations into natural spoken Hindi dubbing lines. "
                "Same meaning, but punchy, emotional, speakable.\n\n"
                "LITERAL HINDI:\n"
            )
        else:
            target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
            polish_system = (
                f"You are a {target_name} dubbing rewriter. Rewrite literal translations into "
                f"natural, conversational {target_name}. Keep meaning, improve flow.\n"
                f"Output ONLY numbered rewritten lines."
            )
            polish_prefix = f"Rewrite into natural spoken {target_name}:\n\n"

        llm_engines = []
        if groq_key:
            llm_engines.append(("Groq", groq_key))
        if sambanova_key:
            llm_engines.append(("SambaNova", sambanova_key))
        if gemini_key:
            llm_engines.append(("Gemini", gemini_key))

        if llm_engines:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            batch_size = 40
            batches = []
            for bi in range(0, total, batch_size):
                batches.append(segments[bi: bi + batch_size])
            total_batches = len(batches)

            engine_label = (
                " + ".join(n for n, _ in llm_engines[:2]) + " turbo"
                if len(llm_engines) >= 2 else llm_engines[0][0]
            )
            self._report("translate", 0.35,
                         f"Stage 2/3: {engine_label} dubbing rewrite ({total_batches} batches)...")

            # Emotion → rewrite mode instructions (added per-segment in the prompt)
            _REWRITE_MODE_HINTS = {
                "punchy":    "[MODE: punchy — short, dramatic, hype energy]",
                "emotional": "[MODE: emotional — soft, heartfelt, slow rhythm]",
                "comedic":   "[MODE: comedic — light, playful, natural sarcasm]",
                "neutral":   "[MODE: neutral — clear, conversational, natural flow]",
            }

            def _polish_one_batch(batch, batch_idx):
                """Send batch to ALL available engines simultaneously, return first good result."""
                lines = []
                for i, seg in enumerate(batch):
                    emotion = seg.get("emotion", "neutral")
                    mode_hint = _REWRITE_MODE_HINTS.get(emotion, "")
                    lines.append(f"{i+1}. {mode_hint} {seg['text_translated']}")
                user_msg = polish_prefix + "\n".join(lines)

                def _call_llm(name, key):
                    if name == "Gemini":
                        try:
                            from google import genai
                            client = genai.Client(api_key=key)
                            r = client.models.generate_content(
                                model="gemini-2.5-flash-preview-05-20",
                                contents=polish_system + "\n\n" + user_msg)
                            return self._parse_numbered_translations(r.text, len(batch))
                        except Exception:
                            return None
                    else:
                        api_url, model = self.TURBO_ENGINE_CONFIG.get(name, (None, None))
                        if not api_url:
                            return None
                        try:
                            import requests as _req
                            resp = _req.post(
                                api_url,
                                headers={"Authorization": f"Bearer {key}",
                                         "Content-Type": "application/json"},
                                json={
                                    "model": model,
                                    "messages": [
                                        {"role": "system", "content": polish_system},
                                        {"role": "user",   "content": user_msg},
                                    ],
                                    "temperature": 0.7,
                                    "max_tokens": 8192,
                                },
                                timeout=60,
                            )
                            resp.raise_for_status()
                            return self._parse_numbered_translations(
                                resp.json()["choices"][0]["message"]["content"], len(batch))
                        except Exception:
                            return None

                # Race all engines — take whichever responds first with a valid result
                with ThreadPoolExecutor(max_workers=len(llm_engines)) as inner_pool:
                    futs = {inner_pool.submit(_call_llm, n, k): n for n, k in llm_engines}
                    for fut in as_completed(futs):
                        result = fut.result()
                        if result:
                            return result, futs[fut]
                return None, None

            # Process all batches in parallel (one worker per engine count)
            completed = 0
            with ThreadPoolExecutor(max_workers=max(2, len(llm_engines))) as outer_pool:
                batch_futures = {
                    outer_pool.submit(_polish_one_batch, batch, bi): (bi, batch)
                    for bi, batch in enumerate(batches)
                }
                for fut in as_completed(batch_futures):
                    bi, batch = batch_futures[fut]
                    polished, winner = fut.result()
                    if polished:
                        for i, seg in enumerate(batch):
                            if polished[i]:
                                seg["text_translated"] = polished[i]
                    completed += 1
                    self._report("translate", 0.35 + 0.50 * (completed / total_batches),
                                 f"Stage 2/3: {completed}/{total_batches} batches rewritten"
                                 + (f" ({winner})" if winner else ""))
        else:
            self._report("translate", 0.85, "No LLM API keys — skipping dubbing rewrite")

        # ── Stage C: Rule engine cleanup ─────────────────────────────────
        if is_hindi:
            self._report("translate", 0.90, "Stage 3/3: Rule engine cleanup...")
            for seg in segments:
                seg["text_translated"] = _hindi_rules.apply(seg["text_translated"])
            self._report("translate", 0.95, "Stage 3/3: Rules applied")

        self._report("translate", 0.98, "NLLB → LLM → Rules pipeline complete!")

    def _translate_single_fallback(self, text: str) -> str:
        """Smart fallback: try Ollama (GPU) first, then Google Translate."""
        if self._ollama_available():
            try:
                import requests as _requests
                target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
                # Pick best Ollama model
                resp = _requests.get("http://localhost:11434/api/tags", timeout=2)
                models = [m["name"] for m in resp.json().get("models", [])]
                preferred = ["hindi-dubbing", "qwen2.5:14b", "qwen2.5:32b", "llama3.1:8b", "gemma2:9b", "mistral:7b"]
                model = None
                for pref in preferred:
                    for m in models:
                        if pref.split(":")[0] in m:
                            model = m
                            break
                    if model:
                        break
                if not model and models:
                    model = models[0]
                if model:
                    prompt = (
                        f"Translate the following text to {target_name}. "
                        f"Use natural, conversational {target_name} — not literal translation. "
                        f"Output ONLY the translation, nothing else.\n\n{text}"
                    )
                    resp = _requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": model, "prompt": prompt, "stream": False},
                        timeout=60,
                    )
                    result = resp.json().get("response", "").strip()
                    if result:
                        return result
            except Exception:
                pass
        return self._translate_single_google(text)

    def _translate_single_google(self, text: str) -> str:
        """Translate a single text with Google Translate."""
        from deep_translator import GoogleTranslator
        try:
            src = self.cfg.source_language if self.cfg.source_language != "auto" else "auto"
            translator = GoogleTranslator(source=src, target=self.cfg.target_language)
            return translator.translate(text) or text
        except Exception:
            return text

    @staticmethod
    def _parse_numbered_translations(text: str, expected_count: int) -> List[str]:
        """Parse numbered translation output from Gemini."""
        lines = text.strip().split("\n")
        translations = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Match: "1. translation" or "1) translation" or "1: translation" or "1- translation"
            match = re.match(r'\s*\d+[\.:\)\-]\s*(?:\[[^\]]*\]\s*)?(.*)', line)
            if match:
                trans = match.group(1).strip()
                translations.append(trans)
        # Pad with empty strings if Gemini returned fewer lines
        while len(translations) < expected_count:
            translations.append("")
        return translations[:expected_count]

    # ── Step 5: Continuous TTS ────────────────────────────────────────────
    def _tts_continuous(self, translated_text: str) -> Path:
        """Synthesize the entire translated narrative as ONE single TTS call."""
        import edge_tts

        out_mp3 = self.cfg.work_dir / "tts_full.mp3"
        out_wav = self.cfg.work_dir / "tts_full.wav"
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        self._report("synthesize", 0.1, "Generating speech (single voice)...")

        async def synthesize():
            communicate = edge_tts.Communicate(translated_text, voice, rate=rate)
            await communicate.save(str(out_mp3))

        asyncio.run(synthesize())

        if not out_mp3.exists() or out_mp3.stat().st_size == 0:
            raise RuntimeError("TTS synthesis produced no audio")

        self._report("synthesize", 0.8, "Converting to WAV...")

        # Convert to WAV
        self._run_proc(
            [
                self._ffmpeg, "-y", "-i", str(out_mp3),
                "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                str(out_wav),
            ],
            check=True, capture_output=True,
        )
        out_mp3.unlink(missing_ok=True)

        return out_wav

    # ── Step 5b: Time-aligned TTS ─────────────────────────────────────────
    def _tts_time_aligned(self, segments, total_duration, prefix="", progress_base=0.0, progress_span=1.0):
        """Natural-flow TTS: generate at natural speed, place at original timestamps.

        No speed manipulation — speech sounds completely natural.
        If a segment runs longer than its slot, it simply overlaps into the next gap.
        """
        import edge_tts
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        # Generate all TTS at natural rate
        async def tts_generate():
            for i, seg in enumerate(segments):
                text = self._prepare_tts_text(
                    seg.get("text_translated", seg.get("text", "")).strip()
                )
                if not text:
                    continue
                mp3 = self.cfg.work_dir / f"{prefix}seg_{i:04d}.mp3"
                try:
                    comm = edge_tts.Communicate(text, voice, rate=rate)
                    await comm.save(str(mp3))
                    seg["_tts_mp3"] = mp3
                except Exception:
                    pass

        asyncio.run(tts_generate())

        # Convert to WAV (no speed changes)
        tts_data = []
        for i, seg in enumerate(segments):
            mp3 = seg.pop("_tts_mp3", None)
            if not mp3 or not mp3.exists():
                continue

            wav = self.cfg.work_dir / f"{prefix}seg_{i:04d}.wav"
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(mp3),
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(wav)],
                check=True, capture_output=True,
            )
            mp3.unlink(missing_ok=True)
            self._enhance_tts_wav(wav)

            tts_dur = self._get_duration(wav)
            tts_data.append({
                "start": seg["start"],
                "wav": wav,
                "duration": tts_dur,
            })

        tts_data = self._truncate_overlaps(tts_data)
        return self._build_timeline(tts_data, total_duration, prefix)

    CROSSFADE_MS = 15  # 15ms crossfade at segment boundaries to prevent clicks

    # Minimum silence gap between sentences (seconds)
    SENTENCE_GAP = 1.0

    def _truncate_overlaps(self, tts_data):
        """REFLOW segments: push overlapping segments forward. NO speech is ever cut.

        Philosophy: every word must be heard completely. If a segment's audio
        runs longer than its time slot, we don't trim it — we push the NEXT
        segment later so there's no overlap, with a 1-second silence gap.

        Rules:
        1. NEVER cut, trim, or truncate any TTS audio
        2. If segment N overflows into segment N+1's time, push N+1 forward
        3. Maintain minimum 1.0s silence gap between all segments
        4. Audio timeline may become longer than original video
        5. Cascade: pushing one segment may push all subsequent ones

        Returns segments with updated start times (no WAV files modified).
        """
        if not tts_data or len(tts_data) < 2:
            return tts_data

        # Skip sentence gap enforcement when disabled
        if not getattr(self.cfg, 'enable_sentence_gap', True):
            return tts_data

        sorted_data = sorted(tts_data, key=lambda s: s.get("start", 0))
        pushed_count = 0

        for idx in range(1, len(sorted_data)):
            prev = sorted_data[idx - 1]
            curr = sorted_data[idx]

            prev_start = prev.get("start", 0)
            prev_dur = prev.get("duration", 0)
            prev_audio_end = prev_start + prev_dur  # when previous audio actually finishes

            curr_start = curr.get("start", 0)

            # Minimum allowed start = previous audio end + gap
            min_start = prev_audio_end + self.SENTENCE_GAP

            if curr_start < min_start:
                # Push this segment forward (don't cut previous)
                shift = min_start - curr_start
                curr["start"] = min_start
                if "end" in curr:
                    curr["end"] = curr["end"] + shift
                pushed_count += 1

        if pushed_count:
            last = sorted_data[-1]
            new_end = last["start"] + last.get("duration", 0) + 0.5
            print(f"[Reflow] Pushed {pushed_count}/{len(sorted_data)} segments forward "
                  f"(1s gaps, no speech cut). Timeline now {new_end:.1f}s", flush=True)

        return sorted_data

    def _build_timeline(self, tts_data, total_duration, prefix=""):
        """Place TTS segments at their reflowed timestamps on a silent audio track.

        Segments are pre-reflowed by _truncate_overlaps() so they don't overlap.
        Each segment plays at FULL duration — no cutting, no trimming.
        The timeline auto-extends if reflowed segments push past total_duration.
        """
        # Calculate actual timeline length (may be longer than video due to reflow)
        if tts_data:
            last = max(tts_data, key=lambda s: s.get("start", 0) + s.get("duration", 0))
            actual_end = last.get("start", 0) + last.get("duration", 0) + 0.5
            total_duration = max(total_duration, actual_end)

        total_samples = int((total_duration + 0.5) * self.SAMPLE_RATE)
        bytes_per_frame = 2 * self.N_CHANNELS  # 16-bit stereo = 4 bytes
        timeline = bytearray(total_samples * bytes_per_frame)

        for seg in tts_data:
            wav_path = seg.get("wav")
            if not wav_path or not Path(wav_path).exists():
                continue
            start_byte = int(seg["start"] * self.SAMPLE_RATE) * bytes_per_frame
            if start_byte < 0:
                start_byte = 0
            if start_byte >= len(timeline):
                continue

            try:
                with wave.open(str(wav_path), 'rb') as w:
                    if w.getnchannels() != self.N_CHANNELS or w.getsampwidth() != 2:
                        continue
                    raw = w.readframes(w.getnframes())
            except Exception:
                continue

            # Place FULL audio — extend timeline buffer if WAV is longer than metadata
            needed_end = start_byte + len(raw)
            if needed_end > len(timeline):
                timeline.extend(b'\x00' * (needed_end - len(timeline)))
            end_byte = start_byte + len(raw)
            copy_len = end_byte - start_byte
            if copy_len <= 0:
                continue

            # Overwrite (not mix) — single voice at any point
            timeline[start_byte:start_byte + copy_len] = raw[:copy_len]

        output = self.cfg.work_dir / f"{prefix}tts_aligned.wav"
        with wave.open(str(output), 'wb') as w:
            w.setnchannels(self.N_CHANNELS)
            w.setsampwidth(2)
            w.setframerate(self.SAMPLE_RATE)
            w.writeframes(bytes(timeline))

        return output

    # ── Speed-fit & no-cut assembly ─────────────────────────────────────

    # Speed limits: slow TTS → speed up to max 1.1x, long TTS → speed up to max 1.25x
    SPEED_MIN = 1.0 / 1.1    # 0.909 — slowest we allow (1.1x slower than natural)
    SPEED_MAX = 1.25          # fastest we allow (1.25x faster than natural)

    # Spec-aligned duration fitting thresholds (per production-grade dubbing spec)
    FIT_PASS_MS = 80          # ≤80ms error → accept as-is (no stretch)
    FIT_STRETCH_MS = 150      # ≤150ms → fine stretch within preferred ratio
    FIT_REWRITE_MS = 250      # >250ms → flag for rewrite (still stretch if within hard limits)
    FIT_RATIO_PREF_MIN = 0.94 # preferred soft stretch limit
    FIT_RATIO_PREF_MAX = 1.06 # preferred soft stretch limit

    def _apply_base_speedup(self, tts_data, speed=1.15):
        """Apply a uniform speedup to ALL TTS segments before assembly.

        This reduces total audio length by speeding up every segment by the
        given factor (default 1.15x). Keeps speech natural-sounding while
        reducing the amount of video slowdown needed.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def speedup_one(idx_tts):
            idx, tts = idx_tts
            wav_path = tts.get("wav")
            if not wav_path or not Path(wav_path).exists():
                return (idx, tts.copy())
            tts_dur = tts.get("duration", 0)
            if tts_dur < 0.1:
                return (idx, tts.copy())

            sped_wav = self.cfg.work_dir / f"basespeed_{idx:04d}.wav"
            try:
                self._run_proc(
                    [self._ffmpeg, "-y", "-i", str(wav_path),
                     "-af", f"atempo={speed}",
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     "-acodec", "pcm_s16le", str(sped_wav)],
                    check=True, capture_output=True,
                )
                new_dur = tts_dur / speed
                return (idx, {**tts, "wav": sped_wav, "duration": new_dur})
            except Exception:
                return (idx, tts.copy())

        # Use as_completed instead of pool.map so we can report progress every
        # ~25 segments. Without this the UI freezes for the entire batch.
        total = len(tts_data)
        results = []
        self._report("synthesize", 0.92, f"Applying {speed}x base speedup to {total} segments...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(speedup_one, item) for item in enumerate(tts_data)]
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % 25 == 0 or done == total:
                    self._report("synthesize", 0.92,
                                 f"[BaseSpeed] {done}/{total} segments sped up")

        results.sort(key=lambda x: x[0])
        print(f"[BaseSpeed] Applied {speed}x speedup to {len(results)} segments", flush=True)
        return [r[1] for r in results]

    def _speed_fit_segments(self, tts_data):
        """Speed-adjust each TTS segment to fit its time slot.

        Rules:
        - If TTS is shorter than slot → slow down up to 1.1x (SPEED_MIN atempo)
        - If TTS is longer than slot → speed up up to 1.25x (SPEED_MAX atempo)
        - NEVER cut or truncate audio — all speech must be heard completely
        - If TTS still doesn't fit after max speed-up, let it overflow into the gap

        Audio Priority mode: 1.15x base speedup is applied before this step.
        Speed-fit still runs to fine-tune each segment. Video slows to match if needed.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fit_one(idx_tts):
            idx, tts = idx_tts
            seg_start = tts["start"]
            seg_end = tts["end"]
            slot_dur = seg_end - seg_start
            tts_dur = tts["duration"]
            tts_wav = tts["wav"]
            _wc = int(tts.get("_expected_words", 0) or 0)

            if slot_dur < 0.1 or tts_dur < 0.1:
                # Long-segment trace: skipped (too short)
                if _wc > 0:
                    self._trace_record_speed_fit(idx, _wc, tts_dur, tts_dur, 1.0, False)
                return (idx, tts.copy())

            ratio = tts_dur / slot_dur  # >1 = TTS longer than slot

            # Close enough — no adjustment needed
            if abs(ratio - 1.0) < 0.05:
                if _wc > 0:
                    self._trace_record_speed_fit(idx, _wc, tts_dur, tts_dur, 1.0, False)
                return (idx, tts.copy())

            # Clamp ratio to our speed limits
            clamped_ratio = max(self.SPEED_MIN, min(ratio, self.SPEED_MAX))
            was_clamped = (clamped_ratio != ratio)

            stretched_wav = self.cfg.work_dir / f"speedfit_{idx:04d}.wav"
            self._time_stretch(tts_wav, clamped_ratio, stretched_wav)

            new_dur = tts_dur / clamped_ratio  # duration after speed change

            # Long-segment trace: record speed-fit outcome. If clamped, the
            # audio could not be stretched all the way to fit the slot, so
            # the resulting WAV may overflow the slot in assembly — this is
            # the most common cause of "TTS read it but final video missing words".
            if _wc > 0:
                self._trace_record_speed_fit(idx, _wc, tts_dur, new_dur, clamped_ratio, was_clamped)

            return (idx, {
                "start": seg_start,
                "end": seg_end,
                "wav": stretched_wav,
                "duration": new_dur,
            })

        # as_completed + per-25 progress so the UI doesn't freeze on long videos
        total = len(tts_data)
        results = []
        self._report("synthesize", 0.94, f"Speed-fitting {total} segments to time slots...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fit_one, item) for item in enumerate(tts_data)]
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % 25 == 0 or done == total:
                    self._report("synthesize", 0.94,
                                 f"[SpeedFit] {done}/{total} segments fitted")

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _build_timeline_no_cut(self, tts_data, total_duration, prefix=""):
        """Place TTS segments on a timeline WITHOUT cutting any audio.

        Uses FFmpeg adelay filter to place each segment at its correct time,
        then amix to combine. Handles arbitrarily long videos without loading
        the entire timeline into RAM (avoids OOM on 12h+ videos).
        """
        # Extend total_duration to fit all reflowed segments
        if tts_data:
            last = max(tts_data, key=lambda s: s.get("start", 0) + s.get("duration", 0))
            actual_end = last.get("start", 0) + last.get("duration", 0) + 0.5
            total_duration = max(total_duration, actual_end)

        output = self.cfg.work_dir / f"{prefix}tts_no_cut.wav"

        if not tts_data:
            # Generate silence for the full duration
            self._run_proc(
                [self._ffmpeg, "-y", "-f", "lavfi",
                 "-i", f"anullsrc=r={self.SAMPLE_RATE}:cl=stereo",
                 "-t", f"{total_duration:.3f}",
                 "-acodec", "pcm_s16le", str(output)],
                check=True, capture_output=True,
            )
            return output

        # For very large segment counts, process in chunks to avoid hitting
        # FFmpeg's filter complexity limits (~500 inputs max)
        CHUNK_SIZE = 200
        if len(tts_data) <= CHUNK_SIZE:
            self._build_timeline_chunk(tts_data, total_duration, output)
        else:
            # Build chunks, then merge
            chunk_paths = []
            for ci in range(0, len(tts_data), CHUNK_SIZE):
                chunk = tts_data[ci:ci + CHUNK_SIZE]
                chunk_out = self.cfg.work_dir / f"{prefix}timeline_chunk_{ci:04d}.wav"
                self._build_timeline_chunk(chunk, total_duration, chunk_out)
                chunk_paths.append(chunk_out)
                self._report("assemble",
                             0.2 + 0.3 * ((ci + CHUNK_SIZE) / len(tts_data)),
                             f"Built timeline chunk {len(chunk_paths)}/{math.ceil(len(tts_data)/CHUNK_SIZE)}...")

            # Merge chunks — ffmpeg amix can only handle ~10 inputs at a time on Windows
            if len(chunk_paths) == 1:
                shutil.move(str(chunk_paths[0]), str(output))
            else:
                # Merge in batches of 10 to avoid ffmpeg input limits
                MERGE_BATCH = 10
                current_paths = list(chunk_paths)
                merge_round = 0
                while len(current_paths) > 1:
                    next_paths = []
                    for bi in range(0, len(current_paths), MERGE_BATCH):
                        batch = current_paths[bi:bi + MERGE_BATCH]
                        if len(batch) == 1:
                            next_paths.append(batch[0])
                            continue
                        merge_out = self.cfg.work_dir / f"{prefix}merge_r{merge_round}_{bi:04d}.wav"
                        inputs = []
                        for cp in batch:
                            inputs.extend(["-i", str(cp)])
                        self._run_proc(
                            [self._ffmpeg, "-y"] + inputs + [
                                "-filter_complex",
                                f"amix=inputs={len(batch)}:duration=longest:normalize=0",
                                "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                                "-acodec", "pcm_s16le", str(merge_out)],
                            check=True, capture_output=True,
                        )
                        next_paths.append(merge_out)
                        # Clean up merged inputs (but not original chunks on first round)
                        if merge_round > 0:
                            for cp in batch:
                                cp.unlink(missing_ok=True)
                    current_paths = next_paths
                    merge_round += 1
                shutil.move(str(current_paths[0]), str(output))
                for cp in chunk_paths:
                    cp.unlink(missing_ok=True)

        return output

    def _build_timeline_chunk(self, tts_data, total_duration, output: Path):
        """Build a WAV timeline for a chunk of TTS segments using FFmpeg adelay + amix.

        Segments are pre-reflowed by _truncate_overlaps() so they don't overlap.
        Each segment plays at full duration — no trimming.
        """
        inputs = []
        filter_parts = []
        valid_idx = 0
        for seg in tts_data:
            wav_path = seg.get("wav")
            if not wav_path or not Path(wav_path).exists():
                continue
            delay_ms = max(0, int(seg.get("start", 0) * 1000))
            inputs.extend(["-i", str(wav_path)])
            idx = len(inputs) // 2 - 1
            filter_parts.append(
                f"[{idx}:a]adelay={delay_ms}|{delay_ms},apad[d{valid_idx}]"
            )
            valid_idx += 1

        if not filter_parts:
            self._run_proc(
                [self._ffmpeg, "-y", "-f", "lavfi",
                 "-i", f"anullsrc=r={self.SAMPLE_RATE}:cl=stereo",
                 "-t", f"{total_duration:.3f}",
                 "-acodec", "pcm_s16le", str(output)],
                check=True, capture_output=True,
            )
            return

        n = len(filter_parts)
        mix_inputs = "".join(f"[d{i}]" for i in range(n))
        filter_complex = ";".join(filter_parts) + f";{mix_inputs}amix=inputs={n}:duration=longest:normalize=0"

        self._run_proc(
            [self._ffmpeg, "-y"] + inputs + [
                "-filter_complex", filter_complex,
                "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                "-acodec", "pcm_s16le", str(output)],
            check=True, capture_output=True,
        )

    # ── Video-adapts-to-audio assembly ──────────────────────────────────
    # Maximum video slowdown: 1.1x (setpts=1.1*PTS makes it 10% slower)
    VIDEO_SLOW_MAX = 1.1

    def _assemble_video_adapts_to_audio(self, video_path, audio_raw, tts_data, total_video_duration):
        """Assemble dubbed video where AUDIO IS MASTER and video adapts per-segment.
        """
        print("[Assembly] Using: _assemble_video_adapts_to_audio (per-segment sync)", flush=True)

        # Apply 1.15x base speedup + duration fitting (unless untouchable or
        # disabled). The new tts_no_time_pressure flag (default ON) is the
        # MASTER off-switch — when set, NO speed manipulation is applied to
        # TTS output regardless of the other flags. The user explicitly asked
        # for this: TTS produces every word at natural pace, assembly handles
        # any slot overflow via the audio_priority path.
        no_pressure = getattr(self.cfg, 'tts_no_time_pressure', True)
        if not no_pressure and not self.cfg.audio_untouchable and getattr(self.cfg, 'enable_duration_fit', True):
            self._report("assemble", 0.03, "Applying 1.15x base speedup...")
            tts_data = self._apply_base_speedup(tts_data, 1.15)
            # Speed-fit to slots
            tts_data = self._speed_fit_segments(tts_data)
        elif no_pressure:
            self._report("assemble", 0.03,
                         "tts_no_time_pressure ON — skipping base speedup + speed-fit "
                         "(natural pace, assembly will adapt video to audio)")
            print("[NO-TIME-PRESSURE] Skipped _apply_base_speedup and _speed_fit_segments — "
                  "TTS audio will play at full natural pace, assembly handles overflow",
                  flush=True)

        """
        Per-segment logic:

        ── PURE STRETCH MODE (no_pressure=True, default per user request) ──
        1. TTS audio plays at NATURAL speed — never sped up, slowed, or cut.
        2. Source video clip [start..end] is stretched (or compressed) by
           setpts={tts_dur/slot_dur}*PTS so its final duration equals the
           audio's natural length. NO upper or lower cap on the stretch ratio.
           Heavy stretches (>2x) just look like slow motion — accepted because
           the user explicitly chose audio fidelity over video pacing.
        3. NO LOOPING. NO FREEZING. NO AUDIO MANIPULATION.
        4. When audio is shorter than the slot, the video is sped up so the
           segment ends exactly when audio ends, then the next segment starts.

        ── LEGACY MODE (no_pressure=False) ──
        1. Audio is sped up AND video is slowed down (split the difference)
        2. Caps: audio 1.50x max, video 2.00x max
        3. If still doesn't fit: video freezes last frame while audio overflows

        ── Both modes ──
        Gaps between speech segments play at normal video speed.
        Result: dubbed video with the audio track fully intact.
        """
        # ══════════════════════════════════════════════════════════════════
        # ── ASSEMBLY MANAGER: Verify + fix timeline before building video ──
        # ══════════════════════════════════════════════════════════════════

        # 1. SORT: Guarantee original video order
        tts_data = sorted(tts_data, key=lambda t: t.get("start", 0))

        # 2. DEDUPLICATE: Remove segments with identical start times (keep longer audio)
        seen_starts = {}  # key -> index in deduped list
        deduped = []
        for tts in tts_data:
            key = round(tts["start"], 2)
            if key in seen_starts:
                old_idx = seen_starts[key]
                if tts.get("duration", 0) > deduped[old_idx].get("duration", 0):
                    deduped[old_idx] = tts  # replace in-place, O(1)
            else:
                seen_starts[key] = len(deduped)
                deduped.append(tts)
        if len(deduped) < len(tts_data):
            print(f"[Assembly Manager] Removed {len(tts_data) - len(deduped)} duplicate segments", flush=True)
        tts_data = deduped

        # 3. VALIDATE: Check each segment has a valid WAV file
        validated = []
        missing_audio = 0
        for tts in tts_data:
            wav = tts.get("wav")
            if not wav or not Path(wav).exists():
                missing_audio += 1
                continue
            if Path(wav).stat().st_size < 500:
                missing_audio += 1
                continue
            # Re-probe actual duration (don't trust cached value)
            actual_dur = self._get_duration(wav)
            if actual_dur < 0.1:
                missing_audio += 1
                continue
            tts["duration"] = actual_dur
            validated.append(tts)

        if missing_audio:
            total_input = len(deduped)
            drop_pct = (missing_audio / max(total_input, 1)) * 100
            print(f"[Assembly Manager] WARNING: {missing_audio}/{total_input} segments "
                  f"({drop_pct:.1f}%) have missing/corrupt audio", flush=True)
            if drop_pct > 20:
                print(f"[Assembly Manager] CRITICAL: >20% segments missing audio — "
                      f"this indicates a TTS failure. Check TTS-SWEEP logs.", flush=True)
        tts_data = validated

        # 4. FIX OVERLAPS: Cascade — keep pushing forward until no overlaps remain
        changed = True
        passes = 0
        while changed and passes < 50:  # safety limit
            changed = False
            passes += 1
            for i in range(1, len(tts_data)):
                prev_end = tts_data[i-1]["start"] + tts_data[i-1]["duration"]
                if tts_data[i]["start"] < prev_end + 0.01:
                    new_start = prev_end + 0.05
                    shift = new_start - tts_data[i]["start"]
                    tts_data[i]["start"] = new_start
                    tts_data[i]["end"] = tts_data[i].get("end", new_start) + shift
                    changed = True
        if passes > 1:
            print(f"[Assembly Manager] Overlap cascade: {passes} passes to resolve", flush=True)

        # 5. VERIFY: Final order check
        order_ok = all(
            tts_data[i]["start"] >= tts_data[i-1]["start"] - 0.01
            for i in range(1, len(tts_data))
        )

        total_audio = sum(t["duration"] for t in tts_data)
        if tts_data:
            print(f"[Assembly Manager] Timeline verified: {len(tts_data)} segments, "
                  f"{total_audio:.0f}s audio, "
                  f"range {tts_data[0]['start']:.1f}s-{tts_data[-1].get('end', 0):.1f}s, "
                  f"order={'OK' if order_ok else 'FIXED'}",
                  flush=True)
        else:
            print("[Assembly Manager] WARNING: No TTS segments available", flush=True)

        # ── Compute proportional shrink ratio for duration matching ──
        # Some segments have TTS audio longer than their slot (overflow) and
        # some have audio shorter (slack). To match the source duration, we
        # need the total overflow to be absorbed by the total slack.
        #
        # _shrink_ratio tells the section builder how much slack from short
        # segments should be consumed to compensate for overflow from long ones:
        #   0.0 → no overflow, short segments keep full slot duration
        #   0.5 → absorb half the slack from each short segment
        #   1.0 → absorb all slack → output exactly matches source
        #   >1.0 → clamped to 1.0 (can't fully compensate; output > source)
        #
        # This produces output ≈ source_duration regardless of per-segment
        # variation, without touching the audio at all.
        _total_overflow = 0.0
        _total_slack = 0.0
        for _t in tts_data:
            _sd = _t.get("end", 0) - _t.get("start", 0)
            _td = _t.get("duration", 0)
            if _sd > 0.1 and _td > _sd * 1.05:
                _total_overflow += (_td - _sd)
            elif _sd > 0.1 and _td < _sd * 0.95:
                _total_slack += (_sd - _td)
        _shrink_ratio = min(1.0, _total_overflow / _total_slack) if _total_slack > 0 else 0.0
        print(f"[Assembly] Duration balancing: overflow={_total_overflow:.1f}s, "
              f"slack={_total_slack:.1f}s, shrink_ratio={_shrink_ratio:.3f} "
              f"({'exact match' if _shrink_ratio <= 1.0 else 'overflow exceeds slack'})",
              flush=True)

        sections = []
        current_pos = 0.0
        audio_pos = 0.0  # running position in the output audio timeline

        # Pre-compute gap behavior from cfg.gap_mode:
        #   "none"  → 0s, pure back-to-back (shortest output)
        #   "micro" → 0.2s breathing room (proven default)
        #   "full"  → keep original silence durations (legacy)
        # Fallback: if gap_mode not set, use auto-detect from rate/pressure settings
        _gap_mode = getattr(self.cfg, 'gap_mode', None)
        if _gap_mode == "none":
            _minimize_gaps = True
            _micro_gap_sec = 0.0
        elif _gap_mode == "full":
            _minimize_gaps = False
            _micro_gap_sec = 0.0
        elif _gap_mode == "micro":
            _minimize_gaps = True
            _micro_gap_sec = 0.2
        else:
            # Auto-detect (legacy): minimize when auto rate + no_time_pressure
            _no_pressure_asm = getattr(self.cfg, 'tts_no_time_pressure', True)
            _auto_mode_asm = (getattr(self.cfg, "tts_rate_mode", "auto") or "auto").lower() == "auto"
            _minimize_gaps = _no_pressure_asm and _auto_mode_asm
            _micro_gap_sec = 0.2
        _gap_label = {"none": "no gaps (back-to-back)", "micro": "micro-gaps (0.2s)",
                      "full": "full original gaps"}.get(_gap_mode or "", "auto-detect")
        print(f"[Assembly] Gap mode: {_gap_label}", flush=True)

        for tts in tts_data:
            seg_start = tts["start"]
            seg_end = tts["end"]
            tts_dur = tts["duration"]
            slot_dur = seg_end - seg_start

            # Gap before this segment.
            # Under auto rate mode + no_time_pressure: SKIP gaps entirely
            # so segments play back-to-back. The gap time between segments
            # was adding ~20-30 seconds to the output (silence, scene
            # transitions, pauses) which made the dubbed video longer than
            # the source. By removing gaps, the output duration is driven
            # purely by the speech segments, which the balanced proportional
            # approach already targets to match the source.
            #
            # Under manual mode: keep gaps at original duration (legacy behavior).
            if seg_start > current_pos + 0.05:
                gap_dur = seg_start - current_pos
                if _minimize_gaps:
                    # Replace full gap with a tiny micro-gap for natural pacing.
                    # IMPORTANT: clamp to min(micro_gap, original_gap) so we
                    # never ADD time that wasn't in the original. If the
                    # original gap was 0.05s and micro_gap is 0.2s, use 0.05s.
                    if _micro_gap_sec > 0:
                        actual_gap = min(_micro_gap_sec, gap_dur)
                        sections.append({
                            "type": "gap",
                            "video_start": current_pos,
                            "video_end": min(current_pos + actual_gap, seg_start),
                            "output_dur": actual_gap,
                        })
                        audio_pos += actual_gap
                else:
                    sections.append({
                        "type": "gap",
                        "video_start": current_pos,
                        "video_end": seg_start,
                        "output_dur": gap_dur,
                    })
                    audio_pos += gap_dur

            # Speech segment — PURE VIDEO STRETCH (no audio touching).
            # Per user request: audio is sacred, video stretches as much as
            # needed to last as long as the audio. No "split the difference"
            # speedup of audio. No looping. Just setpts on the video clip.
            #
            # The legacy "split the difference" path is kept as a fallback for
            # when tts_no_time_pressure is OFF (the old behavior).
            no_pressure = getattr(self.cfg, 'tts_no_time_pressure', True)
            ratio = (tts_dur / slot_dur) if slot_dur > 0.1 else 1.0
            freeze = False
            freeze_target_dur = 0.0

            if no_pressure:
                # ── BALANCED PATH: proportional slot compensation ──
                #
                # Goal: output duration ≈ source duration. Audio is never
                # touched. Video stretches for long segments, compensates
                # for short segments.
                #
                # The _shrink_ratio (computed before this loop at the top of
                # the assembly method) tells us how much slack from short
                # segments should be absorbed to compensate for overflow
                # from long segments:
                #   0.0 = no overflow → short segments keep full slot
                #   0.5 = half the slack is used → partial compression
                #   1.0 = all slack used → exact source duration match
                #   >1.0 = clamped to 1.0 → can't fully compensate,
                #          output will be slightly longer than source
                #
                if slot_dur > 0.1 and tts_dur > slot_dur * 1.05:
                    # Audio LONGER than slot → stretch video to fit audio
                    pts_factor = tts_dur / slot_dur
                    meet_dur = tts_dur
                elif slot_dur > 0.1 and tts_dur < slot_dur * 0.95:
                    # Audio SHORTER than slot → absorb SOME of the slack
                    # to compensate for overflow from long segments.
                    slack = slot_dur - tts_dur
                    absorbed = slack * _shrink_ratio
                    actual_dur = slot_dur - absorbed
                    # Don't shrink below the audio duration (would cut audio)
                    actual_dur = max(actual_dur, tts_dur)
                    pts_factor = actual_dur / slot_dur if slot_dur > 0 else 1.0
                    meet_dur = actual_dur
                else:
                    # Audio ≈ slot (within 5%) — play at natural pace
                    pts_factor = 1.0
                    meet_dur = slot_dur

                if pts_factor > 1.5:
                    print(f"[Assembly] Seg at {seg_start:.1f}s: video stretched "
                          f"{pts_factor:.2f}x to fit {tts_dur:.1f}s audio "
                          f"in {slot_dur:.1f}s slot", flush=True)
            else:
                # ── LEGACY PATH: split-the-difference (audio compress + video slow) ──
                AUDIO_SPEED_MAX = 1.50
                VIDEO_SLOW_MAX = 2.00

                if slot_dur > 0.1 and tts_dur > slot_dur * 1.05:
                    # Split evenly using square root, capped at limits
                    import math
                    sqrt_ratio = math.sqrt(ratio)
                    audio_speed = min(sqrt_ratio, AUDIO_SPEED_MAX)
                    video_slow = ratio / audio_speed
                    if video_slow > VIDEO_SLOW_MAX:
                        video_slow = VIDEO_SLOW_MAX
                        audio_speed = min(ratio / video_slow, AUDIO_SPEED_MAX)

                    # If STILL exceeds both limits — let audio overflow (video freezes last frame)
                    max_combined = AUDIO_SPEED_MAX * VIDEO_SLOW_MAX
                    if ratio > max_combined:
                        audio_speed = AUDIO_SPEED_MAX
                        video_slow = VIDEO_SLOW_MAX
                        freeze = True
                        print(f"[Assembly] Seg at {seg_start:.1f}s: ratio {ratio:.2f}x exceeds "
                              f"max {max_combined:.2f}x — audio will overflow", flush=True)

                    # Speed up the audio WAV file
                    if audio_speed > 1.03:
                        sped_wav = self.cfg.work_dir / f"sped_{hash(str(tts['wav'])) & 0xFFFFFFFF:08x}.wav"
                        self._time_stretch(tts["wav"], audio_speed, sped_wav)
                        tts["wav"] = sped_wav
                        tts_dur = tts_dur / audio_speed

                    pts_factor = video_slow
                    meet_dur = max(tts_dur, slot_dur * video_slow)
                    freeze_target_dur = meet_dur if freeze else 0.0
                else:
                    pts_factor = 1.0
                    meet_dur = tts_dur

            sections.append({
                "type": "speech",
                "video_start": seg_start,
                "video_end": seg_end,
                "pts_factor": pts_factor,
                "freeze": freeze,
                "freeze_target_dur": freeze_target_dur,
                "tts_wav": tts["wav"],
                "tts_dur": tts_dur,
                "output_dur": meet_dur,
            })

            audio_pos += sections[-1]["output_dur"]
            current_pos = max(current_pos, seg_end)  # Never move backward for overlapping segments

        # Trailing gap — minimize in auto mode (same as inter-segment gaps above)
        if current_pos < total_video_duration - 0.05 and not _minimize_gaps:
            gap_dur = total_video_duration - current_pos
            sections.append({
                "type": "gap",
                "video_start": current_pos,
                "video_end": total_video_duration,
                "output_dur": gap_dur,
            })
            audio_pos += gap_dur

        # Build video clips for each section — PARALLEL with NVENC GPU encoding
        num_sections = len(sections)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # NVENC can handle multiple concurrent sessions on modern GPUs (RTX series)
        # RTX 3060+ supports up to 8 concurrent sessions
        nvenc_ok = self._check_nvenc()
        max_workers = 8 if nvenc_ok else 6  # NVENC: 8 sessions, CPU: 6 threads
        encode_args = self._video_encode_args(force_cpu=False)  # NVENC when available
        encode_args_cpu = self._video_encode_args(force_cpu=True)  # CPU for freeze frames
        ffmpeg = self._ffmpeg
        work_dir = self.cfg.work_dir
        vp = str(video_path)
        slow_max = self.VIDEO_SLOW_MAX
        completed_count = [0]

        def build_section(idx_sec):
            idx, sec = idx_sec
            clip = work_dir / f"adapt_{idx:04d}.mp4"
            vs = sec["video_start"]
            ve = sec["video_end"]
            dur = ve - vs

            if dur < 0.05:
                return None

            if sec["type"] == "gap":
                self._run_proc(
                    [ffmpeg, "-y",
                     "-ss", f"{vs:.3f}", "-i", vp,
                     "-t", f"{dur:.3f}", "-an",
                     *encode_args, str(clip)],
                    check=True, capture_output=True,
                )
                return (idx, clip)

            elif sec["type"] == "speech":
                pts_factor = sec["pts_factor"]
                # Output duration = slot × pts_factor (slowed video fills TTS duration)
                output_dur = dur * pts_factor

                if not sec.get("freeze", False):
                    if abs(pts_factor - 1.0) < 0.03:
                        # Normal speed — no filter needed
                        self._run_proc(
                            [ffmpeg, "-y",
                             "-ss", f"{vs:.3f}", "-i", vp,
                             "-t", f"{dur:.3f}", "-an",
                             *encode_args, str(clip)],
                            check=True, capture_output=True,
                        )
                    else:
                        # Slowed video: setpts makes frames play slower.
                        # Input: extract `dur` seconds from source.
                        # setpts stretches it to `dur * pts_factor` seconds.
                        # ffmpeg -t BEFORE -i = input limit, AFTER = output limit.
                        # We need output to be full `output_dur` — so NO output -t limit.
                        self._run_proc(
                            [ffmpeg, "-y",
                             "-ss", f"{vs:.3f}",
                             "-t", f"{dur:.3f}",        # input: extract exactly slot duration
                             "-i", vp,
                             "-filter:v", f"setpts={pts_factor:.6f}*PTS",
                             "-an", *encode_args,
                             str(clip)],                 # output: full slowed duration
                            check=True, capture_output=True,
                        )
                    return (idx, clip)

                else:
                    # Freeze: slow scene + freeze last frame
                    target_dur = sec["freeze_target_dur"]
                    slowed_clip = work_dir / f"adapt_{idx:04d}_slow.mp4"
                    self._run_proc(
                        [ffmpeg, "-y",
                         "-ss", f"{vs:.3f}",
                         "-t", f"{dur:.3f}",        # input limit
                         "-i", vp,
                         "-filter:v", f"setpts={slow_max:.6f}*PTS",
                         "-an", *encode_args, str(slowed_clip)],  # no output limit
                        check=True, capture_output=True,
                    )
                    slowed_dur = dur * slow_max

                    if slowed_dur >= target_dur:
                        return (idx, slowed_clip)

                    freeze_dur = target_dur - slowed_dur
                    last_frame = work_dir / f"adapt_{idx:04d}_lastframe.png"
                    frame_ok = self._run_proc(
                        [ffmpeg, "-y",
                         "-sseof", "-0.1", "-i", str(slowed_clip),
                         "-frames:v", "1", "-update", "1", str(last_frame)],
                        capture_output=True,
                    )
                    if frame_ok.returncode != 0 or not last_frame.exists():
                        self._run_proc(
                            [ffmpeg, "-y", "-i", str(slowed_clip),
                             "-frames:v", "1", "-update", "1", str(last_frame)],
                            capture_output=True,
                        )
                    if not last_frame.exists():
                        return (idx, slowed_clip)

                    # Still image → CPU encode (NVENC chokes on image input)
                    freeze_clip = work_dir / f"adapt_{idx:04d}_freeze.mp4"
                    self._run_proc(
                        [ffmpeg, "-y",
                         "-loop", "1", "-i", str(last_frame),
                         "-t", f"{freeze_dur:.3f}",
                         "-vf", "fps=30",
                         *encode_args_cpu, "-pix_fmt", "yuv420p",
                         str(freeze_clip)],
                        check=True, capture_output=True,
                    )
                    concat_list = work_dir / f"adapt_{idx:04d}_concat.txt"
                    concat_list.write_text(
                        f"file '{str(slowed_clip).replace(chr(92), '/')}'\n"
                        f"file '{str(freeze_clip).replace(chr(92), '/')}'\n"
                    )
                    self._run_proc(
                        [ffmpeg, "-y",
                         "-f", "concat", "-safe", "0",
                         "-i", str(concat_list),
                         *encode_args, str(clip)],
                        check=True, capture_output=True,
                    )
                    return (idx, clip)

            return None

        encoder_name = "NVENC (GPU)" if nvenc_ok else f"libx264 ({self.cfg.encode_preset})"
        self._report("assemble", 0.15,
                     f"Building {num_sections} sections (x{max_workers} parallel, {encoder_name})...")

        clip_results = {}  # idx -> clip_path
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(build_section, (idx, sec)): idx
                       for idx, sec in enumerate(sections)}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    sec_idx, clip_path = result
                    clip_results[sec_idx] = clip_path
                completed_count[0] += 1
                if completed_count[0] % max(max_workers, 5) == 0:
                    self._report("assemble",
                                 0.15 + 0.50 * (completed_count[0] / num_sections),
                                 f"Built {completed_count[0]}/{num_sections} sections...")

        # Sort clips by section index to maintain original order
        clip_paths = [clip_results[i] for i in sorted(clip_results.keys())]
        clip_section_indices = sorted(clip_results.keys())

        if not clip_paths:
            raise RuntimeError("No video sections produced")

        # Concatenate all video clips
        self._report("assemble", 0.70, "Joining video sections...")
        adapted_video = self.cfg.work_dir / "video_adapted.mp4"
        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], adapted_video)
        else:
            self._concatenate_videos(clip_paths, adapted_video)

        # Probe actual durations of each video clip — parallel for speed
        self._report("assemble", 0.72, "Measuring actual clip durations (parallel)...")
        actual_durations = {}

        def probe_dur(clip_sec):
            cp, si = clip_sec
            d = self._get_duration(cp)
            return (si, d) if d > 0 else None

        with ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(probe_dur, zip(clip_paths, clip_section_indices)):
                if result:
                    actual_durations[result[0]] = result[1]

        # Build the TTS audio timeline using ACTUAL video clip durations (not theoretical)
        self._report("assemble", 0.80, "Building audio timeline...")
        audio_timeline_pos = 0.0
        audio_segments = []
        produced_sections = set(clip_section_indices)
        for idx, sec in enumerate(sections):
            if idx not in produced_sections:
                continue
            real_dur = actual_durations.get(idx, sec["output_dur"])
            if sec["type"] == "speech":
                # Verify WAV exists before adding to timeline
                wav_path = sec.get("tts_wav")
                if wav_path and Path(wav_path).exists() and Path(wav_path).stat().st_size > 500:
                    audio_segments.append({
                        "start": audio_timeline_pos,
                        "wav": wav_path,
                        "duration": sec["tts_dur"],
                    })
                else:
                    print(f"[Assembly Manager] Section {idx}: WAV missing/corrupt, "
                          f"gap will be silent", flush=True)
            audio_timeline_pos += real_dur

        # Verify audio timeline integrity
        total_output_dur = audio_timeline_pos
        for i in range(1, len(audio_segments)):
            prev_end = audio_segments[i-1]["start"] + audio_segments[i-1]["duration"]
            curr_start = audio_segments[i]["start"]
            if curr_start < prev_end - 0.01:
                print(f"[Assembly Manager] Audio overlap at {curr_start:.1f}s "
                      f"(prev ends {prev_end:.1f}s) — will be reflowed", flush=True)

        print(f"[Assembly Manager] Audio timeline: {len(audio_segments)} segments "
              f"placed across {total_output_dur:.0f}s video", flush=True)

        audio_segments = self._truncate_overlaps(audio_segments)
        tts_audio = self._build_timeline(audio_segments, total_output_dur, prefix="adapted_")

        # Mix original audio at low volume if requested
        if self.cfg.mix_original:
            self._report("assemble", 0.85, "Mixing original audio...")
            tts_audio = self._mix_audio(audio_raw, tts_audio, self.cfg.original_volume)

        # Mux adapted video + natural TTS audio
        self._report("assemble", 0.90, "Muxing final video...")
        self._mux_replace_audio(adapted_video, tts_audio, self.cfg.output_path)

    # ── Natural TTS + Video sync ────────────────────────────────────────
    # Languages that Chatterbox TTS can pronounce well (English only for now)
    CHATTERBOX_SUPPORTED_LANGS = {"en"}
    # Languages supported by Chatterbox Multilingual
    CHATTERBOX_MTL_LANGS = {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
        "sw", "tr", "zh",
    }

    def _split_segments_at_sentences(self, segments):
        """Split multi-sentence segments so each sentence is its own segment.

        This ensures the 1s gap in assembly applies PER SENTENCE, not just per segment.
        A segment like "यह अच्छा है। अब आगे बढ़ते हैं।" becomes two segments.
        """
        split = []
        for seg in segments:
            text = seg.get("text_translated", seg.get("text", "")).strip()
            if not text:
                split.append(seg)
                continue

            # Split at sentence-ending punctuation
            sentences = re.split(r'(?<=[.!?।॥])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 1:
                split.append(seg)
                continue

            # Distribute time proportionally by character count
            total_chars = sum(len(s) for s in sentences)
            seg_start = seg.get("start", 0)
            seg_dur = seg.get("end", seg_start) - seg_start
            pos = seg_start

            for sent in sentences:
                frac = len(sent) / total_chars if total_chars > 0 else 1.0 / len(sentences)
                sent_dur = seg_dur * frac
                new_seg = dict(seg)
                new_seg["text_translated"] = sent
                new_seg["text"] = sent
                new_seg["start"] = pos
                new_seg["end"] = pos + sent_dur
                split.append(new_seg)
                pos += sent_dur

        if len(split) > len(segments):
            print(f"[SentenceSplit] {len(segments)} segments -> {len(split)} "
                  f"(split multi-sentence segments for 1s per-sentence gaps)", flush=True)
        return split

    def _generate_tts_natural(self, segments):
        """Generate TTS at natural speed. Uses first enabled engine in priority order.

        English target priority (zero-spend):
            Chatterbox-Turbo → Chatterbox Multilingual → XTTS v2 → Edge-TTS

        Non-English target priority:
            CosyVoice 2 → Chatterbox Multilingual → ElevenLabs → XTTS v2 → Edge-TTS
        """
        # Split multi-sentence segments so each sentence gets its own 1s gap
        segments = self._split_segments_at_sentences(segments)

        # Store the split segments so callers can update text_segments
        # to match the expanded list (fixes _tts_manager index mismatch)
        self._split_tts_segments = segments

        # Tag every segment with a unique index — survives TTS engine shuffling
        for i, seg in enumerate(segments):
            seg["_seg_idx"] = i
            # Store the translated text that SHOULD be in the audio
            seg["_expected_text"] = seg.get("text_translated", seg.get("text", ""))

        target = self.cfg.target_language
        is_english = target == "en" or target.startswith("en-")

        # ── English dubbing: zero-spend local stack ──────────────────────
        if is_english:
            return self._generate_tts_english(segments)

        # ── Non-English dubbing ──────────────────────────────────────────
        # QUAD PARALLEL: CosyVoice (10% complex) + XTTS (GPU) + Sarvam (cloud) + Edge (fallback)
        # CosyVoice is handled INSIDE _tts_triple_parallel (not standalone anymore)
        # Triggers when at least Sarvam or XTTS is enabled + Edge as fallback
        sarvam_key = get_sarvam_key()
        sarvam_ready = bool(sarvam_key) and self.cfg.use_sarvam_bulbul
        any_primary = sarvam_ready or self.cfg.use_coqui_xtts or self.cfg.use_cosyvoice
        if any_primary and self.cfg.use_edge_tts:
            try:
                engines = []
                if self.cfg.use_cosyvoice: engines.append("CosyVoice (10%)")
                if self.cfg.use_coqui_xtts: engines.append("XTTS (GPU)")
                if sarvam_ready: engines.append("Sarvam (cloud)")
                engines.append("Edge (fallback)")
                self._report("synthesize", 0.02,
                             f"PARALLEL TTS: {' + '.join(engines)}...")
                return self._tts_triple_parallel(segments)
            except Exception as e:
                self._report("synthesize", 0.05,
                             f"Parallel TTS failed ({e}) — falling back to single engine...")

        # Standalone fallback priority: CosyVoice → Chatterbox → ElevenLabs → XTTS → Google → Edge
        if self.cfg.use_cosyvoice:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("No CUDA GPU available")
                self._report("synthesize", 0.05, "Using CosyVoice 2 standalone (GPU, all segments)...")
                return self._tts_cosyvoice2(segments)
            except Exception as e:
                self._report("synthesize", 0.05,
                             f"CosyVoice 2 failed ({e}) — falling back...")

        if self.cfg.use_chatterbox:
            if target.split("-")[0] in self.CHATTERBOX_MTL_LANGS:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("No CUDA GPU available")
                    self._report("synthesize", 0.05, "Using Chatterbox Multilingual (GPU, voice cloning)...")
                    return self._tts_chatterbox_multilingual(segments)
                except Exception as e:
                    self._report("synthesize", 0.05,
                                 f"Chatterbox Multilingual failed ({e}) — falling back...")
            else:
                target_name = LANGUAGE_NAMES.get(target, target)
                self._report("synthesize", 0.05,
                             f"Chatterbox doesn't support {target_name} — trying next engine...")

        if self.cfg.use_sarvam_bulbul:
            target_base = target.split("-")[0]
            if target_base in self.SARVAM_SUPPORTED_LANGS:
                sarvam_key = get_sarvam_key()
                if sarvam_key:
                    try:
                        self._report("synthesize", 0.05, "Using Sarvam Bulbul v3 (best Hindi, cloud API)...")
                        return self._tts_sarvam_bulbul(segments)
                    except Exception as e:
                        self._report("synthesize", 0.05,
                                     f"Sarvam Bulbul failed ({e}) — falling back...")
                else:
                    self._report("synthesize", 0.05,
                                 "Sarvam Bulbul enabled but no SARVAM_API_KEY in .env — falling back...")
            else:
                target_name = LANGUAGE_NAMES.get(target, target)
                self._report("synthesize", 0.05,
                             f"Sarvam Bulbul doesn't support {target_name} — trying next engine...")

        if self.cfg.use_elevenlabs:
            elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
            if elevenlabs_key:
                self._report("synthesize", 0.05, "Using ElevenLabs for human-like voice...")
                return self._tts_elevenlabs(segments, elevenlabs_key)
            if not self.cfg.use_chatterbox:
                raise RuntimeError("ElevenLabs enabled but ELEVENLABS_API_KEY not set in .env")

        # HYBRID MODE: Coqui XTTS (GPU) + Edge-TTS (cloud) in parallel
        if self.cfg.use_coqui_xtts and self.cfg.use_edge_tts:
            try:
                import torch
                if torch.cuda.is_available():
                    self._report("synthesize", 0.05,
                                 "HYBRID: Coqui XTTS (GPU) + Edge-TTS (cloud) in parallel...")
                    return self._tts_hybrid_coqui_edge(segments)
            except ImportError:
                pass
            except Exception as e:
                self._report("synthesize", 0.05,
                             f"Hybrid TTS failed ({e}) — trying single engine...")

        if self.cfg.use_coqui_xtts:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("No CUDA GPU available for XTTS")
                self._report("synthesize", 0.05, "Using Coqui XTTS v2 (GPU, voice cloning)...")
                return self._tts_coqui_xtts(segments)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._report("synthesize", 0.05,
                             f"Coqui XTTS failed ({e}) — falling back...")

        if self.cfg.use_google_tts:
            google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            if google_creds or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip():
                self._report("synthesize", 0.05, "Using Google Cloud TTS (WaveNet/Neural2)...")
                try:
                    return self._tts_google_cloud(segments)
                except Exception as e:
                    self._report("synthesize", 0.05,
                                 f"Google Cloud TTS failed ({e}) — falling back to Edge-TTS...")
            else:
                self._report("synthesize", 0.05,
                             "Google Cloud TTS enabled but no credentials found — falling back to Edge-TTS...")

        # Fall through to Edge-TTS (supports 70+ languages with native voices)
        voice = self.cfg.tts_voice
        target_name = LANGUAGE_NAMES.get(target, target)
        if self._voice_map:
            voices_used = len(set(self._voice_map.values()))
            self._report("synthesize", 0.05,
                         f"Using Edge-TTS with {voices_used} distinct voices for {target_name}...")
        else:
            self._report("synthesize", 0.05, f"Using Edge-TTS ({voice}) for {target_name}...")
        return self._tts_edge(segments, voice_map=self._voice_map)

    def _generate_tts_english(self, segments):
        """Zero-spend English dubbing TTS stack.

        Priority: Chatterbox-Turbo → Chatterbox Multilingual → XTTS v2 → Edge-TTS
        All local, all free, no API keys needed.
        """
        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            pass

        # 1. Chatterbox-Turbo (350M, English-focused, lowest VRAM)
        if has_gpu:
            try:
                self._report("synthesize", 0.05,
                             "English dub: Chatterbox-Turbo (GPU, 350M, voice cloning)...")
                return self._tts_chatterbox_turbo(segments)
            except Exception as e:
                self._report("synthesize", 0.05,
                             f"Chatterbox-Turbo failed ({e}) — trying Multilingual...")

        # 2. Chatterbox Multilingual (broader model, still free)
        if has_gpu:
            try:
                self._report("synthesize", 0.05,
                             "English dub: Chatterbox Multilingual (GPU, voice cloning)...")
                return self._tts_chatterbox_multilingual(segments)
            except Exception as e:
                self._report("synthesize", 0.05,
                             f"Chatterbox Multilingual failed ({e}) — trying XTTS v2...")

        # 3. XTTS v2 fallback (voice cloning, 16 langs — always tried for English)
        if has_gpu:
            try:
                self._report("synthesize", 0.05,
                             "English dub: XTTS v2 (GPU, voice cloning fallback)...")
                return self._tts_coqui_xtts(segments)
            except Exception as e:
                self._report("synthesize", 0.05,
                             f"XTTS v2 failed ({e}) — falling back to Edge-TTS...")

        # 4. Edge-TTS (cloud, no GPU needed)
        self._report("synthesize", 0.05, "English dub: Edge-TTS (cloud fallback)...")
        return self._tts_edge(segments, voice_map=self._voice_map)

    def _tts_hybrid_coqui_edge(self, segments):
        """HYBRID: Split segments between Coqui XTTS (GPU) and Edge-TTS (cloud).

        Coqui runs on GPU (sequential but high quality voice cloning).
        Edge-TTS runs on cloud (highly parallel, native voices).
        Both run simultaneously in separate threads = ~2x faster.
        """
        import threading

        # Split: even-indexed segments → Coqui XTTS, odd-indexed → Edge-TTS
        coqui_segments = [(i, seg) for i, seg in enumerate(segments) if i % 2 == 0]
        edge_segments = [(i, seg) for i, seg in enumerate(segments) if i % 2 == 1]

        coqui_results = {}
        edge_results = {}
        errors = []

        # Separate work dirs to prevent filename collisions
        coqui_dir = self.cfg.work_dir / "hybrid_coqui"
        edge_dir = self.cfg.work_dir / "hybrid_edge"
        coqui_dir.mkdir(exist_ok=True)
        edge_dir.mkdir(exist_ok=True)

        self._report("synthesize", 0.05,
                     f"Hybrid: {len(coqui_segments)} segs → Coqui GPU, {len(edge_segments)} segs → Edge-TTS cloud")

        def _match_results_to_segments(results, indexed_segments, out_dict):
            """Match TTS results back to original segment indices by start time.
            TTS functions may skip segments (empty text, errors), so positional
            zip would misalign results. Match by start time instead."""
            for result in results:
                for orig_idx, seg in indexed_segments:
                    if orig_idx not in out_dict and abs(result["start"] - seg["start"]) < 0.01:
                        out_dict[orig_idx] = result
                        break

        def run_coqui():
            try:
                coqui_segs_only = [seg for _, seg in coqui_segments]
                results = self._tts_coqui_xtts(coqui_segs_only, work_dir=coqui_dir)
                _match_results_to_segments(results, coqui_segments, coqui_results)
            except Exception as e:
                errors.append(f"Coqui: {e}")

        def run_edge():
            try:
                edge_segs_only = [seg for _, seg in edge_segments]
                results = self._tts_edge(edge_segs_only, voice_map=self._voice_map, work_dir=edge_dir)
                _match_results_to_segments(results, edge_segments, edge_results)
            except Exception as e:
                errors.append(f"Edge: {e}")

        # Launch both in parallel
        t_coqui = threading.Thread(target=run_coqui, name="tts-coqui")
        t_edge = threading.Thread(target=run_edge, name="tts-edge")
        t_coqui.start()
        t_edge.start()
        t_coqui.join()
        t_edge.join()

        # If Coqui failed, Edge may have succeeded for its half
        if errors:
            self._report("synthesize", 0.85,
                         f"Hybrid partial errors: {'; '.join(errors)}")

        # Merge results in original order
        all_results = {**coqui_results, **edge_results}

        # If Coqui failed entirely, re-generate its segments via Edge-TTS
        if not coqui_results and coqui_segments:
            self._report("synthesize", 0.85,
                         "Coqui failed — re-generating its segments via Edge-TTS...")
            coqui_segs_only = [seg for _, seg in coqui_segments]
            fallback_results = self._tts_edge(coqui_segs_only, voice_map=self._voice_map, work_dir=coqui_dir)
            _match_results_to_segments(fallback_results, coqui_segments, all_results)

        # If Edge-TTS failed entirely, re-generate its segments via Edge-TTS retry
        if not edge_results and edge_segments:
            self._report("synthesize", 0.87,
                         "Edge-TTS failed — retrying its segments...")
            edge_segs_only = [seg for _, seg in edge_segments]
            fallback_results = self._tts_edge(edge_segs_only, voice_map=self._voice_map, work_dir=edge_dir)
            _match_results_to_segments(fallback_results, edge_segments, all_results)

        # Sort by original segment index
        tts_data = [all_results[i] for i in sorted(all_results.keys())]
        self._report("synthesize", 0.9,
                     f"Hybrid TTS complete: {len(tts_data)} segments")
        return tts_data

    def _tts_triple_parallel(self, segments):
        """MANAGED QUAD PARALLEL: Tiered queues — quality engines get reserved work,
        Edge only handles overflow.

        Architecture:
        - Queue A (RESERVED): Top 90% complex segments → CosyVoice, XTTS, Sarvam ONLY
        - Queue B (OVERFLOW): Bottom 10% simple segments → Edge-TTS
        - When a quality engine finishes Queue A, it can steal from Queue B too
        - Edge NEVER touches Queue A — guaranteed quality on complex segments

        Quality tiers (merge priority):
          CosyVoice (4) > XTTS (3) > Sarvam (2) > Edge (1)
        """
        import threading
        import queue

        total = len(segments)

        # ── Detect available engines ──
        sarvam_key = get_sarvam_key()
        sarvam_available = bool(sarvam_key) and self.cfg.use_sarvam_bulbul
        sarvam_num_keys = _sarvam_keys.count() if sarvam_available else 0

        xtts_available = False
        if self.cfg.use_coqui_xtts:
            try:
                import torch
                xtts_available = torch.cuda.is_available()
            except ImportError:
                pass

        cosyvoice_available = False
        if self.cfg.use_cosyvoice:
            try:
                import torch
                if torch.cuda.is_available():
                    import sys
                    cv_dir = str(Path(__file__).resolve().parent / "CosyVoice")
                    if cv_dir not in sys.path:
                        sys.path.insert(0, cv_dir)
                    from cosyvoice.cli.cosyvoice import CosyVoice2
                    cosyvoice_available = True
            except Exception:
                pass

        # Count quality engines available
        quality_engines = sum([cosyvoice_available, xtts_available, sarvam_available])

        # ── Build tiered queues — sorted longest first ──
        seg_lengths = [(i, len(seg.get("text_translated", seg.get("text", "")))) for i, seg in enumerate(segments)]
        seg_lengths.sort(key=lambda x: x[1], reverse=True)

        # Reserve 90% for quality engines, 10% overflow for Edge
        # If no quality engines available, everything goes to Edge
        if quality_engines > 0:
            reserved_count = int(total * 0.90)
        else:
            reserved_count = 0
        overflow_count = total - reserved_count

        quality_queue = queue.Queue()   # Queue A: quality engines only
        overflow_queue = queue.Queue()  # Queue B: Edge + quality overflow

        for idx, _ in seg_lengths[:reserved_count]:
            quality_queue.put(idx)
        for idx, _ in seg_lengths[reserved_count:]:
            overflow_queue.put(idx)

        # Thread-safe results
        results_lock = threading.Lock()
        all_results = {}
        errors = []

        engines = []
        if cosyvoice_available: engines.append("CosyVoice")
        if xtts_available: engines.append("XTTS")
        if sarvam_available: engines.append(f"Sarvam (x{sarvam_num_keys})")
        engines.append("Edge")
        self._report("synthesize", 0.03,
                     f"MANAGED PARALLEL: {reserved_count} reserved (quality) + "
                     f"{overflow_count} overflow (Edge) -> {' + '.join(engines)}")

        # ── Per-engine single-segment synthesizers ──
        def _synth_cosyvoice(seg_idx, seg, work_dir):
            try:
                results = self._tts_cosyvoice2([seg], work_dir=work_dir)
                return results[0] if results else None
            except Exception as e:
                errors.append(f"CosyVoice seg {seg_idx}: {e}")
                return None

        def _synth_xtts(seg_idx, seg, work_dir):
            try:
                results = self._tts_coqui_xtts([seg], work_dir=work_dir)
                return results[0] if results else None
            except Exception as e:
                errors.append(f"XTTS seg {seg_idx}: {e}")
                return None

        def _synth_sarvam(seg_idx, seg, work_dir):
            try:
                results = self._tts_sarvam_parallel([seg], work_dir=work_dir)
                return results[0] if results else None
            except Exception as e:
                errors.append(f"Sarvam seg {seg_idx}: {e}")
                return None

        def _synth_edge(seg_idx, seg, work_dir):
            try:
                results = self._tts_edge([seg], voice_map=self._voice_map, work_dir=work_dir)
                return results[0] if results else None
            except Exception as e:
                errors.append(f"Edge seg {seg_idx}: {e}")
                return None

        QUALITY = {"cosyvoice": 4, "xtts": 3, "sarvam": 2, "edge": 1}

        def quality_worker(engine_name, synth_fn):
            """Quality engine: pulls from quality_queue first, then overflow_queue."""
            worker_dir = self.cfg.work_dir / f"dyn_{engine_name}"
            worker_dir.mkdir(exist_ok=True)
            count = 0

            # Phase 1: Process reserved quality segments
            while True:
                try:
                    seg_idx = quality_queue.get_nowait()
                except queue.Empty:
                    break

                seg = segments[seg_idx]
                seg_dir = worker_dir / f"seg_{seg_idx:04d}"
                seg_dir.mkdir(exist_ok=True)

                result = synth_fn(seg_idx, seg, seg_dir)
                if result:
                    # Tag result with segment index + expected text for assembly verification
                    result["_seg_idx"] = seg_idx
                    result["_expected_text"] = seg.get("_expected_text", seg.get("text_translated", ""))
                    with results_lock:
                        existing = all_results.get(seg_idx)
                        if not existing or QUALITY[engine_name] > existing.get("_quality", 0):
                            result["_quality"] = QUALITY[engine_name]
                            result["_engine"] = engine_name
                            all_results[seg_idx] = result
                        done = len(all_results)
                    count += 1

                if done % 5 == 0 or done == total:
                    self._report("synthesize", 0.05 + 0.80 * (done / total),
                                 f"TTS: {done}/{total} ({engine_name}: {count})...")
                pass  # queue.Queue doesn't need task_done

            # Phase 2: Quality engine finished its queue — help with overflow
            while True:
                try:
                    seg_idx = overflow_queue.get_nowait()
                except queue.Empty:
                    break

                seg = segments[seg_idx]
                seg_dir = worker_dir / f"seg_{seg_idx:04d}"
                seg_dir.mkdir(exist_ok=True)

                result = synth_fn(seg_idx, seg, seg_dir)
                if result:
                    result["_seg_idx"] = seg_idx
                    result["_expected_text"] = seg.get("_expected_text", seg.get("text_translated", ""))
                    with results_lock:
                        existing = all_results.get(seg_idx)
                        if not existing or QUALITY[engine_name] > existing.get("_quality", 0):
                            result["_quality"] = QUALITY[engine_name]
                            result["_engine"] = engine_name
                            all_results[seg_idx] = result
                        done = len(all_results)
                    count += 1

                if done % 5 == 0 or done == total:
                    self._report("synthesize", 0.05 + 0.80 * (done / total),
                                 f"TTS: {done}/{total} ({engine_name}: {count}, overflow)...")
                pass  # queue.Queue doesn't need task_done

        def edge_worker():
            """Edge: ONLY pulls from overflow_queue. Never touches quality segments."""
            worker_dir = self.cfg.work_dir / "dyn_edge"
            worker_dir.mkdir(exist_ok=True)
            count = 0

            while True:
                try:
                    seg_idx = overflow_queue.get_nowait()
                except queue.Empty:
                    break

                seg = segments[seg_idx]
                seg_dir = worker_dir / f"seg_{seg_idx:04d}"
                seg_dir.mkdir(exist_ok=True)

                result = _synth_edge(seg_idx, seg, seg_dir)
                if result:
                    result["_seg_idx"] = seg_idx
                    result["_expected_text"] = seg.get("_expected_text", seg.get("text_translated", ""))
                    with results_lock:
                        existing = all_results.get(seg_idx)
                        if not existing or QUALITY["edge"] > existing.get("_quality", 0):
                            result["_quality"] = QUALITY["edge"]
                            result["_engine"] = "edge"
                            all_results[seg_idx] = result
                        done = len(all_results)
                    count += 1

                if done % 5 == 0 or done == total:
                    self._report("synthesize", 0.05 + 0.80 * (done / total),
                                 f"TTS: {done}/{total} (Edge: {count})...")
                pass  # queue.Queue doesn't need task_done

        # Pre-create worker dirs (avoids race conditions on mkdir in threads)
        for eng_name in ["cosyvoice", "xtts", "sarvam", "edge", "fallback"]:
            (self.cfg.work_dir / f"dyn_{eng_name}").mkdir(exist_ok=True)

        # ── Launch workers ──
        threads = []
        if cosyvoice_available:
            threads.append(threading.Thread(
                target=quality_worker, args=("cosyvoice", _synth_cosyvoice),
                name="tts-cosyvoice"))
        if xtts_available:
            threads.append(threading.Thread(
                target=quality_worker, args=("xtts", _synth_xtts),
                name="tts-xtts"))
        if sarvam_available:
            for k in range(min(sarvam_num_keys, 3)):
                threads.append(threading.Thread(
                    target=quality_worker, args=("sarvam", _synth_sarvam),
                    name=f"tts-sarvam-{k}"))
        threads.append(threading.Thread(target=edge_worker, name="tts-edge"))

        worker_names = []
        if cosyvoice_available: worker_names.append("CosyVoice")
        if xtts_available: worker_names.append("XTTS")
        if sarvam_available: worker_names.append(f"Sarvam x{min(sarvam_num_keys, 3)}")
        worker_names.append("Edge (overflow only)")
        self._report("synthesize", 0.04,
                     f"Launching {len(threads)} workers: {' + '.join(worker_names)}")

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors and len(errors) <= 10:
            self._report("synthesize", 0.85,
                         f"TTS partial errors: {len(errors)} (see logs)")

        # ── Fallback: any segments still missing -> Edge-TTS retry ──
        missing = [i for i in range(total) if i not in all_results]
        if missing:
            self._report("synthesize", 0.87,
                         f"Recovering {len(missing)} missing segments via Edge-TTS...")
            fallback_dir = self.cfg.work_dir / "dyn_fallback"
            fallback_dir.mkdir(exist_ok=True)
            for seg_idx in missing:
                result = _synth_edge(seg_idx, segments[seg_idx], fallback_dir)
                if result:
                    result["_quality"] = 1
                    result["_engine"] = "edge"
                    all_results[seg_idx] = result

        tts_data = [all_results[i] for i in sorted(all_results.keys())]

        # Report engine distribution
        stats = []
        for eng in ["cosyvoice", "xtts", "sarvam", "edge"]:
            n = sum(1 for r in all_results.values() if r.get("_engine") == eng)
            if n: stats.append(f"{eng}={n}")
        self._report("synthesize", 0.92,
                     f"Managed TTS complete: {len(tts_data)}/{total} segments "
                     f"({', '.join(stats)})")

        # ── Smooth audio across engines ──
        # Different engines produce different loudness, tone, noise floor.
        # This pass normalizes all segments to consistent loudness + EQ.
        if len(tts_data) > 1:
            tts_data = self._smooth_multi_engine_audio(tts_data)

        return tts_data

    def _tts_chatterbox_turbo(self, segments):
        """Generate TTS using Chatterbox-Turbo — 350M English-focused model.

        Lowest VRAM usage, MIT license, voice cloning from original audio.
        Primary engine for zero-spend English dubbing.
        """
        import torch
        import torchaudio
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        self._report("synthesize", 0.05, "Loading Chatterbox-Turbo (350M) on GPU...")
        model = ChatterboxTurboTTS.from_pretrained(device="cuda")

        # Prepare voice reference from original audio (needs >5s)
        try:
            ref_wav = self._get_voice_ref(min_duration=6, sample_rate=24000)
            if ref_wav:
                self._report("synthesize", 0.06, "Cloning voice from original audio...")
                model.prepare_conditionals(str(ref_wav))
        except Exception as e:
            print(f"[CB-Turbo] Voice cloning setup failed ({e}), using built-in voice", flush=True)

        tts_data = []
        try:
            for i, seg in enumerate(segments):
                text = self._prepare_tts_text(
                    seg.get("text_translated", seg.get("text", "")).strip()
                )
                if not text:
                    continue

                wav_path = self.cfg.work_dir / f"tts_{i:04d}.wav"

                try:
                    wav_tensor = model.generate(text)
                    torchaudio.save(str(wav_path), wav_tensor.cpu(), model.sr)

                    # Resample to pipeline sample rate
                    if model.sr != self.SAMPLE_RATE:
                        resampled = self.cfg.work_dir / f"tts_{i:04d}_rs.wav"
                        self._run_proc(
                            [self._ffmpeg, "-y", "-i", str(wav_path),
                             "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                             str(resampled)],
                            check=True, capture_output=True,
                        )
                        wav_path.unlink(missing_ok=True)
                        resampled.replace(wav_path)
                    self._enhance_tts_wav(wav_path)

                    expected_dur = seg.get("end", 0) - seg.get("start", 0)
                    qc = self._qc_check_wav(wav_path, expected_duration=expected_dur)
                    if not qc["ok"]:
                        print(f"[QC/CB-Turbo] Seg {i} issues: {'; '.join(qc['issues'])}", flush=True)

                except Exception as e:
                    self._report("synthesize", 0.1 + 0.8 * ((i + 1) / len(segments)),
                                 f"CB-Turbo error on seg {i+1}: {e}, skipping...")
                    continue

                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    continue

                tts_dur = self._get_duration(wav_path)
                tts_data.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "wav": wav_path,
                    "duration": tts_dur,
                })
                self._report(
                    "synthesize",
                    0.1 + 0.8 * ((i + 1) / len(segments)),
                    f"Synthesized {i + 1}/{len(segments)} segments (CB-Turbo)...",
                )
        finally:
            try:
                model = None
            except Exception:
                pass
            _hardened_cuda_cleanup()

        return tts_data

    def _tts_chatterbox_multilingual(self, segments):
        """Generate TTS using Chatterbox Multilingual — 23 languages, voice cloning.

        Fallback for when Turbo fails or for non-English targets.
        MIT license, supports voice cloning from a reference clip.
        """
        import torch
        import torchaudio
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        target = self.cfg.target_language.split("-")[0].lower()  # "en-US" → "en", "Hi" → "hi"

        self._report("synthesize", 0.05, "Loading Chatterbox Multilingual on GPU...")
        model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

        # Prepare voice reference from original audio
        try:
            ref_wav = self._get_voice_ref(min_duration=6, sample_rate=24000)
            if ref_wav:
                self._report("synthesize", 0.06, "Cloning voice from original audio...")
                model.prepare_conditionals(str(ref_wav))
        except Exception as e:
            print(f"[CB-MTL] Voice cloning setup failed ({e}), using built-in voice", flush=True)

        tts_data = []
        try:
            for i, seg in enumerate(segments):
                text = self._prepare_tts_text(
                    seg.get("text_translated", seg.get("text", "")).strip()
                )
                if not text:
                    continue

                wav_path = self.cfg.work_dir / f"tts_{i:04d}.wav"

                try:
                    wav_tensor = model.generate(text, language_id=target)
                    torchaudio.save(str(wav_path), wav_tensor.cpu(), model.sr)

                    # Resample to pipeline sample rate
                    if model.sr != self.SAMPLE_RATE:
                        resampled = self.cfg.work_dir / f"tts_{i:04d}_rs.wav"
                        self._run_proc(
                            [self._ffmpeg, "-y", "-i", str(wav_path),
                             "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                             str(resampled)],
                            check=True, capture_output=True,
                        )
                        wav_path.unlink(missing_ok=True)
                        resampled.replace(wav_path)
                    self._enhance_tts_wav(wav_path)

                    expected_dur = seg.get("end", 0) - seg.get("start", 0)
                    qc = self._qc_check_wav(wav_path, expected_duration=expected_dur)
                    if not qc["ok"]:
                        print(f"[QC/CB-MTL] Seg {i} issues: {'; '.join(qc['issues'])}", flush=True)

                except Exception as e:
                    self._report("synthesize", 0.1 + 0.8 * ((i + 1) / len(segments)),
                                 f"CB-Multilingual error on seg {i+1}: {e}, skipping...")
                    continue

                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    continue

                tts_dur = self._get_duration(wav_path)
                tts_data.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "wav": wav_path,
                    "duration": tts_dur,
                })
                self._report(
                    "synthesize",
                    0.1 + 0.8 * ((i + 1) / len(segments)),
                    f"Synthesized {i + 1}/{len(segments)} segments (CB-Multilingual)...",
                )
        finally:
            try:
                model = None
            except Exception:
                pass
            _hardened_cuda_cleanup()

        return tts_data

    def _get_voice_ref(self, min_duration: int = 6, sample_rate: int = 24000) -> Optional[Path]:
        """Extract a voice reference clip from original audio for voice cloning.

        Returns path to a WAV file suitable for Chatterbox voice cloning,
        or None if no audio is available.
        """
        ref_wav = self.cfg.work_dir / f"voice_ref_{sample_rate}.wav"
        if ref_wav.exists():
            return ref_wav

        # Check for user-provided reference first
        user_ref = Path(__file__).resolve().parent / "voices" / "my_voice_refs"
        if user_ref.exists():
            ref_files = list(user_ref.glob("*.wav")) + list(user_ref.glob("*.mp3"))
            if ref_files:
                # Convert to correct sample rate
                self._run_proc(
                    [self._ffmpeg, "-y", "-i", str(ref_files[0]),
                     "-t", "15", "-ar", str(sample_rate), "-ac", "1",
                     str(ref_wav)],
                    check=True, capture_output=True,
                )
                return ref_wav

        # Extract from original audio
        audio_raw = self.cfg.work_dir / "audio_raw.wav"
        if audio_raw.exists():
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(audio_raw),
                 "-t", "15", "-ar", str(sample_rate), "-ac", "1",
                 str(ref_wav)],
                check=True, capture_output=True,
            )
            if ref_wav.exists() and ref_wav.stat().st_size > 0:
                return ref_wav

        return None

    def _tts_cosyvoice2(self, segments, work_dir=None):
        """Generate TTS using CosyVoice 2 — near-ElevenLabs quality, completely free.

        Uses cross-lingual voice cloning:
        - Takes first 10 sec of original English audio as voice reference
        - Extracts speaker tone/pitch/style from that reference
        - Generates Hindi (or any target language) speech with the same voice

        Install: git clone https://github.com/FunAudioLLM/CosyVoice
                 pip install -r CosyVoice/requirements.txt
        """
        if work_dir is None:
            work_dir = self.cfg.work_dir

        # Add CosyVoice to sys.path
        import sys
        cosyvoice_dir = str(Path(__file__).resolve().parent / "CosyVoice")
        matcha_dir = str(Path(__file__).resolve().parent / "CosyVoice" / "third_party" / "Matcha-TTS")
        for p in [cosyvoice_dir, matcha_dir]:
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
            from cosyvoice.utils.file_utils import load_wav as cosyvoice_load_wav
        except ImportError:
            raise RuntimeError(
                "CosyVoice 2 not installed. "
                "Run: git clone --recursive https://github.com/FunAudioLLM/CosyVoice && "
                "pip install -r CosyVoice/requirements.txt"
            )

        import torch
        import torchaudio

        # ── Prepare 16kHz mono voice reference (CosyVoice 2 requires 16kHz) ──
        ref_16k = self.cfg.work_dir / "cosyvoice_ref_16k.wav"
        audio_raw = self.cfg.work_dir / "audio_raw.wav"
        if not ref_16k.exists() and audio_raw.exists():
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(audio_raw),
                 "-t", "10", "-ar", "16000", "-ac", "1", str(ref_16k)],
                check=True, capture_output=True,
            )
        if not ref_16k.exists():
            raise RuntimeError("No voice reference available for CosyVoice 2")

        # ── Load model ────────────────────────────────────────────────────────
        self._report("synthesize", 0.02, "Loading CosyVoice 2 model (0.5B, first run downloads ~1GB)...")

        # Try local first, then HuggingFace
        model_path = Path(__file__).parent / "pretrained_models" / "CosyVoice2-0.5B"
        if not model_path.exists():
            # Check inside CosyVoice clone dir
            model_path = Path(__file__).parent / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B"
        if not model_path.exists():
            model_path = "FunAudioLLM/CosyVoice2-0.5B"  # HF hub ID — auto-downloads

        tts_model = CosyVoice2(str(model_path), load_jit=False, load_trt=False)
        prompt_speech = cosyvoice_load_wav(str(ref_16k), 16000)

        tts_data = []
        total = len(segments)
        try:
            for i, seg in enumerate(segments):
                text = self._prepare_tts_text(
                    seg.get("text_translated", seg.get("text", "")).strip()
                )
                if not text:
                    continue

                wav_path = work_dir / f"tts_{i:04d}.wav"
                try:
                    # Cross-lingual: English voice reference → target language output
                    chunks = []
                    for _, result in tts_model.inference_cross_lingual(
                        text, prompt_speech, stream=False
                    ):
                        chunks.append(result["tts_speech"])

                    if not chunks:
                        continue

                    audio = torch.cat(chunks, dim=-1)
                    torchaudio.save(str(wav_path), audio, tts_model.sample_rate)

                    # Resample to pipeline sample rate
                    if tts_model.sample_rate != self.SAMPLE_RATE:
                        resampled = wav_path.with_suffix(".rs.wav")
                        self._run_proc(
                            [self._ffmpeg, "-y", "-i", str(wav_path),
                             "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                             str(resampled)],
                            check=True, capture_output=True,
                        )
                        wav_path.unlink(missing_ok=True)
                        resampled.replace(wav_path)

                    self._enhance_tts_wav(wav_path)
                    # QC gate
                    expected_dur = seg.get("end", 0) - seg.get("start", 0)
                    qc = self._qc_check_wav(wav_path, expected_duration=expected_dur)
                    if not qc["ok"]:
                        print(f"[QC/CosyVoice2] Seg {i} issues: {'; '.join(qc['issues'])}", flush=True)

                except Exception as e:
                    self._report("synthesize", 0.1 + 0.8 * ((i + 1) / total),
                                 f"CosyVoice2 error seg {i+1}: {e}, skipping...")
                    continue

                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    continue

                tts_dur = self._get_duration(wav_path)
                tts_data.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "wav": wav_path,
                    "duration": tts_dur,
                })
                self._report("synthesize", 0.1 + 0.8 * ((i + 1) / total),
                             f"CosyVoice2: {i+1}/{total} segments (voice cloned)...")
        finally:
            try:
                tts_model = None
            except Exception:
                pass
            _hardened_cuda_cleanup()

        return tts_data

    # ── Sarvam language code mapping ──
    SARVAM_LANG_MAP = {
        "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
        "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN",
        "od": "od-IN", "pa": "pa-IN", "en": "en-IN",
    }
    SARVAM_SUPPORTED_LANGS = set(SARVAM_LANG_MAP.keys())

    def _tts_sarvam_bulbul(self, segments, work_dir=None):
        """Generate TTS using Sarvam AI Bulbul v3 — best Hindi quality, cloud API.

        Features:
        - Purpose-built for Indian languages (11 languages)
        - 38+ Hindi voices (male/female)
        - API key rotation with auto-failover
        - Max 2500 chars per request
        - Returns base64-encoded audio

        Get free API key at https://dashboard.sarvam.ai (Rs 1000 free credits)
        """
        import requests
        import base64

        if work_dir is None:
            work_dir = self.cfg.work_dir

        api_key = get_sarvam_key()
        if not api_key:
            raise RuntimeError(
                "Sarvam Bulbul enabled but no SARVAM_API_KEY set in .env. "
                "Get free key at https://dashboard.sarvam.ai"
            )

        # Map target language
        target = self.cfg.target_language.split("-")[0]
        lang_code = self.SARVAM_LANG_MAP.get(target)
        if not lang_code:
            raise RuntimeError(
                f"Sarvam Bulbul doesn't support language '{target}'. "
                f"Supported: {', '.join(sorted(self.SARVAM_SUPPORTED_LANGS))}"
            )

        # Pick speaker based on target language
        speaker = "shubh"  # Default Hindi male voice — clear, natural
        tts_data = []
        total = len(segments)

        for i, seg in enumerate(segments):
            text = self._prepare_tts_text(
                seg.get("text_translated", seg.get("text", "")).strip()
            )
            if not text:
                continue

            # Sarvam max 2500 chars per request — cut at sentence boundary, never mid-word
            if len(text) > 2500:
                for sep in ['।', '.', '!', '?', ',', ' ']:
                    last = text[:2500].rfind(sep)
                    if last > 2000:
                        text = text[:last + 1]
                        break
                else:
                    text = text[:2500]

            wav_path = work_dir / f"tts_{i:04d}.wav"
            mp3_path = work_dir / f"tts_{i:04d}.mp3"

            # Retry with key rotation on quota errors
            max_retries = _sarvam_keys.count()
            current_key = api_key
            success = False

            for attempt in range(max(max_retries, 2)):
                try:
                    resp = requests.post(
                        "https://api.sarvam.ai/text-to-speech",
                        headers={
                            "api-subscription-key": current_key,
                            "Content-Type": "application/json",
                        },
                        json={
                            "text": text,
                            "target_language_code": lang_code,
                            "speaker": speaker,
                            "model": "bulbul:v3",
                            "pace": 1.0,
                            "temperature": 0.6,
                            "speech_sample_rate": 24000,
                            "output_audio_codec": "mp3",
                        },
                        timeout=30,
                    )

                    if resp.status_code == 429:
                        # Quota exhausted — rotate key
                        _sarvam_keys.report_quota_error(current_key)
                        current_key = get_sarvam_key()
                        self._report("synthesize", 0.1 + 0.8 * ((i + 1) / total),
                                     f"Sarvam key quota hit — rotated to next key (attempt {attempt + 1})...")
                        continue

                    if resp.status_code != 200:
                        raise RuntimeError(f"Sarvam API error {resp.status_code}: {resp.text[:200]}")

                    result = resp.json()
                    audio_b64 = result.get("audios", [None])[0]
                    if not audio_b64:
                        raise RuntimeError("Sarvam returned empty audio")

                    audio_bytes = base64.b64decode(audio_b64)
                    mp3_path.write_bytes(audio_bytes)

                    # Convert MP3 → WAV (48kHz stereo, pipeline standard)
                    self._run_proc(
                        [self._ffmpeg, "-y", "-i", str(mp3_path),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(wav_path)],
                        check=True, capture_output=True,
                    )
                    mp3_path.unlink(missing_ok=True)

                    self._enhance_tts_wav(wav_path)

                    # QC gate
                    expected_dur = seg.get("end", 0) - seg.get("start", 0)
                    qc = self._qc_check_wav(wav_path, expected_duration=expected_dur)
                    if not qc["ok"]:
                        print(f"[QC/Sarvam] Seg {i} issues: {'; '.join(qc['issues'])}", flush=True)

                    success = True
                    break

                except requests.exceptions.Timeout:
                    self._report("synthesize", 0.1 + 0.8 * ((i + 1) / total),
                                 f"Sarvam timeout on seg {i+1}, retrying...")
                    continue
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        _sarvam_keys.report_quota_error(current_key)
                        current_key = get_sarvam_key()
                        continue
                    self._report("synthesize", 0.1 + 0.8 * ((i + 1) / total),
                                 f"Sarvam error seg {i+1}: {e}, skipping...")
                    break

            if not success or not wav_path.exists() or wav_path.stat().st_size == 0:
                continue

            tts_dur = self._get_duration(wav_path)
            tts_data.append({
                "start": seg["start"],
                "end": seg["end"],
                "wav": wav_path,
                "duration": tts_dur,
            })
            self._report("synthesize", 0.1 + 0.8 * ((i + 1) / total),
                         f"Sarvam Bulbul: {i+1}/{total} segments...")

        return tts_data

    def _tts_sarvam_parallel(self, segments, work_dir=None):
        """Parallel Sarvam Bulbul — runs N concurrent HTTP requests (1 per API key).

        Same logic as _tts_sarvam_bulbul but uses ThreadPoolExecutor for concurrency.
        With 5 keys: 5 concurrent requests = ~5x faster than sequential.
        Thread-safe: each worker gets its own rotated key.
        """
        import requests
        import base64
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if work_dir is None:
            work_dir = self.cfg.work_dir

        api_key = get_sarvam_key()
        if not api_key:
            raise RuntimeError(
                "Sarvam Bulbul enabled but no SARVAM_API_KEY set in .env. "
                "Get free key at https://dashboard.sarvam.ai"
            )

        target = self.cfg.target_language.split("-")[0]
        lang_code = self.SARVAM_LANG_MAP.get(target)
        if not lang_code:
            raise RuntimeError(
                f"Sarvam Bulbul doesn't support language '{target}'. "
                f"Supported: {', '.join(sorted(self.SARVAM_SUPPORTED_LANGS))}"
            )

        speaker = "shubh"
        total = len(segments)
        num_keys = _sarvam_keys.count()
        concurrency = max(num_keys, 2)  # At least 2 concurrent, up to num_keys
        completed = [0]

        def synthesize_one(i, seg):
            text = self._prepare_tts_text(
                seg.get("text_translated", seg.get("text", "")).strip()
            )
            if not text:
                return None

            if len(text) > 2500:
                for sep in ['।', '.', '!', '?', ',', ' ']:
                    last = text[:2500].rfind(sep)
                    if last > 2000:
                        text = text[:last + 1]
                        break
                else:
                    text = text[:2500]

            wav_path = work_dir / f"tts_{i:04d}.wav"
            mp3_path = work_dir / f"tts_{i:04d}.mp3"

            current_key = get_sarvam_key()
            max_retries = max(num_keys, 2)

            for attempt in range(max_retries):
                try:
                    resp = requests.post(
                        "https://api.sarvam.ai/text-to-speech",
                        headers={
                            "api-subscription-key": current_key,
                            "Content-Type": "application/json",
                        },
                        json={
                            "text": text,
                            "target_language_code": lang_code,
                            "speaker": speaker,
                            "model": "bulbul:v3",
                            "pace": 1.0,
                            "temperature": 0.6,
                            "speech_sample_rate": 24000,
                            "output_audio_codec": "mp3",
                        },
                        timeout=30,
                    )

                    if resp.status_code == 429:
                        _sarvam_keys.report_quota_error(current_key)
                        current_key = get_sarvam_key()
                        continue

                    if resp.status_code != 200:
                        raise RuntimeError(f"Sarvam API error {resp.status_code}: {resp.text[:200]}")

                    result = resp.json()
                    audio_b64 = result.get("audios", [None])[0]
                    if not audio_b64:
                        raise RuntimeError("Sarvam returned empty audio")

                    audio_bytes = base64.b64decode(audio_b64)
                    mp3_path.write_bytes(audio_bytes)

                    self._run_proc(
                        [self._ffmpeg, "-y", "-i", str(mp3_path),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(wav_path)],
                        check=True, capture_output=True,
                    )
                    mp3_path.unlink(missing_ok=True)

                    self._enhance_tts_wav(wav_path)
                    expected_dur = seg.get("end", 0) - seg.get("start", 0)
                    self._qc_check_wav(wav_path, expected_duration=expected_dur)

                    if wav_path.exists() and wav_path.stat().st_size > 0:
                        tts_dur = self._get_duration(wav_path)
                        return {
                            "start": seg["start"],
                            "end": seg["end"],
                            "wav": wav_path,
                            "duration": tts_dur,
                        }
                    break

                except requests.exceptions.Timeout:
                    continue
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        _sarvam_keys.report_quota_error(current_key)
                        current_key = get_sarvam_key()
                        continue
                    break

            return None

        tts_data = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(synthesize_one, i, seg): i
                for i, seg in enumerate(segments)
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    tts_data.append(result)
                completed[0] += 1
                if completed[0] % max(concurrency, 5) == 0 or completed[0] == total:
                    self._report("synthesize",
                                 0.1 + 0.8 * (completed[0] / total),
                                 f"Sarvam Bulbul: {completed[0]}/{total} "
                                 f"(x{concurrency} parallel)...")

        return tts_data

    def _tts_elevenlabs(self, segments, api_key):
        """Generate TTS using ElevenLabs — paid API, most human-like."""
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=api_key)

        voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
        if not voice_id:
            voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel — clear, natural

        model_id = "eleven_multilingual_v2"

        tts_data = []
        for i, seg in enumerate(segments):
            text = seg.get("text_translated", seg.get("text", "")).strip()
            if not text:
                continue

            mp3 = self.cfg.work_dir / f"tts_{i:04d}.mp3"

            try:
                audio_gen = client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format="mp3_44100_128",
                )
                with open(mp3, "wb") as f:
                    for chunk in audio_gen:
                        f.write(chunk)
            except Exception:
                try:
                    asyncio.run(self._edge_tts_single(text, mp3))
                except Exception:
                    continue

            if not mp3.exists() or mp3.stat().st_size == 0:
                continue

            wav = self.cfg.work_dir / f"tts_{i:04d}.wav"
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(mp3),
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(wav)],
                check=True, capture_output=True,
            )
            mp3.unlink(missing_ok=True)
            tts_dur = self._get_duration(wav)
            tts_data.append({
                "start": seg["start"],
                "end": seg["end"],
                "wav": wav,
                "duration": tts_dur,
            })
            self._report(
                "synthesize",
                0.1 + 0.8 * ((i + 1) / len(segments)),
                f"Synthesized {i + 1}/{len(segments)} segments (ElevenLabs)...",
            )

        return tts_data

    def _tts_google_cloud(self, segments):
        """Generate TTS using Google Cloud Text-to-Speech (WaveNet/Neural2).

        Free tier: 1M characters/month per billing account. Pool rotates across
        multiple service-account JSONs so multiple billing accounts can be used
        together — see backend/dubbing/google_tts_pool.py for env config.
        """
        from google.cloud import texttospeech
        from dubbing.google_tts_pool import load_pool_from_env

        usage_file = Path(__file__).resolve().parent / "google_tts_usage.json"
        pool = load_pool_from_env(usage_file)
        if pool is None:
            raise RuntimeError(
                "Google TTS enabled but no credentials found. Set "
                "GOOGLE_TTS_CREDENTIALS_DIR, GOOGLE_TTS_CREDENTIAL_1..N, "
                "GOOGLE_TTS_CREDENTIALS, or GOOGLE_APPLICATION_CREDENTIALS."
            )
        target = self.cfg.target_language

        # Map target language to Google Cloud voice config
        GOOGLE_VOICE_MAP = {
            "hi": ("hi-IN", "hi-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "bn": ("bn-IN", "bn-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "ta": ("ta-IN", "ta-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "te": ("te-IN", "te-IN-Standard-A", texttospeech.SsmlVoiceGender.FEMALE),
            "mr": ("mr-IN", "mr-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "gu": ("gu-IN", "gu-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "kn": ("kn-IN", "kn-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "ml": ("ml-IN", "ml-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "pa": ("pa-IN", "pa-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "en": ("en-US", "en-US-Neural2-F", texttospeech.SsmlVoiceGender.FEMALE),
            "es": ("es-ES", "es-ES-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "fr": ("fr-FR", "fr-FR-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "de": ("de-DE", "de-DE-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "ja": ("ja-JP", "ja-JP-Neural2-B", texttospeech.SsmlVoiceGender.FEMALE),
            "ko": ("ko-KR", "ko-KR-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "zh": ("cmn-CN", "cmn-CN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "ar": ("ar-XA", "ar-XA-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "pt": ("pt-BR", "pt-BR-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "ru": ("ru-RU", "ru-RU-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "it": ("it-IT", "it-IT-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
        }

        lang_code, voice_name, gender = GOOGLE_VOICE_MAP.get(
            target, (f"{target}-IN", None, texttospeech.SsmlVoiceGender.FEMALE)
        )

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
            ssml_gender=gender,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.SAMPLE_RATE,
        )

        tts_data = []
        for i, seg in enumerate(segments):
            text = seg.get("text_translated", seg.get("text", "")).strip()
            if not text:
                continue

            wav_path = self.cfg.work_dir / f"tts_{i:04d}.wav"

            try:
                response = pool.synthesize(text, voice_params, audio_config)
                with open(wav_path, "wb") as f:
                    f.write(response.audio_content)

                # Ensure correct format (stereo, target sample rate)
                wav_fixed = self.cfg.work_dir / f"tts_{i:04d}_fixed.wav"
                self._run_proc(
                    [self._ffmpeg, "-y", "-i", str(wav_path),
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     str(wav_fixed)],
                    check=True, capture_output=True,
                )
                shutil.move(str(wav_fixed), str(wav_path))

            except Exception as e:
                self._report("synthesize", 0.1 + 0.8 * ((i + 1) / len(segments)),
                             f"Google TTS error on seg {i+1}: {e}, falling back to Edge-TTS...")
                mp3 = self.cfg.work_dir / f"tts_{i:04d}.mp3"
                try:
                    asyncio.run(self._edge_tts_single(text, mp3))
                    self._run_proc(
                        [self._ffmpeg, "-y", "-i", str(mp3),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(wav_path)],
                        check=True, capture_output=True,
                    )
                    mp3.unlink(missing_ok=True)
                except Exception:
                    continue

            if not wav_path.exists() or wav_path.stat().st_size == 0:
                continue

            tts_dur = self._get_duration(wav_path)
            tts_data.append({
                "start": seg["start"],
                "end": seg["end"],
                "wav": wav_path,
                "duration": tts_dur,
            })
            self._report(
                "synthesize",
                0.1 + 0.8 * ((i + 1) / len(segments)),
                f"Synthesized {i + 1}/{len(segments)} segments (Google Cloud TTS)...",
            )

        return tts_data

    # ── Fish Speech 1.5 ────────────────────────────────────────────────
    _fish_speech_dir = Path(__file__).resolve().parent / "fish-speech"

    def _tts_fish_speech(self, segments, work_dir=None):
        """Generate TTS using Fish Speech 1.5 — open-source voice cloning (~3GB VRAM).

        Features:
        - 500M dual-AR transformer + VQGAN decoder
        - Voice cloning from 10-30s reference audio
        - Output: 44100 Hz WAV
        - Languages: en, zh, ja, ko, de, fr, es, ar, ru, nl, it, pl, pt
          (Hindi not official but may work partially)
        - Runs alongside XTTS on 12GB GPU (~3GB + ~3GB = ~6GB)
        """
        import sys
        import torch
        import numpy as np

        if work_dir is None:
            work_dir = self.cfg.work_dir

        # Add fish-speech to sys.path for imports
        fish_dir = str(self._fish_speech_dir)
        if fish_dir not in sys.path:
            sys.path.insert(0, fish_dir)

        device = "cuda"
        ckpt_dir = self._fish_speech_dir / "checkpoints" / "fish-speech-1.5"

        if not ckpt_dir.exists():
            raise RuntimeError(
                f"Fish Speech 1.5 model not found at {ckpt_dir}. "
                "Run: huggingface-cli download fishaudio/fish-speech-1.5 "
                f"--local-dir {ckpt_dir}"
            )

        tts_data = []
        total = len(segments)

        try:
            # Load VQGAN decoder
            self._report("synthesize", 0.02,
                         "Loading Fish Speech 1.5 VQGAN decoder...")
            from tools.vqgan.inference import load_model as load_decoder_model
            decoder_model = load_decoder_model(
                config_name="firefly_gan_vq",
                checkpoint_path=str(ckpt_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
                device=device,
            )

            # Load LLM via thread-safe queue
            self._report("synthesize", 0.05,
                         "Loading Fish Speech 1.5 LLM (dual-AR transformer)...")
            from tools.llama.generate import launch_thread_safe_queue
            from tools.inference_engine import TTSInferenceEngine
            from tools.schema import ServeTTSRequest, ServeReferenceAudio

            llama_queue = launch_thread_safe_queue(
                checkpoint_path=str(ckpt_dir),
                device=device,
                precision=torch.bfloat16,
                compile=False,
            )

            engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                precision=torch.bfloat16,
                compile=False,
            )

            # Build voice reference from original audio
            ref_wav = self.cfg.work_dir / "voice_ref.wav"
            audio_raw = self.cfg.work_dir / "audio_raw.wav"
            if not ref_wav.exists() and audio_raw.exists():
                self._run_proc(
                    [self._ffmpeg, "-y", "-i", str(audio_raw),
                     "-t", "15", "-ar", "44100", "-ac", "1",
                     str(ref_wav)],
                    check=True, capture_output=True,
                )

            ref_audio_bytes = None
            if ref_wav.exists():
                ref_audio_bytes = ref_wav.read_bytes()

            for i, seg in enumerate(segments):
                text = seg.get("text_translated", seg.get("text", "")).strip()
                if not text:
                    continue

                wav_path = work_dir / f"tts_{i:04d}.wav"

                try:
                    # Build request
                    references = []
                    if ref_audio_bytes:
                        references.append(ServeReferenceAudio(
                            audio=ref_audio_bytes,
                            text="",  # No transcript needed for basic cloning
                        ))

                    request = ServeTTSRequest(
                        text=text,
                        references=references,
                        max_new_tokens=1024,
                        chunk_length=200,
                        top_p=0.7,
                        repetition_penalty=1.2,
                        temperature=0.7,
                        format="wav",
                        normalize=True,
                        streaming=False,
                    )

                    # Run inference
                    for result in engine.inference(request):
                        if hasattr(result, 'code') and result.code == "final":
                            sample_rate, audio_array = result.audio
                            import soundfile as sf
                            sf.write(str(wav_path), audio_array, sample_rate)
                            break
                        elif hasattr(result, 'code') and result.code == "error":
                            raise RuntimeError(f"Fish Speech error: {result.error}")

                    if not wav_path.exists() or wav_path.stat().st_size == 0:
                        continue

                    # Resample to pipeline sample rate if needed
                    if sample_rate != self.SAMPLE_RATE:
                        resampled = work_dir / f"tts_{i:04d}_rs.wav"
                        self._run_proc(
                            [self._ffmpeg, "-y", "-i", str(wav_path),
                             "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                             str(resampled)],
                            check=True, capture_output=True,
                        )
                        wav_path.unlink(missing_ok=True)
                        resampled.replace(wav_path)

                    self._enhance_tts_wav(wav_path)
                    tts_dur = self._get_duration(wav_path)

                    tts_data.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "wav": wav_path,
                        "duration": tts_dur,
                    })
                    self._report("synthesize",
                                 0.1 + 0.8 * ((i + 1) / total),
                                 f"Fish Speech: {i+1}/{total} segments (voice clone)...")

                except Exception as e:
                    self._report("synthesize",
                                 0.1 + 0.8 * ((i + 1) / total),
                                 f"Fish Speech error seg {i+1}: {e}, skipping...")
                    continue

        finally:
            # Clean up GPU memory (hardened — see _hardened_cuda_cleanup docstring)
            try:
                engine = None
                llama_queue = None
                decoder_model = None
            except Exception:
                pass
            _hardened_cuda_cleanup()

        return tts_data

    def _tts_coqui_xtts(self, segments, work_dir=None):
        """Generate TTS using Coqui XTTS v2 — open-source voice cloning.

        Can clone the original speaker's voice and speak in Hindi/Bengali/etc.
        Requires GPU. Uses the extracted audio as voice reference.
        """
        if work_dir is None:
            work_dir = self.cfg.work_dir
        import torch
        import torchaudio
        # Accept Coqui TOS automatically (non-interactive server environment)
        os.environ["COQUI_TOS_AGREED"] = "1"
        # Fix Windows encoding: Coqui TTS prints non-ASCII text to stdout/stderr
        os.environ["PYTHONIOENCODING"] = "utf-8"
        import sys, io
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        try:
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
            if hasattr(sys.stderr, 'buffer'):
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except Exception:
            pass  # If wrapping fails, continue with originals
        # PyTorch 2.6+ defaults weights_only=True which breaks Coqui's model loading
        # Use a lock to prevent concurrent pipelines from racing on the global monkey-patch
        import threading
        if not hasattr(Pipeline, '_torch_load_lock'):
            Pipeline._torch_load_lock = threading.Lock()

        from TTS.api import TTS

        self._report("synthesize", 0.02, "Loading XTTS v2 model on GPU (this may take a minute)...")
        tts_data = []
        tts_model = None
        try:
            with Pipeline._torch_load_lock:
                _original_torch_load = torch.load
                torch.load = lambda *args, **kwargs: _original_torch_load(
                    *args, **{**kwargs, "weights_only": False}
                )
                try:
                    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
                finally:
                    torch.load = _original_torch_load

            # Build per-speaker reference map if available
            refs_dir = self.cfg.work_dir / "speaker_refs"
            speaker_ref_map: Dict[str, Path] = {}
            if refs_dir.exists():
                for ref_file in refs_dir.glob("SPEAKER_*.wav"):
                    spk_id = ref_file.stem  # e.g. "SPEAKER_00"
                    speaker_ref_map[spk_id] = ref_file
                if speaker_ref_map:
                    self._report("synthesize", 0.04,
                                 f"Found {len(speaker_ref_map)} per-speaker voice references for cloning")

            # Default reference: single voice from original audio
            ref_wav = self.cfg.work_dir / "voice_ref.wav"
            audio_raw = self.cfg.work_dir / "audio_raw.wav"
            if not ref_wav.exists() and audio_raw.exists():
                self._run_proc(
                    [self._ffmpeg, "-y", "-i", str(audio_raw),
                     "-t", "15", "-ar", "22050", "-ac", "1",
                     str(ref_wav)],
                    check=True, capture_output=True,
                )

            # Check if we also have a user-provided reference in voices/
            user_ref = Path(__file__).resolve().parent / "voices" / "my_voice_refs"
            if user_ref.exists():
                ref_files = list(user_ref.glob("*.wav")) + list(user_ref.glob("*.mp3"))
                if ref_files:
                    ref_wav = ref_files[0]
                    self._report("synthesize", 0.05, f"Using custom voice reference: {ref_wav.name}")

            if not ref_wav.exists() and not speaker_ref_map:
                raise RuntimeError("No voice reference available for XTTS voice cloning")

            # Map language codes for XTTS
            XTTS_LANG_MAP = {
                "hi": "hi", "bn": "bn", "ta": "ta", "te": "te",
                "en": "en", "es": "es", "fr": "fr", "de": "de",
                "it": "it", "pt": "pt", "pl": "pl", "tr": "tr",
                "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
                "zh": "zh-cn", "ja": "ja", "ko": "ko", "hu": "hu",
            }
            xtts_lang = XTTS_LANG_MAP.get(self.cfg.target_language, self.cfg.target_language)
            for i, seg in enumerate(segments):
                text = seg.get("text_translated", seg.get("text", "")).strip()
                if not text:
                    continue

                wav_path = work_dir / f"tts_{i:04d}.wav"

                # Pick per-speaker reference if available, else default
                spk_id = seg.get("speaker_id", "")
                seg_ref = speaker_ref_map.get(spk_id, ref_wav) if speaker_ref_map else ref_wav

                try:
                    tts_model.tts_to_file(
                        text=text,
                        speaker_wav=str(seg_ref),
                        language=xtts_lang,
                        file_path=str(wav_path),
                    )

                    # Convert to target sample rate and stereo
                    wav_fixed = work_dir / f"tts_{i:04d}_fixed.wav"
                    self._run_proc(
                        [self._ffmpeg, "-y", "-i", str(wav_path),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(wav_fixed)],
                        check=True, capture_output=True,
                    )
                    shutil.move(str(wav_fixed), str(wav_path))

                except Exception as e:
                    self._report("synthesize", 0.1 + 0.8 * ((i + 1) / len(segments)),
                                 f"XTTS error on seg {i+1}: {e}, skipping...")
                    continue

                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    continue

                tts_dur = self._get_duration(wav_path)
                tts_data.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "wav": wav_path,
                    "duration": tts_dur,
                })
                self._report(
                    "synthesize",
                    0.1 + 0.8 * ((i + 1) / len(segments)),
                    f"Synthesized {i + 1}/{len(segments)} segments (XTTS v2 voice clone)...",
                )
        finally:
            try:
                tts_model = None
            except Exception:
                pass
            _hardened_cuda_cleanup()
            # Detach wrappers before restoring to prevent GC from closing shared buffer
            if sys.stdout is not _orig_stdout:
                try:
                    sys.stdout.detach()
                except Exception:
                    pass
            if sys.stderr is not _orig_stderr:
                try:
                    sys.stderr.detach()
                except Exception:
                    pass
            sys.stdout, sys.stderr = _orig_stdout, _orig_stderr

        return tts_data

    def _ssml_with_pauses(self, text: str, pause_ms: int = 1000) -> str:
        """Wrap text in SSML with break tags after sentence-ending punctuation.

        Inserts a 1-second pause after every . ! ? । ॥ for natural speech rhythm.
        """
        import html as _html
        # Escape XML special chars in text
        safe = _html.escape(text, quote=False)
        # Insert break tag after sentence-ending punctuation (but not at the very end)
        safe = re.sub(
            r'([.!?।॥])(\s+)',
            rf'\1<break time="{pause_ms}ms"/>\2',
            safe
        )
        return f'<speak>{safe}</speak>'

    async def _edge_tts_single(self, text, mp3_path, voice=None, rate=None):
        """Generate a single segment with edge-tts (results cached by text+voice+rate)."""
        import edge_tts
        _voice = voice or self.cfg.tts_voice
        _rate  = rate  or self.cfg.tts_rate

        # No SSML pauses — silence gaps are handled in assembly (post-processing)
        # Injecting pauses into TTS audio itself causes double-gaps
        tts_input = text
        cache_tag = "edge_tts_plain"

        # Check TTS cache (use original text as key for consistency)
        cached_bytes = _cache.get_tts(text, _voice, _rate, cache_tag)
        if cached_bytes is not None:
            Path(mp3_path).write_bytes(cached_bytes)
            return

        comm = edge_tts.Communicate(tts_input, _voice, rate=_rate)
        await comm.save(str(mp3_path))

        # Store in cache
        try:
            _cache.put_tts(text, _voice, _rate, cache_tag, Path(mp3_path).read_bytes())
        except Exception:
            pass

    def _sarvam_tts_single_mp3(self, text: str, mp3_path) -> bool:
        """Synthesize a single segment via Sarvam Bulbul v3 → MP3.

        Returns True if the file was written successfully, False otherwise.
        Uses API key rotation with auto-failover on quota errors.
        """
        import requests as _requests
        import base64

        api_key = get_sarvam_key()
        if not api_key:
            return False

        target = self.cfg.target_language.split("-")[0]
        lang_code = self.SARVAM_LANG_MAP.get(target)
        if not lang_code:
            return False

        # Sarvam max 2500 chars per request
        if len(text) > 2500:
            for sep in ['\u0964', '.', '!', '?', ',', ' ']:
                last = text[:2500].rfind(sep)
                if last > 2000:
                    text = text[:last + 1]
                    break
            else:
                text = text[:2500]

        mp3_path = Path(mp3_path)
        max_retries = _sarvam_keys.count()
        current_key = api_key

        for attempt in range(max(max_retries, 2)):
            try:
                resp = _requests.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={
                        "api-subscription-key": current_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "target_language_code": lang_code,
                        "speaker": "shubh",
                        "model": "bulbul:v3",
                        "pace": 1.0,
                        "temperature": 0.6,
                        "speech_sample_rate": 24000,
                        "output_audio_codec": "mp3",
                    },
                    timeout=30,
                )
                if resp.status_code == 429:
                    _sarvam_keys.report_quota_error(current_key)
                    current_key = get_sarvam_key()
                    continue
                if resp.status_code != 200:
                    print(f"[SARVAM-FB] API error {resp.status_code}: {resp.text[:200]}", flush=True)
                    return False

                result = resp.json()
                audio_b64 = result.get("audios", [None])[0]
                if not audio_b64:
                    return False

                mp3_path.write_bytes(base64.b64decode(audio_b64))
                if mp3_path.exists() and mp3_path.stat().st_size > 200:
                    return True
                return False

            except _requests.exceptions.Timeout:
                continue
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    _sarvam_keys.report_quota_error(current_key)
                    current_key = get_sarvam_key()
                    continue
                print(f"[SARVAM-FB] Single-segment fallback failed: {e}", flush=True)
                return False
        return False

    # ── TTS text prep pass ────────────────────────────────────────────────
    def _prepare_tts_text(self, text: str) -> str:
        """Speech-optimize translated text before passing to TTS engine.

        Separate from the rewritten translation — this is a speech-rendering pass:
        - Insert commas at natural breath points (after 5+ word chunks without pause)
        - Normalize ellipses for natural pause rhythm
        - Expand digits to spoken form (basic: digits surrounded by spaces)
        - Remove overly long clauses (Hindi verbosity reduction)
        - Apply pronunciation dictionary
        """
        if not text:
            return text

        # Skip all text processing when post_tts_level is "none"
        level = getattr(self.cfg, 'post_tts_level', 'full')
        if level == "none" or getattr(self.cfg, 'audio_untouchable', False):
            return text

        # Expand common abbreviations that TTS reads literally
        text = re.sub(r'\bsec\b', 'seconds', text, flags=re.IGNORECASE)
        text = re.sub(r'\bspec\b', 'special', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmins?\b', 'minutes', text, flags=re.IGNORECASE)
        text = re.sub(r'\bhrs?\b', 'hours', text, flags=re.IGNORECASE)
        text = re.sub(r'\bvs\b', 'versus', text, flags=re.IGNORECASE)
        text = re.sub(r'\betc\b', 'etcetera', text, flags=re.IGNORECASE)

        # Normalize ellipses → short pause comma
        text = re.sub(r'\.{2,}', ',', text)

        # Replace EM-dash with comma-pause
        text = text.replace('—', ', ')
        text = text.replace('–', ', ')

        # Expand common digit patterns (1, 2, ..., 99) into Hindi spoken words
        _DIGIT_HI = {
            '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
            '5': 'पाँच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
            '10': 'दस', '11': 'ग्यारह', '12': 'बारह', '13': 'तेरह',
            '14': 'चौदह', '15': 'पंद्रह', '20': 'बीस', '25': 'पच्चीस',
            '30': 'तीस', '50': 'पचास', '100': 'सौ', '1000': 'हज़ार',
        }
        if self.cfg.target_language in ('hi', 'mr', 'bn', 'gu', 'pa', 'ur'):
            for num, word in sorted(_DIGIT_HI.items(), key=lambda x: -len(x[0])):
                text = re.sub(r'(?<!\w)' + re.escape(num) + r'(?!\w)', word, text)

        # Insert comma breath-pause after every 7+ word run with no punctuation
        words = text.split()
        if len(words) > 8:
            result = []
            run = 0
            for w in words:
                result.append(w)
                run += 1
                if re.search(r'[,।!?]$', w):
                    run = 0
                elif run >= 7:
                    result[-1] = w + ','
                    run = 0
            text = ' '.join(result)

        # Clean up double commas and trailing punctuation artefacts
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r',\s*([।!?])', r'\1', text)
        text = text.strip().rstrip(',')

        # Apply pronunciation dictionary
        text = self._apply_pronunciation(text)

        return text

    # ── Auto-rerender logic ───────────────────────────────────────────────
    def _simplify_text_for_retry(self, text: str, attempt: int) -> str:
        """Return a simplified version of TTS text for retry attempts.

        RULE: NEVER drop words. All text must be spoken.
        attempt 1 — add punctuation pauses to help TTS pronunciation
        attempt 2 — normalize special chars, simplify punctuation
        attempt 3 — return text as-is (let it through, don't lose content)
        """
        if attempt == 1:
            # Add comma pauses to break up long runs
            if ',' not in text and '।' not in text:
                words = text.split()
                if len(words) > 5:
                    mid = len(words) // 2
                    words.insert(mid, ',')
                    text = ' '.join(words)
            return text

        elif attempt == 2:
            # Normalize: remove special chars that confuse TTS, keep all words
            text = re.sub(r'["""\'\'`]', '', text)
            text = re.sub(r'[()[\]{}]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        else:  # attempt 3 — return as-is, never drop words
            return text

    def _rerender_edge_segment(
        self, text: str, voice: str, wav_out: Path, expected_dur: float
    ) -> dict:
        """Attempt to render a single segment with edge-tts up to 3 times.

        Strategy:
          Attempt 0: original text
          Attempt 1: punctuation tweak (add breath comma)
          Attempt 2: shorten text by 20%
          Attempt 3: shorten to 50% / first sentence

        Returns the QC result dict of the best attempt.
        Best = passes QC; if none pass, returns last attempt's QC.
        """
        import edge_tts
        best_qc = None
        mp3_tmp = wav_out.with_suffix(".retry.mp3")
        wav_tmp = wav_out.with_suffix(".retry.wav")

        current_text = text
        for attempt in range(4):
            if attempt > 0:
                current_text = self._simplify_text_for_retry(text, attempt)
                if not current_text:
                    break

            try:
                asyncio.run(self._edge_tts_single(current_text, mp3_tmp, voice=voice))
                self._run_proc(
                    [self._ffmpeg, "-y", "-i", str(mp3_tmp),
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     str(wav_tmp)],
                    check=True, capture_output=True,
                )
                mp3_tmp.unlink(missing_ok=True)
                self._enhance_tts_wav(wav_tmp)
                qc = self._qc_check_wav(wav_tmp, expected_duration=expected_dur)
                qc["attempt"] = attempt
                qc["text_used"] = current_text

                if qc["ok"]:
                    # This attempt passed — use it
                    # unlink destination first: Windows raises WinError 183 if target exists
                    wav_out.unlink(missing_ok=True)
                    wav_tmp.replace(wav_out)
                    return qc

                best_qc = qc  # keep last for fallback

            except Exception as e:
                print(f"[Rerender] attempt {attempt} exception: {e}", flush=True)
                wav_tmp.unlink(missing_ok=True)
                mp3_tmp.unlink(missing_ok=True)

        # No attempt passed — keep last wav if it exists, else original stays
        if wav_tmp.exists():
            wav_out.unlink(missing_ok=True)
            wav_tmp.replace(wav_out)
        mp3_tmp.unlink(missing_ok=True)
        if best_qc:
            best_qc["manual_review"] = True
        return best_qc or {"ok": False, "issues": ["all retries failed"], "manual_review": True}

    def _probe_mp3_duration_fast(self, mp3_path: "Path") -> float:
        """Return the duration of an MP3/WAV file in seconds via ffprobe.

        Used by the inline TTS truncation guard to detect Edge-TTS WebSocket
        drops that silently save partial audio. Returns 0.0 on any error so
        callers can treat "unknown" as "don't retry".
        """
        try:
            ffprobe = self._ffmpeg.replace("ffmpeg", "ffprobe") \
                if "ffmpeg" in self._ffmpeg else "ffprobe"
            result = self._run_proc(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(mp3_path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return float((result.stdout or "0").strip() or 0)
        except Exception:
            pass
        return 0.0

    def _count_speech_bursts_mp3(self, mp3_path: "Path") -> int:
        """Count syllable-like speech bursts in an MP3.

        Pipes the file through ffmpeg as 8 kHz mono PCM, then counts rising
        edges in an RMS envelope using a 20 ms window with an adaptive
        threshold (30% of mean active RMS). Each rising edge approximates
        a syllable boundary.

        Hindi averages ~1.5–2.0 syllables per word, so for an N-word sentence
        you expect ~1.5N to ~2N bursts. The truncation guard uses this to
        detect "second half of sentence missing" cases that the duration
        check sometimes misses.

        Returns 0 on any error — callers should treat 0 as "unknown, skip check".
        """
        try:
            import numpy as np
            import subprocess
            # Pipe ffmpeg output as raw signed 16-bit mono 8 kHz PCM
            result = subprocess.run(
                [self._ffmpeg, "-nostdin", "-i", str(mp3_path),
                 "-f", "s16le", "-ar", "8000", "-ac", "1",
                 "-loglevel", "error", "-"],
                capture_output=True, timeout=15,
            )
            if result.returncode != 0 or not result.stdout:
                return 0
            samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            if len(samples) < 160:  # < 20 ms — nothing to measure
                return 0
            # 20 ms RMS frames at 8 kHz = 160 samples per frame
            frame_len = 160
            n_frames = len(samples) // frame_len
            if n_frames < 2:
                return 0
            trimmed = samples[:n_frames * frame_len].reshape(n_frames, frame_len)
            rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
            # Adaptive threshold: 30% of the mean RMS of the LOUD frames only
            # (using all frames would let trailing silence drag the mean down)
            loud = rms[rms > rms.mean() * 0.5]
            if len(loud) == 0:
                return 0
            threshold = max(0.005, float(loud.mean()) * 0.30)
            above = rms > threshold
            # Count rising edges (silence → speech transitions)
            rising = np.logical_and(above[1:], np.logical_not(above[:-1]))
            bursts = int(rising.sum()) + (1 if above[0] else 0)
            return bursts
        except Exception:
            return 0

    # Counter for truncation incidents (logged once per job in _tts_edge)
    _tts_trunc_count = 0

    def _post_tts_word_match_verify(self, tts_data, segments):
        """Post-TTS exact word-count verification via Whisper-tiny.

        For each generated TTS segment:
        1. Transcribe the WAV with Whisper-tiny in target language
        2. Count the actual words spoken
        3. Compare to seg["_expected_words"] with tolerance window
        4. If mismatch outside tolerance, mark for retry
        5. Re-TTS each mismatched segment (up to 3 attempts each)
        6. Re-verify after retry; if still mismatched, accept whatever we got
           rather than dropping the segment

        Runs ONLY if self.cfg.tts_word_match_verify is True.

        Cost: ~150-300 ms per segment on CPU. Linear with segment count.
        For a 5000-segment job, expect 15-25 minutes added.

        Inputs:
            tts_data: list of dicts from _tts_edge with "wav" and "_seg_idx" keys
            segments: original segments list (used to look up text + re-call TTS)

        Mutates tts_data in place — replaces wav paths for retried segments.
        Returns dict with stats: {checked, mismatched, retried, recovered, final_bad}.
        """
        if not getattr(self.cfg, "tts_word_match_verify", False):
            return None
        if not tts_data:
            return None

        tolerance = float(getattr(self.cfg, "tts_word_match_tolerance", 0.15))
        if tolerance < 0 or tolerance > 1:
            tolerance = 0.15

        # ── Pick the Whisper model based on user setting + hardware ──
        # "auto":  turbo on GPU, tiny on CPU (best speed/accuracy at each tier)
        # "tiny":  always tiny (fastest on CPU, lowest accuracy)
        # "turbo": always turbo (best accuracy, much slower on CPU)
        model_pref = (getattr(self.cfg, "tts_word_match_model", "auto") or "auto").lower()

        # Detect CUDA availability ONCE so model + fallback agree
        _cuda_available = False
        try:
            import torch as _torch
            _cuda_available = bool(_torch.cuda.is_available())
        except Exception:
            _torch = None

        if model_pref == "tiny":
            chosen_model = "tiny"
        elif model_pref == "turbo":
            chosen_model = "large-v3-turbo"
        else:  # auto
            chosen_model = "large-v3-turbo" if _cuda_available else "tiny"

        device = "cuda" if _cuda_available else "cpu"
        compute = "float16" if _cuda_available else "int8"

        # Pre-clean stale GPU allocations from prior steps in this process so
        # we don't OOM on model load when CUDA memory is fragmented
        if _cuda_available and _torch is not None:
            try:
                _torch.cuda.empty_cache()
                _torch.cuda.ipc_collect()
            except Exception:
                pass

        # ── Load Whisper model with GPU→CPU fallback on OOM ──
        wmodel = None
        try:
            from faster_whisper import WhisperModel
            self._report("synthesize", 0.91,
                         f"Word-match verify: loading Whisper {chosen_model} on {device.upper()}...")
            try:
                wmodel = WhisperModel(chosen_model, device=device, compute_type=compute)
                print(f"[WORD-VERIFY] Loaded Whisper {chosen_model} on {device.upper()} ({compute})",
                      flush=True)
            except Exception as gpu_err:
                # GPU OOM or model not found on GPU — fall back to CPU
                err_str = str(gpu_err).lower()
                if device == "cuda" and ("out of memory" in err_str or "cuda" in err_str):
                    print(f"[WORD-VERIFY] GPU load failed ({gpu_err}) — falling back to CPU",
                          flush=True)
                    if _torch is not None:
                        try:
                            _torch.cuda.empty_cache()
                        except Exception:
                            pass
                    # On CPU, downgrade turbo to tiny because turbo is too slow
                    # on CPU (~1.5s/seg) — tiny is the practical choice.
                    if chosen_model == "large-v3-turbo":
                        chosen_model = "tiny"
                        print("[WORD-VERIFY] Auto-downgraded turbo -> tiny on CPU "
                              "(turbo too slow without GPU)", flush=True)
                    wmodel = WhisperModel(chosen_model, device="cpu", compute_type="int8")
                    device = "cpu"
                    print(f"[WORD-VERIFY] Loaded Whisper {chosen_model} on CPU (int8)",
                          flush=True)
                else:
                    raise
        except Exception as e:
            print(f"[WORD-VERIFY] Could not load any Whisper model: {e} — "
                  f"verification disabled for this run", flush=True)
            return None

        target_lang = self.cfg.target_language or "hi"

        def _count_words_in_wav(wav_path):
            """Transcribe a WAV with Whisper and return (word_count, transcript).

            Returns (-1, "") on any error so callers can skip the segment.
            The transcript is captured so the diagnostic dump can show
            expected-vs-actual side-by-side for problem segments.
            """
            try:
                seg_iter, _info = wmodel.transcribe(
                    str(wav_path),
                    language=target_lang,
                    vad_filter=False,        # we want the FULL transcription
                    beam_size=1,             # tiny+greedy = fastest
                    best_of=1,
                    condition_on_previous_text=False,
                    word_timestamps=False,
                )
                text_parts = [s.text.strip() for s in seg_iter if s.text]
                full = " ".join(text_parts)
                return (len(full.split()) if full else 0, full)
            except Exception as e:
                print(f"[WORD-VERIFY] transcribe error: {e}", flush=True)
                return (-1, "")

        # ── Diagnostic dump: collect problem segments for user inspection ──
        # Each problem segment gets:
        #   1. A copy of its final WAV in backend/logs/word_verify_samples_<job_id>/
        #   2. An entry in the manifest JSON with expected_text, whisper_transcript,
        #      expected_words, actual_words, verdict, wav_copy_path
        # The user can read the manifest and listen to the WAVs to verify
        # by ear whether the audio actually has missing words or whether
        # the mismatch is Whisper tokenization drift.
        #
        # Cleanup: evict diagnostic dirs older than 7 days to prevent disk bloat.
        try:
            import time as _evict_time
            logs_dir = Path(__file__).resolve().parent / "logs"
            _evict_cutoff = _evict_time.time() - 7 * 86400
            for d in logs_dir.glob("word_verify_samples_*"):
                try:
                    if d.is_dir() and d.stat().st_mtime < _evict_cutoff:
                        import shutil as _evict_sh
                        _evict_sh.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
        except Exception:
            pass
        _diagnostic_dump = []
        try:
            _job_id_for_dump = self.cfg.work_dir.parent.name if self.cfg.work_dir else "unknown"
        except Exception:
            _job_id_for_dump = "unknown"
        _dump_dir = Path(__file__).resolve().parent / "logs" / f"word_verify_samples_{_job_id_for_dump}"
        try:
            _dump_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        def _save_problem_sample(seg_idx, expected_text, expected_words,
                                  actual_words, transcript, wav_path, verdict):
            """Copy the WAV + record the expected-vs-transcript comparison."""
            try:
                import shutil as _sh
                dst_wav = _dump_dir / f"seg_{seg_idx:04d}_{verdict}.wav"
                if wav_path and Path(wav_path).exists():
                    _sh.copy2(wav_path, dst_wav)
                    wav_str = str(dst_wav)
                else:
                    wav_str = None
                _diagnostic_dump.append({
                    "seg_idx":           int(seg_idx),
                    "verdict":           verdict,
                    "expected_text":     expected_text,
                    "expected_words":    int(expected_words),
                    "whisper_transcript": transcript,
                    "actual_words":      int(actual_words),
                    "off_by":            int(actual_words) - int(expected_words),
                    "wav_copy":          wav_str,
                })
            except Exception as e:
                print(f"[WORD-VERIFY] dump save failed for seg {seg_idx}: {e}", flush=True)

        def _retry_tts_segment(seg, sub_idx, strategy="plain"):
            """Re-call Edge-TTS for a single segment with a specific strategy.

            Keeps the voice as Madhur — the user explicitly chose voice
            consistency over falling back to Google/Fish-Speech. Instead,
            each retry tries a different RATE to coax Edge-TTS into producing
            correct output. Text stays unchanged across rate retries; the
            only text transform is `chunked`, kept as the last-resort option
            for content issues that no rate change can fix.

            Strategies (in escalation order, per user request 2026-04-12):
              plain    — User's selected rate, same text. Original behavior.
              reduced  — Halfway between user's rate and 0%. If user picked
                         +30%, this retries at +15%. Gentler pacing often
                         helps Edge-TTS stream without WebSocket drops.
              natural  — Force rate="+0%". Edge-TTS's most stable rate;
                         used as the last rate-based fallback.
              chunked  — Last resort. Split text at SENTENCE boundaries
                         (never inside a sentence), synthesize each sentence
                         at user's rate, concatenate via ffmpeg. Handles
                         cases where a specific phrase in a long multi-
                         sentence segment is tripping Edge-TTS up.

            Returns the new WAV path or None on failure.
            """
            try:
                import edge_tts
                import asyncio
                raw_text = seg.get("text_translated", seg.get("text", "")).strip()
                if not raw_text:
                    return None
                text = self._prepare_tts_text(raw_text)
                voice = self.cfg.tts_voice  # still Madhur — consistent voice
                rate = self.cfg.tts_rate
                retry_dir = self.cfg.work_dir / "tts_word_verify_retry"
                retry_dir.mkdir(exist_ok=True)
                mp3 = retry_dir / f"retry_{sub_idx:04d}_{strategy}.mp3"

                # ── Apply strategy-specific rate changes (text stays the same) ──
                effective_text = text
                effective_rate = rate

                if strategy == "reduced":
                    # Halve the user's rate toward natural (+0%). Example:
                    # user picked "+30%" -> reduced = "+15%".
                    # user picked "+50%" -> reduced = "+25%".
                    # user picked "-20%" -> reduced = "-10%".
                    # user picked "+0%"  -> reduced = "+0%" (no effective change)
                    import re as _re2
                    m = _re2.match(r"([+-]?)(\d+)%", rate or "+0%")
                    if m:
                        sign = m.group(1) or "+"
                        val = int(m.group(2))
                        if sign == "-":
                            val = -val
                        reduced_val = val // 2  # int division toward zero
                        if reduced_val > 0:
                            effective_rate = f"+{reduced_val}%"
                        elif reduced_val == 0:
                            effective_rate = "+0%"
                        else:
                            effective_rate = f"{reduced_val}%"
                    else:
                        effective_rate = "+0%"  # parse failure, fall to natural

                elif strategy == "natural":
                    # Force natural rate for maximum Edge-TTS reliability.
                    # Edge-TTS WebSocket drops increase with rate variance;
                    # +0% is the most stable.
                    effective_rate = "+0%"

                elif strategy == "chunked":
                    # Last resort: split at SENTENCE boundaries (never inside
                    # a sentence), synthesize each sentence at the user's
                    # chosen rate, concatenate. Handles content-specific
                    # issues no rate change can fix.
                    return _retry_chunked(seg, sub_idx, effective_text, voice, rate, retry_dir)

                # Strategies plain / pauses / natural: single-call path
                async def _do():
                    comm = edge_tts.Communicate(effective_text, voice, rate=effective_rate)
                    await comm.save(str(mp3))

                asyncio.run(_do())
                if not mp3.exists() or mp3.stat().st_size < 200:
                    return None

                # Convert MP3 → WAV at pipeline sample rate
                new_wav = retry_dir / f"retry_{sub_idx:04d}_{strategy}.wav"
                self._run_proc(
                    [self._ffmpeg, "-y", "-i", str(mp3),
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     str(new_wav)],
                    check=True, capture_output=True,
                )
                try:
                    mp3.unlink(missing_ok=True)
                except Exception:
                    pass
                return new_wav if new_wav.exists() else None
            except Exception as e:
                print(f"[WORD-VERIFY] retry ({strategy}) call failed: {e}", flush=True)
                return None

        def _retry_chunked(seg, sub_idx, text, voice, rate, retry_dir):
            """Chunked retry: split text into SENTENCE-level pieces, synthesize
            each sentence separately, concatenate via ffmpeg concat demuxer.

            ── Sentence-scope preservation (user-requested 2026-04-12) ──
            This strategy now splits ONLY at sentence boundaries (।/./!/?),
            NEVER inside a sentence. It reuses the same
            _split_text_on_sentence_boundary helper as the main sentence
            pre-splitter at Stage 4, so both stages are guaranteed consistent:
            each chunk sent to Edge-TTS is exactly one complete sentence.

            Worst case: a single very long sentence (>60 chars, no internal
            punctuation) — it stays as ONE chunk even if it's 200 chars.
            Splitting it mid-word would produce audibly broken prosody and
            break the sentence's natural intonation, which the user has
            explicitly said is unacceptable. We accept the drop risk on the
            single-sentence case rather than chop up natural prosody.

            Falls through to a single-call path when the text contains only
            one sentence (no sentence boundary to split on).
            """
            try:
                import edge_tts
                import asyncio

                # ── Split at sentence boundaries, reusing the existing helper ──
                # Returns a list of (chunk, terminator) tuples where terminator
                # is the `।`/`.`/`!`/`?` character that ended the sentence.
                pieces = self._split_text_on_sentence_boundary(text)

                # If the splitter returned 0 or 1 piece, there's no sentence
                # boundary to split on — the whole text is one sentence.
                # Fall through to a single Edge-TTS call with the full text.
                if len(pieces) < 2:
                    mp3 = retry_dir / f"retry_{sub_idx:04d}_chunked.mp3"
                    async def _do_single():
                        comm = edge_tts.Communicate(text, voice, rate=rate)
                        await comm.save(str(mp3))
                    asyncio.run(_do_single())
                    if not mp3.exists() or mp3.stat().st_size < 200:
                        return None
                    new_wav = retry_dir / f"retry_{sub_idx:04d}_chunked.wav"
                    self._run_proc(
                        [self._ffmpeg, "-y", "-i", str(mp3),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(new_wav)],
                        check=True, capture_output=True,
                    )
                    mp3.unlink(missing_ok=True)
                    return new_wav if new_wav.exists() else None

                # ── Synthesize each sentence separately ──
                # Each piece is one full sentence + its original terminator.
                # Sentence scope is preserved: nothing is ever split mid-sentence.
                print(f"[WORD-VERIFY] chunked retry: splitting into "
                      f"{len(pieces)} sentences (scope preserved)", flush=True)
                chunk_mp3s = []
                ok = True
                for ci, (sentence_text, terminator) in enumerate(pieces):
                    # Reattach the terminator so Edge-TTS generates the natural
                    # sentence-end intonation (same as the main pre-splitter).
                    full_sentence = sentence_text + (terminator or "")
                    c_mp3 = retry_dir / f"retry_{sub_idx:04d}_sent{ci:02d}.mp3"
                    async def _do_sentence(_ct=full_sentence, _cm=c_mp3):
                        comm = edge_tts.Communicate(_ct, voice, rate=rate)
                        await comm.save(str(_cm))
                    try:
                        asyncio.run(_do_sentence())
                        if not c_mp3.exists() or c_mp3.stat().st_size < 200:
                            ok = False
                            break
                        chunk_mp3s.append(c_mp3)
                    except Exception:
                        ok = False
                        break

                if not ok or not chunk_mp3s:
                    # Cleanup partial chunks
                    for p in chunk_mp3s:
                        try: p.unlink(missing_ok=True)
                        except Exception: pass
                    return None

                # Concatenate all sentence mp3s via ffmpeg concat demuxer
                # (no re-encode, instant concatenation, preserves audio quality)
                final_mp3 = retry_dir / f"retry_{sub_idx:04d}_chunked.mp3"
                concat_list = retry_dir / f"retry_{sub_idx:04d}_chunked_list.txt"
                concat_list.write_text(
                    "\n".join(f"file '{p.name}'" for p in chunk_mp3s),
                    encoding="utf-8",
                )
                try:
                    self._run_proc(
                        [self._ffmpeg, "-y", "-f", "concat",
                         "-safe", "0", "-i", str(concat_list),
                         "-c", "copy", str(final_mp3)],
                        check=True, capture_output=True,
                    )
                except Exception as e:
                    print(f"[WORD-VERIFY] chunked concat failed: {e}", flush=True)
                    return None
                finally:
                    # Cleanup intermediate files
                    for p in chunk_mp3s:
                        try: p.unlink(missing_ok=True)
                        except Exception: pass
                    try: concat_list.unlink(missing_ok=True)
                    except Exception: pass

                # Convert concatenated MP3 → WAV at pipeline sample rate
                new_wav = retry_dir / f"retry_{sub_idx:04d}_chunked.wav"
                self._run_proc(
                    [self._ffmpeg, "-y", "-i", str(final_mp3),
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     str(new_wav)],
                    check=True, capture_output=True,
                )
                try:
                    final_mp3.unlink(missing_ok=True)
                except Exception:
                    pass
                return new_wav if new_wav.exists() else None
            except Exception as e:
                print(f"[WORD-VERIFY] chunked retry failed: {e}", flush=True)
                return None

        # Wrap entire scan + retry in try/finally so the model is ALWAYS
        # released — otherwise GPU VRAM (1.5 GB for turbo) leaks until process exit.
        total = len(tts_data)
        result_stats = None
        try:
            self._report("synthesize", 0.92,
                         f"Word-match verify: scanning {total} segments via Whisper {chosen_model}...")
            mismatched_indices = []
            ok_count = 0
            unknown_count = 0
            for i, tts in enumerate(tts_data):
                wav = tts.get("wav")
                if not wav or not Path(wav).exists():
                    continue

                # Look up expected word count from the source segment
                seg_idx = tts.get("_seg_idx", i)
                seg = segments[seg_idx] if 0 <= seg_idx < len(segments) else None
                expected = (seg.get("_expected_words", 0) if seg else 0) or 0
                if expected == 0:
                    continue  # nothing to verify

                actual, actual_text = _count_words_in_wav(wav)
                if actual < 0:
                    unknown_count += 1
                    continue

                # Tolerance window — percent with an absolute ±1 floor.
                # Pure percent collapses to zero for short segments: at 15%,
                # target=2 gives window [2,2] which any Whisper drift fails.
                # Real cause is Whisper vs Python tokenization differences on
                # Hindi particles/compounds, NOT a TTS problem. The ±1 floor
                # lets natural tokenization drift through without flagging.
                pct_lo = int(round(expected * (1.0 - tolerance)))
                pct_hi = int(round(expected * (1.0 + tolerance)))
                lo = max(1, min(pct_lo, expected - 1))
                hi = max(1, max(pct_hi, expected + 1))
                tts["_word_actual"] = actual
                tts["_word_expected"] = expected
                tts["_word_transcript"] = actual_text  # used by dump

                if lo <= actual <= hi:
                    ok_count += 1
                    tts["_word_verify"] = "ok"
                else:
                    tts["_word_verify"] = "mismatch"
                    mismatched_indices.append(i)
                    expected_text_full = (seg.get('text_translated') or seg.get('text') or '')
                    if len(mismatched_indices) <= 10:
                        print(f"[WORD-VERIFY] Seg {i}: expected {expected}, "
                              f"got {actual} (tolerance {lo}-{hi}) — flagged for retry.",
                              flush=True)
                        print(f"[WORD-VERIFY]   expected_text: {expected_text_full[:140]!r}", flush=True)
                        print(f"[WORD-VERIFY]   whisper_heard: {actual_text[:140]!r}", flush=True)

                # Throttled per-100 progress update
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    self._report("synthesize", 0.92,
                                 f"Word-verify scan: {i+1}/{total} "
                                 f"(ok={ok_count}, mismatch={len(mismatched_indices)})")

            if not mismatched_indices:
                self._report("synthesize", 0.94,
                             f"Word verification: all {ok_count}/{total} segments OK")
                print(f"[WORD-VERIFY] All {ok_count} segments passed (tolerance +/-{tolerance*100:.0f}%)",
                      flush=True)
                result_stats = {
                    "checked":    ok_count,
                    "mismatched": 0,
                    "retried":    0,
                    "recovered":  0,
                    "final_bad":  0,
                    "model":      chosen_model,
                    "device":     device,
                }
                return result_stats

            # ── Pass 2: retry mismatched segments (up to 3 attempts each) ──
            self._report("synthesize", 0.93,
                         f"Word verification: retrying {len(mismatched_indices)} "
                         f"mismatched segments (sequential)...")
            recovered = 0
            final_bad = 0
            retried_total = 0

            for ord_, idx in enumerate(mismatched_indices):
                tts = tts_data[idx]
                seg_idx = tts.get("_seg_idx", idx)
                if not (0 <= seg_idx < len(segments)):
                    final_bad += 1
                    continue
                seg = segments[seg_idx]
                expected = tts.get("_word_expected", 0)
                # Same ±1-word floor as pass-1 (see comment above)
                pct_lo = int(round(expected * (1.0 - tolerance)))
                pct_hi = int(round(expected * (1.0 + tolerance)))
                lo = max(1, min(pct_lo, expected - 1))
                hi = max(1, max(pct_hi, expected + 1))

                best_wav = None
                best_actual = tts.get("_word_actual", -1)
                best_transcript = tts.get("_word_transcript", "")
                best_strategy = "original"

                # ── Four escalating retry strategies (rate-based, user-requested) ──
                # Each attempt tries a DIFFERENT rate to coax Edge-TTS into
                # producing the correct output. Voice stays Madhur in all of
                # them. Text stays unchanged until the very last attempt
                # (chunked), which splits only at sentence boundaries.
                # Escalation order:
                #   1. plain    — user's selected rate (baseline retry)
                #   2. reduced  — halfway toward 0% (gentler pacing)
                #   3. natural  — force rate="+0%" (most stable rate)
                #   4. chunked  — sentence-boundary split at user's rate
                retry_strategies = ["plain", "reduced", "natural", "chunked"]
                for attempt_idx, strategy in enumerate(retry_strategies):
                    retried_total += 1
                    new_wav = _retry_tts_segment(
                        seg, ord_ * 10 + attempt_idx, strategy=strategy,
                    )
                    if new_wav is None:
                        continue
                    new_actual, new_transcript = _count_words_in_wav(new_wav)
                    if new_actual < 0:
                        continue
                    # Did this attempt land inside the tolerance window?
                    if lo <= new_actual <= hi:
                        best_wav = new_wav
                        best_actual = new_actual
                        best_transcript = new_transcript
                        best_strategy = strategy
                        print(f"[WORD-VERIFY] Seg {idx} recovered via strategy "
                              f"'{strategy}' (attempt {attempt_idx+1}/4)", flush=True)
                        break
                    # Not yet in window — keep the closest one
                    if best_actual < 0 or abs(new_actual - expected) < abs(best_actual - expected):
                        best_wav = new_wav
                        best_actual = new_actual
                        best_transcript = new_transcript
                        best_strategy = strategy

                expected_text_full = (seg.get('text_translated') or seg.get('text') or '')

                if best_wav is not None and lo <= best_actual <= hi:
                    # Replace the original wav with the rescued one
                    old_wav = tts.get("wav")
                    tts["wav"] = best_wav
                    tts["_word_actual"] = best_actual
                    tts["_word_transcript"] = best_transcript
                    tts["_word_verify"] = "ok_after_retry"
                    recovered += 1
                    print(f"[WORD-VERIFY] Seg {idx} recovered: now {best_actual} words "
                          f"(target {expected}, window {lo}-{hi})", flush=True)
                    # Save diagnostic sample for "recovered" verdict
                    _save_problem_sample(
                        idx, expected_text_full, expected, best_actual,
                        best_transcript, best_wav, "recovered",
                    )
                    # Best-effort cleanup of the old wav
                    try:
                        if old_wav and Path(old_wav).exists() and Path(old_wav) != Path(best_wav):
                            Path(old_wav).unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    # Keep best-effort attempt if we got SOMETHING closer; else
                    # leave the original wav untouched.
                    if best_wav is not None and best_actual >= 0:
                        old_wav = tts.get("wav")
                        if abs(best_actual - expected) < abs(tts.get("_word_actual", 0) - expected):
                            tts["wav"] = best_wav
                            tts["_word_actual"] = best_actual
                            tts["_word_transcript"] = best_transcript
                    tts["_word_verify"] = "still_mismatch"
                    final_bad += 1
                    print(f"[WORD-VERIFY] Seg {idx} STILL MISMATCH after all 4 strategies "
                          f"(best={best_strategy}): {best_actual} words (target {expected})",
                          flush=True)
                    print(f"[WORD-VERIFY]   expected_text: {expected_text_full[:140]!r}", flush=True)
                    print(f"[WORD-VERIFY]   whisper_heard: {tts.get('_word_transcript', '')[:140]!r}", flush=True)
                    # Save diagnostic sample for "still_mismatch" verdict
                    _save_problem_sample(
                        idx, expected_text_full, expected,
                        tts.get("_word_actual", 0),
                        tts.get("_word_transcript", ""),
                        tts.get("wav"),
                        "still_mismatch",
                    )

            # ── Write diagnostic manifest JSON ──
            # Lists every problem segment (recovered + still_mismatch) with
            # expected_text, whisper_transcript, word counts, and the path to
            # the copied WAV. The user can read the manifest and listen to
            # the WAVs to verify by ear whether the audio is actually missing
            # words or whether the mismatch is Whisper tokenization drift.
            if _diagnostic_dump:
                try:
                    import json as _json
                    from datetime import datetime as _dt
                    manifest_path = _dump_dir / "manifest.json"
                    manifest = {
                        "generated_at":   _dt.now().isoformat(),
                        "job_id":         _job_id_for_dump,
                        "model":          chosen_model,
                        "device":         device,
                        "tolerance":      tolerance,
                        "summary": {
                            "checked":             total,
                            "total_ok":            ok_count,
                            "recovered":           recovered,
                            "still_mismatch":      final_bad,
                            "problems_in_manifest": len(_diagnostic_dump),
                        },
                        "how_to_read": (
                            "Each entry is a problem segment. Compare "
                            "'expected_text' (the Hindi text TTS was told to "
                            "speak) with 'whisper_transcript' (what Whisper "
                            "heard). Listen to 'wav_copy' by ear. If both "
                            "texts are semantically the same and the audio "
                            "sounds complete, the mismatch is tokenization "
                            "drift, NOT missing audio. If the transcript is "
                            "clearly shorter than expected_text or the audio "
                            "cuts off mid-word, it's a real TTS drop."
                        ),
                        "segments": _diagnostic_dump,
                    }
                    manifest_path.write_text(
                        _json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"[WORD-VERIFY] Diagnostic dump: {len(_diagnostic_dump)} "
                          f"problem segments saved to {_dump_dir}", flush=True)
                    print(f"[WORD-VERIFY]   Manifest: {manifest_path}", flush=True)
                except Exception as e:
                    print(f"[WORD-VERIFY] Failed to write diagnostic manifest: {e}", flush=True)

            # ── Final summary ──
            msg = (f"Word verification done: {ok_count} ok / "
                   f"{recovered} recovered / {final_bad} still mismatched")
            self._report("synthesize", 0.95, msg)
            print(f"[WORD-VERIFY] {msg} (retried {retried_total} TTS calls, "
                  f"model={chosen_model} device={device})", flush=True)

            result_stats = {
                "checked":    total,
                "mismatched": len(mismatched_indices),
                "retried":    retried_total,
                "recovered":  recovered,
                "final_bad":  final_bad,
                "model":      chosen_model,
                "device":     device,
            }
            return result_stats
        finally:
            # Always free the Whisper model so GPU VRAM is reclaimed for the
            # next pipeline step (assembly, encoding) or the next job.
            #
            # Hardened cleanup (after PID 20328 crashed during this block on
            # 2026-04-09): the original `del wmodel` could race with the
            # assembly step about to grab NVENC and cause a CUDA abort. Now:
            #
            #   1. Drop our reference inside its own try/except so a failed
            #      __del__ on the model never propagates.
            #   2. Force a Python gc.collect() before touching CUDA so any
            #      lingering CUDA tensor references are flushed first.
            #   3. cuda.synchronize() before empty_cache() so all in-flight
            #      kernels complete first — empty_cache() on a busy stream
            #      is what triggered the abort.
            #   4. Sleep briefly so the CUDA driver has time to actually
            #      release the memory blocks before the next pipeline step
            #      tries to allocate them.
            #   5. Each step is independently try/except'd — none can take
            #      down the process even if individual cleanup fails.
            try:
                wmodel = None
            except Exception:
                pass
            try:
                import gc as _gc
                _gc.collect()
            except Exception:
                pass
            if _cuda_available and _torch is not None:
                try:
                    _torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    _torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    _torch.cuda.ipc_collect()
                except Exception:
                    pass
                try:
                    import time as _t
                    _t.sleep(0.5)  # let the CUDA driver settle
                except Exception:
                    pass
            print("[WORD-VERIFY] Cleanup complete (model freed, CUDA cache emptied)",
                  flush=True)

    # ─────────────────────────────────────────────────────────────────────
    # Long-Segment Trace Watchdog
    # ─────────────────────────────────────────────────────────────────────
    # Records the full pipeline lifecycle of every "long" segment (segments
    # over the configured word threshold). Each event is a dict appended
    # to self._long_seg_traces[seg_idx]. At the end of run(), all traces
    # are written to backend/logs/long_segment_trace_<job_id>.json so the
    # user can grep/analyze where each long segment was modified or dropped.
    #
    # The point of this watchdog: when a long segment's audio comes out
    # truncated, the user can't tell if Edge-TTS dropped it, or the WebSocket
    # cut, or the time-stretch hard-clipped it, or the assembly cut it. This
    # records EVERY transformation so the failure point is unambiguous.
    #
    # Cost: a few microseconds per event. Only long segments are recorded.
    # Storage: ~1-3 KB per long segment in the JSON report.

    def _trace_init(self):
        """Initialize the trace store on the Pipeline. Idempotent."""
        if not hasattr(self, "_long_seg_traces"):
            self._long_seg_traces = {}  # seg_idx -> list of event dicts

    def _trace_is_long(self, word_count: int) -> bool:
        """True if this segment qualifies for tracing (configured threshold)."""
        if not getattr(self.cfg, "long_segment_trace", True):
            return False
        threshold = int(getattr(self.cfg, "long_segment_threshold_words", 15) or 15)
        return word_count >= threshold

    def _trace_event(self, seg_idx: int, stage: str, **fields):
        """Append a stage event to a segment's trace. Best-effort, never raises.

        stage: short label like 'pre_tts', 'post_save', 'post_wav_convert',
               'post_truncation_guard', 'post_word_verify', 'post_speed_fit',
               'post_assembly_cut', 'final'.
        fields: any keyword args become stage-specific data on the event.
        """
        try:
            self._trace_init()
            import time as _time
            event = {"stage": stage, "t": round(_time.time(), 3), **fields}
            traces = self._long_seg_traces.setdefault(seg_idx, [])
            traces.append(event)
        except Exception:
            pass  # watchdog must NEVER crash the pipeline

    def _trace_record_pre_tts(self, seg_idx: int, seg: dict, text: str, word_count: int):
        """Record pre-TTS state: text characteristics, expected word count, slot dur."""
        if not self._trace_is_long(word_count):
            return
        try:
            slot_dur = max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
        except Exception:
            slot_dur = 0.0
        # Text fingerprints — useful for spotting mixed-script or special-char issues
        char_count = len(text)
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
        has_latin = any(('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in text)
        is_mixed_script = has_devanagari and has_latin
        # Sentence count — splits on `।` `?` `!` `.` followed by space/EOL
        import re as _re
        sentence_chunks = [s.strip() for s in _re.split(r'(?<=[।?!.])\s+|(?<=[।?!.])$', text) if s.strip()]
        sentence_count = len(sentence_chunks) or 1
        # Required WPM to fit
        required_wpm = (word_count * 60.0 / slot_dur) if slot_dur > 0 else 0.0

        self._trace_event(
            seg_idx, "pre_tts",
            text_preview=text[:120],
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            sentence_words=[len(s.split()) for s in sentence_chunks],
            slot_dur_sec=round(slot_dur, 3),
            required_wpm=round(required_wpm, 1),
            is_mixed_script=is_mixed_script,
            has_devanagari=has_devanagari,
            has_latin=has_latin,
        )

    def _trace_record_after_save(self, seg_idx: int, word_count: int,
                                  mp3_path, exception: Optional[BaseException] = None):
        """Record post-comm.save() state: file size, exception, raised/clean."""
        if not self._trace_is_long(word_count):
            return
        info = {"path": str(mp3_path)}
        try:
            from pathlib import Path as _P
            p = _P(mp3_path)
            if p.exists():
                info["file_size_bytes"] = p.stat().st_size
                info["file_exists"] = True
            else:
                info["file_size_bytes"] = 0
                info["file_exists"] = False
        except Exception:
            info["file_size_bytes"] = -1
            info["file_exists"] = False

        if exception is not None:
            info["exception_type"] = type(exception).__name__
            info["exception_msg"] = str(exception)[:300]
            info["raised"] = True
        else:
            info["raised"] = False

        self._trace_event(seg_idx, "post_save", **info)

    def _trace_record_truncation_guard(self, seg_idx: int, word_count: int,
                                       outcome: str, **measurements):
        """Record truncation guard verdict (duration probe + burst count).

        outcome: 'pass' | 'duration_short' | 'bursts_low' | 'recovered_after_retry'
        measurements: actual_dur, min_dur, actual_bursts, expected_bursts, etc.
        """
        if not self._trace_is_long(word_count):
            return
        self._trace_event(seg_idx, "post_truncation_guard",
                          outcome=outcome, **measurements)

    def _trace_record_wav_convert(self, seg_idx: int, word_count: int,
                                  wav_path, duration_sec: float):
        """Record post-MP3-to-WAV state: final WAV file + measured duration."""
        if not self._trace_is_long(word_count):
            return
        self._trace_event(seg_idx, "post_wav_convert",
                          wav_path=str(wav_path),
                          duration_sec=round(duration_sec, 3))

    def _trace_record_word_verify(self, seg_idx: int, word_count: int,
                                  expected: int, actual: int, outcome: str,
                                  retried: int = 0):
        """Record post-Whisper-verification state.

        outcome: 'ok' | 'mismatch_flagged' | 'recovered' | 'still_mismatch'
        """
        if not self._trace_is_long(word_count):
            return
        self._trace_event(seg_idx, "post_word_verify",
                          expected_words=expected,
                          actual_words=actual,
                          outcome=outcome,
                          retried_attempts=retried)

    def _trace_record_speed_fit(self, seg_idx: int, word_count: int,
                                pre_dur: float, post_dur: float, ratio: float,
                                clamped: bool):
        """Record post-speed-fit state: did time-stretch shrink/extend the audio?

        If clamped=True the time-stretch hit SPEED_MAX or SPEED_MIN and the
        audio may have been shortened to fit the slot. THIS IS THE MOST
        COMMON cause of "TTS read it but final video is missing words".
        """
        if not self._trace_is_long(word_count):
            return
        self._trace_event(seg_idx, "post_speed_fit",
                          pre_dur=round(pre_dur, 3),
                          post_dur=round(post_dur, 3),
                          ratio_applied=round(ratio, 3),
                          clamped_to_limit=clamped,
                          warning="time_stretch_clamped_may_lose_words" if clamped else None)

    def _trace_record_assembly_cut(self, seg_idx: int, word_count: int,
                                   final_dur: float, slot_dur: float, was_cut: bool):
        """Record assembly-stage state: was the audio cut to fit a slot?"""
        if not self._trace_is_long(word_count):
            return
        self._trace_event(seg_idx, "post_assembly",
                          final_dur=round(final_dur, 3),
                          slot_dur=round(slot_dur, 3),
                          was_cut_at_slot=was_cut,
                          warning="audio_truncated_at_slot_boundary" if was_cut else None)

    def _trace_write_report(self, job_id: str = "unknown"):
        """Write the long-segment trace report to backend/logs/.

        Called from run()'s finally block. Best-effort: never raises.

        Each segment's report shows the full lifecycle so you can read down
        the events list and pinpoint which stage modified or dropped audio.
        """
        if not getattr(self.cfg, "long_segment_trace", True):
            return
        if not hasattr(self, "_long_seg_traces") or not self._long_seg_traces:
            return

        # Evict old trace files (>7 days) to prevent disk bloat
        try:
            import time as _tr_time
            from pathlib import Path as _TrP
            logs_dir = _TrP(__file__).resolve().parent / "logs"
            _cutoff = _tr_time.time() - 7 * 86400
            for f in logs_dir.glob("long_segment_trace_*.json"):
                try:
                    if f.stat().st_mtime < _cutoff:
                        f.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            import json as _json
            from pathlib import Path as _P
            from datetime import datetime as _dt
            log_dir = _P(__file__).resolve().parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            out_path = log_dir / f"long_segment_trace_{job_id}.json"

            # Build a summary so the user can see counts at a glance
            traces = self._long_seg_traces
            n = len(traces)
            n_with_truncation = sum(
                1 for t in traces.values()
                if any(e.get("stage") == "post_truncation_guard" and
                       e.get("outcome") not in (None, "pass") for e in t)
            )
            n_with_word_mismatch = sum(
                1 for t in traces.values()
                if any(e.get("stage") == "post_word_verify" and
                       e.get("outcome") in ("mismatch_flagged", "still_mismatch") for e in t)
            )
            n_with_speed_clamp = sum(
                1 for t in traces.values()
                if any(e.get("stage") == "post_speed_fit" and
                       e.get("clamped_to_limit") for e in t)
            )
            n_with_save_error = sum(
                1 for t in traces.values()
                if any(e.get("stage") == "post_save" and e.get("raised") for e in t)
            )

            report = {
                "generated_at": _dt.now().isoformat(),
                "job_id": job_id,
                "threshold_words": int(getattr(self.cfg, "long_segment_threshold_words", 15)),
                "summary": {
                    "long_segments_traced": n,
                    "with_truncation_guard_hit": n_with_truncation,
                    "with_word_mismatch": n_with_word_mismatch,
                    "with_speed_fit_clamp": n_with_speed_clamp,
                    "with_save_error": n_with_save_error,
                },
                "segments": {
                    str(seg_idx): events for seg_idx, events in sorted(traces.items())
                },
            }
            out_path.write_text(_json.dumps(report, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            print(f"[LONG-SEG-TRACE] Wrote report -> {out_path} "
                  f"({n} segments traced, {n_with_truncation} truncation hits, "
                  f"{n_with_word_mismatch} word mismatches, "
                  f"{n_with_speed_clamp} speed-fit clamps, "
                  f"{n_with_save_error} save errors)", flush=True)
        except Exception as e:
            print(f"[LONG-SEG-TRACE] Failed to write report: {e}", flush=True)

    def _calibrate_voice_wpm(self, segments) -> float:
        """Measure the actual WPM of the TTS voice by synthesizing a few
        sample segments at +0% and probing their duration.

        Edge-TTS Madhur (and most synthetic voices) speak faster than the
        static 130 WPM conversational Hindi baseline because they have no
        hesitations, filler words, or breathing pauses. If we use the static
        baseline, the auto rate overestimates `natural_duration` and computes
        a rate that's too aggressive → output is shorter than the source.

        This calibration synthesizes 5 medium-length segments (10-20 words,
        picked to be representative) at +0% rate, probes their actual audio
        duration via ffprobe, and computes:

            actual_wpm = total_sample_words × 60 / total_sample_audio_sec

        Returns the measured WPM, or the static baseline on any failure.
        Cost: ~3-8 seconds (5 Edge-TTS calls + 5 ffprobe calls).
        """
        import edge_tts
        import asyncio
        import time as _time

        t0 = _time.time()
        static_wpm = int(getattr(self.cfg, "tts_rate_target_wpm", 130) or 130)
        voice = self.cfg.tts_voice
        work_dir = self.cfg.work_dir

        # Pick 5 representative segments for calibration. Accept any
        # non-trivial segment (3+ words). Prefer medium-length (8-20 words)
        # but fall back to shorter/longer if none available.
        candidates = []
        for seg in segments:
            wc = seg.get("_expected_words", 0)
            if wc < 3:
                # Also try raw word count from text if _expected_words is 0
                # (can happen if budget hasn't run yet on this code path)
                text = seg.get("text_translated", seg.get("text", "")).strip()
                wc = len(text.split()) if text else 0
            if wc >= 3:
                text = seg.get("text_translated", seg.get("text", "")).strip()
                if text:
                    candidates.append((wc, text))
        # Sort by closeness to 12 words (ideal medium-length), pick up to 5
        candidates.sort(key=lambda x: abs(x[0] - 12))
        samples = candidates[:5]

        if len(samples) < 2:
            print(f"[RATE-AUTO] Not enough medium-length segments for calibration "
                  f"({len(samples)} found, need 2+) — using static baseline "
                  f"{static_wpm} WPM", flush=True)
            return float(static_wpm)

        # Synthesize each sample at +0% and measure duration
        total_words_sampled = 0
        total_duration_sampled = 0.0
        cal_dir = work_dir / "_rate_calibration"
        cal_dir.mkdir(exist_ok=True)

        for ci, (wc, text) in enumerate(samples):
            mp3 = cal_dir / f"cal_{ci:02d}.mp3"
            try:
                # Each calibration call gets its own asyncio.run() — safe in
                # a sync function context (no nesting). The default-arg
                # closure captures text and mp3 path for this iteration.
                async def _do_cal(_t=text, _m=mp3):
                    comm = edge_tts.Communicate(_t, voice, rate="+0%")
                    await comm.save(str(_m))
                asyncio.run(_do_cal())
                if not mp3.exists() or mp3.stat().st_size < 200:
                    print(f"[RATE-AUTO] Calibration sample {ci}: mp3 missing "
                          f"or too small ({mp3.stat().st_size if mp3.exists() else 0} bytes)",
                          flush=True)
                    continue
                dur = self._probe_mp3_duration_fast(mp3)
                if dur > 0.5:
                    total_words_sampled += wc
                    total_duration_sampled += dur
                else:
                    print(f"[RATE-AUTO] Calibration sample {ci}: duration "
                          f"too short ({dur:.2f}s)", flush=True)
            except Exception as _cal_err:
                # Don't silently swallow — log the actual error so we can
                # diagnose why Edge-TTS fails during calibration but works
                # during the main TTS pass.
                print(f"[RATE-AUTO] Calibration sample {ci} FAILED: "
                      f"{type(_cal_err).__name__}: {str(_cal_err)[:200]}",
                      flush=True)
            finally:
                try:
                    mp3.unlink(missing_ok=True)
                except Exception:
                    pass

        # Cleanup calibration dir
        try:
            import shutil
            shutil.rmtree(cal_dir, ignore_errors=True)
        except Exception:
            pass

        if total_words_sampled < 10 or total_duration_sampled < 1.0:
            print(f"[RATE-AUTO] Calibration failed (sampled {total_words_sampled} "
                  f"words / {total_duration_sampled:.1f}s) — using static baseline "
                  f"{static_wpm} WPM", flush=True)
            return float(static_wpm)

        measured_wpm = total_words_sampled * 60.0 / total_duration_sampled
        elapsed = _time.time() - t0
        print(f"[RATE-AUTO] Calibrated voice WPM: {measured_wpm:.0f} "
              f"(sampled {len(samples)} segments, {total_words_sampled} words, "
              f"{total_duration_sampled:.1f}s audio, took {elapsed:.1f}s). "
              f"Static baseline was {static_wpm} WPM.", flush=True)
        return measured_wpm

    def _auto_compute_tts_rate(self, total_words: int,
                               target_duration_sec: float,
                               calibrated_wpm: float = 0.0) -> str:
        """Compute the optimal tts_rate to match the source video duration.

        Math:
            natural_duration = total_words × 60 / voice_wpm
              (how long the ACTUAL voice would take at +0% pace)
            speedup = natural_duration / target_duration_sec
              (how much faster than natural we need to go to fit the slot)
            rate_pct = round((speedup - 1.0) × 100)
              (convert to Edge-TTS "+N%" rate string)

        voice_wpm comes from calibration (preferred) or the static baseline
        tts_rate_target_wpm (fallback). Calibrated values are typically
        150-170 WPM for Edge-TTS Madhur, vs the static 130 WPM baseline.

        Then clamp to [−50%, ceiling]. Ceiling is user-configurable
        (default +50% = 1.5× max per user request). Above the ceiling, the
        cap is returned and the remaining time overflow is absorbed by
        video stretching in assembly (audio_priority path).

        Returns a rate string like "+30%" or "+0%" ready to pass to
        edge_tts.Communicate(..., rate=<this>).
        """
        import re as _re
        # Use calibrated WPM if provided (from voice sampling), else static baseline
        if calibrated_wpm > 0:
            target_wpm = calibrated_wpm
        else:
            target_wpm = float(getattr(self.cfg, "tts_rate_target_wpm", 130) or 130)
        ceiling_str = getattr(self.cfg, "tts_rate_ceiling", "+50%") or "+50%"

        # Parse the ceiling into an integer percent
        m = _re.match(r"([+-]?)(\d+)%", ceiling_str)
        if m:
            ceiling_pct = int(m.group(2))
            if m.group(1) == "-":
                ceiling_pct = -ceiling_pct
        else:
            ceiling_pct = 50  # safe default

        # Sanity: missing data → no speedup
        if target_duration_sec <= 0 or total_words <= 0 or target_wpm <= 0:
            print(f"[RATE-AUTO] Insufficient data to compute rate "
                  f"(words={total_words}, target={target_duration_sec}, "
                  f"wpm={target_wpm}) — defaulting to +0%", flush=True)
            return "+0%"

        natural_duration = total_words * 60.0 / float(target_wpm)
        speedup = natural_duration / float(target_duration_sec)
        rate_pct = int(round((speedup - 1.0) * 100))

        # Clamp to [-50, ceiling]
        was_capped = False
        if rate_pct > ceiling_pct:
            print(f"[RATE-AUTO] Required +{rate_pct}% exceeds ceiling "
                  f"{ceiling_str} — capped. The remaining "
                  f"{rate_pct - ceiling_pct}% will be absorbed by video "
                  f"stretching in assembly.", flush=True)
            rate_pct = ceiling_pct
            was_capped = True
        elif rate_pct < -50:
            rate_pct = -50

        if rate_pct >= 0:
            result = f"+{rate_pct}%"
        else:
            result = f"{rate_pct}%"

        # Compute projected duration for the user to verify
        projected_duration = natural_duration / (1.0 + rate_pct / 100.0)

        print(
            f"[RATE-AUTO] words={total_words}, target_wpm={target_wpm}, "
            f"natural_duration={natural_duration:.0f}s, "
            f"source_duration={target_duration_sec:.0f}s, "
            f"speedup={speedup:.3f}x, computed_rate={result}"
            + (f" (CAPPED at {ceiling_str})" if was_capped else ""),
            flush=True,
        )
        print(
            f"[RATE-AUTO] Projected dubbed duration at {result}: "
            f"{projected_duration:.0f}s ({projected_duration / 60:.2f} min)"
            + (f" — video will stretch to absorb overflow"
               if was_capped else " — matches source"),
            flush=True,
        )
        return result

    def _pretts_word_budget(self, segments):
        """Pre-TTS pass: compute expected word count for every segment BEFORE
        the TTS engine starts. Stores three things on each segment:

          seg["_expected_words"]        — total words across the whole segment
          seg["_sentence_word_counts"]  — list[int], one entry per sentence
          seg["_sentence_count"]        — convenience: len(_sentence_word_counts)

        And returns a job-level budget dict with totals so callers can stash
        them on the Job state for the UI.

        Why per-sentence:
        1. The sentence pre-splitter uses these to decide which segments to
           pre-split before TTS (multi-sentence = at risk of WebSocket drop).
        2. The post-job UI shows "X words across Y segments / Z sentences"
           so the user can see the workload after the dub completes.
        3. Future analysis: per-sentence word counts let us spot abnormally
           long sentences (>40 words) that may need a deeper split.
        """
        import re as _re
        # Sentence-splitter regex: split on `।` (Hindi danda), `.`, `?`, `!`
        # followed by whitespace or end-of-string. Keeps the terminator
        # attached to the preceding clause via the capture group.
        _SENT_SPLIT = _re.compile(r'(?<=[।?!.])\s+|(?<=[।?!.])$')

        total_words      = 0
        total_slot       = 0.0
        total_sentences  = 0
        multi_sent       = 0
        empty_count      = 0
        max_seg_words    = 0
        max_sent_words   = 0

        for seg in segments:
            text = seg.get("text_translated", seg.get("text", "")).strip()
            if not text:
                seg["_expected_words"] = 0
                seg["_sentence_word_counts"] = []
                seg["_sentence_count"] = 0
                empty_count += 1
                continue

            # Per-sentence word counts: split into sentences, count words per
            sentence_chunks = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
            if not sentence_chunks:
                sentence_chunks = [text]
            sent_word_counts = [len(s.split()) for s in sentence_chunks]

            seg_words = sum(sent_word_counts)
            seg["_expected_words"] = seg_words
            seg["_sentence_word_counts"] = sent_word_counts
            seg["_sentence_count"] = len(sent_word_counts)

            total_words += seg_words
            total_sentences += len(sent_word_counts)
            slot = max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
            total_slot += slot
            if seg_words > max_seg_words:
                max_seg_words = seg_words
            if sent_word_counts:
                max_in_seg = max(sent_word_counts)
                if max_in_seg > max_sent_words:
                    max_sent_words = max_in_seg
            if len(sent_word_counts) > 1:
                multi_sent += 1

        # Required WPM = words/min the speaker needs to fit all words in all
        # slots. Hindi natural rate is 100-180 WPM. >200 means heavy speedup
        # needed downstream; >250 is impossible without losing intelligibility.
        required_wpm = (total_words * 60.0 / total_slot) if total_slot > 0 else 0.0
        avg_words_per_sent = (total_words / total_sentences) if total_sentences > 0 else 0.0

        msg = (f"TTS budget: {len(segments)} segments, {total_sentences} sentences, "
               f"{total_words} words, {total_slot:.0f}s slots, ~{required_wpm:.0f} WPM required")
        self._report("synthesize", 0.05, msg)
        print(f"[TTS-BUDGET] {msg}", flush=True)
        print(f"[TTS-BUDGET] Avg words/sentence: {avg_words_per_sent:.1f}, "
              f"longest sentence: {max_sent_words} words, "
              f"longest segment: {max_seg_words} words", flush=True)
        if multi_sent > 0:
            print(f"[TTS-BUDGET] {multi_sent}/{len(segments)} segments contain "
                  f"multiple sentences (truncation risk — pre-splitter will handle)",
                  flush=True)
        if empty_count > 0:
            print(f"[TTS-BUDGET] {empty_count}/{len(segments)} segments are empty "
                  f"(skipped)", flush=True)
        if max_seg_words > 50:
            print(f"[TTS-BUDGET] Largest segment has {max_seg_words} words "
                  f"(heavy TTS call — may need split)", flush=True)
        if required_wpm > 200:
            print(f"[TTS-BUDGET] WARNING: required WPM is {required_wpm:.0f} "
                  f"(>200) — audio will need heavy speedup to fit slots", flush=True)

        budget = {
            "total_segments":     len(segments),
            "total_sentences":    total_sentences,
            "total_words":        total_words,
            "total_slot":         total_slot,
            "required_wpm":       round(required_wpm, 1),
            "avg_words_per_sent": round(avg_words_per_sent, 2),
            "multi_sentence":     multi_sent,
            "empty":              empty_count,
            "max_seg_words":      max_seg_words,
            "max_sent_words":     max_sent_words,
        }
        # Stash on the Pipeline instance so app.py can read it after run()
        # completes and copy to the Job state for the UI.
        self._tts_budget = budget
        return budget

    def _split_text_on_sentence_boundary(self, text: str) -> list:
        """Split Hindi/English text on sentence terminators for safer TTS.

        Used by BOTH:
          - Stage 4 sentence pre-splitter (runs before first TTS call on
            multi-sentence segments)
          - Stage 7 chunked-retry strategy (last-resort retry in the word
            verifier)

        Returns a list of (sub_text, terminator) tuples. Each tuple represents
        exactly ONE sentence — the sub_text is the sentence body without its
        terminator, and the terminator is the `।`/`.`/`!`/`?` character that
        ended it. Callers re-attach the terminator before sending each piece
        to Edge-TTS so the voice model generates the natural sentence-end
        intonation.

        ── 2026-04-12 fix ──
        Previous version had two bugs:
          1. Regex was `([।?!\\.])\\s*` which matched `.` inside abbreviations
             like "U.S.A" and split them incorrectly.
          2. A coalesce step then merged any chunk with ≤ 2 words into the
             previous chunk, destroying legitimate short Hindi sentences like
             "हाँ।" (1 word) or "ठीक है।" (2 words). Result: 3 sentences
             often collapsed to 1.

        Both are now fixed:
          1. Regex requires whitespace OR end-of-string AFTER the terminator,
             so "U.S.A" is never split.
          2. No coalesce — every sentence boundary produces a real split.
             Short sentences stay separate. The sentence pre-splitter is
             allowed to produce as many tiny Edge-TTS calls as the text
             requires; that's the whole point of splitting by sentence.

        Empty parts are filtered out.
        """
        import re as _re
        if not text:
            return []
        # Require whitespace or end-of-string AFTER the terminator. This
        # prevents splitting inside abbreviations like "U.S.A" where the
        # dot is followed immediately by another letter.
        parts = _re.split(r'([।?!\.])(?=\s|$)', text)
        out = []
        i = 0
        while i < len(parts):
            chunk = parts[i].strip() if parts[i] else ""
            term = parts[i + 1] if i + 1 < len(parts) else ""
            if chunk:
                out.append((chunk, term))
            i += 2
        return out

    def _tts_edge(self, segments, voice_map=None, work_dir=None):
        """Generate TTS using edge-tts (free Microsoft voices).
        If voice_map is provided, each segment uses its speaker's assigned voice.
        Uses parallel processing for speed — 120 segments at once.

        Pre-pass: word-count budget (counts words/sentences per segment, logs totals).
        Pre-split: multi-sentence segments are split on `।` before calling Edge-TTS
                  so the WebSocket can't drop at a sentence boundary that isn't
                  in the call. Sub-sentence audio is concatenated with natural pauses.
        Post-check: two-stage truncation guard (duration probe + burst count).
        """
        if work_dir is None:
            work_dir = self.cfg.work_dir
        import edge_tts
        default_voice = self.cfg.tts_voice

        # ── Dynamic worker scaling ──
        # Start at tts_dynamic_start (default 30) instead of always 120.
        # The generate_all loop adjusts this between batches based on the
        # observed failure rate of the previous batch:
        #   failure rate > 10%  → halve concurrency (min tts_dynamic_min)
        #   failure rate < 2%   → grow by 25%   (max tts_dynamic_max)
        # This adapts to whatever Edge-TTS is willing to serve right now,
        # without over-pressuring (which causes WebSocket drops) or
        # under-using (which makes long jobs take forever).
        dyn_enabled = bool(getattr(self.cfg, "tts_dynamic_workers", True))
        dyn_min   = int(getattr(self.cfg, "tts_dynamic_min", 10))
        dyn_max   = int(getattr(self.cfg, "tts_dynamic_max", 120))
        dyn_start = int(getattr(self.cfg, "tts_dynamic_start", 30))
        # Clamp + sanity
        dyn_start = max(dyn_min, min(dyn_max, dyn_start))
        concurrency = dyn_start if dyn_enabled else 120

        _tts_failures = [0]  # mutable counter for async closure
        _tts_truncs = [0]    # mutable counter for truncation retries
        _tts_splits = [0]    # mutable counter for sentence-split rescues
        _sarvam_fallback_count = [0]  # mutable counter for Sarvam TTS fallbacks
        trunc_threshold = float(getattr(self.cfg, "tts_truncation_threshold", 0.30) or 0.0)

        # ── Sarvam Bulbul v3 availability check ──
        # Pre-check so we don't attempt Sarvam fallback on every segment
        # if no API keys are configured.
        _sarvam_fallback_available = False
        _sarvam_fb_key = get_sarvam_key()
        target_for_sarvam = self.cfg.target_language.split("-")[0]
        if _sarvam_fb_key and target_for_sarvam in self.SARVAM_SUPPORTED_LANGS:
            _sarvam_fallback_available = True
            print("[TTS] Sarvam Bulbul v3 available — will use as fallback "
                  "when Edge-TTS fails", flush=True)
        elif not _sarvam_fb_key:
            print("[TTS] No SARVAM_API_KEY found — Sarvam fallback disabled, "
                  "Edge-only mode", flush=True)
        else:
            print(f"[TTS] Sarvam doesn't support '{target_for_sarvam}' — "
                  "Edge-only mode", flush=True)

        # ── Pre-pass: compute word counts and log workload ──
        # MUST run BEFORE the auto rate computation so we have total_words.
        budget = self._pretts_word_budget(segments)

        # ── Auto TTS rate mode ──
        # When tts_rate_mode=="auto" (default), compute the optimal rate
        # from the source video duration + total word count so the dubbed
        # output matches the original runtime. Capped at tts_rate_ceiling.
        # In manual mode, the user's Speech Rate slider value is used as-is.
        rate_mode = (getattr(self.cfg, "tts_rate_mode", "auto") or "auto").lower()
        if rate_mode == "auto":
            source_dur = getattr(self, "_source_video_duration", 0.0) or 0.0
            if source_dur <= 0:
                source_dur = float(budget.get("total_slot", 0.0) or 0.0)
            total_words = int(budget.get("total_words", 0) or 0)

            # ── Auto-calibrate the voice's actual WPM ──
            # Synthesize a few sample segments at +0% to measure how fast
            # Edge-TTS Madhur actually speaks THIS content, instead of
            # relying on the static 130 WPM baseline which underestimates
            # Madhur's pace and causes the output to be shorter than the
            # source. Adds ~5-8 seconds. The measured WPM is typically
            # 150-170 for Madhur vs the static 130 — a ~20-30% difference
            # that directly translates to 20-30% output duration error.
            self._report("synthesize", 0.04,
                         "Auto rate: calibrating voice speed (sampling 5 segments at +0%)...")
            calibrated_wpm = self._calibrate_voice_wpm(segments)

            auto_rate = self._auto_compute_tts_rate(
                total_words, source_dur, calibrated_wpm=calibrated_wpm,
            )
            self._report("synthesize", 0.06,
                         f"Auto rate: {auto_rate} (calibrated {calibrated_wpm:.0f} WPM, "
                         f"matches {source_dur:.0f}s source, {total_words} words)")
            self.cfg.tts_rate = auto_rate
        else:
            print(f"[RATE-AUTO] Mode=manual — using user rate "
                  f"{self.cfg.tts_rate} as-is", flush=True)

        # NOW read the (possibly auto-updated) rate
        rate = self.cfg.tts_rate

        async def generate_one(i, seg, semaphore):
            raw_text = seg.get("text_translated", seg.get("text", "")).strip()
            if not raw_text:
                return
            # TTS text prep: speech-optimize + pronunciation dictionary
            text = self._prepare_tts_text(raw_text)
            seg_voice = default_voice
            if voice_map and "speaker_id" in seg:
                seg_voice = voice_map.get(seg["speaker_id"], default_voice)
            mp3 = work_dir / f"tts_{i:04d}.mp3"

            # ── Use pre-computed word count from the budget pass ──
            # _pretts_word_budget already counted these and stashed them on
            # the segment. Falling back to a live count for safety.
            word_count = seg.get("_expected_words") or (len(text.split()) if text else 1)
            sentence_count = seg.get("_sentence_count", 1)

            # ── Lower-bound duration estimate (for truncation guard) ──
            # Generous: assumes a very fast Hindi speaker at 250 WPM. Real
            # speech is 100-180 WPM, so this lower bound is impossible to
            # beat in practice — meaning the guard CANNOT false-positive on
            # naturally fast speech. It only catches the catastrophic case
            # where the WebSocket dropped mid-stream.
            min_seconds_at_max_wpm = (word_count * 60.0) / 250.0  # 250 WPM ceiling
            # Apply user threshold: 0.30 default = "actual must be >= 30% of
            # the floor". Lower threshold = more permissive, higher = stricter.
            min_acceptable_dur = min_seconds_at_max_wpm * trunc_threshold

            # ── Word-count expectations for the burst guard ──
            # Hindi averages ~1.5–2.0 syllables per word; we use 1.5 as a
            # conservative floor. Each syllable produces roughly one RMS
            # rising edge in the burst counter. So expected_bursts ≈ words × 1.5.
            # The truncation threshold scales this lower bound: 0.30 means
            # "acceptable if we got at least 30% of expected bursts".
            # Hindi syllable density varies by segment length:
            # Long segments (10+ words): ~1.5 syllables per word (mixed
            #   compound verbs, particles, natural pauses → more bursts)
            # Short segments (<10 words): often 1 continuous voicing with
            #   few internal pauses → fewer bursts per word (~1.0).
            # Using 1.5 for short segments causes false positives (the
            #   "bursts 3 < floor 4 for 5 words" pattern from the logs).
            _syl_factor = 1.5 if word_count >= 10 else 1.0
            expected_bursts = max(1, int(word_count * _syl_factor))
            min_acceptable_bursts = max(1, int(expected_bursts * trunc_threshold))

            # ── Long-segment trace: record pre-TTS state ──
            # Cheap, only fires for segments above the configured word threshold.
            self._trace_record_pre_tts(i, seg, text, word_count)

            # ── Sentence pre-splitter (root-cause fix) ──
            # If this segment contains multiple sentences AND the truncation
            # guard is on, split the text on sentence terminators and call
            # Edge-TTS once per sub-sentence. This prevents the WebSocket
            # from being able to drop at a `।` boundary because no individual
            # call contains a `।`. Each sub-mp3 is concatenated with a tiny
            # silence into the final mp3 path.
            #
            # Falls back to single-call path if:
            #   - splitter returns 1 piece (no actual split)
            #   - any sub-call fails (rather than partial audio, do the whole
            #     thing in one call and let the post-check guard catch it)
            use_split = (
                trunc_threshold > 0
                and sentence_count > 1
                and len(text) > 30
            )

            async with semaphore:
                # ── PATH A: sentence-split synthesis (root-cause fix) ──
                if use_split:
                    pieces = self._split_text_on_sentence_boundary(text)
                    if len(pieces) >= 2:
                        try:
                            sub_mp3s = []
                            ok = True
                            for sub_idx, (sub_text, term) in enumerate(pieces):
                                # Re-attach the terminator so the TTS keeps the
                                # natural sentence-end intonation
                                sub_full = sub_text + (term or "")
                                sub_mp3 = work_dir / f"tts_{i:04d}_p{sub_idx}.mp3"
                                # Each sub-call gets its own retry loop (3 attempts)
                                sub_ok = False
                                for sub_attempt in range(3):
                                    try:
                                        comm = edge_tts.Communicate(sub_full, seg_voice, rate=rate)
                                        await comm.save(str(sub_mp3))
                                        if sub_mp3.exists() and sub_mp3.stat().st_size > 200:
                                            sub_ok = True
                                            break
                                    except Exception:
                                        if sub_attempt < 2:
                                            await asyncio.sleep(0.5 * (sub_attempt + 1))
                                if not sub_ok:
                                    ok = False
                                    break
                                sub_mp3s.append(sub_mp3)

                            if ok and sub_mp3s:
                                # Concatenate sub-mp3s into the final mp3 using
                                # ffmpeg's concat demuxer (no re-encode = fast).
                                concat_list = work_dir / f"tts_{i:04d}_concat.txt"
                                concat_list.write_text(
                                    "\n".join(f"file '{p.name}'" for p in sub_mp3s),
                                    encoding="utf-8",
                                )
                                try:
                                    self._run_proc(
                                        [self._ffmpeg, "-y", "-f", "concat",
                                         "-safe", "0", "-i", str(concat_list),
                                         "-c", "copy", str(mp3)],
                                        check=True, capture_output=True,
                                    )
                                    # Cleanup sub-files
                                    for p in sub_mp3s:
                                        try: p.unlink(missing_ok=True)
                                        except Exception: pass
                                    try: concat_list.unlink(missing_ok=True)
                                    except Exception: pass

                                    _tts_splits[0] += 1
                                    if _tts_splits[0] <= 5:
                                        print(f"[TTS-SPLIT] Seg {i}: synthesized "
                                              f"{len(sub_mp3s)} sub-sentences and "
                                              f"concatenated -> {mp3.name}", flush=True)
                                    seg["_tts_mp3"] = mp3
                                    return
                                except Exception as e:
                                    print(f"[TTS-SPLIT] Seg {i} concat failed: {e} "
                                          f"— falling back to single-call path",
                                          flush=True)
                                    # cleanup before fallthrough
                                    for p in sub_mp3s:
                                        try: p.unlink(missing_ok=True)
                                        except Exception: pass
                                    try: concat_list.unlink(missing_ok=True)
                                    except Exception: pass
                            else:
                                # Sub-call failed — clean up partials and fall
                                # through to the single-call path with the full text
                                for p in sub_mp3s:
                                    try: p.unlink(missing_ok=True)
                                    except Exception: pass
                                print(f"[TTS-SPLIT] Seg {i} sub-call failed — "
                                      f"falling back to single-call path", flush=True)
                        except Exception as e:
                            print(f"[TTS-SPLIT] Seg {i} unexpected error: {e} "
                                  f"— falling back to single-call path", flush=True)

                # ── PATH B: single-call synthesis — try Edge once, fallback to Google ──
                edge_ok = False
                try:
                    comm = edge_tts.Communicate(text, seg_voice, rate=rate)
                    await comm.save(str(mp3))

                    # Long-segment trace: post-save state (file size, no exception)
                    self._trace_record_after_save(i, word_count, mp3, exception=None)

                    # ── Inline truncation guard (two-stage) ──
                    truncated = False
                    trunc_reason = ""
                    if trunc_threshold > 0 and mp3.exists():
                        # Stage 1 — duration check
                        actual_dur = self._probe_mp3_duration_fast(mp3)
                        if 0 < actual_dur < min_acceptable_dur:
                            truncated = True
                            trunc_reason = (f"duration {actual_dur:.2f}s "
                                            f"< floor {min_acceptable_dur:.2f}s "
                                            f"({word_count} words)")
                        else:
                            # Stage 2 — burst (≈syllable) count check
                            actual_bursts = self._count_speech_bursts_mp3(mp3)
                            if 0 < actual_bursts < min_acceptable_bursts:
                                truncated = True
                                trunc_reason = (f"bursts {actual_bursts} "
                                                f"< floor {min_acceptable_bursts} "
                                                f"(expected ≈{expected_bursts} "
                                                f"for {word_count} words)")

                    if truncated:
                        _tts_truncs[0] += 1
                        if _tts_truncs[0] <= 10:
                            print(f"[TTS-TRUNC] Seg {i}: {trunc_reason} "
                                  f"— falling back to Google TTS. "
                                  f"Text: {text[:80]!r}", flush=True)
                        self._trace_record_truncation_guard(
                            i, word_count,
                            outcome="duration_short" if "duration" in trunc_reason else "bursts_low",
                            attempt=0, reason=trunc_reason,
                        )
                        try:
                            mp3.unlink(missing_ok=True)
                        except Exception:
                            pass
                        edge_ok = False
                    else:
                        self._trace_record_truncation_guard(
                            i, word_count, outcome="pass", attempt=0,
                        )
                        edge_ok = True
                except Exception as e:
                    self._trace_record_after_save(i, word_count, mp3, exception=e)
                    _tts_failures[0] += 1
                    if _tts_failures[0] <= 5:
                        print(f"[TTS] Seg {i} Edge-TTS failed: {e} "
                              f"— falling back to Google TTS", flush=True)
                    edge_ok = False

                if edge_ok:
                    seg["_tts_mp3"] = mp3
                    return

                # ── Sarvam Bulbul v3 fallback (no more Edge retries) ──
                if _sarvam_fallback_available:
                    sarvam_ok = self._sarvam_tts_single_mp3(text, mp3)
                    if sarvam_ok:
                        _sarvam_fallback_count[0] += 1
                        if _sarvam_fallback_count[0] <= 10:
                            print(f"[TTS-SARVAM] Seg {i}: Sarvam Bulbul fallback succeeded",
                                  flush=True)
                        seg["_tts_mp3"] = mp3
                        return

                # Both failed — mark for post-batch sweep
                seg["_tts_mp3"] = None

        async def generate_all():
            """Dynamic-worker batch loop.

            Each batch uses a fresh semaphore at the current concurrency level.
            After every batch we measure failures + truncs since the last
            checkpoint and adjust concurrency for the NEXT batch:
              fail rate > 10% → halve (min dyn_min)
              fail rate < 2%  → grow 25% (max dyn_max)
              else            → keep
            """
            nonlocal_state = {"current": concurrency}
            total = len(segments)
            done_so_far = 0
            batch_no = 0
            seg_idx = 0
            # Snapshot counters at the start of each batch so we measure
            # ONLY failures that happened during that batch.
            prev_failures = _tts_failures[0]
            prev_truncs = _tts_truncs[0]

            while seg_idx < total:
                cur = nonlocal_state["current"]
                semaphore = asyncio.Semaphore(cur)
                # Build tasks for THIS batch only
                batch_end = min(seg_idx + cur, total)
                batch_tasks = [
                    generate_one(i, segments[i], semaphore)
                    for i in range(seg_idx, batch_end)
                ]
                await asyncio.gather(*batch_tasks)

                seg_idx = batch_end
                done_so_far = batch_end
                batch_no += 1

                # ── Measure failures from this batch ──
                batch_failures = _tts_failures[0] - prev_failures
                batch_truncs   = _tts_truncs[0] - prev_truncs
                batch_size_actual = batch_end - (seg_idx - len(batch_tasks))
                if batch_size_actual <= 0:
                    batch_size_actual = 1
                bad_count = batch_failures + batch_truncs
                bad_rate = bad_count / float(batch_size_actual)

                # Reset prev so the next batch only sees its own counts
                prev_failures = _tts_failures[0]
                prev_truncs = _tts_truncs[0]

                # ── Adjust concurrency for the NEXT batch ──
                if dyn_enabled:
                    old_cur = cur
                    if bad_rate > 0.10:
                        # Heavy failures → halve, but never below dyn_min
                        nonlocal_state["current"] = max(dyn_min, cur // 2)
                        if nonlocal_state["current"] != old_cur:
                            print(f"[TTS-DYN] Batch {batch_no}: bad_rate "
                                  f"{bad_rate*100:.0f}% ({bad_count}/{batch_size_actual}) "
                                  f"-> halving concurrency {old_cur} -> "
                                  f"{nonlocal_state['current']}", flush=True)
                    elif bad_rate < 0.02 and cur < dyn_max:
                        # Clean batch → grow by 25%
                        new_cur = min(dyn_max, max(cur + 1, int(cur * 1.25)))
                        nonlocal_state["current"] = new_cur
                        if new_cur != old_cur:
                            print(f"[TTS-DYN] Batch {batch_no}: bad_rate "
                                  f"{bad_rate*100:.1f}% (clean) -> growing concurrency "
                                  f"{old_cur} -> {new_cur}", flush=True)

                self._report(
                    "synthesize",
                    0.1 + 0.8 * (done_so_far / total),
                    f"Synthesized {done_so_far}/{total} segments "
                    f"(workers={nonlocal_state['current']}, batch_bad={bad_count})...",
                )

        asyncio.run(generate_all())

        # ── POST-BATCH VERIFICATION SWEEP ──────────────────────────────────
        # After all batches complete, find segments that are STILL missing
        # audio. Use Sarvam Bulbul v3 only (no Edge retries — Edge already
        # had its chance).
        missing_indices = []
        for i, seg in enumerate(segments):
            raw_text = seg.get("text_translated", seg.get("text", "")).strip()
            if not raw_text:
                continue  # legitimately empty segment
            mp3_path = seg.get("_tts_mp3")
            if not mp3_path or not Path(mp3_path).exists() or Path(mp3_path).stat().st_size < 200:
                missing_indices.append(i)

        if missing_indices:
            if _sarvam_fallback_available:
                print(f"[TTS-SWEEP] Post-batch sweep: {len(missing_indices)} segments "
                      f"missing audio — retrying with Sarvam Bulbul v3", flush=True)
                self._report("synthesize", 0.86,
                             f"Post-batch sweep: {len(missing_indices)} segments via Sarvam Bulbul v3...")
            else:
                print(f"[TTS-SWEEP] Post-batch sweep: {len(missing_indices)} segments "
                      f"missing audio — Sarvam unavailable, segments will be skipped", flush=True)

            sarvam_rescued = 0
            still_missing = []

            for idx, i in enumerate(missing_indices):
                seg = segments[i]
                raw_text = seg.get("text_translated", seg.get("text", "")).strip()
                if not raw_text:
                    continue
                text = self._prepare_tts_text(raw_text)
                mp3 = work_dir / f"tts_{i:04d}.mp3"

                rescued = False
                if _sarvam_fallback_available:
                    rescued = self._sarvam_tts_single_mp3(text, mp3)
                    if rescued:
                        sarvam_rescued += 1

                if rescued:
                    seg["_tts_mp3"] = mp3
                else:
                    still_missing.append(i)

                if (idx + 1) % 10 == 0 or idx == len(missing_indices) - 1:
                    self._report(
                        "synthesize",
                        0.86 + 0.04 * ((idx + 1) / len(missing_indices)),
                        f"Sweep: {idx + 1}/{len(missing_indices)} processed, "
                        f"{len(still_missing)} still missing",
                    )

            if sarvam_rescued:
                print(f"[TTS-SWEEP] Sarvam Bulbul rescued {sarvam_rescued} segments", flush=True)
            if still_missing:
                print(f"[TTS-SWEEP] CRITICAL: {len(still_missing)} segments permanently "
                      f"failed TTS. Indices: {still_missing}", flush=True)
            else:
                print(f"[TTS-SWEEP] All missing segments recovered!", flush=True)
        else:
            print(f"[TTS-SWEEP] All {len([s for s in segments if (s.get('text_translated', s.get('text', '')).strip())])} "
                  f"segments have audio — no sweep needed", flush=True)

        # Summary: how many segments hit each guard this job
        if _tts_splits[0] > 0:
            self._report("synthesize", 0.84,
                         f"TTS sentence pre-splitter rescued {_tts_splits[0]} multi-sentence "
                         f"segment(s) by splitting on `।` before synthesis")
            print(f"[TTS-SPLIT] Total sentence-splits this job: {_tts_splits[0]}", flush=True)
        if _tts_truncs[0] > 0:
            self._report("synthesize", 0.85,
                         f"TTS truncation guard caught {_tts_truncs[0]} WebSocket drop(s) "
                         f"(threshold={trunc_threshold:.2f})")
            print(f"[TTS-TRUNC] Total truncation retries this job: {_tts_truncs[0]}", flush=True)
        elif trunc_threshold > 0 and _tts_splits[0] == 0:
            print(f"[TTS-TRUNC] Truncation guard active (threshold={trunc_threshold:.2f}), "
                  f"no truncations detected", flush=True)
        if _sarvam_fallback_count[0] > 0:
            self._report("synthesize", 0.855,
                         f"Sarvam Bulbul fallback used for {_sarvam_fallback_count[0]} segment(s)")
            print(f"[TTS-SARVAM] Total Sarvam fallback segments this job: "
                  f"{_sarvam_fallback_count[0]}", flush=True)

        # Parallel MP3→WAV conversion (natural pace — no speedup, video adapts instead)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def convert_and_speed(i, seg):
            mp3 = seg.pop("_tts_mp3", None)
            if not mp3 or not Path(mp3).exists():
                raw_text = seg.get("text_translated", seg.get("text", "")).strip()
                if raw_text:
                    print(f"[TTS-CONVERT] WARNING: Segment {i} has no mp3 file "
                          f"— skipped (text: {raw_text[:60]!r})", flush=True)
                return None
            # Check for 0-byte or corrupt mp3
            if Path(mp3).stat().st_size < 200:
                print(f"[TTS-CONVERT] WARNING: Segment {i} mp3 is too small "
                      f"({Path(mp3).stat().st_size} bytes) — skipped", flush=True)
                return None
            wav = work_dir / f"tts_{i:04d}.wav"
            # MP3→WAV + resample only — NO atempo speedup (audio stays natural)
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(mp3),
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(wav)],
                check=True, capture_output=True,
            )
            mp3.unlink(missing_ok=True)
            # Enhance: silence trim + loudness norm
            self._enhance_tts_wav(wav)
            tts_dur = self._get_duration(wav)

            # Long-segment trace: record post-WAV-convert duration
            try:
                _wc = int(seg.get("_expected_words", 0) or 0)
                if _wc > 0:
                    self._trace_record_wav_convert(i, _wc, wav, tts_dur)
            except Exception:
                pass

            # QC gate + auto-rerender on failure
            expected_dur = seg.get("end", 0) - seg.get("start", 0)
            qc = self._qc_check_wav(wav, expected_duration=expected_dur)
            rerender_count = 0
            manual_review = False

            if not qc["ok"]:
                print(f"[QC] Seg {i} issues: {'; '.join(qc['issues'])} | "
                      f"dur={qc['duration']:.2f}s expected={expected_dur:.2f}s — retrying...",
                      flush=True)
                # Auto-rerender: prepare TTS text and try up to 3 attempts
                tts_text = self._prepare_tts_text(
                    seg.get("text_translated", seg.get("text", "")).strip()
                )
                seg_voice = default_voice
                if voice_map and "speaker_id" in seg:
                    seg_voice = voice_map.get(seg["speaker_id"], default_voice)

                retry_qc = self._rerender_edge_segment(tts_text, seg_voice, wav, expected_dur)
                rerender_count = retry_qc.get("attempt", 0) + 1
                manual_review = retry_qc.get("manual_review", False)

                if manual_review:
                    print(f"[QC] Seg {i} -> MANUAL REVIEW after {rerender_count} rerenders", flush=True)
                else:
                    print(f"[QC] Seg {i} -> passed on attempt {rerender_count}", flush=True)

                tts_dur = self._get_duration(wav)

            return {
                "start": seg["start"],
                "end": seg["end"],
                "wav": wav,
                "duration": tts_dur,
                "_order": i,
                "_seg_idx": seg.get("_seg_idx", i),  # propagate segment index for _tts_manager matching
                "_orig_seg_idx": i,   # original index into segments[] for review queue lookup
                "_already_sped": False,  # no speedup applied — natural pace
                "_rerender_count": rerender_count,
                "_manual_review": manual_review,
                "_qc_ok": qc["ok"],
            }

        tts_data = []
        convert_skipped = 0
        non_empty_count = sum(1 for s in segments
                              if s.get("text_translated", s.get("text", "")).strip())
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(convert_and_speed, i, seg): i for i, seg in enumerate(segments)}
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    tts_data.append(result)
                else:
                    convert_skipped += 1

        if convert_skipped > 0:
            print(f"[TTS-CONVERT] {convert_skipped} segments produced no WAV "
                  f"(of {non_empty_count} non-empty)", flush=True)

        # Sort back to original order and collect metrics
        tts_data.sort(key=lambda x: x.get("_order", 0))
        total_rerenders = 0
        review_queue = []
        for t in tts_data:
            rc = t.pop("_rerender_count", 0)
            mr = t.pop("_manual_review", False)
            orig_idx = t.pop("_orig_seg_idx", None)
            total_rerenders += rc
            t.pop("_order", None)
            t.pop("_qc_ok", None)
            if mr:
                # Look up original segment by its index (not zip position — some segments are empty/skipped)
                orig_seg = segments[orig_idx] if orig_idx is not None and orig_idx < len(segments) else {}
                review_queue.append({
                    "segment_idx": orig_idx,
                    "start": t.get("start", 0),
                    "end": t.get("end", 0),
                    "source_text": orig_seg.get("text", ""),
                    "translated_text": orig_seg.get("text_translated", ""),
                    "emotion": orig_seg.get("emotion", "neutral"),
                    "rerender_count": rc,
                    "issues": ["auto-rerender exhausted"],
                })
        if total_rerenders > 0 or review_queue:
            print(f"[QC] Edge TTS: {total_rerenders} rerenders, "
                  f"{len(review_queue)} for manual review", flush=True)
        if review_queue:
            self._save_manual_review_queue(review_queue)
        return tts_data

    def _build_fitted_audio(self, video_path, audio_raw, tts_data, total_video_duration):
        print("[Assembly] Using: _build_fitted_audio", flush=True)
        """Keep video at original speed, speed-adjust each TTS clip (1.1x–1.25x).

        Rules:
        - Slow TTS (shorter than slot) → slow down up to 1.1x
        - Fast TTS (longer than slot) → speed up up to 1.25x
        - NEVER cut or truncate any audio — all speech must be heard
        """
        num_segs = len(tts_data)
        fitted_segments = []
        duration_fit_enabled = getattr(self.cfg, 'enable_duration_fit', True)

        for idx, tts in enumerate(tts_data):
            self._report("assemble",
                         0.05 + 0.55 * (idx / max(num_segs, 1)),
                         f"Fitting segment {idx + 1}/{num_segs} to scene...")

            seg_start = tts["start"]
            seg_end = tts["end"]
            original_dur = seg_end - seg_start
            tts_dur = tts["duration"]
            tts_wav = tts["wav"]

            # Skip speed fitting when duration fit is disabled — use raw TTS as-is
            if not duration_fit_enabled or original_dur < 0.1 or tts_dur < 0.1:
                fitted_segments.append({
                    "start": seg_start,
                    "wav": tts_wav,
                    "duration": tts_dur,
                })
                continue

            ratio = tts_dur / original_dur  # > 1 = TTS is longer, need to speed up
            gap_ms = abs(tts_dur - original_dur) * 1000

            # Tier 1: within 80ms — accept as-is
            if gap_ms <= self.FIT_PASS_MS:
                fitted_segments.append({
                    "start": seg_start,
                    "wav": tts_wav,
                    "duration": tts_dur,
                })
                continue

            # Tier 2: 80–250ms — stretch within preferred limits
            # Tier 3: >250ms — still stretch if within hard limits, but flag as needing rewrite
            if gap_ms > self.FIT_REWRITE_MS and ratio > self.FIT_RATIO_PREF_MAX:
                print(f"[FIT] Seg {idx}: gap {gap_ms:.0f}ms exceeds rewrite threshold "
                      f"(ratio={ratio:.3f}) — stretching within hard limits", flush=True)

            # Clamp ratio to hard limits
            clamped_ratio = max(self.SPEED_MIN, min(ratio, self.SPEED_MAX))

            stretched_wav = self.cfg.work_dir / f"fitted_{idx:04d}.wav"
            self._time_stretch(tts_wav, clamped_ratio, stretched_wav)

            new_dur = tts_dur / clamped_ratio
            fitted_segments.append({
                "start": seg_start,
                "wav": stretched_wav,
                "duration": new_dur,
            })

        # Truncate overlapping segments before assembly
        fitted_segments = self._truncate_overlaps(fitted_segments)
        # Build audio timeline
        self._report("assemble", 0.65, "Building audio timeline...")
        fitted_audio = self._build_timeline_no_cut(fitted_segments, total_video_duration, prefix="fitted_")

        # Mix original audio at low volume if requested
        if self.cfg.mix_original:
            fitted_audio = self._mix_audio(audio_raw, fitted_audio, self.cfg.original_volume)

        # Mux: original video (untouched) + fitted TTS audio
        self._report("assemble", 0.85, "Muxing final video...")
        self._mux_replace_audio(video_path, fitted_audio, self.cfg.output_path)

    def _build_video_synced(self, video_path, audio_raw, tts_data, total_video_duration):
        print("[Assembly] Using: _build_video_synced", flush=True)
        """Adjust video speed per-segment to match natural TTS duration.

        Instead of changing audio speed, adjust video speed:
        - TTS longer than original → slow video down (setpts > 1)
        - TTS shorter than original → speed video up (setpts < 1)
        - Gaps between speech play at normal speed
        This keeps TTS voices sounding natural.
        """
        # Build sections: alternating gaps and speech segments
        sections = []
        current_pos = 0.0

        for tts in tts_data:
            seg_start = tts["start"]
            seg_end = tts["end"]
            tts_dur = tts["duration"]
            original_dur = seg_end - seg_start

            # Gap before this segment
            if seg_start > current_pos + 0.05:
                sections.append({
                    "type": "gap",
                    "video_start": current_pos,
                    "video_end": seg_start,
                })

            # Speech segment — compute video speed factor
            # setpts factor: tts_dur / original_dur
            # > 1 = slow video down (TTS is longer), < 1 = speed video up (TTS is shorter)
            pts_factor = (tts_dur / original_dur) if original_dur > 0.1 else 1.0
            # Clamp to 1.1x slow / 1.25x fast limits
            pts_factor = max(self.SPEED_MIN, min(pts_factor, self.SPEED_MAX))

            sections.append({
                "type": "speech",
                "video_start": seg_start,
                "video_end": seg_end,
                "pts_factor": pts_factor,
                "tts_wav": tts["wav"],
                "tts_dur": tts_dur,
            })

            current_pos = seg_end

        # Trailing gap after last speech
        if current_pos < total_video_duration - 0.05:
            sections.append({
                "type": "gap",
                "video_start": current_pos,
                "video_end": total_video_duration,
            })

        # Create video clips for each section
        num_sections = len(sections)
        clip_paths = []

        for idx, sec in enumerate(sections):
            self._report("assemble",
                         0.1 + 0.6 * (idx / max(num_sections, 1)),
                         f"Syncing section {idx + 1}/{num_sections}...")

            clip = self.cfg.work_dir / f"vsync_{idx:04d}.mp4"
            vs = sec["video_start"]
            ve = sec["video_end"]
            dur = ve - vs

            if dur < 0.05:
                continue

            pts_factor = sec.get("pts_factor", 1.0)

            if sec["type"] == "gap" or abs(pts_factor - 1.0) < 0.08:
                # No speed change needed — extract at normal speed
                self._run_proc(
                    [self._ffmpeg, "-y",
                     "-ss", f"{vs:.3f}", "-i", str(video_path),
                     "-t", f"{dur:.3f}",
                     "-an",
                     *self._video_encode_args(force_cpu=True),
                     str(clip)],
                    check=True, capture_output=True,
                )
            else:
                # Adjust video speed: setpts=factor*PTS
                # factor > 1 = slow down, factor < 1 = speed up
                self._run_proc(
                    [self._ffmpeg, "-y",
                     "-ss", f"{vs:.3f}", "-i", str(video_path),
                     "-t", f"{dur:.3f}",
                     "-filter:v", f"setpts={pts_factor:.6f}*PTS",
                     "-an",
                     *self._video_encode_args(force_cpu=True),
                     str(clip)],
                    check=True, capture_output=True,
                )

            clip_paths.append(clip)

        if not clip_paths:
            raise RuntimeError("No video sections produced")

        # Concatenate all video clips
        self._report("assemble", 0.75, "Joining video sections...")
        synced_video = self.cfg.work_dir / "video_synced.mp4"
        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], synced_video)
        else:
            self._concatenate_videos(clip_paths, synced_video)

        # Build TTS audio timeline matching the adjusted video timing
        new_pos = 0.0
        audio_segments = []
        for sec in sections:
            dur = sec["video_end"] - sec["video_start"]
            if dur < 0.05:
                continue
            if sec["type"] == "speech":
                audio_segments.append({
                    "start": new_pos,
                    "wav": sec["tts_wav"],
                    "duration": sec["tts_dur"],
                })
                new_pos += sec["tts_dur"]
            else:
                new_pos += dur  # gaps keep original duration

        self._report("assemble", 0.85, "Building audio timeline...")
        audio_segments = self._truncate_overlaps(audio_segments)
        synced_audio = self._build_timeline(audio_segments, new_pos, prefix="synced_")

        # Mix original audio at low volume if requested
        if self.cfg.mix_original:
            synced_audio = self._mix_audio(audio_raw, synced_audio, self.cfg.original_volume)

        # Mux final video + audio
        self._report("assemble", 0.90, "Muxing final video...")
        self._mux_replace_audio(synced_video, synced_audio, self.cfg.output_path)

    @staticmethod
    def _parse_tts_rate(rate_str: str) -> int:
        """Parse edge-tts rate string like '-5%' or '+20%' to integer."""
        return int(rate_str.replace("%", "").replace("+", ""))

    def _time_stretch(self, wav_path: Path, ratio: float, output_path: Path) -> Path:
        """Time-stretch audio (ratio > 1 = speed up). Tries rubberband then atempo."""
        # Try ffmpeg rubberband filter first (better pitch preservation)
        try:
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(wav_path),
                 "-filter:a", f"rubberband=tempo={ratio:.4f}",
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(output_path)],
                check=True, capture_output=True,
            )
            return output_path
        except subprocess.CalledProcessError:
            pass  # rubberband not available, fall back to atempo

        # Fallback: atempo filter (chain for ratios outside 0.5-2.0)
        tempo = ratio
        filters = []
        while tempo > 2.0:
            filters.append("atempo=2.0")
            tempo /= 2.0
        while tempo < 0.5:
            filters.append("atempo=0.5")
            tempo /= 0.5
        filters.append(f"atempo={tempo:.4f}")
        self._run_proc(
            [self._ffmpeg, "-y", "-i", str(wav_path),
             "-filter:a", ",".join(filters),
             "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
             str(output_path)],
            check=True, capture_output=True,
        )
        return output_path

    # ── Duration & tempo adjustment ───────────────────────────────────────
    def _get_duration(self, media_path: Path) -> float:
        """Get duration of a media file in seconds using ffprobe."""
        ffmpeg_path = Path(self._ffmpeg)
        if ffmpeg_path.is_absolute():
            ffprobe = str(ffmpeg_path.parent / "ffprobe")
            if sys.platform == "win32" and not ffprobe.endswith(".exe"):
                ffprobe += ".exe"
        else:
            ffprobe = shutil.which("ffprobe") or "ffprobe"
        try:
            result = self._run_proc(
                [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)],
                capture_output=True, text=True, timeout=15,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _adjust_tempo(self, wav_path: Path, ratio: float) -> Path:
        """Speed up or slow down audio to match video duration.

        ratio = tts_duration / video_duration
        ratio > 1 means TTS is longer → speed up (atempo > 1)
        ratio < 1 means TTS is shorter → slow down (atempo < 1)
        """
        adjusted = self.cfg.work_dir / "tts_adjusted.wav"

        # ffmpeg atempo filter accepts 0.5 to 100.0
        # For values outside 0.5-2.0, chain multiple filters
        tempo = ratio
        filters = []
        while tempo > 2.0:
            filters.append("atempo=2.0")
            tempo /= 2.0
        while tempo < 0.5:
            filters.append("atempo=0.5")
            tempo /= 0.5
        filters.append(f"atempo={tempo:.4f}")

        filter_str = ",".join(filters)
        self._run_proc(
            [
                self._ffmpeg, "-y", "-i", str(wav_path),
                "-filter:a", filter_str,
                "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                str(adjusted),
            ],
            check=True, capture_output=True,
        )
        return adjusted

    def _adjust_video_duration(self, video_path: Path, target_duration: float) -> Path:
        """Adjust video duration to match the dubbed audio using setpts filter.

        If dubbed audio is longer than video → slow down video (scenes last longer).
        If dubbed audio is shorter than video → speed up video (scenes go faster).
        """
        video_duration = self._get_duration(video_path)
        if video_duration <= 0 or target_duration <= 0:
            return video_path

        # PTS factor: >1 slows video down, <1 speeds it up
        pts_factor = target_duration / video_duration
        if abs(pts_factor - 1.0) < 0.02:  # Less than 2% difference, skip
            return video_path

        adjusted = self.cfg.work_dir / "video_adjusted.mp4"
        self._report("assemble", 0.1,
                     f"Adjusting video speed ({1/pts_factor:.2f}x) to match audio...")

        # setpts=PTS*factor changes video timing
        # factor > 1 → slower (stretches video), factor < 1 → faster (compresses video)
        # fps filter re-establishes constant frame rate after pts change
        self._run_proc(
            [
                self._ffmpeg, "-y", "-i", str(video_path),
                "-filter:v", f"setpts={pts_factor:.6f}*PTS",
                "-an",  # Drop original audio (we'll add dubbed audio)
                *self._video_encode_args("18"),
                str(adjusted),
            ],
            check=True, capture_output=True,
        )
        return adjusted

    # ── Audio mixing ─────────────────────────────────────────────────────
    def _separate_background(self, audio_raw: Path) -> Path:
        """Use demucs to extract instrumental/background track (no vocals).
        For long audio (>10min), splits into chunks to avoid GPU OOM.
        Returns path to the no-vocals audio file, or audio_raw as fallback."""
        bg_path = self.cfg.work_dir / "background_music.wav"
        if bg_path.exists():
            return bg_path

        total_duration = self._get_duration(audio_raw)
        CHUNK_MINS = 10  # Process 10-minute chunks to avoid OOM
        CHUNK_SECS = CHUNK_MINS * 60

        try:
            import demucs.separate

            if total_duration <= CHUNK_SECS:
                # Short audio — process in one shot
                return self._demucs_single(audio_raw, bg_path)

            # Long audio — split, process chunks, concatenate
            print(f"[DEMUCS] Audio is {total_duration/60:.0f}min, splitting into {CHUNK_MINS}min chunks...", flush=True)
            num_chunks = math.ceil(total_duration / CHUNK_SECS)
            chunk_bg_paths = []

            for ci in range(num_chunks):
                start = ci * CHUNK_SECS
                chunk_audio = self.cfg.work_dir / f"demucs_chunk_{ci:03d}.wav"
                chunk_bg = self.cfg.work_dir / f"demucs_bg_{ci:03d}.wav"

                # Extract chunk
                self._run_proc(
                    [self._ffmpeg, "-y", "-i", str(audio_raw),
                     "-ss", str(start), "-t", str(CHUNK_SECS),
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     "-acodec", "pcm_s16le", str(chunk_audio)],
                    check=True, capture_output=True,
                )

                # Separate this chunk
                result = self._demucs_single(chunk_audio, chunk_bg)
                chunk_audio.unlink(missing_ok=True)

                if result == chunk_bg and chunk_bg.exists():
                    chunk_bg_paths.append(chunk_bg)
                else:
                    # Demucs failed for this chunk — use original audio chunk as fallback
                    fallback = self.cfg.work_dir / f"demucs_fallback_{ci:03d}.wav"
                    self._run_proc(
                        [self._ffmpeg, "-y", "-i", str(audio_raw),
                         "-ss", str(start), "-t", str(CHUNK_SECS),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         "-acodec", "pcm_s16le", str(fallback)],
                        check=True, capture_output=True,
                    )
                    chunk_bg_paths.append(fallback)

                print(f"[DEMUCS] Chunk {ci+1}/{num_chunks} done", flush=True)

            # Concatenate all chunk backgrounds
            if not chunk_bg_paths:
                return audio_raw
            concat_list = self.cfg.work_dir / "demucs_concat.txt"
            concat_list.write_text(
                "\n".join(f"file '{str(p).replace(chr(92), '/')}'" for p in chunk_bg_paths),
                encoding="utf-8",
            )
            self._run_proc(
                [self._ffmpeg, "-y", "-f", "concat", "-safe", "0",
                 "-i", str(concat_list),
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 "-acodec", "pcm_s16le", str(bg_path)],
                check=True, capture_output=True,
            )

            # Cleanup chunks
            for p in chunk_bg_paths:
                p.unlink(missing_ok=True)
            concat_list.unlink(missing_ok=True)

            print(f"[DEMUCS] Background track extracted: {bg_path}", flush=True)
            return bg_path

        except ImportError:
            print("[DEMUCS] demucs not installed, falling back to raw audio", flush=True)
        except Exception as e:
            print(f"[DEMUCS] Separation failed: {e}, falling back to raw audio", flush=True)
        return audio_raw

    def _demucs_single(self, audio_path: Path, output_path: Path) -> Path:
        """Run demucs on a single audio file and return the no-vocals track."""
        try:
            import demucs.separate
            demucs_out = self.cfg.work_dir / "demucs_out"
            print(f"[DEMUCS] Separating {audio_path.name}...", flush=True)
            demucs.separate.main([
                "--two-stems", "vocals",
                "-n", "htdemucs",
                "-o", str(demucs_out),
                str(audio_path),
            ])
            stem_name = audio_path.stem
            no_vocals = demucs_out / "htdemucs" / stem_name / "no_vocals.wav"
            if no_vocals.exists():
                self._run_proc(
                    [self._ffmpeg, "-y", "-i", str(no_vocals),
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     "-acodec", "pcm_s16le", str(output_path)],
                    check=True, capture_output=True,
                )
                shutil.rmtree(demucs_out, ignore_errors=True)
                return output_path
            print(f"[DEMUCS] no_vocals.wav not found for {audio_path.name}", flush=True)
        except Exception as e:
            print(f"[DEMUCS] Failed on {audio_path.name}: {e}", flush=True)
        return audio_path

    def _mix_audio(self, original: Path, tts: Path, original_vol: float) -> Path:
        # Use background-only track (no vocals) instead of full original
        bg_track = self._separate_background(original)
        mixed = self.cfg.work_dir / "audio_mixed.wav"
        self._run_proc(
            [
                self._ffmpeg, "-y",
                "-i", str(tts),
                "-i", str(bg_track),
                "-filter_complex",
                (
                    f"[0:a]asplit=2[tts_out][tts_sc];"
                    f"[1:a]volume={original_vol}[bg_raw];"
                    f"[bg_raw][tts_sc]sidechaincompress="
                    f"threshold=0.02:ratio=3:attack=20:release=300:makeup=1[bg_duck];"
                    f"[tts_out][bg_duck]amix=inputs=2:duration=longest:dropout_transition=2[out]"
                ),
                "-map", "[out]",
                "-ar", str(self.SAMPLE_RATE),
                "-ac", str(self.N_CHANNELS),
                str(mixed),
            ],
            check=True,
            capture_output=True,
        )
        return mixed

    # ── Video split / concat ─────────────────────────────────────────────
    def _split_video(self, video_path: Path, start: float, duration: float, output_path: Path):
        """Extract a clip from the video using stream copy (fast, no re-encode)."""
        self._run_proc(
            [self._ffmpeg, "-y",
             "-ss", f"{start:.3f}", "-i", str(video_path),
             "-t", f"{duration:.3f}",
             "-c", "copy", "-an",  # copy video only, drop audio
             str(output_path)],
            check=True, capture_output=True,
        )

    def _concatenate_videos(self, video_paths: List[Path], output_path: Path):
        """Concatenate multiple video files using ffmpeg concat demuxer."""
        concat_list = self.cfg.work_dir / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for vp in video_paths:
                # ffmpeg concat needs forward slashes even on Windows
                safe_path = str(vp).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")
        self._run_proc(
            [self._ffmpeg, "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy",
             str(output_path)],
            check=True, capture_output=True,
        )

    # ── Fast assembly: build audio timeline + stream-copy video ────────
    def _assemble_fast_mux(self, video_path, audio_raw, tts_data, total_video_duration):
        """AUDIO-FIRST assembly. TTS audio is NEVER touched after generation.

        Philosophy:
        1. TTS audio is sacred — full duration, untouched, every word heard
        2. 1.15x speedup applied BEFORE this step (in TTS generation)
        3. Place segments sequentially with 1s silence gaps
        4. Build per-segment video clips, each slowed/frozen to match its audio
        5. Output can be ANY length — no constraint from original video

        Flow:
        TTS audio (untouched) → place with 1s gaps → build video per-segment → concat
        """
        print("[Assembly] Using: _assemble_fast_mux (audio-first, per-segment video)", flush=True)
        num_segs = len(tts_data)
        self._report("assemble", 0.02, f"Audio-first assembly: {num_segs} segments...")

        # ── Step 1: Apply 1.15x speedup (skipped under no_time_pressure) ──
        # The new tts_no_time_pressure flag (default ON) takes precedence over
        # all other gates. When set, NO speedup is applied — TTS audio plays
        # at natural pace and the audio-first timeline below stretches to fit.
        no_pressure = getattr(self.cfg, 'tts_no_time_pressure', True)
        if not no_pressure and not self.cfg.audio_untouchable and getattr(self.cfg, 'enable_duration_fit', True):
            self._report("assemble", 0.03, "Applying 1.15x base speedup...")
            tts_data = self._apply_base_speedup(tts_data, 1.15)
        elif no_pressure:
            print("[NO-TIME-PRESSURE] _assemble_fast_mux: skipped 1.15x base speedup",
                  flush=True)

        # ── Step 2: Build audio timeline — place each segment with 1s gap ──
        # Sort by original start time
        sorted_segs = sorted(tts_data, key=lambda s: s.get("start", 0))
        audio_pos = 0.0  # current position in output audio timeline

        for seg in sorted_segs:
            seg["_audio_start"] = audio_pos
            seg["_audio_end"] = audio_pos + seg.get("duration", 0)
            gap = self.SENTENCE_GAP if getattr(self.cfg, 'enable_sentence_gap', True) else 0.0
            audio_pos = seg["_audio_end"] + gap

        total_audio_duration = audio_pos
        gap_label = f"{num_segs}x 1s gaps" if getattr(self.cfg, 'enable_sentence_gap', True) else "no gaps"
        self._report("assemble", 0.10,
                     f"Audio timeline: {total_audio_duration:.0f}s "
                     f"({num_segs} segments + {gap_label})")

        # Build WAV timeline from _audio_start positions
        audio_entries = []
        for seg in sorted_segs:
            wav = seg.get("wav")
            if not wav or not Path(wav).exists():
                continue
            audio_entries.append({
                "start": seg["_audio_start"],
                "wav": wav,
                "duration": seg.get("duration", 0),
            })

        dubbed_audio = self._build_timeline(audio_entries, total_audio_duration, prefix="fast_")
        audio_duration = self._get_duration(dubbed_audio)

        # ── Step 3: Match video to audio (single FFmpeg pass — FAST) ──
        if audio_duration > total_video_duration + 0.5:
            # Audio is longer — slow video uniformly to match
            slowdown = audio_duration / total_video_duration
            self._report("assemble", 0.50,
                         f"Slowing video {slowdown:.2f}x to match {audio_duration:.0f}s audio "
                         f"(original {total_video_duration:.0f}s)...")
            slowed_video = self.cfg.work_dir / "video_slowed.mp4"
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(video_path),
                 "-filter:v", f"setpts={slowdown}*PTS",
                 "-an",
                 *self._video_encode_args(),  # NVENC when available
                 str(slowed_video)],
                check=True, capture_output=True,
            )
            self._report("assemble", 0.85, "Muxing slowed video + dubbed audio...")
            self._mux_replace_audio(slowed_video, dubbed_audio, self.cfg.output_path)
        else:
            # Audio fits — direct mux
            self._report("assemble", 0.85, "Muxing video + dubbed audio...")
            self._mux_replace_audio(video_path, dubbed_audio, self.cfg.output_path)

        final_dur = self._get_duration(self.cfg.output_path)
        self._report("assemble", 0.95,
                     f"Complete! {num_segs} segments, {final_dur:.0f}s output "
                     f"(original {total_video_duration:.0f}s)")

    def _make_freeze_clip(self, video_path: Path, timestamp: float, duration: float, output: Path):
        """Extract a single frame from video at timestamp and create a still video clip."""
        timestamp = max(0, timestamp)
        frame = output.with_suffix(".png")
        self._run_proc(
            [self._ffmpeg, "-y",
             "-ss", f"{timestamp:.3f}", "-i", str(video_path),
             "-frames:v", "1", "-update", "1", str(frame)],
            check=True, capture_output=True,
        )
        if not frame.exists():
            # Fallback: first frame
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(video_path),
                 "-frames:v", "1", "-update", "1", str(frame)],
                check=True, capture_output=True,
            )
        self._run_proc(
            [self._ffmpeg, "-y",
             "-loop", "1", "-i", str(frame),
             "-t", f"{duration:.3f}",
             "-vf", "fps=30",
             *self._video_encode_args(force_cpu=True),  # CPU for still image input
             "-pix_fmt", "yuv420p",
             str(output)],
            check=True, capture_output=True,
        )
        frame.unlink(missing_ok=True)

    # ── Audio enhancement ─────────────────────────────────────────────────
    # Silence WAV (1 second) for punctuation gaps — generated once, reused
    _silence_1s_path = None

    def _get_silence_wav(self):
        """Get or create a 1-second silence WAV file for punctuation gaps."""
        if self._silence_1s_path and self._silence_1s_path.exists():
            return self._silence_1s_path
        silence = self.cfg.work_dir / "_silence_1s.wav"
        if not silence.exists():
            self._run_proc(
                [self._ffmpeg, "-y", "-f", "lavfi",
                 "-i", f"anullsrc=r={self.SAMPLE_RATE}:cl=stereo",
                 "-t", "1.0", "-acodec", "pcm_s16le", str(silence)],
                check=True, capture_output=True,
            )
        self._silence_1s_path = silence
        return silence

    def _insert_punctuation_pauses(self, wav_path: Path, text: str) -> Path:
        """Insert 1-second silence after each sentence-ending punctuation in a WAV.

        For non-SSML engines (XTTS, Sarvam): splits at punctuation boundaries,
        uses ffmpeg silence detection to find natural pauses, then extends them.

        Simpler approach: detect silence regions in the WAV and extend any
        that correspond to sentence boundaries to exactly 1 second.
        """
        if not text or not wav_path.exists():
            return wav_path

        # Count sentence boundaries
        boundaries = len(re.findall(r'[.!?।॥]\s+', text))
        if boundaries == 0:
            return wav_path  # Single sentence, no pauses needed

        # Use ffmpeg silencedetect to find natural pauses
        result = self._run_proc(
            [self._ffmpeg, "-i", str(wav_path),
             "-af", "silencedetect=noise=-40dB:d=0.15",
             "-f", "null", "-"],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
        )

        # Parse silence regions from stderr
        import re as _re
        silences = []
        for line in result.stderr.split('\n'):
            start_m = _re.search(r'silence_start:\s*([\d.]+)', line)
            end_m = _re.search(r'silence_end:\s*([\d.]+)', line)
            if start_m:
                silences.append({'start': float(start_m.group(1))})
            elif end_m and silences and 'end' not in silences[-1]:
                silences[-1]['end'] = float(end_m.group(1))

        if not silences:
            return wav_path  # No silences detected, leave as-is

        # Extend each detected silence to at least 1 second using apad filter
        # Build a concat approach: split at silences, insert 1s gap, rejoin
        silence_wav = self._get_silence_wav()
        total_dur = self._get_duration(wav_path)

        # Build ffmpeg filter to insert silences
        # Strategy: split WAV at each silence midpoint, concat with 1s silence between
        split_points = []
        for s in silences[:boundaries]:  # Only as many as sentence boundaries
            if 'end' in s:
                mid = (s['start'] + s['end']) / 2
                split_points.append(mid)

        if not split_points:
            return wav_path

        # Create parts and concat with silence
        parts = []
        prev = 0
        for idx, sp in enumerate(split_points):
            part_path = wav_path.with_suffix(f'.part{idx}.wav')
            self._run_proc(
                [self._ffmpeg, "-y", "-i", str(wav_path),
                 "-ss", f"{prev:.3f}", "-t", f"{sp - prev:.3f}",
                 "-acodec", "pcm_s16le", str(part_path)],
                check=True, capture_output=True,
            )
            parts.append(part_path)
            parts.append(silence_wav)  # 1s gap
            prev = sp

        # Last part
        last_part = wav_path.with_suffix(f'.part{len(split_points)}.wav')
        self._run_proc(
            [self._ffmpeg, "-y", "-i", str(wav_path),
             "-ss", f"{prev:.3f}",
             "-acodec", "pcm_s16le", str(last_part)],
            check=True, capture_output=True,
        )
        parts.append(last_part)

        # Concat all parts
        concat_list = wav_path.with_suffix('.concat.txt')
        with open(concat_list, 'w') as f:
            for p in parts:
                f.write(f"file '{p}'\n")

        output = wav_path.with_suffix('.paused.wav')
        self._run_proc(
            [self._ffmpeg, "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list),
             "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
             "-acodec", "pcm_s16le", str(output)],
            check=True, capture_output=True,
        )

        # Clean up parts
        for p in parts:
            if p != silence_wav:
                p.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)

        # Replace original
        wav_path.unlink(missing_ok=True)
        output.replace(wav_path)

        return wav_path

    def _enhance_tts_wav(self, wav_path: Path) -> Path:
        """Normalize speech loudness, trim silence, and apply light compression per TTS segment.

        Respects post_tts_level config:
        - "full":    silenceremove + fade + speechnorm + compressor (best quality)
        - "minimal": fade + speechnorm only (lighter, preserves natural dynamics)
        - "none":    zero processing (raw TTS output)
        """
        level = getattr(self.cfg, 'post_tts_level', 'full')

        if level == "none" or getattr(self.cfg, 'audio_untouchable', False):
            return wav_path  # Raw TTS, no modifications

        enhanced = wav_path.with_suffix(".enh.wav")
        try:
            # ── Step A: Remove ALL silence using soundfile+numpy (fastest, 12ms/seg) ──
            # TTS engines insert 0.2-0.5s silence between phrases and at edges.
            # Soundfile+numpy is 5x faster than ffmpeg silenceremove and more precise.
            import soundfile as _sf
            import numpy as _np

            try:
                data, srate = _sf.read(str(wav_path))
                mono = _np.mean(data, axis=1) if data.ndim > 1 else data

                # RMS energy per 10ms frame
                frame_len = int(srate * 0.01)
                n_frames = len(mono) // frame_len
                if n_frames > 0:
                    energy = _np.array([
                        _np.sqrt(_np.mean(mono[i * frame_len:(i + 1) * frame_len] ** 2))
                        for i in range(n_frames)
                    ])
                    threshold = 0.01  # ~-40dB

                    # Find speech regions (min 30ms speech, skip silences > 150ms)
                    regions = []
                    in_speech = False
                    start_f = 0
                    silence_frames = 0
                    MIN_SILENCE_FRAMES = 15  # 150ms = silence to remove

                    for i, e in enumerate(energy):
                        if e > threshold:
                            if not in_speech:
                                start_f = i
                                in_speech = True
                            silence_frames = 0
                        else:
                            if in_speech:
                                silence_frames += 1
                                if silence_frames >= MIN_SILENCE_FRAMES:
                                    # End of speech region
                                    end_f = i - silence_frames + 1
                                    if end_f > start_f:
                                        regions.append((start_f * frame_len, end_f * frame_len))
                                    in_speech = False
                                    silence_frames = 0

                    if in_speech:
                        regions.append((start_f * frame_len, len(data)))

                    if regions:
                        trimmed = _np.concatenate([data[s:e] for s, e in regions])
                        _sf.write(str(wav_path), trimmed, srate)
            except Exception:
                pass  # If numpy trim fails, continue with original WAV

            # ── Step B: Apply ffmpeg audio enhancement (loudness, compression, fade) ──
            if level == "minimal":
                af_chain = (
                    "afade=t=in:st=0:d=0.01,"
                    "speechnorm=e=50:r=0.0001:l=1,"
                    "areverse,afade=t=in:st=0:d=0.01,areverse"
                )
            else:  # "full"
                af_chain = (
                    "afade=t=in:st=0:d=0.01,"
                    "speechnorm=e=50:r=0.0001:l=1,"
                    "acompressor=threshold=-20dB:ratio=3:attack=5:release=50:makeup=2,"
                    "areverse,afade=t=in:st=0:d=0.01,areverse"
                )

            self._run_proc(
                [
                    self._ffmpeg, "-y", "-i", str(wav_path),
                    "-af", af_chain,
                    "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                    str(enhanced),
                ],
                check=True, capture_output=True,
            )
            wav_path.unlink(missing_ok=True)
            enhanced.replace(wav_path)
        except Exception:
            enhanced.unlink(missing_ok=True)  # keep original on failure
        return wav_path

    # ── Librosa quality audio processor (process-isolated) ──────────────
    @staticmethod
    def _librosa_process_worker(args):
        """Process a single segment with librosa in a separate OS process.

        Higher quality than ffmpeg for: time-stretch, trim, spectral smoothing.
        Process-isolated — safe for librosa/numba.
        """
        import librosa
        import numpy as np
        import soundfile as sf

        wav_path, target_sr, n_channels, stretch_rate = args

        try:
            y, sr = librosa.load(wav_path, sr=None, mono=True)

            # 1. Smart silence trim (better than ffmpeg silenceremove)
            y, _ = librosa.effects.trim(y, top_db=30)

            # 2. Time-stretch if needed (phase-vocoder — higher quality than ffmpeg atempo)
            if stretch_rate and abs(stretch_rate - 1.0) > 0.03:
                y = librosa.effects.time_stretch(y, rate=stretch_rate)

            # 3. De-noise: spectral gating (remove TTS hum/artifacts)
            # Simple approach: reduce frequencies below energy threshold
            S = librosa.stft(y)
            mag = np.abs(S)
            # Compute noise floor from quietest 10% of frames
            frame_energy = np.mean(mag, axis=0)
            noise_threshold = np.percentile(frame_energy, 10) * 2
            # Gate: reduce frames below threshold
            mask = np.ones_like(mag)
            for i in range(mag.shape[1]):
                if frame_energy[i] < noise_threshold:
                    mask[:, i] = 0.1  # reduce noise frames to 10%
            S_clean = S * mask
            y = librosa.istft(S_clean, length=len(y))

            # 4. Peak normalize to -1dB (headroom for assembly)
            peak = np.max(np.abs(y))
            if peak > 0.001:
                target_peak = 10 ** (-1 / 20)  # -1dB
                y = y * (target_peak / peak)

            # 5. Fade in/out (5ms — prevent clicks)
            fade_samples = int(sr * 0.005)
            if len(y) > fade_samples * 2:
                y[:fade_samples] *= np.linspace(0, 1, fade_samples)
                y[-fade_samples:] *= np.linspace(1, 0, fade_samples)

            # Write back
            sf.write(wav_path, y, sr)
            duration = len(y) / sr
            return (True, duration)

        except Exception as e:
            return (False, 0)

    def _smooth_librosa_quality(self, tts_data):
        """Quality mode: process all segments with librosa via ProcessPoolExecutor.

        Uses librosa for: smart trim + spectral de-noise + peak normalize + fade.
        Process-isolated — 4 workers, no thread-safety issues.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        total = len(tts_data)
        self._report("synthesize", 0.93,
                     f"Quality mode: librosa processing {total} segments (4 processes)...")

        args_list = []
        indices = []
        for i, tts in enumerate(tts_data):
            wav = tts.get("wav")
            if not wav or not Path(wav).exists():
                continue
            args_list.append((str(wav), self.SAMPLE_RATE, self.N_CHANNELS, None))
            indices.append(i)

        if not args_list:
            return tts_data

        try:
            # submit + as_completed instead of pool.map so we can stream progress
            # every ~25 segments. Without per-item callbacks the UI stays frozen
            # at 93% for 15-20 minutes on long videos (silent-step pattern).
            with ProcessPoolExecutor(max_workers=4) as pool:
                future_to_idx = {
                    pool.submit(self._librosa_process_worker, args): idx
                    for args, idx in zip(args_list, indices)
                }
                processed = 0
                done = 0
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        success, new_dur = fut.result()
                    except Exception:
                        success, new_dur = False, 0
                    if success and new_dur > 0:
                        tts_data[idx]["duration"] = new_dur
                        processed += 1
                    done += 1
                    if done % 25 == 0 or done == total:
                        self._report("synthesize", 0.93,
                                     f"[Librosa] {done}/{total} segments processed ({processed} ok)")

            self._report("synthesize", 0.95,
                         f"Quality mode: {processed}/{total} segments processed with librosa")

        except Exception as e:
            print(f"[Quality Mode] Librosa processing failed: {e} — falling back to fast mode",
                  flush=True)
            return self._smooth_fast_ffmpeg(tts_data)

        return tts_data

    def _smooth_fast_ffmpeg(self, tts_data):
        """Fast mode: process all segments with ffmpeg (parallel threads).

        Uses ffmpeg for: loudnorm + highpass + compressor + fade.
        ThreadPoolExecutor — 8 workers, fast subprocess calls.
        """
        import threading
        total = len(tts_data)
        self._report("synthesize", 0.93,
                     f"Fast mode: ffmpeg smoothing {total} segments (8 threads)...")

        af_chain = (
            "highpass=f=80,"
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=none,"
            "acompressor=threshold=-25dB:ratio=2.5:attack=5:release=80,"
            "afade=t=in:st=0:d=0.005,"
            "areverse,afade=t=in:st=0:d=0.005,areverse"
        )
        ffmpeg = self._ffmpeg
        sr = str(self.SAMPLE_RATE)
        nc = str(self.N_CHANNELS)

        def smooth_one(idx_tts):
            idx, tts = idx_tts
            wav = tts.get("wav")
            if not wav or not Path(wav).exists():
                return (idx, False, 0)
            wav_path = Path(wav)
            tid = threading.current_thread().ident
            output = wav_path.with_suffix(f".s{tid}.wav")
            try:
                self._run_proc(
                    [ffmpeg, "-y", "-i", str(wav_path), "-af", af_chain,
                     "-ar", sr, "-ac", nc, str(output)],
                    check=True, capture_output=True,
                )
                wav_path.unlink(missing_ok=True)
                output.replace(wav_path)
                new_dur = self._get_duration(str(wav_path))
                return (idx, True, new_dur)
            except Exception:
                output.unlink(missing_ok=True)
                return (idx, False, 0)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = min(8, total)
        results: List = []
        completed = 0
        # Emit live progress so the UI doesn't freeze on long videos.
        # 0.93 → 0.95 covers this step.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(smooth_one, (idx, tts)): idx
                       for idx, tts in enumerate(tts_data)}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                except Exception:
                    results.append((futures[fut], False, 0))
                completed += 1
                # Report every ~50 segments OR every 2% step progress
                if completed % 50 == 0 or completed == total:
                    pct = 0.93 + 0.02 * (completed / total)
                    self._report(
                        "synthesize", pct,
                        f"Smoothing audio: {completed}/{total} segments "
                        f"({completed*100//total}%)",
                    )

        smoothed = 0
        for idx, success, new_dur in results:
            if success:
                tts_data[idx]["duration"] = new_dur
                smoothed += 1

        self._report("synthesize", 0.95,
                     f"Fast mode: {smoothed}/{total} segments smoothed")
        return tts_data

    def _smooth_multi_engine_audio(self, tts_data):
        """Route to quality (librosa) or fast (ffmpeg) mode based on config.

        - "quality": librosa ProcessPoolExecutor — smart trim, de-noise, spectral clean
        - "fast": ffmpeg ThreadPoolExecutor — loudnorm, highpass, compressor

        Both take ~3s for 100 segments when parallelized.
        Skipped when post_tts_level is "none" or audio_untouchable is True.
        """
        level = getattr(self.cfg, 'post_tts_level', 'full')
        if level == "none" or getattr(self.cfg, 'audio_untouchable', False):
            return tts_data

        mode = getattr(self.cfg, 'audio_quality_mode', 'fast')
        if mode == "quality":
            return self._smooth_librosa_quality(tts_data)
        else:
            return self._smooth_fast_ffmpeg(tts_data)

    # ── TTS Manager: validate, retry, sequence, normalize, assemble ─────
    # Target words-per-minute by language (for duration validation)
    _TTS_WPM = {"hi": 135, "bn": 130, "ta": 120, "te": 120, "en": 160,
                "es": 150, "fr": 150, "de": 140, "ja": 100, "ko": 110}

    def _tts_manager(self, tts_data, segments):
        """Post-TTS Manager: validate, retry, sequence, normalize, combine.

        Runs AFTER triple parallel TTS, BEFORE assembly. Uses pydub + soundfile.
        All heavy steps are PARALLELIZED with ThreadPoolExecutor for speed.

        Pipeline:
        1. MATCH — link tts_data to segments by _seg_idx or timestamp
        2. RETRY — re-generate missing/failed segments with Edge-TTS
        3. VERIFY — duration + energy check (every segment)
        4. SEQUENCE — sort by segment index
        5. GAP — smart silence (only where original had gaps)

        Removed (redundant):
        - Old validate (soundfile RMS) — merged into verify
        - Normalize (pydub -16dBFS) — _smooth_multi_engine_audio already does this
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import numpy as np

        total = len(segments)
        self._report("synthesize", 0.93, f"TTS Manager: {len(tts_data)} segments...")

        # ── Step 1: MATCH — link tts results to segments ───────────────────
        tts_by_idx = {t.get("_seg_idx"): t for t in tts_data if "_seg_idx" in t}
        tts_by_start = {round(t["start"], 2): t for t in tts_data}

        valid = []
        failed_indices = []
        for idx in range(total):
            seg = segments[idx]
            tts = tts_by_idx.get(idx) or tts_by_start.get(round(seg.get("start", 0), 2))
            if tts and tts.get("wav") and Path(tts["wav"]).exists() and Path(tts["wav"]).stat().st_size > 500:
                valid.append((idx, tts))
            else:
                failed_indices.append(idx)

        self._report("synthesize", 0.94,
                     f"Matched: {len(valid)}/{total}, {len(failed_indices)} missing")

        # ── Step 2: RETRY missing → Edge-TTS ──
        if failed_indices:
            self._report("synthesize", 0.94,
                         f"Retrying {len(failed_indices)} failed segments via Edge-TTS...")
            retry_dir = self.cfg.work_dir / "tts_retry"
            retry_dir.mkdir(exist_ok=True)
            retry_segs = [segments[i] for i in failed_indices]
            # Tag retry segs with their original index for matching back
            for fi, seg in zip(failed_indices, retry_segs):
                seg["_retry_idx"] = fi
            try:
                retry_results = self._tts_edge(retry_segs, voice_map=self._voice_map,
                                               work_dir=retry_dir)
                # Match by _seg_idx first, then timestamp fallback
                for result in retry_results:
                    matched = False
                    for idx in list(failed_indices):
                        seg = segments[idx]
                        # Match by _seg_idx (exact) or timestamp (fallback)
                        if (result.get("_seg_idx") == idx or
                            abs(result["start"] - seg.get("start", 0)) < 0.01):
                            result["_validated"] = True
                            result["_seg_idx"] = idx
                            result["_expected_text"] = seg.get("_expected_text", seg.get("text_translated", ""))
                            valid.append((idx, result))
                            failed_indices.remove(idx)
                            matched = True
                            break
                    if not matched and failed_indices:
                        # Last resort: assign to first unmatched
                        idx = failed_indices.pop(0)
                        result["_validated"] = True
                        result["_seg_idx"] = idx
                        valid.append((idx, result))
            except Exception as e:
                print(f"[TTS Manager] Retry failed: {e}", flush=True)

            if failed_indices:
                print(f"[TTS Manager] WARNING: {len(failed_indices)} still failed: "
                      f"{failed_indices}", flush=True)

        # ── Step 3: VERIFY — Duration + Energy check (every segment) ──────
        # Re-transcription doesn't work for Hindi (Whisper writes in different
        # script/spelling = false failures on perfect audio).
        # Instead: verify audio duration matches word count AND has speech energy.
        # Catches: silent audio, truncated audio, garbled/empty audio.
        #
        # IMPORTANT: The duration check (40%-250% of expected) has a 70%+ false
        # positive rate for Hindi because Hindi WPM varies wildly per sentence.
        # Each false positive triggers a sequential Edge-TTS retry, turning what
        # should be a 5-second cleanup into a 2-hour bottleneck on long videos.
        # OFF by default. Set cfg.enable_tts_verify_retry=True to re-enable.
        MAX_RETRIES = 2 if getattr(self.cfg, 'enable_tts_verify_retry', False) else 0
        if MAX_RETRIES == 0:
            self._report("synthesize", 0.96,
                         f"TTS verify retry disabled (cfg.enable_tts_verify_retry=False) — "
                         f"accepting all {len(valid)} segments")
        hi_wpm = LANGUAGE_WPM.get(self.cfg.target_language, 120)

        # Hard short-circuit: if retries are disabled, skip the per-segment
        # energy/duration scan entirely. The scan reads every wav from disk
        # via soundfile + numpy and is itself ~30-60s on long videos for no
        # benefit when MAX_RETRIES=0.
        skip_verify_scan = (MAX_RETRIES == 0)
        if skip_verify_scan:
            self._report("synthesize", 0.97,
                         f"Skipped per-segment verify ({len(valid)} segments accepted)")
            passed = len(valid)
            regen_count = 0
        else:
            self._report("synthesize", 0.95,
                         f"Verifying {len(valid)} segments (duration + energy)...")

        retry_dir = self.cfg.work_dir / "tts_verify_retry"
        if not skip_verify_scan:
            retry_dir.mkdir(exist_ok=True)
            passed = 0
            regen_count = 0

            # ── Phase 1: PARALLEL SCAN — score every segment with ThreadPool ──
            # Reads each wav, computes duration + energy bursts. Pure I/O, scales
            # well with threads. Was sequential before → 30-60s on long videos.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import soundfile as _sf
            import numpy as _np

            def _scan_one(args):
                vi, (idx, tts) = args
                wav_path = tts.get("wav")
                expected_text = (segments[idx].get("text_translated",
                                  segments[idx].get("text", ""))
                                 if idx < len(segments) else "")
                if not wav_path or not Path(wav_path).exists() or not expected_text:
                    return (vi, idx, True, 0.0, "ok")  # accept missing/empty
                word_count = len(expected_text.split())
                actual_dur = tts.get("duration", 0)
                expected_dur = (word_count / hi_wpm) * 60 if word_count > 0 else 0.5
                dur_ok = (actual_dur >= expected_dur * 0.40) and (actual_dur <= expected_dur * 2.5)

                energy_ok = True
                try:
                    data, sr = _sf.read(str(wav_path))
                    mono = _np.mean(data, axis=1) if data.ndim > 1 else data
                    frame_len = int(sr * 0.03)
                    n_frames = len(mono) // frame_len
                    if n_frames > 0:
                        # Vectorized RMS — much faster than the old per-frame loop
                        trimmed = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
                        energy = _np.sqrt(_np.mean(trimmed ** 2, axis=1))
                        threshold = 0.01
                        # Count bursts: rising edges crossing threshold
                        above = energy > threshold
                        if len(above) > 1:
                            rising = _np.logical_and(above[1:], _np.logical_not(above[:-1]))
                            bursts = int(rising.sum()) + (1 if above[0] else 0)
                        else:
                            bursts = 1 if above[0] else 0
                        min_bursts = max(1, word_count // 3)
                        energy_ok = bursts >= min_bursts
                except Exception:
                    pass

                ok = dur_ok and energy_ok
                reason = ""
                if not ok:
                    parts = []
                    if not dur_ok:
                        parts.append(f"dur {actual_dur:.1f}s vs {expected_dur:.1f}s")
                    if not energy_ok:
                        parts.append("low energy")
                    reason = ", ".join(parts)
                return (vi, idx, ok, expected_dur, reason)

            self._report("synthesize", 0.95,
                         f"Scanning {len(valid)} segments in parallel (32 workers)...")

            scan_results = []
            scan_done = 0
            with ThreadPoolExecutor(max_workers=32) as pool:
                futures = [pool.submit(_scan_one, (vi, v))
                           for vi, v in enumerate(valid)]
                for fut in as_completed(futures):
                    try:
                        scan_results.append(fut.result())
                    except Exception:
                        pass
                    scan_done += 1
                    if scan_done % 200 == 0 or scan_done == len(valid):
                        self._report("synthesize",
                                     0.95 + 0.01 * (scan_done / len(valid)),
                                     f"Scanned {scan_done}/{len(valid)} segments")

            # Sort scan_results by vi to maintain order
            scan_results.sort(key=lambda r: r[0])
            failed_to_retry = [
                (vi, idx, expected_dur)
                for (vi, idx, ok, expected_dur, _reason) in scan_results
                if not ok
            ]
            passed = len(valid) - len(failed_to_retry)
            print(f"[TTS Verify] scan: {passed} passed, {len(failed_to_retry)} flagged",
                  flush=True)

            # ── Phase 2: BATCHED PARALLEL RETRY ────────────────────────────────
            # _tts_edge already runs 120 segments in parallel. So instead of
            # calling it 1-segment-at-a-time (which gives no speedup), we call
            # it ONCE with all the failed segments at once. Was 0.3 seg/sec
            # sequential → ~17 seg/sec batched (50× faster, the same speedup
            # the main TTS step gets).
            if failed_to_retry and MAX_RETRIES > 0:
                self._report("synthesize", 0.96,
                             f"Re-generating {len(failed_to_retry)} flagged segments "
                             f"(parallel x120)...")
                # Build a list of segment dicts to re-TTS
                retry_segs = []
                for vi, idx, expected_dur in failed_to_retry:
                    if idx < len(segments):
                        seg_copy = dict(segments[idx])
                        seg_copy["_retry_vi"] = vi
                        seg_copy["_seg_idx"] = idx
                        seg_copy["_expected_dur"] = expected_dur
                        retry_segs.append(seg_copy)

                for retry_round in range(MAX_RETRIES):
                    if not retry_segs:
                        break
                    round_dir = retry_dir / f"round_{retry_round}"
                    round_dir.mkdir(exist_ok=True)
                    self._report("synthesize", 0.96,
                                 f"Retry round {retry_round+1}/{MAX_RETRIES}: "
                                 f"{len(retry_segs)} segments via Edge-TTS x120")
                    try:
                        results = self._tts_edge(
                            retry_segs,
                            voice_map=self._voice_map,
                            work_dir=round_dir,
                        )
                    except Exception as e:
                        print(f"[TTS Verify] retry round {retry_round} failed: {e}",
                              flush=True)
                        results = []

                    # Match results back, accept those that now pass duration
                    still_failed = []
                    for retry_idx, seg in enumerate(retry_segs):
                        vi = seg["_retry_vi"]
                        expected_dur = seg["_expected_dur"]
                        # Find this seg's result in `results`
                        matched_result = None
                        for r in results:
                            if r.get("_seg_idx") == seg["_seg_idx"]:
                                matched_result = r
                                break
                        if matched_result and matched_result.get("wav"):
                            new_dur = matched_result.get("duration", 0)
                            if (new_dur >= expected_dur * 0.40) and (new_dur <= expected_dur * 2.5):
                                matched_result["_seg_idx"] = seg["_seg_idx"]
                                valid[vi] = (seg["_seg_idx"], matched_result)
                                regen_count += 1
                                continue
                        still_failed.append(seg)
                    retry_segs = still_failed
                    self._report("synthesize", 0.97,
                                 f"Round {retry_round+1}: re-generated "
                                 f"{regen_count} so far, {len(still_failed)} still flagged")

                if retry_segs:
                    print(f"[TTS Verify] After {MAX_RETRIES} rounds, "
                          f"{len(retry_segs)} segments still flagged — accepted anyway",
                          flush=True)

        print(f"[TTS Verify] {len(valid) if not skip_verify_scan else 0} checked, "
              f"{regen_count if not skip_verify_scan else 0} re-generated", flush=True)

        # ── Step 4: SEQUENCE — sort by segment index ───────────────────────
        valid.sort(key=lambda x: x[0])

        # Normalize removed — _smooth_multi_engine_audio already normalizes loudness.
        # Double-normalizing degrades audio quality.

        # ── Step 5: GAP + segment_map ──────────────────────────────────────
        # Smart gaps: only insert silence where the ORIGINAL video had a gap.
        # Continuous narration → no artificial pauses. Gaps between scenes → keep them.
        segment_map = []
        use_gaps = getattr(self.cfg, 'enable_sentence_gap', True)
        ORIGINAL_GAP_THRESHOLD = 0.3  # If original gap was < 0.3s, speaker was continuous
        audio_pos = 0.0

        for i, (seg_idx, tts) in enumerate(valid):
            dur = tts.get("duration", 0)

            # Calculate gap: use original video gap if it existed, otherwise 0
            gap_dur = 0.0
            if use_gaps and i < len(valid) - 1:
                curr_end = tts.get("end", 0)
                next_start = valid[i + 1][1].get("start", 0)
                original_gap = next_start - curr_end
                if original_gap >= ORIGINAL_GAP_THRESHOLD:
                    # Original had a real gap (scene break, pause) → preserve it
                    gap_dur = min(original_gap, 1.5)  # cap at 1.5s
                # else: continuous speech → no gap

            entry = {
                "start": tts.get("start", 0),
                "end": tts.get("end", 0),
                "wav": tts.get("wav"),
                "duration": dur,
                "_audio_start": audio_pos,
                "_audio_end": audio_pos + dur,
            }
            # Preserve flags from TTS engines (e.g. Edge-TTS _already_sped)
            if tts.get("_already_sped"):
                entry["_already_sped"] = True
            if tts.get("_seg_idx") is not None:
                entry["_seg_idx"] = tts["_seg_idx"]
            segment_map.append(entry)
            audio_pos += dur
            if i < len(valid) - 1:
                audio_pos += gap_dur

        # Deep QC removed — soundfile+numpy validation in Step 1 already handles
        # silence detection, duration check, and RMS energy verification.

        self._report("synthesize", 0.99,
                     f"TTS Manager: {len(segment_map)} segments, "
                     f"{audio_pos:.0f}s total ({gap_dur}s gaps)")
        return segment_map

    def _verify_tts_completeness(self, tts_data, text_segments):
        """Verify ALL non-empty segments have audio before proceeding to assembly.

        Logs a clear warning for any segments that are missing. Does NOT block
        assembly (to avoid losing a 90% complete job), but provides visibility
        into exactly which segments are missing and why.

        Returns the count of missing segments.
        """
        # Count non-empty input segments
        non_empty = [i for i, seg in enumerate(text_segments)
                     if seg.get("text_translated", seg.get("text", "")).strip()]
        produced = len(tts_data)
        missing_count = len(non_empty) - produced

        if missing_count > 0:
            # Find which segment indices are missing
            produced_starts = {round(t.get("start", -1), 2) for t in tts_data}
            produced_idxs = {t.get("_seg_idx") for t in tts_data if "_seg_idx" in t}
            missing_details = []
            for i in non_empty:
                seg = text_segments[i]
                seg_start = round(seg.get("start", 0), 2)
                seg_idx = seg.get("_seg_idx", i)
                if seg_idx not in produced_idxs and seg_start not in produced_starts:
                    text_preview = seg.get("text_translated", seg.get("text", ""))[:60]
                    missing_details.append(f"  Seg {i} @ {seg.get('start', 0):.1f}s: {text_preview!r}")
            if missing_details:
                print(f"[TTS-COMPLETE] WARNING: {len(missing_details)} segments "
                      f"missing audio out of {len(non_empty)} non-empty:", flush=True)
                for detail in missing_details[:20]:  # cap log output
                    print(detail, flush=True)
                if len(missing_details) > 20:
                    print(f"  ... and {len(missing_details) - 20} more", flush=True)
        else:
            print(f"[TTS-COMPLETE] All {produced}/{len(non_empty)} non-empty "
                  f"segments have audio — ready for assembly", flush=True)

        return max(0, missing_count)

    # ── Librosa Deep QC (process-isolated, thread-safe) ─────────────────
    @staticmethod
    def _librosa_deep_qc_worker(args):
        """Run in a SEPARATE PROCESS via ProcessPoolExecutor.

        Librosa is NOT thread-safe, but IS process-safe. Each worker gets its
        own memory space — no shared state, no GIL conflicts, no corruption.

        Analyzes:
        1. Speech rate (tempo) — detect unnaturally fast/slow TTS
        2. Silence ratio — detect failed TTS that's mostly quiet
        3. Clipping — detect distortion
        4. Spectral centroid — detect robotic/metallic TTS artifacts
        """
        import librosa
        import numpy as np

        wav_path, expected_dur, seg_idx = args

        result = {
            "seg_idx": seg_idx,
            "ok": True,
            "issues": [],
            "tempo": 0,
            "silence_ratio": 0,
            "clipping": False,
            "spectral_centroid_mean": 0,
        }

        try:
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            duration = len(y) / sr

            if duration < 0.2:
                result["ok"] = False
                result["issues"].append("too_short")
                return result

            # 1. Silence ratio
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            silent_frames = np.sum(rms < 0.01)
            total_frames = len(rms)
            silence_ratio = silent_frames / max(total_frames, 1)
            result["silence_ratio"] = round(float(silence_ratio), 3)
            if silence_ratio > 0.70:
                result["ok"] = False
                result["issues"].append(f"mostly_silent({silence_ratio:.0%})")

            # 2. Clipping
            peak = float(np.max(np.abs(y)))
            if peak > 0.98:
                result["clipping"] = True
                result["issues"].append("clipping")

            # 3. Tempo (speech rate proxy)
            try:
                tempo = librosa.beat.tempo(y=y, sr=sr)[0]
                result["tempo"] = round(float(tempo), 1)
            except Exception:
                pass

            # 4. Spectral centroid (metallic/robotic voice detection)
            try:
                cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                mean_cent = float(np.mean(cent))
                result["spectral_centroid_mean"] = round(mean_cent, 0)
                # Normal speech: 1000-4000 Hz centroid
                # Robotic/metallic: >5000 Hz or <500 Hz
                if mean_cent > 6000 or mean_cent < 300:
                    result["issues"].append(f"unusual_tone({mean_cent:.0f}Hz)")
            except Exception:
                pass

            # 5. Duration vs expected
            if expected_dur > 0.1:
                ratio = duration / expected_dur
                if ratio > 3.0:
                    result["issues"].append(f"too_long({ratio:.1f}x)")
                elif ratio < 0.2:
                    result["issues"].append(f"too_short({ratio:.1f}x)")

        except Exception as e:
            result["ok"] = False
            result["issues"].append(f"load_error:{e}")

        return result

    def _librosa_deep_qc(self, tts_data):
        """Run librosa analysis on all TTS segments using ProcessPoolExecutor.

        Process isolation: each worker is a separate OS process with its own
        memory — no thread-safety issues with librosa/numba/scipy.

        Returns list of issue dicts for segments that failed QC.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        level = getattr(self.cfg, 'post_tts_level', 'full')
        if level == "none" or getattr(self.cfg, 'audio_untouchable', False):
            return []

        total = len(tts_data)
        if total == 0:
            return []

        self._report("synthesize", 0.96,
                     f"Deep QC: analyzing {total} segments with librosa (process-isolated)...")

        # Build args for worker processes
        args_list = []
        for i, tts in enumerate(tts_data):
            wav = tts.get("wav")
            if not wav or not Path(wav).exists():
                continue
            expected_dur = tts.get("end", 0) - tts.get("start", 0)
            seg_idx = tts.get("_seg_idx", i)
            args_list.append((str(wav), expected_dur, seg_idx))

        if not args_list:
            return []

        # ProcessPoolExecutor — each librosa call runs in its own process.
        # submit + as_completed instead of pool.map so progress streams every
        # ~25 segments (silent-step pattern fix).
        workers = min(4, len(args_list))  # 4 processes max (CPU-bound)
        issues_found = []

        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._librosa_deep_qc_worker, args)
                           for args in args_list]
                results = []
                done = 0
                qc_total = len(futures)
                for fut in as_completed(futures):
                    try:
                        r = fut.result()
                        results.append(r)
                    except Exception:
                        pass
                    done += 1
                    if done % 25 == 0 or done == qc_total:
                        self._report("synthesize", 0.96,
                                     f"[Deep QC] {done}/{qc_total} segments analyzed")

            for r in results:
                if r["issues"]:
                    issues_found.append(r)

            if issues_found:
                print(f"[Deep QC] {len(issues_found)}/{total} segments have issues:",
                      flush=True)
                for r in issues_found[:10]:  # show first 10
                    print(f"  Seg {r['seg_idx']}: {', '.join(r['issues'])}",
                          flush=True)
            else:
                print(f"[Deep QC] All {total} segments passed", flush=True)

        except Exception as e:
            print(f"[Deep QC] Process pool failed: {e} — skipping deep QC", flush=True)

        self._report("synthesize", 0.97,
                     f"Deep QC: {total - len(issues_found)}/{total} passed, "
                     f"{len(issues_found)} flagged")
        return issues_found

    def _qc_check_wav(self, wav_path: Path, expected_duration: float = 0.0) -> dict:
        """QC gate: inspect a TTS WAV file for common failure modes.

        Checks:
        - silence_ratio: fraction of near-silent samples (>0.8 = TTS likely failed)
        - clipping: peak amplitude > 0.98 (distortion)
        - duration_ratio: actual / expected (>2.5 or <0.2 = something went wrong)

        Returns dict with keys: ok, silence_ratio, clipping, duration, duration_ratio, issues
        """
        # Skip QC when post_tts_level is "none"
        level = getattr(self.cfg, 'post_tts_level', 'full')
        if level == "none" or getattr(self.cfg, 'audio_untouchable', False):
            return {"ok": True, "silence_ratio": 0.0, "clipping": False,
                    "duration": 0.0, "duration_ratio": 1.0, "issues": []}

        issues = []
        result = {"ok": True, "silence_ratio": 0.0, "clipping": False,
                  "duration": 0.0, "duration_ratio": 1.0, "issues": issues}
        try:
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.getnframes()
                framerate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                nchannels = wf.getnchannels()
                actual_dur = frames / framerate
                result["duration"] = actual_dur

                raw = wf.readframes(frames)

            if expected_duration > 0.1:
                ratio = actual_dur / expected_duration
                result["duration_ratio"] = ratio
                # Under tts_no_time_pressure, NEVER fail QC because TTS audio
                # is "too long for its slot" — we explicitly want full audio
                # and assembly handles the overflow. We still flag silence
                # (ratio < 0.15 = "only 15% of expected") because that
                # indicates a real failure (TTS produced almost nothing).
                no_pressure = getattr(self.cfg, 'tts_no_time_pressure', True)
                if ratio > 2.5 and not no_pressure:
                    issues.append(f"TTS {ratio:.1f}x longer than slot")
                elif ratio < 0.15:
                    issues.append(f"TTS only {ratio:.1%} of expected length")

            # Sample-level checks (mono mix for speed)
            if sampwidth == 2 and frames > 0:
                import struct
                samples_per_frame = nchannels
                n_samples = len(raw) // (sampwidth * samples_per_frame)
                # Unpack first channel only (every nchannels-th sample)
                fmt = f"<{n_samples * samples_per_frame}h"
                all_samples = struct.unpack_from(fmt, raw[:n_samples * sampwidth * samples_per_frame])
                mono = [all_samples[i * nchannels] for i in range(n_samples)]

                if mono:
                    peak = max(abs(s) for s in mono) / 32767.0
                    result["clipping"] = peak > 0.98
                    # Under no_time_pressure: don't FAIL on clipping.
                    # Edge-TTS synthesized voices routinely peak at 1.0 —
                    # it's digital full-scale, not analog distortion. The
                    # old check caused 4+ rerenders per job on segments
                    # that sounded fine, wasting time and sometimes
                    # producing WORSE audio on the retry.
                    if result["clipping"] and not no_pressure:
                        issues.append(f"Clipping detected (peak={peak:.3f})")

                    silent_count = sum(1 for s in mono if abs(s) < 328)  # < 1% of max
                    silence_ratio = silent_count / len(mono)
                    result["silence_ratio"] = silence_ratio
                    if silence_ratio > 0.80:
                        issues.append(f"Mostly silent ({silence_ratio:.0%})")

        except Exception as e:
            issues.append(f"QC read error: {e}")

        result["ok"] = len(issues) == 0
        if len(issues) == 0:
            result["qc_status"] = "pass"
        elif any("manual_review" in str(i).lower() for i in issues):
            result["qc_status"] = "manual_review"
        elif len(issues) == 1 and "duration" in issues[0].lower():
            result["qc_status"] = "pass_with_warning"
        else:
            result["qc_status"] = "rerender"
        return result

    def _save_manual_review_queue(self, review_items: list):
        """Persist segments that need manual review to a JSON file in the job output.

        Saved to: {work_dir}/../manual_review_queue.json (next to dubbed.mp4)
        Each item: {segment_idx, start, end, source_text, translated_text,
                    emotion, issues, rerender_count}
        """
        if not self.cfg.enable_manual_review or not review_items:
            return
        import json as _json
        out_path = self.cfg.work_dir.parent / "manual_review_queue.json"
        try:
            existing = []
            if out_path.exists():
                existing = _json.loads(out_path.read_text(encoding="utf-8"))
            existing.extend(review_items)
            out_path.write_text(_json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ManualReview] {len(review_items)} segments saved -> {out_path}", flush=True)
        except Exception as e:
            print(f"[ManualReview] Failed to save queue: {e}", flush=True)

    # ── Video muxing ─────────────────────────────────────────────────────
    def _mux_replace_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        """Mux video + dubbed audio. Applies EBU R128 loudness normalisation
        (-14 LUFS, YouTube standard) for professional broadcast-level output.

        NEVER cuts audio — if audio is longer than video, the last video frame
        freezes while remaining audio plays out completely.
        """
        # Check if audio extends beyond video — use -shortest only if they match
        audio_dur = self._get_duration(audio_path)
        video_dur = self._get_duration(video_path)

        cmd = [
            self._ffmpeg, "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", self.cfg.audio_bitrate,
        ]

        if audio_dur > video_dur + 1.0:
            # Audio is longer — DON'T let FFmpeg cut it at video end
            # Loop last video frame to cover remaining audio
            print(f"[Mux] Audio ({audio_dur:.0f}s) > Video ({video_dur:.0f}s) "
                  f"— extending video to match audio", flush=True)
            # Re-encode video with loop to match audio length
            cmd = [
                self._ffmpeg, "-y",
                "-stream_loop", "-1",  # loop video
                "-i", str(video_path),
                "-i", str(audio_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                *self._video_encode_args(),  # NVENC when available
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", self.cfg.audio_bitrate,
                "-t", f"{audio_dur:.3f}",  # output = audio length (NOT video length)
            ]

        cmd.append(str(output_path))
        self._run_proc(cmd, check=True, capture_output=True)


async def list_voices(language_filter: str = "hi"):
    """List available edge-tts voices filtered by language."""
    import edge_tts

    voices = await edge_tts.list_voices()
    if language_filter:
        voices = [v for v in voices if v.get("Locale", "").startswith(language_filter)]
    return voices
