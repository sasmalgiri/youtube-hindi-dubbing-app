"""TTS providers and the speaker-preserving router.

Every synthesis goes through ``TTSRouter.synthesize(turn)``, which obtains the
voice from ``SpeakerRegistry.resolve_voice`` -- first pass, retries, rewrites
and fallbacks alike. Retries keep the text and voice unchanged. A provider
fallback uses the *same speaker's* binding on the fallback provider and marks
the clip degraded; ``reroute_mixed_speakers`` then moves the whole speaker to
one provider so a character is not voiced by two different voices.

Paid providers (sarvam, elevenlabs, google) are only constructed when the
owner lists them explicitly; the default provider list is ["edge"].
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import os
import random
import re
import shutil
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import audio
from .contracts import UNKNOWN_SPEAKER, Clip, Turn
from .speaker_registry import SpeakerRegistry, VoiceResolutionError

SENTENCE_SPLIT = re.compile(r"(?<=[।.!?])\s+")

_BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]*>")
_SYMBOLS = re.compile(r"[*#_~^`|♪♫•]+")


def tts_sanitize(text: str) -> str:
    """Remove what a TTS engine would read aloud or choke on.

    LLM output can carry stage directions ("[हँसते हुए]"), markdown, music
    symbols or a leading/trailing ellipsis; Edge reads some of these aloud or
    returns no audio. Only the *spoken* text is cleaned; subtitles keep the
    translation as written. Never returns an empty string for non-empty text.
    """
    if not text:
        return text
    t = re.sub(r"\s*\|\s*", "। ", text)   # LLMs often type "|" for the danda
    t = _BRACKETED.sub(" ", t)
    t = _SYMBOLS.sub(" ", t)
    t = t.replace("\u201c", "").replace("\u201d", "").replace('"', "")
    t = re.sub(r"\s*[—–]\s*", ", ", t)
    t = re.sub(r"\.{3,}|…", "…", t)
    t = re.sub(r"^[\s…,]+|[\s…,]+$", "", t)
    t = t.replace("…", ", ")
    t = re.sub(r"\s*,\s*(,\s*)+", ", ", t)
    t = re.sub(r"\s+([,।.!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if re.search(r"\w", t) else text.strip()


def rate_with_speed(rate: Optional[str], speed: float) -> str:
    """Edge-style prosody rate ("+5%") with an extra speed factor applied."""
    m = re.match(r"^\s*([+-]?\d+)\s*%\s*$", rate or "")
    base = int(m.group(1)) if m else 0
    pct = round(((1 + base / 100.0) * speed - 1) * 100)
    return f"{pct:+d}%"


class TTSProviderError(RuntimeError):
    pass


class TTSFailure(RuntimeError):
    def __init__(self, turn_id: str, history: List[Dict]):
        super().__init__(f"TTS failed for {turn_id}")
        self.turn_id = turn_id
        self.history = history


def split_for_limit(text: str, max_chars: int) -> List[str]:
    """Split at sentence (then clause/space) boundaries. Never drops text."""
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for sent in SENTENCE_SPLIT.split(text):
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = f"{cur} {sent}".strip()
            continue
        if cur:
            parts.append(cur)
        while len(sent) > max_chars:
            cut = max(sent.rfind(",", 0, max_chars), sent.rfind(" ", 0, max_chars))
            cut = cut if cut > 0 else max_chars
            parts.append(sent[:cut + 1].strip())
            sent = sent[cut + 1:].strip()
        cur = sent
    if cur:
        parts.append(cur)
    assert "".join(parts).replace(" ", "") == text.replace(" ", ""), "split lost text"
    return parts


class BaseProvider:
    name = "base"
    max_chars = 3000
    paid = False
    native_rate = False      # accepts binding["rate"] (speaking rate) natively
    retry_backoff_s = 0.0    # base delay before a retry (online, rate-limited services)

    def synthesize_part(self, text: str, binding: Dict, out: Path) -> Path:
        raise NotImplementedError

    def synthesize(self, text: str, binding: Dict, out_wav: Path) -> Path:
        """Synthesize full text to a 48 kHz mono WAV (concatenating parts)."""
        parts = split_for_limit(text, self.max_chars)
        wavs = []
        for i, part in enumerate(parts):
            raw = out_wav.with_name(f"{out_wav.stem}.p{i}.raw")
            got = self.synthesize_part(part, binding, raw)
            w = out_wav.with_name(f"{out_wav.stem}.p{i}.48k.wav")
            audio.to_wav(got, w)
            got.unlink(missing_ok=True)
            wavs.append(w)
        if len(wavs) == 1:
            wavs[0].replace(out_wav)
        else:
            import numpy as np
            chunks = [audio.read_wav(w)[0][:, 0] for w in wavs]
            gap = np.zeros(int(0.12 * audio.SR), dtype=np.float32)
            joined = np.concatenate([x for c in chunks for x in (c, gap)][:-1])
            audio.write_wav(out_wav, joined, audio.SR)
            for w in wavs:
                w.unlink(missing_ok=True)
        return out_wav


class EdgeProvider(BaseProvider):
    name = "edge"
    max_chars = 2000
    native_rate = True
    retry_backoff_s = 2.0    # Edge throttles bursts; immediate retries just fail again
    timeout_s = 45.0         # one stalled websocket must not hang the job

    def synthesize_part(self, text, binding, out):
        try:
            import edge_tts
        except ImportError as e:
            raise TTSProviderError("edge-tts not installed") from e
        mp3 = out.with_suffix(".mp3")
        kwargs = {"rate": binding.get("rate") or "+0%"}
        if binding.get("pitch"):
            kwargs["pitch"] = binding["pitch"]
        try:
            import inspect
            params = inspect.signature(edge_tts.Communicate.__init__).parameters
            if "connect_timeout" in params:
                kwargs.update(connect_timeout=8, receive_timeout=30)
        except (TypeError, ValueError):
            pass

        async def _go():
            await asyncio.wait_for(edge_tts.Communicate(text, binding["voice"], **kwargs).save(str(mp3)),
                                   timeout=self.timeout_s)
        try:
            asyncio.run(_go())
        except Exception as e:
            raise TTSProviderError(f"edge: {type(e).__name__}: {str(e)[:160]}") from e
        if not mp3.exists() or mp3.stat().st_size < 500:
            raise TTSProviderError("edge: empty audio")
        return mp3


class SarvamProvider(BaseProvider):
    name = "sarvam"
    paid = True
    max_chars = 1400  # bulbul:v2 limit is 1500 characters

    def __init__(self):
        self.key = os.environ.get("SARVAM_API_KEY", "").strip()

    def synthesize_part(self, text, binding, out):
        import requests
        if not self.key:
            raise TTSProviderError("SARVAM_API_KEY not set")
        body = {"text": text, "target_language_code": "hi-IN", "speaker": binding["voice"],
                "model": binding.get("model") or "bulbul:v2", "speech_sample_rate": 24000}
        r = requests.post("https://api.sarvam.ai/text-to-speech", json=body, timeout=60,
                          headers={"api-subscription-key": self.key})
        if r.status_code != 200:
            raise TTSProviderError(f"sarvam HTTP {r.status_code}: {r.text[:160]}")
        b64 = (r.json().get("audios") or [None])[0]
        if not b64:
            raise TTSProviderError("sarvam: no audio")
        p = out.with_suffix(".wav")
        p.write_bytes(base64.b64decode(b64))
        return p


class ElevenLabsProvider(BaseProvider):
    name = "elevenlabs"
    paid = True
    max_chars = 5000

    def __init__(self):
        self.key = os.environ.get("ELEVENLABS_API_KEY", "").strip()

    def synthesize_part(self, text, binding, out):
        import requests
        if not self.key:
            raise TTSProviderError("ELEVENLABS_API_KEY not set")
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{binding['voice']}?output_format=mp3_44100_128",
            headers={"xi-api-key": self.key, "Content-Type": "application/json"},
            json={"text": text, "model_id": binding.get("model") or "eleven_multilingual_v2",
                  "language_code": "hi"}, timeout=120)
        if r.status_code != 200:
            raise TTSProviderError(f"elevenlabs HTTP {r.status_code}: {r.text[:160]}")
        p = out.with_suffix(".mp3")
        p.write_bytes(r.content)
        return p


class GoogleProvider(BaseProvider):
    name = "google"
    paid = True
    max_chars = 4500

    def __init__(self):
        self.key = os.environ.get("GOOGLE_TTS_API_KEY", "").strip()

    def synthesize_part(self, text, binding, out):
        import requests
        if not self.key:
            raise TTSProviderError("GOOGLE_TTS_API_KEY not set")
        pitch = 0.0
        if binding.get("pitch"):
            try:  # "+8Hz" -> ~ semitones (rough, only to separate shared voices)
                pitch = max(-20.0, min(20.0, float(binding["pitch"].rstrip("Hz")) / 6.0))
            except ValueError:
                pitch = 0.0
        r = requests.post(
            f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.key}",
            json={"input": {"text": text},
                  "voice": {"languageCode": "hi-IN", "name": binding["voice"]},
                  "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000,
                                  "pitch": pitch}}, timeout=60)
        if r.status_code != 200:
            raise TTSProviderError(f"google HTTP {r.status_code}: {r.text[:160]}")
        p = out.with_suffix(".wav")
        p.write_bytes(base64.b64decode(r.json()["audioContent"]))
        return p


class MockProvider(BaseProvider):
    """Deterministic offline provider for tests and dry runs (tones, not speech)."""
    name = "mock"

    def __init__(self, seconds_per_char: float = 0.06,
                 fail: Optional[Callable[[str, Dict], bool]] = None, name: str = "mock",
                 native_rate: bool = False):
        self.seconds_per_char = seconds_per_char
        self.fail = fail
        self.name = name
        self.native_rate = native_rate
        self.calls: List[Dict] = []
        self._lock = threading.Lock()

    def synthesize_part(self, text, binding, out):
        with self._lock:
            self.calls.append({"text": text, "voice": binding["voice"], "pitch": binding.get("pitch")})
        if self.fail and self.fail(text, binding):
            raise TTSProviderError("mock failure")
        sr = 24000
        dur = max(0.3, len(text) * self.seconds_per_char)
        m = re.match(r"^([+-]?\d+)%$", binding.get("rate") or "")
        if self.native_rate and m:
            dur /= 1 + int(m.group(1)) / 100.0
        f0 = 110.0 + (sum(map(ord, binding["voice"])) % 7) * 25.0
        data = array("h", (int(8000 * math.sin(2 * math.pi * f0 * n / sr)
                               * (0.6 + 0.4 * math.sin(2 * math.pi * 3 * n / sr)))
                           for n in range(int(dur * sr))))
        p = out.with_suffix(".wav")
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(data.tobytes())
        return p


class IndicF5Provider(BaseProvider):
    """EXPERIMENTAL local IndicF5 (ai4bharat/IndicF5, gated, MIT) synthesis.

    Each binding's "voice" is a curated reference id resolved to
    backend/voices/indicf5/<category>/<id>.wav + <id>.txt (exact transcript of
    the reference, in Hindi). Only references you are authorised to use may be
    placed there. Original-speaker cloning is NOT enabled: a clean Hindi
    reference with an exact transcript is required, and arbitrary English
    clips are not assumed to clone reliably into Hindi.
    Not runtime-verified in this repository's CI (needs GPU + gated weights).
    """
    name = "indicf5"
    max_chars = 600
    SR = 24000

    def __init__(self):
        self.model = None
        self._lock = threading.Lock()

    def _load(self):
        if self.model is None:
            try:
                from transformers import AutoModel
            except ImportError as e:
                raise TTSProviderError("transformers not installed (IndicF5 is optional)") from e
            self.model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

    def synthesize_part(self, text, binding, out):
        ref_wav = Path(binding["ref_audio"])
        ref_txt = Path(binding["ref_text"]).read_text(encoding="utf-8").strip()
        with self._lock:  # one GPU generation at a time
            self._load()
            try:
                wav = self.model(text, ref_audio_path=str(ref_wav), ref_text=ref_txt)
            except Exception as e:
                raise TTSProviderError(f"indicf5: {str(e)[:160]}") from e
        import numpy as np
        arr = np.asarray(wav)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        p = out.with_suffix(".wav")
        audio.write_wav(p, arr.reshape(-1), self.SR)
        return p


class IndicParlerProvider(BaseProvider):
    """AI4Bharat Indic Parler-TTS (Apache-2.0, gated: HF_TOKEN), free and local.

    Runs in a persistent worker (workers/parler_worker.py), optionally under
    its own Python (INDIC_PARLER_PYTHON) because parler-tts pins
    transformers==4.46.1. The speaker is chosen by naming it in a voice
    description; the binding's "pitch" field carries a style word
    (low / high) that changes the description. The model card advises
    ~10-12 s per generation, so text is sent in sentence-sized chunks.
    """
    name = "indic_parler"
    max_chars = 160
    STYLE = {None: "a natural, clear tone", "low": "a deep, low-pitched tone",
             "high": "a slightly high-pitched tone"}

    def __init__(self):
        from .local_workers import PersistentWorker
        self.worker = PersistentWorker("parler", init={"model": "ai4bharat/indic-parler-tts"},
                                       timeout=900)

    def description(self, binding: Dict) -> str:
        style = self.STYLE.get(binding.get("pitch"), self.STYLE[None])
        return (f"{binding['voice']}'s voice is expressive and conversational, with {style}, "
                f"speaking at a moderate pace in a close recording with very clear audio.")

    def synthesize_part(self, text, binding, out):
        p = out.with_suffix(".wav")
        seed = sum(map(ord, binding["voice"] + str(binding.get("pitch")))) % 100000
        try:
            self.worker.request({"op": "tts", "text": text, "description": self.description(binding),
                                 "out": str(p), "seed": seed})
        except Exception as e:
            raise TTSProviderError(f"indic_parler: {str(e)[:200]}") from e
        if not p.exists() or p.stat().st_size < 500:
            raise TTSProviderError("indic_parler: empty audio")
        return p

    def close(self):
        self.worker.close()


PROVIDER_CLASSES = {"edge": EdgeProvider, "indic_parler": IndicParlerProvider, "sarvam": SarvamProvider,
                    "elevenlabs": ElevenLabsProvider, "google": GoogleProvider,
                    "indicf5": IndicF5Provider, "mock": MockProvider}


def build_providers(names: Sequence[str]) -> Dict[str, BaseProvider]:
    out = {}
    for n in names:
        if n not in PROVIDER_CLASSES:
            raise ValueError(f"Unknown TTS provider '{n}'")
        out[n] = PROVIDER_CLASSES[n]()
    return out


def tts_cache_key(provider: str, voice: str, pitch: Optional[str], rate: Optional[str],
                  speed: float, spoken_text: str) -> str:
    """Within-job TTS cache key: the same line in the same voice is the same audio."""
    raw = "|".join([provider, voice or "", pitch or "", rate or "", f"{float(speed):.4f}", spoken_text])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TTSRouter:
    """``cache_dir`` (opt-in) keeps every synthesized clip under its
    tts_cache_key, so a resumed run of the SAME job reuses unchanged lines
    instead of calling the provider again. The orchestrator points it into
    the job's own work_dir: nothing is shared across jobs."""

    def __init__(self, providers: Dict[str, BaseProvider], registry: SpeakerRegistry,
                 order: Sequence[str], clip_dir: Path, max_retries: int = 2,
                 pronunciation: Optional[Dict[str, str]] = None,
                 cache_dir: Optional[Path] = None):
        if not order:
            raise ValueError("at least one TTS provider is required")
        self.providers = providers
        self.registry = registry
        self.order = list(order)
        self.clip_dir = Path(clip_dir)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.pronunciation = pronunciation or {}
        self.speaker_provider: Dict[str, str] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hits = 0
        self._cache_keys: Dict[str, str] = {}     # clip_id -> cache key
        self._counter = 0
        self._lock = threading.Lock()

    # within-job cache
    def _cache_get(self, key: str, dst: Path) -> Optional[float]:
        try:
            shutil.copyfile(self.cache_dir / f"{key}.wav", dst)
            dur = audio.probe_duration(dst)
        except Exception:
            return None
        with self._lock:
            self.cache_hits += 1
        return dur

    def _cache_put(self, key: str, src: Path):
        tmp = self.cache_dir / f".{key}.{threading.get_ident()}.tmp"
        try:
            shutil.copyfile(src, tmp)
            os.replace(tmp, self.cache_dir / f"{key}.wav")
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def uncache(self, clip: Clip):
        """Forget a clip's cached audio (it failed the checks): a resumed run
        must synthesize that line again, not reuse the rejected audio."""
        key = self._cache_keys.get(clip.clip_id)
        if key and self.cache_dir is not None:
            try:
                (self.cache_dir / f"{key}.wav").unlink(missing_ok=True)
            except OSError:
                pass

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"c{self._counter:05d}"

    def _spoken_text(self, text: str) -> str:
        for k in sorted(self.pronunciation, key=len, reverse=True):
            if k in text:
                text = text.replace(k, self.pronunciation[k])
        return tts_sanitize(text)

    def supports_native_rate(self, speaker_id: str) -> bool:
        prov = self.providers.get(self.providers_for(speaker_id)[0])
        return bool(prov is not None and prov.native_rate)

    def providers_for(self, speaker_id: str) -> List[str]:
        first = self.speaker_provider.get(speaker_id, self.order[0])
        return [first] + [p for p in self.order if p != first]

    def synthesize(self, turn: Turn, reason: str = "initial",
                   only_provider: Optional[str] = None, speed: float = 1.0,
                   use_cache: bool = True) -> Clip:
        """``speed`` > 1 asks a native-rate provider to speak faster (same
        voice, same pitch); providers without native rate ignore it.
        ``use_cache=False`` (a regeneration) always calls the provider; the
        new audio still replaces the cached one."""
        text = turn.speech_text
        if not text:
            raise TTSFailure(turn.turn_id, [{"error": "empty text"}])
        spoken = self._spoken_text(text)
        history: List[Dict] = []
        policy = "register_unknown" if turn.speaker_id == UNKNOWN_SPEAKER else "strict"
        chain = [only_provider] if only_provider else self.providers_for(turn.speaker_id)
        primary = chain[0]
        for prov_name in chain:
            prov = self.providers.get(prov_name)
            if prov is None:
                continue
            try:
                binding = self.registry.resolve_voice(turn.speaker_id, prov_name, policy)
            except VoiceResolutionError as e:
                history.append({"provider": prov_name, "error": str(e)})
                continue
            if abs(speed - 1.0) > 1e-3 and prov.native_rate:
                binding = dict(binding, rate=rate_with_speed(binding.get("rate"), speed))
            key = None
            if self.cache_dir is not None:
                key = tts_cache_key(prov_name, binding["voice"], binding.get("pitch"),
                                    binding.get("rate"), speed, spoken)
                if use_cache and (self.cache_dir / f"{key}.wav").exists():
                    clip_id = self._next_id()
                    final = self.clip_dir / f"{turn.turn_id}_{clip_id}.wav"
                    dur = self._cache_get(key, final)
                    if dur is not None:
                        self._cache_keys[clip_id] = key
                        return self._clip(turn, clip_id, prov_name, primary, binding, spoken, final,
                                          dur, history + [{"reason": reason, "cache": "hit"}])
            for attempt in range(self.max_retries + 1):
                if attempt and prov.retry_backoff_s > 0:
                    time.sleep(min(10.0, prov.retry_backoff_s * 2 ** (attempt - 1))
                               + random.uniform(0, 0.5))
                clip_id = self._next_id()
                raw = self.clip_dir / f"{turn.turn_id}_{clip_id}_raw.wav"
                final = self.clip_dir / f"{turn.turn_id}_{clip_id}.wav"
                try:
                    prov.synthesize(spoken, binding, raw)
                    dur = audio.trim_silence(raw, final)
                    raw.unlink(missing_ok=True)
                except Exception as e:
                    history.append({"provider": prov_name, "voice": binding["voice"],
                                    "attempt": attempt + 1, "error": str(e)[:200]})
                    continue
                if key is not None:
                    self._cache_put(key, final)
                    self._cache_keys[clip_id] = key
                return self._clip(turn, clip_id, prov_name, primary, binding, spoken, final, dur,
                                  history + [{"reason": reason}])
        raise TTSFailure(turn.turn_id, history)

    def _clip(self, turn: Turn, clip_id: str, prov_name: str, primary: str, binding: Dict,
              spoken: str, final: Path, dur: float, history: List[Dict]) -> Clip:
        # Anything not voiced by the configured primary provider is degraded.
        degraded = prov_name != self.order[0]
        if prov_name != primary:
            errors = [h["error"] for h in history if h.get("error")]
            self.registry.record_fallback(turn.speaker_id, primary, prov_name,
                                          turn.turn_id, errors[-1] if errors else "")
        return Clip(clip_id=clip_id, turn_id=turn.turn_id, speaker_id=turn.speaker_id,
                    provider=prov_name, voice=binding["voice"], model=binding.get("model", ""),
                    voice_params={"pitch": binding.get("pitch"), "variant": binding.get("variant"),
                                  "rate": binding.get("rate")},
                    spoken_text=spoken, path=str(final), natural_duration=dur,
                    final_duration=dur, degraded=degraded, retry_history=history)

    def reroute_mixed_speakers(self, turns: Dict[str, Turn], clips: Dict[str, Clip],
                               synth: Optional[Callable[[Turn, str, str], Clip]] = None) -> List[Dict]:
        """If a speaker's clips came from >1 provider, re-voice the whole
        speaker with the fallback provider. Returns unresolved mixes.

        A speaker voiced *entirely* by a fallback provider (every call to the
        primary failed, e.g. each Indic Parler line fell back to Edge) is
        consistent already, but it is still pinned to that provider: later
        regenerations (fit, verification) go to providers_for(speaker)[0]
        and must use the provider that actually works, not retry the broken
        one. The orchestrator reports every pinned speaker (speaker_provider)
        as voiced by a fallback provider.

        `synth(turn, reason, provider)` should run the same acceptance checks
        as the first pass; without it clips are accepted unchecked."""
        by_spk: Dict[str, Dict[str, List[str]]] = {}
        for tid, c in clips.items():
            by_spk.setdefault(c.speaker_id, {}).setdefault(c.provider, []).append(tid)
        unresolved = []
        for spk, provs in by_spk.items():
            if len(provs) < 2:
                only = next(iter(provs))
                if only != self.order[0]:
                    self.speaker_provider[spk] = only
                continue
            target = max((p for p in provs if p != self.order[0]),
                         key=lambda p: len(provs[p]), default=None)
            if target is None:
                continue
            self.speaker_provider[spk] = target
            failed = []
            for prov, tids in provs.items():
                if prov == target:
                    continue
                for tid in tids:
                    try:
                        if synth is not None:
                            new = synth(turns[tid], "speaker_reroute", target)
                        else:
                            new = self.synthesize(turns[tid], reason="speaker_reroute", only_provider=target)
                            new.accepted = True
                    except TTSFailure:
                        failed.append(tid)
                        continue
                    if not new.accepted:
                        failed.append(tid)
                        continue
                    new.degraded = True
                    clips[tid] = new
            if failed:
                unresolved.append({"speaker_id": spk, "providers": sorted(provs), "turn_ids": failed})
        return unresolved
