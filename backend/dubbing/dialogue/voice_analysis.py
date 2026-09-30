"""Audio-based voice-category suggestion.

This replaces the single 165 Hz threshold of the legacy pipeline with a
confidence-scored suggestion that keeps an explicit *unknown* state:

  * F0 is aggregated over several clean (non-overlapping) clips of a speaker.
  * Too little voiced audio -> unknown (no guess).
  * Median F0 in the ambiguous band -> unknown.
  * Very high median F0 -> "child_like" suggestion with reduced confidence.

The result describes how a voice *sounds* so a TTS voice of a similar register
can be chosen. It is not a statement about anyone's gender, and it is never
inferred from video appearance.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .contracts import (CATEGORY_CHILD, CATEGORY_FEMALE, CATEGORY_MALE,
                        CATEGORY_UNKNOWN)

# Tuning values (documented, not established quality guarantees).
MIN_VOICED_SECONDS = 1.5
MALE_MAX_F0 = 150.0          # median F0 at or below -> male_like
FEMALE_MIN_F0 = 175.0        # median F0 at or above -> female_like
CHILD_MIN_F0 = 285.0         # median F0 at or above -> child_like
MIN_CONFIDENCE = 0.35
MAX_ANALYSIS_SECONDS = 40.0


def read_wav_mono(path: Path, start: float = 0.0, end: Optional[float] = None):
    """Read a PCM WAV region as float32 mono numpy array. Returns (samples, sr)."""
    import numpy as np
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        s0 = max(0, min(int(start * sr), n))
        s1 = n if end is None else max(s0, min(int(end * sr), n))
        wf.setpos(s0)
        raw = wf.readframes(s1 - s0)
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width {sw}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def estimate_f0_track(samples, sr: int, fmin: float = 60.0, fmax: float = 450.0,
                      frame_s: float = 0.04, hop_s: float = 0.01,
                      min_corr: float = 0.5) -> List[float]:
    """Vectorised normalised-autocorrelation F0 estimator. Returns voiced F0s."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float32)
    flen = int(frame_s * sr)
    hop = max(1, int(hop_s * sr))
    if len(x) < flen * 2:
        return []
    n_frames = 1 + (len(x) - flen) // hop
    idx = np.arange(flen)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = (frames ** 2).mean(axis=1)
    if not np.any(energy > 0):
        return []
    # Keep frames within ~26 dB of the loud frames (speech, not pauses).
    thresh = max(1e-6, 0.0025 * float(np.percentile(energy, 95)))
    keep = energy > thresh
    frames = frames[keep]
    if len(frames) == 0:
        return []
    win = np.hanning(flen).astype(np.float32)
    frames = frames * win
    nfft = 1 << int(np.ceil(np.log2(2 * flen)))
    spec = np.fft.rfft(frames, n=nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :flen]
    ac0 = ac[:, :1] + 1e-12
    ac = ac / ac0
    lag_min = max(1, int(sr / fmax))
    lag_max = min(flen - 1, int(sr / fmin))
    seg = ac[:, lag_min:lag_max]
    best = seg.argmax(axis=1)
    peak = seg[np.arange(len(seg)), best]
    lags = best + lag_min
    voiced = peak >= min_corr
    return [float(sr / l) for l in lags[voiced]]


def classify_f0(f0s: Sequence[float], voiced_seconds: float) -> Tuple[str, Optional[float], Dict]:
    """Map aggregated F0 values to (category, confidence, evidence)."""
    import statistics
    evidence: Dict = {"voiced_seconds": round(voiced_seconds, 2), "n_f0": len(f0s)}
    if voiced_seconds < MIN_VOICED_SECONDS or len(f0s) < 20:
        evidence["reason"] = "insufficient_voiced_audio"
        return CATEGORY_UNKNOWN, None, evidence
    med = statistics.median(f0s)
    evidence["median_f0_hz"] = round(med, 1)
    amount = min(1.0, voiced_seconds / 6.0)
    if med >= CHILD_MIN_F0:
        cat = CATEGORY_CHILD
        margin = (med - CHILD_MIN_F0) / 60.0
        amount *= 0.7   # adult voices can reach this range when excited
    elif med >= FEMALE_MIN_F0:
        cat = CATEGORY_FEMALE
        margin = (med - FEMALE_MIN_F0) / 40.0
    elif med <= MALE_MAX_F0:
        cat = CATEGORY_MALE
        margin = (MALE_MAX_F0 - med) / 40.0
    else:
        evidence["reason"] = "ambiguous_f0_band"
        return CATEGORY_UNKNOWN, None, evidence
    conf = round(max(0.0, min(1.0, 0.4 + 0.6 * min(1.0, margin))) * amount, 3)
    if conf < MIN_CONFIDENCE:
        evidence["reason"] = "low_confidence"
        evidence["suggested"] = cat
        return CATEGORY_UNKNOWN, conf, evidence
    return cat, conf, evidence


def analyze_speaker(audio_path: Path, ranges: Sequence[Tuple[float, float]]
                    ) -> Tuple[str, Optional[float], Dict]:
    """Aggregate F0 over several clean ranges of one speaker."""
    f0s: List[float] = []
    voiced_total = 0.0
    used = 0
    # Prefer longer ranges (cleaner pitch), cap total analysed audio.
    for s, e in sorted(ranges, key=lambda r: r[1] - r[0], reverse=True):
        if e - s < 0.4:
            continue
        if voiced_total >= MAX_ANALYSIS_SECONDS:
            break
        samples, sr = read_wav_mono(audio_path, s, e)
        track = estimate_f0_track(samples, sr)
        f0s.extend(track)
        voiced_total += len(track) * 0.01
        used += 1
    cat, conf, ev = classify_f0(f0s, voiced_total)
    ev["clips_used"] = used
    return cat, conf, ev


def classify_gender_ml(audio_path: Path, ranges_by_speaker: Dict[str, Sequence[Tuple[float, float]]],
                       timeout: float = 900.0) -> Dict[str, float]:
    """p(male) per speaker from the wav2vec2 gender classifier.

    Runs in the same isolated child process as the legacy multi-speaker path
    (pipeline._diarize_child_worker in gender-only mode). Pitch alone is
    ambiguous for many voices (a male Edge voice measured 153 Hz, a female
    180 Hz); the classifier is ~0.999 confident on both, also under music.
    Returns {} if the classifier cannot run (callers fall back to F0).
    """
    import json
    import multiprocessing as mp
    import os
    import tempfile
    import time
    try:
        from pipeline import _diarize_child_worker
    except Exception:
        return {}
    ranges = {k: [list(r) for r in v] for k, v in ranges_by_speaker.items() if v}
    if not ranges:
        return {}
    fd, result_path = tempfile.mkstemp(suffix=".json", prefix="dlg_gender_")
    os.close(fd)
    try:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        p = mp.get_context("spawn").Process(
            target=_diarize_child_worker,
            args=(str(audio_path), "", device, result_path, ranges), daemon=True)
        p.start()
        t0 = time.time()
        while p.is_alive():
            p.join(1.0)
            if time.time() - t0 > timeout:
                p.kill()
                p.join(5)
                return {}
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        if data.get("error") is not None:
            return {}
        return {k: float(v) for k, v in (data.get("p_male") or {}).items()}
    finally:
        try:
            Path(result_path).unlink(missing_ok=True)
        except Exception:
            pass
