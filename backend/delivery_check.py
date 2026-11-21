"""Delivery check for a finished dub (tracker Q05, P07).

Runs before a job counts as Complete or is charged. Two kinds of findings:

  problems  block delivery: the time is returned and the job ends as an error
            (no decode, no audio, too short, silent, shorter than paid for,
            subtitles unreadable or out of bounds)
  warnings  delivered, but shown on the job page so the creator can look
            (clipping, long silent gaps inside speech, repeated subtitle
            lines, subtitles that are not Hindi script, overlapping cues,
            optional spoken-language check)

Plus metrics (duration, peak, loudness, gaps, cue count) and a sha256 of the
file, recorded with the job. Everything is ffmpeg/ffprobe; the optional
spoken-language check (VOICEDUB_VERIFY_ASR=1) transcribes three short
samples with local faster-whisper and never calls a paid service.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

SILENCE_DB = -60.0          # max volume below this = silent output
CLIP_PEAK_DB = -0.1         # true peak at/above this with many clipped samples = clipping
CLIP_MIN_SAMPLES = 50
GAP_DB = -45.0              # quieter than this counts as a gap
GAP_SECONDS = 8.0           # a gap this long while subtitles say someone speaks = missing line
SHORT_FRACTION = 0.8        # output shorter than this share of the paid seconds = problem
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
TS = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def _run(args: List[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── subtitles ──
def _ts(s: str) -> float:
    m = TS.search(s)
    if not m:
        raise ValueError(f"bad timestamp {s!r}")
    h, mi, se, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000


def parse_srt(raw: bytes) -> List[Tuple[float, float, str]]:
    """Strict-enough SRT reader. Raises ValueError with a short reason."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("subtitles are not UTF-8")
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        i = 1 if "-->" not in lines[0] else 0
        if i >= len(lines) or "-->" not in lines[i]:
            raise ValueError(f"cue without a time line: {lines[0][:40]!r}")
        a, b = lines[i].split("-->", 1)
        cues.append((_ts(a), _ts(b), " ".join(lines[i + 1:]).strip()))
    if not cues:
        raise ValueError("subtitles are empty")
    return cues


def check_srt(raw: bytes, duration: float, target_lang: str = "hi") -> Tuple[List[str], List[str], Dict]:
    problems, warnings = [], []
    try:
        cues = parse_srt(raw)
    except ValueError as e:
        return [str(e)], [], {}
    bad_order = sum(1 for s, e, _ in cues if e <= s)
    if bad_order:
        problems.append(f"{bad_order} subtitle cue(s) end before they start")
    past = [s for s, _, _ in cues if s > duration + 1.0]
    if past:
        problems.append(f"{len(past)} subtitle cue(s) start after the audio ends")
    overl = sum(1 for (s1, e1, _), (s2, _, _) in zip(cues, cues[1:]) if s2 < e1 - 0.05)
    if overl:
        warnings.append(f"{overl} subtitle cue(s) overlap the next one")
    empty = sum(1 for *_, t in cues if not t)
    if empty:
        warnings.append(f"{empty} empty subtitle cue(s)")
    reps = sum(1 for (_, _, a), (_, _, b) in zip(cues, cues[1:]) if a and a == b and len(a) > 3)
    if reps:
        warnings.append(f"{reps} subtitle line(s) repeated back to back")
    if target_lang == "hi":
        text = " ".join(t for *_, t in cues)
        dev, lat = len(DEVANAGARI.findall(text)), len(LATIN.findall(text))
        if dev + lat and dev / (dev + lat) < 0.6:
            warnings.append(f"subtitles are mostly not in Hindi script ({dev * 100 // (dev + lat)}% Devanagari)")
    return problems, warnings, {"cues": len(cues), "speech_spans": [(s, e) for s, e, t in cues if t]}


# ── audio ──
def _probe(path: Path) -> Dict:
    pr = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], 60)
    if pr.returncode != 0:
        raise ValueError("file does not decode")
    return json.loads(pr.stdout or "{}")


def _audio_stats(path: Path) -> Dict:
    """One decode pass: decode errors, volumedetect, astats peak/clipping, silences."""
    r = _run(["ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-i", str(path), "-map", "0:a:0",
              "-af", f"volumedetect,astats=metadata=0:reset=0,silencedetect=noise={GAP_DB}dB:d={GAP_SECONDS}",
              "-f", "null", "-"])
    err = r.stderr or ""
    out: Dict = {"decode_errors": len(re.findall(r"(?i)error while decoding|invalid data found", err))}
    m = re.search(r"max_volume:\s*(-?[\d.]+|-inf) dB", err)
    out["max_volume_db"] = None if not m else (-999.0 if m.group(1) == "-inf" else float(m.group(1)))
    m = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", err)
    out["mean_volume_db"] = None if not m or m.group(1) == "-inf" else float(m.group(1))
    overall = err.split("Overall", 1)[-1] if "Overall" in err else err
    m = re.search(r"Peak level dB:\s*(-?[\d.]+|-inf)", overall)
    out["peak_db"] = None if not m or m.group(1) == "-inf" else float(m.group(1))
    m = re.search(r"Peak count:\s*(\d+)", overall)
    out["peak_count"] = int(m.group(1)) if m else 0
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", err)]
    out["gaps"] = [(max(0.0, s), e) for s, e in zip(starts, ends)]
    return out


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def spoken_language_check(path: Path, duration: float, asr_factory: Optional[Callable] = None) -> Optional[str]:
    """Optional: transcribe three 8 s samples; warn when little Hindi script comes back."""
    if asr_factory is None:
        if os.environ.get("VOICEDUB_VERIFY_ASR", "").strip() not in ("1", "true", "yes"):
            return None
        from dubbing.dialogue.verify import WhisperHindiASR
        asr_factory = WhisperHindiASR
    import tempfile
    asr, texts = asr_factory(), []
    try:
        with tempfile.TemporaryDirectory() as td:
            for i, at in enumerate((duration * 0.2, duration * 0.5, duration * 0.8)):
                wav = Path(td) / f"s{i}.wav"
                _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-t", "8", "-i", str(path),
                      "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(wav)], 120)
                if wav.exists():
                    texts.append(asr.transcribe(str(wav)))
    finally:
        getattr(asr, "close", lambda: None)()
    text = " ".join(texts)
    dev, lat = len(DEVANAGARI.findall(text)), len(LATIN.findall(text))
    if dev + lat == 0:
        return "no intelligible speech found in three samples"
    if dev / (dev + lat) < 0.5:
        return f"spoken audio does not sound like Hindi ({dev * 100 // (dev + lat)}% Devanagari in samples)"
    return None


def check(path: Path, *, srt: Optional[Path] = None, paid_seconds: Optional[float] = None,
          target_lang: str = "hi", min_seconds: float = 1.0, asr_factory: Optional[Callable] = None) -> Dict:
    """Returns {"ok", "problems", "warnings", "metrics"}; never raises."""
    problems: List[str] = []
    warnings: List[str] = []
    metrics: Dict = {}
    try:
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return {"ok": False, "problems": ["empty file"], "warnings": [], "metrics": {}}
        try:
            info = _probe(path)
        except ValueError as e:
            return {"ok": False, "problems": [str(e)], "warnings": [], "metrics": {}}
        if not any(st.get("codec_type") == "audio" for st in info.get("streams", [])):
            return {"ok": False, "problems": ["no audio track"], "warnings": [], "metrics": {}}
        dur = float(info.get("format", {}).get("duration") or 0)
        metrics["duration"] = round(dur, 2)
        metrics["sha256"] = sha256(path)
        if dur < min_seconds:
            problems.append(f"too short ({dur:.1f}s)")
        if paid_seconds and dur < paid_seconds * SHORT_FRACTION:
            problems.append(f"output is {dur:.0f}s but {paid_seconds:.0f}s were paid for")
        st = _audio_stats(path)
        metrics.update({k: st[k] for k in ("max_volume_db", "mean_volume_db", "peak_db", "peak_count", "decode_errors")})
        if st["max_volume_db"] is None:
            problems.append("audio could not be measured")
        elif st["max_volume_db"] < SILENCE_DB:
            problems.append("audio is silent")
        if st["decode_errors"]:
            problems.append(f"{st['decode_errors']} decode error(s) in the audio")
        if st["peak_db"] is not None and st["peak_db"] >= CLIP_PEAK_DB and st["peak_count"] >= CLIP_MIN_SAMPLES:
            warnings.append(f"audio clips ({st['peak_count']} samples at full scale)")
        gaps = st["gaps"]
        metrics["long_gaps"] = len(gaps)
        if srt is not None and Path(srt).exists():
            p, w, m = check_srt(Path(srt).read_bytes(), dur, target_lang)
            problems += p
            warnings += w
            spans = m.pop("speech_spans", [])
            metrics.update(m)
            silent_lines = [g for g in gaps if any(_overlap(g, s) > GAP_SECONDS * 0.75 for s in spans)]
            if silent_lines:
                warnings.append(f"{len(silent_lines)} long silence(s) where the subtitles say someone speaks "
                                f"(first at {silent_lines[0][0]:.0f}s) — a line may be missing")
        if not problems:
            lang = spoken_language_check(path, dur, asr_factory)
            if lang:
                warnings.append(lang)
    except subprocess.TimeoutExpired:
        problems.append("check timed out")
    except Exception as e:   # the check itself must never crash a job
        problems.append(f"check failed ({type(e).__name__})")
    return {"ok": not problems, "problems": problems, "warnings": warnings, "metrics": metrics}
