"""Word-level timing refinement for dubbing segments.

A single, tiered strategy for getting the tightest word-level timestamps we can,
degrading gracefully instead of silently dropping to coarse timing:

  Tier 1 — WhisperX forced alignment (wav2vec2). Most precise word boundaries.
  Tier 2 — Local faster-whisper with native ``word_timestamps`` on the GPU.
           Used when WhisperX is unavailable OR produces an *unhealthy*
           alignment (the "when whisperx is not proper" case). Looser than
           Tier 1 (DTW on attention vs. phoneme alignment) but far better than
           whole-segment timing.
  Tier 3 — The original segment-level timing. Always returned if 1 and 2 fail.

The heavy local-Whisper pass is INJECTED as a callable (``local_whisper``) so
this module never imports the multiprocessing worker or any pipeline internals —
it stays small and unit-testable. pipeline.py binds that callable to
faster-whisper large-v3 (see PipelineConfig.whisper_fallback_model).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional


def _has_word_timing(segments: List[Dict], min_coverage: float = 0.5) -> bool:
    """True when a reasonable fraction of segments carry word-level entries."""
    if not segments:
        return False
    with_words = sum(1 for s in segments if s.get("words"))
    return (with_words / len(segments)) >= min_coverage


def alignment_healthy(aligned: List[Dict], original: List[Dict]) -> bool:
    """Heuristic check: did WhisperX actually produce usable word timing?

    Rejects the degenerate output forced alignment can emit — no words, a
    collapsed segment count, or word spans that are out of segment bounds,
    non-monotonic, or zero/negative length. This is what "whisperx is not
    proper" means concretely, so we can fall back on it deterministically.
    """
    if not aligned:
        return False
    # Most segments should have words.
    if not _has_word_timing(aligned, min_coverage=0.6):
        return False
    # Segment count shouldn't collapse vs the source transcript.
    if original and len(aligned) < 0.5 * len(original):
        return False

    good_words = 0
    total_words = 0
    for seg in aligned:
        s0 = float(seg.get("start", 0.0) or 0.0)
        s1 = float(seg.get("end", 0.0) or 0.0)
        prev_end = s0 - 0.05
        for w in (seg.get("words") or []):
            total_words += 1
            ws, we = w.get("start"), w.get("end")
            if ws is None or we is None:
                continue
            ws, we = float(ws), float(we)
            # positive span, inside segment bounds (tolerant), monotonic
            if (we >= ws
                    and ws >= s0 - 0.15
                    and we <= s1 + 0.15
                    and ws >= prev_end - 0.20):
                good_words += 1
                prev_end = we
    if total_words == 0:
        return False
    return (good_words / total_words) >= 0.8


def align_whisperx(wav_path, segments: List[Dict], language: str, device: str,
                   report: Optional[Callable] = None) -> Optional[List[Dict]]:
    """Run WhisperX forced alignment. Returns normalised aligned segments, or
    ``None`` on any failure (not installed, model load, align error, empty)."""
    try:
        import whisperx  # noqa: F401
    except Exception:
        if report:
            report("transcribe", 0.97,
                   "WhisperX not available — will use local Whisper for word timing")
        return None
    try:
        import whisperx
        lang = language if language and language != "auto" else "en"
        if report:
            report("transcribe", 0.97, "Running WhisperX forced alignment...")
        model_a, meta = whisperx.load_align_model(language_code=lang, device=device)
        result = whisperx.align(segments, model_a, meta, str(wav_path), device,
                                return_char_alignments=False)
        refined = result.get("segments") if isinstance(result, dict) else None
        if not refined:
            return None
        out: List[Dict] = []
        for seg in refined:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", 0.0) or 0.0)
            entry = {**seg, "start": start, "end": end,
                     "text": (seg.get("text", "") or "").strip()}
            words = seg.get("words")
            if words:
                fixed = []
                for w in words:
                    ws = w.get("start")
                    we = w.get("end")
                    fixed.append({
                        "word": (w.get("word", "") or "").strip(),
                        "start": float(ws) if ws is not None else start,
                        "end": float(we) if we is not None else end,
                    })
                entry["words"] = fixed
            out.append(entry)
        return out
    except Exception as e:
        if report:
            report("transcribe", 0.97,
                   f"WhisperX alignment failed ({str(e)[:60]}) — falling back")
        return None


def refine(wav_path, segments: List[Dict], *,
           language: str = "en", device: str = "cpu",
           use_whisperx: bool = True, gpu_fallback: bool = True,
           local_whisper: Optional[Callable[[], List[Dict]]] = None,
           report: Optional[Callable] = None) -> List[Dict]:
    """Return the best-available word-timed segments (Tier 1 → 2 → 3).

    ``local_whisper`` is a no-arg callable that re-transcribes the audio with a
    local faster-whisper model (already ``word_timestamps=True``); the caller
    binds the model choice. It is only invoked when Tiers 1 fails AND we don't
    already have word-level timing.
    """
    if not segments:
        return segments

    # Tier 1 — WhisperX forced alignment.
    if use_whisperx:
        aligned = align_whisperx(wav_path, segments, language, device, report)
        if aligned and alignment_healthy(aligned, segments):
            if report:
                report("transcribe", 0.99,
                       f"WhisperX alignment OK ({len(aligned)} segments)")
            return aligned
        if report:
            report("transcribe", 0.97, "WhisperX alignment not usable")

    # Already word-level (e.g. local transcription) — keep it, don't re-run.
    if _has_word_timing(segments, min_coverage=0.6):
        return segments

    # Tier 2 — local Whisper on the GPU for native word timestamps.
    if gpu_fallback and local_whisper is not None:
        try:
            if report:
                report("transcribe", 0.90,
                       "Re-transcribing locally on GPU for word-level timing...")
            local_segs = local_whisper()
            if local_segs and _has_word_timing(local_segs, min_coverage=0.5):
                if report:
                    report("transcribe", 0.99,
                           f"Local word-level timing OK ({len(local_segs)} segments)")
                return local_segs
        except Exception as e:
            if report:
                report("transcribe", 0.90,
                       f"Local word-timing fallback failed ({str(e)[:60]})")

    # Tier 3 — original segment-level timing.
    return segments
