"""Verification: cheap per-clip checks, coverage gate, identity, content.

Coverage counts unique required turn IDs with an accepted clip. Extra or
duplicate clips can never compensate for a missing turn.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import audio
from .contracts import Clip, Turn
from .speaker_registry import SpeakerRegistry, VoiceResolutionError
from .text_checks import compare_hindi

MIN_CHARS_PER_S = 3.0     # slower than this -> padded/garbled audio suspected
MAX_CHARS_PER_S = 30.0    # faster than this -> truncated audio suspected
MIN_RMS = 0.003
MAX_CLIP_FRACTION = 0.001


def check_clip(clip: Clip) -> Dict:
    p = Path(clip.path)
    res: Dict = {"ok": True, "problems": []}
    if not p.exists() or p.stat().st_size < 100:
        return {"ok": False, "problems": ["missing_file"]}
    try:
        st = audio.audio_stats(p)
    except Exception as e:
        return {"ok": False, "problems": [f"undecodable: {str(e)[:80]}"]}
    res.update({k: round(v, 5) if isinstance(v, float) else v for k, v in st.items()})
    if st["duration"] < 0.15:
        res["problems"].append("too_short")
    if st["rms"] < MIN_RMS:
        res["problems"].append("near_silent")
    if st["clip_fraction"] > MAX_CLIP_FRACTION:
        res["problems"].append("clipping")
    chars = len("".join(clip.spoken_text.split()))
    dur = max(clip.natural_duration, 1e-3)
    cps = chars / dur
    res["chars_per_s"] = round(cps, 2)
    if chars >= 8 and cps > MAX_CHARS_PER_S:
        res["problems"].append("suspected_truncation")
    if chars >= 8 and cps < MIN_CHARS_PER_S:
        res["problems"].append("suspected_padding_or_garbage")
    res["ok"] = not res["problems"]
    return res


def coverage(turns: Sequence[Turn], clips_by_turn: Dict[str, Clip],
             all_clip_records: Sequence[Clip] = ()) -> Dict:
    required = [t for t in turns if t.required and t.speech_text]
    untranslated = [t for t in turns if t.required and not t.speech_text]
    generated = sorted(tid for tid, c in clips_by_turn.items() if c.accepted)
    req_ids = [t.turn_id for t in turns if t.required]
    missing = []
    for t in turns:
        if not t.required:
            continue
        c = clips_by_turn.get(t.turn_id)
        if c is None or not c.accepted:
            missing.append({"turn_id": t.turn_id, "speaker_id": t.speaker_id,
                            "start": t.source_start, "end": t.source_end,
                            "reason": ("no_translation" if not t.speech_text else
                                       "no_clip" if c is None else
                                       "clip_rejected: " + ",".join(c.verification.get("problems", [])))})
    counts: Dict[str, int] = {}
    for c in all_clip_records:
        if c.accepted:
            counts[c.turn_id] = counts.get(c.turn_id, 0) + 1
    duplicates = [{"turn_id": k, "accepted_clips": v} for k, v in counts.items() if v > 1]
    return {"required_turn_ids": req_ids, "generated_turn_ids": generated,
            "missing": missing, "duplicates": duplicates,
            "required_with_text": len(required), "untranslated": len(untranslated)}


def identity_check(turns: Dict[str, Turn], clips_by_turn: Dict[str, Clip],
                   registry: SpeakerRegistry) -> List[Dict]:
    out = []
    for tid, c in clips_by_turn.items():
        t = turns.get(tid)
        if t is None:
            out.append({"turn_id": tid, "problem": "clip_for_unknown_turn"})
            continue
        if c.speaker_id != t.speaker_id:
            out.append({"turn_id": tid, "problem": "speaker_mismatch",
                        "turn_speaker": t.speaker_id, "clip_speaker": c.speaker_id})
            continue
        try:
            b = registry.resolve_voice(c.speaker_id, c.provider, policy="strict")
        except VoiceResolutionError as e:
            out.append({"turn_id": tid, "problem": "unresolvable", "detail": str(e)})
            continue
        if b["voice"] != c.voice or b.get("pitch") != c.voice_params.get("pitch"):
            out.append({"turn_id": tid, "problem": "voice_not_bound_voice",
                        "expected": b["voice"], "got": c.voice})
    return out


def timing_collisions(clips: Sequence[Clip]) -> List[Dict]:
    out = []
    by_track: Dict[int, List[Clip]] = {}
    for c in clips:
        by_track.setdefault(c.track, []).append(c)
    for tr, lst in by_track.items():
        lst.sort(key=lambda c: c.scheduled_start)
        for a, b in zip(lst, lst[1:]):
            if b.scheduled_start < a.scheduled_end - 1e-3:
                out.append({"track": tr, "a": a.turn_id, "b": b.turn_id,
                            "overlap_s": round(a.scheduled_end - b.scheduled_start, 3)})
    return out


# ── content verification (Hindi re-ASR) ────────────────────────────────────
class WhisperHindiASR:
    """faster-whisper Hindi transcription with bounded, OOM-aware batching."""

    def __init__(self, model: str = "auto"):
        self.model_name = model
        self.model = None
        self.device = "cpu"

    def load(self):
        from faster_whisper import WhisperModel
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
        except Exception:
            pass
        name = self.model_name
        if name == "auto":
            name = "large-v3-turbo" if self.device == "cuda" else "small"
        self.model_name = name
        compute = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(name, device=self.device, compute_type=compute)

    def transcribe(self, path: str) -> str:
        if self.model is None:
            self.load()
        segs, _ = self.model.transcribe(path, language="hi", beam_size=1,
                                        vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs)

    def close(self):
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass


def verify_content(turns: Dict[str, Turn], clips: Dict[str, Clip], asr,
                   resynth: Optional[Callable[[Turn, str], Clip]] = None,
                   refit: Optional[Callable[[Turn, Clip, Clip], None]] = None,
                   max_wer: float = 0.25, max_regens: int = 1,
                   cancel_check: Callable[[], bool] = lambda: False,
                   on_progress: Callable[[float, str], None] = lambda p, m: None) -> List[Dict]:
    """Re-ASR every accepted clip; regenerate (same speaker voice) once on a
    clear mismatch. Persistent mismatches are recorded, not looped on, since
    the verifier itself can be wrong."""
    warnings: List[Dict] = []
    ids = sorted(tid for tid, c in clips.items() if c.accepted)
    for n, tid in enumerate(ids):
        if cancel_check():
            raise RuntimeError("Job cancelled by user")
        t, c = turns[tid], clips[tid]
        for attempt in range(max_regens + 1):
            try:
                heard = asr.transcribe(c.path)
            except Exception as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                    try:
                        asr.close()
                    except Exception:
                        pass
                    asr.model_name = "small"
                    asr.device = "cpu"
                    warnings.append({"type": "verifier_oom_downgrade", "turn_id": tid})
                    heard = asr.transcribe(c.path)
                else:
                    raise
            cmp = compare_hindi(t.speech_text, heard, t.protected_terms)
            c.verification["content"] = {"heard": heard, **cmp, "attempt": attempt + 1}
            substantive = [s for s in cmp["substitutions"] if s["substantive"]]
            bad = cmp["wer"] > max_wer or cmp["critical"] or substantive
            if not bad:
                break
            if attempt < max_regens and resynth is not None:
                try:
                    new = resynth(t, "content_mismatch")
                    if refit:
                        refit(t, c, new)
                    new.retry_history = c.retry_history + new.retry_history
                    clips[tid] = c = new
                    continue
                except Exception:
                    pass
            warnings.append({"type": "content_mismatch", "turn_id": tid,
                             "speaker_id": t.speaker_id, "wer": cmp["wer"],
                             "omissions": cmp["omissions"][:10],
                             "insertions": cmp["insertions"][:10],
                             "repetitions": cmp["repetitions"][:10],
                             "substitutions": substantive[:10],
                             "critical": cmp["critical"],
                             "note": "persisted after regeneration; may be verifier error"})
            break
        on_progress((n + 1) / max(1, len(ids)), f"Verified {n + 1}/{len(ids)} clips")
    return warnings
