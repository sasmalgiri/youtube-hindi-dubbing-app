"""Timing fit for the dialogue profile.

The video keeps its original speed. For each turn (in time order):
  1. synthesize at natural rate, measure;
  2. use genuinely free time after the turn (up to the next required,
     non-overlapping turn) as slack;
  3. if still too long, ask for a shorter *faithful* rewrite and regenerate
     (bounded);
  4. then apply a modest pitch-preserving stretch (<= max_stretch);
  5. if it still does not fit, keep the full audio (never truncated), allow a
     small pre-roll into preceding silence, and record the exact overflow.

Source times on turns are never modified; results go to Clip.scheduled_*.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import audio
from .contracts import Clip, Turn


@dataclass
class FitConfig:
    max_stretch: float = 1.15       # tuning value; confirm by listening tests
    guard_s: float = 0.05           # silence kept before the next turn
    preroll_s: float = 0.25         # may start this much earlier into silence
    max_rewrites: int = 2
    overflow_tolerance_s: float = 0.15
    draft_overflow_s: float = 0.6   # unresolved overflow above this -> draft


def compute_windows(turns: Sequence[Turn], media_duration: float, guard_s: float = 0.05
                    ) -> Dict[str, Tuple[float, float]]:
    """turn_id -> (start, available_end) from immutable source timing.

    The window ends at the next later-starting required turn that is not a
    deliberate overlap partner, or at the end of the media.
    """
    req = sorted([t for t in turns if t.required], key=lambda t: (t.source_start, t.source_end))
    out: Dict[str, Tuple[float, float]] = {}
    for i, t in enumerate(req):
        nxt = media_duration
        for u in req[i + 1:]:
            if u.turn_id in t.overlaps_with:
                continue
            if u.source_start >= t.source_start:
                nxt = min(nxt, u.source_start)
                break
        end = max(t.source_end, nxt - guard_s)
        out[t.turn_id] = (t.source_start, min(end, media_duration) if media_duration > 0 else end)
    return out


def plan_speed(natural: float, slack: float, max_stretch: float) -> Tuple[float, float]:
    """Return (speed, overflow_seconds) for a clip in a slot."""
    if slack <= 0:
        return max_stretch, natural / max_stretch
    if natural <= slack:
        return 1.0, 0.0
    need = natural / slack
    if need <= max_stretch:
        return need, 0.0
    return max_stretch, natural / max_stretch - slack


def fit_all(turns: Sequence[Turn], clips: Dict[str, Clip], media_duration: float,
            resynth: Callable[[Turn, str], Clip],
            rewrite: Optional[Callable[[Turn, str, float], Optional[str]]],
            cfg: FitConfig, cancel_check: Callable[[], bool] = lambda: False,
            on_progress: Callable[[float, str], None] = lambda p, m: None) -> List[Dict]:
    """Fit every clip in place. Returns timing deviation records."""
    windows = compute_windows(turns, media_duration, cfg.guard_s)
    deviations: List[Dict] = []
    ordered = sorted([t for t in turns if t.turn_id in clips], key=lambda t: t.source_start)
    last_end = 0.0
    for n, t in enumerate(ordered):
        if cancel_check():
            raise RuntimeError("Job cancelled by user")
        clip = clips[t.turn_id]
        start, avail_end = windows.get(t.turn_id, (t.source_start, t.source_end))
        slack = avail_end - start
        t.budget_s = round(slack, 3)
        speed, overflow = plan_speed(clip.natural_duration, slack, cfg.max_stretch)
        rewrites = 0
        while overflow > cfg.overflow_tolerance_s and rewrite and rewrites < cfg.max_rewrites:
            rewrites += 1
            ratio = max(0.4, (slack * cfg.max_stretch) / clip.natural_duration * 0.95)
            new_text = rewrite(t, t.speech_text, ratio)
            if not new_text:
                break
            old_fit, old_display = t.hi_fit, t.hi_display
            t.hi_fit = new_text
            try:
                new_clip = resynth(t, "duration_rewrite")
            except Exception:
                t.hi_fit = old_fit
                break
            if new_clip.natural_duration >= clip.natural_duration:
                t.hi_fit = old_fit
                continue
            # subtitles show what is actually spoken; hi_raw keeps the original
            if old_display in ("", old_fit):
                t.hi_display = new_text
            new_clip.retry_history = clip.retry_history + new_clip.retry_history
            clip = new_clip
            clips[t.turn_id] = clip
            t.add_flag("duration_rewrite")
            speed, overflow = plan_speed(clip.natural_duration, slack, cfg.max_stretch)
        # stretch
        if abs(speed - 1.0) > 1e-3:
            src = Path(clip.path)
            dst = src.with_name(src.stem + "_fit.wav")
            audio.time_stretch(src, dst, speed)
            clip.path = str(dst)
            clip.final_duration = audio.probe_duration(dst)
        else:
            clip.final_duration = clip.natural_duration
        clip.stretch = round(speed, 4)
        sched = start
        if overflow > cfg.overflow_tolerance_s:
            # pre-roll into silence left by the previous turn (never before it ends)
            earliest = max(start - cfg.preroll_s, last_end + cfg.guard_s)
            if earliest < start:
                shift = min(start - earliest, overflow)
                sched = start - shift
                overflow -= shift
        clip.scheduled_start = round(sched, 3)
        clip.scheduled_end = round(sched + clip.final_duration, 3)
        last_end = max(last_end, clip.scheduled_end) if not t.overlaps_with else last_end
        rec = {"turn_id": t.turn_id, "speaker_id": t.speaker_id,
               "source_start": t.source_start, "source_end": t.source_end,
               "scheduled_start": clip.scheduled_start, "scheduled_end": clip.scheduled_end,
               "stretch": clip.stretch, "rewrites": rewrites}
        overflow = max(0.0, clip.scheduled_end - avail_end)
        if media_duration > 0 and clip.scheduled_end > media_duration:
            rec["beyond_media_end_s"] = round(clip.scheduled_end - media_duration, 3)
        if overflow > cfg.overflow_tolerance_s:
            rec["overflow_s"] = round(overflow, 3)
            rec["severity"] = "draft" if overflow > cfg.draft_overflow_s else "warning"
            t.add_flag("timing_overflow")
            deviations.append(rec)
        elif clip.stretch > 1.0 or sched < start or rewrites:
            rec["severity"] = "info"
            deviations.append(rec)
        on_progress((n + 1) / max(1, len(ordered)), f"Fitted {n + 1}/{len(ordered)} turns")
    return deviations


def assign_tracks(clips: Sequence[Clip], gap: float = 0.0) -> int:
    """Greedy track assignment so no two clips overlap on one track."""
    ends: List[float] = []
    for c in sorted(clips, key=lambda c: (c.scheduled_start, c.scheduled_end)):
        for i, e in enumerate(ends):
            if c.scheduled_start >= e + gap - 1e-6:
                c.track = i
                ends[i] = c.scheduled_end
                break
        else:
            c.track = len(ends)
            ends.append(c.scheduled_end)
    return len(ends)
