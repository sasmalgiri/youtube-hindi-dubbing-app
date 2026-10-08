"""Elastic tempo fit: schedule a supplied Hindi script onto the original timeline.

Why this exists
---------------
Tempo Match (`Pipeline._tempo_fit_segments`) fits each cue's speech INSIDE its
own slot and treats the next cue's start as sacred. That is right when the
slots are real speech spans with real gaps (Whisper). A script supplied as an
SRT (ChatGPT, a translator, an uploaded file) is different: its cues are
usually contiguous, rounded to whole seconds, and its text was never measured
against the audio. One 61-minute script measured 54 min of speech against
61 min of video, yet 17% of its cues needed more than 1.25x to fit their own
slot, 5% more than 1.5x and one 3x: a strict per-slot fit forces unintelligible
speed on those cues while the neighbouring cues sit half empty.

A human dubber would do the opposite: speak a little faster through a dense
stretch, let the speech lag the picture by a second or two, and catch up in the
next pause. This module plans exactly that, from measured durations only:

    for every cue choose (speed v, start offset o) minimising
        speed cost      (grows steeply above `v_soft`, hard ceiling `v_max`)
      + lateness cost   (speech starting after its cue, quadratic)
      + earliness cost  (speech starting before its cue, quadratic)
    subject to  no overlap: start >= previous finish + gap.

It is a dynamic program over (cue, start-offset bucket): the optimal schedule
for the cost model, deterministic, and fast (a second or two for 500 cues).
The video is never retimed, so the output keeps the source length.

The planner is pure (no I/O, no ffmpeg): the caller measures, then applies the
plan with atempo / native Edge rate and places clips at `Placement.start`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CueSpec:
    """One cue to place. All values in seconds, measured, not estimated."""
    start: float          # cue start on the original timeline
    end: float            # cue end (slot end); informational
    speech: float         # speech seconds at natural speed (silence-trimmed)
    pause: float = 0.0    # inter-sentence pauses inside the cue (never sped up)


@dataclass(frozen=True)
class FitParams:
    v_max: float = 1.40       # hard ceiling on the total speed-up of one cue
    v_soft: float = 1.20      # above this the speech starts to sound rushed
    v_step: float = 0.05      # speed choices: 1.00, 1.05, ... v_max
    gap: float = 0.15         # silence kept between consecutive cues' clips
    early_max: float = 0.6    # a clip may begin this much before its cue (when dead air precedes it)
    late_cap: float = 20.0    # offsets above this are not bucketed separately
    bucket: float = 0.2       # start-offset resolution of the dynamic program
    # cost weights (unit-less "badness"; only the ratios matter)
    w_speed: float = 30.0     # (v-1)^2
    w_rushed: float = 400.0   # (v-v_soft)^2 beyond the soft limit
    w_late: float = 1.2       # offset^2 for speech after its cue
    w_early: float = 3.0      # offset^2 for speech before its cue
    w_end: float = 50.0       # per second the last clip ends after the media


@dataclass
class Placement:
    index: int
    speed: float              # total speed-up applied to the cue's speech (1.0 = natural)
    start: float              # when the cue's first clip starts on the output timeline
    duration: float           # planned duration of the whole cue (speech/speed + pauses)
    offset: float             # start - cue.start  (>0: late, <0: early)

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Plan:
    placements: List[Placement]
    cost: float = 0.0
    stats: Dict[str, float] = field(default_factory=dict)


# A search node: best known way to reach "cue i starts at t".
#   (cost so far, exact start time t, previous node, speed chosen for the previous cue)
_Node = Tuple[float, float, Optional[tuple], Optional[float]]


def _speeds(p: FitParams) -> List[float]:
    out, k = [], 0
    while True:
        v = round(1.0 + k * p.v_step, 4)
        if v > p.v_max + 1e-9:
            break
        out.append(v)
        k += 1
    return out


def plan_elastic(cues: Sequence[CueSpec], media_end: float,
                 params: Optional[FitParams] = None) -> Plan:
    """Optimal speed + start for every cue. `cues` must be sorted by start."""
    p = params or FitParams()
    n = len(cues)
    if n == 0:
        return Plan([], 0.0, {})
    speeds = _speeds(p)
    inv_q = 1.0 / p.bucket
    floors = (-p.early_max, -p.early_max / 2.0, 0.0)

    def speed_cost(v: float) -> float:
        c = p.w_speed * (v - 1.0) ** 2
        if v > p.v_soft:
            c += p.w_rushed * (v - p.v_soft) ** 2
        return c

    spd_cost = {v: speed_cost(v) for v in speeds}

    def offset_cost(o: float) -> float:
        return p.w_late * o * o if o > 0 else p.w_early * o * o

    # layer for cue 0: it may start on time or a little early
    layer: Dict[int, tuple] = {}
    for fl in floors:
        key = int(round(fl * inv_q))
        if key not in layer:
            layer[key] = (0.0, cues[0].start + fl, None, None)

    for i in range(n - 1):
        c = cues[i]
        s_next = cues[i + 1].start
        nxt: Dict[int, tuple] = {}
        for node in layer.values():
            cost, t, _, _ = node
            base = cost + offset_cost(t - c.start)
            for v in speeds:
                fin = t + c.speech / v + c.pause
                cc = base + spd_cost[v]
                bound = fin + p.gap - s_next          # smallest allowed offset of the next cue
                for fl in floors:
                    o = max(bound, fl)
                    key = int(round(min(o, p.late_cap) * inv_q))
                    cur = nxt.get(key)
                    if cur is None or cc < cur[0]:
                        nxt[key] = (cc, s_next + o, node, v)
        layer = nxt

    # last cue + the end-of-media penalty
    c = cues[-1]
    best_total, best_node, best_v = None, None, None
    for node in layer.values():
        cost, t, _, _ = node
        base = cost + offset_cost(t - c.start)
        for v in speeds:
            fin = t + c.speech / v + c.pause
            cc = base + spd_cost[v] + p.w_end * max(0.0, fin - media_end)
            if best_total is None or cc < best_total:
                best_total, best_node, best_v = cc, node, v

    # walk back through the chosen nodes
    chosen: List[Tuple[float, float]] = [(best_node[1], best_v)]       # (start, speed) of the last cue
    node = best_node
    while node[2] is not None:
        prev, v_prev = node[2], node[3]
        chosen.append((prev[1], v_prev))
        node = prev
    chosen.reverse()

    out: List[Placement] = []
    for i, (t, v) in enumerate(chosen):
        c = cues[i]
        out.append(Placement(index=i, speed=v, start=t, duration=c.speech / v + c.pause,
                             offset=t - c.start))
    return Plan(out, best_total, summarize(out, media_end))


def summarize(placements: Sequence[Placement], media_end: float) -> Dict[str, float]:
    n = len(placements)
    if n == 0:
        return {}
    late = [pl.offset for pl in placements if pl.offset > 0]
    speeds = [pl.speed for pl in placements]
    return {
        "cues": n,
        "speed_mean": round(sum(speeds) / n, 3),
        "speed_max": round(max(speeds), 3),
        "faster_than_1.1": sum(1 for v in speeds if v > 1.1 + 1e-9),
        "faster_than_1.25": sum(1 for v in speeds if v > 1.25 + 1e-9),
        "faster_than_1.35": sum(1 for v in speeds if v > 1.35 + 1e-9),
        "late_gt_1s": sum(1 for o in late if o > 1.0),
        "late_gt_2s": sum(1 for o in late if o > 2.0),
        "late_gt_4s": sum(1 for o in late if o > 4.0),
        "late_max": round(max([0.0] + late), 2),
        "ends_at": round(placements[-1].end, 2),
        "media_end": round(media_end, 2),
    }
