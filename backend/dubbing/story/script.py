"""The story script: what gets generated, who says what, and when.

A Story is a JSON file the user can read and edit before anything is
generated. Timing is derived, never typed: every line's duration comes from
its Hindi word count at a natural speaking pace, each scene lasts as long
as its lines plus a little air, and a scene is cut into fixed-length shots
(the clip length the local model produces).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

WORDS_PER_SEC = 2.3          # Hindi Edge voices at natural rate (measured: 6648 words = 48 min)
LINE_GAP_S = 0.6             # breath between two lines of a scene
SCENE_LEAD_S = 1.0           # the picture is seen before the first line
SCENE_TAIL_S = 1.2           # ... and after the last one
SHOT_SECONDS = 3.0           # one generated clip (Wan 2.1: 49 frames at 16 fps; 81 frames peaks past 12 GB VRAM)
MIN_SCENE_S = 4.0


@dataclass
class Character:
    id: str                   # SPEAKER_00 style, stable across the story
    name: str
    gender: str               # male | female
    look: str = ""            # visual description used in every prompt (clothes, age, build)
    voice: str = ""           # Edge voice id; empty = picked by gender


@dataclass
class Line:
    character: str            # Character.id ("" = narrator)
    text: str                 # Hindi, as it will be spoken
    emotion: str = ""

    @property
    def seconds(self) -> float:
        words = len(re.findall(r"\S+", self.text))
        return max(0.8, words / WORDS_PER_SEC)


@dataclass
class Scene:
    id: str
    setting: str              # one paragraph of visual description (English, for the generator)
    characters: List[str] = field(default_factory=list)   # Character.ids on screen
    lines: List[Line] = field(default_factory=list)
    action: str = ""          # what happens visually (English)
    shots: List[str] = field(default_factory=list)   # prompt per generated clip (filled by plan_shots)

    @property
    def speech_seconds(self) -> float:
        return sum(l.seconds for l in self.lines) + LINE_GAP_S * max(0, len(self.lines) - 1)

    @property
    def seconds(self) -> float:
        return max(MIN_SCENE_S, self.speech_seconds + SCENE_LEAD_S + SCENE_TAIL_S)

    @property
    def n_shots(self) -> int:
        return max(1, math.ceil(self.seconds / SHOT_SECONDS))


@dataclass
class Story:
    title: str
    cast: List[Character]
    scenes: List[Scene]
    style: str = "cinematic, realistic, Indian short drama, natural lighting, 24fps"
    source_text: str = ""

    # ── persistence ───────────────────────────────────────────────────────
    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: Dict) -> "Story":
        cast = [Character(**c) for c in d.get("cast", [])]
        scenes = []
        for s in d.get("scenes", []):
            lines = [Line(**l) for l in s.get("lines", [])]
            scenes.append(Scene(id=s["id"], setting=s.get("setting", ""), characters=list(s.get("characters", [])),
                                lines=lines, action=s.get("action", ""), shots=list(s.get("shots", []))))
        return cls(title=d.get("title", "Untitled"), cast=cast, scenes=scenes,
                   style=d.get("style", cls.style), source_text=d.get("source_text", ""))

    @classmethod
    def load(cls, path: Path) -> "Story":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ── derived ───────────────────────────────────────────────────────────
    def character(self, cid: str) -> Optional[Character]:
        return next((c for c in self.cast if c.id == cid), None)

    @property
    def total_seconds(self) -> float:
        return sum(s.seconds for s in self.scenes)

    @property
    def total_shots(self) -> int:
        return sum(s.n_shots for s in self.scenes)

    def validate(self) -> List[str]:
        problems = []
        if not self.scenes:
            problems.append("no scenes")
        ids = {c.id for c in self.cast}
        if len(ids) != len(self.cast):
            problems.append("duplicate character ids")
        for c in self.cast:
            if c.gender not in ("male", "female"):
                problems.append(f"{c.name}: gender must be male or female (got {c.gender!r})")
        for s in self.scenes:
            if not s.setting.strip():
                problems.append(f"{s.id}: empty setting")
            for l in s.lines:
                if l.character and l.character not in ids:
                    problems.append(f"{s.id}: line by unknown character {l.character}")
                if not any(ch.isalnum() for ch in l.text):
                    problems.append(f"{s.id}: a line with nothing to speak")
        return problems


# ── shots ──────────────────────────────────────────────────────────────────
def plan_shots(story: Story) -> None:
    """One prompt per generated clip. Every shot of a scene shares the setting
    and the cast's looks (the only consistency a text-to-video model offers);
    the camera varies a little so consecutive clips do not look like loops."""
    angles = ["medium shot", "close-up on the speaker's face", "wide shot of the room",
              "over-the-shoulder shot", "slow push-in, medium shot"]
    for s in story.scenes:
        people = []
        for cid in s.characters:
            c = story.character(cid)
            if c:
                people.append(f"{c.name} ({c.gender}, {c.look})" if c.look else f"{c.name} ({c.gender})")
        who = "; ".join(people)
        s.shots = []
        for k in range(s.n_shots):
            parts = [s.setting.strip().rstrip(".")]
            if who:
                parts.append("Characters: " + who)
            if s.action.strip():
                parts.append(s.action.strip().rstrip("."))
            parts.append(angles[k % len(angles)])
            parts.append(story.style)
            s.shots.append(". ".join(parts) + ".")


# ── timing + SRT ───────────────────────────────────────────────────────────
def timeline(story: Story) -> List[Dict]:
    """Absolute cue times on the assembled video: [{scene, character, text, start, end}]."""
    cues, t = [], 0.0
    for s in story.scenes:
        dur = s.n_shots * SHOT_SECONDS          # the scene is exactly its clips
        air = dur - s.speech_seconds
        lead = min(SCENE_LEAD_S, air / 2) if air > 0 else 0.0
        pos = t + lead
        for l in s.lines:
            cues.append({"scene": s.id, "character": l.character, "text": l.text,
                         "start": round(pos, 3), "end": round(pos + l.seconds, 3)})
            pos += l.seconds + LINE_GAP_S
        t += dur
    return cues


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, r = divmod(ms, 3600000)
    m, r = divmod(r, 60000)
    s, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(story: Story, path: Path, speaker_labels: bool = True) -> List[Dict]:
    """The dub script: [SPEAKER_XX] labels let the dub path give each character its voice."""
    cues = timeline(story)
    out = []
    for i, c in enumerate(cues, 1):
        label = f"[{c['character']}] " if speaker_labels and c["character"] else ""
        out.append(f"{i}\n{_ts(c['start'])} --> {_ts(c['end'])}\n{label}{c['text']}\n")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return cues
