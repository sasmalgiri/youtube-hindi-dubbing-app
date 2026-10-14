"""Clips -> one silent video on the script's timeline (+ a title card for a
missing clip so timing never shifts), and the Hindi SRT with speaker labels.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from .generate import ClipResult, FPS
from .script import SHOT_SECONDS, Story, write_srt


def _run(cmd: List[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")


def _placeholder(path: Path, text: str, width: int, height: int, ffmpeg: str) -> None:
    """A dark card: the slot is kept, the gap is visible (no drawtext: Windows
    ffmpeg builds often have no fontconfig)."""
    _run([ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c=0x202020:s={width}x{height}:r={FPS}:d={SHOT_SECONDS}",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])


def assemble(story: Story, clips: List[ClipResult], out_dir: Path, ffmpeg: str = "ffmpeg",
             width: int = 832, height: int = 480) -> Dict[str, Path]:
    """Returns {"video": silent mp4, "srt": Hindi SRT}. Missing clips become cards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key = {(c.scene_id, c.index): c for c in clips}
    parts: List[Path] = []
    missing = []
    for s in story.scenes:
        for k in range(s.n_shots):
            c = by_key.get((s.id, k))
            if c and not c.error and c.path.exists():
                parts.append(c.path)
            else:
                card = out_dir / f"missing_{s.id}_{k}.mp4"
                _placeholder(card, f"{s.id} shot {k + 1} not generated", width, height, ffmpeg)
                parts.append(card)
                missing.append(f"{s.id} shot {k + 1}")
    # Uniform re-encode through the concat demuxer (clips share size/fps, but
    # a re-encode guarantees one timebase and exact SHOT_SECONDS per part).
    lst = out_dir / "concat.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    video = out_dir / "story_silent.mp4"
    _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
          "-vf", f"scale={width}:{height},fps={FPS}", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
          "-pix_fmt", "yuv420p", "-an", str(video)])
    srt = out_dir / "story_hi.srt"
    write_srt(story, srt, speaker_labels=True)
    shutil.copy2(srt, out_dir / "story_hi_labels.srt")
    return {"video": video, "srt": srt, "missing": missing}   # type: ignore[dict-item]
