"""Clips for every shot, from the local Wan worker (dubbing/story/wan_worker.py).

The worker loads the model once; this side feeds it one shot at a time,
turns its step events into progress lines (a 5 s clip takes minutes on a
12 GB GPU: a silent wait looks like a crash), skips clips that already
exist (a re-run or a single-shot regeneration), and kills the child on
cancel. Failures are per clip and reported, never hidden: a scene whose
clip failed is listed in `ClipResult.error` and the caller decides.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .script import SHOT_SECONDS, Story

WORKER = Path(__file__).resolve().parent / "wan_worker.py"
FPS = 16
FRAMES = int(SHOT_SECONDS * FPS) + 1       # 49: Wan wants 4k+1 frames
DEFAULT_STEPS = int(os.environ.get("STORY_VIDEO_STEPS", "20"))   # measured: 25 steps = 8.3 min per 3 s clip on an RTX 3060


@dataclass
class ClipResult:
    scene_id: str
    index: int
    path: Path
    seconds: float = 0.0
    error: str = ""


class WanGenerator:
    def __init__(self, clip_dir: Path, steps: int = DEFAULT_STEPS, width: int = 832, height: int = 480,
                 python: Optional[str] = None):
        self.clip_dir = clip_dir
        self.steps, self.width, self.height = steps, width, height
        self.python = python or os.environ.get("STORY_VIDEO_PYTHON") or sys.executable
        self._proc: Optional[subprocess.Popen] = None

    # ── lifecycle ──
    def start(self, on_progress: Callable[[float, str], None]) -> None:
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self._proc = subprocess.Popen([self.python, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=env,
                                      cwd=str(WORKER.parent))
        t0 = time.time()
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("the video model process ended before it was ready (out of memory? "
                                   "model not downloaded?)")
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "ready":
                on_progress(0.0, f"Video model loaded in {time.time() - t0:.0f}s ({ev.get('model')})")
                return

    def close(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            p.stdin.write(json.dumps({"quit": True}) + "\n")
            p.stdin.flush()
            p.wait(timeout=20)
        except Exception:
            pass
        if p.poll() is None:
            p.kill()

    # ── generation ──
    def generate_all(self, story: Story, on_progress: Callable[[float, str], None],
                     cancel_check: Callable[[], bool] = lambda: False,
                     only: Optional[set] = None) -> List[ClipResult]:
        """Every shot of every scene (or just the (scene_id, index) pairs in `only`)."""
        todo = [(s.id, k, prompt) for s in story.scenes for k, prompt in enumerate(s.shots)
                if only is None or (s.id, k) in only]
        results: List[ClipResult] = []
        total = len(todo)
        done = 0
        times: List[float] = []
        for sid, k, prompt in todo:
            out = self.clip_dir / f"{sid}_{k}.mp4"
            if out.exists() and out.stat().st_size > 1000 and (sid, k) not in (only or set()):
                results.append(ClipResult(sid, k, out, 0.0))
                done += 1
                continue
            if cancel_check():
                self.close()
                raise RuntimeError("Job cancelled by user")
            eta = (sum(times) / len(times) * (total - done)) if times else 0.0
            label = (f"Clip {done + 1}/{total} ({sid} shot {k + 1})"
                     + (f", about {eta / 60:.0f} min left" if eta else ""))
            seed = (hash(sid) & 0xFFFF) * 97 + k
            r = self._one(sid, k, prompt, out, seed, lambda frac: on_progress(
                (done + frac) / max(total, 1), f"{label}: step {int(frac * self.steps)}/{self.steps}"),
                cancel_check)
            results.append(r)
            if r.seconds:
                times.append(r.seconds)
            done += 1
            on_progress(done / max(total, 1), f"{label}: " + (f"done in {r.seconds:.0f}s" if not r.error else f"FAILED ({r.error[:60]})"))
        return results

    def _one(self, sid: str, k: int, prompt: str, out: Path, seed: int,
             on_frac: Callable[[float], None], cancel_check: Callable[[], bool]) -> ClipResult:
        assert self._proc is not None, "start() first"
        req = {"id": f"{sid}_{k}", "prompt": prompt, "frames": FRAMES, "steps": self.steps, "seed": seed,
               "width": self.width, "height": self.height, "fps": FPS, "out": str(out)}
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        # read events; poll cancel between them (a step takes seconds, so this is responsive)
        box: Dict[str, Optional[dict]] = {"last": None}

        def reader():
            for line in self._proc.stdout:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "step":
                    on_frac(ev["step"] / max(ev["of"], 1))
                elif ev.get("event") in ("done", "error"):
                    box["last"] = ev
                    return
            box["last"] = {"event": "error", "error": "the video model process died (out of memory?)"}

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        while t.is_alive():
            t.join(1.0)
            if cancel_check():
                self._proc.kill()
                raise RuntimeError("Job cancelled by user")
        ev = box["last"] or {}
        if ev.get("event") == "done" and out.exists():
            return ClipResult(sid, k, out, float(ev.get("seconds", 0)))
        return ClipResult(sid, k, out, 0.0, ev.get("error", "no result"))
