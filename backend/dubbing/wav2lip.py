"""Wav2Lip lip-sync post-processing wrapper.

Wav2Lip is NOT shipped with this repo (the model + code is ~500MB). User must:
  1. Clone https://github.com/Rudrabha/Wav2Lip into backend/wav2lip/
  2. Download wav2lip_gan.pth into backend/wav2lip/checkpoints/
  3. pip install -r backend/wav2lip/requirements.txt
  4. Toggle use_wav2lip=True in DubbingSettings

Call apply_lipsync() after the dubbed MP4 is assembled. On any failure
(missing repo, missing checkpoint, no face detected, subprocess error)
returns False — the caller keeps the original assembled video.

Wav2Lip performs best on:
  - Single front-facing speaker in a close-up shot
  - Clean audio with clear speech (our dubbed Hindi qualifies)
  - GPU runtime (CPU is ~10-20× realtime, impractical for real videos)
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

# Match tqdm-style "NN%" inside a Wav2Lip inference.py progress line.
# Wav2Lip uses two tqdm bars: face detection pass, then frame generation pass.
_TQDM_PCT = re.compile(r"(\d{1,3})%\|")

_WAV2LIP_REPO_DIR = Path(__file__).resolve().parent.parent / "wav2lip"
_CHECKPOINT_DIR = _WAV2LIP_REPO_DIR / "checkpoints"


def _pick_checkpoint() -> Optional[Path]:
    """Prefer wav2lip_gan.pth (better quality); fall back to wav2lip.pth."""
    for name in ("wav2lip_gan.pth", "wav2lip.pth"):
        p = _CHECKPOINT_DIR / name
        if p.exists():
            return p
    return None


def is_available() -> bool:
    """True when the Wav2Lip repo and at least one checkpoint are present."""
    return (
        _WAV2LIP_REPO_DIR.exists()
        and (_WAV2LIP_REPO_DIR / "inference.py").exists()
        and _pick_checkpoint() is not None
    )


def availability_reason() -> str:
    """Human-readable explanation of why is_available() is False."""
    if not _WAV2LIP_REPO_DIR.exists():
        return (
            f"Wav2Lip repo not found at {_WAV2LIP_REPO_DIR}. "
            f"Clone https://github.com/Rudrabha/Wav2Lip into that path."
        )
    if not (_WAV2LIP_REPO_DIR / "inference.py").exists():
        return f"Wav2Lip repo at {_WAV2LIP_REPO_DIR} is missing inference.py."
    if _pick_checkpoint() is None:
        return (
            f"No Wav2Lip checkpoint in {_CHECKPOINT_DIR}. "
            f"Download wav2lip_gan.pth (recommended) or wav2lip.pth."
        )
    return "Wav2Lip is available."


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def apply_lipsync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    work_dir: Path,
    checkpoint: Optional[Path] = None,
    report: Optional[Callable[[str, float, str], None]] = None,
    timeout_seconds: int = 3600,
    nosmooth: bool = True,
) -> bool:
    """Run Wav2Lip inference to re-sync the video's lips to the given audio.

    Returns True iff a valid output file was produced at output_path.
    On any failure, returns False and the caller keeps the original video.

    Args:
        video_path: input MP4 (dubbed, pre-lipsync)
        audio_path: dubbed audio (typically the same track already muxed in)
        output_path: where the lip-synced MP4 will be written
        work_dir: scratch directory (unused today, reserved)
        checkpoint: optional override; defaults to wav2lip_gan.pth if present
        report: optional callback(stage, progress, message)
        timeout_seconds: subprocess hard cap (default 1 hour)
        nosmooth: pass --nosmooth to inference.py (recommended for talking heads)
    """
    if not is_available():
        if report:
            report("lipsync", 0.0, availability_reason())
        return False

    if not video_path.exists():
        if report:
            report("lipsync", 0.0, f"Wav2Lip: input video not found at {video_path}")
        return False
    if not audio_path.exists():
        if report:
            report("lipsync", 0.0, f"Wav2Lip: audio not found at {audio_path}")
        return False

    ckpt = checkpoint or _pick_checkpoint()
    if ckpt is None:
        if report:
            report("lipsync", 0.0, availability_reason())
        return False

    gpu = _has_gpu()
    if not gpu and report:
        report("lipsync", 0.05,
               "Wav2Lip: no GPU detected — CPU inference is ~10-20× realtime. "
               "Consider disabling use_wav2lip for long videos.")

    if report:
        report("lipsync", 0.10,
               f"Wav2Lip: applying lip sync with {ckpt.name} "
               f"({'GPU' if gpu else 'CPU'})...")

    cmd = [
        sys.executable,
        "-u",  # unbuffered stdout so tqdm lines reach us in real time
        str(_WAV2LIP_REPO_DIR / "inference.py"),
        "--checkpoint_path", str(ckpt),
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(output_path),
    ]
    if nosmooth:
        cmd.append("--nosmooth")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_WAV2LIP_REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # tqdm writes to stderr — merge into one stream
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except FileNotFoundError as e:
        if report:
            report("lipsync", 1.0, f"Wav2Lip subprocess not launchable: {e}")
        return False
    except Exception as e:
        if report:
            report("lipsync", 1.0, f"Wav2Lip subprocess error: {e}")
        return False

    # Wav2Lip runs two tqdm phases in order: face detection, then generation.
    # We track phase transitions by watching for percent resets (e.g., 100→0).
    phase = 0          # 0 = face detection, 1 = lip generation
    last_reported = -1 # last percent we forwarded (throttle)
    last_line = ""
    # Collect last ~40 lines for the failure tail (bounded — no memory runaway).
    tail: list[str] = []

    # Watchdog: `for raw in proc.stdout` is blocking I/O. If Wav2Lip stalls
    # silently (GPU driver hang, stuck face detector) no new line arrives and
    # an inline deadline check would never fire. A timer thread kills the
    # process at the deadline, which closes stdout and lets the for-loop exit
    # naturally. Set a flag so we can distinguish timeout from other failures.
    timed_out = {"hit": False}

    def _kill_on_deadline() -> None:
        timed_out["hit"] = True
        try:
            proc.kill()
        except Exception:
            pass

    watchdog = threading.Timer(timeout_seconds, _kill_on_deadline)
    watchdog.daemon = True
    watchdog.start()

    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                last_line = line
                tail.append(line)
                if len(tail) > 40:
                    tail.pop(0)
            m = _TQDM_PCT.search(line)
            if m:
                pct = int(m.group(1))
                # Detect phase flip: percent dropped by > 30 after being near 100
                if last_reported > 70 and pct < 30:
                    phase = min(phase + 1, 1)
                    last_reported = -1
                # Throttle to ~5% granularity so we don't spam the UI
                if pct // 5 != last_reported // 5 and report:
                    last_reported = pct
                    phase_weight = 0.30 if phase == 0 else 0.60
                    phase_base = 0.10 if phase == 0 else 0.40
                    overall = phase_base + phase_weight * (pct / 100.0)
                    label = "face detection" if phase == 0 else "lip generation"
                    report("lipsync", overall, f"Wav2Lip {label}: {pct}%")
        proc.wait(timeout=30)
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        if report:
            report("lipsync", 1.0, f"Wav2Lip stream error: {e}. Last line: {last_line}")
        return False
    finally:
        watchdog.cancel()

    if timed_out["hit"]:
        if report:
            report("lipsync", 1.0,
                   f"Wav2Lip timed out after {timeout_seconds // 60} min — "
                   f"keeping original video")
        return False

    if proc.returncode != 0:
        err_tail = "\n".join(tail[-8:])
        if report:
            report("lipsync", 1.0,
                   f"Wav2Lip exited with code {proc.returncode}. Tail: {err_tail.strip()}")
        return False

    if not output_path.exists() or output_path.stat().st_size == 0:
        if report:
            report("lipsync", 1.0, "Wav2Lip produced no output file — keeping original")
        return False

    if report:
        report("lipsync", 1.0, "Wav2Lip: lip sync applied successfully")
    return True
