"""Small FFmpeg/numpy audio helpers for the dialogue profile."""
from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Dict, Optional, Sequence

SR = 48000


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError("FFmpeg not found. Install it (winget install ffmpeg) and ensure it is on PATH.")


def run_ffmpeg(args: Sequence[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-y", *map(str, args)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({res.returncode}): {res.stderr[-800:]}")
    return res


@functools.lru_cache(maxsize=1)
def ffmpeg_filters() -> str:
    res = subprocess.run([find_ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True)
    return res.stdout


def has_filter(name: str) -> bool:
    return bool(re.search(rf"\s{re.escape(name)}\s", ffmpeg_filters()))


def probe_duration(path: Path) -> float:
    """Duration via WAV header when possible, else ffmpeg stderr parse."""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            with wave.open(str(p), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass
    res = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(p)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr)
    if not m:
        raise RuntimeError(f"Could not probe duration of {p.name}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def probe_streams(path: Path) -> Dict:
    res = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    info = {"video": bool(re.search(r"Stream #\S+.*Video:", res.stderr)),
            "audio": bool(re.search(r"Stream #\S+.*Audio:", res.stderr))}
    m = re.search(r"Audio:.*?(\d+) Hz, (mono|stereo|[\d.]+ channels)", res.stderr)
    if m:
        info["sample_rate"] = int(m.group(1))
        info["channels"] = 1 if m.group(2) == "mono" else 2 if m.group(2) == "stereo" else m.group(2)
    try:
        info["duration"] = probe_duration(path)
    except Exception:
        pass
    return info


def to_wav(src: Path, dst: Path, sr: int = SR, channels: int = 1) -> Path:
    run_ffmpeg(["-i", str(src), "-vn", "-ac", str(channels), "-ar", str(sr),
                "-acodec", "pcm_s16le", str(dst)])
    return dst


def read_wav(path: Path):
    """Return (float32 array shape (n, ch), sr)."""
    import numpy as np
    with wave.open(str(path), "rb") as wf:
        sr, ch, sw, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
        raw = wf.readframes(n)
    if sw != 2:
        raise ValueError(f"expected 16-bit PCM: {path}")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return data.reshape(-1, ch), sr


def write_wav(path: Path, data, sr: int):
    import numpy as np
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(arr.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def audio_stats(path: Path) -> Dict:
    import numpy as np
    data, sr = read_wav(path)
    mono = data.mean(axis=1) if data.size else data.reshape(-1)
    n = len(mono)
    if n == 0:
        return {"duration": 0.0, "rms": 0.0, "peak": 0.0, "clip_fraction": 0.0, "sample_rate": sr,
                "channels": data.shape[1] if data.ndim == 2 else 1}
    return {"duration": n / float(sr),
            "rms": float(np.sqrt(np.mean(mono ** 2))),
            "peak": float(np.max(np.abs(data))),
            "clip_fraction": float(np.mean(np.abs(data) >= 0.999)),
            "sample_rate": sr, "channels": data.shape[1]}


def trim_silence(path: Path, out: Path, threshold_db: float = -50.0, keep_head_s: float = 0.04,
                 keep_tail_s: float = 0.12) -> float:
    """Trim leading/trailing silence only (never inside speech). Returns new duration.

    The tail keeps more padding than the head (soft Hindi endings such as a
    breathy "है" decay below the threshold), and short fades avoid clicks at
    the cut points (padding values follow pyVideoTrans' remove_silence_wav).
    """
    import numpy as np
    data, sr = read_wav(path)
    mono = np.abs(data.mean(axis=1))
    if len(mono) == 0:
        write_wav(out, data, sr)
        return 0.0
    thr = 10 ** (threshold_db / 20.0)
    win = max(1, int(0.01 * sr))
    env = np.convolve(mono, np.ones(win) / win, mode="same")
    idx = np.nonzero(env > thr)[0]
    if len(idx) == 0:
        write_wav(out, data, sr)
        return len(data) / float(sr)
    s = max(0, idx[0] - int(keep_head_s * sr))
    e = min(len(data), idx[-1] + int(keep_tail_s * sr))
    seg = np.array(data[s:e], dtype=np.float32)
    fi, fo = min(len(seg), int(0.005 * sr)), min(len(seg), int(0.015 * sr))
    if s > 0 and fi > 1:
        seg[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    if e < len(data) and fo > 1:
        seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None]
    write_wav(out, seg, sr)
    return (e - s) / float(sr)


def speech_level_gain(x, sr: int, target_dbfs: float = -20.0, max_gain_db: float = 9.0,
                      peak_cap: float = 0.89) -> float:
    """Linear gain bringing a clip's *active speech* RMS toward target_dbfs.

    TTS voices (and pitch variants) come out at different loudness, so one
    speaker can sound much louder than another before the final loudnorm,
    which only fixes the overall mix. Bounded to +/-max_gain_db and so the
    peak stays under peak_cap.
    """
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 1.0
    win = max(1, int(0.02 * sr))
    n = len(x) // win
    if n == 0:
        return 1.0
    frames = x[:n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    loud = rms[rms > max(1e-4, float(np.max(rms)) * 0.1)]
    if loud.size == 0:
        return 1.0
    level_db = 20 * np.log10(float(np.sqrt(np.mean(loud ** 2))))
    gain_db = max(-max_gain_db, min(max_gain_db, target_dbfs - level_db))
    gain = 10 ** (gain_db / 20.0)
    peak = float(np.max(np.abs(x)))
    if peak > 0 and peak * gain > peak_cap:
        gain = peak_cap / peak
    return float(gain)


def time_stretch(src: Path, dst: Path, speed: float) -> Path:
    """Pitch-preserving tempo change. speed>1 = faster/shorter."""
    if abs(speed - 1.0) < 1e-3:
        shutil.copyfile(src, dst)
        return dst
    if has_filter("rubberband"):
        flt = f"rubberband=tempo={speed:.5f}:pitchq=quality"
    else:
        flt = f"atempo={speed:.5f}"
    run_ffmpeg(["-i", str(src), "-filter:a", flt, "-ar", str(SR), "-ac", "1",
                "-acodec", "pcm_s16le", str(dst)])
    return dst


def loudnorm_two_pass(src: Path, dst: Path, target_i: float = -17.0, tp: float = -1.5,
                      lra: float = 11.0, channels: int = 2) -> Dict:
    """Measured (two-pass) EBU R128 normalisation followed by a peak limiter."""
    first = run_ffmpeg(["-i", str(src), "-af",
                        f"loudnorm=I={target_i}:TP={tp}:LRA={lra}:print_format=json",
                        "-f", "null", "-"])
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", first.stderr, re.DOTALL)
    measured = json.loads(m.group(0)) if m else {}
    if measured and measured.get("input_i") not in (None, "-inf"):
        flt = (f"loudnorm=I={target_i}:TP={tp}:LRA={lra}"
               f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
               f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
               f":offset={measured.get('target_offset', 0)}:linear=true")
    else:
        flt = f"loudnorm=I={target_i}:TP={tp}:LRA={lra}"
    run_ffmpeg(["-i", str(src), "-af", f"{flt},alimiter=limit=0.89:level=false",
                "-ar", str(SR), "-ac", str(channels), "-acodec", "pcm_s16le", str(dst)])
    return {"target_i": target_i, "true_peak": tp, "measured": measured}
