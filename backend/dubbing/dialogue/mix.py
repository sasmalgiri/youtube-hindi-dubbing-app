"""Timestamped multi-track dialogue render, background policy, mux, subtitles.

Background policy (dialogue profile):
  * "auto"  : try Demucs speech/music separation; if it is unavailable or
              fails, output clean Hindi-only audio and report it.
  * "demucs": same as auto but the failure is also a job warning.
  * "none"  : Hindi-only audio (no background bed).
The original English audio is never used as a "background" stand-in.
Separation is an *estimate* of the non-speech bed, not a clean M&E track.
The dialogue path separates in a child process (separate_in_child), so a
cancel stops it at once.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import audio
from .contracts import Clip, Turn


def render_dialogue(clips: Sequence[Clip], duration: float, out_dir: Path,
                    sr: int = audio.SR, level_dbfs: Optional[float] = -20.0) -> Dict:
    """Place every accepted clip at its scheduled time. Writes one stem per
    track plus the summed dialogue bus. Nothing is cut: if clips extend past
    `duration`, the bus is extended and the excess is reported.

    With ``level_dbfs`` each clip's active-speech level is brought toward
    that value (bounded) so speakers/voices sit at an even loudness before
    the final mix normalisation."""
    import numpy as np
    accepted = [c for c in clips if c.accepted]
    end = max([duration] + [c.scheduled_end for c in accepted])
    n = int(round(end * sr)) + 1
    n_tracks = max([c.track for c in accepted], default=-1) + 1
    tracks = [np.zeros(n, dtype=np.float32) for _ in range(max(1, n_tracks))]
    gains: List[float] = []
    for c in accepted:
        data, csr = audio.read_wav(Path(c.path))
        if csr != sr:
            tmp = Path(c.path).with_name(Path(c.path).stem + f"_{sr}.wav")
            audio.to_wav(Path(c.path), tmp, sr=sr)
            data, csr = audio.read_wav(tmp)
        mono = data.mean(axis=1)
        if level_dbfs is not None:
            g = audio.speech_level_gain(mono, sr, target_dbfs=level_dbfs)
            mono = mono * g
            c.voice_params["level_gain_db"] = round(20 * float(np.log10(max(g, 1e-6))), 2)
            gains.append(c.voice_params["level_gain_db"])
        s = int(round(c.scheduled_start * sr))
        e = min(n, s + len(mono))
        tracks[c.track][s:e] += mono[:e - s]
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for i, tr in enumerate(tracks):
        p = out_dir / f"dialogue_track_{i}.wav"
        audio.write_wav(p, tr, sr)
        stems.append(str(p))
    bus = np.sum(tracks, axis=0)
    peak = float(np.max(np.abs(bus))) if bus.size else 0.0
    if peak > 0.98:
        bus = bus * (0.98 / peak)
    bus_path = out_dir / "dialogue_bus.wav"
    audio.write_wav(bus_path, bus, sr)
    return {"bus": str(bus_path), "stems": stems, "tracks": len(tracks),
            "rendered_duration": round(n / sr, 3),
            "beyond_media_end_s": round(max(0.0, end - duration), 3),
            "pre_normalise_peak": round(peak, 4),
            "clip_level_gain_db": ({"min": min(gains), "max": max(gains)} if gains else None)}


SEPARATOR_MODELS = (
    # audio-separator (MIT; wraps UVR MDX / Roformer models, downloads weights
    # on first use). BS-Roformer first, then UVR MDX-Net Inst HQ 4 (the model
    # SoniTranslate uses for its voiceless background), then Demucs.
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "UVR-MDX-NET-Inst_HQ_4.onnx",
)


def separate_background(original: Path, work: Path, policy: str = "auto",
                        cancel_check: Optional[Callable[[], bool]] = None) -> Dict:
    """Split the original audio into a background bed and a vocals stem.

    Returns {"status", "background": path|None, "vocals": path|None,
    "detail"}. The vocals stem is used for speaker detection / voice
    analysis (music and effects otherwise create false speakers); the
    background is mixed under the Hindi dialogue. Separation is an estimate.
    ``cancel_check`` is polled between separator attempts (each one runs for
    about the length of the video); a cancel raises RuntimeError.
    """
    if policy == "none":
        return {"status": "disabled", "background": None, "vocals": None, "detail": "policy=none"}

    def check_cancel():
        if cancel_check and cancel_check():
            raise RuntimeError("Job cancelled by user")

    errors = []
    try:
        import audio_separator  # noqa: F401
        has_as = True
    except ImportError:
        has_as = False
    if has_as:
        models = [m for m in [os.environ.get("DIALOGUE_SEPARATOR_MODEL", "").strip()] if m] \
            + list(SEPARATOR_MODELS)
        for model in models:
            check_cancel()
            try:
                return _audio_separator(original, work, model)
            except Exception as e:
                errors.append(f"{model}: {str(e)[:120]}")
    try:
        import demucs  # noqa: F401
    except ImportError:
        detail = ("no separator installed (pip install \"audio-separator[gpu]\" or demucs); "
                  "output is Hindi dialogue only")
        if errors:
            detail = "audio-separator failed (" + "; ".join(errors) + "); demucs not installed"
        return {"status": "unavailable" if not errors else "failed", "background": None,
                "vocals": None, "detail": detail}
    out = work / "background_estimate.wav"
    check_cancel()
    try:
        res = _demucs_api(original, out, work)
    except Exception as api_err:
        check_cancel()
        try:
            res = _demucs_cli(original, out, work)
        except Exception as cli_err:
            return {"status": "failed", "background": None, "vocals": None,
                    "detail": f"demucs failed ({str(api_err)[:120]} / {str(cli_err)[:120]}); "
                              f"output is Hindi dialogue only"}
    if errors:
        res["detail"] += " (audio-separator failed: " + "; ".join(errors) + ")"
    return res


def separate_in_child(original: Path, work: Path, policy: str = "auto",
                      cancel_check: Optional[Callable[[], bool]] = None,
                      timeout: Optional[float] = None) -> Dict:
    """separate_background in a spawned child process (the dialogue default).

    Separation runs for about the length of the video. In a child, a cancel
    kills it within a second instead of after the whole video, a native crash
    or CUDA OOM kills only the child (reported as a failed separation), and
    all of its VRAM is back before Whisper and pyannote start. Same result
    dict as separate_background.
    """
    if policy == "none" or not (_installed("audio_separator") or _installed("demucs")):
        return separate_background(original, work, policy)   # nothing heavy to isolate
    import json
    import multiprocessing as mp
    import tempfile
    import time

    if cancel_check and cancel_check():
        raise RuntimeError("Job cancelled by user")

    if timeout is None:   # bounds a hang only; a cancel ends it at any time
        try:
            timeout = max(3600.0, 10.0 * audio.probe_duration(original))
        except Exception:
            timeout = 6 * 3600.0
    fd, result_path = tempfile.mkstemp(suffix=".json", prefix="dlg_separate_")
    os.close(fd)
    try:
        p = mp.get_context("spawn").Process(
            target=_separation_child, args=(str(original), str(work), policy, result_path),
            daemon=True)
        p.start()
        t0 = time.time()
        while p.is_alive():
            p.join(1.0)
            if cancel_check and cancel_check():
                p.kill()
                p.join(5)
                raise RuntimeError("Job cancelled by user")
            if time.time() - t0 > timeout:
                p.kill()
                p.join(5)
                return {"status": "failed", "background": None, "vocals": None,
                        "detail": f"separation timed out after {int(timeout)}s; "
                                  f"output is Hindi dialogue only"}
        data: Dict = {}
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
        if data.get("status"):
            return data
        return {"status": "failed", "background": None, "vocals": None,
                "detail": "separation process failed ("
                          + str(data.get("error") or f"exit code {p.exitcode}")[:200]
                          + "); output is Hindi dialogue only"}
    finally:
        try:
            Path(result_path).unlink(missing_ok=True)
        except Exception:
            pass


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _separation_child(original: str, work: str, policy: str, result_path: str) -> None:
    """Child-process body of separate_in_child (top level so spawn can pickle it)."""
    import json
    import sys

    def finish(code: int):
        # The result is on disk: skip interpreter teardown, where Windows CUDA
        # libraries can fail-fast while unloading (as in _pyannote_child).
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os._exit(code)

    try:
        try:
            # torch (and its cuDNN) before anything that could load
            # CTranslate2, as in every process that runs Roformer/Demucs.
            importlib.import_module("torch")
        except Exception:
            pass
        res = separate_background(Path(original), Path(work), policy)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(res, f)
    except BaseException as e:
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({"error": f"{type(e).__name__}: {e}"}, f)
        except Exception:
            pass
        finish(1)
    finish(0)


def _audio_separator(original: Path, work: Path, model: str) -> Dict:
    from audio_separator.separator import Separator
    backend_dir = Path(__file__).resolve().parents[2]
    out_dir = work / "separation"
    out_dir.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(out_dir), output_format="WAV",
                    model_file_dir=str(backend_dir / "models" / "separation"))
    try:
        sep.load_model(model_filename=model)
        names = sep.separate(str(original))
    finally:
        # Separation now runs first: release the model so its VRAM is free
        # for Whisper, pyannote and the local TTS/MT workers.
        del sep
        _free_gpu()
    files = [Path(f) if Path(f).is_absolute() else out_dir / f for f in names]
    voc = next((f for f in files if "(vocals)" in f.name.lower()), None)
    bed = next((f for f in files if "(instrumental)" in f.name.lower()
                or "(no_vocals)" in f.name.lower() or "(no vocals)" in f.name.lower()), None)
    if bed is None or not bed.exists():
        raise RuntimeError(f"no instrumental stem in {[f.name for f in files]}")
    background = work / "background_estimate.wav"
    audio.to_wav(bed, background, channels=2)
    vocals = None
    if voc is not None and voc.exists():
        vocals = work / "vocals_estimate.wav"
        audio.to_wav(voc, vocals, channels=2)
    return {"status": "ok", "background": str(background),
            "vocals": str(vocals) if vocals else None,
            "detail": f"audio-separator {model}"}


def _free_gpu():
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _demucs_api(original: Path, out: Path, work: Path) -> Dict:
    import os
    import demucs.api
    model = os.environ.get("DIALOGUE_DEMUCS_MODEL", "htdemucs")
    sep = demucs.api.Separator(model=model, overlap=0.25, segment=7, shifts=0,
                               progress=False)
    _, stems = sep.separate_audio_file(str(original))
    bed = None
    for name, wav in stems.items():
        if name == "vocals":
            continue
        bed = wav if bed is None else bed + wav
    if bed is None:
        raise RuntimeError("no non-vocal stems")
    demucs.api.save_audio(bed, str(out), samplerate=sep.samplerate)
    vocals = None
    if "vocals" in stems:
        vocals = work / "vocals_estimate.wav"
        demucs.api.save_audio(stems["vocals"], str(vocals), samplerate=sep.samplerate)
    del sep
    _free_gpu()
    return {"status": "ok", "background": str(out), "vocals": str(vocals) if vocals else None,
            "detail": f"demucs.api {model} (overlap 0.25, segment 7s); sum of non-vocal stems"}


def _demucs_cli(original: Path, out: Path, work: Path) -> Dict:
    import demucs.separate
    dst = work / "demucs_out"
    demucs.separate.main(["--two-stems", "vocals", "-n", "htdemucs", "--overlap", "0.25",
                          "-o", str(dst), str(original)])
    nv = dst / "htdemucs" / original.stem / "no_vocals.wav"
    if not nv.exists():
        raise RuntimeError("no_vocals.wav not produced")
    audio.to_wav(nv, out, channels=2)
    vocals = None
    vv = dst / "htdemucs" / original.stem / "vocals.wav"
    if vv.exists():
        vocals = work / "vocals_estimate.wav"
        audio.to_wav(vv, vocals, channels=2)
    shutil.rmtree(dst, ignore_errors=True)
    return {"status": "ok", "background": str(out), "vocals": str(vocals) if vocals else None,
            "detail": "demucs CLI --two-stems vocals (htdemucs, overlap 0.25)"}


def final_mix(dialogue_bus: Path, background: Optional[Path], out: Path,
              duration: float, bg_gain: float = 0.8, target_lufs: float = -17.0,
              vocals_key: Optional[Path] = None) -> Dict:
    """Duck background under dialogue, normalise loudness, limit peaks.

    With ``vocals_key`` (the separated English vocals stem) the background
    gets a second, gentle duck wherever the original speech was: separation
    leaves faint English residue in the bed, which is otherwise exposed
    where the Hindi line is shorter than the English one.
    """
    pre = out.with_name("mix_pre_norm.wav")
    dur = f"{duration:.3f}"
    duck_residue = False
    if background:
        inputs = ["-i", str(dialogue_bus), "-i", str(background)]
        bg = "bg"
        duck_residue = bool(vocals_key and Path(vocals_key).exists())
        flt = (f"[0:a]aformat=channel_layouts=stereo,apad,atrim=0:{dur},asplit=2[dlg][sc];"
               f"[1:a]aformat=channel_layouts=stereo,apad,atrim=0:{dur},volume={bg_gain}[bg];")
        if duck_residue:
            inputs += ["-i", str(vocals_key)]
            flt += (f"[2:a]aformat=channel_layouts=stereo,apad,atrim=0:{dur}[vk];"
                    f"[bg][vk]sidechaincompress=threshold=0.03:ratio=2.5:attack=10:release=250[bgv];")
            bg = "bgv"
        flt += (f"[{bg}][sc]sidechaincompress=threshold=0.05:ratio=6:attack=20:release=350[duck];"
                f"[dlg][duck]amix=inputs=2:duration=first:normalize=0[m]")
        audio.run_ffmpeg(inputs + ["-filter_complex", flt, "-map", "[m]", "-ar", str(audio.SR),
                                   "-ac", "2", "-acodec", "pcm_s16le", str(pre)])
    else:
        audio.run_ffmpeg(["-i", str(dialogue_bus), "-af",
                          f"aformat=channel_layouts=stereo,apad,atrim=0:{dur}",
                          "-ar", str(audio.SR), "-ac", "2", "-acodec", "pcm_s16le", str(pre)])
    info = audio.loudnorm_two_pass(pre, out, target_i=target_lufs)
    pre.unlink(missing_ok=True)
    st = audio.audio_stats(out)
    info.update({"duration": round(st["duration"], 3), "peak": round(st["peak"], 4),
                 "clip_fraction": st["clip_fraction"], "channels": st["channels"],
                 "sample_rate": st["sample_rate"], "background": bool(background),
                 "residue_duck": duck_residue})
    return info


def mux(video: Path, mix_wav: Path, out: Path, bitrate: str = "192k",
        subtitles: Optional[Path] = None) -> Path:
    args = ["-i", str(video), "-i", str(mix_wav)]
    if subtitles:
        args += ["-i", str(subtitles)]
    args += ["-map", "0:v:0", "-map", "1:a:0"]
    if subtitles:
        args += ["-map", "2:s:0", "-c:s", "mov_text", "-metadata:s:s:0", "language=hin"]
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", bitrate,
             "-metadata:s:a:0", "language=hin", "-movflags", "+faststart", str(out)]
    audio.run_ffmpeg(args)
    return out


# ── subtitles ─────────────────────────────────────────────────────────────
def _ts(t: float, sep: str = ",") -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def wrap_lines(text: str, width: int = 42) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > 2:  # rebalance into two lines rather than dropping text
        half = (len(words) + 1) // 2
        lines = [" ".join(words[:half]), " ".join(words[half:])]
    return "\n".join(lines)


def subtitle_cues(turns: Dict[str, Turn], clips: Dict[str, Clip]) -> List[Dict]:
    """Cues timed to the *scheduled* dub audio (what is actually heard)."""
    cues = []
    for tid, c in clips.items():
        if not c.accepted:
            continue
        t = turns[tid]
        cues.append({"start": c.scheduled_start, "end": c.scheduled_end,
                     "text": wrap_lines(t.hi_display or t.speech_text), "turn_id": tid})
    return sorted(cues, key=lambda x: x["start"])


def write_srt(cues: Sequence[Dict], path: Path, key: str = "text"):
    lines = []
    for i, c in enumerate(cues, 1):
        lines.append(f"{i}\n{_ts(c['start'])} --> {_ts(c['end'])}\n{c[key]}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(cues: Sequence[Dict], path: Path, key: str = "text"):
    lines = ["WEBVTT", ""]
    for c in cues:
        lines += [f"{_ts(c['start'], '.')} --> {_ts(c['end'], '.')}", c[key], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def verify_subtitles(cues: Sequence[Dict], clips: Dict[str, Clip]) -> List[Dict]:
    """Every cue must match the final scheduled audio of its turn."""
    out = []
    for c in cues:
        clip = clips.get(c["turn_id"])
        if clip is None or abs(clip.scheduled_start - c["start"]) > 0.01 \
                or abs(clip.scheduled_end - c["end"]) > 0.01:
            out.append({"turn_id": c["turn_id"], "problem": "subtitle_not_aligned_to_audio"})
    return out
