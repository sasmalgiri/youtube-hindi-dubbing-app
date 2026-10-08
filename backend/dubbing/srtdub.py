"""SRT Direct Pipeline — translated SRT → TTS verbatim → concat (0 gap) → stretch 1-10× → mux.

You provide a perfect translated SRT; we:
  1. Download the source video
  2. Parse SRT cues verbatim (no merging, no splitting, no cleanup)
  3. Edge-TTS each cue as written
  4. Concatenate all TTS clips back-to-back with ZERO gap
  5. Stretch the video to match the concatenated audio length, capped 1-10×
  6. If audio is STILL longer than stretched video, freeze the last frame
     to extend the video — never trim audio
  7. Mux

NO translation. NO ASR. NO text cleanup. Audio is never modified or trimmed.
"""
from __future__ import annotations
import asyncio
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Callable, Optional
from concurrent.futures import ThreadPoolExecutor


# ── Executable resolver (same logic as wordchunk — skips broken venv shims) ──

def _resolve_executable(name: str) -> str:
    ext = ".exe" if sys.platform == "win32" else ""
    full_name = name + ext

    def _works(path: str) -> bool:
        try:
            r = subprocess.run([path, "--version"], capture_output=True,
                               timeout=5, text=True)
            return r.returncode == 0 and bool((r.stdout or r.stderr or "").strip())
        except Exception:
            return False

    venv_path = Path(sys.executable).parent / full_name
    if venv_path.exists() and _works(str(venv_path)):
        return str(venv_path)
    found = shutil.which(name)
    if found and _works(found):
        return found
    scripts_path = Path(sys.executable).parent / "Scripts" / full_name
    if scripts_path.exists() and _works(str(scripts_path)):
        return str(scripts_path)
    if sys.platform == "win32":
        userprofile = os.environ.get("USERPROFILE", str(Path.home()))
        for cand in [
            Path(userprofile) / "AppData" / "Local" / "Programs" / "Python" / "Python310" / "Scripts" / full_name,
            Path(userprofile) / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "Scripts" / full_name,
            Path(userprofile) / "AppData" / "Local" / "Programs" / "Python" / "Python312" / "Scripts" / full_name,
            Path("C:/Program Files/ffmpeg/bin") / full_name,
            Path("C:/ffmpeg/bin") / full_name,
        ]:
            if cand.exists() and _works(str(cand)):
                return str(cand)
    return name


# ── SRT parsing ─────────────────────────────────────────────────────────

# Match a single cue. Cue number is OPTIONAL (some tools export numberless
# SRTs). Timestamps accept both ``,`` (standard) and ``.`` (non-standard).
_SRT_TIMELINE_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*$"
)


def _srt_ts_to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_srt(srt_text: str) -> List[Dict]:
    """Parse SRT text into cues {idx, start, end, text} — verbatim, no cleanup.

    Tolerant parser: cue numbers optional, BOM stripped, accepts ``,`` or ``.``
    as the millisecond separator, collapses multi-line cue text into a single
    space-joined string.
    """
    # Strip UTF-8 BOM if present
    if srt_text.startswith("\ufeff"):
        srt_text = srt_text.lstrip("\ufeff")
    norm = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not norm:
        return []
    lines = norm.split("\n")

    cues: List[Dict] = []
    i = 0
    auto_idx = 0
    while i < len(lines):
        m = _SRT_TIMELINE_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        start = _srt_ts_to_seconds(m.group(1))
        end = _srt_ts_to_seconds(m.group(2))
        # Previous line may be a cue number — if it's a bare integer, use it
        auto_idx += 1
        idx = auto_idx
        if i > 0:
            prev = lines[i - 1].strip()
            if prev.isdigit():
                try:
                    idx = int(prev)
                except ValueError:
                    pass
        # Collect text lines until blank line or next timeline
        text_lines: List[str] = []
        i += 1
        while i < len(lines):
            ln = lines[i]
            if _SRT_TIMELINE_RE.match(ln.strip()):
                # Look-back: if prev collected line is a bare number, treat
                # it as the NEXT cue's index (not part of this cue's text).
                if text_lines and text_lines[-1].strip().isdigit():
                    text_lines.pop()
                break
            if ln.strip() == "":
                # Empty line — end of cue text
                i += 1
                break
            text_lines.append(ln)
            i += 1
        raw = "\n".join(text_lines)
        # Strip HTML-style formatting tags (<i>, <b>, <font color="...">)
        text = re.sub(r"<[^>]+>", "", raw).strip()
        # Collapse internal newlines/tabs to a single space
        text = re.sub(r"\s+", " ", text)
        # Drop control chars (NUL, bidi overrides, etc.) that can break
        # Edge-TTS or render wrongly; keep printable characters + space only.
        text = "".join(ch for ch in text if ch >= " " and ch != "\u200b" and ch != "\ufeff")
        text = text.strip()
        if text:
            cues.append({"idx": idx, "start": start, "end": end, "text": text})
    return cues


# ── Video download ──────────────────────────────────────────────────────

def _download_video(url: str, video_path: Path, ytdlp: str) -> None:
    """Fetch a video into ``video_path``. Handles:
      - HTTP(S) URLs via yt-dlp (2 format fallbacks)
      - Uploaded files marked as ``upload:filename.ext`` (finds the saved
        upload in the same work_dir)
      - Plain local file paths
    """
    if video_path.exists() and video_path.stat().st_size > 0:
        return

    # Non-URL: local upload or absolute path
    is_url = bool(re.match(r"^https?://", url or ""))
    if not is_url:
        # Upload case: upload endpoint saved the file as work_dir/source.EXT
        # where work_dir is the parent of video_path.
        for candidate in list(video_path.parent.glob("source.*")) + [Path(url or "")]:
            try:
                if candidate and candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                    import shutil as _sh
                    _sh.copy2(candidate, video_path)
                    if video_path.exists() and video_path.stat().st_size > 0:
                        return
            except Exception:
                continue
        raise RuntimeError(
            f"Non-URL source '{url}' and no uploaded file found in {video_path.parent}")

    # HTTPS URL path: yt-dlp with 2 format fallbacks
    last_err = ""
    for cmd in (
        [ytdlp, "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
         "--merge-output-format", "mp4", "--no-warnings", "-o", str(video_path), url],
        [ytdlp, "-f", "best", "--no-warnings", "-o", str(video_path), url],
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=600,
                                   text=True, encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                err_lines = (proc.stderr or proc.stdout or "").strip().splitlines()
                last_err = "; ".join(err_lines[-3:]) if err_lines else f"rc={proc.returncode}"
        except subprocess.TimeoutExpired:
            last_err = "yt-dlp timed out after 600s"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if video_path.exists() and video_path.stat().st_size > 0:
            return
    raise RuntimeError(f"Failed to download video from {url} — {last_err}")


# ── TTS ─────────────────────────────────────────────────────────────────

async def _edge_tts_async(text: str, out_mp3: Path, voice: str,
                           rate: str, timeout: float = 90.0) -> bool:
    """Edge-TTS with a hard timeout so a single stuck request can't freeze
    the whole batch. Returns True on success (non-empty MP3 written)."""
    import edge_tts
    try:
        comm = edge_tts.Communicate(text, voice, rate=rate)
        await asyncio.wait_for(comm.save(str(out_mp3)), timeout=timeout)
        return out_mp3.exists() and out_mp3.stat().st_size > 0
    except Exception:
        return False


def _tts_all_cues(cues: List[Dict], work_dir: Path, voice: str, rate: str,
                  ffmpeg: str, progress: Callable, cancel_check: Callable) -> List[Dict]:
    """Two-phase: (1) async Edge-TTS → MP3 files in parallel;
    (2) sync ffmpeg MP3→WAV via a thread pool. Keeping ffmpeg out of the
    asyncio loop is critical — otherwise ``subprocess.run`` blocks the loop
    and the Semaphore(30) concurrency collapses to serial execution."""
    total = len(cues)
    done = [0]
    lock = __import__("threading").Lock()

    # ── Phase 1: async TTS → MP3 ──────────────────────────────────────
    async def _tts_one(i: int, cue: Dict, sem: asyncio.Semaphore):
        if cancel_check():
            return
        async with sem:
            if cancel_check():
                return
            text = cue["text"].strip()
            if not text or not any(ch.isalnum() for ch in text):
                # empty, or punctuation only ("...", a music mark): nothing to speak
                cue["_mp3"] = None
                return
            mp3 = work_dir / f"sd_tts_{i:04d}.mp3"
            ok = await _edge_tts_async(text, mp3, voice, rate)
            if not ok and not cancel_check():
                ok = await _edge_tts_async(text, mp3, voice, rate)
            cue["_mp3"] = mp3 if (ok and mp3.exists() and mp3.stat().st_size > 0) else None

    async def _run_tts():
        sem = asyncio.Semaphore(30)
        await asyncio.gather(*[_tts_one(i, c, sem) for i, c in enumerate(cues)],
                              return_exceptions=True)
    asyncio.run(_run_tts())

    if cancel_check():
        raise RuntimeError("Cancelled by user")

    # ── Phase 2: parallel ffmpeg MP3 → WAV (thread pool, no event loop) ──
    def _convert_one(args):
        i, cue = args
        if cancel_check():
            return
        mp3 = cue.get("_mp3")
        if not mp3 or not mp3.exists():
            cue["wav"] = None
            return
        wav = work_dir / f"sd_tts_{i:04d}.wav"
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(mp3),
                 "-ar", "48000", "-ac", "2", "-acodec", "pcm_s16le",
                 str(wav)],
                check=True, capture_output=True, timeout=30)
            mp3.unlink(missing_ok=True)
            cue["wav"] = wav
        except Exception:
            cue["wav"] = None
        finally:
            with lock:
                done[0] += 1
                n = done[0]
            if n % 10 == 0 or n == total:
                progress("synthesize", 0.1 + 0.8 * (n / total),
                         f"[{int((n/total)*100)}%] TTS {n}/{total} cues")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_convert_one, enumerate(cues)))

    # B8: warn when cues were dropped — silent drops shift the whole timeline
    dropped = [i for i, c in enumerate(cues) if not c.get("wav")]
    if dropped:
        print(f"[SrtDub] WARNING: {len(dropped)}/{total} cues produced no audio "
              f"(indices: {dropped[:10]}{'...' if len(dropped) > 10 else ''}). "
              f"The final audio timeline will be SHORTER than expected.", flush=True)

    return cues


# ── Audio assembly (zero-gap concat) ────────────────────────────────────

def _read_wav_pcm(wav_path: Path) -> bytes:
    try:
        with open(wav_path, "rb") as f:
            riff = f.read(12)
            if riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
                return b""
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                cid = hdr[:4]
                csize = struct.unpack('<I', hdr[4:])[0]
                if cid == b'data':
                    return f.read(csize)
                f.seek(csize, 1)
    except Exception:
        pass
    return b""


def _concat_wavs_zero_gap(cues: List[Dict], out_wav: Path,
                          sample_rate: int = 48000,
                          ffmpeg: str = "ffmpeg") -> float:
    """Zero-gap concatenation of all valid cue WAVs. Writes a regular WAV
    via a Python-built RIFF header when size ≤ 4 GB; for larger totals,
    falls back to ffmpeg's concat demuxer with RF64 output (WAV spec's
    32-bit size field can't represent ≥4 GB)."""
    n_channels = 2
    loaded: List[bytes] = []
    valid_paths: List[Path] = []
    for c in cues:
        wav = c.get("wav")
        if not wav or not wav.exists():
            continue
        pcm = _read_wav_pcm(wav)
        if pcm:
            loaded.append(pcm)
            valid_paths.append(Path(wav))
    if not loaded:
        raise RuntimeError("No TTS cues produced valid audio")

    estimated_size = sum(len(p) for p in loaded)
    MAX_WAV_SIZE = (2 ** 32) - 128

    if estimated_size > MAX_WAV_SIZE:
        print(f"[SrtDub] Timeline > 4 GB ({estimated_size/1e9:.2f} GB) — "
              f"using ffmpeg concat with RF64", flush=True)
        del loaded
        list_path = out_wav.with_suffix(".list.txt")
        with open(list_path, "w", encoding="utf-8") as lf:
            for p in valid_paths:
                lf.write(f"file '{p.as_posix()}'\n")
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-c:a", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(n_channels),
             "-rf64", "auto",
             str(out_wav)],
            check=True, capture_output=True, timeout=3600)
        list_path.unlink(missing_ok=True)
        return _get_duration(out_wav, ffmpeg)

    timeline = b"".join(loaded)  # zero gap — back-to-back
    total_samples = len(timeline) // (n_channels * 2)
    data_size = len(timeline)
    with open(out_wav, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', n_channels))
        f.write(struct.pack('<I', sample_rate))
        bits = 16
        block = n_channels * (bits // 8)
        f.write(struct.pack('<I', sample_rate * block))
        f.write(struct.pack('<H', block))
        f.write(struct.pack('<H', bits))
        f.write(b"data")
        f.write(struct.pack('<I', data_size))
        f.write(timeline)
    return total_samples / sample_rate


# ── Video stretch + freeze extension ────────────────────────────────────

def _get_duration(path: Path, ffmpeg: str = "ffmpeg") -> float:
    try:
        ffprobe = ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in ffmpeg else "ffprobe"
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip() or 0)
    except Exception:
        return 0.0


def _stretch_and_extend(video_in: Path, target_dur: float, max_stretch: float,
                         out_path: Path, ffmpeg: str,
                         vx_filter: str = "") -> None:
    """Adapt the video so its duration equals ``target_dur`` (the final
    audio length). Three regimes:

      1. audio > video           → stretch via ``setpts`` (up to ``max_stretch``),
                                   then freeze-pad last frame if still short.
      2. audio == video (±50ms)  → stream-copy if no vx_filter, else re-encode.
      3. audio < video           → trim video to ``target_dur`` via ``-t``.

    ``vx_filter`` is an ffmpeg -vf chain string (e.g. ``"hflip,hue=h=5"``).
    When provided, ALL regimes re-encode to apply the visual transform.

    Audio is NEVER trimmed. The original ``video_in`` file is NEVER moved
    or modified — we always write to ``out_path``.
    """
    if target_dur <= 0:
        raise RuntimeError(f"Invalid target duration: {target_dur:.3f}s")
    orig = _get_duration(video_in, ffmpeg)
    if orig <= 0:
        raise RuntimeError("Cannot probe source video duration")

    out_path.unlink(missing_ok=True)
    FFMPEG_TIMEOUT = 3600
    has_vx = bool(vx_filter)

    def _x264(extra_vf: str = "") -> list:
        """Build the common libx264 encoder args with a combined vf chain."""
        vf_parts = [p for p in (vx_filter, extra_vf) if p]
        vf_str = ",".join(vf_parts)
        args = []
        if vf_str:
            args += ["-vf", vf_str]
        args += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                 "-pix_fmt", "yuv420p", "-an"]
        return args

    ratio = target_dur / orig

    # Regime 2: near-exact match
    if abs(ratio - 1.0) < 0.01:
        if has_vx:
            # Re-encode to apply visual transforms even when no stretch needed
            subprocess.run(
                [ffmpeg, "-y", "-i", str(video_in), *_x264(), str(out_path)],
                check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
        else:
            # Stream-copy (fast, lossless)
            subprocess.run(
                [ffmpeg, "-y", "-i", str(video_in), "-c", "copy", "-an", str(out_path)],
                check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
        return

    # Regime 3: audio SHORTER than video → trim.
    # Always re-encode: `-c:v copy` + `-t` trims at the nearest keyframe,
    # which can overshoot by 100-500ms and cause A/V desync at the tail.
    # Re-encoding lets us cut exactly at target_dur on any frame boundary.
    if ratio < 1.0:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video_in),
             "-t", f"{target_dur:.3f}",
             *_x264(),
             str(out_path)],
            check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
        return

    # Regime 1: audio LONGER than video → stretch (setpts) + optional freeze-pad
    apply_ratio = min(ratio, max_stretch)
    tmp_slow = out_path.with_name(out_path.stem + "_slow.mp4")
    tmp_slow.unlink(missing_ok=True)
    # setpts MUST be last in the chain so zoom/crop/hflip apply before timing change
    subprocess.run(
        [ffmpeg, "-y", "-i", str(video_in),
         *_x264(f"setpts={apply_ratio:.4f}*PTS"),
         str(tmp_slow)],
        check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)

    slow_dur = _get_duration(tmp_slow, ffmpeg)
    if slow_dur <= 0:
        raise RuntimeError("Stretched video probe returned 0 duration")
    if slow_dur + 0.05 < target_dur:
        # Still short after max stretch — freeze-pad the tail.
        # vx_filter already applied during stretch, so tpad only on the tail;
        # do NOT re-apply vx_filter here or it compounds.
        pad = target_dur - slow_dur
        subprocess.run(
            [ffmpeg, "-y", "-i", str(tmp_slow),
             "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f}",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-an",
             str(out_path)],
            check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
        tmp_slow.unlink(missing_ok=True)
    else:
        tmp_slow.rename(out_path)


# ── Public entry point ─────────────────────────────────────────────────

def _apply_atempo(in_wav: Path, out_wav: Path, speed: float, ffmpeg: str) -> None:
    """Apply ffmpeg atempo filter to speed/slow audio without pitch shift.
    atempo supports 0.5..100.0 per instance; for bigger ranges chain multiple.
    """
    # Guard: non-positive speed would cause an infinite loop in the chain
    # builder below (the "< 0.5" loop would never converge on speed <= 0).
    if not speed or speed <= 0:
        raise ValueError(f"atempo speed must be > 0, got {speed}")
    if abs(speed - 1.0) < 0.01:
        # No change — just copy
        import shutil as _sh
        _sh.copy2(in_wav, out_wav)
        return
    # Build atempo chain (handles speeds outside 0.5..2.0 by chaining)
    chain = []
    remaining = speed
    while remaining > 2.0:
        chain.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        chain.append("atempo=0.5")
        remaining /= 0.5
    chain.append(f"atempo={remaining:.4f}")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(in_wav),
         "-filter:a", ",".join(chain),
         "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
         str(out_wav)],
        check=True, capture_output=True, timeout=900)


def _build_vx_filter(vx_hflip: bool, vx_hue: float, vx_zoom: float) -> str:
    """Build a video-filter chain for visual transforms that break Content ID.
    Returns a comma-separated ffmpeg -vf string, or "" if all transforms off.

    Dimensions are forced even via ``trunc(X/2)*2`` — libx264 with yuv420p
    requires even width + height. Otherwise encoding fails on sources whose
    dimensions don't divide cleanly by the zoom factor (e.g. 1366×768).
    """
    parts = []
    if vx_hflip:
        parts.append("hflip")
    if vx_hue and abs(vx_hue) > 0.01:
        parts.append(f"hue=h={vx_hue:.2f}")
    if vx_zoom and abs(vx_zoom - 1.0) > 0.001:
        # Zoom in by vx_zoom, then crop back to (original size, rounded even).
        # Force even dims on both scale and crop to keep libx264 happy.
        parts.append(
            f"scale=trunc(iw*{vx_zoom:.4f}/2)*2:trunc(ih*{vx_zoom:.4f}/2)*2")
        parts.append(
            f"crop=trunc(iw/{vx_zoom:.4f}/2)*2:trunc(ih/{vx_zoom:.4f}/2)*2")
    return ",".join(parts)


def run_srtdub(
    source_url: str,
    srt_content: str,
    work_dir: Path,
    output_path: Path,
    tts_voice: str = "hi-IN-SwaraNeural",
    tts_rate: str = "+0%",
    audio_bitrate: str = "320k",
    max_stretch: float = 10.0,
    # Post-TTS audio speedup via atempo (1.0 = unchanged, 1.25 = 25% faster)
    audio_speed: float = 1.25,
    # Visual transforms (Content-ID evasion)
    vx_hflip: bool = True,      # horizontal mirror
    vx_hue: float = 5.0,        # hue shift degrees (-30..+30)
    vx_zoom: float = 1.05,      # zoom+crop (1.0 = off, 1.05 = 5% zoom)
    on_progress: Callable = None,
    cancel_check: Callable = None,
    ffmpeg: str = "ffmpeg",
    ytdlp: str = "yt-dlp",
) -> Path:
    progress = on_progress or (lambda *_: None)
    is_cancelled = cancel_check or (lambda: False)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate inputs up-front with clear messages
    if not source_url or not source_url.strip():
        raise RuntimeError("source_url is required (YouTube URL)")
    if not srt_content or not srt_content.strip():
        raise RuntimeError(
            "SRT content is empty — paste your translated SRT into "
            "'Translated SRT content' (or upload a .srt file).")
    if not tts_voice:
        raise RuntimeError("tts_voice is required")

    max_stretch = max(1.0, min(20.0, float(max_stretch or 20.0)))
    # Clamp audio_speed — atempo technically supports 0.5–100, but 4× is
    # the practical ceiling before speech becomes unintelligible.
    audio_speed = max(0.5, min(4.0, float(audio_speed or 1.0)))
    # Clamp visual transform params
    vx_hue = max(-180.0, min(180.0, float(vx_hue or 0.0)))
    vx_zoom = max(1.0, min(1.5, float(vx_zoom or 1.0)))

    ffmpeg = _resolve_executable(ffmpeg) if ffmpeg == "ffmpeg" else ffmpeg
    ytdlp = _resolve_executable(ytdlp) if ytdlp == "yt-dlp" else ytdlp
    print(f"[SrtDub] Using ffmpeg={ffmpeg}, yt-dlp={ytdlp}", flush=True)
    print(f"[SrtDub] audio_speed={audio_speed:.2f}× "
          f"visual: hflip={vx_hflip} hue={vx_hue:.1f}° zoom={vx_zoom:.3f}×",
          flush=True)

    def _raise_if_cancelled():
        if is_cancelled():
            raise RuntimeError("Cancelled by user")

    try:
        # 1. Parse SRT
        progress("transcribe", 0.0, "[0%] Parsing SRT...")
        cues = _parse_srt(srt_content)
        if not cues:
            raise RuntimeError(
                "No cues parsed from the SRT — check the format. "
                "Expected: lines of '00:00:01,000 --> 00:00:04,000' followed "
                "by the subtitle text.")
        LONG = 500
        long_count = sum(1 for c in cues if len(c["text"]) > LONG)
        if long_count:
            print(f"[SrtDub] WARNING: {long_count}/{len(cues)} cues exceed "
                  f"{LONG} chars — Edge-TTS may truncate these", flush=True)
        progress("transcribe", 1.0, f"[100%] Parsed {len(cues)} SRT cues")
        _raise_if_cancelled()

        # 2. Download video
        progress("download", 0.0, "[0%] Downloading video...")
        video_path = work_dir / "video.mp4"
        _download_video(source_url, video_path, ytdlp)
        progress("download", 1.0, "[100%] Video downloaded")
        _raise_if_cancelled()

        # 3. TTS each cue verbatim
        progress("synthesize", 0.0, f"[0%] Edge-TTS on {len(cues)} cues...")
        cues = _tts_all_cues(cues, work_dir, tts_voice, tts_rate, ffmpeg, progress, is_cancelled)
        cues_with_audio = [c for c in cues if c.get("wav")]
        if not cues_with_audio:
            raise RuntimeError("All TTS attempts failed — no audio produced")
        progress("synthesize", 1.0,
                 f"[100%] {len(cues_with_audio)}/{len(cues)} TTS clips ready")
        _raise_if_cancelled()

        # 4. Concat audio with zero gap
        progress("assemble", 0.0, "[0%] Concatenating audio (zero gap)...")
        timeline_wav = work_dir / "sd_timeline.wav"
        audio_dur = _concat_wavs_zero_gap(cues_with_audio, timeline_wav, ffmpeg=ffmpeg)
        progress("assemble", 0.2, f"[20%] Audio (natural): {audio_dur:.1f}s")
        _raise_if_cancelled()

        # 4b. Apply post-TTS audio speedup (atempo) — default 1.25×
        if abs(audio_speed - 1.0) > 0.01:
            progress("assemble", 0.25,
                     f"[25%] Speeding audio up to {audio_speed:.2f}× (atempo)...")
            sped_wav = work_dir / "sd_timeline_sped.wav"
            _apply_atempo(timeline_wav, sped_wav, audio_speed, ffmpeg)
            timeline_wav = sped_wav
            # Probe the actual sped WAV for accurate duration — atempo can
            # introduce tiny rounding vs. the mathematical dur/speed estimate.
            probed = _get_duration(timeline_wav, ffmpeg)
            if probed > 0:
                audio_dur = probed
            else:
                audio_dur = audio_dur / audio_speed
            progress("assemble", 0.3,
                     f"[30%] Audio after {audio_speed:.2f}× speedup: {audio_dur:.1f}s")
        _raise_if_cancelled()

        # 5. Stretch video (cap at max_stretch) + freeze-pad if needed,
        # plus visual transforms (hflip / hue / zoom) to break Content ID
        vx_filter = _build_vx_filter(vx_hflip, vx_hue, vx_zoom)
        if vx_filter:
            print(f"[SrtDub] Visual filter chain: {vx_filter}", flush=True)
        progress("assemble", 0.4,
                 f"[40%] Stretching video (cap {max_stretch}×) + visual transforms...")
        adapted_video = work_dir / "sd_video_adapted.mp4"
        _stretch_and_extend(video_path, audio_dur, max_stretch, adapted_video,
                             ffmpeg, vx_filter=vx_filter)
        if not adapted_video.exists() or adapted_video.stat().st_size < 1024:
            raise RuntimeError("Video adaptation produced no file or an empty file")
        adapted_dur = _get_duration(adapted_video, ffmpeg)
        if adapted_dur <= 0:
            raise RuntimeError("Adapted video has unreadable duration")
        progress("assemble", 0.7,
                 f"[70%] Video: {adapted_dur:.1f}s (audio: {audio_dur:.1f}s)")
        _raise_if_cancelled()

        # 6. Mux — video is already adapted to target_dur. Add -shortest so
        # any last-frame duration rounding from setpts/trim/atempo doesn't
        # leave a silent audio tail or frozen video tail. The pre-mux
        # adjustment has already sized them within a few ms of each other.
        output_path.unlink(missing_ok=True)
        mux_proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(adapted_video), "-i", str(timeline_wav),
             "-c:v", "copy", "-c:a", "aac", "-b:a", audio_bitrate or "320k",
             "-map", "0:v:0", "-map", "1:a:0",
             "-shortest",
             "-movflags", "+faststart",
             str(output_path)],
            capture_output=True, timeout=1800,
            text=True, encoding="utf-8", errors="replace")
        if (mux_proc.returncode != 0
                or not output_path.exists()
                or output_path.stat().st_size < 1024):
            err_tail = (mux_proc.stderr or "").strip().splitlines()[-5:]
            raise RuntimeError(
                "Final mux failed — " + ("; ".join(err_tail) if err_tail else
                f"rc={mux_proc.returncode}, file missing or empty"))

        progress("assemble", 1.0, "[100%] Done!")
        return output_path

    finally:
        # Always clean up TTS intermediates, even on exception / cancel
        for pat in ["sd_tts_*.wav", "sd_tts_*.mp3", "sd_video_adapted_slow.mp4", "sd_timeline_sped.wav"]:
            for f in work_dir.glob(pat):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
