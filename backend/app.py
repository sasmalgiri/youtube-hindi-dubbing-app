"""
YouTube Hindi Dubbing API
=========================
FastAPI server with SSE progress, YouTube URL input, and translation support.
"""
from __future__ import annotations

import os
# Fix Windows console encoding for Hindi/Devanagari text
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from pathlib import Path as _Path
# Load .env file for GEMINI_API_KEY etc.
_env_file = _Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import asyncio
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
from sse_starlette.sse import EventSourceResponse

from pipeline import Pipeline, PipelineConfig, list_voices, DEFAULT_VOICES
from metrics import get_metrics
from jobstore import JobStore

# ── Crash Reporter (active under any launch path) ───────────────────────────
# Generates a crash report for every backend run. Captures:
#   1. C-level crashes (segfault, SIGABRT) via faulthandler
#   2. Uncaught main-thread exceptions via sys.excepthook
#   3. Uncaught worker-thread exceptions via threading.excepthook
#   4. Per-job pipeline failures via _crash_dump_job() called from _run_job
#   5. Clean / dirty exit marker via atexit
#
# Each backend launch gets its own file: backend/logs/crashes/run_<TS>_pid<PID>.log
# Active even when launched via `python -m uvicorn app:app` — unlike the file
# logger inside __main__ which only attaches under `python app.py`.
import atexit as _atexit
import faulthandler as _faulthandler
import datetime as _crash_dt
import traceback as _crash_tb

_CRASH_DIR = _Path(__file__).resolve().parent / "logs" / "crashes"
try:
    _CRASH_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

_RUN_ID  = _crash_dt.datetime.now().strftime("%Y%m%d_%H%M%S")
_RUN_PID = os.getpid()
_CRASH_LOG = _CRASH_DIR / f"run_{_RUN_ID}_pid{_RUN_PID}.log"
_FAULT_LOG = _CRASH_DIR / f"run_{_RUN_ID}_pid{_RUN_PID}.faulthandler"

# Track whether we exited cleanly so atexit can label the marker
_clean_exit = {"value": False}

def _crash_write(label: str, body: str) -> None:
    """Append a labeled section to this run's crash log. Never raises."""
    try:
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n========= {label} @ {_crash_dt.datetime.now().isoformat()} =========\n")
            f.write(body)
            if not body.endswith("\n"):
                f.write("\n")
    except Exception:
        pass  # crash reporter must never crash

def _crash_dump_job(job, exc: BaseException) -> None:
    """Called from _run_job's except block. Dumps full job context + traceback."""
    try:
        body  = f"Job ID:        {getattr(job, 'id', '?')}\n"
        body += f"Source URL:    {getattr(job, 'source_url', '?')}\n"
        body += f"Current step:  {getattr(job, 'current_step', '?')}\n"
        body += f"Step progress: {getattr(job, 'step_progress', '?')}\n"
        body += f"Last message:  {getattr(job, 'message', '?')}\n"
        body += f"Target lang:   {getattr(job, 'target_language', '?')}\n"
        body += f"Step times:    {getattr(job, 'step_times', {})}\n"
        req = getattr(job, "original_req", None)
        if req is not None:
            try:
                body += f"\nRequest:\n{req.dict() if hasattr(req,'dict') else req}\n"
            except Exception:
                body += f"\nRequest: <unserializable>\n"
        body += f"\nException type: {type(exc).__name__}\n"
        body += f"Exception args: {exc.args}\n"
        body += "\nTraceback:\n"
        body += "".join(_crash_tb.format_exception(type(exc), exc, exc.__traceback__))
        _crash_write(f"JOB CRASH {getattr(job, 'id', '?')}", body)
    except Exception:
        pass

# ── 1. Faulthandler — C-level crash (segfault, SIGABRT) traceback dumps ──
try:
    _fh_file = open(_FAULT_LOG, "a", encoding="utf-8", buffering=1)
    _faulthandler.enable(file=_fh_file, all_threads=True)
except Exception as _e:
    print(f"[CRASH-REPORTER] faulthandler enable failed: {_e}", flush=True)

# ── 2. Uncaught main-thread exceptions ──
def _main_excepthook(exc_type, exc_value, exc_tb):
    body = "".join(_crash_tb.format_exception(exc_type, exc_value, exc_tb))
    _crash_write("MAIN-THREAD UNCAUGHT", body)
    print(f"[CRASH-REPORTER] Wrote main-thread crash -> {_CRASH_LOG}", flush=True)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _main_excepthook

# ── 3. Uncaught worker-thread exceptions (Python 3.8+) ──
def _thread_excepthook(args):
    body = "".join(_crash_tb.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    body += f"\nThread: {args.thread.name if args.thread else '<unknown>'}\n"
    _crash_write("WORKER-THREAD UNCAUGHT", body)
    print(f"[CRASH-REPORTER] Wrote thread crash -> {_CRASH_LOG}", flush=True)

try:
    threading.excepthook = _thread_excepthook
except AttributeError:
    pass  # Python <3.8

# ── 4. Process exit marker (clean vs dirty) ──
def _atexit_marker():
    label = "PROCESS EXIT (clean)" if _clean_exit["value"] else "PROCESS EXIT (dirty / no clean shutdown)"
    _crash_write(label, f"PID {_RUN_PID}")

_atexit.register(_atexit_marker)

# ── 5. Initial run marker ──
_crash_write("RUN START",
    f"PID:        {_RUN_PID}\n"
    f"Python:     {sys.version.splitlines()[0]}\n"
    f"Platform:   {sys.platform}\n"
    f"CWD:        {os.getcwd()}\n"
    f"Argv:       {sys.argv}\n"
    f"Fault log:  {_FAULT_LOG}\n"
)

print(f"[CRASH-REPORTER] Active. Run log: {_CRASH_LOG}", flush=True)

# ── Hinglish AI Training Hook ────────────────────────────────────────────────
HINGLISH_TRAINER_URL = os.environ.get("HINGLISH_TRAINER_URL", "http://localhost:8100")


def _send_training_data(source_srt_path: Path, translated_srt_path: Path, source_lang: str = "en"):
    """Send SRT pair to hinglish-ai-model trainer (fire-and-forget)."""
    def _send():
        try:
            import urllib.request
            import urllib.error
            boundary = "----HinglishTrainerBoundary"
            parts = []
            for field_name, filepath in [("source_srt", source_srt_path), ("translated_srt", translated_srt_path)]:
                content = filepath.read_bytes()
                parts.append(
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{field_name}"; filename="{filepath.name}"\r\n'
                    f"Content-Type: application/octet-stream\r\n\r\n"
                )
                parts.append(content)
                parts.append(b"\r\n")
            # Add source_language field
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="source_language"\r\n\r\n'
                f"{source_lang}\r\n"
                f"--{boundary}--\r\n"
            )
            body = b""
            for p in parts:
                body += p.encode("utf-8") if isinstance(p, str) else p

            req = urllib.request.Request(
                f"{HINGLISH_TRAINER_URL}/api/upload-translated-srt",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            result = json.loads(resp.read())
            print(f"[TRAINING] Sent {result.get('extracted', 0)} pairs to trainer "
                  f"({result.get('total', '?')} total)", flush=True)
        except Exception as e:
            # Silently fail — trainer might not be running
            print(f"[TRAINING] Could not send to trainer: {e}", flush=True)

    threading.Thread(target=_send, daemon=True).start()


# ── Types ────────────────────────────────────────────────────────────────────

JobState = Literal["queued", "running", "done", "error", "waiting_for_srt", "review_transcription", "review_translation"]


@dataclass
class Job:
    id: str
    state: JobState = "queued"
    current_step: str = ""
    step_progress: float = 0.0
    overall_progress: float = 0.0
    message: str = "Queued"
    error: Optional[str] = None
    result_path: Optional[Path] = None
    source_url: str = ""
    video_title: str = ""
    target_language: str = "hi"
    segments: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    events: List[Dict] = field(default_factory=list)
    original_req: Optional[Any] = None  # Store original request for resume
    saved_folder: Optional[str] = None  # Path to titled output folder
    saved_video: Optional[str] = None   # Path to saved video file
    description: Optional[str] = None   # YouTube description
    qa_score: Optional[float] = None    # Transcription QA score (0-1)
    # Word/sentence budget — populated from pipeline._tts_budget after a
    # successful run() so the UI can show "X words across Y sentences"
    total_words: int = 0
    total_sentences: int = 0
    avg_words_per_sent: float = 0.0
    max_seg_words: int = 0
    max_sent_words: int = 0
    step_times: Dict[str, float] = field(default_factory=dict)  # step -> elapsed seconds
    _step_start: float = 0.0  # internal: when current step started
    chain_languages: List[str] = field(default_factory=list)  # Remaining languages in chain
    chain_parent_id: Optional[str] = None  # Parent job ID in chain
    cancel_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    pipeline_ref: Optional[Any] = None  # Reference to Pipeline for step-by-step state
    worker_thread: Optional[Any] = None  # threading.Thread running this job — used by cancel


class JobCreateRequest(BaseModel):
    url: str
    source_language: str = "en"   # English source by default — pipeline is English→Hindi
    target_language: str = "hi"
    voice: str = "hi-IN-SwaraNeural"
    asr_model: str = "groq-whisper"
    translation_engine: str = "google"
    tts_rate: str = "+0%"
    mix_original: bool = False
    original_volume: float = 0.10
    use_cosyvoice: bool = False
    use_chatterbox: bool = False
    use_fish_speech: bool = False
    use_indic_parler: bool = False
    use_sarvam_bulbul: bool = False
    use_elevenlabs: bool = False
    use_google_tts: bool = False
    use_coqui_xtts: bool = False
    use_edge_tts: bool = True
    prefer_youtube_subs: bool = True
    use_yt_translate: bool = True   # ON: try YouTube Hindi auto-translate first (best quality)
    multi_speaker: bool = False
    transcribe_only: bool = False
    audio_priority: bool = True
    audio_untouchable: bool = False  # TTS output is NEVER modified — no trim, no norm, no speed change
    video_slow_to_match: bool = True  # Slow video uniformly to match longer dubbed audio
    post_tts_level: str = "minimal"    # "minimal" | "full" | "none"
    audio_quality_mode: str = "fast"  # "fast" (ffmpeg) | "quality" (librosa)
    enable_sentence_gap: bool = False  # OFF: no artificial gaps, continuous audio
    enable_duration_fit: bool = False  # OFF: assembly split-the-diff handles timing
    audio_bitrate: str = "192k"
    encode_preset: str = "fast"
    # Download mode: "remux" (current default — instant container swap, may fail on
    # exotic codecs) OR "encode" (old behavior — re-encodes via ffmpeg merge,
    # slower but always works regardless of source codec compatibility)
    download_mode: str = "remux"  # "remux" | "encode"
    split_duration: int = 30
    dub_duration: int = 0
    fast_assemble: bool = False
    dub_chain: List[str] = []
    enable_manual_review: bool = False
    use_whisperx: bool = True          # WhisperX forced alignment for tighter word timestamps (on by default; whisperx installed)
    simplify_english: bool = False     # OFF: translation 35% word cap handles it
    enable_tts_verify_retry: bool = False  # OFF: 70% false-positive rate on Hindi turns 5s cleanup into 2h bottleneck
    # Inline TTS truncation guard: catches Edge-TTS WebSocket drops that
    # silently save partial audio. Range 0.0-1.0:
    #   0.00 = OFF, 0.30 = default (catastrophic only), 0.70 = aggressive
    tts_truncation_threshold: float = 0.30
    # Keep noun subjects of every sentence in the source language. Pronoun
    # subjects (he, she, it, they, ...) are still translated normally.
    # Requires spaCy + en_core_web_sm. Off by default.
    keep_subject_english: bool = False
    # Post-TTS exact word-count verification: transcribe each WAV with
    # Whisper and re-run TTS for any segment outside the tolerance.
    # ON by default — exact per-segment word match with retry on mismatch.
    # Model: "auto" picks turbo on GPU, tiny on CPU.
    tts_word_match_verify: bool = True
    tts_word_match_tolerance: float = 0.15
    tts_word_match_model: str = "auto"  # "auto" | "tiny" | "turbo"
    tts_word_match_max_segments: int = 1000  # auto-off above N; 0 = no cap
    # Wav2Lip lip-sync post-process. Requires backend/wav2lip/ + checkpoint + GPU.
    use_wav2lip: bool = False
    # Long-segment trace watchdog: records full lifecycle of every long
    # segment to a JSON report. Cheap, on by default.
    long_segment_trace: bool = True
    long_segment_threshold_words: int = 15
    # No time pressure on TTS: TTS gets every word, post-processing handles slots.
    tts_no_time_pressure: bool = True
    # Dynamic worker scaling — adapts to Edge-TTS rate limits between batches.
    tts_dynamic_workers: bool = True
    tts_dynamic_min: int = 10
    tts_dynamic_max: int = 120
    tts_dynamic_start: int = 30
    # Auto TTS rate mode: match source video duration automatically.
    # "auto" (default) = compute tts_rate from word count + source duration.
    # "manual"         = use the Speech Rate slider value as-is.
    tts_rate_mode: str = "auto"
    tts_rate_ceiling: str = "+25%"
    tts_rate_target_wpm: int = 130
    purge_on_new_url: bool = False     # When True: delete prior job's work_dir + caches when a different URL is submitted
    step_by_step: bool = False         # Pause after transcription & translation for review
    use_new_pipeline: bool = False     # Use new modular pipeline (experimental)
    pipeline_mode: str = "classic"     # "classic" | "hybrid" | "new" | "oneflow" | "wordchunk" | "srtdub"
    # ── WordChunk mode options ──
    wc_chunk_size: int = 8              # 4 | 8 | 12 words per TTS chunk
    wc_max_stretch: float = 20.0        # 1.0–20.0× max video slowdown
    wc_transcript: str = ""             # Optional user-pasted transcript — bypasses YouTube subs fetch
    # ── SRT Direct mode options ──
    sd_srt_content: str = ""            # Full SRT content (cues verbatim) — required for srtdub mode
    sd_max_stretch: float = 20.0        # 1.0–20.0× max video slowdown; freeze-pads if still short
    sd_audio_speed: float = 1.25        # Post-TTS atempo multiplier (1.0 = unchanged, 1.25 = 25% faster)
    sd_vx_hflip: bool = True            # Horizontal mirror (Content-ID evasion)
    sd_vx_hue: float = 5.0              # Hue shift in degrees (-30..+30)
    sd_vx_zoom: float = 1.05            # Zoom then crop back (1.0 = off, 1.05 = 5% zoom)
    srt_needs_translation: bool = False  # True = English SRT needs translation before TTS
    transcript_srt_content: str = ""     # Paste/upload English SRT to skip transcription → go straight to translation
    preset_name: str = ""                 # Name of preset used (for display only)
    # ── AV Sync Modules ──
    av_sync_mode: str = "original"       # "original" | "capped" | "audio_first"
    max_audio_speedup: float = 1.30      # cap for "capped" mode
    min_video_speed: float = 0.70        # floor before flagging
    slot_verify: str = "off"             # "off" | "dry_run" | "auto_fix"
    # ── Global Stretch ──
    use_global_stretch: bool = False     # uniform video slowdown
    global_stretch_speedup: float = 1.25 # TTS speedup for global stretch
    # ── Segmenter ──
    segmenter: str = "dp"                # "dp" | "sentence"
    segmenter_buffer_pct: float = 0.20   # Hindi expansion buffer
    max_sentences_per_cue: int = 2       # max sentences per segment
    # ── YouTube Transcript Mode ──
    # How YouTube subs are structured before feeding to the proven pipeline:
    #   "yt_timeline"      — Option 1: YouTube text + YouTube's own timelines.
    #                        Merge into sentences, group 2 per segment, redistribute
    #                        slots proportionally by word count. Fast (no Whisper).
    #   "whisper_timeline" — Option 2: YouTube text + Whisper timestamps.
    #                        Run Whisper for precise speech timelines, replace its
    #                        text with YouTube's (better quality). Slower but exact.
    yt_transcript_mode: str = "yt_timeline"
    # Segment split mode for YouTube subs:
    #   "sentence"  — group 2 complete sentences per segment (needs punctuation)
    #   "wordcount" — split by ~20 words per segment (uniform, no punctuation needed)
    yt_segment_mode: str = "sentence"
    # Use YouTube subs as reference to correct Whisper transcription text.
    # Whisper keeps its precise timestamps, only the TEXT is replaced with
    # YouTube's (higher quality). The proven pipeline flow stays identical.
    yt_text_correction: bool = True
    # How to replace Whisper text with YouTube subs:
    #   "full" — total replacement (all words from YouTube)
    #   "diff" — word-level diff, only swap words that differ (keeps Whisper punctuation)
    yt_replace_mode: str = "diff"
    # TTS chunk size: split translated text into N-word chunks before TTS.
    # 0 = off (use full segments as-is, best prosody).
    # 4/8/12 = chunk size (smaller = no truncation but choppier sound).
    tts_chunk_words: int = 0
    # Gap mode between segments in assembly:
    #   "none"   — 0s, pure back-to-back (shortest output)
    #   "micro"  — 0.2s breathing room (proven default)
    #   "full"   — keep original silence durations (legacy)
    gap_mode: str = "micro"

    @validator("target_language", "source_language")
    def validate_language(cls, v):
        import re
        if not re.match(r"^[a-zA-Z]{2,5}(-[a-zA-Z]{2,5})?$|^auto$", v):
            raise ValueError(f"Invalid language code: {v}")
        return v


# ── Step weights for overall progress ────────────────────────────────────────

STEP_ORDER = ["download", "extract", "transcribe", "translate", "synthesize", "assemble"]
STEP_WEIGHTS = {
    "download": 0.15,
    "extract": 0.05,
    "transcribe": 0.25,
    "translate": 0.15,
    "synthesize": 0.30,
    "assemble": 0.10,
}

# ── Storage ──────────────────────────────────────────────────────────────────

JOBS: Dict[str, Job] = {}
MAX_JOBS = 200
# Only run one pipeline at a time to avoid resource contention
_pipeline_semaphore = threading.Semaphore(1)
BASE_DIR = Path(__file__).resolve().parent
# Use a short temp path on Windows to avoid 260-char path limit (WinError 206)
if os.name == "nt":
    _short_root = Path(os.environ.get("VOICEDUB_WORK", "C:/tmp/vd"))
    try:
        _short_root.mkdir(parents=True, exist_ok=True)
        WORK_ROOT = _short_root
    except (PermissionError, OSError):
        print(f"[WARN] Cannot create {_short_root}, falling back to local work dir", flush=True)
        WORK_ROOT = BASE_DIR / "work"
else:
    WORK_ROOT = BASE_DIR / "work"
OUTPUTS = WORK_ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Final saved dubbed videos go here, organized by title
# Use D: drive folder if available (user's preferred location), else fallback to local
_preferred_save = Path("D:/Shirshendu sasmal/youtube dubbed")
SAVED_DIR = _preferred_save if _preferred_save.exists() else BASE_DIR / "dubbed_outputs"
SAVED_DIR.mkdir(parents=True, exist_ok=True)

# ── Job persistence: SQLite ─────────────────────────────────────────────────
# Single local SQLite store. Simple, fast, no network dependency. The previous
# Supabase secondary writer was removed because it added no value for a
# single-machine workflow and the supabase package's websockets dependency
# was broken on this Python install.
_store = JobStore(BASE_DIR / "jobs.db")
_store.load_all(JOBS)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="YouTube Hindi Dubbing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (needed for Colab ngrok tunnel)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output files for video playback
OUTPUTS.mkdir(parents=True, exist_ok=True)
app.mount("/static/jobs", StaticFiles(directory=str(OUTPUTS)), name="job-files")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _calc_overall(step: str, step_progress: float) -> float:
    """Calculate overall progress from current step and its progress."""
    overall = 0.0
    for s in STEP_ORDER:
        if s == step:
            overall += STEP_WEIGHTS.get(s, 0) * step_progress
            break
        overall += STEP_WEIGHTS.get(s, 0)
    return min(overall, 1.0)


def _sanitize_filename(name: str) -> str:
    """Convert a video title to a safe folder/file name."""
    # Remove null bytes and control characters
    name = re.sub(r'[\x00-\x1f\x7f]', '', name)
    # Remove or replace characters unsafe on Windows/Linux/Mac
    name = re.sub(r'[<>:"/\\|?*#%&{}$!\'`@^+= ,;]', ' ', name)
    # Remove leading/trailing dots and spaces (Windows issue)
    name = name.strip('. ')
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    # Remove emoji and non-BMP characters that cause filesystem issues
    name = re.sub(r'[^\x20-\x7E\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]', '', name)
    # Truncate to avoid filesystem limits (keep at word boundary)
    if len(name) > 100:
        name = name[:100].rsplit(' ', 1)[0]
    # Avoid reserved Windows names
    reserved = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3'}
    if name.upper().split('.')[0] in reserved:
        name = f"_{name}"
    return name or "Untitled"



def _save_to_titled_folder(job: Job):
    """Move dubbed video + SRT to a titled folder in dubbed_outputs/, then delete work dir."""
    if not job.result_path or not job.result_path.exists():
        return

    title = _sanitize_filename(job.video_title or "Untitled")
    lang = job.target_language or "hi"

    # Create folder: dubbed_outputs/<Title> [LANG Dubbed] (job_id)
    folder_name = f"{title} [{lang.upper()} Dubbed] ({job.id})"
    folder = SAVED_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    # Move video with title as filename
    video_name = f"{title} - {lang.upper()} Dubbed.mp4"
    saved_video = folder / video_name
    shutil.move(str(job.result_path), str(saved_video))

    # Move SRT if available
    srt_src = job.result_path.parent / f"subtitles_{lang}.srt"
    if srt_src.exists():
        srt_name = f"{title} - {lang.upper()} Dubbed.srt"
        shutil.move(str(srt_src), str(folder / srt_name))

    # Move manual review queue if present (must happen before job_dir cleanup)
    mrq_src = OUTPUTS / job.id / "manual_review_queue.json"
    if mrq_src.exists():
        shutil.move(str(mrq_src), str(folder / "manual_review_queue.json"))

    # Update job to point to new location
    job.result_path = saved_video
    job.saved_folder = str(folder)
    job.saved_video = str(saved_video)

    # Delete the entire work/outputs/<job_id> directory to free space
    job_work_dir = OUTPUTS / job.id
    if job_work_dir.exists():
        shutil.rmtree(job_work_dir, ignore_errors=True)
        print(f"[CLEANUP] Deleted work directory {job_work_dir}")


def _generate_youtube_description(job: Job) -> str:
    """Generate a 10-line YouTube summary description using Groq or fallback."""
    title = job.video_title or "Untitled"
    lang_names = {
        "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
        "mr": "Marathi", "es": "Spanish", "fr": "French", "de": "German",
        "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "pt": "Portuguese",
    }
    lang_name = lang_names.get(job.target_language, job.target_language)

    # Collect translated text from segments for context
    translated_texts = []
    for seg in (job.segments or [])[:20]:
        t = seg.get("text_translated") or seg.get("text", "")
        if t.strip():
            translated_texts.append(t.strip())
    context = " ".join(translated_texts[:15])

    prompt = (
        f"You are a YouTube description writer. Write a compelling 10-line YouTube video description "
        f"for a video titled \"{title}\" that has been dubbed into {lang_name}.\n\n"
        f"Content context from the video:\n{context[:1500]}\n\n"
        f"Rules:\n"
        f"- Line 1: Hook/attention-grabbing summary of the video\n"
        f"- Lines 2-4: Brief synopsis of what happens in the video\n"
        f"- Line 5: Mention it's dubbed in {lang_name}\n"
        f"- Lines 6-8: Relevant hashtags and keywords\n"
        f"- Line 9: Call to action (like, subscribe, share)\n"
        f"- Line 10: Credits/disclaimer about AI dubbing\n"
        f"- Write in English\n"
        f"- Each line should be a separate paragraph\n"
        f"- Keep it engaging and SEO-friendly\n"
        f"- Output ONLY the description, no extra text"
    )

    # Try Groq first (free, fast)
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            import requests
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": "You are a professional YouTube description writer."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                desc = resp.json()["choices"][0]["message"]["content"].strip()
                return desc
        except Exception:
            pass

    # Fallback: generate a simple template description
    return (
        f"{title} - {lang_name} Dubbed Version\n\n"
        f"Watch this amazing video now dubbed in {lang_name}!\n\n"
        f"This video has been professionally dubbed using AI voice technology "
        f"to bring you the best viewing experience in {lang_name}.\n\n"
        f"Original content translated and voiced with natural-sounding {lang_name} narration.\n\n"
        f"#{lang_name}Dubbed #{lang_name} #AIDubbing #HindiDubbing #YouTubeDubbing\n\n"
        f"If you enjoyed this dubbed version, please Like, Subscribe, and Share!\n\n"
        f"Turn on notifications to never miss a new dubbed video.\n\n"
        f"This video was dubbed using AI voice technology. "
        f"Original content belongs to the respective creators."
    )


def _get_video_duration(ffmpeg_path: str, video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    ffprobe = str(Path(ffmpeg_path).parent / "ffprobe") if Path(ffmpeg_path).is_absolute() else "ffprobe"
    if sys.platform == "win32" and not ffprobe.endswith(".exe"):
        ffprobe += ".exe"
    try:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _split_video(ffmpeg_path: str, video_path: Path, split_mins: int, output_dir: Path) -> List[Path]:
    """Split a video into parts of split_mins duration each. Returns list of part paths."""
    duration = _get_video_duration(ffmpeg_path, video_path)
    if duration <= 0:
        return [video_path]

    split_secs = split_mins * 60
    num_parts = math.ceil(duration / split_secs)

    if num_parts <= 1:
        return [video_path]

    parts = []
    for i in range(num_parts):
        start = i * split_secs
        part_path = output_dir / f"part_{i+1:02d}.mp4"
        cmd = [
            ffmpeg_path, "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-t", f"{split_secs:.3f}",
            "-c", "copy",  # stream copy = instant, no re-encode
            "-avoid_negative_ts", "make_zero",
            str(part_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if part_path.exists() and part_path.stat().st_size > 0:
            parts.append(part_path)

    return parts


def _make_progress_callback(job: Job):
    """Create a progress callback that updates the job and appends events."""
    _prev_step = [job.current_step]  # mutable for closure

    def callback(step: str, progress: float, message: str):
        now = time.time()

        # Track step timing: when step changes, record previous step duration
        if step != _prev_step[0]:
            if _prev_step[0] and job._step_start > 0:
                job.step_times[_prev_step[0]] = round(now - job._step_start, 1)
            job._step_start = now
            _prev_step[0] = step

        # First call — init start time
        if job._step_start == 0:
            job._step_start = now

        # Update live elapsed for current step
        job.step_times[step] = round(now - job._step_start, 1)

        job.current_step = step
        job.step_progress = progress
        job.overall_progress = _calc_overall(step, progress)
        job.message = message

        # Update job state for step-by-step pauses
        if job.pipeline_ref and job.pipeline_ref.paused_at:
            if job.pipeline_ref.paused_at == "review_transcription":
                job.state = "review_transcription"
                job.segments = job.pipeline_ref.segments  # Expose segments for review
            elif job.pipeline_ref.paused_at == "review_translation":
                job.state = "review_translation"
                job.segments = job.pipeline_ref.segments

        job.events.append({
            "step": step,
            "progress": round(progress, 3),
            "overall": round(job.overall_progress, 3),
            "message": message,
        })
        # Cap events list to prevent unbounded memory growth
        if len(job.events) > 500:
            job.events = job.events[-500:]
    return callback


def _translate_srt_content(
    srt_content: str,
    source_lang: str = "en",
    target_lang: str = "hi",
    on_progress=None,
) -> str:
    """Translate an SRT string in-place: keep timestamps, translate text lines.

    Uses Google Translate (deep_translator) with 20 parallel workers.
    Returns a new SRT string with translated text.
    """
    import re
    from deep_translator import GoogleTranslator
    from concurrent.futures import ThreadPoolExecutor, as_completed

    progress = on_progress or (lambda *_: None)

    # Parse SRT into blocks: (index_line, timestamp_line, text_lines)
    blocks = []
    raw_blocks = re.split(r"\n\n+", srt_content.strip())
    ts_re = re.compile(r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}")
    for block in raw_blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # Find the timestamp line
        ts_idx = None
        for i, line in enumerate(lines):
            if ts_re.search(line):
                ts_idx = i
                break
        if ts_idx is None:
            continue
        idx_part = "\n".join(lines[:ts_idx]) if ts_idx > 0 else ""
        ts_line = lines[ts_idx]
        text_part = "\n".join(lines[ts_idx + 1:]).strip()
        if text_part:
            blocks.append((idx_part, ts_line, text_part))

    if not blocks:
        return srt_content  # nothing to translate, return as-is

    total = len(blocks)
    translated_texts = [""] * total
    completed = [0]

    def translate_one(idx: int, text: str):
        for attempt in range(3):
            try:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                result = translator.translate(text)
                if result and "<html" not in result.lower():
                    return idx, result
            except Exception:
                pass
            import time; time.sleep(1.0 * (attempt + 1))
        return idx, text  # fallback: keep original

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(translate_one, i, b[2]): i for i, b in enumerate(blocks)}
        for future in as_completed(futures):
            idx, result = future.result()
            translated_texts[idx] = result
            completed[0] += 1
            if completed[0] % 10 == 0 or completed[0] == total:
                progress("translate", completed[0] / total,
                         f"Translated {completed[0]}/{total} cues...")

    # Rebuild SRT
    out_lines = []
    for i, (idx_part, ts_line, _orig_text) in enumerate(blocks):
        if idx_part:
            out_lines.append(idx_part)
        else:
            out_lines.append(str(i + 1))
        out_lines.append(ts_line)
        out_lines.append(translated_texts[i])
        out_lines.append("")  # blank separator

    return "\n".join(out_lines)


def _run_job(job: Job, req: JobCreateRequest):
    """Run the dubbing pipeline in a background thread."""
    # ── Status helper: every line below sets a visible message AND persists ──
    # Without these explicit setup messages the UI sees "Starting..." for the
    # entire 30-90 second gap between job submission and the first pipeline
    # _report() call, which is indistinguishable from a hang. Each call below
    # gives the UI something concrete and proves the worker thread is alive.
    def _setup_status(msg: str, pct: float = 0.0):
        job.current_step = "setup"
        job.step_progress = pct
        job.overall_progress = pct * 0.02   # setup is 0-2% of overall progress
        job.message = msg
        try:
            _store.save(job)
        except Exception:
            pass

    # ── TTS ENGINE LOCK: Edge-TTS only ──
    # Lock to Edge-TTS as the sole engine. All other engines are forced OFF
    # so the user can't accidentally flip a UI toggle and get CosyVoice/etc.
    req.use_edge_tts       = True
    req.use_cosyvoice      = False
    req.use_chatterbox     = False
    req.use_indic_parler   = False
    req.use_sarvam_bulbul  = False
    req.use_elevenlabs     = False
    req.use_google_tts     = False
    req.use_coqui_xtts     = False
    req.use_fish_speech    = False

    # ── VOICE SELECTION: auto per language + user override ──
    # If user explicitly picked a voice in the UI → honor their choice.
    # If voice is empty or the old default (SwaraNeural) → auto-select the
    # best voice for the target language:
    #   Hindi  -> hi-IN-MadhurNeural (user's preferred male voice)
    #   Bengali -> bn-IN-TanishaaNeural
    #   Tamil  -> ta-IN-PallaviNeural
    #   etc.   -> DEFAULT_VOICES[target_language]
    # This way the voice picker UI stays functional: users CAN choose a
    # different voice (e.g., Swara for female Hindi) and it will be honored.
    _PREFERRED_VOICES = {
        "hi": "hi-IN-MadhurNeural",   # user's preferred Hindi male voice
    }
    if not req.voice or req.voice == "hi-IN-SwaraNeural":
        # User didn't explicitly pick — auto-select for target language.
        # Prefer Madhur for Hindi; for other languages use DEFAULT_VOICES.
        req.voice = _PREFERRED_VOICES.get(
            req.target_language,
            DEFAULT_VOICES.get(req.target_language, "hi-IN-MadhurNeural"),
        )

    # ── NO TIME PRESSURE on TTS ──
    # When the master flag is on (default ON per user request), the slot/duration
    # gates are bypassed in assembly and QC. The user can still pick a faster
    # baseline rate via the Speech Rate slider in the UI — that gets passed to
    # Edge-TTS at synthesis time as a native SSML prosody-rate setting (sounds
    # natural at faster rates, not chipmunk).
    #   - audio_priority = True (assembly adapts video to audio, never reverse)
    #   - enable_duration_fit = False (extra belt-and-suspenders)
    # NOTE: tts_rate is NOT forced here. The user's chosen rate (default "+0%",
    # range -50% to +75%) flows through to Edge-TTS unchanged.
    if getattr(req, 'tts_no_time_pressure', True):
        req.audio_priority      = True
        req.enable_duration_fit = False

    job.message = "Waiting in queue..."
    _setup_status("Waiting for pipeline slot...", 0.05)
    _t_start = time.time()   # defined before try so error handler can always use it
    pipeline = None           # defined before try so completion handler can always access it
    _pipeline_semaphore.acquire()
    try:
        job.state = "running"
        _setup_status("Initializing pipeline (Edge-TTS Madhur locked)...", 0.10)

        job_dir = OUTPUTS / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        work_dir = job_dir / "work"
        work_dir.mkdir(exist_ok=True)
        _setup_status("Created working directory...", 0.20)

        out_path = job_dir / "dubbed.mp4"

        # Voice was auto-selected or user-chosen above. Use it as-is.
        voice = req.voice

        # ── SPLIT MODE: Split video into parts and dub each ──────────
        # SKIP split mode for pipelines that have their own single-pass
        # assembly: oneflow, wordchunk, and srtdub all manage video/audio
        # themselves and don't want the classic Pipeline invoked per-part.
        _split_skipped_for = ("oneflow", "wordchunk", "srtdub")
        _current_mode = getattr(req, 'pipeline_mode', 'classic')
        if req.split_duration > 0 and _current_mode not in _split_skipped_for:
            _setup_status(f"Split mode: video will be processed in "
                          f"{req.split_duration}-minute chunks", 0.30)
            _run_job_split(job, req, voice)
            return
        if req.split_duration > 0 and _current_mode in _split_skipped_for:
            print(f"[Route] split_duration={req.split_duration} ignored — "
                  f"pipeline_mode={_current_mode} has its own assembly", flush=True)

        _setup_status("Building pipeline configuration...", 0.40)

        cfg = PipelineConfig(
            source=req.url,
            work_dir=work_dir,
            output_path=out_path,
            source_language=req.source_language,
            target_language=req.target_language,
            asr_model=req.asr_model,
            translation_engine=req.translation_engine,
            tts_voice=voice,
            tts_rate=req.tts_rate,
            mix_original=req.mix_original,
            original_volume=req.original_volume,
            use_cosyvoice=req.use_cosyvoice,
            use_chatterbox=req.use_chatterbox,
            use_indic_parler=req.use_indic_parler,
            use_sarvam_bulbul=req.use_sarvam_bulbul,
            use_elevenlabs=req.use_elevenlabs,
            use_google_tts=req.use_google_tts,
            use_coqui_xtts=req.use_coqui_xtts,
            use_fish_speech=req.use_fish_speech,
            use_edge_tts=req.use_edge_tts,
            prefer_youtube_subs=req.prefer_youtube_subs,
            use_yt_translate=req.use_yt_translate,
            multi_speaker=req.multi_speaker,
            transcribe_only=req.transcribe_only,
            audio_priority=req.audio_priority,
            audio_untouchable=req.audio_untouchable,
            video_slow_to_match=req.video_slow_to_match,
            post_tts_level=req.post_tts_level,
            audio_quality_mode=req.audio_quality_mode,
            enable_sentence_gap=req.enable_sentence_gap,
            enable_duration_fit=req.enable_duration_fit,
            audio_bitrate=req.audio_bitrate,
            encode_preset=req.encode_preset,
            download_mode=getattr(req, 'download_mode', 'remux'),
            split_duration=req.split_duration,
            dub_duration=req.dub_duration,
            fast_assemble=req.fast_assemble,
            enable_manual_review=req.enable_manual_review,
            use_whisperx=req.use_whisperx,
            simplify_english=req.simplify_english,
            step_by_step=req.step_by_step,
            enable_tts_verify_retry=req.enable_tts_verify_retry,
            tts_truncation_threshold=getattr(req, 'tts_truncation_threshold', 0.30),
            keep_subject_english=getattr(req, 'keep_subject_english', False),
            tts_word_match_verify=getattr(req, 'tts_word_match_verify', False),
            tts_word_match_tolerance=getattr(req, 'tts_word_match_tolerance', 0.15),
            tts_word_match_model=getattr(req, 'tts_word_match_model', 'auto'),
            tts_word_match_max_segments=getattr(req, 'tts_word_match_max_segments', 1000),
            use_wav2lip=getattr(req, 'use_wav2lip', False),
            long_segment_trace=getattr(req, 'long_segment_trace', True),
            long_segment_threshold_words=getattr(req, 'long_segment_threshold_words', 15),
            tts_no_time_pressure=getattr(req, 'tts_no_time_pressure', True),
            tts_dynamic_workers=getattr(req, 'tts_dynamic_workers', True),
            tts_dynamic_min=getattr(req, 'tts_dynamic_min', 10),
            tts_dynamic_max=getattr(req, 'tts_dynamic_max', 120),
            tts_dynamic_start=getattr(req, 'tts_dynamic_start', 30),
            tts_rate_mode=getattr(req, 'tts_rate_mode', 'auto'),
            tts_rate_ceiling=getattr(req, 'tts_rate_ceiling', '+25%'),
            tts_rate_target_wpm=getattr(req, 'tts_rate_target_wpm', 130),
            srt_needs_translation=req.srt_needs_translation,
            # ── AV Sync Modules ──
            av_sync_mode=getattr(req, 'av_sync_mode', 'original'),
            max_audio_speedup=getattr(req, 'max_audio_speedup', 1.30),
            min_video_speed=getattr(req, 'min_video_speed', 0.70),
            slot_verify=getattr(req, 'slot_verify', 'off'),
            use_global_stretch=getattr(req, 'use_global_stretch', False),
            global_stretch_speedup=getattr(req, 'global_stretch_speedup', 1.25),
            segmenter=getattr(req, 'segmenter', 'dp'),
            segmenter_buffer_pct=getattr(req, 'segmenter_buffer_pct', 0.20),
            max_sentences_per_cue=getattr(req, 'max_sentences_per_cue', 2),
            yt_transcript_mode=getattr(req, 'yt_transcript_mode', 'yt_timeline'),
            yt_segment_mode=getattr(req, 'yt_segment_mode', 'sentence'),
            yt_text_correction=getattr(req, 'yt_text_correction', True),
            yt_replace_mode=getattr(req, 'yt_replace_mode', 'diff'),
            tts_chunk_words=getattr(req, 'tts_chunk_words', 0),
            gap_mode=getattr(req, 'gap_mode', 'micro'),
        )
        _setup_status("Pipeline configuration ready", 0.60)

        get_metrics().record_job_start(job.id, req.url, {
            "source_language": req.source_language,
            "target_language": req.target_language,
            "tts_engine": (
                "cosyvoice" if req.use_cosyvoice else
                "chatterbox" if req.use_chatterbox else
                "elevenlabs" if req.use_elevenlabs else
                "edge_tts"
            ),
            "asr_model": req.asr_model,
            "translation_engine": req.translation_engine,
        })

        pipeline_mode = getattr(req, 'pipeline_mode', 'classic')
        progress_cb = _make_progress_callback(job)
        _setup_status(f"Loading pipeline (mode: {pipeline_mode})...", 0.80)

        # ── Enforce mode constraints on backend ──
        if pipeline_mode == "oneflow":
            # ═══ ONEFLOW: Groq → Google → Edge-TTS → 1.15x → video adapts ═══
            from dubbing.oneflow import run_oneflow
            try:
                run_oneflow(
                    source_url=req.url,
                    work_dir=OUTPUTS / job.id / "work",
                    output_path=out_path,
                    target_language=req.target_language,
                    tts_voice=req.voice,
                    tts_rate=req.tts_rate,
                    audio_bitrate=req.audio_bitrate,
                    on_progress=progress_cb,
                    cancel_check=job.cancel_event.is_set,
                )
                job.result_path = out_path
                job.video_title = req.url.split("/")[-1]
            except Exception as e:
                raise RuntimeError(f"OneFlow failed: {e}")
            # Skip all other pipeline paths
            job.segments = []

        elif pipeline_mode == "srtdub":
            # ═══ SRT DIRECT: verbatim SRT → TTS → 0-gap concat → stretch 1-10× + freeze-pad ═══
            from dubbing.srtdub import run_srtdub

            # ── If English transcript SRT provided, translate it first ──
            _transcript_srt = getattr(req, "transcript_srt_content", "").strip()
            _srt_content = getattr(req, "sd_srt_content", "") or ""
            if _transcript_srt:
                progress_cb("translate", 0.0, "Translating English transcript SRT...")
                _srt_content = _translate_srt_content(
                    _transcript_srt,
                    source_lang=req.source_language if req.source_language != "auto" else "en",
                    target_lang=req.target_language,
                    on_progress=progress_cb,
                )
                progress_cb("translate", 1.0, "Translation complete — proceeding to TTS...")

            try:
                run_srtdub(
                    source_url=req.url,
                    srt_content=_srt_content,
                    work_dir=OUTPUTS / job.id / "work",
                    output_path=out_path,
                    tts_voice=req.voice,
                    tts_rate=req.tts_rate,
                    audio_bitrate=req.audio_bitrate,
                    audio_speed=float(getattr(req, "sd_audio_speed", 1.25) or 1.25),
                    vx_hflip=bool(getattr(req, "sd_vx_hflip", True)),
                    vx_hue=float(getattr(req, "sd_vx_hue", 5.0) or 0.0),
                    vx_zoom=float(getattr(req, "sd_vx_zoom", 1.05) or 1.0),
                    max_stretch=float(getattr(req, "sd_max_stretch", 20.0) or 20.0),
                    on_progress=progress_cb,
                    cancel_check=job.cancel_event.is_set,
                )
                job.result_path = out_path
                job.video_title = req.url.split("/")[-1]
            except Exception as e:
                raise RuntimeError(f"SrtDub failed: {e}")
            job.segments = []

        elif pipeline_mode == "wordchunk":
            # ═══ WORDCHUNK: YouTube Hindi VTT → N-word TTS chunks → super-stretch ═══
            from dubbing.wordchunk import run_wordchunk
            try:
                run_wordchunk(
                    source_url=req.url,
                    work_dir=OUTPUTS / job.id / "work",
                    output_path=out_path,
                    target_language=req.target_language,
                    source_language=req.source_language or "en",
                    tts_voice=req.voice,
                    tts_rate=req.tts_rate,
                    audio_bitrate=req.audio_bitrate,
                    chunk_size=int(getattr(req, "wc_chunk_size", 8) or 8),
                    max_stretch=float(getattr(req, "wc_max_stretch", 20.0) or 20.0),
                    transcript_override=getattr(req, "wc_transcript", "") or "",
                    dub_duration_min=int(getattr(req, "dub_duration", 0) or 0),
                    on_progress=progress_cb,
                    cancel_check=job.cancel_event.is_set,
                )
                job.result_path = out_path
                job.video_title = req.url.split("/")[-1]
            except Exception as e:
                raise RuntimeError(f"WordChunk failed: {e}")
            job.segments = []

        elif pipeline_mode == "new":
            # The new DP pipeline has some options it can't consume. Normalize.
            #
            # IMPORTANT (2026-04-12): prefer_youtube_subs is NO LONGER disabled
            # for the new pipeline. YouTube's transcript has proper sentence
            # boundaries, punctuation, and capitalization — it's BETTER input
            # than any ASR (Whisper, Parakeet, Google ASR). Using YouTube subs
            # eliminates the fragment-merging problem entirely because the
            # sentences come pre-segmented correctly from YouTube.
            if (req.asr_model or "").lower() == "groq-whisper":
                req.asr_model = "parakeet"
            # req.prefer_youtube_subs — LEFT ALONE (user's choice flows through)
            # req.use_yt_translate — LEFT ALONE (user's choice flows through)
            req.transcribe_only = False
            req.multi_speaker = False
            req.step_by_step = False
            req.simplify_english = False
            req.dub_chain = []
        elif pipeline_mode == "hybrid":
            # Hybrid keeps old ASR + translation, but disables old simplify (DP cues replace it)
            req.simplify_english = False

        if pipeline_mode == "new":
            # ═══ NEW MODULAR PIPELINE ═══
            # Everything new: Parakeet + WhisperX + DP cues + glossary
            # Old shell only for download + TTS + assembly
            from dubbing.runner import DubbingRunner
            runner = DubbingRunner(
                work_dir=OUTPUTS / job.id / "work",
                source_lang=req.source_language if req.source_language != "auto" else "en",
                target_lang=req.target_language,
                on_progress=progress_cb,
            )
            pipeline = Pipeline(cfg, on_progress=progress_cb,
                                cancel_check=job.cancel_event.is_set)
            job.pipeline_ref = pipeline

            # ── YouTube subs cascade: if found, use the CLASSIC proven pipeline ──
            # When YouTube subs are available, we bypass the entire "new" pipeline
            # (DubbingRunner, SentenceSplit, AV sync) and use the classic
            # Pipeline.run() flow instead. This is the PROVEN process that
            # achieved ±5 second duration matching. YouTube subs just replace
            # the Whisper step — everything else stays identical.
            #
            # The "new" pipeline's DubbingRunner adds SentenceSplit which
            # re-fragments the already-good YouTube sentences into 930 tiny
            # pieces, destroying the sentence scope. By falling through to
            # Pipeline.run(), we get: merge → translate → TTS → assembly
            # with auto rate, proportional balancing, micro-gaps — all proven.
            import re as _yt_re
            _is_url = bool(_yt_re.match(r"^https?://", req.url or ""))
            _use_classic_for_yt = (
                _is_url
                and (req.use_yt_translate or req.prefer_youtube_subs
                     or getattr(req, 'yt_text_correction', False))
            )

            if _use_classic_for_yt:
                # Let Pipeline.run() handle EVERYTHING — it has the YouTube
                # cascade built in (lines 1817-1860), sentence merge, fragment
                # merge, translation, TTS, assembly, auto rate, proportional
                # balancing, micro-gaps. This is the proven path.
                progress_cb("transcribe", 0.0,
                            "YouTube subs available -> using proven classic pipeline flow...")

                # ── Transcript upload: skip transcription in new-pipeline classic path too ──
                _transcript_content_new = getattr(req, "transcript_srt_content", "").strip()
                if _transcript_content_new:
                    _transcript_path_new = work_dir / "transcript_upload.srt"
                    _transcript_path_new.write_text(_transcript_content_new, encoding="utf-8")
                    progress_cb("transcribe", 0.5, "Transcript SRT provided — skipping transcription...")
                    pipeline.run_from_source_srt(_transcript_path_new)
                else:
                    pipeline.run()

                # Copy results to job
                job.video_title = pipeline.video_title or "Untitled"
                job.segments = pipeline.segments
                job.qa_score = pipeline.qa_score
                _budget = getattr(pipeline, "_tts_budget", None)
                if _budget:
                    job.total_words = int(_budget.get("total_words", 0))
                    job.total_sentences = int(_budget.get("total_sentences", 0))
                    job.avg_words_per_sent = float(_budget.get("avg_words_per_sent", 0.0))

            else:
                # No YouTube subs available → use DubbingRunner (new pipeline)
                pipeline.download_and_extract()
                wav_path = OUTPUTS / job.id / "work" / "audio_raw.wav"

            # ── DubbingRunner ASR path (only when YouTube subs weren't found) ──
            if not _use_classic_for_yt:
                glossary_path = OUTPUTS / job.id / "work" / "glossary.json"

                _asr_choice = (req.asr_model or "parakeet").lower()
                use_parakeet = (_asr_choice == "parakeet")
                _whisper_size_map = {
                    "parakeet":       "large-v3",
                    "base":           "base",
                    "small":          "small",
                    "medium":         "medium",
                    "large-v3":       "large-v3",
                    "large-v3-turbo": "large-v3-turbo",
                    "groq-whisper":   "large-v3",
                }
                whisper_model_size = _whisper_size_map.get(_asr_choice, "large-v3")

                def translate_fn(text, hints):
                    segs = [{"text": text, "start": 0, "end": hints.get("duration_ms", 3000) / 1000}]
                    pipeline._translate_segments(segs)
                    return segs[0].get("text_translated", text)

                segments = runner.run_full(
                    wav_path, translate_fn=translate_fn,
                    glossary_path=glossary_path, use_parakeet=use_parakeet,
                    whisper_model_size=whisper_model_size,
                    segmenter=getattr(cfg, 'segmenter', 'dp'),
                    buffer_pct=getattr(cfg, 'segmenter_buffer_pct', 0.20),
                    max_sentences_per_cue=getattr(cfg, 'max_sentences_per_cue', 2),
                )
                runner.export_srt(OUTPUTS / job.id / "work" / f"transcript_{req.target_language}.srt")
                runner.export_source_srt(OUTPUTS / job.id / "work" / "transcript_source.srt")

                # TTS + assembly (old shell renders TTS wavs)
                pipeline.segments = segments
                pipeline._run_tts_and_assembly(segments, wav_path)

            # ── POST-TTS: AV Sync modules (only for DubbingRunner path) ──
            if not _use_classic_for_yt:
                _av_mode = getattr(cfg, 'av_sync_mode', 'original')
                _use_gs = getattr(cfg, 'use_global_stretch', False)
                _audio_untouchable = getattr(cfg, 'audio_untouchable', False)
                tts_wav_dir = OUTPUTS / job.id / "work" / "tts_wavs"

                if _audio_untouchable:
                    if _av_mode != "original" or _use_gs:
                        progress_cb("assembly", 0.85,
                                    "Audio Untouchable is ON -- skipping AV sync modules")
                        _av_mode = "original"
                        _use_gs = False

                if _av_mode != "original" and _use_gs:
                    progress_cb("assembly", 0.85,
                                "Both Slot Recompute and Global Stretch -- using Global Stretch")
                    _av_mode = "original"

                if (_av_mode != "original" or _use_gs) and tts_wav_dir.is_dir():
                    if _av_mode != "original":
                        progress_cb("assembly", 0.85, f"Running slot recompute (mode={_av_mode})...")
                        runner.recompute_slots(
                            tts_wav_dir=tts_wav_dir,
                            av_sync_mode=_av_mode,
                            max_audio_speedup=getattr(cfg, 'max_audio_speedup', 1.30),
                            min_video_speed=getattr(cfg, 'min_video_speed', 0.70),
                            slot_verify_mode=getattr(cfg, 'slot_verify', 'off'),
                        )
                        runner.export_srt(
                            OUTPUTS / job.id / "work" / f"transcript_{req.target_language}_revised.srt",
                            use_revised_timeline=True,
                        )

                    if _use_gs:
                        original_dur = pipeline._get_duration(wav_path) if hasattr(pipeline, '_get_duration') else 0
                        if original_dur > 0:
                            gs_result = runner.compute_global_stretch(
                                tts_wav_dir=tts_wav_dir,
                                original_video_duration=original_dur,
                                audio_speedup=getattr(cfg, 'global_stretch_speedup', 1.25),
                            )
                            progress_cb("assembly", 0.90,
                                        f"Global stretch: {gs_result.video_speed:.2f}x video speed")

        elif pipeline_mode == "hybrid":
            # ═══ HYBRID: Old shell + new core ═══
            # Old: download, extract, ASR (Whisper), TTS, assembly, retry, fallbacks
            # New: DP cue building, glossary, Hindi fitting, QC gates
            # Best of both: proven infrastructure + quality-critical intelligence
            from dubbing.runner import DubbingRunner
            from dubbing import glossary, glossary_builder, cue_builder, qc, fit_hi, format_hi, tts_bridge
            from dubbing.asr_runner import normalize_words
            from dubbing.contracts import Word

            pipeline = Pipeline(cfg, on_progress=progress_cb,
                                cancel_check=job.cancel_event.is_set,
                                pause_event=job.pause_event if req.step_by_step else None)
            job.pipeline_ref = pipeline

            # Step 1-2: Download + Extract (old shell — proven)
            pipeline.download_and_extract()

            # Step 3: ASR (old shell — Whisper/YouTube subs, all options work)
            pipeline._run_transcription()

            # ─── NEW CORE TAKES OVER ───
            # Convert old segments to Word objects
            words = []
            for seg in pipeline.segments:
                if seg.get("words"):
                    for w in seg["words"]:
                        words.append(Word(
                            text=w.get("word", w.get("text", "")),
                            start=w.get("start", 0), end=w.get("end", 0),
                            source="whisper",
                        ))
                else:
                    for word_text in seg.get("text", "").split():
                        words.append(Word(
                            text=word_text,
                            start=seg.get("start", 0), end=seg.get("end", 0),
                            source="whisper",
                        ))
            words = normalize_words(words)

            # Glossary
            glossary_path = OUTPUTS / job.id / "work" / "glossary.json"
            if glossary_path.exists():
                terms = glossary.load_glossary(glossary_path)
            else:
                terms = glossary_builder.extract_terms_from_words(words)
                glossary_builder.save_glossary(terms, glossary_path)

            progress_cb("transcribe", 0.75, f"Glossary: {len(terms)} terms")

            # Tag words + DP cue build
            words = glossary.tag_words(words, terms)
            progress_cb("transcribe", 0.8, "Building DP cues...")
            cues = cue_builder.build_cues(words)
            cues = glossary.tag_cues(cues, terms)
            cues = qc.english_qc(cues)
            issues = qc.count_issues(cues)
            progress_cb("transcribe", 0.9,
                        f"DP cues: {len(cues)} | QC: {issues['pass_rate']:.0%} pass")

            # Convert cues back to pipeline segments for translation
            text_segments = []
            for cue in cues:
                text_segments.append({
                    "start": cue.start, "end": cue.end,
                    "text": cue.text_clean_en,
                    "text_original": cue.text_original,
                    "speaker_id": cue.speaker,
                    "_protected_terms": cue.protected_terms,
                })

            pipeline.segments = text_segments

            # Step 4: Translate (old shell — all 12+ engines, proven)
            progress_cb("translate", 0.0, "Translating with old engine...")
            pipeline._translate_segments(text_segments)
            pipeline.segments = text_segments

            # ─── NEW CORE: Hindi fitting + QC ───
            # BUG FIX: cues and text_segments may have different lengths
            # Use index-based iteration, not zip()
            n_match = min(len(cues), len(text_segments))
            for i in range(n_match):
                cues[i].text_hi_raw = text_segments[i].get("text_translated", "")
                cues[i].text_clean_en = text_segments[i].get("text", "")  # FIX: was text_en_clean (typo)

            cues = fit_hi.fit_cues(cues, terms)
            cues = glossary.validate_hindi(cues, terms)
            cues = format_hi.format_cues(cues)
            cues = qc.pre_tts_qc(cues)

            pre_issues = qc.count_issues(cues)
            progress_cb("translate", 0.95,
                        f"Hindi fit + QC: {pre_issues['pass_rate']:.0%} pass")

            # Write fitted Hindi back to pipeline segments (index-based, not zip)
            for i in range(n_match):
                text_segments[i]["text_translated"] = cues[i].text_hi_fit or cues[i].text_hi_raw

            # Export debug artifacts
            tts_bridge.export_json(cues, OUTPUTS / job.id / "work" / "cues_debug.json")
            tts_bridge.export_csv(cues, OUTPUTS / job.id / "work" / "cues_elevenlabs.csv")

            # Step 5-6: TTS + Assembly (old shell — proven, all engines)
            audio_raw_path = OUTPUTS / job.id / "work" / "audio_raw.wav"
            if not audio_raw_path.exists():
                raise RuntimeError(f"audio_raw.wav not found: {audio_raw_path}")
            pipeline._run_tts_and_assembly(text_segments, audio_raw_path)

        elif pipeline_mode not in ("oneflow", "wordchunk", "srtdub"):
            # ═══ CLASSIC MONOLITH PIPELINE ═══
            pipeline = Pipeline(cfg, on_progress=progress_cb,
                                cancel_check=job.cancel_event.is_set,
                                pause_event=job.pause_event if req.step_by_step else None)
            job.pipeline_ref = pipeline

            # ── Transcript upload: skip transcription → translate directly ──
            _transcript_content = getattr(req, "transcript_srt_content", "").strip()
            if _transcript_content:
                _transcript_path = work_dir / "transcript_upload.srt"
                _transcript_path.write_text(_transcript_content, encoding="utf-8")
                _setup_status("Transcript SRT provided — skipping transcription...", 0.85)
                pipeline.run_from_source_srt(_transcript_path)
            else:
                pipeline.run()

        # OneFlow / WordChunk set their own job data — skip pipeline access
        if pipeline_mode not in ("oneflow", "wordchunk", "srtdub"):
            job.video_title = pipeline.video_title or "Untitled"
            job.segments = pipeline.segments
            job.qa_score = pipeline.qa_score
            # Copy TTS budget metrics (computed by _pretts_word_budget) onto
            # the Job so they show up in the API response and the UI.
            _budget = getattr(pipeline, "_tts_budget", None)
            if _budget:
                job.total_words        = int(_budget.get("total_words", 0))
                job.total_sentences    = int(_budget.get("total_sentences", 0))
                job.avg_words_per_sent = float(_budget.get("avg_words_per_sent", 0.0))
                job.max_seg_words      = int(_budget.get("max_seg_words", 0))
                job.max_sent_words     = int(_budget.get("max_sent_words", 0))

        if req.transcribe_only:
            job.overall_progress = 1.0
            job.state = "waiting_for_srt"
            job.message = "Transcription complete. Download SRT and upload translation."
            job.events.append({"type": "complete", "state": "waiting_for_srt"})
            _store.save(job)
            return

        if not out_path.exists():
            raise RuntimeError("Pipeline finished but output file not found")

        job.result_path = out_path

        # Stash QA report path before work dir is cleaned up
        _qa_src = OUTPUTS / job.id / "work" / "qa_report.txt"
        _qa_report_text = None
        if _qa_src.exists():
            try:
                _qa_report_text = _qa_src.read_text(encoding="utf-8")
            except Exception:
                pass

        # Auto-save to titled folder (deletes work dir)
        try:
            _save_to_titled_folder(job)
        except Exception as save_err:
            print(f"[WARN] Failed to save to titled folder: {save_err}")

        # Save QA report to output folder
        if _qa_report_text and job.saved_folder:
            try:
                (Path(job.saved_folder) / "qa_report.txt").write_text(_qa_report_text, encoding="utf-8")
            except Exception:
                pass

        # Generate YouTube description
        try:
            job.description = _generate_youtube_description(job)
            if job.saved_folder:
                desc_path = Path(job.saved_folder) / "description.txt"
                desc_path.write_text(job.description, encoding="utf-8")
        except Exception as desc_err:
            print(f"[WARN] Failed to generate description: {desc_err}")

        # Send completed translation to hinglish-ai trainer
        try:
            work_dir = OUTPUTS / job.id / "work"
            src_srt = work_dir / "transcript_source.srt"
            tr_srt = work_dir / f"transcript_{req.target_language}.srt"
            if not src_srt.exists():
                # Fallback: look for any cached transcription SRT
                for f in work_dir.glob("transcript_*.srt"):
                    if req.target_language not in f.name:
                        src_srt = f
                        break
            if src_srt.exists() and tr_srt.exists():
                _send_training_data(src_srt, tr_srt, source_lang=req.source_language)
        except Exception:
            pass

        job.overall_progress = 1.0
        job.state = "done"
        qa_msg = f" (QA: {job.qa_score:.0%})" if job.qa_score is not None else ""
        job.message = f"Complete{qa_msg}"
        job.events.append({"type": "complete", "state": "done"})
        _store.save(job)

        # Record job metrics to Supabase (fire-and-forget)
        _render_time = time.time() - _t_start
        _pipeline_exists = pipeline_mode not in ("oneflow", "wordchunk", "srtdub") and 'pipeline' in locals()
        _segs = (pipeline.segments if _pipeline_exists else job.segments) or []
        # Read manual review queue to count segments that needed review
        _mrq_path = OUTPUTS / job.id / "manual_review_queue.json"
        _mrq_path_saved = Path(job.saved_folder) / "manual_review_queue.json" if job.saved_folder else None
        _manual_review_count = 0
        for _mp in [_mrq_path_saved, _mrq_path]:
            if _mp and _mp.exists():
                try:
                    import json as _json
                    _manual_review_count = len(_json.loads(_mp.read_text(encoding="utf-8")))
                    break
                except Exception:
                    pass
        get_metrics().record_job_complete(job.id, "done", {
            "total_segments": len(_segs),
            "pass_rate_first_try": 1.0 if not _segs else
                max(0.0, (len(_segs) - _manual_review_count) / len(_segs)),
            "manual_review_count": _manual_review_count,
            "total_render_time_s": _render_time,
            "video_title": job.video_title,
        })
        if _segs:
            get_metrics().record_segments(job.id, [
                {
                    "segment_idx": i,
                    "start_time": s.get("start", 0),
                    "end_time": s.get("end", 0),
                    "source_text": s.get("text", ""),
                    "translated_text": s.get("text_translated", ""),
                    "emotion": s.get("emotion", "neutral"),
                }
                for i, s in enumerate(_segs)
            ])

        # Mark this URL as completed so it won't be re-queued on restart
        if job.source_url:
            # Only mark completed if no chain remaining (final language done)
            if not job.chain_languages:
                _mark_url_completed(job.source_url)

        # Chain dubbing: if more languages remain, queue next step
        if job.chain_languages and job.saved_video:
            _queue_chain_next(job)

    except Exception as e:
        import traceback
        err_text = f"[JOB ERROR] {e}\n{traceback.format_exc()}"
        print(err_text, flush=True)
        # Dump full job context + traceback to the per-run crash log
        _crash_dump_job(job, e)
        # Long-segment trace: write whatever we collected before the crash so
        # the user can still see WHERE the crash happened relative to long
        # segments. Best-effort.
        try:
            if pipeline is not None and hasattr(pipeline, "_trace_write_report"):
                pipeline._trace_write_report(job.id)
        except Exception:
            pass
        # Clean up work directory but keep error log in saved folder if available
        try:
            job_dir = OUTPUTS / job.id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            pass
        job.state = "error"
        job.error = str(e)
        job.message = f"Error: {e}"
        job.events.append({"type": "complete", "state": "error", "error": str(e)})
        _store.save(job)
        try:
            get_metrics().record_job_complete(job.id, "error", {"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            _purge_global_caches()
        except Exception:
            pass
        _pipeline_semaphore.release()


def _run_job_split(job: Job, req: JobCreateRequest, voice: str):
    """Split a long video into parts and dub each one separately.
    Called from within _run_job when split_duration > 0."""
    import math as _math

    job_dir = OUTPUTS / job.id
    work_dir = job_dir / "work"
    work_dir.mkdir(exist_ok=True)
    split_dir = work_dir / "splits"
    split_dir.mkdir(exist_ok=True)

    callback = _make_progress_callback(job)

    # Step 1: Download the video first
    callback("download", 0.0, "Downloading video for splitting...")
    dl_cfg = PipelineConfig(
        source=req.url,
        work_dir=work_dir,
        output_path=job_dir / "dubbed.mp4",
        source_language=req.source_language,
        target_language=req.target_language,
    )
    dl_pipeline = Pipeline(dl_cfg, on_progress=callback, cancel_check=job.cancel_event.is_set)
    job.pipeline_ref = dl_pipeline   # cancel handler can kill in-flight subprocesses
    dl_pipeline._ensure_ffmpeg()
    video_path = dl_pipeline._ingest_source(req.url)
    job.video_title = dl_pipeline.video_title or "Untitled"
    callback("download", 1.0, f"Downloaded: {video_path.name}")

    if job.cancel_event.is_set():
        raise RuntimeError("Job cancelled")

    # Step 2: Split the video
    callback("extract", 0.0, f"Splitting video into {req.split_duration}-min parts...")
    ffmpeg_path = dl_pipeline._ffmpeg
    parts = _split_video(ffmpeg_path, video_path, req.split_duration, split_dir)
    num_parts = len(parts)
    callback("extract", 1.0, f"Split into {num_parts} parts")

    if num_parts <= 1:
        # Video is shorter than split duration — run normally
        callback("extract", 1.0, "Video is shorter than split duration, running as single part...")
        out_path = job_dir / "dubbed.mp4"
        cfg = PipelineConfig(
            source=str(video_path), work_dir=work_dir, output_path=out_path,
            source_language=req.source_language, target_language=req.target_language,
            asr_model=req.asr_model, translation_engine=req.translation_engine,
            tts_voice=voice, tts_rate=req.tts_rate,
            mix_original=req.mix_original, original_volume=req.original_volume,
            use_cosyvoice=req.use_cosyvoice,
            use_chatterbox=req.use_chatterbox,
            use_indic_parler=req.use_indic_parler,
            use_sarvam_bulbul=req.use_sarvam_bulbul,
            use_elevenlabs=req.use_elevenlabs,
            use_google_tts=req.use_google_tts,
            use_coqui_xtts=req.use_coqui_xtts,
            use_fish_speech=req.use_fish_speech,
            use_edge_tts=req.use_edge_tts,
            prefer_youtube_subs=False,
            use_yt_translate=req.use_yt_translate,
            multi_speaker=req.multi_speaker,
            transcribe_only=req.transcribe_only,
            audio_priority=req.audio_priority,
            audio_untouchable=req.audio_untouchable,
            video_slow_to_match=req.video_slow_to_match,
            post_tts_level=req.post_tts_level,
            audio_quality_mode=req.audio_quality_mode,
            enable_sentence_gap=req.enable_sentence_gap,
            enable_duration_fit=req.enable_duration_fit,
            audio_bitrate=req.audio_bitrate,
            encode_preset=req.encode_preset,
            download_mode=getattr(req, 'download_mode', 'remux'),
            dub_duration=req.dub_duration,
            fast_assemble=req.fast_assemble,
            enable_manual_review=req.enable_manual_review,
            use_whisperx=req.use_whisperx,
            simplify_english=req.simplify_english,
            step_by_step=req.step_by_step,
            enable_tts_verify_retry=req.enable_tts_verify_retry,
            tts_truncation_threshold=getattr(req, 'tts_truncation_threshold', 0.30),
            keep_subject_english=getattr(req, 'keep_subject_english', False),
            tts_word_match_verify=getattr(req, 'tts_word_match_verify', False),
            tts_word_match_tolerance=getattr(req, 'tts_word_match_tolerance', 0.15),
            tts_word_match_model=getattr(req, 'tts_word_match_model', 'auto'),
            tts_word_match_max_segments=getattr(req, 'tts_word_match_max_segments', 1000),
            use_wav2lip=getattr(req, 'use_wav2lip', False),
            long_segment_trace=getattr(req, 'long_segment_trace', True),
            long_segment_threshold_words=getattr(req, 'long_segment_threshold_words', 15),
            tts_no_time_pressure=getattr(req, 'tts_no_time_pressure', True),
            tts_dynamic_workers=getattr(req, 'tts_dynamic_workers', True),
            tts_dynamic_min=getattr(req, 'tts_dynamic_min', 10),
            tts_dynamic_max=getattr(req, 'tts_dynamic_max', 120),
            tts_dynamic_start=getattr(req, 'tts_dynamic_start', 30),
            tts_rate_mode=getattr(req, 'tts_rate_mode', 'auto'),
            tts_rate_ceiling=getattr(req, 'tts_rate_ceiling', '+25%'),
            tts_rate_target_wpm=getattr(req, 'tts_rate_target_wpm', 130),
            srt_needs_translation=req.srt_needs_translation,
            av_sync_mode=getattr(req, 'av_sync_mode', 'original'),
            max_audio_speedup=getattr(req, 'max_audio_speedup', 1.30),
            min_video_speed=getattr(req, 'min_video_speed', 0.70),
            slot_verify=getattr(req, 'slot_verify', 'off'),
            use_global_stretch=getattr(req, 'use_global_stretch', False),
            global_stretch_speedup=getattr(req, 'global_stretch_speedup', 1.25),
            segmenter=getattr(req, 'segmenter', 'dp'),
            segmenter_buffer_pct=getattr(req, 'segmenter_buffer_pct', 0.20),
            max_sentences_per_cue=getattr(req, 'max_sentences_per_cue', 2),
            yt_transcript_mode=getattr(req, 'yt_transcript_mode', 'yt_timeline'),
            yt_segment_mode=getattr(req, 'yt_segment_mode', 'sentence'),
            yt_text_correction=getattr(req, 'yt_text_correction', True),
            yt_replace_mode=getattr(req, 'yt_replace_mode', 'diff'),
            tts_chunk_words=getattr(req, 'tts_chunk_words', 0),
            gap_mode=getattr(req, 'gap_mode', 'micro'),
        )
        p = Pipeline(cfg, on_progress=callback, cancel_check=job.cancel_event.is_set)
        job.pipeline_ref = p   # so DELETE handler can _kill_all_procs() instantly
        p.video_title = job.video_title
        p.run()
        job.result_path = out_path
        job.segments = p.segments
        job.video_title = p.video_title or job.video_title
        job.qa_score = p.qa_score
        job.overall_progress = 1.0
        job.state = "done"
        job.message = "Complete"
        job.events.append({"type": "complete", "state": "done"})
        try:
            _save_to_titled_folder(job)
        except Exception:
            pass
        _store.save(job)
        return

    # Step 3: Process each part
    output_parts = []
    for part_idx, part_path in enumerate(parts):
        part_num = part_idx + 1
        part_label = f"Part {part_num}/{num_parts}"

        if job.cancel_event.is_set():
            raise RuntimeError("Job cancelled")

        # Create work dir for this part
        part_work = work_dir / f"part_{part_num:02d}"
        part_work.mkdir(exist_ok=True)
        part_out = job_dir / f"dubbed_part{part_num:02d}.mp4"

        # Progress callback that scales to this part's position in overall progress
        # Parts share transcribe (25%), translate (15%), synthesize (30%), assemble (10%) = 80%
        # Download (15%) + extract (5%) already done = 20%
        part_base = 0.20 + 0.80 * (part_idx / num_parts)
        part_range = 0.80 / num_parts

        def _part_callback(step, progress, message, _base=part_base, _range=part_range, _label=part_label):
            step_w = STEP_WEIGHTS.get(step, 0.1)
            step_idx = STEP_ORDER.index(step) if step in STEP_ORDER else 0
            # Map step progress to overall part progress
            step_offset = sum(STEP_WEIGHTS.get(s, 0) for s in STEP_ORDER[:step_idx])
            overall = _base + _range * (step_offset + step_w * progress) / 0.80
            job.current_step = step
            job.step_progress = progress
            job.overall_progress = min(overall, 0.99)
            job.message = f"[{_label}] {message}"
            job.events.append({
                "step": step,
                "progress": round(progress, 3),
                "overall": round(job.overall_progress, 3),
                "message": f"[{_label}] {message}",
            })
            if len(job.events) > 500:
                job.events = job.events[-500:]

        _part_callback("transcribe", 0.0, f"Starting {part_label}...")

        cfg = PipelineConfig(
            source=str(part_path),
            work_dir=part_work,
            output_path=part_out,
            source_language=req.source_language,
            target_language=req.target_language,
            asr_model=req.asr_model,
            translation_engine=req.translation_engine,
            tts_voice=voice,
            tts_rate=req.tts_rate,
            mix_original=req.mix_original,
            original_volume=req.original_volume,
            use_cosyvoice=req.use_cosyvoice,
            use_chatterbox=req.use_chatterbox,
            use_indic_parler=req.use_indic_parler,
            use_sarvam_bulbul=req.use_sarvam_bulbul,
            use_elevenlabs=req.use_elevenlabs,
            use_google_tts=req.use_google_tts,
            use_coqui_xtts=req.use_coqui_xtts,
            use_fish_speech=req.use_fish_speech,
            use_edge_tts=req.use_edge_tts,
            prefer_youtube_subs=False,
            use_yt_translate=req.use_yt_translate,
            multi_speaker=req.multi_speaker,
            transcribe_only=req.transcribe_only,
            audio_priority=req.audio_priority,
            audio_untouchable=req.audio_untouchable,
            video_slow_to_match=req.video_slow_to_match,
            post_tts_level=req.post_tts_level,
            audio_quality_mode=req.audio_quality_mode,
            enable_sentence_gap=req.enable_sentence_gap,
            enable_duration_fit=req.enable_duration_fit,
            audio_bitrate=req.audio_bitrate,
            encode_preset=req.encode_preset,
            download_mode=getattr(req, 'download_mode', 'remux'),
            dub_duration=req.dub_duration,
            fast_assemble=req.fast_assemble,
            enable_manual_review=req.enable_manual_review,
            use_whisperx=req.use_whisperx,
            simplify_english=req.simplify_english,
            step_by_step=req.step_by_step,
            enable_tts_verify_retry=req.enable_tts_verify_retry,
            tts_truncation_threshold=getattr(req, 'tts_truncation_threshold', 0.30),
            keep_subject_english=getattr(req, 'keep_subject_english', False),
            tts_word_match_verify=getattr(req, 'tts_word_match_verify', False),
            tts_word_match_tolerance=getattr(req, 'tts_word_match_tolerance', 0.15),
            tts_word_match_model=getattr(req, 'tts_word_match_model', 'auto'),
            tts_word_match_max_segments=getattr(req, 'tts_word_match_max_segments', 1000),
            use_wav2lip=getattr(req, 'use_wav2lip', False),
            long_segment_trace=getattr(req, 'long_segment_trace', True),
            long_segment_threshold_words=getattr(req, 'long_segment_threshold_words', 15),
            tts_no_time_pressure=getattr(req, 'tts_no_time_pressure', True),
            tts_dynamic_workers=getattr(req, 'tts_dynamic_workers', True),
            tts_dynamic_min=getattr(req, 'tts_dynamic_min', 10),
            tts_dynamic_max=getattr(req, 'tts_dynamic_max', 120),
            tts_dynamic_start=getattr(req, 'tts_dynamic_start', 30),
            tts_rate_mode=getattr(req, 'tts_rate_mode', 'auto'),
            tts_rate_ceiling=getattr(req, 'tts_rate_ceiling', '+25%'),
            tts_rate_target_wpm=getattr(req, 'tts_rate_target_wpm', 130),
            srt_needs_translation=req.srt_needs_translation,
            av_sync_mode=getattr(req, 'av_sync_mode', 'original'),
            max_audio_speedup=getattr(req, 'max_audio_speedup', 1.30),
            min_video_speed=getattr(req, 'min_video_speed', 0.70),
            slot_verify=getattr(req, 'slot_verify', 'off'),
            use_global_stretch=getattr(req, 'use_global_stretch', False),
            global_stretch_speedup=getattr(req, 'global_stretch_speedup', 1.25),
            segmenter=getattr(req, 'segmenter', 'dp'),
            segmenter_buffer_pct=getattr(req, 'segmenter_buffer_pct', 0.20),
            max_sentences_per_cue=getattr(req, 'max_sentences_per_cue', 2),
            yt_transcript_mode=getattr(req, 'yt_transcript_mode', 'yt_timeline'),
            yt_segment_mode=getattr(req, 'yt_segment_mode', 'sentence'),
            yt_text_correction=getattr(req, 'yt_text_correction', True),
            yt_replace_mode=getattr(req, 'yt_replace_mode', 'diff'),
            tts_chunk_words=getattr(req, 'tts_chunk_words', 0),
            gap_mode=getattr(req, 'gap_mode', 'micro'),
        )

        pipeline = Pipeline(cfg, on_progress=_part_callback,
                           cancel_check=job.cancel_event.is_set)
        job.pipeline_ref = pipeline   # cancel handler can kill in-flight subprocesses
        pipeline.video_title = f"{job.video_title} - Part {part_num}"
        pipeline.run()

        # Accumulate per-part TTS budget into the Job totals so the UI shows
        # the cumulative count across the whole split job, not just one part.
        _budget = getattr(pipeline, "_tts_budget", None)
        if _budget:
            job.total_words      += int(_budget.get("total_words", 0))
            job.total_sentences  += int(_budget.get("total_sentences", 0))
            _seg_max = int(_budget.get("max_seg_words", 0))
            _sent_max = int(_budget.get("max_sent_words", 0))
            if _seg_max > job.max_seg_words:
                job.max_seg_words = _seg_max
            if _sent_max > job.max_sent_words:
                job.max_sent_words = _sent_max
            # Recompute avg from accumulated totals (more accurate than averaging averages)
            if job.total_sentences > 0:
                job.avg_words_per_sent = round(job.total_words / job.total_sentences, 2)

        if part_out.exists():
            output_parts.append((part_num, part_out))
            print(f"[SPLIT] Part {part_num}/{num_parts} complete: {part_out}", flush=True)

    if not output_parts:
        raise RuntimeError("No parts were produced")

    # Save all parts to titled folders — sanitize title for Windows filesystem
    base_title = _sanitize_filename(job.video_title or "Untitled")
    saved_parts = []
    for part_num, part_out in output_parts:
        part_title = f"{base_title} - Part {part_num}"
        dest_dir = SAVED_DIR / part_title
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{part_title}.mp4"
        shutil.copy2(part_out, dest_path)
        saved_parts.append(str(dest_path))
        print(f"[SPLIT] Saved: {dest_path}", flush=True)

    job.result_path = output_parts[0][1]  # First part for preview
    job.saved_folder = str(SAVED_DIR / base_title) if len(output_parts) == 1 else str(SAVED_DIR)
    job.saved_video = saved_parts[0] if saved_parts else None
    job.overall_progress = 1.0
    job.state = "done"
    job.message = f"Complete — {len(output_parts)} parts dubbed!"
    job.events.append({"type": "complete", "state": "done",
                       "parts": len(output_parts)})
    _store.save(job)

    if job.source_url:
        _mark_url_completed(job.source_url)

    # Cleanup work directory
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass


def _queue_chain_next(parent_job: Job):
    """Queue the next language in a dub chain, using the parent's output as input."""
    next_lang = parent_job.chain_languages[0]
    remaining = parent_job.chain_languages[1:]

    print(f"[CHAIN] Job {parent_job.id} done ({parent_job.target_language}). "
          f"Queuing next: {next_lang} (remaining: {remaining})", flush=True)

    job_id = uuid.uuid4().hex[:12]
    # Use the saved video from previous step as input
    input_path = parent_job.saved_video

    # Create request using the output video as source — carry all settings from original
    _orig = parent_job.original_req
    req = JobCreateRequest(
        url=input_path,
        source_language=parent_job.target_language,  # Previous output language
        target_language=next_lang,
        prefer_youtube_subs=False,  # No YouTube subs for local file
        asr_model=_orig.asr_model if _orig else "large-v3",
        translation_engine=_orig.translation_engine if _orig else "auto",
        tts_rate=_orig.tts_rate if _orig else "+0%",
        use_cosyvoice=getattr(_orig, 'use_cosyvoice', False) if _orig else False,
        use_chatterbox=getattr(_orig, 'use_chatterbox', False) if _orig else False,
        use_indic_parler=getattr(_orig, 'use_indic_parler', False) if _orig else False,
        use_sarvam_bulbul=getattr(_orig, 'use_sarvam_bulbul', False) if _orig else False,
        use_elevenlabs=getattr(_orig, 'use_elevenlabs', False) if _orig else False,
        use_google_tts=getattr(_orig, 'use_google_tts', False) if _orig else False,
        use_coqui_xtts=getattr(_orig, 'use_coqui_xtts', False) if _orig else False,
        use_fish_speech=getattr(_orig, 'use_fish_speech', False) if _orig else False,
        use_edge_tts=getattr(_orig, 'use_edge_tts', True) if _orig else True,
        mix_original=getattr(_orig, 'mix_original', False) if _orig else False,
        original_volume=getattr(_orig, 'original_volume', 0.10) if _orig else 0.10,
        multi_speaker=getattr(_orig, 'multi_speaker', False) if _orig else False,
        audio_priority=getattr(_orig, 'audio_priority', True) if _orig else True,
        audio_untouchable=getattr(_orig, 'audio_untouchable', False) if _orig else False,
        video_slow_to_match=getattr(_orig, 'video_slow_to_match', True) if _orig else True,
        post_tts_level=getattr(_orig, 'post_tts_level', 'minimal') if _orig else 'minimal',
        audio_quality_mode=getattr(_orig, 'audio_quality_mode', 'fast') if _orig else 'fast',
        enable_sentence_gap=getattr(_orig, 'enable_sentence_gap', False) if _orig else False,
        enable_duration_fit=getattr(_orig, 'enable_duration_fit', False) if _orig else False,
        audio_bitrate=getattr(_orig, 'audio_bitrate', '192k') if _orig else '192k',
        encode_preset=getattr(_orig, 'encode_preset', 'fast') if _orig else 'fast',
        download_mode=getattr(_orig, 'download_mode', 'remux') if _orig else 'remux',
        dub_duration=getattr(_orig, 'dub_duration', 0) if _orig else 0,
        fast_assemble=getattr(_orig, 'fast_assemble', False) if _orig else False,
        enable_manual_review=getattr(_orig, 'enable_manual_review', False) if _orig else False,
        use_whisperx=getattr(_orig, 'use_whisperx', False) if _orig else False,
        simplify_english=getattr(_orig, 'simplify_english', False) if _orig else False,
        tts_no_time_pressure=getattr(_orig, 'tts_no_time_pressure', True) if _orig else True,
        tts_rate_mode=getattr(_orig, 'tts_rate_mode', 'auto') if _orig else 'auto',
        tts_rate_ceiling=getattr(_orig, 'tts_rate_ceiling', '+25%') if _orig else '+25%',
        tts_rate_target_wpm=getattr(_orig, 'tts_rate_target_wpm', 130) if _orig else 130,
        tts_truncation_threshold=getattr(_orig, 'tts_truncation_threshold', 0.30) if _orig else 0.30,
        tts_word_match_verify=getattr(_orig, 'tts_word_match_verify', True) if _orig else True,
        tts_word_match_tolerance=getattr(_orig, 'tts_word_match_tolerance', 0.15) if _orig else 0.15,
        tts_word_match_model=getattr(_orig, 'tts_word_match_model', 'auto') if _orig else 'auto',
        tts_dynamic_workers=getattr(_orig, 'tts_dynamic_workers', True) if _orig else True,
        tts_dynamic_min=getattr(_orig, 'tts_dynamic_min', 10) if _orig else 10,
        tts_dynamic_max=getattr(_orig, 'tts_dynamic_max', 120) if _orig else 120,
        tts_dynamic_start=getattr(_orig, 'tts_dynamic_start', 30) if _orig else 30,
        long_segment_trace=getattr(_orig, 'long_segment_trace', True) if _orig else True,
        long_segment_threshold_words=getattr(_orig, 'long_segment_threshold_words', 15) if _orig else 15,
        keep_subject_english=getattr(_orig, 'keep_subject_english', False) if _orig else False,
        gap_mode=getattr(_orig, 'gap_mode', 'micro') if _orig else 'micro',
        tts_chunk_words=getattr(_orig, 'tts_chunk_words', 0) if _orig else 0,
        yt_text_correction=getattr(_orig, 'yt_text_correction', True) if _orig else True,
        yt_replace_mode=getattr(_orig, 'yt_replace_mode', 'diff') if _orig else 'diff',
    )

    job = Job(
        id=job_id,
        source_url=parent_job.source_url,  # Keep original URL for tracking
        target_language=next_lang,
        chain_languages=remaining,
        chain_parent_id=parent_job.id,
    )
    job.original_req = req
    job.video_title = parent_job.video_title  # Carry title forward
    JOBS[job_id] = job
    _store.save(job)

    parent_job.message = f"Complete — next: dubbing to {next_lang.upper()}"

    t = threading.Thread(target=_run_job, args=(job, req), daemon=True)
    t.start()
    job.worker_thread = t


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Translation Glossary API ──────────────────────────────────────────────────
GLOSSARY_PATH = Path(__file__).parent / "translation_glossary.json"

def _load_glossary() -> dict:
    try:
        if GLOSSARY_PATH.exists():
            import json
            data = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        pass
    return {}

def _save_glossary(glossary: dict):
    import json
    data = {"_comment": "Translation Glossary: English word -> Hindi output. Edit freely.", **glossary}
    GLOSSARY_PATH.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")

@app.get("/api/glossary")
def get_glossary():
    return _load_glossary()

@app.post("/api/glossary")
async def add_glossary_entry(request: Request):
    body = await request.json()
    english = body.get("english", "").strip().lower()
    hindi = body.get("hindi", "").strip()
    if not english or not hindi:
        return {"error": "Both 'english' and 'hindi' fields required"}, 400
    glossary = _load_glossary()
    glossary[english] = hindi
    _save_glossary(glossary)
    return {"ok": True, "glossary": glossary}

@app.delete("/api/glossary/{word}")
def delete_glossary_entry(word: str):
    glossary = _load_glossary()
    word_lower = word.strip().lower()
    if word_lower in glossary:
        del glossary[word_lower]
        _save_glossary(glossary)
        return {"ok": True, "glossary": glossary}
    return {"error": f"'{word}' not in glossary"}


@app.get("/api/cache")
def cache_stats():
    """Return content-hash cache statistics (entry counts and disk usage)."""
    import cache as _cache_mod
    return _cache_mod.cache_stats()


@app.delete("/api/cache")
def clear_cache(older_than_days: int = 0):
    """Clear cached ASR/translation/TTS entries.
    older_than_days=0 clears everything; N clears entries not accessed in N days."""
    import cache as _cache_mod
    return _cache_mod.clear_cache(older_than_days)


@app.get("/api/voices")
async def voices(lang: str = "hi") -> Any:
    return await list_voices(lang)


@app.get("/api/cookies/status")
def cookies_status():
    """Verify the YouTube cookies file is present and recognized as a Premium
    subscriber. Returns a JSON status the UI can show to the user.

    Test method: runs `yt-dlp -F --cookies <file> <test_url>` against a known
    public video and looks for "Premium" tags in the output. If any format row
    is marked Premium, the cookies are valid AND the account is Premium.

    Status values:
      - "ok"           — file present, Premium recognized
      - "logged_in"    — file present, recognized as logged-in but not Premium
      - "anonymous"    — file present but YouTube doesn't recognize the session
      - "missing"      — no cookies.txt file in any of the expected locations
      - "error"        — yt-dlp failed for an unrelated reason (network etc.)
    """
    # Find the cookies file using the same logic as Pipeline._find_cookies_file
    cookie_paths = [
        BASE_DIR / "cookies.txt",
        Path.home() / "cookies.txt",
    ]
    cookie_file = None
    for p in cookie_paths:
        if p.exists():
            cookie_file = str(p)
            break

    if not cookie_file:
        return {
            "status": "missing",
            "message": "No cookies.txt found. Save your YouTube cookies to "
                       f"{BASE_DIR / 'cookies.txt'} to enable Premium downloads.",
            "checked_paths": [str(p) for p in cookie_paths],
            "premium": False,
        }

    # Run yt-dlp -F against a known stable test video
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    try:
        # Find yt-dlp on PATH
        ytdlp_exe = shutil.which("yt-dlp") or "yt-dlp"
        result = subprocess.run(
            [ytdlp_exe, "-F", "--cookies", cookie_file,
             "--no-warnings", "--quiet", test_url],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"yt-dlp failed: {(result.stderr or '')[:300]}",
                "cookie_file": cookie_file,
                "premium": False,
            }
        out = result.stdout or ""

        # Check for Premium markers in the format list. yt-dlp prints
        # "(Premium)" in the NOTES column for Premium-gated formats.
        is_premium = "Premium" in out

        # Check for "drm" or other login-required indicators
        has_formats = ("mp4" in out.lower()) or ("webm" in out.lower())

        if is_premium:
            return {
                "status": "ok",
                "message": "✅ Cookies recognized as YouTube Premium. Full bandwidth available.",
                "cookie_file": cookie_file,
                "premium": True,
            }
        if has_formats:
            return {
                "status": "logged_in",
                "message": "Cookies work but YouTube doesn't see this account as Premium. "
                           "Verify you exported from a Premium-subscribed browser session.",
                "cookie_file": cookie_file,
                "premium": False,
            }
        return {
            "status": "anonymous",
            "message": "Cookies file present but YouTube treats requests as anonymous. "
                       "The cookies may be expired — re-export from your browser.",
            "cookie_file": cookie_file,
            "premium": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "yt-dlp timed out (>30s). Network or YouTube issue.",
            "cookie_file": cookie_file,
            "premium": False,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Verification failed: {e}",
            "cookie_file": cookie_file,
            "premium": False,
        }


@app.get("/api/jobs")
def list_jobs():
    """List all jobs, newest first."""
    jobs = sorted(list(JOBS.values()), key=lambda j: getattr(j, 'created_at', 0), reverse=True)
    result = []
    for j in jobs:
        try:
            result.append({
                "id": j.id,
                "state": getattr(j, "state", "unknown"),
                "current_step": getattr(j, "current_step", ""),
                "step_progress": getattr(j, "step_progress", 0),
                "overall_progress": getattr(j, "overall_progress", 0),
                "step_times": getattr(j, "step_times", {}),
                "message": getattr(j, "message", ""),
                "error": getattr(j, "error", None),
                "source_url": getattr(j, "source_url", ""),
                "video_title": getattr(j, "video_title", ""),
                "target_language": getattr(j, "target_language", ""),
                "created_at": getattr(j, "created_at", 0),
                "saved_folder": getattr(j, "saved_folder", None),
                "saved_video": getattr(j, "saved_video", None),
                "description": getattr(j, "description", None),
                "qa_score": getattr(j, "qa_score", None),
                "chain_languages": getattr(j, "chain_languages", []),
                "chain_parent_id": getattr(j, "chain_parent_id", None),
                "total_words": getattr(j, "total_words", None),
                "total_sentences": getattr(j, "total_sentences", None),
            })
        except Exception:
            # Skip corrupt jobs instead of crashing the entire list
            result.append({"id": getattr(j, "id", "?"), "state": "error",
                           "message": "Job data corrupted", "error": "corrupt"})
    return result


# ── Always-purge end-of-job hook ──────────────────────────────────────────────
# User requested NO REUSE of any cached data. Every successful or failed job
# wipes the global content-hash caches under backend/cache/ so the next job
# starts from scratch. The work_dir under OUTPUTS/<job_id>/ is already wiped
# by _save_to_titled_folder on success, and by the error handler on failure.

def _purge_global_caches() -> int:
    """Wipe backend/cache/{asr,translation,tts}/. Returns bytes freed."""
    cache_root = BASE_DIR / "cache"
    freed = 0
    for sub in ("asr", "translation", "tts"):
        d = cache_root / sub
        if not d.exists():
            continue
        for f in d.glob("*"):
            try:
                if f.is_file():
                    freed += f.stat().st_size
                    f.unlink()
            except OSError:
                pass
    if freed > 0:
        print(f"[CACHE-PURGE] Wiped {freed/1024/1024:.0f} MB of "
              f"backend/cache/ (no-reuse mode)", flush=True)
    return freed


# ── Per-URL disk purge ───────────────────────────────────────────────────────
# Tracks the most recently submitted URL so we can wipe its work artifacts
# (multi-GB) when a different URL is submitted next, keeping disk usage bounded.
_last_submitted_url: Optional[str] = None
_last_url_lock = threading.Lock()


def _purge_url_artifacts(url: str, *, also_caches: bool = False) -> Dict[str, int]:
    """Delete every work artifact tied to a specific source URL.

    Frees the multi-GB intermediate files (raw audio, source.mp4, TTS wavs)
    that the pipeline writes per job. Optionally also wipes the global
    content-hash caches under backend/cache/, but those are usually safe
    to keep across videos because they're SHA-keyed.

    Returns a small dict with bytes/files freed for logging.
    """
    if not url:
        return {"bytes": 0, "dirs": 0, "cache_files": 0}

    freed_bytes = 0
    freed_dirs = 0

    # 1. Find all jobs in JOBS that match this source_url and wipe their
    # work_dir on disk + dubbed_outputs folder + DB row.
    matching_ids = [jid for jid, j in list(JOBS.items())
                    if (j.source_url or "") == url]
    for jid in matching_ids:
        job_dir = OUTPUTS / jid
        if job_dir.exists():
            try:
                # Measure size before delete
                for p in job_dir.rglob("*"):
                    if p.is_file():
                        try:
                            freed_bytes += p.stat().st_size
                        except OSError:
                            pass
                shutil.rmtree(job_dir, ignore_errors=True)
                freed_dirs += 1
            except Exception as e:
                print(f"[PURGE] failed to remove {job_dir}: {e}", flush=True)

        # Drop from in-memory + DB so the UI doesn't show ghost entries
        JOBS.pop(jid, None)
        try:
            _store.delete(jid)
        except Exception:
            pass

    # 2. Optionally clean the global content-hash caches.
    cache_files = 0
    if also_caches:
        cache_root = BASE_DIR / "cache"
        for sub in ("asr", "translation", "tts"):
            d = cache_root / sub
            if not d.exists():
                continue
            for f in d.glob("*"):
                try:
                    if f.is_file():
                        freed_bytes += f.stat().st_size
                        f.unlink()
                        cache_files += 1
                except OSError:
                    pass

    return {"bytes": freed_bytes, "dirs": freed_dirs, "cache_files": cache_files}


def _maybe_purge_previous_url(new_url: str, enabled: bool):
    """Called at the top of create_job. If `enabled` is True and the new URL
    is different from the previously submitted one, purge the prior URL's
    work artifacts (frees multi-GB) before starting the new job."""
    global _last_submitted_url
    if not enabled:
        # Still update the tracker so a future enable can act on it
        with _last_url_lock:
            _last_submitted_url = new_url
        return

    with _last_url_lock:
        prev = _last_submitted_url
        _last_submitted_url = new_url

    if prev and prev != new_url:
        result = _purge_url_artifacts(prev, also_caches=False)
        if result["bytes"] > 0 or result["dirs"] > 0:
            print(
                f"[PURGE] new URL submitted, cleaned previous: "
                f"{result['dirs']} dirs, "
                f"{result['bytes']/1024/1024:.0f} MB freed "
                f"(prev URL: {prev[:80]})",
                flush=True,
            )


def _cleanup_old_jobs():
    """Remove oldest completed/errored jobs and orphaned work directories."""
    # Clean in-memory jobs exceeding limit
    if len(JOBS) > MAX_JOBS:
        completed = sorted(
            [(jid, j) for jid, j in list(JOBS.items()) if j.state in ("done", "error")],
            key=lambda x: x[1].created_at,
        )
        while len(JOBS) > MAX_JOBS and completed:
            jid, _ = completed.pop(0)
            JOBS.pop(jid, None)
            _store.delete(jid)
            job_dir = OUTPUTS / jid
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)

    # Clean orphaned work dirs on disk (not tracked in JOBS, older than 2 hours)
    if OUTPUTS.exists():
        now = time.time()
        for d in OUTPUTS.iterdir():
            if d.is_dir() and d.name not in JOBS:
                try:
                    age = now - d.stat().st_mtime
                    if age > 7200:  # 2 hours old
                        shutil.rmtree(d, ignore_errors=True)
                        print(f"[CLEANUP] Removed orphaned work dir: {d.name}", flush=True)
                except Exception:
                    pass


@app.post("/api/jobs")
def create_job(req: JobCreateRequest):
    """Create a new dubbing job from a YouTube URL."""
    _cleanup_old_jobs()
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    # If user opted in: when the URL changes from the previous submission,
    # delete the prior video's work_dir and saved files (frees multi-GB).
    _maybe_purge_previous_url(url, enabled=getattr(req, "purge_on_new_url", False))

    # Handle dub chain: e.g. ["en", "hi"] means dub to English first, then Hindi
    if req.dub_chain and len(req.dub_chain) >= 2:
        first_lang = req.dub_chain[0]
        remaining = req.dub_chain[1:]
        req.target_language = first_lang
        # Force YouTube subs for first step (use existing English subs)
        if first_lang == "en":
            req.prefer_youtube_subs = True
    else:
        remaining = []

    job_id = uuid.uuid4().hex[:12]
    job = Job(
        id=job_id,
        source_url=url,
        target_language=req.target_language,
        chain_languages=remaining,
    )
    job.original_req = req
    JOBS[job_id] = job
    _store.save(job)

    t = threading.Thread(target=_run_job, args=(job, req), daemon=True)
    t.start()
    job.worker_thread = t

    return {"id": job_id}


@app.post("/api/jobs/upload")
async def create_job_upload(
    file: UploadFile = File(...),
    source_language: str = Form("en"),
    target_language: str = Form("hi"),
    asr_model: str = Form("groq-whisper"),
    translation_engine: str = Form("google"),
    tts_rate: str = Form("+0%"),
    mix_original: str = Form("false"),
    original_volume: float = Form(0.10),
    use_cosyvoice: str = Form("false"),
    use_chatterbox: str = Form("false"),
    use_indic_parler: str = Form("false"),
    use_sarvam_bulbul: str = Form("false"),
    use_elevenlabs: str = Form("false"),
    use_google_tts: str = Form("false"),
    use_coqui_xtts: str = Form("false"),
    use_edge_tts: str = Form("true"),
    prefer_youtube_subs: str = Form("true"),
    use_yt_translate: str = Form("false"),
    multi_speaker: str = Form("false"),
    transcribe_only: str = Form("false"),
    audio_priority: str = Form("true"),
    audio_untouchable: str = Form("false"),
    video_slow_to_match: str = Form("true"),
    post_tts_level: str = Form("minimal"),
    enable_sentence_gap: str = Form("false"),
    enable_duration_fit: str = Form("false"),
    audio_bitrate: str = Form("192k"),
    encode_preset: str = Form("fast"),
    split_duration: int = Form(30),
    dub_duration: int = Form(0),
    fast_assemble: str = Form("false"),
    enable_manual_review: str = Form("false"),
    use_whisperx: str = Form("true"),
    simplify_english: str = Form("false"),
    step_by_step: str = Form("false"),
    voice: str = Form("hi-IN-SwaraNeural"),
    # ── AV Sync Modules ──
    av_sync_mode: str = Form("original"),
    max_audio_speedup: float = Form(1.30),
    min_video_speed: float = Form(0.70),
    slot_verify: str = Form("off"),
    use_global_stretch: str = Form("false"),
    global_stretch_speedup: float = Form(1.25),
    segmenter: str = Form("dp"),
    segmenter_buffer_pct: float = Form(0.20),
    max_sentences_per_cue: int = Form(2),
    yt_transcript_mode: str = Form("yt_timeline"),
    yt_segment_mode: str = Form("sentence"),
    yt_text_correction: str = Form("true"),
    yt_replace_mode: str = Form("diff"),
    tts_chunk_words: int = Form(0),
    gap_mode: str = Form("micro"),
    preset_name: str = Form(""),
    # Pipeline mode + mode-specific fields (WordChunk + SRT Direct)
    pipeline_mode: str = Form("classic"),
    wc_chunk_size: int = Form(8),
    wc_max_stretch: float = Form(20.0),
    wc_transcript: str = Form(""),
    sd_srt_content: str = Form(""),
    sd_max_stretch: float = Form(20.0),
):
    """Create a dubbing job from an uploaded video file."""
    _cleanup_old_jobs()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    work_dir = job_dir / "work"
    work_dir.mkdir(exist_ok=True)

    # Save uploaded file
    ext = Path(file.filename).suffix or ".mp4"
    saved_path = work_dir / f"source{ext}"
    with open(saved_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)

    # Validate via Pydantic BEFORE adding to JOBS to prevent zombie entries
    def _bool(v) -> bool:
        return str(v).lower() in ("true", "1", "yes")

    try:
        req = JobCreateRequest(
            url=str(saved_path),
            source_language=source_language,
            target_language=target_language,
            voice=voice,
            asr_model=asr_model,
            translation_engine=translation_engine,
            tts_rate=tts_rate,
            mix_original=_bool(mix_original),
            original_volume=original_volume,
            use_cosyvoice=_bool(use_cosyvoice), use_chatterbox=_bool(use_chatterbox),
            use_indic_parler=_bool(use_indic_parler),
            use_sarvam_bulbul=_bool(use_sarvam_bulbul),
            use_elevenlabs=_bool(use_elevenlabs),
            use_google_tts=_bool(use_google_tts),
            use_coqui_xtts=_bool(use_coqui_xtts),
            use_fish_speech=False,
            use_edge_tts=_bool(use_edge_tts),
            prefer_youtube_subs=_bool(prefer_youtube_subs),
            use_yt_translate=_bool(use_yt_translate),
            multi_speaker=_bool(multi_speaker),
            transcribe_only=_bool(transcribe_only),
            audio_priority=_bool(audio_priority),
            audio_untouchable=_bool(audio_untouchable),
            video_slow_to_match=_bool(video_slow_to_match),
            post_tts_level=post_tts_level,
            audio_quality_mode="fast",
            enable_sentence_gap=_bool(enable_sentence_gap),
            enable_duration_fit=_bool(enable_duration_fit),
            audio_bitrate=audio_bitrate,
            encode_preset=encode_preset,
            split_duration=split_duration,
            dub_duration=dub_duration,
            fast_assemble=_bool(fast_assemble),
            enable_manual_review=_bool(enable_manual_review),
            use_whisperx=_bool(use_whisperx),
            simplify_english=_bool(simplify_english),
            step_by_step=_bool(step_by_step),
            srt_needs_translation=_bool(srt_needs_translation) if 'srt_needs_translation' in dir() else False,
            # ── AV Sync Modules ──
            av_sync_mode=av_sync_mode,
            max_audio_speedup=max_audio_speedup,
            min_video_speed=min_video_speed,
            slot_verify=slot_verify,
            use_global_stretch=_bool(use_global_stretch),
            global_stretch_speedup=global_stretch_speedup,
            segmenter=segmenter,
            segmenter_buffer_pct=segmenter_buffer_pct,
            max_sentences_per_cue=max_sentences_per_cue,
            yt_transcript_mode=yt_transcript_mode,
            yt_segment_mode=yt_segment_mode,
            yt_text_correction=_bool(yt_text_correction),
            yt_replace_mode=yt_replace_mode,
            tts_chunk_words=tts_chunk_words,
            gap_mode=gap_mode,
            preset_name=preset_name,
            pipeline_mode=pipeline_mode,
            wc_chunk_size=wc_chunk_size,
            wc_max_stretch=wc_max_stretch,
            wc_transcript=wc_transcript,
            sd_srt_content=sd_srt_content,
            sd_max_stretch=sd_max_stretch,
        )
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    job = Job(id=job_id, source_url=f"upload:{file.filename}", target_language=target_language)
    job.original_req = req
    JOBS[job_id] = job
    _store.save(job)

    t = threading.Thread(target=_run_job, args=(job, req), daemon=True)
    t.start()
    job.worker_thread = t

    return {"id": job_id}


def _run_job_with_srt(job: Job, req: JobCreateRequest, srt_path: Path):
    """Download/extract the video, then run TTS+assembly from the provided SRT."""
    # ── Apply same overrides as _run_job so behavior is consistent ──
    req.use_edge_tts = True
    req.use_cosyvoice = req.use_chatterbox = req.use_indic_parler = False
    req.use_sarvam_bulbul = req.use_elevenlabs = req.use_google_tts = False
    req.use_coqui_xtts = req.use_fish_speech = False
    # Voice: auto-select per target language (same logic as _run_job)
    _PREFERRED_SRT = {"hi": "hi-IN-MadhurNeural"}
    if not req.voice or req.voice == "hi-IN-SwaraNeural":
        req.voice = _PREFERRED_SRT.get(
            req.target_language,
            DEFAULT_VOICES.get(req.target_language, "hi-IN-MadhurNeural"),
        )
    if getattr(req, 'tts_no_time_pressure', True):
        req.audio_priority = True
        req.enable_duration_fit = False

    job.message = "Waiting in queue..."
    _pipeline_semaphore.acquire()
    try:
        job.state = "running"
        job.message = "Starting (SRT provided, skipping transcription)..."
        _store.save(job)

        job_dir = OUTPUTS / job.id
        work_dir = job_dir / "work"
        out_path = job_dir / "dubbed.mp4"

        voice = req.voice

        cfg = PipelineConfig(
            source=req.url,
            work_dir=work_dir,
            output_path=out_path,
            source_language=req.source_language,
            target_language=req.target_language,
            asr_model=req.asr_model,
            translation_engine=req.translation_engine,
            tts_voice=voice,
            tts_rate=req.tts_rate,
            mix_original=req.mix_original,
            original_volume=req.original_volume,
            use_cosyvoice=req.use_cosyvoice,
            use_chatterbox=req.use_chatterbox,
            use_indic_parler=req.use_indic_parler,
            use_sarvam_bulbul=req.use_sarvam_bulbul,
            use_elevenlabs=req.use_elevenlabs,
            use_google_tts=req.use_google_tts,
            use_coqui_xtts=req.use_coqui_xtts,
            use_fish_speech=req.use_fish_speech,
            use_edge_tts=req.use_edge_tts,
            prefer_youtube_subs=req.prefer_youtube_subs,
            use_yt_translate=req.use_yt_translate,
            multi_speaker=req.multi_speaker,
            transcribe_only=req.transcribe_only,
            audio_priority=req.audio_priority,
            audio_untouchable=req.audio_untouchable,
            video_slow_to_match=req.video_slow_to_match,
            post_tts_level=req.post_tts_level,
            audio_quality_mode=req.audio_quality_mode,
            enable_sentence_gap=req.enable_sentence_gap,
            enable_duration_fit=req.enable_duration_fit,
            audio_bitrate=req.audio_bitrate,
            encode_preset=req.encode_preset,
            download_mode=getattr(req, 'download_mode', 'remux'),
            dub_duration=req.dub_duration,
            fast_assemble=req.fast_assemble,
            enable_manual_review=req.enable_manual_review,
            use_whisperx=req.use_whisperx,
            simplify_english=req.simplify_english,
            step_by_step=req.step_by_step,
            enable_tts_verify_retry=req.enable_tts_verify_retry,
            tts_truncation_threshold=getattr(req, 'tts_truncation_threshold', 0.30),
            keep_subject_english=getattr(req, 'keep_subject_english', False),
            tts_word_match_verify=getattr(req, 'tts_word_match_verify', False),
            tts_word_match_tolerance=getattr(req, 'tts_word_match_tolerance', 0.15),
            tts_word_match_model=getattr(req, 'tts_word_match_model', 'auto'),
            tts_word_match_max_segments=getattr(req, 'tts_word_match_max_segments', 1000),
            use_wav2lip=getattr(req, 'use_wav2lip', False),
            long_segment_trace=getattr(req, 'long_segment_trace', True),
            long_segment_threshold_words=getattr(req, 'long_segment_threshold_words', 15),
            tts_no_time_pressure=getattr(req, 'tts_no_time_pressure', True),
            tts_dynamic_workers=getattr(req, 'tts_dynamic_workers', True),
            tts_dynamic_min=getattr(req, 'tts_dynamic_min', 10),
            tts_dynamic_max=getattr(req, 'tts_dynamic_max', 120),
            tts_dynamic_start=getattr(req, 'tts_dynamic_start', 30),
            tts_rate_mode=getattr(req, 'tts_rate_mode', 'auto'),
            tts_rate_ceiling=getattr(req, 'tts_rate_ceiling', '+25%'),
            tts_rate_target_wpm=getattr(req, 'tts_rate_target_wpm', 130),
            srt_needs_translation=req.srt_needs_translation,
            # ── AV Sync Modules ──
            av_sync_mode=getattr(req, 'av_sync_mode', 'original'),
            max_audio_speedup=getattr(req, 'max_audio_speedup', 1.30),
            min_video_speed=getattr(req, 'min_video_speed', 0.70),
            slot_verify=getattr(req, 'slot_verify', 'off'),
            use_global_stretch=getattr(req, 'use_global_stretch', False),
            global_stretch_speedup=getattr(req, 'global_stretch_speedup', 1.25),
            segmenter=getattr(req, 'segmenter', 'dp'),
            segmenter_buffer_pct=getattr(req, 'segmenter_buffer_pct', 0.20),
            max_sentences_per_cue=getattr(req, 'max_sentences_per_cue', 2),
            yt_transcript_mode=getattr(req, 'yt_transcript_mode', 'yt_timeline'),
            yt_segment_mode=getattr(req, 'yt_segment_mode', 'sentence'),
            yt_text_correction=getattr(req, 'yt_text_correction', True),
            yt_replace_mode=getattr(req, 'yt_replace_mode', 'diff'),
            tts_chunk_words=getattr(req, 'tts_chunk_words', 0),
            gap_mode=getattr(req, 'gap_mode', 'micro'),
        )

        pipeline = Pipeline(cfg, on_progress=_make_progress_callback(job),
                           cancel_check=job.cancel_event.is_set)
        job.pipeline_ref = pipeline   # cancel handler can kill in-flight subprocesses

        # Step 1-2: Download + extract audio (pipeline handles this)
        pipeline.download_and_extract()

        job.video_title = pipeline.video_title or "Untitled"

        # Check if SRT needs translation (source English) or is already translated
        srt_needs_translation = getattr(req, 'srt_needs_translation', False)
        if srt_needs_translation:
            # English SRT → translate → TTS → assembly
            pipeline.run_from_source_srt(srt_path)
        else:
            # Already translated SRT → TTS + assembly only
            pipeline.run_from_srt(srt_path)

        if not out_path.exists():
            raise RuntimeError("Pipeline finished but output file not found")

        job.result_path = out_path
        job.segments = pipeline.segments

        # Auto-save to titled folder
        try:
            _save_to_titled_folder(job)
        except Exception as save_err:
            print(f"[WARN] Failed to save to titled folder: {save_err}")

        # Generate YouTube description
        try:
            job.description = _generate_youtube_description(job)
            if job.saved_folder:
                desc_path = Path(job.saved_folder) / "description.txt"
                desc_path.write_text(job.description, encoding="utf-8")
        except Exception:
            pass

        # Send SRT pair to hinglish-ai trainer for auto-training
        try:
            work_dir = OUTPUTS / job.id / "work"
            source_srt = work_dir / "transcript_source.srt"
            if source_srt.exists() and srt_path.exists():
                _send_training_data(source_srt, srt_path, source_lang=req.source_language)
        except Exception:
            pass

        job.overall_progress = 1.0
        job.state = "done"
        job.message = "Complete"
        job.events.append({"type": "complete", "state": "done"})
        _store.save(job)

        if job.source_url:
            _mark_url_completed(job.source_url)

    except Exception as e:
        import traceback
        try:
            (OUTPUTS / job.id / "error.log").write_text(
                f"[SRT-JOB ERROR] {e}\n{traceback.format_exc()}", encoding="utf-8"
            )
        except OSError:
            pass
        try:
            job_dir = OUTPUTS / job.id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            pass
        job.state = "error"
        job.error = str(e)
        job.message = f"Error: {e}"
        job.events.append({"type": "complete", "state": "error", "error": str(e)})
        _store.save(job)
    finally:
        try:
            _purge_global_caches()
        except Exception:
            pass
        _pipeline_semaphore.release()


@app.post("/api/jobs/with-srt")
async def create_job_with_srt(
    srt_file: UploadFile = File(...),
    url: str = Form(""),
    video_file: Optional[UploadFile] = File(None),
    source_language: str = Form("en"),
    target_language: str = Form("hi"),
    asr_model: str = Form("groq-whisper"),
    translation_engine: str = Form("google"),
    tts_rate: str = Form("+0%"),
    mix_original: str = Form("false"),
    original_volume: float = Form(0.10),
    use_cosyvoice: str = Form("false"),
    use_chatterbox: str = Form("false"),
    use_indic_parler: str = Form("false"),
    use_sarvam_bulbul: str = Form("false"),
    use_elevenlabs: str = Form("false"),
    use_google_tts: str = Form("false"),
    use_coqui_xtts: str = Form("false"),
    use_edge_tts: str = Form("true"),
    prefer_youtube_subs: str = Form("true"),
    use_yt_translate: str = Form("false"),
    multi_speaker: str = Form("false"),
    audio_priority: str = Form("true"),
    audio_untouchable: str = Form("false"),
    video_slow_to_match: str = Form("true"),
    post_tts_level: str = Form("minimal"),
    enable_sentence_gap: str = Form("false"),
    enable_duration_fit: str = Form("false"),
    audio_bitrate: str = Form("192k"),
    encode_preset: str = Form("fast"),
    split_duration: int = Form(30),
    dub_duration: int = Form(0),
    fast_assemble: str = Form("false"),
    enable_manual_review: str = Form("false"),
    use_whisperx: str = Form("true"),
    simplify_english: str = Form("false"),
    step_by_step: str = Form("false"),
    srt_needs_translation: str = Form("false"),
    voice: str = Form("hi-IN-SwaraNeural"),
    # ── AV Sync Modules ──
    av_sync_mode: str = Form("original"),
    max_audio_speedup: float = Form(1.30),
    min_video_speed: float = Form(0.70),
    slot_verify: str = Form("off"),
    use_global_stretch: str = Form("false"),
    global_stretch_speedup: float = Form(1.25),
    segmenter: str = Form("dp"),
    segmenter_buffer_pct: float = Form(0.20),
    max_sentences_per_cue: int = Form(2),
    yt_transcript_mode: str = Form("yt_timeline"),
    yt_segment_mode: str = Form("sentence"),
    yt_text_correction: str = Form("true"),
    yt_replace_mode: str = Form("diff"),
    tts_chunk_words: int = Form(0),
    gap_mode: str = Form("micro"),
    preset_name: str = Form(""),
    # Pipeline mode + mode-specific fields (WordChunk + SRT Direct)
    pipeline_mode: str = Form("classic"),
    wc_chunk_size: int = Form(8),
    wc_max_stretch: float = Form(20.0),
    wc_transcript: str = Form(""),
    sd_srt_content: str = Form(""),
    sd_max_stretch: float = Form(20.0),
):
    """Create a dubbing job from a video (URL or file) + SRT file.
    If srt_needs_translation=true: English SRT → translate → TTS → assembly.
    If srt_needs_translation=false: Already translated SRT → TTS → assembly.
    Skips transcription and translation — goes straight to TTS + assembly."""
    _cleanup_old_jobs()

    if not url and (not video_file or not video_file.filename):
        raise HTTPException(status_code=400, detail="Provide either a URL or a video file")
    if not srt_file.filename:
        raise HTTPException(status_code=400, detail="SRT file is required")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    work_dir = job_dir / "work"
    work_dir.mkdir(exist_ok=True)

    # Determine video source
    source_url = url.strip()
    if video_file and video_file.filename:
        # Save uploaded video
        ext = Path(video_file.filename).suffix or ".mp4"
        saved_video_path = work_dir / f"source{ext}"
        with open(saved_video_path, "wb") as f:
            while chunk := await video_file.read(1024 * 1024):
                f.write(chunk)
        source_url = str(saved_video_path)
        display_source = f"upload:{video_file.filename}"
    else:
        display_source = source_url

    # Save uploaded SRT
    srt_path = work_dir / "translated_upload.srt"
    with open(srt_path, "wb") as f:
        while chunk := await srt_file.read(1024 * 1024):
            f.write(chunk)

    def _bool(v) -> bool:
        return str(v).lower() in ("true", "1", "yes")

    try:
        req = JobCreateRequest(
            url=source_url,
            source_language=source_language,
            target_language=target_language,
            voice=voice,
            asr_model=asr_model,
            translation_engine=translation_engine,
            tts_rate=tts_rate,
            mix_original=_bool(mix_original),
            original_volume=original_volume,
            use_cosyvoice=_bool(use_cosyvoice), use_chatterbox=_bool(use_chatterbox),
            use_indic_parler=_bool(use_indic_parler),
            use_sarvam_bulbul=_bool(use_sarvam_bulbul),
            use_elevenlabs=_bool(use_elevenlabs),
            use_google_tts=_bool(use_google_tts),
            use_coqui_xtts=_bool(use_coqui_xtts),
            use_fish_speech=False,
            use_edge_tts=_bool(use_edge_tts),
            prefer_youtube_subs=_bool(prefer_youtube_subs),
            use_yt_translate=_bool(use_yt_translate),
            multi_speaker=_bool(multi_speaker),
            audio_priority=_bool(audio_priority),
            audio_untouchable=_bool(audio_untouchable),
            video_slow_to_match=_bool(video_slow_to_match),
            post_tts_level=post_tts_level,
            audio_quality_mode="fast",
            enable_sentence_gap=_bool(enable_sentence_gap),
            enable_duration_fit=_bool(enable_duration_fit),
            audio_bitrate=audio_bitrate,
            encode_preset=encode_preset,
            split_duration=split_duration,
            dub_duration=dub_duration,
            fast_assemble=_bool(fast_assemble),
            enable_manual_review=_bool(enable_manual_review),
            use_whisperx=_bool(use_whisperx),
            simplify_english=_bool(simplify_english),
            step_by_step=_bool(step_by_step),
            srt_needs_translation=_bool(srt_needs_translation) if 'srt_needs_translation' in dir() else False,
            # ── AV Sync Modules ──
            av_sync_mode=av_sync_mode,
            max_audio_speedup=max_audio_speedup,
            min_video_speed=min_video_speed,
            slot_verify=slot_verify,
            use_global_stretch=_bool(use_global_stretch),
            global_stretch_speedup=global_stretch_speedup,
            segmenter=segmenter,
            segmenter_buffer_pct=segmenter_buffer_pct,
            max_sentences_per_cue=max_sentences_per_cue,
            yt_transcript_mode=yt_transcript_mode,
            yt_segment_mode=yt_segment_mode,
            yt_text_correction=_bool(yt_text_correction),
            yt_replace_mode=yt_replace_mode,
            tts_chunk_words=tts_chunk_words,
            gap_mode=gap_mode,
            preset_name=preset_name,
            pipeline_mode=pipeline_mode,
            wc_chunk_size=wc_chunk_size,
            wc_max_stretch=wc_max_stretch,
            wc_transcript=wc_transcript,
            sd_srt_content=sd_srt_content,
            sd_max_stretch=sd_max_stretch,
        )
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    job = Job(id=job_id, source_url=display_source, target_language=target_language)
    job.original_req = req
    JOBS[job_id] = job
    _store.save(job)

    t = threading.Thread(target=_run_job_with_srt, args=(job, req, srt_path), daemon=True)
    t.start()
    job.worker_thread = t

    return {"id": job_id}


def _job_config(job: Job) -> Dict[str, Any]:
    """Extract the config/settings used for this job."""
    try:
        return _job_config_inner(job)
    except Exception:
        return {"error": "Failed to read job config"}

def _job_config_inner(job: Job) -> Dict[str, Any]:
    req = job.original_req
    if not req:
        return {}
    if getattr(req, 'use_coqui_xtts', False) and getattr(req, 'use_edge_tts', False):
        sarvam_on = getattr(req, 'use_sarvam_bulbul', False)
        if sarvam_on:
            tts = "Triple Parallel (XTTS+Sarvam+Edge)"
        else:
            tts = "Dual Parallel (XTTS+Edge)"
    elif getattr(req, 'use_chatterbox', False):
        tts = "Chatterbox"
    elif getattr(req, 'use_sarvam_bulbul', False):
        tts = "Sarvam Bulbul v3"
    elif getattr(req, 'use_indic_parler', False):
        tts = "Indic Parler-TTS"
    elif getattr(req, 'use_elevenlabs', False):
        tts = "ElevenLabs"
    elif getattr(req, 'use_cosyvoice', False):
        tts = "CosyVoice 2"
    elif getattr(req, 'use_coqui_xtts', False):
        tts = "Coqui XTTS"
    elif getattr(req, 'use_google_tts', False):
        tts = "Google TTS"
    else:
        tts = "Edge-TTS"
    engine_labels = {
        "auto":         "Auto",
        "gemma4":       "Gemma 4 (31B)",
        "turbo":        "Turbo (Groq+SambaNova)",
        "gemini":       "Gemini",
        "groq":         "Groq",
        "sambanova":    "SambaNova",
        "google_polish":"Google+LLM Polish",
        "ollama":       "Ollama",
        "google":       "Google Translate",
        "hinglish":     "Hinglish AI",
        "nllb_polish":  "IndicTrans2+",
        "nllb":         "IndicTrans2",
        "chain_dub":    "Chain Dub",
        "seamless":     "SeamlessM4T v2",
    }
    # Show actual transcription source, not just Whisper model
    pipeline_mode = getattr(req, "pipeline_mode", "classic")
    if pipeline_mode == "new":
        _asr = (getattr(req, "asr_model", "parakeet") or "parakeet").lower()
        if _asr == "parakeet":
            asr_label = "Parakeet + Whisper large-v3 + DP Cues"
        else:
            # Whisper-only run at the size the user picked.
            asr_label = f"Whisper {_asr} + DP Cues"
    elif getattr(req, "use_yt_translate", False):
        asr_label = "YouTube Auto-Translate"
    elif getattr(req, "prefer_youtube_subs", False):
        asr_label = "YouTube Subtitles"
    elif pipeline_mode == "hybrid":
        asr_label = f"Whisper {getattr(req, 'asr_model', 'large-v3')} + DP Cues"
    else:
        asr_label = f"Whisper {getattr(req, 'asr_model', 'large-v3')}"

    return {
        # ── Core labels (always shown) ──
        "asr_model": asr_label,
        "translation_engine": engine_labels.get(getattr(req, "translation_engine", "auto"), "Auto"),
        "tts_engine": tts,
        "tts_rate": getattr(req, "tts_rate", "+0%"),
        "audio_bitrate": getattr(req, "audio_bitrate", "192k"),
        "encode_preset": getattr(req, "encode_preset", "veryfast"),
        "post_tts_level": getattr(req, "post_tts_level", "full"),
        "audio_quality_mode": getattr(req, "audio_quality_mode", "fast"),
        "download_mode": getattr(req, "download_mode", "remux"),
        "split_duration": getattr(req, "split_duration", 30),
        "dub_duration": getattr(req, "dub_duration", 0),
        "pipeline_mode": getattr(req, "pipeline_mode", "classic"),
        "source_language": getattr(req, "source_language", "en"),
        "target_language": getattr(req, "target_language", "hi"),
        "voice": getattr(req, "voice", "hi-IN-SwaraNeural"),
        "original_volume": getattr(req, "original_volume", 0.10),
        # ── Boolean toggles ──
        "audio_priority": getattr(req, "audio_priority", False),
        "audio_untouchable": getattr(req, "audio_untouchable", False),
        "video_slow_to_match": getattr(req, "video_slow_to_match", True),
        "mix_original": getattr(req, "mix_original", False),
        "multi_speaker": getattr(req, "multi_speaker", False),
        "fast_assemble": getattr(req, "fast_assemble", False),
        "enable_sentence_gap": getattr(req, "enable_sentence_gap", True),
        "enable_duration_fit": getattr(req, "enable_duration_fit", True),
        "prefer_youtube_subs": getattr(req, "prefer_youtube_subs", False),
        "use_yt_translate": getattr(req, "use_yt_translate", False),
        "use_whisperx": getattr(req, "use_whisperx", False),
        "simplify_english": getattr(req, "simplify_english", True),
        "enable_manual_review": getattr(req, "enable_manual_review", True),
        "transcribe_only": getattr(req, "transcribe_only", False),
        "use_sarvam_bulbul": getattr(req, "use_sarvam_bulbul", False),
        "enable_tts_verify_retry": getattr(req, "enable_tts_verify_retry", False),
        "keep_subject_english": getattr(req, "keep_subject_english", False),
        "tts_word_match_verify": getattr(req, "tts_word_match_verify", True),
        "long_segment_trace": getattr(req, "long_segment_trace", True),
        "tts_no_time_pressure": getattr(req, "tts_no_time_pressure", True),
        "tts_dynamic_workers": getattr(req, "tts_dynamic_workers", True),
        "purge_on_new_url": getattr(req, "purge_on_new_url", False),
        "step_by_step": getattr(req, "step_by_step", False),
        "use_new_pipeline": getattr(req, "use_new_pipeline", False),
        "srt_needs_translation": getattr(req, "srt_needs_translation", False),
        # ── Numeric thresholds ──
        "tts_truncation_threshold": getattr(req, "tts_truncation_threshold", 0.30),
        "tts_word_match_tolerance": getattr(req, "tts_word_match_tolerance", 0.15),
        "tts_word_match_model": getattr(req, "tts_word_match_model", "auto"),
        "long_segment_threshold_words": getattr(req, "long_segment_threshold_words", 15),
        "tts_dynamic_min": getattr(req, "tts_dynamic_min", 10),
        "tts_dynamic_max": getattr(req, "tts_dynamic_max", 120),
        "tts_dynamic_start": getattr(req, "tts_dynamic_start", 30),
        # ── AV Sync Modules ──
        "preset_name": getattr(req, "preset_name", ""),
        "av_sync_mode": getattr(req, "av_sync_mode", "original"),
        "max_audio_speedup": getattr(req, "max_audio_speedup", 1.30),
        "min_video_speed": getattr(req, "min_video_speed", 0.70),
        "slot_verify": getattr(req, "slot_verify", "off"),
        "use_global_stretch": getattr(req, "use_global_stretch", False),
        "global_stretch_speedup": getattr(req, "global_stretch_speedup", 1.25),
        "segmenter": getattr(req, "segmenter", "dp"),
        "segmenter_buffer_pct": getattr(req, "segmenter_buffer_pct", 0.20),
        "max_sentences_per_cue": getattr(req, "max_sentences_per_cue", 2),
        "yt_transcript_mode": getattr(req, "yt_transcript_mode", "yt_timeline"),
        "yt_segment_mode": getattr(req, "yt_segment_mode", "sentence"),
        "yt_text_correction": getattr(req, "yt_text_correction", True),
        "yt_replace_mode": getattr(req, "yt_replace_mode", "diff"),
        "tts_chunk_words": getattr(req, "tts_chunk_words", 0),
        "gap_mode": getattr(req, "gap_mode", "micro"),
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "state": job.state,
        "current_step": job.current_step,
        "step_progress": round(job.step_progress, 3),
        "overall_progress": round(job.overall_progress, 3),
        "step_times": job.step_times,
        "message": job.message,
        "error": job.error,
        "source_url": job.source_url,
        "video_title": job.video_title,
        "target_language": job.target_language,
        "created_at": job.created_at,
        "config": _job_config(job),
        "saved_folder": job.saved_folder,
        "saved_video": job.saved_video,
        "description": job.description,
        "qa_score": job.qa_score,
        "chain_languages": job.chain_languages,
        "chain_parent_id": job.chain_parent_id,
        # Word/sentence budget — populated by _pretts_word_budget after TTS finishes
        "total_words":        job.total_words,
        "total_sentences":    job.total_sentences,
        "avg_words_per_sent": job.avg_words_per_sent,
        "max_seg_words":      job.max_seg_words,
        "max_sent_words":     job.max_sent_words,
    }


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """SSE endpoint for real-time progress updates."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_index = 0
        sse_start = time.time()
        max_sse_secs = 1440000  # 400h max SSE connection
        while True:
            if time.time() - sse_start > max_sse_secs:
                yield {"data": json.dumps({"type": "complete", "state": "error", "error": "SSE timeout"})}
                return
            # Detect list reset (e.g. resume cleared events) — resync
            if last_index > len(job.events):
                last_index = 0
            if last_index < len(job.events):
                for event in job.events[last_index:]:
                    yield {"data": json.dumps(event)}
                    if event.get("type") == "complete":
                        return
                last_index = len(job.events)
            if job.state in ("done", "error", "waiting_for_srt"):
                if last_index >= len(job.events):
                    yield {"data": json.dumps({"type": "complete", "state": job.state})}
                    return
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.get("/api/jobs/{job_id}/transcript")
def get_transcript(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # Prefer live pipeline segments (has latest state during step-by-step review)
    if job.pipeline_ref and job.pipeline_ref.segments:
        return {"segments": job.pipeline_ref.segments}
    return {"segments": job.segments}


class CompareRequest(BaseModel):
    engines: List[str] = []  # e.g. ["groq", "gemini", "google"] — empty = all available
    max_segments: int = 10


@app.post("/api/jobs/{job_id}/compare-translations")
def compare_translations(job_id: str, body: CompareRequest):
    """Run selected translation engines on a sample of segments for comparison.

    Returns: { engines: { engine_label: [{ text, text_translated }] }, available: [...] }
    """
    import copy
    import traceback

    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get segments from pipeline or job
    segments = []
    if job.pipeline_ref and job.pipeline_ref.segments:
        segments = job.pipeline_ref.segments
    elif job.segments:
        segments = job.segments

    if not segments:
        raise HTTPException(status_code=400, detail="No transcribed segments available")

    # Detect available engines
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    all_engines = {}
    if groq_key:
        all_engines["groq"] = "Groq"
    if sambanova_key:
        all_engines["sambanova"] = "SambaNova"
    if gemini_key:
        all_engines["gemini"] = "Gemini"
    all_engines["google"] = "Google"
    if groq_key or sambanova_key or gemini_key:
        all_engines["google_polish"] = "Google+Polish"
    # IndicTrans2 (raw local meaning model — no LLM key needed, runs on GPU)
    all_engines["nllb"] = "IndicTrans2"
    # IndicTrans2+ (local model + LLM polish — needs at least one LLM key)
    if groq_key or sambanova_key or gemini_key:
        all_engines["nllb_polish"] = "IndicTrans2+"
        all_engines["chain_dub"] = "Chain Dub"
    if groq_key and sambanova_key:
        all_engines["turbo"] = "Turbo"

    # If max_segments=0, just return available engines (no translation)
    if body.max_segments <= 0:
        return {
            "engines": {},
            "segment_count": 0,
            "available": [{"key": k, "label": v} for k, v in all_engines.items()],
        }

    sample = [s for s in segments if s.get("text", "").strip()][:body.max_segments]
    if not sample:
        raise HTTPException(status_code=400, detail="No text segments found")

    # Filter to selected engines (or use all if none specified)
    if body.engines:
        engines_to_run = [(all_engines[k], k) for k in body.engines if k in all_engines]
    else:
        engines_to_run = [(label, key) for key, label in all_engines.items()]

    if not engines_to_run:
        raise HTTPException(status_code=400, detail="No valid engines selected")

    req = job.original_req
    target_lang = req.target_language if req else "hi"
    source_lang = (req.source_language if req and req.source_language != "auto" else "en")

    from pipeline import PipelineConfig, Pipeline

    # Reuse existing work dir so IndicTrans2 model cache works
    work_dir = OUTPUTS / job_id / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    def run_engine(label, engine_key):
        try:
            # Redirect stdout/stderr to a UTF-8 string buffer for this thread
            # This prevents Windows cp1252 crashes when Hindi text is printed
            import sys as _sys, io as _io
            _old_stdout, _old_stderr = _sys.stdout, _sys.stderr
            _sys.stdout = _io.StringIO()
            _sys.stderr = _io.StringIO()
            try:
                cfg = PipelineConfig(
                    source="compare",
                    target_language=target_lang,
                    source_language=source_lang,
                    translation_engine=engine_key,
                    work_dir=work_dir,
                    output_path=work_dir / f"compare_{engine_key}.mp4",
                )
                p = Pipeline(cfg)
                engine_segs = [copy.deepcopy(s) for s in sample]
                for s in engine_segs:
                    s.pop("text_translated", None)

                p._translate_segments(engine_segs)
            finally:
                _sys.stdout, _sys.stderr = _old_stdout, _old_stderr

            return label, [
                {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", ""),
                 "text_translated": s.get("text_translated", "")}
                for s in engine_segs
            ]
        except Exception as e:
            try:
                traceback.print_exc()
            except Exception:
                pass
            return label, [
                {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", ""),
                 "text_translated": f"[ERROR: {str(e)[:80]}]"}
                for s in sample
            ]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(run_engine, label, key): label for label, key in engines_to_run}
            for fut in as_completed(futures):
                label, translated = fut.result()
                results[label] = translated
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compare failed: {str(e)[:200]}")

    return {
        "engines": results,
        "segment_count": len(sample),
        "available": [{"key": k, "label": v} for k, v in all_engines.items()],
    }


@app.post("/api/jobs/{job_id}/continue")
def continue_job(job_id: str):
    """Resume a step-by-step job after reviewing transcription or translation."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state not in ("review_transcription", "review_translation"):
        raise HTTPException(status_code=400, detail=f"Job is not paused for review (state={job.state})")
    job.state = "running"
    job.pause_event.set()  # Unblock the pipeline thread
    return {"status": "resumed", "from_state": job.state}


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state != "done" or not job.result_path:
        raise HTTPException(status_code=409, detail="Job not complete")
    if not job.result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found (cleaned up)")

    title = _sanitize_filename(job.video_title) if job.video_title else f"dubbed_{job_id}"
    return FileResponse(
        path=str(job.result_path),
        media_type="video/mp4",
        filename=f"{title} - Dubbed.mp4",
    )


@app.get("/api/jobs/{job_id}/srt")
def get_srt(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    srt_path = OUTPUTS / job_id / f"subtitles_{job.target_language}.srt"
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="Subtitles not found")

    return FileResponse(
        path=str(srt_path),
        media_type="text/plain",
        filename=f"subtitles_{job_id}.srt",
    )


@app.get("/api/jobs/{job_id}/source-srt")
def get_source_srt(job_id: str):
    """Download the source-language SRT (for manual translation)."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    srt_path = OUTPUTS / job_id / "work" / "transcript_source.srt"
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="Source SRT not found — run with Transcribe Only first")

    return FileResponse(
        path=str(srt_path),
        media_type="text/plain",
        filename=f"source_{job_id}.srt",
    )


@app.get("/api/jobs/{job_id}/qa")
def get_qa_report(job_id: str):
    """Get the QA report for a job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Try work dir first, then saved folder
    qa_path = OUTPUTS / job_id / "work" / "qa_report.txt"
    if not qa_path.exists() and job.saved_folder:
        qa_path = Path(job.saved_folder) / "qa_report.txt"
    if not qa_path.exists():
        raise HTTPException(status_code=404, detail="QA report not available")

    report = qa_path.read_text(encoding="utf-8")
    return {"qa_score": job.qa_score, "report": report}


def _run_resume(job: Job):
    """Resume pipeline from uploaded translated SRT in a background thread."""
    job.message = "Waiting in queue..."
    _pipeline_semaphore.acquire()
    try:
        job.state = "running"
        job.message = "Resuming from uploaded SRT..."
        _store.save(job)
        job.events.clear()  # Clear in-place so SSE generators tracking last_index stay consistent

        job_dir = OUTPUTS / job.id
        work_dir = job_dir / "work"
        out_path = job_dir / "dubbed.mp4"
        translated_srt = work_dir / "translated_upload.srt"

        # Restore TTS settings from original request.
        # Apply same hard locks as _run_job for consistency.
        req = job.original_req
        # Voice: auto-select per target language (same as _run_job)
        _PREFERRED_RESUME = {"hi": "hi-IN-MadhurNeural"}
        voice = _PREFERRED_RESUME.get(
            job.target_language,
            DEFAULT_VOICES.get(job.target_language, "hi-IN-MadhurNeural"),
        )
        if req and req.voice and req.voice != "hi-IN-SwaraNeural":
            voice = req.voice  # user explicitly chose — honor it

        # ── Align fallback defaults with JobCreateRequest defaults ──
        # (Fixed 2026-04-12: old defaults were wrong — use_cosyvoice=True,
        # enable_sentence_gap=True, enable_duration_fit=True, etc. all
        # contradicted the actual JobCreateRequest defaults.)
        cfg = PipelineConfig(
            source="resume",
            work_dir=work_dir,
            output_path=out_path,
            source_language=req.source_language if req else "en",
            target_language=job.target_language,
            asr_model=req.asr_model if req else "groq-whisper",
            translation_engine=req.translation_engine if req else "google",
            tts_voice=voice,
            tts_rate=req.tts_rate if req else "+0%",
            use_cosyvoice=False,
            use_chatterbox=False,
            use_indic_parler=False,
            use_sarvam_bulbul=False,
            use_elevenlabs=False,
            use_google_tts=False,
            use_coqui_xtts=False,
            use_fish_speech=False,
            use_edge_tts=True,
            mix_original=req.mix_original if req else False,
            original_volume=req.original_volume if req else 0.10,
            audio_priority=True,
            audio_untouchable=getattr(req, 'audio_untouchable', False) if req else False,
            post_tts_level=getattr(req, 'post_tts_level', 'none') if req else 'none',
            audio_quality_mode=getattr(req, 'audio_quality_mode', 'fast') if req else 'fast',
            enable_sentence_gap=getattr(req, 'enable_sentence_gap', False) if req else False,
            enable_duration_fit=False,
            audio_bitrate=req.audio_bitrate if req else "192k",
            encode_preset=req.encode_preset if req else "veryfast",
            download_mode=getattr(req, 'download_mode', 'remux') if req else 'remux',
            dub_duration=getattr(req, 'dub_duration', 0) if req else 0,
            fast_assemble=req.fast_assemble if req else True,
            multi_speaker=req.multi_speaker if req else False,
            enable_manual_review=req.enable_manual_review if req else True,
            use_whisperx=req.use_whisperx if req else False,
            simplify_english=req.simplify_english if req else True,
            tts_no_time_pressure=getattr(req, 'tts_no_time_pressure', True) if req else True,
            tts_rate_mode=getattr(req, 'tts_rate_mode', 'auto') if req else 'auto',
            tts_rate_ceiling=getattr(req, 'tts_rate_ceiling', '+25%') if req else '+25%',
            tts_rate_target_wpm=getattr(req, 'tts_rate_target_wpm', 130) if req else 130,
            tts_truncation_threshold=getattr(req, 'tts_truncation_threshold', 0.30) if req else 0.30,
            tts_word_match_verify=getattr(req, 'tts_word_match_verify', True) if req else True,
            tts_word_match_tolerance=getattr(req, 'tts_word_match_tolerance', 0.15) if req else 0.15,
            tts_word_match_model=getattr(req, 'tts_word_match_model', 'auto') if req else 'auto',
            tts_word_match_max_segments=getattr(req, 'tts_word_match_max_segments', 1000) if req else 1000,
            use_wav2lip=getattr(req, 'use_wav2lip', False) if req else False,
            tts_dynamic_workers=getattr(req, 'tts_dynamic_workers', True) if req else True,
            tts_dynamic_min=getattr(req, 'tts_dynamic_min', 10) if req else 10,
            tts_dynamic_max=getattr(req, 'tts_dynamic_max', 120) if req else 120,
            tts_dynamic_start=getattr(req, 'tts_dynamic_start', 30) if req else 30,
            video_slow_to_match=getattr(req, 'video_slow_to_match', True) if req else True,
            gap_mode=getattr(req, 'gap_mode', 'micro') if req else 'micro',
            tts_chunk_words=getattr(req, 'tts_chunk_words', 0) if req else 0,
            long_segment_trace=getattr(req, 'long_segment_trace', True) if req else True,
            long_segment_threshold_words=getattr(req, 'long_segment_threshold_words', 15) if req else 15,
            enable_tts_verify_retry=getattr(req, 'enable_tts_verify_retry', False) if req else False,
            keep_subject_english=getattr(req, 'keep_subject_english', False) if req else False,
        )

        pipeline = Pipeline(cfg, on_progress=_make_progress_callback(job),
                           cancel_check=job.cancel_event.is_set)
        pipeline.run_from_srt(translated_srt)

        if not out_path.exists():
            raise RuntimeError("Pipeline finished but output file not found")

        job.result_path = out_path
        job.segments = pipeline.segments
        job.video_title = job.video_title or "Untitled"

        # Auto-save to titled folder
        try:
            _save_to_titled_folder(job)
        except Exception:
            pass

        # Generate YouTube description
        try:
            job.description = _generate_youtube_description(job)
            if job.saved_folder:
                desc_path = Path(job.saved_folder) / "description.txt"
                desc_path.write_text(job.description, encoding="utf-8")
        except Exception:
            pass

        job.overall_progress = 1.0
        job.state = "done"
        job.message = "Complete"
        job.events.append({"type": "complete", "state": "done"})
        _store.save(job)

        # Mark this URL as completed so it won't be re-queued on restart
        if job.source_url:
            _mark_url_completed(job.source_url)

    except Exception as e:
        import traceback
        try:
            (OUTPUTS / job.id / "error.log").write_text(
                f"[RESUME ERROR] {e}\n{traceback.format_exc()}", encoding="utf-8"
            )
        except OSError:
            pass
        # Clean up failed job's work directory
        try:
            job_dir = OUTPUTS / job.id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            pass
        job.state = "error"
        job.error = str(e)
        job.message = f"Error: {e}"
        job.events.append({"type": "complete", "state": "error", "error": str(e)})
        _store.save(job)
    finally:
        try:
            _purge_global_caches()
        except Exception:
            pass
        _pipeline_semaphore.release()


@app.post("/api/jobs/{job_id}/resume-with-srt")
async def resume_with_srt(job_id: str, file: UploadFile = File(...)):
    """Upload a translated SRT and resume the pipeline (TTS + assembly)."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state != "waiting_for_srt":
        raise HTTPException(status_code=409, detail=f"Job is '{job.state}', not waiting for SRT")

    # Transition state immediately to prevent duplicate resume from concurrent requests
    job.state = "running"

    work_dir = OUTPUTS / job_id / "work"
    srt_path = work_dir / "translated_upload.srt"
    try:
        with open(srt_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    except Exception:
        job.state = "waiting_for_srt"  # Revert so user can retry
        raise

    # Send SRT pair to hinglish-ai trainer (fire-and-forget)
    source_srt = work_dir / "transcript_source.srt"
    if source_srt.exists():
        src_lang = getattr(job.original_req, "source_language", "en") if job.original_req else "en"
        _send_training_data(source_srt, srt_path, source_lang=src_lang)

    t = threading.Thread(target=_run_resume, args=(job,), daemon=True)
    t.start()

    return {"id": job_id, "state": "running"}


@app.get("/api/jobs/{job_id}/original")
def get_original_video(job_id: str):
    """Serve the original downloaded video for preview."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    work_dir = OUTPUTS / job_id / "work"
    source = work_dir / "source.mp4"
    if not source.exists():
        sources = list(work_dir.glob("source.*"))
        source = sources[0] if sources else None

    if not source or not source.exists():
        raise HTTPException(status_code=404, detail="Original video not found")

    import mimetypes
    mime = mimetypes.guess_type(str(source))[0] or "video/mp4"
    return FileResponse(path=str(source), media_type=mime)


@app.get("/api/jobs/{job_id}/description")
def get_description(job_id: str):
    """Get the YouTube description for a completed job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "description": job.description or "",
        "video_title": job.video_title,
        "saved_folder": job.saved_folder,
    }


@app.get("/api/outputs")
def list_outputs():
    """List all saved dubbed outputs (titled folders)."""
    outputs = []
    if SAVED_DIR.exists():
        try:
            folders = list(SAVED_DIR.iterdir())
        except OSError:
            return outputs
        for folder in folders:
            try:
                if not folder.is_dir():
                    continue
                videos = list(folder.glob("*.mp4"))
                desc_file = folder / "description.txt"
                outputs.append({
                    "folder_name": folder.name,
                    "folder_path": str(folder),
                    "video_file": str(videos[0]) if videos else None,
                    "has_description": desc_file.exists(),
                    "has_srt": bool(list(folder.glob("*.srt"))),
                    "created": folder.stat().st_mtime,
                })
            except OSError:
                continue  # folder deleted mid-scan — skip
        outputs.sort(key=lambda x: x.get("created", 0), reverse=True)
    return outputs


class PurgeRequest(BaseModel):
    url: Optional[str] = None  # if given, purge artifacts for THIS URL only
    also_caches: bool = False  # if True, also wipe content-hash caches


@app.post("/api/cache/purge")
def purge_cache(req: PurgeRequest):
    """Manually free disk by removing work artifacts.

    - With `url`: removes all jobs/work_dirs tied to that source URL.
    - Without `url`: removes ALL jobs in error/done state and their work_dirs.
    - With `also_caches=True`: also wipes backend/cache/{asr,translation,tts}/.

    Returns bytes/files freed.
    """
    if req.url:
        result = _purge_url_artifacts(req.url, also_caches=req.also_caches)
        return {
            "status": "ok",
            "scope": "url",
            "url": req.url,
            "freed_mb": round(result["bytes"] / 1024 / 1024, 1),
            "dirs_removed": result["dirs"],
            "cache_files_removed": result["cache_files"],
        }

    # Global purge: remove all completed/errored jobs' work_dirs
    freed = 0
    dirs = 0
    for jid in [jid for jid, j in list(JOBS.items())
                if j.state in ("done", "error")]:
        job_dir = OUTPUTS / jid
        if job_dir.exists():
            try:
                for p in job_dir.rglob("*"):
                    if p.is_file():
                        try:
                            freed += p.stat().st_size
                        except OSError:
                            pass
                shutil.rmtree(job_dir, ignore_errors=True)
                dirs += 1
            except Exception:
                pass

    cache_files = 0
    if req.also_caches:
        cache_root = BASE_DIR / "cache"
        for sub in ("asr", "translation", "tts"):
            d = cache_root / sub
            if not d.exists():
                continue
            for f in d.glob("*"):
                try:
                    if f.is_file():
                        freed += f.stat().st_size
                        f.unlink()
                        cache_files += 1
                except OSError:
                    pass

    return {
        "status": "ok",
        "scope": "all",
        "freed_mb": round(freed / 1024 / 1024, 1),
        "dirs_removed": dirs,
        "cache_files_removed": cache_files,
    }


@app.get("/api/jobs/{job_id}/manual-review")
def get_manual_review_queue(job_id: str):
    """Return segments flagged for manual review for a given job.

    Returns the contents of manual_review_queue.json if it exists.
    Empty list if the job passed QC or manual review is disabled.
    """
    # Look in saved titled folder first, then in work dir
    import json as _json
    locations = []
    if job_id in JOBS and JOBS[job_id].saved_folder:
        locations.append(Path(JOBS[job_id].saved_folder) / "manual_review_queue.json")
    locations.append(OUTPUTS / job_id / "manual_review_queue.json")

    for loc in locations:
        try:
            if loc.exists():
                items = _json.loads(loc.read_text(encoding="utf-8"))
                return {"job_id": job_id, "count": len(items), "items": items}
        except Exception:
            continue
    return {"job_id": job_id, "count": 0, "items": []}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """Cancel a running job or delete a finished job.

    Cancel flow (running jobs) — RETURNS IMMEDIATELY, no blocking on the API
    thread. The previous version blocked the HTTP request for up to 8 seconds
    waiting for the worker thread, and longer if the worker was wedged in a
    native call (whisper, ffmpeg). The frontend would show "Cancelling..."
    forever and never know if the cancel succeeded.

    New flow:
      1. Signal cancel + kill subprocesses synchronously (fast, ~milliseconds)
      2. Mark state=error in DB synchronously so the UI sees "Cancelled" instantly
      3. Return 200 to the HTTP request
      4. A background thread joins the worker, escalates to thread-level kill
         if the worker is still alive after 5s, and removes the work directory
         once the worker has stopped touching files
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.state in ("running", "queued"):
        # ── Phase 1 (synchronous, fast): signal + mark cancelled ──
        # Signal cancellation so the pipeline's _check_cancelled() raises at
        # its next checkpoint
        job.cancel_event.set()

        # Kill in-flight subprocesses (yt-dlp, ffmpeg, ...) immediately.
        # Without this, the worker thread blocks inside subprocess.wait() for
        # the entire duration of the current child process.
        pipeline = getattr(job, "pipeline_ref", None)
        if pipeline is not None and hasattr(pipeline, "_kill_all_procs"):
            try:
                pipeline._kill_all_procs()
            except Exception as e:
                print(f"[CANCEL] _kill_all_procs failed for {job_id}: {e}", flush=True)

        # Mark cancelled in the DB IMMEDIATELY so the UI updates on next poll.
        # The actual worker thread shutdown happens asynchronously below.
        job.state = "error"
        job.error = "Cancelled by user"
        job.message = "Cancelled"
        job.events.append({"type": "complete", "state": "error", "error": "Cancelled by user"})
        try:
            _store.save(job)
        except Exception as e:
            print(f"[CANCEL] _store.save failed for {job_id}: {e}", flush=True)

        # ── Phase 2 (background): join worker + cleanup work dir ──
        # Spawn a daemon thread so the HTTP DELETE returns immediately.
        # Without this the API stalls until the worker actually exits, which
        # on a wedged native call (CUDA hang, ffmpeg input lock) can be minutes.
        def _async_cleanup():
            worker = getattr(job, "worker_thread", None)
            if worker is not None and worker.is_alive():
                try:
                    # Generous join — most cancels complete in <1s after
                    # _kill_all_procs returns. If the worker is still alive
                    # after 30s it's stuck in a native call we can't reach.
                    worker.join(timeout=30.0)
                    if worker.is_alive():
                        print(f"[CANCEL] Worker thread for {job_id} did not exit "
                              f"in 30s — proceeding with cleanup anyway. The "
                              f"thread will be reaped when the process exits.",
                              flush=True)
                except Exception as e:
                    print(f"[CANCEL] join failed for {job_id}: {e}", flush=True)

            # Remove the work directory. If the worker is genuinely still
            # alive and writing files, ignore_errors=True keeps cleanup
            # best-effort instead of crashing.
            job_dir = OUTPUTS / job_id
            if job_dir.exists():
                try:
                    shutil.rmtree(job_dir, ignore_errors=False)
                except Exception as e:
                    print(f"[CANCEL] rmtree {job_dir} failed: {e} — "
                          f"retrying with ignore_errors", flush=True)
                    shutil.rmtree(job_dir, ignore_errors=True)
            print(f"[CANCEL] Async cleanup complete for {job_id}", flush=True)

        threading.Thread(target=_async_cleanup, daemon=True,
                         name=f"cancel-cleanup-{job_id[:8]}").start()

        return {"status": "cancelled"}

    # Not running — just a plain delete of a finished/failed job entry
    JOBS.pop(job_id, None)
    _store.delete(job_id)

    job_dir = OUTPUTS / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    return {"status": "deleted"}


# ── Saved Links (persistent) ─────────────────────────────────────────────────

LINKS_FILE = BASE_DIR / "saved_links.json"
COMPLETED_FILE = BASE_DIR / "completed_urls.json"
_links_lock = threading.Lock()


def _load_links() -> List[Dict]:
    if LINKS_FILE.exists():
        try:
            return json.loads(LINKS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_links(links: List[Dict]):
    LINKS_FILE.write_text(json.dumps(links, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_completed_urls() -> List[str]:
    if COMPLETED_FILE.exists():
        try:
            return json.loads(COMPLETED_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to load completed URLs: {e}", flush=True)
    return []


def _mark_url_completed(url: str):
    """Add a URL to the completed list (persisted to disk)."""
    with _links_lock:
        try:
            urls = _load_completed_urls()
            if url not in urls:
                urls.append(url)
                COMPLETED_FILE.write_text(json.dumps(urls, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Failed to mark URL completed: {e}", flush=True)


def _fetch_yt_title(url: str) -> str:
    """Fetch YouTube video title via yt-dlp. Returns empty string on failure."""
    try:
        ytdlp = shutil.which("yt-dlp")
        if not ytdlp:
            return ""
        cmd = [ytdlp]
        node = shutil.which("node")
        if node:
            cmd += ["--js-runtimes", f"node:{node}"]
        cmd += ["--dump-single-json", "--no-download", url]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode == 0 and r.stdout:
            data = json.loads(r.stdout.decode("utf-8", errors="replace"))
            title = data.get("title", "")
            # Sanitize: remove filesystem-unsafe chars
            return re.sub(r'[\\/:*?"<>|]', '', title).strip()
    except Exception:
        pass
    return ""


def _bg_fetch_title(link_id: str, url: str):
    """Background thread: fetch title and update the link in saved_links.json."""
    title = _fetch_yt_title(url)
    if not title:
        return
    with _links_lock:
        links = _load_links()
        for link in links:
            if link["id"] == link_id:
                link["title"] = title
                break
        _save_links(links)


@app.get("/api/links")
def get_links():
    links = _load_links()
    completed = set(_load_completed_urls())
    for link in links:
        link["completed"] = link["url"] in completed
    return links


class LinkAdd(BaseModel):
    url: str
    title: Optional[str] = None
    preset: Optional[Dict] = None


@app.post("/api/links")
def add_link(req: LinkAdd):
    with _links_lock:
        links = _load_links()
        # Deduplicate by URL — but update preset if provided
        for l in links:
            if l["url"] == req.url:
                if req.preset:
                    l["preset"] = req.preset
                    _save_links(links)
                return {"status": "exists", "links": links}
        link_id = uuid.uuid4().hex[:12]
        links.append({
            "id": link_id,
            "url": req.url,
            "title": req.title or "",
            "added_at": time.time(),
            "preset": req.preset or {},
        })
        _save_links(links)
    # Fetch title in background if not provided
    if not req.title:
        threading.Thread(target=_bg_fetch_title, args=(link_id, req.url), daemon=True).start()
    return {"status": "added", "links": links}


class LinkPresetUpdate(BaseModel):
    preset: Dict


@app.patch("/api/links/{link_id}")
def update_link_preset(link_id: str, req: LinkPresetUpdate):
    with _links_lock:
        links = _load_links()
        for l in links:
            if l["id"] == link_id:
                l["preset"] = req.preset
                _save_links(links)
                return {"status": "updated", "links": links}
    raise HTTPException(status_code=404, detail="Link not found")


@app.delete("/api/links/{link_id}")
def delete_link(link_id: str):
    with _links_lock:
        links = _load_links()
        links = [l for l in links if l["id"] != link_id]
        _save_links(links)
    return {"status": "deleted", "links": links}


# ════════════════════════════════════════════════════════════════════════
# PRESETS — named pipeline configurations (max 3)
# ════════════════════════════════════════════════════════════════════════
from dubbing.presets import (
    save_preset as _save_preset,
    load_preset as _load_preset,
    list_presets as _list_presets,
    delete_preset as _delete_preset,
)
from dubbing.builtin_presets import (
    list_builtin_presets as _list_builtin_presets,
    get_builtin_preset as _get_builtin_preset,
)

@app.get("/api/presets")
def get_presets():
    """List all saved presets (name + slug)."""
    return {"presets": _list_presets(WORK_ROOT)}


@app.get("/api/presets/builtin")
def get_builtin_presets_list():
    """List built-in read-only quality presets."""
    return {"presets": _list_builtin_presets()}


@app.get("/api/presets/builtin/{slug}")
def get_builtin_preset_detail(slug: str):
    """Load a built-in preset's full settings by slug."""
    data = _get_builtin_preset(slug)
    if not data:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": f"Built-in preset '{slug}' not found"})
    return data

@app.get("/api/presets/{slug}")
def get_preset(slug: str):
    """Load a preset's full settings by slug."""
    try:
        data = _load_preset(WORK_ROOT, slug)
    except (ValueError, OSError) as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": str(e)})
    if not data:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": f"Preset '{slug}' not found"})
    return data

@app.post("/api/presets")
def create_preset(body: dict = Body(...)):
    """Save a named preset. Body: { name: string, settings: {...} }"""
    name = body.get("name", "").strip()
    settings = body.get("settings", {})
    if not name:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": "Preset name is required"})
    try:
        result = _save_preset(WORK_ROOT, name, settings)
        return result
    except ValueError as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.delete("/api/presets/{slug}")
def remove_preset(slug: str):
    deleted = _delete_preset(WORK_ROOT, slug)
    if not deleted:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": f"Preset '{slug}' not found"})
    return {"status": "deleted", "slug": slug}


@app.on_event("startup")
def _on_startup():
    """Server startup hook — recover any jobs left in `running` state from a
    previous crash, wipe stale work directories + caches (no-reuse mode),
    then announce readiness."""
    # Auto-recover stuck jobs: if the backend crashed mid-job, the DB will still
    # show state="running" but no worker thread is alive to update it. Mark them
    # failed on startup so the UI doesn't show ghost "Starting..." entries forever.
    try:
        recovered = 0
        for jid, job in list(JOBS.items()):
            if getattr(job, "state", None) == "running":
                job.state = "error"
                job.error = "Server restarted"
                job.message = (
                    "Backend was restarted while this job was running — "
                    "please resubmit."
                )
                _store.save(job)
                recovered += 1
        if recovered:
            print(f"[STARTUP] Recovered {recovered} stuck running job(s) -> marked as error",
                  flush=True)
    except Exception as e:
        print(f"[STARTUP] Stuck-job recovery failed: {e}", flush=True)

    # NO-REUSE mode: every startup wipes the global content-hash caches AND
    # any orphaned work directories from crashed jobs. Guarantees a clean slate.
    try:
        freed_cache = _purge_global_caches()
        freed_work = 0
        if OUTPUTS.exists():
            for d in list(OUTPUTS.iterdir()):
                if d.is_dir() and d.name not in JOBS:
                    try:
                        for p in d.rglob("*"):
                            if p.is_file():
                                try:
                                    freed_work += p.stat().st_size
                                except OSError:
                                    pass
                        shutil.rmtree(d, ignore_errors=True)
                    except Exception:
                        pass
        if freed_cache > 0 or freed_work > 0:
            print(f"[STARTUP] No-reuse cleanup: cache={freed_cache/1024/1024:.0f} MB, "
                  f"orphan work={freed_work/1024/1024:.0f} MB freed", flush=True)
    except Exception as e:
        print(f"[STARTUP] Startup cleanup failed: {e}", flush=True)

    print("[STARTUP] Server ready.", flush=True)


# ── Direct-launch entry point ─────────────────────────────────────────────
# Allows `python app.py` to start the server. Without this, the obvious launch
# command silently imports the module and exits — leaving any submitted job
# stuck in "Starting..." forever because no worker thread is alive.
#
# Adds three crash-recovery features:
#   1. File-based logging (backend/logs/backend.log) — persists across crashes
#      so we can read the traceback after the cmd window closes.
#   2. Port-conflict check up front — refuses to start if port 8000 is held,
#      with a clear error pointing at the offending PID instead of silently
#      losing the bind race.
#   3. Tee print() to both stdout AND the log file so cmd-window output and
#      file output stay in sync.
if __name__ == "__main__":
    import socket
    import sys
    import datetime as _dt
    import uvicorn

    host = os.environ.get("VOICEDUB_HOST", "0.0.0.0")
    port = int(os.environ.get("VOICEDUB_PORT", "8000"))

    # ── Persistent log file ──
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "backend.log"

    # Tee class — writes to both terminal and log file.
    # MUST forward isatty/fileno/encoding to the first (real terminal) stream
    # so libraries like uvicorn that probe stdout for TTY-ness don't crash.
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                try:
                    st.write(s)
                    st.flush()
                except Exception:
                    pass
        def flush(self):
            for st in self.streams:
                try:
                    st.flush()
                except Exception:
                    pass
        def isatty(self):
            try:
                return self.streams[0].isatty()
            except Exception:
                return False
        def fileno(self):
            # Some libs call fileno() on stdout. Delegate to the real terminal.
            return self.streams[0].fileno()
        @property
        def encoding(self):
            try:
                return self.streams[0].encoding
            except Exception:
                return "utf-8"
        def writable(self):
            return True
        def readable(self):
            return False
        def seekable(self):
            return False
        def close(self):
            # Don't actually close — these are sys.__stdout__ etc.
            pass

    log_file = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
    log_file.write(f"\n\n========= BACKEND LAUNCH {_dt.datetime.now().isoformat()} =========\n")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    # ── Port-conflict pre-check ──
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        s.bind(("127.0.0.1", port))
        s.close()
    except OSError:
        s.close()
        print(
            f"\n[FATAL] Port {port} is already in use.\n"
            f"        Another backend process is holding it.\n"
            f"        Run:  netstat -ano | findstr :{port}\n"
            f"        Then: taskkill /F /PID <pid>\n"
            f"        Or use a different port: set VOICEDUB_PORT=8001\n",
            flush=True,
        )
        sys.exit(1)

    print(f"[LAUNCH] Starting VoiceDub backend on http://{host}:{port}", flush=True)
    print(f"[LAUNCH] Log file: {log_path}", flush=True)

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("[SHUTDOWN] Ctrl+C received — closing down", flush=True)
        _clean_exit["value"] = True
    except Exception as _e:
        import traceback
        print(f"\n[FATAL] Backend crashed with unhandled exception: {_e}", flush=True)
        traceback.print_exc()
        # Re-raise so the cmd window keeps the traceback visible
        raise
    finally:
        print(f"[SHUTDOWN] Backend exited at {_dt.datetime.now().isoformat()}",
              flush=True)
        try:
            log_file.close()
        except Exception:
            pass
