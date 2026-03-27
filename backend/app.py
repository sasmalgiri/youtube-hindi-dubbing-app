"""
YouTube Hindi Dubbing API
=========================
FastAPI server with SSE progress, YouTube URL input, and translation support.
"""
from __future__ import annotations

import os
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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
from sse_starlette.sse import EventSourceResponse

from pipeline import Pipeline, PipelineConfig, list_voices, DEFAULT_VOICES
from metrics import get_metrics
from jobstore import JobStore

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

JobState = Literal["queued", "running", "done", "error", "waiting_for_srt"]


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
    chain_languages: List[str] = field(default_factory=list)  # Remaining languages in chain
    chain_parent_id: Optional[str] = None  # Parent job ID in chain
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobCreateRequest(BaseModel):
    url: str
    source_language: str = "auto"
    target_language: str = "hi"
    voice: str = "hi-IN-SwaraNeural"
    asr_model: str = "large-v3"
    translation_engine: str = "auto"
    tts_rate: str = "+0%"
    mix_original: bool = False
    original_volume: float = 0.10
    use_cosyvoice: bool = True
    use_chatterbox: bool = False
    use_elevenlabs: bool = False
    use_google_tts: bool = False
    use_coqui_xtts: bool = False
    use_edge_tts: bool = True
    prefer_youtube_subs: bool = False
    use_yt_translate: bool = False
    multi_speaker: bool = False
    transcribe_only: bool = False
    audio_priority: bool = True
    audio_bitrate: str = "192k"
    encode_preset: str = "veryfast"
    split_duration: int = 0     # 0 = no split, 30/40 = split every N minutes
    fast_assemble: bool = True  # True = instant in-memory, False = ffmpeg (preserves overlaps)
    dub_chain: List[str] = []  # e.g. ["en", "hi"] — dub through languages sequentially
    enable_manual_review: bool = True  # Save manual_review_queue.json for failed segments
    use_whisperx: bool = False         # WhisperX forced alignment for tighter word timestamps

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

# ── SQLite job persistence ────────────────────────────────────────────────────
# Survives server restarts: jobs reload from DB on startup.
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
    def callback(step: str, progress: float, message: str):
        job.current_step = step
        job.step_progress = progress
        job.overall_progress = _calc_overall(step, progress)
        job.message = message
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


def _run_job(job: Job, req: JobCreateRequest):
    """Run the dubbing pipeline in a background thread."""
    job.message = "Waiting in queue..."
    _t_start = time.time()   # defined before try so error handler can always use it
    pipeline = None           # defined before try so completion handler can always access it
    _pipeline_semaphore.acquire()
    try:
        job.state = "running"
        job.message = "Starting..."
        _store.save(job)

        job_dir = OUTPUTS / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        work_dir = job_dir / "work"
        work_dir.mkdir(exist_ok=True)

        out_path = job_dir / "dubbed.mp4"

        # Use the requested voice, or pick default for target language
        voice = req.voice
        if not voice or voice == "hi-IN-SwaraNeural":
            voice = DEFAULT_VOICES.get(req.target_language, req.voice)

        # ── SPLIT MODE: Split video into parts and dub each ──────────
        if req.split_duration > 0:
            _run_job_split(job, req, voice)
            return

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
            use_cosyvoice=req.use_cosyvoice, use_chatterbox=req.use_chatterbox,
            use_elevenlabs=req.use_elevenlabs,
            use_google_tts=req.use_google_tts,
            use_coqui_xtts=req.use_coqui_xtts,
            use_edge_tts=req.use_edge_tts,
            prefer_youtube_subs=req.prefer_youtube_subs,
            use_yt_translate=req.use_yt_translate,
            multi_speaker=req.multi_speaker,
            transcribe_only=req.transcribe_only,
            audio_priority=req.audio_priority,
            audio_bitrate=req.audio_bitrate,
            encode_preset=req.encode_preset,
            split_duration=req.split_duration,
            fast_assemble=req.fast_assemble,
            enable_manual_review=req.enable_manual_review,
            use_whisperx=req.use_whisperx,
        )

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

        pipeline = Pipeline(cfg, on_progress=_make_progress_callback(job),
                            cancel_check=job.cancel_event.is_set)
        pipeline.run()

        job.video_title = pipeline.video_title or "Untitled"
        job.segments = pipeline.segments
        job.qa_score = pipeline.qa_score

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
        _segs = (pipeline.segments if pipeline else None) or []
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
            use_cosyvoice=req.use_cosyvoice, use_chatterbox=req.use_chatterbox, use_elevenlabs=req.use_elevenlabs,
            use_google_tts=req.use_google_tts, use_coqui_xtts=req.use_coqui_xtts,
            use_edge_tts=req.use_edge_tts, prefer_youtube_subs=False,
            multi_speaker=req.multi_speaker, audio_priority=req.audio_priority,
            audio_bitrate=req.audio_bitrate, encode_preset=req.encode_preset,
            fast_assemble=req.fast_assemble,
            enable_manual_review=req.enable_manual_review,
            use_whisperx=req.use_whisperx,
        )
        p = Pipeline(cfg, on_progress=callback, cancel_check=job.cancel_event.is_set)
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
            use_cosyvoice=req.use_cosyvoice, use_chatterbox=req.use_chatterbox,
            use_elevenlabs=req.use_elevenlabs,
            use_google_tts=req.use_google_tts,
            use_coqui_xtts=req.use_coqui_xtts,
            use_edge_tts=req.use_edge_tts,
            prefer_youtube_subs=False,
            multi_speaker=req.multi_speaker,
            audio_priority=req.audio_priority,
            audio_bitrate=req.audio_bitrate,
            encode_preset=req.encode_preset,
            fast_assemble=req.fast_assemble,
            enable_manual_review=req.enable_manual_review,
            use_whisperx=req.use_whisperx,
        )

        pipeline = Pipeline(cfg, on_progress=_part_callback,
                           cancel_check=job.cancel_event.is_set)
        pipeline.video_title = f"{job.video_title} - Part {part_num}"
        pipeline.run()

        if part_out.exists():
            output_parts.append((part_num, part_out))
            print(f"[SPLIT] Part {part_num}/{num_parts} complete: {part_out}", flush=True)

    if not output_parts:
        raise RuntimeError("No parts were produced")

    # Save all parts to titled folders
    base_title = job.video_title
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
        use_cosyvoice=_orig.use_cosyvoice if _orig else True,
        use_chatterbox=_orig.use_chatterbox if _orig else False,
        use_elevenlabs=_orig.use_elevenlabs if _orig else False,
        use_google_tts=_orig.use_google_tts if _orig else False,
        use_coqui_xtts=_orig.use_coqui_xtts if _orig else False,
        use_edge_tts=_orig.use_edge_tts if _orig else True,
        mix_original=_orig.mix_original if _orig else False,
        original_volume=_orig.original_volume if _orig else 0.10,
        multi_speaker=_orig.multi_speaker if _orig else False,
        audio_priority=_orig.audio_priority if _orig else True,
        audio_bitrate=_orig.audio_bitrate if _orig else "192k",
        encode_preset=_orig.encode_preset if _orig else "veryfast",
        fast_assemble=_orig.fast_assemble if _orig else True,
        enable_manual_review=_orig.enable_manual_review if _orig else True,
        use_whisperx=_orig.use_whisperx if _orig else False,
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


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


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


@app.get("/api/jobs")
def list_jobs():
    """List all jobs, newest first."""
    jobs = sorted(list(JOBS.values()), key=lambda j: j.created_at, reverse=True)
    return [
        {
            "id": j.id,
            "state": j.state,
            "current_step": j.current_step,
            "step_progress": j.step_progress,
            "overall_progress": j.overall_progress,
            "message": j.message,
            "error": j.error,
            "source_url": j.source_url,
            "video_title": j.video_title,
            "target_language": j.target_language,
            "created_at": j.created_at,
            "saved_folder": j.saved_folder,
            "saved_video": j.saved_video,
            "description": j.description,
            "qa_score": j.qa_score,
            "chain_languages": j.chain_languages,
            "chain_parent_id": j.chain_parent_id,
        }
        for j in jobs
    ]


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

    return {"id": job_id}


@app.post("/api/jobs/upload")
async def create_job_upload(
    file: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form("hi"),
    asr_model: str = Form("large-v3"),
    translation_engine: str = Form("auto"),
    tts_rate: str = Form("+0%"),
    mix_original: str = Form("false"),
    original_volume: float = Form(0.10),
    use_cosyvoice: str = Form("true"),
    use_chatterbox: str = Form("false"),
    use_elevenlabs: str = Form("false"),
    use_google_tts: str = Form("false"),
    use_coqui_xtts: str = Form("false"),
    use_edge_tts: str = Form("false"),
    prefer_youtube_subs: str = Form("false"),
    use_yt_translate: str = Form("false"),
    multi_speaker: str = Form("false"),
    transcribe_only: str = Form("false"),
    audio_priority: str = Form("true"),
    audio_bitrate: str = Form("192k"),
    encode_preset: str = Form("veryfast"),
    split_duration: int = Form(0),
    fast_assemble: str = Form("true"),
    enable_manual_review: str = Form("true"),
    use_whisperx: str = Form("false"),
    voice: str = Form("hi-IN-SwaraNeural"),
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
            use_elevenlabs=_bool(use_elevenlabs),
            use_google_tts=_bool(use_google_tts),
            use_coqui_xtts=_bool(use_coqui_xtts),
            use_edge_tts=_bool(use_edge_tts),
            prefer_youtube_subs=_bool(prefer_youtube_subs),
            use_yt_translate=_bool(use_yt_translate),
            multi_speaker=_bool(multi_speaker),
            transcribe_only=_bool(transcribe_only),
            audio_priority=_bool(audio_priority),
            audio_bitrate=audio_bitrate,
            encode_preset=encode_preset,
            split_duration=split_duration,
            fast_assemble=_bool(fast_assemble),
            enable_manual_review=_bool(enable_manual_review),
            use_whisperx=_bool(use_whisperx),
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

    return {"id": job_id}


def _run_job_with_srt(job: Job, req: JobCreateRequest, srt_path: Path):
    """Download/extract the video, then run TTS+assembly from the provided SRT."""
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
        if not voice or voice == "hi-IN-SwaraNeural":
            voice = DEFAULT_VOICES.get(req.target_language, req.voice)

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
            use_cosyvoice=req.use_cosyvoice, use_chatterbox=req.use_chatterbox,
            use_elevenlabs=req.use_elevenlabs,
            use_google_tts=req.use_google_tts,
            use_coqui_xtts=req.use_coqui_xtts,
            use_edge_tts=req.use_edge_tts,
            multi_speaker=req.multi_speaker,
            audio_priority=req.audio_priority,
            audio_bitrate=req.audio_bitrate,
            encode_preset=req.encode_preset,
            fast_assemble=req.fast_assemble,
            enable_manual_review=req.enable_manual_review,
            use_whisperx=req.use_whisperx,
        )

        pipeline = Pipeline(cfg, on_progress=_make_progress_callback(job),
                           cancel_check=job.cancel_event.is_set)

        # Step 1-2: Download + extract audio (pipeline handles this)
        pipeline.download_and_extract()

        job.video_title = pipeline.video_title or "Untitled"

        # Step 3-6: Skip transcribe/translate, run TTS + assembly from SRT
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
        _pipeline_semaphore.release()


@app.post("/api/jobs/with-srt")
async def create_job_with_srt(
    srt_file: UploadFile = File(...),
    url: str = Form(""),
    video_file: Optional[UploadFile] = File(None),
    source_language: str = Form("auto"),
    target_language: str = Form("hi"),
    asr_model: str = Form("large-v3"),
    translation_engine: str = Form("auto"),
    tts_rate: str = Form("+0%"),
    mix_original: str = Form("false"),
    original_volume: float = Form(0.10),
    use_cosyvoice: str = Form("true"),
    use_chatterbox: str = Form("false"),
    use_elevenlabs: str = Form("false"),
    use_google_tts: str = Form("false"),
    use_coqui_xtts: str = Form("false"),
    use_edge_tts: str = Form("false"),
    prefer_youtube_subs: str = Form("false"),
    use_yt_translate: str = Form("false"),
    multi_speaker: str = Form("false"),
    audio_priority: str = Form("true"),
    audio_bitrate: str = Form("192k"),
    encode_preset: str = Form("veryfast"),
    split_duration: int = Form(0),
    fast_assemble: str = Form("true"),
    enable_manual_review: str = Form("true"),
    use_whisperx: str = Form("false"),
    voice: str = Form("hi-IN-SwaraNeural"),
):
    """Create a dubbing job from a video (URL or file) + pre-translated SRT file.
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
            use_elevenlabs=_bool(use_elevenlabs),
            use_google_tts=_bool(use_google_tts),
            use_coqui_xtts=_bool(use_coqui_xtts),
            use_edge_tts=_bool(use_edge_tts),
            prefer_youtube_subs=_bool(prefer_youtube_subs),
            use_yt_translate=_bool(use_yt_translate),
            multi_speaker=_bool(multi_speaker),
            audio_priority=_bool(audio_priority),
            audio_bitrate=audio_bitrate,
            encode_preset=encode_preset,
            split_duration=split_duration,
            fast_assemble=_bool(fast_assemble),
            enable_manual_review=_bool(enable_manual_review),
            use_whisperx=_bool(use_whisperx),
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

    return {"id": job_id}


def _job_config(job: Job) -> Dict[str, Any]:
    """Extract the config/settings used for this job."""
    req = job.original_req
    if not req:
        return {}
    if req.use_coqui_xtts and req.use_edge_tts:
        tts = "Hybrid (Coqui+Edge)"
    elif req.use_chatterbox:
        tts = "Chatterbox"
    elif req.use_elevenlabs:
        tts = "ElevenLabs"
    elif req.use_coqui_xtts:
        tts = "Coqui XTTS"
    elif req.use_google_tts:
        tts = "Google TTS"
    else:
        tts = "Edge-TTS"
    engine_labels = {
        "auto": "Auto", "turbo": "Turbo (Groq+SambaNova)", "gemini": "Gemini",
        "groq": "Groq", "sambanova": "SambaNova", "google_polish": "Google+LLM Polish",
        "ollama": "Ollama", "google": "Google Translate", "hinglish": "Hinglish AI",
    }
    return {
        "asr_model": getattr(req, "asr_model", "large-v3"),
        "translation_engine": engine_labels.get(getattr(req, "translation_engine", "auto"), "Auto"),
        "tts_engine": tts,
        "audio_priority": getattr(req, "audio_priority", False),
        "audio_bitrate": getattr(req, "audio_bitrate", "192k"),
        "encode_preset": getattr(req, "encode_preset", "veryfast"),
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
    return {"segments": job.segments}


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state != "done" or not job.result_path:
        raise HTTPException(status_code=409, detail="Job not complete")

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

        # Restore TTS settings from original request
        req = job.original_req
        # Replicate voice defaulting logic from _run_job
        voice = req.voice if req else "hi-IN-SwaraNeural"
        if not voice or voice == "hi-IN-SwaraNeural":
            voice = DEFAULT_VOICES.get(job.target_language, voice)
        cfg = PipelineConfig(
            source="resume",
            work_dir=work_dir,
            output_path=out_path,
            source_language=req.source_language if req else "auto",
            target_language=job.target_language,
            asr_model=req.asr_model if req else "large-v3-turbo",
            translation_engine=req.translation_engine if req else "auto",
            tts_voice=voice,
            tts_rate=req.tts_rate if req else "+0%",
            use_cosyvoice=req.use_cosyvoice if req else True,
            use_chatterbox=req.use_chatterbox if req else False,
            use_elevenlabs=req.use_elevenlabs if req else False,
            use_google_tts=req.use_google_tts if req else False,
            use_coqui_xtts=req.use_coqui_xtts if req else False,
            use_edge_tts=req.use_edge_tts if req else True,
            mix_original=req.mix_original if req else False,
            original_volume=req.original_volume if req else 0.10,
            audio_priority=req.audio_priority if req else True,
            audio_bitrate=req.audio_bitrate if req else "192k",
            encode_preset=req.encode_preset if req else "veryfast",
            fast_assemble=req.fast_assemble if req else True,
            multi_speaker=req.multi_speaker if req else False,
            enable_manual_review=req.enable_manual_review if req else True,
            use_whisperx=req.use_whisperx if req else False,
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
        src_lang = job.original_req.source_language if job.original_req else "en"
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
        for folder in sorted(SAVED_DIR.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True):
            if folder.is_dir():
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
    return outputs


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
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.state in ("running", "queued"):
        # Signal the pipeline to stop at the next cancel checkpoint
        job.cancel_event.set()
        job.state = "error"
        job.error = "Cancelled by user"
        job.message = "Cancelled"
        job.events.append({"type": "complete", "state": "error", "error": "Cancelled by user"})
        _store.save(job)
        # Clean up work directory
        job_dir = OUTPUTS / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        return {"status": "cancelled"}

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
        except Exception:
            return []
    return []


def _mark_url_completed(url: str):
    """Add a URL to the completed list (persisted to disk)."""
    with _links_lock:
        urls = _load_completed_urls()
        if url not in urls:
            urls.append(url)
            COMPLETED_FILE.write_text(json.dumps(urls, indent=2, ensure_ascii=False), encoding="utf-8")


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


@app.on_event("startup")
def _on_startup():
    """Server startup hook — placeholder for future startup tasks."""
    print("[STARTUP] Server ready.", flush=True)
