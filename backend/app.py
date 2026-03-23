"""
VoiceDub Backend API
====================
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
import re
import shutil
import subprocess
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
    use_chatterbox: bool = True
    use_elevenlabs: bool = False
    use_google_tts: bool = False
    use_coqui_xtts: bool = False
    use_edge_tts: bool = False
    prefer_youtube_subs: bool = False
    multi_speaker: bool = False
    transcribe_only: bool = False
    audio_priority: bool = True
    audio_bitrate: str = "192k"
    encode_preset: str = "veryfast"
    dub_chain: List[str] = []  # e.g. ["en", "hi"] — dub through languages sequentially

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
WORK_ROOT = BASE_DIR / "work"
OUTPUTS = WORK_ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Final saved dubbed videos go here, organized by title
# Use D: drive folder if available (user's preferred location), else fallback to local
_preferred_save = Path("D:/Shirshendu sasmal/youtube dubbed")
SAVED_DIR = _preferred_save if _preferred_save.exists() else BASE_DIR / "dubbed_outputs"
SAVED_DIR.mkdir(parents=True, exist_ok=True)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="VoiceDub API")

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
        f"#{lang_name}Dubbed #{lang_name} #AIDubbing #VoiceDub #YouTubeDubbing\n\n"
        f"If you enjoyed this dubbed version, please Like, Subscribe, and Share!\n\n"
        f"Turn on notifications to never miss a new dubbed video.\n\n"
        f"This video was dubbed using AI voice technology. "
        f"Original content belongs to the respective creators."
    )


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
    return callback


def _run_job(job: Job, req: JobCreateRequest):
    """Run the dubbing pipeline in a background thread."""
    job.message = "Waiting in queue..."
    _pipeline_semaphore.acquire()
    try:
        job.state = "running"
        job.message = "Starting..."

        job_dir = OUTPUTS / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        work_dir = job_dir / "work"
        work_dir.mkdir(exist_ok=True)

        out_path = job_dir / "dubbed.mp4"

        # Use the requested voice, or pick default for target language
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
            use_chatterbox=req.use_chatterbox,
            use_elevenlabs=req.use_elevenlabs,
            use_google_tts=req.use_google_tts,
            use_coqui_xtts=req.use_coqui_xtts,
            use_edge_tts=req.use_edge_tts,
            prefer_youtube_subs=req.prefer_youtube_subs,
            multi_speaker=req.multi_speaker,
            transcribe_only=req.transcribe_only,
            audio_priority=req.audio_priority,
            audio_bitrate=req.audio_bitrate,
            encode_preset=req.encode_preset,
        )

        pipeline = Pipeline(cfg, on_progress=_make_progress_callback(job))
        pipeline.run()

        job.video_title = pipeline.video_title or "Untitled"
        job.segments = pipeline.segments
        job.qa_score = pipeline.qa_score

        if req.transcribe_only:
            job.overall_progress = 1.0
            job.state = "waiting_for_srt"
            job.message = "Transcription complete. Download SRT and upload translation."
            job.events.append({"type": "complete", "state": "waiting_for_srt"})
            return

        if not out_path.exists():
            raise RuntimeError("Pipeline finished but output file not found")

        job.result_path = out_path

        # Auto-save to titled folder
        try:
            _save_to_titled_folder(job)
        except Exception as save_err:
            print(f"[WARN] Failed to save to titled folder: {save_err}")

        # Generate YouTube description
        try:
            job.description = _generate_youtube_description(job)
            # Save description file to the titled folder
            if job.saved_folder:
                desc_path = Path(job.saved_folder) / "description.txt"
                desc_path.write_text(job.description, encoding="utf-8")
        except Exception as desc_err:
            print(f"[WARN] Failed to generate description: {desc_err}")

        job.overall_progress = 1.0
        job.state = "done"
        job.message = "Complete"
        job.events.append({"type": "complete", "state": "done"})

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
        try:
            (OUTPUTS / job.id / "error.log").write_text(
                f"[JOB ERROR] {e}\n{traceback.format_exc()}", encoding="utf-8"
            )
        except OSError:
            pass
        # Clean up failed job's work directory too
        try:
            job_dir = OUTPUTS / job.id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
                print(f"[CLEANUP] Deleted failed job work directory {job_dir}")
        except Exception:
            pass
        job.state = "error"
        job.error = str(e)
        job.message = f"Error: {e}"
        job.events.append({"type": "complete", "state": "error", "error": str(e)})
    finally:
        _pipeline_semaphore.release()


def _queue_chain_next(parent_job: Job):
    """Queue the next language in a dub chain, using the parent's output as input."""
    next_lang = parent_job.chain_languages[0]
    remaining = parent_job.chain_languages[1:]

    print(f"[CHAIN] Job {parent_job.id} done ({parent_job.target_language}). "
          f"Queuing next: {next_lang} (remaining: {remaining})", flush=True)

    job_id = uuid.uuid4().hex[:12]
    # Use the saved video from previous step as input
    input_path = parent_job.saved_video

    # Create request using the output video as source
    req = JobCreateRequest(
        url=input_path,
        source_language=parent_job.target_language,  # Previous output language
        target_language=next_lang,
        prefer_youtube_subs=False,  # No YouTube subs for local file
        use_chatterbox=parent_job.original_req.use_chatterbox if parent_job.original_req else True,
        use_edge_tts=parent_job.original_req.use_edge_tts if parent_job.original_req else False,
        mix_original=parent_job.original_req.mix_original if parent_job.original_req else False,
        original_volume=parent_job.original_req.original_volume if parent_job.original_req else 0.10,
        audio_priority=parent_job.original_req.audio_priority if parent_job.original_req else True,
        audio_bitrate=parent_job.original_req.audio_bitrate if parent_job.original_req else "192k",
        encode_preset=parent_job.original_req.encode_preset if parent_job.original_req else "veryfast",
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

    parent_job.message = f"Complete — next: dubbing to {next_lang.upper()}"

    t = threading.Thread(target=_run_job, args=(job, req), daemon=True)
    t.start()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


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
        }
        for j in jobs
    ]


def _cleanup_old_jobs():
    """Remove oldest completed/errored jobs when limit is exceeded."""
    if len(JOBS) <= MAX_JOBS:
        return
    completed = sorted(
        [(jid, j) for jid, j in list(JOBS.items()) if j.state in ("done", "error")],
        key=lambda x: x[1].created_at,
    )
    while len(JOBS) > MAX_JOBS and completed:
        jid, _ = completed.pop(0)
        JOBS.pop(jid, None)
        job_dir = OUTPUTS / jid
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)


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
    mix_original: bool = Form(False),
    original_volume: float = Form(0.10),
    use_chatterbox: bool = Form(True),
    use_elevenlabs: bool = Form(False),
    use_google_tts: bool = Form(False),
    use_coqui_xtts: bool = Form(False),
    use_edge_tts: bool = Form(False),
    prefer_youtube_subs: bool = Form(False),
    multi_speaker: bool = Form(False),
    transcribe_only: bool = Form(False),
    audio_priority: bool = Form(True),
    audio_bitrate: str = Form("192k"),
    encode_preset: str = Form("veryfast"),
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
    try:
        req = JobCreateRequest(
            url=str(saved_path),
            source_language=source_language,
            target_language=target_language,
            voice=voice,
            asr_model=asr_model,
            translation_engine=translation_engine,
            tts_rate=tts_rate,
            mix_original=mix_original,
            original_volume=original_volume,
            use_chatterbox=use_chatterbox,
            use_elevenlabs=use_elevenlabs,
            use_google_tts=use_google_tts,
            use_coqui_xtts=use_coqui_xtts,
            use_edge_tts=use_edge_tts,
            prefer_youtube_subs=prefer_youtube_subs,
            multi_speaker=multi_speaker,
            transcribe_only=transcribe_only,
            audio_priority=audio_priority,
            audio_bitrate=audio_bitrate,
            encode_preset=encode_preset,
        )
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    job = Job(id=job_id, source_url=f"upload:{file.filename}", target_language=target_language)
    job.original_req = req
    JOBS[job_id] = job

    t = threading.Thread(target=_run_job, args=(job, req), daemon=True)
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
        "groq": "Groq", "sambanova": "SambaNova",
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
    }


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """SSE endpoint for real-time progress updates."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_index = 0
        while True:
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


def _run_resume(job: Job):
    """Resume pipeline from uploaded translated SRT in a background thread."""
    job.message = "Waiting in queue..."
    _pipeline_semaphore.acquire()
    try:
        job.state = "running"
        job.message = "Resuming from uploaded SRT..."
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
            target_language=job.target_language,
            tts_voice=voice,
            tts_rate=req.tts_rate if req else "+0%",
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
            multi_speaker=req.multi_speaker if req else False,
        )

        pipeline = Pipeline(cfg, on_progress=_make_progress_callback(job))
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


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.state in ("running", "queued"):
        raise HTTPException(status_code=409, detail="Cannot delete a running job — wait for it to finish or error out")

    JOBS.pop(job_id, None)

    job_dir = OUTPUTS / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    return {"status": "deleted"}


# ── Saved Links (persistent) ─────────────────────────────────────────────────

LINKS_FILE = BASE_DIR / "saved_links.json"
COMPLETED_FILE = BASE_DIR / "completed_urls.json"


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
    links = _load_links()
    for link in links:
        if link["id"] == link_id:
            link["title"] = title
            break
    _save_links(links)


@app.get("/api/links")
def get_links():
    return _load_links()


class LinkAdd(BaseModel):
    url: str
    title: Optional[str] = None


@app.post("/api/links")
def add_link(req: LinkAdd):
    links = _load_links()
    # Deduplicate by URL
    if any(l["url"] == req.url for l in links):
        return {"status": "exists", "links": links}
    link_id = uuid.uuid4().hex[:12]
    links.append({
        "id": link_id,
        "url": req.url,
        "title": req.title or "",
        "added_at": time.time(),
    })
    _save_links(links)
    # Fetch title in background if not provided
    if not req.title:
        threading.Thread(target=_bg_fetch_title, args=(link_id, req.url), daemon=True).start()
    return {"status": "added", "links": links}


@app.delete("/api/links/{link_id}")
def delete_link(link_id: str):
    links = _load_links()
    links = [l for l in links if l["id"] != link_id]
    _save_links(links)
    return {"status": "deleted", "links": links}


# ── Auto-resume incomplete links on startup ─────────────────────────────────

def _resume_incomplete_links():
    """Check saved links vs completed URLs and re-queue incomplete ones."""
    links = _load_links()
    completed = set(_load_completed_urls())

    pending = [l for l in links if l["url"] not in completed]
    if not pending:
        print("[STARTUP] All saved links already completed.")
        return

    print(f"[STARTUP] Found {len(pending)} incomplete links, queuing them...")
    for link in pending:
        url = link["url"]
        # Skip if already in the current job queue
        if any(j.source_url == url and j.state in ("queued", "running") for j in JOBS.values()):
            print(f"[STARTUP]   Skipping (already queued): {url}")
            continue

        job_id = uuid.uuid4().hex[:12]
        req = JobCreateRequest(url=url)
        job = Job(id=job_id, source_url=url, target_language=req.target_language)
        job.original_req = req
        JOBS[job_id] = job

        t = threading.Thread(target=_run_job, args=(job, req), daemon=True)
        t.start()
        print(f"[STARTUP]   Queued: {url} -> job {job_id}")


# Run auto-resume after a short delay to let the server start
def _startup_resume():
    time.sleep(3)
    _resume_incomplete_links()

threading.Thread(target=_startup_resume, daemon=True).start()
