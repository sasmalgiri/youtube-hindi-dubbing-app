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
            os.environ.setdefault(_k.strip(), _v.strip())

import asyncio
import json
import re
import shutil
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
from pydantic import BaseModel
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
BASE_DIR = Path(__file__).resolve().parent
WORK_ROOT = BASE_DIR / "work"
OUTPUTS = WORK_ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Final saved dubbed videos go here, organized by title
SAVED_DIR = BASE_DIR / "dubbed_outputs"
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
    """Copy dubbed video + SRT to a titled folder in dubbed_outputs/."""
    if not job.result_path or not job.result_path.exists():
        return

    title = _sanitize_filename(job.video_title or "Untitled")
    lang = job.target_language or "hi"

    # Create folder: dubbed_outputs/<Title> [Hindi Dubbed]
    folder_name = f"{title} [{lang.upper()} Dubbed]"
    folder = SAVED_DIR / folder_name

    # If folder exists, add job id suffix to avoid collision
    if folder.exists():
        folder = SAVED_DIR / f"{folder_name} ({job.id})"
    folder.mkdir(parents=True, exist_ok=True)

    # Copy video with title as filename
    video_name = f"{title} - {lang.upper()} Dubbed.mp4"
    saved_video = folder / video_name
    shutil.copy2(job.result_path, saved_video)

    # Copy SRT if available
    srt_src = job.result_path.parent / f"subtitles_{lang}.srt"
    if srt_src.exists():
        srt_name = f"{title} - {lang.upper()} Dubbed.srt"
        shutil.copy2(srt_src, folder / srt_name)

    job.saved_folder = str(folder)
    job.saved_video = str(saved_video)


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

        if req.transcribe_only:
            job.overall_progress = 1.0
            job.state = "waiting_for_srt"
            job.message = "Transcription complete. Download SRT and upload translation."
            job.events.append({"type": "complete", "state": "waiting_for_srt"})
            return

        if not out_path.exists():
            raise RuntimeError("Pipeline finished but output file not found")

        job.result_path = out_path
        job.video_title = pipeline.video_title or "Untitled"
        job.segments = pipeline.segments

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

    except Exception as e:
        import traceback
        _Path("/tmp/voicedub_error.log").write_text(
            f"[JOB ERROR] {e}\n{traceback.format_exc()}", encoding="utf-8"
        )
        job.state = "error"
        job.error = str(e)
        job.message = f"Error: {e}"
        job.events.append({"type": "complete", "state": "error", "error": str(e)})


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
    jobs = sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)
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
        }
        for j in jobs
    ]


@app.post("/api/jobs")
def create_job(req: JobCreateRequest):
    """Create a new dubbing job from a YouTube URL."""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, source_url=url, target_language=req.target_language)
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
    voice: str = Form("hi-IN-SwaraNeural"),
):
    """Create a dubbing job from an uploaded video file."""
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

    job = Job(id=job_id, source_url=f"upload:{file.filename}", target_language=target_language)
    JOBS[job_id] = job

    req = JobCreateRequest(
        url=str(saved_path),
        source_language=source_language,
        target_language=target_language,
        voice=voice,
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
    )
    job.original_req = req

    t = threading.Thread(target=_run_job, args=(job, req), daemon=True)
    t.start()

    return {"id": job_id}


def _job_config(job: Job) -> Dict[str, Any]:
    """Extract the config/settings used for this job."""
    req = job.original_req
    if not req:
        return {}
    tts = "Chatterbox" if req.use_chatterbox else "ElevenLabs" if req.use_elevenlabs else "Coqui XTTS" if req.use_coqui_xtts else "Google TTS" if req.use_google_tts else "Edge-TTS" if req.use_edge_tts else "Edge-TTS"
    engine_labels = {
        "auto": "Auto", "turbo": "Turbo (Multi-Engine)", "gemini": "Gemini",
        "groq": "Groq", "sambanova": "SambaNova", "together": "Together AI",
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
    try:
        job.state = "running"
        job.message = "Resuming from uploaded SRT..."
        job.events = []  # Reset events for fresh SSE stream

        job_dir = OUTPUTS / job.id
        work_dir = job_dir / "work"
        out_path = job_dir / "dubbed.mp4"
        translated_srt = work_dir / "translated_upload.srt"

        # Restore TTS settings from original request
        req = job.original_req
        cfg = PipelineConfig(
            source="resume",
            work_dir=work_dir,
            output_path=out_path,
            target_language=job.target_language,
            tts_voice=req.voice if req else "hi-IN-SwaraNeural",
            tts_rate=req.tts_rate if req else "+0%",
            use_chatterbox=req.use_chatterbox if req else False,
            use_elevenlabs=req.use_elevenlabs if req else False,
            use_google_tts=req.use_google_tts if req else False,
            use_coqui_xtts=req.use_coqui_xtts if req else False,
            use_edge_tts=req.use_edge_tts if req else True,
            mix_original=req.mix_original if req else False,
            original_volume=req.original_volume if req else 0.10,
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

    except Exception as e:
        import traceback
        _Path("/tmp/voicedub_error.log").write_text(
            f"[RESUME ERROR] {e}\n{traceback.format_exc()}", encoding="utf-8"
        )
        job.state = "error"
        job.error = str(e)
        job.message = f"Error: {e}"
        job.events.append({"type": "complete", "state": "error", "error": str(e)})


@app.post("/api/jobs/{job_id}/resume-with-srt")
async def resume_with_srt(job_id: str, file: UploadFile = File(...)):
    """Upload a translated SRT and resume the pipeline (TTS + assembly)."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state != "waiting_for_srt":
        raise HTTPException(status_code=409, detail=f"Job is '{job.state}', not waiting for SRT")

    work_dir = OUTPUTS / job_id / "work"
    srt_path = work_dir / "translated_upload.srt"
    with open(srt_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

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

    return FileResponse(path=str(source), media_type="video/mp4")


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
    job = JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = OUTPUTS / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    return {"status": "deleted"}
