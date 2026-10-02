"""The single orchestration path for the `hindi_dialogue` profile.

URL/file -> acquire -> extract -> text source (ASR / YouTube subs / SRT)
-> audio diarization (always, whatever the text source) -> word attribution
-> speaker-aware turns -> speaker registry + voice bindings -> contextual
translation -> TTS via the resolver -> bounded fit -> verification ->
multi-track mix -> mux -> MP4 + SRT/VTT + report.

Every stage records its outcome in the JobReport. Partial assets are kept on
failure. Nothing is reused across jobs (each job works in its own work_dir;
the legacy cross-job ASR cache is bypassed).
"""
from __future__ import annotations

import inspect
import json
import os
import platform
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import audio, fit, mix, verify
from .contracts import (STATUS_CANCELLED, STATUS_FAILED, UNKNOWN_SPEAKER,
                        CATEGORY_FEMALE, CATEGORY_MALE, CATEGORY_UNKNOWN, Clip, JobReport, Turn)
from .diarization import (DiarizationResult, DiarizationUnavailable,
                          assign_single_speaker, assign_words, hf_token_from_env,
                          run_pyannote)
from .report import derive_status, write_report
from .speaker_registry import SpeakerRegistry, ensure_unknown_speaker
from .translation import (DialogueTranslator, _not_hindi, build_mt_engines,
                          default_llm_clients, load_glossary)
from .tts import TTSFailure, TTSRouter, build_providers
from .turns import (align_text_to_words, build_turns, cues_from_turn_like,
                    turns_from_translated_cues, words_from_asr_segments,
                    words_from_cues)
from .voice_analysis import MIN_CONFIDENCE, analyze_speaker, classify_gender_ml

BACKEND_DIR = Path(__file__).resolve().parents[2]
ProgressCB = Callable[[str, float, str], None]

# The one speaker of a single-voice run (speaker detection off): a known
# speaker, not UNKNOWN, so the narrator's voice is still analysed.
SINGLE_SPEAKER = "SPEAKER_00"
# Below this share of required lines with Hindi text the job fails before
# TTS: a dub of the background alone is not a draft worth producing.
MIN_TRANSLATED_SHARE = 0.1
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_PCT_PREFIX = re.compile(r"^\[\d+%\]\s*")


@dataclass
class DialogueConfig:
    source: str
    work_dir: Path
    output_dir: Path
    source_srt: Optional[Path] = None        # English SRT: text source, audio still analysed
    translated_srt: Optional[Path] = None    # Hindi SRT: translation skipped
    use_youtube_subs: bool = True
    asr: str = "auto"                        # auto | local | groq
    asr_model: str = "large-v3"
    diarization_model: str = "community-1"
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    tts_providers: List[str] = field(default_factory=lambda: ["edge"])
    translation_engines: List[str] = field(default_factory=lambda: ["gemini", "groq", "cerebras"])
    mt_engines: List[str] = field(default_factory=list)   # sentence-level MT fallbacks: indictrans2 | google_basic
    allow_basic_translation_fallback: bool = True
    ollama_model: str = ""                   # model for the local Ollama translator
    diarization: bool = True                 # False = single voice by choice (not a failure)
    duration_rewrite: bool = True            # LLM shortens lines that do not fit
    background: str = "auto"                 # auto | demucs | none
    # Which audio speaker detection / voice analysis listens to. "vocals" =
    # the separated vocals stem (music and effects otherwise create false
    # speakers and wrong gender/pitch), falling back to the mix when no
    # separator ran. ASR stays on the mix unless asr_on_vocals is set.
    analysis_audio: str = "vocals"          # vocals | mix
    asr_on_vocals: bool = False
    asr_decode: str = "accurate"            # accurate (beam 5, punctuation prompt) | fast (greedy)
    content_verify: str = "auto"             # auto | on | off
    verifier_model: str = "auto"
    max_stretch: float = 1.15
    story_brief: bool = True                 # one whole-transcript LLM call: genders, aap/tum, names
    native_rate: bool = True                 # speed up via the TTS engine's own rate first (Edge)
    max_rewrites: int = 2
    max_tts_retries: int = 2
    tts_workers: int = 6
    target_lufs: float = -17.0
    audio_bitrate: str = "192k"
    unknown_voice_category: str = "male_like"
    limit_seconds: float = 0.0               # >0: dub only the first N seconds (trims media)
    embed_subtitles: bool = True
    modules: Optional[Dict[str, Any]] = None  # resolved module matrix (recorded in the report)

    def public(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        d["source"] = _redact(self.source)   # never persist signed URLs / tokens
        return d


@dataclass
class Components:
    """Injectable stage implementations (defaults wrap the real backends)."""
    acquire: Callable[[DialogueConfig, Path], Path]
    asr: Optional[Callable[[Path], List[Dict]]]
    diarize: Optional[Callable[[Path], DiarizationResult]]
    fetch_subtitles: Optional[Callable[[str], Optional[List[Dict]]]]
    llm_clients: List[Any]
    tts_providers: Dict[str, Any]
    content_asr_factory: Optional[Callable[[], Any]]
    separate: Callable[[Path, Path, str], Dict]
    basic_translate: Optional[Callable[[str], str]] = None
    notes: Dict[str, str] = field(default_factory=dict)
    mt_engines: Optional[List[Any]] = None   # sentence-level MT engines (name + translate_batch)


@dataclass
class DialogueResult:
    status: str
    output_video: Optional[Path]
    report_json: Path
    report_md: Path
    subtitles: Optional[Path]
    reasons: List[str]
    turns: List[Turn]


class Cancelled(RuntimeError):
    pass


def _is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s or ""))


# ── default components ────────────────────────────────────────────────────
def _legacy_pipeline(cfg: DialogueConfig, on_progress: Optional[ProgressCB] = None,
                     cancel_check: Optional[Callable[[], bool]] = None):
    """A legacy Pipeline instance used only for link download/subtitles/local ASR.

    With the job's cancel_check its subprocesses (yt-dlp, ffmpeg) are killed
    on cancel; on_progress receives its (step, fraction, message) reports."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from pipeline import Pipeline, PipelineConfig
    pc = PipelineConfig(source=cfg.source, work_dir=cfg.work_dir,
                        output_path=cfg.output_dir / "dubbed.mp4",
                        source_language="en", target_language="hi",
                        asr_model=cfg.asr_model, use_whisperx=False)
    return Pipeline(pc, on_progress=on_progress, cancel_check=cancel_check)


class AcquireError(RuntimeError):
    pass


def default_acquire(cfg: DialogueConfig, work: Path, on_progress: Optional[ProgressCB] = None,
                    cancel_check: Optional[Callable[[], bool]] = None,
                    on_legacy_pipeline: Optional[Callable[[Any], None]] = None) -> Path:
    src = Path(cfg.source)
    if not _is_url(cfg.source):
        if not src.exists():
            raise AcquireError(f"Input file not found: {src}")
        dst = work / f"source{src.suffix.lower() or '.mp4'}"
        if src.resolve() != dst.resolve():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        return dst
    try:
        p = _legacy_pipeline(cfg, on_progress=on_progress, cancel_check=cancel_check)
        if on_legacy_pipeline:
            on_legacy_pipeline(p)
        p._ensure_ffmpeg()
        video = Path(p._ingest_source(cfg.source))
    except Cancelled:
        raise
    except Exception as e:
        if cancel_check and cancel_check():
            # yt-dlp was killed by the cancel: not a download problem
            raise Cancelled("Job cancelled by user") from e
        msg = str(e)
        hint = ("The link could not be downloaded. If the video is private, age-restricted, "
                "members-only or region-locked, export YouTube cookies to backend/cookies.txt "
                "(see docs/dialogue/README.md) or download the video yourself and upload the file. "
                "Some links cannot be supported at all.")
        detail = re.sub(r"https?://\S+", "<url>", msg)[:300]
        raise AcquireError(f"{hint} Detail: {detail}") from e
    _write_source_title(work, getattr(p, "video_title", ""))
    return video


def _write_source_title(work: Path, title: str) -> None:
    """The downloaded video's title as one line in work/source_title.txt:
    app.py names the job and its output folder from it."""
    title = " ".join((title or "").split())
    if not title:
        return   # yt-dlp gave no title: app.py names the job from the link
    try:
        (work / "source_title.txt").write_text(title, encoding="utf-8")
    except OSError as e:
        print(f"[hindi_dialogue] could not write source_title.txt ({e}); "
              f"the job is named from the link", flush=True)


WHISPER_PUNCT_PROMPT = "Hello. Yes, I know! What did you say? Okay, let's go."


def _groq_word_asr(wav: Path) -> List[Dict]:
    import requests
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    if wav.stat().st_size > 24 * 1024 * 1024:
        raise RuntimeError("audio exceeds Groq 25 MB upload limit")
    with open(wav, "rb") as f:
        r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                          headers={"Authorization": f"Bearer {key}"},
                          data=[("model", "whisper-large-v3"), ("response_format", "verbose_json"),
                                ("language", "en"), ("prompt", WHISPER_PUNCT_PROMPT),
                                ("timestamp_granularities[]", "word"),
                                ("timestamp_granularities[]", "segment")],
                          files={"file": (wav.name, f, "audio/wav")}, timeout=600)
    r.raise_for_status()
    data = r.json()
    return _words_into_segments(data.get("segments", []), data.get("words") or [])


def _words_into_segments(segments: List[Dict], words: List[Dict], tol: float = 0.01) -> List[Dict]:
    """Attach Groq's flat word list to its segments, each word to exactly ONE
    segment: the one whose [start, end) holds the word's start, else the
    nearest within `tol`. A ±tol window on both ends put a word starting
    exactly on a boundary into both neighbours, so turns read "That's That's
    convenient." and the translator kept the doubled English word."""
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip(),
             "words": []} for s in segments]
    for w in words:
        ws = float(w["start"])
        best, best_d = None, tol
        for i, s in enumerate(segs):
            if s["start"] <= ws < s["end"]:
                best = i
                break
            d = min(abs(ws - s["start"]), abs(ws - s["end"]))
            if d <= best_d:
                best, best_d = i, d
        if best is not None:
            segs[best]["words"].append({"word": w["word"], "start": ws, "end": float(w["end"])})
    return segs


def default_components(cfg: DialogueConfig) -> Components:
    notes: Dict[str, str] = {}

    def asr(wav: Path, on_progress: Optional[ProgressCB] = None,
            cancel_check: Optional[Callable[[], bool]] = None,
            on_legacy_pipeline: Optional[Callable[[Any], None]] = None) -> List[Dict]:
        mode = cfg.asr
        if mode in ("auto", "groq") and os.environ.get("GROQ_API_KEY"):
            try:
                segs = _groq_word_asr(wav)
                notes["asr"] = "groq whisper-large-v3 (word timestamps)"
                return segs
            except Exception as e:
                if mode == "groq":
                    raise
                notes["asr_fallback"] = f"groq failed ({str(e)[:100]}); used local faster-whisper"
        p = _legacy_pipeline(cfg, on_progress=on_progress, cancel_check=cancel_check)
        if on_legacy_pipeline:
            on_legacy_pipeline(p)
        segs = p._transcribe_local(wav, decode=cfg.asr_decode)  # child process, no cache
        notes["asr"] = f"local faster-whisper {p.cfg.asr_model if p.cfg.asr_model not in ('groq-whisper','groq','parakeet') else 'medium'}"
        return segs

    def diarize(wav: Path, seg_bounds=None, heartbeat: Optional[Callable[[float], None]] = None,
                cancel_check: Optional[Callable[[], bool]] = None) -> DiarizationResult:
        return run_pyannote(wav, hf_token_from_env(), cfg.diarization_model,
                            cfg.num_speakers, cfg.min_speakers, cfg.max_speakers,
                            heartbeat=heartbeat, seg_bounds=seg_bounds, cancel_check=cancel_check)

    def fetch_subs(url: str, cancel_check: Optional[Callable[[], bool]] = None,
                   on_legacy_pipeline: Optional[Callable[[Any], None]] = None):
        from pipeline import Pipeline
        if not hasattr(Pipeline, "_fetch_youtube_subtitles"):
            # YouTube-subtitle input was removed from the legacy pipeline
            # (092e758): speech recognition is the text source, as for files.
            return None
        p = _legacy_pipeline(cfg, cancel_check=cancel_check)
        if on_legacy_pipeline:
            on_legacy_pipeline(p)
        return p._fetch_youtube_subtitles(url)

    def content_asr():
        return verify.WhisperHindiASR(cfg.verifier_model)

    try:
        import faster_whisper  # noqa: F401
        has_fw = True
    except ImportError:
        has_fw = False

    return Components(
        acquire=default_acquire, asr=asr, diarize=diarize if cfg.diarization else None,
        fetch_subtitles=fetch_subs,
        llm_clients=default_llm_clients(cfg.translation_engines, ollama_model=cfg.ollama_model),
        tts_providers=build_providers(cfg.tts_providers),
        content_asr_factory=content_asr if has_fw else None,
        separate=mix.separate_in_child, notes=notes,
        mt_engines=build_mt_engines(cfg.mt_engines))


def _supported_kwargs(fn: Callable, **kw) -> Dict[str, Any]:
    """The keyword arguments `fn` accepts: the default stage components take
    the progress/cancel hooks, injected ones may keep the bare signature."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def _new_children_killer() -> Callable[[], None]:
    """Returns a function that kills the multiprocessing children started
    after this call. The legacy local-Whisper child and the gender-classifier
    child are waited on with a plain join() that polls no cancel flag; killing
    the child ends that wait, and the cancel checkpoint after it ends the job."""
    import multiprocessing as mp
    before = set(mp.active_children())

    def kill():
        for child in mp.active_children():
            if child not in before:
                try:
                    child.kill()
                except Exception:
                    pass
    return kill


def _speakers_off_reason(cfg: DialogueConfig) -> str:
    """Why the module resolver -- not the user -- switched speaker detection
    off (e.g. no HF token), or "" when one voice was the user's choice."""
    for ch in (cfg.modules or {}).get("changes") or []:
        if ch.get("stage") == "speakers" and ch.get("action") == "deactivated" \
                and ch.get("choice") == "pyannote":
            fix = f"; fix: {ch['fix']}" if ch.get("fix") else ""
            return f"{ch.get('reason') or 'cannot run here'}{fix}"
    return ""


def _translation_errors(tr: DialogueTranslator) -> str:
    """Why the translation engines produced no Hindi, one entry per engine."""
    seen: Dict[str, str] = {}
    for w in tr.warnings:
        eng = w.get("engine")
        if not eng or eng in seen:
            continue
        if w.get("type") in ("engine_error", "engine_marked_down", "story_brief_failed"):
            seen[eng] = str(w.get("detail", ""))[:160]
        elif w.get("type") == "engine_skipped":
            seen[eng] = ("skipped: " + str(w.get("reason") or "marked down"))[:160]
        elif w.get("type") in ("malformed", "malformed_item", "empty_translation", "missing_id"):
            seen[eng] = w["type"].replace("_", " ")
    if seen:
        return "; ".join(f"{k}: {v}" for k, v in seen.items())
    if not tr.clients and not tr.mt_engines:
        return "no translation engine is available"
    return "the engines returned no Hindi text"


# ── orchestrator ──────────────────────────────────────────────────────────
class DialogueOrchestrator:
    # dialogue stage -> UI step (app.py STEP_ORDER). A stage missing here is
    # passed through under its own name, which the UI counts as 100%.
    STEP_MAP = {"acquire": "download", "extract": "extract", "separate": "extract",
                "transcribe": "transcribe", "diarize": "transcribe", "turns": "transcribe",
                "speakers": "transcribe", "translate": "translate", "synthesize": "synthesize",
                "fit": "synthesize", "verify": "synthesize", "mix": "assemble"}
    # Stages sharing a UI step each get a slice of it, so the bar only moves
    # forward (a stage restarting at 0% would drag it back to the step start).
    STAGE_SPAN = {"extract": (0.0, 0.1), "separate": (0.1, 1.0),
                  "transcribe": (0.0, 0.45), "diarize": (0.45, 0.85), "turns": (0.85, 0.9),
                  "speakers": (0.9, 1.0),
                  "synthesize": (0.0, 0.6), "fit": (0.6, 0.8), "verify": (0.8, 1.0)}
    HEARTBEAT_S = 5.0   # elapsed-time report interval during long blocking calls

    def __init__(self, cfg: DialogueConfig, components: Optional[Components] = None,
                 on_progress: Optional[ProgressCB] = None,
                 cancel_check: Optional[Callable[[], bool]] = None,
                 on_legacy_pipeline: Optional[Callable[[Any], None]] = None):
        self.cfg = cfg
        self.c = components or default_components(cfg)
        self.on_progress = on_progress or (lambda s, p, m: None)
        self.cancel_check = cancel_check or (lambda: False)
        self.on_legacy_pipeline = on_legacy_pipeline
        self.report = JobReport()
        self.turns: List[Turn] = []
        self.clips: Dict[str, Clip] = {}
        self.all_clips: List[Clip] = []
        self.registry = SpeakerRegistry(unknown_default_category=cfg.unknown_voice_category)
        self._stage_frac: Dict[str, float] = {}
        self._progress_lock = threading.Lock()

    # helpers
    def _progress(self, stage: str, frac: float, msg: str):
        """Report `frac` of a dialogue stage as progress of its UI step, never
        lower than already shown (heartbeats and legacy reports interleave),
        with the [N%] prefix every status message carries."""
        lo, hi = self.STAGE_SPAN.get(stage, (0.0, 1.0))
        with self._progress_lock:
            frac = max(self._stage_frac.get(stage, 0.0), max(0.0, min(1.0, frac)))
            self._stage_frac[stage] = frac
            p = lo + (hi - lo) * frac
            if msg and "%" not in msg:
                msg = f"[{int(round(p * 100))}%] {msg}"
            try:
                self.on_progress(self.STEP_MAP.get(stage, stage), p, msg)
            except Exception:
                pass

    def _legacy_progress(self, stage: str) -> ProgressCB:
        """Progress callback for a legacy Pipeline working inside `stage`. Its
        step names and [N%] prefix are its own, so the report is re-based on
        the stage and held below 100% until the stage really ends. It is also
        a cancel checkpoint: raising here stops the legacy Whisper fallback
        ladder from loading the next model after a cancel."""
        def cb(_step: str, frac: float, msg: str):
            if self.cancel_check():
                raise Cancelled("Job cancelled by user")
            self._progress(stage, min(0.95, float(frac)), _PCT_PREFIX.sub("", msg or ""))
        return cb

    def _register_legacy(self, pipeline) -> None:
        """Hand each legacy Pipeline to the caller: app.py keeps it as
        job.pipeline_ref, so its cancel handler kills yt-dlp/ffmpeg at once."""
        if self.on_legacy_pipeline is None:
            return
        try:
            self.on_legacy_pipeline(pipeline)
        except Exception:
            pass

    @contextmanager
    def _heartbeat(self, stage: str, label: str, expected_s: float = 0.0,
                   on_cancel: Optional[Callable[[], None]] = None):
        """While a long blocking call runs (separation, ASR, the gender model,
        the final mix), report "<label>... Ns" every HEARTBEAT_S, paced toward
        95% of the stage over `expected_s` (an estimate; 0 = keep the bar):
        a bar that sits still for minutes looks like a crash. With `on_cancel`
        the cancel flag is polled every second and, once set, on_cancel runs
        at every poll (it kills child processes that poll no cancel flag
        themselves, including a fallback started after the first kill)."""
        stop = threading.Event()
        t0 = time.time()

        def run():
            next_beat = t0 + self.HEARTBEAT_S
            cancelled = False
            while not stop.wait(min(1.0, self.HEARTBEAT_S)):
                now = time.time()
                if on_cancel is not None and (cancelled or self.cancel_check()):
                    cancelled = True
                    try:
                        on_cancel()
                    except Exception:
                        pass
                if now >= next_beat:
                    next_beat = now + self.HEARTBEAT_S
                    el = now - t0
                    frac = min(0.95, el / expected_s) if expected_s > 0 else 0.0
                    self._progress(stage, frac, f"{label}... {el:.0f}s")

        th = threading.Thread(target=run, name=f"dialogue-heartbeat-{stage}", daemon=True)
        th.start()
        try:
            yield
        finally:
            stop.set()
            th.join(timeout=2.0)

    def _check_cancel(self):
        if self.cancel_check():
            raise Cancelled("Job cancelled by user")

    @contextmanager
    def _stage(self, name: str):
        st = self.report.stage(name)
        st.status = "running"
        t0 = time.time()
        self._progress(name, 0.0, f"{name}...")
        try:
            yield st
            if st.status == "running":
                st.status = "ok"
        except BaseException as e:
            st.status = "failed"
            st.detail = (st.detail + " | " if st.detail else "") + str(e)[:300]
            raise
        finally:
            st.seconds = round(time.time() - t0, 2)
            rss = _peak_rss_mb()
            if rss:
                st.data["peak_rss_mb"] = rss
            self._progress(name, 1.0, f"{name}: {st.status}")

    def run(self) -> DialogueResult:
        cfg = self.cfg
        cfg.work_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.report.config = cfg.public()
        self.report.environment = _environment()
        aborted = ""
        try:
            self._run_stages()
        except Cancelled:
            aborted = STATUS_CANCELLED
        except BaseException as e:  # noqa: BLE001 - record and keep partial assets
            # A stage whose child process the cancel killed may fail with its
            # own error: the user's cancel is still the reason it stopped.
            if "cancelled by user" in str(e).lower() or self.cancel_check():
                aborted = STATUS_CANCELLED
            else:
                aborted = STATUS_FAILED
                self.report.unresolved_failures.append(f"{type(e).__name__}: {str(e)[:500]}")
        finally:
            self._finalise(aborted)
        jp = cfg.output_dir / "report.json"
        mp = cfg.output_dir / "report.md"
        video = Path(self.report.outputs["video"]) if self.report.outputs.get("video") else None
        subs = Path(self.report.outputs["subtitles_srt"]) if self.report.outputs.get("subtitles_srt") else None
        return DialogueResult(self.report.final_status, video, jp, mp, subs,
                              self.report.status_reasons, self.turns)

    # ── stages ─────────────────────────────────────────────────────────
    def _run_stages(self):
        cfg, r, w = self.cfg, self.report, self.cfg.work_dir

        with self._stage("acquire") as st:
            video = self.c.acquire(cfg, w, **_supported_kwargs(
                self.c.acquire, on_progress=self._legacy_progress("acquire"),
                cancel_check=self.cancel_check, on_legacy_pipeline=self._register_legacy))
            st.detail = video.name
            r.input_identity = {"source": _redact(cfg.source), "file": video.name,
                                "bytes": video.stat().st_size}
        self._check_cancel()

        with self._stage("extract") as st:
            info = audio.probe_streams(video)
            if not info.get("audio"):
                raise RuntimeError("input has no audio stream")
            if cfg.limit_seconds and info.get("duration", 0) > cfg.limit_seconds:
                cut = w / f"source_first{int(cfg.limit_seconds)}s{video.suffix}"
                audio.run_ffmpeg(["-i", str(video), "-t", f"{cfg.limit_seconds:.3f}",
                                  "-c", "copy", str(cut)])
                video = cut
                r.limitations.append(f"only the first {cfg.limit_seconds:.0f}s were dubbed (limit_seconds)")
            with self._heartbeat("extract", "Extracting audio",
                                 expected_s=10.0 + 0.02 * float(info.get("duration") or 0)):
                audio_48k = audio.to_wav(video, w / "original_48k.wav", sr=48000, channels=2)
                audio_16k = audio.to_wav(video, w / "original_16k_mono.wav", sr=16000, channels=1)
            media_dur = audio.probe_duration(audio_48k)
            vinfo = audio.probe_streams(video)
            if vinfo.get("video"):
                media_dur = vinfo.get("duration", media_dur)
            r.input_identity.update({"duration_s": round(media_dur, 3), "has_video": vinfo.get("video")})
            st.detail = f"{media_dur:.1f}s"
        self.video, self.audio_48k, self.audio_16k, self.media_dur = video, audio_48k, audio_16k, media_dur
        self._check_cancel()

        # Separate once, up front: the vocals stem feeds speaker detection,
        # gender/pitch analysis and reference clips; the background bed is
        # reused by the final mix (no second separation pass).
        self.sep = None
        self.vocals_16k = self.vocals_48k = None
        if cfg.background != "none":
            with self._stage("separate") as st:
                # About as long as the video on this GPU: the heartbeat keeps
                # the bar alive, and the default separator runs in a child
                # process that a cancel kills within a second.
                with self._heartbeat("separate", "Separating voice from music",
                                     expected_s=1.2 * media_dur):
                    sep = self.c.separate(audio_48k, w, cfg.background, **_supported_kwargs(
                        self.c.separate, cancel_check=self.cancel_check))
                self.sep = sep
                st.detail = f"{sep.get('status')}: {str(sep.get('detail', ''))[:160]}"
                if sep.get("status") != "ok":
                    st.status = "degraded"
                voc = sep.get("vocals")
                if voc and Path(voc).exists():
                    try:
                        self.vocals_48k = Path(voc)
                        self.vocals_16k = audio.to_wav(self.vocals_48k, w / "vocals_16k_mono.wav",
                                                       sr=16000, channels=1)
                    except Exception as e:
                        self.vocals_48k = self.vocals_16k = None
                        st.data["vocals_error"] = str(e)[:200]
                st.data["analysis_audio"] = "vocals" if self._use_vocals() else "mix"
            self._check_cancel()

        # text source
        mode, cues = self._text_source()
        asr_words = []
        asr_segs = []
        with self._stage("transcribe") as st:
            st.data["text_source"] = mode
            if mode != "translated_srt":
                if self.c.asr is None:
                    if mode == "asr":
                        raise RuntimeError("no ASR backend available (install faster-whisper or set GROQ_API_KEY)")
                    st.status = "degraded"
                    st.detail = "no ASR: subtitle timing estimated from cue times"
                    r.limitations.append("ASR unavailable: subtitle words timed by cue interpolation")
                else:
                    try:
                        wav = self.vocals_16k if (cfg.asr_on_vocals and self.vocals_16k) else audio_16k
                        # Local Whisper runs in a child that reports nothing
                        # until it is done: heartbeat, and kill it on cancel.
                        with self._heartbeat("transcribe", "Transcribing speech",
                                             expected_s=30.0 + 0.25 * media_dur,
                                             on_cancel=_new_children_killer()):
                            segs = self.c.asr(wav, **_supported_kwargs(
                                self.c.asr, on_progress=self._legacy_progress("transcribe"),
                                cancel_check=self.cancel_check,
                                on_legacy_pipeline=self._register_legacy))
                        asr_segs = segs
                        asr_words = words_from_asr_segments(segs)
                        st.detail = f"{len(asr_words)} words; " + self.c.notes.get("asr", "")
                        if self.c.notes.get("asr_fallback"):
                            r.limitations.append(self.c.notes["asr_fallback"])
                    except Exception as e:
                        if mode == "asr" or isinstance(e, Cancelled) or self.cancel_check():
                            raise
                        st.status = "degraded"
                        st.detail = f"ASR failed ({str(e)[:120]}); subtitle timing estimated"
                        r.limitations.append("ASR failed: subtitle words timed by cue interpolation")
            if mode == "asr" and not asr_words:
                raise RuntimeError("No speech detected in the audio")
        self._check_cancel()

        with self._stage("diarize") as st:
            diar = None
            if not cfg.diarization:
                st.status = "skipped"
                st.detail = "single voice selected (speaker detection off)"
                r.limitations.append("speaker detection was switched off: one voice for all lines")
                why = _speakers_off_reason(cfg)
                if why:
                    # Multi-speaker was asked for but cannot run on this PC:
                    # one voice is a degraded result, not a clean "completed".
                    st.status = "degraded"
                    st.detail = f"speaker detection could not run on this PC ({why})"
                    r.unresolved_failures.append(
                        f"speaker detection could not run on this PC ({why}): every line is "
                        f"voiced by one voice")
            elif self.c.diarize is None:
                st.status = "skipped"
                st.detail = "no diarization backend"
            else:
                try:
                    # Transcript line boundaries let diarization re-check each
                    # line (minor characters); every hook is passed only to
                    # backends that take it.
                    expected = 60.0 + 0.15 * media_dur
                    hooks: Dict[str, Any] = {
                        "heartbeat": lambda el: self._progress(
                            "diarize", min(0.95, el / expected), f"Detecting speakers... {el:.0f}s"),
                        "cancel_check": self.cancel_check}
                    if asr_segs:
                        hooks["seg_bounds"] = [(float(x["start"]), float(x["end"])) for x in asr_segs]
                    diar = self.c.diarize(self._analysis_16k(),
                                          **_supported_kwargs(self.c.diarize, **hooks))
                    st.detail = f"{diar.backend}: {len(diar.speakers)} speakers"
                    r.model_versions["diarization"] = diar.backend
                except DiarizationUnavailable as e:
                    st.status = "degraded"
                    st.detail = str(e)[:300]
            if diar is None and cfg.diarization:
                r.unresolved_failures.append(
                    "speaker diarization unavailable: all dialogue voiced by the default voice "
                    "(speakers not separated)")
        self.diar = diar
        self._check_cancel()

        with self._stage("turns") as st:
            # Speaker detection off: the whole video is one known speaker, so
            # the registry still analyses that voice (a woman narrator gets a
            # female voice, not the unknown-speaker default).
            if mode == "translated_srt":
                self.turns = turns_from_translated_cues(
                    cues, diar, default_speaker=UNKNOWN_SPEAKER if cfg.diarization else SINGLE_SPEAKER)
            else:
                if mode == "asr":
                    words = asr_words
                else:
                    words = align_text_to_words(cues, asr_words) if asr_words else words_from_cues(cues)
                if cfg.diarization:
                    stats = assign_words(words, diar)
                else:
                    stats = assign_single_speaker(words, SINGLE_SPEAKER)
                st.data["attribution"] = stats
                self.turns = build_turns(words)
                # unknown one/two-word fragments given to a neighbouring line
                stats["joined_adjacent_turn"] = sum(
                    1 for x in words if x.attribution.get("method") == "adjacent_turn")
                (w / "words.json").write_text(json.dumps([x.to_dict() for x in words],
                                                         ensure_ascii=False), encoding="utf-8")
            req = [t for t in self.turns if t.required]
            st.detail = f"{len(self.turns)} turns ({len(req)} required)"
            for t in self.turns:
                if t.overlaps_with and t.turn_id < min(t.overlaps_with):
                    r.unresolved_overlaps.append({
                        "turn_ids": [t.turn_id] + t.overlaps_with,
                        "start": t.source_start, "end": t.source_end,
                        "note": "simultaneous speech detected; each side is voiced on its own track "
                                "but overlapped words may be missing from the transcript"})
        self._check_cancel()

        with self._stage("speakers") as st:
            # The gender classifier loads a 1 GB model in a child process.
            with self._heartbeat("speakers", "Analysing each speaker's voice", expected_s=45.0,
                                 on_cancel=_new_children_killer()):
                self._build_registry()
            st.detail = f"{len(self.registry.speakers)} speakers"
        self._check_cancel()

        with self._stage("translate") as st:
            if mode == "translated_srt":
                st.status = "skipped"
                st.detail = "Hindi SRT supplied"
            else:
                glossary = load_glossary(BACKEND_DIR / "translation_glossary.json")
                tr = DialogueTranslator(
                    self.c.llm_clients, glossary=glossary,
                    allow_basic_fallback=cfg.allow_basic_translation_fallback,
                    basic_fallback=self.c.basic_translate,
                    mt_engines=self.c.mt_engines,
                    on_progress=lambda p, m: self._progress("translate", p, m),
                    cancel_check=self.cancel_check, story_brief=cfg.story_brief)
                if not self.c.llm_clients:
                    st.status = "degraded"
                    r.limitations.append("no LLM translation engine available; "
                                         "line-by-line (non-contextual) translation used")
                hints = {sid: rec.voice_category for sid, rec in self.registry.speakers.items()}
                self.translator = tr       # before translate(): _finalise reports its warnings
                r.translation_warnings += tr.translate(self.turns, hints)
                for t in self.turns:
                    if "translation_uncertain" in t.flags:
                        r.translation_warnings.append({"type": "translation_uncertain", "id": t.turn_id})
                self.translator = tr
                if tr.brief:
                    st.data["story_brief"] = tr.brief
                    self._check_brief_genders(tr.brief)
                r.model_versions["translation"] = sorted(tr.engines_used)
                if "indictrans2" in tr.engines_used:
                    n = sum(1 for t in self.turns if "sentence_level_mt" in t.flags)
                    r.limitations.append(f"{n} line(s) translated line by line by IndicTrans2 "
                                         f"(no dialogue context: check gendered verb forms)")
                st.detail = f"engines: {', '.join(sorted(tr.engines_used)) or 'none'}"
                # With (almost) no Hindi there is nothing to voice: fail here
                # with the engines' errors instead of mixing a background-only
                # "draft" that hides why the translation failed.
                req = [t for t in self.turns if t.required]
                # The translator's own acceptance rule: a line of digits or
                # punctuation ("3... 2... 1...") is a valid translation.
                usable = [t for t in req if t.speech_text and not _not_hindi(t.speech_text)
                          and "translation_failed" not in t.flags]
                if req and len(usable) < max(1.0, MIN_TRANSLATED_SHARE * len(req)):
                    raise RuntimeError(f"Translation failed for {len(req) - len(usable)}/{len(req)} "
                                       f"lines: {_translation_errors(tr)}")
        self._check_cancel()

        with self._stage("synthesize") as st:
            self._synthesize_all()
            st.detail = f"{sum(1 for c in self.clips.values() if c.accepted)} clips accepted"
        self._check_cancel()

        with self._stage("fit") as st:
            translator = getattr(self, "translator", None)
            rewrite = (translator.rewrite_shorter
                       if cfg.duration_rewrite and translator and translator.clients else None)
            if rewrite is None:
                r.limitations.append("line shortening off (disabled or no LLM): overlong "
                                     "turns are only sped up (max %.2fx)" % cfg.max_stretch)
            accepted = {k: v for k, v in self.clips.items() if v.accepted}
            r.timing_deviations = fit.fit_all(
                self.turns, accepted, self.media_dur, self._resynth, rewrite,
                fit.FitConfig(max_stretch=cfg.max_stretch, max_rewrites=cfg.max_rewrites),
                self.cancel_check, lambda p, m: self._progress("fit", p, m),
                native_rate=self._native_rate if cfg.native_rate else None)
            self.clips.update(accepted)
            st.detail = f"{sum(1 for d in r.timing_deviations if d.get('severity') != 'info')} timing issue(s)"
        self._check_cancel()

        with self._stage("verify") as st:
            self._verify()
            st.detail = (f"coverage {len(r.generated_turn_ids)}/{len(r.required_turn_ids)}; "
                         f"{len(r.content_warnings)} content warning(s)")
        self._check_cancel()

        # Local model workers (Indic Parler, IndicTrans2) are done: free the GPU
        # (separation already ran up front; a fallback separation may run in the mix).
        self._close_local_models()

        with self._stage("mix") as st:
            with self._heartbeat("mix", "Mixing the Hindi audio and writing the video",
                                 expected_s=30.0 + 0.1 * media_dur):
                self._mix_and_mux(st)

    def _text_source(self):
        cfg = self.cfg
        from srt_utils import parse_srt
        if cfg.translated_srt:
            cues = parse_srt(Path(cfg.translated_srt), text_key="text_translated")
            if not cues:
                raise RuntimeError("translated SRT is empty or invalid")
            return "translated_srt", cues
        if cfg.source_srt:
            cues = cues_from_turn_like(parse_srt(Path(cfg.source_srt), text_key="text"))
            if not cues:
                raise RuntimeError("English SRT is empty or invalid")
            return "english_srt", cues
        if cfg.use_youtube_subs and _is_url(cfg.source) and self.c.fetch_subtitles:
            try:
                subs = self.c.fetch_subtitles(cfg.source, **_supported_kwargs(
                    self.c.fetch_subtitles, cancel_check=self.cancel_check,
                    on_legacy_pipeline=self._register_legacy))
            except Exception as e:
                self._check_cancel()
                subs = None
                self.report.limitations.append(
                    f"YouTube subtitles could not be read ({str(e)[:120]}); "
                    f"the text comes from speech recognition")
            if subs:
                cues = cues_from_turn_like(subs)
                if cfg.limit_seconds:
                    cues = [c for c in cues if c["start"] < cfg.limit_seconds]
                if cues:
                    return "youtube_subs", cues
        return "asr", []

    def _build_registry(self):
        diar = self.diar
        speech = diar.speech_seconds() if diar else {}
        refs = self.cfg.work_dir / "speaker_refs"
        speaker_ranges = {}
        for spk in sorted({t.speaker_id for t in self.turns}):
            if spk == UNKNOWN_SPEAKER:
                continue
            ranges = diar.clean_ranges(spk) if diar else []
            if not ranges:  # speaker from SRT label only: use its turn spans
                ranges = [(t.source_start, t.source_end) for t in self.turns
                          if t.speaker_id == spk and "multi_speaker_cue" not in t.flags]
            speaker_ranges[spk] = ranges
        # Gender: wav2vec2 classifier first (one isolated child for all
        # speakers); F0 is kept as evidence and is the explicit, reported
        # fallback only when the classifier cannot run.
        p_male = classify_gender_ml(self._analysis_16k(), speaker_ranges)
        for spk in sorted({t.speaker_id for t in self.turns}):
            if spk == UNKNOWN_SPEAKER:
                ensure_unknown_speaker(self.registry)
                continue
            ranges = speaker_ranges[spk]
            try:
                cat, conf, ev = analyze_speaker(self._analysis_16k(), ranges)
            except Exception as e:
                cat, conf, ev = CATEGORY_UNKNOWN, None, {"reason": f"analysis_failed: {e}"}
            if spk in p_male:
                pm = p_male[spk]
                ml_conf = round(abs(pm - 0.5) * 2, 3)
                ev = dict(ev, f0_category=cat, classifier="wav2vec2-gender", p_male=round(pm, 4))
                cat = ((CATEGORY_MALE if pm >= 0.5 else CATEGORY_FEMALE)
                       if ml_conf >= MIN_CONFIDENCE else CATEGORY_UNKNOWN)
                conf = ml_conf
            else:
                ev = dict(ev, classifier="unavailable (F0 decision used)")
            origin = ""
            if not diar:
                single = not self.cfg.diarization and spk == SINGLE_SPEAKER
                origin = "single_voice" if single else "srt_label"
            rec = self.registry.register(
                spk, voice_category=cat, category_confidence=conf, category_evidence=ev,
                total_speech_s=round(speech.get(spk, sum(e - s for s, e in ranges)), 2),
                embedding_provenance=(diar.backend if diar and spk in diar.embeddings else None),
                mapping_origin=origin)
            self._save_reference(rec, ranges, refs)
        for prov in self.c.tts_providers:
            self.registry.bind_provider(prov)
        for t in self.turns:
            t.voice_category_hint = self.registry.speakers[t.speaker_id].voice_category
        self.registry.save(self.cfg.work_dir / "speakers.json")

    def _check_brief_genders(self, brief: Dict):
        """The transcript says one gender, the voice analysis another: the
        voice or the Hindi verb forms may be wrong. Reported, not overridden."""
        cat_of = {"male": CATEGORY_MALE, "female": CATEGORY_FEMALE}
        for sid, v in (brief.get("speakers") or {}).items():
            rec = self.registry.speakers.get(sid)
            want = cat_of.get(v.get("gender"))
            if not rec or not want or rec.voice_category not in (CATEGORY_MALE, CATEGORY_FEMALE):
                continue
            if rec.voice_category != want:
                self.report.content_warnings.append({
                    "type": "speaker_gender_disagreement", "speaker_id": sid,
                    "voice_category": rec.voice_category, "transcript_gender": v.get("gender"),
                    "evidence": v.get("evidence", "")})

    def _use_vocals(self) -> bool:
        return bool(self.cfg.analysis_audio == "vocals" and getattr(self, "vocals_16k", None))

    def _analysis_16k(self) -> Path:
        return self.vocals_16k if self._use_vocals() else self.audio_16k

    def _save_reference(self, rec, ranges, refs: Path):
        """Longest clean range (<=12 s) + its transcript, for review/optional cloning."""
        if not ranges:
            return
        s, e = max(ranges, key=lambda r: r[1] - r[0])
        e = min(e, s + 12.0)
        try:
            refs.mkdir(exist_ok=True)
            p = refs / f"{rec.speaker_id}.wav"
            audio.run_ffmpeg(["-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i",
                              str(self.vocals_48k if self._use_vocals() else self.audio_48k),
                              "-ac", "1", "-ar", "24000", str(p)])
            rec.reference_clip = str(p)
            words = []
            wp = self.cfg.work_dir / "words.json"
            if wp.exists():
                words = [x for x in json.loads(wp.read_text(encoding="utf-8"))
                         if x["start"] >= s - 0.05 and x["end"] <= e + 0.05]
            rec.reference_transcript = " ".join(x["text"] for x in words) or None
        except Exception:
            pass

    # synthesis
    def _new_router(self) -> TTSRouter:
        pron = {}
        try:
            data = json.loads((BACKEND_DIR / "pronunciation.json").read_text(encoding="utf-8"))
            pron = {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, str)}
        except Exception:
            pass
        return TTSRouter(self.c.tts_providers, self.registry, list(self.c.tts_providers),
                         self.cfg.work_dir / "clips", max_retries=self.cfg.max_tts_retries,
                         pronunciation=pron)

    def _synth_checked(self, t: Turn, reason: str, only_provider: Optional[str] = None,
                       speed: float = 1.0) -> Clip:
        """Synthesize + cheap checks; one regeneration (same voice) on failure."""
        clip = self.router.synthesize(t, reason=reason, only_provider=only_provider, speed=speed)
        chk = verify.check_clip(clip)
        if not chk["ok"]:
            again = self.router.synthesize(t, reason="cheap_check_regen:" + ",".join(chk["problems"]),
                                           only_provider=only_provider or clip.provider, speed=speed)
            chk2 = verify.check_clip(again)
            again.retry_history = clip.retry_history + again.retry_history
            clip, chk = again, chk2
        clip.verification["cheap"] = chk
        clip.accepted = chk["ok"]
        self.all_clips.append(clip)
        return clip

    def _resynth(self, t: Turn, reason: str) -> Clip:
        c = self._synth_checked(t, reason, only_provider=self.router.providers_for(t.speaker_id)[0])
        if not c.accepted:
            raise RuntimeError(f"regenerated clip for {t.turn_id} failed checks")
        return c

    def _native_rate(self, t: Turn, speed: float) -> Optional[Clip]:
        """Same voice, faster native speaking rate (no stretching artefacts)."""
        if not self.router.supports_native_rate(t.speaker_id):
            return None
        c = self._synth_checked(t, f"native_rate:{speed:.3f}",
                                only_provider=self.router.providers_for(t.speaker_id)[0], speed=speed)
        return c if c.accepted else None

    def _synthesize_all(self):
        self.router = self._new_router()
        todo = [t for t in self.turns if t.required and t.speech_text]
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.tts_workers)) as ex:
            futs = {ex.submit(self._synth_checked, t, "initial"): t for t in todo}
            for f in as_completed(futs):
                t = futs[f]
                try:
                    self.clips[t.turn_id] = f.result()
                except TTSFailure as e:
                    t.add_flag("tts_failed")
                    self.report.unresolved_failures.append(
                        f"tts_failed {t.turn_id}: {json.dumps(e.history[-2:], ensure_ascii=False)[:300]}")
                done += 1
                self._progress("synthesize", done / max(1, len(todo)), f"Synthesized {done}/{len(todo)} turns")
                if self.cancel_check():
                    for other in futs:
                        other.cancel()
                    raise Cancelled("Job cancelled by user")
        turn_map = {t.turn_id: t for t in self.turns}
        accepted = {k: v for k, v in self.clips.items() if v.accepted}
        unresolved = self.router.reroute_mixed_speakers(
            turn_map, accepted, synth=lambda t, why, prov: self._synth_checked(t, why, only_provider=prov))
        self.clips.update(accepted)
        for u in unresolved:
            self.report.unresolved_failures.append(
                f"speaker_mixed_providers {u['speaker_id']}: turns {u['turn_ids']} still use "
                f"a different provider voice")
        for spk in self.router.speaker_provider:
            self.report.limitations.append(
                f"speaker {spk} voiced entirely by fallback provider "
                f"'{self.router.speaker_provider[spk]}' (degraded)")

    # verification
    def _refit(self, t: Turn, old: Clip, new: Clip):
        windows = fit.compute_windows(self.turns, self.media_dur)
        start, end = windows.get(t.turn_id, (t.source_start, t.source_end))
        speed, overflow = fit.plan_speed(new.natural_duration, end - old.scheduled_start,
                                         self.cfg.max_stretch)
        if abs(speed - 1.0) > 1e-3:
            dst = Path(new.path).with_name(Path(new.path).stem + "_fit.wav")
            audio.time_stretch(Path(new.path), dst, speed)
            new.path = str(dst)
            new.final_duration = audio.probe_duration(dst)
        else:
            new.final_duration = new.natural_duration
        new.stretch = round(speed, 4)
        new.scheduled_start = old.scheduled_start
        new.scheduled_end = round(old.scheduled_start + new.final_duration, 3)
        if overflow > 0.15:
            self.report.timing_deviations.append({
                "turn_id": t.turn_id, "overflow_s": round(overflow, 3),
                "severity": "draft" if overflow > 0.6 else "warning",
                "scheduled_start": new.scheduled_start, "scheduled_end": new.scheduled_end,
                "reason": "after content regeneration"})

    def _mark_superseded(self):
        """Clips replaced by a rewrite/regeneration/reroute are kept for the
        record but are no longer accepted, so they are not counted twice."""
        active = {id(c) for c in self.clips.values()}
        for c in self.all_clips:
            if c.accepted and id(c) not in active:
                c.accepted = False
                c.verification["superseded"] = True

    def _verify(self):
        r, cfg = self.report, self.cfg
        turn_map = {t.turn_id: t for t in self.turns}
        run_content = cfg.content_verify == "on" or (cfg.content_verify == "auto"
                                                     and self.c.content_asr_factory is not None)
        if cfg.content_verify == "on" and self.c.content_asr_factory is None:
            raise RuntimeError("content_verify=on but faster-whisper is not installed")
        if run_content:
            asr = self.c.content_asr_factory()
            try:
                accepted = {k: v for k, v in self.clips.items() if v.accepted}
                r.content_warnings += verify.verify_content(
                    turn_map, accepted, asr, resynth=self._resynth, refit=self._refit,
                    cancel_check=self.cancel_check,
                    on_progress=lambda p, m: self._progress("verify", p, m))
                self.clips.update(accepted)
                r.model_versions["content_verifier"] = getattr(asr, "model_name", "custom")
            finally:
                if hasattr(asr, "close"):
                    asr.close()
        else:
            r.limitations.append("Hindi re-ASR content verification not run "
                                 "(faster-whisper unavailable or disabled): spoken words unverified")
            r.content_warnings.append({"type": "content_verification_skipped"})
        self._mark_superseded()
        cov = verify.coverage(self.turns, self.clips, self.all_clips)
        r.required_turn_ids = cov["required_turn_ids"]
        r.generated_turn_ids = cov["generated_turn_ids"]
        r.missing_turns = cov["missing"]
        r.duplicate_clips = cov["duplicates"]
        accepted = {k: v for k, v in self.clips.items() if v.accepted}
        r.identity_violations = verify.identity_check(turn_map, accepted, self.registry)
        degraded = [c.turn_id for c in accepted.values() if c.degraded]
        if degraded:
            r.content_warnings.append({"type": "degraded_provider_clips", "turn_ids": degraded})

    def _mix_and_mux(self, st):
        r, cfg, w = self.report, self.cfg, self.cfg.work_dir
        out = cfg.output_dir
        accepted = [c for c in self.clips.values() if c.accepted]
        n_tracks = fit.assign_tracks(accepted)
        collisions = verify.timing_collisions(accepted)
        if collisions:
            r.unresolved_failures.append(f"track collisions: {collisions[:5]}")
        rend = mix.render_dialogue(accepted, self.media_dur, w / "stems")
        st.data.update({"tracks": n_tracks, "render": {k: v for k, v in rend.items() if k != 'stems'}})
        sep = getattr(self, "sep", None)
        if sep is None:
            sep = self.c.separate(self.audio_48k, w, cfg.background)
        r.separation = sep
        if sep["status"] in ("failed", "unavailable") and cfg.background == "demucs":
            r.unresolved_failures.append(f"background separation {sep['status']}: {sep['detail']}")
        elif sep["status"] in ("failed", "unavailable"):
            r.limitations.append(f"no background bed: {sep['detail']}")
        final_wav = out / "hindi_mix.wav"
        mixinfo = mix.final_mix(Path(rend["bus"]), Path(sep["background"]) if sep.get("background") else None,
                                final_wav, self.media_dur, target_lufs=cfg.target_lufs,
                                vocals_key=Path(sep["vocals"]) if sep.get("vocals") else None)
        st.data["mix"] = mixinfo
        cues = mix.subtitle_cues({t.turn_id: t for t in self.turns}, {c.turn_id: c for c in accepted})
        srt, vtt = out / "subtitles_hi.srt", out / "subtitles_hi.vtt"
        mix.write_srt(cues, srt)
        mix.write_vtt(cues, vtt)
        bad = mix.verify_subtitles(cues, {c.turn_id: c for c in accepted})
        if bad:
            r.unresolved_failures.append(f"subtitle/audio mismatch: {bad[:5]}")
        src_cues = [{"start": t.source_start, "end": t.source_end,
                     "text": f"[{t.speaker_id}] {t.source_text}"} for t in self.turns if t.source_text]
        mix.write_srt(src_cues, out / "transcript_en_speakers.srt")
        outputs = {"audio_mix": str(final_wav), "subtitles_srt": str(srt), "subtitles_vtt": str(vtt),
                   "source_transcript": str(out / "transcript_en_speakers.srt")}
        if self.video and audio.probe_streams(self.video).get("video"):
            mp4 = out / "dubbed_hi.mp4"
            mix.mux(self.video, final_wav, mp4, cfg.audio_bitrate,
                    subtitles=srt if cfg.embed_subtitles else None)
            info = audio.probe_streams(mp4)
            if not (info.get("video") and info.get("audio")):
                raise RuntimeError("muxed MP4 is missing a stream")
            if abs(info.get("duration", 0) - self.media_dur) > 0.5:
                r.unresolved_failures.append(
                    f"output duration {info.get('duration')}s differs from source {self.media_dur:.2f}s")
            outputs["video"] = str(mp4)
        for p in rend["stems"]:
            shutil.copy2(p, out / Path(p).name)
        r.outputs.update(outputs)
        st.detail = f"{n_tracks} dialogue track(s); background={sep['status']}"

    def _close_local_models(self):
        for obj in list((self.c.tts_providers or {}).values()) + list(self.c.mt_engines or []):
            close = getattr(obj, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass

    # finalise
    def _finalise(self, aborted: str):
        self._close_local_models()
        r, out = self.report, self.cfg.output_dir
        try:
            (out / "turns.json").write_text(json.dumps([t.to_dict() for t in self.turns],
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
            (out / "clips.json").write_text(json.dumps([c.to_dict() for c in self.all_clips],
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
            self.registry.save(out / "speakers.json")
        except Exception as e:
            r.unresolved_failures.append(f"could not write artefacts: {e}")
        r.speakers = [rec.to_dict() for rec in self.registry.speakers.values()]
        reuse = {}
        for prov in self.c.tts_providers:
            for v, s in self.registry.voice_reuse(prov).items():
                reuse[f"{prov}:{v}"] = s
        r.voice_reuse = reuse
        for k, v in self.c.notes.items():
            r.model_versions.setdefault(k, v)
        # translate() handed over only the warnings that existed when it
        # returned; the fit stage's rewrites can add more (an engine marked
        # down, an API key refused) and those belong in the report too.
        tr = getattr(self, "translator", None)
        if tr is not None:
            have = {id(w) for w in r.translation_warnings}
            r.translation_warnings += [w for w in tr.warnings if id(w) not in have]
        r.final_status, r.status_reasons = derive_status(r, aborted)
        write_report(r, out)


def run_dialogue(cfg: DialogueConfig, on_progress: Optional[ProgressCB] = None,
                 cancel_check: Optional[Callable[[], bool]] = None,
                 components: Optional[Components] = None,
                 on_legacy_pipeline: Optional[Callable[[Any], None]] = None) -> DialogueResult:
    """`on_legacy_pipeline` is called with every legacy pipeline.Pipeline the
    run creates (link download, subtitles, local Whisper); app.py keeps it as
    job.pipeline_ref so a cancel can kill its subprocesses at once."""
    orch = DialogueOrchestrator(cfg, components, on_progress, cancel_check)
    # Set after construction: wrappers of __init__ keep the 4-argument call.
    orch.on_legacy_pipeline = on_legacy_pipeline
    return orch.run()


# ── environment / resource helpers ────────────────────────────────────────
def _redact(s: str) -> str:
    """Drop query strings (signed URLs, tokens) from logged sources, keeping a
    YouTube video id so the report still identifies the input."""
    if not _is_url(s) or "?" not in s:
        return s
    base, _, query = s.partition("?")
    vid = re.search(r"(?:^|&)v=([\w-]{6,})", query)
    return f"{base}?v={vid.group(1)}" if vid else f"{base}?<redacted>"


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(v / 1024.0 if sys.platform != "darwin" else v / 1048576.0, 1)
    except Exception:
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 1048576.0, 1)
        except Exception:
            return None


def _environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        env["torch"] = None
    return env
