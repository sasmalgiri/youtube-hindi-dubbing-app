"""Optional, experimental: make each Hindi clip sound like its original speaker.

OpenVoice v2's tone colour converter (myshell-ai/OpenVoice, MIT) changes
only the voice colour (timbre) of a finished TTS clip toward a reference of
the original speaker; the Hindi words, pronunciation and timing stay the TTS
engine's. It runs in a persistent worker (workers/openvoice_worker.py),
normally under its own Python (OPENVOICE_PYTHON), because OpenVoice pins
numpy 1.22 and librosa 0.9.1 -- the same arrangement as Indic Parler-TTS
(see local_workers.py).

The orchestrator owns the policy: it cuts each speaker's references, calls
prepare() once per speaker and convert() after each clip is synthesized,
and reports failures as limitations. This side only answers honestly:
  * a speaker with less than MIN_REFERENCE_S of reference speech is not
    prepared (an embedding from a second or two of audio is unstable), and
    its clips are never converted;
  * convert() returns False -- the TTS clip stays as it was -- for any
    failure, and True only for an output with exactly the input's sample
    rate and length, so the dub's timing cannot change.
Why a speaker or clip was left unconverted is kept in `notes` / `errors`.
"""
from __future__ import annotations

import hashlib
import re
import threading
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import audio
from .local_workers import PersistentWorker, WorkerError

MIN_REFERENCE_S = 3.0      # total reference speech a speaker needs to be converted
MIN_CLIP_S = 0.3           # shorter clips are left as they are (the worker refuses them too)
MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "openvoice"
MAX_ERRORS = 20


def _seconds(path: Path) -> float:
    try:
        return audio.probe_duration(path)
    except Exception:
        return 0.0


def _wav_shape(path: Path) -> Optional[Tuple[int, int, int]]:
    """(sample rate, frames, channels) from the WAV header, None if unreadable."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getframerate(), wf.getnframes(), wf.getnchannels()
    except Exception:
        return None


def _digest(paths: Sequence[Path]) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(Path(p).read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


class OpenVoiceMatcher:
    """The orchestrator's voice matcher (name / prepare / convert / close)."""
    name = "openvoice"

    def __init__(self, cache_dir: Optional[Path] = None, models_dir: Optional[Path] = None,
                 tau: float = 0.3):
        # Speaker embeddings are cached inside this job's work_dir only.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.worker = PersistentWorker("openvoice", init={"models_dir": str(models_dir or MODELS_DIR),
                                                          "tau": tau}, timeout=900)
        self.prepared: Dict[str, bool] = {}
        self._prepare_reqs: Dict[str, Dict] = {}
        self.notes: Dict[str, str] = {}       # speaker_id -> why it keeps the TTS voice
        self.errors: List[str] = []
        self._lock = threading.Lock()

    def _error(self, msg: str):
        with self._lock:
            if msg not in self.errors and len(self.errors) < MAX_ERRORS:
                self.errors.append(msg)

    def prepare(self, speaker_id: str, reference_wavs: List[Path]) -> bool:
        """Extract (and cache) the speaker's voice embedding. False = this
        speaker keeps the Hindi TTS voice."""
        refs = [Path(p) for p in reference_wavs or [] if Path(p).is_file()]
        total = sum(_seconds(p) for p in refs)
        if total < MIN_REFERENCE_S:
            self.prepared[speaker_id] = False
            self.notes[speaker_id] = (f"only {total:.1f} s of clean reference speech (needs "
                                      f"{MIN_REFERENCE_S:g} s)")
            return False
        req = {"op": "prepare", "speaker": speaker_id, "refs": [str(p) for p in refs]}
        if self.cache_dir:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", speaker_id)
            req["se_path"] = str(self.cache_dir / f"se_{safe}_{_digest(refs)}.pth")
        try:
            self.worker.request(req)
        except Exception as e:
            self.prepared[speaker_id] = False
            self.notes[speaker_id] = f"reference could not be analysed: {str(e)[:200]}"
            self._error(f"{speaker_id}: {str(e)[:300]}")
            return False
        self.prepared[speaker_id] = True
        self._prepare_reqs[speaker_id] = req
        self.notes.pop(speaker_id, None)
        return True

    def convert(self, in_wav: Path, out_wav: Path, speaker_id: str,
                source_key: Optional[str] = None) -> bool:
        """Write `in_wav` with the speaker's voice colour to `out_wav` (may be
        the same file). False = nothing usable was written; keep the clip.

        source_key (optional) names the TTS voice that spoke the clip; its
        clips then share one averaged source embedding instead of each
        clip's own."""
        if not self.prepared.get(speaker_id):
            return False
        in_wav, out_wav = Path(in_wav), Path(out_wav)
        shape = _wav_shape(in_wav)
        if shape is None:
            self._error(f"{in_wav.name}: not a readable WAV clip")
            return False
        if shape[1] < MIN_CLIP_S * shape[0]:
            return False
        req = {"op": "convert", "in": str(in_wav), "out": str(out_wav), "speaker": speaker_id,
               "source_key": source_key}
        try:
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.worker.request(req)
            except WorkerError as e:
                # A worker restarted after a crash has lost the speakers
                # prepared so far: prepare this one again (cached) and retry.
                if "was not prepared" not in str(e):
                    raise
                self.worker.request(self._prepare_reqs[speaker_id])
                self.worker.request(req)
        except Exception as e:
            self._error(f"{speaker_id}: {str(e)[:300]}")
            return False
        if _wav_shape(out_wav) != shape:
            # The worker keeps rate and length exactly; anything else would
            # shift the dub's timing.
            self._error(f"{speaker_id}: converted clip {out_wav.name} does not keep the clip's "
                        f"sample rate/length")
            if out_wav.resolve() != in_wav.resolve():
                out_wav.unlink(missing_ok=True)
            return False
        return True

    def close(self) -> None:
        self.worker.close()


def runtime_status() -> Tuple[bool, str]:
    """(runnable, reason) of the OpenVoice runtime on this PC (cached)."""
    from .local_workers import runtime_status as status
    return status("openvoice")


def make_voice_matcher(cfg) -> Optional[OpenVoiceMatcher]:
    """The matcher cfg.voice_match asks for, or None when it is off or its
    runtime cannot run here (runtime_status() says why)."""
    if str(getattr(cfg, "voice_match", "openvoice") or "off").strip().lower() != "openvoice":
        return None
    if not runtime_status()[0]:
        return None
    work_dir = getattr(cfg, "work_dir", None)
    return OpenVoiceMatcher(cache_dir=Path(work_dir) / "voice_match" if work_dir else None)
