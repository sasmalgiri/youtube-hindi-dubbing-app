"""Per-job checkpoint and within-job caches for resuming a dialogue run.

work_dir/checkpoint/state.json holds what the stages up to and including
"translate" produced (media paths, transcript, speakers, turns, voices, the
report entries so far). It is rewritten atomically (temp file + os.replace)
after each of those stages, so a crash mid-write never leaves half a file.

A resumed run (DialogueConfig.resume) restores that state and continues with
synthesize -> fit -> verify -> mix, which always run again. Unchanged lines
reuse their audio from work_dir/tts_cache/ (TTSRouter) and their shortened
text from work_dir/rewrite_cache.json (RewriteCache).

The owner's rule holds: nothing is shared across jobs. A checkpoint is only
read from the job's own work_dir and only for the same source identity, and
a fresh (non-resume) run clears the checkpoint and both caches first.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import Turn, WordRecord
from .diarization import DiarizationResult

VERSION = 1
# Stages whose results are checkpointed, in run order.
STAGES = ("extract", "separate", "transcribe", "diarize", "turns", "speakers", "translate")


def checkpoint_path(work_dir: Path) -> Path:
    return Path(work_dir) / "checkpoint" / "state.json"


def tts_cache_dir(work_dir: Path) -> Path:
    return Path(work_dir) / "tts_cache"


def rewrite_cache_path(work_dir: Path) -> Path:
    return Path(work_dir) / "rewrite_cache.json"


def write_json_atomic(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _sha1_file(p: Optional[Path]) -> Optional[str]:
    try:
        return _sha1_bytes(Path(p).read_bytes()) if p else None
    except OSError:
        return None


def _is_url(source: Optional[str]) -> bool:
    return bool(re.match(r"^https?://", source or ""))


def _source_key(source: Optional[str]) -> str:
    """A local path in one spelling (absolute, links resolved, the OS's case
    rules: D:\\v\\a.mp4 and d:/v/a.mp4 are one file); a URL as given."""
    s = (source or "").strip()
    if not s or _is_url(s):
        return s
    try:
        return os.path.normcase(os.path.realpath(os.path.abspath(s)))
    except (OSError, ValueError):
        return s


def source_identity(source: str, limit_seconds: float = 0.0, source_srt: Optional[Path] = None,
                    translated_srt: Optional[Path] = None) -> Dict[str, Any]:
    """What makes a checkpoint belong to this input: the source (hashed, so
    a signed URL is never written down), the dubbed length and the text
    files supplied, plus a local source file's size."""
    ident: Dict[str, Any] = {"source_sha1": _sha1_bytes(_source_key(source).encode("utf-8")),
                             "limit_seconds": float(limit_seconds or 0.0),
                             "source_srt_sha1": _sha1_file(source_srt),
                             "translated_srt_sha1": _sha1_file(translated_srt)}
    if source and not _is_url(source):
        try:
            ident["bytes"] = Path(source).stat().st_size
        except OSError:
            pass
    return ident


def _work_copy(work_dir: Path, source: Optional[str], files: Dict[str, Any]) -> Optional[Path]:
    """The job's own copy of a local source (default_acquire's work/source<ext>
    hard link or copy), else the checkpointed video."""
    if source and not _is_url(source):
        p = Path(work_dir) / f"source{Path(source).suffix.lower() or '.mp4'}"
        if p.is_file():
            return p
    v = resolve((files or {}).get("video"), work_dir)
    return v if v is not None and v.is_file() else None


def _same_content(a: Path, b: Path, chunk: int = 1 << 20) -> bool:
    """Same file, or same size and the same first/middle/last megabyte."""
    try:
        if os.path.samefile(a, b):
            return True
        size = a.stat().st_size
        if size != b.stat().st_size:
            return False
        with open(a, "rb") as fa, open(b, "rb") as fb:
            for off in sorted({0, max(0, size // 2 - chunk // 2), max(0, size - chunk)}):
                fa.seek(off)
                fb.seek(off)
                if fa.read(chunk) != fb.read(chunk):
                    return False
        return True
    except OSError:
        return False


def same_source(stored: Dict[str, Any], current: Dict[str, Any], work_dir: Path,
                source: Optional[str] = None, files: Optional[Dict[str, Any]] = None) -> bool:
    """Whether a checkpoint's identity `stored` is this input `current`.

    The length and text files must match exactly. A local source matches by
    its path in any spelling (also a checkpoint written before paths were
    normalised), unless the file there now has another size; an original
    that was moved away or deleted still matches when the job's own copy is
    there; one under another path matches when it holds the same media as
    the job's own copy."""
    if source is None:
        return stored == current           # nothing to compare a local file with
    for k in ("limit_seconds", "source_srt_sha1", "translated_srt_sha1"):
        if stored.get(k) != current.get(k):
            return False
    # also the hash of the path as typed: checkpoints from before normalisation
    names = {current.get("source_sha1"), _sha1_bytes((source or "").encode("utf-8"))}
    if not source or _is_url(source):
        return stored.get("source_sha1") in names
    had, has = stored.get("bytes"), current.get("bytes")
    if stored.get("source_sha1") in names:
        if has is not None:
            return had is None or had == has
        # the original is gone: the job's own copy is all a resume reads
        copy = _work_copy(work_dir, source, files or {})
        return copy is not None and (had is None or copy.stat().st_size == had)
    copy = _work_copy(work_dir, source, files or {})
    return (has is not None and has == had and copy is not None
            and _same_content(Path(source), copy))


def save(work_dir: Path, state: Dict[str, Any]) -> Path:
    return write_json_atomic(checkpoint_path(work_dir),
                             dict(state, version=VERSION, saved_at=round(time.time(), 3)))


def load(work_dir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """(state, "") or (None, why it cannot be used)."""
    p = checkpoint_path(work_dir)
    if not p.exists():
        return None, "no checkpoint in this job's work folder"
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"checkpoint unreadable ({str(e)[:80]})"
    if not isinstance(state, dict) or state.get("version") != VERSION:
        return None, "checkpoint from another version"
    return state, ""


def stages_on_disk(work_dir: Path) -> Optional[List[str]]:
    """The stages the checkpoint in work_dir says it finished, whatever its
    version or identity; [] when there is none, None when it is unreadable."""
    p = checkpoint_path(work_dir)
    if not p.exists():
        return []
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
        return [str(s) for s in state.get("completed") or []]
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def unusable_reason(state: Dict[str, Any], identity: Dict[str, Any], work_dir: Path,
                    source: Optional[str] = None) -> str:
    """"" when `state` can resume this job, else the reason it cannot.
    `source` (the job's source as given) lets a local file match in another
    spelling or after it was moved (see same_source)."""
    if not same_source(state.get("identity") or {}, identity, work_dir, source,
                       state.get("files") or {}):
        return "the checkpoint belongs to a different source"
    if not completed_prefix(state):
        return "the checkpoint holds no finished stage"
    for name, rel_path in (state.get("files") or {}).items():
        if rel_path and not resolve(rel_path, work_dir).exists():
            return f"{name} file is missing"
    return ""


def completed_prefix(state: Dict[str, Any]) -> List[str]:
    """The leading checkpointed stages that finished (resume continues after them)."""
    done = set(state.get("completed") or [])
    out = []
    for s in STAGES:
        if s not in done:
            break
        out.append(s)
    return out


def clear(work_dir: Path):
    """A fresh run starts from scratch: drop any earlier checkpoint and cache."""
    shutil.rmtree(checkpoint_path(work_dir).parent, ignore_errors=True)
    shutil.rmtree(tts_cache_dir(work_dir), ignore_errors=True)
    try:
        rewrite_cache_path(work_dir).unlink(missing_ok=True)
    except OSError:
        pass


# ── paths: relative to the work folder when inside it ──────────────────────
def rel(p: Optional[Path], work_dir: Path) -> Optional[str]:
    if not p:
        return None
    p = Path(p)
    try:
        return p.resolve().relative_to(Path(work_dir).resolve()).as_posix()
    except ValueError:
        return str(p)


def resolve(s: Optional[str], work_dir: Path) -> Optional[Path]:
    if not s:
        return None
    p = Path(s)
    return p if p.is_absolute() else Path(work_dir) / p


# ── stage results <-> JSON ─────────────────────────────────────────────────
def diar_to_dict(d: Optional[DiarizationResult]) -> Optional[Dict[str, Any]]:
    if d is None:
        return None
    return {"regular": [[float(s), float(e), str(k)] for s, e, k in d.regular],
            "exclusive": [[float(s), float(e), str(k)] for s, e, k in d.exclusive],
            "embeddings": {str(k): [float(x) for x in v] for k, v in (d.embeddings or {}).items()},
            "backend": d.backend, "detail": d.detail}


def diar_from_dict(d: Optional[Dict[str, Any]]) -> Optional[DiarizationResult]:
    if d is None:
        return None
    return DiarizationResult([tuple(x) for x in d.get("regular") or []],
                             [tuple(x) for x in d.get("exclusive") or []],
                             dict(d.get("embeddings") or {}), d.get("backend", "none"),
                             d.get("detail", ""))


def words_from_list(items: Sequence[Dict[str, Any]]) -> List[WordRecord]:
    return [WordRecord(**w) for w in items or []]


def turns_from_list(items: Sequence[Dict[str, Any]]) -> List[Turn]:
    return [Turn(**t) for t in items or []]


# ── LLM line-shortening cache ──────────────────────────────────────────────
class RewriteCache:
    """Shortened lines of THIS job, keyed by (turn_id, current Hindi, target
    ratio rounded to 0.05): a resumed run asks the LLM again only for lines
    that changed. Only accepted rewrites are kept: neither a refusal (it may
    have been an outage) nor a rewrite the fit rejected because it did not
    make the line shorter (the orchestrator stores a rewrite only after the
    fit has kept it)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.data: Dict[str, str] = {}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data = {k: v for k, v in loaded.items() if isinstance(v, str)}
        except (OSError, ValueError):
            pass

    @staticmethod
    def key(turn_id: str, current_hi: str, ratio: float) -> str:
        return f"{turn_id}|{round(float(ratio) / 0.05) * 0.05:.2f}|{current_hi}"

    def get(self, turn_id: str, current_hi: str, ratio: float) -> Optional[str]:
        with self._lock:
            return self.data.get(self.key(turn_id, current_hi, ratio))

    def put(self, turn_id: str, current_hi: str, ratio: float, new_hi: str):
        with self._lock:
            self.data[self.key(turn_id, current_hi, ratio)] = new_hi
            self._write()

    def delete(self, turn_id: str, current_hi: str, ratio: float):
        """Forget a rewrite the fit rejected (it did not make the line shorter)."""
        with self._lock:
            if self.data.pop(self.key(turn_id, current_hi, ratio), None) is not None:
                self._write()

    def _write(self):
        try:
            write_json_atomic(self.path, self.data)
        except OSError:
            pass
