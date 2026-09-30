"""Diarization adapter + word-level speaker attribution.

Backends (tried in order of `model` preference):
  * pyannote/speaker-diarization-community-1  (pyannote.audio 4.x: `token=`,
    returns DiarizeOutput with .speaker_diarization, .exclusive_speaker_diarization
    and .speaker_embeddings)
  * pyannote/speaker-diarization-3.1          (pyannote.audio 3.x: `use_auth_token=`,
    returns an Annotation; exclusive track derived here)

Both the regular (overlap-aware) and the exclusive track are kept: the
exclusive track drives word attribution, the regular track flags simultaneous
speech. Words that cannot be attributed stay unknown (speaker_id=None).
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .contracts import WordRecord

Seg = Tuple[float, float, str]

MODEL_IDS = {
    "community-1": "pyannote/speaker-diarization-community-1",
    "3.1": "pyannote/speaker-diarization-3.1",
}


class DiarizationUnavailable(RuntimeError):
    """Raised with an actionable message when diarization cannot run."""


@dataclass
class DiarizationResult:
    regular: List[Seg]
    exclusive: List[Seg]
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    backend: str = "none"
    detail: str = ""

    @property
    def speakers(self) -> List[str]:
        return sorted({s for _, _, s in self.regular} | {s for _, _, s in self.exclusive})

    def speech_seconds(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for s, e, spk in self.exclusive or self.regular:
            out[spk] = out.get(spk, 0.0) + max(0.0, e - s)
        return out

    def clean_ranges(self, speaker: str, min_len: float = 0.5) -> List[Tuple[float, float]]:
        """Exclusive ranges of `speaker` that do not intersect other speakers."""
        own = [(s, e) for s, e, k in self.exclusive or self.regular if k == speaker]
        others = [(s, e) for s, e, k in self.regular if k != speaker]
        out = []
        for s, e in own:
            if e - s < min_len:
                continue
            if any(os_ < e and oe > s for os_, oe in others):
                continue
            out.append((s, e))
        return out


# ── exclusive-track derivation ─────────────────────────────────────────────
def derive_exclusive(regular: Sequence[Seg]) -> List[Seg]:
    """Collapse overlaps so at most one speaker is active at any instant.

    In an overlapped region the speaker whose covering segment is shortest
    (usually the interjection) wins. pyannote 4 supplies its own exclusive
    track; this is only used for 3.x output.
    """
    if not regular:
        return []
    bounds = sorted({t for s, e, _ in regular for t in (s, e)})
    out: List[Seg] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= 1e-6:
            continue
        mid = (a + b) / 2
        active = [(e - s, spk) for s, e, spk in regular if s <= mid < e]
        if not active:
            continue
        spk = min(active)[1]
        if out and out[-1][2] == spk and abs(out[-1][1] - a) < 1e-6:
            out[-1] = (out[-1][0], b, spk)
        else:
            out.append((a, b, spk))
    return out


def _hf_error_message(err: Exception, model_id: str) -> str:
    msg = str(err)
    low = msg.lower()
    if any(k in low for k in ("401", "403", "gated", "unauthorized", "access to model")):
        return (f"Cannot access {model_id}: accept the model's user conditions at "
                f"https://huggingface.co/{model_id} (and its segmentation dependency) "
                f"with the account whose token is in HF_TOKEN.")
    return f"{model_id} failed: {msg[:300]}"


def run_pyannote(wav_path: Path, hf_token: str, model: str = "community-1",
                 num_speakers: Optional[int] = None,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None,
                 heartbeat: Optional[Callable[[float], None]] = None,
                 timeout: float = 3600.0) -> DiarizationResult:
    """Run pyannote with the requested model, falling back to the other one.

    Each attempt runs in a spawned CHILD process (_pyannote_child): a native
    crash or CUDA OOM kills only the child, all GPU memory is returned, and
    the child can hide NeMo (pyannote imports it optionally, and NeMo 2.7
    crashes at import on torch 2.4 with an AttributeError pyannote does not
    catch). Raises DiarizationUnavailable with an actionable message if
    neither model runs.
    """
    if not hf_token:
        raise DiarizationUnavailable(
            "HF_TOKEN is not set. Create a Hugging Face read token, accept the "
            "conditions of pyannote/speaker-diarization-community-1, and put "
            "HF_TOKEN=... in backend/.env.")
    try:
        from importlib.metadata import version as _pkg_version
        version = _pkg_version("pyannote.audio")
    except Exception as e:
        raise DiarizationUnavailable(
            "pyannote.audio is not installed (pip install -r backend/requirements-dialogue.txt)") from e
    major = int(version.split(".")[0]) if version[:1].isdigit() else 0

    order = [model] + [m for m in MODEL_IDS if m != model]
    errors: List[str] = []
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            kwargs["max_speakers"] = int(max_speakers)

    for key in order:
        model_id = MODEL_IDS[key]
        if key == "community-1" and major < 4:
            errors.append(f"{model_id} needs pyannote.audio>=4 (installed {version})")
            continue
        try:
            data = _run_child(wav_path, hf_token, key, kwargs, heartbeat, timeout)
            return DiarizationResult(
                [tuple(x) for x in data["regular"]], [tuple(x) for x in data["exclusive"]],
                data.get("embeddings") or {}, backend=f"pyannote-{key}",
                detail=data.get("detail", ""))
        except DiarizationUnavailable:
            raise
        except Exception as e:  # try next backend
            errors.append(_hf_error_message(e, model_id))
    raise DiarizationUnavailable("; ".join(errors) or "diarization failed")


def _run_child(wav_path: Path, hf_token: str, key: str, kwargs: Dict,
               heartbeat: Optional[Callable[[float], None]], timeout: float) -> Dict:
    """Spawn _pyannote_child; heartbeat every 10 s; returns its JSON result."""
    import json
    import multiprocessing as mp
    import tempfile

    fd, result_path = tempfile.mkstemp(suffix=".json", prefix="dlg_diarize_")
    os.close(fd)
    try:
        p = mp.get_context("spawn").Process(
            target=_pyannote_child, args=(str(wav_path), hf_token, key, kwargs, result_path),
            daemon=True)
        p.start()
        t0 = time.time()
        next_beat = t0 + 10.0
        while p.is_alive():
            p.join(1.0)
            now = time.time()
            if now - t0 > timeout:
                p.kill()
                p.join(5)
                raise RuntimeError(f"diarization timed out after {int(timeout)}s")
            if heartbeat and now >= next_beat:
                next_beat = now + 10.0
                try:
                    heartbeat(now - t0)
                except Exception:
                    pass
        data = {}
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
        # A complete result is trusted whatever the exit code: it is written
        # before the child exits (Windows CUDA teardown can fail-fast).
        if data.get("error") is None and "regular" in data:
            return data
        raise RuntimeError(data.get("error") or f"diarization child exited with code {p.exitcode}")
    finally:
        try:
            Path(result_path).unlink(missing_ok=True)
        except Exception:
            pass


def _pyannote_child(wav_path: str, hf_token: str, key: str, kwargs: Dict,
                    result_path: str) -> None:
    """Child-process body of run_pyannote (top level so spawn can pickle it)."""
    import json
    import sys
    import warnings
    try:
        sys.modules["nemo"] = None  # see run_pyannote docstring
        # Audio goes in memory, so pyannote 4's torchcodec warning is moot.
        warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed")
        import torch
        import pyannote.audio as pa
        major = int(pa.__version__.split(".")[0])
        model_id = MODEL_IDS[key]
        if key == "3.1" and major >= 4:
            # 3.1's recipe (agglomerative clustering) needs no PLDA, but
            # pyannote 4's from_pretrained would also fetch the gated
            # community-1 PLDA. Build it explicitly from its config.yaml.
            import pyannote.audio.pipelines.speaker_diarization as _sd
            _orig_get_plda = _sd.get_plda
            _sd.get_plda = (lambda plda, **kw:
                            None if plda is None else _orig_get_plda(plda, **kw))
            pipe = _sd.SpeakerDiarization(
                segmentation="pyannote/segmentation-3.0",
                embedding="pyannote/wespeaker-voxceleb-resnet34-LM",
                embedding_exclude_overlap=True, clustering="AgglomerativeClustering",
                plda=None, embedding_batch_size=32, segmentation_batch_size=32,
                token=hf_token)
            pipe.instantiate({"clustering": {"method": "centroid", "min_cluster_size": 12,
                                             "threshold": 0.7045654963945799},
                              "segmentation": {"min_duration_off": 0.0}})
        else:
            from pyannote.audio import Pipeline as PyannotePipeline
            if major >= 4:
                pipe = PyannotePipeline.from_pretrained(model_id, token=hf_token)
            else:
                pipe = PyannotePipeline.from_pretrained(model_id, use_auth_token=hf_token)
            if pipe is None:
                raise RuntimeError("from_pretrained returned None (gated model not accepted?)")
        note = ""
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
        audio_input = _audio_input(Path(wav_path))
        try:
            output = pipe(audio_input, **kwargs)
        except RuntimeError as oom:
            if "out of memory" not in str(oom).lower():
                raise
            # Bounded, reported fallback: same model on CPU (slower).
            torch.cuda.empty_cache()
            pipe.to(torch.device("cpu"))
            output = pipe(audio_input, **kwargs)
            note = "; CUDA OOM -> reran on CPU"
        res = _convert_output(output, key)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"regular": res.regular, "exclusive": res.exclusive,
                       "embeddings": res.embeddings, "detail": res.detail + note,
                       "error": None}, f)
    except Exception as e:
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({"error": f"{type(e).__name__}: {e}"}, f)
        except Exception:
            pass
        os._exit(1)
    os._exit(0)


def _audio_input(wav_path: Path):
    """In-memory waveform avoids torchcodec/soundfile decoding differences."""
    try:
        import numpy as np
        import torch
        from .voice_analysis import read_wav_mono
        samples, sr = read_wav_mono(Path(wav_path))
        return {"waveform": torch.from_numpy(np.ascontiguousarray(samples))[None, :],
                "sample_rate": sr}
    except Exception:
        return str(wav_path)


def _annotation_to_segs(annotation) -> List[Seg]:
    segs: List[Seg] = []
    if hasattr(annotation, "itertracks"):
        for turn, _, spk in annotation.itertracks(yield_label=True):
            segs.append((float(turn.start), float(turn.end), str(spk)))
    else:  # iterable of (turn, speaker)
        for turn, spk in annotation:
            segs.append((float(turn.start), float(turn.end), str(spk)))
    return sorted(segs)


def _convert_output(output, key: str) -> DiarizationResult:
    if hasattr(output, "speaker_diarization"):
        regular = _annotation_to_segs(output.speaker_diarization)
        excl_ann = getattr(output, "exclusive_speaker_diarization", None)
        exclusive = _annotation_to_segs(excl_ann) if excl_ann is not None else derive_exclusive(regular)
        embeddings: Dict[str, List[float]] = {}
        emb = getattr(output, "speaker_embeddings", None)
        if emb is not None:
            labels = list(output.speaker_diarization.labels())
            for i, lab in enumerate(labels):
                if i < len(emb):
                    embeddings[str(lab)] = [float(x) for x in emb[i]]
        return DiarizationResult(regular, exclusive, embeddings,
                                 backend=f"pyannote-{key}", detail="DiarizeOutput")
    regular = _annotation_to_segs(output)
    return DiarizationResult(regular, derive_exclusive(regular), {},
                             backend=f"pyannote-{key}", detail="Annotation (exclusive derived)")


def _release(obj):
    try:
        del obj
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


# ── chunk reconciliation ───────────────────────────────────────────────────
def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def reconcile_chunks(chunks: Sequence[Tuple[float, DiarizationResult]],
                     threshold: float = 0.6) -> DiarizationResult:
    """Merge independently diarized chunks into one global speaker space.

    Each chunk's local labels are mapped to the most similar global speaker by
    embedding cosine similarity (>= threshold), otherwise a new global speaker
    is created. Without embeddings, identities cannot be reconciled and a
    ValueError is raised rather than silently resetting character identity.
    """
    global_emb: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}
    regular: List[Seg] = []
    exclusive: List[Seg] = []
    for offset, res in chunks:
        if res.speakers and not res.embeddings:
            raise ValueError("chunk reconciliation requires speaker embeddings")
        mapping: Dict[str, str] = {}
        for local in res.speakers:
            emb = res.embeddings[local]
            best, best_sim = None, threshold
            for gid, gemb in global_emb.items():
                if gid in mapping.values():
                    continue
                sim = _cos(emb, gemb)
                if sim >= best_sim:
                    best, best_sim = gid, sim
            if best is None:
                best = f"SPEAKER_{len(global_emb):02d}"
                global_emb[best] = list(emb)
                counts[best] = 1
            else:  # running mean of embeddings
                n = counts[best]
                global_emb[best] = [(g * n + x) / (n + 1) for g, x in zip(global_emb[best], emb)]
                counts[best] = n + 1
            mapping[local] = best
        regular += [(s + offset, e + offset, mapping[k]) for s, e, k in res.regular]
        exclusive += [(s + offset, e + offset, mapping[k]) for s, e, k in res.exclusive]
    return DiarizationResult(sorted(_dedupe(regular)), sorted(_dedupe(exclusive)),
                             global_emb, backend="reconciled", detail=f"{len(chunks)} chunks")


def _dedupe(segs: List[Seg]) -> List[Seg]:
    """Merge same-speaker segments that overlap (from chunk overlap windows)."""
    out: List[Seg] = []
    for s, e, k in sorted(segs, key=lambda x: (x[2], x[0])):
        if out and out[-1][2] == k and s <= out[-1][1] + 0.05:
            out[-1] = (out[-1][0], max(out[-1][1], e), k)
        else:
            out.append((s, e, k))
    return out


# ── word attribution ───────────────────────────────────────────────────────
def assign_words(words: List[WordRecord], diar: Optional[DiarizationResult],
                 min_share: float = 0.5, nearest_tol: float = 0.3,
                 overlap_share: float = 0.3) -> Dict[str, int]:
    """Assign speaker_id per word in place. Returns attribution statistics.

    * Primary evidence: overlap with the exclusive track (share of the word).
    * A word with little overlap may take the nearest exclusive segment within
      `nearest_tol` seconds; otherwise it stays unknown (speaker_id=None).
    * A word covered by >=2 speakers in the regular track is flagged overlap.
    """
    stats = {"attributed": 0, "nearest": 0, "unknown": 0, "overlap": 0}
    if diar is None or not (diar.exclusive or diar.regular):
        for w in words:
            w.speaker_id = None
            w.attribution = {"method": "none", "reason": "no_diarization"}
            stats["unknown"] += 1
        return stats
    excl = diar.exclusive or derive_exclusive(diar.regular)
    for w in words:
        dur = max(w.end - w.start, 0.02)
        shares: Dict[str, float] = {}
        for s, e, k in excl:
            if e <= w.start or s >= w.end:
                continue
            shares[k] = shares.get(k, 0.0) + (min(e, w.end) - max(s, w.start)) / dur
        reg_cover: Dict[str, float] = {}
        for s, e, k in diar.regular:
            if e <= w.start or s >= w.end:
                continue
            reg_cover[k] = reg_cover.get(k, 0.0) + (min(e, w.end) - max(s, w.start)) / dur
        w.overlap = sum(1 for v in reg_cover.values() if v >= overlap_share) >= 2
        if w.overlap:
            stats["overlap"] += 1
        if shares:
            best = max(shares, key=shares.get)
            if shares[best] >= min_share or (len(shares) == 1 and shares[best] >= 0.2):
                w.speaker_id = best
                w.attribution = {"method": "exclusive_overlap",
                                 "share": round(min(1.0, shares[best]), 3)}
                stats["attributed"] += 1
                continue
        # nearest segment within tolerance
        best_k, best_d = None, nearest_tol
        mid = (w.start + w.end) / 2
        for s, e, k in excl:
            d = 0.0 if s <= mid <= e else min(abs(mid - s), abs(mid - e))
            if d <= best_d:
                best_k, best_d = k, d
        if best_k is not None:
            w.speaker_id = best_k
            w.attribution = {"method": "nearest_segment", "distance_s": round(best_d, 3)}
            stats["nearest"] += 1
        else:
            w.speaker_id = None
            w.attribution = {"method": "none", "reason": "no_speaker_near_word",
                             "shares": {k: round(v, 3) for k, v in shares.items()}}
            stats["unknown"] += 1
    return stats


def hf_token_from_env() -> str:
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()
