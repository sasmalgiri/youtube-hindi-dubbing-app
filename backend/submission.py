"""Creator submission (W2) and funded jobs (W4, W6 core).

Flow: upload → **draft** (bounded streamed upload, checksum, real probe)
→ **quote** (server-side, bound to the draft and the balance) → **rights
declaration** (fresh, unchecked by default, versioned) → **submit**
(idempotent per account + key; the quote is single-use; seconds reserved in
the ledger in the same request). The job's thread is wrapped by
``run_billed`` so a delivered job settles once and anything else releases.

A quote with cash due is refused until Razorpay (W5) can collect it: nothing
here charges money.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

RIGHTS_VERSION = "2026-11-20.1"
RIGHTS_TEXT = "I own this material or have the rights needed to create this dub and use it as intended."

MAX_UPLOAD_BYTES = int(os.environ.get("VOICEDUB_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
MAX_SOURCE_SECONDS = int(os.environ.get("VOICEDUB_MAX_SOURCE_MINUTES", "180")) * 60
ALLOWED_EXT = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mp3", ".wav", ".m4a")

SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT,
    path        TEXT NOT NULL,
    filename    TEXT NOT NULL,
    size        INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    duration    REAL NOT NULL,
    probe       TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS quotes (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT,
    draft_id    TEXT NOT NULL,
    body        TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    used_job_id TEXT
);
CREATE TABLE IF NOT EXISTS rights (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id    TEXT,
    draft_id    TEXT NOT NULL,
    version     TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT,
    body        TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions (
    owner_key   TEXT PRIMARY KEY,          -- owner_id || ':' || idempotency key
    job_id      TEXT,
    quote_id    TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
"""


class SubmitError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ── probe ────────────────────────────────────────────────────────────────
def _ffprobe() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def probe_media(path: Path) -> Dict[str, Any]:
    """Real container facts (Q01). Raises SubmitError for unusable media."""
    try:
        r = subprocess.run([_ffprobe(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SubmitError(f"Could not inspect the file ({type(e).__name__})", 422)
    if r.returncode != 0:
        raise SubmitError("This file could not be read as audio or video (damaged or unsupported).", 422)
    data = json.loads(r.stdout or "{}")
    streams = data.get("streams") or []
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    video = [s for s in streams if s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic") != 1]
    try:
        duration = float((data.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if not audio:
        raise SubmitError("This file has no audio track, so there is nothing to dub.", 422)
    if duration <= 0:
        raise SubmitError("The file's duration could not be measured (damaged or still uploading?).", 422)
    if duration > MAX_SOURCE_SECONDS:
        raise SubmitError(f"The source is {duration / 60:.0f} min; the limit is {MAX_SOURCE_SECONDS // 60} min.", 413)
    rot = 0
    for s in video:
        tags = s.get("tags") or {}
        for sd in s.get("side_data_list") or []:
            if "rotation" in sd:
                rot = int(sd["rotation"])
        if "rotate" in tags:
            rot = int(tags["rotate"])
    warnings = []
    if len(audio) > 1:
        warnings.append(f"{len(audio)} audio tracks: the first one is used")
    return {
        "duration": duration,
        "container": (data.get("format") or {}).get("format_name", ""),
        "video": [{"codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
                   "fps": s.get("avg_frame_rate")} for s in video],
        "audio": [{"codec": s.get("codec_name"), "channels": s.get("channels"), "sample_rate": s.get("sample_rate"),
                   "language": (s.get("tags") or {}).get("language")} for s in audio],
        "rotation": rot,
        "warnings": warnings,
    }


# ── store ────────────────────────────────────────────────────────────────
class SubmissionStore:
    def __init__(self, db_path: Path, drafts_dir: Path, clock: Callable[[], float] = time.time):
        self.db_path = Path(db_path)
        self.drafts_dir = Path(drafts_dir)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def _now(self) -> int:
        return int(self._clock())

    def _x(self, sql: str, args=()):
        """Every statement under one lock: the connection is shared by request threads."""
        with self._lock:
            return self._conn.execute(sql, args)

    def _one(self, sql: str, args=()):
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    def close(self) -> None:
        self._conn.close()

    def owner_counts(self, owner_id: Optional[str]) -> Dict[str, int]:
        return {"drafts": self._one("SELECT COUNT(*) FROM drafts WHERE owner_id IS ?", (owner_id,))[0],
                "batches": self._one("SELECT COUNT(*) FROM batches WHERE owner_id IS ?", (owner_id,))[0]}

    def purge_owner(self, owner_id: str) -> Dict[str, int]:
        """L04: remove an account's uploads, quotes and batches. Rights
        confirmations are kept (evidence of what the account declared)."""
        rows = self._x("SELECT path FROM drafts WHERE owner_id=?", (owner_id,)).fetchall()
        for (path,) in rows:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        n = self.owner_counts(owner_id)
        self._x("DELETE FROM drafts WHERE owner_id=?", (owner_id,))
        self._x("DELETE FROM quotes WHERE owner_id=?", (owner_id,))
        self._x("DELETE FROM batches WHERE owner_id=?", (owner_id,))
        self._x("DELETE FROM submissions WHERE owner_key LIKE ?", (f"{owner_id}:%",))
        return n

    # drafts
    async def save_upload(self, upload, owner_id: Optional[str]) -> Dict[str, Any]:
        """Stream to disk with a byte limit and an incremental checksum, then probe."""
        name = Path(upload.filename or "").name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise SubmitError(f"Unsupported file type {ext or '(none)'}; use {', '.join(ALLOWED_EXT)}", 415)
        did = "d_" + secrets.token_hex(8)
        part = self.drafts_dir / f"{did}{ext}.part"
        final = self.drafts_dir / f"{did}{ext}"
        h, size = hashlib.sha256(), 0
        try:
            with open(part, "wb") as f:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise SubmitError(f"The file is larger than {MAX_UPLOAD_BYTES // 2**20} MB", 413)
                    h.update(chunk)
                    f.write(chunk)
            if size == 0:
                raise SubmitError("The upload was empty (interrupted?)", 400)
            os.replace(part, final)
            probe = probe_media(final)
        except BaseException:
            for p in (part, final):
                try:
                    p.unlink()
                except OSError:
                    pass
            raise
        self._x("INSERT INTO drafts(id, owner_id, path, filename, size, sha256, duration, probe, created_at) "
                           "VALUES (?,?,?,?,?,?,?,?,?)",
                           (did, owner_id, str(final), name[:200], size, h.hexdigest(), probe["duration"],
                            json.dumps(probe), self._now()))
        return self.draft(did, owner_id)

    def draft(self, draft_id: str, owner_id: Optional[str]) -> Dict[str, Any]:
        r = self._one("SELECT id, owner_id, path, filename, size, sha256, duration, probe, created_at FROM drafts WHERE id=?",
                      (draft_id,))
        if r is None or r[1] != owner_id:
            raise SubmitError("Draft not found", 404)
        return {"id": r[0], "path": r[2], "filename": r[3], "size": r[4], "sha256": r[5], "duration": r[6],
                "probe": json.loads(r[7]), "created_at": r[8]}

    # quotes
    def make_quote(self, *, draft_id: str, owner_id: Optional[str], languages: List[str], minutes: int,
                   style: str, quality: str, batch: bool, available: Dict[str, int]) -> Dict[str, Any]:
        from billing import catalog
        from billing.quote import QuoteRefused, make_quote
        d = self.draft(draft_id, owner_id)
        end = None if not minutes else float(minutes) * 60.0
        try:
            q = make_quote(account_id=owner_id or "local", source_id=d["sha256"], source_duration=d["duration"],
                           languages=languages, style=style, quality=quality, start=0, end=end, batch=batch,
                           available=available, now=self._now())
        except (QuoteRefused, catalog.NotPriced) as e:
            raise SubmitError(str(e), 422)
        body = q.to_dict()
        body["minutes"] = int(minutes or 0)
        body["draft_id"] = draft_id
        qid = "q_" + secrets.token_hex(8)
        self._x("INSERT INTO quotes(id, owner_id, draft_id, body, created_at) VALUES (?,?,?,?,?)",
                           (qid, owner_id, draft_id, json.dumps(body), self._now()))
        body["id"] = qid
        body["display"] = {
            "cash": catalog.fmt_money(q.cash_minor), "tax": catalog.fmt_money(q.tax_minor),
            "standard": catalog.fmt_seconds(q.standard_seconds), "premium": catalog.fmt_seconds(q.premium_seconds),
            "billed": catalog.fmt_seconds(sum(l.billable_seconds for l in q.lines)),
        }
        return body

    def quote(self, quote_id: str, owner_id: Optional[str]) -> Dict[str, Any]:
        r = self._one("SELECT owner_id, body, used_job_id FROM quotes WHERE id=?", (quote_id,))
        if r is None or r[0] != owner_id:
            raise SubmitError("Quote not found", 404)
        body = json.loads(r[1])
        body["id"], body["used_job_id"] = quote_id, r[2]
        return body

    # rights (L01)
    def record_rights(self, owner_id: Optional[str], draft_id: str, accepted: bool, version: str) -> int:
        if accepted is not True:
            raise SubmitError("Confirm that you own this material or have the rights to dub it.", 400)
        if version != RIGHTS_VERSION:
            raise SubmitError("The rights statement changed; read it again and confirm.", 409)
        cur = self._x("INSERT INTO rights(owner_id, draft_id, version, text, created_at) VALUES (?,?,?,?,?)",
                                 (owner_id, draft_id, version, RIGHTS_TEXT, self._now()))
        return cur.lastrowid

    # idempotent submit bookkeeping
    def claim(self, owner_id: Optional[str], key: str, quote_id: str) -> Optional[str]:
        """None = this request owns the submission; a job id = it was already submitted."""
        if not key or len(key) > 100:
            raise SubmitError("A submission key is required", 400)
        ok = f"{owner_id or 'local'}:{key}"
        try:
            self._x("INSERT INTO submissions(owner_key, job_id, quote_id, created_at) VALUES (?,?,?,?)",
                               (ok, None, quote_id, self._now()))
            return None
        except sqlite3.IntegrityError:
            for _ in range(50):                    # the first request may still be creating the job
                r = self._one("SELECT job_id FROM submissions WHERE owner_key=?", (ok,))
                if r and r[0]:
                    return r[0]
                time.sleep(0.1)
            raise SubmitError("This submission is still being processed; refresh in a moment.", 409)

    def complete_claim(self, owner_id: Optional[str], key: str, job_id: str) -> None:
        self._x("UPDATE submissions SET job_id=? WHERE owner_key=?", (job_id, f"{owner_id or 'local'}:{key}"))

    def drop_claim(self, owner_id: Optional[str], key: str) -> None:
        self._x("DELETE FROM submissions WHERE owner_key=? AND job_id IS NULL", (f"{owner_id or 'local'}:{key}",))

    # batches (W8: S04, M06) — server records, so a batch survives logout and restarts
    def create_batch(self, owner_id: Optional[str], body: Dict[str, Any]) -> str:
        bid = "b_" + secrets.token_hex(8)
        now = self._now()
        self._x("INSERT INTO batches(id, owner_id, body, created_at, updated_at) VALUES (?,?,?,?,?)",
                (bid, owner_id, json.dumps(body), now, now))
        return bid

    def batch(self, batch_id: str, owner_id: Optional[str]) -> Dict[str, Any]:
        r = self._one("SELECT owner_id, body, created_at FROM batches WHERE id=?", (batch_id,))
        if r is None or r[0] != owner_id:
            raise SubmitError("Batch not found", 404)
        body = json.loads(r[1])
        body["id"], body["created_at"] = batch_id, r[2]
        return body

    def save_batch(self, batch_id: str, body: Dict[str, Any]) -> None:
        body = {k: v for k, v in body.items() if k not in ("id", "created_at")}
        self._x("UPDATE batches SET body=?, updated_at=? WHERE id=?", (json.dumps(body), self._now(), batch_id))

    def batches(self, owner_id: Optional[str]) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT id, body, created_at FROM batches WHERE owner_id IS ? "
                                      "ORDER BY created_at DESC", (owner_id,)).fetchall()
        out = []
        for bid, body, created in rows:
            b = json.loads(body)
            b["id"], b["created_at"] = bid, created
            out.append(b)
        return out

    def use_quote(self, quote_id: str, job_id: str) -> bool:
        cur = self._x("UPDATE quotes SET used_job_id=? WHERE id=? AND used_job_id IS NULL", (job_id, quote_id))
        return cur.rowcount == 1


# ── billing around a job's thread ────────────────────────────────────────
def run_billed(job, ledger_factory: Callable[[], Any], run: Callable[[], None], save: Callable[[Any], None]) -> None:
    """Run the job; settle its reservation once if delivered, otherwise release it."""
    try:
        run()
    finally:
        close_billing(job, ledger_factory, save)


def _store_delivery(job, b: Dict) -> None:
    """Copy the checked output to private storage when configured. A storage
    failure never undoes delivery: the file is still on this PC's disk and the
    error is recorded for the operator (retried by the next download)."""
    import storage
    if not storage.configured():
        return
    srt = getattr(job, "subtitles_path", None)
    try:
        b["stored"] = storage.put_job_files(getattr(job, "owner_id", None), job.id,
                                            {"video": Path(str(job.result_path)), "srt": Path(srt) if srt else None})
        b.pop("store_error", None)
    except Exception as e:
        b["store_error"] = f"{type(e).__name__}: {str(e)[:160]}"


def verify_result(path: Path, min_seconds: float = 1.0, silence_db: float = -60.0) -> Optional[str]:
    """Check a delivered file before it counts as Complete: it decodes, has an
    audio stream, is long enough and is not silent. Returns a reason or None."""
    import re
    try:
        if path.stat().st_size == 0:
            return "empty file"
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                            capture_output=True, text=True, timeout=60)
        if pr.returncode != 0:
            return "file does not decode"
        info = json.loads(pr.stdout or "{}")
        if not any(st.get("codec_type") == "audio" for st in info.get("streams", [])):
            return "no audio track"
        dur = float(info.get("format", {}).get("duration") or 0)
        if dur < min_seconds:
            return f"too short ({dur:.1f}s)"
        vd = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
                             "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True, timeout=600)
        m = re.search(r"max_volume:\s*(-?[\d.]+|-inf) dB", vd.stderr or "")
        if not m:
            return "audio could not be measured"
        if m.group(1) == "-inf" or float(m.group(1)) < silence_db:
            return "audio is silent"
        return None
    except Exception as e:                    # a check that cannot run must not pass
        return f"check failed ({type(e).__name__})"


def close_billing(job, ledger_factory: Callable[[], Any], save: Callable[[Any], None]) -> Optional[str]:
    b = getattr(job, "billing", None)
    if not b or b.get("closed"):
        return None
    led = ledger_factory()
    try:
        delivered = job.state == "done" and bool(getattr(job, "result_path", None)) and \
            Path(str(job.result_path)).exists()
        if delivered:
            problem = verify_result(Path(str(job.result_path)))
            if problem:                       # P07: never "Complete" (or charged) on a bad output
                delivered = False
                job.state = "error"
                job.error = f"Result failed the delivery check: {problem}"
                job.message = "The output did not pass the delivery check; your time was returned."
        if delivered:
            led.settle(b["account"], b["reserve_key"])
            b["closed"] = "settled"
            _store_delivery(job, b)
        else:
            led.release(b["account"], b["reserve_key"], note=f"job {job.id} ended as {job.state}")
            b["closed"] = "released"
        b["closed_at"] = int(time.time())
    finally:
        led.close()
    save(job)
    return b["closed"]
