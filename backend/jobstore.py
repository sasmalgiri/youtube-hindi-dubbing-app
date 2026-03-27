"""
SQLite-backed job persistence for the YouTube Hindi Dubbing API.

Keeps the in-memory JOBS dict in sync with a local SQLite database so that
all jobs survive server restarts.  Only serialisable fields are persisted
(the cancel_event threading.Event and events SSE list are re-created on load).

Usage (in app.py):
    from jobstore import JobStore
    store = JobStore(DB_PATH)
    store.load_all(JOBS)          # on startup: populate JOBS from DB
    store.save(job)               # after any state change
    store.delete(job_id)          # when a job is deleted
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import at runtime; Job is imported lazily in load_all()
    from app import Job

# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    state       TEXT NOT NULL DEFAULT 'queued',
    payload     TEXT NOT NULL,          -- full JSON blob
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
"""

# Fields that live only in memory (not persisted)
_SKIP_FIELDS = {"events", "cancel_event"}

# Fields that are Path objects and need str conversion
_PATH_FIELDS = {"result_path"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _job_to_dict(job: "Job") -> dict:
    """Serialise a Job dataclass to a plain JSON-safe dict."""
    d: Dict[str, Any] = {}
    for k, v in job.__dict__.items():
        if k in _SKIP_FIELDS:
            continue
        if k in _PATH_FIELDS:
            d[k] = str(v) if v is not None else None
        elif hasattr(v, "__dict__"):
            # Pydantic model (original_req)
            try:
                d[k] = v.dict()
            except Exception:
                d[k] = None
        else:
            d[k] = v
    return d


def _dict_to_job(data: dict) -> "Job":
    """Deserialise a dict back to a Job dataclass instance."""
    from app import Job, JobCreateRequest  # deferred to avoid circular import at module load

    # Re-hydrate Path fields
    for field in _PATH_FIELDS:
        if data.get(field):
            data[field] = Path(data[field])

    # Re-hydrate original_req (Pydantic model)
    if isinstance(data.get("original_req"), dict):
        try:
            data["original_req"] = JobCreateRequest(**data["original_req"])
        except Exception:
            data["original_req"] = None

    # Strip keys unknown to the dataclass
    import dataclasses
    known = {f.name for f in dataclasses.fields(Job)}
    data = {k: v for k, v in data.items() if k in known}

    # cancel_event and events are transient — create fresh instances
    job = Job(**data)
    return job


# ── JobStore ──────────────────────────────────────────────────────────────────

class JobStore:
    """Thread-safe SQLite-backed job store."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # allow concurrent reads
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def load_all(self, jobs_dict: Dict[str, Any]) -> int:
        """
        Populate *jobs_dict* from the database.
        Returns number of jobs loaded.
        Jobs in state 'running' are reset to 'error' (they crashed mid-run).
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, state, payload FROM jobs ORDER BY created_at ASC"
            ).fetchall()

        loaded = 0
        for job_id, state, payload_json in rows:
            try:
                data = json.loads(payload_json)
                # Normalise state: treat null/unknown values as error
                effective_state = data.get("state") or state or "error"
                if effective_state not in ("queued", "running", "done", "error", "waiting_for_srt"):
                    effective_state = "error"
                data["state"] = effective_state
                # Jobs that were 'running' when the server died → mark as error
                if effective_state == "running":
                    data["state"] = "error"
                    data["message"] = "Server restarted while job was running"
                    data["error"] = "Server restarted"
                # Ensure list fields default correctly when missing from old DB rows
                if not isinstance(data.get("chain_languages"), list):
                    data["chain_languages"] = []
                if not isinstance(data.get("segments"), list):
                    data["segments"] = []
                job = _dict_to_job(data)
                jobs_dict[job_id] = job
                loaded += 1
            except Exception as e:
                print(f"[JobStore] Failed to load job {job_id}: {e}", flush=True)

        print(f"[JobStore] Loaded {loaded} jobs from {self.db_path}", flush=True)
        return loaded

    def save(self, job: "Job") -> None:
        """Upsert a job into the database (call after any state change)."""
        try:
            d = _job_to_dict(job)
            payload = json.dumps(d, ensure_ascii=False, default=str)
            now = time.time()
            with self._lock:
                self._conn.execute(
                    """INSERT INTO jobs (id, state, payload, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                           state = excluded.state,
                           payload = excluded.payload,
                           updated_at = excluded.updated_at""",
                    (job.id, job.state, payload, job.created_at, now),
                )
                self._conn.commit()
        except Exception as e:
            print(f"[JobStore] Save error for job {job.id}: {e}", flush=True)

    def delete(self, job_id: str) -> None:
        """Remove a job from the database."""
        try:
            with self._lock:
                self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                self._conn.commit()
        except Exception as e:
            print(f"[JobStore] Delete error for job {job_id}: {e}", flush=True)

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass
