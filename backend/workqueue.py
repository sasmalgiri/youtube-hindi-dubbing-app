"""Durable work queue for funded jobs (W3: R01, S03, T08).

Every funded job is a row in a SQLite table, so a queued or running job is
not lost when the server stops:

* queued   -> waits for a free slot (global limit, and per-account limit)
* running  -> holds a lease that a heartbeat renews while the job runs
* finished / failed / cancelled -> terminal

After a restart, ``recover()`` puts interrupted ``running`` rows back in the
queue while they have attempts left, and gives up (``on_give_up``) after the
retry ceiling. A job that ends with an error is *not* retried: an error is a
result; only an interruption is retried.

The queue never touches money. The caller's ``run`` closes billing exactly
once (submission.close_billing is idempotent), and ``on_give_up`` releases
the reservation of a job that will not run again.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS work (
    job_id      TEXT PRIMARY KEY,
    account     TEXT NOT NULL,
    state       TEXT NOT NULL,           -- queued | running | finished | failed | cancelled
    attempts    INTEGER NOT NULL DEFAULT 0,
    lease_until INTEGER NOT NULL DEFAULT 0,
    enqueued_at INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    note        TEXT
);
CREATE INDEX IF NOT EXISTS work_state ON work(state, enqueued_at);
"""


class WorkQueue:
    def __init__(self, db_path: Path, run: Callable[[str], None], *,
                 max_global: int = 1, max_per_account: int = 1, max_attempts: int = 2,
                 lease_seconds: int = 120, on_give_up: Optional[Callable[[str, str], None]] = None,
                 clock: Callable[[], float] = time.time):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self._run = run
        self.max_global = max_global
        self.max_per_account = max_per_account
        self.max_attempts = max_attempts
        self.lease_seconds = lease_seconds
        self._on_give_up = on_give_up
        self._clock = clock
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._threads: Dict[str, threading.Thread] = {}

    # ── storage helpers ──
    def _now(self) -> int:
        return int(self._clock())

    def _x(self, sql: str, args=()):
        with self._lock:
            return self._conn.execute(sql, args)

    def row(self, job_id: str) -> Optional[dict]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM work WHERE job_id=?", (job_id,))
            r = cur.fetchone()
            return dict(zip([c[0] for c in cur.description], r)) if r else None

    # ── API ──
    def enqueue(self, job_id: str, account: str) -> bool:
        """Idempotent: the same job twice is one row. Returns True if new."""
        now = self._now()
        cur = self._x("INSERT OR IGNORE INTO work(job_id, account, state, enqueued_at, updated_at) "
                      "VALUES (?,?, 'queued', ?, ?)", (job_id, account, now, now))
        self._wake.set()
        return cur.rowcount == 1

    def cancel(self, job_id: str) -> bool:
        """Cancel a job that has not started. A running job is cancelled by the pipeline."""
        cur = self._x("UPDATE work SET state='cancelled', updated_at=?, note='cancelled before start' "
                      "WHERE job_id=? AND state='queued'", (self._now(), job_id))
        return cur.rowcount == 1

    def position(self, job_id: str) -> Optional[int]:
        """1-based place in the queue, None if not queued."""
        r = self.row(job_id)
        if not r or r["state"] != "queued":
            return None
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM work WHERE state='queued' AND "
                                   "(enqueued_at < ? OR (enqueued_at = ? AND job_id < ?))",
                                   (r["enqueued_at"], r["enqueued_at"], job_id)).fetchone()[0]
        return n + 1

    def counts(self) -> dict:
        with self._lock:
            return dict(self._conn.execute("SELECT state, COUNT(*) FROM work GROUP BY state").fetchall())

    def recover(self) -> List[str]:
        """After a restart: every 'running' row was interrupted. Requeue it while
        attempts remain, otherwise give up. Returns the requeued job ids."""
        requeued: List[str] = []
        with self._lock:
            rows = self._conn.execute("SELECT job_id, attempts FROM work WHERE state='running'").fetchall()
        for job_id, attempts in rows:
            if attempts < self.max_attempts:
                self._x("UPDATE work SET state='queued', lease_until=0, updated_at=?, note='requeued after restart' "
                        "WHERE job_id=? AND state='running'", (self._now(), job_id))
                requeued.append(job_id)
            else:
                self._give_up(job_id, f"interrupted {attempts} times")
        return requeued

    def queued_ids(self) -> List[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute(
                "SELECT job_id FROM work WHERE state='queued' ORDER BY enqueued_at, job_id").fetchall()]

    def _give_up(self, job_id: str, why: str) -> None:
        self._x("UPDATE work SET state='failed', updated_at=?, note=? WHERE job_id=?", (self._now(), why, job_id))
        if self._on_give_up:
            try:
                self._on_give_up(job_id, why)
            except Exception as e:  # never let a callback kill the dispatcher
                print(f"[QUEUE] give-up callback failed for {job_id}: {e}", flush=True)

    # ── dispatcher ──
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="workqueue", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Tests: wait until nothing is queued or running."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            c = self.counts()
            if not c.get("queued") and not c.get("running"):
                return True
            time.sleep(0.05)
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._dispatch()
            except Exception as e:
                print(f"[QUEUE] dispatch error: {e}", flush=True)
            self._wake.wait(0.5)
            self._wake.clear()

    def _dispatch(self) -> None:
        now = self._now()
        with self._lock:
            # a lease that ran out while this process lives = a dead worker thread
            for job_id, attempts in self._conn.execute(
                    "SELECT job_id, attempts FROM work WHERE state='running' AND lease_until < ?", (now,)).fetchall():
                t = self._threads.get(job_id)
                if t is None or not t.is_alive():
                    if attempts < self.max_attempts:
                        self._conn.execute("UPDATE work SET state='queued', updated_at=?, note='lease expired' "
                                           "WHERE job_id=?", (now, job_id))
                    else:
                        self._give_up(job_id, "lease expired after the last attempt")
            running = self._conn.execute("SELECT account, COUNT(*) FROM work WHERE state='running' "
                                         "GROUP BY account").fetchall()
            per_acct = dict(running)
            total = sum(per_acct.values())
            picks = []
            for job_id, account in self._conn.execute(
                    "SELECT job_id, account FROM work WHERE state='queued' ORDER BY enqueued_at, job_id").fetchall():
                if total >= self.max_global:
                    break
                if per_acct.get(account, 0) >= self.max_per_account:
                    continue
                self._conn.execute("UPDATE work SET state='running', attempts=attempts+1, lease_until=?, "
                                   "updated_at=? WHERE job_id=? AND state='queued'",
                                   (now + self.lease_seconds, now, job_id))
                per_acct[account] = per_acct.get(account, 0) + 1
                total += 1
                picks.append(job_id)
        for job_id in picks:
            t = threading.Thread(target=self._work, args=(job_id,), name=f"job-{job_id}", daemon=True)
            self._threads[job_id] = t
            t.start()

    def _work(self, job_id: str) -> None:
        done = threading.Event()

        def beat():
            while not done.wait(max(1, self.lease_seconds // 3)):
                self._x("UPDATE work SET lease_until=? WHERE job_id=? AND state='running'",
                        (self._now() + self.lease_seconds, job_id))

        hb = threading.Thread(target=beat, daemon=True)
        hb.start()
        note = None
        try:
            self._run(job_id)
        except Exception as e:
            note = f"{type(e).__name__}: {e}"[:300]
        finally:
            done.set()
            self._x("UPDATE work SET state='finished', updated_at=?, note=? WHERE job_id=? AND state='running'",
                    (self._now(), note, job_id))
            self._threads.pop(job_id, None)
            self._wake.set()

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._conn.close()
