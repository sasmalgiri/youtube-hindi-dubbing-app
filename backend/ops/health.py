"""Owner view of the running service, and the alerts derived from it.

    python -m ops.health            # print the snapshot and any alerts

snapshot() reads only the state databases and the disk; nothing here changes
data. alerts() turns a snapshot into problems worth a message. AlertSink
records each alert in STATE_DIR/alerts.jsonl and emails VOICEDUB_ALERT_EMAIL
when SMTP is configured, sending the same alert again at most every
REPEAT_S seconds.
"""
from __future__ import annotations

import json
import os
import shutil
import smtplib
import sqlite3
import threading
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Dict, List, Optional

# thresholds (env overrides so the owner can tune without code)
DISK_MIN_GB = float(os.environ.get("VOICEDUB_ALERT_DISK_GB", "20"))
FAILS_24H = int(os.environ.get("VOICEDUB_ALERT_FAILS_24H", "5"))
QUEUE_STUCK_S = int(os.environ.get("VOICEDUB_ALERT_QUEUE_STUCK_S", "1800"))
BACKUP_MAX_AGE_H = float(os.environ.get("VOICEDUB_ALERT_BACKUP_H", "30"))
REPEAT_S = 6 * 3600


def _q(db: Path, sql: str, args=()) -> List[tuple]:
    if not db.exists():
        return []
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    try:
        return c.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        c.close()


def _last_backup(dest: Path) -> Optional[int]:
    best = None
    for m in dest.glob("backup_*/backup.json") if dest.exists() else []:
        try:
            t = int(json.loads(m.read_text(encoding="utf-8")).get("created_at") or 0)
        except (ValueError, OSError):
            continue
        best = max(best or 0, t)
    return best


def snapshot(state: Path, backups: Optional[Path] = None, now: Optional[float] = None,
             jobs: Optional[List[Dict]] = None) -> Dict:
    """jobs: in-memory job summaries [{id, state, created_at, error, owner_id}] (the app passes them)."""
    state = Path(state)
    now = int(now or time.time())
    backups = Path(backups) if backups else Path(os.environ.get("VOICEDUB_BACKUP_DIR") or state / "backups")
    queue = {s: n for s, n in _q(state / "queue.db", "SELECT state, COUNT(*) FROM work GROUP BY state")}
    oldest = _q(state / "queue.db", "SELECT MIN(enqueued_at) FROM work WHERE state='queued'")
    running_stale = _q(state / "queue.db",
                       "SELECT COUNT(*) FROM work WHERE state='running' AND lease_until>0 AND lease_until<?", (now,))
    orders = {s: (n, a) for s, n, a in _q(state / "payments.db",
                                          "SELECT status, COUNT(*), COALESCE(SUM(amount_minor),0) FROM orders "
                                          "WHERE created_at>=? GROUP BY status", (now - 30 * 86400,))}
    by_cur = [{"currency": c, "paid": n, "amount_minor": a} for c, n, a in _q(
        state / "payments.db", "SELECT currency, COUNT(*), SUM(amount_minor) FROM orders "
                               "WHERE status='paid' AND created_at>=? GROUP BY currency", (now - 30 * 86400,))]
    jobs = jobs or []
    day = [j for j in jobs if (j.get("created_at") or 0) >= now - 86400]
    failed = [j for j in day if j.get("state") == "error"]
    du = shutil.disk_usage(state if state.exists() else Path("."))
    last_bk = _last_backup(backups)
    return {
        "at": now,
        "queue": {"by_state": queue, "oldest_queued_s": (now - oldest[0][0]) if oldest and oldest[0][0] else 0,
                  "expired_leases": running_stale[0][0] if running_stale else 0},
        "jobs_24h": {"total": len(day), "failed": len(failed), "done": sum(1 for j in day if j.get("state") == "done"),
                     "recent_failures": [{"id": j["id"], "error": (j.get("error") or "")[:200]} for j in failed[-10:]]},
        "orders_30d": {s: {"count": n, "amount_minor": a} for s, (n, a) in orders.items()},
        "paid_by_currency_30d": by_cur,
        "disk": {"free_gb": round(du.free / 2**30, 1), "total_gb": round(du.total / 2**30, 1)},
        "backup": {"dir": str(backups), "last_at": last_bk, "age_h": round((now - last_bk) / 3600, 1) if last_bk else None},
    }


def alerts(snap: Dict, reconcile_problems: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
    out = []
    def add(key, level, text):
        out.append({"key": key, "level": level, "text": text})
    if snap["disk"]["free_gb"] < DISK_MIN_GB:
        add("disk", "critical", f"Disk nearly full: {snap['disk']['free_gb']} GB free (alert below {DISK_MIN_GB:g} GB)")
    if snap["jobs_24h"]["failed"] >= FAILS_24H:
        add("failures", "warning", f"{snap['jobs_24h']['failed']} dubs failed in the last 24 h")
    if snap["queue"]["oldest_queued_s"] > QUEUE_STUCK_S:
        add("queue", "critical", f"Queue stuck: oldest job waiting {snap['queue']['oldest_queued_s'] // 60} min")
    if snap["queue"]["expired_leases"]:
        add("leases", "warning", f"{snap['queue']['expired_leases']} running job(s) past their lease (worker died?)")
    age = snap["backup"]["age_h"]
    if age is None:
        add("backup", "critical", "No backup found in " + snap["backup"]["dir"])
    elif age > BACKUP_MAX_AGE_H:
        add("backup", "critical", f"Last backup is {age} h old")
    disputed = snap["orders_30d"].get("disputed", {}).get("count", 0)
    if disputed:
        add("disputes", "warning", f"{disputed} payment dispute(s) in the last 30 days")
    n = sum(len(v) for v in (reconcile_problems or {}).values())
    if n:
        add("reconcile", "critical", f"Billing reconciliation found {n} problem(s)")
    return out


def _email(to: str, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, os.environ["SMTP_FROM"], to
    msg.set_content(body)
    port = int(os.environ.get("SMTP_PORT") or 587)
    with smtplib.SMTP(os.environ["SMTP_HOST"], port, timeout=20) as s:
        if port != 25:
            s.starttls()
        if os.environ.get("SMTP_USER"):
            s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASS", ""))
        s.send_message(msg)


class AlertSink:
    def __init__(self, state: Path, clock: Callable[[], float] = time.time,
                 send: Optional[Callable[[str, str, str], None]] = None):
        self.path = Path(state) / "alerts.jsonl"
        self.clock = clock
        self.to = (os.environ.get("VOICEDUB_ALERT_EMAIL") or "").strip()
        if send is None and self.to and os.environ.get("SMTP_HOST") and os.environ.get("SMTP_FROM"):
            send = _email
        self.send = send
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def emit(self, items: List[Dict]) -> List[Dict]:
        """Record new alerts; returns the ones actually raised (not repeats)."""
        now, raised = self.clock(), []
        with self._lock:
            for a in items:
                k = a["key"] + ":" + a["text"].split(":")[0]
                if now - self._last.get(k, -1e18) < REPEAT_S:
                    continue
                self._last[k] = now
                rec = {**a, "at": int(now), "emailed": False}
                if self.send and self.to:
                    try:
                        self.send(self.to, f"[VoiceDub {a['level']}] {a['text']}", a["text"])
                        rec["emailed"] = True
                    except Exception as e:
                        rec["email_error"] = str(e)[:200]
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[ALERT] {a['level']}: {a['text']}", flush=True)
                raised.append(rec)
        return raised

    def recent(self, n: int = 50) -> List[Dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(x) for x in reversed(lines) if x.strip()]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Service health snapshot and alerts")
    ap.add_argument("--state", default=os.environ.get("VOICEDUB_STATE_DIR") or "state")
    a = ap.parse_args(argv)
    snap = snapshot(Path(a.state))
    print(json.dumps({"snapshot": snap, "alerts": alerts(snap)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
