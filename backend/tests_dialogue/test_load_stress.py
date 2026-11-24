"""Load + soak test of the paid path (tracker T08, R01): many accounts submit
at once, dubs succeed/fail/produce bad output at random, the server restarts
on a timer; afterwards no job is lost, nothing runs twice and every second
of prepaid time is accounted for.

Default: a short load run inside the normal suite (~30-60 s).
Soak:    VOICEDUB_STRESS_SECONDS=86400 python -m pytest tests_dialogue/test_load_stress.py -q
         (restarts every VOICEDUB_STRESS_RESTART_S, default 20 s)
The dubbing itself is a fake runner (real files, real ffmpeg checks); no provider is called.
"""
import os
import random
import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

from tests_dialogue.test_submission import env, media, grant, upload, rights  # noqa: F401

ACCOUNTS = int(os.environ.get("VOICEDUB_STRESS_ACCOUNTS", "12"))
JOBS_PER_ROUND = 2
SOAK_S = float(os.environ.get("VOICEDUB_STRESS_SECONDS", "0"))
RESTART_S = float(os.environ.get("VOICEDUB_STRESS_RESTART_S", "20"))


class ChaosRunner:
    """Per job: 75% good output, 10% pipeline error, 10% silent output, 5% garbage.
    Seeded by job id so a re-run of the same job behaves the same."""

    def __init__(self):
        self.calls = Counter()
        self.lock = threading.Lock()

    def __call__(self, job, req):
        with self.lock:
            self.calls[job.id] += 1
        rnd = random.Random(job.id)
        time.sleep(rnd.uniform(0.02, 0.25))
        roll = rnd.random()
        out = Path(req.url).parent / "dubbed.mp4"
        if roll < 0.75:
            shutil.copy2(req.url, out)
            job.result_path, job.state = out, "done"
        elif roll < 0.85:
            job.state, job.error = "error", "simulated pipeline failure"
        elif roll < 0.95:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                            "-t", "5", "-c:a", "aac", str(out)], check=True)
            job.result_path, job.state = out, "done"
        else:
            out.write_bytes(b"x" * 2000)
            job.result_path, job.state = out, "done"


class Serving:
    """A real restart stops serving requests before it reloads; requests wait."""

    def __init__(self):
        self.cv = threading.Condition()
        self.open, self.inflight = True, 0

    def enter(self):
        with self.cv:
            self.cv.wait_for(lambda: self.open)
            self.inflight += 1

    def leave(self):
        with self.cv:
            self.inflight -= 1
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.open = False
            self.cv.wait_for(lambda: self.inflight == 0)

    def reopen(self):
        with self.cv:
            self.open = True
            self.cv.notify_all()


SERVING = Serving()


def _restart(env):
    """Stop serving, let running jobs finish, reload everything from disk."""
    SERVING.close()
    try:
        _reload(env)
    finally:
        SERVING.reopen()


def _reload(env):
    app = env["app"]
    q = env["queue"]
    q.stop()
    q.wait_idle(60)
    app.JOBS.clear()
    app._store.load_all(app.JOBS)
    app._settle_loaded_jobs(app.JOBS, app._store)
    app._recover_queue(app.JOBS)
    app._reconcile_billing(app.JOBS)
    q.start()


def test_many_accounts_with_restarts_lose_nothing(env, media, monkeypatch):
    app = env["app"]
    runner = ChaosRunner()
    monkeypatch.setattr(app, "_job_runner", lambda: runner)
    GRANT = 200_000
    users = []
    for i in range(ACCOUNTS):
        c, uid = env["client"](f"user{i}@example.com")
        grant(env, uid, GRANT, f"pay:t{i}")
        did = upload(c, media["ok"]).json()["id"]
        users.append((c, uid, did))
    v = rights(users[0][0])

    submitted, errors, lock = [], [], threading.Lock()
    stop_at = time.time() + SOAK_S

    def account_worker(c, uid, did, n0):
        n = n0
        while True:
            for _ in range(JOBS_PER_ROUND):
                n += 1
                SERVING.enter()
                try:
                    q = c.post("/api/quotes", json={"draft_id": did})
                    if q.status_code != 200:
                        with lock:
                            errors.append(f"quote {q.status_code} {q.text[:80]}")
                        continue
                    r = c.post(f"/api/drafts/{did}/submit", json={"quote_id": q.json()["id"], "idempotency_key": f"{uid}-{n}",
                                                                  "rights_accepted": True, "rights_version": v})
                    if r.status_code == 200:
                        with lock:
                            submitted.append((uid, r.json()["id"]))
                    else:
                        with lock:
                            errors.append(f"submit {r.status_code} {r.text[:80]}")
                except Exception as e:
                    with lock:
                        errors.append(repr(e)[:120])
                finally:
                    SERVING.leave()
            if time.time() >= stop_at:
                return
            time.sleep(random.uniform(0.0, 0.3))

    threads = [threading.Thread(target=account_worker, args=(c, uid, did, 0)) for c, uid, did in users]
    for t in threads:
        t.start()
    restarts = 0
    while any(t.is_alive() for t in threads):
        time.sleep(min(RESTART_S, 3.0) if SOAK_S else 0.5)
        if SOAK_S and time.time() < stop_at:
            _restart(env)
            restarts += 1
    for t in threads:
        t.join()
    _restart(env)          # one more restart with work still queued
    restarts += 1

    deadline = time.time() + max(120, len(submitted) * 0.5)
    while time.time() < deadline:
        open_ = [j for _, j in submitted if not (app.JOBS.get(j) and (app.JOBS[j].billing or {}).get("closed"))]
        if not open_:
            break
        time.sleep(0.25)

    assert not errors, errors[:5]
    assert len(submitted) == ACCOUNTS * JOBS_PER_ROUND or SOAK_S
    # no job lost, every reservation closed exactly once
    closed = Counter()
    for uid, j in submitted:
        job = app.JOBS.get(j)
        assert job is not None, f"job {j} lost"
        assert (job.billing or {}).get("closed") in ("settled", "released"), (j, job.state, job.billing)
        closed[job.billing["closed"]] += 1
        assert (job.billing["closed"] == "settled") == (job.state == "done"), (j, job.state, job.billing["closed"])
    # nothing ran twice
    twice = {j: n for j, n in runner.calls.items() if n > 1}
    assert not twice, twice
    # every second accounted for
    from billing.ledger import Ledger
    led = Ledger(env["tmp"] / "ledger.db")
    try:
        for c, uid, _ in users:
            b = led.balance(uid, "standard")
            mine = [j for u, j in submitted if u == uid]
            settled = sum(app.JOBS[j].billing["seconds"] for j in mine if app.JOBS[j].billing["closed"] == "settled")
            assert b.reserved == 0, (uid, b)
            assert b.consumed == settled, (uid, b.consumed, settled)
            assert b.spendable == GRANT - settled, (uid, b)
            assert led.invariants(uid) == []
    finally:
        led.close()
    from billing import reconcile
    assert sum(len(x) for x in reconcile.run(env["tmp"], active_jobs=set()).values()) == 0
    print(f"\n[stress] accounts={ACCOUNTS} jobs={len(submitted)} settled={closed['settled']} "
          f"released={closed['released']} restarts={restarts} soak_s={SOAK_S}")
