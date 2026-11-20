"""W3: durable queue for funded jobs (R01, S03, Q00, T08).

Unit tests drive WorkQueue directly; the app tests reuse the submission
fixture (real ledger, real ffmpeg clips, pipeline replaced by a stand-in)."""
import threading
import time

import pytest

import workqueue
from tests_dialogue.test_submission import (balance, env, grant, media, rights, upload,  # noqa: F401
                                            wait_closed)


def _q(tmp_path, run, **kw):
    q = workqueue.WorkQueue(tmp_path / "q.db", run, **kw)
    return q


def test_enqueue_is_idempotent_and_runs_once(tmp_path):
    ran = []
    q = _q(tmp_path, ran.append)
    assert q.enqueue("j1", "a") is True
    assert q.enqueue("j1", "a") is False
    q.start()
    assert q.wait_idle(5)
    assert ran == ["j1"] and q.row("j1")["state"] == "finished"
    q.close()


def test_global_and_per_account_limits(tmp_path):
    gate = threading.Event()
    live, peak, lock = {}, {"all": 0, "a": 0}, threading.Lock()

    def run(job_id):
        acct = job_id[0]
        with lock:
            live[job_id] = acct
            peak["all"] = max(peak["all"], len(live))
            peak["a"] = max(peak["a"], sum(1 for v in live.values() if v == "a"))
        gate.wait(5)
        with lock:
            live.pop(job_id)

    q = _q(tmp_path, run, max_global=2, max_per_account=1)
    for j in ("a1", "a2", "a3", "b1", "c1"):
        q.enqueue(j, j[0])
    q.start()
    time.sleep(1.2)
    assert q.counts().get("running") == 2
    assert q.position("a2") is not None
    gate.set()
    assert q.wait_idle(10)
    assert peak == {"all": 2, "a": 1}
    q.close()


def test_restart_requeues_interrupted_then_gives_up_at_the_ceiling(tmp_path):
    given_up = []
    q = _q(tmp_path, lambda j: None, max_attempts=2, on_give_up=lambda j, why: given_up.append((j, why)))
    q.enqueue("j1", "a")
    # simulate: picked up, then the process died (no finish)
    q._x("UPDATE work SET state='running', attempts=1 WHERE job_id='j1'")
    q.close()
    q2 = _q(tmp_path, lambda j: None, max_attempts=2, on_give_up=lambda j, why: given_up.append((j, why)))
    assert q2.recover() == ["j1"] and q2.row("j1")["state"] == "queued"
    q2._x("UPDATE work SET state='running', attempts=2 WHERE job_id='j1'")
    assert q2.recover() == []
    assert q2.row("j1")["state"] == "failed" and given_up and given_up[0][0] == "j1"
    q2.close()


def test_cancel_only_before_start(tmp_path):
    q = _q(tmp_path, lambda j: None)
    q.enqueue("j1", "a")
    assert q.cancel("j1") is True and q.cancel("j1") is False
    assert q.row("j1")["state"] == "cancelled"
    q.close()


def test_a_crashing_run_is_finished_not_retried(tmp_path):
    calls = []

    def run(j):
        calls.append(j)
        raise RuntimeError("boom")

    q = _q(tmp_path, run)
    q.enqueue("j1", "a")
    q.start()
    assert q.wait_idle(5)
    r = q.row("j1")
    assert calls == ["j1"] and r["state"] == "finished" and "boom" in r["note"]
    q.close()


# ── through the app ──

def _submit(c, did, key="k1"):
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    return c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": key,
                                                     "rights_accepted": True, "rights_version": rights(c)})


def test_funded_job_writes_a_manifest_and_settles(env, media):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    did = upload(c, media["ok"]).json()["id"]
    q = c.post("/api/quotes", json={"draft_id": did}).json()
    assert "queue" in q and q["queue"]["workers"] >= 1
    jid = _submit(c, did).json()["id"]
    assert wait_closed(env, jid) == "settled"
    m = c.get(f"/api/jobs/{jid}/manifest").json()
    assert m["manifest_version"] == 1 and m["source"]["sha256"] and m["targets"] == ["hi"]
    assert m["consent"]["rights_version"] and m["quote"]["standard_seconds"] == 91
    # another account cannot read it
    c2, _ = env["client"]("b@example.com")
    assert c2.get(f"/api/jobs/{jid}/manifest").status_code == 404


def test_cancel_a_queued_job_returns_its_time(env, media):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    env["runner"].gate = threading.Event()        # first job holds the only slots
    env["queue"].max_global = 1
    did = upload(c, media["ok"]).json()["id"]
    j1 = _submit(c, did, "k1").json()["id"]
    did2 = upload(c, media["ok"]).json()["id"]
    r2 = _submit(c, did2, "k2").json()
    assert r2["queue_position"] == 1 or c.get(f"/api/jobs/{r2['id']}/queue").json()["state"] == "queued"
    assert c.delete(f"/api/jobs/{r2['id']}").json()["status"] == "cancelled"
    assert wait_closed(env, r2["id"]) == "released"
    env["runner"].gate.set()
    assert wait_closed(env, j1) == "settled"
    assert balance(env, uid).spendable == 3_600 - 91


def test_restart_resumes_a_queued_funded_job(env, media):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    app = env["app"]
    env["queue"].stop()                           # the server "stops" before the job is picked up
    did = upload(c, media["ok"]).json()["id"]
    jid = _submit(c, did).json()["id"]
    assert app.JOBS[jid].state == "queued"
    # restart: jobs reload from the store (which marks unfinished jobs as error) …
    app.JOBS.clear()
    app._store.load_all(app.JOBS)
    app._settle_loaded_jobs(app.JOBS, app._store)
    app._recover_queue(app.JOBS)
    app._reconcile_billing(app.JOBS)
    assert app.JOBS[jid].state == "queued"           # … but the paid job is back in the queue, not released
    assert balance(env, uid).reserved == 91
    env["queue"].start()
    assert wait_closed(env, jid) == "settled"
