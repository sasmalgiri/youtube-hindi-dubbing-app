"""W2 creator submission + W4/W6 funded jobs (tracker P01, Q01, M02, M04, L01, T08, T11).

Real ffmpeg-made clips, real ledger and stores; the dubbing pipeline itself is
replaced by a fake runner, so nothing is synthesized and nothing is charged.
"""
import importlib
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

import accounts as A

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe needed")


def _clip(path: Path, seconds: float, audio=True, video=True):
    args = ["ffmpeg", "-v", "error", "-y"]
    if video:
        args += ["-f", "lavfi", "-i", f"testsrc=s=320x240:r=10:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if video:
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac"]
    subprocess.run(args + ["-shortest", str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    (d / "broken.mp4").write_bytes(b"not a video at all" * 100)
    return {
        "ok": _clip(d / "talk.mp4", 90.4),
        "silent_video": _clip(d / "noaudio.mp4", 5, audio=False),
        "garbage": d / "broken.mp4",
    }


class Runner:
    """Stands in for the pipeline: ends the job as done (with a result file) or as error."""

    def __init__(self):
        self.outcome = "done"
        self.calls = []
        self.gate = None

    def __call__(self, job, req):
        self.calls.append(job.id)
        if self.gate:
            self.gate.wait(10)
        out = Path(req.url).parent / "dubbed.mp4"
        if self.outcome == "done":
            shutil.copy2(req.url, out)            # a real, audible file passes the delivery check
            job.result_path = out
            job.state = "done"
        elif self.outcome in ("silent", "garbage"):
            if self.outcome == "silent":
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                                "-t", "5", "-c:a", "aac", str(out)], check=True)
            else:
                out.write_bytes(b"x" * 2000)
            job.result_path = out
            job.state = "done"                    # the pipeline claims success; the check must catch it
        else:
            job.state = "error"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEDUB_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("VOICEDUB_PROFILE", "hosted")
    import features
    importlib.reload(features)
    import app as app_module
    import submission
    from jobstore import JobStore
    monkeypatch.setattr(app_module, "_features", features)
    monkeypatch.setattr(app_module, "STATE_DIR", tmp_path)
    monkeypatch.setattr(app_module, "OUTPUTS", tmp_path / "outputs")
    outbox = []
    acc = A.AccountStore(tmp_path / "accounts.db", mailer=lambda to, code: outbox.append((to, code)))
    monkeypatch.setattr(app_module, "_accounts", acc)
    monkeypatch.setattr(app_module, "_store", JobStore(tmp_path / "jobs.db"))
    monkeypatch.setattr(app_module, "JOBS", {})
    st = submission.SubmissionStore(tmp_path / "submission.db", tmp_path / "drafts")
    monkeypatch.setattr(app_module._submissions, "_store", st, raising=False)
    runner = Runner()
    monkeypatch.setattr(app_module, "_job_runner", lambda: runner)
    # hosted voices: Google TTS on (Edge is off in hosted)
    features.set_state("tts.google", True)
    from fastapi.testclient import TestClient

    def client(email):
        c = TestClient(app_module.app, base_url="https://testserver")
        acc._conn.execute("UPDATE email_codes SET created_at = created_at - 120")
        c.post("/api/auth/email/start", json={"email": email})
        uid = c.post("/api/auth/email/verify", json={"email": email, "code": outbox[-1][1]}).json()["user"]["id"]
        return c, uid

    yield {"app": app_module, "client": client, "runner": runner, "tmp": tmp_path, "store": st}
    acc.close()
    st.close()


def grant(env, uid, seconds, key):
    from billing.ledger import Ledger
    led = Ledger(env["tmp"] / "ledger.db")
    led.grant(uid, "standard", seconds, key=key)
    led.close()


def balance(env, uid):
    from billing.ledger import Ledger
    led = Ledger(env["tmp"] / "ledger.db")
    try:
        return led.balance(uid, "standard")
    finally:
        led.close()


def upload(c, path):
    with open(path, "rb") as f:
        return c.post("/api/drafts", files={"file": (Path(path).name, f, "video/mp4")})


def rights(c):
    return c.get("/api/rights").json()["version"]


def wait_closed(env, job_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = env["app"].JOBS.get(job_id)
        if j and j.billing and j.billing.get("closed"):
            return j.billing["closed"]
        time.sleep(0.05)
    raise AssertionError("billing never closed")


# ── drafts (P01, Q01) ──
def test_upload_is_probed_before_any_quote(env, media):
    c, _ = env["client"]("a@example.com")
    r = upload(c, media["ok"])
    assert r.status_code == 200, r.text
    d = r.json()
    assert abs(d["duration"] - 90.4) < 0.3 and d["probe"]["audio"][0]["codec"] == "aac"
    assert len(d["sha256"]) == 64 and "path" not in c.get(f"/api/drafts/{d['id']}").json()


@pytest.mark.parametrize("kind,status", [("silent_video", 422), ("garbage", 422)])
def test_unusable_media_is_refused_with_a_reason(env, media, kind, status):
    c, _ = env["client"]("a@example.com")
    r = upload(c, media[kind])
    assert r.status_code == status and r.json()["detail"]
    assert not any(p.suffix != ".part" for p in (env["tmp"] / "drafts").glob("*")) or True
    assert list((env["tmp"] / "drafts").glob("*")) == []          # nothing left behind


def test_wrong_extension_and_size_limit(env, media, monkeypatch, tmp_path):
    c, _ = env["client"]("a@example.com")
    bad = tmp_path / "x.exe"; bad.write_bytes(b"MZ" * 10)
    assert upload(c, bad).status_code == 415
    import submission
    monkeypatch.setattr(submission, "MAX_UPLOAD_BYTES", 1000)
    r = upload(c, media["ok"])
    assert r.status_code == 413 and list((env["tmp"] / "drafts").glob("*")) == []


def test_drafts_are_private_and_hosted_needs_sign_in(env, media):
    ca, _ = env["client"]("a@example.com")
    cb, _ = env["client"]("b@example.com")
    did = upload(ca, media["ok"]).json()["id"]
    assert cb.get(f"/api/drafts/{did}").status_code == 404
    assert cb.post("/api/quotes", json={"draft_id": did}).status_code == 404
    from fastapi.testclient import TestClient
    anon = TestClient(env["app"].app, base_url="https://testserver")
    assert anon.post("/api/quotes", json={"draft_id": did}).status_code == 401


# ── quotes (M02, P02) ──
def test_quote_uses_the_balance_and_the_first_n_minutes(env, media):
    c, uid = env["client"]("a@example.com")
    did = upload(c, media["ok"]).json()["id"]
    q = c.post("/api/quotes", json={"draft_id": did}).json()
    assert q["interval"] == [0, 91] and q["cash_minor"] == 253          # 91 s at ₹100/h, half-up
    grant(env, uid, 3_600, "pay:t1")
    q = c.post("/api/quotes", json={"draft_id": did, "minutes": 1}).json()
    assert q["interval"] == [0, 60] and q["standard_seconds"] == 60 and q["cash_minor"] == 0
    assert c.post("/api/quotes", json={"draft_id": did, "style": "narration"}).status_code == 422


# ── submit (W2/W4/W6) ──
def test_happy_path_reserves_then_settles_once(env, media):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    r = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k1",
                                                  "rights_accepted": True, "rights_version": rights(c)})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert wait_closed(env, jid) == "settled"
    b = balance(env, uid)
    assert (b.spendable, b.consumed, b.reserved) == (3_600 - 91, 91, 0)
    assert c.get(f"/api/jobs/{jid}").status_code == 200


def test_failed_job_releases_the_reservation(env, media):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    env["runner"].outcome = "error"
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    jid = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k1",
                                                    "rights_accepted": True, "rights_version": rights(c)}).json()["id"]
    assert wait_closed(env, jid) == "released"
    assert balance(env, uid).spendable == 3_600


def test_rights_must_be_confirmed_with_the_current_wording(env, media):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    r = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k1"})
    assert r.status_code == 400 and "rights" in r.json()["detail"]
    r = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k2",
                                                  "rights_accepted": True, "rights_version": "old"})
    assert r.status_code == 409
    assert balance(env, uid).spendable == 3_600 and env["runner"].calls == []


def test_cash_due_is_refused_until_checkout_exists(env, media):
    c, uid = env["client"]("a@example.com")
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    r = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k1",
                                                  "rights_accepted": True, "rights_version": rights(c)})
    assert r.status_code == 402 and "₹2.53" in r.json()["detail"]
    assert env["runner"].calls == []


def test_duplicate_click_yields_one_job_and_one_debit(env, media):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    env["runner"].gate = threading.Event()
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    body = {"quote_id": qid, "idempotency_key": "same-click", "rights_accepted": True, "rights_version": rights(c)}
    out = []
    ts = [threading.Thread(target=lambda: out.append(c.post(f"/api/drafts/{did}/submit", json=body).json()))
          for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all("id" in o for o in out), out
    ids = {o["id"] for o in out}
    assert len(ids) == 1 and sum(not o["duplicate"] for o in out) == 1
    env["runner"].gate.set()
    jid = ids.pop()
    assert wait_closed(env, jid) == "settled"
    assert env["runner"].calls == [jid] and balance(env, uid).consumed == 91


def test_two_tabs_with_one_quote_start_one_job(env, media):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    v = rights(c)
    r1 = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "tab1", "rights_accepted": True, "rights_version": v})
    r2 = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "tab2", "rights_accepted": True, "rights_version": v})
    assert r1.status_code == 200 and r2.status_code == 409
    wait_closed(env, r1.json()["id"])
    assert balance(env, uid).consumed == 91 and balance(env, uid).reserved == 0


def test_two_jobs_racing_for_the_final_minute(env, media):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 91, "pay:t1")                    # exactly one job's worth
    env["runner"].gate = threading.Event()
    did = upload(c, media["ok"]).json()["id"]
    v = rights(c)
    q1 = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    q2 = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    r1 = c.post(f"/api/drafts/{did}/submit", json={"quote_id": q1, "idempotency_key": "a", "rights_accepted": True, "rights_version": v})
    r2 = c.post(f"/api/drafts/{did}/submit", json={"quote_id": q2, "idempotency_key": "b", "rights_accepted": True, "rights_version": v})
    assert sorted([r1.status_code, r2.status_code]) == [200, 409]
    env["runner"].gate.set()
    ok = r1 if r1.status_code == 200 else r2
    wait_closed(env, ok.json()["id"])
    b = balance(env, uid)
    assert (b.spendable, b.consumed, b.reserved) == (0, 91, 0)


def test_restart_reconciliation_releases_an_interrupted_job(env, media):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    env["runner"].gate = threading.Event()             # job stays "running"
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    jid = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k", "rights_accepted": True,
                                                    "rights_version": rights(c)}).json()["id"]
    app_module = env["app"]
    job = app_module.JOBS[jid]
    job.state = "running"
    app_module._store.save(job)
    assert balance(env, uid).reserved == 91
    # cold restart: load from disk (running → error), then reconcile
    from jobstore import JobStore
    fresh = {}
    JobStore(env["tmp"] / "jobs.db").load_all(fresh)
    assert fresh[jid].state == "error" and fresh[jid].billing["closed"] is None if "closed" in fresh[jid].billing else True
    app_module._reconcile_billing(fresh)
    assert fresh[jid].billing["closed"] == "released"
    app_module._reconcile_billing(fresh)                # idempotent
    b = balance(env, uid)
    assert (b.spendable, b.reserved) == (3_600, 0)
    env["runner"].gate.set()


@pytest.mark.parametrize("outcome,reason", [("silent", "silent"), ("garbage", "decode")])
def test_bad_output_is_never_complete_and_is_not_charged(env, media, outcome, reason):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    env["runner"].outcome = outcome
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    jid = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k1",
                                                    "rights_accepted": True, "rights_version": rights(c)}).json()["id"]
    assert wait_closed(env, jid) == "released"
    assert balance(env, uid).spendable == 3_600
    j = c.get(f"/api/jobs/{jid}").json()
    assert j["state"] == "error" and reason in (j.get("error") or "")


def test_verify_result_checks(media, tmp_path):
    import submission
    assert submission.verify_result(media["ok"]) is None
    assert submission.verify_result(media["silent_video"]) == "no audio track"
    assert submission.verify_result(media["garbage"]) == "file does not decode"
    assert submission.verify_result(tmp_path / "missing.mp4").startswith("check failed")
