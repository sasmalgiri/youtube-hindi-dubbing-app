"""L04/P09: account deletion removes jobs, files, uploads and personal data;
keeps financial records; refuses while a dub is running; P07 skipped steps;
P08 delivered count; presets that use a switched-off module are marked."""
import json

from tests_dialogue.test_submission import env, media, grant, upload, rights, wait_closed  # noqa: F401


def _done_job(env, c, uid, media):
    grant(env, uid, 3_600, "pay:t1")
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    jid = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k1",
                                                    "rights_accepted": True, "rights_version": rights(c)}).json()["id"]
    assert wait_closed(env, jid) == "settled"
    return jid, did


def test_preview_then_delete_removes_everything_personal(env, media):
    c, uid = env["client"]("a@example.com")
    jid, did = _done_job(env, c, uid, media)
    c.post("/api/drafts", files={"file": ("x.mp4", open(media["ok"], "rb"), "video/mp4")})   # an unused upload
    p = c.get("/api/account/deletion").json()
    assert p["jobs"] == 1 and p["drafts"] == 2 and p["active_jobs"] == [] and p["kept"]
    assert p["unused_seconds"]["standard"] == 3_600 - 91

    assert c.post("/api/account/delete", json={"confirm_email": "wrong@example.com"}).status_code == 400
    assert env["app"].JOBS.get(jid) is not None
    r = c.post("/api/account/delete", json={"confirm_email": "A@Example.com"})
    assert r.status_code == 200 and r.json()["jobs_removed"] == 1 and r.json()["drafts"] == 2

    assert env["app"].JOBS.get(jid) is None
    assert not list((env["tmp"] / "drafts").glob("*.mp4"))
    assert c.get("/api/auth/me").json().get("user") is None          # signed out
    assert c.get(f"/api/jobs/{jid}").status_code in (401, 404)
    log = [json.loads(x) for x in (env["tmp"] / "deletions.jsonl").read_text().splitlines()]
    assert log[-1]["user_id"] == uid and "a@example.com" not in json.dumps(log)
    # the ledger is kept, and the same email can start over as a new account
    from billing.ledger import Ledger
    led = Ledger(env["tmp"] / "ledger.db")
    assert led.balance(uid, "standard").consumed == 91
    led.close()
    c2, uid2 = env["client"]("a@example.com")
    assert uid2 != uid and c2.get("/api/jobs").json() == []


def test_delete_refused_while_a_dub_runs(env, media):
    import threading
    c, uid = env["client"]("a@example.com")
    env["runner"].gate = threading.Event()
    grant(env, uid, 3_600, "pay:t1")
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    jid = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k1",
                                                    "rights_accepted": True, "rights_version": rights(c)}).json()["id"]
    import time
    for _ in range(100):
        if env["app"].JOBS[jid].state == "running":
            break
        time.sleep(0.05)
    try:
        r = c.post("/api/account/delete", json={"confirm_email": "a@example.com"})
        assert r.status_code == 409 and "running" in r.json()["detail"]
    finally:
        env["runner"].gate.set()
    wait_closed(env, jid)


def test_skipped_steps_and_delivered_count(env, media):
    c, uid = env["client"]("a@example.com")
    jid, _ = _done_job(env, c, uid, media)
    sk = c.get(f"/api/jobs/{jid}").json()["skipped_steps"]
    assert sk["download"].startswith("Uploaded file")
    app = env["app"]
    import features
    features.set_state("surface.global_lists", True)   # off by default in hosted
    app.JOBS[jid].source_url = "https://example.com/v"
    app._save_links([{"id": "l1", "url": "https://example.com/v", "title": "", "added_at": 0, "preset": {}},
                     {"id": "l2", "url": "https://example.com/other", "title": "", "added_at": 0, "preset": {}}])
    links = {x["id"]: x for x in c.get("/api/links").json()}
    assert links["l1"]["delivered"] == 1 and links["l2"]["delivered"] == 0
    c2, _ = env["client"]("b@example.com")
    assert {x["id"]: x for x in c2.get("/api/links").json()}["l1"]["delivered"] == 0   # not B's job


def test_presets_using_switched_off_modules_are_marked(env):
    c, _ = env["client"]("a@example.com")
    app = env["app"]
    app._save_preset(app.WORK_ROOT, "Eleven", {"use_elevenlabs": True})
    app._save_preset(app.WORK_ROOT, "Plain", {"asr_model": "groq-whisper"})
    got = {p["name"]: p for p in c.get("/api/presets").json()["presets"]}
    assert "tts.elevenlabs" in (got["Eleven"]["blocked"] or "")      # ElevenLabs is off in hosted
    assert got["Plain"]["blocked"] is None
