"""W7: expiring download links and progress events that survive a restart."""
import time

from tests_dialogue.test_submission import env, grant, media, rights, upload, wait_closed  # noqa: F401


def _done_job(env, media, email="a@example.com"):
    c, uid = env["client"](email)
    grant(env, uid, 3_600, f"pay:{email}")
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    jid = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k1",
                                                    "rights_accepted": True, "rights_version": rights(c)}).json()["id"]
    assert wait_closed(env, jid) == "settled"
    return c, jid


def test_link_downloads_without_cookie_and_expires(env, media, monkeypatch):
    c, jid = _done_job(env, media)
    links = c.get(f"/api/jobs/{jid}/links").json()
    from fastapi.testclient import TestClient
    anon = TestClient(env["app"].app, base_url="https://testserver")
    r = anon.get(links["video"])
    assert r.status_code == 200 and len(r.content) > 1000
    # tampered token
    assert anon.get(links["video"][:-2] + "00").status_code == 404
    assert anon.get(links["video"].replace(".video.", ".srt.")).status_code == 404
    # expiry
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 10_000)
    assert anon.get(links["video"]).status_code == 410


def test_links_are_owner_only(env, media):
    c, jid = _done_job(env, media)
    c2, _ = env["client"]("b@example.com")
    assert c2.get(f"/api/jobs/{jid}/links").status_code == 404


def test_events_survive_a_store_round_trip(env):
    app = env["app"]
    job = app.Job(id="evtjob", source_url="x", target_language="hi")
    job.events = [{"type": "progress", "step": "transcribe", "overall": 0.3}] * 250
    app._store.save(job)
    jobs = {}
    app._store.load_all(jobs)
    ev = jobs["evtjob"].events
    assert len(ev) == 200 and ev[-1]["step"] == "transcribe"
