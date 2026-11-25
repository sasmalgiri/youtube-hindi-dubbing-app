"""Hosted hardening from the route audit: no unbilled create routes, no free
paid-TTS previews, bounded re-voices, no cross-account eviction, and unknown
job ids always 404."""
from tests_dialogue.test_submission import env, media, grant, upload, rights, wait_closed  # noqa: F401


def test_legacy_upload_routes_refused_before_body(env, media):
    c, _ = env["client"]("a@example.com")
    for path in ("/api/jobs/upload", "/api/jobs/with-srt"):
        r = c.post(path, files={"file": ("x.mp4", open(media["ok"], "rb"), "video/mp4")})
        assert r.status_code == 404, path
    assert env["app"].JOBS == {}


def test_voice_preview_needs_sign_in_and_ignores_text(env, monkeypatch):
    from fastapi.testclient import TestClient
    anon = TestClient(env["app"].app, base_url="https://testserver")
    assert anon.post("/api/dialogue/voice-preview", json={"provider": "edge", "voice": "x", "text": "free"}).status_code == 401


def test_unknown_job_id_is_404(env):
    c, _ = env["client"]("a@example.com")
    for p in ("/api/jobs/deadbeef0000/manifest", "/api/jobs/deadbeef0000/queue", "/api/jobs/deadbeef0000"):
        assert c.get(p).status_code == 404, p


def test_revoice_is_capped_per_dub(env, media, monkeypatch):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:r1")
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    jid = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k1",
                                                    "rights_accepted": True, "rights_version": rights(c)}).json()["id"]
    assert wait_closed(env, jid) == "settled"
    app = env["app"]
    monkeypatch.setattr(app, "MAX_REVOICES", 2)
    job = app.JOBS[jid]
    job.billing = {**job.billing, "revoices": 2}
    r = c.post(f"/api/jobs/{jid}/dialogue/revoice", json={})
    assert r.status_code == 429 and "re-voiced 2 times" in r.json()["detail"]


def test_hosted_never_evicts_by_count(env, monkeypatch):
    app = env["app"]
    monkeypatch.setattr(app, "MAX_JOBS", 0)
    j = app.Job(id="old1", source_url="x", target_language="hi")
    j.state, j.owner_id = "done", "someone"
    app.JOBS["old1"] = j
    app._cleanup_old_jobs()
    assert "old1" in app.JOBS
