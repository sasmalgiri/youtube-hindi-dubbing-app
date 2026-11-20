"""Module on/off matrix (backend/features.py): tracker X01, A06, S02, S03, L08, R07."""
import importlib
import json

import pytest


@pytest.fixture
def feat(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEDUB_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("VOICEDUB_PROFILE", "local")
    import features
    importlib.reload(features)
    return features


def test_every_module_id_is_unique_and_grouped(feat):
    groups = {g for g, _ in feat.GROUPS}
    assert len(feat.MODULES) == len(feat._M)
    assert all(m.group in groups for m in feat._M)


def test_local_profile_keeps_todays_behaviour(feat):
    """Nothing the owner uses today is switched off by default locally."""
    assert all(feat.is_on(m.id) for m in feat._M)


def test_hosted_is_deny_by_default_and_locks_cannot_be_overridden(feat, monkeypatch):
    monkeypatch.setenv("VOICEDUB_PROFILE", "hosted")
    assert not feat.is_on("input.url")            # P01/L01
    assert not feat.is_on("tts.edge")             # L06
    assert not feat.is_on("surface.cookies")      # S02
    assert feat.is_on("input.upload")
    with pytest.raises(PermissionError):
        feat.set_state("integration.trainer", True)
    # even a hand-edited state file cannot unlock a locked module
    (feat._state_file()).write_text(json.dumps({"hosted": {"clone.coqui_xtts": True}}), encoding="utf-8")
    assert not feat.is_on("clone.coqui_xtts")


def test_switch_persists_per_profile_and_reset_restores(feat, monkeypatch):
    assert feat.set_state("pipeline.oneflow", False) is False
    assert not feat.is_on("pipeline.oneflow")
    monkeypatch.setenv("VOICEDUB_PROFILE", "hosted")
    assert feat.is_on("pipeline.classic")         # hosted state untouched by the local switch
    monkeypatch.setenv("VOICEDUB_PROFILE", "local")
    feat.reset()
    assert feat.is_on("pipeline.oneflow")


def test_unknown_module_is_on_and_unknown_switch_raises(feat):
    assert feat.is_on("mt.some_future_engine")
    with pytest.raises(KeyError):
        feat.set_state("nope.nope", False)


def test_job_refusal_names_every_switched_off_choice(feat):
    feat.set_state("pipeline.oneflow", False)
    feat.set_state("tts.elevenlabs", False)
    req = {"pipeline_mode": "oneflow", "use_elevenlabs": True, "use_edge_tts": True, "asr_model": "groq-whisper"}
    msg = feat.job_refusal(req, "url")
    assert "OneFlow (pipeline.oneflow)" in msg and "ElevenLabs (tts.elevenlabs)" in msg
    assert "Edge" not in msg
    assert feat.job_refusal({"pipeline_mode": "classic"}, "upload") is None


def test_false_boolean_field_does_not_select_a_module(feat):
    feat.set_state("clone.coqui_xtts", False)
    assert feat.job_refusal({"use_coqui_xtts": False}, "upload") is None
    assert feat.job_refusal({"use_coqui_xtts": True}, "upload") is not None


def test_list_field_selects_only_when_non_empty(feat):
    feat.set_state("option.dub_chain", False)
    assert feat.job_refusal({"dub_chain": []}, "url") is None
    assert feat.job_refusal({"dub_chain": ["en", "hi"]}, "url") is not None


def test_srt_job_from_a_link_is_also_a_link_job(feat):
    feat.set_state("input.url", False)
    assert feat.job_refusal({}, ("srt",)) is None
    assert "Video link" in feat.job_refusal({}, ("srt", "url"))


def test_route_gate_local_403_hosted_404(feat, monkeypatch):
    feat.set_state("surface.outputs", False)
    assert feat.route_refusal("GET", "/api/outputs")[0] == 403
    assert feat.route_refusal("GET", "/api/jobs/abc123") is None
    monkeypatch.setenv("VOICEDUB_PROFILE", "hosted")
    assert feat.route_refusal("GET", "/api/cookies/status") == (404, "Not Found")
    assert feat.route_refusal("POST", "/api/jobs/abc/compare-translations")[0] == 404
    assert feat.route_refusal("GET", "/static/jobs/abc/result.mp4")[0] == 404   # T04
    assert feat.route_refusal("GET", "/static/jobs") is None                      # prefix needs a deeper path


def test_trainer_forwarding_is_impossible_in_hosted(feat, monkeypatch):
    assert feat.trainer_allowed()
    feat.set_state("integration.trainer", False)
    assert not feat.trainer_allowed()
    feat.reset()
    monkeypatch.setenv("VOICEDUB_PROFILE", "hosted")
    monkeypatch.setenv("HINGLISH_TRAINER_URL", "")
    assert not feat.trainer_allowed()


def test_bad_profile_value_falls_back_to_local(feat, monkeypatch):
    monkeypatch.setenv("VOICEDUB_PROFILE", "  HOSTED ")
    assert feat.profile() == "hosted"
    monkeypatch.setenv("VOICEDUB_PROFILE", "staging")
    assert feat.profile() == "local"


# ── API (FastAPI TestClient) ──
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEDUB_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("VOICEDUB_PROFILE", "local")
    import features
    importlib.reload(features)
    import app as app_module
    monkeypatch.setattr(app_module, "_features", features)
    from fastapi.testclient import TestClient
    return TestClient(app_module.app), features


def test_api_lists_and_switches_modules(client):
    c, feat = client
    r = c.get("/api/features")
    assert r.status_code == 200 and r.json()["profile"] == "local"
    assert c.patch("/api/features/surface.outputs", json={"on": False}).json() == {"id": "surface.outputs", "on": False}
    r = c.get("/api/outputs")
    assert r.status_code == 403 and "surface.outputs" in r.json()["detail"]
    assert c.patch("/api/features/nope", json={"on": False}).status_code == 404


def test_api_refuses_a_job_that_selects_a_switched_off_module(client):
    c, feat = client
    feat.set_state("pipeline.oneflow", False)
    r = c.post("/api/jobs", json={"url": "https://example.com/v.mp4", "pipeline_mode": "oneflow"})
    assert r.status_code == 400 and "pipeline.oneflow" in r.json()["detail"]


def test_api_hosted_switches_need_the_admin_token(client, monkeypatch):
    c, feat = client
    monkeypatch.setenv("VOICEDUB_PROFILE", "hosted")
    monkeypatch.delenv("VOICEDUB_ADMIN_TOKEN", raising=False)
    assert c.patch("/api/features/pipeline.classic", json={"on": False}).status_code == 403
    monkeypatch.setenv("VOICEDUB_ADMIN_TOKEN", "s3cret")
    assert c.patch("/api/features/pipeline.classic", json={"on": False}, headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert c.patch("/api/features/pipeline.classic", json={"on": False}, headers={"X-Admin-Token": "s3cret"}).status_code == 200
    assert c.patch("/api/features/clone.cosyvoice", json={"on": True}, headers={"X-Admin-Token": "s3cret"}).status_code == 409
