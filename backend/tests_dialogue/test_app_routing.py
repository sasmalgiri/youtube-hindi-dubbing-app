"""API routing: upload + with-srt jobs in pipeline_mode=hindi_dialogue reach the
shared dialogue orchestrator and expose an honest result status + report."""
import json

import pytest

from dubbing.dialogue import orchestrator as orch_mod

from test_orchestrator_e2e import HAVE_FFMPEG, HINDI, SCRIPT, _components, _make_media

pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod
    calls = {}
    holder = {}

    def fake_components(cfg):
        comps = _components(calls, orch_holder=holder)
        return comps

    real_init = orch_mod.DialogueOrchestrator.__init__

    def init(self, cfg, components=None, on_progress=None, cancel_check=None):
        cfg.tts_providers = ["mock"]
        real_init(self, cfg, components, on_progress, cancel_check)
        holder["orch"] = self

    monkeypatch.setattr(orch_mod, "default_components", fake_components)
    monkeypatch.setattr(orch_mod.DialogueOrchestrator, "__init__", init)
    monkeypatch.setattr(app_mod, "SAVED_DIR", tmp_path / "saved")
    (tmp_path / "saved").mkdir()
    return TestClient(app_mod.app), app_mod, calls


def _wait(app_mod, job_id):
    job = app_mod.JOBS[job_id]
    job.worker_thread.join(timeout=120)
    return job


def test_upload_routes_to_dialogue_profile(client, tmp_path):
    c, app_mod, calls = client
    media = _make_media(tmp_path)
    with open(media, "rb") as f:
        r = c.post("/api/jobs/upload", files={"file": ("clip.mp4", f, "video/mp4")},
                   data={"pipeline_mode": "hindi_dialogue", "multi_speaker": "false"})
    assert r.status_code == 200, r.text
    job = _wait(app_mod, r.json()["id"])
    assert job.state == "done" and job.result_status == "completed", (job.message, job.status_reasons)
    assert calls["diarize"] == 1                       # diarization ran although multi_speaker=false
    assert any("diarization always runs" in e.get("message", "") for e in job.events)
    body = c.get(f"/api/jobs/{job.id}").json()
    assert body["result_status"] == "completed" and body["report_path"].endswith("report.md")
    rep = c.get(f"/api/jobs/{job.id}/report?fmt=json").json()
    assert rep["generated_turn_ids"] == ["t0001", "t0002", "t0003", "t0004"]
    assert {s["speaker_id"] for s in json.loads(json.dumps(rep["speakers"]))} == {"SPEAKER_00", "SPEAKER_01"}
    assert job.saved_video and job.saved_video.endswith("dubbed_hi.mp4")


def test_with_srt_hindi_routes_to_dialogue_profile(client, tmp_path):
    c, app_mod, calls = client
    media = _make_media(tmp_path)
    srt = "\n".join(f"{i}\n00:00:0{int(s)},{int((s % 1) * 1000):03d} --> "
                    f"00:00:0{int(e)},{int((e % 1) * 1000):03d}\n{HINDI[t]}\n"
                    for i, (_, _, s, e, t) in enumerate(SCRIPT, 1))
    r = c.post("/api/jobs/with-srt",
               files={"srt_file": ("hi.srt", srt.encode("utf-8"), "text/plain")},
               data={"url": str(media), "pipeline_mode": "hindi_dialogue"})
    assert r.status_code == 200, r.text
    job = _wait(app_mod, r.json()["id"])
    assert job.result_status == "completed", job.status_reasons
    assert calls["diarize"] == 1 and "asr" not in calls
    assert [s["speaker_id"] for s in job.segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]
