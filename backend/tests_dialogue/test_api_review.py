"""API side of the hindi_dialogue review / re-voice / voice-picker flow.

run_dialogue is replaced by a fake that behaves like the orchestrator's
contract: with cfg.review_before_voice it calls the review hook with a
"before_voice" packet, writes this job's checkpoint, the "after_run" packet
and its clips, and returns a result. The API is tested against that."""
import dataclasses
import json
import threading
import time
import types
import wave
from pathlib import Path

import pytest

from dubbing.dialogue import orchestrator as orch
from dubbing.dialogue import tts as dialogue_tts

# DialogueConfig fields of the shared contract (added by the orchestrator side).
CONTRACT_FIELDS = {"review_before_voice": False, "turn_edits": {}, "voice_overrides": {},
                   "speaker_merges": {}, "resume": False, "keep_original_audio": False,
                   "english_subtitles": True, "burn_subtitles": False, "container": "mp4",
                   "voice_match": "off"}
EDITS = {"turn_edits": {"t0001": {"hi": "नमस्ते दोस्त।"}, "t0002": {"delete": True}},
         "voice_overrides": {"SPEAKER_01": {"provider": "edge", "voice": "hi-IN-SwaraNeural",
                                            "pitch": "+20Hz"}},
         "speaker_merges": {"SPEAKER_02": "SPEAKER_00"}}


def _wav(path: Path, seconds: float = 0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * int(24000 * seconds))
    return path


def _packet(stage, clip=None):
    return {"job_stage": stage, "media_duration": 4.0,
            "turns": [{"turn_id": "t0001", "speaker_id": "SPEAKER_00", "start": 0.0, "end": 1.5,
                       "english": "Hello friend.", "hindi": "नमस्ते मित्र।", "flags": [],
                       "required": True, "clip": clip, "overflow_s": None},
                      {"turn_id": "t0002", "speaker_id": "SPEAKER_01", "start": 2.0, "end": 3.5,
                       "english": "Hi.", "hindi": "हाय।", "flags": [], "required": True,
                       "clip": None, "overflow_s": None}],
            "speakers": [{"speaker_id": "SPEAKER_00", "voice_category": "male_like",
                          "category_confidence": 0.9, "total_speech_s": 1.5, "turns": 1,
                          "provider": "edge", "voice": "hi-IN-MadhurNeural", "pitch": None,
                          "variant": 0, "reference_clip": None, "override": False}],
            "voice_options": {}}


class FakeDialogue:
    """Stands in for orchestrator.run_dialogue (contract signature)."""

    def __init__(self):
        self.calls = []
        self.statuses = ["completed", "completed_with_warnings"]

    def __call__(self, cfg, on_progress=None, cancel_check=None, components=None,
                 on_legacy_pipeline=None, review=None):
        from dubbing.dialogue.speaker_registry import SpeakerRegistry
        call = {"cfg": cfg}
        self.calls.append(call)
        work, out = Path(cfg.work_dir), Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        status = self.statuses[min(len(self.calls), len(self.statuses)) - 1]
        if cfg.review_before_voice and review is not None:
            packet = _packet("before_voice")
            (work / "review.json").write_text(json.dumps(packet), encoding="utf-8")
            call["edits"] = review(packet)
            if cancel_check():
                status = "cancelled"
        (work / "checkpoint").mkdir(exist_ok=True)
        (work / "checkpoint" / "state.json").write_text("{}", encoding="utf-8")
        _wav(out / "clips" / "t0001.wav")
        _wav(out / "clips" / "ref_SPEAKER_00.wav")
        (out / "review.json").write_text(json.dumps(_packet("after_run", clip="t0001.wav")),
                                         encoding="utf-8")
        reg = SpeakerRegistry()
        reg.register("SPEAKER_00", voice_category="male_like", total_speech_s=1.5)
        reg.bind_provider("edge")
        reg.save(out / "speakers.json")
        srt = out / "subtitles_hi.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,500\nनमस्ते\n", encoding="utf-8")
        md = out / "report.md"
        md.write_text(f"# run {len(self.calls)}", encoding="utf-8")
        video = out / f"dubbed_hi.{cfg.container}"
        video.write_bytes(b"video")
        turns = [types.SimpleNamespace(turn_id="t0001", speaker_id="SPEAKER_00", source_start=0.0,
                                       source_end=1.5, source_text="Hello friend.",
                                       speech_text="नमस्ते मित्र।")]
        return orch.DialogueResult(status=status, output_video=video,
                                   report_json=out / "report.json", report_md=md,
                                   subtitles=srt, reasons=[], turns=turns)


@pytest.fixture()
def api(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod
    from test_modules import FakeProbe

    class _NoStore:
        def save(self, job):
            pass

        def delete(self, job_id):
            pass

    for d in ("saved", "outputs"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(app_mod, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(app_mod, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(app_mod, "JOBS", {})
    monkeypatch.setattr(app_mod, "_store", _NoStore())
    monkeypatch.setattr(app_mod, "_purge_global_caches", lambda: 0)
    monkeypatch.setattr(app_mod, "_mark_url_completed", lambda url: None)
    monkeypatch.setattr(app_mod, "_dialogue_probe", lambda max_age=60.0: FakeProbe())
    # The contract's DialogueConfig fields, whichever side of that change is merged.
    have = {f.name for f in dataclasses.fields(orch.DialogueConfig)}
    missing = [(n, type(d), dataclasses.field(default_factory=dict) if isinstance(d, dict)
                else dataclasses.field(default=d))
               for n, d in CONTRACT_FIELDS.items() if n not in have]
    if missing:
        monkeypatch.setattr(orch, "DialogueConfig", dataclasses.make_dataclass(
            "DialogueConfig", missing, bases=(orch.DialogueConfig,)))
    fake = FakeDialogue()
    monkeypatch.setattr(orch, "run_dialogue", fake)
    return types.SimpleNamespace(c=TestClient(app_mod.app), app=app_mod, fake=fake, tmp=tmp_path)


def _upload(api, **data):
    form = {"pipeline_mode": "hindi_dialogue", **data}
    r = api.c.post("/api/jobs/upload", files={"file": ("My Clip.mp4", b"video", "video/mp4")},
                   data=form)
    assert r.status_code == 200, r.text
    return api.app.JOBS[r.json()["id"]]


def _wait_state(job, state, timeout=10.0):
    end = time.time() + timeout
    while job.state != state and time.time() < end:
        time.sleep(0.02)
    assert job.state == state, (job.state, job.message)


def _finish(job):
    job.worker_thread.join(timeout=30)
    assert not job.worker_thread.is_alive()
    return job


# ── review before voicing ────────────────────────────────────────────────────

def test_review_pause_round_trip_and_resume(api):
    job = _upload(api, dialogue_review="true")
    _wait_state(job, "review_translation")
    assert job.message == "Review the Hindi lines and voices, then continue"
    body = api.c.get(f"/api/jobs/{job.id}").json()
    assert body["state"] == "review_translation"
    assert body["dialogue_review"]["job_stage"] == "before_voice"
    assert [s["turn_id"] for s in api.c.get(f"/api/jobs/{job.id}/transcript").json()["segments"]] \
        == ["t0001", "t0002"]                            # lines under review are visible
    live = api.c.get(f"/api/jobs/{job.id}/dialogue/review").json()
    assert live["job_stage"] == "before_voice" and live["turns"][0]["hindi"] == "नमस्ते मित्र।"
    # bad edits are refused and the job keeps waiting
    for bad in ({"turn_edits": {"t0001": {"english": "x"}}},
                {"turn_edits": {"t0001": {"hi": "  "}}},
                {"voice_overrides": {"SPEAKER_00": {"provider": "edge", "voice": "nobody"}}},
                {"voice_overrides": {"SPEAKER_00": {"category": "robot"}}},
                {"speaker_merges": {"SPEAKER_00": "SPEAKER_00"}}):
        assert api.c.post(f"/api/jobs/{job.id}/dialogue/review", json=bad).status_code == 400, bad
    assert job.state == "review_translation"
    r = api.c.post(f"/api/jobs/{job.id}/dialogue/review", json=EDITS)
    assert r.status_code == 200 and r.json() == {"status": "resumed"}
    _finish(job)
    assert api.c.post(f"/api/jobs/{job.id}/dialogue/review", json=EDITS).status_code == 409
    call, = api.fake.calls
    assert call["cfg"].review_before_voice is True
    assert call["edits"] == EDITS                        # the hook handed over the edits
    assert job.state == "done" and job.result_status == "completed"
    assert job.dialogue_review is None and job.dialogue_edits == EDITS
    # after the run: the finished packet from the job folder
    after = api.c.get(f"/api/jobs/{job.id}/dialogue/review").json()
    assert after["job_stage"] == "after_run" and after["turns"][0]["clip"] == "t0001.wav"
    assert api.c.get(f"/api/jobs/{job.id}").json()["dialogue_review"] is None


def test_continue_resumes_a_paused_dialogue_job_without_edits(api):
    job = _upload(api, dialogue_review="true")
    _wait_state(job, "review_translation")
    r = api.c.post(f"/api/jobs/{job.id}/continue")
    assert r.json() == {"status": "resumed", "from_state": "review_translation"}
    _finish(job)
    assert api.fake.calls[0]["edits"] is None and job.state == "done"


def test_cancel_while_paused_stops_the_waiting_job(api):
    job = _upload(api, dialogue_review="true")
    _wait_state(job, "review_translation")
    assert api.c.delete(f"/api/jobs/{job.id}").json() == {"status": "cancelled"}
    _finish(job)                                         # the hook saw the cancel flag
    assert api.fake.calls[0]["edits"] is None
    assert job.state == "error" and job.result_status == "cancelled"
    assert api.c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={}).status_code == 409


def test_review_endpoints_404_and_409(api):
    assert api.c.get("/api/jobs/nope/dialogue/review").status_code == 404
    job = _upload(api)                                   # no review requested
    _finish(job)
    assert "edits" not in api.fake.calls[0]              # never paused
    assert api.c.post(f"/api/jobs/{job.id}/dialogue/review", json={}).status_code == 409
    (api.app.OUTPUTS / job.id / "dialogue_out" / "review.json").unlink()
    saved = api.c.get(f"/api/jobs/{job.id}/dialogue/review").json()   # the saved copy
    assert saved["job_stage"] == "after_run"
    (Path(job.saved_folder) / "review.json").unlink()
    assert api.c.get(f"/api/jobs/{job.id}/dialogue/review").status_code == 404


# ── request options reach DialogueConfig ─────────────────────────────────────

def test_upload_form_options_reach_dialogue_config(api):
    overrides = {"SPEAKER_01": {"provider": "edge", "voice": "hi-IN-SwaraNeural", "pitch": "+20Hz"},
                 "SPEAKER_02": {"category": "female_like"}}
    job = _finish(_upload(api, dialogue_keep_original_audio="true", dialogue_english_subtitles="false",
                          dialogue_burn_subtitles="true", dialogue_container="MKV",
                          dialogue_voice_overrides_json=json.dumps(overrides)))
    cfg = api.fake.calls[0]["cfg"]
    assert (cfg.keep_original_audio, cfg.english_subtitles, cfg.burn_subtitles, cfg.container) == \
        (True, False, True, "mkv")
    assert cfg.voice_overrides == overrides and cfg.review_before_voice is False
    assert job.saved_video.endswith("dubbed_hi.mkv")
    r = api.c.get(f"/api/jobs/{job.id}/result")
    assert r.status_code == 200 and r.headers["content-type"] == "video/x-matroska"
    conf = api.c.get(f"/api/jobs/{job.id}").json()["config"]
    assert conf["dialogue_container"] == "mkv" and conf["dialogue_burn_subtitles"] is True


def test_defaults_leave_the_dialogue_config_unchanged(api):
    _finish(_upload(api))
    cfg = api.fake.calls[0]["cfg"]
    assert {k: getattr(cfg, k) for k in CONTRACT_FIELDS} == CONTRACT_FIELDS


def test_with_srt_form_options_reach_dialogue_config(api, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    r = api.c.post("/api/jobs/with-srt",
                   files={"srt_file": ("hi.srt", "1\n00:00:00,000 --> 00:00:01,000\nनमस्ते\n".encode(),
                                       "text/plain")},
                   data={"url": str(media), "pipeline_mode": "hindi_dialogue",
                         "dialogue_review": "true", "dialogue_container": "mkv"})
    assert r.status_code == 200, r.text
    job = api.app.JOBS[r.json()["id"]]
    _wait_state(job, "review_translation")
    api.c.post(f"/api/jobs/{job.id}/continue")
    _finish(job)
    cfg = api.fake.calls[0]["cfg"]
    assert cfg.review_before_voice is True and cfg.container == "mkv"
    assert Path(cfg.translated_srt).name == "translated_upload.srt"


def test_bad_voice_overrides_json_is_refused(api):
    r = api.c.post("/api/jobs/upload", files={"file": ("a.mp4", b"v", "video/mp4")},
                   data={"pipeline_mode": "hindi_dialogue",
                         "dialogue_voice_overrides_json": '{"SPEAKER_00": {"provider": "edge", '
                                                          '"voice": "en-US-Nobody"}}'})
    assert r.status_code == 422 and "en-US-Nobody" in r.text
    assert api.app.JOBS == {} and list(api.app.OUTPUTS.iterdir()) == []
    r = api.c.post("/api/jobs", json={"url": "https://example.com/v", "pipeline_mode": "hindi_dialogue",
                                      "dialogue_voice_overrides_json": "{not json"})
    assert r.status_code == 422


# ── re-voice ─────────────────────────────────────────────────────────────────

def test_revoice_reuses_the_job_and_refreshes_its_outputs(api):
    job = _upload(api, dialogue_review="true")
    _wait_state(job, "review_translation")
    api.c.post(f"/api/jobs/{job.id}/dialogue/review", json=EDITS)
    _finish(job)
    first_cfg = api.fake.calls[0]["cfg"]
    first_folder = Path(job.saved_folder)
    assert first_folder.name.endswith(f"[HI Dialogue completed] ({job.id})")
    assert api.c.get(f"/api/jobs/{job.id}").json()["dialogue_revoice_ready"] is True

    more = {"turn_edits": {"t0001": {"speaker_id": "SPEAKER_01"}, "t0003": {"hi": "ठीक है।"}},
            "voice_overrides": {"SPEAKER_00": {"category": "female_like"}},
            "container": "mkv", "keep_original_audio": True}
    r = api.c.post(f"/api/jobs/{job.id}/dialogue/revoice", json=more)
    assert r.status_code == 200 and r.json() == {"status": "started"}
    _finish(job)
    cfg = api.fake.calls[1]["cfg"]
    assert cfg.resume is True and cfg.review_before_voice is False
    assert (cfg.work_dir, cfg.output_dir) == (first_cfg.work_dir, first_cfg.output_dir)
    # every edit so far, later ones winning field by field
    assert cfg.turn_edits == {"t0001": {"hi": "नमस्ते दोस्त।", "speaker_id": "SPEAKER_01"},
                              "t0002": {"delete": True}, "t0003": {"hi": "ठीक है।"}}
    assert cfg.voice_overrides == {**EDITS["voice_overrides"], "SPEAKER_00": {"category": "female_like"}}
    assert cfg.speaker_merges == EDITS["speaker_merges"]
    assert (cfg.container, cfg.keep_original_audio) == ("mkv", True)
    # the same job, a refreshed outcome and saved copy
    assert job.state == "done" and job.result_status == "completed_with_warnings"
    assert not first_folder.exists()
    assert Path(job.saved_folder).name.endswith(f"[HI Dialogue completed_with_warnings] ({job.id})")
    assert job.saved_video.endswith("dubbed_hi.mkv") and Path(job.saved_video).exists()
    assert job.result_path.name == "dubbed_hi.mkv"
    assert (Path(job.saved_folder) / "report.md").read_text(encoding="utf-8") == "# run 2"
    assert [e for e in job.events if e.get("type") == "complete"] == \
        [{"type": "complete", "state": "done", "result_status": "completed_with_warnings"}]


def test_revoice_refusals(api):
    assert api.c.post("/api/jobs/nope/dialogue/revoice", json={}).status_code == 404
    job = _finish(_upload(api))
    job.state = "running"
    r = api.c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={})
    assert r.status_code == 409 and "running" in r.json()["detail"]
    job.state = "done"
    assert api.c.post(f"/api/jobs/{job.id}/dialogue/revoice",
                      json={"container": "avi"}).status_code == 400
    assert api.c.post(f"/api/jobs/{job.id}/dialogue/revoice",
                      json={"turn_edits": {"t1": {"delete": "yes"}}}).status_code == 400
    (api.app.OUTPUTS / job.id / "work" / "checkpoint" / "state.json").unlink()
    r = api.c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={})
    assert r.status_code == 409 and "checkpoint" in r.json()["detail"]
    assert api.c.get(f"/api/jobs/{job.id}").json()["dialogue_revoice_ready"] is False
    classic = api.app.Job(id="classic00001", state="done",
                          original_req=api.app.JobCreateRequest(url="https://example.com/v"))
    api.app.JOBS[classic.id] = classic
    assert api.c.post(f"/api/jobs/{classic.id}/dialogue/revoice", json={}).status_code == 409
    assert len(api.fake.calls) == 1                      # nothing was started


def test_revoice_waits_for_the_pipeline_slot(api):
    job = _finish(_upload(api))
    api.app._pipeline_semaphore.acquire()                # another job is running
    try:
        assert api.c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={}).status_code == 200
        time.sleep(0.3)
        assert job.state == "queued" and len(api.fake.calls) == 1
        assert api.c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={}).status_code == 409
    finally:
        api.app._pipeline_semaphore.release()
    _finish(job)
    assert len(api.fake.calls) == 2 and api.fake.calls[1]["cfg"].resume is True


def test_merged_edits_stay_resolved(api):
    merge = api.app._merge_dialogue_edits
    assert merge({"speaker_merges": {"A": "B"}}, {"speaker_merges": {"B": "C"}}) == \
        {"speaker_merges": {"A": "C", "B": "C"}}
    assert merge({"speaker_merges": {"A": "B"}}, {"speaker_merges": {"C": "A"}}) == \
        {"speaker_merges": {"A": "B", "C": "B"}}
    assert merge({"turn_edits": {"t1": {"delete": True}}}, {"turn_edits": {"t1": {"delete": False}}}) == \
        {"turn_edits": {"t1": {"delete": False}}}
    assert merge({}, {}) == {}


# ── clips ────────────────────────────────────────────────────────────────────

def test_clip_endpoint_serves_clips_and_rejects_other_names(api):
    job = _finish(_upload(api))
    r = api.c.get(f"/api/jobs/{job.id}/dialogue/clip/t0001.wav")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    for bad in ("report.md", "t 1.wav", "t0001.wav.txt", "..%5Cspeakers.json"):
        assert api.c.get(f"/api/jobs/{job.id}/dialogue/clip/{bad}").status_code == 400, bad
    for traversal in ("..%2F..%2Fsubtitles_hi.wav", "../speakers.wav"):
        assert api.c.get(f"/api/jobs/{job.id}/dialogue/clip/{traversal}").status_code in (400, 404)
    assert api.c.get(f"/api/jobs/{job.id}/dialogue/clip/t9999.wav").status_code == 404
    # the saved copy serves it once the job folder's clip is gone
    (api.app.OUTPUTS / job.id / "dialogue_out" / "clips" / "ref_SPEAKER_00.wav").unlink()
    assert api.c.get(f"/api/jobs/{job.id}/dialogue/clip/ref_SPEAKER_00.wav").status_code == 200


# ── voices + preview ─────────────────────────────────────────────────────────

def test_voices_are_exactly_the_bindable_slots(api, monkeypatch):
    from dubbing.hindi_voices import FEMALE_SLOTS, MALE_SLOTS
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    v = api.c.get("/api/dialogue/voices").json()
    assert "mock" not in v and {"edge", "sarvam", "google"} <= set(v)
    edge = v["edge"]
    assert edge["paid"] is False and edge["preview"] is True
    slots = lambda opts: [o["voice"] + (f"|{o['pitch']}" if o["pitch"] else "") for o in opts]
    assert slots(edge["male_like"]) == MALE_SLOTS and slots(edge["female_like"]) == FEMALE_SLOTS
    assert edge["male_like"][5] == {"voice": "hi-IN-MadhurNeural", "pitch": "+20Hz",
                                    "label": "Madhur +20Hz"}
    assert v["sarvam"]["paid"] is True and v["sarvam"]["preview"] is False
    assert {o["voice"] for o in v["sarvam"]["female_like"]} >= {"anushka", "vidya"}


class _FakeEdge:
    paid = False
    made = []

    def synthesize(self, text, binding, out_wav):
        _FakeEdge.made.append((text, dict(binding)))
        return _wav(Path(out_wav))


def test_voice_preview_synthesizes_once_and_caches(api, monkeypatch):
    monkeypatch.setattr(_FakeEdge, "made", [])
    monkeypatch.setitem(dialogue_tts.PROVIDER_CLASSES, "edge", _FakeEdge)
    body = {"provider": "edge", "voice": "hi-IN-MadhurNeural", "pitch": "+20Hz"}
    r = api.c.post("/api/dialogue/voice-preview", json=body)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav", r.text
    assert r.content[:4] == b"RIFF"
    (text, binding), = _FakeEdge.made
    assert text == "नमस्ते, यह मेरी आवाज़ है।"
    assert (binding["voice"], binding["pitch"]) == ("hi-IN-MadhurNeural", "+20Hz")
    assert api.c.post("/api/dialogue/voice-preview", json=body).status_code == 200
    assert len(_FakeEdge.made) == 1                      # served from the cache
    cached = list((api.app.OUTPUTS / "_voice_previews").glob("*.wav"))
    assert len(cached) == 1 and cached[0].name.startswith("edge_")
    api.c.post("/api/dialogue/voice-preview", json={**body, "text": "दूसरी लाइन"})
    assert len(_FakeEdge.made) == 2 and _FakeEdge.made[1][0] == "दूसरी लाइन"


def test_voice_preview_refusals(api, monkeypatch):
    monkeypatch.setattr(_FakeEdge, "made", [])
    monkeypatch.setitem(dialogue_tts.PROVIDER_CLASSES, "edge", _FakeEdge)
    called = []

    class PaidNoKey:
        paid = True
        key = ""

        def synthesize(self, *a):
            called.append(a)

    monkeypatch.setitem(dialogue_tts.PROVIDER_CLASSES, "sarvam", PaidNoKey)
    post = lambda **b: api.c.post("/api/dialogue/voice-preview", json=b)
    assert post(provider="edge", voice="en-US-Nobody").status_code == 400
    assert post(provider="edge", voice="hi-IN-MadhurNeural", pitch="+50Hz").status_code == 400
    assert post(provider="mock", voice="mock-male-1").status_code == 400
    assert post(provider="edge", voice="hi-IN-MadhurNeural", text="क" * 301).status_code == 400
    r = post(provider="sarvam", voice="abhilash")
    assert r.status_code == 409 and "key" in r.json()["detail"]
    assert post(provider="indic_parler", voice="Rohit").status_code == 409
    assert called == [] and _FakeEdge.made == []         # nothing was synthesized

    class Broken(_FakeEdge):
        def synthesize(self, text, binding, out_wav):
            raise RuntimeError("edge: empty audio")
    monkeypatch.setitem(dialogue_tts.PROVIDER_CLASSES, "edge", Broken)
    r = post(provider="edge", voice="hi-IN-SwaraNeural")
    assert r.status_code == 502 and "empty audio" in r.json()["detail"]
    assert list((api.app.OUTPUTS / "_voice_previews").glob("*")) == []   # no partial file cached


# ── persistence ──────────────────────────────────────────────────────────────

def test_review_state_and_options_survive_the_job_store(tmp_path):
    import app as app_mod
    from jobstore import JobStore
    store = JobStore(tmp_path / "jobs.db")
    req = app_mod.JobCreateRequest(url="https://example.com/v", pipeline_mode="hindi_dialogue",
                                   dialogue_review=True, dialogue_container="mkv",
                                   dialogue_keep_original_audio=True, dialogue_english_subtitles=False,
                                   dialogue_burn_subtitles=True,
                                   dialogue_voice_overrides_json=json.dumps(
                                       {"SPEAKER_00": {"category": "female_like"}}))
    job = app_mod.Job(id="persist00001", state="done", original_req=req,
                      dialogue_review=_packet("before_voice"), dialogue_edits=EDITS)
    store.save(job)
    loaded = {}
    store.load_all(loaded)
    got = loaded[job.id]
    assert got.dialogue_review == _packet("before_voice") and got.dialogue_edits == EDITS
    r = got.original_req
    assert (r.dialogue_review, r.dialogue_container, r.dialogue_keep_original_audio,
            r.dialogue_english_subtitles, r.dialogue_burn_subtitles) == (True, "mkv", True, False, True)
    assert json.loads(r.dialogue_voice_overrides_json) == {"SPEAKER_00": {"category": "female_like"}}
    # a job that was paused for review when the server stopped
    paused = app_mod.Job(id="paused000001", state="review_translation", original_req=req,
                         dialogue_review=_packet("before_voice"))
    store.save(paused)
    loaded = {}
    store.load_all(loaded)
    app_mod._settle_loaded_jobs(loaded, store)
    got = loaded[paused.id]
    assert got.state == "error" and got.dialogue_review is None and "review" in got.error
    assert isinstance(got.pause_event, threading.Event)
    store.close()
