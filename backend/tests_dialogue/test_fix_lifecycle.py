"""A re-voice never destroys the last good result; restarts, cancels during
the final mix and burned-in subtitles behave.

* orchestrator: each run starts with an empty output folder (the earlier
  run's files are moved aside), keeps nothing stale, and a run that does not
  finish puts the earlier outputs and the checkpoint they match back.
* app: a failed, cancelled or interrupted (server restart) re-voice leaves the
  job exactly as it was: result, saved copy, edits, job folder, checkpoint.
* restart: a job is never stripped of its request while app.py is importing.
* mux: burned-in Hindi is not shown twice, a burn failure falls back to soft
  subtitles, and a cancel stops the encode.

Same fakes as the neighbouring tests: test_api_review (fake run_dialogue),
test_e2e_review_revoice (API + the real orchestrator and FFmpeg),
test_review_resume (orchestrator on synthetic media), test_mux_outputs.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from dubbing.dialogue import audio, checkpoint
from dubbing.dialogue import mix as mix_mod
from dubbing.dialogue import orchestrator as orch
from dubbing.dialogue import tts as dialogue_tts
from dubbing.dialogue.tts import MockProvider

from test_api_review import FakeDialogue, _FakeEdge, _packet, api  # noqa: F401  (fixture)
from test_api_review import _finish as api_finish
from test_api_review import _upload as api_upload
from test_api_review import _wait_state as api_wait_state
from test_e2e_review_revoice import EN, e2e  # noqa: F401  (fixture)
from test_e2e_review_revoice import _finish as e2e_finish
from test_e2e_review_revoice import _upload as e2e_upload
from test_mux_outputs import BURN_SRT, _probe, media, needs_libass  # noqa: F401  (fixture)
from test_orchestrator_e2e import HAVE_FFMPEG, HINDI, _make_media
from test_review_resume import _orch

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")
BACKEND = Path(__file__).resolve().parents[1]
NEW_T2 = "मैं पूरी रात स्टेशन पर थी।"


def _bytes(p: Path) -> bytes:
    return Path(p).read_bytes()


# ── orchestrator: per-run deliverables ───────────────────────────────────────

@needs_ffmpeg
def test_a_failed_revoice_puts_back_the_previous_outputs_and_checkpoint(tmp_path, monkeypatch):
    prov = MockProvider()
    res1 = _orch(tmp_path, provider=prov)[0].run()
    assert res1.status == "completed", res1.reasons
    out, work = tmp_path / "out", tmp_path / "work"
    before = {p.relative_to(out).as_posix(): _bytes(p) for p in out.rglob("*") if p.is_file()}
    ckpt = checkpoint.checkpoint_path(work).read_text(encoding="utf-8")

    def broken_mux(*a, **kw):
        raise RuntimeError("no space left on device")
    monkeypatch.setattr(mix_mod, "mux", broken_mux)
    orch2, _ = _orch(tmp_path, provider=prov, cfg={"resume": True, "turn_edits": {"t0002": {"hi": NEW_T2}}})
    res2 = orch2.run()
    assert res2.status == "failed" and res2.output_video is None
    # every file of the earlier run is back, byte for byte, and nothing else
    after = {p.relative_to(out).as_posix(): _bytes(p) for p in out.rglob("*") if p.is_file()}
    assert after == before
    # ...with the checkpoint they belong to (the failed run's edit is not in it)
    assert checkpoint.checkpoint_path(work).read_text(encoding="utf-8") == ckpt
    assert NEW_T2 not in ckpt
    # the failed attempt's own report is kept apart, and says why
    assert res2.report_md.parent == work / "failed_run"
    assert "no space left on device" in res2.report_md.read_text(encoding="utf-8")
    assert json.loads(res2.report_json.read_text(encoding="utf-8"))["final_status"] == "failed"
    assert not (work / "previous_run").exists()


@needs_ffmpeg
def test_a_new_run_never_mixes_with_the_previous_runs_files(tmp_path):
    prov = MockProvider()
    assert _orch(tmp_path, provider=prov)[0].run().status == "completed"
    out = tmp_path / "out"
    (out / "dialogue_track_7.wav").write_bytes(b"stem of a line deleted since")
    res2 = _orch(tmp_path, provider=prov, cfg={"resume": True, "container": "mkv"})[0].run()
    assert res2.status == "completed", res2.reasons
    assert not (out / "dialogue_track_7.wav").exists()
    assert (out / "dubbed_hi.mkv").is_file() and not (out / "dubbed_hi.mp4").exists()
    assert not (tmp_path / "work" / "previous_run").exists()     # released once the run finished


def test_moving_the_previous_outputs_aside_and_back(tmp_path, monkeypatch):
    """The CLI's default work folder is inside the output folder; a put-back
    that stopped half way (a file in use) carries on when called again."""
    out = tmp_path / "out"
    work = out / "work"
    ckpt = checkpoint.checkpoint_path(work)
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("run 1", encoding="utf-8")
    (out / "clips").mkdir()
    (out / "clips" / "t0001.wav").write_bytes(b"1")
    (out / "dubbed_hi.mp4").write_bytes(b"video 1")
    assert orch.stash_previous_outputs(work, out) is True
    assert sorted(p.name for p in out.iterdir()) == ["work"]           # the work folder stays
    ckpt.write_text("run 2", encoding="utf-8")
    (out / "dubbed_hi.mkv").write_bytes(b"half a video")
    prev = work / orch.PREVIOUS_RUN_DIR
    # a first put-back stopped after one file
    real_move = orch.shutil.move
    calls = []

    def flaky_move(src, dst):
        calls.append(src)
        if len(calls) == 3:
            raise PermissionError("file in use")
        return real_move(src, dst)
    with monkeypatch.context() as m:
        m.setattr(orch.shutil, "move", flaky_move)
        with pytest.raises(PermissionError):
            orch.recover_interrupted_run(work, out)
    assert prev.is_dir()
    failed = orch.recover_interrupted_run(work, out)
    assert sorted(p.name for p in out.iterdir()) == ["clips", "dubbed_hi.mp4", "work"]
    assert (out / "dubbed_hi.mp4").read_bytes() == b"video 1" and (out / "clips" / "t0001.wav").is_file()
    assert ckpt.read_text(encoding="utf-8") == "run 1"
    assert (failed / "dubbed_hi.mkv").read_bytes() == b"half a video"
    assert not prev.exists() and orch.recover_interrupted_run(work, out) is None


@needs_ffmpeg
def test_a_run_that_cannot_move_the_previous_outputs_changes_nothing(tmp_path, monkeypatch):
    prov = MockProvider()
    assert _orch(tmp_path, provider=prov)[0].run().status == "completed"
    out = tmp_path / "out"
    before = {p.relative_to(out).as_posix(): _bytes(p) for p in out.rglob("*") if p.is_file()}
    real_move, moved = orch.shutil.move, []

    def move(src, dst):
        if Path(src).name == "subtitles_hi.srt":
            raise PermissionError("the file is open in another program")
        moved.append(src)
        return real_move(src, dst)
    monkeypatch.setattr(orch.shutil, "move", move)
    calls = len(prov.calls)
    res = _orch(tmp_path, provider=prov, cfg={"resume": True})[0].run()
    assert res.status == "failed" and "could not be moved aside" in " ".join(res.reasons)
    assert moved and len(prov.calls) == calls                    # some files moved, then put back
    assert {p.relative_to(out).as_posix(): _bytes(p) for p in out.rglob("*") if p.is_file()} == before
    assert res.report_md.parent == tmp_path / "work" / "failed_run"


@needs_ffmpeg
def test_a_failed_first_run_still_writes_the_review_packet(tmp_path, monkeypatch):
    def broken_final_mix(*a, **kw):
        raise RuntimeError("mix exploded")
    monkeypatch.setattr(mix_mod, "final_mix", broken_final_mix)
    res = _orch(tmp_path)[0].run()
    assert res.status == "failed"
    packet = json.loads((tmp_path / "out" / "review.json").read_text(encoding="utf-8"))
    assert packet["job_stage"] == "after_run"
    assert [t["hindi"] for t in packet["turns"]] == list(HINDI.values())
    assert {s["speaker_id"] for s in packet["speakers"]} == {"SPEAKER_00", "SPEAKER_01"}


@needs_ffmpeg
def test_a_cancel_during_the_final_mux_cancels_the_run(tmp_path, monkeypatch):
    state = {"cancel": False}
    real_mux = mix_mod.mux

    def mux(*a, **kw):
        got = real_mux(*a, **kw)
        state["cancel"] = True          # Cancel pressed while the video was being written
        return got
    monkeypatch.setattr(mix_mod, "mux", mux)
    res = _orch(tmp_path, cancel=lambda: state["cancel"])[0].run()
    assert res.status == "cancelled"


@needs_ffmpeg
def test_a_burn_in_failure_falls_back_to_soft_subtitles(tmp_path, monkeypatch):
    real = audio.has_filter
    monkeypatch.setattr(audio, "has_filter", lambda name: False if name == "subtitles" else real(name))
    res = _orch(tmp_path, cfg={"burn_subtitles": True})[0].run()
    assert res.status == "completed_with_warnings", res.reasons
    assert any("burn" in r for r in res.reasons), res.reasons
    streams, _ = _probe(res.output_video)
    subs = [s for s in streams if s["type"] == "subtitle"]
    assert subs and (subs[0]["lang"], subs[0]["default"]) == ("hin", True)


@needs_libass
def test_burned_mp4_gets_no_soft_subtitle_track_and_says_so(tmp_path):
    res = _orch(tmp_path, cfg={"burn_subtitles": True})[0].run()
    assert res.status == "completed", res.reasons
    streams, _ = _probe(res.output_video)
    assert [s["type"] for s in streams] == ["video", "audio"]
    rep = json.loads(res.report_json.read_text(encoding="utf-8"))
    assert any("MKV" in x for x in rep["limitations"]), rep["limitations"]
    assert (tmp_path / "out" / "subtitles_hi.srt").is_file()


# ── mux ──────────────────────────────────────────────────────────────────────

@needs_libass
def test_burned_hindi_is_not_also_a_default_soft_track(media):
    mkv = media["dir"] / "burned.mkv"
    mix_mod.mux(media["video"], media["wav"], mkv, subtitles=media["hi"], burn_subtitles=media["hi"],
                extra_subtitles=[(media["en"], "eng", "English")], container="mkv")
    streams, _ = _probe(mkv)
    assert [(s["type"], s["lang"], s["default"]) for s in streams if s["type"] == "subtitle"] == \
        [("subtitle", "hin", False), ("subtitle", "eng", False)]
    # an MP4 player shows its first subtitle track whatever its flags: none is added
    mp4 = media["dir"] / "burned.mp4"
    mix_mod.mux(media["video"], media["wav"], mp4, subtitles=media["hi"], burn_subtitles=media["hi"],
                extra_subtitles=[(media["en"], "eng", "English")])
    streams, _ = _probe(mp4)
    assert [s["type"] for s in streams] == ["video", "audio"]


@needs_libass
def test_mux_stops_the_encode_on_cancel(tmp_path):
    video = tmp_path / "long.mp4"
    audio.run_ffmpeg(["-f", "lavfi", "-i", "testsrc=size=1280x720:rate=25:duration=30",
                      "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(video)])
    wav = tmp_path / "mix.wav"
    audio.run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=220:duration=30:sample_rate=48000",
                      "-ac", "2", "-acodec", "pcm_s16le", str(wav)])
    burn = tmp_path / "burn.srt"
    burn.write_text(BURN_SRT, encoding="utf-8")
    out = tmp_path / "dubbed_hi.mp4"
    t0 = time.time()
    with pytest.raises(RuntimeError, match="cancelled by user"):
        mix_mod.mux(video, wav, out, burn_subtitles=burn, cancel_check=lambda: time.time() - t0 > 0.3)
    assert time.time() - t0 < 5.0
    assert not out.exists() and not [p for p in tmp_path.iterdir() if "partial" in p.name]


@needs_ffmpeg
def test_a_failed_mux_leaves_no_half_written_video(media, monkeypatch):
    out = media["dir"] / "dubbed_hi.mp4"
    out.write_bytes(b"the earlier, good video")
    real = audio.run_ffmpeg

    def failing(args, *a, **kw):
        real(args, *a, **kw)            # writes (part of) the file...
        raise RuntimeError("ffmpeg failed (1): disk full")
    monkeypatch.setattr(audio, "run_ffmpeg", failing)
    with pytest.raises(RuntimeError):
        mix_mod.mux(media["video"], media["wav"], out, subtitles=media["hi"])
    assert out.read_bytes() == b"the earlier, good video"
    assert sorted(p.name for p in media["dir"].iterdir() if "dubbed" in p.name) == ["dubbed_hi.mp4"]


# ── API: a re-voice that does not finish changes nothing ─────────────────────

def _job_state(job):
    keep = ("state", "result_status", "status_reasons", "result_path", "saved_folder", "saved_video",
            "subtitles_path", "report_path", "speakers", "segments", "dialogue_edits")
    return {k: getattr(job, k) for k in keep}


@needs_ffmpeg
def test_a_failed_revoice_keeps_the_last_good_result(e2e, monkeypatch):
    c, app_mod = e2e.c, e2e.app
    job = e2e_finish(e2e_upload(e2e, _make_media(e2e.tmp)))
    assert job.state == "done" and job.result_status == "completed", (job.message, job.status_reasons)
    before, message = _job_state(job), job.message
    saved_video = _bytes(job.saved_video)
    packet = c.get(f"/api/jobs/{job.id}/dialogue/review").json()
    ckpt = app_mod._dialogue_checkpoint(job).read_text(encoding="utf-8")

    def broken_mux(*a, **kw):
        raise RuntimeError("no space left on device")
    monkeypatch.setattr(mix_mod, "mux", broken_mux)
    r = c.post(f"/api/jobs/{job.id}/dialogue/revoice",
               json={"turn_edits": {"t0002": {"hi": NEW_T2}}, "container": "mkv", "burn_subtitles": True})
    assert r.status_code == 200, r.text
    e2e_finish(job)
    # the job is exactly what it was, and says the re-voice failed
    assert _job_state(job) == before
    assert job.message != message and "Re-voice failed" in job.message
    assert "no space left on device" in job.message
    assert (job.original_req.dialogue_container, job.original_req.dialogue_burn_subtitles) == ("mp4", False)
    # the saved copy is untouched and still the only one
    assert _bytes(job.saved_video) == saved_video
    assert [p.name for p in app_mod.SAVED_DIR.iterdir()] == [Path(job.saved_folder).name]
    assert c.get(f"/api/jobs/{job.id}/result").status_code == 200
    # the review packet and the checkpoint the next re-voice starts from agree
    assert c.get(f"/api/jobs/{job.id}/dialogue/review").json() == packet
    assert app_mod._dialogue_checkpoint(job).read_text(encoding="utf-8") == ckpt
    status = c.get(f"/api/jobs/{job.id}").json()
    assert status["dialogue_revoice_ready"] is True and status["dialogue_revoice_running"] is False
    assert [e for e in job.events if e.get("type") == "complete"][-1]["state"] == "done"


@needs_ffmpeg
def test_stopping_a_running_revoice_restores_the_finished_job(e2e):
    c, app_mod, tts = e2e.c, e2e.app, e2e.tts
    job = e2e_finish(e2e_upload(e2e, _make_media(e2e.tmp)))
    assert job.state == "done" and job.result_status == "completed"
    before, message = _job_state(job), job.message
    saved_video = _bytes(job.saved_video)
    packet = c.get(f"/api/jobs/{job.id}/dialogue/review").json()
    ckpt = app_mod._dialogue_checkpoint(job).read_text(encoding="utf-8")

    started, gate = threading.Event(), threading.Event()
    real = tts.synthesize_part

    def slow(text, binding, out):
        started.set()
        gate.wait(60)
        return real(text, binding, out)
    tts.synthesize_part = slow
    r = c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={"turn_edits": {"t0002": {"hi": NEW_T2}}})
    assert r.status_code == 200, r.text
    assert started.wait(120)
    assert c.get(f"/api/jobs/{job.id}").json()["dialogue_revoice_running"] is True
    r = c.delete(f"/api/jobs/{job.id}")
    assert r.status_code == 200 and r.json() == {"status": "revoice_cancelled"}
    # at once: the finished job is back
    assert _job_state(job) == before and job.message == message
    assert c.get(f"/api/jobs/{job.id}").json()["dialogue_revoice_running"] is False
    gate.set()
    tts.synthesize_part = real
    e2e_finish(job)
    assert _job_state(job) == before and job.message == message
    assert app_mod._dialogue_checkpoint(job).read_text(encoding="utf-8") == ckpt
    assert _bytes(job.saved_video) == saved_video
    assert [p.name for p in app_mod.SAVED_DIR.iterdir()] == [Path(job.saved_folder).name]
    assert c.get(f"/api/jobs/{job.id}/result").status_code == 200
    assert c.get(f"/api/jobs/{job.id}/dialogue/review").json() == packet
    time.sleep(0.5)                                  # a job folder cleanup would have run by now
    assert (app_mod.OUTPUTS / job.id / "dialogue_out" / job.result_path.name).is_file()
    assert c.get(f"/api/jobs/{job.id}").json()["dialogue_revoice_ready"] is True
    # ...and it can be re-voiced again
    r = c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={"turn_edits": {"t0002": {"hi": NEW_T2}}})
    assert r.status_code == 200, r.text
    e2e_finish(job)
    assert job.state == "done" and job.result_status == "completed", (job.message, job.status_reasons)
    assert NEW_T2 in c.get(f"/api/jobs/{job.id}/srt").text


@needs_ffmpeg
def test_stopping_a_queued_revoice_restores_the_finished_job(e2e):
    c, app_mod = e2e.c, e2e.app
    job = e2e_finish(e2e_upload(e2e, _make_media(e2e.tmp)))
    before, message = _job_state(job), job.message
    app_mod._pipeline_semaphore.acquire()            # another job holds the slot
    try:
        r = c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={"turn_edits": {"t0002": {"hi": NEW_T2}}})
        assert r.status_code == 200, r.text
        time.sleep(0.3)
        assert job.state == "queued" and c.get(f"/api/jobs/{job.id}").json()["dialogue_revoice_running"]
        r = c.delete(f"/api/jobs/{job.id}")
        assert r.status_code == 200 and r.json() == {"status": "revoice_cancelled"}
        assert _job_state(job) == before and job.message == message
        job.worker_thread.join(timeout=10)           # it stops waiting for the slot
        assert not job.worker_thread.is_alive()
    finally:
        app_mod._pipeline_semaphore.release()
    assert _job_state(job) == before and job.message == message
    assert app_mod._dialogue_checkpoint(job).is_file()
    assert c.get(f"/api/jobs/{job.id}/result").status_code == 200
    assert c.get(f"/api/jobs/{job.id}").json()["dialogue_revoice_ready"] is True
    assert len(e2e.configs) == 1                     # the re-voice never ran


def test_a_restart_during_a_queued_revoice_puts_the_job_back(api, tmp_path):
    from jobstore import JobStore
    app_mod = api.app
    job = api_finish(api_upload(api))
    before = _job_state(job)
    app_mod._pipeline_semaphore.acquire()
    try:
        assert api.c.post(f"/api/jobs/{job.id}/dialogue/revoice",
                          json={"turn_edits": {"t0001": {"hi": "नया"}}}).status_code == 200
        time.sleep(0.2)
        assert job.state == "queued"
        store = JobStore(tmp_path / "jobs.db")
        store.save(job)                              # what jobs.db holds when the server stops
        loaded = {}
        store.load_all(loaded)
        app_mod._settle_loaded_jobs(loaded, store)
        got = loaded[job.id]
        assert _job_state(got) == before
        assert "restart" in got.message.lower()
        assert app_mod._dialogue_revoice_ready(got) is True
        reloaded = {}
        store.load_all(reloaded)                     # the fix-up was saved
        assert reloaded[job.id].state == "done" and reloaded[job.id].dialogue_revoice_prev is None
        store.close()
        api.c.delete(f"/api/jobs/{job.id}")          # stop the in-process worker of this test
    finally:
        app_mod._pipeline_semaphore.release()
    job.worker_thread.join(timeout=10)


def test_a_job_queued_when_the_server_stopped_is_not_stuck(tmp_path):
    import app as app_mod
    from jobstore import JobStore
    store = JobStore(tmp_path / "jobs.db")
    store.save(app_mod.Job(id="queued000001", state="queued",
                           original_req=app_mod.JobCreateRequest(url="https://example.com/v")))
    loaded = {}
    store.load_all(loaded)
    assert loaded["queued000001"].state == "error" and "restart" in loaded["queued000001"].message.lower()
    store.close()


# ── API: restart, review, preview, output options ────────────────────────────

def test_jobs_with_voice_overrides_survive_a_restart(tmp_path):
    """jobs.db is read while app.py is still being imported (uvicorn app:app)."""
    import app as app_mod
    from jobstore import JobStore
    store = JobStore(tmp_path / "jobs.db")
    req = app_mod.JobCreateRequest(url="https://example.com/v", pipeline_mode="hindi_dialogue",
                                   dialogue_voice_overrides_json=json.dumps(
                                       {"SPEAKER_00": {"category": "female_like"}}))
    store.save(app_mod.Job(id="restart00001", state="done", original_req=req))
    store.close()
    code = ("import app; j = app.JOBS['restart00001']; "
            "print('KEPT' if app._is_dialogue_job(j) else 'LOST', j.original_req.dialogue_voice_overrides_json)")
    env = dict(os.environ, VOICEDUB_STATE_DIR=str(tmp_path), VOICEDUB_WORK=str(tmp_path / "work"))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(BACKEND), env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=300)
    assert "KEPT" in r.stdout and "female_like" in r.stdout, r.stdout + r.stderr[-2000:]


def test_a_stored_request_naming_a_voice_no_longer_offered_is_kept(tmp_path):
    import app as app_mod
    from jobstore import JobStore
    store = JobStore(tmp_path / "jobs.db")
    req = app_mod.JobCreateRequest(url="https://example.com/v", pipeline_mode="hindi_dialogue")
    job = app_mod.Job(id="oldvoice0001", state="done", original_req=req)
    store.save(job)
    # a later version dropped that voice from the curated pool
    raw = json.loads(store._conn.execute("SELECT payload FROM jobs").fetchone()[0])
    raw["original_req"]["dialogue_voice_overrides_json"] = json.dumps(
        {"SPEAKER_00": {"provider": "edge", "voice": "hi-IN-RetiredNeural"}})
    store._conn.execute("UPDATE jobs SET payload = ?", (json.dumps(raw),))
    store._conn.commit()
    loaded = {}
    store.load_all(loaded)
    got = loaded[job.id].original_req
    assert got is not None and app_mod._is_dialogue_job(loaded[job.id])
    assert "RetiredNeural" in got.dialogue_voice_overrides_json
    store.close()


def test_the_review_pause_event_carries_its_state(api):
    job = api_upload(api, dialogue_review="true")
    api_wait_state(job, "review_translation")
    review = [e for e in job.events if e.get("type") == "review"]
    assert review and review[-1]["state"] == "review_translation"
    api.c.post(f"/api/jobs/{job.id}/dialogue/review", json={})
    api_finish(job)


def test_review_packet_of_a_job_stopped_before_voicing(api):
    """Server restarted during the review pause: the before_voice packet is still served."""
    app_mod = api.app
    req = app_mod.JobCreateRequest(url="upload:My Clip.mp4", pipeline_mode="hindi_dialogue",
                                   dialogue_review=True)
    job = app_mod.Job(id="paused000002", state="error", original_req=req,
                      message="Server restarted while the job waited for review")
    app_mod.JOBS[job.id] = job
    work = app_mod.OUTPUTS / job.id / "work"
    (work / "checkpoint").mkdir(parents=True)
    (work / "checkpoint" / "state.json").write_text("{}", encoding="utf-8")
    (work / "review.json").write_text(json.dumps(_packet("before_voice")), encoding="utf-8")
    r = api.c.get(f"/api/jobs/{job.id}/dialogue/review")
    assert r.status_code == 200 and r.json() == _packet("before_voice")
    assert api.c.get(f"/api/jobs/{job.id}").json()["dialogue_revoice_ready"] is True
    # an after-run packet, once there is one, is served first
    out = app_mod.OUTPUTS / job.id / "dialogue_out"
    out.mkdir()
    (out / "review.json").write_text(json.dumps(_packet("after_run")), encoding="utf-8")
    assert api.c.get(f"/api/jobs/{job.id}/dialogue/review").json()["job_stage"] == "after_run"


def test_voice_preview_accepts_child_voices(api, monkeypatch):
    monkeypatch.setattr(_FakeEdge, "made", [])
    monkeypatch.setitem(dialogue_tts.PROVIDER_CLASSES, "edge", _FakeEdge)
    post = lambda **b: api.c.post("/api/dialogue/voice-preview", json=b)
    for voice in ("hi-IN-SwaraNeural", "pt-BR-ThalitaMultilingualNeural", "en-US-EmmaMultilingualNeural"):
        r = post(provider="edge", voice=voice, pitch="+25Hz")
        assert r.status_code == 200, (voice, r.text)
    assert [(b["voice"], b["pitch"]) for _, b in _FakeEdge.made][0] == ("hi-IN-SwaraNeural", "+25Hz")
    assert post(provider="edge", voice="hi-IN-SwaraNeural", pitch="+50Hz").status_code == 400
    assert post(provider="edge", voice="en-US-Nobody", pitch="+25Hz").status_code == 400


def test_revoice_output_options_win_over_module_params(api):
    modules = json.dumps({"params": {"keep_original_audio": True, "container": "mkv"}})
    job = api_finish(api_upload(api, dialogue_modules_json=modules))
    cfg1 = api.fake.calls[0]["cfg"]
    assert (cfg1.keep_original_audio, cfg1.container) == (True, "mkv")
    assert api.c.get(f"/api/jobs/{job.id}").json()["config"]["dialogue_container"] == "mkv"
    r = api.c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={"keep_original_audio": False,
                                                                 "container": "mp4"})
    assert r.status_code == 200, r.text
    api_finish(job)
    cfg2 = api.fake.calls[1]["cfg"]
    assert (cfg2.keep_original_audio, cfg2.container) == (False, "mp4")
    # later re-voices keep them
    assert api.c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={}).status_code == 200
    api_finish(job)
    cfg3 = api.fake.calls[2]["cfg"]
    assert (cfg3.keep_original_audio, cfg3.container) == (False, "mp4")
    config = api.c.get(f"/api/jobs/{job.id}").json()["config"]
    assert (config["dialogue_keep_original_audio"], config["dialogue_container"]) == (False, "mp4")


def test_a_cancel_landing_as_the_run_returns_keeps_the_job_cancelled(api, monkeypatch):
    marked = []
    monkeypatch.setattr(api.app, "_mark_url_completed", marked.append)
    fake = FakeDialogue()

    def run(cfg, on_progress=None, cancel_check=None, components=None, on_legacy_pipeline=None,
            review=None):
        res = fake(cfg, on_progress, cancel_check, components, on_legacy_pipeline, review)
        for j in api.app.JOBS.values():
            j.cancel_event.set()                 # Cancel pressed just as the run returned
        return res
    monkeypatch.setattr(orch, "run_dialogue", run)
    job = api_finish(api_upload(api))
    assert job.state == "error" and job.result_status == "cancelled", (job.state, job.result_status)
    assert job.result_path is None and marked == []
