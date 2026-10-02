"""End to end through the API: review -> edits -> outputs -> re-voice -> cancel.

Everything below the HTTP calls is the merged product code: app.py's
endpoints, job threads and review hook, the real DialogueOrchestrator
(checkpoint, within-job TTS cache, edits, review packets) and the real
mix.mux / FFmpeg, on the synthetic two-speaker clip of test_orchestrator_e2e.

Only the model backends are fake (as in test_app_routing): scripted ASR,
diarization, LLM and Hindi re-ASR, a silent background bed, and a
tone-generating TTS registered under the name "edge" so that the voices
GET /api/dialogue/voices offers are the voices the job binds. These tests
check routing, identity, edits and bookkeeping, not dubbing quality.
"""
import json
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubbing.dialogue import audio
from dubbing.dialogue import orchestrator as orch_mod
from dubbing.dialogue.orchestrator import RESUMED_DETAIL
from dubbing.dialogue.tts import MockProvider

from test_mux_outputs import _probe
from test_orchestrator_e2e import (DURATION, HAVE_FFMPEG, HINDI, SCRIPT, FakeLLM,
                                   _components, _make_media)

pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")

# Stages a re-voice takes from the job's own checkpoint (in run order).
CHECKPOINTED = ["acquire", "extract", "separate", "transcribe", "diarize", "turns", "speakers",
                "translate"]
EN = [en for *_, en in SCRIPT]                       # t0001..t0004, English
EDITED_T1 = "कल रात तुम किसके साथ थे?"               # review: new Hindi for t0001
REVOICED_T2 = "मैं पूरी रात स्टेशन पर थी।"            # re-voice: new Hindi for t0002


class CountingLLM(FakeLLM):
    """FakeLLM that records what it was asked (translate / brief / rewrite)."""

    def __init__(self):
        self.requests = []

    def complete(self, system, user):
        req = json.loads(user)
        self.requests.append("translate" if "turns" in req else
                             "brief" if "transcript" in req else "rewrite")
        return super().complete(system, user)


@pytest.fixture()
def e2e(monkeypatch, tmp_path):
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
    # A fully equipped PC (the real probe reflects this container).
    monkeypatch.setattr(app_mod, "_dialogue_probe", lambda max_age=60.0: FakeProbe())

    calls, holder, configs = {}, {}, []
    tts = MockProvider(name="edge")      # one provider for every run of the job: calls add up
    llm = CountingLLM()

    def fake_components(cfg):
        comps = _components(calls, orch_holder=holder)
        comps.tts_providers = {"edge": tts}
        comps.llm_clients = [llm]

        def silent_background(original, work, policy):
            calls["separate"] = calls.get("separate", 0) + 1
            out = work / "background_estimate.wav"
            audio.run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", str(DURATION),
                              str(out)])
            return {"status": "ok", "background": str(out), "detail": "test bed (silence)"}
        comps.separate = silent_background
        return comps

    real_init = orch_mod.DialogueOrchestrator.__init__

    def init(self, cfg, components=None, on_progress=None, cancel_check=None, **kw):
        real_init(self, cfg, components, on_progress, cancel_check, **kw)
        holder["orch"] = self            # the fake Hindi re-ASR reads this run's turns
        configs.append(cfg)

    monkeypatch.setattr(orch_mod, "default_components", fake_components)
    monkeypatch.setattr(orch_mod.DialogueOrchestrator, "__init__", init)
    ns = SimpleNamespace(c=TestClient(app_mod.app), app=app_mod, tts=tts, llm=llm, calls=calls,
                         configs=configs, tmp=tmp_path)
    yield ns
    # Never leave a worker holding the (module-wide) pipeline slot behind.
    for job in list(app_mod.JOBS.values()):
        worker = job.worker_thread
        if worker is not None and worker.is_alive():
            job.cancel_event.set()
            job.pause_event.set()
            worker.join(timeout=60)


# ── helpers ──────────────────────────────────────────────────────────────────

def _upload(e2e, media, **form):
    with open(media, "rb") as f:
        r = e2e.c.post("/api/jobs/upload", files={"file": ("Two Friends.mp4", f, "video/mp4")},
                       data={"pipeline_mode": "hindi_dialogue", **form})
    assert r.status_code == 200, r.text
    return e2e.app.JOBS[r.json()["id"]]


def _wait_state(job, state, timeout=180.0):
    end = time.time() + timeout
    while job.state != state and time.time() < end:
        if job.state == "error" or (job.worker_thread is not None and not job.worker_thread.is_alive()):
            break
        time.sleep(0.05)
    assert job.state == state, (job.state, job.message, job.error, job.status_reasons)


def _finish(job, timeout=300.0):
    job.worker_thread.join(timeout=timeout)
    assert not job.worker_thread.is_alive(), f"job stuck: {job.state} {job.message}"
    return job


def _subtitle_text(video: Path, index: int, tmp: Path) -> str:
    """The text of the video's index-th subtitle stream (FFmpeg converts it to SRT)."""
    out = tmp / f"{video.stem}_{video.suffix[1:]}_s{index}.srt"
    audio.run_ffmpeg(["-i", str(video), "-map", f"0:s:{index}", "-f", "srt", str(out)])
    return out.read_text(encoding="utf-8")


def _rms(video: Path, audio_index: int, start: float, dur: float, tmp: Path) -> float:
    """RMS of one audio stream of the video over [start, start + dur]."""
    import numpy as np
    out = tmp / f"{video.stem}_a{audio_index}_{start:.2f}.wav"
    audio.run_ffmpeg(["-i", str(video), "-map", f"0:a:{audio_index}", "-ss", f"{start:.3f}",
                      "-t", f"{dur:.3f}", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out)])
    data, _ = audio.read_wav(out)
    return float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0


def _stage(rep, name):
    return next(s for s in rep["stages"] if s["name"] == name)


def _accepted_clips(out_dir: Path):
    clips = [x for x in json.loads((out_dir / "clips.json").read_text(encoding="utf-8")) if x["accepted"]]
    for x in clips:   # never truncated: only sped up, within the bound
        assert x["stretch"] <= 1.15 + 1e-3, x
        assert abs(x["final_duration"] * x["stretch"] - x["natural_duration"]) < 0.06, x
    return clips


def _timing(packet):
    return [(t["turn_id"], t["start"], t["end"]) for t in packet["turns"]]


# ── the whole flow ───────────────────────────────────────────────────────────

def test_review_edits_outputs_and_revoice_through_the_api(e2e):
    c, app_mod, tts = e2e.c, e2e.app, e2e.tts
    media = _make_media(e2e.tmp)

    # 1. a hindi_dialogue upload with review, the original audio kept, English subtitles on
    job = _upload(e2e, media, dialogue_review="true", dialogue_keep_original_audio="true",
                  dialogue_english_subtitles="true")

    # 2. it pauses after translation with the review packet of the script
    _wait_state(job, "review_translation")
    assert tts.calls == []                               # nothing voiced before the review
    cfg1 = e2e.configs[0]
    assert (cfg1.review_before_voice, cfg1.keep_original_audio, cfg1.english_subtitles,
            cfg1.container, cfg1.resume) == (True, True, True, "mp4", False)
    status = c.get(f"/api/jobs/{job.id}").json()
    assert status["state"] == "review_translation"
    assert status["dialogue_review"]["job_stage"] == "before_voice"
    packet = c.get(f"/api/jobs/{job.id}/dialogue/review").json()
    assert packet["job_stage"] == "before_voice"
    assert [(t["turn_id"], t["speaker_id"], t["english"], t["hindi"]) for t in packet["turns"]] == \
        [(f"t{i:04d}", spk, en, HINDI[en]) for i, (spk, _, _, _, en) in enumerate(SCRIPT, 1)]
    for t, (_, _, s, e, _) in zip(packet["turns"], SCRIPT):
        assert abs(t["start"] - s) < 0.06 and abs(t["end"] - e) < 0.06    # the source timing
        assert t["required"] is True and t["clip"] is None and t["overflow_s"] is None
    spk = {s["speaker_id"]: s for s in packet["speakers"]}
    assert set(spk) == {"SPEAKER_00", "SPEAKER_01"}
    assert (spk["SPEAKER_00"]["voice_category"], spk["SPEAKER_01"]["voice_category"]) == \
        ("male_like", "female_like")
    assert spk["SPEAKER_00"]["turns"] == spk["SPEAKER_01"]["turns"] == 2
    assert {s["provider"] for s in spk.values()} == {"edge"}
    assert not any(s["override"] for s in spk.values())
    assert spk["SPEAKER_00"]["voice"] != spk["SPEAKER_01"]["voice"]  # one voice per speaker
    assert packet["media_duration"] == pytest.approx(DURATION, abs=0.3)
    assert [s["turn_id"] for s in c.get(f"/api/jobs/{job.id}/transcript").json()["segments"]] == \
        ["t0001", "t0002", "t0003", "t0004"]

    # 3. edit one Hindi line, delete one turn, give SPEAKER_01 another offered voice
    voices = c.get("/api/dialogue/voices").json()
    assert "mock" not in voices
    assert packet["voice_options"]["edge"] == {k: voices["edge"][k] for k in ("male_like", "female_like")}
    taken = {(s["voice"], s["pitch"]) for s in spk.values()}
    pick = next(o for o in voices["edge"]["female_like"]
                if o["pitch"] and (o["voice"], o["pitch"]) not in taken)
    edits = {"turn_edits": {"t0001": {"hi": EDITED_T1}, "t0003": {"delete": True}},
             "voice_overrides": {"SPEAKER_01": {"provider": "edge", "voice": pick["voice"],
                                                "pitch": pick["pitch"]}}}
    r = c.post(f"/api/jobs/{job.id}/dialogue/review", json=edits)
    assert r.status_code == 200 and r.json() == {"status": "resumed"}, r.text

    # 4. the dub: honest status, every edit in every output
    _finish(job)
    assert job.state == "done" and job.result_status == "completed", (job.message, job.status_reasons)
    assert job.dialogue_review is None and job.dialogue_edits == edits
    status = c.get(f"/api/jobs/{job.id}").json()
    assert (status["state"], status["result_status"], status["status_reasons"]) == ("done", "completed", [])
    assert status["dialogue_review"] is None and status["dialogue_revoice_ready"] is True
    rep = c.get(f"/api/jobs/{job.id}/report?fmt=json").json()
    assert rep["final_status"] == "completed" and rep["status_reasons"] == []
    assert rep["required_turn_ids"] == rep["generated_turn_ids"] == ["t0001", "t0002", "t0004"]
    assert rep["missing_turns"] == [] and rep["identity_violations"] == []
    assert _stage(rep, "review")["detail"].startswith("3 edit(s) applied")
    assert sorted(e["kind"] for e in rep["applied_edits"] if not e.get("ignored")) == \
        ["turn_edit", "turn_edit", "voice_override"]

    # exactly the kept lines were voiced, each in its speaker's (overridden) voice
    s0 = (spk["SPEAKER_00"]["voice"], spk["SPEAKER_00"]["pitch"])
    s1 = (pick["voice"], pick["pitch"])
    assert sorted((x["text"], x["voice"], x["pitch"]) for x in tts.calls) == sorted([
        (EDITED_T1, *s0), (HINDI[EN[1]], *s1), (HINDI[EN[3]], *s1)])
    speakers = {s["speaker_id"]: s for s in rep["speakers"]}
    b1 = speakers["SPEAKER_01"]["provider_voices"]["edge"]
    assert (b1["voice"], b1.get("pitch"), b1.get("override")) == (*s1, True)
    assert speakers["SPEAKER_01"]["mapping_origin"] == "user"
    out_dir = app_mod.OUTPUTS / job.id / "dialogue_out"
    clips = _accepted_clips(out_dir)
    assert {x["turn_id"]: (x["voice"], x["voice_params"]["pitch"]) for x in clips} == \
        {"t0001": s0, "t0002": s1, "t0004": s1}
    card = {s["speaker"]: s["voice"] for s in status["speakers"]}
    assert card["SPEAKER_01"] == f"{pick['voice']}|{pick['pitch']}"

    # the video: Hindi default + original second audio, Hindi + English subtitle streams
    assert job.result_path.name == "dubbed_hi.mp4" and job.saved_video.endswith("dubbed_hi.mp4")
    saved_video = Path(job.saved_video)
    assert saved_video.is_file() and Path(job.saved_folder).name.endswith(
        f"[HI Dialogue completed] ({job.id})")
    streams, fmt = _probe(saved_video)
    assert "mp4" in fmt
    assert [s["type"] for s in streams] == ["video", "audio", "audio", "subtitle", "subtitle"]
    assert [(s["codec"], s["lang"], s["default"]) for s in streams[1:]] == [
        ("aac", "hin", True), ("aac", "eng", False), ("mov_text", "hin", True), ("mov_text", "eng", False)]
    assert streams[1]["sample_rate"] == 48000                           # the Hindi mix
    assert streams[2]["sample_rate"] == 16000                           # the source's own audio
    assert streams[2]["tags"].get("handler_name") == "Original"
    info = audio.probe_streams(saved_video)
    assert abs(info["duration"] - DURATION) < 0.3                       # source timing, nothing cut
    # English is never the Hindi track's background: the deleted line's slot
    # (t0003, 5.0-5.6 s) is silent there but speech in the original track.
    hindi_gap = _rms(saved_video, 0, 5.05, 0.5, e2e.tmp)
    original_gap = _rms(saved_video, 1, 5.05, 0.5, e2e.tmp)
    assert original_gap > 0.02 and hindi_gap < 0.05 * original_gap, (hindi_gap, original_gap)
    hi_track = _subtitle_text(saved_video, 0, e2e.tmp)
    en_track = _subtitle_text(saved_video, 1, e2e.tmp)
    assert EDITED_T1 in hi_track and HINDI[EN[0]] not in hi_track and HINDI[EN[2]] not in hi_track
    assert EN[0] in en_track and EN[2] not in en_track
    srt = c.get(f"/api/jobs/{job.id}/srt").text
    assert srt.count("-->") == 3 and EDITED_T1 in srt
    assert HINDI[EN[0]] not in srt and HINDI[EN[2]] not in srt          # edited away / deleted
    en_srt = (out_dir / "subtitles_en.srt").read_text(encoding="utf-8")
    assert en_srt.count("-->") == 3 and EN[2] not in en_srt

    # 5. after the run: the finished packet with clips to listen to
    after = c.get(f"/api/jobs/{job.id}/dialogue/review").json()
    assert after["job_stage"] == "after_run"
    assert _timing(after) == _timing(packet)            # source timing never changes
    by_id = {t["turn_id"]: t for t in after["turns"]}
    assert by_id["t0001"]["hindi"] == EDITED_T1 and "edited" in by_id["t0001"]["flags"]
    assert by_id["t0003"]["required"] is False and "deleted_by_user" in by_id["t0003"]["flags"]
    assert by_id["t0003"]["clip"] is None
    assert {tid: t["clip"] for tid, t in by_id.items() if tid != "t0003"} == \
        {"t0001": "t0001.wav", "t0002": "t0002.wav", "t0004": "t0004.wav"}
    after_spk = {s["speaker_id"]: s for s in after["speakers"]}
    assert (after_spk["SPEAKER_01"]["voice"], after_spk["SPEAKER_01"]["pitch"]) == s1
    assert after_spk["SPEAKER_01"]["override"] is True and after_spk["SPEAKER_00"]["override"] is False
    r = c.get(f"/api/jobs/{job.id}/dialogue/clip/t0001.wav")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
    got = e2e.tmp / "t0001_from_api.wav"
    got.write_bytes(r.content)
    with wave.open(str(got), "rb") as wf:
        assert wf.getnframes() / wf.getframerate() > 0.3
    ref = after_spk["SPEAKER_00"]["reference_clip"]
    assert ref and c.get(f"/api/jobs/{job.id}/dialogue/clip/{ref}").status_code == 200
    assert c.get(f"/api/jobs/{job.id}/dialogue/clip/t0003.wav").status_code == 404   # deleted: no clip
    for bad in ("report.json", "t0001.wav.txt", "t 1.wav", "..%5Chindi_mix.wav"):
        assert c.get(f"/api/jobs/{job.id}/dialogue/clip/{bad}").status_code == 400, bad
    for traversal in ("..%2Fhindi_mix.wav", "..%2F..%2Fwork%2Foriginal_48k.wav", "../hindi_mix.wav",
                      "%2E%2E%2Fhindi_mix.wav", "%2Fetc%2Fpasswd.wav"):
        r = c.get(f"/api/jobs/{job.id}/dialogue/clip/{traversal}")
        assert r.status_code in (400, 404) and r.content[:4] != b"RIFF", traversal

    # 6. re-voice: one more Hindi edit, MKV this time
    calls_before, llm_before = len(tts.calls), len(e2e.llm.requests)
    first_saved_folder = Path(job.saved_folder)
    r = c.post(f"/api/jobs/{job.id}/dialogue/revoice",
               json={"turn_edits": {"t0002": {"hi": REVOICED_T2}}, "container": "mkv"})
    assert r.status_code == 200 and r.json() == {"status": "started"}, r.text
    _finish(job)
    assert job.state == "done" and job.result_status == "completed", (job.message, job.status_reasons)
    cfg2 = e2e.configs[-1]
    assert cfg2.resume is True and cfg2.review_before_voice is False and cfg2.container == "mkv"
    assert (cfg2.work_dir, cfg2.output_dir) == (cfg1.work_dir, cfg1.output_dir)   # this job's own
    rep2 = c.get(f"/api/jobs/{job.id}/report?fmt=json").json()
    assert rep2["final_status"] == "completed" and rep2["status_reasons"] == []
    # everything up to translation came from the checkpoint: no ASR, diarization,
    # separation or translation call was made again
    assert rep2["resumed_from_checkpoint"] == CHECKPOINTED
    for name in CHECKPOINTED:
        st = _stage(rep2, name)
        assert (st["status"], st["detail"]) == ("skipped", RESUMED_DETAIL), name
    assert e2e.calls == {"asr": 1, "diarize": 1, "separate": 1}
    assert [x for x in e2e.llm.requests[llm_before:] if x != "rewrite"] == []
    # only the changed line was voiced again, in SPEAKER_01's chosen voice
    assert [(x["text"], x["voice"], x["pitch"]) for x in tts.calls[calls_before:]] == [(REVOICED_T2, *s1)]
    assert _stage(rep2, "synthesize")["data"]["tts_cache_hits"] == 2
    assert rep2["required_turn_ids"] == rep2["generated_turn_ids"] == ["t0001", "t0002", "t0004"]
    assert rep2["identity_violations"] == [] and rep2["missing_turns"] == []
    # earlier edits still hold
    srt2 = c.get(f"/api/jobs/{job.id}/srt").text
    assert EDITED_T1 in srt2 and REVOICED_T2 in srt2 and HINDI[EN[2]] not in srt2
    assert HINDI[EN[1]] not in srt2 and srt2.count("-->") == 3
    after2_packet = c.get(f"/api/jobs/{job.id}/dialogue/review").json()
    assert after2_packet["job_stage"] == "after_run" and _timing(after2_packet) == _timing(packet)
    after2 = {t["turn_id"]: t for t in after2_packet["turns"]}
    assert after2["t0002"]["hindi"] == REVOICED_T2 and after2["t0001"]["hindi"] == EDITED_T1
    assert "deleted_by_user" in after2["t0003"]["flags"] and after2["t0003"]["clip"] is None
    clips2 = _accepted_clips(out_dir)
    assert {x["turn_id"]: (x["voice"], x["voice_params"]["pitch"]) for x in clips2} == \
        {"t0001": s0, "t0002": s1, "t0004": s1}
    assert job.dialogue_edits["turn_edits"] == {"t0001": {"hi": EDITED_T1}, "t0002": {"hi": REVOICED_T2},
                                                "t0003": {"delete": True}}
    # the MKV is the job's result and its saved video; the old MP4 is gone
    assert job.result_path.name == "dubbed_hi.mkv" and job.result_path.is_file()
    assert not (out_dir / "dubbed_hi.mp4").exists()
    assert job.saved_video.endswith("dubbed_hi.mkv") and Path(job.saved_video).is_file()
    # one refreshed saved copy of this job (replaced, not merged with the first run's)
    assert [p.name for p in app_mod.SAVED_DIR.iterdir() if p.name.endswith(f"({job.id})")] == \
        [first_saved_folder.name] == [Path(job.saved_folder).name]
    assert not list(Path(job.saved_folder).glob("dubbed_hi.mp4"))
    status = c.get(f"/api/jobs/{job.id}").json()
    assert status["saved_video"] == job.saved_video and status["result_status"] == "completed"
    r = c.get(f"/api/jobs/{job.id}/result")
    assert r.status_code == 200 and r.headers["content-type"] == "video/x-matroska"
    assert ".mkv" in r.headers["content-disposition"]
    streams2, fmt2 = _probe(Path(job.saved_video))
    assert "matroska" in fmt2
    assert [(s["type"], s["codec"], s["lang"], s["default"]) for s in streams2[1:]] == [
        ("audio", "aac", "hin", True), ("audio", "aac", "eng", False), ("subtitle", "subrip", "hin", True),
        ("subtitle", "subrip", "eng", False)]
    assert streams2[0]["type"] == "video" and streams2[2]["sample_rate"] == 16000
    assert abs(audio.probe_streams(Path(job.saved_video))["duration"] - DURATION) < 0.3
    mkv_hi = _subtitle_text(Path(job.saved_video), 0, e2e.tmp)
    assert REVOICED_T2 in mkv_hi and EDITED_T1 in mkv_hi and HINDI[EN[2]] not in mkv_hi
    mkv_en = _subtitle_text(Path(job.saved_video), 1, e2e.tmp)
    assert EN[1] in mkv_en and EN[2] not in mkv_en
    assert [e for e in job.events if e.get("type") == "complete"] == \
        [{"type": "complete", "state": "done", "result_status": "completed"}]


# ── cancel while paused for review ───────────────────────────────────────────

def test_cancel_while_paused_for_review_stops_the_job(e2e):
    c, app_mod = e2e.c, e2e.app
    job = _upload(e2e, _make_media(e2e.tmp), dialogue_review="true")
    _wait_state(job, "review_translation")
    assert c.delete(f"/api/jobs/{job.id}").json() == {"status": "cancelled"}
    _finish(job, timeout=30)                            # not stuck waiting for a review
    assert job.state == "error" and job.result_status == "cancelled", (job.state, job.result_status)
    assert e2e.tts.calls == []                          # nothing was voiced
    assert job.dialogue_review is None
    # honest about what is left: no dubbed video, a saved copy (if any) says cancelled
    assert job.result_path is None and job.saved_video is None
    assert c.get(f"/api/jobs/{job.id}/result").status_code == 409
    if job.saved_folder:
        assert Path(job.saved_folder).name.endswith(f"[HI Dialogue cancelled] ({job.id})")
    # the pipeline slot the paused job held is free again
    assert app_mod._pipeline_semaphore.acquire(timeout=5)
    app_mod._pipeline_semaphore.release()
    status = c.get(f"/api/jobs/{job.id}").json()
    assert status["state"] == "error" and status["result_status"] == "cancelled"
    assert status["dialogue_review"] is None and status["dialogue_revoice_ready"] is False
    assert c.post(f"/api/jobs/{job.id}/dialogue/review", json={}).status_code == 409
    assert c.post(f"/api/jobs/{job.id}/continue").status_code == 400
    assert c.post(f"/api/jobs/{job.id}/dialogue/revoice", json={}).status_code == 409
    rep = c.get(f"/api/jobs/{job.id}/report?fmt=json")
    if rep.status_code == 200:                          # the folder may already be cleaned up
        assert rep.json()["final_status"] == "cancelled"
    end = time.time() + 15
    while (app_mod.OUTPUTS / job.id).exists() and time.time() < end:
        time.sleep(0.1)
    assert not (app_mod.OUTPUTS / job.id).exists()      # the cancel's cleanup ran


# ── a second job of the same video starts from nothing ───────────────────────

def test_a_second_job_reuses_nothing_and_default_options_apply(e2e):
    c, tts = e2e.c, e2e.tts
    media = _make_media(e2e.tmp)
    first = _finish(_upload(e2e, media))
    second = _finish(_upload(e2e, media))
    for job in (first, second):
        assert job.state == "done" and job.result_status == "completed", (job.message, job.status_reasons)
    # every stage ran again for the second job: nothing came from the first one
    assert e2e.calls == {"asr": 2, "diarize": 2, "separate": 2}
    assert e2e.llm.requests.count("translate") == 2
    assert sorted(x["text"] for x in tts.calls) == sorted([HINDI[en] for en in EN] * 2)
    rep = c.get(f"/api/jobs/{second.id}/report?fmt=json").json()
    assert rep["resumed_from_checkpoint"] == [] and "tts_cache_hits" not in _stage(rep, "synthesize")["data"]
    assert not any(s["status"] == "skipped" for s in rep["stages"] if s["name"] in CHECKPOINTED)
    cfg_a, cfg_b = e2e.configs
    assert cfg_a.work_dir != cfg_b.work_dir and cfg_a.output_dir != cfg_b.output_dir
    # defaults: no review pause, no original audio track, mp4, nothing burned in;
    # English subtitles are on by default (docs/dialogue/README.md)
    assert (cfg_b.review_before_voice, cfg_b.keep_original_audio, cfg_b.burn_subtitles,
            cfg_b.container, cfg_b.resume, cfg_b.turn_edits, cfg_b.voice_overrides) == \
        (False, False, False, "mp4", False, {}, {})
    assert not any(e.get("type") == "review" for e in second.events)
    streams, fmt = _probe(Path(second.saved_video))
    assert "mp4" in fmt and second.saved_video.endswith("dubbed_hi.mp4")
    assert [(s["type"], s["lang"]) for s in streams[1:]] == [("audio", "hin"), ("subtitle", "hin"),
                                                             ("subtitle", "eng")]
