"""Classic pipeline + API fixes: numbered LLM replies are read back by number
(a skipped line never shifts the batch), split jobs keep every part's status
and subtitles, an uploaded transcript is never dropped by split mode, the
English source lock, and what the job page reads for dialogue jobs
(Characters card, title, subtitles, inline report)."""
import json
import sys
import types
from pathlib import Path

import pytest

import pipeline as pipeline_mod
from pipeline import Pipeline, PipelineConfig, _check_numbered_reply

HI = ["नमस्ते।", "तुम कहाँ जा रहे हो?", "मेरा इंतज़ार करो।"]


@pytest.fixture()
def pipe(tmp_path):
    cfg = PipelineConfig(source="x.mp4", work_dir=tmp_path / "w", output_path=tmp_path / "o.mp4",
                         tts_voice="hi-IN-MadhurNeural", translation_engine="groq")
    return Pipeline(cfg)


def _segs():
    return [{"start": float(i), "end": i + 0.9, "text": t}
            for i, t in enumerate(["Hello there.", "Where are you going?", "Wait for me."])]


class _FakeGroq:
    """groq.Groq stand-in answering each request from a list (str or exception)."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

        def create(**kw):
            self.calls += 1
            r = self.replies.pop(0)
            if isinstance(r, Exception):
                raise r
            msg = types.SimpleNamespace(content=r)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))


def _install_groq(monkeypatch, fake):
    mod = types.ModuleType("groq")
    mod.Groq = lambda api_key=None: fake
    monkeypatch.setitem(sys.modules, "groq", mod)
    monkeypatch.setattr(pipeline_mod.time, "sleep", lambda s: None)


# ── numbered replies ─────────────────────────────────────────────────────────

def test_numbered_reply_is_read_back_by_number():
    # line 2 merged away: the old in-order parser put line 3 on segment 2
    assert Pipeline._parse_numbered_translations("1. a\n3. c", 3) == ["a", "", "c"]
    assert Pipeline._parse_numbered_translations("2) b\n1: a\n3- c", 3) == ["a", "b", "c"]
    # echoed [hint] tags dropped; the first of a repeated number wins
    assert Pipeline._parse_numbered_translations("1. [12w] a\n1. x\n2. b\n9. z", 2) == ["a", "b"]


def test_numbered_reply_problems_and_trust():
    assert _check_numbered_reply("1. a\n2. b", 2)[1:] == ("", True)
    assert _check_numbered_reply("1. a\n3. c", 3)[1:] == ("missing 2", True)
    # a split line renumbers the rest: none of its lines can be placed
    assert _check_numbered_reply("1. a\n2. b1\n3. b2\n4. c", 3)[1:] == ("unexpected 4", False)
    assert _check_numbered_reply("1. a\n1. b", 2)[1:] == ("missing 2; repeated 1", False)
    # a symbol-only source line ("♪") may come back empty
    assert _check_numbered_reply("1. a\n2.", 2, ["Hi.", "♪♪"])[1] == ""


def test_groq_rerequests_a_batch_that_skips_a_line(pipe, monkeypatch):
    fake = _FakeGroq([f"1. {HI[0]}\n3. {HI[2]}", "\n".join(f"{i}. {t}" for i, t in enumerate(HI, 1))])
    _install_groq(monkeypatch, fake)
    monkeypatch.setattr(pipe, "_translate_single_fallback", lambda t: pytest.fail("no fallback"))
    segs = _segs()
    pipe._translate_segments_groq(segs, "k")
    assert fake.calls == 2
    assert [s["text_translated"] for s in segs] == HI
    assert pipe.result_warnings == []


def test_groq_translates_only_the_missing_line_after_retries(pipe, monkeypatch):
    fake = _FakeGroq([f"1. {HI[0]}\n3. {HI[2]}"] * 3)
    _install_groq(monkeypatch, fake)
    asked = []
    monkeypatch.setattr(pipe, "_translate_single_fallback", lambda t: asked.append(t) or HI[1])
    segs = _segs()
    pipe._translate_segments_groq(segs, "k")
    assert fake.calls == 3 and asked == ["Where are you going?"]
    assert [s["text_translated"] for s in segs] == HI       # line 3 stays on segment 3
    assert any("one by one" in w for w in pipe.result_warnings)


def test_line_still_english_is_reported_not_passed_off_as_hindi(pipe, monkeypatch):
    _install_groq(monkeypatch, _FakeGroq([RuntimeError("503 unavailable")] * 3))
    monkeypatch.setattr(pipe, "_translate_single_fallback",
                        lambda t: HI[0] if t == "Hello there." else t)   # fallback fails twice
    segs = _segs()
    pipe._translate_segments_groq(segs, "k")
    assert segs[0]["text_translated"] == HI[0]
    english = [w for w in pipe.result_warnings if "left in ENGLISH" in w]
    assert len(english) == 2 and "Where are you going?" in english[0]
    assert [s["text"] for s in pipe._english_left] == ["Where are you going?", "Wait for me."]


def test_translation_fails_loudly_when_most_lines_stay_english(pipe, monkeypatch):
    _install_groq(monkeypatch, _FakeGroq([RuntimeError("401 invalid key")] * 3))
    monkeypatch.setattr(pipeline_mod, "get_groq_key", lambda: "k")
    monkeypatch.setattr(pipe, "_translate_single_fallback", lambda t: t)
    with pytest.raises(RuntimeError, match="Translation failed for 3/3 lines"):
        pipe._translate_segments(_segs())


def test_sambanova_retry_keeps_its_own_key(pipe, monkeypatch):
    import requests
    replies = [f"1. {HI[0]}", f"1. {HI[0]}", "\n".join(f"{i}. {t}" for i, t in enumerate(HI, 1))]
    keys = []

    class _Resp:
        status_code = 200

        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": self.text}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        keys.append(headers["Authorization"])
        return _Resp(replies.pop(0))
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(pipeline_mod.time, "sleep", lambda s: None)
    url, model = Pipeline.TURBO_ENGINE_CONFIG["SambaNova"]
    assert pipe._translate_batch_openai_compat(_segs(), url, "samba", model, "SambaNova") == HI
    assert keys == ["Bearer samba"] * 3      # never swapped for a Groq key


def test_turbo_retries_an_empty_reply_batch_with_its_other_engine(pipe, monkeypatch):
    # gpt-oss can answer with no content at all: that is no partial reply, so
    # Turbo must hand the batch to SambaNova, not voice it line by line.
    import requests
    groq_url = Pipeline.TURBO_ENGINE_CONFIG["Groq"][0]
    asked = []

    class _Resp:
        status_code = 200

        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": self.text}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        asked.append("groq" if url == groq_url else "samba")
        return _Resp(None if url == groq_url else "\n".join(f"{i}. {t}" for i, t in enumerate(HI, 1)))
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(pipeline_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(pipeline_mod, "get_groq_key", lambda: "")
    monkeypatch.setattr(pipe, "_translate_single_fallback", lambda t: pytest.fail("no fallback"))
    segs = _segs()
    pipe._translate_segments_turbo(segs, [("Groq", "g"), ("SambaNova", "s")])
    assert asked == ["groq"] * 3 + ["samba"]
    assert [s["text_translated"] for s in segs] == HI and pipe.result_warnings == []


# ── API / job routing ────────────────────────────────────────────────────────

@pytest.fixture()
def app_mod(monkeypatch, tmp_path):
    import app as app_mod

    class _NoStore:
        def save(self, job):
            pass

        def delete(self, job_id):
            pass
    (tmp_path / "saved").mkdir()
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(app_mod, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(app_mod, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(app_mod, "_store", _NoStore())
    monkeypatch.setattr(app_mod, "_purge_global_caches", lambda: 0)
    return app_mod


class _FakeLegacyPipeline:
    """Classic Pipeline stand-in: writes the outputs a real run leaves behind."""
    runs = []

    def __init__(self, cfg, on_progress=None, cancel_check=None, pause_event=None):
        self.cfg, self.video_title, self._ffmpeg, self.paused_at = cfg, "Long Video", "ffmpeg", None
        self.result_status, self.result_warnings = "completed", []
        self.speaker_summary, self.speaker_warning, self._tts_budget = [], None, None
        self.segments, self.qa_score = [], None

    def _ensure_ffmpeg(self):
        pass

    def _ingest_source(self, url):
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        p = self.cfg.work_dir / "source.mp4"
        p.write_bytes(b"v")
        return p

    def _write_outputs(self, srt_text):
        self.cfg.output_path.write_bytes(b"dub")
        (self.cfg.output_path.parent / f"subtitles_{self.cfg.target_language}.srt").write_text(
            srt_text, encoding="utf-8")

    def run(self):
        self.runs.append(self.cfg.source)
        self._write_outputs(f"part {len(self.runs)}")
        if len(self.runs) == 2:
            self.result_status = "draft_incomplete"
            self.result_warnings = ["1 of 9 segments have NO dubbed audio"]


def test_auto_source_language_means_english(app_mod):
    assert [app_mod._english_source(x) for x in ("auto", "", None, "en", "hi")] == \
        ["en", "en", "en", "en", "hi"]


def test_split_job_keeps_each_parts_outcome_and_subtitles(app_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(_FakeLegacyPipeline, "runs", [])
    monkeypatch.setattr(app_mod, "Pipeline", _FakeLegacyPipeline)
    monkeypatch.setattr(app_mod, "_split_video",
                        lambda ff, video, mins, out: [out / "part_01.mp4", out / "part_02.mp4"])
    job = app_mod.Job(id="splitjob0001")
    (app_mod.OUTPUTS / job.id).mkdir()
    app_mod._run_job_split(job, app_mod.JobCreateRequest(url="https://example.com/v"), "hi-IN-MadhurNeural")
    saved = tmp_path / "saved"
    srts = [saved / f"Long Video - Part {n}" / f"Long Video - Part {n}.srt" for n in (1, 2)]
    assert [p.read_text(encoding="utf-8") for p in srts] == ["part 1", "part 2"]
    assert job.subtitles_path == str(srts[0])
    assert job.state == "done" and job.result_status == "draft_incomplete"
    assert job.status_reasons == ["Part 2: 1 of 9 segments have NO dubbed audio"]
    assert "DRAFT" in job.message


def test_uploaded_transcript_is_not_dropped_by_split_mode(app_mod, monkeypatch):
    from metrics import NoOpMetrics
    seen = {}

    class FakePipeline(_FakeLegacyPipeline):
        def run(self):
            pytest.fail("transcription must be skipped when a transcript is supplied")

        def run_from_source_srt(self, path):
            seen["source_language"] = self.cfg.source_language
            seen["transcript"] = Path(path).read_text(encoding="utf-8")
            self._write_outputs("hindi srt")

    monkeypatch.setattr(app_mod, "Pipeline", FakePipeline)
    monkeypatch.setattr(app_mod, "_run_job_split", lambda *a: pytest.fail("split mode must not run"))
    monkeypatch.setattr(app_mod, "get_metrics", lambda: NoOpMetrics())
    monkeypatch.setattr(app_mod, "_generate_youtube_description", lambda job: "desc")
    srt = "1\n00:00:00,000 --> 00:00:02,000\nHello there.\n"
    req = app_mod.JobCreateRequest(url="https://example.com/v", source_language="auto",
                                   split_duration=30, transcript_srt_content=srt)
    job = app_mod.Job(id="transcript001")
    app_mod._run_job(job, req)
    assert job.state == "done", job.error
    assert seen == {"source_language": "en", "transcript": srt.strip()}
    assert any("split_duration=30 ignored" in e.get("message", "") for e in job.events)
    assert Path(job.subtitles_path).read_text(encoding="utf-8") == "hindi srt"   # saved folder


def test_uploaded_video_job_keeps_the_transcript(app_mod, monkeypatch):
    # The UI sends the transcript with the video file; the upload endpoint
    # used to drop the field, so the video was transcribed anyway.
    from fastapi.testclient import TestClient
    started = []
    monkeypatch.setattr(app_mod, "JOBS", {})
    monkeypatch.setattr(app_mod, "_run_job", lambda job, req: started.append(req))
    srt = "1\n00:00:00,000 --> 00:00:02,000\nHello there.\n"
    r = TestClient(app_mod.app).post(
        "/api/jobs/upload", files={"file": ("clip.mp4", b"video", "video/mp4")},
        data={"transcript_srt_content": srt, "source_language": "auto"})
    assert r.status_code == 200, r.text
    app_mod.JOBS[r.json()["id"]].worker_thread.join(timeout=10)
    assert started[0].transcript_srt_content == srt and started[0].source_language == "en"


def test_cancelled_resume_keeps_the_job_folder_until_its_worker_exits(app_mod, monkeypatch):
    import threading
    import time as _time
    from fastapi.testclient import TestClient
    release = threading.Event()
    monkeypatch.setattr(app_mod, "_run_resume", lambda job: release.wait(10))
    job = app_mod.Job(id="resume000001", state="waiting_for_srt")
    job.worker_thread = threading.Thread(target=lambda: None)   # the finished phase-1 worker
    job.worker_thread.start()
    job.worker_thread.join()
    job_dir = app_mod.OUTPUTS / job.id
    (job_dir / "work").mkdir(parents=True)
    monkeypatch.setitem(app_mod.JOBS, job.id, job)
    c = TestClient(app_mod.app)
    r = c.post(f"/api/jobs/{job.id}/resume-with-srt",
               files={"file": ("hi.srt", "1\n00:00:00,000 --> 00:00:01,000\nनमस्ते\n".encode(), "text/plain")})
    assert r.status_code == 200, r.text
    assert c.delete(f"/api/jobs/{job.id}").json() == {"status": "cancelled"}
    _time.sleep(0.5)
    assert job_dir.exists()                      # the resume worker is still running
    release.set()
    for _ in range(100):
        if not job_dir.exists():
            break
        _time.sleep(0.05)
    assert not job_dir.exists()                  # removed once it exited


def _speakers_json(tmp_path, speakers):
    p = tmp_path / "speakers.json"
    p.write_text(json.dumps({"speakers": speakers}), encoding="utf-8")
    return p


def test_dialogue_speakers_match_the_classic_characters_card(app_mod, tmp_path):
    from dubbing.dialogue.speaker_registry import SpeakerRegistry, ensure_unknown_speaker
    reg = SpeakerRegistry()                      # the real speakers.json format
    reg.register("SPEAKER_00", voice_category="male_like", total_speech_s=41.26)
    reg.register("SPEAKER_01", voice_category="female_like", total_speech_s=12.0)
    ensure_unknown_speaker(reg)
    reg.bind_provider("edge")
    reg.save(tmp_path / "speakers.json")
    rows = app_mod._dialogue_speakers(tmp_path / "speakers.json", ["edge"])
    assert rows == [
        {"speaker": "SPEAKER_00", "gender": "male", "voice": "hi-IN-MadhurNeural",
         "voice_label": "Madhur", "seconds": 41.3, "reused": False},
        {"speaker": "SPEAKER_01", "gender": "female", "voice": "hi-IN-SwaraNeural",
         "voice_label": "Swara", "seconds": 12.0, "reused": False},
    ]                                           # UNKNOWN placeholder (0 s) left out
    # UNKNOWN voices every line when speaker detection is off: then it is listed
    turns = [types.SimpleNamespace(speaker_id="UNKNOWN", source_start=0.0, source_end=2.5)]
    unknown = app_mod._dialogue_speakers(tmp_path / "speakers.json", ["edge"], turns)[-1]
    assert unknown["speaker"] == "UNKNOWN" and unknown["seconds"] == 2.5


def test_dialogue_speakers_pitch_variant_reuse_and_provider_order(app_mod, tmp_path):
    p = _speakers_json(tmp_path, {"SPEAKER_03": {
        "voice_category": "female_like", "total_speech_s": 3.0,
        "provider_voices": {"edge": {"voice": "hi-IN-SwaraNeural", "pitch": "+20Hz",
                                     "indistinguishable_reuse": True}}}})
    row, = app_mod._dialogue_speakers(p, ["sarvam", "edge"])   # first provider WITH a binding
    assert (row["voice"], row["voice_label"], row["reused"]) == \
        ("hi-IN-SwaraNeural|+20Hz", "Swara +20Hz", True)


def test_dialogue_job_gets_title_characters_subtitles_and_cancel_hook(app_mod, monkeypatch):
    from dubbing.dialogue import orchestrator as orch
    from dubbing.dialogue.speaker_registry import SpeakerRegistry
    from test_modules import FakeProbe
    monkeypatch.setattr(app_mod, "_dialogue_probe", lambda max_age=60.0: FakeProbe())
    hooked = []

    def fake_run_dialogue(cfg, on_progress=None, cancel_check=None, components=None,
                          on_legacy_pipeline=None):
        legacy = types.SimpleNamespace(paused_at=None, segments=[])
        on_legacy_pipeline(legacy)               # the download step's Pipeline
        hooked.append(job.pipeline_ref is legacy)
        (cfg.work_dir / "source_title.txt").write_text("Real Title\n", encoding="utf-8")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        reg = SpeakerRegistry()
        reg.register("SPEAKER_00", voice_category="female_like", total_speech_s=5.0)
        reg.bind_provider((cfg.tts_providers or ["edge"])[0])
        reg.save(cfg.output_dir / "speakers.json")
        srt = cfg.output_dir / "subtitles_hi.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nनमस्ते\n", encoding="utf-8")
        md = cfg.output_dir / "report.md"
        md.write_text("# report", encoding="utf-8")
        return orch.DialogueResult(status="completed", output_video=None,
                                   report_json=cfg.output_dir / "report.json", report_md=md,
                                   subtitles=srt, reasons=[], turns=[])

    monkeypatch.setattr(orch, "run_dialogue", fake_run_dialogue)
    job = app_mod.Job(id="dialogue0001")
    app_mod._run_dialogue_mode(job, app_mod.JobCreateRequest(
        url="https://www.youtube.com/watch?v=abc123", pipeline_mode="hindi_dialogue"))
    assert hooked == [True] and job.pipeline_ref is None
    assert job.state == "done" and job.video_title == "Real Title"
    assert [(s["speaker"], s["gender"]) for s in job.speakers] == [("SPEAKER_00", "female")]
    assert Path(job.subtitles_path).parent == Path(job.saved_folder)


def test_subtitles_and_report_are_served_after_the_job_folder_is_gone(app_mod, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    srt = tmp_path / "saved" / "T [HI Dubbed] (x)" / "T - HI Dubbed.srt"
    srt.parent.mkdir(parents=True)
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nनमस्ते\n", encoding="utf-8")
    report = tmp_path / "report.md"
    report.write_text("# Report\n", encoding="utf-8")
    job = app_mod.Job(id="srtjob000001", state="done", subtitles_path=str(srt),
                      report_path=str(report))
    monkeypatch.setitem(app_mod.JOBS, job.id, job)
    c = TestClient(app_mod.app)
    r = c.get(f"/api/jobs/{job.id}/srt")
    assert r.status_code == 200 and "नमस्ते" in r.content.decode("utf-8")
    assert c.get(f"/api/jobs/{job.id}").json()["subtitles_path"] == str(srt)
    r = c.get(f"/api/jobs/{job.id}/report")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert r.headers["content-disposition"].startswith("inline")   # opens in the tab


def test_job_saved_before_subtitles_path_still_offers_its_subtitles(app_mod, monkeypatch, tmp_path):
    # The job page shows the Subtitles button only when the payload names a
    # file: a job saved before subtitles_path existed reports its saved SRT.
    from fastapi.testclient import TestClient
    folder = tmp_path / "saved" / "Old [HI Dubbed] (old000000001)"
    folder.mkdir(parents=True)
    (folder / "transcript_en_speakers.srt").write_text("1\n", encoding="utf-8")   # not the dub
    srt = folder / "Old - HI Dubbed.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nनमस्ते\n", encoding="utf-8")
    old = app_mod.Job(id="old000000001", state="done", saved_folder=str(folder))
    none = app_mod.Job(id="none00000001", state="done")
    monkeypatch.setitem(app_mod.JOBS, old.id, old)
    monkeypatch.setitem(app_mod.JOBS, none.id, none)
    c = TestClient(app_mod.app)
    assert c.get(f"/api/jobs/{old.id}").json()["subtitles_path"] == str(srt)
    assert "नमस्ते" in c.get(f"/api/jobs/{old.id}/srt").content.decode("utf-8")
    assert c.get(f"/api/jobs/{none.id}").json()["subtitles_path"] is None
    assert c.get(f"/api/jobs/{none.id}/srt").status_code == 404
