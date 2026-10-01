"""End-to-end orchestration on synthetic media with mocked model backends.

These tests exercise routing, identity, coverage, reporting and real FFmpeg
rendering/muxing. They do NOT measure dubbing quality: the TTS here is a
tone generator and ASR/diarization/translation are scripted.
"""
import json
import math
import wave
from array import array
from pathlib import Path

import pytest

from dubbing.dialogue import audio
from dubbing.dialogue.diarization import DiarizationResult, DiarizationUnavailable
from dubbing.dialogue.orchestrator import (Components, DialogueConfig,
                                           DialogueOrchestrator, default_acquire)
from dubbing.dialogue.tts import MockProvider

try:
    audio.find_ffmpeg()
    HAVE_FFMPEG = True
except RuntimeError:
    HAVE_FFMPEG = False
pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")

# (speaker, f0, start, end, english words)
SCRIPT = [
    ("SPEAKER_00", 115.0, 0.3, 2.3, "Where were you last night?"),
    ("SPEAKER_01", 225.0, 2.8, 4.6, "I was at the station."),
    ("SPEAKER_00", 115.0, 5.0, 5.6, "Why?"),
    ("SPEAKER_01", 225.0, 6.0, 8.4, "Riya asked me to wait for 2 hours."),
]
HINDI = {
    "Where were you last night?": "कल रात तुम कहाँ थे?",
    "I was at the station.": "मैं स्टेशन पर थी।",
    "Why?": "क्यों?",
    "Riya asked me to wait for 2 hours.": "रिया ने मुझे 2 घंटे रुकने को कहा।",
}
DURATION = 10.0


def _make_media(tmp: Path) -> Path:
    sr = 16000
    data = array("h", [0] * int(DURATION * sr))
    for _, f0, s, e, _ in SCRIPT:
        for n in range(int(s * sr), int(e * sr)):
            v = sum(math.sin(2 * math.pi * f0 * k * n / sr) / k for k in range(1, 6))
            data[n] = int(5000 * v)
    wav = tmp / "dialogue.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    mp4 = tmp / "input.mp4"
    audio.run_ffmpeg(["-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={DURATION}",
                      "-i", str(wav), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-shortest", str(mp4)])
    return mp4


def _asr_segments():
    segs = []
    for spk, _, s, e, text in SCRIPT:
        toks = text.split()
        step = (e - s) / len(toks)
        segs.append({"start": s, "end": e, "text": text,
                     "words": [{"word": w, "start": s + i * step, "end": s + (i + 1) * step}
                               for i, w in enumerate(toks)]})
    return segs


def _diar():
    segs = [(s - 0.05, e + 0.05, spk) for spk, _, s, e, _ in SCRIPT]
    return DiarizationResult(segs, segs, {"SPEAKER_00": [1, 0], "SPEAKER_01": [0, 1]},
                             backend="mock")


class FakeLLM:
    model_id = "fake:llm"

    def complete(self, system, user):
        req = json.loads(user)
        if "transcript" in req:    # whole-video story brief
            return json.dumps({"speakers": {"SPEAKER_00": {"gender": "male", "evidence": "he"},
                                            "SPEAKER_01": {"gender": "female", "evidence": "she"}},
                               "address": {"SPEAKER_00 -> SPEAKER_01": "tum"},
                               "names": {"Riya": "रिया"}, "summary": "two friends"})
        if "turns" not in req:     # rewrite request
            return json.dumps({"hi": req["hindi"]}, ensure_ascii=False)
        return json.dumps({"translations": [{"id": t["id"], "hi": HINDI[t["text"]]} for t in req["turns"]],
                           "names": {"Riya": "रिया"}}, ensure_ascii=False)


class FakeHindiASR:
    """Returns exactly what each clip was meant to say, except overrides."""

    def __init__(self, turns_text, overrides=None):
        self.turns_text = turns_text
        self.overrides = overrides or {}
        self.model_name = "fake-asr"

    def transcribe(self, path):
        tid = Path(path).name.split("_")[0]
        return self.overrides.get(tid, self.turns_text.get(tid, ""))


def _components(calls, *, fail_text=None, diar_ok=True, subs=None, verifier_overrides=None,
                orch_holder=None):
    def diarize(wav):
        calls["diarize"] = calls.get("diarize", 0) + 1
        if not diar_ok:
            raise DiarizationUnavailable("HF_TOKEN is not set (simulated)")
        return _diar()

    def asr(wav):
        calls["asr"] = calls.get("asr", 0) + 1
        return _asr_segments()

    def content_asr():
        turns = orch_holder["orch"].turns if orch_holder else []
        return FakeHindiASR({t.turn_id: t.speech_text for t in turns}, verifier_overrides)

    return Components(
        acquire=default_acquire, asr=asr, diarize=diarize,
        fetch_subtitles=(lambda url: subs) if subs is not None else None,
        llm_clients=[FakeLLM()],
        tts_providers={"mock": MockProvider(fail=(lambda t, b: fail_text in t) if fail_text else None)},
        content_asr_factory=content_asr,
        separate=lambda orig, work, policy: {"status": "failed", "background": None,
                                             "detail": "simulated separation failure"},
        basic_translate=lambda s: "")


def _go(tmp_path, **kw):
    holder = {}
    cfg_kw = kw.pop("cfg", {})
    comps = _components(kw.pop("calls", {}), orch_holder=holder, **kw)
    media = _make_media(tmp_path)
    cfg = DialogueConfig(source=str(media), work_dir=tmp_path / "work", output_dir=tmp_path / "out",
                         tts_providers=["mock"], **cfg_kw)
    orch = DialogueOrchestrator(cfg, comps)
    holder["orch"] = orch
    return orch, orch.run()


def test_mixed_dialogue_end_to_end(tmp_path):
    orch, res = _go(tmp_path)
    rep = json.loads(res.report_json.read_text(encoding="utf-8"))
    assert res.status == "completed", rep["status_reasons"]
    # four turns, alternating speakers, each voiced by its speaker's bound voice
    spk = {s["speaker_id"]: s for s in rep["speakers"]}
    assert spk["SPEAKER_00"]["voice_category"] == "male_like"
    assert spk["SPEAKER_01"]["voice_category"] == "female_like"
    clips = {c.turn_id: c for c in orch.clips.values()}
    assert [clips[t.turn_id].voice for t in orch.turns] == \
        ["mock-male-1", "mock-female-1", "mock-male-1", "mock-female-1"]
    assert rep["required_turn_ids"] == rep["generated_turn_ids"] == ["t0001", "t0002", "t0003", "t0004"]
    # outputs: video with both streams, original duration, subtitles aligned to dub audio
    info = audio.probe_streams(res.output_video)
    assert info["video"] and info["audio"] and abs(info["duration"] - DURATION) < 0.3
    srt = res.subtitles.read_text(encoding="utf-8")
    assert srt.count("-->") == 4 and "रिया" in srt
    # whole-video brief ran, agrees with the voices -> no disagreement warning
    tr = [s for s in rep["stages"] if s["name"] == "translate"][0]
    assert tr["data"]["story_brief"]["address"] == {"SPEAKER_00 -> SPEAKER_01": "tum"}
    assert not [w for w in rep["content_warnings"] if w.get("type") == "speaker_gender_disagreement"]
    # separation failed -> Hindi-only mix, never English as "background"
    assert rep["separation"]["status"] == "failed"
    mixinfo = [s for s in rep["stages"] if s["name"] == "mix"][0]["data"]["mix"]
    assert mixinfo["background"] is False
    # source timing untouched and scheduling recorded separately
    for t in orch.turns:
        c = clips[t.turn_id]
        assert c.scheduled_start >= t.source_start - 0.25


def test_youtube_subtitles_still_run_audio_diarization(tmp_path):
    # YouTube cues carry no speaker labels and group two speakers in one cue
    subs = [{"start": 0.3, "end": 4.6, "text": "Where were you last night? I was at the station."},
            {"start": 5.0, "end": 8.4, "text": "Why? Riya asked me to wait for 2 hours."}]
    calls, holder = {}, {}
    comps = _components(calls, subs=subs, orch_holder=holder)
    media = _make_media(tmp_path)
    comps.acquire = lambda cfg, w: media           # stand-in for the link download
    cfg = DialogueConfig(source="https://www.youtube.com/watch?v=abcdefgh", work_dir=tmp_path / "work",
                         output_dir=tmp_path / "out", tts_providers=["mock"])
    orch = DialogueOrchestrator(cfg, comps)
    holder["orch"] = orch
    res = orch.run()
    transcribe = [s for s in orch.report.stages if s.name == "transcribe"][0]
    assert transcribe.data["text_source"] == "youtube_subs"
    assert calls["diarize"] == 1 and calls["asr"] == 1
    assert [t.speaker_id for t in orch.turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]
    assert orch.turns[3].source_text == "Riya asked me to wait for 2 hours."
    assert res.status == "completed", res.reasons


def test_subtitle_text_source_is_attributed_by_audio(tmp_path):
    srt = tmp_path / "en.srt"
    # one cue spanning two speakers: audio attribution must split it
    srt.write_text("1\n00:00:00,300 --> 00:00:04,600\nWhere were you last night? I was at the station.\n\n"
                   "2\n00:00:05,000 --> 00:00:05,600\nWhy?\n\n"
                   "3\n00:00:06,000 --> 00:00:08,400\nRiya asked me to wait for 2 hours.\n",
                   encoding="utf-8")
    calls = {}
    orch, res = _go(tmp_path, calls=calls, cfg={"source_srt": srt})
    assert calls["diarize"] == 1
    assert [(t.speaker_id, t.source_text) for t in orch.turns] == [
        ("SPEAKER_00", "Where were you last night?"), ("SPEAKER_01", "I was at the station."),
        ("SPEAKER_00", "Why?"), ("SPEAKER_01", "Riya asked me to wait for 2 hours.")]
    assert res.status == "completed", res.reasons


def test_translated_srt_input_uses_audio_speakers(tmp_path):
    srt = tmp_path / "hi.srt"
    srt.write_text("\n".join(f"{i}\n00:00:0{int(s)},{int((s % 1) * 1000):03d} --> "
                             f"00:00:0{int(e)},{int((e % 1) * 1000):03d}\n{HINDI[text]}\n"
                             for i, (_, _, s, e, text) in enumerate(SCRIPT, 1)), encoding="utf-8")
    calls = {}
    orch, res = _go(tmp_path, calls=calls, cfg={"translated_srt": srt})
    assert calls["diarize"] == 1 and "asr" not in calls
    assert [t.speaker_id for t in orch.turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]
    stages = {s.name: s.status for s in orch.report.stages}
    assert stages["translate"] == "skipped"


def test_missing_turn_makes_honest_draft(tmp_path):
    orch, res = _go(tmp_path, fail_text="क्यों")
    rep = json.loads(res.report_json.read_text(encoding="utf-8"))
    assert res.status == "draft_incomplete"
    assert [m["turn_id"] for m in rep["missing_turns"]] == ["t0003"]
    assert res.output_video and res.output_video.exists()   # partial asset kept for review


def test_no_diarization_is_reported_not_hidden(tmp_path):
    orch, res = _go(tmp_path, diar_ok=False)
    assert res.status == "completed_with_warnings"
    assert any("diarization unavailable" in r for r in res.reasons)
    assert {t.speaker_id for t in orch.turns} == {"UNKNOWN"}


def test_wrong_speech_same_word_count_is_regenerated_then_reported(tmp_path):
    orch, res = _go(tmp_path, verifier_overrides={"t0002": "मैं स्कूल पर थी।"})
    rep = json.loads(res.report_json.read_text(encoding="utf-8"))
    cw = [w for w in rep["content_warnings"] if w.get("type") == "content_mismatch"]
    assert [w["turn_id"] for w in cw] == ["t0002"]
    assert res.status == "completed_with_warnings"
    # regeneration kept the female voice
    assert orch.clips["t0002"].voice == "mock-female-1"
    # the replaced clip is kept for the record but not counted twice
    assert rep["duplicate_clips"] == []
    superseded = [c for c in orch.all_clips if c.verification.get("superseded")]
    assert [c.turn_id for c in superseded] == ["t0002"]


def test_cancel_keeps_partial_assets(tmp_path):
    holder = {}
    comps = _components({}, orch_holder=holder)
    media = _make_media(tmp_path)
    cfg = DialogueConfig(source=str(media), work_dir=tmp_path / "work", output_dir=tmp_path / "out",
                         tts_providers=["mock"])
    state = {"n": 0}

    def cancel():
        state["n"] += 1
        return state["n"] > 6
    orch = DialogueOrchestrator(cfg, comps, cancel_check=cancel)
    holder["orch"] = orch
    res = orch.run()
    assert res.status == "cancelled"
    assert (tmp_path / "work" / "original_48k.wav").exists()
    assert json.loads(res.report_json.read_text(encoding="utf-8"))["final_status"] == "cancelled"


def test_separation_runs_once_first_and_vocals_drive_speaker_analysis(tmp_path):
    """Separate-first (as pyVideoTrans/SoniTranslate do): the vocals stem
    feeds diarization; the background bed is reused by the mix."""
    calls, holder, seen = {}, {}, {}
    comps = _components(calls, orch_holder=holder)
    media = _make_media(tmp_path)

    def separate(orig, work, policy):
        calls["separate"] = calls.get("separate", 0) + 1
        voc = work / "vocals_estimate.wav"
        bed = work / "background_estimate.wav"
        audio.to_wav(orig, voc, channels=2)
        audio.run_ffmpeg(["-f", "lavfi", "-i", f"anoisesrc=d={DURATION}:a=0.01",
                          "-ac", "2", "-ar", "48000", str(bed)])
        return {"status": "ok", "background": str(bed), "vocals": str(voc), "detail": "fake"}

    def diarize(wav):
        seen["diarize_input"] = Path(wav).name
        return _diar()

    comps.separate, comps.diarize = separate, diarize
    cfg = DialogueConfig(source=str(media), work_dir=tmp_path / "work", output_dir=tmp_path / "out",
                         tts_providers=["mock"])
    orch = DialogueOrchestrator(cfg, comps)
    holder["orch"] = orch
    res = orch.run()
    rep = json.loads(res.report_json.read_text(encoding="utf-8"))
    assert calls["separate"] == 1
    assert seen["diarize_input"] == "vocals_16k_mono.wav"
    names = [s["name"] for s in rep["stages"]]
    assert names.index("separate") < names.index("diarize")
    mixinfo = [s for s in rep["stages"] if s["name"] == "mix"][0]["data"]["mix"]
    assert mixinfo["background"] is True and mixinfo["residue_duck"] is True
    assert res.status in ("completed", "completed_with_warnings"), res.reasons


def test_analysis_audio_mix_keeps_diarization_on_original(tmp_path):
    seen = {}
    holder = {}
    comps = _components({}, orch_holder=holder)
    media = _make_media(tmp_path)

    def separate(orig, work, policy):
        voc = work / "vocals_estimate.wav"
        audio.to_wav(orig, voc, channels=2)
        return {"status": "failed", "background": None, "vocals": str(voc), "detail": "bed failed"}

    def diarize(wav):
        seen["in"] = Path(wav).name
        return _diar()

    comps.separate, comps.diarize = separate, diarize
    cfg = DialogueConfig(source=str(media), work_dir=tmp_path / "work", output_dir=tmp_path / "out",
                         tts_providers=["mock"], analysis_audio="mix")
    orch = DialogueOrchestrator(cfg, comps)
    holder["orch"] = orch
    orch.run()
    assert seen["in"] == "original_16k_mono.wav"
