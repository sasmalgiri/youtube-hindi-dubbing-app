"""Review before voicing, re-voicing a finished job (resume from its own
checkpoint), the within-job caches, voice matching and the output options.

Same synthetic media and mocked backends as test_orchestrator_e2e.py: these
tests check routing, identity and bookkeeping, not dubbing quality.
"""
import json
import shutil
import sys
import types
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubbing.dialogue import checkpoint
from dubbing.dialogue import mix as mix_mod
from dubbing.dialogue import orchestrator as orch_mod
from dubbing.dialogue.contracts import Turn
from dubbing.dialogue.diarization import DiarizationResult
from dubbing.dialogue.orchestrator import (RESUMED_DETAIL, DialogueConfig,
                                           DialogueOrchestrator, default_components)
from dubbing.dialogue.speaker_registry import SpeakerRegistry, VoiceResolutionError
from dubbing.dialogue.translation import TranslationEngineError
from dubbing.dialogue.tts import MockProvider, TTSRouter

from test_orchestrator_e2e import HAVE_FFMPEG, HINDI, SCRIPT, FakeLLM, _components, _make_media

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")

CHECKPOINTED = ["acquire", "extract", "separate", "transcribe", "diarize", "turns", "speakers",
                "translate"]


class CountingLLM(FakeLLM):
    def __init__(self):
        self.requests = []

    def complete(self, system, user):
        req = json.loads(user)
        self.requests.append("translate" if "turns" in req else
                             "brief" if "transcript" in req else "rewrite")
        return super().complete(system, user)


def _diar3():
    """The last line is a third speaker (the same woman, split by diarization).
    SPEAKER_01 then has too little speech for a voice category ("unknown");
    the story brief knows from the text that she is a woman."""
    spk = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_02"]
    segs = [(s - 0.05, e + 0.05, k) for (_, _, s, e, _), k in zip(SCRIPT, spk)]
    return DiarizationResult(segs, segs, {k: [1.0, 0.0] for k in set(spk)}, backend="mock")


def _orch(tmp_path, *, cfg=None, review=None, cancel=None, provider=None, llm=None, diar=None,
          voice_matcher=None):
    """A fresh orchestrator on the shared synthetic media (made once per test,
    so a re-voice run sees the same source)."""
    holder, calls = {}, {}
    comps = _components(calls, orch_holder=holder)
    if provider is not None:
        comps.tts_providers = {"mock": provider}
    if llm is not None:
        comps.llm_clients = [llm]
    if diar is not None:
        def diarize(wav):
            calls["diarize"] = calls.get("diarize", 0) + 1
            return diar
        comps.diarize = diarize
    comps.voice_matcher = voice_matcher
    media = tmp_path / "input.mp4"
    if not media.exists():
        _make_media(tmp_path)
    c = DialogueConfig(source=str(media), work_dir=tmp_path / "work", output_dir=tmp_path / "out",
                       tts_providers=["mock"], **(cfg or {}))
    orch = DialogueOrchestrator(c, comps, cancel_check=cancel, review=review)
    holder["orch"] = orch
    return orch, calls


def _report(res):
    return json.loads(res.report_json.read_text(encoding="utf-8"))


def _clip_of(orch, tid):
    return orch.clips[tid]


# ── review before voicing ─────────────────────────────────────────────────
@needs_ffmpeg
def test_review_edits_reach_every_output(tmp_path):
    seen = {}
    new_line = "कल रात तुम किसके साथ थे?"

    def review(packet):
        seen["packet"] = packet
        seen["file"] = json.loads((tmp_path / "work" / "review.json").read_text(encoding="utf-8"))
        seen["calls_before"] = len(prov.calls)
        return {"turn_edits": {"t0001": {"hi": new_line},
                               "t0002": {"delete": True},
                               "t0003": {"speaker_id": "SPEAKER_02"}},
                "speaker_merges": {"SPEAKER_01": "SPEAKER_02"},
                "voice_overrides": {"SPEAKER_00": {"provider": "mock", "voice": "mock-male-2",
                                                   "pitch": None}}}

    prov = MockProvider()
    orch, _ = _orch(tmp_path, cfg={"review_before_voice": True}, review=review, provider=prov,
                    diar=_diar3())
    res = orch.run()
    rep = _report(res)

    # the packet the reviewer saw (also written to work/review.json)
    p = seen["packet"]
    assert p["job_stage"] == "before_voice" and seen["file"]["job_stage"] == "before_voice"
    assert [t["turn_id"] for t in p["turns"]] == ["t0001", "t0002", "t0003", "t0004"]
    t1 = p["turns"][0]
    assert t1["english"] == "Where were you last night?" and t1["hindi"] == HINDI[t1["english"]]
    assert t1["clip"] is None and t1["overflow_s"] is None and t1["required"] is True
    spk = {s["speaker_id"]: s for s in p["speakers"]}
    assert set(spk) == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}
    assert spk["SPEAKER_00"]["voice"] == "mock-male-1" and spk["SPEAKER_00"]["override"] is False
    assert spk["SPEAKER_00"]["provider"] == "mock" and spk["SPEAKER_00"]["turns"] == 2
    ref = spk["SPEAKER_00"]["reference_clip"]
    assert ref == "ref_SPEAKER_00.wav" and (tmp_path / "out" / "clips" / ref).exists()
    assert [v["voice"] for v in p["voice_options"]["mock"]["female_like"]] == \
        ["mock-female-1", "mock-female-2"]
    assert p["media_duration"] > 9.5
    assert seen["calls_before"] == 0     # nothing was voiced before the review

    # every edit is in the dub
    assert res.status == "completed", res.reasons
    turns = {t.turn_id: t for t in orch.turns}
    assert turns["t0001"].hi_display == new_line and "edited" in turns["t0001"].flags
    assert _clip_of(orch, "t0001").spoken_text == new_line
    assert _clip_of(orch, "t0001").voice == "mock-male-2"                # voice override
    assert turns["t0002"].required is False and "deleted_by_user" in turns["t0002"].flags
    assert "t0002" not in orch.clips and "t0002" not in rep["required_turn_ids"]
    assert rep["missing_turns"] == []                                    # not counted as missing
    assert turns["t0003"].speaker_id == "SPEAKER_02"                     # relabelled
    assert _clip_of(orch, "t0003").voice == "mock-female-1"
    assert turns["t0002"].speaker_id == "SPEAKER_02"                     # merged (and deleted)
    assert turns["t0004"].speaker_id == "SPEAKER_02"
    assert _clip_of(orch, "t0004").voice == "mock-female-1"
    assert "SPEAKER_01" not in orch.registry.speakers
    assert {s["speaker_id"] for s in rep["speakers"]} == {"SPEAKER_00", "SPEAKER_02"}
    # SPEAKER_01's voice category was unknown, the brief says she is a woman: no gender warning
    assert not [w for w in rep["content_warnings"] if w["type"] == "speaker_merge_gender_check"]
    assert rep["identity_violations"] == []
    s0 = [s for s in rep["speakers"] if s["speaker_id"] == "SPEAKER_00"][0]
    assert s0["mapping_origin"] == "user" and s0["provider_voices"]["mock"]["override"] is True
    srt = (tmp_path / "out" / "subtitles_hi.srt").read_text(encoding="utf-8")
    assert new_line in srt and HINDI["I was at the station."] not in srt
    en = (tmp_path / "out" / "subtitles_en.srt").read_text(encoding="utf-8")
    assert "Where were you last night?" in en and "I was at the station." not in en
    assert "SPEAKER" not in en
    kinds = sorted(e["kind"] for e in rep["applied_edits"])
    assert kinds == ["speaker_merge", "turn_edit", "turn_edit", "turn_edit", "voice_override"]
    assert all(e["via"] == "review" and not e.get("ignored") for e in rep["applied_edits"])
    review_stage = [s for s in rep["stages"] if s["name"] == "review"][0]
    assert review_stage["detail"].startswith("5 edit(s) applied")

    # after-run packet: accepted clips copied for listening
    after = json.loads((tmp_path / "out" / "review.json").read_text(encoding="utf-8"))
    assert after["job_stage"] == "after_run"
    by_id = {t["turn_id"]: t for t in after["turns"]}
    assert by_id["t0001"]["clip"] == "t0001.wav" and by_id["t0002"]["clip"] is None
    assert (tmp_path / "out" / "clips" / "t0001.wav").exists()
    assert {s["speaker_id"] for s in after["speakers"]} == {"SPEAKER_00", "SPEAKER_02"}
    assert [s for s in after["speakers"] if s["speaker_id"] == "SPEAKER_00"][0]["override"] is True


@needs_ffmpeg
def test_merging_speakers_of_different_categories_is_flagged(tmp_path):
    orch, _ = _orch(tmp_path, cfg={"speaker_merges": {"SPEAKER_01": "SPEAKER_00"}})
    res = orch.run()
    rep = _report(res)
    assert {t.speaker_id for t in orch.turns} == {"SPEAKER_00"}
    assert {c.voice for c in orch.clips.values() if c.accepted} == {"mock-male-1"}
    w = [x for x in rep["content_warnings"] if x["type"] == "speaker_merge_gender_check"]
    assert len(w) == 1 and w[0]["from_category"] == "female_like" and w[0]["turn_ids"] == ["t0002", "t0004"]
    assert res.status == "completed_with_warnings"
    assert [e["via"] for e in rep["applied_edits"]] == ["request"]


@needs_ffmpeg
def test_bad_edits_are_reported_not_applied(tmp_path):
    orch, _ = _orch(tmp_path, cfg={
        "turn_edits": {"t9999": {"hi": "x"}, "t0001": {"speaker_id": "SPEAKER_77"}},
        "voice_overrides": {"SPEAKER_00": {"provider": "edge", "voice": "hi-IN-MadhurNeural"},
                            "NOBODY": {"category": "female_like"}}})
    res = orch.run()
    rep = _report(res)
    assert res.status == "completed", res.reasons
    assert all(e.get("ignored") for e in rep["applied_edits"]) and len(rep["applied_edits"]) == 4
    assert orch.turns[0].speaker_id == "SPEAKER_00" and _clip_of(orch, "t0001").voice == "mock-male-1"


@needs_ffmpeg
def test_cancel_during_review_then_resume(tmp_path):
    state = {"cancel": False}

    def review(packet):
        state["cancel"] = True          # the user pressed Cancel while reviewing
        return {"turn_edits": {"t0001": {"hi": "रद्द"}}}

    prov = MockProvider()
    orch, _ = _orch(tmp_path, cfg={"review_before_voice": True}, review=review,
                    cancel=lambda: state["cancel"], provider=prov)
    res = orch.run()
    assert res.status == "cancelled"
    assert not prov.calls and "synthesize" not in {s.name for s in orch.report.stages}
    assert orch.turns[0].hi_raw == HINDI["Where were you last night?"]   # edits not applied
    saved = json.loads(checkpoint.checkpoint_path(tmp_path / "work").read_text(encoding="utf-8"))
    assert saved["completed"][-1] == "translate"

    # the cancelled job is re-voiced from its checkpoint
    orch2, calls2 = _orch(tmp_path, cfg={"resume": True}, provider=prov)
    res2 = orch2.run()
    assert res2.status == "completed", res2.reasons
    assert calls2 == {} and len(prov.calls) == 4
    assert orch2.report.resumed_from_checkpoint == CHECKPOINTED


# ── re-voicing a finished job ─────────────────────────────────────────────
@needs_ffmpeg
def test_resume_skips_to_voicing_and_reuses_unchanged_lines(tmp_path):
    prov, llm = MockProvider(), CountingLLM()
    orch1, calls1 = _orch(tmp_path, provider=prov, llm=llm)
    res1 = orch1.run()
    assert res1.status == "completed", res1.reasons
    assert len(prov.calls) == 4 and calls1 == {"asr": 1, "diarize": 1}
    first_llm = len(llm.requests)
    assert (tmp_path / "work" / "checkpoint" / "state.json").exists()

    # re-voice with one changed line: only that line is synthesized again
    new_line = "मैं पूरी रात स्टेशन पर थी।"
    orch2, calls2 = _orch(tmp_path, provider=prov, llm=llm,
                          cfg={"resume": True, "turn_edits": {"t0002": {"hi": new_line}}})
    res2 = orch2.run()
    rep2 = _report(res2)
    assert res2.status == "completed", res2.reasons
    assert calls2 == {}                                      # no ASR, no diarization
    assert [r for r in llm.requests[first_llm:] if r != "rewrite"] == []   # no translation
    assert len(prov.calls) == 5 and prov.calls[-1]["text"] == new_line
    assert rep2["resumed_from_checkpoint"] == CHECKPOINTED
    st = {s["name"]: s for s in rep2["stages"]}
    for name in CHECKPOINTED:
        assert st[name]["status"] == "skipped" and st[name]["detail"] == RESUMED_DETAIL
    assert st["translate"]["data"]["checkpoint"]["data"]["story_brief"]["address"]
    assert st["synthesize"]["data"]["tts_cache_hits"] == 3
    hits = [c.turn_id for c in orch2.clips.values()
            if any(h.get("cache") == "hit" for h in c.retry_history)]
    assert sorted(hits) == ["t0001", "t0003", "t0004"]
    assert _clip_of(orch2, "t0002").spoken_text == new_line
    assert new_line in (tmp_path / "out" / "subtitles_hi.srt").read_text(encoding="utf-8")
    assert rep2["input_identity"] == _report(res1)["input_identity"]
    assert "Resumed run" in res2.report_md.read_text(encoding="utf-8")
    assert [e["via"] for e in rep2["applied_edits"]] == ["request"]

    # a second re-voice keeps the first one's edit and re-voices only SPEAKER_00
    orch3, _ = _orch(tmp_path, provider=prov, llm=llm,
                     cfg={"resume": True, "voice_overrides": {"SPEAKER_00": {"category": "female_like"}}})
    res3 = orch3.run()
    rep3 = _report(res3)
    # the story brief says SPEAKER_00 is a man: the Hindi may carry male forms
    assert res3.status == "completed_with_warnings", res3.reasons
    assert [(w["type"], w["speaker_id"], w.get("set_by_user")) for w in rep3["content_warnings"]] == \
        [("speaker_gender_disagreement", "SPEAKER_00", True)]
    assert len(prov.calls) == 7
    assert {c["voice"] for c in prov.calls[-2:]} == {"mock-female-2"}   # a voice SPEAKER_01 does not use
    assert _clip_of(orch3, "t0002").spoken_text == new_line              # earlier edit kept
    assert _clip_of(orch3, "t0002").voice == "mock-female-1"
    assert rep3["identity_violations"] == []
    assert [e["kind"] for e in rep3["applied_edits"]] == ["turn_edit", "voice_override"]


class DownLLM:
    model_id = "groq:test-model"

    def complete(self, system, user):
        raise TranslationEngineError("groq HTTP 503: overloaded")


@needs_ffmpeg
def test_failed_translation_resumes_from_the_last_finished_stage(tmp_path):
    orch1, _ = _orch(tmp_path, llm=DownLLM())
    res1 = orch1.run()
    assert res1.status == "failed"
    saved = json.loads(checkpoint.checkpoint_path(tmp_path / "work").read_text(encoding="utf-8"))
    assert saved["completed"][-1] == "speakers"

    llm = CountingLLM()
    orch2, calls2 = _orch(tmp_path, llm=llm, cfg={"resume": True})
    res2 = orch2.run()
    assert res2.status == "completed", res2.reasons
    assert calls2 == {} and "translate" in llm.requests
    assert orch2.report.resumed_from_checkpoint == CHECKPOINTED[:-1]
    assert {s.name: s.status for s in orch2.report.stages}["translate"] == "ok"


@needs_ffmpeg
def test_resume_never_uses_another_sources_checkpoint(tmp_path):
    prov = MockProvider()
    orch1, _ = _orch(tmp_path, provider=prov)
    orch1.run()
    orch2, calls2 = _orch(tmp_path, provider=prov, cfg={"resume": True, "limit_seconds": 9.0})
    res2 = orch2.run()
    assert res2.status in ("completed", "completed_with_warnings"), res2.reasons
    assert calls2 == {"asr": 1, "diarize": 1} and orch2.report.resumed_from_checkpoint == []
    assert any("different source" in x for x in orch2.report.limitations)
    assert len(prov.calls) == 8                       # the caches were cleared too


@needs_ffmpeg
def test_a_fresh_run_clears_the_previous_runs_caches(tmp_path):
    prov = MockProvider()
    _orch(tmp_path, provider=prov)[0].run()
    _orch(tmp_path, provider=prov)[0].run()            # not a resume: from scratch
    assert len(prov.calls) == 8


# ── voice matching ────────────────────────────────────────────────────────
class FakeMatcher:
    name = "fake-match"

    def __init__(self, fail=False, refuse=()):
        self.fail, self.refuse = fail, set(refuse)
        self.prepared, self.converted, self.closed = {}, [], 0

    def prepare(self, speaker_id, reference_wavs):
        self.prepared[speaker_id] = [Path(p) for p in reference_wavs]
        return speaker_id not in self.refuse

    def convert(self, in_wav, out_wav, speaker_id):
        self.converted.append((Path(in_wav).name, speaker_id))
        if self.fail:
            raise RuntimeError("converter crashed")
        shutil.copyfile(in_wav, out_wav)
        return True

    def close(self):
        self.closed += 1


@needs_ffmpeg
def test_voice_matcher_converts_every_clip(tmp_path):
    vm = FakeMatcher()
    orch, _ = _orch(tmp_path, voice_matcher=vm, cfg={"voice_match": "openvoice"})
    res = orch.run()
    assert res.status == "completed", res.reasons
    assert set(vm.prepared) == {"SPEAKER_00", "SPEAKER_01"}
    for wavs in vm.prepared.values():
        assert 1 <= len(wavs) <= 3
        for w in wavs:
            with wave.open(str(w), "rb") as wf:
                assert wf.getframerate() == 24000 and wf.getnchannels() == 1
                assert wf.getnframes() / 24000 <= 12.0
    assert len(vm.converted) == len(orch.all_clips) == 4
    assert {sp for _, sp in vm.converted} == {"SPEAKER_00", "SPEAKER_01"}
    assert all(c.voice_params.get("voice_matched") for c in orch.clips.values())
    assert vm.closed == 1
    assert orch.report.model_versions["voice_match"] == "fake-match"


@needs_ffmpeg
def test_voice_matcher_failure_is_not_fatal(tmp_path):
    vm = FakeMatcher(fail=True, refuse={"SPEAKER_01"})
    orch, _ = _orch(tmp_path, voice_matcher=vm)
    res = orch.run()
    assert res.status == "completed", res.reasons
    assert [sp for _, sp in vm.converted] == ["SPEAKER_00", "SPEAKER_00"]   # refused speaker skipped
    assert not any(c.voice_params.get("voice_matched") for c in orch.clips.values())
    lim = " | ".join(orch.report.limitations)
    assert "voice matching failed on 2 clip(s)" in lim and "SPEAKER_01 (no usable reference)" in lim


def test_default_components_report_a_missing_voice_matcher(tmp_path, monkeypatch):
    cfg = DialogueConfig(source="x.mp4", work_dir=tmp_path, output_dir=tmp_path,
                         tts_providers=["mock"], translation_engines=[], voice_match="openvoice")
    monkeypatch.setitem(sys.modules, "dubbing.dialogue.voice_match", None)    # not installed
    comps = default_components(cfg)
    assert comps.voice_matcher is None and "not installed" in comps.notes["voice_match"]
    fake = types.ModuleType("dubbing.dialogue.voice_match")
    fake.make_voice_matcher = lambda c: None                                   # cannot run here
    monkeypatch.setitem(sys.modules, "dubbing.dialogue.voice_match", fake)
    comps = default_components(cfg)
    assert comps.voice_matcher is None and "cannot run" in comps.notes["voice_match"]
    built = FakeMatcher()
    fake.make_voice_matcher = lambda c: built
    assert default_components(cfg).voice_matcher is built
    cfg.voice_match = "off"
    assert default_components(cfg).voice_matcher is None


# ── output options ────────────────────────────────────────────────────────
@needs_ffmpeg
def test_mux_gets_the_output_options(tmp_path, monkeypatch):
    real_mux = mix_mod.mux
    seen = []

    def fake_mux(video, mix_wav, out, bitrate="192k", subtitles=None, **kw):
        seen.append(dict(kw, out=Path(out), subtitles=subtitles))
        tmp = Path(out).with_name("muxed_for_test.mp4")
        real_mux(video, mix_wav, tmp, bitrate, subtitles=subtitles)
        tmp.replace(out)
        return Path(out)

    monkeypatch.setattr(mix_mod, "mux", fake_mux)
    orch, _ = _orch(tmp_path, cfg={"keep_original_audio": True, "burn_subtitles": True,
                                   "container": "mkv"})
    res = orch.run()
    out = tmp_path / "out"
    assert res.status == "completed", res.reasons
    assert len(seen) == 1
    kw = seen[0]
    assert kw["out"] == out / "dubbed_hi.mkv" and res.output_video == out / "dubbed_hi.mkv"
    assert kw["subtitles"] == out / "subtitles_hi.srt"
    assert kw["original_audio"] is True and kw["container"] == "mkv"
    assert kw["burn_subtitles"] == out / "subtitles_hi.srt"
    assert kw["extra_subtitles"] == [(out / "subtitles_en.srt", "eng", "English")]
    en = (out / "subtitles_en.srt").read_text(encoding="utf-8")
    assert en.count("-->") == 4 and "[SPEAKER" not in en and "00:00:00,300 --> 00:00:02,300" in en

    # defaults: mp4, no original audio, no burn; English off -> no English track or file
    seen.clear()
    orch2, _ = _orch(tmp_path, cfg={"english_subtitles": False})
    res2 = orch2.run()
    assert seen[0]["out"] == out / "dubbed_hi.mp4" and not (out / "dubbed_hi.mkv").exists()
    assert seen[0]["original_audio"] is False and seen[0]["burn_subtitles"] is None
    assert seen[0]["extra_subtitles"] is None and seen[0]["container"] == "mp4"
    assert not (out / "subtitles_en.srt").exists() and res2.status == "completed"


# ── units: registry overrides, caches, checkpoint identity, CLI ───────────
def _reg():
    reg = SpeakerRegistry()
    reg.register("A", voice_category="male_like", total_speech_s=10)
    reg.register("B", voice_category="male_like", total_speech_s=5)
    reg.register("C", voice_category="female_like", total_speech_s=3)
    reg.bind_provider("mock")
    return reg


def test_category_override_repicks_a_free_voice_and_keeps_the_others():
    reg = _reg()
    got = reg.override_category("B", "female_like", ["mock"])
    assert got["mock"]["voice"] == "mock-female-2" and got["mock"]["override"] is True
    assert reg.resolve_voice("B", "mock")["voice"] == "mock-female-2"
    assert reg.speakers["A"].provider_voices["mock"]["voice"] == "mock-male-1"
    assert reg.speakers["C"].provider_voices["mock"]["voice"] == "mock-female-1"
    b = reg.speakers["B"]
    assert b.voice_category == "female_like" and b.mapping_origin == "user"
    assert b.category_evidence["user_override"] == {"from": "male_like", "to": "female_like"}
    # no free female voice left: the voice is shared and marked as such
    got = reg.override_category("A", "female_like", ["mock"])
    assert got["mock"]["indistinguishable_reuse"] is True
    # a speaker already in that category keeps its voice
    before = dict(reg.speakers["C"].provider_voices["mock"])
    got = reg.override_category("C", "female_like", ["mock"])
    assert got["mock"]["voice"] == before["voice"]


def test_explicit_voice_override():
    reg = _reg()
    b = reg.override_voice("C", "mock", "mock-female-2|+20Hz")
    assert (b["voice"], b["pitch"]) == ("mock-female-2", "+20Hz") and "indistinguishable_reuse" not in b
    b = reg.override_voice("B", "mock", "mock-male-1")       # A's voice: allowed, but marked
    assert b["indistinguishable_reuse"] is True
    assert reg.speakers["A"].provider_voices["mock"]["voice"] == "mock-male-1"
    with pytest.raises(VoiceResolutionError):
        reg.override_voice("A", "indicf5", "not-a-curated-reference")
    with pytest.raises(VoiceResolutionError):
        reg.override_voice("NOBODY", "mock", "mock-male-1")
    gone = reg.merge_speaker("B", "A")
    assert gone.speaker_id == "B" and "B" not in reg.speakers
    assert reg.speakers["A"].total_speech_s == 15


def test_voice_options_shape():
    opts = SpeakerRegistry().voice_options("edge")
    assert opts["male_like"][0] == {"voice": "hi-IN-MadhurNeural", "pitch": None, "label": "Madhur"}
    assert {"voice": "hi-IN-MadhurNeural", "pitch": "+20Hz", "label": "Madhur +20Hz"} in opts["male_like"]
    assert opts["female_like"][0]["label"] == "Swara"


@needs_ffmpeg
def test_tts_cache_is_opt_in_and_regeneration_bypasses_it(tmp_path):
    prov = MockProvider()
    reg = _reg()
    t = Turn("t0001", "A", 0.0, 2.0, hi_fit="नमस्ते, आप कैसे हैं?")
    plain = TTSRouter({"mock": prov}, reg, ["mock"], tmp_path / "clips")
    plain.synthesize(t)
    plain.synthesize(t)
    assert len(prov.calls) == 2                               # no cache_dir: unchanged behaviour
    router = TTSRouter({"mock": prov}, reg, ["mock"], tmp_path / "clips2", cache_dir=tmp_path / "cache")
    a = router.synthesize(t)
    b = router.synthesize(t)
    assert len(prov.calls) == 3 and router.cache_hits == 1
    assert b.path != a.path and Path(b.path).exists() and b.voice == a.voice
    assert b.retry_history[-1] == {"reason": "initial", "cache": "hit"}
    router.synthesize(t, use_cache=False)                     # a regeneration
    assert len(prov.calls) == 4
    router.synthesize(t, speed=1.1)                           # another rate is another clip
    assert len(prov.calls) == 5
    router.uncache(b)
    router.synthesize(t)
    assert len(prov.calls) == 6


def test_rewrite_cache_rounds_the_ratio_and_persists(tmp_path):
    p = tmp_path / "rewrite_cache.json"
    cache = checkpoint.RewriteCache(p)
    cache.put("t0001", "लंबी पंक्ति", 0.52, "छोटी")
    assert cache.get("t0001", "लंबी पंक्ति", 0.51) == "छोटी"
    assert cache.get("t0001", "लंबी पंक्ति", 0.6) is None
    assert cache.get("t0002", "लंबी पंक्ति", 0.52) is None
    assert checkpoint.RewriteCache(p).get("t0001", "लंबी पंक्ति", 0.5) == "छोटी"


def test_checkpoint_identity_never_records_the_url(tmp_path):
    url = "https://www.youtube.com/watch?v=abcdefgh&sig=SECRET"
    ident = checkpoint.source_identity(url)
    assert "SECRET" not in json.dumps(ident) and "youtube" not in json.dumps(ident)
    assert ident != checkpoint.source_identity(url, limit_seconds=60)
    srt = tmp_path / "hi.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nनमस्ते\n", encoding="utf-8")
    a = checkpoint.source_identity(url, translated_srt=srt)
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nबदला\n", encoding="utf-8")
    assert a != checkpoint.source_identity(url, translated_srt=srt)


def test_cli_flags_and_review_hook(tmp_path, monkeypatch):
    from dubbing.dialogue import __main__ as cli
    captured = {}

    def fake_run(cfg, on_progress=None, review=None, **kw):
        captured.update(cfg=cfg, review=review)
        return SimpleNamespace(status="completed", reasons=[], output_video=None, report_md=None)

    monkeypatch.setattr(cli, "_load_env", lambda: None)
    monkeypatch.setattr(orch_mod, "run_dialogue", fake_run)
    edits = tmp_path / "edits.json"
    edits.write_text(json.dumps({"turn_edits": {"t0001": {"delete": True}},
                                 "speaker_merges": {"SPEAKER_02": "SPEAKER_01"}}), encoding="utf-8")
    rc = cli.main(["dub", "clip.mp4", "--out", str(tmp_path / "o"), "--review", "--resume",
                   "--edits", str(edits), "--keep-original-audio", "--no-english-subs",
                   "--burn-subs", "--container", "mkv"])
    cfg = captured["cfg"]
    assert rc == 0 and cfg.review_before_voice and cfg.resume
    assert cfg.keep_original_audio and not cfg.english_subtitles and cfg.burn_subtitles
    assert cfg.container == "mkv" and cfg.turn_edits == {"t0001": {"delete": True}}
    assert cfg.speaker_merges == {"SPEAKER_02": "SPEAKER_01"} and cfg.voice_overrides == {}

    hook = captured["review"]
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert hook({"turns": [], "speakers": []}) is None                 # no edits file: no changes
    cfg.work_dir.mkdir(parents=True)
    (cfg.work_dir / "review_edits.json").write_text(
        json.dumps({"turn_edits": {"t0002": {"hi": "हाँ।"}}}), encoding="utf-8")
    assert hook({"turns": [], "speakers": []}) == {"turn_edits": {"t0002": {"hi": "हाँ।"}}}

    captured.clear()
    cli.main(["dub", "clip.mp4", "--out", str(tmp_path / "o2")])
    cfg = captured["cfg"]
    assert captured["review"] is None and not cfg.resume and cfg.english_subtitles
    assert cfg.container == "mp4" and not cfg.keep_original_audio and not cfg.burn_subtitles
