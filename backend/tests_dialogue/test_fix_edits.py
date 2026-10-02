"""Edits, re-voicing and voice matching: the review findings fixed.

Same synthetic media and scripted backends as test_orchestrator_e2e.py.
Each test reproduces one finding: an override on the job's second TTS
provider, a voice of another category, a re-voice whose checkpoint cannot
be used, a deleted line restored, warnings a reviewer settled, re-sent edits,
merge chains, the rewrite cache, reference clips of a moved job folder, the
source identity, and voice-matching failures part way through a job.
"""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from dubbing.dialogue import checkpoint, fit
from dubbing.dialogue.contracts import Clip, Turn
from dubbing.dialogue.orchestrator import DialogueConfig, DialogueOrchestrator
from dubbing.dialogue.speaker_registry import SpeakerRegistry
from dubbing.dialogue.tts import MockProvider

from test_orchestrator_e2e import HAVE_FFMPEG, HINDI, FakeLLM, _components, _make_media
from test_review_resume import CHECKPOINTED, _diar3

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")

EN = ["Where were you last night?", "I was at the station.", "Why?",
      "Riya asked me to wait for 2 hours."]
NEW_T2 = "मैं पूरी रात स्टेशन पर थी।"


def _build(tmp_path, *, work="work", out="out", source=None, cfg=None, providers=None, llm=None,
           diar=None, voice_matcher=None, verifier_overrides=None, review=None):
    """An orchestrator on the shared synthetic media (made by the `media`
    fixture, never re-made here: a test may move or delete it)."""
    holder, calls = {}, {}
    comps = _components(calls, orch_holder=holder, verifier_overrides=verifier_overrides)
    if providers is not None:
        comps.tts_providers = dict(providers)
    if llm is not None:
        comps.llm_clients = [llm]
    if diar is not None:
        def diarize(wav):
            calls["diarize"] = calls.get("diarize", 0) + 1
            return diar
        comps.diarize = diarize
    comps.voice_matcher = voice_matcher
    c = DialogueConfig(source=str(source or tmp_path / "input.mp4"), work_dir=tmp_path / work,
                       output_dir=tmp_path / out, tts_providers=list(comps.tts_providers),
                       **(cfg or {}))
    orch = DialogueOrchestrator(c, comps, review=review)
    holder["orch"] = orch
    return orch, calls


@pytest.fixture()
def media(tmp_path):
    return _make_media(tmp_path)


def _report(res):
    return json.loads(res.report_json.read_text(encoding="utf-8"))


def _after_packet(tmp_path, out="out"):
    return json.loads((tmp_path / out / "review.json").read_text(encoding="utf-8"))


def _accepted(orch, speaker=None):
    return [c for c in orch.clips.values() if c.accepted and (speaker is None or c.speaker_id == speaker)]


def _entries(rep, kind=None):
    return [e for e in rep["applied_edits"] if kind is None or e["kind"] == kind]


# ── core-edits#1 / web#3: a voice on the job's second provider ────────────
@needs_ffmpeg
def test_voice_on_the_second_provider_pins_the_speaker_to_it(tmp_path, media):
    mock, sarvam = MockProvider(), MockProvider(name="sarvam")
    provs = {"mock": mock, "sarvam": sarvam}
    pick = {"SPEAKER_01": {"provider": "sarvam", "voice": "anushka", "pitch": None}}
    orch, _ = _build(tmp_path, providers=provs, cfg={"voice_overrides": pick})
    res = orch.run()
    rep = _report(res)
    assert res.status == "completed", res.reasons
    assert {(c.provider, c.voice, c.degraded) for c in _accepted(orch, "SPEAKER_01")} == \
        {("sarvam", "anushka", False)}
    assert {(c.provider, c.voice) for c in _accepted(orch, "SPEAKER_00")} == {("mock", "mock-male-1")}
    assert {c["voice"] for c in sarvam.calls} == {"anushka"} and len(sarvam.calls) == 2
    assert not [x for x in rep["limitations"] if "fallback provider" in x]
    (e,) = _entries(rep, "voice_override")
    assert not e.get("ignored") and e["provider_first"] == {"from": "mock", "to": "sarvam"}
    after = {s["speaker_id"]: s for s in _after_packet(tmp_path)["speakers"]}
    assert (after["SPEAKER_01"]["provider"], after["SPEAKER_01"]["voice"],
            after["SPEAKER_01"]["override"]) == ("sarvam", "anushka", True)
    assert after["SPEAKER_00"]["provider"] == "mock"
    state = json.loads(checkpoint.checkpoint_path(tmp_path / "work").read_text(encoding="utf-8"))
    assert state["registry"]["provider_pins"] == {"SPEAKER_01": "sarvam"}

    # the choice holds on a re-voice that does not send it again
    orch2, _ = _build(tmp_path, providers=provs, cfg={"resume": True})
    res2 = orch2.run()
    assert res2.status == "completed", res2.reasons
    assert {(c.provider, c.voice) for c in _accepted(orch2, "SPEAKER_01")} == {("sarvam", "anushka")}

    # a voice on the job's first provider again: back to it
    back = {"SPEAKER_01": {"provider": "mock", "voice": "mock-female-2", "pitch": None}}
    orch3, _ = _build(tmp_path, providers=provs, cfg={"resume": True, "voice_overrides": back})
    res3 = orch3.run()
    assert res3.status == "completed", res3.reasons
    assert {(c.provider, c.voice, c.degraded) for c in _accepted(orch3, "SPEAKER_01")} == \
        {("mock", "mock-female-2", False)}
    assert orch3.registry.provider_pins == {}


# ── core-edits#4: a voice of another category ─────────────────────────────
def test_a_voice_of_another_category_moves_the_speaker_and_its_other_voices():
    reg = SpeakerRegistry()
    reg.register("A", voice_category="male_like", total_speech_s=10)
    reg.register("B", voice_category="female_like", total_speech_s=5)
    for p in ("mock", "sarvam", "edge"):
        reg.bind_provider(p)
    reg.override_voice("A", "mock", "mock-female-2", providers=["mock", "sarvam"])
    a = reg.speakers["A"]
    assert a.voice_category == "female_like"
    assert a.category_evidence["user_override"] == {"from": "male_like", "to": "female_like"}
    assert a.provider_voices["sarvam"]["voice"] == "manisha"          # B has anushka
    assert a.provider_voices["edge"]["category_used"] == "male_like"   # not in `providers`: untouched
    # a voice of the speaker's own category leaves the other providers alone
    before = dict(reg.speakers["B"].provider_voices["sarvam"])
    reg.override_voice("B", "mock", "mock-female-1", providers=["mock", "sarvam"])
    assert reg.speakers["B"].provider_voices["sarvam"] == before
    # Edge's raised-pitch Swara is the child slot; a provider without child
    # voices keeps the woman's voice the speaker already has
    reg.override_voice("B", "edge", "hi-IN-SwaraNeural", "+25Hz", providers=["edge", "sarvam"])
    assert reg.speakers["B"].voice_category == "child_like"
    assert reg.speakers["B"].provider_voices["sarvam"] == before


@needs_ffmpeg
def test_fallback_after_a_voice_of_another_category_keeps_that_category(tmp_path, media):
    mock = MockProvider(fail=lambda text, b: b["voice"] == "mock-female-2")   # rejects the pick
    sarvam = MockProvider(name="sarvam")
    pick = {"SPEAKER_00": {"provider": "mock", "voice": "mock-female-2", "pitch": None}}
    orch, _ = _build(tmp_path, providers={"mock": mock, "sarvam": sarvam},
                     cfg={"voice_overrides": pick})
    res = orch.run()
    rep = _report(res)
    assert orch.registry.speakers["SPEAKER_00"].voice_category == "female_like"
    # the fallback provider speaks in a woman's voice, as the reviewer chose
    assert {(c.provider, c.voice) for c in _accepted(orch, "SPEAKER_00")} == {("sarvam", "manisha")}
    assert {(c.provider, c.voice) for c in _accepted(orch, "SPEAKER_01")} == {("mock", "mock-female-1")}
    assert [x for x in rep["unresolved_failures"]
            if x.startswith("your voice choice for SPEAKER_00 (mock: mock-female-2) could not be used")]
    assert res.status == "completed_with_warnings"
    (e,) = _entries(rep, "voice_override")
    assert e["category"]["from"] == "male_like" and e["category"]["to"] == "female_like"


# ── core-edits#2 / core-resume#3: a checkpoint that cannot be used ────────
@needs_ffmpeg
@pytest.mark.parametrize("breakage", ["work_file_missing", "other_version"])
def test_a_revoice_never_applies_old_ids_to_a_recomputed_transcript(tmp_path, media, breakage):
    prov = MockProvider()
    assert _build(tmp_path, providers={"mock": prov})[0].run().status == "completed"
    w = tmp_path / "work"
    if breakage == "work_file_missing":
        (w / "original_16k_mono.wav").unlink()
    else:
        p = checkpoint.checkpoint_path(w)
        state = json.loads(p.read_text(encoding="utf-8"))
        state["version"] = 0                       # an app update changed the format
        p.write_text(json.dumps(state), encoding="utf-8")
    kept = checkpoint.checkpoint_path(w).read_bytes()
    n = len(prov.calls)
    orch, calls = _build(tmp_path, providers={"mock": prov},
                         cfg={"resume": True, "turn_edits": {"t0002": {"delete": True}},
                              "speaker_merges": {"SPEAKER_01": "SPEAKER_00"}})
    res = orch.run()
    assert res.status == "failed"
    assert any("checkpoint cannot be used" in x and "Start a new job" in x for x in res.reasons), res.reasons
    assert calls == {} and len(prov.calls) == n          # no ASR, no diarization, nothing voiced
    assert checkpoint.checkpoint_path(w).read_bytes() == kept
    assert _report(res)["applied_edits"] == []


@needs_ffmpeg
def test_without_edits_an_unusable_checkpoint_still_means_a_full_run(tmp_path, media):
    prov = MockProvider()
    _build(tmp_path, providers={"mock": prov})[0].run()
    (tmp_path / "work" / "original_16k_mono.wav").unlink()
    orch, calls = _build(tmp_path, providers={"mock": prov}, cfg={"resume": True})
    res = orch.run()
    assert res.status == "completed", res.reasons
    assert calls == {"asr": 1, "diarize": 1} and orch.report.resumed_from_checkpoint == []
    assert any("audio_16k file is missing" in x for x in orch.report.limitations)


# ── core-resume#6: source identity and a checkpoint kept until replaced ───
def test_source_identity_is_normalised_and_accepts_older_checkpoints(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x" * 100)
    w = tmp_path / "w"
    w.mkdir()
    ident = checkpoint.source_identity(str(f))
    assert checkpoint.source_identity(str(w / ".." / "a.mp4")) == ident    # another spelling
    legacy = dict(ident, source_sha1=hashlib.sha1(str(f).encode("utf-8")).hexdigest())
    assert checkpoint.same_source(legacy, ident, w, source=str(f))           # written before the fix
    f.write_bytes(b"y" * 50)                                                 # another file there now
    assert not checkpoint.same_source(ident, checkpoint.source_identity(str(f)), w, source=str(f))
    url = "https://www.youtube.com/watch?v=abcdefgh&sig=SECRET"
    u = checkpoint.source_identity(url)
    assert checkpoint.same_source(u, checkpoint.source_identity(url), w, source=url)
    assert not checkpoint.same_source(u, checkpoint.source_identity(url + "x"), w, source=url + "x")


@needs_ffmpeg
def test_resume_finds_the_checkpoint_in_any_spelling_and_after_the_original_moved(tmp_path, media):
    prov = MockProvider()
    assert _build(tmp_path, providers={"mock": prov})[0].run().status == "completed"
    # the same file, spelled another way
    orch, calls = _build(tmp_path, providers={"mock": prov}, cfg={"resume": True},
                         source=tmp_path / "work" / ".." / "input.mp4")
    assert orch.run().status == "completed" and calls == {}
    assert orch.report.resumed_from_checkpoint == CHECKPOINTED
    # the original moved away: the job's own copy is all a resume reads
    moved = tmp_path / "moved.mp4"
    media.rename(moved)
    orch, calls = _build(tmp_path, providers={"mock": prov}, cfg={"resume": True}, source=media)
    res = orch.run()
    assert res.status == "completed", res.reasons
    assert calls == {} and orch.report.resumed_from_checkpoint == CHECKPOINTED
    # ... and under its new name it is still the same media
    orch, calls = _build(tmp_path, providers={"mock": prov}, cfg={"resume": True}, source=moved)
    assert orch.run().status == "completed" and calls == {}
    assert orch.report.resumed_from_checkpoint == CHECKPOINTED


@needs_ffmpeg
def test_a_mismatched_checkpoint_is_kept_when_the_full_run_cannot_start(tmp_path, media):
    prov = MockProvider()
    _build(tmp_path, providers={"mock": prov})[0].run()
    p = checkpoint.checkpoint_path(tmp_path / "work")
    kept = p.read_bytes()
    orch, _ = _build(tmp_path, providers={"mock": prov}, cfg={"resume": True},
                     source=tmp_path / "elsewhere" / "other.mp4")        # not there
    res = orch.run()
    assert res.status == "failed" and any("Input file not found" in x for x in res.reasons)
    assert any("different source" in x for x in orch.report.limitations)
    assert p.read_bytes() == kept                                       # the job can still be re-voiced
    assert (tmp_path / "work" / "tts_cache").is_dir() and any((tmp_path / "work" / "tts_cache").iterdir())


# ── core-edits#3: deleted lines, restore, "deleted" in the packet ─────────
@needs_ffmpeg
def test_a_deleted_line_is_marked_deleted_and_can_be_restored(tmp_path, media):
    prov = MockProvider()
    seen = {}

    def review(packet):
        seen["packet"] = packet
        return {"turn_edits": {"t0002": {"delete": True}}}

    orch1, _ = _build(tmp_path, providers={"mock": prov}, cfg={"review_before_voice": True},
                      review=review)
    assert orch1.run().status == "completed"
    assert [t["deleted"] for t in seen["packet"]["turns"]] == [False] * 4
    after = {t["turn_id"]: t for t in _after_packet(tmp_path)["turns"]}
    assert after["t0002"]["deleted"] is True and after["t0002"]["clip"] is None
    assert "deleted_by_user" in after["t0002"]["flags"] and after["t0001"]["deleted"] is False

    # new Hindi typed for the line, which stays deleted: not voiced, reported as ignored
    n = len(prov.calls)
    orch2, _ = _build(tmp_path, providers={"mock": prov},
                      cfg={"resume": True, "turn_edits": {"t0002": {"delete": True, "hi": NEW_T2}}})
    res2 = orch2.run()
    rep2 = _report(res2)
    assert res2.status == "completed", res2.reasons
    assert "t0002" not in orch2.clips and len(prov.calls) == n
    t2 = next(t for t in orch2.turns if t.turn_id == "t0002")
    assert t2.speech_text == HINDI[EN[1]] and "edited" not in t2.flags
    (e,) = [x for x in _entries(rep2, "turn_edit") if x["turn_id"] == "t0002"]
    assert e["changes"] == {"delete": True}
    assert any("line is deleted; restore it first" in x for x in e["notes"])
    assert [s for s in rep2["stages"] if s["name"] == "review"][0]["detail"].startswith("0 edit(s) applied")
    # the same with only the Hindi (no delete flag sent): the whole edit is ignored
    orch2b, _ = _build(tmp_path, providers={"mock": prov},
                       cfg={"resume": True, "turn_edits": {"t0002": {"hi": NEW_T2}}})
    orch2b.run()
    assert "t0002" not in orch2b.clips and len(prov.calls) == n

    # restored, with that Hindi: voiced again
    orch3, _ = _build(tmp_path, providers={"mock": prov},
                      cfg={"resume": True, "turn_edits": {"t0002": {"delete": False, "hi": NEW_T2}}})
    res3 = orch3.run()
    rep3 = _report(res3)
    assert res3.status == "completed", res3.reasons
    assert orch3.clips["t0002"].accepted and orch3.clips["t0002"].spoken_text == NEW_T2
    after3 = {t["turn_id"]: t for t in _after_packet(tmp_path)["turns"]}
    assert after3["t0002"]["deleted"] is False and after3["t0002"]["required"] is True
    assert after3["t0002"]["clip"] == "t0002.wav"
    (e3,) = [x for x in _entries(rep3, "turn_edit") if x["turn_id"] == "t0002"]
    assert e3["changes"]["delete"] is False and e3["changes"]["hi"]["to"] == NEW_T2
    assert "t0002" in rep3["required_turn_ids"] and rep3["missing_turns"] == []


# ── core-edits#5: warnings the reviewer settled ───────────────────────────
class DropsTheNumber(FakeLLM):
    """Translates t0004 without its "2": a critical-token warning."""

    def complete(self, system, user):
        return super().complete(system, user).replace("2 घंटे", "घंटों")


class BriefSaysMale(FakeLLM):
    """The story brief calls SPEAKER_01 a man; her voice says otherwise."""

    def complete(self, system, user):
        out = super().complete(system, user)
        if "transcript" in json.loads(user):
            d = json.loads(out)
            d["speakers"]["SPEAKER_01"]["gender"] = "male"
            out = json.dumps(d)
        return out


@needs_ffmpeg
def test_a_deleted_or_rewritten_line_settles_its_translation_warning(tmp_path, media):
    prov, llm = MockProvider(), DropsTheNumber()
    res1 = _build(tmp_path, providers={"mock": prov}, llm=llm)[0].run()
    assert res1.status == "completed_with_warnings" and res1.reasons == ["1 translation warning(s)"]

    def revoice(edits):
        orch, _ = _build(tmp_path, providers={"mock": prov}, llm=llm, cfg={"resume": True, **edits})
        res = orch.run()
        crit = [w for w in _report(res)["translation_warnings"] if w["type"] == "critical_tokens"]
        return orch, res, crit

    orch, res, crit = revoice({"turn_edits": {"t0004": {"delete": True}}})
    assert res.status == "completed", res.reasons
    assert len(crit) == 1 and crit[0]["resolved"] == "line deleted by the reviewer"
    # restored: its translation is back, and so is the warning
    orch, res, crit = revoice({"turn_edits": {"t0004": {"delete": False}}})
    assert res.status == "completed_with_warnings" and not crit[0].get("resolved")
    # rewritten by the reviewer: settled, and the line no longer carries the flag
    orch, res, crit = revoice({"turn_edits": {"t0004": {"hi": HINDI[EN[3]]}}})
    assert res.status == "completed", res.reasons
    assert crit[0]["resolved"] == "line rewritten by the reviewer"
    t4 = next(t for t in orch.turns if t.turn_id == "t0004")
    assert "critical_token_warning" not in t4.flags and "edited" in t4.flags


@needs_ffmpeg
def test_a_category_set_to_the_transcript_gender_settles_the_disagreement(tmp_path, media):
    prov, llm = MockProvider(), BriefSaysMale()
    res1 = _build(tmp_path, providers={"mock": prov}, llm=llm)[0].run()
    w1 = [w for w in _report(res1)["content_warnings"] if w["type"] == "speaker_gender_disagreement"]
    assert res1.status == "completed_with_warnings" and len(w1) == 1
    assert w1[0]["speaker_id"] == "SPEAKER_01" and w1[0]["voice_category"] == "female_like"
    orch, _ = _build(tmp_path, providers={"mock": prov}, llm=llm,
                     cfg={"resume": True, "voice_overrides": {"SPEAKER_01": {"category": "male_like"}}})
    res2 = orch.run()
    assert res2.status == "completed", res2.reasons
    assert not [w for w in _report(res2)["content_warnings"] if w["type"] == "speaker_gender_disagreement"]


# ── core-edits#7: one entry per edit, re-sent edits are not "ignored" ─────
@needs_ffmpeg
def test_resent_edits_keep_one_entry_each_and_are_not_reported_ignored(tmp_path, media):
    prov = MockProvider()
    edited = "कल रात तुम किसके साथ थे?"
    edits = {"turn_edits": {"t0003": {"speaker_id": "SPEAKER_02"}, "t0001": {"hi": edited}},
             "speaker_merges": {"SPEAKER_02": "SPEAKER_01"},
             "voice_overrides": {"SPEAKER_00": {"provider": "mock", "voice": "mock-male-2", "pitch": None}}}
    orch1, _ = _build(tmp_path, providers={"mock": prov}, diar=_diar3(), cfg=dict(edits))
    rep1 = _report(orch1.run())
    assert len(rep1["applied_edits"]) == 4 and not [e for e in rep1["applied_edits"] if e.get("ignored")]
    first = {(e["kind"], e.get("turn_id") or e.get("from") or e.get("speaker_id")): e
             for e in rep1["applied_edits"]}

    # the API sends every earlier edit again on each re-voice, here with one new line
    again = dict(edits, turn_edits={**edits["turn_edits"], "t0002": {"hi": NEW_T2}})
    for applied, in_effect in ((1, 4), (0, 5)):
        orch, _ = _build(tmp_path, providers={"mock": prov}, diar=_diar3(), cfg={"resume": True, **again})
        res = orch.run()
        rep = _report(res)
        keys = [(e["kind"], e.get("turn_id") or e.get("from") or e.get("speaker_id"))
                for e in rep["applied_edits"]]
        assert len(keys) == len(set(keys)) == 5, rep["applied_edits"]
        assert not [e for e in rep["applied_edits"] if e.get("ignored")], rep["applied_edits"]
        for k, e in first.items():                     # still the first run's own record
            assert rep["applied_edits"][keys.index(k)] == e
        review = [s for s in rep["stages"] if s["name"] == "review"][0]
        assert review["detail"] == f"{applied} edit(s) applied; {in_effect} earlier edit(s) already in effect"
        assert {t.turn_id: t.speaker_id for t in orch.turns}["t0003"] == "SPEAKER_01"
        assert orch.clips["t0002"].spoken_text == NEW_T2


# ── core-edits#8: merge chains ────────────────────────────────────────────
@needs_ffmpeg
@pytest.mark.parametrize("merges", [{"SPEAKER_02": "SPEAKER_01", "SPEAKER_00": "SPEAKER_02"},
                                    {"SPEAKER_00": "SPEAKER_02", "SPEAKER_02": "SPEAKER_01"}])
def test_merge_chains_do_not_depend_on_their_order(tmp_path, media, merges):
    orch, _ = _build(tmp_path, diar=_diar3(), cfg={"speaker_merges": merges})
    rep = _report(orch.run())
    assert {t.speaker_id for t in orch.turns} == {"SPEAKER_01"}
    assert set(orch.registry.speakers) == {"SPEAKER_01"}
    assert len({c.voice for c in _accepted(orch)}) == 1
    merged = {e["from"]: e for e in _entries(rep, "speaker_merge")}
    assert not [e for e in merged.values() if e.get("ignored")]
    assert merged["SPEAKER_00"]["into"] == "SPEAKER_01" and merged["SPEAKER_00"]["requested_into"] == "SPEAKER_02"


@needs_ffmpeg
def test_a_merge_cycle_is_ignored(tmp_path, media):
    orch, _ = _build(tmp_path, cfg={"speaker_merges": {"SPEAKER_00": "SPEAKER_01",
                                                       "SPEAKER_01": "SPEAKER_00"}})
    rep = _report(orch.run())
    assert set(orch.registry.speakers) == {"SPEAKER_00", "SPEAKER_01"}
    assert [e["ignored"] for e in _entries(rep, "speaker_merge")] == ["these merges form a cycle"] * 2


# ── core-resume#2: the rewrite cache never hands back a rejected rewrite ──
def test_a_rejected_rewrite_is_neither_served_again_nor_cached(tmp_path):
    cfg = DialogueConfig(source="x.mp4", work_dir=tmp_path / "work", output_dir=tmp_path / "out",
                         tts_providers=["mock"])
    cfg.work_dir.mkdir()
    orch = DialogueOrchestrator(cfg, _components({}))
    orig = "यह एक बहुत लंबी पंक्ति है जो अपनी जगह में नहीं आती"
    longer, short = "यह पंक्ति अपनी जगह में नहीं आती", "छोटी पंक्ति"
    natural = {orig: 4.0, longer: 4.5, short: 2.0}      # the first rewrite speaks LONGER
    asked = []

    def llm(t, current_hi, ratio):
        asked.append(current_hi)
        return [longer, short][len(asked) - 1] if len(asked) <= 2 else None

    def clip(t, text):
        return Clip(clip_id=f"c{len(asked)}_{t.turn_id}", turn_id=t.turn_id, speaker_id=t.speaker_id,
                    provider="mock", voice="v", spoken_text=text, path="unused.wav",
                    natural_duration=natural.get(text, 0.5), accepted=True)

    def run_fit(rewrite):
        t1 = Turn("t0001", "SPEAKER_00", 0.0, 2.0, hi_raw=orig, hi_fit=orig, hi_display=orig)
        t2 = Turn("t0002", "SPEAKER_00", 2.5, 3.5, hi_raw="हाँ", hi_fit="हाँ", hi_display="हाँ")
        clips = {"t0001": clip(t1, orig), "t0002": clip(t2, "हाँ")}
        cached = orch._cached_rewrite(rewrite)
        devs = fit.fit_all([t1, t2], clips, 10.0, lambda t, why: clip(t, t.speech_text), cached,
                           fit.FitConfig(max_stretch=1.15, max_rewrites=2))
        getattr(cached, "commit", lambda turns: None)([t1, t2])
        return t1, devs

    t1, devs = run_fit(llm)
    assert asked == [orig, orig]                        # the second request reached the LLM
    assert t1.speech_text == short and not [d for d in devs if d.get("severity") == "draft"]
    cache = checkpoint.RewriteCache(checkpoint.rewrite_cache_path(cfg.work_dir))
    assert list(cache.data.values()) == [short]         # only the rewrite the fit kept

    # a re-voice of the job: the kept rewrite comes from the cache, no LLM call
    t1b, _ = run_fit(lambda t, c, r: pytest.fail("the LLM was asked again"))
    assert t1b.speech_text == short


def test_a_cached_rewrite_the_fit_rejects_is_forgotten(tmp_path):
    cfg = DialogueConfig(source="x.mp4", work_dir=tmp_path / "work", output_dir=tmp_path / "out",
                         tts_providers=["mock"])
    cfg.work_dir.mkdir()
    orch = DialogueOrchestrator(cfg, _components({}))
    t = Turn("t0001", "SPEAKER_00", 0.0, 2.0, hi_raw="लंबी पंक्ति", hi_fit="लंबी पंक्ति")
    checkpoint.RewriteCache(checkpoint.rewrite_cache_path(cfg.work_dir)).put(
        "t0001", "लंबी पंक्ति", 0.6, "पुरानी")
    answers = iter(["नई"])
    cached = orch._cached_rewrite(lambda t, c, r: next(answers))
    assert cached(t, "लंबी पंक्ति", 0.6) == "पुरानी"     # once from the cache ...
    assert cached(t, "लंबी पंक्ति", 0.6) == "नई"         # ... asked again: the LLM
    t.hi_fit = "नई"
    cached.commit([t])
    assert checkpoint.RewriteCache(checkpoint.rewrite_cache_path(cfg.work_dir)).get(
        "t0001", "लंबी पंक्ति", 0.6) == "नई"


# ── core-resume#5: reference clips of a moved job folder ──────────────────
@needs_ffmpeg
def test_reference_clips_survive_a_moved_job_folder(tmp_path, media):
    prov = MockProvider()
    assert _build(tmp_path, providers={"mock": prov})[0].run().status == "completed"
    state = json.loads(checkpoint.checkpoint_path(tmp_path / "work").read_text(encoding="utf-8"))
    assert state["registry"]["speakers"]["SPEAKER_00"]["reference_clip"] == "speaker_refs/SPEAKER_00.wav"
    (tmp_path / "moved").mkdir()
    shutil.move(str(tmp_path / "work"), str(tmp_path / "moved" / "work"))
    shutil.move(str(tmp_path / "out"), str(tmp_path / "moved" / "out"))
    orch, calls = _build(tmp_path, work="moved/work", out="moved/out", providers={"mock": prov},
                         cfg={"resume": True})
    res = orch.run()
    assert res.status == "completed", res.reasons
    assert calls == {} and orch.report.resumed_from_checkpoint == CHECKPOINTED
    spk = {s["speaker_id"]: s for s in _after_packet(tmp_path, "moved/out")["speakers"]}
    for sid in ("SPEAKER_00", "SPEAKER_01"):
        assert spk[sid]["reference_clip"] == f"ref_{sid}.wav"
        assert (tmp_path / "moved" / "out" / "clips" / f"ref_{sid}.wav").is_file()


# ── mux-voicematch#1: voice matching that fails part way ──────────────────
class RealisticMatcher:
    """Behaves like OpenVoiceMatcher: never raises. After `works_for`
    conversions its worker is gone (CUDA out of memory): convert() returns
    False and the reason is in `errors`; a speaker it cannot prepare gets
    its reason in `notes`."""
    name = "openvoice"

    def __init__(self, works_for=99, refuse=None):
        self.works_for, self.refuse = works_for, dict(refuse or {})
        self.notes, self.errors, self.calls = {}, [], []

    def prepare(self, speaker_id, reference_wavs):
        if speaker_id in self.refuse:
            self.notes[speaker_id] = self.refuse[speaker_id]
            return False
        return True

    def convert(self, in_wav, out_wav, speaker_id):
        self.calls.append((Path(in_wav).name, speaker_id))
        if len(self.calls) > self.works_for:
            msg = f"{speaker_id}: openvoice: RuntimeError: CUDA out of memory"
            if msg not in self.errors:
                self.errors.append(msg)
            return False
        shutil.copyfile(in_wav, out_wav)
        return True

    def close(self):
        pass


def _matched_by_speaker(orch):
    out = {}
    for c in _accepted(orch):
        out.setdefault(c.speaker_id, set()).add(bool(c.voice_params.get("voice_matched")))
    return out


@needs_ffmpeg
def test_voice_matching_failing_mid_synthesis_keeps_one_voice_per_speaker(tmp_path, media):
    vm = RealisticMatcher(works_for=1)
    orch, _ = _build(tmp_path, voice_matcher=vm, cfg={"voice_match": "openvoice"})
    res = orch.run()
    rep = _report(res)
    by_spk = _matched_by_speaker(orch)
    assert all(len(v) == 1 for v in by_spk.values()), by_spk         # never two voices for one speaker
    assert by_spk == {"SPEAKER_00": {False}, "SPEAKER_01": {False}}
    lim = " | ".join(rep["limitations"])
    assert "of SPEAKER_0" in lim and "lines keep the stock voice (one voice per speaker)" in lim
    assert "CUDA out of memory" in lim                                 # the matcher's own reason
    assert rep["model_versions"]["voice_match"].startswith("openvoice (not applied")
    for c in _accepted(orch):
        assert "_vm" not in Path(c.path).name


@needs_ffmpeg
def test_voice_matching_failing_after_the_fit_is_undone_for_that_speaker(tmp_path, media):
    # 4 lines converted; the verifier then regenerates t0001 (SPEAKER_00) and the
    # converter has died: SPEAKER_00 goes back to the stock voice, SPEAKER_01 keeps its match
    vm = RealisticMatcher(works_for=4)
    orch, _ = _build(tmp_path, voice_matcher=vm, cfg={"voice_match": "openvoice"},
                     verifier_overrides={"t0001": "बिल्कुल अलग बात"})
    res = orch.run()
    rep = _report(res)
    assert len(vm.calls) == 6                                         # 4 + the regeneration + one retry
    assert _matched_by_speaker(orch) == {"SPEAKER_00": {False}, "SPEAKER_01": {True}}
    t3 = orch.clips["t0003"]
    assert t3.voice_params.get("voice_match_undone") and "_vm" not in Path(t3.path).name
    assert any("failed on 1 of SPEAKER_00's 2 line(s)" in x for x in rep["limitations"])
    assert rep["model_versions"]["voice_match"] == "openvoice"
    assert not [x for x in rep["unresolved_failures"] if x.startswith("voice_match_mixed")]


@needs_ffmpeg
def test_voice_matching_reports_the_matchers_own_reason_for_a_speaker(tmp_path, media):
    vm = RealisticMatcher(refuse={"SPEAKER_01": "reference could not be analysed: worker could not start"})
    orch, _ = _build(tmp_path, voice_matcher=vm, cfg={"voice_match": "openvoice"})
    res = orch.run()
    assert res.status == "completed", res.reasons
    assert _matched_by_speaker(orch) == {"SPEAKER_00": {True}, "SPEAKER_01": {False}}
    assert any("SPEAKER_01 (reference could not be analysed: worker could not start)" in x
               for x in orch.report.limitations)
