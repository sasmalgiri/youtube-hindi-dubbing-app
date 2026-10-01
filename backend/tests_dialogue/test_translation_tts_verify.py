"""Translation contracts, TTS identity under retries/fallback, fit, verification, status."""
import json

import pytest

from dubbing.dialogue import fit, report, verify
from dubbing.dialogue.contracts import (CATEGORY_FEMALE, CATEGORY_MALE,
                                        STATUS_COMPLETED,
                                        STATUS_COMPLETED_WITH_WARNINGS,
                                        STATUS_DRAFT_INCOMPLETE, Clip,
                                        JobReport, Turn)
from dubbing.dialogue.speaker_registry import SpeakerRegistry
from dubbing.dialogue.text_checks import (compare_hindi, english_numbers,
                                          translation_critical_issues)
from dubbing.dialogue.translation import (DialogueTranslator,
                                          validate_translations)
from dubbing.dialogue.tts import MockProvider, TTSRouter, split_for_limit


# ── translation validation ─────────────────────────────────────────────────
def test_validation_detects_missing_duplicate_unknown_and_context_echo():
    payload = {"translations": [
        {"id": "t0002", "hi": "हाँ।"},
        {"id": "t0003", "hi": "एक"}, {"id": "t0003", "hi": "दो"},   # conflicting duplicate
        {"id": "t0009", "hi": "???"},                              # unknown
        {"id": "t0001", "hi": "संदर्भ"},                            # context echo
    ]}
    good, errors = validate_translations(payload, ["t0002", "t0003", "t0004"], ["t0001"])
    assert set(good) == {"t0002"}
    types = {(e["type"], e.get("id")) for e in errors}
    assert ("duplicate_id_conflict", "t0003") in types
    assert ("unknown_id", "t0009") in types
    assert ("missing_id", "t0004") in types
    assert ("context_echo_ignored", "t0001") in types


class ScriptedLLM:
    """Replays JSON responses; records requests."""
    model_id = "fake:llm"

    def __init__(self, responder):
        self.responder = responder
        self.requests = []

    def complete(self, system, user):
        req = json.loads(user)
        self.requests.append(req)
        return json.dumps(self.responder(req), ensure_ascii=False)


def _turns(n, spk=("A", "B")):
    return [Turn(f"t{i:04d}", spk[i % 2], i * 2.0, i * 2.0 + 1.5, source_text=f"Line {chr(96 + i)} here.")
            for i in range(1, n + 1)]


def test_translator_retries_only_missing_ids_and_never_emits_context_twice():
    calls = {"n": 0}

    def responder(req):
        calls["n"] += 1
        ids = [t["id"] for t in req["turns"]]
        if calls["n"] == 1:   # drop the last id on the first call
            ids = ids[:-1]
        return {"translations": [{"id": i, "hi": f"पंक्ति {i}"} for i in ids]}

    llm = ScriptedLLM(responder)
    turns = _turns(5)
    tr = DialogueTranslator([llm], batch_size=5, allow_basic_fallback=False)
    tr.translate(turns)
    assert all(t.hi_raw == f"पंक्ति {t.turn_id}" for t in turns)
    assert [t["id"] for t in llm.requests[1]["turns"]] == ["t0005"]
    assert any(w["type"] == "missing_id" for w in tr.warnings)


def test_translator_batches_carry_context_but_translate_each_turn_once():
    llm = ScriptedLLM(lambda req: {"translations": [{"id": t["id"], "hi": "ठीक"} for t in req["turns"]]})
    turns = _turns(7)
    DialogueTranslator([llm], batch_size=3, allow_basic_fallback=False).translate(turns)
    translated = [t["id"] for r in llm.requests for t in r["turns"]]
    assert sorted(translated) == [t.turn_id for t in turns]   # each exactly once
    assert llm.requests[1]["context_before"], "second batch must see earlier dialogue"
    assert all(c["id"] not in [t["id"] for t in llm.requests[1]["turns"]]
               for c in llm.requests[1]["context_before"])


def test_voice_hint_is_passed_only_as_hint_and_speaker_ids_are_sent():
    llm = ScriptedLLM(lambda req: {"translations": [{"id": t["id"], "hi": "हाँ"} for t in req["turns"]]})
    turns = _turns(2)
    DialogueTranslator([llm], allow_basic_fallback=False).translate(
        turns, {"B": CATEGORY_MALE, "A": "unknown"})
    sent = llm.requests[0]["turns"]
    assert sent[0]["speaker"] == "B" and sent[0]["voice_hint"] == CATEGORY_MALE
    assert "voice_hint" not in sent[1]


def test_no_llm_uses_flagged_basic_fallback():
    turns = _turns(2)
    tr = DialogueTranslator([], basic_fallback=lambda s: "अनुवाद")
    tr.translate(turns)
    assert all("non_contextual_translation" in t.flags for t in turns)


def test_translation_failure_is_flagged_not_invented():
    turns = _turns(1)
    tr = DialogueTranslator([], allow_basic_fallback=False)
    tr.translate(turns)
    assert turns[0].hi_raw == "" and "translation_failed" in turns[0].flags


def test_critical_token_retry_fixes_dropped_number():
    state = {"n": 0}

    def responder(req):
        state["n"] += 1
        if "reviewer_note" in req:
            return {"translations": [{"id": "t0001", "hi": "मुझे 3 टिकट चाहिए।"}]}
        return {"translations": [{"id": "t0001", "hi": "मुझे टिकट चाहिए।"}]}

    t = Turn("t0001", "A", 0, 2, source_text="I need 3 tickets.")
    DialogueTranslator([ScriptedLLM(responder)], allow_basic_fallback=False).translate([t])
    assert "3" in t.hi_raw and "critical_token_warning" not in t.flags


def test_rewrite_shorter_rejects_dropping_facts():
    t = Turn("t0001", "A", 0, 2, source_text="I will not pay 500 rupees.")
    bad = ScriptedLLM(lambda req: {"hi": "मैं पैसे दूँगा।"})
    good = ScriptedLLM(lambda req: {"hi": "500 रुपये नहीं दूँगा।"})
    current = "मैं 500 रुपये बिल्कुल नहीं दूँगा, समझे आप?"
    assert DialogueTranslator([bad]).rewrite_shorter(t, current, 0.7) is None
    assert DialogueTranslator([good]).rewrite_shorter(t, current, 0.7) == "500 रुपये नहीं दूँगा।"


# ── text checks ──────────────────────────────────────────────────────────
def test_same_word_count_different_words_is_detected():
    exp = "वह कल दिल्ली नहीं जाएगा"
    heard = "वह कल मुंबई भी जाएगा"          # same count, different meaning
    cmp = compare_hindi(exp, heard)
    assert cmp["expected_words"] == cmp["heard_words"]
    assert cmp["wer"] > 0.25
    assert any(c["type"] == "negation_differs" for c in cmp["critical"])


def test_hindi_normalisation_tolerates_legit_variants():
    cmp = compare_hindi("मैं ज़रूर आऊँगा। २ बजे।", "मैं जरूर आऊंगा, 2 बजे")
    assert cmp["wer"] == 0 and not cmp["critical"]


def test_numbers_and_negation_translation_checks():
    assert english_numbers("twenty-five people and 3,000 rupees") >= {25.0, 3000.0}
    issues = translation_critical_issues("I don't have 20 dollars", "मेरे पास डॉलर हैं")
    assert {i["type"] for i in issues} == {"number_missing", "negation_missing"}
    assert translation_critical_issues("I don't have 20 dollars", "मेरे पास 20 डॉलर नहीं हैं") == []


# ── TTS router ───────────────────────────────────────────────────────────
def _reg():
    reg = SpeakerRegistry()
    reg.register("M", voice_category=CATEGORY_MALE, total_speech_s=10)
    reg.register("F", voice_category=CATEGORY_FEMALE, total_speech_s=8)
    return reg


def test_female_turn_failing_initially_retries_in_female_voice(tmp_path):
    fails = {"n": 0}

    def fail(text, binding):
        if binding["voice"] == "mock-female-1" and fails["n"] < 2:
            fails["n"] += 1
            return True
        return False
    prov = MockProvider(fail=fail)
    reg = _reg()
    router = TTSRouter({"mock": prov}, reg, ["mock"], tmp_path, max_retries=2)
    clip = router.synthesize(Turn("t0001", "F", 0, 2, hi_fit="नमस्ते, आप कैसे हैं?"))
    assert clip.voice == "mock-female-1" and clip.speaker_id == "F"
    assert [c["voice"] for c in prov.calls] == ["mock-female-1"] * 3
    assert not clip.degraded


def test_provider_fallback_preserves_speaker_and_reroutes_whole_speaker(tmp_path):
    primary = MockProvider(name="mock", fail=lambda text, b: "गिरो" in text)
    backup = MockProvider(name="mock2")
    reg = _reg()
    reg.pools["mock2"] = {"model": "m2", "supports_pitch": False,
                          CATEGORY_MALE: ["b-male"], CATEGORY_FEMALE: ["b-female"]}
    router = TTSRouter({"mock": primary, "mock2": backup}, reg, ["mock", "mock2"], tmp_path, max_retries=1)
    turns = {"t1": Turn("t1", "F", 0, 1, hi_fit="पहली बात"),
             "t2": Turn("t2", "F", 2, 3, hi_fit="गिरो मत"),
             "t3": Turn("t3", "M", 4, 5, hi_fit="ठीक है")}
    clips = {tid: router.synthesize(t) for tid, t in turns.items()}
    assert clips["t2"].provider == "mock2" and clips["t2"].voice == "b-female" and clips["t2"].degraded
    assert reg.speakers["F"].fallback_history
    unresolved = router.reroute_mixed_speakers(turns, clips)
    assert unresolved == []
    assert clips["t1"].accepted and clips["t1"].degraded   # rerouted clip usable + labelled
    assert {c.voice for tid, c in clips.items() if c.speaker_id == "F"} == {"b-female"}
    assert clips["t3"].provider == "mock"          # other speakers untouched


def test_long_text_split_never_loses_text():
    text = "यह पहला वाक्य है। " * 200
    parts = split_for_limit(text.strip(), 500)
    assert all(len(p) <= 500 for p in parts)
    assert "".join(parts).replace(" ", "") == text.strip().replace(" ", "")


# ── fit ──────────────────────────────────────────────────────────────────
def test_windows_use_immutable_source_times_and_skip_overlap_partners():
    a = Turn("t1", "A", 0.0, 2.0)
    b = Turn("t2", "B", 1.5, 2.5)
    c = Turn("t3", "A", 4.0, 5.0)
    a.overlaps_with, b.overlaps_with = ["t2"], ["t1"]
    w = fit.compute_windows([a, b, c], 10.0, guard_s=0.05)
    assert w["t1"] == (0.0, pytest.approx(3.95))   # t2 is an overlap partner, next is t3
    assert w["t3"] == (4.0, pytest.approx(9.95))


def test_plan_speed_bounds():
    assert fit.plan_speed(1.0, 2.0, 1.15) == (1.0, 0.0)
    s, o = fit.plan_speed(2.2, 2.0, 1.15)
    assert s == pytest.approx(1.1) and o == 0.0
    s, o = fit.plan_speed(3.0, 2.0, 1.15)
    assert s == 1.15 and o == pytest.approx(3.0 / 1.15 - 2.0)


def test_overlong_turn_gets_faithful_rewrite_before_stretch(tmp_path):
    reg = _reg()
    prov = MockProvider(seconds_per_char=0.1)
    router = TTSRouter({"mock": prov}, reg, ["mock"], tmp_path)
    t = Turn("t1", "M", 0.0, 1.0, hi_fit="बहुत लंबी पंक्ति जो समय में नहीं आती")
    nxt = Turn("t2", "F", 1.2, 2.0, hi_fit="हाँ")
    clips = {x.turn_id: router.synthesize(x) for x in (t, nxt)}
    before = t.source_start, t.source_end
    devs = fit.fit_all([t, nxt], clips, 5.0, lambda tt, r: router.synthesize(tt, r),
                       lambda tt, cur, ratio: "छोटी पंक्ति", fit.FitConfig())
    assert t.hi_fit == "छोटी पंक्ति" and "duration_rewrite" in t.flags
    assert t.hi_display == "छोटी पंक्ति" and t.hi_raw == ""   # subtitle follows speech
    assert (t.source_start, t.source_end) == before
    assert clips["t1"].scheduled_end <= 1.15 + 1e-6
    assert not [d for d in devs if d.get("severity") in ("warning", "draft")]


def test_unfittable_turn_is_kept_whole_and_reported(tmp_path):
    reg = _reg()
    router = TTSRouter({"mock": MockProvider(seconds_per_char=0.2)}, reg, ["mock"], tmp_path)
    t = Turn("t1", "M", 0.0, 1.0, hi_fit="यह बहुत लंबी पंक्ति है जो नहीं आएगी")
    nxt = Turn("t2", "F", 1.1, 2.0, hi_fit="हाँ")
    clips = {x.turn_id: router.synthesize(x) for x in (t, nxt)}
    natural = clips["t1"].natural_duration
    devs = fit.fit_all([t, nxt], clips, 10.0, lambda tt, r: router.synthesize(tt, r), None, fit.FitConfig())
    d = [x for x in devs if x["turn_id"] == "t1"][0]
    assert d["severity"] == "draft" and d["overflow_s"] > 0.6
    assert clips["t1"].final_duration == pytest.approx(natural / 1.15, rel=0.03)  # not truncated


# ── verification / status ────────────────────────────────────────────────
def _clip(tid, spk="M", accepted=True):
    return Clip(clip_id="c" + tid, turn_id=tid, speaker_id=spk, provider="mock",
                voice="mock-male-1", spoken_text="x", path="/nonexistent", accepted=accepted)


def test_missing_turn_plus_duplicate_clip_is_detected():
    turns = [Turn("t1", "M", 0, 1, hi_fit="क"), Turn("t2", "M", 2, 3, hi_fit="ख")]
    c1a, c1b = _clip("t1"), _clip("t1")
    cov = verify.coverage(turns, {"t1": c1a}, [c1a, c1b])
    assert [m["turn_id"] for m in cov["missing"]] == ["t2"]
    assert cov["duplicates"] == [{"turn_id": "t1", "accepted_clips": 2}]
    r = JobReport(missing_turns=cov["missing"], duplicate_clips=cov["duplicates"])
    assert report.derive_status(r)[0] == STATUS_DRAFT_INCOMPLETE


def test_identity_check_flags_global_voice_substitution():
    reg = _reg()
    reg.bind_provider("mock")
    turns = {"t1": Turn("t1", "F", 0, 1)}
    wrong = _clip("t1", spk="F")          # voiced with the male voice
    assert verify.identity_check(turns, {"t1": wrong}, reg)[0]["problem"] == "voice_not_bound_voice"


def test_status_derivation_levels():
    assert report.derive_status(JobReport())[0] == STATUS_COMPLETED
    r = JobReport(timing_deviations=[{"turn_id": "t1", "severity": "warning", "overflow_s": 0.2}])
    assert report.derive_status(r)[0] == STATUS_COMPLETED_WITH_WARNINGS
    r = JobReport(unresolved_failures=["speaker_mixed_providers F: turns ['t2']"])
    assert report.derive_status(r)[0] == STATUS_DRAFT_INCOMPLETE
    assert report.derive_status(JobReport(), aborted="cancelled")[0] == "cancelled"


# ── adopted from SoniTranslate / pyVideoTrans review ─────────────────────
def test_tts_sanitize_strips_directions_and_symbols_keeps_words():
    from dubbing.dialogue.tts import rate_with_speed, tts_sanitize
    assert tts_sanitize("[हँसते हुए] अरे... तुम *यहाँ* हो?") == "अरे, तुम यहाँ हो?"
    assert tts_sanitize("...क्या — सच में?…") == "क्या, सच में?"
    assert tts_sanitize("मैं ठीक हूँ।") == "मैं ठीक हूँ।"
    assert rate_with_speed("+0%", 1.1) == "+10%" and rate_with_speed("-5%", 1.1) == "+4%"


def test_router_speaks_sanitized_text(tmp_path):
    prov = MockProvider()
    router = TTSRouter({"mock": prov}, _reg(), ["mock"], tmp_path)
    clip = router.synthesize(Turn("t1", "M", 0, 2, hi_fit="(धीरे से) नमस्ते"))
    assert prov.calls[-1]["text"] == "नमस्ते" and clip.spoken_text == "नमस्ते"


def test_retry_backoff_only_for_providers_that_ask_for_it(tmp_path, monkeypatch):
    import dubbing.dialogue.tts as tts_mod
    sleeps = []
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: sleeps.append(s))
    n = {"i": 0}

    def fail(text, b):
        n["i"] += 1
        return n["i"] < 3
    prov = MockProvider(fail=fail)
    TTSRouter({"mock": prov}, _reg(), ["mock"], tmp_path, max_retries=2).synthesize(
        Turn("t1", "M", 0, 2, hi_fit="ठीक है"))
    assert sleeps == []
    n["i"] = 0
    prov.retry_backoff_s = 2.0
    TTSRouter({"mock": prov}, _reg(), ["mock"], tmp_path, max_retries=2).synthesize(
        Turn("t2", "M", 0, 2, hi_fit="ठीक है"))
    assert len(sleeps) == 2 and 2.0 <= sleeps[0] <= 2.5 and 4.0 <= sleeps[1] <= 4.5


def test_previous_overrun_delays_next_clip_instead_of_talking_over_it(tmp_path):
    router = TTSRouter({"mock": MockProvider(seconds_per_char=0.2)}, _reg(), ["mock"], tmp_path)
    a = Turn("t1", "M", 0.0, 1.0, hi_fit="यह लंबी पंक्ति है")
    b = Turn("t2", "F", 1.1, 4.0, hi_fit="हाँ")
    clips = {x.turn_id: router.synthesize(x) for x in (a, b)}
    devs = fit.fit_all([a, b], clips, 10.0, lambda tt, r: router.synthesize(tt, r), None, fit.FitConfig())
    first_end = clips["t1"].scheduled_end
    assert first_end > 1.1                      # t1 overran into t2's start
    d2 = [d for d in devs if d["turn_id"] == "t2"][0]
    assert 0 < d2["delayed_s"] <= 0.4 + 1e-6    # bounded delay, recorded
    assert clips["t2"].scheduled_start == pytest.approx(min(first_end + 0.05, 1.5), abs=2e-3)
    assert (b.source_start, b.source_end) == (1.1, 4.0)


def test_native_rate_is_used_before_stretching_and_total_speed_is_capped(tmp_path):
    prov = MockProvider(seconds_per_char=0.1, native_rate=True)
    router = TTSRouter({"mock": prov}, _reg(), ["mock"], tmp_path)
    t = Turn("t1", "M", 0.0, 1.0, hi_fit="नमस्ते आप कैसे हैं")
    nxt = Turn("t2", "F", 1.65, 3.0, hi_fit="हाँ")
    clips = {x.turn_id: router.synthesize(x) for x in (t, nxt)}
    natural = clips["t1"].natural_duration
    assert natural > 1.65
    devs = fit.fit_all([t, nxt], clips, 5.0, lambda tt, r: router.synthesize(tt, r), None, fit.FitConfig(),
                       native_rate=lambda tt, sp: router.synthesize(tt, "native", speed=sp))
    c = clips["t1"]
    assert c.voice_params["rate"] not in (None, "+0%")
    d = [x for x in devs if x["turn_id"] == "t1"][0]
    assert d["native_rate"] > 1.0
    assert natural / c.final_duration <= 1.15 + 0.03     # native x stretch within the cap


def test_last_turn_may_pre_roll_further_so_media_end_does_not_cut_it(tmp_path):
    router = TTSRouter({"mock": MockProvider(seconds_per_char=0.2)}, _reg(), ["mock"], tmp_path)
    t = Turn("t1", "M", 8.0, 9.0, hi_fit="यह अंतिम लंबी पंक्ति")
    clips = {"t1": router.synthesize(t)}
    fit.fit_all([t], clips, 10.0, lambda tt, r: router.synthesize(tt, r), None, fit.FitConfig())
    assert clips["t1"].scheduled_start < 8.0 - 0.25
    assert clips["t1"].scheduled_start >= 8.0 - 1.5 - 1e-6


def test_trim_keeps_soft_tail_and_level_gain_evens_voices(tmp_path):
    import numpy as np
    from dubbing.dialogue import audio
    sr = 48000
    tone = 0.3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
    tail = 0.004 * np.sin(2 * np.pi * 200 * np.arange(int(0.08 * sr)) / sr)   # ~ -48 dBFS
    x = np.concatenate([np.zeros(sr // 2), tone, tail, np.zeros(sr // 2)]).astype(np.float32)
    src, out = tmp_path / "a.wav", tmp_path / "b.wav"
    audio.write_wav(src, x, sr)
    dur = audio.trim_silence(src, out)
    assert 1.08 <= dur <= 1.0 + 0.08 + 0.04 + 0.12 + 0.02
    quiet, loud = 0.2 * tone, 1.2 * tone   # about -27 and -12 dBFS RMS
    gq, gl = audio.speech_level_gain(quiet, sr), audio.speech_level_gain(loud, sr)
    assert gq > 1 > gl
    lvl = lambda v: 20 * np.log10(np.sqrt(np.mean(v ** 2)))
    assert abs(lvl(quiet * gq) - lvl(loud * gl)) < 1.0


class _BriefLLM:
    model_id = "fake:brief"

    def __init__(self, brief):
        self.brief, self.requests = brief, []

    def complete(self, system, user):
        req = json.loads(user)
        self.requests.append(req)
        if "transcript" in req:
            return json.dumps(self.brief, ensure_ascii=False)
        return json.dumps({"translations": [{"id": t["id"], "hi": "ठीक है"} for t in req["turns"]]},
                          ensure_ascii=False)


def test_story_brief_is_built_once_cleaned_and_sent_with_every_batch():
    llm = _BriefLLM({"speakers": {"A": {"gender": "female", "evidence": "'she said'"},
                                  "B": {"gender": "robot"}, "ZZ": {"gender": "male"}},
                     "address": {"A -> B": "aap", "B -> A": "yo"},
                     "names": {"Riya": "रिया", "Bob": "Bob"}, "summary": "an interview"})
    tr = DialogueTranslator([llm], batch_size=2, story_brief=True, allow_basic_fallback=False)
    turns = _turns(5)
    tr.translate(turns)
    briefs = [r for r in llm.requests if "transcript" in r]
    assert len(briefs) == 1 and "[B] Line a here." in briefs[0]["transcript"]
    assert tr.brief["speakers"] == {"A": {"gender": "female", "evidence": "'she said'"}}
    assert tr.brief["address"] == {"A -> B": "aap"}
    assert tr.name_map == {"Riya": "रिया"}          # non-Devanagari spelling rejected
    batches = [r for r in llm.requests if "turns" in r]
    assert len(batches) == 3 and all(r["story_brief"] == tr.brief for r in batches)


def test_story_brief_failure_is_non_fatal():
    class Broken(_BriefLLM):
        def complete(self, system, user):
            if "transcript" in json.loads(user):
                raise RuntimeError("boom")
            return super().complete(system, user)
    tr = DialogueTranslator([Broken({})], story_brief=True, allow_basic_fallback=False)
    turns = _turns(3)
    warnings = tr.translate(turns)
    assert all(t.hi_raw for t in turns) and tr.brief == {}
    assert any(w["type"] == "story_brief_failed" for w in warnings)


def test_pipe_becomes_danda_and_first_turn_at_zero_is_not_delayed(tmp_path):
    from dubbing.dialogue.tts import tts_sanitize
    assert tts_sanitize("मैं आ गया | चलो") == "मैं आ गया। चलो"
    router = TTSRouter({"mock": MockProvider()}, _reg(), ["mock"], tmp_path)
    t = Turn("t1", "M", 0.0, 2.0, hi_fit="हाँ")
    clips = {"t1": router.synthesize(t)}
    devs = fit.fit_all([t], clips, 5.0, lambda tt, r: router.synthesize(tt, r), None, fit.FitConfig())
    assert clips["t1"].scheduled_start == 0.0 and not devs
