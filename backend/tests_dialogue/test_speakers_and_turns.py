"""Identity, attribution and turn-construction invariants."""
import math
import wave
from array import array

import pytest

from dubbing.dialogue.contracts import (CATEGORY_FEMALE, CATEGORY_MALE,
                                        CATEGORY_UNKNOWN, UNKNOWN_SPEAKER, Turn,
                                        WordRecord)
from dubbing.dialogue.diarization import (DiarizationResult, assign_words,
                                          derive_exclusive, reconcile_chunks)
from dubbing.dialogue.speaker_registry import (SpeakerRegistry,
                                               VoiceResolutionError)
from dubbing.dialogue.turns import (align_text_to_words, build_turns,
                                    turns_from_translated_cues,
                                    words_from_asr_segments)
from dubbing.dialogue.voice_analysis import analyze_speaker, classify_f0


def W(i, text, s, e, spk=None, overlap=False):
    return WordRecord(word_id=f"w{i:05d}", text=text, start=s, end=e,
                      speaker_id=spk, overlap=overlap)


# ── Turn immutability ──────────────────────────────────────────────────────
def test_turn_source_timing_is_immutable():
    t = Turn("t0001", "SPEAKER_00", 1.0, 2.0)
    with pytest.raises(AttributeError):
        t.source_start = 1.5
    with pytest.raises(AttributeError):
        t.turn_id = "t0002"
    t.hi_fit = "ठीक है"          # other fields remain writable
    t.source_end = 2.0            # assigning the same value is allowed


# ── Registry ───────────────────────────────────────────────────────────────
def _registry(spec):
    reg = SpeakerRegistry()
    for sid, cat, secs in spec:
        reg.register(sid, voice_category=cat, total_speech_s=secs)
    return reg


def test_alternating_male_female_are_stable():
    reg = _registry([("SPEAKER_00", CATEGORY_MALE, 30), ("SPEAKER_01", CATEGORY_FEMALE, 20)])
    a = reg.resolve_voice("SPEAKER_00", "edge")
    b = reg.resolve_voice("SPEAKER_01", "edge")
    assert a["voice"] == "hi-IN-MadhurNeural"
    assert b["voice"] == "hi-IN-SwaraNeural"
    # repeated resolution (e.g. retries) never changes the binding
    for _ in range(5):
        assert reg.resolve_voice("SPEAKER_01", "edge") == b
        assert reg.resolve_voice("SPEAKER_00", "edge") == a


def test_two_male_two_female_four_identities_with_reported_reuse():
    reg = _registry([("SPEAKER_00", CATEGORY_MALE, 40), ("SPEAKER_01", CATEGORY_FEMALE, 35),
                     ("SPEAKER_02", CATEGORY_MALE, 20), ("SPEAKER_03", CATEGORY_FEMALE, 10)])
    edge = {s: reg.resolve_voice(s, "edge") for s in reg.speakers}
    keys = {(b["voice"], b["pitch"]) for b in edge.values()}
    assert len(keys) == 4, "each character needs a distinct (voice, variant)"
    reuse = reg.voice_reuse("edge")
    assert set(reuse) == {"hi-IN-MadhurNeural", "hi-IN-SwaraNeural"}
    # Sarvam has several voices per category -> unique voices, no reuse
    sarvam = {s: reg.resolve_voice(s, "sarvam")["voice"] for s in reg.speakers}
    assert len(set(sarvam.values())) == 4
    assert reg.voice_reuse("sarvam") == {}


def test_main_speaker_gets_base_voice_regardless_of_id_order():
    reg = _registry([("SPEAKER_00", CATEGORY_MALE, 5), ("SPEAKER_01", CATEGORY_MALE, 50)])
    assert reg.resolve_voice("SPEAKER_01", "edge")["pitch"] is None
    assert reg.resolve_voice("SPEAKER_00", "edge")["pitch"] is not None


def test_unknown_speaker_strict_vs_register():
    reg = _registry([("SPEAKER_00", CATEGORY_MALE, 5)])
    with pytest.raises(VoiceResolutionError):
        reg.resolve_voice("SPEAKER_09", "edge")
    b = reg.resolve_voice(UNKNOWN_SPEAKER, "edge", policy="register_unknown")
    assert reg.speakers[UNKNOWN_SPEAKER].classification_unknown
    assert reg.speakers[UNKNOWN_SPEAKER].mapping_origin == "default_unknown"
    assert b["voice"]


def test_registry_persistence_roundtrip(tmp_path):
    reg = _registry([("SPEAKER_00", CATEGORY_MALE, 5), ("SPEAKER_01", CATEGORY_FEMALE, 4)])
    before = {s: reg.resolve_voice(s, "edge") for s in reg.speakers}
    reg.save(tmp_path / "speakers.json")
    reg2 = SpeakerRegistry.load(tmp_path / "speakers.json")
    after = {s: reg2.resolve_voice(s, "edge") for s in reg2.speakers}
    assert before == after


# ── Voice category ─────────────────────────────────────────────────────────
def test_classify_keeps_unknown_states():
    assert classify_f0([120.0] * 100, voiced_seconds=0.5)[0] == CATEGORY_UNKNOWN
    assert classify_f0([162.0] * 300, voiced_seconds=5)[0] == CATEGORY_UNKNOWN
    cat, conf, _ = classify_f0([110.0] * 300, voiced_seconds=5)
    assert cat == CATEGORY_MALE and conf >= 0.35
    cat, conf, _ = classify_f0([220.0] * 300, voiced_seconds=5)
    assert cat == CATEGORY_FEMALE and conf >= 0.35


def _write_tone(path, segments, sr=16000):
    """segments: list of (duration_s, freq_hz or 0)."""
    data = array("h")
    for dur, f in segments:
        for n in range(int(dur * sr)):
            if f:
                # harmonic-rich voiced-like signal
                v = sum(math.sin(2 * math.pi * f * k * n / sr) / k for k in range(1, 6))
                data.append(int(6000 * v))
            else:
                data.append(0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def test_analyze_speaker_on_synthetic_voices(tmp_path):
    p = tmp_path / "a.wav"
    _write_tone(p, [(3.0, 115.0), (3.0, 230.0)])
    male = analyze_speaker(p, [(0.0, 3.0)])
    female = analyze_speaker(p, [(3.0, 6.0)])
    assert male[0] == CATEGORY_MALE, male
    assert female[0] == CATEGORY_FEMALE, female


# ── Word attribution ───────────────────────────────────────────────────────
def test_one_asr_segment_spanning_two_speakers_splits_into_two_turns():
    seg = {"start": 0.0, "end": 3.0, "text": "Are you coming? Yes I am.",
           "words": [{"word": w, "start": s, "end": e} for w, s, e in [
               ("Are", 0.0, 0.3), ("you", 0.3, 0.5), ("coming?", 0.5, 1.0),
               ("Yes", 1.3, 1.6), ("I", 1.6, 1.8), ("am.", 1.8, 2.2)]]}
    words = words_from_asr_segments([seg])
    diar = DiarizationResult(regular=[(0.0, 1.1, "SPEAKER_00"), (1.2, 2.3, "SPEAKER_01")],
                             exclusive=[(0.0, 1.1, "SPEAKER_00"), (1.2, 2.3, "SPEAKER_01")])
    assign_words(words, diar)
    turns = build_turns(words)
    assert [(t.speaker_id, t.source_text) for t in turns] == [
        ("SPEAKER_00", "Are you coming?"), ("SPEAKER_01", "Yes I am.")]


def test_uncertain_words_stay_unknown_not_speaker_00():
    words = [W(1, "hello", 5.0, 5.4)]
    diar = DiarizationResult(regular=[(0.0, 1.0, "SPEAKER_01")],
                             exclusive=[(0.0, 1.0, "SPEAKER_01")])
    stats = assign_words(words, diar)
    assert words[0].speaker_id is None and stats["unknown"] == 1
    turns = build_turns(words)
    assert turns[0].speaker_id == UNKNOWN_SPEAKER and "speaker_unknown" in turns[0].flags


def test_no_diarization_leaves_everything_unknown():
    words = [W(1, "hi", 0, 0.3), W(2, "there", 0.3, 0.6)]
    assign_words(words, None)
    assert all(w.speaker_id is None for w in words)


def test_short_replies_and_interruption_are_not_merged():
    words = [W(1, "I", 0.0, 0.2, "A"), W(2, "was", 0.2, 0.4, "A"), W(3, "saying", 0.4, 0.8, "A"),
             W(4, "No!", 0.85, 1.1, "B"),
             W(5, "that", 1.15, 1.4, "A"), W(6, "we", 1.4, 1.6, "A"), W(7, "go.", 1.6, 1.9, "A"),
             W(8, "Yes.", 2.0, 2.3, "B")]
    turns = build_turns(words)
    assert [t.speaker_id for t in turns] == ["A", "B", "A", "B"]
    assert turns[1].source_text == "No!" and turns[3].source_text == "Yes."
    for t in turns:
        assert len({w for w in t.word_ids}) == len(t.word_ids)


def test_overlapping_speech_keeps_both_turns_and_flags_overlap():
    words = [W(1, "We", 0.0, 0.3, "A"), W(2, "should", 0.3, 0.6, "A", True),
             W(3, "Wait", 0.5, 0.8, "B", True), W(4, "leave", 0.6, 0.9, "A", True),
             W(5, "now.", 0.9, 1.2, "A")]
    turns = build_turns(words)
    a = [t for t in turns if t.speaker_id == "A"]
    b = [t for t in turns if t.speaker_id == "B"]
    assert len(a) == 1 and a[0].source_text == "We should leave now."
    assert len(b) == 1 and b[0].turn_id in a[0].overlaps_with


def test_nonlexical_turn_is_not_required():
    words = [W(1, "[laughter]", 0.0, 0.8, "A"), W(2, "Okay.", 2.0, 2.4, "B")]
    turns = build_turns(words)
    assert turns[0].required is False and "nonlexical" in turns[0].flags
    assert turns[1].required is True


def test_long_monologue_is_split_at_a_boundary():
    words, t = [], 0.0
    for i in range(60):
        txt = "word." if i == 29 else "word"
        words.append(W(i + 1, txt, t, t + 0.3, "A"))
        t += 0.32
    turns = build_turns(words, max_turn_s=15.0, sentence_split_min_s=100)
    assert len(turns) == 2 and turns[0].source_text.endswith("word.")


def test_derive_exclusive_has_no_overlap():
    excl = derive_exclusive([(0, 5, "A"), (2, 3, "B")])
    assert excl == [(0, 2, "A"), (2, 3, "B"), (3, 5, "A")]


def test_chunk_reconciliation_keeps_identity_across_chunks():
    c1 = DiarizationResult([(0, 5, "S0"), (5, 9, "S1")], [(0, 5, "S0"), (5, 9, "S1")],
                           {"S0": [1, 0, 0], "S1": [0, 1, 0]})
    # chunk 2 labels are swapped locally
    c2 = DiarizationResult([(0, 4, "S0"), (4, 8, "S1")], [(0, 4, "S0"), (4, 8, "S1")],
                           {"S0": [0, 0.9, 0.1], "S1": [0.95, 0.05, 0]})
    res = reconcile_chunks([(0.0, c1), (10.0, c2)])
    assert len(res.speakers) == 2
    first = {k for s, e, k in res.exclusive if s < 5}
    later = {k for s, e, k in res.exclusive if 14 <= s < 18}
    assert first == later, "speaker at 0s and 14s must be the same global identity"
    with pytest.raises(ValueError):
        reconcile_chunks([(0.0, DiarizationResult([(0, 1, "X")], [(0, 1, "X")]))])


# ── Subtitle text sources ─────────────────────────────────────────────────
def test_youtube_text_aligned_to_audio_words_keeps_subtitle_text():
    cues = [{"start": 0.0, "end": 2.5, "text": "Hello, Riya. How are you?"},
            {"start": 2.5, "end": 4.0, "text": "I'm fine."}]
    asr = words_from_asr_segments([{"start": 0, "end": 4, "text": "", "words": [
        {"word": "hello", "start": 0.1, "end": 0.4}, {"word": "rhea", "start": 0.4, "end": 0.8},
        {"word": "how", "start": 1.0, "end": 1.2}, {"word": "are", "start": 1.2, "end": 1.3},
        {"word": "you", "start": 1.3, "end": 1.6}, {"word": "I'm", "start": 2.8, "end": 3.0},
        {"word": "fine", "start": 3.0, "end": 3.4}]}])
    words = align_text_to_words(cues, asr)
    assert " ".join(w.text for w in words) == "Hello, Riya. How are you? I'm fine."
    riya = words[1]
    assert riya.timing_estimated and 0.4 <= riya.start <= riya.end <= 1.0
    assert words[0].start == pytest.approx(0.1)
    diar = DiarizationResult([(0, 2.0, "A"), (2.6, 3.6, "B")], [(0, 2.0, "A"), (2.6, 3.6, "B")])
    assign_words(words, diar)
    turns = build_turns(words)
    assert [t.speaker_id for t in turns] == ["A", "B"]


def test_translated_cues_take_speakers_from_audio_and_flag_mixed_cues():
    cues = [{"start": 0.0, "end": 2.0, "text_translated": "नमस्ते।"},
            {"start": 2.0, "end": 5.0, "text_translated": "ठीक है, चलो।", "speaker_id": "SPEAKER_07"}]
    diar = DiarizationResult([(0, 2, "A"), (2, 3.5, "B"), (3.5, 5, "A")],
                             [(0, 2, "A"), (2, 3.5, "B"), (3.5, 5, "A")])
    turns = turns_from_translated_cues(cues, diar)
    assert turns[0].speaker_id == "A"
    assert turns[1].speaker_id == "B" and "multi_speaker_cue" in turns[1].flags
    # the SRT label is used only if diarization is unavailable
    turns2 = turns_from_translated_cues(cues, None)
    assert turns2[1].speaker_id == "SPEAKER_07" and turns2[0].speaker_id == UNKNOWN_SPEAKER
