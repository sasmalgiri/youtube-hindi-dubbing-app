"""Regression tests for identity/honesty fixes in the legacy Pipeline."""
import inspect

import pytest

from pipeline import Pipeline, PipelineConfig


@pytest.fixture()
def pipe(tmp_path):
    cfg = PipelineConfig(source="x.mp4", work_dir=tmp_path / "w", output_path=tmp_path / "o.mp4",
                         tts_voice="hi-IN-MadhurNeural")
    return Pipeline(cfg)


def seg(start, end, text, spk=None, **kw):
    d = {"start": start, "end": end, "text": text, **kw}
    if spk:
        d["speaker_id"] = spk
    return d


def test_merge_never_crosses_known_speaker_change(pipe):
    out = pipe._merge_broken_sentences([
        seg(0, 1, "Are you", "A"), seg(1.1, 2, "coming", "B"), seg(2.1, 3, "tonight?", "B")])
    assert [(s["speaker_id"], s["text"]) for s in out] == [("A", "Are you"), ("B", "coming tonight?")]


def test_merge_unknown_then_adopts_speaker_and_stops_at_next_change(pipe):
    out = pipe._merge_broken_sentences([
        seg(0, 1, "well"), seg(1.1, 2, "I think", "A"), seg(2.1, 3, "so.", "B")])
    assert [(s.get("speaker_id"), s["text"]) for s in out] == [("A", "well I think"), ("B", "so.")]


def test_grouping_respects_speakers_even_with_unlabelled_sentences(pipe):
    sents = [seg(0, 1, "One.", "A"), seg(1, 2, "Two."), seg(2, 3, "Three.", "B"), seg(3, 4, "Four.", "B")]
    groups = pipe._group_sentences_by_count(sents, target_per_group=2)
    assert [(g.get("speaker_id"), g["text"]) for g in groups] == [("A", "One. Two."), ("B", "Three. Four.")]


def test_combining_different_speakers_is_refused(pipe):
    with pytest.raises(ValueError):
        pipe._combine_sentence_group([seg(0, 1, "Hi.", "A"), seg(1, 2, "Hey.", "B")])


def test_retry_voice_is_the_segment_speaker_voice(pipe):
    pipe._voice_map = {"A": "hi-IN-MadhurNeural", "B": "hi-IN-SwaraNeural"}
    assert pipe._voice_for_segment({"speaker_id": "B"}) == "hi-IN-SwaraNeural"
    assert pipe._voice_for_segment({}) == "hi-IN-MadhurNeural"
    src = inspect.getsource(Pipeline._post_tts_word_match_verify)
    assert "voice = self._voice_for_segment(seg)" in src
    assert "voice = self.cfg.tts_voice" not in src


def test_sarvam_fallback_speaker_matches_voice_register(pipe):
    assert pipe._sarvam_speaker_for_voice("hi-IN-SwaraNeural") == "ishita"
    assert pipe._sarvam_speaker_for_voice("hi-IN-MadhurNeural") == "shubh"


def test_unattributed_segment_is_not_assigned_speaker_00(pipe):
    segs = [seg(0, 1, "x"), seg(5, 6, "y")]
    pipe._assign_speaker_to_segments(segs, {"SPEAKER_01": [(0, 1.2)]})
    assert segs[0]["speaker_id"] == "SPEAKER_01"
    assert "speaker_id" not in segs[1] and segs[1]["_speaker_unknown"]


def test_completeness_counts_unique_segments_not_files(pipe, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    segs = [seg(0, 1, "a", text_translated="क", _seg_idx=0),
            seg(1, 2, "b", text_translated="ख", _seg_idx=1)]
    tts = [{"start": 0, "wav": wav, "_seg_idx": 0}, {"start": 0, "wav": wav, "_seg_idx": 0}]
    assert pipe._verify_tts_completeness(tts, segs) == 1   # duplicate cannot mask the gap
    assert pipe.result_status == "draft_incomplete"
    assert any("segment 1" in w for w in pipe.result_warnings)


def test_failed_separation_never_returns_original_as_background(pipe, tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_demucs(name, *a, **k):
        if name.startswith("demucs"):
            raise ImportError("no demucs")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_demucs)
    monkeypatch.setattr(pipe, "_get_duration", lambda p: 30.0)
    orig = tmp_path / "orig.wav"
    tts = tmp_path / "tts.wav"
    assert pipe._separate_background(orig) is None
    assert pipe._mix_audio(orig, tts, 0.1) == tts          # dub only, no English bed
    assert any("NOT used" in w for w in pipe.result_warnings)
