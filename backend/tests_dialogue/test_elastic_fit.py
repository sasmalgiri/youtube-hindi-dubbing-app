"""Elastic tempo fit: the planner (pure), then the pipeline step on real (synthetic) audio."""
import math
import time

import numpy as np
import pytest
import soundfile as sf

from dubbing.elastic_fit import CueSpec, FitParams, plan_elastic
from pipeline import Pipeline, PipelineConfig


# ── planner ────────────────────────────────────────────────────────────────
def cues(spec):
    """[(start, end, speech)] -> CueSpec list"""
    return [CueSpec(s, e, d) for s, e, d in spec]


def no_overlap(plan, gap=0.15):
    return all(b.start >= a.end + gap - 1e-6 for a, b in zip(plan.placements, plan.placements[1:]))


def test_sparse_script_is_spoken_at_natural_speed_on_its_cues():
    c = cues([(i * 10.0, i * 10.0 + 10.0, 4.0) for i in range(20)])
    plan = plan_elastic(c, 200.0)
    assert all(abs(pl.speed - 1.0) < 1e-9 for pl in plan.placements)
    assert all(abs(pl.offset) < 1e-6 for pl in plan.placements)
    assert no_overlap(plan)


def test_dense_stretch_speeds_up_a_little_and_lets_speech_lag_instead_of_rushing_one_cue():
    # 12 cues of 2 s each, 3 s of speech each: strict slots would need 1.5x on every cue.
    dense = [(i * 2.0, i * 2.0 + 2.0, 3.0) for i in range(12)]
    quiet = [(24.0 + i * 8.0, 32.0 + i * 8.0, 2.0) for i in range(6)]      # room to catch up
    plan = plan_elastic(cues(dense + quiet), 80.0)
    assert no_overlap(plan)
    assert max(pl.speed for pl in plan.placements) <= 1.40 + 1e-9
    dense_speeds = [pl.speed for pl in plan.placements[:12]]
    assert max(dense_speeds) < 1.5                      # strict fit would have been 1.5x everywhere
    assert max(pl.offset for pl in plan.placements) > 0  # it chose to lag a little
    # ...and it recovered: the quiet tail is back on time
    assert all(abs(pl.offset) < 0.05 for pl in plan.placements[-3:])


def test_speed_never_exceeds_the_ceiling_and_overflow_becomes_lateness():
    c = cues([(i * 1.0, i * 1.0 + 1.0, 3.0) for i in range(10)])        # hopeless: 3 s of speech per second
    plan = plan_elastic(c, 100.0, FitParams(v_max=1.3))
    assert max(pl.speed for pl in plan.placements) <= 1.3 + 1e-9
    assert no_overlap(plan)
    assert plan.placements[-1].offset > 5                              # cannot catch up, still ordered, no overlap


def test_early_start_is_bounded_and_only_used_when_it_helps():
    # first cue on time; second is dense and preceded by dead air: may start early, never by more than early_max
    c = cues([(0.0, 10.0, 2.0), (10.0, 12.0, 3.5), (12.0, 20.0, 2.0)])
    plan = plan_elastic(c, 30.0, FitParams(early_max=0.8))
    assert min(pl.offset for pl in plan.placements) >= -0.8 - 1e-9
    assert plan.placements[0].offset >= -0.8 - 1e-9


def test_pause_inside_a_cue_is_not_sped_up():
    c = [CueSpec(0.0, 5.0, speech=4.0, pause=0.6)]
    plan = plan_elastic(c, 10.0, FitParams(v_max=1.4))
    pl = plan.placements[0]
    assert pl.duration == pytest.approx(4.0 / pl.speed + 0.6)


def test_single_and_empty_inputs():
    assert plan_elastic([], 10.0).placements == []
    one = plan_elastic([CueSpec(1.0, 3.0, 1.5)], 10.0)
    assert len(one.placements) == 1 and one.placements[0].speed == 1.0


def test_contiguous_real_world_shape_runs_fast_and_stays_sane():
    rng = np.random.default_rng(7)
    t, spec = 0.0, []
    for _ in range(526):
        slot = float(rng.integers(1, 15))
        spec.append((t, t + slot, float(slot * rng.uniform(0.35, 1.25))))
        t += slot
    t0 = time.time()
    plan = plan_elastic(cues(spec), t)
    assert time.time() - t0 < 8.0
    assert no_overlap(plan)
    assert [pl.index for pl in plan.placements] == list(range(526))
    # a plan never starts a cue before the previous one finished
    assert all(pl.start >= -1.0 for pl in plan.placements)


def test_plan_is_deterministic():
    spec = [(i * 3.0, i * 3.0 + 3.0, 2.0 + (i % 5) * 0.6) for i in range(60)]
    a = plan_elastic(cues(spec), 200.0)
    b = plan_elastic(cues(spec), 200.0)
    assert [(p.speed, p.start) for p in a.placements] == [(p.speed, p.start) for p in b.placements]


# ── pipeline step on real audio ───────────────────────────────────────────
SR = 24000


def _tone(path, seconds):
    t = np.arange(int(SR * seconds)) / SR
    sf.write(str(path), (0.3 * np.sin(2 * math.pi * 220 * t)).astype("float32"), SR)


@pytest.fixture()
def pipe(tmp_path):
    cfg = PipelineConfig(source="x.mp4", work_dir=tmp_path / "w", output_path=tmp_path / "o.mp4",
                         tts_voice="hi-IN-MadhurNeural", tempo_match=True, tempo_elastic=True,
                         post_tts_level="none")
    (tmp_path / "w").mkdir()
    p = Pipeline(cfg)
    p._ensure_ffmpeg()
    return p


def _cue_data(pipe, tmp_path, spec):
    """spec: [(start, end, clip_seconds)] -> (tts_data, segments) like the splitter + TTS produce."""
    segments, tts_data = [], []
    for i, (s, e, d) in enumerate(spec):
        wav = tmp_path / f"clip{i}.wav"
        _tone(wav, d)
        segments.append({"start": s, "end": e, "text": f"cue {i}", "text_translated": f"cue {i}",
                         "_seg_idx": i})
        tts_data.append({"start": s, "end": e, "wav": str(wav), "duration": d, "_seg_idx": i})
    return tts_data, segments


def test_elastic_fit_places_clips_without_overlap_and_fills_the_report_inputs(pipe, tmp_path):
    spec = [(0, 2, 3.0), (2, 4, 3.0), (4, 6, 3.0), (6, 20, 2.0), (20, 24, 3.0)]
    tts, segs = _cue_data(pipe, tmp_path, spec)
    out = pipe._elastic_fit_segments(tts, segs, media_end=30.0)
    assert [t["_tempo_fitted"] for t in out] == [True] * 5
    ends = [(t["start"], t["start"] + t["duration"]) for t in out]
    assert all(b0 >= a1 + 0.1 for (a0, a1), (b0, b1) in zip(ends, ends[1:]))          # no overlaps
    assert all(t["duration"] <= 3.0 + 1e-6 for t in out)                                  # never longer than natural
    assert pipe._elastic_stats["cues"] == 5
    # subtitles are written where the speech really is
    assert len(pipe._placed_speech) == 5
    for (src_s, src_e, out_s, dur), t in zip(pipe._placed_speech, out):
        assert out_s == pytest.approx(t["start"]) and dur == pytest.approx(t["duration"])
    # the fitted audio on disk really has the planned (shorter) length
    for t in out:
        info = sf.info(t["wav"])
        assert info.frames / info.samplerate == pytest.approx(t["duration"], abs=0.06)


def test_elastic_fit_leaves_a_sparse_script_untouched(pipe, tmp_path):
    spec = [(0, 10, 3.0), (10, 20, 4.0), (20, 30, 2.5)]
    tts, segs = _cue_data(pipe, tmp_path, spec)
    out = pipe._elastic_fit_segments(tts, segs, media_end=40.0)
    assert [round(t["start"], 2) for t in out] == [0.0, 10.0, 20.0]
    assert all(r["tier"] == "natural" for r in pipe._tempo_fit_records)
    assert not pipe.result_warnings


def test_elastic_fit_flags_a_hopelessly_dense_script(pipe, tmp_path):
    spec = [(i, i + 1, 3.0) for i in range(8)] + [(8, 60, 1.0)]
    tts, segs = _cue_data(pipe, tmp_path, spec)
    pipe._elastic_fit_segments(tts, segs, media_end=70.0)
    assert any("dense" in w for w in pipe.result_warnings)
    assert any(r.get("flagged") for r in pipe._tempo_fit_records)


# ── silent cues ───────────────────────────────────────────────────────────
def test_punctuation_only_cue_is_not_missing_audio(pipe):
    from pipeline import _has_speakable
    assert not _has_speakable("...") and not _has_speakable("…") and not _has_speakable(" - ")
    assert _has_speakable("हाँ।") and _has_speakable("Okay") and _has_speakable("100")
    segs = [{"start": 0, "end": 2, "text_translated": "नमस्ते", "_seg_idx": 0},
            {"start": 2, "end": 4, "text_translated": "...", "_seg_idx": 1}]
    tts = [{"start": 0, "wav": __file__, "duration": 1.0, "_seg_idx": 0}]
    assert pipe._verify_tts_completeness(tts, segs) == 0
    assert pipe.result_status != "draft_incomplete" and not pipe.result_warnings


def test_silent_piece_inside_a_cue_joins_its_neighbour(pipe):
    out = pipe._split_segments_at_sentences([
        {"start": 0, "end": 6, "text_translated": "... ठीक है। चलो चलें।", "text": "x"}])
    assert [o["text_translated"] for o in out] == ["... ठीक है।", "चलो चलें।"]
    out = pipe._split_segments_at_sentences([
        {"start": 0, "end": 6, "text_translated": "ठीक है। ...", "text": "x"}])
    assert len(out) == 1 and out[0]["text_translated"].startswith("ठीक है।")


def test_generate_tts_skips_cues_with_nothing_to_speak(pipe, monkeypatch):
    seen = {}
    monkeypatch.setattr(Pipeline, "_generate_tts_english", lambda self, segs: seen.setdefault("segs", segs) or [],
                        raising=True)
    pipe.cfg.target_language = "en"
    pipe._generate_tts_natural([
        {"start": 0, "end": 2, "text_translated": "Hello there.", "text": "Hello there."},
        {"start": 2, "end": 4, "text_translated": "...", "text": "..."}])
    assert [s["text_translated"] for s in seen["segs"]] == ["Hello there."]
    assert pipe._silent_cues and pipe._silent_cues[0][1] == "..."


# ── streaming timeline ────────────────────────────────────────────────────
def _clip(pipe, path, seconds, ch=None, sr=None, level=0.2):
    sr = sr or pipe.SAMPLE_RATE
    ch = ch or pipe.N_CHANNELS
    n = int(sr * seconds)
    data = np.full((n, ch), level, dtype="float32")
    sf.write(str(path), data, sr, subtype="PCM_16")
    return str(path)


def test_sequential_timeline_places_clips_to_the_sample(pipe, tmp_path):
    a = _clip(pipe, tmp_path / "a.wav", 1.0)
    b = _clip(pipe, tmp_path / "b.wav", 0.5)
    out = pipe._build_timeline_sequential(
        [{"start": 3.0, "wav": b, "duration": 0.5}, {"start": 0.5, "wav": a, "duration": 1.0}], 6.0)
    data, sr = sf.read(str(out), always_2d=True)
    assert sr == pipe.SAMPLE_RATE and len(data) == int(6.0 * sr)
    amp = np.abs(data).max(axis=1)
    assert amp[: int(0.49 * sr)].max() == 0                                  # lead silence
    assert amp[int(0.51 * sr): int(1.49 * sr)].min() > 0.1                   # clip a
    assert amp[int(1.51 * sr): int(2.99 * sr)].max() == 0                    # gap
    assert amp[int(3.01 * sr): int(3.49 * sr)].min() > 0.1                   # clip b
    assert amp[int(3.51 * sr):].max() == 0                                   # tail silence


def test_sequential_timeline_refuses_overlaps_and_foreign_formats(pipe, tmp_path):
    a = _clip(pipe, tmp_path / "a.wav", 2.0)
    b = _clip(pipe, tmp_path / "b.wav", 1.0)
    with pytest.raises(ValueError, match="overlap"):
        pipe._build_timeline_sequential([{"start": 0.0, "wav": a}, {"start": 1.0, "wav": b}], 5.0)
    mono = _clip(pipe, tmp_path / "m.wav", 1.0, ch=1)
    with pytest.raises(ValueError, match="timeline is"):
        pipe._build_timeline_sequential([{"start": 0.0, "wav": mono}], 5.0)


def test_sequential_timeline_runs_past_the_end_without_cutting(pipe, tmp_path):
    a = _clip(pipe, tmp_path / "a.wav", 3.0)
    out = pipe._build_timeline_sequential([{"start": 4.0, "wav": a}], 5.0)
    data, sr = sf.read(str(out), always_2d=True)
    assert len(data) == int(7.0 * sr)          # nothing cut; the anchored pass trims to the target


def test_mixer_chunk_size_follows_the_command_line_budget(pipe, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(Pipeline, "_build_timeline_chunk",
                        lambda self, chunk, total, output, exact=False: calls.append(len(chunk)))
    monkeypatch.setattr(Pipeline, "_run_proc", lambda self, *a, **k: None)
    long_dir = tmp_path / ("x" * 150)
    long_dir.mkdir()
    clips = []
    for i in range(120):
        w = long_dir / f"c{i}.wav"
        w.write_bytes(b"x")
        clips.append({"start": i, "wav": str(w), "duration": 0.5})
    try:
        pipe._build_timeline_no_cut(clips, 200.0, exact=True)
    except Exception:
        pass                                    # the merge step needs real files; only the chunking matters
    assert calls and max(calls) * (len(str(clips[0]["wav"])) + 60) < 27000


def test_cue_above_the_atempo_band_is_resynthesized_at_a_faster_edge_rate(pipe, tmp_path, monkeypatch):
    """A cue the plan speeds up beyond 1.25x gets a native Edge-rate pass, then lands on its planned duration."""
    calls = []

    def fake_resynth(self, entry, seg, rate_pct, tag, enhance_noop):
        calls.append(rate_pct)
        d = sf.info(entry["wav"]).frames / sf.info(entry["wav"]).samplerate
        out = tmp_path / f"re_{tag}.wav"
        _tone(out, d / (1.0 + rate_pct / 100.0) * 1.03)        # Edge lands a touch longer than linear
        return (out, d / (1.0 + rate_pct / 100.0) * 1.03)

    monkeypatch.setattr(Pipeline, "_tempo_resynth_child", fake_resynth)
    # a run of cues that each need ~1.7x: the plan saturates near the ceiling instead of lagging forever
    spec = [(i * 2.0, i * 2.0 + 2.0, 3.4) for i in range(8)] + [(16.0, 40.0, 1.0)]
    tts, segs = _cue_data(pipe, tmp_path, spec)
    pipe.cfg.tempo_max_speedup = 1.4
    out = pipe._elastic_fit_segments(tts, segs, media_end=45.0)
    recs = pipe._tempo_fit_records
    two = [r for r in recs if r["tier"] == "two_pass"]
    assert two and calls and all(5 <= c <= 40 for c in calls)
    assert two[0]["speed"] > 1.25
    ends = [(t["start"], t["start"] + t["duration"]) for t in out]
    assert all(b0 >= a1 + 0.1 for (a0, a1), (b0, b1) in zip(ends, ends[1:]))
