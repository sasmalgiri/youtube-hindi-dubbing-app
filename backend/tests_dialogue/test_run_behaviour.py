"""Dialogue run behaviour: progress the UI can trust, a cancel that stops the
long stages, and one voice per character (no orphan-word voice, an analysed
single-voice narrator, an honest failure when nothing was translated)."""
import importlib.util
import math
import multiprocessing as mp
import time
import wave
from array import array

import pytest

from dubbing.dialogue import audio
from dubbing.dialogue import orchestrator as orch_mod
from dubbing.dialogue.contracts import UNKNOWN_SPEAKER, WordRecord
from dubbing.dialogue.diarization import (DiarizationResult, assign_single_speaker,
                                          assign_words, run_pyannote)
from dubbing.dialogue.mix import separate_in_child
from dubbing.dialogue.orchestrator import (DialogueConfig, DialogueOrchestrator,
                                           default_acquire, run_dialogue)
from dubbing.dialogue.translation import TranslationEngineError
from dubbing.dialogue.turns import build_turns, turns_from_translated_cues

from test_orchestrator_e2e import (DURATION, HAVE_FFMPEG, SCRIPT, _asr_segments,
                                   _components, _diar, _go, _make_media)

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")


def W(i, text, s, e, spk=None):
    return WordRecord(word_id=f"w{i:05d}", text=text, start=s, end=e, speaker_id=spk)


def _cancel_after(seconds):
    t0 = time.time()
    return lambda: time.time() - t0 > seconds


def _silent_wav(path, seconds=3.0, sr=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\0\0" * int(seconds * sr))
    return path


def _tone_media(tmp, tones):
    """MP4 whose audio has a harmonic tone (f0, start, end) per line."""
    sr = 16000
    data = array("h", [0] * int(DURATION * sr))
    for f0, s, e in tones:
        for n in range(int(s * sr), int(e * sr)):
            data[n] = int(5000 * sum(math.sin(2 * math.pi * f0 * k * n / sr) / k for k in range(1, 6)))
    wav = tmp / "tones.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    mp4 = tmp / "tones.mp4"
    audio.run_ffmpeg(["-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={DURATION}",
                      "-i", str(wav), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-shortest", str(mp4)])
    return mp4


# ── word attribution ──────────────────────────────────────────────────────
def test_word_touching_only_one_speaker_is_theirs():
    # ASR stretched the line's first word over the silence before it: 6% of
    # it is in the segment and its midpoint is 0.4 s outside.
    words = [W(1, "I", 0.2, 1.05)]
    diar = DiarizationResult([(1.0, 3.0, "B")], [(1.0, 3.0, "B")])
    stats = assign_words(words, diar)
    assert words[0].speaker_id == "B" and stats["unknown"] == 0


def test_nearest_segment_is_measured_from_the_word_edges():
    words = [W(1, "So,", 1.0, 1.85)]          # edge 0.15 s from B, midpoint 0.58 s
    diar = DiarizationResult([(2.0, 3.0, "B")], [(2.0, 3.0, "B")])
    assign_words(words, diar)
    assert words[0].speaker_id == "B"
    assert words[0].attribution == {"method": "nearest_segment", "distance_s": 0.15}


def test_word_under_two_speakers_without_majority_goes_to_the_larger_share():
    words = [W(1, "well", 0.75, 1.35)]         # 42% under A, 25% under B
    diar = DiarizationResult([(0.0, 1.0, "A"), (1.2, 3.0, "B")],
                             [(0.0, 1.0, "A"), (1.2, 3.0, "B")])
    assign_words(words, diar)
    assert words[0].speaker_id == "A"


def test_single_voice_words_are_one_known_speaker():
    words = [W(1, "Hello", 0.0, 0.4), W(2, "there.", 0.4, 0.8)]
    stats = assign_single_speaker(words)
    assert {w.speaker_id for w in words} == {"SPEAKER_00"} and stats["unknown"] == 0
    assert words[0].attribution["method"] == "single_voice"


# ── turns: orphaned words get a character's voice ─────────────────────────
def test_orphan_first_word_joins_the_next_speakers_line():
    words = [W(1, "Where", 0.0, 0.3, "A"), W(2, "were", 0.3, 0.5, "A"), W(3, "you?", 0.5, 0.9, "A"),
             W(4, "I", 1.3, 1.4), W(5, "was", 1.45, 1.7, "B"), W(6, "home.", 1.7, 2.1, "B")]
    turns = build_turns(words)
    assert [(t.speaker_id, t.source_text) for t in turns] == [
        ("A", "Where were you?"), ("B", "I was home.")]
    assert "speaker_from_adjacent_turn" in turns[1].flags
    assert words[3].attribution["method"] == "adjacent_turn"


def test_orphan_sentence_end_joins_the_previous_line():
    words = [W(1, "I", 0.0, 0.2, "A"), W(2, "went", 0.2, 0.6, "A"), W(3, "home.", 0.65, 1.0),
             W(4, "Really?", 1.6, 2.0, "B")]
    assert [(t.speaker_id, t.source_text) for t in build_turns(words)] == [
        ("A", "I went home."), ("B", "Really?")]


def test_orphan_far_from_its_preferred_neighbour_joins_the_close_one():
    # "I" does not end a sentence, but the next line is 3 s away and the
    # previous one ended 50 ms before it.
    words = [W(1, "and", 0.0, 0.3, "A"), W(2, "then", 0.3, 0.6, "A"), W(3, "I", 0.65, 0.75),
             W(4, "Hello.", 3.75, 4.2, "B")]
    assert [(t.speaker_id, t.source_text) for t in build_turns(words)] == [
        ("A", "and then I"), ("B", "Hello.")]


def test_long_unattributed_speech_keeps_the_unknown_voice():
    words = [W(1, "Hi.", 0.0, 0.3, "A"), W(2, "this", 0.5, 0.7), W(3, "is", 0.7, 0.8),
             W(4, "someone", 0.8, 1.2), W(5, "else.", 1.2, 1.5)]
    turns = build_turns(words)
    assert [t.speaker_id for t in turns] == ["A", UNKNOWN_SPEAKER]
    assert "speaker_unknown" in turns[1].flags


def test_unlabelled_hindi_cues_are_the_single_voice_when_detection_is_off():
    cues = [{"start": 0.0, "end": 2.0, "text_translated": "नमस्ते।"},
            {"start": 2.0, "end": 4.0, "text_translated": "चलो।", "speaker_id": "SPEAKER_07"}]
    turns = turns_from_translated_cues(cues, None, default_speaker="SPEAKER_00")
    assert [t.speaker_id for t in turns] == ["SPEAKER_00", "SPEAKER_07"]
    assert "single_voice" in turns[0].flags and "speaker_unknown" not in turns[0].flags


# ── cancel reaches child processes ────────────────────────────────────────
def test_children_killer_stops_only_children_started_after_it():
    ctx = mp.get_context("spawn")
    old = ctx.Process(target=time.sleep, args=(60,), daemon=True)
    old.start()
    try:
        kill = orch_mod._new_children_killer()
        new = ctx.Process(target=time.sleep, args=(60,), daemon=True)
        new.start()
        kill()
        new.join(10)
        assert new.exitcode is not None and old.is_alive()
    finally:
        old.kill()
        old.join(5)


@pytest.mark.skipif(importlib.util.find_spec("pyannote") is None, reason="pyannote.audio required")
def test_cancel_kills_the_pyannote_child(tmp_path):
    before = set(mp.active_children())
    t0 = time.time()
    with pytest.raises(RuntimeError, match="cancelled by user"):
        run_pyannote(_silent_wav(tmp_path / "a.wav"), "hf_test_token", cancel_check=_cancel_after(0.3))
    assert time.time() - t0 < 30
    assert not [c for c in mp.active_children() if c not in before]


@pytest.mark.skipif(not any(importlib.util.find_spec(m) for m in ("audio_separator", "demucs")),
                    reason="a separator is required")
def test_cancel_kills_the_separation_child(tmp_path):
    before = set(mp.active_children())
    t0 = time.time()
    with pytest.raises(RuntimeError, match="cancelled by user"):
        separate_in_child(_silent_wav(tmp_path / "a.wav"), tmp_path, "auto",
                          cancel_check=_cancel_after(0.3))
    assert time.time() - t0 < 30
    assert not [c for c in mp.active_children() if c not in before]


def test_a_killed_download_is_a_cancel_not_a_link_error(tmp_path, monkeypatch):
    class KilledDownload:
        video_title = ""

        def _ensure_ffmpeg(self):
            pass

        def _ingest_source(self, src):
            raise RuntimeError("yt-dlp failed: process killed")

    monkeypatch.setattr(orch_mod, "_legacy_pipeline",
                        lambda cfg, on_progress=None, cancel_check=None: KilledDownload())
    cfg = DialogueConfig(source="https://www.youtube.com/watch?v=abcdefgh", work_dir=tmp_path,
                         output_dir=tmp_path)
    with pytest.raises(orch_mod.Cancelled):
        default_acquire(cfg, tmp_path, cancel_check=lambda: True)
    with pytest.raises(orch_mod.AcquireError):
        default_acquire(cfg, tmp_path, cancel_check=lambda: False)
    assert not (tmp_path / "source_title.txt").exists()


@needs_ffmpeg
def test_cancel_during_separation_stops_before_transcription(tmp_path):
    calls, started = {}, {}
    comps = _components(calls)

    def separate(orig, work, policy, cancel_check=None):
        started["at"] = time.time()
        while not cancel_check():
            if time.time() - started["at"] > 10:
                raise AssertionError("the job's cancel flag never reached the separator")
            time.sleep(0.02)
        raise RuntimeError("Job cancelled by user")

    comps.separate = separate
    cfg = DialogueConfig(source=str(_make_media(tmp_path)), work_dir=tmp_path / "work",
                         output_dir=tmp_path / "out", tts_providers=["mock"])
    res = DialogueOrchestrator(cfg, comps, cancel_check=lambda: "at" in started).run()
    assert res.status == "cancelled"
    assert "asr" not in calls and "diarize" not in calls


@needs_ffmpeg
def test_cancel_kills_a_blocking_asr_child_and_stops_its_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setattr(DialogueOrchestrator, "HEARTBEAT_S", 0.05)
    comps = _components({})
    state = {"started": []}

    def asr(wav, on_progress=None):
        # Like pipeline._transcribe_local: each model runs in a child joined
        # without cancel polling, then a progress report before the next one.
        for model in ("large-v3", "medium"):
            p = mp.get_context("spawn").Process(target=time.sleep, args=(60,), daemon=True)
            p.start()
            state["started"].append(model)
            p.join(60)
            on_progress("transcribe", 0.15, f"Whisper {model} failed (exit {p.exitcode}); next")
        raise AssertionError("the fallback ladder was not stopped")

    comps.asr = asr
    cfg = DialogueConfig(source=str(_make_media(tmp_path)), work_dir=tmp_path / "work",
                         output_dir=tmp_path / "out", tts_providers=["mock"], background="none")
    t0 = time.time()
    res = DialogueOrchestrator(cfg, comps, cancel_check=lambda: bool(state["started"])).run()
    assert res.status == "cancelled"
    assert state["started"] == ["large-v3"] and time.time() - t0 < 45


# ── progress the UI can trust ─────────────────────────────────────────────
@needs_ffmpeg
def test_url_job_progress_only_moves_forward_and_legacy_pipelines_reach_the_caller(
        tmp_path, monkeypatch):
    media = _make_media(tmp_path)
    made = []

    class FakeLegacy:
        """Stands in for pipeline.Pipeline on the link-download path."""
        video_title = "  A Story \n Title "

        def __init__(self, on_progress, cancel_check):
            self.on_progress, self.cancel_check = on_progress, cancel_check
            made.append(self)

        def _ensure_ffmpeg(self):
            pass

        def _ingest_source(self, src):
            self.on_progress("download", 0.5, "[50%] Downloading: 5 / 10 MB (50%)")
            self.on_progress("download", 1.0, "Downloading: 10 / 10 MB (100%)")
            return media

    monkeypatch.setattr(orch_mod, "_legacy_pipeline",
                        lambda cfg, on_progress=None, cancel_check=None: FakeLegacy(on_progress,
                                                                                    cancel_check))
    monkeypatch.setattr(DialogueOrchestrator, "HEARTBEAT_S", 0.05)
    seen = {}
    comps = _components({})

    def separate(orig, work, policy, cancel_check=None):
        seen["separate_cancel"] = cancel_check is not None
        time.sleep(0.5)
        return {"status": "failed", "background": None, "vocals": None, "detail": "test"}

    def diarize(wav, heartbeat=None, cancel_check=None):
        seen["diarize_hooks"] = (heartbeat is not None, cancel_check is not None)
        heartbeat(12.0)
        return _diar()

    comps.separate, comps.diarize = separate, diarize
    events, handed = [], []
    cfg = DialogueConfig(source="https://www.youtube.com/watch?v=abcdefgh", work_dir=tmp_path / "work",
                         output_dir=tmp_path / "out", tts_providers=["mock"], content_verify="off")
    res = run_dialogue(cfg, on_progress=lambda s, p, m: events.append((s, p, m)),
                       components=comps, on_legacy_pipeline=handed.append)
    assert res.status != "failed", res.reasons
    # the download's Pipeline went to the caller (app.py: job.pipeline_ref) with the cancel flag
    assert handed == made and len(made) == 1 and made[0].cancel_check is not None
    assert (tmp_path / "work" / "source_title.txt").read_text(encoding="utf-8") == "A Story Title"
    assert seen == {"separate_cancel": True, "diarize_hooks": (True, True)}

    order = ["download", "extract", "transcribe", "translate", "synthesize", "assemble"]
    assert {s for s, _, _ in events} <= set(order)       # nothing the UI would count as 100%
    last = (-1, -1.0)
    for s, p, m in events:
        assert (order.index(s), p) >= last, f"progress went back to {s} {p:.3f} ({m})"
        last = (order.index(s), p)
        assert "%" in m, m
    download = [p for s, p, _ in events if s == "download"]
    assert max(download[:-1]) <= 0.95 and download[-1] == 1.0
    assert any(s == "extract" and p < 1.0 and "Separating voice from music..." in m
               for s, p, m in events)
    assert any(s == "transcribe" and "Detecting speakers... 12s" in m for s, p, m in events)


# ── one voice, analysed; honest failures ──────────────────────────────────
@needs_ffmpeg
def test_single_voice_narrator_is_one_analysed_speaker(tmp_path):
    lines = [(s, e, text) for spk, _, s, e, text in SCRIPT if spk == "SPEAKER_01"]   # the woman
    media = _tone_media(tmp_path, [(225.0, s, e) for s, e, _ in lines])
    holder = {}
    comps = _components({}, orch_holder=holder)
    comps.asr = lambda wav: [seg for seg in _asr_segments() if seg["text"] in {t for _, _, t in lines}]
    cfg = DialogueConfig(source=str(media), work_dir=tmp_path / "work", output_dir=tmp_path / "out",
                         tts_providers=["mock"], diarization=False, story_brief=False)
    orch = DialogueOrchestrator(cfg, comps)
    holder["orch"] = orch
    res = orch.run()
    assert {t.speaker_id for t in orch.turns} == {"SPEAKER_00"}
    rec = orch.registry.speakers["SPEAKER_00"]
    assert rec.voice_category == "female_like" and rec.mapping_origin == "single_voice"
    assert UNKNOWN_SPEAKER not in orch.registry.speakers
    assert {c.voice for c in orch.clips.values() if c.accepted} == {"mock-female-1"}
    # one voice was the user's choice: a limitation, not a warning
    assert res.status == "completed", res.reasons


@needs_ffmpeg
def test_speaker_detection_switched_off_by_the_resolver_is_a_warning(tmp_path):
    orch, res = _go(tmp_path, cfg={"diarization": False, "modules": {"changes": [
        {"stage": "speakers", "action": "deactivated", "choice": "pyannote",
         "reason": "missing: HF_TOKEN", "fix": "free token from huggingface.co"}]}})
    assert res.status == "completed_with_warnings"
    assert any("speaker detection could not run on this PC (missing: HF_TOKEN" in r
               for r in res.reasons), res.reasons
    assert {t.speaker_id for t in orch.turns} == {"SPEAKER_00"}
    assert [s.status for s in orch.report.stages if s.name == "diarize"] == ["degraded"]


class BrokenLLM:
    model_id = "groq:test-model"

    def complete(self, system, user):
        raise TranslationEngineError("groq HTTP 401: invalid API key")


@needs_ffmpeg
def test_no_translated_lines_fails_before_tts_with_the_engine_errors(tmp_path):
    holder = {}
    comps = _components({}, orch_holder=holder)
    comps.llm_clients = [BrokenLLM()]           # and the basic fallback returns nothing
    cfg = DialogueConfig(source=str(_make_media(tmp_path)), work_dir=tmp_path / "work",
                         output_dir=tmp_path / "out", tts_providers=["mock"])
    orch = DialogueOrchestrator(cfg, comps)
    holder["orch"] = orch
    res = orch.run()
    assert res.status == "failed"
    assert any("Translation failed for 4/4 lines" in r and "HTTP 401" in r for r in res.reasons), \
        res.reasons
    assert orch.all_clips == [] and res.output_video is None
    stages = {s.name: s.status for s in orch.report.stages}
    assert stages["translate"] == "failed" and "synthesize" not in stages
