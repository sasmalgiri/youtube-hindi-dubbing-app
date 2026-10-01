"""Persistent local-model workers (protocol, crash handling) and the providers
that use them. A stand-in worker replaces the real model scripts, which need
gated weights and a GPU."""
import sys
import textwrap

import pytest

from dubbing.dialogue.contracts import CATEGORY_FEMALE, CATEGORY_MALE, Turn
from dubbing.dialogue.local_workers import PersistentWorker, WorkerError
from dubbing.dialogue.mt_engines import IndicTrans2MT
from dubbing.dialogue.speaker_registry import SpeakerRegistry
from dubbing.dialogue.translation import DialogueTranslator
from dubbing.dialogue.tts import IndicParlerProvider, TTSRouter

FAKE = textwrap.dedent('''
    import json, sys, math, wave, array
    proto = sys.stdout
    sys.stdout = sys.stderr
    print("library noise that must not reach the protocol stream")
    def reply(**kw):
        proto.write(json.dumps(kw, ensure_ascii=False) + "\\n"); proto.flush()
    for line in sys.stdin:
        req = json.loads(line)
        op = req["op"]
        if op == "quit":
            reply(ok=True); break
        if op == "init":
            reply(ok=True, info={"device": "cpu", "sampling_rate": 24000})
        elif op == "translate":
            reply(ok=True, texts=["हिं:" + t for t in req["texts"]])
        elif op == "tts":
            if "FAIL" in req["text"]:
                reply(ok=False, error="model error"); continue
            if "CRASH" in req["text"]:
                sys.exit(3)
            sr = 24000
            f0 = 120 if "Rohit" in req["description"] or "Aman" in req["description"] else 220
            data = array.array("h", (int(6000 * math.sin(2 * math.pi * f0 * n / sr))
                                     for n in range(int(0.05 * len(req["text"]) * sr))))
            with wave.open(req["out"], "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes(data.tobytes())
            reply(ok=True, out=req["out"], sampling_rate=sr)
''')


@pytest.fixture()
def fake_script(tmp_path):
    p = tmp_path / "fake_worker.py"
    p.write_text(FAKE, encoding="utf-8")
    return p


def test_worker_roundtrip_ignores_library_prints(fake_script):
    w = PersistentWorker("indictrans2", script=fake_script, python=sys.executable, timeout=30)
    assert w.request({"op": "translate", "texts": ["hello"]})["texts"] == ["हिं:hello"]
    assert w.info["device"] == "cpu"
    w.close()


def test_worker_error_and_crash_are_reported_and_recoverable(fake_script, tmp_path):
    w = PersistentWorker("parler", script=fake_script, python=sys.executable, timeout=30)
    with pytest.raises(WorkerError, match="model error"):
        w.request({"op": "tts", "text": "FAIL", "description": "Rohit", "out": str(tmp_path / "a.wav")})
    with pytest.raises(WorkerError):
        w.request({"op": "tts", "text": "CRASH", "description": "Rohit", "out": str(tmp_path / "b.wav")})
    # the next request restarts the worker
    r = w.request({"op": "tts", "text": "नमस्ते", "description": "Divya", "out": str(tmp_path / "c.wav")})
    assert r["sampling_rate"] == 24000
    w.close()


def test_indictrans2_engine_through_translator(fake_script):
    mt = IndicTrans2MT()
    mt.worker = PersistentWorker("indictrans2", script=fake_script, python=sys.executable, timeout=30)
    turns = [Turn("t0001", "A", 0, 1, source_text="Hello there."),
             Turn("t0002", "B", 1, 2, source_text="Hi.")]
    tr = DialogueTranslator([], mt_engines=[mt])
    tr.translate(turns)
    assert [t.hi_raw for t in turns] == ["हिं:Hello there.", "हिं:Hi."]
    assert all("sentence_level_mt" in t.flags for t in turns)
    assert not any(w["type"] == "sentence_level_mt" for w in tr.warnings)   # limitation, not per-turn warning
    mt.close()


def test_indic_parler_speakers_stay_bound_per_character(fake_script, tmp_path):
    reg = SpeakerRegistry()
    reg.register("M1", voice_category=CATEGORY_MALE, total_speech_s=30)
    reg.register("F1", voice_category=CATEGORY_FEMALE, total_speech_s=20)
    reg.register("M2", voice_category=CATEGORY_MALE, total_speech_s=10)
    prov = IndicParlerProvider()
    prov.worker = PersistentWorker("parler", script=fake_script, python=sys.executable, timeout=30)
    router = TTSRouter({"indic_parler": prov}, reg, ["indic_parler"], tmp_path)
    voices = {}
    for tid, spk in (("t1", "M1"), ("t2", "F1"), ("t3", "M2"), ("t4", "M1")):
        c = router.synthesize(Turn(tid, spk, 0, 1, hi_fit="यह एक परीक्षण वाक्य है।"))
        voices.setdefault(spk, set()).add(c.voice)
    assert voices == {"M1": {"Rohit"}, "F1": {"Divya"}, "M2": {"Aman"}}
    assert "Rohit's voice" in prov.description(reg.resolve_voice("M1", "indic_parler"))
    prov.close()
