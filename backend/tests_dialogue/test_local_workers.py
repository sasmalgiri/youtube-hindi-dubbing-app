"""Persistent local-model workers (protocol, crash handling, runtime checks)
and the providers that use them. A stand-in worker replaces the real model
scripts, which need gated weights and a GPU; the real scripts' start-up is
exercised against stub libraries."""
import json
import os
import sys
import textwrap

import pytest

from dubbing.dialogue import local_workers as lw
from dubbing.dialogue.contracts import CATEGORY_FEMALE, CATEGORY_MALE, Turn
from dubbing.dialogue.local_workers import PersistentWorker, WorkerError
from dubbing.dialogue.mt_engines import IndicTrans2MT
from dubbing.dialogue.speaker_registry import SpeakerRegistry
from dubbing.dialogue.translation import DialogueTranslator
from dubbing.dialogue.tts import IndicParlerProvider, MockProvider, TTSRouter

FAKE = textwrap.dedent('''
    import json, os, sys, math, wave, array
    proto = sys.stdout
    sys.stdout = sys.stderr
    print("library noise that must not reach the protocol stream")
    if os.environ.get("FAKE_STARTS"):
        with open(os.environ["FAKE_STARTS"], "a") as f:
            f.write("start\\n")
    def reply(**kw):
        proto.write(json.dumps(kw, ensure_ascii=False) + "\\n"); proto.flush()
    loaded = False
    for line in sys.stdin:
        req = json.loads(line)
        op = req["op"]
        if op == "quit":
            reply(ok=True); break
        if op == "init":
            if os.environ.get("FAKE_INIT_ERROR"):
                reply(ok=False, error=os.environ["FAKE_INIT_ERROR"]); continue
            loaded = True
            reply(ok=True, info={"device": "cpu", "sampling_rate": 24000})
        elif not loaded:
            reply(ok=False, error="KeyError: 'torch'")   # a half-initialised worker
        elif op == "translate":
            if "NOISE" in req["texts"]:
                proto.write("a C library printed this\\n"); proto.flush()
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


def test_model_load_failure_keeps_its_reason_and_is_not_reloaded_per_line(fake_script, tmp_path,
                                                                          monkeypatch):
    # e.g. a gated checkpoint: every later line must name that cause (not a
    # half-initialised worker's KeyError) and the slow load is not repeated.
    starts = tmp_path / "starts.txt"
    monkeypatch.setenv("FAKE_STARTS", str(starts))
    monkeypatch.setenv("FAKE_INIT_ERROR", "OSError: 403 Client Error (gated repo)")
    w = PersistentWorker("parler", script=fake_script, python=sys.executable, timeout=30)
    for _ in range(3):
        with pytest.raises(WorkerError, match="failed to start.*403"):
            w.request({"op": "tts", "text": "नमस्ते", "description": "Rohit",
                       "out": str(tmp_path / "a.wav")})
    assert starts.read_text().count("start") == 1
    w.close()


def test_stray_stdout_line_restarts_the_worker_instead_of_shifting_replies(fake_script):
    w = PersistentWorker("indictrans2", script=fake_script, python=sys.executable, timeout=30)
    with pytest.raises(WorkerError, match="non-protocol line"):
        w.request({"op": "translate", "texts": ["NOISE"]})
    # answered by a fresh worker, not by the stale reply to the previous request
    assert w.request({"op": "translate", "texts": ["hello"]})["texts"] == ["हिं:hello"]
    w.close()


def test_parler_that_cannot_run_is_replaced_visibly_and_speakers_pinned(fake_script, tmp_path,
                                                                        monkeypatch):
    monkeypatch.setenv("FAKE_INIT_ERROR", "RuntimeError: expected m1 and m2 to have the same dtype")
    reg = SpeakerRegistry()
    reg.register("M1", voice_category=CATEGORY_MALE, total_speech_s=30)
    reg.register("F1", voice_category=CATEGORY_FEMALE, total_speech_s=20)
    reg.pools["mock2"] = {"model": "m2", "supports_pitch": False,
                          CATEGORY_MALE: ["b-male"], CATEGORY_FEMALE: ["b-female"]}
    prov = IndicParlerProvider()
    prov.worker = PersistentWorker("parler", script=fake_script, python=sys.executable, timeout=30)
    router = TTSRouter({"indic_parler": prov, "mock2": MockProvider(name="mock2", native_rate=True)},
                       reg, ["indic_parler", "mock2"], tmp_path, max_retries=1)
    turns = {f"t{i}": Turn(f"t{i}", spk, i, i + 1, hi_fit="यह एक परीक्षण वाक्य है।")
             for i, spk in enumerate(("M1", "F1", "M1"))}
    clips = {tid: router.synthesize(t) for tid, t in turns.items()}
    assert {c.provider for c in clips.values()} == {"mock2"} and all(c.degraded for c in clips.values())
    assert router.reroute_mixed_speakers(turns, clips) == []
    # fit/verify regenerate with the provider that works, at its native rate
    assert router.speaker_provider == {"M1": "mock2", "F1": "mock2"}
    assert router.providers_for("M1")[0] == "mock2" and router.supports_native_rate("M1")
    assert "same dtype" in reg.speakers["M1"].fallback_history[0]["reason"]
    prov.close()


# ── runtime check: "runnable" only when it really is ─────────────────────────
def test_unmet_requirements_reads_the_library_pins(monkeypatch):
    import importlib.metadata as md
    installed = {"transformers": "4.51.3", "torch": "2.4.1+cu121"}

    def version(name):
        if name not in installed:
            raise md.PackageNotFoundError(name)
        return installed[name]
    monkeypatch.setattr(md, "requires", lambda dist: [
        "transformers<=4.46.1,>=4.46.1", "torch", 'black~=23.1; extra == "dev"', "sentencepiece"])
    monkeypatch.setattr(md, "version", version)
    assert lw.unmet_requirements("parler-tts") == [
        "transformers<=4.46.1,>=4.46.1 (installed: 4.51.3)", "sentencepiece (installed: none)"]


def test_parler_in_the_main_python_with_wrong_transformers_needs_its_own_env(monkeypatch):
    monkeypatch.delenv("INDIC_PARLER_PYTHON", raising=False)
    monkeypatch.setattr(lw, "_RUNTIME_CACHE", {})
    monkeypatch.setattr(lw, "_module_installed", lambda name: True)   # imports would pass
    monkeypatch.setattr(lw, "unmet_requirements",
                        lambda dist: ["transformers<=4.46.1,>=4.46.1 (installed: 4.51.3)"])
    ok, why = lw.runtime_status("parler")
    assert not ok and not lw.runtime_available("parler")
    assert "needs its own Python env (INDIC_PARLER_PYTHON)" in why and "4.51.3" in why
    monkeypatch.setattr(lw, "_RUNTIME_CACHE", {})
    monkeypatch.setattr(lw, "unmet_requirements", lambda dist: [])
    assert lw.runtime_available("parler")


def _fake_dist(site, requires):
    d = site / "fakedist-1.0.dist-info"
    d.mkdir(parents=True)
    (d / "METADATA").write_text("Metadata-Version: 2.1\nName: fakedist\nVersion: 1.0\n"
                                f"Requires-Dist: {requires}\n", encoding="utf-8")
    return site


def test_separate_interpreter_is_checked_inside_that_interpreter(monkeypatch, tmp_path):
    # The same Python under another spelling, so the subprocess check runs for
    # real; a fake distribution stands in for parler-tts and its pins.
    alt = os.path.join(os.path.dirname(sys.executable), ".", os.path.basename(sys.executable))
    monkeypatch.setenv("INDIC_PARLER_PYTHON", alt)
    monkeypatch.setitem(lw.DISTRIBUTION, "parler", "fakedist")
    monkeypatch.setitem(lw.CHECK_MODULES, "parler", ["json"])

    def status(site):
        monkeypatch.setenv("PYTHONPATH", str(site))
        monkeypatch.setattr(lw, "_RUNTIME_CACHE", {})
        return lw.runtime_status("parler")

    ok, why = status(_fake_dist(tmp_path / "good", "pytest>=1"))
    assert ok, why
    ok, why = status(_fake_dist(tmp_path / "pinned", "pytest<1"))
    assert not ok and "does not satisfy fakedist: pytest<1" in why
    monkeypatch.setitem(lw.CHECK_MODULES, "parler", ["json", "no_such_module_xyz"])
    ok, why = status(tmp_path / "good")
    assert not ok and "no_such_module_xyz" in why
    monkeypatch.setenv("INDIC_PARLER_PYTHON", str(tmp_path / "missing" / "python.exe"))
    ok, why = status(tmp_path / "good")
    assert not ok and "missing interpreter" in why


# ── the real worker scripts, started against stub libraries ──────────────────
STUB_TORCH = textwrap.dedent('''
    import os

    class _DType:
        def __init__(self, name):
            self.name = name

        def __str__(self):
            return "torch." + self.name

    float32, float16, bfloat16 = _DType("float32"), _DType("float16"), _DType("bfloat16")

    class cuda:
        @staticmethod
        def is_available():
            return os.environ.get("STUB_CUDA") == "1"

        @staticmethod
        def is_bf16_supported():
            return os.environ.get("STUB_BF16") == "1"
''')
STUB_TRANSFORMERS = textwrap.dedent('''
    import json, os

    def log(**kw):
        with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\\n")

    class Model:
        def __init__(self):
            enc = type("Enc", (), {"_name_or_path": "google/flan-t5-large"})()
            self.config = type("Cfg", (), {"sampling_rate": 44100, "text_encoder": enc})()

        def to(self, *args, **kwargs):
            log(call="to", args=[str(a) for a in args], kwargs={k: str(v) for k, v in kwargs.items()})
            return self

        def eval(self):
            return self

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return object()

    class AutoModelForSeq2SeqLM:
        @staticmethod
        def from_pretrained(name, **kwargs):
            log(call="from_pretrained", name=name, kwargs=sorted(kwargs))
            return Model()
''')
STUB_PARLER = textwrap.dedent('''
    from transformers import Model, log

    class ParlerTTSForConditionalGeneration:
        @staticmethod
        def from_pretrained(name, **kwargs):
            log(call="from_pretrained", name=name, kwargs=sorted(kwargs))
            return Model()
''')


@pytest.fixture()
def stub_libs(tmp_path, monkeypatch):
    root = tmp_path / "stubs"
    files = {"torch/__init__.py": STUB_TORCH, "transformers/__init__.py": STUB_TRANSFORMERS,
             "parler_tts/__init__.py": STUB_PARLER, "IndicTransToolkit/__init__.py": "",
             "IndicTransToolkit/processor.py":
                 "class IndicProcessor:\n    def __init__(self, inference=True):\n        pass\n"}
    for rel, src in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(src, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PYTHONPATH", str(root))
    monkeypatch.setenv("STUB_LOG", str(log))
    for var in ("INDIC_PARLER_PYTHON", "INDICTRANS2_PYTHON", "INDICTRANS2_MODEL", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return lambda: ([json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
                    if log.exists() else [])


def _start_real_worker(kind, init=None):
    w = PersistentWorker(kind, init=init, python=sys.executable, timeout=60)
    w.request({"op": "quit"})   # runs the worker's real init first
    w.close()
    return w.info


@pytest.mark.parametrize("bf16, dtype", [("1", "bfloat16"), ("0", "float32")])
def test_parler_worker_casts_the_whole_model_after_loading(stub_libs, monkeypatch, bf16, dtype):
    # from_pretrained(torch_dtype=bf16) left the sub-models in fp32, so every
    # line failed with a dtype mismatch and silently fell back to Edge.
    monkeypatch.setenv("STUB_CUDA", "1")
    monkeypatch.setenv("STUB_BF16", bf16)
    info = _start_real_worker("parler", {"model": "ai4bharat/indic-parler-tts"})
    assert info["device"] == "cuda" and info["dtype"] == dtype
    calls = stub_libs()
    assert [c["kwargs"] for c in calls if c["call"] == "from_pretrained"] == \
        [["attn_implementation", "token"]]                    # no torch_dtype
    assert [c for c in calls if c["call"] == "to"] == \
        [{"call": "to", "args": ["cuda"], "kwargs": {"dtype": f"torch.{dtype}"}}]


def test_indictrans2_worker_defaults_to_the_accessible_1b_model(stub_libs, monkeypatch):
    assert _start_real_worker("indictrans2")["model"] == "ai4bharat/indictrans2-en-indic-1B"
    monkeypatch.setenv("INDICTRANS2_MODEL", "ai4bharat/indictrans2-en-indic-dist-200M")
    assert _start_real_worker("indictrans2")["model"] == "ai4bharat/indictrans2-en-indic-dist-200M"
