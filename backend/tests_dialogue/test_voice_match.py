"""'Sound like the original speaker' (OpenVoice tone colour): the matcher's
protocol with a stand-in worker, its gates and failure handling, and the
real worker script started against stub libraries (the real model needs a
separate venv and a checkpoint download)."""
import array
import json
import math
import sys
import textwrap
import types
import wave
from pathlib import Path

import pytest

from dubbing.dialogue import local_workers as lw
from dubbing.dialogue import voice_match
from dubbing.dialogue.local_workers import PersistentWorker
from dubbing.dialogue.voice_match import MIN_REFERENCE_S, OpenVoiceMatcher, make_voice_matcher

FAKE = textwrap.dedent('''
    import json, os, sys, wave, array
    proto = sys.stdout
    sys.stdout = sys.stderr
    print("library noise that must not reach the protocol stream")
    def log(**kw):
        with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\\n")
    def reply(**kw):
        proto.write(json.dumps(kw, ensure_ascii=False) + "\\n"); proto.flush()
    targets = {}
    for line in sys.stdin:
        req = json.loads(line)
        op = req["op"]
        log(**req)
        if op == "quit":
            reply(ok=True); break
        if op == "init":
            if os.environ.get("FAKE_INIT_ERROR"):
                reply(ok=False, error=os.environ["FAKE_INIT_ERROR"]); continue
            reply(ok=True, info={"device": "cpu", "sampling_rate": 22050, "version": "v2"})
        elif op == "prepare":
            if "FAIL" in req["speaker"]:
                reply(ok=False, error="RuntimeError: reference encoder failed"); continue
            if req.get("se_path"):
                os.makedirs(os.path.dirname(req["se_path"]), exist_ok=True)
                open(req["se_path"], "w").write("se")
            targets[req["speaker"]] = len(req["refs"])
            reply(ok=True, speaker=req["speaker"], cached=False)
        elif op == "convert":
            if "CRASH" in req["in"]:
                sys.exit(3)
            if req["speaker"] not in targets:
                reply(ok=False, error="RuntimeError: speaker %s was not prepared" % req["speaker"])
                continue
            if "ERR" in req["in"]:
                reply(ok=False, error="RuntimeError: out of memory"); continue
            with wave.open(req["in"], "rb") as wf:
                params = wf.getparams()
                data = array.array("h", wf.readframes(wf.getnframes()))
            data = array.array("h", (v // 2 for v in data))       # "converted"
            if "LONG" in req["in"]:
                data.extend([0] * 4800)                            # breaks the timing
            tmp = req["out"] + ".tmp"
            with wave.open(tmp, "wb") as wf:
                wf.setparams(params)
                wf.writeframes(data.tobytes())
            os.replace(tmp, req["out"])
            reply(ok=True, out=req["out"], sampling_rate=params.framerate)
''')


def _tone(path: Path, seconds: float, sr: int = 24000, f0: float = 150.0) -> Path:
    data = array.array("h", (int(8000 * math.sin(2 * math.pi * f0 * n / sr))
                             for n in range(int(seconds * sr))))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return path


def _frames(path: Path):
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate(), array.array("h", wf.readframes(wf.getnframes()))


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_openvoice.py"
    script.write_text(FAKE, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))

    def make(cache_dir=None):
        m = OpenVoiceMatcher(cache_dir=cache_dir, models_dir=tmp_path / "models")
        m.worker = PersistentWorker("openvoice", init=m.worker.init, script=script,
                                    python=sys.executable, timeout=30)
        return m

    make.calls = lambda: ([json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
                          if log.exists() else [])
    return make


def test_prepare_convert_close_protocol(fake, tmp_path):
    m = fake(cache_dir=tmp_path / "work" / "voice_match")
    assert m.name == "openvoice"
    refs = [_tone(tmp_path / "ref_A_1.wav", 2.0), _tone(tmp_path / "ref_A_2.wav", 1.5)]
    assert m.prepare("SPEAKER_00", refs) is True
    clip = _tone(tmp_path / "t0001.wav", 1.2, sr=48000)
    out = tmp_path / "out" / "t0001_vm.wav"
    assert m.convert(clip, out, "SPEAKER_00") is True
    sr_in, x = _frames(clip)
    sr_out, y = _frames(out)
    assert sr_out == sr_in == 48000 and len(y) == len(x)         # timing unchanged
    assert y != x                                                 # the converted audio
    assert m.convert(clip, clip, "SPEAKER_00") is True            # in place
    assert len(_frames(clip)[1]) == len(x)
    calls = fake.calls()
    assert [c["op"] for c in calls] == ["init", "prepare", "convert", "convert"]
    assert calls[0]["models_dir"] == str(tmp_path / "models") and calls[0]["tau"] == 0.3
    assert calls[1]["speaker"] == "SPEAKER_00" and calls[1]["refs"] == [str(p) for p in refs]
    se = Path(calls[1]["se_path"])                                # cached in this job's work_dir
    assert se.parent == tmp_path / "work" / "voice_match" and se.name.startswith("se_SPEAKER_00_")
    assert se.exists()
    assert calls[2]["in"] == str(clip) and calls[2]["out"] == str(out)
    assert calls[2]["source_key"] is None and not m.errors
    m.close()
    assert m.worker._proc is None
    assert fake.calls()[-1]["op"] == "quit"


def test_short_reference_is_left_unconverted_without_loading_the_model(fake, tmp_path):
    m = fake()
    short = [_tone(tmp_path / "r1.wav", 1.5), _tone(tmp_path / "r2.wav", 1.4),
             tmp_path / "missing.wav"]                            # 2.9 s in all
    assert MIN_REFERENCE_S == 3.0
    assert m.prepare("SPEAKER_01", short) is False
    assert "2.9 s" in m.notes["SPEAKER_01"]
    clip = _tone(tmp_path / "t0002.wav", 1.0)
    out = tmp_path / "t0002_vm.wav"
    assert m.convert(clip, out, "SPEAKER_01") is False and not out.exists()
    assert m.convert(clip, out, "NEVER_PREPARED") is False
    assert fake.calls() == []                                     # no worker, no model load
    assert m.prepare("SPEAKER_01", short[:2] + [_tone(tmp_path / "r3.wav", 0.2)]) is True
    m.close()


def test_failures_return_false_and_keep_the_clip(fake, tmp_path):
    m = fake()
    refs = [_tone(tmp_path / "ref.wav", 4.0)]
    assert m.prepare("FAIL_SPK", refs) is False
    assert "reference encoder failed" in m.notes["FAIL_SPK"] and m.errors
    assert m.prepare("S1", refs) is True

    err = _tone(tmp_path / "ERR.wav", 1.0)
    before = err.read_bytes()
    assert m.convert(err, err, "S1") is False and err.read_bytes() == before
    assert any("out of memory" in e for e in m.errors)

    long_clip = _tone(tmp_path / "LONG.wav", 1.0)
    out = tmp_path / "LONG_vm.wav"
    assert m.convert(long_clip, out, "S1") is False and not out.exists()
    assert any("sample rate/length" in e for e in m.errors)

    tiny = _tone(tmp_path / "tiny.wav", 0.1)                      # too short to convert
    assert m.convert(tiny, tmp_path / "tiny_vm.wav", "S1") is False
    not_wav = tmp_path / "x.wav"
    not_wav.write_bytes(b"not audio")
    assert m.convert(not_wav, tmp_path / "x_vm.wav", "S1") is False

    # a crashed worker restarts and the speaker is prepared again on its own
    crash = _tone(tmp_path / "CRASH.wav", 1.0)
    assert m.convert(crash, tmp_path / "c_vm.wav", "S1") is False
    ok = _tone(tmp_path / "t0003.wav", 1.0)
    assert m.convert(ok, tmp_path / "t0003_vm.wav", "S1") is True
    ops = [c["op"] for c in fake.calls()]
    assert ops[-4:] == ["init", "convert", "prepare", "convert"]
    m.close()


def test_worker_that_cannot_load_fails_every_call_softly(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_INIT_ERROR", "OSError: checkpoint download failed")
    m = fake()
    assert m.prepare("S1", [_tone(tmp_path / "ref.wav", 4.0)]) is False
    assert m.prepare("S2", [_tone(tmp_path / "ref2.wav", 4.0)]) is False
    assert all("checkpoint download failed" in e for e in m.errors)
    ops = [c["op"] for c in fake.calls()]
    assert ops.count("init") == 1 and "prepare" not in ops        # not reloaded per speaker
    m.close()


def test_make_voice_matcher_needs_the_runtime(monkeypatch, tmp_path):
    cfg = types.SimpleNamespace(voice_match="openvoice", work_dir=tmp_path / "work")
    monkeypatch.setattr(lw, "_RUNTIME_CACHE", {})
    monkeypatch.setenv("OPENVOICE_PYTHON", str(tmp_path / "missing" / "python.exe"))
    assert make_voice_matcher(cfg) is None
    ok, why = voice_match.runtime_status()
    assert not ok and "OPENVOICE_PYTHON" in why

    monkeypatch.setattr(lw, "runtime_status", lambda kind, timeout=90.0: (True, "runnable"))
    m = make_voice_matcher(cfg)
    assert isinstance(m, OpenVoiceMatcher) and m.cache_dir == tmp_path / "work" / "voice_match"
    assert m.worker.kind == "openvoice" and m.worker._proc is None   # nothing started yet
    assert make_voice_matcher(types.SimpleNamespace(voice_match="off", work_dir=tmp_path)) is None


def test_openvoice_runtime_is_registered_like_the_other_local_models(monkeypatch):
    assert lw.INTERPRETER_ENV["openvoice"] == "OPENVOICE_PYTHON"
    assert (lw.WORKER_DIR / lw.SCRIPT["openvoice"]).is_file()
    assert lw.CHECK_MODULES["openvoice"] == ["openvoice", "torch"]
    monkeypatch.delenv("OPENVOICE_PYTHON", raising=False)
    monkeypatch.setattr(lw, "_RUNTIME_CACHE", {})
    monkeypatch.setattr(lw, "_module_installed", lambda name: False)
    ok, why = lw.runtime_status("openvoice")
    assert not ok and "openvoice" in why


def test_doctor_lists_openvoice_as_an_optional_warning(monkeypatch, tmp_path):
    from dubbing.dialogue import preflight
    monkeypatch.delenv("OPENVOICE_PYTHON", raising=False)
    monkeypatch.setattr(lw, "runtime_status", lambda kind, timeout=90.0:
                        (False, "not installed in this Python: openvoice, torch"))

    def line():
        res = preflight.run_preflight(work_root=tmp_path)
        return [c for c in res["checks"] if c["check"].startswith("OpenVoice")][0]

    c = line()
    assert c["level"] == "warning" and "setup_local_ai.bat" in c["detail"]   # never blocking
    assert "optional" in c["check"]
    monkeypatch.setenv("OPENVOICE_PYTHON", str(tmp_path / "py.exe"))
    monkeypatch.setattr(lw, "runtime_status", lambda kind, timeout=90.0:
                        (False, "OPENVOICE_PYTHON points to a missing interpreter: x"))
    assert "missing interpreter" in line()["detail"]
    monkeypatch.setattr(lw, "runtime_status", lambda kind, timeout=90.0: (True, "runnable in py"))
    c = line()
    assert c["level"] == "ok" and c["detail"] == "runnable in py"


def test_openvoice_pins_that_the_converter_never_imports_are_not_required(monkeypatch):
    import importlib.metadata as md
    installed = {"numpy": "1.22.0", "librosa": "0.9.1"}

    def version(name):
        if name not in installed:
            raise md.PackageNotFoundError(name)
        return installed[name]
    monkeypatch.setattr(md, "requires", lambda dist: [
        "librosa==0.9.1", "numpy==1.22.0", "faster-whisper==0.9.0", "gradio==3.48.0",
        "whisper-timestamped==1.14.2", "wavmark==0.0.3"])
    monkeypatch.setattr(md, "version", version)
    assert lw.unmet_requirements("MyShell-OpenVoice") == ["wavmark==0.0.3 (installed: none)"]
    assert len(lw.unmet_requirements("parler-tts")) == 4           # only OpenVoice's are skipped


# ── the real worker script, started against stub libraries ──────────────────
STUB_TORCH = textwrap.dedent('''
    import json, os

    class T:
        def __init__(self, v):
            self.v = v

        def to(self, device):
            return self

        def mean(self, dim):
            return T(sum(t.v for t in self.v) / len(self.v))

    def stack(ts):
        return T(list(ts))

    def load(path, map_location=None, weights_only=False):
        with open(path) as f:
            return T(json.load(f))

    class cuda:
        @staticmethod
        def is_available():
            return os.environ.get("STUB_CUDA") == "1"
''')
STUB_OPENVOICE_API = textwrap.dedent('''
    import json, os, wave
    import numpy as np
    import torch

    def log(**kw):
        with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\\n")

    class _NS:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class ToneColorConverter:
        def __init__(self, config_path, device="cuda:0"):
            cfg = json.load(open(config_path))
            self.hps = _NS(data=_NS(sampling_rate=cfg["sampling_rate"]))
            self.version = "v2"
            log(call="init", config=config_path, device=device)

        def load_ckpt(self, path):
            log(call="load_ckpt", path=path)

        def extract_se(self, ref_wav_list, se_save_path=None):
            se = torch.T(sum(os.path.getsize(p) for p in ref_wav_list) / 1000.0)
            if se_save_path:
                json.dump(se.v, open(se_save_path, "w"))
            log(call="extract_se", refs=ref_wav_list, save=se_save_path)
            return se

        def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3,
                    message="default"):
            with wave.open(audio_src_path, "rb") as wf:
                sr, n = wf.getframerate(), wf.getnframes()
                x = np.frombuffer(wf.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
            m = int(round(n * 22050 / sr)) + int(os.environ.get("STUB_EXTRA", "0"))
            y = np.interp(np.linspace(0, n - 1, m), np.arange(n), x).astype(np.float32) * 0.5
            log(call="convert", src_se=src_se.v, tgt_se=tgt_se.v, tau=tau, message=message,
                output_path=output_path)
            return y
''')
STUB_LIBROSA = textwrap.dedent('''
    import numpy as np

    def resample(y, orig_sr, target_sr):
        m = int(round(len(y) * target_sr / orig_sr))
        return np.interp(np.linspace(0, len(y) - 1, m), np.arange(len(y)), y)
''')
STUB_HUB = textwrap.dedent('''
    import json, os

    def hf_hub_download(repo_id, filename, local_dir=None):
        path = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"sampling_rate": 22050}) if filename.endswith(".json") else "ckpt")
        with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as f:
            f.write(json.dumps({"call": "download", "repo": repo_id, "file": filename}) + "\\n")
        return path
''')


@pytest.fixture()
def stub_openvoice(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    root = tmp_path / "stubs"
    files = {"torch/__init__.py": STUB_TORCH, "openvoice/__init__.py": "",
             "openvoice/api.py": STUB_OPENVOICE_API, "librosa/__init__.py": STUB_LIBROSA,
             "huggingface_hub/__init__.py": STUB_HUB}
    for rel, src in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(src, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PYTHONPATH", str(root))
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.delenv("OPENVOICE_PYTHON", raising=False)

    def make(cache_dir=None):
        m = OpenVoiceMatcher(cache_dir=cache_dir, models_dir=tmp_path / "models")
        m.worker = PersistentWorker("openvoice", init=m.worker.init, python=sys.executable,
                                    timeout=60)
        return m

    make.calls = lambda: ([json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
                          if log.exists() else [])
    return make


@pytest.mark.parametrize("sr, extra", [(48000, 200), (24000, -150), (22050, 0)])
def test_real_worker_keeps_rate_and_exact_length(stub_openvoice, tmp_path, monkeypatch, sr, extra):
    # The converter runs at 22.05 kHz and its output can be a hop off: the
    # clip must come back at its own rate with exactly its own length.
    monkeypatch.setenv("STUB_EXTRA", str(extra))
    m = stub_openvoice(cache_dir=tmp_path / "work" / "voice_match")
    assert m.prepare("S1", [_tone(tmp_path / "ref1.wav", 2.0), _tone(tmp_path / "ref2.wav", 2.0)]), \
        m.errors
    clip = _tone(tmp_path / "t0001.wav", 1.37, sr=sr)
    n = len(_frames(clip)[1])
    assert m.convert(clip, clip, "S1") is True, m.errors
    rate, y = _frames(clip)
    assert rate == sr and len(y) == n
    assert max(abs(v) for v in y) > 1000                          # level matched, not silence
    assert m.worker.info["sampling_rate"] == 22050 and m.worker.info["device"] == "cpu"
    calls = stub_openvoice.calls()
    assert [c["file"] for c in calls if c["call"] == "download"] == \
        ["converter/config.json", "converter/checkpoint.pth"]
    assert all(c["repo"] == "myshell-ai/OpenVoiceV2" for c in calls if c["call"] == "download")
    conv = [c for c in calls if c["call"] == "convert"][0]
    assert conv["tau"] == 0.3 and conv["message"] == "@MyShell" and conv["output_path"] is None
    m.close()


def test_real_worker_refuses_a_retimed_conversion(stub_openvoice, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_EXTRA", "2205")                      # 100 ms longer
    m = stub_openvoice()
    assert m.prepare("S1", [_tone(tmp_path / "ref.wav", 3.5)])
    clip = _tone(tmp_path / "t0001.wav", 1.0)
    before = clip.read_bytes()
    assert m.convert(clip, clip, "S1") is False and clip.read_bytes() == before
    assert any("ms off the clip's length" in e for e in m.errors)
    m.close()


def test_real_worker_caches_embeddings_and_downloads_once(stub_openvoice, tmp_path):
    cache = tmp_path / "work" / "voice_match"
    refs = [_tone(tmp_path / "ref1.wav", 2.0), _tone(tmp_path / "ref2.wav", 2.0)]
    m = stub_openvoice(cache_dir=cache)
    assert m.prepare("S1", refs)
    # one TTS voice: its first clips form one averaged source embedding
    for i, secs in enumerate((1.2, 0.5, 1.5, 2.0, 1.1)):
        assert m.convert(_tone(tmp_path / f"c{i}.wav", secs), tmp_path / f"o{i}.wav", "S1",
                         source_key="edge|hi-IN-MadhurNeural|")
    m.close()
    calls = stub_openvoice.calls()
    per_clip = [c for c in calls if c["call"] == "extract_se" and c["save"] is None]
    assert len(per_clip) == 4                                     # 3 pooled (>= 1 s) + 1 short
    src = [c["src_se"] for c in calls if c["call"] == "convert"]
    assert src[1] == src[0]                                       # a short clip uses the pool
    assert src[2] != src[0] and src[3] != src[2]                  # the pool grows to 3 clips
    assert src[4] == src[3]                                       # then stays fixed
    # a new worker (e.g. a resumed run) reuses the saved embedding and the checkpoint
    m2 = stub_openvoice(cache_dir=cache)
    assert m2.prepare("S1", refs)
    m2.close()
    calls = stub_openvoice.calls()
    assert sum(c["call"] == "download" for c in calls) == 2
    assert sum(c["call"] == "extract_se" and c["save"] is not None for c in calls) == 1
