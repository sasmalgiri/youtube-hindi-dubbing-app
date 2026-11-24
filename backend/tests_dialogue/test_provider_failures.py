"""T08: voice provider failures. 429/5xx are retried briefly; a provider that
stays down hands the line to Sarvam; a line nobody can voice fails a hosted
job (so its time is returned) instead of delivering a dub with a missing line."""
import base64
import shutil
import subprocess

import pytest

from dubbing import chirp_tts as C

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")


class R:
    def __init__(self, code, body=None, headers=None):
        self.status_code, self._b, self.headers = code, body or {}, headers or {}
        self.text = str(body)

    def json(self):
        return self._b


def _wav(tmp_path):
    w = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=0.5", str(w)], check=True)
    return w.read_bytes()


def test_rate_limit_then_success_is_retried(tmp_path):
    wav, calls, slept = _wav(tmp_path), [], []
    replies = [R(429, headers={"Retry-After": "2"}), R(503), R(200, {"audioContent": base64.b64encode(wav).decode()})]

    def post(url, params, json, timeout):
        calls.append(1)
        return replies.pop(0)
    assert C.synthesize("नमस्ते", "hi-IN-Chirp3-HD-Kore", key="k", post=post, sleep=slept.append) == wav
    assert len(calls) == 3 and slept == [2.0, 2.0]


def test_long_retry_after_means_quota_spent_no_waiting():
    slept = []
    with pytest.raises(C.ChirpError) as e:
        C.synthesize("x", "hi-IN-Chirp3-HD-Kore", key="k", sleep=slept.append,
                     post=lambda *a, **k: R(429, headers={"Retry-After": "3600"}))
    assert "429" in str(e.value) and slept == []


def test_network_errors_retry_then_give_up():
    n = []

    def post(*a, **k):
        n.append(1)
        raise ConnectionError("reset")
    with pytest.raises(C.ChirpError) as e:
        C.synthesize("x", "v", key="k", post=post, sleep=lambda s: None)
    assert len(n) == C.MAX_ATTEMPTS and "unreachable" in str(e.value)


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    from pipeline import Pipeline, PipelineConfig
    monkeypatch.setenv("GOOGLE_TTS_API_KEY", "k")
    cfg = PipelineConfig(source="x.mp4", work_dir=tmp_path / "w", output_path=tmp_path / "o.mp4", target_language="hi")
    cfg.work_dir.mkdir(parents=True)
    return Pipeline(cfg)


SEGS = [{"start": 0.0, "end": 1.0, "text_translated": "पहली लाइन"},
        {"start": 1.0, "end": 2.0, "text_translated": "दूसरी लाइन"}]


def test_outage_falls_back_to_sarvam_per_line(pipe, tmp_path, monkeypatch):
    wav = _wav(tmp_path)

    def synth(text, voice, sr):
        if "दूसरी" in text:
            raise C.ChirpError("Google TTS HTTP 503")
        return wav
    monkeypatch.setattr(C, "synthesize", synth)

    def sarvam(text, mp3):
        shutil.copy2(tmp_path / "tone.wav", mp3)
        return True
    monkeypatch.setattr(pipe, "_sarvam_tts_single_mp3", sarvam)
    out = pipe._tts_google_rest([dict(s) for s in SEGS])
    assert len(out) == 2


def test_total_outage_fails_a_hosted_job_but_not_a_local_one(pipe, tmp_path, monkeypatch):
    wav = _wav(tmp_path)
    monkeypatch.setattr(C, "synthesize", lambda t, v, sr: wav if "पहली" in t else (_ for _ in ()).throw(C.ChirpError("503")))
    monkeypatch.setattr(pipe, "_sarvam_tts_single_mp3", lambda text, mp3: False)
    monkeypatch.setattr(pipe, "_edge_tts_single", None, raising=False)
    monkeypatch.setenv("VOICEDUB_PROFILE", "hosted")
    with pytest.raises(RuntimeError, match="could not be voiced"):
        pipe._tts_google_rest([dict(s) for s in SEGS])
    monkeypatch.setenv("VOICEDUB_PROFILE", "local")

    async def edge_fails(text, mp3):
        raise RuntimeError("edge down")
    monkeypatch.setattr(pipe, "_edge_tts_single", edge_fails)
    out = pipe._tts_google_rest([dict(s) for s in SEGS])
    assert len(out) == 1                      # local: delivered with a reported gap
