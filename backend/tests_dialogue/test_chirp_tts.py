"""Chirp 3 HD over REST: request shape, key handling, errors (no network)."""
import base64

import pytest

from dubbing import chirp_tts as C


class R:
    def __init__(self, code, body):
        self.status_code, self._b = code, body
        self.text = str(body)

    def json(self):
        return self._b


def test_request_has_no_pitch_and_derives_language():
    b = C.request_body("नमस्ते", "hi-IN-Chirp3-HD-Kore", 44100)
    assert b["voice"] == {"languageCode": "hi-IN", "name": "hi-IN-Chirp3-HD-Kore"}
    assert "pitch" not in b["audioConfig"] and b["audioConfig"]["sampleRateHertz"] == 44100


def test_voice_from_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_TTS_VOICE_HI", raising=False)
    assert C.voice_for("hi") == "hi-IN-Chirp3-HD-Kore"
    monkeypatch.setenv("GOOGLE_TTS_VOICE_HI", "hi-IN-Chirp3-HD-Charon")
    assert C.voice_for("hi") == "hi-IN-Chirp3-HD-Charon"


def test_synthesize_ok_and_key_sent_as_param_only():
    seen = {}

    def post(url, params, json, timeout):
        seen.update(url=url, params=params, json=json)
        return R(200, {"audioContent": base64.b64encode(b"RIFFdata").decode()})

    assert C.synthesize("hello", "hi-IN-Chirp3-HD-Kore", key="k123", post=post) == b"RIFFdata"
    assert seen["params"] == {"key": "k123"} and "k123" not in seen["url"]


def test_errors_are_short_and_keyless():
    def post(url, params, json, timeout):
        return R(403, {"error": "API not enabled"})

    with pytest.raises(C.ChirpError) as e:
        C.synthesize("x", "hi-IN-Chirp3-HD-Kore", key="secretkey", post=post)
    assert "403" in str(e.value) and "secretkey" not in str(e.value)
    with pytest.raises(C.ChirpError):
        C.synthesize("x", "v", key="")
