"""Classic Gemini key rotation: a key Google refuses (403 'reported as leaked',
revoked, invalid) is dropped for the run; a rate-limited key only cools down."""
import pipeline


class _Refused(Exception):
    code = 403


def _rotator(monkeypatch, n=3):
    for i in range(2, 20):
        monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    for i in range(2, n + 1):
        monkeypatch.setenv(f"GEMINI_API_KEY_{i}", f"k{i}")
    return pipeline._GeminiKeyRotator()


def test_refused_key_is_never_handed_out_again(monkeypatch, capsys):
    r = _rotator(monkeypatch)
    assert r.count() == 3
    r.report_rate_limit("k2", _Refused("403 PERMISSION_DENIED: Your API key was reported as leaked"))
    assert r.count() == 2
    assert "k2" not in {r.get_key() for _ in range(12)}
    out = capsys.readouterr().out
    assert "GEMINI_API_KEY_2 refused" in out and "k2" not in out.replace("GEMINI_API_KEY_2", "")


def test_rate_limit_only_cools_the_key_down(monkeypatch):
    r = _rotator(monkeypatch)
    r.report_rate_limit("k1", Exception("429 RESOURCE_EXHAUSTED: quota"))
    assert r.count() == 3
    assert "k1" not in {r.get_key() for _ in range(6)}     # 30 s cooldown


def test_every_key_refused_gives_no_key(monkeypatch):
    r = _rotator(monkeypatch, n=2)
    r.report_rate_limit("k1", Exception("400 API_KEY_INVALID"))
    r.report_rate_limit("k2", _Refused("denied"))
    assert r.count() == 0
    assert r.get_key() == ""


def test_refused_detection():
    assert pipeline._is_refused_key_error(_Refused("x"))
    assert pipeline._is_refused_key_error(Exception("API key not valid. Please pass a valid API key."))
    assert not pipeline._is_refused_key_error(Exception("503 UNAVAILABLE: model overloaded"))
    assert not pipeline._is_refused_key_error(Exception("429 RESOURCE_EXHAUSTED"))
