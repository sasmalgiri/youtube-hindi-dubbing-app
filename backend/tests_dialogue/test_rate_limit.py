"""Hosted request limits: sign-in codes, quotes and the catch-all are capped
per client; webhooks are never limited; local profile is untouched."""
import ratelimit
from tests_dialogue.test_submission import env  # noqa: F401


def test_window_slides():
    t = [0.0]
    lim = ratelimit.Limiter(rules=(("/api/x", "POST", 2),), clock=lambda: t[0])
    assert lim.check("a", "/api/x", "POST") is None
    assert lim.check("a", "/api/x", "POST") is None
    assert lim.check("a", "/api/x", "POST") >= 1
    assert lim.check("b", "/api/x", "POST") is None          # other clients unaffected
    assert lim.check("a", "/api/x", "GET") is None           # rule is per method
    t[0] = 61
    assert lim.check("a", "/api/x", "POST") is None


def test_first_matching_rule_wins():
    lim = ratelimit.Limiter()
    assert lim.rule("/api/auth/email/start", "POST") == ("/api/auth/email/start", 5)
    assert lim.rule("/api/jobs", "GET") == ("/api/", 600)
    assert lim.rule("/static/x", "GET") is None


def test_cf_header_only_when_trusted(monkeypatch):
    class R:
        headers = {"cf-connecting-ip": "9.9.9.9"}
        class client: host = "1.2.3.4"
    monkeypatch.delenv("VOICEDUB_TRUST_CF", raising=False)
    assert ratelimit.client_of(R) == "1.2.3.4"
    monkeypatch.setenv("VOICEDUB_TRUST_CF", "1")
    assert ratelimit.client_of(R) == "9.9.9.9"


def test_hosted_email_codes_are_limited_per_client(env):
    from fastapi.testclient import TestClient
    c = TestClient(env["app"].app, base_url="https://testserver")
    codes = [c.post("/api/auth/email/start", json={"email": f"u{i}@example.com"}).status_code for i in range(7)]
    assert codes[:5] == [200] * 5 and codes[5] == 429
    r = c.post("/api/auth/email/start", json={"email": "z@example.com"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_webhooks_never_limited(env):
    from fastapi.testclient import TestClient
    c = TestClient(env["app"].app, base_url="https://testserver")
    env["app"]._limiter.rules = (("/api/", None, 1),)
    codes = [c.post("/api/webhooks/razorpay", content=b"{}").status_code for _ in range(3)]
    assert 429 not in codes
