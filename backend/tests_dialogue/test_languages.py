"""Worldwide release: language catalog, per-language voices, and one funded job
per target language (a 2-language quote used to charge for 2 and dub 1)."""
import pytest

import languages as L
from dubbing import chirp_tts
from tests_dialogue.test_submission import env, media, grant, upload, rights, wait_closed, balance  # noqa: F401


def test_catalog_entries_are_consistent():
    from dubbing.mt_guard import SCRIPT
    for code, lang in L.LANGUAGES.items():
        assert lang.code == code and lang.name and lang.native
        assert code in SCRIPT, f"translation guard has no script for {code}"
        if lang.chirp_locale:
            assert lang.chirp_voice().endswith("-Chirp3-HD-Kore")
    assert L.get("hi-IN").code == "hi" and L.get("xx") is None
    assert L.sellable("hi")                                   # flagship: baseline + owner decision
    assert not L.sellable("es") and "reviewer" in L.status("es")["why"]


def test_voice_per_language(monkeypatch):
    monkeypatch.delenv("GOOGLE_TTS_VOICE_ES", raising=False)
    assert chirp_tts.voice_for("hi") == "hi-IN-Chirp3-HD-Kore"
    assert chirp_tts.voice_for("es") == "es-US-Chirp3-HD-Kore"
    assert chirp_tts.voice_for("ja-JP") == "ja-JP-Chirp3-HD-Kore"
    monkeypatch.setenv("GOOGLE_TTS_VOICE_ES", "es-ES-Chirp3-HD-Charon")
    assert chirp_tts.voice_for("es") == "es-ES-Chirp3-HD-Charon"
    with pytest.raises(chirp_tts.ChirpError):
        chirp_tts.voice_for("pa")                             # no Chirp voice: Sarvam only


def test_hosted_lists_languages_and_refuses_unsellable_ones(env, media):
    c, uid = env["client"]("a@example.com")
    langs = {x["code"]: x for x in c.get("/api/languages").json()["languages"]}
    assert langs["hi"]["available"] and langs["hi"]["flagship"]
    assert not langs["es"]["available"] and langs["es"]["missing"]
    grant(env, uid, 3_600, "pay:t1")
    did = upload(c, media["ok"]).json()["id"]
    r = c.post("/api/quotes", json={"draft_id": did, "languages": ["es"]})
    assert r.status_code == 422 and "Spanish is not available yet" in r.json()["detail"]
    assert c.post("/api/quotes", json={"draft_id": did, "languages": ["xx"]}).status_code == 422


def test_two_language_quote_makes_two_jobs_each_charged_once(env, media, monkeypatch):
    monkeypatch.setattr(L, "sellable", lambda code: code in ("hi", "bn"))
    monkeypatch.setattr(L, "status", lambda code: {"sellable": code in ("hi", "bn"), "why": ""})
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    did = upload(c, media["ok"]).json()["id"]
    q = c.post("/api/quotes", json={"draft_id": did, "languages": ["hi", "bn"]}).json()
    assert q["standard_seconds"] == 182
    r = c.post(f"/api/drafts/{did}/submit", json={"quote_id": q["id"], "idempotency_key": "k1",
                                                  "rights_accepted": True, "rights_version": rights(c)}).json()
    assert len(r["ids"]) == 2 and r["id"] == r["ids"][0]
    app = env["app"]
    assert [app.JOBS[j].target_language for j in r["ids"]] == ["hi", "bn"]
    for j in r["ids"]:
        assert wait_closed(env, j) == "settled"
        assert app.JOBS[j].billing["seconds"] == 91
    b = balance(env, uid)
    assert (b.consumed, b.reserved, b.spendable) == (182, 0, 3_600 - 182)
    again = c.post(f"/api/drafts/{did}/submit", json={"quote_id": q["id"], "idempotency_key": "k1",
                                                      "rights_accepted": True, "rights_version": rights(c)}).json()
    assert again == {"id": r["id"], "duplicate": True}


def test_failed_second_reservation_releases_the_first(env, media, monkeypatch):
    monkeypatch.setattr(L, "sellable", lambda code: True)
    monkeypatch.setattr(L, "status", lambda code: {"sellable": True, "why": ""})
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 200, "pay:t1")
    did = upload(c, media["ok"]).json()["id"]
    q = c.post("/api/quotes", json={"draft_id": did, "languages": ["hi", "bn"]}).json()
    from billing.ledger import Ledger
    led = Ledger(env["tmp"] / "ledger.db")
    led.reserve(uid, "someone-else", 50)                      # balance drops after the quote
    led.close()
    r = c.post(f"/api/drafts/{did}/submit", json={"quote_id": q["id"], "idempotency_key": "k1",
                                                  "rights_accepted": True, "rights_version": rights(c)})
    assert r.status_code == 409
    b = balance(env, uid)
    assert b.reserved == 50 and b.spendable == 150             # only the unrelated hold remains
