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
    assert not L.sellable("es") and "reviewer" in L.status("es")["why"]


def test_every_language_needs_a_voice_bound_signoff(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "SIGNOFFS", tmp_path / "s.json")
    assert not L.sellable("hi") and "sign-off" in L.status("hi")["why"]          # no flagship exception
    (tmp_path / "s.json").write_text('{"hi": {"by": "a,b", "date": "2026-12-01", "voice": "hi-IN-Old"}}')
    assert not L.sellable("hi") and "current voice" in L.status("hi")["why"]
    (tmp_path / "s.json").write_text('{"hi": {"by": "a,b", "date": "2026-12-01", "voice": "%s"}}' % L.get("hi").chirp_voice())
    assert L.sellable("hi")


def test_preview_languages_are_dubbable_not_sellable(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "SIGNOFFS", tmp_path / "none.json")
    monkeypatch.delenv("VOICEDUB_PREVIEW_LANGUAGES", raising=False)
    assert L.refusal("hi", hosted=True)
    monkeypatch.setenv("VOICEDUB_PREVIEW_LANGUAGES", "hi")
    assert L.refusal("hi", hosted=True) is None and not L.sellable("hi")
    d = {x["code"]: x for x in L.describe(True)}
    assert d["hi"]["available"] and not d["hi"]["sellable"] and not d["es"]["available"]


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


def test_hindi_glossary_and_pronunciation_never_reach_other_languages(tmp_path, monkeypatch):
    import json
    import pipeline
    from pipeline import Pipeline, PipelineConfig
    g = tmp_path / "glossary.json"          # the real glossary is gitignored: bring our own
    g.write_text(json.dumps({"_note": "x", "mother": "माँ", "YouTube": "YouTube"}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(pipeline, "GLOSSARY_PATH", g)
    def pipe(code):
        return Pipeline(PipelineConfig(source="x.mp4", work_dir=tmp_path / code, output_path=tmp_path / "o.mp4",
                                       source_language="en", target_language=code))
    hi, es = pipe("hi"), pipe("es")
    assert any(not str(v).isascii() for v in hi._glossary.values())
    assert es._glossary == {"youtube": "YouTube"}
    assert es._apply_pronunciation("Uso ChatGPT") == "Uso ChatGPT"
    assert hi._apply_pronunciation("ChatGPT") != "ChatGPT"
