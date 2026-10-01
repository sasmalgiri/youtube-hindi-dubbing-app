"""Translation robustness: every API key used (429 -> next key, Retry-After),
failing engines marked down and skipped, Ollama's native chat API, Google's
429 no longer swallowed, non-Devanagari replies rejected. Offline: requests.post
and time.sleep are replaced by scripted fakes."""
import json
import time

import pytest
import requests

from dubbing.dialogue import translation
from dubbing.dialogue.contracts import Turn
from dubbing.dialogue.mt_engines import GoogleBasicMT
from dubbing.dialogue.translation import (DialogueTranslator,
                                          EngineUnavailableError,
                                          OpenAICompatClient,
                                          TranslationCancelled,
                                          TranslationEngineError,
                                          validate_translations)


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(payload or {}, ensure_ascii=False)

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


def _ok(content):
    return FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})


@pytest.fixture()
def http(monkeypatch):
    """Scripted requests.post (responses or exceptions, in order) + recorded sleeps."""
    state = {"calls": [], "script": [], "sleeps": []}

    def post(url, headers=None, json=None, timeout=None):
        state["calls"].append({"url": url, "headers": dict(headers or {}), "json": json,
                               "timeout": timeout})
        item = state["script"].pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(translation.time, "sleep", lambda s: state["sleeps"].append(s))
    return state


@pytest.fixture()
def keys(monkeypatch):
    """keys(ENV, v1, v2, ...) sets ENV, ENV_2, ... and clears every other ENV_n."""
    def set_keys(env, *values):
        monkeypatch.delenv(env, raising=False)
        for i in range(2, 20):
            monkeypatch.delenv(f"{env}_{i}", raising=False)
        for i, v in enumerate(values):
            monkeypatch.setenv(env if i == 0 else f"{env}_{i + 1}", v)
    return set_keys


def _turns(n):
    return [Turn(f"t{i:04d}", "AB"[i % 2], i * 2.0, i * 2.0 + 1.5, source_text=f"Line {chr(96 + i)} here.")
            for i in range(1, n + 1)]


# ── API keys / retry cycle ────────────────────────────────────────────────
def test_every_numbered_key_is_loaded(keys):
    keys("GEMINI_API_KEY", "k1", "k2", "", "k4", "k1")      # blank and duplicate skipped
    c = OpenAICompatClient("gemini")
    assert c.keys == ["k1", "k2", "k4"]
    assert c.key_envs == ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_4"]
    assert c.available


def test_429_moves_to_the_next_key_at_once_and_calls_spread_over_keys(http, keys):
    keys("GROQ_API_KEY", "k1", "k2")
    http["script"] = [FakeResponse(429), _ok('{"hi": "हाँ"}'), _ok("{}")]
    c = OpenAICompatClient("groq")
    assert c.complete("s", "u") == '{"hi": "हाँ"}'
    c.complete("s", "u")
    assert [x["headers"]["Authorization"] for x in http["calls"]] == \
        ["Bearer k1", "Bearer k2", "Bearer k1"]
    assert http["sleeps"] == []


def test_backs_off_only_after_every_key_is_limited_honouring_retry_after(http, keys):
    keys("GEMINI_API_KEY", "k1", "k2")
    http["script"] = [FakeResponse(429, headers={"Retry-After": "7"}),
                      FakeResponse(429, text='[{"error": {"details": [{"retryDelay": "4s"}]}}]'),
                      _ok("{}")]
    events = []
    c = OpenAICompatClient("gemini")
    c.on_event = events.append
    assert c.complete("s", "u") == "{}"
    assert sum(http["sleeps"]) == 7          # one wait: the longest the server asked for
    assert [e["type"] for e in events] == ["llm_wait"]
    assert "429 on all 2 keys" in events[0]["reason"]


def test_retry_after_beyond_the_cap_gives_up_instead_of_stalling(http, keys):
    keys("GROQ_API_KEY", "k1")
    http["script"] = [FakeResponse(429, headers={"Retry-After": "3600"})]
    with pytest.raises(EngineUnavailableError, match="3600"):
        OpenAICompatClient("groq").complete("s", "u")
    assert http["sleeps"] == []


def test_5xx_retry_cycle_ends_in_engine_unavailable(http, keys):
    keys("GEMINI_API_KEY", "k1", "k2")
    http["script"] = [FakeResponse(503)] * 3
    with pytest.raises(EngineUnavailableError, match="HTTP 503"):
        OpenAICompatClient("gemini").complete("s", "u")
    assert len(http["calls"]) == 3           # server side: no key hopping
    assert sum(http["sleeps"]) == 3 + 6


def test_bad_request_is_not_an_outage(http, keys):
    keys("GROQ_API_KEY", "k1", "k2")
    http["script"] = [FakeResponse(400, text="context too long")]
    with pytest.raises(TranslationEngineError) as e:
        OpenAICompatClient("groq").complete("s", "u")
    assert not isinstance(e.value, EngineUnavailableError) and len(http["calls"]) == 1


def test_rejected_key_is_dropped_and_reported_by_name_only(http, keys):
    keys("GROQ_API_KEY", "secret-1", "secret-2")
    http["script"] = [FakeResponse(401, text='{"error": "invalid_api_key"}'), _ok("{}"), _ok("{}"),
                      FakeResponse(403, text="forbidden")]
    events = []
    c = OpenAICompatClient("groq")
    c.on_event = events.append
    c.complete("s", "u")
    c.complete("s", "u")
    assert [x["headers"]["Authorization"] for x in http["calls"]] == \
        ["Bearer secret-1", "Bearer secret-2", "Bearer secret-2"]
    assert events == [{"type": "llm_key_rejected", "key": "GROQ_API_KEY", "status": 401,
                       "engine": c.model_id}]
    with pytest.raises(EngineUnavailableError, match="every API key was rejected"):
        c.complete("s", "u")
    assert "secret" not in json.dumps(events)


def test_cancel_stops_the_backoff_at_once(http, keys):
    keys("GROQ_API_KEY", "k1")
    http["script"] = [FakeResponse(503)]
    c = OpenAICompatClient("groq")
    c.cancel_check = lambda: True
    with pytest.raises(TranslationCancelled, match="cancelled by user"):
        c.complete("s", "u")
    assert http["sleeps"] == []


# ── Ollama native chat ────────────────────────────────────────────────────
def test_ollama_uses_native_chat_with_large_context_and_json(http, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    c = OpenAICompatClient("ollama", model="hinglish-translator:latest")
    http["script"] = [FakeResponse(200, {"model": c.model, "done": True, "done_reason": "stop",
                                         "message": {"role": "assistant", "content": '{"hi": "हाँ"}'}})]
    assert c.complete("sys", "usr") == '{"hi": "हाँ"}'
    call = http["calls"][0]
    assert call["url"] == "http://localhost:11434/api/chat"
    body = call["json"]
    assert body["model"] == "hinglish-translator:latest"
    assert body["stream"] is False and body["format"] == "json"
    assert body["options"]["num_ctx"] == 8192
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert "response_format" not in body and "Authorization" not in call["headers"]
    assert call["timeout"] >= 600

    http["script"] = [FakeResponse(200, {"message": {"content": '{"translations": ['},
                                         "done": True, "done_reason": "length"})]
    with pytest.raises(TranslationEngineError, match="cut off"):
        c.complete("sys", "usr")

    n = len(http["calls"])
    http["script"] = [requests.exceptions.ReadTimeout()]
    with pytest.raises(EngineUnavailableError):      # not re-sent: it would be as slow again
        c.complete("sys", "usr")
    assert len(http["calls"]) == n + 1


# ── translator: engine health ─────────────────────────────────────────────
class DownLLM:
    """An engine whose whole retry cycle fails (Gemini answering 503)."""
    model_id = "fake:down"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        raise EngineUnavailableError("gemini failed after retries (HTTP 503)")


class GoodLLM:
    model_id = "fake:good"

    def __init__(self):
        self.requests = []

    def complete(self, system, user):
        req = json.loads(user)
        self.requests.append(req)
        if "transcript" in req:
            return json.dumps({"summary": "two friends talk"})
        if "english" in req:
            return json.dumps({"hi": "छोटी पंक्ति"}, ensure_ascii=False)
        out = []
        for t in req["turns"]:
            hi = "ठीक है।"
            if "3" in t["text"]:      # drops the number until the reviewer note names it
                hi = "मुझे 3 टिकट चाहिए।" if "reviewer_note" in req else "मुझे टिकट चाहिए।"
            out.append({"id": t["id"], "hi": hi})
        return json.dumps({"translations": out}, ensure_ascii=False)


def test_engine_down_is_skipped_everywhere_reported_and_retried_later():
    down, good = DownLLM(), GoodLLM()
    progress = []
    tr = DialogueTranslator([down, good], batch_size=2, story_brief=True, allow_basic_fallback=False,
                            on_progress=lambda p, m: progress.append(m))
    turns = _turns(5)
    turns[3].source_text = "I need 3 tickets."
    tr.translate(turns)

    assert down.calls == 1                       # failed once (brief); never retried after that
    assert all(t.hi_raw for t in turns) and "3" in turns[3].hi_raw
    assert any("reviewer_note" in r for r in good.requests)   # critical pass: first HEALTHY engine
    marked = [w for w in tr.warnings if w["type"] == "engine_marked_down"]
    assert len(marked) == 1 and marked[0]["engine"] == "fake:down" and "503" in marked[0]["detail"]
    skipped = [w for w in tr.warnings if w["type"] == "engine_skipped"]
    assert len(skipped) == 1 and skipped[0]["engine"] == "fake:down"
    assert skipped[0]["skipped_calls"] == 4      # 3 batches + 1 critical-token retry
    assert skipped[0]["where"] == ["translation", "critical-token retry"]

    b = progress.index("Building story brief...")
    assert any(m.startswith("fake:down unavailable") for m in progress[b:])
    assert progress.index("Story brief ready (fake:good)") > b
    assert progress.index("Translating turns 1-2 of 5...") > progress.index("Story brief ready (fake:good)")

    long_hi = "यह बहुत लंबी पंक्ति है जिसे छोटा करना है"      # no number word ("एक" counts)
    assert tr.rewrite_shorter(turns[0], long_hi, 0.5) == "छोटी पंक्ति"
    assert down.calls == 1 and "line shortening" in skipped[0]["where"]

    tr._clock = lambda: time.monotonic() + translation.ENGINE_DOWN_S + 1   # down period over
    tr.rewrite_shorter(turns[0], long_hi, 0.5)
    assert down.calls == 2
    assert [w for w in tr.warnings if w["type"] == "engine_marked_down"][-1]["failures"] == 2


def test_translator_reports_client_waits_and_rejected_keys(http, keys):
    keys("GROQ_API_KEY", "k1", "k2")
    reply = json.dumps({"translations": [{"id": "t0001", "hi": "हाँ।"}]}, ensure_ascii=False)
    http["script"] = [FakeResponse(401), FakeResponse(503), _ok(reply)]
    client = OpenAICompatClient("groq")
    progress = []
    tr = DialogueTranslator([client], allow_basic_fallback=False,
                            on_progress=lambda p, m: progress.append(m))
    turns = _turns(1)
    tr.translate(turns)
    assert turns[0].hi_raw == "हाँ।"
    assert {"type": "llm_key_rejected", "engine": client.model_id, "key": "GROQ_API_KEY",
            "status": 401} in tr.warnings
    assert f"{client.model_id}: HTTP 503; retrying in 3s" in progress


def test_cancel_during_backoff_cancels_the_job_not_just_the_engine(http, keys):
    keys("GROQ_API_KEY", "k1")
    http["script"] = [FakeResponse(503)]
    checks = {"n": 0}

    def cancel_check():
        checks["n"] += 1
        return checks["n"] > 1        # cancelled after translation started

    fallback = []
    tr = DialogueTranslator([OpenAICompatClient("groq")], cancel_check=cancel_check,
                            basic_fallback=lambda s: fallback.append(s) or "अनुवाद")
    with pytest.raises(TranslationCancelled):
        tr.translate(_turns(1))
    assert fallback == []             # no fallback engine run for a cancelled job


# ── replies that are not Hindi ────────────────────────────────────────────
def test_reply_without_devanagari_counts_as_empty():
    payload = {"translations": [{"id": "a", "hi": "Main theek hoon."},
                                {"id": "b", "hi": "3... 2... 1..."},
                                {"id": "c", "hi": "हाँ।"}]}
    good, errors = validate_translations(payload, ["a", "b", "c"])
    assert set(good) == {"b", "c"}
    assert {"type": "empty_translation", "id": "a", "detail": "no Devanagari"} in errors


def test_english_reply_is_retried_then_falls_back_flagged():
    class EnglishLLM:
        model_id = "fake:english"
        calls = 0

        def complete(self, system, user):
            EnglishLLM.calls += 1
            req = json.loads(user)
            return json.dumps({"translations": [{"id": t["id"], "hi": "Okay."} for t in req["turns"]]})

    turns = _turns(1)
    DialogueTranslator([EnglishLLM()], basic_fallback=lambda s: "ठीक है").translate(turns)
    assert EnglishLLM.calls == 3
    assert turns[0].hi_raw == "ठीक है" and "non_contextual_translation" in turns[0].flags


def test_latin_name_spellings_and_latin_rewrites_are_not_used():
    class LatinBits:
        model_id = "fake:latin"

        def complete(self, system, user):
            req = json.loads(user)
            if "english" in req:                       # rewrite request
                return json.dumps({"hi": "Chhoti line"})
            return json.dumps({"translations": [{"id": t["id"], "hi": "ठीक है।"} for t in req["turns"]],
                               "names": {"Bob": "Bob", "Riya": "रिया"}}, ensure_ascii=False)

    tr = DialogueTranslator([LatinBits()], allow_basic_fallback=False)
    turns = _turns(1)
    tr.translate(turns)
    assert tr.name_map == {"Riya": "रिया"}
    assert tr.rewrite_shorter(turns[0], "यह बहुत लंबी पंक्ति है जिसे छोटा करना है", 0.5) is None
    assert turns[0].translation_attempts[-1]["reason"] == "rewrite_not_devanagari"


# ── Google basic fallback ─────────────────────────────────────────────────
class TooManyRequests(Exception):
    """Stands in for deep_translator.exceptions.TooManyRequests."""


def test_google_basic_stops_at_429_and_raises_when_nothing_came_back():
    calls = []

    def fn(text):
        calls.append(text)
        raise TooManyRequests("Server Error: You made too many requests to the server.")

    with pytest.raises(RuntimeError, match="google_basic: TooManyRequests"):
        GoogleBasicMT(fn).translate_batch(["a", "b", "c"])
    assert calls == ["a"]                 # stopped calling after the 429


def test_google_basic_partial_429_pads_and_keeps_the_first_error():
    def fn(text):
        if text == "b":
            raise TooManyRequests("Server Error: You made too many requests to the server.")
        return "हिंदी " + text

    mt = GoogleBasicMT(fn)
    assert mt.translate_batch(["a", "b", "c"]) == ["हिंदी a", "", ""]
    assert mt.last_error.startswith("TooManyRequests")


def test_google_block_is_recorded_per_turn_through_the_translator():
    def fn(text):
        if "Line a" not in text:
            raise TooManyRequests("Server Error: You made too many requests to the server.")
        return "अनुवाद"

    turns = _turns(3)
    tr = DialogueTranslator([], basic_fallback=fn)
    tr.translate(turns)
    assert turns[0].hi_raw == "अनुवाद"
    for t in turns[1:]:
        assert "translation_failed" in t.flags
        assert t.translation_attempts[-1]["ok"] is False
        assert "TooManyRequests" in t.translation_attempts[-1]["error"]
    errs = [w for w in tr.warnings if w["type"] == "engine_error" and w["engine"] == "google_basic"]
    assert errs and errs[0]["detail"].startswith("2 line(s) not translated: TooManyRequests")

    blocked = _turns(2)              # nothing comes back: raised, recorded per turn
    tr = DialogueTranslator([], basic_fallback=lambda s: fn("x"))
    tr.translate(blocked)
    for t in blocked:
        assert len(t.translation_attempts) == 1 and t.translation_attempts[0]["ok"] is False
        assert t.translation_attempts[0]["error"].startswith("google_basic: TooManyRequests")
    assert any(w["type"] == "engine_error" and w["detail"].startswith("google_basic: TooManyRequests")
               for w in tr.warnings)
