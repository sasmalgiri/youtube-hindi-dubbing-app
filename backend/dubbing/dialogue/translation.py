"""Contextual, structured English -> Hindi dialogue translation.

* Batches of turns are sent as JSON keyed by immutable turn IDs, with
  read-only context before (already translated) and after (source only).
* Responses are validated: missing / duplicate / unknown IDs are detected;
  context-only IDs echoed back are ignored (never emitted twice); missing IDs
  are retried a bounded number of times, alone.
* Critical-token checks (numbers, negation, names, glossary) run per turn; a
  failing turn is retried once with the issue named, then kept with a warning.
* Duration is a soft constraint: `rewrite_shorter` must keep every critical
  token or the rewrite is rejected.
* A per-turn non-contextual fallback (Google via deep-translator) is used
  only if no LLM engine works, and every such turn is flagged.
* An engine that fails a whole retry cycle (5xx, 429 on every key, network)
  is marked down for ENGINE_DOWN_S and skipped by batches, retries, rewrites
  and the brief; both are recorded in the warnings.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .contracts import CATEGORY_UNKNOWN, Turn
from .text_checks import (critical_tokens_hindi, extract_protected_terms,
                          translation_critical_issues)

HINDI_WORDS_PER_SECOND = 2.2   # ~130 wpm conversational Hindi (tuning value)

SYSTEM_PROMPT = """You translate English film/video dialogue into natural spoken Hindi for dubbing.
Rules:
1. Translate ONLY the items in "turns". "context_before" and "context_after" are read-only context: never output them.
2. Return exactly one JSON object: {"translations":[{"id":"<turn id>","hi":"<Hindi>","uncertain":false,"note":""}],"names":{"<English name>":"<Devanagari spelling>"}}. One entry per turn id, same ids, no extra ids.
3. Preserve meaning, negation, relationships, humour, politeness/formality (aap/tum/tu as the relationship implies) and the speaker's intent. Use everyday spoken Hindi (Hindustani), not Sanskritised formal Hindi. Common English loanwords Indians say in speech are fine.
4. Keep every name, quantity, date and key fact. Write numbers, quantities and dates as digits (e.g. 3, 25, 1999).
5. Names: write in Devanagari consistently; reuse the spellings in "known_names". Keep the protected terms listed per turn.
6. Grammatical gender: use gender only when the text/context makes it clear. "voice_hint" is a guess from the audio of the speaker's voice, not verified identity; use it only as a weak hint for the speaker's own first-person forms and never for other people. If unclear, prefer a natural gender-neutral construction and set "uncertain": true.
7. "max_seconds"/"target_words" are soft length targets for lip timing. Prefer concise natural phrasing, but NEVER drop facts, names, numbers or negation to meet them.
8. Do not invent content. If the source is unintelligible or a fragment, translate what is there and set "uncertain": true with a short note.
9. Short replies stay short ("Yes." -> "हाँ।"). Keep exclamations/questions as such.
10. "story_brief", when present, is read-only knowledge of the WHOLE video: use its speaker genders (they come from textual evidence and outrank voice_hint) for each speaker's own gendered forms, its "address" register to keep aap/tum/tu consistent between the same two speakers, and its name spellings."""

BRIEF_PROMPT = """You are preparing a Hindi dub of a video. Read the whole English transcript (one line per turn: "[SPEAKER_ID] text") and return ONE JSON object:
{"speakers":{"<SPEAKER_ID>":{"gender":"male|female|unknown","evidence":"<short quote or reason from the text>"}},
 "address":{"<SPEAKER_A> -> <SPEAKER_B>":"aap|tum|tu"},
 "names":{"<English name>":"<Devanagari spelling>"},
 "summary":"<at most 60 words: setting, relationships, tone>"}
Give a gender only with textual evidence (pronouns used about the speaker, their name, forms of address such as "sir", "ma'am", "Mom", self-reference); otherwise "unknown". Never guess from stereotypes. Choose the address register Hindi speakers would use given the relationship (stranger/elder/boss -> aap, friends/partners -> tum, very close or rude -> tu)."""

REWRITE_PROMPT = """Shorten this Hindi dubbing line so it can be spoken faster, keeping the meaning of the English source.
Keep EVERY name, number (as digits), negation and key fact. Remove only filler/redundancy; prefer shorter synonyms.
Return JSON: {"hi":"<shorter Hindi>"}."""


class TranslationEngineError(RuntimeError):
    pass


class EngineUnavailableError(TranslationEngineError):
    """The engine itself is out (5xx/overloaded, rate-limited on every key,
    unreachable, every key rejected), not just one bad request. The
    translator then skips it for ENGINE_DOWN_S instead of paying the whole
    retry cycle again on every batch."""


class TranslationCancelled(BaseException):
    """The job was cancelled while an engine was backing off. A BaseException
    (like asyncio.CancelledError) so the per-engine `except Exception`
    fallbacks don't take it for an engine failure and call the next engine;
    the orchestrator recognises the "cancelled by user" message."""


# An engine that failed a whole retry cycle is skipped this long: an outage
# then costs one retry cycle, not one per batch, retry, rewrite and brief
# (Gemini 503s stretched 84 lines to 8.5 min), and a long job still gives
# it another chance later.
ENGINE_DOWN_S = 600.0


def _retry_after_s(r) -> float:
    """Seconds the server asked us to wait: the Retry-After header (seconds or
    an HTTP date), else the "retryDelay" Gemini puts in its error body; 0 if none."""
    v = str(r.headers.get("Retry-After") or "").strip()
    if v:
        try:
            return max(0.0, float(v))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                return max(0.0, parsedate_to_datetime(v).timestamp() - time.time())
            except Exception:
                pass
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', r.text or "")
    return float(m.group(1)) if m else 0.0


def _key_rejected(r) -> bool:
    """This key (not the request) was refused: 401/403, or Gemini's 400
    API_KEY_INVALID. Another key of the same provider may still work."""
    if r.status_code in (401, 403):
        return True
    text = r.text or ""
    return r.status_code == 400 and ("API_KEY_INVALID" in text or "API key not valid" in text)


# ── LLM clients ────────────────────────────────────────────────────────────
class OpenAICompatClient:
    """Minimal chat client for OpenAI-compatible chat-completions endpoints,
    and for Ollama's native /api/chat (see OLLAMA_NUM_CTX)."""

    ENDPOINTS = {
        "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                   "GEMINI_API_KEY", "DIALOGUE_GEMINI_MODEL", "gemini-3.5-flash"),  # 2.5 retired Sept 2026
        "groq": ("https://api.groq.com/openai/v1/chat/completions",
                 "GROQ_API_KEY", "DIALOGUE_GROQ_MODEL", "openai/gpt-oss-120b"),  # llama-3.3-70b-versatile retired Sept 2026
        "cerebras": ("https://api.cerebras.ai/v1/chat/completions",
                     "CEREBRAS_API_KEY", "DIALOGUE_CEREBRAS_MODEL", "llama-3.3-70b"),
        "openai": ("https://api.openai.com/v1/chat/completions",
                   "OPENAI_API_KEY", "DIALOGUE_OPENAI_MODEL", "gpt-4o"),
        "ollama": ("http://localhost:11434/api/chat",
                   "", "OLLAMA_MODEL", ""),
    }
    # Ollama's OpenAI-compatible endpoint cannot raise the context window, and
    # model defaults (2048; 1024 in hinglish-translator) are smaller than the
    # ~480-token system prompt plus a 20-turn JSON batch: Ollama then silently
    # drops the start of the prompt and the reply loses ids / breaks the JSON.
    OLLAMA_NUM_CTX = 8192
    ROUNDS = 3                  # retry rounds for 5xx / network / every key rate-limited
    MAX_RETRY_AFTER_S = 60.0    # asked to wait longer = quota spent: give up, don't stall the job

    def __init__(self, name: str, timeout: Optional[float] = None, model: Optional[str] = None):
        if name not in self.ENDPOINTS:
            raise ValueError(f"Unknown engine {name}")
        url, key_env, model_env, default_model = self.ENDPOINTS[name]
        self.name = name
        if name == "ollama":
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
            url = (host if host.startswith("http") else "http://" + host) + "/api/chat"
        self.url = url
        # Every non-empty KEY, KEY_2 .. KEY_19, the .env keys the classic
        # pipeline rotates: one free-tier key's rate limit must not stall the job.
        self.needs_key = bool(key_env)
        self.keys: List[str] = []
        self.key_envs: List[str] = []       # names only, for reports -- never the keys
        for env in ([key_env] + [f"{key_env}_{i}" for i in range(2, 20)]) if key_env else []:
            k = os.environ.get(env, "").strip()
            if k and k not in self.keys:
                self.keys.append(k)
                self.key_envs.append(env)
        self._next_key = 0
        self.model = (model or "").strip() or os.environ.get(model_env, "").strip() or default_model
        # A local model on a 12 GB GPU can need minutes for a 20-turn batch.
        self.timeout = timeout if timeout is not None else (600.0 if name == "ollama" else 120.0)
        # Set by DialogueTranslator: waits / rejected keys become progress lines
        # and warnings, and a cancelled job stops backing off at once.
        self.on_event: Optional[Callable[[Dict[str, Any]], None]] = None
        self.cancel_check: Optional[Callable[[], bool]] = None

    @property
    def available(self) -> bool:
        if self.name == "ollama":
            return bool(self.model)
        return bool(self.keys)

    @property
    def model_id(self) -> str:
        return f"{self.name}:{self.model}"

    def complete(self, system: str, user: str) -> str:
        import requests
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.name == "ollama":
            body = {"model": self.model, "messages": messages, "stream": False, "format": "json",
                    "options": {"temperature": 0.2, "num_ctx": self.OLLAMA_NUM_CTX}}
        else:
            body = {"model": self.model, "temperature": 0.2, "messages": messages}
            if self.name in ("openai", "groq", "gemini"):
                body["response_format"] = {"type": "json_object"}
        last = ""
        for rnd in range(self.ROUNDS):
            if self.needs_key and not self.keys:
                raise EngineUnavailableError(f"{self.name}: every API key was rejected")
            wait = 0.0
            for _ in range(max(1, len(self.keys))):     # each key at most once per round
                idx = self._next_key % len(self.keys) if self.keys else -1
                headers = {"Content-Type": "application/json"}
                if idx >= 0:
                    headers["Authorization"] = f"Bearer {self.keys[idx]}"
                try:
                    r = requests.post(self.url, headers=headers, json=body, timeout=self.timeout)
                except Exception as e:  # network error: not key-specific, back off
                    last = f"network: {type(e).__name__}"
                    if self.name == "ollama" and isinstance(e, requests.exceptions.ReadTimeout):
                        # a local generation this slow is as slow again: don't re-send it
                        raise EngineUnavailableError(
                            f"ollama gave no answer within {self.timeout:.0f}s") from e
                    break
                if r.status_code == 200:
                    if self.keys:   # spread calls over the keys: each has its own quota
                        self._next_key = (idx + 1) % len(self.keys)
                    return self._content(r)
                if r.status_code == 429:
                    # this key's quota: the next key has its own, so retry at once
                    last = "HTTP 429" + (f" on all {len(self.keys)} keys" if len(self.keys) > 1 else "")
                    wait = max(wait, _retry_after_s(r))
                    if self.keys:
                        self._next_key = (idx + 1) % len(self.keys)
                    continue
                if r.status_code >= 500:
                    last = f"HTTP {r.status_code}"
                    wait = max(wait, _retry_after_s(r))
                    break   # server side: another key will not help
                if idx >= 0 and _key_rejected(r):
                    env = self.key_envs.pop(idx)
                    self.keys.pop(idx)
                    self._emit(type="llm_key_rejected", key=env, status=r.status_code)
                    if not self.keys:
                        raise EngineUnavailableError(
                            f"{self.name}: every API key was rejected (last {env}: HTTP {r.status_code})")
                    self._next_key = idx % len(self.keys)   # the slot now holds the next key
                    continue
                raise TranslationEngineError(f"{self.name} HTTP {r.status_code}: {r.text[:200]}")
            if rnd == self.ROUNDS - 1:
                break
            delay = wait or 3.0 * (rnd + 1)
            if delay > self.MAX_RETRY_AFTER_S:
                raise EngineUnavailableError(f"{self.name}: {last}; server asks to wait {delay:.0f}s")
            self._emit(type="llm_wait", seconds=delay, reason=last)
            self._sleep(delay)
        raise EngineUnavailableError(f"{self.name} failed after retries ({last})")

    def _content(self, r) -> str:
        try:
            data = r.json()
            msg = data["message"] if self.name == "ollama" else data["choices"][0]["message"]
            content = msg["content"] or ""
        except Exception as e:
            raise TranslationEngineError(f"{self.name}: unexpected response ({type(e).__name__}): "
                                         f"{(r.text or '')[:200]}") from e
        if self.name == "ollama" and data.get("done_reason") == "length":
            # cut-off JSON would only surface as a confusing parse error
            raise TranslationEngineError(f"ollama reply cut off at the length limit "
                                         f"(num_ctx {self.OLLAMA_NUM_CTX})")
        return content

    def _emit(self, **ev):
        if self.on_event:
            try:
                self.on_event(dict(ev, engine=self.model_id))
            except Exception:
                pass

    def _sleep(self, seconds: float):
        """Back off in <= 1 s slices so a cancelled job stops waiting at once."""
        while seconds > 0:
            if self.cancel_check and self.cancel_check():
                raise TranslationCancelled("Job cancelled by user")
            step = min(1.0, seconds)
            time.sleep(step)
            seconds -= step


def default_llm_clients(order: Sequence[str], ollama_model: str = "") -> List[OpenAICompatClient]:
    out = []
    for name in order:
        if name in OpenAICompatClient.ENDPOINTS:
            c = OpenAICompatClient(name, model=ollama_model if name == "ollama" else None)
            if c.available:
                out.append(c)
    return out


def build_mt_engines(names: Sequence[str]) -> List[Any]:
    """Sentence-level MT engines in priority order (see mt_engines.py)."""
    from .mt_engines import make_mt_engine
    return [make_mt_engine(n) for n in names]


def google_basic_translate(text: str) -> str:
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="en", target="hi").translate(text) or ""


# ── validation ─────────────────────────────────────────────────────────────
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def _not_hindi(text: str) -> bool:
    """Letters but no Devanagari: the model answered in English/Latin (or
    another script), useless to a Hindi voice. Digits/punctuation only
    ("3... 2... 1...") has no wrong-language text and passes."""
    return not _DEVANAGARI.search(text) and any(ch.isalpha() for ch in text)


def parse_json_object(raw: str) -> Dict:
    s = (raw or "").strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()   # reasoning models
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def validate_translations(payload: Dict, expected_ids: Sequence[str],
                          context_ids: Sequence[str] = ()) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Return ({id: item}, errors). Ambiguous IDs are treated as missing."""
    errors: List[Dict] = []
    items = payload.get("translations")
    if not isinstance(items, list):
        return {}, [{"type": "malformed", "detail": "no translations list"}]
    expected = set(expected_ids)
    ctx = set(context_ids)
    seen: Dict[str, List[Dict]] = {}
    for it in items:
        if not isinstance(it, dict):
            errors.append({"type": "malformed_item"})
            continue
        tid = str(it.get("id", "")).strip()
        if tid in ctx and tid not in expected:
            errors.append({"type": "context_echo_ignored", "id": tid})
            continue
        if tid not in expected:
            errors.append({"type": "unknown_id", "id": tid})
            continue
        seen.setdefault(tid, []).append(it)
    good: Dict[str, Dict] = {}
    for tid, lst in seen.items():
        texts = {str(x.get("hi", "")).strip() for x in lst}
        if len(lst) > 1 and len(texts) > 1:
            errors.append({"type": "duplicate_id_conflict", "id": tid})
            continue
        if len(lst) > 1:
            errors.append({"type": "duplicate_id_identical", "id": tid})
        item = lst[0]
        hi = str(item.get("hi", "")).strip()
        if not hi:
            errors.append({"type": "empty_translation", "id": tid})
            continue
        if _not_hindi(hi):      # retried like an empty line
            errors.append({"type": "empty_translation", "id": tid, "detail": "no Devanagari"})
            continue
        good[tid] = item
    for tid in expected_ids:
        if tid not in good and not any(e.get("id") == tid for e in errors):
            errors.append({"type": "missing_id", "id": tid})
    return good, errors


# ── translator ─────────────────────────────────────────────────────────────
class DialogueTranslator:
    def __init__(self, clients: Sequence, glossary: Optional[Dict[str, str]] = None,
                 batch_size: int = 20, context_before: int = 4, context_after: int = 3,
                 max_attempts: int = 3, allow_basic_fallback: bool = True,
                 basic_fallback: Optional[Callable[[str], str]] = None,
                 on_progress: Optional[Callable[[float, str], None]] = None,
                 cancel_check: Optional[Callable[[], bool]] = None,
                 mt_engines: Optional[Sequence[Any]] = None, story_brief: bool = False):
        self.clients = list(clients)
        self.glossary = glossary or {}
        self.batch_size = batch_size
        self.context_before = context_before
        self.context_after = context_after
        self.max_attempts = max_attempts
        self.allow_basic_fallback = allow_basic_fallback
        self.basic_fallback = basic_fallback or google_basic_translate
        # Sentence-level MT fallbacks tried in order after the LLMs. None keeps
        # the original behaviour: Google basic when allow_basic_fallback.
        if mt_engines is None:
            from .mt_engines import GoogleBasicMT
            mt_engines = [GoogleBasicMT(self.basic_fallback)] if allow_basic_fallback else []
        self.mt_engines = list(mt_engines)
        self.on_progress = on_progress or (lambda p, m: None)
        self.cancel_check = cancel_check or (lambda: False)
        self.name_map: Dict[str, str] = {}
        self.story_brief = story_brief
        self.brief: Dict[str, Any] = {}
        self.warnings: List[Dict] = []
        self.engines_used: set = set()
        # Engine health, keyed by id(client): (down until, reason), failed
        # retry cycles, and the one "engine_skipped" warning per engine.
        self._down: Dict[int, Tuple[float, str]] = {}
        self._failures: Dict[int, int] = {}
        self._skipped: Dict[int, Dict] = {}
        self._clock = time.monotonic
        self._frac = 0.0
        self._in_translate = False
        for c in self.clients:
            if hasattr(c, "on_event"):      # OpenAICompatClient
                c.on_event = self._client_event
                c.cancel_check = self.cancel_check

    # ── engine health ─────────────────────────────────────────────────
    def _call(self, client, system: str, user: str) -> str:
        try:
            return client.complete(system, user)
        except EngineUnavailableError as e:
            self._mark_down(client, str(e))
            raise

    def _mark_down(self, client, reason: str):
        k = id(client)
        self._failures[k] = self._failures.get(k, 0) + 1
        self._down[k] = (self._clock() + ENGINE_DOWN_S, reason[:200])
        self.warnings.append({"type": "engine_marked_down", "engine": client.model_id,
                              "detail": reason[:200], "down_for_s": int(ENGINE_DOWN_S),
                              "failures": self._failures[k]})
        self._note(f"{client.model_id} unavailable ({reason[:80]}); "
                   f"skipping it for {int(ENGINE_DOWN_S // 60)} min")

    def _skip_if_down(self, client, where: str) -> bool:
        """True (and counted in one warning per engine) while marked down."""
        down = self._down.get(id(client))
        if not down or down[0] <= self._clock():
            return False
        w = self._skipped.get(id(client))
        if w is None:
            w = self._skipped[id(client)] = {"type": "engine_skipped", "engine": client.model_id,
                                             "reason": "", "skipped_calls": 0, "where": []}
            self.warnings.append(w)
        w["reason"] = down[1]
        w["skipped_calls"] += 1
        if where not in w["where"]:
            w["where"].append(where)
        return True

    def _first_healthy(self, where: str) -> List:
        """[the first engine not marked down], or [] if all are down."""
        for c in self.clients:
            if not self._skip_if_down(c, where):
                return [c]
        return []

    def _progress(self, frac: float, msg: str):
        self._frac = frac
        self.on_progress(frac, msg)

    def _note(self, msg: str):
        # Only while translate() runs: from rewrite_shorter (fit stage) the
        # line would show up under the translate step.
        if self._in_translate:
            self.on_progress(self._frac, msg)

    def _client_event(self, ev: Dict[str, Any]):
        if ev.get("type") == "llm_key_rejected":
            self.warnings.append({k: ev.get(k) for k in ("type", "engine", "key", "status")})
        elif ev.get("type") == "llm_wait":
            self._note(f"{ev.get('engine')}: {ev.get('reason')}; "
                       f"retrying in {float(ev.get('seconds') or 0):.0f}s")

    # prompt payload
    def _turn_payload(self, t: Turn, speaker_hints: Dict[str, str]) -> Dict:
        secs = max(t.source_duration, 0.3)
        d = {"id": t.turn_id, "speaker": t.speaker_id, "text": t.source_text,
             "max_seconds": round(secs, 2),
             "target_words": max(1, int(round(secs * HINDI_WORDS_PER_SECOND)))}
        hint = speaker_hints.get(t.speaker_id, CATEGORY_UNKNOWN)
        if hint != CATEGORY_UNKNOWN:
            d["voice_hint"] = hint
        if t.protected_terms:
            d["protected"] = t.protected_terms
        if t.overlaps_with:
            d["overlaps_with"] = t.overlaps_with
        return d

    def _request(self, client, batch: List[Turn], before: List[Turn], after: List[Turn],
                 speaker_hints: Dict[str, str], note: str = "") -> Tuple[Dict[str, Dict], List[Dict]]:
        user = {
            "known_names": self.name_map,
            "glossary": {k: v for k, v in self.glossary.items()
                         if any(k.lower() in t.source_text.lower() for t in batch)},
            "context_before": [{"id": t.turn_id, "speaker": t.speaker_id, "text": t.source_text,
                                "hi": t.hi_raw} for t in before],
            "turns": [self._turn_payload(t, speaker_hints) for t in batch],
            "context_after": [{"id": t.turn_id, "speaker": t.speaker_id, "text": t.source_text}
                              for t in after],
        }
        if self.brief:
            user = {"story_brief": self.brief, **user}
        if note:
            user["reviewer_note"] = note
        raw = self._call(client, SYSTEM_PROMPT, json.dumps(user, ensure_ascii=False))
        payload = parse_json_object(raw)
        names = payload.get("names")
        if isinstance(names, dict):
            for k, v in names.items():
                # Devanagari only, as for the brief's names: a Latin spelling would
                # be pushed into every later batch as the one to reuse.
                if isinstance(k, str) and isinstance(v, str) and k not in self.name_map \
                        and _DEVANAGARI.search(v):
                    self.name_map[k] = v
        ctx_ids = [t.turn_id for t in before + after]
        return validate_translations(payload, [t.turn_id for t in batch], ctx_ids)

    def translate(self, turns: List[Turn], speaker_hints: Optional[Dict[str, str]] = None) -> List[Dict]:
        """Fill hi_raw/hi_fit/hi_display on required turns. Returns warnings."""
        self._in_translate = True
        try:
            self._translate_turns(turns, speaker_hints or {})
        finally:
            self._in_translate = False
        return self.warnings

    def _translate_turns(self, turns: List[Turn], speaker_hints: Dict[str, str]):
        todo = [t for t in turns if t.required and t.source_text.strip() and not t.hi_raw]
        for t in todo:
            t.protected_terms = extract_protected_terms(t.source_text, self.glossary)
        order = {t.turn_id: i for i, t in enumerate(turns)}
        if self.story_brief and self.clients and len(todo) >= 2 and not self.brief:
            # one whole-transcript LLM call: say so, or the step looks frozen
            self._progress(0.0, "Building story brief...")
            brief = self.build_brief(turns)
            self._progress(0.0, f"Story brief ready ({brief['engine']})" if brief else
                           "No story brief (see translation warnings); using local context only")
        done = 0
        for b0 in range(0, len(todo), self.batch_size):
            if self.cancel_check():
                raise RuntimeError("Job cancelled by user")
            batch = todo[b0:b0 + self.batch_size]
            self._progress(done / max(1, len(todo)),
                           f"Translating turns {done + 1}-{done + len(batch)} of {len(todo)}...")
            first, last = order[batch[0].turn_id], order[batch[-1].turn_id]
            before = [t for t in turns[max(0, first - self.context_before):first] if t.hi_raw]
            after = turns[last + 1:last + 1 + self.context_after]
            self._translate_batch(batch, before, after, speaker_hints)
            done += len(batch)
            self._progress(done / max(1, len(todo)), f"Translated {done}/{len(todo)} turns")
        self._critical_pass(todo, turns, speaker_hints)
        for t in todo:
            if t.hi_raw:
                t.hi_fit = t.hi_fit or t.hi_raw
                t.hi_display = t.hi_display or t.hi_raw

    def _translate_batch(self, batch, before, after, speaker_hints):
        pending = list(batch)
        for client in self.clients:
            if not pending:
                return
            if self._skip_if_down(client, "translation"):
                continue
            for attempt in range(self.max_attempts):
                if not pending:
                    return
                try:
                    good, errors = self._request(client, pending, before, after, speaker_hints)
                except Exception as e:
                    self.warnings.append({"type": "engine_error", "engine": client.model_id,
                                          "detail": str(e)[:200]})
                    break  # try next engine
                for err in errors:
                    if err["type"] != "context_echo_ignored":
                        err = dict(err, engine=client.model_id, attempt=attempt + 1)
                        self.warnings.append(err)
                for t in pending:
                    if t.turn_id in good:
                        it = good[t.turn_id]
                        t.hi_raw = str(it["hi"]).strip()
                        t.translation_attempts.append({"engine": client.model_id, "ok": True,
                                                       "uncertain": bool(it.get("uncertain"))})
                        if it.get("uncertain"):
                            t.add_flag("translation_uncertain")
                            if it.get("note"):
                                t.translation_attempts[-1]["note"] = str(it["note"])[:200]
                        self.engines_used.add(client.model_id)
                pending = [t for t in pending if not t.hi_raw]
        for engine in self.mt_engines:
            if not pending:
                break
            try:
                outs = engine.translate_batch([t.source_text for t in pending])
            except Exception as e:
                for t in pending:
                    t.translation_attempts.append({"engine": engine.name, "ok": False,
                                                   "error": str(e)[:200]})
                self.warnings.append({"type": "engine_error", "engine": engine.name,
                                      "detail": str(e)[:200]})
                continue
            # A partial failure (e.g. Google 429 mid-batch) is recorded per turn
            # with the engine's first error, not left as silent blanks.
            err = getattr(engine, "last_error", "") or ""
            missed = 0
            for t, out in zip(pending, outs):
                out = (out or "").strip()
                if not out:
                    missed += 1
                    t.translation_attempts.append({"engine": engine.name, "ok": False,
                                                   "error": (err or "no output")[:200]})
                    continue
                t.hi_raw = out
                t.add_flag(engine.flag)
                t.translation_attempts.append({"engine": engine.name, "ok": True})
                self.engines_used.add(engine.name)
                if engine.warn_per_turn:
                    self.warnings.append({"type": engine.flag, "id": t.turn_id})
            if missed and err:
                self.warnings.append({"type": "engine_error", "engine": engine.name,
                                      "detail": f"{missed} line(s) not translated: {err}"[:200]})
            pending = [t for t in pending if not t.hi_raw]
        for t in batch:
            if not t.hi_raw:
                t.add_flag("translation_failed")
                self.warnings.append({"type": "translation_failed", "id": t.turn_id})

    def _critical_pass(self, todo: List[Turn], turns: List[Turn], speaker_hints):
        """Retry turns failing critical-token checks once, then keep + warn."""
        failing = []
        for t in todo:
            if not t.hi_raw:
                continue
            issues = translation_critical_issues(t.source_text, t.hi_raw, self.name_map, self.glossary)
            if issues:
                failing.append((t, issues))
        order = {t.turn_id: i for i, t in enumerate(turns)}
        for k, (t, issues) in enumerate(failing, 1):
            if self.cancel_check():
                raise RuntimeError("Job cancelled by user")
            fixed = False
            # the first engine NOT marked down (clients[0] kept retrying a dead Gemini)
            for client in self._first_healthy("critical-token retry"):
                self._progress(1.0, f"Re-translating line {k}/{len(failing)} "
                                    f"(dropped name/number/negation)")
                i = order[t.turn_id]
                note = ("Previous attempt had these problems, fix them: "
                        + json.dumps(issues, ensure_ascii=False) + f". Previous: {t.hi_raw}")
                saved = t.hi_raw
                try:
                    t.hi_raw = ""
                    good, _ = self._request(client, [t], [x for x in turns[max(0, i - 3):i] if x.hi_raw],
                                            turns[i + 1:i + 3], speaker_hints, note=note)
                    cand = str(good.get(t.turn_id, {}).get("hi", "")).strip()
                except Exception:
                    cand = ""
                new_issues = translation_critical_issues(t.source_text, cand, self.name_map,
                                                         self.glossary) if cand else issues
                if cand and len(new_issues) < len(issues):
                    t.hi_raw = cand
                    t.translation_attempts.append({"engine": client.model_id, "ok": True,
                                                   "reason": "critical_token_retry"})
                    issues = new_issues
                    fixed = not new_issues
                else:
                    t.hi_raw = saved
            if not fixed and issues:
                t.add_flag("critical_token_warning")
                self.warnings.append({"type": "critical_tokens", "id": t.turn_id, "issues": issues})

    # ── whole-video brief ─────────────────────────────────────────────
    def build_brief(self, turns: List[Turn], max_chars: int = 40000) -> Dict[str, Any]:
        """One LLM call over the whole transcript (pyVideoTrans passes the full
        text as global context to every batch; a compact brief gives Hindi the
        part that matters -- speaker gender, aap/tum register, names -- at a
        fraction of the tokens). Failure is non-fatal: batches then rely on
        local context as before."""
        lines, size = [], 0
        for t in turns:
            if not t.source_text.strip():
                continue
            line = f"[{t.speaker_id}] {t.source_text.strip()}"
            size += len(line) + 1
            if size > max_chars:
                break
            lines.append(line)
        speakers = sorted({t.speaker_id for t in turns})
        user = json.dumps({"transcript": "\n".join(lines), "speaker_ids": speakers}, ensure_ascii=False)
        for client in self.clients:
            if self._skip_if_down(client, "story brief"):
                continue
            try:
                data = parse_json_object(self._call(client, BRIEF_PROMPT, user))
            except Exception as e:
                self.warnings.append({"type": "story_brief_failed", "engine": client.model_id,
                                      "detail": str(e)[:160]})
                continue
            brief = _clean_brief(data, speakers)
            if not brief:
                continue
            for k, v in brief.get("names", {}).items():
                self.name_map.setdefault(k, v)
            brief["engine"] = client.model_id
            self.brief = {k: v for k, v in brief.items() if k != "engine"}
            return brief
        return {}

    # ── faithful shortening ───────────────────────────────────────────
    def rewrite_shorter(self, t: Turn, current_hi: str, target_ratio: float) -> Optional[str]:
        """Ask for a shorter faithful Hindi line. Returns None if rejected."""
        if not self.clients:
            return None
        before = critical_tokens_hindi(current_hi)
        target_words = max(1, int(len(current_hi.split()) * target_ratio))
        user = json.dumps({"english": t.source_text, "hindi": current_hi,
                           "target_words": target_words, "protected": t.protected_terms,
                           "known_names": self.name_map}, ensure_ascii=False)
        for client in self.clients:
            if self._skip_if_down(client, "line shortening"):
                continue
            try:
                cand = str(parse_json_object(self._call(client, REWRITE_PROMPT, user)).get("hi", "")).strip()
            except Exception:
                continue
            if not cand or len(cand) >= len(current_hi):
                continue
            if _not_hindi(cand):
                t.translation_attempts.append({"engine": client.model_id, "ok": False,
                                               "reason": "rewrite_not_devanagari"})
                continue
            after = critical_tokens_hindi(cand)
            if after["numbers"] != before["numbers"] or after["negations"] < before["negations"]:
                t.translation_attempts.append({"engine": client.model_id, "ok": False,
                                               "reason": "rewrite_dropped_critical_tokens"})
                continue
            if translation_critical_issues(t.source_text, cand, self.name_map, self.glossary) \
                    and not translation_critical_issues(t.source_text, current_hi, self.name_map, self.glossary):
                continue
            t.translation_attempts.append({"engine": client.model_id, "ok": True,
                                           "reason": "duration_rewrite"})
            return cand
        return None


def _clean_brief(data: Any, speaker_ids: Sequence[str]) -> Dict[str, Any]:
    """Keep only well-formed fields of a story brief (it is model output)."""
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Any] = {}
    spk = data.get("speakers")
    if isinstance(spk, dict):
        clean = {}
        for sid, v in spk.items():
            if sid in speaker_ids and isinstance(v, dict) and \
                    v.get("gender") in ("male", "female", "unknown"):
                clean[sid] = {"gender": v["gender"], "evidence": str(v.get("evidence", ""))[:160]}
        if clean:
            out["speakers"] = clean
    addr = data.get("address")
    if isinstance(addr, dict):
        clean = {str(k)[:60]: v for k, v in addr.items() if v in ("aap", "tum", "tu")}
        if clean:
            out["address"] = clean
    names = data.get("names")
    if isinstance(names, dict):
        clean = {k: v for k, v in names.items()
                 if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
                 and re.search(r"[\u0900-\u097F]", v)}
        if clean:
            out["names"] = clean
    if isinstance(data.get("summary"), str) and data["summary"].strip():
        out["summary"] = data["summary"].strip()[:500]
    return out


def load_glossary(path) -> Dict[str, str]:
    try:
        from pathlib import Path
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, str)}
    except Exception:
        pass
    return {}
