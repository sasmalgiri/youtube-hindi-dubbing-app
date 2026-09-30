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
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
9. Short replies stay short ("Yes." -> "हाँ।"). Keep exclamations/questions as such."""

REWRITE_PROMPT = """Shorten this Hindi dubbing line so it can be spoken faster, keeping the meaning of the English source.
Keep EVERY name, number (as digits), negation and key fact. Remove only filler/redundancy; prefer shorter synonyms.
Return JSON: {"hi":"<shorter Hindi>"}."""


class TranslationEngineError(RuntimeError):
    pass


# ── LLM clients ────────────────────────────────────────────────────────────
class OpenAICompatClient:
    """Minimal chat-completions client for OpenAI-compatible endpoints."""

    ENDPOINTS = {
        "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                   "GEMINI_API_KEY", "DIALOGUE_GEMINI_MODEL", "gemini-2.5-flash"),
        "groq": ("https://api.groq.com/openai/v1/chat/completions",
                 "GROQ_API_KEY", "DIALOGUE_GROQ_MODEL", "llama-3.3-70b-versatile"),
        "cerebras": ("https://api.cerebras.ai/v1/chat/completions",
                     "CEREBRAS_API_KEY", "DIALOGUE_CEREBRAS_MODEL", "llama-3.3-70b"),
        "openai": ("https://api.openai.com/v1/chat/completions",
                   "OPENAI_API_KEY", "DIALOGUE_OPENAI_MODEL", "gpt-4o"),
        "ollama": ("http://localhost:11434/v1/chat/completions",
                   "", "OLLAMA_MODEL", ""),
    }

    def __init__(self, name: str, timeout: float = 120.0):
        if name not in self.ENDPOINTS:
            raise ValueError(f"Unknown engine {name}")
        url, key_env, model_env, default_model = self.ENDPOINTS[name]
        self.name = name
        self.url = url
        self.key = os.environ.get(key_env, "").strip() if key_env else ""
        self.model = os.environ.get(model_env, "").strip() or default_model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        if self.name == "ollama":
            return bool(self.model)
        return bool(self.key)

    @property
    def model_id(self) -> str:
        return f"{self.name}:{self.model}"

    def complete(self, system: str, user: str) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        body = {"model": self.model, "temperature": 0.2,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        if self.name in ("openai", "groq", "gemini"):
            body["response_format"] = {"type": "json_object"}
        last = None
        for attempt in range(3):
            try:
                r = requests.post(self.url, headers=headers, json=body, timeout=self.timeout)
            except Exception as e:  # network error
                last = f"network: {type(e).__name__}"
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise TranslationEngineError(f"{self.name} HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            return data["choices"][0]["message"]["content"] or ""
        raise TranslationEngineError(f"{self.name} failed after retries ({last})")


def default_llm_clients(order: Sequence[str]) -> List[OpenAICompatClient]:
    out = []
    for name in order:
        if name in OpenAICompatClient.ENDPOINTS:
            c = OpenAICompatClient(name)
            if c.available:
                out.append(c)
    return out


def google_basic_translate(text: str) -> str:
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="en", target="hi").translate(text) or ""


# ── validation ─────────────────────────────────────────────────────────────
def parse_json_object(raw: str) -> Dict:
    s = (raw or "").strip()
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
        if not str(item.get("hi", "")).strip():
            errors.append({"type": "empty_translation", "id": tid})
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
                 cancel_check: Optional[Callable[[], bool]] = None):
        self.clients = list(clients)
        self.glossary = glossary or {}
        self.batch_size = batch_size
        self.context_before = context_before
        self.context_after = context_after
        self.max_attempts = max_attempts
        self.allow_basic_fallback = allow_basic_fallback
        self.basic_fallback = basic_fallback or google_basic_translate
        self.on_progress = on_progress or (lambda p, m: None)
        self.cancel_check = cancel_check or (lambda: False)
        self.name_map: Dict[str, str] = {}
        self.warnings: List[Dict] = []
        self.engines_used: set = set()

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
        if note:
            user["reviewer_note"] = note
        raw = client.complete(SYSTEM_PROMPT, json.dumps(user, ensure_ascii=False))
        payload = parse_json_object(raw)
        names = payload.get("names")
        if isinstance(names, dict):
            for k, v in names.items():
                if isinstance(k, str) and isinstance(v, str) and k not in self.name_map:
                    self.name_map[k] = v
        ctx_ids = [t.turn_id for t in before + after]
        return validate_translations(payload, [t.turn_id for t in batch], ctx_ids)

    def translate(self, turns: List[Turn], speaker_hints: Optional[Dict[str, str]] = None) -> List[Dict]:
        """Fill hi_raw/hi_fit/hi_display on required turns. Returns warnings."""
        speaker_hints = speaker_hints or {}
        todo = [t for t in turns if t.required and t.source_text.strip() and not t.hi_raw]
        for t in todo:
            t.protected_terms = extract_protected_terms(t.source_text, self.glossary)
        order = {t.turn_id: i for i, t in enumerate(turns)}
        done = 0
        for b0 in range(0, len(todo), self.batch_size):
            if self.cancel_check():
                raise RuntimeError("Job cancelled by user")
            batch = todo[b0:b0 + self.batch_size]
            first, last = order[batch[0].turn_id], order[batch[-1].turn_id]
            before = [t for t in turns[max(0, first - self.context_before):first] if t.hi_raw]
            after = turns[last + 1:last + 1 + self.context_after]
            self._translate_batch(batch, before, after, speaker_hints)
            done += len(batch)
            self.on_progress(done / max(1, len(todo)), f"Translated {done}/{len(todo)} turns")
        self._critical_pass(todo, turns, speaker_hints)
        for t in todo:
            if t.hi_raw:
                t.hi_fit = t.hi_fit or t.hi_raw
                t.hi_display = t.hi_display or t.hi_raw
        return self.warnings

    def _translate_batch(self, batch, before, after, speaker_hints):
        pending = list(batch)
        for client in self.clients:
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
        if pending and self.allow_basic_fallback:
            for t in pending:
                try:
                    t.hi_raw = (self.basic_fallback(t.source_text) or "").strip()
                except Exception as e:
                    t.translation_attempts.append({"engine": "google_basic", "ok": False,
                                                   "error": str(e)[:200]})
                    continue
                if t.hi_raw:
                    t.add_flag("non_contextual_translation")
                    t.translation_attempts.append({"engine": "google_basic", "ok": True})
                    self.engines_used.add("google_basic")
                    self.warnings.append({"type": "non_contextual_translation", "id": t.turn_id})
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
        for t, issues in failing:
            fixed = False
            for client in self.clients[:1]:
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
            try:
                cand = str(parse_json_object(client.complete(REWRITE_PROMPT, user)).get("hi", "")).strip()
            except Exception:
                continue
            if not cand or len(cand) >= len(current_hi):
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
