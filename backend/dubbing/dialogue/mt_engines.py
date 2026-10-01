"""Sentence-level machine-translation engines used after (or instead of) the
contextual LLM translators.

They translate each turn on its own, so they cannot use dialogue context
(gender, formality, names across lines). Every turn they translate is
flagged; the Google fallback is also reported as a warning because it is a
degraded path, while IndicTrans2 is a deliberate local choice and is listed
once as a limitation.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence


def _rate_limited(e: BaseException) -> bool:
    """Google's free endpoint answered 429 (deep_translator: TooManyRequests)."""
    if type(e).__name__ == "TooManyRequests":
        return True
    if getattr(getattr(e, "response", None), "status_code", None) == 429:
        return True
    return "too many requests" in str(e).lower()


class GoogleBasicMT:
    name = "google_basic"
    flag = "non_contextual_translation"
    warn_per_turn = True

    def __init__(self, fn: Optional[Callable[[str], str]] = None):
        if fn is None:
            from .translation import google_basic_translate
            fn = google_basic_translate
        self.fn = fn
        self.last_error = ""    # first error of the last batch (read by the translator)

    def translate_batch(self, texts: Sequence[str]) -> List[str]:
        """Line by line. Swallowing every error made an IP block (429) look
        like blank translations: the first error is kept, and raised when
        nothing came back, so the translator records it per turn."""
        out: List[str] = []
        first_error = ""
        for t in texts:
            try:
                out.append(self.fn(t) or "")
            except Exception as e:
                first_error = first_error or f"{type(e).__name__}: {e}"[:200]
                if _rate_limited(e):
                    break   # every further line would 429 too and prolong the block
                out.append("")
        out += [""] * (len(texts) - len(out))
        self.last_error = first_error
        if texts and not any(o.strip() for o in out):
            raise RuntimeError(f"google_basic: {first_error or 'no translation returned'}")
        return out


class IndicTrans2MT:
    """AI4Bharat IndicTrans2 English->Hindi, local and unlimited.

    Runs in a persistent worker process (workers/indictrans2_worker.py),
    optionally under its own Python (INDICTRANS2_PYTHON) because it needs
    transformers 4.x while other local models pin different versions.
    """
    name = "indictrans2"
    flag = "sentence_level_mt"
    warn_per_turn = False

    def __init__(self, model_id: Optional[str] = None):
        from .local_workers import PersistentWorker
        init = {"model": model_id} if model_id else {}
        self.worker = PersistentWorker("indictrans2", init=init, timeout=1800)

    def translate_batch(self, texts: Sequence[str]) -> List[str]:
        return list(self.worker.request({"op": "translate", "texts": list(texts)})["texts"])

    def close(self):
        self.worker.close()


def make_mt_engine(name: str):
    if name == "google_basic":
        return GoogleBasicMT()
    if name == "indictrans2":
        return IndicTrans2MT()
    raise ValueError(f"Unknown MT engine '{name}'")
