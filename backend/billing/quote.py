"""Quotes (tracker M02, P02, P06).

A quote is bound to everything that changes its price: account, source version,
selected interval, target language(s), style, quality, price version, currency,
tax treatment, which credits apply, expiry and the maximum cash. Any change
gives a different ``binding`` hash, so a stale quote can never be used for a
changed draft (``check_still_valid``).

Credit application:
  * single job: Standard seconds first, then Premium, then cash;
  * batch child: Premium only; anything uncovered is refused (a batch is
    never funded from Standard or from per-job cash, §4 "Batch").
The quote never spends anything; the ledger reserves at acceptance.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from . import catalog
from .catalog import PREMIUM, STANDARD

QUOTE_TTL_SECONDS = 30 * 60


@dataclass
class QuoteLine:
    language: str
    source_seconds: int            # selected interval, whole seconds
    billable_seconds: int          # after the style/quality multiplier
    standard_seconds: int = 0      # covered by Standard credits
    premium_seconds: int = 0       # covered by Premium credits
    cash_seconds: int = 0          # not covered: paid in cash
    cash_minor: int = 0


@dataclass
class Quote:
    account_id: str
    source_id: str                 # source checksum/version
    interval: List[int]            # [start_s, end_s] actually billed
    languages: List[str]
    style: str
    quality: str
    batch: bool
    currency: str
    price_version: str
    lines: List[QuoteLine]
    created_at: int
    expires_at: int
    standard_seconds: int = 0
    premium_seconds: int = 0
    cash_seconds: int = 0
    cash_minor: int = 0            # job charge (tax-inclusive)
    tax_minor: int = 0             # GST share of cash_minor
    max_cash_minor: int = 0        # never charge more than this for this quote
    notes: List[str] = field(default_factory=list)
    binding: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class QuoteRefused(ValueError):
    pass


def clip_interval(source_duration: float, start: Optional[float], end: Optional[float]) -> List[int]:
    """Whole-second interval inside the real source.

    Start rounds down and end rounds up, so a fractional second the creator
    selected is always included (and billed). A request past the end of a
    shorter source is clipped to the source (P02 "shorter-than-requested").
    """
    if source_duration is None or source_duration <= 0:
        raise QuoteRefused("the source has no measurable duration (probe it first)")
    dur = math.ceil(float(source_duration))
    s = 0 if start is None else int(math.floor(max(0.0, float(start))))
    e = dur if end is None else int(math.ceil(min(float(end), float(source_duration))))
    if s >= dur:
        raise QuoteRefused(f"start {start}s is past the end of the source ({source_duration:.1f}s)")
    if e <= s:
        raise QuoteRefused("the selected interval is empty")
    return [s, e]


def _binding(**kw) -> str:
    raw = json.dumps(kw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def make_quote(*, account_id: str, source_id: str, source_duration: float, languages: Sequence[str],
               style: str = "dialogue", quality: str = "standard", start: Optional[float] = None,
               end: Optional[float] = None, batch: bool = False, available: Optional[Dict[str, int]] = None,
               now: int, currency: str = catalog.CURRENCY) -> Quote:
    """``available``: spendable seconds per credit type from the ledger (may be empty)."""
    langs = [l.strip() for l in languages if l and l.strip()]
    if not langs:
        raise QuoteRefused("choose at least one target language")
    if len(set(langs)) != len(langs):
        raise QuoteRefused("a target language is listed twice")
    interval = clip_interval(source_duration, start, end)
    src_s = interval[1] - interval[0]
    per_lang = catalog.billable_seconds(src_s, style, quality)   # raises NotPriced
    avail = {STANDARD: 0, PREMIUM: 0}
    avail.update({k: max(0, int(v)) for k, v in (available or {}).items() if k in avail})

    lines: List[QuoteLine] = []
    std_left, prem_left = avail[STANDARD], avail[PREMIUM]
    for lang in langs:
        ln = QuoteLine(language=lang, source_seconds=src_s, billable_seconds=per_lang)
        need = per_lang
        if not batch:
            ln.standard_seconds = min(need, std_left); std_left -= ln.standard_seconds; need -= ln.standard_seconds
        ln.premium_seconds = min(need, prem_left); prem_left -= ln.premium_seconds; need -= ln.premium_seconds
        ln.cash_seconds = need
        ln.cash_minor = catalog.cash_minor(need, currency) if need else 0
        lines.append(ln)

    q = Quote(account_id=account_id, source_id=source_id, interval=interval, languages=langs, style=style,
              quality=quality, batch=batch, currency=currency, price_version=catalog.PRICE_VERSION, lines=lines,
              created_at=int(now), expires_at=int(now) + QUOTE_TTL_SECONDS)
    q.standard_seconds = sum(l.standard_seconds for l in lines)
    q.premium_seconds = sum(l.premium_seconds for l in lines)
    q.cash_seconds = sum(l.cash_seconds for l in lines)
    q.cash_minor = sum(l.cash_minor for l in lines)
    q.tax_minor = catalog.tax_inclusive_share(q.cash_minor)
    q.max_cash_minor = q.cash_minor
    if batch and q.cash_seconds:
        raise QuoteRefused(
            f"a batch needs Premium time: {catalog.fmt_seconds(q.cash_seconds)} more Premium seconds are needed "
            f"(Standard time and per-job cash cannot fund a batch)")
    if q.cash_seconds:
        q.notes.append(f"{catalog.fmt_seconds(q.cash_seconds)} not covered by your balance: "
                       f"{catalog.fmt_money(q.cash_minor, currency)} incl. GST")
    q.binding = _binding(account_id=account_id, source_id=source_id, interval=interval, languages=langs,
                         style=style, quality=quality, batch=batch, currency=currency,
                         price_version=catalog.PRICE_VERSION, std=q.standard_seconds, prem=q.premium_seconds,
                         cash=q.cash_minor, expires_at=q.expires_at)
    return q


def check_still_valid(q: Quote, *, now: int, account_id: str, source_id: str, interval: Sequence[int],
                      languages: Sequence[str], style: str, quality: str, batch: bool) -> None:
    """Raise QuoteRefused when the quote expired, the price version moved, or the draft changed."""
    if now >= q.expires_at:
        raise QuoteRefused("the quote expired; get a new quote")
    if q.price_version != catalog.PRICE_VERSION:
        raise QuoteRefused("prices changed since this quote; get a new quote")
    same = (q.account_id == account_id and q.source_id == source_id and list(q.interval) == list(interval)
            and list(q.languages) == list(languages) and q.style == style and q.quality == quality
            and q.batch == batch)
    if not same:
        raise QuoteRefused("the draft changed after this quote; get a new quote")
