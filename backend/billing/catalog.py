"""The approved Indian catalog (tracker M01).

    ₹400 → 14,400 Standard seconds
    ₹500 → 14,400 Premium seconds (single jobs or batches)
    ₹100 →  3,600 Standard seconds
No renewal, no expiry, no checkout after a calendar boundary (balances persist
through inactivity).

One **base hour** = one hour of selected source in one target language, in
Standard single-voice dialogue. Other styles and Highest Quality are quoted
separately: until their multipliers are approved they are **not priced**, and
a quote for them is refused rather than guessed (P06).

Rounding rule (documented once, applied per line): a line's cash is
``ceil_half_up(seconds × rate_minor_per_hour / 3600)`` in minor units. Totals are
sums of already-rounded lines, so lines always reconcile with the total.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

PRICE_VERSION = "inr-2026-11-v1"
CURRENCY = "INR"
MINOR_PER_MAJOR = {"INR": 100, "USD": 100, "EUR": 100, "GBP": 100}
SYMBOL = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}

STANDARD = "standard"
PREMIUM = "premium"
CREDIT_TYPES = (STANDARD, PREMIUM)


@dataclass(frozen=True)
class Pack:
    id: str
    credit_type: str
    seconds: int
    price_minor: int
    currency: str = CURRENCY


PACKS: Dict[str, Pack] = {p.id: p for p in (
    Pack("std_4h", STANDARD, 14_400, 40_000),
    Pack("prem_4h", PREMIUM, 14_400, 50_000),
    Pack("std_1h", STANDARD, 3_600, 10_000),
)}

# Cash rate for time not covered by credits: ₹100 per base hour
# (15 min Standard in one language = ₹25 before credits).
BASE_RATE_MINOR_PER_HOUR = {CURRENCY: 10_000}

# (style, quality) → base-seconds multiplier as an exact fraction (num, den).
# Only the approved combination is priced. Add rows when the founder approves
# them; never infer a price.
STYLES = ("dialogue", "narration", "multi_speaker")
QUALITIES = ("standard", "highest")
MULTIPLIERS: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("dialogue", "standard"): (1, 1),
}

# Price lists per currency (worldwide release). INR is the approved list
# (tracker M01). USD/EUR/GBP are PROPOSED by engineering and stay off until the
# owner approves them by listing the currency in VOICEDUB_CURRENCIES
# (e.g. "INR,USD"). Pack ids and seconds are the same everywhere; only the
# price differs. A currency's base rate prices uncovered seconds in quotes.
PRICE_LISTS: Dict[str, Dict] = {
    "INR": {"version": PRICE_VERSION, "proposed": False, "base_rate": 10_000,
            "packs": {"std_4h": 40_000, "prem_4h": 50_000, "std_1h": 10_000}},
    "USD": {"version": "usd-2026-11-proposed", "proposed": True, "base_rate": 500,
            "packs": {"std_4h": 1_600, "prem_4h": 2_000, "std_1h": 500}},
    "EUR": {"version": "eur-2026-11-proposed", "proposed": True, "base_rate": 450,
            "packs": {"std_4h": 1_500, "prem_4h": 1_900, "std_1h": 450}},
    "GBP": {"version": "gbp-2026-11-proposed", "proposed": True, "base_rate": 400,
            "packs": {"std_4h": 1_300, "prem_4h": 1_600, "std_1h": 400}},
}
EUROZONE = {"AT", "BE", "HR", "CY", "EE", "FI", "FR", "DE", "GR", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PT", "SK", "SI", "ES"}


def enabled_currencies():
    import os
    want = [c.strip().upper() for c in os.environ.get("VOICEDUB_CURRENCIES", "INR").split(",") if c.strip()]
    return [c for c in want if c in PRICE_LISTS] or ["INR"]


def currency_for_country(country) -> str:
    """The currency a buyer in this country pays in (it may not be enabled yet)."""
    cc = (country or "IN").upper()
    if cc == "IN":
        return "INR"
    if cc in EUROZONE:
        return "EUR"
    if cc == "GB":
        return "GBP"
    return "USD"


def sale_currency(country):
    """The enabled currency for this buyer, or None when their country is not on sale yet."""
    c = currency_for_country(country)
    return c if c in enabled_currencies() else None


def pack_price(pack_id: str, currency: str) -> int:
    pl = PRICE_LISTS.get(currency)
    if pl is None or currency not in enabled_currencies():
        raise NotPriced(f"no approved price in {currency}")
    return pl["packs"][pack_id]


def price_version(currency: str) -> str:
    return PRICE_LISTS[currency]["version"]


# Indian display is tax-inclusive. The GST share of a tax-inclusive amount is
# reported for receipts; the treatment itself awaits the accountant (tracker M09/L06).
GST_RATE = (18, 100)


class NotPriced(ValueError):
    """The requested style/quality/currency has no approved price."""


def multiplier(style: str, quality: str) -> Tuple[int, int]:
    if style not in STYLES:
        raise NotPriced(f"unknown style {style!r}")
    if quality not in QUALITIES:
        raise NotPriced(f"unknown quality {quality!r}")
    m = MULTIPLIERS.get((style, quality))
    if m is None:
        raise NotPriced(f"{style} / {quality} is quoted separately and has no approved price yet")
    return m


def billable_seconds(source_seconds: int, style: str, quality: str) -> int:
    """Base seconds for one language: source seconds × the exact multiplier, rounded up."""
    num, den = multiplier(style, quality)
    return -(-source_seconds * num // den)


def cash_minor(seconds: int, currency: str = CURRENCY) -> int:
    """Rounded once per line, half-up, in minor units."""
    rate = BASE_RATE_MINOR_PER_HOUR.get(currency)
    if rate is None and currency in PRICE_LISTS and currency in enabled_currencies():
        rate = PRICE_LISTS[currency]["base_rate"]
    if rate is None:
        raise NotPriced(f"no price in {currency}")
    return (seconds * rate * 2 + 3600) // (2 * 3600)


def tax_inclusive_share(total_minor: int, rate: Tuple[int, int] = GST_RATE) -> int:
    """GST included in a tax-inclusive amount, half-up: total × r / (1 + r)."""
    num, den = rate
    return (total_minor * num * 2 + (den + num)) // (2 * (den + num))


def fmt_money(minor: int, currency: str = CURRENCY) -> str:
    per = MINOR_PER_MAJOR[currency]
    sign = "-" if minor < 0 else ""
    major, rest = divmod(abs(minor), per)
    sym = SYMBOL.get(currency, currency + " ")
    return f"{sign}{sym}{major:,}" + (f".{rest:02d}" if rest else "")


def fmt_seconds(seconds: int) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def pack(pack_id: str) -> Pack:
    p = PACKS.get(pack_id)
    if p is None:
        raise KeyError(f"unknown pack {pack_id}")
    return p


def find_pack(credit_type: str, seconds: Optional[int] = None) -> Optional[Pack]:
    for p in PACKS.values():
        if p.credit_type == credit_type and (seconds is None or p.seconds == seconds):
            return p
    return None
