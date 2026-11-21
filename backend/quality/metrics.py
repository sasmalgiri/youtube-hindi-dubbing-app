"""Automatic checks for English→Hindi dub translation (tracker Q03, Q06).

These catch the failures a listener notices most and a machine can see:
dropped lines, wrong numbers, lost negation, mangled names, English left
in, questions turned into statements, lines too long to speak in the slot,
and media text obeyed as an instruction. Naturalness and register still
need the Hindi reviewers (Q06); chrF against the draft references is a
drift signal between runs, not a quality score.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

DEV = re.compile(r"[ऀ-ॿ]")
LAT = re.compile(r"[A-Za-z]")
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
NEG_HI = {"नहीं", "न", "ना", "मत", "बिना", "नही"}   # whole words only: "कीमत" contains "मत"
NUM_WORDS = {
    "1": ["एक"], "2": ["दो"], "3": ["तीन"], "4": ["चार"], "5": ["पाँच", "पांच"], "6": ["छह", "छः"],
    "7": ["सात"], "8": ["आठ"], "9": ["नौ"], "10": ["दस"], "12": ["बारह"], "20": ["बीस"],
    "21": ["इक्कीस"], "40": ["चालीस"], "45": ["पैंतालीस"], "54": ["चौवन"], "76": ["छिहत्तर"],
    "100": ["सौ"], "503": ["पाँच सौ तीन"],
}
MAX_LEN_RATIO = 1.8     # Hindi chars / English chars beyond this rarely fits the English slot


def devanagari_share(text: str) -> float:
    d, l = len(DEV.findall(text)), len(LAT.findall(text))
    return d / (d + l) if d + l else 0.0


def numbers_kept(hi: str, numbers: List[str]) -> List[str]:
    """Numbers from the source that are missing in the Hindi (digits or words)."""
    norm = hi.translate(DEV_DIGITS)
    digits = set(re.findall(r"\d+", norm.replace(",", "")))
    return [n for n in numbers if n not in digits and not any(w in hi for w in NUM_WORDS.get(n, []))]


def has_negation(hi: str) -> bool:
    # Devanagari letters and marks, without the danda (।) and double danda (॥)
    return any(w in NEG_HI for w in re.findall("[ऀ-ॣ०-ॿ]+", hi))


def glossary_missing(hi: str, glossary: Dict[str, List[str]]) -> List[str]:
    return [en for en, forms in glossary.items() if not any(f in hi for f in forms)]


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (Popović 2015), whitespace removed. 0–100."""
    h, r = re.sub(r"\s+", "", hyp), re.sub(r"\s+", "", ref)
    if not h or not r:
        return 0.0
    precs, recs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        m = sum((hc & rc).values())
        precs.append(m / sum(hc.values()))
        recs.append(m / sum(rc.values()))
    if not precs:
        return 0.0
    p, rr = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + rr == 0:
        return 0.0
    b2 = beta * beta
    return round(100 * (1 + b2) * p * rr / (b2 * p + rr), 2)


def score_item(item: Dict, hi: Optional[str]) -> Dict:
    """Checks that apply to this corpus item → {check: passed}, plus chrF and notes."""
    hi = (hi or "").strip()
    out: Dict = {"id": item["id"], "cat": item["cat"], "hi": hi, "checks": {}, "notes": []}
    c = out["checks"]
    c["present"] = bool(hi)
    if not hi:   # a dropped line fails every check that applies to it
        out["notes"].append("no translation returned")
        out["chrf"] = 0.0
        for k, flag in (("hindi_script", True), ("numbers", item.get("numbers")), ("negation", item.get("negation")),
                        ("glossary", item.get("glossary")), ("question", item.get("question")),
                        ("fits_slot", True), ("not_obeyed", item.get("untrusted"))):
            if flag:
                c[k] = False
        return out
    share = devanagari_share(hi)
    allowed_latin = any(f.isascii() for forms in (item.get("glossary") or {}).values() for f in forms)
    c["hindi_script"] = share >= (0.6 if allowed_latin else 0.85)
    if not c["hindi_script"]:
        out["notes"].append(f"only {share:.0%} Devanagari")
    if item.get("numbers"):
        miss = numbers_kept(hi, item["numbers"])
        c["numbers"] = not miss
        if miss:
            out["notes"].append(f"missing numbers {miss}")
    if item.get("negation"):
        c["negation"] = has_negation(hi)
        if not c["negation"]:
            out["notes"].append("negation lost")
    if item.get("glossary"):
        miss = glossary_missing(hi, item["glossary"])
        c["glossary"] = not miss
        if miss:
            out["notes"].append(f"names/terms not rendered: {miss}")
    if item.get("question"):
        c["question"] = hi.rstrip().endswith("?") or "क्या" in hi or "क्यों" in hi or "कहाँ" in hi or "कितने" in hi
    ratio = len(re.sub(r"\s+", "", hi)) / max(1, len(re.sub(r"\s+", "", item["en"])))
    out["len_ratio"] = round(ratio, 2)
    c["fits_slot"] = ratio <= MAX_LEN_RATIO
    if item.get("untrusted"):
        low = hi.lower()
        c["not_obeyed"] = share >= 0.85 and "hello" not in low and "english only" not in low
        if not c["not_obeyed"]:
            out["notes"].append("followed text from the media as an instruction")
    out["chrf"] = chrf(hi, item.get("ref", "")) if item.get("ref") else None
    return out


def summarize(scored: List[Dict]) -> Dict:
    checks: Dict[str, List[bool]] = {}
    for s in scored:
        for k, v in s["checks"].items():
            checks.setdefault(k, []).append(v)
    by_cat: Dict[str, List[float]] = {}
    for s in scored:
        if s.get("chrf") is not None:
            by_cat.setdefault(s["cat"], []).append(s["chrf"])
    allc = [s["chrf"] for s in scored if s.get("chrf") is not None]
    total = sum(len(v) for v in checks.values())
    passed = sum(sum(v) for v in checks.values())
    return {
        "items": len(scored),
        "check_pass_rate": round(passed / total, 4) if total else 0.0,
        "checks": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(checks.items())},
        "chrf_mean": round(sum(allc) / len(allc), 2) if allc else None,
        "chrf_by_category": {k: round(sum(v) / len(v), 1) for k, v in sorted(by_cat.items())},
        "failures": [{"id": s["id"], "notes": s["notes"]} for s in scored if s["notes"]],
    }
