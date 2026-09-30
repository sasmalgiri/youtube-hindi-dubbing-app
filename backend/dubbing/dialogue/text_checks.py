"""Text normalisation and critical-token checks (English and Hindi).

Used for two distinct checks that must not be confused:
  * translation faithfulness (English source -> Hindi text): numbers,
    negation and protected names survive translation;
  * TTS fidelity (intended Hindi -> re-ASR Hindi): the audio says the text.
Re-ASR agreement says nothing about whether the Hindi translates the English.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Set

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_PUNCT = re.compile(r"[।॥.,!?;:\"'“”‘’()\[\]{}<>…—–\-/\\|*~`@#$%^&+=_]+")
_ZW = re.compile(r"[​‌‍﻿]")

HINDI_NUMBER_WORDS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "छह": 6, "छः": 6, "छे": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14,
    "पंद्रह": 15, "पन्द्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "पच्चीस": 25, "तीस": 30, "चालीस": 40, "पचास": 50, "साठ": 60, "सत्तर": 70, "अस्सी": 80,
    "नब्बे": 90, "सौ": 100, "हजार": 1000, "लाख": 100000, "करोड": 10000000,
}
EN_NUMBER_WORDS = {
    # "one" is excluded: too often a pronoun ("the one who...").
    "zero": 0, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000, "lakh": 100000, "crore": 10000000,
}
EN_NEGATION = re.compile(
    r"\b(not|no|never|nobody|nothing|none|nowhere|neither|nor|without|cannot|can't|won't|"
    r"don't|doesn't|didn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|shouldn't|"
    r"wouldn't|couldn't|mustn't|ain't)\b|n't\b", re.IGNORECASE)
HI_NEGATION = {"नहीं", "न", "ना", "मत", "बिना", "नहि"}


def normalize_hindi(text: str) -> str:
    """NFC, strip zero-width chars/punctuation/nukta, unify candrabindu, digits."""
    t = unicodedata.normalize("NFC", text or "")
    t = _ZW.sub("", t)
    t = t.replace("़", "")               # nukta: ज़ -> ज
    t = t.replace("ँ", "ं")         # candrabindu -> anusvara
    t = t.translate(DEVANAGARI_DIGITS)
    t = _PUNCT.sub(" ", t)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    return " ".join(t.lower().split())


def hindi_tokens(text: str) -> List[str]:
    return normalize_hindi(text).split()


def _num_values_from_digits(text: str) -> Set[float]:
    vals: Set[float] = set()
    for m in re.finditer(r"\d+(?:[.,]\d+)*", text.translate(DEVANAGARI_DIGITS)):
        raw = m.group(0)
        s = raw.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{2,3})+", raw) else raw.replace(",", ".")
        try:
            vals.add(float(s))
        except ValueError:
            pass
    return vals


def english_numbers(text: str) -> Set[float]:
    vals = _num_values_from_digits(text)
    toks = re.findall(r"[a-z]+", (text or "").lower())
    for tok in toks:
        if tok in EN_NUMBER_WORDS:
            vals.add(float(EN_NUMBER_WORDS[tok]))
    for a, b in re.findall(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]"
                           r"(one|two|three|four|five|six|seven|eight|nine)\b", (text or "").lower()):
        vals.add(float(EN_NUMBER_WORDS[a] + (1 if b == "one" else EN_NUMBER_WORDS[b])))
    return vals


def hindi_numbers(text: str) -> Set[float]:
    vals = _num_values_from_digits(text)
    for tok in hindi_tokens(text):
        if tok in HINDI_NUMBER_WORDS:
            vals.add(float(HINDI_NUMBER_WORDS[tok]))
    return vals


def has_english_negation(text: str) -> bool:
    return bool(EN_NEGATION.search(text or ""))


def hindi_negation_count(text: str) -> int:
    return sum(1 for t in hindi_tokens(text) if t in HI_NEGATION)


def extract_protected_terms(text: str, glossary: Optional[Dict[str, str]] = None) -> List[str]:
    """Names/acronyms (capitalised, not sentence-initial) + glossary hits."""
    terms: List[str] = []
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]*|[.!?]", text or "")
    sentence_start = True
    for tok in toks:
        if tok in ".!?":
            sentence_start = True
            continue
        if (tok[0].isupper() and not sentence_start and tok.lower() not in {"i", "i'm", "i'll", "i've", "i'd"}) \
                or (len(tok) >= 2 and tok.isupper()):
            if tok not in terms:
                terms.append(tok)
        sentence_start = False
    if glossary:
        low = (text or "").lower()
        for k in glossary:
            if re.search(r"\b" + re.escape(k.lower()) + r"\b", low) and k not in terms:
                terms.append(k)
    return terms


def translation_critical_issues(source_en: str, hindi: str,
                                name_map: Optional[Dict[str, str]] = None,
                                glossary: Optional[Dict[str, str]] = None) -> List[Dict]:
    """Critical-token checks English -> Hindi. Returns a list of issues."""
    issues: List[Dict] = []
    en_nums = english_numbers(source_en)
    hi_nums = hindi_numbers(hindi)
    missing = sorted(v for v in en_nums if v not in hi_nums)
    if missing:
        issues.append({"type": "number_missing", "values": missing})
    if has_english_negation(source_en) and hindi_negation_count(hindi) == 0:
        issues.append({"type": "negation_missing"})
    hn = normalize_hindi(hindi)
    for term, target in (glossary or {}).items():
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", (source_en or "").lower()):
            if normalize_hindi(target) not in hn and term.lower() not in hn:
                issues.append({"type": "glossary_term_missing", "term": term, "expected": target})
    for name, target in (name_map or {}).items():
        if re.search(r"\b" + re.escape(name) + r"\b", source_en or ""):
            if normalize_hindi(target) not in hn and name.lower() not in hn:
                issues.append({"type": "name_missing", "term": name, "expected": target})
    return issues


def critical_tokens_hindi(hindi: str) -> Dict[str, object]:
    return {"numbers": hindi_numbers(hindi), "negations": hindi_negation_count(hindi)}


def compare_hindi(expected: str, heard: str, protected: Iterable[str] = ()) -> Dict:
    """TTS fidelity: compare intended Hindi with re-ASR Hindi.

    Reports omissions, insertions, substitutions, repetitions, and critical
    token differences (numbers, negation, protected terms) separately.
    """
    exp = hindi_tokens(expected)
    got = hindi_tokens(heard)
    sm = difflib.SequenceMatcher(a=exp, b=got, autojunk=False)
    omissions, insertions, substitutions = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "delete":
            omissions += exp[i1:i2]
        elif op == "insert":
            insertions += got[j1:j2]
        elif op == "replace":
            e_s, h_s = " ".join(exp[i1:i2]), " ".join(got[j1:j2])
            sim = difflib.SequenceMatcher(a=e_s, b=h_s, autojunk=False).ratio()
            substitutions.append({"expected": e_s, "heard": h_s, "similarity": round(sim, 2),
                                  # near-spellings are usually ASR/orthography noise;
                                  # dissimilar replacements are wrong words
                                  "substantive": sim < 0.5})
    exp_pairs = set(zip(exp, exp[1:]))
    repetitions = [got[i] for i in range(1, len(got))
                   if got[i] == got[i - 1] and (got[i - 1], got[i]) not in exp_pairs]
    edits = len(omissions) + len(insertions) + sum(
        max(len(s["expected"].split()), len(s["heard"].split())) for s in substitutions)
    wer = edits / max(1, len(exp))
    critical: List[Dict] = []
    en, gn = hindi_numbers(expected), hindi_numbers(heard)
    if en != gn:
        critical.append({"type": "numbers_differ", "expected": sorted(en), "heard": sorted(gn)})
    if hindi_negation_count(expected) != hindi_negation_count(heard):
        critical.append({"type": "negation_differs",
                         "expected": hindi_negation_count(expected),
                         "heard": hindi_negation_count(heard)})
    hn = normalize_hindi(heard)
    for term in protected:
        nt = normalize_hindi(term)
        if nt and nt in normalize_hindi(expected) and nt not in hn:
            critical.append({"type": "protected_term_missing", "term": term})
    return {"wer": round(wer, 3), "omissions": omissions, "insertions": insertions,
            "substitutions": substitutions, "repetitions": repetitions,
            "critical": critical, "expected_words": len(exp), "heard_words": len(got)}
