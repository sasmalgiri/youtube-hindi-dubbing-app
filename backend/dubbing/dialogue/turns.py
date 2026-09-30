"""Speaker-aware dialogue turn construction.

Turns are cut from attributed words at: speaker changes (including
known <-> unknown), pauses, sentence ends, and a hard duration cap. Two
different known speakers are never merged into one turn, and short replies
("Yes.", "No!") survive as their own turns.

Text sources:
  * ASR words (word timestamps)                       -> words_from_asr_segments
  * YouTube / English SRT text + ASR words from audio -> align_text_to_words
  * SRT text without any ASR                          -> words_from_cues (estimated timing)
  * Pre-translated Hindi SRT cues                     -> turns_from_translated_cues
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence

from .contracts import UNKNOWN_SPEAKER, Turn, WordRecord
from .diarization import DiarizationResult, derive_exclusive

SENTENCE_END = re.compile(r"[.!?।…][\"'\)\]”’]*$")
CLAUSE_END = re.compile(r"[,;:—-][\"'\)\]]*$")
NONLEXICAL = re.compile(r"^[\[\(♪*].*[\]\)♪*]$|^♪+$")


def _norm_token(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"[^\w']+", "", t)


# ── word sources ───────────────────────────────────────────────────────────
def _even_words(text: str, start: float, end: float, prefix: str, counter: List[int]
                ) -> List[WordRecord]:
    toks = [t for t in text.split() if t.strip()]
    if not toks:
        return []
    span = max(end - start, 0.05 * len(toks))
    # weight by character length so long words get more time
    weights = [max(1, len(t)) for t in toks]
    total = float(sum(weights))
    out, t = [], start
    for tok, wgt in zip(toks, weights):
        d = span * wgt / total
        counter[0] += 1
        out.append(WordRecord(word_id=f"{prefix}{counter[0]:05d}", text=tok,
                              start=round(t, 3), end=round(t + d, 3),
                              timing_estimated=True,
                              nonlexical=bool(NONLEXICAL.match(tok))))
        t += d
    return out


def words_from_asr_segments(segments: Iterable[Dict]) -> List[WordRecord]:
    """ASR segments ({start,end,text,words?}) -> WordRecords (w00001...)."""
    counter = [0]
    out: List[WordRecord] = []
    for seg in segments:
        ws = seg.get("words") or []
        if ws:
            for w in ws:
                txt = (w.get("word") or w.get("text") or "").strip()
                if not txt:
                    continue
                counter[0] += 1
                conf = w.get("probability", w.get("score"))
                out.append(WordRecord(
                    word_id=f"w{counter[0]:05d}", text=txt,
                    start=float(w.get("start", seg.get("start", 0.0))),
                    end=float(w.get("end", seg.get("end", 0.0))),
                    alignment_confidence=float(conf) if conf is not None else None,
                    nonlexical=bool(NONLEXICAL.match(txt))))
        else:
            out += _even_words(seg.get("text", ""), float(seg.get("start", 0.0)),
                               float(seg.get("end", 0.0)), "w", counter)
    out.sort(key=lambda w: (w.start, w.end))
    return out


def words_from_cues(cues: Iterable[Dict], text_key: str = "text") -> List[WordRecord]:
    counter = [0]
    out: List[WordRecord] = []
    for c in cues:
        out += _even_words(c.get(text_key, ""), float(c["start"]), float(c["end"]), "s", counter)
    return out


def align_text_to_words(cues: Sequence[Dict], asr_words: Sequence[WordRecord],
                        text_key: str = "text", max_drift: float = 3.0) -> List[WordRecord]:
    """Keep subtitle *text*, take timing from audio ASR words.

    Subtitle tokens matched to ASR tokens (difflib alignment on normalised
    tokens) take the ASR word's times. Unmatched tokens are interpolated
    between neighbouring anchors inside their cue and flagged timing_estimated.
    No subtitle token is dropped and no ASR-only token is added.
    """
    sub: List[Dict] = []
    for ci, c in enumerate(cues):
        for tok in (c.get(text_key) or "").split():
            sub.append({"text": tok, "cue": ci})
    if not sub:
        return []
    if not asr_words:
        return words_from_cues(cues, text_key)
    a = [_norm_token(s["text"]) for s in sub]
    b = [_norm_token(w.text) for w in asr_words]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    match: Dict[int, int] = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            match[blk.a + k] = blk.b + k
    out: List[WordRecord] = []
    for i, s in enumerate(sub):
        cue = cues[s["cue"]]
        cs, ce = float(cue["start"]), float(cue["end"])
        j = match.get(i)
        if j is not None:
            aw = asr_words[j]
            if aw.start < cs - max_drift or aw.end > ce + max_drift:
                j = None
        rec = WordRecord(word_id=f"s{i + 1:05d}", text=s["text"], start=cs, end=cs,
                         nonlexical=bool(NONLEXICAL.match(s["text"])))
        if j is not None:
            aw = asr_words[j]
            rec.start, rec.end = aw.start, aw.end
            rec.alignment_confidence = aw.alignment_confidence
            rec.attribution = {"text_source": "subtitle", "timing_source": "asr_word"}
        else:
            rec.timing_estimated = True
            rec.attribution = {"text_source": "subtitle", "timing_source": "interpolated"}
        out.append(rec)
    _interpolate_unanchored(out, sub, cues)
    return out


def _interpolate_unanchored(words: List[WordRecord], sub: List[Dict], cues: Sequence[Dict]):
    n = len(words)
    i = 0
    while i < n:
        if not words[i].timing_estimated:
            i += 1
            continue
        j = i
        while j < n and words[j].timing_estimated and sub[j]["cue"] == sub[i]["cue"]:
            j += 1
        cue = cues[sub[i]["cue"]]
        lo = float(cue["start"])
        hi = float(cue["end"])
        if i > 0 and sub[i - 1]["cue"] == sub[i]["cue"] and not words[i - 1].timing_estimated:
            lo = words[i - 1].end
        if j < n and sub[j]["cue"] == sub[i]["cue"] and not words[j].timing_estimated:
            hi = words[j].start
        if hi < lo:
            hi = lo + 0.05 * (j - i)
        step = (hi - lo) / (j - i)
        for k in range(i, j):
            words[k].start = round(lo + step * (k - i), 3)
            words[k].end = round(lo + step * (k - i + 1), 3)
        i = j


# ── turn building ──────────────────────────────────────────────────────────
def _spk(w: WordRecord) -> str:
    return w.speaker_id or UNKNOWN_SPEAKER


def build_turns(words: Sequence[WordRecord], max_pause: float = 0.7,
                sentence_split_min_s: float = 3.0, sentence_pause_s: float = 0.3,
                max_turn_s: float = 15.0) -> List[Turn]:
    """Group attributed words into dialogue turns (see module docstring)."""
    words = sorted(words, key=lambda w: (w.start, w.end))
    groups: List[List[WordRecord]] = []
    # One open turn per speaker. A non-overlapped word from another speaker
    # closes every other open turn (a real speaker change); words inside
    # simultaneous speech (w.overlap) do not, so interleaved overlap words
    # keep both speakers' turns intact instead of shattering them.
    open_turns: Dict[str, List[WordRecord]] = {}
    for w in words:
        spk = _spk(w)
        if not w.overlap:
            for other in [k for k in open_turns if k != spk]:
                groups.append(open_turns.pop(other))
        cur = open_turns.get(spk)
        if cur:
            prev = cur[-1]
            pause = w.start - prev.end
            cur_dur = prev.end - cur[0].start
            if (pause > max_pause
                    or (SENTENCE_END.search(prev.text)
                        and (cur_dur >= sentence_split_min_s or pause >= sentence_pause_s))):
                groups.append(open_turns.pop(spk))
                cur = None
        open_turns.setdefault(spk, []).append(w)
    groups += list(open_turns.values())
    groups.sort(key=lambda g: (g[0].start, g[0].end))

    final: List[List[WordRecord]] = []
    for g in groups:
        final += _split_long(g, max_turn_s)

    turns: List[Turn] = []
    for n, g in enumerate(final, 1):
        spk = _spk(g[0])
        assert all(_spk(w) == spk for w in g), "turn spans multiple speakers"
        for w in g:
            if NONLEXICAL.match(w.text):
                w.nonlexical = True
        lexical = [w for w in g if not w.nonlexical]
        t = Turn(turn_id=f"t{n:04d}", speaker_id=spk,
                 source_start=round(g[0].start, 3),
                 source_end=round(max(w.end for w in g), 3),
                 word_ids=[w.word_id for w in g],
                 source_text=" ".join(w.text for w in g).strip(),
                 required=bool(lexical))
        if not lexical:
            t.add_flag("nonlexical")
        if spk == UNKNOWN_SPEAKER:
            t.add_flag("speaker_unknown")
        if any(w.overlap for w in g):
            t.add_flag("contains_overlap")
        if any(w.timing_estimated for w in g):
            t.add_flag("timing_estimated")
        turns.append(t)
    mark_overlaps(turns)
    return turns


def _split_long(g: List[WordRecord], max_turn_s: float) -> List[List[WordRecord]]:
    if len(g) < 2 or (g[-1].end - g[0].start) <= max_turn_s:
        return [g]
    mid_t = (g[0].start + g[-1].end) / 2
    best_i, best_score = None, None
    for i in range(1, len(g)):
        prev = g[i - 1]
        pause = g[i].start - prev.end
        score = pause * 4.0
        if SENTENCE_END.search(prev.text):
            score += 3.0
        elif CLAUSE_END.search(prev.text):
            score += 1.5
        score -= abs(prev.end - mid_t) / max(max_turn_s, 1.0)
        if best_score is None or score > best_score:
            best_i, best_score = i, score
    return _split_long(g[:best_i], max_turn_s) + _split_long(g[best_i:], max_turn_s)


def mark_overlaps(turns: List[Turn], min_overlap_s: float = 0.1):
    for i, a in enumerate(turns):
        for b in turns[i + 1:]:
            if b.source_start >= a.source_end:
                break
            if a.speaker_id == b.speaker_id:
                continue
            ov = min(a.source_end, b.source_end) - max(a.source_start, b.source_start)
            if ov >= min_overlap_s:
                if b.turn_id not in a.overlaps_with:
                    a.overlaps_with.append(b.turn_id)
                if a.turn_id not in b.overlaps_with:
                    b.overlaps_with.append(a.turn_id)


# ── pre-translated Hindi cues ──────────────────────────────────────────────
def turns_from_translated_cues(cues: Sequence[Dict], diar: Optional[DiarizationResult],
                               text_key: str = "text_translated",
                               second_speaker_share: float = 0.25) -> List[Turn]:
    """One turn per Hindi cue; speaker from audio diarization over the cue span.

    A Hindi cue cannot be split word-by-word against English audio, so cues
    spanning a speaker change are flagged `multi_speaker_cue` instead.
    SRT [SPEAKER_XX] labels are used only when diarization is unavailable.
    """
    excl = []
    if diar is not None:
        excl = diar.exclusive or derive_exclusive(diar.regular)
    turns: List[Turn] = []
    for n, c in enumerate(sorted(cues, key=lambda c: c["start"]), 1):
        text = (c.get(text_key) or c.get("text") or "").strip()
        s, e = float(c["start"]), float(c["end"])
        shares: Dict[str, float] = {}
        for a, b, k in excl:
            ov = min(b, e) - max(a, s)
            if ov > 0:
                shares[k] = shares.get(k, 0.0) + ov
        flags = ["pre_translated"]
        if shares:
            total = sum(shares.values())
            ranked = sorted(shares.items(), key=lambda kv: -kv[1])
            spk = ranked[0][0]
            if len(ranked) > 1 and ranked[1][1] / total >= second_speaker_share:
                flags.append("multi_speaker_cue")
        elif c.get("speaker_id"):
            spk = c["speaker_id"]
            flags.append("speaker_from_srt_label")
        else:
            spk = UNKNOWN_SPEAKER
            flags.append("speaker_unknown")
        t = Turn(turn_id=f"t{n:04d}", speaker_id=spk, source_start=s, source_end=e,
                 source_text=c.get("text_source", ""), hi_raw=text, hi_fit=text,
                 hi_display=text, required=bool(text))
        for f in flags:
            t.add_flag(f)
        turns.append(t)
    mark_overlaps(turns)
    return turns


def cues_from_turn_like(segments: Iterable[Dict]) -> List[Dict]:
    """Normalise arbitrary {start,end,text} dicts (YT subs / SRT) into cues."""
    out = []
    for s in segments:
        txt = (s.get("text") or "").strip()
        if txt:
            out.append({"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)),
                        "text": txt, **({"speaker_id": s["speaker_id"]} if s.get("speaker_id") else {})})
    return out
