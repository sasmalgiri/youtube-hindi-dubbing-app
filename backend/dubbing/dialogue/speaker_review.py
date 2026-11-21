"""Which lines' speaker is in doubt (tracker P05).

Diarization is never certain, and a wrong speaker is the error a viewer hears
first (a man's line in a woman's voice). The review shows these lines first,
each with a plain reason, so the reviewer checks a handful instead of all.

Reasons (any number per line):
  unknown       no speaker could be found for the words
  borrowed      a short fragment given the speaker of the neighbouring line
  overlap       someone else speaks at the same time
  sandwich      a short line between two lines of one other speaker (A/b/A):
                often the same person, split by a diarization flicker
  very_short    under MIN_SECONDS: too little audio to tell voices apart

Gender/age categories are never a reason and never shown as fact; they are
a guess from the voice ("female-like"), labelled as such in the UI.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

MIN_SECONDS = 0.8
SANDWICH_MAX_SECONDS = 2.0
SANDWICH_MAX_GAP = 1.5

LABELS = {
    "unknown": "no speaker found for these words",
    "borrowed": "short fragment given the next/previous speaker",
    "overlap": "another person speaks at the same time",
    "sandwich": "short line between two lines of another speaker",
    "very_short": "too short to tell voices apart",
}


def speaker_doubts(turns: Sequence[Dict]) -> Dict[str, List[str]]:
    """turns: dicts with turn_id, speaker_id, start, end, flags (review-packet
    shape, in time order). Returns {turn_id: [reason codes]} for doubtful lines."""
    live = [t for t in turns if not t.get("deleted")]
    out: Dict[str, List[str]] = {}
    for i, t in enumerate(live):
        flags = set(t.get("flags") or ())
        r: List[str] = []
        if "speaker_unknown" in flags:
            r.append("unknown")
        if "speaker_from_adjacent_turn" in flags:
            r.append("borrowed")
        if "contains_overlap" in flags:
            r.append("overlap")
        dur = float(t.get("end", 0)) - float(t.get("start", 0))
        if 0 < i < len(live) - 1:
            a, b = live[i - 1], live[i + 1]
            if (a.get("speaker_id") == b.get("speaker_id") != t.get("speaker_id")
                    and dur <= SANDWICH_MAX_SECONDS
                    and float(t["start"]) - float(a["end"]) <= SANDWICH_MAX_GAP
                    and float(b["start"]) - float(t["end"]) <= SANDWICH_MAX_GAP):
                r.append("sandwich")
        if dur < MIN_SECONDS and "nonlexical" not in flags:
            r.append("very_short")
        if r:
            out[t["turn_id"]] = r
    return out


def annotate(packet: Dict) -> Dict:
    """Add speaker_doubt (codes) to each turn and a summary count, in place."""
    doubts = speaker_doubts(packet.get("turns") or [])
    for t in packet.get("turns") or []:
        t["speaker_doubt"] = doubts.get(t["turn_id"], [])
    packet["speaker_doubt_labels"] = LABELS
    packet["speaker_doubt_count"] = len(doubts)
    return packet
