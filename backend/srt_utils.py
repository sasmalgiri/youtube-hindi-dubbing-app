import re
from pathlib import Path
from typing import List, Dict


def _parse_time(ts: str) -> float:
    """Parse SRT timestamp '00:01:23,456' to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


def parse_srt(srt_path: Path, text_key: str = "text_translated") -> List[Dict]:
    """Parse an SRT file into segments: [{"start": float, "end": float, text_key: str}, ...]"""
    content = srt_path.read_text(encoding="utf-8")
    segments = []
    # Split on blank lines to get blocks
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        # Line 1: index (skip), Line 2: timestamps, Line 3+: text
        ts_match = re.search(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})", lines[1])
        if not ts_match:
            continue
        start = _parse_time(ts_match.group(1))
        end = _parse_time(ts_match.group(2))
        text = " ".join(lines[2:]).strip()
        if text:
            segments.append({"start": start, "end": end, text_key: text})
    return segments


def _fmt_time(t: float) -> str:
    if t < 0:
        t = 0
    ms = int(round(t * 1000.0))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segments: List[Dict], out_path: Path, text_key: str = "text"):
    lines = []
    for i, seg in enumerate(sorted(segments, key=lambda s: s["start"]), start=1):
        start = _fmt_time(seg["start"])
        end = _fmt_time(seg["end"])
        text = seg.get(text_key, "").strip()
        if text:
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
