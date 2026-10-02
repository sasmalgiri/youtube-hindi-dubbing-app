"""Final status derivation and report rendering (JSON + Markdown)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from .contracts import (STATUS_CANCELLED, STATUS_COMPLETED,
                        STATUS_COMPLETED_WITH_WARNINGS, STATUS_DRAFT_INCOMPLETE,
                        STATUS_FAILED, JobReport, worst_status)


def derive_status(r: JobReport, aborted: str = "") -> Tuple[str, List[str]]:
    """Return (status, reasons). A clean `completed` requires full coverage,
    no identity problems, no unresolved timing overflow and no warnings."""
    if aborted == STATUS_CANCELLED:
        return STATUS_CANCELLED, ["cancelled by user"]
    if aborted == STATUS_FAILED:
        return STATUS_FAILED, list(r.unresolved_failures) or ["failed"]
    status, reasons = STATUS_COMPLETED, []

    def bump(s, why):
        nonlocal status
        status = worst_status(status, s)
        reasons.append(why)

    if r.missing_turns:
        ids = ", ".join(m["turn_id"] for m in r.missing_turns[:12])
        bump(STATUS_DRAFT_INCOMPLETE, f"{len(r.missing_turns)} required turn(s) have no accepted audio: {ids}")
    if r.identity_violations:
        bump(STATUS_DRAFT_INCOMPLETE, f"{len(r.identity_violations)} clip(s) not in their speaker's bound voice")
    mixed = [f for f in r.unresolved_failures if f.startswith("speaker_mixed_providers")]
    if mixed:
        bump(STATUS_DRAFT_INCOMPLETE, "; ".join(mixed))
    drafts = [d for d in r.timing_deviations if d.get("severity") == "draft"]
    if drafts:
        bump(STATUS_DRAFT_INCOMPLETE,
             f"{len(drafts)} turn(s) overflow their slot beyond tolerance: "
             + ", ".join(f"{d['turn_id']} (+{d['overflow_s']}s)" for d in drafts[:8]))
    beyond = [d for d in r.timing_deviations if d.get("beyond_media_end_s")]
    if beyond:
        bump(STATUS_DRAFT_INCOMPLETE, "dub audio extends past the end of the video: "
             + ", ".join(d["turn_id"] for d in beyond))
    warn = [d for d in r.timing_deviations if d.get("severity") == "warning"]
    if warn:
        bump(STATUS_COMPLETED_WITH_WARNINGS, f"{len(warn)} turn(s) slightly overflow their slot")
    if r.content_warnings:
        bump(STATUS_COMPLETED_WITH_WARNINGS, f"{len(r.content_warnings)} content/speech warning(s)")
    tw = [w for w in r.translation_warnings
          if w.get("type") in ("critical_tokens", "non_contextual_translation", "translation_uncertain")]
    if tw:
        bump(STATUS_COMPLETED_WITH_WARNINGS, f"{len(tw)} translation warning(s)")
    if r.unresolved_overlaps:
        bump(STATUS_COMPLETED_WITH_WARNINGS,
             f"{len(r.unresolved_overlaps)} overlapping-speech region(s) may be incompletely transcribed")
    other = [f for f in r.unresolved_failures if not f.startswith("speaker_mixed_providers")]
    for f in other:
        bump(STATUS_COMPLETED_WITH_WARNINGS, f)
    return status, reasons


def write_report(r: JobReport, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "report.json"
    jp.write_text(json.dumps(r.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    mp = out_dir / "report.md"
    mp.write_text(render_markdown(r), encoding="utf-8")
    return jp, mp


def render_markdown(r: JobReport) -> str:
    L = ["# Hindi dialogue dub report", "",
         f"**Status:** `{r.final_status}`", ""]
    if r.status_reasons:
        L += ["**Why:**"] + [f"- {x}" for x in r.status_reasons] + [""]
    L += ["## Input", ""] + [f"- {k}: {v}" for k, v in r.input_identity.items()] + [""]
    if r.resumed_from_checkpoint:
        L += ["**Resumed run:** these stages were taken from this job's own checkpoint (nothing "
              "from other jobs): " + ", ".join(r.resumed_from_checkpoint) + ".", ""]
    L += ["## Stages", "", "| Stage | Status | Seconds | Detail |", "| --- | --- | --- | --- |"]
    for s in r.stages:
        L.append(f"| {s.name} | {s.status} | {s.seconds:.1f} | {s.detail.replace('|', '/')} |")
    L += ["", "## Speakers", "", "| Speaker | Voice category (audio suggestion) | Confidence | Speech s | Voices |",
          "| --- | --- | --- | --- | --- |"]
    for sp in r.speakers:
        voices = "; ".join(f"{p}:{b['voice']}" + (f" ({b['pitch']})" if b.get("pitch") else "")
                           for p, b in sp.get("provider_voices", {}).items())
        conf = sp.get("category_confidence")
        L.append(f"| {sp['speaker_id']} | {sp['voice_category']} | "
                 f"{'-' if conf is None else conf} | {sp.get('total_speech_s', 0):.1f} | {voices} |")
    if r.voice_reuse:
        L += ["", "**Voice reuse (not unique voices):**"]
        L += [f"- {v}: {', '.join(s)}" for v, s in r.voice_reuse.items()]
    L += ["", "## Coverage", "",
          f"- Required turns: {len(r.required_turn_ids)}",
          f"- Turns with accepted audio: {len(r.generated_turn_ids)}",
          f"- Missing: {len(r.missing_turns)}"]
    for m in r.missing_turns[:50]:
        L.append(f"  - {m['turn_id']} [{m['start']:.2f}-{m['end']:.2f}s] {m['speaker_id']}: {m['reason']}")
    if r.duplicate_clips:
        L.append(f"- Duplicate accepted clips (do not count toward coverage): {r.duplicate_clips}")

    def section(title, items, fmt=lambda x: json.dumps(x, ensure_ascii=False)):
        if items:
            L.extend(["", f"## {title}", ""] + [f"- {fmt(x)}" for x in items[:100]])
            if len(items) > 100:
                L.append(f"- ... {len(items) - 100} more in report.json")

    section("Edits applied (review / re-voice)", r.applied_edits)
    section("Identity problems", r.identity_violations)
    section("Timing (non-info)", [d for d in r.timing_deviations if d.get("severity") != "info"])
    section("Translation warnings", r.translation_warnings)
    section("Content / speech warnings", r.content_warnings)
    section("Unresolved overlapping speech", r.unresolved_overlaps)
    L += ["", "## Background separation", "", f"- {json.dumps(r.separation, ensure_ascii=False)}"]
    section("Unresolved failures", r.unresolved_failures, fmt=str)
    section("Limitations of this run", r.limitations, fmt=str)
    L += ["", "## Outputs", ""] + [f"- {k}: {v}" for k, v in r.outputs.items()]
    L += ["", "## Environment", ""] + [f"- {k}: {v}" for k, v in r.environment.items()]
    L += ["", "_Automated checks cannot certify perceived voice quality, Hindi naturalness "
               "or lip timing; listen before publishing._", ""]
    return "\n".join(L)
