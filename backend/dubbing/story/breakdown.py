"""Plain story text -> Story (cast, scenes, Hindi lines) in one LLM pass.

The LLM only structures and translates; it never invents timing (derived in
script.py) or prompts for the video model (plan_shots). The reply is strict
JSON; a malformed reply is retried once with the parser's complaint, then
the job fails with that complaint rather than guessing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, List, Optional

from .script import Character, Line, Scene, Story, plan_shots

SYSTEM = """You turn a short story into a production script for an AI-generated Hindi short drama.
Reply with ONE JSON object and nothing else:
{
 "title": "...",
 "cast": [{"id":"SPEAKER_00","name":"...","gender":"male|female","look":"age, build, hair, clothes (English, 12 words max)"}],
 "scenes": [
   {"id":"S01",
    "setting":"one English sentence of the place, time of day, mood, for a video generator",
    "action":"one English sentence of what the characters visibly do",
    "characters":["SPEAKER_00"],
    "lines":[{"character":"SPEAKER_00","text":"Hindi dialogue in Devanagari","emotion":"angry|sad|happy|calm|afraid|excited|neutral"}]}
 ]
}
Rules:
- Every person who speaks is in "cast" with a stable id SPEAKER_00, SPEAKER_01, ...; a narrator's line has character "" (empty).
- Dialogue must be natural spoken Hindi in Devanagari. Keep names and brand names as they are. Translate English narration into Hindi too.
- Cover the WHOLE story, first sentence to last: every piece of speech in the story becomes a line, and the narration between
  speeches becomes narrator lines (character ""), so nothing the reader learns is lost. One scene per paragraph or change of
  place; 1 to 6 lines per scene; split long speeches into lines of at most 25 words. A 300-word story has about 15-25 lines.
- "setting" and "action" are English, concrete and visual (no dialogue, no names of emotions), because a video model reads them.
- Keep the story's events and order; do not add new plot.
"""


def _llm_json(llm, system: str, user: str) -> dict:
    raw = llm.complete(system, user)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("the model did not return a JSON object")
    return json.loads(m.group(0))


def break_down(story_text: str, llm, title_hint: str = "",
               on_progress: Optional[Callable[[float, str], None]] = None) -> Story:
    """story_text (any language) -> validated Story with shots planned."""
    report = on_progress or (lambda p, m: None)
    text = story_text.strip()
    if len(text) < 40:
        raise ValueError("the story is too short to break down")
    user = (f"Title hint: {title_hint}\n\n" if title_hint else "") + "STORY:\n" + text[:60000]
    report(0.05, "Breaking the story into scenes and Hindi dialogue...")
    last_err = ""
    for attempt in range(2):
        try:
            data = _llm_json(llm, SYSTEM, user if not last_err else
                             user + f"\n\nYour previous reply was rejected: {last_err}. Reply with valid JSON only.")
            story = _to_story(data, text)
            problems = story.validate()
            n_lines = sum(len(sc.lines) for sc in story.scenes)
            n_sent = max(1, len(re.findall(r"[.!?।]+(?:\s|$)", text)))
            if n_lines < max(3, n_sent // 3):
                problems.append(f"only {n_lines} lines for a story of about {n_sent} sentences: the whole story "
                                f"must be covered, speech and narration")
            if problems:
                raise ValueError("; ".join(problems[:5]))
            plan_shots(story)
            report(1.0, f"{len(story.scenes)} scenes, {len(story.cast)} characters, "
                        f"{story.total_shots} clips (~{story.total_seconds / 60:.1f} min)")
            return story
        except (ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
            last_err = str(e)[:300]
            report(0.3, f"Script needs a second pass: {last_err[:80]}")
    raise RuntimeError(f"Could not build a script from the story: {last_err}")


def _to_story(d: dict, source: str) -> Story:
    cast = [Character(id=str(c["id"]).strip(), name=str(c.get("name", "")).strip(),
                      gender=str(c.get("gender", "")).strip().lower(), look=str(c.get("look", "")).strip())
            for c in d.get("cast", [])]
    scenes: List[Scene] = []
    for i, s in enumerate(d.get("scenes", []), 1):
        lines = [Line(character=str(l.get("character", "")).strip(), text=str(l.get("text", "")).strip(),
                      emotion=str(l.get("emotion", "")).strip())
                 for l in s.get("lines", []) if str(l.get("text", "")).strip()]
        scenes.append(Scene(id=str(s.get("id") or f"S{i:02d}"), setting=str(s.get("setting", "")).strip(),
                            characters=[str(c) for c in s.get("characters", [])], lines=lines,
                            action=str(s.get("action", "")).strip()))
    return Story(title=str(d.get("title") or "Untitled").strip(), cast=cast, scenes=scenes, source_text=source)


def read_story_file(path: Path) -> str:
    """.txt / .md / .docx -> text"""
    if path.suffix.lower() == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8-sig", errors="replace")
