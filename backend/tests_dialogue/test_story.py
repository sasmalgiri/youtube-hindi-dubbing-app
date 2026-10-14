"""Story -> video: script timing, LLM breakdown parsing, worker protocol, assembly."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dubbing.story import generate as gen_mod
from dubbing.story.assemble import assemble
from dubbing.story.breakdown import break_down
from dubbing.story.generate import ClipResult, WanGenerator
from dubbing.story.script import (LINE_GAP_S, SHOT_SECONDS, Character, Line, Scene, Story,
                                  plan_shots, timeline, write_srt)


def _story():
    return Story(title="Test", cast=[Character("SPEAKER_00", "Raj", "male", "30, tall, blue shirt"),
                                     Character("SPEAKER_01", "Priya", "female", "28, red saree")],
                 scenes=[Scene("S01", "A small tea shop at dawn", ["SPEAKER_00", "SPEAKER_01"],
                               [Line("SPEAKER_00", "क्या तुम आज आओगी?", "calm"), Line("SPEAKER_01", "हाँ, ज़रूर।", "happy")],
                               action="Raj pours tea; Priya smiles"),
                         Scene("S02", "A busy street", ["SPEAKER_00"],
                               [Line("SPEAKER_00", " ".join(["बहुत"] * 30))])])


# ── script ──
def test_timing_is_derived_from_words_and_scenes_are_whole_shots():
    st = _story()
    plan_shots(st)
    assert st.scenes[0].n_shots == 2 and st.scenes[1].n_shots >= 3          # 5.2 s -> 2 clips; 30 words ~13 s -> 3 clips
    assert len(st.scenes[1].shots) == st.scenes[1].n_shots
    assert "Raj (male, 30, tall, blue shirt)" in st.scenes[0].shots[0]
    cues = timeline(st)
    assert cues[0]["start"] >= 0 and cues[1]["start"] >= cues[0]["end"] + LINE_GAP_S - 1e-6
    assert cues[2]["start"] >= st.scenes[0].n_shots * SHOT_SECONDS          # scene 2 starts after scene 1's clips
    assert abs(st.total_seconds - sum(s.seconds for s in st.scenes)) < 1e-6


def test_srt_has_speaker_labels_and_round_trips(tmp_path):
    st = _story()
    plan_shots(st)
    p = tmp_path / "s.srt"
    write_srt(st, p)
    txt = p.read_text(encoding="utf-8")
    assert "[SPEAKER_00] क्या तुम आज आओगी?" in txt and "[SPEAKER_01] हाँ, ज़रूर।" in txt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from srt_utils import parse_srt
    cues = parse_srt(p, text_key="text_translated")
    assert [c["speaker_id"] for c in cues] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    st.save(tmp_path / "story.json")
    back = Story.load(tmp_path / "story.json")
    assert back.cast[1].name == "Priya" and back.scenes[1].shots == st.scenes[1].shots


def test_validate_catches_bad_input():
    st = _story()
    st.cast[0].gender = "boy"
    st.scenes[0].lines.append(Line("SPEAKER_09", "..."))
    probs = st.validate()
    assert any("gender" in p for p in probs) and any("unknown character" in p for p in probs) \
        and any("nothing to speak" in p for p in probs)


# ── breakdown ──
class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append(user)
        return self.replies.pop(0)


GOOD = json.dumps({"title": "चाय", "cast": [{"id": "SPEAKER_00", "name": "Raj", "gender": "male", "look": "tall"}],
                   "scenes": [{"id": "S01", "setting": "tea shop", "action": "pours tea", "characters": ["SPEAKER_00"],
                               "lines": [{"character": "SPEAKER_00", "text": "नमस्ते।", "emotion": "calm"},
                                         {"character": "", "text": "राज चाय बनाता है।"},
                                         {"character": "SPEAKER_00", "text": "आइए, बैठिए।"}]}]}, ensure_ascii=False)


def test_breakdown_parses_and_plans_shots():
    llm = FakeLLM(["Here you go:\n" + GOOD])
    st = break_down("Raj runs a tea shop. " * 5, llm, title_hint="x")
    assert st.title == "चाय" and st.scenes[0].shots and st.cast[0].gender == "male"


def test_breakdown_retries_once_with_the_complaint_then_fails():
    llm = FakeLLM(["not json at all", json.dumps({"title": "t", "cast": [], "scenes": []})])
    with pytest.raises(RuntimeError, match="no scenes"):
        break_down("Some story text that is long enough to pass the minimum length check.", llm)
    assert "rejected" in llm.calls[1]


# ── generator protocol (fake worker, no model) ──
FAKE_WORKER = '''
import json, sys, os
print(json.dumps({"event": "ready", "model": "fake"})); sys.stdout.flush()
for line in sys.stdin:
    r = json.loads(line)
    if r.get("quit"): break
    if "FAIL" in r["prompt"]:
        print(json.dumps({"event": "error", "id": r["id"], "error": "boom"})); sys.stdout.flush(); continue
    for i in range(r["steps"]):
        print(json.dumps({"event": "step", "id": r["id"], "step": i + 1, "of": r["steps"]})); sys.stdout.flush()
    open(r["out"], "wb").write(b"0" * 2000)
    print(json.dumps({"event": "done", "id": r["id"], "seconds": 0.1, "out": r["out"]})); sys.stdout.flush()
'''


def test_generator_runs_every_shot_reports_progress_and_keeps_failures_visible(tmp_path, monkeypatch):
    w = tmp_path / "fake_worker.py"
    w.write_text(FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(gen_mod, "WORKER", w)
    st = _story()
    st.scenes[1].setting = "FAIL here"
    plan_shots(st)
    g = WanGenerator(tmp_path / "clips", steps=3)
    msgs = []
    g.start(lambda p, m: msgs.append(m))
    res = g.generate_all(st, lambda p, m: msgs.append(m))
    g.close()
    assert len(res) == st.total_shots
    ok = [r for r in res if not r.error]
    assert len(ok) == st.scenes[0].n_shots and all(r.path.exists() for r in ok)
    assert all(r.error == "boom" for r in res if r.scene_id == "S02")
    assert any("step 2/3" in m for m in msgs) and any("FAILED" in m for m in msgs)
    # a second run skips what exists
    g2 = WanGenerator(tmp_path / "clips", steps=3)
    g2.start(lambda p, m: None)
    res2 = g2.generate_all(st, lambda p, m: None)
    g2.close()
    assert res2[0].seconds == 0.0 and not res2[0].error


# ── assembly (real ffmpeg, tiny clips) ──
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg")
def test_assemble_keeps_timing_with_a_card_for_a_missing_clip(tmp_path):
    st = _story()
    st.scenes[1].lines = [Line("SPEAKER_00", "ठीक है।")]        # 1 shot each
    plan_shots(st)
    clips = []
    for s in st.scenes:
        for k in range(s.n_shots):
            out = tmp_path / f"{s.id}_{k}.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=s=832x480:r=16:d={SHOT_SECONDS}",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
            clips.append(ClipResult(s.id, k, out, 1.0))
    clips = [c if (c.scene_id, c.index) != ("S02", 0) else ClipResult("S02", 0, tmp_path / "nope.mp4", 0.0, "boom") for c in clips]
    res = assemble(st, clips, tmp_path / "out")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(res["video"])], capture_output=True, text=True).stdout)
    assert abs(dur - st.total_shots * SHOT_SECONDS) < 0.2
    assert res["missing"] == ["S02 shot 1"]
    assert (tmp_path / "out" / "story_hi.srt").read_text(encoding="utf-8").count("-->") == 3
