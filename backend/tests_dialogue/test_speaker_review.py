"""P05: lines whose speaker is in doubt are marked with plain reasons."""
from dubbing.dialogue.speaker_review import annotate, speaker_doubts


def T(i, spk, s, e, flags=(), deleted=False):
    return {"turn_id": f"t{i}", "speaker_id": spk, "start": s, "end": e, "flags": list(flags), "deleted": deleted}


def test_sandwich_flags_and_short_lines():
    turns = [T(1, "A", 0, 4), T(2, "B", 4.3, 5.5), T(3, "A", 5.8, 9),          # A/b/A flicker
             T(4, "B", 10, 14), T(5, "A", 14.2, 14.6),                          # very short
             T(6, "B", 15, 19, ["speaker_unknown"]), T(7, "A", 20, 24, ["contains_overlap"]),
             T(8, "B", 25, 29, ["speaker_from_adjacent_turn"]), T(9, "C", 30, 33)]
    d = speaker_doubts(turns)
    assert d["t2"] == ["sandwich"]
    assert "very_short" in d["t5"]
    assert d["t6"] == ["unknown"] and d["t7"] == ["overlap"] and d["t8"] == ["borrowed"]
    assert "t1" not in d and "t3" not in d and "t9" not in d


def test_long_middle_line_and_wide_gaps_are_not_sandwiches():
    assert speaker_doubts([T(1, "A", 0, 4), T(2, "B", 4.2, 8), T(3, "A", 8.2, 12)]) == {}      # B speaks 3.8 s
    assert speaker_doubts([T(1, "A", 0, 4), T(2, "B", 9, 10.5), T(3, "A", 15, 19)]) == {}      # far apart


def test_deleted_lines_are_ignored_and_annotate_is_idempotent():
    turns = [T(1, "A", 0, 4), T(2, "B", 4.2, 5, deleted=True), T(3, "A", 5.2, 9)]
    assert speaker_doubts(turns) == {}
    p = {"turns": [T(1, "A", 0, 4), T(2, "B", 4.3, 5.5), T(3, "A", 5.8, 9)]}
    annotate(p); annotate(p)
    assert [t["speaker_doubt"] for t in p["turns"]] == [[], ["sandwich"], []] and p["speaker_doubt_count"] == 1
