"""The deliverable SRT follows the dubbed timeline when assembly re-timed the
video (audio is master), and keeps the source timings otherwise."""
from pipeline import Pipeline

SRC = "1\n00:00:00,400 --> 00:00:01,900\nsource-timed\n"


def _pipe(placed, segs):
    p = Pipeline.__new__(Pipeline)          # no __init__: only the SRT helper is used
    if placed is not None:
        p._placed_speech = placed
    p._split_tts_segments = segs
    return p


SEGS = [
    {"start": 0.4, "end": 1.9, "text_translated": "अरे, माफ़ करना।"},
    {"start": 1.9, "end": 4.4, "text_translated": "क्या ये चेक-इन की कतार है?"},
    {"start": 4.4, "end": 13.7, "text_translated": "हां, यही सही कतार है।"},
]


def test_cues_sit_where_the_audio_was_placed(tmp_path):
    src = tmp_path / "src.srt"
    src.write_text(SRC, encoding="utf-8")
    out = tmp_path / "out.srt"
    # reflowed/stretched output positions differ from the source spans
    placed = [(0.4, 1.9, 0.0, 1.5), (1.9, 4.4, 2.7, 2.0), (4.4, 13.7, 6.1, 2.2)]
    _pipe(placed, SEGS)._write_output_srt(src, out)
    txt = out.read_text(encoding="utf-8")
    assert "00:00:02,700 --> 00:00:04,700\nक्या ये चेक-इन की कतार है?" in txt
    assert "00:00:06,100 --> 00:00:08,300\nहां, यही सही कतार है।" in txt
    assert "source-timed" not in txt


def test_without_placements_the_source_timed_srt_is_kept(tmp_path):
    src = tmp_path / "src.srt"
    src.write_text(SRC, encoding="utf-8")
    out = tmp_path / "out.srt"
    _pipe(None, SEGS)._write_output_srt(src, out)          # e.g. Tempo Match assembly
    assert out.read_text(encoding="utf-8") == SRC


def test_unmatched_lines_fall_back_to_source_timings_visibly(tmp_path, capsys):
    src = tmp_path / "src.srt"
    src.write_text(SRC, encoding="utf-8")
    out = tmp_path / "out.srt"
    placed = [(0.4, 1.9, 0.0, 1.5), (9.0, 9.5, 2.7, 2.0), (20.0, 21.0, 6.1, 2.2)]
    _pipe(placed, SEGS)._write_output_srt(src, out)
    assert out.read_text(encoding="utf-8") == SRC
    assert "keep the source timings" in capsys.readouterr().out
