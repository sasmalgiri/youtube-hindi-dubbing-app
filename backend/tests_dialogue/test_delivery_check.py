"""Q05 delivery check on real ffmpeg-made files (no network, no models)."""
import shutil
import subprocess
from pathlib import Path

import pytest

import delivery_check as D

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("dc")
    ff("-f", "lavfi", "-i", "sine=frequency=300:duration=30", "-af", "volume=0.3", "-c:a", "aac", str(d / "ok.m4a"))
    # loud square-ish wave driven past full scale -> clipping
    ff("-f", "lavfi", "-i", "sine=frequency=200:duration=10", "-af", "volume=20", "-c:a", "pcm_s16le", str(d / "clip.wav"))
    # 5 s tone, 12 s silence, 5 s tone
    ff("-f", "lavfi", "-i", "sine=frequency=300:duration=5", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=12",
       "-f", "lavfi", "-i", "sine=frequency=300:duration=5",
       "-filter_complex", "[0:a]aresample=44100,aformat=channel_layouts=mono[a];[2:a]aresample=44100,aformat=channel_layouts=mono[c];[a][1:a][c]concat=n=3:v=0:a=1",
       "-c:a", "pcm_s16le", str(d / "gap.wav"))
    ff("-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=5", "-c:a", "pcm_s16le", str(d / "silent.wav"))
    return d


def srt(path: Path, cues):
    body = "\n\n".join(f"{i}\n{a} --> {b}\n{t}" for i, (a, b, t) in enumerate(cues, 1))
    path.write_text(body + "\n", encoding="utf-8")
    return path


def test_good_output_passes_with_metrics(clips, tmp_path):
    s = srt(tmp_path / "ok.srt", [("00:00:01,000", "00:00:04,000", "नमस्ते दोस्तों"),
                                 ("00:00:05,000", "00:00:09,000", "आज हम बात करेंगे")])
    r = D.check(clips / "ok.m4a", srt=s, paid_seconds=30)
    assert r["ok"] and r["warnings"] == [], r
    assert r["metrics"]["cues"] == 2 and len(r["metrics"]["sha256"]) == 64 and abs(r["metrics"]["duration"] - 30) < 0.5


def test_blocking_problems(clips, tmp_path):
    assert D.check(clips / "silent.wav")["problems"] == ["audio is silent"]
    assert any("paid for" in p for p in D.check(clips / "ok.m4a", paid_seconds=120)["problems"])
    bad = tmp_path / "bad.srt"
    bad.write_bytes("1\n00:00:01,000 --> 00:00:02,000\n\xe0 latin1".encode("latin-1"))
    assert D.check(clips / "ok.m4a", srt=bad)["problems"] == ["subtitles are not UTF-8"]
    late = srt(tmp_path / "late.srt", [("00:05:00,000", "00:05:02,000", "बहुत देर")])
    assert any("after the audio ends" in p for p in D.check(clips / "ok.m4a", srt=late)["problems"])


def test_warnings_clipping_gap_repeats_script(clips, tmp_path):
    assert any("clips" in w for w in D.check(clips / "clip.wav")["warnings"])
    s = srt(tmp_path / "g.srt", [("00:00:00,500", "00:00:04,500", "पहली बात"),
                                ("00:00:06,000", "00:00:16,000", "यह लाइन गायब है"),
                                ("00:00:18,000", "00:00:21,000", "यह लाइन गायब है")])
    w = D.check(clips / "gap.wav", srt=s)["warnings"]
    assert any("missing" in x for x in w) and any("repeated" in x for x in w), w
    en = srt(tmp_path / "en.srt", [("00:00:01,000", "00:00:03,000", "This was never translated at all")])
    assert any("Hindi script" in x for x in D.check(clips / "ok.m4a", srt=en)["warnings"])


def test_optional_spoken_language_check_uses_injected_asr(clips):
    class Eng:
        def transcribe(self, p): return "hello this is english"
    class Hin:
        def transcribe(self, p): return "नमस्ते यह हिंदी है"
    assert any("Hindi" in w for w in D.check(clips / "ok.m4a", asr_factory=Eng)["warnings"])
    assert D.check(clips / "ok.m4a", asr_factory=Hin)["warnings"] == []
