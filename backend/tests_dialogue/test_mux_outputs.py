"""mix.mux output tracks on tiny synthetic media with real FFmpeg.

Streams are read back from FFmpeg's own stream dump (``ffmpeg -i``: the same
libavformat probe ffprobe uses, and ffprobe is not always installed next to
ffmpeg). The filter-path escaping helper is checked as a pure unit test
against a model of FFmpeg's two-level option parsing.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

from dubbing.dialogue import audio, mix

try:
    audio.find_ffmpeg()
    HAVE_FFMPEG = True
except RuntimeError:
    HAVE_FFMPEG = False
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")
needs_libass = pytest.mark.skipif(not (HAVE_FFMPEG and audio.has_filter("subtitles")),
                                  reason="ffmpeg without the libass subtitles filter")

DURATION = 2.0
HI_SRT = "1\n00:00:00,000 --> 00:00:01,800\nनमस्ते दुनिया\n"
EN_SRT = "1\n00:00:00,000 --> 00:00:01,800\nHello world\n"
# ASCII, large and white so a burned cue is easy to find on a black frame
BURN_SRT = "1\n00:00:00,000 --> 00:00:01,400\nHELLO WORLD\n"


def _make_video(path: Path, with_audio: bool = True) -> Path:
    args = ["-f", "lavfi", "-i", f"color=c=black:size=160x120:rate=10:duration={DURATION}"]
    if with_audio:  # source audio at 44.1 kHz, the Hindi mix at 48 kHz: tells them apart
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={DURATION}:sample_rate=44100"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if with_audio:
        args += ["-c:a", "aac", "-shortest"]
    audio.run_ffmpeg(args + [str(path)])
    return path


@pytest.fixture
def media(tmp_path):
    video = _make_video(tmp_path / "input.mp4")
    wav = tmp_path / "hindi_mix.wav"
    audio.run_ffmpeg(["-f", "lavfi", "-i", f"sine=frequency=220:duration={DURATION}:sample_rate=48000",
                      "-ac", "2", "-acodec", "pcm_s16le", str(wav)])
    hi, en = tmp_path / "subtitles_hi.srt", tmp_path / "subtitles_en.srt"
    hi.write_text(HI_SRT, encoding="utf-8")
    en.write_text(EN_SRT, encoding="utf-8")
    return {"dir": tmp_path, "video": video, "wav": wav, "hi": hi, "en": en}


def _probe(path: Path):
    """[{type, codec, lang, default, sample_rate, tags}] in stream order + format."""
    res = subprocess.run([audio.find_ffmpeg(), "-hide_banner", "-i", str(path)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    fmt = re.search(r"Input #0, ([^ ]+), from", res.stderr).group(1)
    streams, cur = [], None
    for line in res.stderr.splitlines():
        m = re.match(r"\s*Stream #0:(\d+)(?:\[\w+\])?(?:\((\w+)\))?: (Video|Audio|Subtitle): (\w+)(.*)$",
                     line)
        if m:
            idx, lang, kind, codec, rest = m.groups()
            assert int(idx) == len(streams)
            sr = re.search(r"(\d+) Hz", rest)
            cur = {"type": kind.lower(), "codec": codec, "lang": lang, "default": "(default)" in rest,
                   "sample_rate": int(sr.group(1)) if sr else None, "tags": {}}
            streams.append(cur)
            continue
        m = re.match(r"\s{6,}(\w+)\s*: (.*)$", line)
        if cur is not None and m:
            cur["tags"][m.group(1).lower()] = m.group(2).strip()
        elif not line.startswith(" " * 4):
            cur = None
    return streams, fmt


def _name(stream) -> str:
    """Track name: mkv stores ``title``, mp4 players show ``handler_name``."""
    return stream["tags"].get("title") or stream["tags"].get("handler_name")


def _gray_frame(path: Path, t: float = 0.5) -> bytes:
    res = subprocess.run([audio.find_ffmpeg(), "-hide_banner", "-ss", str(t), "-i", str(path),
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True)
    assert res.returncode == 0 and res.stdout
    return res.stdout


def _h264(path: Path) -> bytes:
    """The video stream's raw H.264 bitstream (no decode)."""
    res = subprocess.run([audio.find_ffmpeg(), "-hide_banner", "-i", str(path), "-map", "0:v:0",
                          "-c:v", "copy", "-f", "h264", "-"], capture_output=True)
    assert res.returncode == 0 and res.stdout
    return res.stdout


def _bright_pixels(frame: bytes) -> int:
    return sum(1 for b in frame if b > 128)


def _faststart(path: Path) -> bool:
    data = path.read_bytes()
    return 0 <= data.find(b"moov") < data.find(b"mdat")


@needs_ffmpeg
def test_old_call_form_is_unchanged(media):
    out = media["dir"] / "dubbed_hi.mp4"
    assert mix.mux(media["video"], media["wav"], out, "192k", subtitles=media["hi"]) == out
    streams, fmt = _probe(out)
    assert "mp4" in fmt
    assert [s["type"] for s in streams] == ["video", "audio", "subtitle"]
    v, a, s = streams
    assert v["codec"] == "h264"
    assert a["codec"] == "aac" and a["lang"] == "hin" and a["default"] and a["sample_rate"] == 48000
    assert _name(a) == "Hindi"
    assert s["codec"] == "mov_text" and s["lang"] == "hin" and s["default"]
    assert _faststart(out)
    # video copied, not re-encoded: the H.264 bitstream is byte-identical
    assert _h264(out) == _h264(media["video"])
    # and without subtitles: just video + Hindi audio
    out2 = media["dir"] / "plain.mp4"
    mix.mux(media["video"], media["wav"], out2)
    assert [s["type"] for s in _probe(out2)[0]] == ["video", "audio"]


@needs_ffmpeg
def test_mp4_original_audio_and_english_subtitles(media):
    out = media["dir"] / "dubbed_hi.mp4"
    mix.mux(media["video"], media["wav"], out, subtitles=media["hi"], original_audio=True,
            extra_subtitles=[(media["en"], "eng", "English")])
    streams, fmt = _probe(out)
    assert "mp4" in fmt
    assert [s["type"] for s in streams] == ["video", "audio", "audio", "subtitle", "subtitle"]
    _, hi_a, orig_a, hi_s, en_s = streams
    # Hindi (the 48 kHz mix) first and default; the source's own audio second, not default
    assert (hi_a["lang"], _name(hi_a), hi_a["default"], hi_a["sample_rate"]) == ("hin", "Hindi", True, 48000)
    assert (orig_a["lang"], _name(orig_a), orig_a["default"], orig_a["sample_rate"]) == \
        ("eng", "Original", False, 44100)
    assert (hi_s["codec"], hi_s["lang"], _name(hi_s), hi_s["default"]) == ("mov_text", "hin", "Hindi", True)
    assert (en_s["codec"], en_s["lang"], _name(en_s), en_s["default"]) == \
        ("mov_text", "eng", "English", False)
    assert _faststart(out)


@needs_ffmpeg
def test_mkv_container_uses_srt_streams(media):
    out = media["dir"] / "dubbed_hi.mkv"
    mix.mux(media["video"], media["wav"], out, subtitles=media["hi"], original_audio=True,
            extra_subtitles=[(media["en"], "eng", "English")], container="mkv")
    streams, fmt = _probe(out)
    assert "matroska" in fmt
    assert [s["type"] for s in streams] == ["video", "audio", "audio", "subtitle", "subtitle"]
    _, hi_a, orig_a, hi_s, en_s = streams
    assert hi_a["default"] and not orig_a["default"]
    assert (hi_a["tags"].get("title"), orig_a["tags"].get("title")) == ("Hindi", "Original")
    assert (hi_a["lang"], orig_a["lang"]) == ("hin", "eng")
    assert (hi_s["codec"], hi_s["lang"], hi_s["tags"].get("title"), hi_s["default"]) == \
        ("subrip", "hin", "Hindi", True)
    assert (en_s["codec"], en_s["lang"], en_s["tags"].get("title"), en_s["default"]) == \
        ("subrip", "eng", "English", False)
    # the container option decides the format, not the file name
    odd = media["dir"] / "named_wrong.mp4"
    mix.mux(media["video"], media["wav"], odd, container="mkv")
    assert "matroska" in _probe(odd)[1]


@needs_ffmpeg
@pytest.mark.parametrize("container", ["mkv", "mp4"])
def test_english_only_subtitles(media, container):
    out = media["dir"] / f"dubbed_hi.{container}"
    mix.mux(media["video"], media["wav"], out, extra_subtitles=[(media["en"], "eng", "English")],
            container=container)
    streams, _ = _probe(out)
    assert [s["type"] for s in streams] == ["video", "audio", "subtitle"]
    assert streams[2]["lang"] == "eng" and _name(streams[2]) == "English"
    if container == "mkv":  # never auto-shown; FFmpeg's mp4 muxer always enables the
        assert not streams[2]["default"]  # first track of a type, so mp4 reads back default


@needs_ffmpeg
def test_original_audio_skipped_when_source_is_silent_video(media):
    silent = _make_video(media["dir"] / "no_audio.mp4", with_audio=False)
    out = media["dir"] / "dubbed_hi.mp4"
    mix.mux(silent, media["wav"], out, original_audio=True)
    streams, _ = _probe(out)
    assert [s["type"] for s in streams] == ["video", "audio"]
    assert streams[1]["lang"] == "hin" and streams[1]["default"]


@needs_ffmpeg
def test_srt_without_cues_is_left_out(media):
    empty = media["dir"] / "empty.srt"
    empty.write_text("", encoding="utf-8")
    out = media["dir"] / "dubbed_hi.mp4"
    mix.mux(media["video"], media["wav"], out, subtitles=empty,
            extra_subtitles=[(empty, "eng", "English")], burn_subtitles=empty)
    streams, _ = _probe(out)
    assert [s["type"] for s in streams] == ["video", "audio"]
    assert _h264(out) == _h264(media["video"])  # nothing to burn -> copied


@needs_libass
def test_burned_subtitles_reencode_the_picture(media):
    # a folder name full of filter-syntax characters proves the path escaping
    tricky = media["dir"] / ("sub's dir [1],x;y=z" + ("" if os.name == "nt" else ":w"))
    tricky.mkdir()
    burn = tricky / "burn.srt"
    burn.write_text(BURN_SRT, encoding="utf-8")
    out = media["dir"] / "dubbed_hi.mp4"
    mix.mux(media["video"], media["wav"], out, burn_subtitles=burn)
    streams, _ = _probe(out)
    # burned only: no soft subtitle stream
    assert [s["type"] for s in streams] == ["video", "audio"]
    assert streams[0]["codec"] == "h264"
    assert _h264(out) != _h264(media["video"])  # re-encoded
    assert _bright_pixels(_gray_frame(media["video"])) == 0
    assert _bright_pixels(_gray_frame(out)) > 20  # the cue is drawn into the frame
    assert _bright_pixels(_gray_frame(out, t=1.7)) == 0  # ...and gone after it ends
    assert _faststart(out)


@needs_libass
def test_burn_handles_odd_sized_video(media):
    odd = media["dir"] / "odd.mkv"
    audio.run_ffmpeg(["-f", "lavfi", "-i", f"color=c=black:size=161x121:rate=10:duration={DURATION}",
                      "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv444p", str(odd)])
    burn = media["dir"] / "burn.srt"
    burn.write_text(BURN_SRT, encoding="utf-8")
    out = media["dir"] / "dubbed_hi.mp4"
    mix.mux(odd, media["wav"], out, burn_subtitles=burn)
    res = subprocess.run([audio.find_ffmpeg(), "-hide_banner", "-i", str(out)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert re.search(r"Video: h264 .*yuv420p.*, 160x120", res.stderr)


@needs_libass
def test_burned_and_soft_subtitles_together_mkv(media):
    burn = media["dir"] / "burn.srt"
    burn.write_text(BURN_SRT, encoding="utf-8")
    out = media["dir"] / "dubbed_hi.mkv"
    mix.mux(media["video"], media["wav"], out, subtitles=media["hi"], burn_subtitles=burn,
            container="mkv")
    streams, fmt = _probe(out)
    assert "matroska" in fmt
    assert [s["type"] for s in streams] == ["video", "audio", "subtitle"]
    assert streams[0]["codec"] == "h264" and streams[2]["codec"] == "subrip"
    assert _bright_pixels(_gray_frame(out)) > 20


# ── filter-path escaping (pure) ─────────────────────────────────────────────
def _av_get_token(buf: str, term: str):
    """Python model of libavutil av_get_token(): quotes and backslash
    escapes, stops at an unescaped ``term`` character. Returns (token, rest)."""
    out, end, i = [], 0, 0
    while i < len(buf) and buf[i] in " \n\t\r":
        i += 1
    while i < len(buf) and buf[i] not in term:
        c = buf[i]
        i += 1
        if c == "\\" and i < len(buf):
            out.append(buf[i])
            i += 1
            end = len(out)
        elif c == "'":
            while i < len(buf) and buf[i] != "'":
                out.append(buf[i])
                i += 1
            if i < len(buf):
                i += 1
                end = len(out)
        else:
            out.append(c)
    tok = "".join(out)
    return tok[:end] + tok[end:].rstrip(" \n\t\r"), buf[i:]


def _ffmpeg_reads(escaped: str) -> str:
    """What the subtitles filter receives from ``-vf subtitles=filename=<escaped>``."""
    opts, rest = _av_get_token(f"filename={escaped}", "[],;")  # filtergraph level
    assert rest == ""
    key, _, value = opts.partition("=")
    assert key == "filename"
    value, rest = _av_get_token(value, ":")  # filter option level
    assert rest == ""
    return value


def test_filter_path_windows_drive_and_backslashes():
    p = r"C:\Users\me\x y\subs.srt"
    esc = mix._filter_path(p, windows=True)
    assert esc == r"C\\:/Users/me/x y/subs.srt"
    assert _ffmpeg_reads(esc) == "C:/Users/me/x y/subs.srt"
    tricky = r"D:\dub jobs\it's [v2], final; ok\subtitles_hi.srt"
    assert _ffmpeg_reads(mix._filter_path(tricky, windows=True)) == \
        "D:/dub jobs/it's [v2], final; ok/subtitles_hi.srt"


def test_filter_path_posix_keeps_backslashes_literal():
    for p in ["/tmp/out/subtitles_hi.srt", "/home/me/x y/a:b/subs.srt",
              "/srv/it's [v2], final; ok/s=1.srt", "/odd/back\\slash/subs.srt"]:
        assert _ffmpeg_reads(mix._filter_path(p, windows=False)) == p
