"""CLI for the Hindi dialogue profile.

    cd backend
    python -m dubbing.dialogue doctor
    python -m dubbing.dialogue dub "https://www.youtube.com/watch?v=..." --out ..\\dubs\\my_video
    python -m dubbing.dialogue dub D:\\videos\\clip.mp4 --out ..\\dubs\\clip
    python -m dubbing.dialogue dub clip.mp4 --srt-en clip.en.srt       # English text supplied
    python -m dubbing.dialogue dub clip.mp4 --srt-hi clip.hi.srt       # Hindi text supplied
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(BACKEND / ".env")
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dubbing.dialogue")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="check dependencies, credentials presence, GPU, disk")
    d.add_argument("--providers", default="edge")
    r = sub.add_parser("dub", help="dub a URL or local video into Hindi")
    r.add_argument("source")
    r.add_argument("--out", type=Path, default=None, help="output folder")
    r.add_argument("--work", type=Path, default=None, help="work folder (default: <out>/work)")
    r.add_argument("--srt-en", type=Path, default=None, help="English SRT to use as text source")
    r.add_argument("--srt-hi", type=Path, default=None, help="Hindi SRT (skips translation)")
    r.add_argument("--no-youtube-subs", action="store_true")
    r.add_argument("--asr", default="auto", choices=["auto", "local", "groq"])
    r.add_argument("--asr-model", default="large-v3")
    r.add_argument("--speakers", type=int, default=None, help="exact number of speakers, if known")
    r.add_argument("--min-speakers", type=int, default=None)
    r.add_argument("--max-speakers", type=int, default=None)
    r.add_argument("--providers", default="edge",
                   help="comma list in priority order, e.g. edge or sarvam,edge (paid providers only if listed)")
    r.add_argument("--engines", default="gemini,groq,cerebras", help="LLM translation engines in order")
    r.add_argument("--background", default="auto", choices=["auto", "demucs", "none"])
    r.add_argument("--verify", default="auto", choices=["auto", "on", "off"])
    r.add_argument("--max-stretch", type=float, default=1.15)
    r.add_argument("--limit-seconds", type=float, default=0.0, help="dub only the first N seconds")
    args = ap.parse_args(argv)
    _load_env()

    if args.cmd == "doctor":
        from dubbing.dialogue.preflight import format_preflight, run_preflight
        res = run_preflight(providers=args.providers.split(","))
        print(format_preflight(res))
        return 0 if res["ready"] else 2

    from dubbing.dialogue.orchestrator import DialogueConfig, run_dialogue
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or (BACKEND / "dubbed_outputs" / f"dialogue_{stamp}")
    cfg = DialogueConfig(
        source=args.source, work_dir=args.work or (out / "work"), output_dir=out,
        source_srt=args.srt_en, translated_srt=args.srt_hi,
        use_youtube_subs=not args.no_youtube_subs, asr=args.asr, asr_model=args.asr_model,
        num_speakers=args.speakers, min_speakers=args.min_speakers, max_speakers=args.max_speakers,
        tts_providers=[p.strip() for p in args.providers.split(",") if p.strip()],
        translation_engines=[e.strip() for e in args.engines.split(",") if e.strip()],
        background=args.background, content_verify=args.verify, max_stretch=args.max_stretch,
        limit_seconds=args.limit_seconds)

    def progress(step, frac, msg):
        print(f"[{step:>10}] {int(frac * 100):3d}%  {msg}", flush=True)

    res = run_dialogue(cfg, on_progress=progress)
    print()
    print(f"Status : {res.status}")
    for why in res.reasons:
        print(f"  - {why}")
    print(f"Video  : {res.output_video}")
    print(f"Report : {res.report_md}")
    return 0 if res.status in ("completed", "completed_with_warnings") else 1


if __name__ == "__main__":
    sys.exit(main())
