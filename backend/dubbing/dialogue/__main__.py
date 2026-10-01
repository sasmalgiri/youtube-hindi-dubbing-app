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

# Load torch (and its bundled cuDNN 9) BEFORE anything in this process can
# import faster-whisper/CTranslate2. If CTranslate2 runs on the GPU first, the
# next cuDNN-heavy torch op (Demucs) dies with "Could not load symbol
# cudnnGetLibConfig" (exit 127) -- reproduced 2026-10-01; torch-first fixes it.
try:
    import torch  # noqa: F401
except Exception:
    pass

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(BACKEND / ".env")
    except Exception:
        pass


def _overrides(args):
    ov = {"params": {}}
    for item in args.set:
        stage, _, val = item.partition("=")
        ov[stage.strip()] = [v.strip() for v in val.split(",") if v.strip()]
    if args.allow_paid:
        ov["params"]["allow_paid"] = True
    if args.local_only:
        ov["params"]["local_only"] = True
    if getattr(args, "ollama_model", ""):
        ov["params"]["ollama_model"] = args.ollama_model
    if getattr(args, "speakers", None):
        ov["params"]["num_speakers"] = args.speakers
    return ov


def _print_resolution(res):
    print(f"\nPreset '{res.preset}' on this PC:")
    for stage, sel in res.selections.items():
        print(f"  {stage:<17} {', '.join(sel) if sel else '(skipped)'}")
    for ch in res.changes:
        print(f"  * {ch['stage']}: {ch['action']} {ch['choice']} — {ch['reason']}"
              + (f"\n      fix: {ch['fix']}" if ch.get("fix") else ""))
    for w in res.warnings:
        print(f"  ! {w}")
    for b in res.blocking:
        print(f"  BLOCKED: {b}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dubbing.dialogue")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="check dependencies, credentials presence, GPU, disk")
    d.add_argument("--providers", default="edge")
    m = sub.add_parser("modules", help="show the module matrix, presets and what a preset "
                       "resolves to on this PC")
    m.add_argument("--preset", default="free-online")
    m.add_argument("--set", action="append", default=[], metavar="STAGE=a,b",
                   help="override a stage, e.g. --set voices=indic_parler,edge")
    m.add_argument("--allow-paid", action="store_true")
    m.add_argument("--local-only", action="store_true")
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
    r.add_argument("--preset", default=None,
                   help="module preset (see `modules`): free-online, free-local, fast-draft, "
                        "single-narrator, premium-voices, hindi-srt-revoice")
    r.add_argument("--set", action="append", default=[], metavar="STAGE=a,b",
                   help="override a stage of the preset, e.g. --set background=none")
    r.add_argument("--allow-paid", action="store_true", help="allow paid services")
    r.add_argument("--local-only", action="store_true", help="no cloud AI services")
    r.add_argument("--ollama-model", default="", help="Ollama model for local translation")
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

    if args.cmd == "modules":
        from dubbing.dialogue.modules import PRESETS, describe_matrix, resolve
        matrix = describe_matrix()
        for st in matrix["stages"]:
            print(f"\n{st['label']}  [{st['id']}]{'  (ordered chain)' if st['multi'] else ''}")
            for c in st["choices"]:
                mark = "OK  " if c["available"] else "--  "
                miss = "" if c["available"] else "  missing: " + ", ".join(x["name"] for x in c["missing"])
                print(f"  {mark}{c['id']:<14} {c['cost']:<9} {c['label']}{miss}")
        print("\nPresets: " + ", ".join(p["id"] for p in PRESETS))
        res = resolve(args.preset, _overrides(args), ctx={"source_kind": "file"})
        _print_resolution(res)
        return 0 if res.ok else 2

    from dubbing.dialogue.orchestrator import DialogueConfig, run_dialogue
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or (BACKEND / "dubbed_outputs" / f"dialogue_{stamp}")
    module_kw = {}
    modules = None
    if args.preset or args.set or args.allow_paid or args.local_only or args.ollama_model:
        from dubbing.dialogue.modules import resolve
        import re as _re
        res = resolve(args.preset or "free-online", _overrides(args),
                      ctx={"source_kind": "url" if _re.match(r"^https?://", args.source) else "file",
                           "files": {"english_srt": bool(args.srt_en), "hindi_srt": bool(args.srt_hi)}})
        _print_resolution(res)
        if not res.ok:
            return 2
        module_kw, modules = res.config, res.to_dict()
    legacy_kw = dict(
        use_youtube_subs=not args.no_youtube_subs, asr=args.asr,
        num_speakers=args.speakers,
        tts_providers=[p.strip() for p in args.providers.split(",") if p.strip()],
        translation_engines=[e.strip() for e in args.engines.split(",") if e.strip()],
        background=args.background, content_verify=args.verify, max_stretch=args.max_stretch,
    ) if not module_kw else {}
    cfg = DialogueConfig(
        source=args.source, work_dir=args.work or (out / "work"), output_dir=out,
        source_srt=args.srt_en, translated_srt=args.srt_hi, asr_model=args.asr_model,
        min_speakers=args.min_speakers, max_speakers=args.max_speakers,
        limit_seconds=args.limit_seconds, modules=modules, **legacy_kw, **module_kw)

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
