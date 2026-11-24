"""Hindi translation benchmark (tracker Q03, Q06).

Runs the corpus through the real pipeline translation path
(``Pipeline._translate_segments``) for each engine whose key is configured,
scores every line with ``quality.metrics`` and writes a JSON + Markdown
report. Compare two runs with --baseline to see whether a change helped.

    python -m quality.translation_bench --engines groq,gemini
    python -m quality.translation_bench --engines groq --baseline reports/old.json --fail-drop 0.03

Calls the providers (free tiers where they exist); never runs in the test
suite. One engine = one translation request per corpus batch, same as a job.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import metrics

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "translation_corpus_v1.jsonl"
REPORTS = HERE / "reports"


def load_corpus(path: Path = CORPUS) -> List[Dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def pipeline_translator(engine: str) -> Callable[[List[str]], List[str]]:
    """Translate a list of English lines with the pipeline's own code path."""
    import tempfile
    from pipeline import Pipeline, PipelineConfig

    def run(lines: List[str]) -> List[str]:
        with tempfile.TemporaryDirectory() as td:
            cfg = PipelineConfig(source="bench.mp4", work_dir=Path(td) / "w", output_path=Path(td) / "o.mp4",
                                 translation_engine=engine, source_language="en", target_language="hi")
            p = Pipeline(cfg)
            # one segment per line, spaced apart so fragment merging never joins them
            segs = [{"start": i * 10.0, "end": i * 10.0 + 6.0, "text": t} for i, t in enumerate(lines)]
            p._translate_segments(segs)
            by_start = {round(s["start"], 1): s.get("text_translated", "") for s in segs}
            return [by_start.get(round(i * 10.0, 1), "") for i in range(len(lines))]
    return run


def run_engine(name: str, translate: Callable[[List[str]], List[str]], corpus: List[Dict]) -> Dict:
    t0 = time.time()
    err = None
    try:
        out = translate([c["en"] for c in corpus])
    except Exception as e:
        out, err = [""] * len(corpus), f"{type(e).__name__}: {str(e)[:200]}"
    scored = [metrics.score_item(c, h) for c, h in zip(corpus, out)]
    return {"engine": name, "seconds": round(time.time() - t0, 1), "error": err,
            "summary": metrics.summarize(scored), "items": scored}


def compare(cur: Dict, base: Dict) -> Dict[str, Dict]:
    out = {}
    b = {r["engine"]: r["summary"] for r in base.get("runs", [])}
    for r in cur.get("runs", []):
        if r["engine"] in b:
            s, o = r["summary"], b[r["engine"]]
            out[r["engine"]] = {"check_pass_rate": round(s["check_pass_rate"] - o["check_pass_rate"], 4),
                                "chrf_mean": round((s["chrf_mean"] or 0) - (o["chrf_mean"] or 0), 2)}
    return out


def markdown(report: Dict) -> str:
    L = [f"# Translation benchmark — corpus {report['corpus']} ({report['items']} lines)", "",
         f"Run {report['when']}. Automatic checks only; naturalness needs the Hindi reviewers.", "",
         "| Engine | Checks passed | chrF (drift) | Seconds | Error |", "|---|---|---|---|---|"]
    for r in report["runs"]:
        s = r["summary"]
        L.append(f"| {r['engine']} | {s['check_pass_rate']:.1%} | {s['chrf_mean']} | {r['seconds']} | {r['error'] or ''} |")
    for r in report["runs"]:
        L += ["", f"## {r['engine']}", "", "| Check | Passed |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in r["summary"]["checks"].items()]
        if r["summary"]["failures"]:
            L += ["", "Failures:"]
            L += [f"- `{f['id']}`: {'; '.join(f['notes'])}" for f in r["summary"]["failures"]]
    if report.get("vs_baseline"):
        L += ["", "## Change vs baseline", "", "| Engine | Δ checks | Δ chrF |", "|---|---|---|"]
        L += [f"| {e} | {d['check_pass_rate']:+.1%} | {d['chrf_mean']:+.2f} |" for e, d in report["vs_baseline"].items()]
    return "\n".join(L) + "\n"


def main(argv=None, translators: Optional[Dict[str, Callable]] = None) -> int:
    ap = argparse.ArgumentParser(description="Hindi translation benchmark")
    ap.add_argument("--engines", default="groq", help="comma list, e.g. groq,gemini,cerebras")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--out", default=str(REPORTS))
    ap.add_argument("--baseline", default=None, help="earlier report JSON to compare with")
    ap.add_argument("--fail-drop", type=float, default=None, help="exit 1 if check pass rate drops by more than this")
    a = ap.parse_args(argv)
    if translators is None:          # real engines: same keys as the server (backend/.env), like app.py
        from setup_check import _load_env_file
        _load_env_file()
    corpus = load_corpus(Path(a.corpus))
    runs = []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"translation_{time.strftime('%Y%m%d_%H%M%S')}"
    report = {"corpus": Path(a.corpus).stem, "items": len(corpus),
              "when": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": runs}

    def save():   # after every engine, so a hung provider never loses earlier results
        (out / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / f"{stem}.md").write_text(markdown(report), encoding="utf-8")

    for e in [x.strip() for x in a.engines.split(",") if x.strip()]:
        tr = (translators or {}).get(e) or pipeline_translator(e)
        print(f"[bench] {e}: translating {len(corpus)} lines...", flush=True)
        runs.append(run_engine(e, tr, corpus))
        s = runs[-1]["summary"]
        print(f"[bench] {e}: checks {s['check_pass_rate']:.1%}, chrF {s['chrf_mean']}"
              f"{' — ' + runs[-1]['error'] if runs[-1]['error'] else ''}", flush=True)
        save()
    rc = 0
    if a.baseline:
        report["vs_baseline"] = compare(report, json.loads(Path(a.baseline).read_text(encoding="utf-8")))
        if a.fail_drop is not None and any(d["check_pass_rate"] < -a.fail_drop for d in report["vs_baseline"].values()):
            rc = 1
    save()
    print(f"[bench] report: {out / (stem + '.md')}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
