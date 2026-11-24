"""Backup and restore of the service state (tracker T09, R01).

    python -m ops.backup create  [--state DIR] [--dest DIR] [--with-outputs]
    python -m ops.backup restore BACKUP_DIR --target DIR
    python -m ops.backup verify  BACKUP_DIR --target DIR

create   every *.db through SQLite's online backup API (consistent while the
         server runs), plus manifests/, drafts/ and deletions.jsonl (outputs/ with
         --with-outputs); writes backup.json with row counts per table, sha256 of
         every file and each account's prepaid balances.
restore  copies a backup into an EMPTY state dir, then runs verify.
verify   compares a state dir with backup.json: same tables and row counts,
         same file hashes, same balances per account and credit type, same job
         ids and owners. Exit 1 on any difference.
Secrets (.env) are never part of a backup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

FILE_DIRS = ("manifests", "drafts")
FILE_NAMES = ("deletions.jsonl", "features.json", "saved_links.json", "completed_urls.json")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _counts(db: Path) -> Dict[str, int]:
    c = sqlite3.connect(str(db))
    try:
        names = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {n: c.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in sorted(names)}
    finally:
        c.close()


def _balances(state: Path) -> Dict[str, Dict[str, List[int]]]:
    db = state / "ledger.db"
    if not db.exists():
        return {}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from billing.ledger import Ledger
    led = Ledger(db)
    try:
        out: Dict[str, Dict[str, List[int]]] = {}
        for a in sorted({e.account_id for e in led.entries()}):
            out[a] = {t: [b.spendable, b.reserved, b.consumed, b.debt] for t, b in led.balances(a).items()}
        return out
    finally:
        led.close()


def _jobs(state: Path) -> Dict[str, Optional[str]]:
    db = state / "jobs.db"
    if not db.exists():
        return {}
    c = sqlite3.connect(str(db))
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(jobs)")]
        if "owner_id" in cols:
            return {r[0]: r[1] for r in c.execute("SELECT id, owner_id FROM jobs")}
        return {r[0]: None for r in c.execute("SELECT id FROM jobs")}
    except sqlite3.OperationalError:
        return {}
    finally:
        c.close()


def _files(root: Path, with_outputs: bool) -> List[Path]:
    out = [root / n for n in FILE_NAMES if (root / n).is_file()]
    for d in FILE_DIRS + (("outputs",) if with_outputs else ()):
        if (root / d).is_dir():
            out += [p for p in (root / d).rglob("*") if p.is_file()]
    return out


def describe(state: Path, with_outputs: bool = False) -> Dict:
    dbs = sorted(p for p in state.glob("*.db") if p.is_file())
    return {
        "databases": {p.name: _counts(p) for p in dbs},
        "files": {str(p.relative_to(state)).replace("\\", "/"): _sha(p) for p in _files(state, with_outputs)},
        "balances": _balances(state),
        "jobs": _jobs(state),
    }


def create(state: Path, dest: Path, with_outputs: bool = False) -> Path:
    state, dest = Path(state), Path(dest)
    out = dest / f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    for db in sorted(state.glob("*.db")):
        src = sqlite3.connect(str(db))
        dst = sqlite3.connect(str(out / db.name))
        try:
            src.backup(dst)          # consistent snapshot even while writers are active
        finally:
            dst.close()
            src.close()
    for f in _files(state, with_outputs):
        rel = f.relative_to(state)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / rel)
    info = describe(out, with_outputs)
    info.update({"created_at": int(time.time()), "source": str(state), "with_outputs": with_outputs})
    (out / "backup.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return out


def verify(backup: Path, target: Path) -> List[str]:
    want = json.loads((Path(backup) / "backup.json").read_text(encoding="utf-8"))
    got = describe(Path(target), want.get("with_outputs", False))
    diffs = []
    for name, counts in want["databases"].items():
        if got["databases"].get(name) != counts:
            diffs.append(f"{name}: rows {got['databases'].get(name)} != {counts}")
    for rel, h in want["files"].items():
        if got["files"].get(rel) != h:
            diffs.append(f"file {rel}: {'missing' if rel not in got['files'] else 'different content'}")
    if got["balances"] != want["balances"]:
        diffs.append("prepaid balances differ")
    if got["jobs"] != want["jobs"]:
        diffs.append("job ids or owners differ")
    return diffs


def restore(backup: Path, target: Path) -> List[str]:
    backup, target = Path(backup), Path(target)
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"refusing to restore into a non-empty folder: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for p in backup.rglob("*"):
        if p.is_file() and p.name != "backup.json":
            (target / p.relative_to(backup)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target / p.relative_to(backup))
    return verify(backup, target)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Backup / restore the service state")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--state", default=os.environ.get("VOICEDUB_STATE_DIR") or str(Path(__file__).resolve().parents[1]))
    c.add_argument("--dest", default="backups")
    c.add_argument("--with-outputs", action="store_true")
    for name in ("restore", "verify"):
        r = sub.add_parser(name)
        r.add_argument("backup")
        r.add_argument("--target", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "create":
        out = create(Path(a.state), Path(a.dest), a.with_outputs)
        info = json.loads((out / "backup.json").read_text(encoding="utf-8"))
        print(f"backup: {out} ({len(info['databases'])} databases, {len(info['files'])} files, "
              f"{len(info['balances'])} accounts)")
        return 0
    diffs = restore(Path(a.backup), Path(a.target)) if a.cmd == "restore" else verify(Path(a.backup), Path(a.target))
    for d in diffs:
        print("DIFF", d)
    print("restore verified: identical" if not diffs else f"{len(diffs)} difference(s)")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
