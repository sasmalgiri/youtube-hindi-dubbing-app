"""Check that TODO.md and TRACKER.md cover each other exactly.

Every tracker row ID must appear in some TODO task's Covers column, and
every ID a task names must exist in the tracker. Exit 1 on any mismatch.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
tracker = (HERE / "TRACKER.md").read_text(encoding="utf-8")
todo = (HERE / "TODO.md").read_text(encoding="utf-8")

row_ids = set(re.findall(r"^\| ([A-Z]\d{2}) \|", tracker, re.M))
covered = {}
for line in todo.splitlines():
    m = re.match(r"^\| (\d+) \| .* \| ([A-Z0-9, ]+) \| [^|]* \|$", line)
    if m:
        for rid in re.findall(r"[A-Z]\d{2}", m.group(2)):
            covered.setdefault(rid, []).append(int(m.group(1)))

missing = sorted(row_ids - covered.keys())
unknown = sorted(covered.keys() - row_ids)
dupes = {k: v for k, v in covered.items() if len(v) > 1}
print(f"tracker rows: {len(row_ids)} | TODO tasks: {len({t for v in covered.values() for t in v})} | "
      f"rows covered: {len(row_ids & covered.keys())}")
if dupes:
    print("rows covered by more than one task (allowed, check intent):",
          ", ".join(f"{k}->{v}" for k, v in sorted(dupes.items())))
if missing:
    print("NOT COVERED by any task:", ", ".join(missing))
if unknown:
    print("tasks name rows that do not exist:", ", ".join(unknown))
sys.exit(1 if missing or unknown else 0)
