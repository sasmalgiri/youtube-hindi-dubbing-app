"""Daily reconciliation (W5: M07). Read-only: it reports, it never repairs.

Checks
  * every account's ledger invariants (seconds conserved, no over-release)
  * paid order  <-> ledger grant  ``pay:<payment_id>`` (amount of seconds, account)
  * ledger payment grant without a paid order (orphan / duplicate credit)
  * orders stuck in ``created`` longer than STALE_SECONDS (abandoned or a lost webhook)
  * open reservations (reserved, not settled/released) older than STALE_SECONDS
    whose job is not queued/running (lost credit)
  * negative spendable / debt per account

Usage:  python -m billing.reconcile [--state DIR] [--json]
Exit code 1 when anything is found, so a scheduler can alert on it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from .ledger import Ledger

STALE_SECONDS = 24 * 3600


def _rows(db: Path, sql: str, args=()) -> List[tuple]:
    if not db.exists():
        return []
    c = sqlite3.connect(str(db))
    try:
        return c.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        c.close()


def run(state: Path, now: Optional[int] = None, active_jobs: Optional[Set[str]] = None) -> Dict[str, List[str]]:
    now = int(now or time.time())
    out: Dict[str, List[str]] = {"ledger": [], "orders": [], "reservations": [], "accounts": []}
    led = Ledger(state / "ledger.db")
    try:
        entries = led.entries()
        accounts = sorted({e.account_id for e in entries})
        for a in accounts:
            for p in led.invariants(a):
                out["ledger"].append(f"{a}: {p}")
            for t, b in led.balances(a).items():
                if b.debt > 0:
                    out["accounts"].append(f"{a}/{t}: owes {b.debt}s after a chargeback")
        grants = {e.business_key: e for e in entries if e.kind == "grant" and e.business_key.startswith("pay:")}

        orders = _rows(state / "payments.db",
                       "SELECT id, owner_id, credit_type, seconds, status, payment_id, created_at FROM orders")
        paid_keys = set()
        for oid, owner, ctype, secs, status, pid, created in orders:
            if status in ("paid", "refunded", "disputed") and pid:
                k = f"pay:{pid}"
                paid_keys.add(k)
                g = grants.get(k)
                if g is None:
                    out["orders"].append(f"{oid}: paid ({pid}) but no ledger grant")
                elif g.account_id != owner or g.seconds != secs or g.credit_type != ctype:
                    out["orders"].append(f"{oid}: grant does not match the order ({g.account_id}/{g.credit_type}/{g.seconds})")
            elif status == "created" and now - int(created) > STALE_SECONDS:
                out["orders"].append(f"{oid}: still 'created' after {(now - int(created)) // 3600}h "
                                     f"(abandoned, or a webhook was lost — check the provider)")
        if orders:   # only meaningful once checkout exists; manual test grants use other keys
            for k, g in grants.items():
                if k not in paid_keys and not k.startswith("pay:t") and not k.startswith("pay:ui"):
                    out["orders"].append(f"{k}: payment grant to {g.account_id} with no paid order")

        active = active_jobs if active_jobs is not None else {
            r[0] for r in _rows(state / "queue.db", "SELECT job_id FROM work WHERE state IN ('queued','running')")}
        by_ref: Dict[str, Dict[str, int]] = {}
        first_seen: Dict[str, int] = {}
        for e in entries:
            if e.kind in ("reserve", "settle", "release"):
                d = by_ref.setdefault(f"{e.account_id}|{e.ref}", {"reserve": 0, "settle": 0, "release": 0})
                d[e.kind] += e.seconds
                if e.kind == "reserve":
                    first_seen.setdefault(f"{e.account_id}|{e.ref}", e.created_at)
        for key, d in by_ref.items():
            open_s = d["reserve"] - d["settle"] - d["release"]
            acct, ref = key.split("|", 1)
            if open_s > 0 and ref not in active and now - first_seen.get(key, now) > STALE_SECONDS:
                out["reservations"].append(f"{acct}/{ref}: {open_s}s reserved for over a day and the job is not running")
    finally:
        led.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=None, help="state dir (default: VOICEDUB_STATE_DIR or backend/)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    import os
    state = Path(a.state or os.environ.get("VOICEDUB_STATE_DIR") or Path(__file__).resolve().parents[1])
    res = run(state)
    n = sum(len(v) for v in res.values())
    if a.json:
        print(json.dumps({"problems": n, **res}, indent=2))
    else:
        print(f"Reconciliation of {state}: {n} problem(s)")
        for k, v in res.items():
            for line in v:
                print(f"  [{k}] {line}")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
