"""Immutable prepaid-time ledger (tracker M04, M05, M06, T02).

Every change is an appended entry with a unique **business key**. Replaying
the same operation (a retried webhook, a duplicate worker completion) finds
the existing key and has no further effect: retries have the economic effect
of one attempt. Entries are never updated or deleted.

Kinds and their effect on one account's balance of one credit type:

    grant       +available            pack purchased (key: payment id)
    reserve     −available +reserved  funded job accepted (key: job/child id)
    settle      −reserved  +consumed  delivered (once per reservation)
    release     −reserved  +available failed / cancelled before delivery
    refund      −available            unused pack refunded (cash goes back via M05)
    chargeback  −available            disputed payment; may push the raw balance
                                      below zero, which is reported as debt, never
                                      as spendable time, and blocks new reserves
    adjust      ±available            operator repair with an audit note

Derived balance: ``available = grant + adjust + release − reserve − refund − chargeback``,
``reserved = reserve − settle − release``, ``consumed = settle``.
Spendable = max(0, available); debt = max(0, −available).

A reservation for a single job may draw from both credit types (Standard
first). Each part is its own entry, so a release restores exactly the
original grant type (§4 "Restore to the original grant type").
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .catalog import CREDIT_TYPES, PREMIUM, STANDARD

KINDS = ("grant", "reserve", "settle", "release", "refund", "chargeback", "adjust")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('grant','reserve','settle','release','refund','chargeback','adjust')),
    credit_type  TEXT NOT NULL CHECK (credit_type IN ('standard','premium')),
    seconds      INTEGER NOT NULL CHECK (seconds >= 0),
    sign         INTEGER NOT NULL DEFAULT 1 CHECK (sign IN (1, -1)),
    business_key TEXT NOT NULL UNIQUE,
    ref          TEXT NOT NULL DEFAULT '',
    note         TEXT NOT NULL DEFAULT '',
    created_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_account ON ledger(account_id, credit_type);
CREATE INDEX IF NOT EXISTS ledger_ref ON ledger(ref);
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger
BEGIN SELECT RAISE(ABORT, 'ledger entries are immutable'); END;
CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger
BEGIN SELECT RAISE(ABORT, 'ledger entries are immutable'); END;
"""


class LedgerError(Exception):
    pass


class InsufficientBalance(LedgerError):
    pass


@dataclass(frozen=True)
class Balance:
    available: int      # raw (may be negative after a chargeback)
    reserved: int
    consumed: int

    @property
    def spendable(self) -> int:
        return max(0, self.available)

    @property
    def debt(self) -> int:
        return max(0, -self.available)


@dataclass(frozen=True)
class Entry:
    seq: int
    account_id: str
    kind: str
    credit_type: str
    seconds: int
    sign: int
    business_key: str
    ref: str
    note: str
    created_at: int


class Ledger:
    def __init__(self, db_path: Path, clock=time.time):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(SCHEMA)

    # One connection per thread; BEGIN IMMEDIATE serialises writers across
    # threads *and* processes (two API workers, two tabs) on the same file.
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=FULL")
            self._local.conn = c
        return c

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    # ── reads ──
    def _balance(self, c: sqlite3.Connection, account_id: str, credit_type: str) -> Balance:
        rows = c.execute("SELECT kind, SUM(seconds * sign) FROM ledger WHERE account_id=? AND credit_type=? GROUP BY kind",
                         (account_id, credit_type)).fetchall()
        t = {k: int(v or 0) for k, v in rows}
        available = (t.get("grant", 0) + t.get("adjust", 0) + t.get("release", 0)
                     - t.get("reserve", 0) - t.get("refund", 0) - t.get("chargeback", 0))
        reserved = t.get("reserve", 0) - t.get("settle", 0) - t.get("release", 0)
        return Balance(available=available, reserved=reserved, consumed=t.get("settle", 0))

    def balance(self, account_id: str, credit_type: str) -> Balance:
        return self._balance(self._conn(), account_id, credit_type)

    def balances(self, account_id: str) -> Dict[str, Balance]:
        return {t: self.balance(account_id, t) for t in CREDIT_TYPES}

    def spendable(self, account_id: str) -> Dict[str, int]:
        return {t: b.spendable for t, b in self.balances(account_id).items()}

    def entries(self, account_id: Optional[str] = None, ref: Optional[str] = None) -> List[Entry]:
        q, args = "SELECT * FROM ledger WHERE 1=1", []
        if account_id is not None:
            q += " AND account_id=?"; args.append(account_id)
        if ref is not None:
            q += " AND ref=?"; args.append(ref)
        return [Entry(*r) for r in self._conn().execute(q + " ORDER BY seq", args).fetchall()]

    def get(self, business_key: str) -> Optional[Entry]:
        r = self._conn().execute("SELECT * FROM ledger WHERE business_key=?", (business_key,)).fetchone()
        return Entry(*r) if r else None

    # ── writes ──
    def _insert(self, c, account_id, kind, credit_type, seconds, key, ref="", note="", sign=1) -> Entry:
        if kind not in KINDS or credit_type not in CREDIT_TYPES:
            raise LedgerError(f"bad entry {kind}/{credit_type}")
        if not isinstance(seconds, int) or seconds < 0:
            raise LedgerError("seconds must be a non-negative integer")
        cur = c.execute(
            "INSERT INTO ledger(account_id, kind, credit_type, seconds, sign, business_key, ref, note, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (account_id, kind, credit_type, seconds, sign, key, ref, note, int(self._clock())))
        return Entry(cur.lastrowid, account_id, kind, credit_type, seconds, sign, key, ref, note, int(self._clock()))

    def _tx(self, fn):
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            out = fn(c)
            c.execute("COMMIT")
            return out
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def _existing(self, c, key: str, account_id: str, kind: str) -> Optional[Entry]:
        r = c.execute("SELECT * FROM ledger WHERE business_key=?", (key,)).fetchone()
        if r is None:
            return None
        e = Entry(*r)
        if e.account_id != account_id or e.kind != kind:
            raise LedgerError(f"business key {key} already used for {e.kind} on another account")
        return e

    def grant(self, account_id: str, credit_type: str, seconds: int, key: str, ref: str = "", note: str = "") -> Entry:
        """Idempotent on ``key`` (use the verified payment id, never a client value)."""
        def fn(c):
            return self._existing(c, key, account_id, "grant") or \
                self._insert(c, account_id, "grant", credit_type, seconds, key, ref, note)
        return self._tx(fn)

    def reserve(self, account_id: str, job_key: str, seconds: int,
                order: Sequence[str] = (STANDARD, PREMIUM), note: str = "") -> Dict[str, int]:
        """Reserve ``seconds`` for one job, drawing credit types in ``order``.

        All-or-nothing in one transaction: either the whole amount is reserved
        or nothing is and InsufficientBalance is raised. Idempotent on job_key:
        a repeat returns the original split without reserving again.
        """
        if seconds <= 0:
            raise LedgerError("a reservation needs a positive number of seconds")

        def fn(c):
            prior = {t: self._existing(c, f"res:{job_key}:{t}", account_id, "reserve") for t in CREDIT_TYPES}
            if any(prior.values()):
                return {t: (e.seconds if e else 0) for t, e in prior.items()}
            need, split = seconds, {t: 0 for t in CREDIT_TYPES}
            for t in order:
                take = min(need, self._balance(c, account_id, t).spendable)
                split[t], need = take, need - take
            if need:
                raise InsufficientBalance(f"{need} seconds short")
            for t, s in split.items():
                if s:
                    self._insert(c, account_id, "reserve", t, s, f"res:{job_key}:{t}", job_key, note)
            return split
        return self._tx(fn)

    def _reservation(self, c, account_id: str, job_key: str) -> Dict[str, Tuple[int, int, int]]:
        """credit_type → (reserved, settled, released) for one job."""
        out = {}
        for t in CREDIT_TYPES:
            vals = []
            for kind in ("reserve", "settle", "release"):
                r = c.execute("SELECT COALESCE(SUM(seconds),0) FROM ledger WHERE account_id=? AND credit_type=? "
                              "AND kind=? AND ref=?", (account_id, t, kind, job_key)).fetchone()
                vals.append(int(r[0]))
            out[t] = tuple(vals)
        return out

    def settle(self, account_id: str, job_key: str, seconds: Optional[int] = None, note: str = "") -> Dict[str, int]:
        """Delivered. Settles once (key ``set:job``); a shorter delivery settles
        ``seconds`` and releases the rest in the same transaction."""
        def fn(c):
            prior = {t: self._existing(c, f"set:{job_key}:{t}", account_id, "settle") for t in CREDIT_TYPES}
            if any(prior.values()):
                return {t: (e.seconds if e else 0) for t, e in prior.items()}
            res = self._reservation(c, account_id, job_key)
            open_ = {t: r - s - rel for t, (r, s, rel) in res.items()}
            total_open = sum(open_.values())
            if total_open <= 0:
                raise LedgerError(f"nothing reserved for {job_key}")
            want = total_open if seconds is None else int(seconds)
            if want < 0 or want > total_open:
                raise LedgerError(f"cannot settle {want}s of a {total_open}s reservation")
            done = {t: 0 for t in CREDIT_TYPES}
            # settle the type reserved first first (Standard before Premium), release the remainder
            for t in CREDIT_TYPES:
                take = min(want, open_[t])
                if take:
                    self._insert(c, account_id, "settle", t, take, f"set:{job_key}:{t}", job_key, note)
                done[t], want = take, want - take
                rest = open_[t] - take
                if rest:
                    self._insert(c, account_id, "release", t, rest, f"rel:{job_key}:{t}", job_key, "unused part of delivery")
            return done
        return self._tx(fn)

    def release(self, account_id: str, job_key: str, note: str = "") -> Dict[str, int]:
        """Failed or cancelled before delivery: the whole open reservation goes
        back to the credit type it came from. Idempotent; a no-op after settle."""
        def fn(c):
            res = self._reservation(c, account_id, job_key)
            out = {}
            for t, (r, s, rel) in res.items():
                open_ = r - s - rel
                key = f"rel:{job_key}:{t}"
                if open_ > 0 and self._existing(c, key, account_id, "release") is None:
                    self._insert(c, account_id, "release", t, open_, key, job_key, note)
                    out[t] = open_
                else:
                    out[t] = 0
            return out
        return self._tx(fn)

    def refund_unused(self, account_id: str, grant_key: str, key: str, note: str = "") -> Entry:
        """Refund a whole pack only while every second of it is still spendable."""
        def fn(c):
            prior = self._existing(c, key, account_id, "refund")
            if prior:
                return prior
            g = self._existing(c, grant_key, account_id, "grant")
            if g is None:
                raise LedgerError(f"no grant {grant_key}")
            if self._balance(c, account_id, g.credit_type).spendable < g.seconds:
                raise InsufficientBalance("part of this pack is used or reserved: refund needs the partial-consumption policy")
            return self._insert(c, account_id, "refund", g.credit_type, g.seconds, key, grant_key, note)
        return self._tx(fn)

    def chargeback(self, account_id: str, grant_key: str, key: str, note: str = "") -> Entry:
        """Remove a disputed pack even when part of it is reserved by running
        work. The shortfall shows as debt; releases of those reservations pay
        the debt down instead of becoming spendable again."""
        def fn(c):
            prior = self._existing(c, key, account_id, "chargeback")
            if prior:
                return prior
            g = self._existing(c, grant_key, account_id, "grant")
            if g is None:
                raise LedgerError(f"no grant {grant_key}")
            return self._insert(c, account_id, "chargeback", g.credit_type, g.seconds, key, grant_key, note)
        return self._tx(fn)

    def adjust(self, account_id: str, credit_type: str, seconds: int, key: str, note: str) -> Entry:
        """Operator repair. ``seconds`` may be negative; a note is required (audit)."""
        if not note.strip():
            raise LedgerError("an adjustment needs a note")

        def fn(c):
            return self._existing(c, key, account_id, "adjust") or self._insert(
                c, account_id, "adjust", credit_type, abs(int(seconds)), key, "", note, sign=1 if seconds >= 0 else -1)
        return self._tx(fn)

    # ── audit ──
    def invariants(self, account_id: str) -> List[str]:
        """Problems found (empty = consistent)."""
        c = self._conn()
        bad = []
        for t in CREDIT_TYPES:
            b = self._balance(c, account_id, t)
            if b.reserved < 0:
                bad.append(f"{t}: negative reserved {b.reserved}")
            grants = c.execute("SELECT COALESCE(SUM(seconds*sign),0) FROM ledger WHERE account_id=? AND credit_type=? "
                               "AND kind IN ('grant','adjust')", (account_id, t)).fetchone()[0]
            removed = c.execute("SELECT COALESCE(SUM(seconds),0) FROM ledger WHERE account_id=? AND credit_type=? "
                                "AND kind IN ('refund','chargeback')", (account_id, t)).fetchone()[0]
            if b.available + b.reserved + b.consumed != grants - removed:
                bad.append(f"{t}: seconds not conserved")
        for (ref,) in c.execute("SELECT DISTINCT ref FROM ledger WHERE account_id=? AND kind='reserve'", (account_id,)):
            for t, (r, s, rel) in self._reservation(c, account_id, ref).items():
                if s + rel > r:
                    bad.append(f"{ref}/{t}: settled+released {s + rel} > reserved {r}")
        return bad

    def export(self, account_id: str) -> str:
        """Deterministic dump for byte-for-byte replay comparison (E01)."""
        return json.dumps([[e.kind, e.credit_type, e.seconds * e.sign, e.business_key, e.ref]
                           for e in self.entries(account_id)], separators=(",", ":"))
