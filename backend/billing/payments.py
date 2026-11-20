"""Prepaid-time checkout through Razorpay (W5: M03, M05, M07, T02).

Money flow, all idempotent:

  POST /api/checkout          -> local order row + Razorpay order (amount from the
                                 server catalog, never from the client)
  browser Razorpay Checkout   -> {order_id, payment_id, signature}
  POST /api/checkout/verify   -> HMAC check, then ``fulfil``
  POST /api/webhooks/razorpay -> HMAC check of the raw body, then ``fulfil``
                                 (payment.captured / order.paid), refunds,
                                 disputes

``fulfil`` grants the pack once: the ledger key is ``pay:<payment_id>``, the
order row moves to ``paid`` under a conditional UPDATE, and the receipt
number is allocated in the same transaction. A browser verify and a webhook
for the same payment, in any order or both twice, give one grant and one
receipt. Keys come from the environment only; without them checkout reports
"not configured" and nothing else changes.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import catalog

API = "https://api.razorpay.com/v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id            TEXT PRIMARY KEY,           -- our order id
    owner_id      TEXT NOT NULL,
    pack_id       TEXT NOT NULL,
    credit_type   TEXT NOT NULL,
    seconds       INTEGER NOT NULL,
    amount_minor  INTEGER NOT NULL,
    tax_minor     INTEGER NOT NULL,
    currency      TEXT NOT NULL,
    price_version TEXT NOT NULL,
    rzp_order_id  TEXT UNIQUE,
    payment_id    TEXT UNIQUE,
    status        TEXT NOT NULL,              -- created | paid | refunded | disputed | failed
    receipt_no    TEXT UNIQUE,
    created_at    INTEGER NOT NULL,
    paid_at       INTEGER,
    note          TEXT
);
CREATE TABLE IF NOT EXISTS receipt_seq (fy TEXT PRIMARY KEY, n INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS webhook_events (
    id          TEXT PRIMARY KEY,             -- x-razorpay-event-id (or body hash)
    event       TEXT NOT NULL,
    received_at INTEGER NOT NULL,
    outcome     TEXT
);
"""


class PaymentError(Exception):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def keys() -> Dict[str, str]:
    return {"key_id": os.environ.get("RAZORPAY_KEY_ID", "").strip(),
            "key_secret": os.environ.get("RAZORPAY_KEY_SECRET", "").strip(),
            "webhook_secret": os.environ.get("RAZORPAY_WEBHOOK_SECRET", "").strip()}


def configured() -> bool:
    k = keys()
    return bool(k["key_id"] and k["key_secret"])


def test_mode() -> bool:
    return keys()["key_id"].startswith("rzp_test_")


def payment_signature_ok(order_id: str, payment_id: str, signature: str, secret: str) -> bool:
    if not (order_id and payment_id and signature and secret):
        return False
    want = hmac.new(secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(want, signature)


def webhook_signature_ok(raw_body: bytes, signature: str, secret: str) -> bool:
    if not (signature and secret):
        return False
    want = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(want, signature)


def _http_create_order(amount_minor: int, currency: str, receipt: str, notes: Dict[str, str]) -> Dict[str, Any]:
    import requests
    k = keys()
    r = requests.post(f"{API}/orders", auth=(k["key_id"], k["key_secret"]), timeout=20,
                      json={"amount": amount_minor, "currency": currency, "receipt": receipt, "notes": notes})
    if r.status_code >= 400:
        raise PaymentError(f"Razorpay refused the order (HTTP {r.status_code})", 502)
    return r.json()


def financial_year(ts: int) -> str:
    t = time.gmtime(ts + 19_800)                   # IST
    start = t.tm_year if t.tm_mon >= 4 else t.tm_year - 1
    return f"{start}-{str(start + 1)[-2:]}"


class PaymentStore:
    def __init__(self, path: Path, ledger_factory: Callable[[], Any],
                 create_order: Callable[..., Dict[str, Any]] = _http_create_order,
                 clock: Callable[[], float] = time.time):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self._ledger = ledger_factory
        self._create_order = create_order
        self._clock = clock

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _now(self) -> int:
        return int(self._clock())

    def _row(self, sql: str, args=()) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, args)
            r = cur.fetchone()
            return dict(zip([c[0] for c in cur.description], r)) if r else None

    # ── checkout ──
    def create(self, owner_id: str, pack_id: str) -> Dict[str, Any]:
        if not configured():
            raise PaymentError("Checkout is not configured yet", 503)
        try:
            p = catalog.pack(pack_id)
        except KeyError:
            raise PaymentError("Unknown pack", 404)
        oid = "o_" + secrets.token_hex(8)
        tax = catalog.tax_inclusive_share(p.price_minor)
        now = self._now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO orders(id, owner_id, pack_id, credit_type, seconds, amount_minor, tax_minor, currency, "
                "price_version, status, created_at) VALUES (?,?,?,?,?,?,?,?,?, 'created', ?)",
                (oid, owner_id, p.id, p.credit_type, p.seconds, p.price_minor, tax, p.currency,
                 catalog.PRICE_VERSION, now))
        try:
            rz = self._create_order(p.price_minor, p.currency, oid, {"order": oid, "pack": p.id})
        except PaymentError:
            self._mark(oid, "failed", "order creation failed")
            raise
        except Exception as e:
            self._mark(oid, "failed", f"order creation failed: {type(e).__name__}")
            raise PaymentError("Could not reach the payment provider; try again", 502)
        if int(rz.get("amount", -1)) != p.price_minor:
            self._mark(oid, "failed", "provider amount mismatch")
            raise PaymentError("Payment provider returned a different amount", 502)
        with self._lock:
            self._conn.execute("UPDATE orders SET rzp_order_id=? WHERE id=?", (rz["id"], oid))
        return {"order_id": oid, "razorpay_order_id": rz["id"], "key_id": keys()["key_id"],
                "amount_minor": p.price_minor, "currency": p.currency, "pack_id": p.id,
                "display": {"amount": catalog.fmt_money(p.price_minor), "tax": catalog.fmt_money(tax),
                            "time": catalog.fmt_seconds(p.seconds)}, "test_mode": test_mode()}

    def _mark(self, oid: str, status: str, note: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE orders SET status=?, note=? WHERE id=? AND status='created'", (status, note, oid))

    def verify(self, owner_id: str, rzp_order_id: str, payment_id: str, signature: str) -> Dict[str, Any]:
        if not payment_signature_ok(rzp_order_id, payment_id, signature, keys()["key_secret"]):
            raise PaymentError("Payment could not be verified", 400)
        o = self._row("SELECT * FROM orders WHERE rzp_order_id=?", (rzp_order_id,))
        if o is None or o["owner_id"] != owner_id:
            raise PaymentError("Order not found", 404)
        return self.fulfil(rzp_order_id, payment_id, int(o["amount_minor"]), source="checkout")

    def fulfil(self, rzp_order_id: str, payment_id: str, amount_minor: int, source: str) -> Dict[str, Any]:
        """Grant the pack once and issue its receipt. Safe to call any number of times."""
        o = self._row("SELECT * FROM orders WHERE rzp_order_id=?", (rzp_order_id,))
        if o is None:
            raise PaymentError("Order not found", 404)
        if int(amount_minor) != int(o["amount_minor"]):
            raise PaymentError("Paid amount does not match the order", 409)
        if o["payment_id"] and o["payment_id"] != payment_id:
            raise PaymentError("Order already paid by another payment", 409)
        led = self._ledger()
        try:
            led.grant(o["owner_id"], o["credit_type"], int(o["seconds"]), key=f"pay:{payment_id}",
                      ref=o["id"], note=f"{o['pack_id']} via {source}")
        finally:
            led.close()
        now = self._now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute("SELECT status, receipt_no FROM orders WHERE id=?", (o["id"],)).fetchone()
                if cur[0] in ("created", "failed"):
                    fy = financial_year(now)
                    self._conn.execute("INSERT OR IGNORE INTO receipt_seq(fy, n) VALUES (?, 0)", (fy,))
                    self._conn.execute("UPDATE receipt_seq SET n = n + 1 WHERE fy=?", (fy,))
                    n = self._conn.execute("SELECT n FROM receipt_seq WHERE fy=?", (fy,)).fetchone()[0]
                    rno = f"VD/{fy}/{n:06d}"
                    self._conn.execute("UPDATE orders SET status='paid', payment_id=?, receipt_no=?, paid_at=? "
                                       "WHERE id=?", (payment_id, rno, now, o["id"]))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return self.order(o["id"], o["owner_id"])

    # ── webhooks ──
    def webhook(self, raw_body: bytes, signature: str, event_id: str = "") -> str:
        if not webhook_signature_ok(raw_body, signature, keys()["webhook_secret"]):
            raise PaymentError("Bad webhook signature", 400)
        body = json.loads(raw_body.decode("utf-8"))
        ev = body.get("event", "")
        eid = event_id or hashlib.sha256(raw_body).hexdigest()
        with self._lock:
            seen = self._conn.execute("SELECT outcome FROM webhook_events WHERE id=?", (eid,)).fetchone()
        if seen:
            return f"duplicate ({seen[0]})"
        pl = body.get("payload", {})
        pay = (pl.get("payment") or {}).get("entity") or {}
        outcome = "ignored"
        if ev in ("payment.captured", "order.paid") and pay.get("order_id"):
            self.fulfil(pay["order_id"], pay["id"], int(pay.get("amount", -1)), source=f"webhook {ev}")
            outcome = "granted"
        elif ev == "refund.processed":
            ref = (pl.get("refund") or {}).get("entity") or {}
            outcome = self._refunded(ref.get("payment_id", ""), ref.get("id", ""))
        elif ev.startswith("payment.dispute.") and ev in ("payment.dispute.created", "payment.dispute.lost"):
            dis = (pl.get("dispute") or {}).get("entity") or {}
            outcome = self._disputed(dis.get("payment_id") or pay.get("id", ""), dis.get("id", eid))
        elif ev == "payment.failed" and pay.get("order_id"):
            with self._lock:
                self._conn.execute("UPDATE orders SET note=? WHERE rzp_order_id=? AND status='created'",
                                   (f"payment failed: {pay.get('error_description') or ''}"[:200], pay["order_id"]))
            outcome = "noted failure"
        with self._lock:
            self._conn.execute("INSERT OR IGNORE INTO webhook_events(id, event, received_at, outcome) VALUES (?,?,?,?)",
                               (eid, ev, self._now(), outcome))
        return outcome

    def _refunded(self, payment_id: str, refund_id: str) -> str:
        o = self._row("SELECT * FROM orders WHERE payment_id=?", (payment_id,))
        if o is None:
            return "refund for unknown payment"
        led = self._ledger()
        try:
            led.refund_unused(o["owner_id"], f"pay:{payment_id}", key=f"refund:{refund_id}", note="Razorpay refund")
        except Exception as e:
            # time already used: take it back as debt via chargeback semantics
            led.chargeback(o["owner_id"], f"pay:{payment_id}", key=f"refund:{refund_id}",
                           note=f"refund after use ({type(e).__name__})")
        finally:
            led.close()
        with self._lock:
            self._conn.execute("UPDATE orders SET status='refunded' WHERE id=?", (o["id"],))
        return "refunded"

    def _disputed(self, payment_id: str, dispute_id: str) -> str:
        o = self._row("SELECT * FROM orders WHERE payment_id=?", (payment_id,))
        if o is None:
            return "dispute for unknown payment"
        led = self._ledger()
        try:
            led.chargeback(o["owner_id"], f"pay:{payment_id}", key=f"dispute:{dispute_id}", note="Razorpay dispute")
        finally:
            led.close()
        with self._lock:
            self._conn.execute("UPDATE orders SET status='disputed' WHERE id=?", (o["id"],))
        return "charged back"

    # ── reads ──
    def order(self, oid: str, owner_id: str) -> Dict[str, Any]:
        o = self._row("SELECT * FROM orders WHERE id=?", (oid,))
        if o is None or o["owner_id"] != owner_id:
            raise PaymentError("Order not found", 404)
        return self._public(o)

    def orders(self, owner_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM orders WHERE owner_id=? AND status!='created' "
                                     "AND status!='failed' ORDER BY created_at DESC, rowid DESC", (owner_id,))
            cols = [c[0] for c in cur.description]
            return [self._public(dict(zip(cols, r))) for r in cur.fetchall()]

    @staticmethod
    def _public(o: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": o["id"], "pack_id": o["pack_id"], "credit_type": o["credit_type"], "seconds": o["seconds"],
                "amount_minor": o["amount_minor"], "tax_minor": o["tax_minor"], "currency": o["currency"],
                "status": o["status"], "receipt_no": o["receipt_no"], "paid_at": o["paid_at"],
                "payment_id": o["payment_id"],
                "display": {"amount": catalog.fmt_money(o["amount_minor"]), "tax": catalog.fmt_money(o["tax_minor"]),
                            "net": catalog.fmt_money(o["amount_minor"] - o["tax_minor"]),
                            "time": catalog.fmt_seconds(o["seconds"])}}


def receipt_html(o: Dict[str, Any], buyer_email: str) -> str:
    import html
    seller = html.escape(os.environ.get("VOICEDUB_SELLER_NAME", "VoiceDub"))
    gstin = html.escape(os.environ.get("VOICEDUB_SELLER_GSTIN", ""))
    addr = html.escape(os.environ.get("VOICEDUB_SELLER_ADDRESS", ""))
    when = time.strftime("%d %b %Y %H:%M IST", time.gmtime((o["paid_at"] or 0) + 19_800))
    d = o["display"]
    kind = "Premium" if o["credit_type"] == "premium" else "Standard"
    status = "" if o["status"] == "paid" else f"<p class=s>Status: {html.escape(o['status'])}</p>"
    return f"""<!doctype html><html><head><meta charset=utf-8><title>Receipt {html.escape(o['receipt_no'] or '')}</title>
<style>body{{font:14px system-ui,sans-serif;max-width:640px;margin:32px auto;padding:0 16px;color:#111}}
table{{width:100%;border-collapse:collapse}}td{{padding:6px 0;border-bottom:1px solid #ddd}}td:last-child{{text-align:right}}
.s{{color:#b45309}}@media print{{button{{display:none}}}}</style></head><body>
<h1>Receipt</h1><p><strong>{seller}</strong><br>{addr}{'<br>GSTIN ' + gstin if gstin else ''}</p>
<p>Receipt no. <strong>{html.escape(o['receipt_no'] or '')}</strong><br>Date {when}<br>Billed to {html.escape(buyer_email)}<br>
Payment ref {html.escape(o['payment_id'] or '')}</p>{status}
<table><tr><td>{kind} prepaid dubbing time, {d['time']}</td><td>{d['net']}</td></tr>
<tr><td>GST 18% (included)</td><td>{d['tax']}</td></tr>
<tr><td><strong>Total paid</strong></td><td><strong>{d['amount']}</strong></td></tr></table>
{('<p>' + html.escape(os.environ.get('VOICEDUB_RECEIPT_NOTE', '')) + '</p>') if os.environ.get('VOICEDUB_RECEIPT_NOTE') else ''}
<button onclick="print()">Print / save as PDF</button></body></html>"""
