"""W5: Razorpay checkout without network (provider order call replaced; signatures real HMAC)."""
import hashlib
import hmac
import json
import threading

import pytest

from billing import payments as P
from billing.ledger import Ledger
from tests_dialogue.test_submission import env  # noqa: F401

KEY_SECRET, HOOK_SECRET = "test_secret_xyz", "hook_secret_abc"


@pytest.fixture
def pay(env, monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_dummy")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", KEY_SECRET)
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", HOOK_SECRET)
    made = []

    def fake_order(amount, currency, receipt, notes):
        made.append(receipt)
        return {"id": f"order_{receipt}", "amount": amount, "currency": currency}

    st = P.PaymentStore(env["tmp"] / "payments.db", lambda: Ledger(env["tmp"] / "ledger.db"), create_order=fake_order)
    monkeypatch.setattr(env["app"]._payments_store, "_store", st, raising=False)
    env["made"] = made
    yield env
    st.close()


def sig(order_id, payment_id):
    return hmac.new(KEY_SECRET.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()


def hook(c, event, payload, eid=None):
    raw = json.dumps({"event": event, "payload": payload}).encode()
    s = hmac.new(HOOK_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    h = {"x-razorpay-signature": s, "content-type": "application/json"}
    if eid:
        h["x-razorpay-event-id"] = eid
    return c.post("/api/webhooks/razorpay", content=raw, headers=h)


def bal(env, uid, t="standard"):
    led = Ledger(env["tmp"] / "ledger.db")
    try:
        return led.balance(uid, t)
    finally:
        led.close()


def captured(order, pid, amount):
    return {"payment": {"entity": {"id": pid, "order_id": order, "amount": amount}}}


def test_catalog_lists_packs_and_checkout_state(pay):
    c, _ = pay["client"]("a@example.com")
    cat = c.get("/api/catalog").json()
    assert {p["id"] for p in cat["packs"]} == {"std_4h", "prem_4h", "std_1h"}
    assert cat["checkout"] == {"configured": True, "test_mode": True, "simulated": False,
                               "script_url": "https://checkout.razorpay.com/v1/checkout.js"}


def test_not_configured_refuses_cleanly(pay, monkeypatch):
    monkeypatch.delenv("RAZORPAY_KEY_SECRET")
    c, _ = pay["client"]("a@example.com")
    assert c.post("/api/checkout", json={"pack_id": "std_1h"}).status_code == 503


def test_checkout_verify_grants_once_with_receipt(pay):
    c, uid = pay["client"]("a@example.com")
    o = c.post("/api/checkout", json={"pack_id": "std_4h"}).json()
    assert o["amount_minor"] == 40_000 and o["key_id"] == "rzp_test_dummy"
    body = {"razorpay_order_id": o["razorpay_order_id"], "razorpay_payment_id": "pay_1",
            "razorpay_signature": sig(o["razorpay_order_id"], "pay_1")}
    r1 = c.post("/api/checkout/verify", json=body).json()
    r2 = c.post("/api/checkout/verify", json=body).json()
    assert r1["status"] == "paid" and r1["receipt_no"] == r2["receipt_no"] and r1["receipt_no"].startswith("VD/")
    assert bal(pay, uid).spendable == 14_400
    # the webhook for the same payment arrives later: no second grant
    assert hook(c, "payment.captured", captured(o["razorpay_order_id"], "pay_1", 40_000)).status_code == 200
    assert bal(pay, uid).spendable == 14_400
    rec = c.get(f"/api/account/receipts/{o['order_id']}")
    assert rec.status_code == 200 and "₹400" in rec.text and "₹61.02" in rec.text
    assert [x["id"] for x in c.get("/api/account/orders").json()] == [o["order_id"]]
    c2, _ = pay["client"]("b@example.com")
    assert c2.get(f"/api/account/receipts/{o['order_id']}").status_code == 404


def test_forged_signature_and_other_users_order_are_refused(pay):
    c, uid = pay["client"]("a@example.com")
    o = c.post("/api/checkout", json={"pack_id": "std_1h"}).json()
    bad = {"razorpay_order_id": o["razorpay_order_id"], "razorpay_payment_id": "pay_x", "razorpay_signature": "00" * 32}
    assert c.post("/api/checkout/verify", json=bad).status_code == 400
    c2, _ = pay["client"]("b@example.com")
    good = dict(bad, razorpay_signature=sig(o["razorpay_order_id"], "pay_x"))
    assert c2.post("/api/checkout/verify", json=good).status_code == 404
    assert bal(pay, uid).spendable == 0


def test_webhook_first_then_browser_and_duplicate_events(pay):
    c, uid = pay["client"]("a@example.com")
    o = c.post("/api/checkout", json={"pack_id": "prem_4h"}).json()
    ev = captured(o["razorpay_order_id"], "pay_9", 50_000)
    assert hook(c, "order.paid", ev, eid="evt_1").json()["outcome"] == "granted"
    assert hook(c, "order.paid", ev, eid="evt_1").json()["outcome"].startswith("duplicate")
    assert hook(c, "payment.captured", ev, eid="evt_2").json()["outcome"] == "granted"   # idempotent grant
    body = {"razorpay_order_id": o["razorpay_order_id"], "razorpay_payment_id": "pay_9",
            "razorpay_signature": sig(o["razorpay_order_id"], "pay_9")}
    assert c.post("/api/checkout/verify", json=body).json()["status"] == "paid"
    assert bal(pay, uid, "premium").spendable == 14_400


def test_bad_webhook_signature_and_wrong_amount(pay):
    c, uid = pay["client"]("a@example.com")
    o = c.post("/api/checkout", json={"pack_id": "std_1h"}).json()
    raw = json.dumps({"event": "order.paid", "payload": captured(o["razorpay_order_id"], "p", 10_000)}).encode()
    assert c.post("/api/webhooks/razorpay", content=raw, headers={"x-razorpay-signature": "nope"}).status_code == 400
    assert hook(c, "order.paid", captured(o["razorpay_order_id"], "p", 1)).status_code == 409
    assert bal(pay, uid).spendable == 0


def test_concurrent_verify_and_webhook_one_grant(pay):
    c, uid = pay["client"]("a@example.com")
    o = c.post("/api/checkout", json={"pack_id": "std_1h"}).json()
    body = {"razorpay_order_id": o["razorpay_order_id"], "razorpay_payment_id": "pay_r",
            "razorpay_signature": sig(o["razorpay_order_id"], "pay_r")}
    out = []
    ts = [threading.Thread(target=lambda: out.append(c.post("/api/checkout/verify", json=body).status_code)) for _ in range(3)]
    ts += [threading.Thread(target=lambda i=i: out.append(hook(c, "payment.captured", captured(o["razorpay_order_id"], "pay_r", 10_000), eid=f"e{i}").status_code)) for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert all(s == 200 for s in out), out
    assert bal(pay, uid).spendable == 3_600


def test_refund_of_unused_pack_and_dispute_after_use(pay):
    c, uid = pay["client"]("a@example.com")
    o = c.post("/api/checkout", json={"pack_id": "std_1h"}).json()
    c.post("/api/checkout/verify", json={"razorpay_order_id": o["razorpay_order_id"], "razorpay_payment_id": "pay_a",
                                         "razorpay_signature": sig(o["razorpay_order_id"], "pay_a")})
    assert hook(c, "refund.processed", {"refund": {"entity": {"id": "rfnd_1", "payment_id": "pay_a"}}}).json()["outcome"] == "refunded"
    assert bal(pay, uid).spendable == 0
    o2 = c.post("/api/checkout", json={"pack_id": "std_1h"}).json()
    c.post("/api/checkout/verify", json={"razorpay_order_id": o2["razorpay_order_id"], "razorpay_payment_id": "pay_b",
                                         "razorpay_signature": sig(o2["razorpay_order_id"], "pay_b")})
    led = Ledger(pay["tmp"] / "ledger.db")
    led.reserve(uid, "job1", 600)
    led.close()
    assert hook(c, "payment.dispute.created", {"dispute": {"entity": {"id": "disp_1", "payment_id": "pay_b"}}}).json()["outcome"] == "charged back"
    b = bal(pay, uid)
    assert b.spendable == 0 and b.debt == 600
    assert [x["status"] for x in c.get("/api/account/orders").json()] == ["disputed", "refunded"]


def test_financial_year_boundary():
    import calendar
    assert P.financial_year(calendar.timegm((2027, 3, 31, 18, 29, 0)) ) == "2026-27"
    assert P.financial_year(calendar.timegm((2027, 3, 31, 18, 31, 0))) == "2027-28"   # 00:01 IST 1 Apr


def test_reconciliation_clean_then_flags_problems(pay):
    from billing import reconcile
    c, uid = pay["client"]("a@example.com")
    o = c.post("/api/checkout", json={"pack_id": "std_1h"}).json()
    c.post("/api/checkout/verify", json={"razorpay_order_id": o["razorpay_order_id"], "razorpay_payment_id": "pay_ok",
                                         "razorpay_signature": sig(o["razorpay_order_id"], "pay_ok")})
    assert sum(len(v) for v in reconcile.run(pay["tmp"]).values()) == 0
    led = Ledger(pay["tmp"] / "ledger.db")
    led.grant(uid, "standard", 60, key="pay:ghost")                 # credit without a paid order
    led.reserve(uid, "lostjob", 100)                                 # reservation of a job that is gone
    led.close()
    c.post("/api/checkout", json={"pack_id": "std_1h"})              # abandoned order
    import time as _t
    res = reconcile.run(pay["tmp"], now=int(_t.time()) + 2 * 86_400, active_jobs=set())
    assert any("pay:ghost" in x for x in res["orders"])
    assert any("still 'created'" in x for x in res["orders"])
    assert any("lostjob" in x for x in res["reservations"])
    assert reconcile.main(["--state", str(pay["tmp"])]) == 1


def test_simulator_url_only_honoured_with_test_keys(monkeypatch):
    monkeypatch.setenv("RAZORPAY_API_BASE", "http://127.0.0.1:8790/v1")
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_x")
    assert P.simulated() and P.api_base() == "http://127.0.0.1:8790/v1"
    assert P.checkout_script_url() == "http://127.0.0.1:8790/v1/checkout.js"
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_x")
    assert not P.simulated() and P.api_base() == P.API
    assert P.checkout_script_url() == "https://checkout.razorpay.com/v1/checkout.js"
