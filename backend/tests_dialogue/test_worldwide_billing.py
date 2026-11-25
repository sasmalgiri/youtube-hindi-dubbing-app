"""Worldwide selling: currency by billing country, proposed price lists stay
off until enabled, tax rule per country on orders and receipts."""
import sqlite3

from billing import catalog, tax
from billing import payments as P
from tests_dialogue.test_payments import pay, sig, bal, hook, captured  # noqa: F401
from tests_dialogue.test_submission import env  # noqa: F401


def test_currency_for_country_and_enabled_lists(monkeypatch):
    assert [catalog.currency_for_country(c) for c in ("IN", "US", "DE", "GB", "BR", None)] == \
        ["INR", "USD", "EUR", "GBP", "USD", "INR"]
    monkeypatch.delenv("VOICEDUB_CURRENCIES", raising=False)
    assert catalog.enabled_currencies() == ["INR"] and catalog.sale_currency("US") is None
    monkeypatch.setenv("VOICEDUB_CURRENCIES", "INR,USD,XYZ")
    assert catalog.enabled_currencies() == ["INR", "USD"] and catalog.sale_currency("US") == "USD"
    assert catalog.fmt_money(1_600, "USD") == "$16" and catalog.fmt_money(450, "EUR") == "€4.50"
    assert tax.rule("IN")["rate"] == (18, 100) and tax.tax_minor(40_000, tax.rule("IN")) == 6_102
    assert tax.rule("DE")["needs_accountant"] and tax.tax_minor(1_500, tax.rule("DE")) == 0
    assert "EU VAT" in tax.rule("DE")["note"]


def test_us_buyer_not_on_sale_until_usd_enabled(pay, monkeypatch):
    monkeypatch.delenv("VOICEDUB_CURRENCIES", raising=False)
    c, uid = pay["client"]("us@example.com")
    assert c.put("/api/account/billing-country", json={"country": "us"}).json() == \
        {"country": "US", "currency": None, "on_sale": False}
    cat = c.get("/api/catalog").json()
    assert cat["country"] == "US" and cat["on_sale"] is False and "price_minor" not in cat["packs"][0]
    assert c.post("/api/checkout", json={"pack_id": "std_1h"}).status_code == 409
    assert c.put("/api/account/billing-country", json={"country": "USA"}).status_code == 400


def test_usd_order_has_no_gst_and_says_so(pay, monkeypatch):
    monkeypatch.setenv("VOICEDUB_CURRENCIES", "INR,USD")
    c, uid = pay["client"]("us@example.com")
    c.put("/api/account/billing-country", json={"country": "US"})
    cat = c.get("/api/catalog").json()
    assert cat["currency"] == "USD" and {p["id"]: p["display"]["price"] for p in cat["packs"]}["std_4h"] == "$16"
    o = c.post("/api/checkout", json={"pack_id": "std_4h"}).json()
    assert (o["amount_minor"], o["currency"], o["country"]) == (1_600, "USD", "US")
    body = {"razorpay_order_id": o["razorpay_order_id"], "razorpay_payment_id": "pay_us",
            "razorpay_signature": sig(o["razorpay_order_id"], "pay_us")}
    assert c.post("/api/checkout/verify", json=body).json()["status"] == "paid"
    assert bal(pay, uid).spendable == 14_400                      # same seconds as the INR pack
    rec = c.get(f"/api/account/receipts/{o['order_id']}").text
    assert "$16" in rec and "No GST: export of services" in rec and "GST 18%" not in rec
    # an Indian buyer still pays INR with GST included
    c2, _ = pay["client"]("in@example.com")
    c2.put("/api/account/billing-country", json={"country": "IN"})
    o2 = c2.post("/api/checkout", json={"pack_id": "std_4h"}).json()
    assert (o2["amount_minor"], o2["currency"], o2["display"]["tax"]) == (40_000, "INR", "₹61.02")
    # preview another country without changing the account
    assert c2.get("/api/catalog?country=DE").json()["on_sale"] is False


def test_old_payments_db_gains_new_columns(tmp_path):
    db = tmp_path / "payments.db"
    old = P._SCHEMA
    c = sqlite3.connect(str(db))
    c.executescript(old)
    c.execute("INSERT INTO orders(id, owner_id, pack_id, credit_type, seconds, amount_minor, tax_minor, currency, "
              "price_version, status, created_at) VALUES ('o_old','u','std_1h','standard',3600,10000,1525,'INR','v','paid',1)")
    c.commit()
    c.close()
    from billing.ledger import Ledger
    st = P.PaymentStore(db, lambda: Ledger(tmp_path / "ledger.db"), create_order=lambda *a: {})
    try:
        o = st.order("o_old", "u")
        assert o["country"] == "IN" and o["display"]["tax_label"] == "GST 18% (included)"
    finally:
        st.close()


import pytest as _pytest


@_pytest.fixture(autouse=True)
def _hindi_signed_off(monkeypatch):
    """Checkout opens only once a language is signed off; these tests are about payments."""
    import languages
    monkeypatch.setattr(languages, "sellable", lambda code: code == "hi")
