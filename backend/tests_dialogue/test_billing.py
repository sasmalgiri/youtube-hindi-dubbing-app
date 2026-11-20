"""Catalog, quotes and ledger (tracker M01, M02, M04, M05, M06, T02, E01 ledger part).

Pure library tests: no payment provider, no accounts. The figures come from the
release instructions §4 and the first end-to-end acceptance run.
"""
import random
import sqlite3
import threading

import pytest

from billing import catalog
from billing.catalog import PREMIUM, STANDARD, NotPriced
from billing.ledger import InsufficientBalance, Ledger, LedgerError
from billing.quote import QuoteRefused, check_still_valid, clip_interval, make_quote

NOW = 1_800_000_000
YEAR = 365 * 24 * 3600


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def led(tmp_path, clock):
    lg = Ledger(tmp_path / "ledger.db", clock=clock)
    yield lg
    lg.close()


def buy(led, acct, pack_id, payment_id):
    p = catalog.pack(pack_id)
    return led.grant(acct, p.credit_type, p.seconds, key=f"pay:{payment_id}", ref=payment_id)


# ── catalog / arithmetic (§4 "Known arithmetic") ──
def test_catalog_packs():
    assert (catalog.pack("std_4h").seconds, catalog.pack("std_4h").price_minor) == (14_400, 40_000)
    assert (catalog.pack("prem_4h").seconds, catalog.pack("prem_4h").price_minor) == (14_400, 50_000)
    assert (catalog.pack("std_1h").seconds, catalog.pack("std_1h").price_minor) == (3_600, 10_000)


def test_fifteen_minutes_standard_one_language_is_25_rupees_before_credits():
    q = make_quote(account_id="A", source_id="s1", source_duration=900, languages=["hi"], now=NOW)
    assert q.cash_minor == 2_500 and catalog.fmt_money(q.cash_minor) == "₹25"
    assert q.standard_seconds == 0 and q.cash_seconds == 900


def test_thirty_minutes_into_two_languages_consumes_3600_base_seconds():
    q = make_quote(account_id="A", source_id="s1", source_duration=1800, languages=["hi", "bn"],
                   available={STANDARD: 14_400}, now=NOW)
    assert q.standard_seconds == 3_600 and q.cash_minor == 0
    assert [l.billable_seconds for l in q.lines] == [1800, 1800]


def test_400_buyer_using_two_and_a_half_hours_has_5400_a_year_later_then_9000(led, clock):
    buy(led, "A", "std_4h", "p1")
    led.reserve("A", "job1", 9_000)
    led.settle("A", "job1")
    clock.t += YEAR
    assert led.spendable("A")[STANDARD] == 5_400
    buy(led, "A", "std_1h", "p2")
    assert led.spendable("A")[STANDARD] == 9_000


def test_lines_round_once_and_reconcile_with_total():
    q = make_quote(account_id="A", source_id="s", source_duration=601.2, languages=["hi", "bn", "ta"], now=NOW)
    assert q.cash_minor == sum(l.cash_minor for l in q.lines)
    assert all(l.cash_minor == catalog.cash_minor(l.cash_seconds) for l in q.lines)


def test_money_and_time_are_integers_in_their_own_units():
    q = make_quote(account_id="A", source_id="s", source_duration=61.4, languages=["hi"], now=NOW)
    for v in (q.cash_minor, q.tax_minor, q.cash_seconds, q.lines[0].billable_seconds):
        assert isinstance(v, int)
    assert q.currency == "INR"


def test_gst_share_of_a_tax_inclusive_amount():
    assert catalog.tax_inclusive_share(11_800) == 1_800
    assert catalog.tax_inclusive_share(40_000) == 6_102


def test_unapproved_style_or_quality_is_not_priced():
    with pytest.raises(NotPriced):
        make_quote(account_id="A", source_id="s", source_duration=60, languages=["hi"], style="narration", now=NOW)
    with pytest.raises(NotPriced):
        make_quote(account_id="A", source_id="s", source_duration=60, languages=["hi"], quality="highest", now=NOW)


# ── quote binding (§4 "Quote", P02) ──
def test_interval_clipping_fractional_and_shorter_source():
    assert clip_interval(600.4, None, None) == [0, 601]           # fractional second billed once
    assert clip_interval(300.0, 0, 600) == [0, 300]               # shorter than requested
    assert clip_interval(600.0, 10.7, 70.2) == [10, 71]
    with pytest.raises(QuoteRefused):
        clip_interval(300, 400, None)
    with pytest.raises(QuoteRefused):
        clip_interval(0, None, None)


def test_single_job_uses_standard_first_then_premium_then_cash():
    q = make_quote(account_id="A", source_id="s", source_duration=600, languages=["hi"],
                   available={STANDARD: 200, PREMIUM: 300}, now=NOW)
    assert (q.standard_seconds, q.premium_seconds, q.cash_seconds) == (200, 300, 100)
    assert q.cash_minor == catalog.cash_minor(100) and q.max_cash_minor == q.cash_minor


def test_batch_uses_premium_only_and_refuses_when_unfunded():
    q = make_quote(account_id="A", source_id="s", source_duration=600, languages=["hi"], batch=True,
                   available={STANDARD: 99_999, PREMIUM: 600}, now=NOW)
    assert (q.standard_seconds, q.premium_seconds, q.cash_seconds) == (0, 600, 0)
    with pytest.raises(QuoteRefused, match="Premium"):
        make_quote(account_id="A", source_id="s", source_duration=600, languages=["hi"], batch=True,
                   available={STANDARD: 99_999, PREMIUM: 599}, now=NOW)


def test_changed_draft_or_expiry_or_price_version_invalidates_the_quote(monkeypatch):
    kw = dict(account_id="A", source_id="s", interval=[0, 600], languages=["hi"], style="dialogue",
              quality="standard", batch=False)
    q = make_quote(account_id="A", source_id="s", source_duration=600, languages=["hi"], now=NOW)
    check_still_valid(q, now=NOW + 60, **kw)
    with pytest.raises(QuoteRefused, match="draft changed"):
        check_still_valid(q, now=NOW + 60, **{**kw, "languages": ["hi", "bn"]})
    with pytest.raises(QuoteRefused, match="draft changed"):
        check_still_valid(q, now=NOW + 60, **{**kw, "source_id": "s-v2"})
    with pytest.raises(QuoteRefused, match="expired"):
        check_still_valid(q, now=q.expires_at, **kw)
    monkeypatch.setattr(catalog, "PRICE_VERSION", "inr-2027-v2")
    with pytest.raises(QuoteRefused, match="prices changed"):
        check_still_valid(q, now=NOW + 60, **kw)
    q2 = make_quote(account_id="A", source_id="s", source_duration=600, languages=["hi", "bn"], now=NOW)
    assert q2.binding != q.binding


def test_duplicate_or_empty_language_list_refused():
    with pytest.raises(QuoteRefused):
        make_quote(account_id="A", source_id="s", source_duration=60, languages=["hi", "hi"], now=NOW)
    with pytest.raises(QuoteRefused):
        make_quote(account_id="A", source_id="s", source_duration=60, languages=[" "], now=NOW)


# ── ledger ──
def test_e01_ledger_figures(led):
    """₹400 → 10 min → 13,800; ₹500 Premium, two 10-min children, one fails →
    600 settle and 13,800 remain; replayed events change nothing; the rerun
    consumes the restored 600 once."""
    buy(led, "A", "std_4h", "pay_1")
    led.reserve("A", "job_10min", 600)
    led.settle("A", "job_10min")
    assert led.spendable("A")[STANDARD] == 13_800

    buy(led, "A", "prem_4h", "pay_2")
    for child in ("b1:c1", "b1:c2"):
        assert led.reserve("A", child, 600, order=(PREMIUM,)) == {STANDARD: 0, PREMIUM: 600}
    led.settle("A", "b1:c1")
    led.release("A", "b1:c2", note="failed before delivery")
    b = led.balances("A")
    assert b[PREMIUM].consumed == 600 and b[PREMIUM].spendable == 13_800
    assert b[STANDARD].spendable == 13_800

    before = led.export("A")
    buy(led, "A", "prem_4h", "pay_2")             # duplicate payment notification
    led.settle("A", "b1:c1")                      # duplicate completion
    led.release("A", "b1:c2")                     # duplicate failure
    assert led.export("A") == before

    led.reserve("A", "b1:c2:retry1", 600, order=(PREMIUM,))
    led.settle("A", "b1:c2:retry1")
    led.settle("A", "b1:c2:retry1")
    assert led.balances("A")[PREMIUM].spendable == 13_200
    assert led.invariants("A") == []


def test_reserve_is_all_or_nothing_and_idempotent(led):
    buy(led, "A", "std_1h", "p")
    with pytest.raises(InsufficientBalance):
        led.reserve("A", "big", 3_601)
    assert led.spendable("A")[STANDARD] == 3_600 and led.entries("A", ref="big") == []
    assert led.reserve("A", "j", 600) == led.reserve("A", "j", 600) == {STANDARD: 600, PREMIUM: 0}
    assert led.spendable("A")[STANDARD] == 3_000


def test_single_job_reservation_spans_types_and_release_restores_each(led):
    buy(led, "A", "std_1h", "p1")
    buy(led, "A", "prem_4h", "p2")
    assert led.reserve("A", "j", 4_000) == {STANDARD: 3_600, PREMIUM: 400}
    led.release("A", "j")
    assert led.spendable("A") == {STANDARD: 3_600, PREMIUM: 14_400}


def test_shorter_delivery_settles_part_and_releases_the_rest(led):
    buy(led, "A", "std_1h", "p")
    led.reserve("A", "j", 600)
    assert led.settle("A", "j", seconds=450) == {STANDARD: 450, PREMIUM: 0}
    assert led.balances("A")[STANDARD].spendable == 3_150
    with pytest.raises(LedgerError):
        led.reserve("A", "x", 1) and led.settle("A", "x", seconds=5)


def test_cancel_racing_completion_has_one_effect(led):
    buy(led, "A", "std_1h", "p")
    led.reserve("A", "j", 600)
    led.settle("A", "j")
    assert led.release("A", "j") == {STANDARD: 0, PREMIUM: 0}         # cancel lost the race: no-op
    led.reserve("A", "k", 600)
    led.release("A", "k")
    with pytest.raises(LedgerError):                                   # completion lost the race: refused
        led.settle("A", "k")
    assert led.balances("A")[STANDARD].consumed == 600 and led.invariants("A") == []


def test_two_workers_compete_for_the_final_minute(tmp_path):
    db = tmp_path / "l.db"
    Ledger(db).grant("A", STANDARD, 60, key="pay:p")
    wins, losses = [], []
    barrier = threading.Barrier(8)

    def worker(i):
        lg = Ledger(db)
        barrier.wait()
        try:
            lg.reserve("A", f"job{i}", 60)
            wins.append(i)
        except InsufficientBalance:
            losses.append(i)
        finally:
            lg.close()

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(wins) == 1 and len(losses) == 7
    assert Ledger(db).spendable("A")[STANDARD] == 0


def test_chargeback_during_a_running_batch_creates_no_spendable_credit(led):
    buy(led, "A", "prem_4h", "p")
    led.reserve("A", "c1", 10_000, order=(PREMIUM,))
    led.chargeback("A", "pay:p", key="cb:p")
    b = led.balance("A", PREMIUM)
    assert b.spendable == 0 and b.debt == 10_000
    with pytest.raises(InsufficientBalance):
        led.reserve("A", "c2", 1, order=(PREMIUM,))
    led.release("A", "c1")                                  # the child fails: pays the debt down
    b = led.balance("A", PREMIUM)
    assert b.spendable == 0 and b.debt == 0
    led.chargeback("A", "pay:p", key="cb:p")                # duplicate dispute event
    assert led.balance("A", PREMIUM).debt == 0


def test_refund_of_an_untouched_pack_and_refusal_after_use(led):
    buy(led, "A", "std_1h", "p1")
    led.refund_unused("A", "pay:p1", key="rf:p1")
    led.refund_unused("A", "pay:p1", key="rf:p1")
    assert led.spendable("A")[STANDARD] == 0
    buy(led, "A", "std_1h", "p2")
    led.reserve("A", "j", 1)
    with pytest.raises(InsufficientBalance, match="partial-consumption"):
        led.refund_unused("A", "pay:p2", key="rf:p2")


def test_business_key_cannot_be_reused_for_another_account(led):
    buy(led, "A", "std_1h", "p")
    with pytest.raises(LedgerError):
        led.grant("B", STANDARD, 3_600, key="pay:p")


def test_entries_are_immutable(led, tmp_path):
    buy(led, "A", "std_1h", "p")
    c = sqlite3.connect(str(tmp_path / "ledger.db"))
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        c.execute("UPDATE ledger SET seconds = 99999")
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        c.execute("DELETE FROM ledger")
    c.close()


def test_adjustment_needs_a_note(led):
    with pytest.raises(LedgerError):
        led.adjust("A", STANDARD, 60, key="adj:1", note=" ")
    led.adjust("A", STANDARD, -60, key="adj:2", note="duplicate grant repair #123")
    assert led.balance("A", STANDARD).debt == 60


@pytest.mark.parametrize("seed", range(25))
def test_random_operation_sequences_conserve_seconds(tmp_path, seed):
    """Property test: any sequence of grants, reservations, settlements,
    releases, retries and refunds keeps every invariant, and each operation
    applied twice has the effect of once."""
    rnd = random.Random(seed)
    lg = Ledger(tmp_path / f"p{seed}.db")
    accounts = ["A", "B"]
    jobs, grants = {a: [] for a in accounts}, {a: [] for a in accounts}
    for step in range(120):
        a = rnd.choice(accounts)
        op = rnd.choice(["grant", "reserve", "reserve", "settle", "release", "refund", "retry"])
        try:
            if op == "grant":
                p = rnd.choice(list(catalog.PACKS.values()))
                k = f"pay:{a}:{step}"
                lg.grant(a, p.credit_type, p.seconds, key=k); grants[a].append(k)
            elif op == "reserve":
                j = f"{a}-job{step}"
                lg.reserve(a, j, rnd.randint(1, 5_000), order=rnd.choice([(STANDARD, PREMIUM), (PREMIUM,)]))
                jobs[a].append(j)
            elif op == "settle" and jobs[a]:
                j = rnd.choice(jobs[a])
                lg.settle(a, j, seconds=None if rnd.random() < 0.7 else rnd.randint(0, 50))
            elif op == "release" and jobs[a]:
                lg.release(a, rnd.choice(jobs[a]))
            elif op == "refund" and grants[a]:
                g = rnd.choice(grants[a]); lg.refund_unused(a, g, key=f"rf:{g}")
            elif op == "retry" and jobs[a]:
                j = rnd.choice(jobs[a])
                lg.release(a, j); after1 = lg.export(a); lg.release(a, j)
                assert lg.export(a) == after1
        except (InsufficientBalance, LedgerError):
            pass
        for acct in accounts:
            assert lg.invariants(acct) == []
            for b in lg.balances(acct).values():
                assert b.spendable >= 0 and b.reserved >= 0
    lg.close()
