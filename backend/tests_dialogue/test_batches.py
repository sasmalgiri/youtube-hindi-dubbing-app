"""W8: Premium batches as server records (S04, M06, T11 'failed batch child at final Premium second')."""
import threading

from tests_dialogue.test_submission import balance, env, media, rights, upload, wait_closed  # noqa: F401


def grant_premium(env, uid, seconds, key):
    from billing.ledger import Ledger
    led = Ledger(env["tmp"] / "ledger.db")
    led.grant(uid, "premium", seconds, key=key)
    led.close()


def prem(env, uid):
    from billing.ledger import Ledger
    led = Ledger(env["tmp"] / "ledger.db")
    try:
        return led.balance(uid, "premium")
    finally:
        led.close()


def grant_standard(env, uid, seconds, key):
    from billing.ledger import Ledger
    led = Ledger(env["tmp"] / "ledger.db")
    led.grant(uid, "standard", seconds, key=key)
    led.close()


def _drafts(c, media, n):
    return [upload(c, media["ok"]).json()["id"] for _ in range(n)]


def _start(c, bid, key="bk1"):
    return c.post(f"/api/batches/{bid}/submit", json={"idempotency_key": key, "rights_accepted": True,
                                                      "rights_version": rights(c)})


def test_itemized_quote_and_full_run_settles_each_child(env, media):
    c, uid = env["client"]("a@example.com")
    grant_premium(env, uid, 3_600, "pay:p1")
    ids = _drafts(c, media, 2)
    b = c.post("/api/batches", json={"draft_ids": ids}).json()
    assert [k["premium_seconds"] for k in b["children"]] == [91, 91] and b["premium_seconds"] == 182
    assert _start(c, b["id"]).json()["duplicate"] is False
    assert _start(c, b["id"]).json()["duplicate"] is True          # double click: one batch start
    v = c.get(f"/api/batches/{b['id']}").json()
    for k in v["children"]:
        assert wait_closed(env, k["job_id"]) == "settled"
    p = prem(env, uid)
    assert (p.spendable, p.consumed, p.reserved) == (3_600 - 182, 182, 0)
    assert c.get("/api/batches").json()[0]["id"] == b["id"]


def test_standard_only_balance_cannot_fund_a_batch(env, media):
    c, uid = env["client"]("a@example.com")
    grant_standard(env, uid, 36_000, "pay:s1")
    ids = _drafts(c, media, 1)
    r = c.post("/api/batches", json={"draft_ids": ids})
    assert r.status_code == 422 and "Premium" in r.json()["detail"]


def test_last_premium_second_is_all_or_nothing(env, media):
    c, uid = env["client"]("a@example.com")
    grant_premium(env, uid, 182, "pay:p1")                          # exactly two children
    ids = _drafts(c, media, 3)
    r = c.post("/api/batches", json={"draft_ids": ids})
    assert r.status_code == 422 and "Item 3" in r.json()["detail"]
    b = c.post("/api/batches", json={"draft_ids": ids[:2]}).json()
    # balance drops after the quote (another batch spends a second) -> the start reserves nothing
    from billing.ledger import Ledger
    led = Ledger(env["tmp"] / "ledger.db")
    led.reserve(uid, "other", 1, order=("premium",))
    led.close()
    r = _start(c, b["id"])
    assert r.status_code == 409
    assert prem(env, uid).reserved == 1                             # only the other job's second
    assert c.get(f"/api/batches/{b['id']}").json()["status"] == "quoted"


def test_mixed_result_returns_only_the_failed_part_and_retries_it(env, media):
    c, uid = env["client"]("a@example.com")
    grant_premium(env, uid, 3_600, "pay:p1")
    ids = _drafts(c, media, 2)
    runner = env["runner"]
    outcomes = iter(["done", "error"])
    orig = runner.__call__.__func__

    class Mixed:
        def __call__(self, job, req):
            runner.outcome = next(outcomes, "done")
            orig(runner, job, req)

    env["app"]._job_runner = lambda: Mixed()
    env["queue"].max_global = 1
    b = c.post("/api/batches", json={"draft_ids": ids}).json()
    _start(c, b["id"])
    kids = c.get(f"/api/batches/{b['id']}").json()["children"]
    assert sorted(wait_closed(env, k["job_id"]) for k in kids) == ["released", "settled"]
    assert prem(env, uid).spendable == 3_600 - 91
    r = c.post(f"/api/batches/{b['id']}/retry-failed", json={"idempotency_key": "r1", "rights_accepted": True,
                                                               "rights_version": rights(c)})
    assert r.status_code == 200 and r.json()["retried"] == 1
    v = c.get(f"/api/batches/{b['id']}").json()
    retried = [k for k in v["children"] if k["attempts"]]
    assert len(retried) == 1 and wait_closed(env, retried[0]["job_id"]) == "settled"
    assert prem(env, uid).spendable == 3_600 - 182


def test_batches_are_private_and_survive_sign_out(env, media):
    c, uid = env["client"]("a@example.com")
    grant_premium(env, uid, 3_600, "pay:p1")
    b = c.post("/api/batches", json={"draft_ids": _drafts(c, media, 1)}).json()
    c2, _ = env["client"]("b@example.com")
    assert c2.get(f"/api/batches/{b['id']}").status_code == 404
    assert c2.get("/api/batches").json() == []
    c.post("/api/auth/logout")
    assert c.get(f"/api/batches/{b['id']}").status_code == 401
    c3, _ = env["client"]("a@example.com")                          # signs in again
    assert c3.get(f"/api/batches/{b['id']}").json()["status"] == "quoted"
