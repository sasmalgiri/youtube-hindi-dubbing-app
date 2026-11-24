"""T09 drill: back up a live state (paid jobs, payments, ledger, accounts) while
the queue runs, restore into a fresh folder, and get the same service back."""
import threading

import pytest

from ops import backup as B
from tests_dialogue.test_submission import env, media, grant, upload, rights, wait_closed  # noqa: F401


def _submit(c, did, key):
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    return c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": key,
                                                     "rights_accepted": True, "rights_version": rights(c)}).json()["id"]


def test_backup_during_work_restores_identically(env, media, tmp_path_factory):
    app = env["app"]
    users = []
    for i in range(3):
        c, uid = env["client"](f"u{i}@example.com")
        grant(env, uid, 10_000, f"pay:t{i}")
        did = upload(c, media["ok"]).json()["id"]
        jid = _submit(c, did, "k1")
        users.append((c, uid, did, jid))
    for c, uid, did, jid in users:
        wait_closed(env, jid)
    # take the backup while more work is being submitted and run
    env["runner"].gate = threading.Event()
    busy = [_submit(c, did, "k2") for c, uid, did, _ in users]
    dest = tmp_path_factory.mktemp("backups")
    out = B.create(env["tmp"], dest)
    env["runner"].gate.set()
    for j in busy:
        wait_closed(env, j)

    target = tmp_path_factory.mktemp("restored") / "state"
    assert B.restore(out, target) == []
    info = B.describe(target)
    assert set(info["jobs"]) >= {u[3] for u in users}
    assert {info["jobs"][u[3]] for u in users} == {u[1] for u in users}            # owners intact
    assert all(info["balances"][u[1]]["standard"][2] >= 91 for u in users)          # consumed kept
    assert "ledger.db" in info["databases"] and "accounts.db" in info["databases"]

    # the restored ledger is internally consistent
    from billing.ledger import Ledger
    led = Ledger(target / "ledger.db")
    try:
        assert all(led.invariants(u[1]) == [] for u in users)
    finally:
        led.close()
    # the restored job store loads the same jobs with their owners
    from jobstore import JobStore
    jobs = {}
    js = JobStore(target / "jobs.db")
    js.load_all(jobs)
    assert {j.id: j.owner_id for j in jobs.values() if j.id in info["jobs"]} == info["jobs"]


def test_restore_refuses_non_empty_target_and_verify_catches_tampering(env, media, tmp_path):
    c, uid = env["client"]("a@example.com")
    grant(env, uid, 3_600, "pay:t1")
    out = B.create(env["tmp"], tmp_path / "b")
    tgt = tmp_path / "t"
    tgt.mkdir()
    (tgt / "x").write_text("keep me")
    with pytest.raises(SystemExit):
        B.restore(out, tgt)
    clean = tmp_path / "clean"
    assert B.restore(out, clean) == []
    from billing.ledger import Ledger
    led = Ledger(clean / "ledger.db")
    led.grant(uid, "standard", 60, key="pay:tampered")
    led.close()
    diffs = B.verify(out, clean)
    assert any("ledger.db" in d for d in diffs) and "prepaid balances differ" in diffs
    assert B.main(["verify", str(out), "--target", str(clean)]) == 1
