"""W1 accounts and ownership (tracker S01, P09, T04, T11 "A/B isolation").

No pipeline runs here: jobs are inserted directly into the store, so the
tests exercise sign-in, sessions and the ownership gate only.
"""
import importlib

import pytest

import accounts as A


class Clock:
    def __init__(self, t=1_800_000_000):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def outbox():
    return []


@pytest.fixture
def store(tmp_path, clock, outbox):
    s = A.AccountStore(tmp_path / "accounts.db", clock=clock, mailer=lambda to, code: outbox.append((to, code)))
    yield s
    s.close()


def sign_in(store, outbox, email):
    store.start_email(email)
    return store.verify_email(email, outbox[-1][1])


# ── email codes ──
def test_email_code_signs_in_and_creates_one_user(store, outbox):
    t1 = sign_in(store, outbox, "Creator@Example.com")
    s = store.session(t1)
    assert s and s.user.email == "creator@example.com" and s.method == "email"
    outbox.clear()
    store._conn.execute("UPDATE email_codes SET created_at = created_at - 120")
    t2 = sign_in(store, outbox, "creator@example.com")
    assert store.session(t2).user.id == s.user.id


def test_code_is_single_use_and_hashed(store, outbox):
    store.start_email("a@example.com")
    code = outbox[-1][1]
    store.verify_email("a@example.com", code)
    with pytest.raises(A.AuthError):
        store.verify_email("a@example.com", code)
    stored = store._conn.execute("SELECT code_hash FROM email_codes").fetchone()[0]
    assert code not in stored


def test_five_wrong_codes_burn_the_code(store, outbox):
    store.start_email("a@example.com")
    good = outbox[-1][1]
    wrong = "000000" if good != "000000" else "111111"
    for _ in range(A.CODE_MAX_ATTEMPTS):
        with pytest.raises(A.AuthError):
            store.verify_email("a@example.com", wrong)
    with pytest.raises(A.AuthError):
        store.verify_email("a@example.com", good)


def test_code_expires(store, outbox, clock):
    store.start_email("a@example.com")
    clock.t += A.CODE_TTL + 1
    with pytest.raises(A.AuthError):
        store.verify_email("a@example.com", outbox[-1][1])


def test_resend_throttle_and_hourly_cap(store, outbox, clock):
    store.start_email("a@example.com")
    with pytest.raises(A.AuthError) as e:
        store.start_email("a@example.com")
    assert e.value.status == 429
    for _ in range(A.CODE_MAX_PER_HOUR - 1):
        clock.t += A.CODE_RESEND_GAP
        store.start_email("a@example.com")
    clock.t += A.CODE_RESEND_GAP
    with pytest.raises(A.AuthError, match="hour"):
        store.start_email("a@example.com")


def test_bad_email_and_no_mailer(tmp_path):
    s = A.AccountStore(tmp_path / "x.db", mailer=None)
    with pytest.raises(A.AuthError, match="valid email"):
        s.start_email("not-an-email")
    with pytest.raises(A.AuthError) as e:
        s.start_email("a@example.com")
    assert e.value.status == 503


# ── Google ──
def _google(info):
    return lambda credential, client_id: info


def test_google_sign_in_links_to_the_same_email_account(store, outbox):
    t_email = sign_in(store, outbox, "g@example.com")
    info = {"aud": "cid", "iss": "https://accounts.google.com", "sub": "123", "email": "g@example.com",
            "email_verified": True, "name": "G"}
    t_google = store.verify_google("tok", "cid", verifier=_google(info))
    assert store.session(t_google).user.id == store.session(t_email).user.id
    assert store.session(t_google).method == "google"


@pytest.mark.parametrize("bad", [
    {"aud": "other", "iss": "accounts.google.com", "sub": "1", "email": "x@example.com", "email_verified": True},
    {"aud": "cid", "iss": "evil.example", "sub": "1", "email": "x@example.com", "email_verified": True},
    {"aud": "cid", "iss": "accounts.google.com", "sub": "1", "email": "x@example.com", "email_verified": False},
])
def test_google_rejects_wrong_audience_issuer_or_unverified_email(store, bad):
    with pytest.raises(A.AuthError):
        store.verify_google("tok", "cid", verifier=_google(bad))


def test_google_not_configured(store):
    with pytest.raises(A.AuthError) as e:
        store.verify_google("tok", "", verifier=_google({}))
    assert e.value.status == 503


def test_google_verifier_exception_is_a_plain_failure(store):
    def boom(c, cid):
        raise ValueError("Token expired, 1700000000 < 1800000000")
    with pytest.raises(A.AuthError, match="Google sign-in failed"):
        store.verify_google("tok", "cid", verifier=boom)


# ── sessions ──
def test_sessions_expire_slide_and_revoke(store, outbox, clock):
    tok = sign_in(store, outbox, "s@example.com")
    uid = store.session(tok).user.id
    clock.t += A.SESSION_TTL - 10
    assert store.session(tok)                       # still valid, and slides
    clock.t += A.SESSION_TTL - 10
    assert store.session(tok)
    clock.t += A.SESSION_TTL + 1
    assert store.session(tok) is None               # expired after inactivity
    t1 = store.create_session(store.get_user(uid))
    t2 = store.create_session(store.get_user(uid))
    assert len(store.sessions(uid)) == 2
    sid1 = store.session(t1).id
    assert store.revoke(uid, sid1) == 1 and store.session(t1) is None and store.session(t2)
    assert store.revoke(uid) == 1 and store.session(t2) is None


def test_session_token_is_stored_hashed(store, outbox):
    tok = sign_in(store, outbox, "h@example.com")
    assert tok not in str(store._conn.execute("SELECT * FROM sessions").fetchall())


def test_deleted_account_cannot_sign_in_with_old_session(store, outbox):
    tok = sign_in(store, outbox, "d@example.com")
    store.delete_account_request(store.session(tok).user.id)
    assert store.session(tok) is None


# ── API: ownership gate ──
@pytest.fixture
def api(tmp_path, monkeypatch, outbox):
    monkeypatch.setenv("VOICEDUB_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("VOICEDUB_PROFILE", "hosted")
    import features
    importlib.reload(features)
    import app as app_module
    from jobstore import JobStore
    monkeypatch.setattr(app_module, "_features", features)
    acc = A.AccountStore(tmp_path / "accounts.db", mailer=lambda to, code: outbox.append((to, code)))
    monkeypatch.setattr(app_module, "_accounts", acc)
    js = JobStore(tmp_path / "jobs.db")
    monkeypatch.setattr(app_module, "_store", js)
    jobs = {}
    monkeypatch.setattr(app_module, "JOBS", jobs)
    from fastapi.testclient import TestClient
    yield app_module, TestClient(app_module.app, base_url="https://testserver"), js, jobs, features
    acc.close()


def _client_signed_in(app_module, outbox, email):
    from fastapi.testclient import TestClient
    c = TestClient(app_module.app, base_url="https://testserver")
    assert c.post("/api/auth/email/start", json={"email": email}).status_code == 200
    r = c.post("/api/auth/email/verify", json={"email": email, "code": outbox[-1][1]})
    assert r.status_code == 200, r.text
    return c, r.json()["user"]["id"]


def _add_job(app_module, js, jobs, job_id, owner):
    j = app_module.Job(id=job_id, source_url=f"upload:{job_id}.mp4", target_language="hi", state="done")
    j.owner_id = owner
    jobs[job_id] = j
    js.save(j)
    return j


def test_hosted_needs_sign_in_for_every_job_route(api):
    app_module, anon, js, jobs, _ = api
    _add_job(app_module, js, jobs, "aaa111", "u_x")
    for path in ("/api/jobs", "/api/jobs/aaa111", "/api/jobs/aaa111/events", "/api/jobs/aaa111/result",
                 "/static/jobs/aaa111/result.mp4"):
        assert anon.get(path).status_code == 401, path
    assert anon.post("/api/jobs", json={"url": "x"}).status_code == 401


def test_two_accounts_see_disjoint_jobs_even_after_a_cold_restart(api, outbox, tmp_path):
    app_module, _, js, jobs, _ = api
    ca, ua = _client_signed_in(app_module, outbox, "a@example.com")
    cb, ub = _client_signed_in(app_module, outbox, "b@example.com")
    _add_job(app_module, js, jobs, "jobA01", ua)
    _add_job(app_module, js, jobs, "jobB01", ub)

    def check():
        assert [j["id"] for j in ca.get("/api/jobs").json()] == ["jobA01"]
        assert [j["id"] for j in cb.get("/api/jobs").json()] == ["jobB01"]
        assert ca.get("/api/jobs/jobA01").status_code == 200
        for path in ("/api/jobs/jobA01", "/api/jobs/jobA01/events", "/api/jobs/jobA01/result",
                     "/api/jobs/jobA01/srt", "/api/jobs/jobA01/report", "/api/jobs/jobA01/description"):
            assert cb.get(path).status_code == 404, path          # 404, never 403: ids cannot be probed
        assert cb.delete("/api/jobs/jobA01").status_code == 404
        assert "jobA01" in jobs

    check()
    # cold restart: a new store reads the same database into an empty dict
    from jobstore import JobStore
    jobs.clear()
    JobStore(tmp_path / "jobs.db").load_all(jobs)
    assert jobs["jobA01"].owner_id == ua and jobs["jobB01"].owner_id == ub
    check()


def test_sign_out_everywhere_ends_every_session(api, outbox):
    app_module, _, js, jobs, _ = api
    c1, uid = _client_signed_in(app_module, outbox, "m@example.com")
    from fastapi.testclient import TestClient
    c2 = TestClient(app_module.app, base_url="https://testserver")
    app_module._accounts._conn.execute("UPDATE email_codes SET created_at = created_at - 120")
    c2.post("/api/auth/email/start", json={"email": "m@example.com"})
    c2.post("/api/auth/email/verify", json={"email": "m@example.com", "code": outbox[-1][1]})
    assert len(c1.get("/api/auth/sessions").json()["sessions"]) == 2
    assert c1.post("/api/auth/sessions/revoke-all").json() == {"revoked": 2}
    assert c2.get("/api/auth/me").json() == {"user": None}
    assert c2.get("/api/jobs").status_code == 401


def test_session_cookie_is_httponly_and_secure_in_hosted(api, outbox):
    app_module, c, *_ = api
    c.post("/api/auth/email/start", json={"email": "c@example.com"})
    r = c.post("/api/auth/email/verify", json={"email": "c@example.com", "code": outbox[-1][1]})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "secure" in cookie and "samesite=lax" in cookie


def test_local_profile_keeps_the_desktop_app_working_without_sign_in(api, monkeypatch, outbox):
    app_module, anon, js, jobs, features = api
    monkeypatch.setenv("VOICEDUB_PROFILE", "local")
    _add_job(app_module, js, jobs, "legacy1", None)               # made before accounts existed
    assert [j["id"] for j in anon.get("/api/jobs").json()] == ["legacy1"]
    assert anon.get("/api/jobs/legacy1").status_code == 200
    # a job assigned to someone is hidden from a different signed-in account
    cb, ub = _client_signed_in(app_module, outbox, "other@example.com")
    _add_job(app_module, js, jobs, "owned1", "u_someone_else")
    assert cb.get("/api/jobs/owned1").status_code == 404
    assert {j["id"] for j in cb.get("/api/jobs").json()} == {"legacy1"}


def test_balance_endpoint_reads_the_ledger(api, outbox, tmp_path):
    app_module, *_ = api
    c, uid = _client_signed_in(app_module, outbox, "bal@example.com")
    from billing.ledger import Ledger
    led = Ledger(tmp_path / "ledger.db")
    led.grant(uid, "standard", 14_400, key="pay:test1")
    led.close()
    monkey_state = app_module.STATE_DIR
    try:
        app_module.STATE_DIR = tmp_path
        b = c.get("/api/account/balance").json()
    finally:
        app_module.STATE_DIR = monkey_state
    assert b["standard"]["spendable_seconds"] == 14_400 and b["premium"]["spendable_seconds"] == 0


def test_legacy_assignment_cli_dry_run_then_apply(tmp_path, monkeypatch, outbox, capsys):
    monkeypatch.setenv("VOICEDUB_STATE_DIR", str(tmp_path))
    s = A.AccountStore(tmp_path / "accounts.db", mailer=lambda to, code: outbox.append((to, code)))
    sign_in(s, outbox, "owner@example.com")
    s.close()
    import app as app_module
    from jobstore import JobStore
    js = JobStore(tmp_path / "jobs.db")
    for jid in ("old1", "old2"):
        js.save(app_module.Job(id=jid, source_url="x", target_language="hi", state="done"))
    assert A._cli(["assign-legacy", "--email", "owner@example.com"]) == 0
    assert "2 unassigned" in capsys.readouterr().out
    jobs = {}
    JobStore(tmp_path / "jobs.db").load_all(jobs)
    assert all(j.owner_id is None for j in jobs.values())           # dry run changed nothing
    assert A._cli(["assign-legacy", "--email", "owner@example.com", "--apply"]) == 0
    jobs = {}
    JobStore(tmp_path / "jobs.db").load_all(jobs)
    assert all(j.owner_id and j.owner_id.startswith("u_") for j in jobs.values())
