"""Owner overview and alerts: thresholds fire, repeats are suppressed, email
is used only when configured, and the admin route needs the token."""
import json
import time

from ops import health
from tests_dialogue.test_submission import env  # noqa: F401


def _snap(tmp_path, **kw):
    s = health.snapshot(tmp_path, backups=tmp_path / "bk", **kw)
    return s


def test_no_backup_and_failures_raise_alerts(tmp_path, monkeypatch):
    now = time.time()
    jobs = [{"id": f"j{i}", "state": "error", "created_at": now - 60, "error": "boom"} for i in range(5)]
    s = _snap(tmp_path, jobs=jobs, now=now)
    keys = {a["key"] for a in health.alerts(s)}
    assert {"backup", "failures"} <= keys
    assert s["jobs_24h"]["failed"] == 5 and s["jobs_24h"]["recent_failures"][0]["error"] == "boom"


def test_fresh_backup_and_quiet_day_is_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "DISK_MIN_GB", 0)
    b = tmp_path / "bk" / "backup_x"
    b.mkdir(parents=True)
    (b / "backup.json").write_text(json.dumps({"created_at": int(time.time()) - 3600}))
    s = _snap(tmp_path)
    assert s["backup"]["age_h"] == 1.0 and health.alerts(s) == []
    assert health.alerts(s, {"orders": ["o1: paid but no grant"]})[0]["key"] == "reconcile"


def test_stuck_queue(tmp_path):
    import workqueue
    q = workqueue.WorkQueue(tmp_path / "queue.db", lambda j: None)
    q.enqueue("j1", "a")
    q._x("UPDATE work SET enqueued_at=? WHERE job_id='j1'", (int(time.time()) - 7200,))
    q.close()
    s = _snap(tmp_path)
    assert s["queue"]["by_state"] == {"queued": 1}
    assert any(a["key"] == "queue" for a in health.alerts(s))


def test_sink_dedupes_and_emails_only_when_configured(tmp_path, monkeypatch):
    t = [1000.0]
    sent = []
    monkeypatch.setenv("VOICEDUB_ALERT_EMAIL", "owner@example.com")
    sink = health.AlertSink(tmp_path, clock=lambda: t[0], send=lambda to, sub, body: sent.append((to, sub)))
    a = [{"key": "disk", "level": "critical", "text": "Disk nearly full: 3 GB free"}]
    assert len(sink.emit(a)) == 1 and len(sink.emit(a)) == 0
    t[0] += health.REPEAT_S + 1
    assert len(sink.emit(a)) == 1 and len(sent) == 2
    assert sink.recent()[0]["emailed"] is True
    monkeypatch.delenv("VOICEDUB_ALERT_EMAIL")
    quiet = health.AlertSink(tmp_path / "q")
    assert quiet.emit(a)[0]["emailed"] is False


def test_admin_overview_needs_token(env, monkeypatch):
    c, _ = env["client"]("a@example.com")
    assert c.get("/api/admin/overview").status_code == 403
    monkeypatch.setenv("VOICEDUB_ADMIN_TOKEN", "t0k")
    r = c.get("/api/admin/overview", headers={"x-admin-token": "t0k"})
    assert r.status_code == 200
    body = r.json()
    assert {"snapshot", "reconcile", "alerts", "recent_alerts"} <= set(body)
