"""Resumable uploads: large files arrive in pieces under the tunnel's request
cap; pieces may come out of order or twice; the result is an ordinary draft."""
import submission
from tests_dialogue.test_submission import env, media  # noqa: F401


def _pieces(path, size):
    data = open(path, "rb").read()
    return data, [data[i:i + size] for i in range(0, len(data), size)]


def test_out_of_order_and_retried_chunks_make_the_same_draft(env, media, monkeypatch):
    monkeypatch.setattr(submission, "CHUNK_BYTES", 64 * 1024)
    c, _ = env["client"]("big@example.com")
    data, parts = _pieces(media["ok"], 64 * 1024)
    assert len(parts) >= 3
    u = c.post("/api/uploads", json={"filename": "talk.mp4", "size": len(data)}).json()
    assert u["chunks"] == len(parts) and u["chunk_size"] == 64 * 1024
    order = list(range(len(parts)))[::-1]
    for i in order:
        assert c.put(f"/api/uploads/{u['id']}/chunks/{i}", content=parts[i]).status_code == 200
    assert c.put(f"/api/uploads/{u['id']}/chunks/1", content=parts[1]).status_code == 200     # retry: harmless
    assert c.get(f"/api/uploads/{u['id']}").json()["received"] == list(range(len(parts)))
    d = c.post(f"/api/uploads/{u['id']}/complete").json()
    direct = c.post("/api/drafts", files={"file": ("talk.mp4", data, "video/mp4")}).json()
    assert d["sha256"] == direct["sha256"] and d["size"] == len(data) and d["duration"] > 80
    assert c.post("/api/quotes", json={"draft_id": d["id"]}).status_code == 200            # usable like any draft
    assert c.get(f"/api/uploads/{u['id']}").status_code == 404                              # consumed


def test_missing_short_and_foreign_chunks_are_refused(env, media, monkeypatch):
    monkeypatch.setattr(submission, "CHUNK_BYTES", 64 * 1024)
    a, _ = env["client"]("a@example.com")
    b, _ = env["client"]("b@example.com")
    data, parts = _pieces(media["ok"], 64 * 1024)
    u = a.post("/api/uploads", json={"filename": "talk.mp4", "size": len(data)}).json()
    assert a.put(f"/api/uploads/{u['id']}/chunks/0", content=parts[0][:100]).status_code == 400     # incomplete
    assert a.put(f"/api/uploads/{u['id']}/chunks/999", content=b"x").status_code == 400             # out of range
    assert b.put(f"/api/uploads/{u['id']}/chunks/0", content=parts[0]).status_code == 404           # not B's
    a.put(f"/api/uploads/{u['id']}/chunks/0", content=parts[0])
    r = a.post(f"/api/uploads/{u['id']}/complete")
    assert r.status_code == 409 and "missing" in r.json()["detail"]


def test_limits_type_and_expiry(env, monkeypatch):
    c, _ = env["client"]("lim@example.com")
    assert c.post("/api/uploads", json={"filename": "x.exe", "size": 10}).status_code == 415
    big = submission.MAX_UPLOAD_BYTES + 1
    assert c.post("/api/uploads", json={"filename": "x.mp4", "size": big}).status_code == 413
    u = c.post("/api/uploads", json={"filename": "x.mp4", "size": 1000}).json()
    st = env["store"]
    st._x("UPDATE uploads SET created_at = created_at - ?", (submission.UPLOAD_TTL_S + 1,))
    assert st.expire_uploads() == 1 and not (st.drafts_dir / f"{u['id']}.upload").exists()


def test_account_deletion_removes_unfinished_uploads(env):
    c, uid = env["client"]("del@example.com")
    u = c.post("/api/uploads", json={"filename": "x.mp4", "size": 1000}).json()
    env["store"].purge_owner(uid)
    assert not (env["store"].drafts_dir / f"{u['id']}.upload").exists()
    assert env["store"]._one("SELECT COUNT(*) FROM uploads")[0] == 0
