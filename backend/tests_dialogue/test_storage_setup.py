"""R01 private storage (fake S3 client) and the launch readiness check."""
import io

import pytest

import storage
from tests_dialogue.test_submission import env, media, grant, upload, rights, wait_closed  # noqa: F401


class FakeS3:
    def __init__(self, fail=False):
        self.objects, self.fail = {}, fail

    def upload_file(self, path, bucket, key):
        if self.fail:
            raise ConnectionError("bucket unreachable")
        self.objects[key] = open(path, "rb").read()

    def generate_presigned_url(self, op, ExpiresIn, Params):
        return f"https://bucket.example/{Params['Key']}?exp={ExpiresIn}"

    def list_objects_v2(self, Bucket, Prefix, **kw):
        return {"Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)], "IsTruncated": False}

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop(o["Key"], None)

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setenv("VOICEDUB_S3_BUCKET", "private-test")
    monkeypatch.setattr(storage, "_client_override", fake)
    return fake


def _funded(env, c, uid, media):
    grant(env, uid, 3_600, "pay:t1")
    did = upload(c, media["ok"]).json()["id"]
    qid = c.post("/api/quotes", json={"draft_id": did}).json()["id"]
    jid = c.post(f"/api/drafts/{did}/submit", json={"quote_id": qid, "idempotency_key": "k1",
                                                    "rights_accepted": True, "rights_version": rights(c)}).json()["id"]
    assert wait_closed(env, jid) == "settled"
    return jid


def test_delivery_is_copied_and_links_redirect_privately(env, media, s3):
    c, uid = env["client"]("a@example.com")
    jid = _funded(env, c, uid, media)
    assert f"{uid}/{jid}/video.mp4" in s3.objects
    link = c.get(f"/api/jobs/{jid}/links").json()["video"]
    r = c.get(link, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith(f"https://bucket.example/{uid}/{jid}/video.mp4")
    assert c.post("/api/account/delete", json={"confirm_email": "a@example.com"}).status_code == 200
    assert not any(k.startswith(f"{uid}/") for k in s3.objects)


def test_storage_failure_keeps_delivery_and_retries_on_download(env, media, s3):
    s3.fail = True
    c, uid = env["client"]("a@example.com")
    jid = _funded(env, c, uid, media)
    b = env["app"].JOBS[jid].billing
    assert b["closed"] == "settled" and "bucket unreachable" in b["store_error"]
    link = c.get(f"/api/jobs/{jid}/links").json()["video"]
    assert c.get(link, follow_redirects=False).status_code == 200      # served from this PC meanwhile
    s3.fail = False
    assert c.get(link, follow_redirects=False).status_code == 302      # retried, now in the bucket
    assert "store_error" not in env["app"].JOBS[jid].billing


def test_storage_self_test(s3):
    assert storage.self_test() is None and not s3.objects


def test_setup_check_reports_missing_and_never_prints_secrets(monkeypatch, capsys):
    import setup_check
    for c in setup_check.CHECKS:
        for k in c.keys:
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(setup_check, "_load_env_file", lambda: None)
    assert setup_check.main([]) == 1
    monkeypatch.setenv("RAZORPAY_KEY_ID", "wrong_prefix")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "s3cr3t-value")
    res = {r["name"]: r for r in setup_check.run()}
    assert res["Razorpay keys"]["status"] == "bad format"
    assert res["Google TTS key"]["status"] == "missing"
    setup_check.main([])
    assert "s3cr3t-value" not in capsys.readouterr().out
    monkeypatch.setenv("VOICEDUB_SELLER_GSTIN", "22AAAAA0000A1Z5")
    assert {r["name"]: r for r in setup_check.run()}["Seller GSTIN"]["status"] == "present"
