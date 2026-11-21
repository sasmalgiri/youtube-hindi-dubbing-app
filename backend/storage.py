"""Private object storage for delivered files (tracker R01, S01).

Off unless a bucket is configured; then every delivered video and SRT is
copied to the bucket under ``<owner>/<job>/`` after the delivery check, and
download links redirect to a short-lived presigned URL. The bucket must be
private; nothing here ever makes an object public. Works with any
S3-compatible store (Cloudflare R2, AWS S3, Backblaze B2, MinIO).

Env:
  VOICEDUB_S3_BUCKET        bucket name (turns storage on)
  VOICEDUB_S3_ACCESS_KEY    access key id
  VOICEDUB_S3_SECRET_KEY    secret
  VOICEDUB_S3_ENDPOINT      e.g. https://<account>.r2.cloudflarestorage.com (empty = AWS)
  VOICEDUB_S3_REGION        default "auto"
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

_client_override: Any = None   # tests put a fake client here
PRESIGN_SECONDS = 300


def _env(k: str) -> str:
    return os.environ.get(k, "").strip()


def configured() -> bool:
    return bool(_env("VOICEDUB_S3_BUCKET") and (_client_override is not None or
                                                (_env("VOICEDUB_S3_ACCESS_KEY") and _env("VOICEDUB_S3_SECRET_KEY"))))


def bucket() -> str:
    return _env("VOICEDUB_S3_BUCKET")


def client():
    if _client_override is not None:
        return _client_override
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=_env("VOICEDUB_S3_ENDPOINT") or None,
                        region_name=_env("VOICEDUB_S3_REGION") or "auto",
                        aws_access_key_id=_env("VOICEDUB_S3_ACCESS_KEY"),
                        aws_secret_access_key=_env("VOICEDUB_S3_SECRET_KEY"),
                        config=Config(signature_version="s3v4", retries={"max_attempts": 3}))


def job_prefix(owner: Optional[str], job_id: str) -> str:
    return f"{owner or 'local'}/{job_id}/"


def put_job_files(owner: Optional[str], job_id: str, files: Dict[str, Optional[Path]]) -> Dict[str, str]:
    """Upload {kind: path}; returns {kind: key}. Raises on failure (caller records it)."""
    c, out = client(), {}
    for kind, path in files.items():
        if not path or not Path(path).exists():
            continue
        key = job_prefix(owner, job_id) + f"{kind}{Path(path).suffix.lower()}"
        c.upload_file(str(path), bucket(), key)
        out[kind] = key
    return out


def presigned(key: str, filename: str) -> str:
    return client().generate_presigned_url(
        "get_object", ExpiresIn=PRESIGN_SECONDS,
        Params={"Bucket": bucket(), "Key": key,
                "ResponseContentDisposition": f'attachment; filename="{filename}"'})


def delete_prefix(prefix: str) -> int:
    c, n = client(), 0
    token = None
    while True:
        kw = {"Bucket": bucket(), "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        page = c.list_objects_v2(**kw)
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            c.delete_objects(Bucket=bucket(), Delete={"Objects": keys})
            n += len(keys)
        if not page.get("IsTruncated"):
            return n
        token = page.get("NextContinuationToken")


def self_test() -> Optional[str]:
    """Write, read back and delete one small object. None when it works."""
    c, key = client(), "_voicedub_setup_check/probe.txt"
    c.put_object(Bucket=bucket(), Key=key, Body=b"ok")
    body = c.get_object(Bucket=bucket(), Key=key)["Body"].read()
    c.delete_object(Bucket=bucket(), Key=key)
    return None if body == b"ok" else "read back a different body"
