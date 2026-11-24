"""Launch readiness: which owner-supplied settings are present, and (with
--live) whether each one actually works. Never prints a secret.

Usage (from backend/):
    python setup_check.py            # presence + format only, no network
    python setup_check.py --live     # also calls each provider read-only
    python setup_check.py --json

Live calls are read-only and free: Razorpay lists one order, Google lists
the Hindi voices, SMTP logs in without sending, Sarvam is not called (it
bills per call), storage writes and deletes one tiny object.
Exit code 0 when everything needed for the hosted pilot is ready.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent


def _load_env_file() -> None:
    f = HERE / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env(k: str) -> str:
    return os.environ.get(k, "").strip()


class Check:
    def __init__(self, name: str, unlocks: str, required: bool, keys: List[str],
                 fmt: Optional[Callable[[], Optional[str]]] = None, live: Optional[Callable[[], Optional[str]]] = None):
        self.name, self.unlocks, self.required, self.keys = name, unlocks, required, keys
        self.fmt, self.live = fmt, live

    def run(self, live: bool) -> Dict:
        missing = [k for k in self.keys if not env(k)]
        out = {"name": self.name, "unlocks": self.unlocks, "required": self.required, "missing": missing}
        if missing:
            out["status"] = "missing"
            return out
        problem = self.fmt() if self.fmt else None
        if problem:
            out.update(status="bad format", detail=problem)
            return out
        if live and self.live:
            try:
                problem = self.live()
            except Exception as e:   # network errors: say what kind, never the key
                problem = f"{type(e).__name__}: {str(e)[:120]}"
            out.update(status="failed" if problem else "works", detail=problem or "")
            return out
        out["status"] = "present"
        return out


# ── provider checks ──
def _rzp_fmt():
    kid = env("RAZORPAY_KEY_ID")
    if not kid.startswith(("rzp_test_", "rzp_live_")):
        return "RAZORPAY_KEY_ID should start with rzp_test_ or rzp_live_"
    return None


def _rzp_live():
    import requests
    from billing.payments import api_base
    r = requests.get(f"{api_base()}/orders", params={"count": 1},
                     auth=(env("RAZORPAY_KEY_ID"), env("RAZORPAY_KEY_SECRET")), timeout=20)
    if r.status_code == 401:
        return "Razorpay refused the key id / secret (HTTP 401)"
    return None if r.status_code == 200 else f"Razorpay HTTP {r.status_code}"


def _chirp_live():
    import requests
    r = requests.get("https://texttospeech.googleapis.com/v1/voices", params={"languageCode": "hi-IN", "key": env("GOOGLE_TTS_API_KEY")},
                     timeout=20)
    if r.status_code != 200:
        return f"Google TTS HTTP {r.status_code} (is the Text-to-Speech API enabled for this key?)"
    names = {v.get("name") for v in r.json().get("voices", [])}
    want = env("GOOGLE_TTS_VOICE_HI") or "hi-IN-Chirp3-HD-Kore"
    return None if want in names else f"voice {want} not offered to this key"


def _oauth_fmt():
    return None if env("GOOGLE_OAUTH_CLIENT_ID").endswith(".apps.googleusercontent.com") \
        else "should end with .apps.googleusercontent.com"


def _smtp_live():
    import smtplib
    port = int(env("SMTP_PORT") or 587)
    cls = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
    with cls(env("SMTP_HOST"), port, timeout=20) as s:
        if port != 465:
            s.starttls()
        if env("SMTP_USER"):
            s.login(env("SMTP_USER"), env("SMTP_PASS"))
    return None


def _gstin_fmt():
    import re
    g = env("VOICEDUB_SELLER_GSTIN")
    return None if re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]", g) else "GSTIN should be 15 characters like 22AAAAA0000A1Z5"


def _storage_live():
    import storage
    return storage.self_test()


CHECKS = [
    Check("Razorpay keys", "Buying prepaid time (checkout, receipts)", True,
          ["RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET"], _rzp_fmt, _rzp_live),
    Check("Razorpay webhook secret", "Payments confirmed even if the browser closes; refunds and disputes", True,
          ["RAZORPAY_WEBHOOK_SECRET"]),
    Check("Google TTS key", "Chirp 3 HD Hindi voice (Standard)", True, ["GOOGLE_TTS_API_KEY"], None, _chirp_live),
    Check("Sarvam key", "Fallback voice when Google fails on a line", False, ["SARVAM_API_KEY"]),
    Check("Google sign-in", "Sign in with Google", False, ["GOOGLE_OAUTH_CLIENT_ID"], _oauth_fmt),
    Check("Email (SMTP)", "Email sign-in codes reach real inboxes", True, ["SMTP_HOST", "SMTP_FROM"], None, _smtp_live),
    Check("Seller name and address", "Name and address printed on receipts", True,
          ["VOICEDUB_SELLER_NAME", "VOICEDUB_SELLER_ADDRESS"]),
    Check("Seller GSTIN", "GST number on receipts", False, ["VOICEDUB_SELLER_GSTIN"], _gstin_fmt),
    Check("Allowed web origin", "Browser access from your public site (Cloudflare Tunnel URL)", True,
          ["VOICEDUB_CORS_ORIGINS"]),
    Check("Private storage", "Finished files kept in a private bucket (otherwise this PC's disk)", False,
          ["VOICEDUB_S3_BUCKET", "VOICEDUB_S3_ACCESS_KEY", "VOICEDUB_S3_SECRET_KEY"], None, _storage_live),
]


def run(live: bool = False) -> List[Dict]:
    return [c.run(live) for c in CHECKS]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Launch readiness check")
    ap.add_argument("--live", action="store_true", help="call each provider (read-only)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    _load_env_file()
    res = run(a.live)
    blocking = [r for r in res if r["required"] and r["status"] not in ("present", "works")]
    if a.json:
        print(json.dumps({"ready": not blocking, "checks": res}, indent=2))
    else:
        mark = {"works": "OK  ", "present": "set ", "missing": "--  ", "bad format": "BAD ", "failed": "FAIL"}
        for r in res:
            extra = f" ({', '.join(r['missing'])})" if r["missing"] else (f" — {r['detail']}" if r.get("detail") else "")
            print(f"[{mark[r['status']]}] {r['name']}{'' if r['required'] else ' (optional)'}: {r['unlocks']}{extra}")
        print(f"\n{'READY for the hosted pilot' if not blocking else f'{len(blocking)} required item(s) not ready'}"
              f"{'' if a.live else ' (format only; run with --live to test each key)'}")
    return 0 if not blocking else 1


if __name__ == "__main__":
    sys.exit(main())
