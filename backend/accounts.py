"""Accounts, sessions and sign-in (W1; tracker S01, P09, T04).

Sign-in methods (owner decision 2026-11-20): **Google sign-in**, with an
**email one-time code** as the fallback. No passwords are stored.

* Sessions: a random 256-bit token in an HttpOnly cookie; only its SHA-256 is
  stored, so a database leak does not leak live sessions. 30-day sliding
  expiry, revocable one by one or all at once ("sign out everywhere").
* Email codes: 6 digits, stored hashed, 10-minute expiry, 5 wrong tries burn
  the code, at most 5 codes per address per hour and one per 60 s.
* Google: the ID token is verified against GOOGLE_OAUTH_CLIENT_ID (signature,
  audience, expiry) and must carry a verified email.

Profiles (features.py): in **hosted**, every job route needs a session and a
job is visible only to its owner. In **local** (the desktop app), signing in is
optional; jobs made before accounts existed stay *unassigned* and private to
this PC until the owner assigns them (``python -m accounts assign-legacy``).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import smtplib
import sqlite3
import threading
import time
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Dict, List, Optional

SESSION_COOKIE = "vd_session"
SESSION_TTL = 30 * 24 * 3600
CODE_TTL = 10 * 60
CODE_MAX_ATTEMPTS = 5
CODE_RESEND_GAP = 60
CODE_MAX_PER_HOUR = 5
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,24}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,
    google_sub  TEXT UNIQUE,
    name        TEXT NOT NULL DEFAULT '',
    created_at  INTEGER NOT NULL,
    deleted_at  INTEGER
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    id          TEXT NOT NULL UNIQUE,
    user_id     TEXT NOT NULL REFERENCES users(id),
    created_at  INTEGER NOT NULL,
    last_seen   INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL,
    revoked_at  INTEGER,
    user_agent  TEXT NOT NULL DEFAULT '',
    method      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS email_codes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT NOT NULL,
    code_hash   TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    used_at     INTEGER
);
CREATE INDEX IF NOT EXISTS email_codes_email ON email_codes(email, created_at);
"""


class AuthError(Exception):
    """Shown to the user as is (no internal detail)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class User:
    id: str
    email: str
    name: str
    created_at: int


@dataclass(frozen=True)
class Session:
    id: str
    user: User
    created_at: int
    last_seen: int
    expires_at: int
    user_agent: str
    method: str


def normalise_email(email: str) -> str:
    e = (email or "").strip().lower()
    if not EMAIL_RE.match(e):
        raise AuthError("Enter a valid email address")
    return e


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── mail ─────────────────────────────────────────────────────────────────
def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_FROM"))


def send_code_email(to: str, code: str) -> None:
    """SMTP when configured (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM)."""
    msg = EmailMessage()
    msg["Subject"] = f"Your sign-in code: {code}"
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = to
    msg.set_content(f"Your sign-in code is {code}. It expires in 10 minutes.\n\n"
                    f"If you did not ask for this code, ignore this email.")
    port = int(os.environ.get("SMTP_PORT") or 587)
    with smtplib.SMTP(os.environ["SMTP_HOST"], port, timeout=20) as s:
        if port != 25:
            s.starttls()
        if os.environ.get("SMTP_USER"):
            s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASS", ""))
        s.send_message(msg)


# ── store ────────────────────────────────────────────────────────────────
class AccountStore:
    def __init__(self, db_path: Path, clock: Callable[[], float] = time.time,
                 mailer: Optional[Callable[[str, str], None]] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._mailer = mailer
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        # keyed hash for codes: a stolen DB alone cannot brute-force 6 digits offline
        self._pepper = os.environ.get("VOICEDUB_AUTH_PEPPER") or self._local_pepper()

    def _local_pepper(self) -> str:
        f = self.db_path.with_suffix(".pepper")
        try:
            return f.read_text(encoding="utf-8").strip()
        except OSError:
            p = secrets.token_hex(32)
            f.write_text(p, encoding="utf-8")
            return p

    def _now(self) -> int:
        return int(self._clock())

    def close(self) -> None:
        self._conn.close()

    # users
    def _user_row(self, row) -> Optional[User]:
        return User(id=row[0], email=row[1], name=row[3] or "", created_at=row[4]) if row else None

    def get_user(self, user_id: str) -> Optional[User]:
        r = self._conn.execute("SELECT * FROM users WHERE id=? AND deleted_at IS NULL", (user_id,)).fetchone()
        return self._user_row(r)

    def user_by_email(self, email: str) -> Optional[User]:
        r = self._conn.execute("SELECT * FROM users WHERE email=? AND deleted_at IS NULL", (email,)).fetchone()
        return self._user_row(r)

    def _upsert_user(self, email: str, name: str = "", google_sub: Optional[str] = None) -> User:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                r = None
                if google_sub:
                    r = self._conn.execute("SELECT * FROM users WHERE google_sub=? AND deleted_at IS NULL", (google_sub,)).fetchone()
                if r is None:
                    r = self._conn.execute("SELECT * FROM users WHERE email=? AND deleted_at IS NULL", (email,)).fetchone()
                if r is None:
                    uid = "u_" + secrets.token_hex(8)
                    self._conn.execute("INSERT INTO users(id, email, google_sub, name, created_at) VALUES (?,?,?,?,?)",
                                       (uid, email, google_sub, name, self._now()))
                    r = self._conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
                elif google_sub and not r[2]:
                    self._conn.execute("UPDATE users SET google_sub=? WHERE id=?", (google_sub, r[0]))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return self._user_row(r)

    # sessions
    def create_session(self, user: User, user_agent: str = "", method: str = "") -> str:
        token = secrets.token_urlsafe(32)
        now = self._now()
        self._conn.execute(
            "INSERT INTO sessions(token_hash, id, user_id, created_at, last_seen, expires_at, user_agent, method) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (_sha(token), "s_" + secrets.token_hex(6), user.id, now, now, now + SESSION_TTL, user_agent[:200], method))
        return token

    def session(self, token: Optional[str]) -> Optional[Session]:
        if not token or len(token) > 200:
            return None
        r = self._conn.execute(
            "SELECT s.id, s.user_id, s.created_at, s.last_seen, s.expires_at, s.revoked_at, s.user_agent, s.method "
            "FROM sessions s WHERE s.token_hash=?", (_sha(token),)).fetchone()
        now = self._now()
        if r is None or r[5] is not None or r[4] <= now:
            return None
        user = self.get_user(r[1])
        if user is None:
            return None
        if now - r[3] > 300:          # sliding expiry, written at most every 5 minutes
            self._conn.execute("UPDATE sessions SET last_seen=?, expires_at=? WHERE id=?", (now, now + SESSION_TTL, r[0]))
        return Session(id=r[0], user=user, created_at=r[2], last_seen=now, expires_at=now + SESSION_TTL,
                       user_agent=r[6], method=r[7])

    def sessions(self, user_id: str) -> List[Dict]:
        now = self._now()
        rows = self._conn.execute(
            "SELECT id, created_at, last_seen, expires_at, user_agent, method FROM sessions "
            "WHERE user_id=? AND revoked_at IS NULL AND expires_at>? ORDER BY last_seen DESC", (user_id, now)).fetchall()
        return [{"id": r[0], "created_at": r[1], "last_seen": r[2], "expires_at": r[3], "user_agent": r[4], "method": r[5]}
                for r in rows]

    def revoke(self, user_id: str, session_id: Optional[str] = None) -> int:
        """One session (by id) or all of the user's sessions."""
        now = self._now()
        live = "revoked_at IS NULL AND expires_at > ?"
        if session_id:
            cur = self._conn.execute(f"UPDATE sessions SET revoked_at=? WHERE user_id=? AND id=? AND {live}",
                                     (now, user_id, session_id, now))
        else:
            cur = self._conn.execute(f"UPDATE sessions SET revoked_at=? WHERE user_id=? AND {live}", (now, user_id, now))
        return cur.rowcount

    def revoke_token(self, token: Optional[str]) -> None:
        if token:
            self._conn.execute("UPDATE sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
                               (self._now(), _sha(token)))

    # email codes
    def _code_hash(self, email: str, code: str) -> str:
        return hmac.new(self._pepper.encode(), f"{email}:{code}".encode(), hashlib.sha256).hexdigest()

    def start_email(self, email: str) -> None:
        """Send a code. Raises AuthError on throttling or when no way to deliver it exists."""
        email = normalise_email(email)
        now = self._now()
        rows = self._conn.execute("SELECT created_at FROM email_codes WHERE email=? AND created_at>? ORDER BY created_at DESC",
                                  (email, now - 3600)).fetchall()
        if rows and now - rows[0][0] < CODE_RESEND_GAP:
            raise AuthError(f"Wait {CODE_RESEND_GAP - (now - rows[0][0])} s before asking for another code", 429)
        if len(rows) >= CODE_MAX_PER_HOUR:
            raise AuthError("Too many codes for this address; try again in an hour", 429)
        if self._mailer is None:
            raise AuthError("Email sign-in is not configured on this server", 503)
        code = f"{secrets.randbelow(1_000_000):06d}"
        self._conn.execute("INSERT INTO email_codes(email, code_hash, created_at, expires_at) VALUES (?,?,?,?)",
                           (email, self._code_hash(email, code), now, now + CODE_TTL))
        self._mailer(email, code)

    def verify_email(self, email: str, code: str, user_agent: str = "") -> str:
        """Returns a session token. One message for every failure (no oracle)."""
        email = normalise_email(email)
        code = (code or "").strip()
        bad = AuthError("That code is wrong or has expired. Ask for a new one.", 401)
        if not re.fullmatch(r"\d{6}", code):
            raise bad
        now = self._now()
        with self._lock:
            r = self._conn.execute(
                "SELECT id, code_hash, expires_at, attempts, used_at FROM email_codes WHERE email=? "
                "ORDER BY created_at DESC LIMIT 1", (email,)).fetchone()
            if r is None or r[4] is not None or r[2] <= now or r[3] >= CODE_MAX_ATTEMPTS:
                raise bad
            if not hmac.compare_digest(r[1], self._code_hash(email, code)):
                self._conn.execute("UPDATE email_codes SET attempts=attempts+1 WHERE id=?", (r[0],))
                raise bad
            cur = self._conn.execute("UPDATE email_codes SET used_at=? WHERE id=? AND used_at IS NULL", (now, r[0]))
            if cur.rowcount != 1:
                raise bad
        user = self._upsert_user(email)
        return self.create_session(user, user_agent, "email")

    # google
    def verify_google(self, credential: str, client_id: str, user_agent: str = "",
                      verifier: Optional[Callable[[str, str], Dict]] = None) -> str:
        if not client_id:
            raise AuthError("Google sign-in is not configured on this server", 503)
        verify = verifier or _google_verify
        try:
            info = verify(credential, client_id)
        except AuthError:
            raise
        except Exception:
            raise AuthError("Google sign-in failed; try again", 401)
        if info.get("aud") != client_id or info.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
            raise AuthError("Google sign-in failed; try again", 401)
        if not info.get("email_verified"):
            raise AuthError("Your Google account's email is not verified", 401)
        email = normalise_email(info.get("email", ""))
        user = self._upsert_user(email, name=str(info.get("name") or "")[:100], google_sub=str(info["sub"]))
        return self.create_session(user, user_agent, "google")

    def delete_account_request(self, user_id: str) -> None:
        """Mark deleted and revoke sessions. Jobs, files and balances are handled
        by the deletion workflow (tracker L04, P09); this only closes access."""
        self._conn.execute("UPDATE users SET deleted_at=? WHERE id=? AND deleted_at IS NULL", (self._now(), user_id))
        self.revoke(user_id)


def _google_verify(credential: str, client_id: str) -> Dict:
    from google.auth.transport import requests as g_requests
    from google.oauth2 import id_token
    return id_token.verify_oauth2_token(credential, g_requests.Request(), client_id)


# ── legacy assignment CLI ────────────────────────────────────────────────
def _cli(argv: Optional[List[str]] = None) -> int:
    """python -m accounts assign-legacy --email you@example.com [--apply]

    Lists jobs that have no owner (made before accounts existed) and, with
    --apply, assigns them to the account with that email. Dry run by default.
    """
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="python -m accounts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assign-legacy")
    a.add_argument("--email", required=True)
    a.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    state = Path(os.environ.get("VOICEDUB_STATE_DIR") or Path(__file__).resolve().parent)
    store = AccountStore(state / "accounts.db")
    user = store.user_by_email(normalise_email(args.email))
    if user is None:
        print(f"No account for {args.email}: sign in once first.")
        return 1
    db = sqlite3.connect(str(state / "jobs.db"))
    rows = db.execute("SELECT id, payload FROM jobs").fetchall()
    todo = []
    for jid, payload in rows:
        d = json.loads(payload)
        if not d.get("owner_id"):
            todo.append((jid, d.get("video_title") or d.get("source_url") or ""))
    print(f"{len(todo)} unassigned job(s) of {len(rows)}:")
    for jid, title in todo[:50]:
        print(f"  {jid}  {str(title)[:70]}")
    if not args.apply:
        print("Dry run. Add --apply to assign them to", user.email)
        return 0
    for jid, _ in todo:
        payload = json.loads(db.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0])
        payload["owner_id"] = user.id
        db.execute("UPDATE jobs SET payload=?, owner_id=? WHERE id=?", (json.dumps(payload, ensure_ascii=False), user.id, jid))
    db.commit()
    print(f"Assigned {len(todo)} job(s) to {user.email}. Restart the backend to load them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
