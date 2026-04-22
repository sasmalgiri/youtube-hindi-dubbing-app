"""Google Cloud TTS credential pool with monthly quota rotation.

Rotates across multiple service-account JSONs (one per GCP billing account),
persisting char usage per credential per month so the free tier is respected
across process restarts. When a credential would exceed quota for the current
call, the pool skips to the next; when all are exhausted, synthesize() raises
and the caller falls through to Edge-TTS.

Env vars (any of, first match wins):
    GOOGLE_TTS_CREDENTIALS_DIR     directory containing *.json service accounts
    GOOGLE_TTS_CREDENTIALS         comma-separated paths
    GOOGLE_TTS_CREDENTIAL_1..N     numbered paths
    GOOGLE_APPLICATION_CREDENTIALS single path (legacy fallback)
    GOOGLE_TTS_QUOTA_PER_CREDENTIAL char quota per cred per month (default 950000)
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Google Neural2 free tier is 1M chars/month per billing account.
# We leave a 5% safety margin so we never spill into paid billing mid-call.
DEFAULT_QUOTA_PER_CREDENTIAL = 950_000


class GoogleTTSPool:
    def __init__(
        self,
        credential_paths: List[Path],
        usage_file: Path,
        quota_per_cred: int = DEFAULT_QUOTA_PER_CREDENTIAL,
    ):
        if not credential_paths:
            raise ValueError("GoogleTTSPool requires at least one credential path")
        self.credential_paths = credential_paths
        self.usage_file = usage_file
        self.quota_per_cred = quota_per_cred
        self._lock = threading.Lock()
        self._clients: dict = {}
        self._usage = self._load_usage()

    @staticmethod
    def _current_month() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _load_usage(self) -> dict:
        """Read the usage file; if it was truncated by a crash mid-write, try
        the *.bak fallback before giving up. Losing the counter means the pool
        starts over-spending the free tier, so the fallback is load-bearing."""
        for candidate in (self.usage_file, self.usage_file.with_suffix(self.usage_file.suffix + ".bak")):
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        return data
                except Exception:
                    continue
        return {}

    def _save_usage(self) -> None:
        """Atomic write: dump to a sibling temp, fsync, rename into place.
        Keeps the previous successful write as a *.bak so a crashed rename
        never loses the counter. Without this, a kill between open+write was
        enough to zero out the quota tracking for the month."""
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.usage_file.with_suffix(self.usage_file.suffix + ".tmp")
        bak_path = self.usage_file.with_suffix(self.usage_file.suffix + ".bak")
        payload = json.dumps(self._usage, indent=2, ensure_ascii=False)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # some filesystems / Windows edge cases don't support fsync on files
        if self.usage_file.exists():
            try:
                os.replace(str(self.usage_file), str(bak_path))
            except OSError:
                pass
        os.replace(str(tmp_path), str(self.usage_file))

    def _month_usage(self) -> dict:
        return self._usage.setdefault(self._current_month(), {})

    def _get_client(self, path: Path):
        if path not in self._clients:
            from google.cloud import texttospeech
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(str(path))
            self._clients[path] = texttospeech.TextToSpeechClient(credentials=creds)
        return self._clients[path]

    def _pick_credential(self, needed_chars: int, skip: set) -> Optional[Path]:
        usage = self._month_usage()
        for p in self.credential_paths:
            if str(p) in skip:
                continue
            used = usage.get(str(p), 0)
            if used + needed_chars <= self.quota_per_cred:
                return p
        return None

    def _record_usage(self, path: Path, chars: int) -> None:
        usage = self._month_usage()
        key = str(path)
        usage[key] = usage.get(key, 0) + chars
        self._save_usage()

    def synthesize(self, text: str, voice_params, audio_config):
        """Synthesize with auto-rotation. Raises if all credentials exhausted."""
        from google.cloud import texttospeech

        chars = len(text)
        tried: set = set()
        last_error: Optional[Exception] = None

        while True:
            with self._lock:
                cred_path = self._pick_credential(chars, tried)
            if cred_path is None:
                raise RuntimeError(
                    f"All {len(self.credential_paths)} Google TTS credentials "
                    f"exhausted for {self._current_month()} "
                    f"(needed {chars} chars, quota {self.quota_per_cred}/cred). "
                    f"Last error: {last_error}"
                )
            tried.add(str(cred_path))
            try:
                client = self._get_client(cred_path)
                response = client.synthesize_speech(
                    input=texttospeech.SynthesisInput(text=text),
                    voice=voice_params,
                    audio_config=audio_config,
                )
                with self._lock:
                    self._record_usage(cred_path, chars)
                return response
            except Exception as e:
                last_error = e
                continue

    def usage_report(self) -> dict:
        """Return {credential_path: chars_used} for the current month."""
        with self._lock:
            month = self._month_usage()
            return {str(p): month.get(str(p), 0) for p in self.credential_paths}


def load_pool_from_env(usage_file: Path) -> Optional[GoogleTTSPool]:
    """Load credential pool from env vars. Returns None if none configured."""
    paths: List[Path] = []
    seen: set = set()

    def _add(p_str: str) -> None:
        p_str = p_str.strip()
        if not p_str:
            return
        p = Path(p_str)
        if p.exists() and str(p.resolve()) not in seen:
            paths.append(p)
            seen.add(str(p.resolve()))

    for i in range(1, 21):
        _add(os.environ.get(f"GOOGLE_TTS_CREDENTIAL_{i}", ""))

    csv_val = os.environ.get("GOOGLE_TTS_CREDENTIALS", "")
    if csv_val:
        for p in csv_val.split(","):
            _add(p)

    dir_val = os.environ.get("GOOGLE_TTS_CREDENTIALS_DIR", "").strip()
    if dir_val:
        d = Path(dir_val)
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                _add(str(p))

    if not paths:
        _add(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))

    if not paths:
        return None

    try:
        quota = int(os.environ.get("GOOGLE_TTS_QUOTA_PER_CREDENTIAL", DEFAULT_QUOTA_PER_CREDENTIAL))
    except ValueError:
        quota = DEFAULT_QUOTA_PER_CREDENTIAL

    return GoogleTTSPool(paths, usage_file, quota_per_cred=quota)
