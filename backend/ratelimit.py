"""Per-client request limits for the hosted profile (sliding one-minute window).

Rules are (path prefix, method or None, requests per minute), first match wins.
The client is the socket peer; behind Cloudflare Tunnel set VOICEDUB_TRUST_CF=1
so CF-Connecting-IP is used instead (only then: the header is trivially forged).
Memory only: a restart forgets the counts, which is fine for abuse limits.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Sequence, Tuple

RULES: Sequence[Tuple[str, Optional[str], int]] = (
    ("/api/auth/email/start", "POST", 5),
    ("/api/auth/email/verify", "POST", 10),
    ("/api/auth/google", "POST", 10),
    ("/api/drafts", "POST", 20),
    ("/api/quotes", "POST", 30),
    ("/api/uploads/", "PUT", 120),          # 50 MB pieces: a 2 GB file is 41 requests
    ("/api/uploads", "POST", 30),
    ("/api/checkout", "POST", 20),
    ("/api/account/delete", "POST", 5),
    ("/api/", None, 600),
)


class Limiter:
    def __init__(self, rules=RULES, window: float = 60.0, clock=time.monotonic):
        self.rules, self.window, self.clock = rules, window, clock
        self._hits: Dict[Tuple[str, str], Deque[float]] = {}
        self._lock = threading.Lock()

    def rule(self, path: str, method: str):
        for prefix, m, n in self.rules:
            if path.startswith(prefix) and (m is None or m == method):
                return prefix, n
        return None

    def check(self, client: str, path: str, method: str) -> Optional[int]:
        """None if allowed, else seconds to wait."""
        r = self.rule(path, method)
        if r is None:
            return None
        prefix, n = r
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault((client, prefix), deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= n:
                return max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            if len(self._hits) > 50000:          # bound memory under a spray of addresses
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[k]
        return None


def client_of(request) -> str:
    if os.environ.get("VOICEDUB_TRUST_CF") == "1":
        ip = request.headers.get("cf-connecting-ip")
        if ip:
            return ip.strip()
    return request.client.host if request.client else "?"
