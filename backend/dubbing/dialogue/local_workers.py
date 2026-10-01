"""Persistent worker processes for local AI models with conflicting deps.

Indic Parler-TTS pins transformers==4.46.1 while IndicTrans2/IndicTransToolkit
need transformers>=4.51 (and both break on transformers 5). Each model
therefore runs in its own long-lived child process, optionally under its own
Python interpreter (a separate venv):

    INDIC_PARLER_PYTHON=C:\\...\\venv-parler\\Scripts\\python.exe
    INDICTRANS2_PYTHON=C:\\...\\venv-indictrans2\\Scripts\\python.exe

If unset, the current interpreter is used. The model is loaded once per job
and the process is closed afterwards, which also frees GPU memory before the
next stage (Demucs, Whisper verification).

Protocol: one JSON request per line on stdin, one JSON reply per line on the
worker's private protocol stream (stdout; the worker redirects library
prints to stderr).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

WORKER_DIR = Path(__file__).resolve().parent / "workers"
INTERPRETER_ENV = {"parler": "INDIC_PARLER_PYTHON", "indictrans2": "INDICTRANS2_PYTHON"}
SCRIPT = {"parler": "parler_worker.py", "indictrans2": "indictrans2_worker.py"}
CHECK_MODULES = {"parler": ["parler_tts", "transformers", "torch"],
                 "indictrans2": ["IndicTransToolkit", "transformers", "torch"]}


class WorkerError(RuntimeError):
    pass


def interpreter_for(kind: str) -> str:
    path = os.environ.get(INTERPRETER_ENV[kind], "").strip()
    return path if path else sys.executable


_RUNTIME_CACHE: Dict[str, bool] = {}


def runtime_available(kind: str, timeout: float = 90.0) -> bool:
    """Can the configured interpreter import the model's libraries? (cached)"""
    py = interpreter_for(kind)
    key = f"{kind}:{py}"
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]
    ok = False
    mods = CHECK_MODULES[kind]
    if py == sys.executable:
        import importlib.util
        ok = all(importlib.util.find_spec(m) is not None for m in mods)
    elif Path(py).exists():
        try:
            r = subprocess.run([py, "-c", "import " + ", ".join(mods)],
                               capture_output=True, timeout=timeout)
            ok = r.returncode == 0
        except Exception:
            ok = False
    _RUNTIME_CACHE[key] = ok
    return ok


class PersistentWorker:
    def __init__(self, kind: str, init: Optional[Dict[str, Any]] = None, timeout: float = 600.0,
                 script: Optional[Path] = None, python: Optional[str] = None):
        self.kind = kind
        self.script = Path(script) if script else WORKER_DIR / SCRIPT[kind]
        self.python = python
        self.init = init or {}
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.info: Dict[str, Any] = {}

    def _start(self):
        py = self.python or interpreter_for(self.kind)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self._proc = subprocess.Popen([py, str(self.script)], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=None, env=env,
                                      text=True, encoding="utf-8", bufsize=1)
        reply = self._roundtrip({"op": "init", **self.init})
        self.info = reply.get("info", {})

    def _roundtrip(self, req: Dict[str, Any]) -> Dict[str, Any]:
        assert self._proc and self._proc.stdin and self._proc.stdout
        try:
            self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise WorkerError(f"{self.kind} worker exited: {e}")
        result: Dict[str, Any] = {}
        reader = threading.Thread(target=lambda: result.update(
            line=self._proc.stdout.readline()), daemon=True)
        reader.start()
        reader.join(self.timeout)
        line = result.get("line")
        if reader.is_alive() or not line:
            self.close()
            raise WorkerError(f"{self.kind} worker gave no reply "
                              f"({'timeout' if reader.is_alive() else 'process exited'})")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise WorkerError(f"{self.kind}: {reply.get('error', 'unknown error')}")
        return reply

    def request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            return self._roundtrip(req)

    def close(self):
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            if p.poll() is None and p.stdin:
                p.stdin.write(json.dumps({"op": "quit"}) + "\n")
                p.stdin.flush()
                p.wait(timeout=20)
        except Exception:
            pass
        if p.poll() is None:
            p.kill()
