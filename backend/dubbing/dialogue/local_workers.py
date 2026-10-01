"""Persistent worker processes for local AI models with conflicting deps.

Indic Parler-TTS pins transformers==4.46.1 while IndicTrans2/IndicTransToolkit
need transformers>=4.51 (and both break on transformers 5). Each model
therefore runs in its own long-lived child process, optionally under its own
Python interpreter (a separate venv):

    INDIC_PARLER_PYTHON=C:\\...\\venv-parler\\Scripts\\python.exe
    INDICTRANS2_PYTHON=C:\\...\\venv-indictrans2\\Scripts\\python.exe

If unset, the current interpreter is used. runtime_status() tells whether
the configured interpreter can really run a model: the libraries must be
installed AND satisfy the version pins in the library's own metadata (an
import check alone passes for parler-tts under the app's newer
transformers, a combination parler-tts does not support). The model is
loaded once per job and the process is closed afterwards, which also frees
GPU memory before the next stage (Demucs, Whisper verification).

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
from typing import Any, Dict, List, Optional, Tuple

# This module must stay importable on its own (standard library only at
# module level): the runtime check runs it inside the model's interpreter.
WORKER_DIR = Path(__file__).resolve().parent / "workers"
INTERPRETER_ENV = {"parler": "INDIC_PARLER_PYTHON", "indictrans2": "INDICTRANS2_PYTHON"}
SCRIPT = {"parler": "parler_worker.py", "indictrans2": "indictrans2_worker.py"}
CHECK_MODULES = {"parler": ["parler_tts", "transformers", "torch"],
                 "indictrans2": ["IndicTransToolkit", "transformers", "torch"]}
# The package whose own metadata pins what its model code needs. parler-tts
# pins transformers==4.46.1: under the main app's newer transformers it
# still imports but runs on an unsupported stack (it already broke there
# once: mixed bf16/fp32 weights), so an import check alone would wrongly
# call it runnable.
DISTRIBUTION = {"parler": "parler-tts", "indictrans2": "IndicTransToolkit"}
SETUP_HINT = {"parler": "; run setup_local_ai.bat", "indictrans2": ""}


class WorkerError(RuntimeError):
    pass


def interpreter_for(kind: str) -> str:
    path = os.environ.get(INTERPRETER_ENV[kind], "").strip()
    return path if path else sys.executable


def _module_installed(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def unmet_requirements(dist: str) -> List[str]:
    """Requirements that `dist` declares in its own package metadata and the
    installed packages do not satisfy (like `pip check`, optional extras
    ignored). Reads metadata only; raises PackageNotFoundError if `dist`
    itself is not installed."""
    from importlib import metadata
    from packaging.requirements import Requirement
    bad = []
    for raw in metadata.requires(dist) or []:
        req = Requirement(raw)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        try:
            have = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            have = None
        if have is None or not req.specifier.contains(have, prereleases=True):
            bad.append(f"{req.name}{req.specifier} (installed: {have or 'none'})")
    return bad


# Runs inside a separate interpreter: import the libraries for real, then
# apply this module's requirement check to that environment.
_CHECK_SCRIPT = (
    "import importlib, runpy, sys\n"
    "for m in sys.argv[3:]:\n"
    "    importlib.import_module(m)\n"
    "bad = runpy.run_path(sys.argv[1])['unmet_requirements'](sys.argv[2])\n"
    "print('; '.join(bad) if bad else 'OK')\n")

_RUNTIME_CACHE: Dict[str, Tuple[bool, str]] = {}


def runtime_status(kind: str, timeout: float = 90.0) -> Tuple[bool, str]:
    """(runnable, reason): can the configured interpreter actually run this
    model's worker? The reason says what is wrong and how to fix it. Cached
    per interpreter."""
    py = interpreter_for(kind)
    key = f"{kind}:{py}"
    if key not in _RUNTIME_CACHE:
        _RUNTIME_CACHE[key] = _check_runtime(kind, py, timeout)
    return _RUNTIME_CACHE[key]


def runtime_available(kind: str, timeout: float = 90.0) -> bool:
    """Can the configured interpreter run the model's worker? (cached)"""
    return runtime_status(kind, timeout)[0]


def _check_runtime(kind: str, py: str, timeout: float) -> Tuple[bool, str]:
    mods, dist, var = CHECK_MODULES[kind], DISTRIBUTION[kind], INTERPRETER_ENV[kind]
    if py == sys.executable:
        # The main app's interpreter: never import the model libraries into
        # the server process (slow, and they may clash) -- metadata only.
        missing = [m for m in mods if not _module_installed(m)]
        if missing:
            return False, f"not installed in this Python: {', '.join(missing)}"
        try:
            bad = unmet_requirements(dist)
        except Exception as e:
            return False, f"cannot check what {dist} requires: {type(e).__name__}: {e}"
        if bad:
            return False, (f"needs its own Python env ({var}): {dist} requires "
                           f"{'; '.join(bad)}{SETUP_HINT[kind]}")
        return True, f"runnable in this Python ({py})"
    if not Path(py).exists():
        return False, f"{var} points to a missing interpreter: {py}"
    try:
        r = subprocess.run([py, "-c", _CHECK_SCRIPT, str(Path(__file__).resolve()), dist, *mods],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, cwd=str(WORKER_DIR),
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    except subprocess.TimeoutExpired:
        return False, f"{var} interpreter did not finish importing {', '.join(mods)} in {timeout:.0f} s"
    except Exception as e:
        return False, f"{var} interpreter could not be run: {type(e).__name__}: {e}"
    out = (r.stdout or "").strip().splitlines()
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()
        return False, (f"{var} interpreter cannot load the model libraries: "
                       f"{err[-1][:300] if err else f'exit code {r.returncode}'}")
    if not out or out[-1] != "OK":
        return False, f"{var} env does not satisfy {dist}: {out[-1][:300] if out else 'no answer'}"
    return True, f"runnable in {py}"


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
        self.init_error: Optional[str] = None

    def _start(self):
        if self.init_error:
            raise WorkerError(self.init_error)
        py = self.python or interpreter_for(self.kind)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            self._proc = subprocess.Popen([py, str(self.script)], stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, stderr=None, env=env,
                                          text=True, encoding="utf-8", bufsize=1)
            reply = self._roundtrip({"op": "init", **self.init})
        except (WorkerError, OSError) as e:
            # A model that failed to load (gated repo, wrong library version,
            # out of memory, load timeout) fails the same way for the next
            # line. Keep the real reason and fail fast with it for the rest
            # of the job, instead of reloading for minutes per request or
            # letting later requests reach a half-initialised worker (whose
            # errors read "KeyError: 'torch'" and hide the cause).
            self.close()
            self.init_error = f"{self.kind} worker failed to start: {e}"
            raise WorkerError(self.init_error) from e
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
        try:
            reply = json.loads(line)
            if not isinstance(reply, dict):
                raise ValueError("not a JSON object")
        except ValueError:
            # Something other than the worker's reply reached the protocol
            # stream: replies would now be out of step with requests.
            self.close()
            raise WorkerError(f"{self.kind} worker wrote a non-protocol line: {line.strip()[:200]!r}")
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
