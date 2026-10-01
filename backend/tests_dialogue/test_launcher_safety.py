"""desktop.py launcher safety: never a second backend next to a running one,
never kill a program that is not VoiceDub, never run pip on its own."""
import importlib.util
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def desktop():
    saved = dict(os.environ)          # desktop.py sets env vars at import time
    try:
        spec = importlib.util.spec_from_file_location("voicedub_desktop", ROOT / "desktop.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture()
def serve():
    servers = []

    def start(routes):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = routes.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return srv.server_port

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


NETSTAT = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1196
  TCP    0.0.0.0:8000           0.0.0.0:0              LISTENING       4242
  TCP    127.0.0.1:8000         127.0.0.1:53011        ESTABLISHED     4242
  TCP    127.0.0.1:53011        127.0.0.1:8000         ESTABLISHED     777
  TCP    127.0.0.1:8000         127.0.0.1:52000        TIME_WAIT       0
  TCP    0.0.0.0:18000          0.0.0.0:0              LISTENING       9999
  TCP    [::]:3000              [::]:0                 ABHOEREN        5150
  TCP    [::]:8000              [::]:0                 LISTENING       4242
  TCP    0.0.0.0:3000           0.0.0.0:0              LISTENING       4
  UDP    0.0.0.0:8000           *:*                                    3333
"""


def test_netstat_parsing_finds_only_the_listener(desktop):
    assert desktop.parse_listening_pids(NETSTAT, 8000) == [4242]      # IPv4+IPv6, once
    assert desktop.parse_listening_pids(NETSTAT, 3000) == [5150]      # localised state; never PID 4
    assert desktop.parse_listening_pids(NETSTAT, 18000) == [9999]     # :8000 is not :18000
    assert desktop.parse_listening_pids(NETSTAT, 9) == []


def test_only_our_backend_is_recognised(desktop, serve):
    ok = json.dumps({"status": "ok"})
    ours = serve({"/api/health": ok,
                  "/openapi.json": json.dumps({"info": {"title": desktop.BACKEND_TITLE}})})
    stranger = serve({"/api/health": ok,
                      "/openapi.json": json.dumps({"info": {"title": "Other API"}})})
    bare = serve({"/api/health": ok})
    assert desktop.is_voicedub_backend(ours)
    assert not desktop.is_voicedub_backend(stranger)
    assert not desktop.is_voicedub_backend(bare)


def test_jobs_still_in_the_old_backend_are_found(desktop, serve, monkeypatch):
    jobs = [{"id": "a1", "state": "running", "video_title": "Talk"},
            {"id": "b2", "state": "review_translation", "video_title": ""},
            {"id": "c3", "state": "queued", "video_title": "Next"},
            {"id": "d4", "state": "done", "video_title": "Old"},
            {"id": "e5", "state": "waiting_for_srt", "video_title": "Kept in jobs.db"}]
    port = serve({"/api/jobs": json.dumps(jobs)})
    assert desktop._active_jobs(port) == ["Talk", "b2", "Next"]
    assert desktop._active_jobs(serve({})) == []          # no job list: nothing to ask about

    def no_answer(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", no_answer)
    assert desktop.confirm_stop_backend(port) is False    # unanswered: old backend kept
    assert desktop.confirm_stop_backend(serve({"/api/jobs": "[]"})) is True


def test_identity_strings_match_the_app(desktop):
    app = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")
    layout = (ROOT / "web" / "src" / "app" / "layout.tsx").read_text(encoding="utf-8")
    assert f'FastAPI(title="{desktop.BACKEND_TITLE}")' in app
    assert f"title: '{desktop.FRONTEND_TITLE}'" in layout


def test_a_stranger_on_the_port_is_never_killed(desktop, monkeypatch):
    calls = []
    monkeypatch.setattr(desktop, "is_port_in_use", lambda port: True)
    monkeypatch.setattr(desktop, "listening_pids", lambda port: [1234])
    monkeypatch.setattr(desktop, "_process_name", lambda pid: "node.exe")
    monkeypatch.setattr(desktop.subprocess, "run", lambda *a, **k: calls.append(a))
    assert desktop.inspect_port(3000, "frontend", lambda port: False) is None
    assert desktop.inspect_port(8000, "backend", lambda port: True) == [1234]
    assert calls == []                      # inspecting never stops anything
    monkeypatch.setattr(desktop, "listening_pids", lambda port: [])
    assert desktop.inspect_port(8000, "backend", lambda port: True) is None   # pid unknown


def test_free_port_needs_no_action(desktop, monkeypatch):
    monkeypatch.setattr(desktop, "is_port_in_use", lambda port: False)
    assert desktop.inspect_port(8000, "backend", lambda port: pytest.fail("probed")) == []


def test_missing_packages_print_a_constrained_command_and_never_run_pip(desktop, monkeypatch, capsys):
    def fake_import(name, *a):          # no real imports (faster_whisper loads CUDA DLLs)
        if name in ("webview", "uvicorn"):
            raise ImportError(name)
        return object()

    def no_process(*a, **k):
        raise AssertionError(f"launcher started a process: {a}")

    monkeypatch.setattr(desktop.importlib, "import_module", fake_import)
    monkeypatch.setattr(desktop.subprocess, "run", no_process)
    monkeypatch.setattr(desktop.subprocess, "Popen", no_process)
    assert desktop.check_python_packages() is False
    out = capsys.readouterr().out
    cmd = next(line for line in out.splitlines() if "-m pip install" in line)
    assert f'-c "{desktop.CONSTRAINTS_FILE}"' in cmd
    assert '"uvicorn[standard]==0.34.0"' in cmd and '"pywebview>=5.0"' in cmd
    assert not re.search(r"\s-r\s", cmd)


def test_requirements_no_longer_pull_torch_or_whisperx():
    active = [line.split("#", 1)[0].strip() for line in
              (ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    names = {re.split(r"[\[<>=!~ ]", line, maxsplit=1)[0].lower() for line in active if line}
    assert not names & {"torch", "torchaudio", "whisperx"}
    dialogue = (ROOT / "backend" / "requirements-dialogue.txt").read_text(encoding="utf-8")
    assert re.search(r"^-c constraints\.txt$", dialogue, re.M)
