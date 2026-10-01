"""
VoiceDub Desktop App
====================
Self-contained launcher that:
  1. Checks all dependencies (Python packages, Node.js, FFmpeg)
  2. Reports missing Python packages with the exact pip command to run
     (never runs pip itself: see check_python_packages)
  3. Starts FastAPI backend (port 8000) + Next.js frontend (port 3000),
     replacing an old VoiceDub server that still holds either port
  4. Opens a native desktop window via pywebview
  5. Cleans up everything on close

Works on any Windows PC — just copy the folder and run VoiceDub.bat.
"""
import os
import re
import sys
import json
import time
import shutil
import signal
import subprocess
import urllib.error
import urllib.request
import importlib

# ── Constants ────────────────────────────────────────────────────────────────
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["COQUI_TOS_AGREED"] = "1"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(APP_DIR, "backend")
FRONTEND_DIR = os.path.join(APP_DIR, "web")
PYTHON = sys.executable
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
CONSTRAINTS_FILE = os.path.join(BACKEND_DIR, "constraints.txt")
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

processes = []


# ── Helpers ──────────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    icons = {"INFO": "  ", "OK": "  [OK]", "WARN": "  [!]", "ERR": "  [X]", "STEP": "  >>"}
    print(f"{icons.get(level, '  ')} {msg}")


def run_cmd(cmd, check=False, capture=True):
    """Run a command and return (success, stdout)."""
    try:
        r = subprocess.run(
            cmd, capture_output=capture, text=True, timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return r.returncode == 0, r.stdout or ""
    except Exception as e:
        return False, str(e)


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def wait_for_server(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ── Ports ────────────────────────────────────────────────────────────────────
# Always 8000 (backend) and 3000 (frontend). This launcher used to move to the
# next free port, which started a SECOND backend next to a running one; that
# backend's startup hook marks the first one's running jobs as failed and
# wipes the caches in the shared jobs.db. Now an old VoiceDub server holding a
# port is stopped the way start.bat does it (netstat -ano -> taskkill), and a
# port held by anything that does not answer as VoiceDub is never touched.
BACKEND_TITLE = "YouTube Hindi Dubbing API"   # FastAPI(title=...) in backend/app.py
FRONTEND_TITLE = "YouTube Hindi Dubbing"      # metadata.title in web/src/app/layout.tsx
# The backend (uvicorn --host 0.0.0.0) listens on IPv4 only, and on Windows
# "localhost" tries ::1 first: each request then waits ~2 s for the refusal.
BACKEND_HOST = "127.0.0.1"


def parse_listening_pids(netstat_text, port):
    """PIDs LISTENING on TCP `port` in `netstat -ano` output (IPv4 and IPv6)."""
    pids = []
    for line in netstat_text.splitlines():
        parts = line.split()
        # Proto  Local-Address  Foreign-Address  State  PID. The state name is
        # localised; a listening socket's foreign address is always 0:0.
        if (len(parts) == 5 and parts[0].upper() == "TCP"
                and parts[1].rsplit(":", 1)[-1] == str(port)
                and (parts[3].upper() == "LISTENING" or parts[2] in ("0.0.0.0:0", "[::]:0"))
                and parts[4].isdigit() and int(parts[4]) > 4    # never 0 (Idle) / 4 (System)
                and int(parts[4]) not in pids):
            pids.append(int(parts[4]))
    return pids


def listening_pids(port):
    """PIDs listening on `port`, from `netstat -ano` like start.bat ([] if unknown)."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, timeout=60,
                             creationflags=NO_WINDOW).stdout
    except Exception:
        return []
    return parse_listening_pids(out.decode("utf-8", errors="replace"), port)


def _process_name(pid):
    """Image name of `pid` for messages ("?" when unknown)."""
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, timeout=30, creationflags=NO_WINDOW).stdout
        name = out.decode("utf-8", errors="replace").strip().split(",")[0].strip('"')
        return name if name and not name.upper().startswith("INFO:") else "?"
    except Exception:
        return "?"


def _http_get(url, timeout):
    """(status, body) of a GET, or None when nothing answers."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return None


def is_voicedub_backend(port):
    """/api/health says ok AND the API title is ours: a bare health endpoint
    returning {"status": "ok"} is common to many local servers."""
    health = _http_get(f"http://{BACKEND_HOST}:{port}/api/health", 10)
    if not health or health[0] != 200:
        return False
    spec = _http_get(f"http://{BACKEND_HOST}:{port}/openapi.json", 30)
    if not spec or spec[0] != 200:
        return False
    try:
        return (json.loads(health[1]).get("status") == "ok"
                and json.loads(spec[1]).get("info", {}).get("title") == BACKEND_TITLE)
    except (ValueError, AttributeError):
        return False


def is_voicedub_frontend(port):
    """Our Next.js UI, recognised by its page title. A dev server compiles a
    page on its first request, hence the long timeout."""
    page = _http_get(f"http://localhost:{port}/", 60)
    return bool(page) and page[0] == 200 and f">{FRONTEND_TITLE}<" in page[1]


def inspect_port(port, what, is_ours):
    """[] when `port` is free; the LISTENING PIDs when an old VoiceDub `what`
    server holds it; None, after saying why, when anything else holds it --
    that program is left running."""
    if not is_port_in_use(port):
        return []
    log(f"Port {port} is busy - checking whether an old VoiceDub {what} holds it...", "STEP")
    pids = listening_pids(port)
    ids = ", ".join(map(str, pids))
    if is_ours(port):
        if pids:
            log(f"An old VoiceDub {what} is running on port {port} (PID {ids}) "
                f"- it will be replaced", "WARN")
            return pids
        log(f"An old VoiceDub {what} answers on port {port}, but netstat -ano does not "
            f"show its process. Close it, then start VoiceDub again.", "ERR")
        return None
    who = ", ".join(f"PID {p} ({_process_name(p)})" for p in pids) or "a process netstat does not show"
    kill = "taskkill /F " + (" ".join(f"/PID {p}" for p in pids) or "/PID <pid>")
    log(f"Port {port} is used by {who}, which does not answer as the VoiceDub {what}.", "ERR")
    log("It was left running. Close that program - or, if it is an old VoiceDub window that "
        f"stopped responding, end it with: {kill} - then start VoiceDub again.", "INFO")
    return None


# Jobs that live only in the backend process: a job paused for review
# (step-by-step) waits there on its pipeline thread, and a restarted backend
# turns it into an error. ("waiting_for_srt" survives a restart in jobs.db.)
ACTIVE_JOB_STATES = ("running", "queued", "review_transcription", "review_translation")


def _active_jobs(port):
    """Titles of the jobs the backend on `port` is still working on."""
    jobs = _http_get(f"http://{BACKEND_HOST}:{port}/api/jobs", 15)
    try:
        return [j.get("video_title") or j.get("id", "?") for j in json.loads(jobs[1])
                if j.get("state") in ACTIVE_JOB_STATES]
    except (TypeError, ValueError, AttributeError):
        return []


def confirm_stop_backend(port):
    """Stopping the old backend ends the jobs it is working on: ask first."""
    busy = _active_jobs(port)
    if not busy:
        return True
    log(f"It is still working on {len(busy)} job(s): {', '.join(busy)[:200]}", "WARN")
    log("Stopping it ends them; they would have to be resubmitted.", "WARN")
    try:
        input("  Press Enter to stop it anyway, or close this window to keep it running... ")
        return True
    except (EOFError, KeyboardInterrupt):
        print()
        log("Left the old backend running.", "INFO")
        return False


def stop_old_server(port, what, pids):
    """Stop an old VoiceDub server like start.bat does (taskkill /F; /T also
    ends its ffmpeg / Whisper / diarization children, which would otherwise
    keep the GPU busy). True once the port is free."""
    log(f"Stopping the old VoiceDub {what} on port {port}...", "STEP")
    if what == "backend":
        log("If start-backend-stable.bat started it, close that window too: "
            "it restarts the backend by itself.", "INFO")
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, creationflags=NO_WINDOW)
    deadline = time.time() + 15
    while is_port_in_use(port):
        if time.time() > deadline:
            log(f"Port {port} is still busy after stopping the old {what}.", "ERR")
            return False
        time.sleep(0.5)
    time.sleep(2)   # like start.bat: give Windows a moment to release the socket
    log(f"Port {port} is free", "OK")
    return True


# ── Dependency Checks ────────────────────────────────────────────────────────
def _requirement_spec(pip_name):
    """`pip_name`'s line in backend/requirements.txt (e.g. "edge-tts==7.2.7"),
    so the printed command installs the version the app expects."""
    try:
        with open(os.path.join(BACKEND_DIR, "requirements.txt"), encoding="utf-8") as f:
            for line in f:
                spec = line.split("#", 1)[0].strip()
                name = re.split(r"[\[<>=!~;@ ]", spec, maxsplit=1)[0]
                if spec and name.lower().replace("_", "-") == pip_name.lower():
                    return spec
    except OSError:
        pass
    return pip_name


def check_python_packages():
    """Check that the key Python packages import. Never runs pip: on the
    owner's PC `pip install -r requirements.txt` would break the working GPU
    setup (faster-whisper's CPU "onnxruntime" dependency overwrites
    onnxruntime-gpu; with whisperx listed, it would also swap the CUDA torch
    2.4.1+cu121 for a CPU torch 2.8). Prints the exact command, pinned by
    backend/constraints.txt, for the user to run instead."""
    log("Checking Python packages...")

    # Quick check: try importing key packages
    critical_packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "edge_tts": "edge-tts",
        "faster_whisper": "faster-whisper",
        "webview": "pywebview",
    }

    missing = []
    for mod_name, pip_name in critical_packages.items():
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        log(f"Missing Python packages: {', '.join(missing)}", "ERR")
        log("They are not installed automatically. Run this, then start VoiceDub again", "INFO")
        log("(backend/constraints.txt keeps torch, numpy and onnxruntime-gpu as they are):", "INFO")
        # Specs are quoted (">=" would redirect); the interpreter only when it
        # must be: a quoted first word is not a command in PowerShell.
        specs = " ".join(f'"{_requirement_spec(m)}"' for m in missing)
        py = f'"{PYTHON}"' if " " in PYTHON else PYTHON
        print(f'\n    {py} -m pip install -c "{CONSTRAINTS_FILE}" {specs}\n')
        log("If pip reports a conflict, or --dry-run shows it would install torch, numpy", "INFO")
        log("or onnxruntime, install that package with --no-deps instead.", "INFO")
        return False

    log("Python packages ready", "OK")
    return True


def check_node():
    """Check if Node.js is installed."""
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    ok, out = run_cmd([npm_cmd, "--version"])
    if ok:
        log(f"Node.js/npm found (v{out.strip()})", "OK")
        return True

    log("Node.js not found!", "ERR")
    log("Install from: https://nodejs.org/en/download/", "INFO")
    log("Or run: winget install OpenJS.NodeJS.LTS", "INFO")
    return False


def check_ffmpeg():
    """Check if FFmpeg is available."""
    # Check PATH
    if shutil.which("ffmpeg"):
        log("FFmpeg found in PATH", "OK")
        return True

    # Check winget install location
    local = os.environ.get("LOCALAPPDATA", "")
    if not local:
        userprofile = os.environ.get("USERPROFILE", "")
        if userprofile:
            local = os.path.join(userprofile, "AppData", "Local")

    if local:
        winget_dir = os.path.join(local, "Microsoft", "WinGet", "Packages")
        if os.path.exists(winget_dir):
            for d in os.listdir(winget_dir):
                if "FFmpeg" in d:
                    for root, dirs, files in os.walk(os.path.join(winget_dir, d)):
                        if "ffmpeg.exe" in files:
                            ffmpeg_dir = root
                            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
                            log(f"FFmpeg found at {ffmpeg_dir}", "OK")
                            return True

    log("FFmpeg not found!", "WARN")
    log("Install with: winget install Gyan.FFmpeg", "INFO")
    log("Edge-TTS will still work, but some features need FFmpeg.", "INFO")
    return False


def check_node_modules():
    """Check and install frontend npm packages."""
    nm_dir = os.path.join(FRONTEND_DIR, "node_modules")
    if os.path.exists(nm_dir) and os.path.exists(os.path.join(nm_dir, "next")):
        log("Frontend packages ready", "OK")
        return True

    log("Installing frontend packages (first run)...", "STEP")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    ok, _ = run_cmd([npm_cmd, "install"], capture=False)
    if ok:
        log("Frontend packages installed", "OK")
    else:
        # Try with cwd
        try:
            subprocess.run(
                [npm_cmd, "install"], cwd=FRONTEND_DIR,
                timeout=300, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            log("Frontend packages installed", "OK")
            ok = True
        except Exception as e:
            log(f"npm install failed: {e}", "ERR")
            return False
    return ok


def check_env_file():
    """Create .env from .env.example if it doesn't exist."""
    env_file = os.path.join(BACKEND_DIR, ".env")
    example_file = os.path.join(BACKEND_DIR, ".env.example")

    if os.path.exists(env_file):
        log(".env file exists", "OK")
        return True

    if os.path.exists(example_file):
        shutil.copy2(example_file, env_file)
        log("Created .env from .env.example — add your API keys there", "WARN")
    else:
        # Create minimal .env
        with open(env_file, "w") as f:
            f.write("# Add your API keys here\n")
            f.write("# GEMINI_API_KEY=your_key\n")
            f.write("# OPENAI_API_KEY=your_key\n")
        log("Created empty .env — add API keys in backend/.env", "WARN")

    return True


def check_gpu():
    """Check GPU/CUDA availability."""
    try:
        ok, out = run_cmd([PYTHON, "-c",
            "import torch; print(f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU-only')"
        ])
        if ok and out.strip() and out.strip() != "CPU-only":
            log(f"GPU: {out.strip()}", "OK")
        else:
            log("GPU: CPU-only mode (Edge-TTS will work, Coqui XTTS needs GPU)", "INFO")
    except Exception:
        log("GPU: Could not detect (CPU mode)", "INFO")


# ── Server Management ────────────────────────────────────────────────────────
# Server output goes to log files, never to subprocess.PIPE: nothing reads
# those pipes while the app runs, so once the OS pipe buffer (a few KB) fills
# up, the server's next print() blocks forever — the backend froze mid-job.
LOG_DIR = os.path.join(BACKEND_DIR, "logs")
BACKEND_LOG = os.path.join(LOG_DIR, "desktop-backend.log")
FRONTEND_LOG = os.path.join(LOG_DIR, "desktop-frontend.log")


def _open_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a", encoding="utf-8", errors="replace")
    f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    f.flush()
    return f


def _log_tail(path, n=2000):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()[-n:]
    except OSError:
        return ""


def start_backend(port):
    """Start the FastAPI backend (output -> backend/logs/desktop-backend.log)."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "app:app",
         "--host", "0.0.0.0", "--port", str(port)],
        cwd=BACKEND_DIR,
        env=env,
        stdout=_open_log(BACKEND_LOG),
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    processes.append(proc)
    return proc


def start_frontend(port, backend_port):
    """Start the Next.js frontend (output -> backend/logs/desktop-frontend.log).
    BACKEND_PORT tells next.config.mjs where to proxy /api (kept explicit
    although the launcher now always runs the backend on 8000)."""
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["BACKEND_PORT"] = str(backend_port)
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    proc = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "-p", str(port)],
        cwd=FRONTEND_DIR,
        env=env,
        stdout=_open_log(FRONTEND_LOG),
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    processes.append(proc)
    return proc


def cleanup():
    """Kill all child processes."""
    for proc in processes:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print()
    print("  " + "=" * 48)
    print("   VoiceDub — YouTube Video Dubbing")
    print("  " + "=" * 48)
    print()

    # ── Step 1: Check dependencies ──
    log("Checking dependencies...", "STEP")
    print()

    checks_ok = True

    if not check_python_packages():
        checks_ok = False
    if not check_node():
        checks_ok = False
    check_ffmpeg()  # Non-fatal
    check_env_file()
    check_gpu()

    if not checks_ok:
        print()
        log("Some required dependencies are missing. Fix them and try again.", "ERR")
        input("\n  Press Enter to exit...")
        sys.exit(1)

    # Install node_modules if needed (after node check passes)
    old_cwd = os.getcwd()
    os.chdir(FRONTEND_DIR)
    if not check_node_modules():
        os.chdir(old_cwd)
        log("Frontend setup failed.", "ERR")
        input("\n  Press Enter to exit...")
        sys.exit(1)
    os.chdir(old_cwd)

    print()

    # ── Step 2: Claim ports 8000 / 3000 (see "Ports" above) ──
    # Both are inspected before anything is stopped, so a stranger on 3000
    # never costs the user their running backend.
    old_backend = inspect_port(BACKEND_PORT, "backend", is_voicedub_backend)
    old_frontend = (inspect_port(FRONTEND_PORT, "frontend", is_voicedub_frontend)
                    if old_backend is not None else None)
    if old_backend is None or old_frontend is None:
        input("\n  Press Enter to exit...")
        sys.exit(1)
    if old_backend and not confirm_stop_backend(BACKEND_PORT):
        input("\n  Press Enter to exit...")
        sys.exit(1)
    for port, what, pids in ((BACKEND_PORT, "backend", old_backend),
                             (FRONTEND_PORT, "frontend", old_frontend)):
        if pids and not stop_old_server(port, what, pids):
            input("\n  Press Enter to exit...")
            sys.exit(1)

    # ── Step 3: Start servers ──
    log(f"Starting backend on port {BACKEND_PORT}...", "STEP")
    backend = start_backend(BACKEND_PORT)
    # The health check only proves that *something* answers on the port; if
    # our process has already exited, that is not this launch's backend.
    if (not wait_for_server(f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health")
            or backend.poll() is not None):
        log("Backend failed to start!", "ERR")
        out = _log_tail(BACKEND_LOG)
        if out:
            print(f"\n  Backend output (end of {BACKEND_LOG}):\n{out}")
        cleanup()
        input("\n  Press Enter to exit...")
        sys.exit(1)
    log(f"Backend running on port {BACKEND_PORT} (log: {BACKEND_LOG})", "OK")

    log(f"Starting frontend on port {FRONTEND_PORT}...", "STEP")
    frontend = start_frontend(FRONTEND_PORT, BACKEND_PORT)
    if (not wait_for_server(f"http://localhost:{FRONTEND_PORT}", timeout=60)
            or frontend.poll() is not None):
        log("Frontend failed to start!", "ERR")
        out = _log_tail(FRONTEND_LOG)
        if out:
            print(f"\n  Frontend output (end of {FRONTEND_LOG}):\n{out}")
        cleanup()
        input("\n  Press Enter to exit...")
        sys.exit(1)
    log(f"Frontend running on port {FRONTEND_PORT}", "OK")

    print()
    print("  " + "=" * 48)
    print("   VoiceDub is ready!")
    print(f"   http://localhost:{FRONTEND_PORT}")
    print("  " + "=" * 48)
    print()

    # ── Step 4: Open native window ──
    try:
        import webview

        # pywebview ignores file downloads unless this is on: the Download
        # video / subtitles / report buttons silently did nothing.
        try:
            webview.settings["ALLOW_DOWNLOADS"] = True
        except Exception:
            pass

        webview.create_window(
            title="VoiceDub",
            url=f"http://localhost:{FRONTEND_PORT}",
            width=1300,
            height=900,
            min_size=(900, 600),
            resizable=True,
            text_select=True,
        )

        webview.start(debug=False, private_mode=False)

    except ImportError:
        log("pywebview not available — opening in browser", "WARN")
        import webbrowser
        webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
        print("  Close this window or press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    except Exception as e:
        log(f"Window error: {e} — opening in browser", "WARN")
        import webbrowser
        webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
        print("  Close this window or press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    finally:
        print()
        log("Shutting down servers...", "STEP")
        cleanup()
        log("Goodbye!", "OK")


if __name__ == "__main__":
    main()
