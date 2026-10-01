"""
VoiceDub Desktop App
====================
Self-contained launcher that:
  1. Checks all dependencies (Python packages, Node.js, FFmpeg)
  2. Auto-installs missing Python packages
  3. Starts FastAPI backend + Next.js frontend
  4. Opens a native desktop window via pywebview
  5. Cleans up everything on close

Works on any Windows PC — just copy the folder and run VoiceDub.bat.
"""
import os
import sys
import time
import shutil
import signal
import subprocess
import threading
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


def find_free_port(start_port):
    """Find a free port starting from start_port."""
    import socket
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
        port += 1
    return start_port


# ── Dependency Checks ────────────────────────────────────────────────────────
def check_python_packages():
    """Check and install missing Python packages from requirements.txt."""
    req_file = os.path.join(BACKEND_DIR, "requirements.txt")
    if not os.path.exists(req_file):
        log("requirements.txt not found!", "ERR")
        return False

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
        log(f"Installing missing packages: {', '.join(missing)}", "STEP")
        ok, out = run_cmd([PYTHON, "-m", "pip", "install", "-r", req_file, "--quiet"])
        if not ok:
            log(f"pip install failed. Run manually:\n    {PYTHON} -m pip install -r {req_file}", "ERR")
            return False
        log("Python packages installed", "OK")
    else:
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
    BACKEND_PORT tells next.config.mjs where to proxy /api (the backend may
    not be on 8000 if that port was busy)."""
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

    # ── Step 2: Find free ports ──
    global BACKEND_PORT, FRONTEND_PORT
    BACKEND_PORT = find_free_port(8000)
    FRONTEND_PORT = find_free_port(3000)

    # ── Step 3: Start servers ──
    log(f"Starting backend on port {BACKEND_PORT}...", "STEP")
    start_backend(BACKEND_PORT)
    if not wait_for_server(f"http://localhost:{BACKEND_PORT}/api/health"):
        log("Backend failed to start!", "ERR")
        out = _log_tail(BACKEND_LOG)
        if out:
            print(f"\n  Backend output (end of {BACKEND_LOG}):\n{out}")
        cleanup()
        input("\n  Press Enter to exit...")
        sys.exit(1)
    log(f"Backend running on port {BACKEND_PORT} (log: {BACKEND_LOG})", "OK")

    log(f"Starting frontend on port {FRONTEND_PORT}...", "STEP")
    start_frontend(FRONTEND_PORT, BACKEND_PORT)
    if not wait_for_server(f"http://localhost:{FRONTEND_PORT}", timeout=60):
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
    print(f"   VoiceDub is ready!")
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

        window = webview.create_window(
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
