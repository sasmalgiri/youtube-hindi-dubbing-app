@echo off
chcp 65001 >nul 2>&1
title VoiceDub - Setup
color 0E

echo.
echo   ================================================
echo    VoiceDub - First Time Setup
echo   ================================================
echo.

:: ── Find Python ──
set PYTHON=
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe
) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe
) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON=python
    ) else (
        echo   [ERROR] Python 3.10+ not found!
        echo   Install from: https://python.org/downloads
        echo   Or run: winget install Python.Python.3.10
        pause
        exit /b 1
    )
)
echo   [OK] Python: %PYTHON%
%PYTHON% --version

:: ── Check Node.js ──
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   Node.js not found. Installing...
    winget install OpenJS.NodeJS.LTS -e --accept-source-agreements --accept-package-agreements
    if %errorlevel% neq 0 (
        echo   [ERROR] Node.js install failed. Install manually from https://nodejs.org
        pause
        exit /b 1
    )
    echo   [NOTE] Close and reopen this terminal, then run setup.bat again.
    pause
    exit /b 0
) else (
    echo   [OK] Node.js found
    node --version
)

:: ── Check FFmpeg ──
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   FFmpeg not found. Installing...
    winget install Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    echo   [NOTE] FFmpeg installed. Restart terminal for PATH changes.
) else (
    echo   [OK] FFmpeg found
)

:: ── Python packages ──
:: Never `pip install -r` over a working install. On the owner's PC that would
:: swap the CUDA torch 2.4.1+cu121 for a CPU torch (whisperx / pyannote.audio 4
:: declare torch 2.8) and overwrite onnxruntime-gpu with the CPU "onnxruntime"
:: that faster-whisper declares. So pip runs only when the app's packages are
:: missing, and always with backend\constraints.txt: torch, numpy,
:: onnxruntime-gpu... stay at the working versions or pip stops with a
:: conflict. The cu121 index is where the pinned CUDA torch lives.
echo.
echo   Checking Python packages...
cd /d "%~dp0backend"
%PYTHON% -c "import fastapi, uvicorn, edge_tts, faster_whisper, webview, yt_dlp" >nul 2>&1
if %errorlevel%==0 goto :python_ready
echo   Some are missing - installing backend\requirements.txt, pinned by
echo   backend\constraints.txt. This can take a while; pip shows its progress.
%PYTHON% -m pip install --upgrade pip -c constraints.txt
%PYTHON% -m pip install -r requirements.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 goto :python_failed
echo   [OK] Python packages installed
goto :python_done
:python_failed
echo.
echo   [ERROR] pip could not install the packages - see the messages above.
echo   Nothing was forced: pinned packages were left as they were.
pause
exit /b 1
:python_ready
echo   [OK] Python packages already installed - pip not run, nothing changed
:python_done

:: The CPU "onnxruntime" next to onnxruntime-gpu overwrites the GPU build's
:: files and the CUDA provider disappears: say so now, not as a slow job later.
%PYTHON% -c "import importlib.metadata as m; m.version('onnxruntime'); m.version('onnxruntime-gpu')" >nul 2>&1
if %errorlevel% neq 0 goto :ort_ok
echo.
echo   [WARNING] onnxruntime (CPU) is installed next to onnxruntime-gpu and has
echo             overwritten it: the GPU is not used. Repair with:
echo     %PYTHON% -m pip uninstall -y onnxruntime
echo     %PYTHON% -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2
:ort_ok

:: ── GPU (CUDA) PyTorch ──
:: Never touches a CUDA torch that is already installed: the working one on
:: the owner's PC is torch 2.4.1+cu121 (backend\constraints.txt), and another
:: torch breaks pyannote / IndicTrans2 / CTranslate2 there. A missing or
:: CPU-only torch is switched to that pinned CUDA build only after a "y".
:: The old "GPU packages" step is gone: it installed torch 2.6.0+cu126 over the
:: working torch for Coqui XTTS / Chatterbox, which the app (Edge-TTS only)
:: no longer uses.
echo.
%PYTHON% -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)" >nul 2>&1
if %errorlevel% neq 0 goto :gpu_ask
%PYTHON% -c "import torch; print('  [OK] CUDA PyTorch', torch.__version__, '- GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not visible right now')"
echo   [OK] Left as it is - setup never reinstalls an existing CUDA PyTorch
goto :gpu_done
:gpu_ask
echo   PyTorch with CUDA is not installed (missing, or a CPU-only build).
set INSTALL_GPU=
set /p INSTALL_GPU="  Install CUDA PyTorch 2.4.1 (cu121, about 2.5 GB) for an NVIDIA GPU? (y/N): "
if /i not "%INSTALL_GPU%"=="y" goto :gpu_done
echo.
echo   Installing PyTorch 2.4.1 + CUDA 12.1, pinned by backend\constraints.txt...
%PYTHON% -m pip install -c "%~dp0backend\constraints.txt" torch==2.4.1+cu121 torchaudio==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
%PYTHON% -c "import torch; print('  PyTorch', torch.__version__, '- CUDA available:', torch.cuda.is_available())"
:gpu_done

:: ── Install frontend packages ──
echo.
echo   Installing frontend packages...
cd /d "%~dp0web"
:: Output and errors stay visible: "--quiet 2>nul" hid a failed install and
:: still printed [OK].
call npm install
if %errorlevel% neq 0 (
    echo   [ERROR] npm install failed - see the messages above.
    pause
    exit /b 1
)
echo   [OK] Frontend packages installed

:: ── Create .env ──
:: Labels instead of an if-block: a ")" inside an echo (the key hints below)
:: closed the block early, so both branches ran.
cd /d "%~dp0backend"
if exist .env goto :env_exists
if exist .env.example (
    copy .env.example .env >nul
    echo   [OK] Created .env from .env.example
) else (
    echo # Add your API keys here> .env
    echo   [OK] Created empty .env
)
echo.
echo   IMPORTANT: Edit backend\.env and add your API keys!
echo   At minimum, add one translation API key:
echo     GROQ_API_KEY=your_key     (free - the default translator)
echo     GEMINI_API_KEY=your_key   (free tier at aistudio.google.com)
echo     OPENAI_API_KEY=your_key   (paid, best quality)
goto :env_done
:env_exists
echo   [OK] .env already exists
:env_done

echo.
echo   ================================================
echo    Setup complete!
echo.
echo    1. Edit backend\.env with your API keys
echo    2. Start VoiceDub with:
echo         VoiceDub.bat - desktop window (recommended)
echo         start.bat    - backend + frontend in your browser
echo.
echo    Adding a Python package later? Keep the working
echo    CUDA setup safe with the constraints file:
echo      pip install -c backend\constraints.txt PACKAGE
echo   ================================================
echo.
pause
