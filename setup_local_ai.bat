@echo off
chcp 65001 >nul 2>&1
title VoiceDub - Local AI models (optional)
REM ===========================================================================
REM  OPTIONAL. Only needed for Indic Parler-TTS voices ("Free - Local AI
REM  (unlimited)" preset). parler-tts pins transformers==4.46.1 while the main
REM  app runs a newer transformers, so it gets its OWN Python environment:
REM    - Indic Parler-TTS (free local Hindi voices)  -> backend\.venvs\parler
REM  and its path is recorded in backend\.env (INDIC_PARLER_PYTHON).
REM
REM  The main app's Python is never changed: no pip install/upgrade of torch,
REM  numpy, transformers, onnxruntime or anything else runs there. Every pip
REM  command below uses the venv's own python.exe.
REM
REM  Before running:
REM    1. Python 3.10 installed - the one the app runs on (py -3.10, or
REM       ...\AppData\Local\Programs\Python\Python310\python.exe).
REM    2. On huggingface.co (same account as HF_TOKEN in backend\.env) accept:
REM         https://huggingface.co/ai4bharat/indic-parler-tts
REM    3. For local translation install Ollama (https://ollama.com) and run:
REM         ollama pull gemma3:12b
REM       then set OLLAMA_MODEL=gemma3:12b in backend\.env.
REM
REM  IndicTrans2 needs no separate environment: it runs in the main app's
REM  Python when IndicTransToolkit is installed there. Its default model is
REM  ai4bharat/indictrans2-en-indic-1B (accept its terms on huggingface.co).
REM
REM  Not yet run end to end on this PC: if a step fails, run that pip command
REM  by hand and send the error. To start over, delete backend\.venvs\parler.
REM ===========================================================================

set "ROOT=%~dp0"
set "VENV=%ROOT%backend\.venvs\parler"
set "PYV=%VENV%\Scripts\python.exe"

REM Python 3.10 is what this PC has (3.10 and 3.13; no 3.11) and what the app
REM runs on. It is only used to create the venv and to run the final check.
set "BASEPY="
py -3.10 --version >nul 2>&1
if not errorlevel 1 set "BASEPY=py -3.10"
if not defined BASEPY if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set BASEPY="%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined BASEPY (
    echo   [ERROR] Python 3.10 not found. Install it: winget install Python.Python.3.10
    pause
    exit /b 1
)

if not exist "%PYV%" (
    echo   Creating %VENV% with Python 3.10 ...
    %BASEPY% -m venv "%VENV%" || goto :fail
)
"%PYV%" -m pip install --upgrade pip || goto :fail

REM torch + torchaudio from the CUDA 12.1 index, the same version as the main
REM app (known to work with this GPU and driver), then held there by a
REM constraints file: parler-tts' audio codec needs torchaudio, and a plain
REM PyPI torchaudio would pull a newer, CPU-only torch into the venv.
echo   Installing PyTorch 2.4.1 + torchaudio (CUDA 12.1) into the venv...
"%PYV%" -m pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || goto :fail
(
    echo torch==2.4.1
    echo torchaudio==2.4.1
) > "%VENV%\constraints.txt"

echo   Installing Indic Parler-TTS 0.2.3 (pins transformers 4.46.1) into the venv...
"%PYV%" -m pip install parler-tts==0.2.3 numpy -c "%VENV%\constraints.txt" || goto :fail
"%PYV%" -c "import parler_tts, torch; print('  parler ok, cuda =', torch.cuda.is_available())" || goto :fail

REM Ask the app's own check (imports + parler-tts version pins) whether the
REM app will accept this venv, before recording it.
echo   Checking the venv the way the app does...
set "INDIC_PARLER_PYTHON=%PYV%"
%BASEPY% -c "import runpy, sys; ok, why = runpy.run_path(sys.argv[1])['runtime_status']('parler'); print('  app check:', why); sys.exit(0 if ok else 1)" "%ROOT%backend\dubbing\dialogue\local_workers.py" || goto :fail

REM An empty line first: backend\.env may not end with a newline, and the new
REM entry must not be glued onto its last line.
findstr /b /c:"INDIC_PARLER_PYTHON=" "%ROOT%backend\.env" >nul 2>&1
if errorlevel 1 (
    >> "%ROOT%backend\.env" echo.
    >> "%ROOT%backend\.env" echo INDIC_PARLER_PYTHON=%PYV%
    echo   Added INDIC_PARLER_PYTHON to backend\.env
)
echo.
echo   Done. Restart VoiceDub, then check with:
echo     cd backend ^&^& %BASEPY% -m dubbing.dialogue modules --preset free-local
pause
exit /b 0

:fail
echo.
echo   [ERROR] A step failed (see above). The main app's Python was not changed.
pause
exit /b 1
