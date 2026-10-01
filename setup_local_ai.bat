@echo off
chcp 65001 >nul 2>&1
title VoiceDub - Local AI models (optional)
REM ===========================================================================
REM  OPTIONAL. Only needed for the "Free - Local AI (unlimited)" preset.
REM  Creates SEPARATE Python environments for local models whose libraries
REM  need different transformers versions than the main app:
REM    - Indic Parler-TTS (free local Hindi voices)  -> backend\.venvs\parler
REM  and records their paths in backend\.env.
REM
REM  Before running:
REM    1. Python 3.11 installed (py -3.11 must work).
REM    2. On huggingface.co (same account as HF_TOKEN in backend\.env) accept:
REM         https://huggingface.co/ai4bharat/indic-parler-tts
REM    3. For local translation install Ollama (https://ollama.com) and run:
REM         ollama pull gemma3:12b
REM       then set OLLAMA_MODEL=gemma3:12b in backend\.env.
REM
REM  IndicTrans2 is NOT set up here: its IndicTransToolkit package ships
REM  Linux/macOS wheels only. Use Ollama on Windows (or WSL for IndicTrans2).
REM
REM  This script was written without access to a Windows PC: if a step fails,
REM  run that pip command by hand and send the error.
REM ===========================================================================

set "ROOT=%~dp0"
set "VENV=%ROOT%backend\.venvs\parler"
set "PYV=%VENV%\Scripts\python.exe"

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Python 3.11 not found. Install it: winget install Python.Python.3.11
    pause
    exit /b 1
)

if not exist "%PYV%" (
    echo   Creating %VENV% ...
    py -3.11 -m venv "%VENV%" || goto :fail
)
"%PYV%" -m pip install --upgrade pip || goto :fail
echo   Installing PyTorch with CUDA 12.1 (change cu121 to match your driver if needed)...
"%PYV%" -m pip install torch --index-url https://download.pytorch.org/whl/cu121 || goto :fail
echo   Installing Indic Parler-TTS (pins transformers 4.46.1)...
"%PYV%" -m pip install git+https://github.com/huggingface/parler-tts.git numpy || goto :fail
"%PYV%" -c "import parler_tts, torch; print('parler ok, cuda =', torch.cuda.is_available())" || goto :fail

findstr /b /c:"INDIC_PARLER_PYTHON=" "%ROOT%backend\.env" >nul 2>&1
if errorlevel 1 (
    echo INDIC_PARLER_PYTHON=%PYV%>> "%ROOT%backend\.env"
    echo   Added INDIC_PARLER_PYTHON to backend\.env
)
echo.
echo   Done. Check with:  cd backend ^&^& python -m dubbing.dialogue modules --preset free-local
pause
exit /b 0

:fail
echo.
echo   [ERROR] A step failed (see above). Nothing else was changed.
pause
exit /b 1
