@echo off
title VoiceDub - YouTube Video Dubbing
color 0A

echo.
echo  ============================================
echo   VoiceDub - YouTube Video Dubbing App
echo  ============================================
echo.

:: Find Python
set PYTHON=
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON=python
    ) else (
        echo [ERROR] Python not found! Install Python 3.10+ from python.org
        pause
        exit /b 1
    )
)

echo  [OK] Python: %PYTHON%

:: Check Node.js
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found! Install from nodejs.org
    pause
    exit /b 1
)
echo  [OK] Node.js found

:: Check FFmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] FFmpeg not in PATH. Install: winget install Gyan.FFmpeg
)

:: Set UTF-8 encoding for Hindi/non-Latin TTS
set PYTHONIOENCODING=utf-8
chcp 65001 >nul 2>&1

:: Start Hinglish AI Trainer (if project exists)
set HINGLISH_DIR=%~dp0..\hinglish-ai-model
if exist "%HINGLISH_DIR%\app.py" (
    echo  Starting Hinglish AI trainer on http://localhost:8100 ...
    start "Hinglish AI Trainer" /min cmd /k "cd /d "%HINGLISH_DIR%" && %PYTHON% app.py"
    echo  [OK] Hinglish AI Trainer starting
) else (
    echo  [SKIP] Hinglish AI trainer not found at %HINGLISH_DIR%
)

:: Start Backend (cmd /k keeps window open if server crashes so you can see the error)
echo.
echo  Starting backend server on http://localhost:8000 ...
cd /d "%~dp0backend"
start "VoiceDub Backend" /min cmd /k "%PYTHON% -m uvicorn app:app --host 0.0.0.0 --port 8000"

:: Wait for backend to be ready
timeout /t 3 /nobreak >nul

:: Start Frontend
echo  Starting frontend on http://localhost:3000 ...
cd /d "%~dp0web"
if not exist node_modules (
    echo  Installing frontend dependencies...
    call npm install
)
start "VoiceDub Frontend" /min cmd /k "npm run dev"

:: Wait for frontend
timeout /t 5 /nobreak >nul

echo.
echo  ============================================
echo   VoiceDub is running!
echo.
echo   App:      http://localhost:3000
echo   Backend:  http://localhost:8000
echo   Trainer:  http://localhost:8100
echo.
echo   Close this window to stop all servers.
echo  ============================================
echo.

:: Open browser
start http://localhost:3000

:: Keep window open - when user closes it, kill servers
echo  Press any key to stop servers...
pause >nul

:: Cleanup - kill the server windows
taskkill /fi "WINDOWTITLE eq Hinglish AI Trainer" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq VoiceDub Backend" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq VoiceDub Frontend" /f >nul 2>&1
echo  Servers stopped. Goodbye!
