@echo off
title LipSync AI Chat Launcher

echo ============================================================
echo   LipSync AI Chat Launcher
echo ============================================================

set "APP_DIR=C:\dev\memorial-api"
if not exist "C:\dev\memorial-api\main.py" set "APP_DIR=C:\Users\yamada\OneDrive - Shizuoka University\myfolder\memorial-api"

echo [1/3] Starting AivisSpeech Engine (Port 10101)...
start "AivisSpeech" /min "%APP_DIR%\AivisSpeech-Engine-Windows-x64-1.2.0\Windows-x64\run.exe" --host 127.0.0.1 --port 10101

timeout /t 3 /nobreak > nul

echo [2/3] Starting FastAPI Server (Port 8000)...
set PYTHONUTF8=1
start "FastAPI" /min /d "%APP_DIR%" C:\Users\yamada\anaconda3\envs\grave\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

timeout /t 3 /nobreak > nul

echo [3/3] Opening browser...
start http://localhost:8000

echo ============================================================
echo   LipSync AI Chat is running!
echo   Browser opened at http://localhost:8000
echo ============================================================
timeout /t 3 > nul
exit
