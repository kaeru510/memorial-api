@echo off
chcp 65001 > nul
title LipSync AI Chat - Local Server Launcher
echo ============================================================
echo   🚀 LipSync AI Chat - ローカルサーバー一括起動
echo ============================================================

echo 🔊 [1/2] AivisSpeech 音声合成エンジンを起動中 (ポート 10101)...
start "AivisSpeech" "C:\dev\memorial-api\AivisSpeech-Engine-Windows-x64-1.2.0\Windows-x64\run.exe" --host 127.0.0.1 --port 10101

timeout /t 3 /nobreak > nul

echo ⚡ [2/2] FastAPI メインサーバーを起動中 (ポート 8000)...
set PYTHONUTF8=1
start "FastAPI (Uvicorn)" "C:\Users\yamada\anaconda3\envs\grave\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

echo.
echo ============================================================
echo   ✅ ローカルサーバーの起動が完了しました！
echo.
echo   👉 ブラウザ: http://localhost:8000
echo   🔑 Basic認証: ユーザー名=a / パスワード=a
echo.
echo   ※ 動画生成のために Google Colab 側のセルも実行してください。
echo ============================================================
pause
