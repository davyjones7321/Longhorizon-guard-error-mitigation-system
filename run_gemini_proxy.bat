@echo off
title LongHorizon Guard - Gemini Proxy
echo ====================================================
echo  Starting LongHorizon Guard Proxy (Gemini Endpoint)
echo  Port: 8000
echo  Upstream: https://generativelanguage.googleapis.com/v1beta/openai/
echo ====================================================
cd /d "%~dp0"
python -m longhorizon_guard.cli proxy --port 8000 --upstream https://generativelanguage.googleapis.com/v1beta/openai/
pause
