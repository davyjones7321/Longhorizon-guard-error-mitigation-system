# LongHorizon Guard - Gemini Proxy Launcher
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Starting LongHorizon Guard Proxy (Gemini Endpoint) " -ForegroundColor Green
Write-Host " Port: 8000" -ForegroundColor Yellow
Write-Host " Upstream: https://generativelanguage.googleapis.com/v1beta/openai/" -ForegroundColor Yellow
Write-Host "====================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot
python -m longhorizon_guard.cli proxy --port 8000 --upstream https://generativelanguage.googleapis.com/v1beta/openai/
