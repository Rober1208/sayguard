@echo off
rem SayGuard warehouse mode (11 synchronized cameras on one floor plan). Close other GPU-heavy programs first.
rem Needs the streams from scripts/fetch_smartspaces.py and scripts/make_site_streams.py (see docs/SITE_MODE.md).
setlocal
cd /d "%~dp0"
if not defined SAYGUARD_WORK set "SAYGUARD_WORK=D:\ugen-work"
set "PY=%SAYGUARD_WORK%\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
set "OLLAMA=%SAYGUARD_WORK%\tools\ollama\ollama.exe"
if not exist "%OLLAMA%" set "OLLAMA=ollama"
if exist "%SAYGUARD_WORK%\ollama-models" set "OLLAMA_MODELS=%SAYGUARD_WORK%\ollama-models"
set "OLLAMA_HOST=127.0.0.1:11434"
start "Ollama (LLM)" "%OLLAMA%" serve
rem the page reconnects by itself while the models load (~20 s)
start "" "http://127.0.0.1:8000/?lang=en"
"%PY%" -m sayguard serve --config configs\warehouse.yaml %*
if errorlevel 1 pause
