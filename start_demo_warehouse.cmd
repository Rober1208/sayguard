@echo off
rem SayGuard warehouse mode (11 synchronized cameras on one floor plan). Close other GPU-heavy programs first.
rem Needs the streams from scripts/fetch_smartspaces.py and scripts/make_site_streams.py (see docs/SITE_MODE.md).
set OLLAMA_MODELS=D:\ugen-work\ollama-models
set OLLAMA_HOST=127.0.0.1:11434
start "Ollama (LLM)" D:\ugen-work\tools\ollama\ollama.exe serve
cd /d "%~dp0"
rem the page reconnects by itself while the models load (~20 s)
start "" "http://127.0.0.1:8000/?lang=en"
D:\ugen-work\venv\Scripts\python.exe -m sayguard serve --config configs\warehouse.yaml
