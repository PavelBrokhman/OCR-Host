@echo off
rem Start pavlogix-paddle, our own host: serve_pavlogix.bat [PORT]
rem Every language PaddleOCR reads, chosen per request; GET /info lists them by model.
setlocal
cd /d "%~dp0"
set PORT=%~1
if "%PORT%"=="" set PORT=8090
.venv\Scripts\python.exe -m pavlogix_paddle --port %PORT% --device gpu
