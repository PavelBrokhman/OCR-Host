@echo off
rem Start the OCR server for one language: serve.bat LANG [PORT]
rem A server serves one language; another language is another start (or another port).
setlocal
cd /d "%~dp0"
if "%~1"=="" (
    echo usage: serve.bat LANG [PORT]
    exit /b 2
)
set LANG_CODE=%~1
set PORT=%~2
if "%PORT%"=="" set PORT=8081
if not exist configs mkdir configs
.venv\Scripts\python.exe make_config.py %LANG_CODE% configs\OCR-%LANG_CODE%.yaml || exit /b 1
.venv\Scripts\paddlex.exe --serve --pipeline configs\OCR-%LANG_CODE%.yaml --device gpu --host 127.0.0.1 --port %PORT%
