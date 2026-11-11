@echo off
rem OCR-Host: PaddleOCR on the GPU in its own environment.
rem
rem paddlepaddle on PyPI is a CPU build only. The GPU build is another package, paddlepaddle-gpu,
rem on Paddle's own index, one index per CUDA version. cu126 needs an NVIDIA driver of 550.54.14
rem or newer (PaddleOCR docs/version3.x/paddlepaddle_installation.en.md); the CUDA libraries come
rem with the package, no CUDA install is needed on the machine.
rem 3.3.1 and paddleocr 3.7.0 are what AI-Gateway's CPU install has, so results can be compared.
setlocal
cd /d "%~dp0"

set PADDLE=3.3.1
set INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu126/
set PADDLEOCR=3.7.0

if not exist .venv (
    py -3.13 -m venv .venv || exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
.venv\Scripts\python.exe -m pip install paddlepaddle-gpu==%PADDLE% -i %INDEX% || exit /b 1
.venv\Scripts\python.exe -m pip install paddleocr==%PADDLEOCR% || exit /b 1
.venv\Scripts\python.exe -m pip install fastapi uvicorn pytest || exit /b 1
.venv\Scripts\python.exe check_gpu.py
