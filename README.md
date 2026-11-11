# OCR-Host

PaddleOCR as a service of its own, the way Ollama is one: its own process, its own Python
environment, the GPU. AI-Gateway talks to it over HTTP instead of loading PaddleOCR into its own
process (AI-Gateway `docs/design/connector-and-engine-hosts.md`, Jira GATEWAY-41).

Status: **trying the ready server first** - PaddleX's `paddlex --serve`. It has to give what the
Gateway's OCR contract needs: recognition with the language chosen per request, and detection only.
If it cannot, a small host of our own around `paddleocr` takes its place.

## Install

```
install.bat
```

Makes `.venv` (Python 3.13) with `paddlepaddle-gpu` from Paddle's own CUDA index, `paddleocr` and
PaddleX's serving plugin, then runs `check_gpu.py`. The GPU build of Paddle is a separate package
from the CPU one on PyPI, one per CUDA version; `install.bat` names the version it takes and why.
