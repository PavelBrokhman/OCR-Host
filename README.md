# OCR-Host

PaddleOCR as a service of its own, the way Ollama is one: its own process, its own Python
environment, the GPU. AI-Gateway talks to it over HTTP instead of loading PaddleOCR into its own
process (AI-Gateway `docs/design/connector-and-engine-hosts.md`, Jira GATEWAY-41). The Gateway
reaches it through the profiles `pavlogix-paddle` and `pavlogix-paddle-detect`.

The host is **pavlogix-paddle** (`pavlogix_paddle/`): every language PaddleOCR reads, chosen per
request, and detection on a route of its own. It uses PaddleOCR directly, in its own process;
PaddleOCR 3.x is built on PaddleX and released in step with it, so PaddleX works underneath as a
library. Named by the engine, not the function: text goes through PaddleOCR, and other Paddle
functions (documents, ...) can join it later.

## Run

```
serve_pavlogix.bat [PORT]    default 8090
```

| Route | |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /info` | versions and the device in use; the devices: `[{"id": "cpu"}, {"id": "gpu:0", "name", "compute", "memory_gb"}]`; the languages by recognition model: `{"functions": {"ocr": {"models": {"PP-OCRv6_medium_rec": ["ch", "en", "japan", ...], ...}}}}` |
| `POST /ocr` | `{"image": base64, "language": "ch"}` -> `{"language", "model", "words": [{"text", "confidence", "box"}]}` |
| `POST /ocr/detect` | `{"image": base64}` -> `{"boxes": [...]}` |

PaddleOCR picks the model by script, not locale: one PP-OCRv6 model reads Chinese, English,
Japanese and 48 Latin languages; Russian, Korean, Arabic... have their own. PaddleOCR has no
public list of this, so `languages.py` asks its language sets and its model-picking method, and
`tests/test_languages.py` fails if an upgrade moves them. An engine is built on first use and kept,
one per model (ch, en and japan share one), about 1.2 GB of the card each.

The host runs on the first card Paddle sees, else the CPU (`--device` picks one by hand). The
devices are the Paddle build's business, not PaddleOCR's: the CPU build sees no card, a
paddlepaddle-gpu build the cards its CUDA version and the driver allow. Choosing the build to fit
the machine is for the install, later (Jira GATEWAY-47). The same GPU build runs on the CPU too:
the three sample clips took 136 / 99 / 106 s there, with the same cues as on the card.

Tests: `.venv\Scripts\python.exe -m pytest` (the engines are faked; the language list is real).

## Measured 2026-11-11

RTX 2070 SUPER, clip `left` of `D:\Projects-Shared\Video\Samples\ocr-logo-left-right\`, its
subtitle band, `PP-OCRv6_medium` detection and recognition (what `lang=ch` gives):

| | detection | recognition |
|---|---|---|
| PaddleOCR in-process, GPU (`measure.py`, 20 frames) | 0.030 s/frame | 0.048 s/frame |
| PaddleOCR in-process, CPU, same venv | 0.51 s/frame | 1.30 s/frame |
| pavlogix-paddle, GPU, HTTP included, one client | 0.031 s/frame | 0.044 s/frame |

The three sample clips through VideoToSRT and the Gateway: 40 / 30 / 33 s, the same cues as on the
CPU.

**PaddleX's ready server** (`paddlex --serve`, a pipeline yaml exported by PaddleOCR) was tried
first and dropped: its language is fixed when it starts, it has no detection-only mode, and it
says nothing about which model it runs. Recognition through it took 0.062 s/frame.

## Install

```
install.bat
```

Makes `.venv` (Python 3.13) with `paddlepaddle-gpu` from Paddle's own CUDA index, `paddleocr`, the
web server and pytest, then runs `check_gpu.py`. The GPU build of Paddle is a separate package
from the CPU one on PyPI, one per CUDA version; `install.bat` names the version it takes and why.
