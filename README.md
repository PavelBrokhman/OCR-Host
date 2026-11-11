# OCR-Host

PaddleOCR as a service of its own, the way Ollama is one: its own process, its own Python
environment, the GPU. AI-Gateway talks to it over HTTP instead of loading PaddleOCR into its own
process (AI-Gateway `docs/design/connector-and-engine-hosts.md`, Jira GATEWAY-41).

Two kinds of host, each reached by the Gateway through a profile of its own (Pavel, 2026-11-11):

- **PaddleX's ready server** (`paddlex --serve`) - here now. It takes its language when it starts,
  so a server serves one language; several servers, on this machine or another, are normal.
- **A host of our own** with the language chosen per request - later; its name is not chosen yet.

## The ready server, measured 2026-11-11

RTX 2070 SUPER, clip `left` of `D:\Projects-Shared\Video\Samples\ocr-logo-left-right\`, its
subtitle band, 20 frames, `PP-OCRv6_medium` detection and recognition (what `lang=ch` gives):

| | detection | recognition |
|---|---|---|
| PaddleOCR in-process, GPU (`measure.py`) | 0.030 s/frame | 0.048 s/frame |
| PaddleOCR in-process, CPU, same venv | 0.51 s/frame | 1.30 s/frame |
| the ready server, GPU, HTTP included (`measure_server.py`) | - | 0.062 s/frame |

The server gave the same texts as in-process. Its answer has `rec_texts`, `rec_scores`, `rec_polys`
and `dt_polys`; it has no detection-only mode, but `dt_polys` of the same answer are the boxes,
and on the GPU full OCR costs only ~0.02 s/frame more than detection. One server takes about
1.2 GB of the card.

## Run

```
serve.bat LANG [PORT]      e.g. serve.bat ch 8081
```

`make_config.py` has PaddleOCR export the server's configuration for the language, so which models a
language needs stays PaddleOCR's table, not a copy here.

## Install

```
install.bat
```

Makes `.venv` (Python 3.13) with `paddlepaddle-gpu` from Paddle's own CUDA index, `paddleocr` and
PaddleX's serving plugin, then runs `check_gpu.py`. The GPU build of Paddle is a separate package
from the CPU one on PyPI, one per CUDA version; `install.bat` names the version it takes and why.
