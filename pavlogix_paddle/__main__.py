"""usage: python -m pavlogix_paddle [--port 8090] [--device gpu]"""

import argparse
from importlib.metadata import version

import uvicorn

from .app import create_app
from .engines import PaddleEngines
from .languages import LanguageCatalogue

DEFAULT_PORT = 8090
HOST = "127.0.0.1"


def main() -> None:
    ap = argparse.ArgumentParser(prog="pavlogix_paddle")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--device", default="gpu")
    a = ap.parse_args()

    versions = {"paddleocr": version("paddleocr"), "paddlex": version("paddlex"), "device": a.device}
    app = create_app(LanguageCatalogue.from_paddleocr(), PaddleEngines(a.device), versions)
    uvicorn.run(app, host=HOST, port=a.port)


if __name__ == "__main__":
    main()
