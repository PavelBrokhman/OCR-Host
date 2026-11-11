"""usage: python -m pavlogix_paddle [--port 8090] [--device cpu|gpu:0]

Without --device the host takes the first card Paddle sees, else the CPU.
"""

import argparse
from importlib.metadata import version

import uvicorn

from .app import create_app
from .devices import Devices
from .engines import PaddleEngines
from .languages import LanguageCatalogue

DEFAULT_PORT = 8090
HOST = "127.0.0.1"


def main() -> None:
    ap = argparse.ArgumentParser(prog="pavlogix_paddle")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--device")
    a = ap.parse_args()

    devices = Devices.from_paddle()
    device = a.device or devices.default()
    versions = {"paddleocr": version("paddleocr"), "paddlex": version("paddlex"), "device": device}
    app = create_app(LanguageCatalogue.from_paddleocr(), PaddleEngines(device), versions,
                     devices.list())
    uvicorn.run(app, host=HOST, port=a.port)


if __name__ == "__main__":
    main()
