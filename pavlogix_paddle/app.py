"""The HTTP side: /health, /info, /ocr, /ocr/detect.

The image travels as base64 in JSON. The language is per request;
the language code goes back as given - turning it into ISO for translation is the Gateway's job.
"""

from __future__ import annotations

import base64
import binascii
from typing import Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .languages import LanguageCatalogue


class Engines(Protocol):
    def recognize(self, image: bytes, lang: str, model: str) -> list[dict]: ...
    def detect(self, image: bytes) -> list[list[list[float]]]: ...


class OcrRequest(BaseModel):
    image: str      # base64
    language: str   # a PaddleOCR code, e.g. "ch"


class DetectRequest(BaseModel):
    image: str


def _decode(image: str) -> bytes:
    try:
        return base64.b64decode(image, validate=True)
    except binascii.Error as e:
        raise HTTPException(400, f"image is not base64: {e}") from e


def create_app(catalogue: LanguageCatalogue, engines: Engines, versions: dict) -> FastAPI:
    app = FastAPI(title="pavlogix-paddle")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info():
        return {"engine": versions,
                "functions": {"ocr": {"models": catalogue.models(), "detection": True}}}

    @app.post("/ocr")
    def ocr(req: OcrRequest):
        model = catalogue.model_for(req.language)
        if model is None:
            raise HTTPException(400, f"no model reads {req.language!r} here; see GET /info")
        words = engines.recognize(_decode(req.image), req.language, model)
        return {"language": req.language, "model": model, "words": words}

    @app.post("/ocr/detect")
    def detect(req: DetectRequest):
        return {"boxes": engines.detect(_decode(req.image))}

    return app
