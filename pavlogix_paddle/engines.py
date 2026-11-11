"""PaddleOCR engines, built on first use and kept: one per recognition model, not per language,
since languages that share a model share the engine (ch, en and japan are one PP-OCRv6 model).

Built the way AI-Gateway's in-process paddleocr provider builds them, so results compare: no page
orientation, no unwarping, no text-line orientation, oneDNN off. Each engine stays on the card
(about 1.2 GB for PP-OCRv6); nothing is unloaded yet.
"""

from __future__ import annotations

import cv2
import numpy as np


class PaddleEngines:
    def __init__(self, device: str):
        self._device = device
        self._readers: dict[str, object] = {}   # recognition model -> PaddleOCR
        self._detector = None

    def recognize(self, image: bytes, lang: str, model: str) -> list[dict]:
        reader = self._readers.get(model)
        if reader is None:
            from paddleocr import PaddleOCR
            reader = PaddleOCR(lang=lang, device=self._device, enable_mkldnn=False,
                               use_doc_orientation_classify=False, use_doc_unwarping=False,
                               use_textline_orientation=False)
            self._readers[model] = reader
        words = []
        for res in reader.predict(_to_array(image)):
            for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
                words.append({"text": text, "confidence": float(score), "box": poly.tolist()})
        return words

    def detect(self, image: bytes) -> list[list[list[float]]]:
        if self._detector is None:
            from paddleocr import TextDetection
            self._detector = TextDetection(device=self._device, enable_mkldnn=False)
        return [poly.tolist() for res in self._detector.predict(_to_array(image))
                for poly in res["dt_polys"]]


def _to_array(image: bytes) -> np.ndarray:
    array = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError("the image could not be decoded")
    return array
