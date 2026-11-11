"""The HTTP side of pavlogix-paddle. The engines are faked; the language list is a small fixed one."""

import base64

from fastapi.testclient import TestClient

from pavlogix_paddle.app import create_app
from pavlogix_paddle.languages import LanguageCatalogue

TABLE = {"ch": "rec-v6", "en": "rec-v6", "ru": "rec-eslav"}
BOX = [[0, 0], [10, 0], [10, 5], [0, 5]]
IMAGE = base64.b64encode(b"fake-png").decode("ascii")


class FakeEngines:
    def __init__(self):
        self.recognized = []

    def recognize(self, image: bytes, lang: str, model: str) -> list[dict]:
        self.recognized.append((image, lang, model))
        return [{"text": "拜托了啊", "confidence": 0.99, "box": BOX}]

    def detect(self, image: bytes) -> list[list[list[float]]]:
        return [BOX]


DEVICES = [{"id": "cpu"}, {"id": "gpu:0", "name": "RTX", "compute": "7.5", "memory_gb": 8.0}]


def _client(engines=None):
    catalogue = LanguageCatalogue(list(TABLE), TABLE.get)
    versions = {"paddleocr": "3.7.0", "paddlex": "3.7.2", "device": "gpu:0"}
    return TestClient(create_app(catalogue, engines or FakeEngines(), versions, DEVICES))


def test_health():
    assert _client().get("/health").json() == {"status": "ok"}


def test_info_lists_the_languages_under_their_models():
    info = _client().get("/info").json()
    assert info["engine"] == {"paddleocr": "3.7.0", "paddlex": "3.7.2", "device": "gpu:0"}
    assert info["functions"]["ocr"]["models"] == {"rec-v6": ["ch", "en"], "rec-eslav": ["ru"]}
    assert info["functions"]["ocr"]["detection"] is True


def test_info_lists_the_devices_the_host_can_use():
    assert _client().get("/info").json()["devices"] == DEVICES


def test_ocr_reads_with_the_model_of_the_language():
    engines = FakeEngines()
    r = _client(engines).post("/ocr", json={"image": IMAGE, "language": "en"})

    assert r.status_code == 200
    assert r.json() == {"language": "en", "model": "rec-v6",
                        "words": [{"text": "拜托了啊", "confidence": 0.99, "box": BOX}]}
    assert engines.recognized == [(b"fake-png", "en", "rec-v6")]


def test_a_language_with_no_model_is_refused_by_name():
    r = _client().post("/ocr", json={"image": IMAGE, "language": "klingon"})
    assert r.status_code == 400
    assert "klingon" in r.json()["detail"]


def test_an_image_that_is_not_base64_is_refused():
    r = _client().post("/ocr", json={"image": "not base64!", "language": "ch"})
    assert r.status_code == 400


def test_detect_returns_the_boxes():
    r = _client().post("/ocr/detect", json={"image": IMAGE})
    assert r.status_code == 200
    assert r.json() == {"boxes": [BOX]}
