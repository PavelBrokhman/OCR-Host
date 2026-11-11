"""Which recognition model reads which language - asked of the PaddleOCR installed here.

PaddleOCR picks a model by the script, not the locale: one PP-OCRv6 model reads Chinese, English,
Japanese and the Latin languages, Russian has its own, and so on. It has no public list of this;
the knowledge is its language sets (`paddleocr._utils.langs`) and the method that maps a code to
model names. Both are asked here, so the list follows the installed version instead of a copy
going stale (VideOCR keeps its own copy and still maps to PP-OCRv5). tests/test_languages.py
fails if an upgrade moves them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

# The codes PaddleOCR names only inside its model-picking method, not in a language set.
SINGLE_MODEL_LANGS = ("ch", "chinese_cht", "en", "japan", "korean", "th", "el", "te", "ta", "ka")


class LanguageCatalogue:
    def __init__(self, codes: Iterable[str], resolve: Callable[[str], str | None]):
        self._model = {}
        for code in sorted(set(codes)):
            model = resolve(code)
            if model is not None:
                self._model[code] = model

    @classmethod
    def from_paddleocr(cls) -> "LanguageCatalogue":
        from paddleocr import PaddleOCR
        from paddleocr._utils import langs

        sets = [v for v in vars(langs).values() if isinstance(v, frozenset)]
        codes = set(SINGLE_MODEL_LANGS).union(*sets)

        def resolve(code: str) -> str | None:
            # Uses nothing of the instance; called unbound so no engine is built.
            _, rec = PaddleOCR._get_ocr_model_names(None, code, None)
            return rec

        return cls(codes, resolve)

    def model_for(self, code: str) -> str | None:
        return self._model.get(code)

    def models(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for code, model in self._model.items():
            grouped.setdefault(model, []).append(code)
        return grouped
