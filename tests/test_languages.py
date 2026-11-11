"""The language list of pavlogix-paddle comes from the PaddleOCR installed here, not from a copy.

These run against the real PaddleOCR (no model is loaded): if an upgrade renames what the list is
built from, they fail here instead of the Gateway offering languages nobody can read.
"""

from pavlogix_paddle.languages import LanguageCatalogue

V6 = "PP-OCRv6_medium_rec"


def test_one_model_reads_chinese_english_japanese_and_latin():
    cat = LanguageCatalogue.from_paddleocr()
    for code in ("ch", "chinese_cht", "en", "japan", "fr", "de"):
        assert cat.model_for(code) == V6, code


def test_other_scripts_have_their_own_models():
    cat = LanguageCatalogue.from_paddleocr()
    assert cat.model_for("ru") == "eslav_PP-OCRv5_mobile_rec"
    assert cat.model_for("korean") == "korean_PP-OCRv5_mobile_rec"
    assert cat.model_for("ar") == "arabic_PP-OCRv5_mobile_rec"


def test_a_code_paddleocr_does_not_know_has_no_model():
    assert LanguageCatalogue.from_paddleocr().model_for("klingon") is None


def test_models_list_every_language_under_its_model():
    cat = LanguageCatalogue.from_paddleocr()
    models = cat.models()
    assert {"ch", "en", "japan"} <= set(models[V6])
    for model, codes in models.items():
        assert all(cat.model_for(c) == model for c in codes)


def test_grouping_uses_whatever_the_resolver_says():
    table = {"a": "m1", "b": "m1", "c": "m2"}
    cat = LanguageCatalogue(["c", "b", "a", "x"], table.get)
    assert cat.models() == {"m1": ["a", "b"], "m2": ["c"]}
