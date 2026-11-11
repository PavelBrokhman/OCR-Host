"""Write the PaddleX server configuration for one recognition language.

usage: make_config.py LANG OUT_YAML

The server takes its language when it starts, not per request, so a server serves one language.
Which models a language needs is PaddleOCR's own table (`lang` -> detection and recognition
models); PaddleOCR builds the pipeline and exports it, so that table is not copied here. The
options are the ones AI-Gateway's paddleocr provider uses: no page orientation, no unwarping,
no text-line orientation. Images are not sent back.
"""

import sys

from paddleocr import PaddleOCR

SERVING = "\nServing:\n  visualize: False\n"


def main() -> None:
    lang, out = sys.argv[1], sys.argv[2]
    ocr = PaddleOCR(lang=lang, enable_mkldnn=False, use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    ocr.export_paddlex_config_to_yaml(out)
    with open(out, "a", encoding="utf-8") as f:
        f.write(SERVING)
    print(out)


if __name__ == "__main__":
    main()
