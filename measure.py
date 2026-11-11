"""Time per frame of PaddleOCR on this machine, the two passes VideoToSRT uses.

usage: measure.py CLIP [--device gpu|cpu] [--frames N] [--lang ch]

Cuts N frames of the subtitle band (the crop of the ocr-logo-left-right sample) out of CLIP with
ffmpeg, then times detection only (TextDetection) and recognition (PaddleOCR with the language),
built the way AI-Gateway's paddleocr provider builds them, so the numbers compare with the
2.7-4.6 s per recognized frame measured on the CPU.
"""

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

BAND = (260, 975, 1400, 105)  # x, y, w, h of the sample's subtitle band
WARM_UP = 2                   # first calls load models and kernels; not timed


def cut_frames(clip: str, n: int, out: Path) -> list[Path]:
    x, y, w, h = BAND
    paths = []
    for i in range(n):
        p = out / f"f{i:03d}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(1 + i * 2), "-i", clip,
                        "-frames:v", "1", "-vf", f"crop={w}:{h}:{x}:{y}", str(p)], check=True)
        paths.append(p)
    return paths


def timed(engine, frames: list[Path]) -> tuple[float, list]:
    for p in frames[:WARM_UP]:
        list(engine.predict(str(p)))
    results = []
    start = time.perf_counter()
    for p in frames:
        results.append(list(engine.predict(str(p))))
    return (time.perf_counter() - start) / len(frames), results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--lang", default="ch")
    a = ap.parse_args()

    from paddleocr import PaddleOCR, TextDetection

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "out") as tmp:
        frames = cut_frames(a.clip, a.frames, Path(tmp))
        det = TextDetection(device=a.device, enable_mkldnn=False)
        det_s, _ = timed(det, frames)
        rec = PaddleOCR(lang=a.lang, device=a.device, enable_mkldnn=False,
                        use_doc_orientation_classify=False, use_doc_unwarping=False,
                        use_textline_orientation=False)
        rec_s, results = timed(rec, frames)

    print(f"device {a.device}, {len(frames)} frames")
    print(f"detection only: {det_s:.3f} s per frame")
    print(f"recognition:    {rec_s:.3f} s per frame")
    for i, res in enumerate(results):
        texts = [t for r in res for t in r["rec_texts"]]
        print(f"  {1 + i * 2:3d}s  {' / '.join(texts)}")


if __name__ == "__main__":
    main()
