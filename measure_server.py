"""Time per frame through the running OCR server, HTTP included.

usage: measure_server.py CLIP [--url http://127.0.0.1:8081] [--frames N]

The same frames as measure.py (the sample's subtitle band, one every 2 s), sent as Base64 to the
server's POST /ocr, so the numbers and texts compare with the in-process ones.
"""

import argparse
import base64
import json
import tempfile
import time
import urllib.request
from pathlib import Path

from measure import WARM_UP, cut_frames


def post(url: str, png: Path) -> dict:
    body = json.dumps({"file": base64.b64encode(png.read_bytes()).decode(), "fileType": 1,
                       "visualize": False}).encode()
    req = urllib.request.Request(f"{url}/ocr", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--frames", type=int, default=20)
    a = ap.parse_args()

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "out") as tmp:
        frames = cut_frames(a.clip, a.frames, Path(tmp))
        for p in frames[:WARM_UP]:
            post(a.url, p)
        start = time.perf_counter()
        answers = [post(a.url, p) for p in frames]
        per_frame = (time.perf_counter() - start) / len(frames)

    print(f"server {a.url}, {len(frames)} frames: {per_frame:.3f} s per frame, HTTP included")
    first = answers[0]["result"]["ocrResults"][0]["prunedResult"]
    print("fields:", sorted(first))
    for i, ans in enumerate(answers):
        res = ans["result"]["ocrResults"][0]["prunedResult"]
        texts = res.get("rec_texts", [])
        scores = [round(s, 2) for s in res.get("rec_scores", [])]
        print(f"  {1 + i * 2:3d}s  {' / '.join(texts)}  {scores}")


if __name__ == "__main__":
    main()
