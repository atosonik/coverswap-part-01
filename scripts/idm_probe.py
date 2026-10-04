"""Smoke test for IDM-VTON try-on.

Usage:
    .venv\\Scripts\\python.exe scripts\\download_idm_vton.py
    .venv\\Scripts\\python.exe scripts\\idm_probe.py [person.jpg] [garment.jpg] [category]

Writes outputs/idm_result.png and outputs/idm_compare.png
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from coverswap import config  # noqa: E402
from coverswap.idm_tryon import IDMVTONBackend  # noqa: E402

# Fallback demo images from IDM-VTON paper repo / common samples
DEFAULT_PERSON = ROOT / "samples" / "bus.jpg"
DEFAULT_GARMENT = ROOT / "samples" / "clothes" / "shirt_white.jpg"


def note(message: str) -> None:
    print(f"  {message}", flush=True)


def imread(path: Path) -> np.ndarray:
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        raise FileNotFoundError(path)
    return data


def imwrite(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


def main() -> int:
    person_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PERSON
    garment_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GARMENT
    category = sys.argv[3] if len(sys.argv) > 3 else "upper_body"

    if not person_path.exists():
        print(f"missing person photo: {person_path}", flush=True)
        print("Tip: run scripts/download_samples.py or pass your own image path.", flush=True)
        return 2
    if not garment_path.exists():
        print(f"missing garment photo: {garment_path}", flush=True)
        return 2

    person_bgr = imread(person_path)
    garment_bgr = imread(garment_path)
    person = Image.fromarray(person_bgr[..., ::-1])
    garment = Image.fromarray(garment_bgr[..., ::-1])

    print(f"person:  {person_path.name} {person.size[0]}x{person.size[1]}", flush=True)
    print(f"garment: {garment_path.name} {garment.size[0]}x{garment.size[1]}", flush=True)
    print(f"device:  {config.describe_device()}", flush=True)
    print(f"category: {category}", flush=True)

    backend = IDMVTONBackend(low_vram=True)
    steps = int(os.environ.get("IDM_STEPS", "30"))
    width = int(os.environ.get("IDM_WIDTH", "768"))
    height = int(os.environ.get("IDM_HEIGHT", "1024"))

    start = time.perf_counter()
    result = backend.run(
        person=person,
        garment=garment,
        category=category,
        garment_description=os.environ.get("IDM_GARMENT_DESC", "shirt"),
        steps=steps,
        width=width,
        height=height,
        progress=note,
    )
    elapsed = time.perf_counter() - start
    print(f"IDM-VTON took {elapsed:.1f}s", flush=True)

    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    imwrite(out / "idm_result.png", result_bgr)

    h = max(person_bgr.shape[0], result_bgr.shape[0], garment_bgr.shape[0])

    def fit(img):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    strip = np.hstack([fit(person_bgr), fit(result_bgr), fit(garment_bgr)])
    imwrite(out / "idm_compare.png", strip)

    print(f"wrote {out / 'idm_result.png'}", flush=True)
    print(f"wrote {out / 'idm_compare.png'}", flush=True)
    print("IDM_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
