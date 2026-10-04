"""Full pipeline check: detect -> mask -> try-on -> composite.

Downloads the try-on weights on first run, then swaps a garment onto one
person and writes the result plus a before/after strip.

Usage: python scripts/tryon_probe.py [photo] [garment] [category]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coverswap import config  # noqa: E402
from coverswap.pipeline import CoverSwapPipeline, imread, imwrite  # noqa: E402

CATVTON = ROOT / "third_party" / "CatVTON" / "resource" / "demo" / "example"
DEFAULT_PHOTO = ROOT / "samples" / "bus.jpg"
DEFAULT_GARMENT = CATVTON / "condition" / "upper" / "24083449_54173465_2048.jpg"


def note(message: str) -> None:
    print(f"  {message}", flush=True)


def main() -> int:
    photo = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PHOTO
    garment_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GARMENT
    category = sys.argv[3] if len(sys.argv) > 3 else "upper_body"

    if not photo.exists():
        print(f"missing photo: {photo}")
        return 2
    if not garment_path.exists():
        print(f"missing garment: {garment_path}")
        return 2

    image = imread(photo)
    garment = imread(garment_path)
    print(f"photo:   {photo.name} {image.shape[1]}x{image.shape[0]}", flush=True)
    print(f"garment: {garment_path.name} {garment.shape[1]}x{garment.shape[0]}", flush=True)

    pipeline = CoverSwapPipeline()
    print(f"device:  {config.describe_device()}", flush=True)

    people = pipeline.detect_people(image, progress=note)
    if not people:
        print("NO PEOPLE FOUND")
        return 1

    # The biggest person has the most pixels for the model to work with.
    person = max(people, key=lambda p: p.area)
    print(f"target:  {person.label} box={person.box}", flush=True)

    settings = config.TryOnSettings(
        category=category,
        steps=int(os.environ.get("PROBE_STEPS", 30)),
        guidance_scale=float(os.environ.get("PROBE_CFG", 4.0)),
    )
    print(f"settings: {settings.width}x{settings.height} steps={settings.steps} "
          f"cfg={settings.guidance_scale} category={settings.category}", flush=True)

    start = time.perf_counter()
    result = pipeline.swap_person(image, person, garment, settings, progress=note)
    print(f"swap took {time.perf_counter() - start:.1f}s", flush=True)

    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    imwrite(out / "tryon_result.png", result)

    # Side-by-side of just the affected person, which is easier to judge.
    box = person.padded_box(image.shape)
    before = image[box[1]:box[3], box[0]:box[2]]
    after = result[box[1]:box[3], box[0]:box[2]]
    height = max(before.shape[0], after.shape[0])

    def fit(img):
        scale = height / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), height))

    ref = cv2.resize(garment, (int(garment.shape[1] * height / garment.shape[0]), height))
    strip = np.hstack([fit(before), fit(after), ref])
    imwrite(out / "tryon_compare.png", strip)

    print(f"wrote {out / 'tryon_result.png'}", flush=True)
    print(f"wrote {out / 'tryon_compare.png'}", flush=True)
    print("TRYON_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
