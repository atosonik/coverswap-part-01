"""End-to-end check of stages 1, 2 and 4 on a real photo.

Runs detection, segmentation and mask building, then writes an annotated image.
The try-on model is not touched, so this stays fast and needs no large weights.

Usage: python scripts/detect_probe.py <image> [category]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coverswap import clothing, config  # noqa: E402
from coverswap.pipeline import CoverSwapPipeline, imread, imwrite  # noqa: E402

PALETTE = [
    (255, 140, 79), (142, 207, 62), (84, 180, 255),
    (199, 110, 255), (250, 139, 167), (225, 208, 77),
]


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "samples" / "bus.jpg"
    category = sys.argv[2] if len(sys.argv) > 2 else "upper_body"

    image = imread(image_path)
    print(f"image: {image_path.name} {image.shape[1]}x{image.shape[0]}", flush=True)

    pipeline = CoverSwapPipeline()
    print(f"device: {pipeline.device} ({config.describe_device()})", flush=True)

    start = time.perf_counter()
    people = pipeline.detect_people(image, progress=lambda m: print(f"  {m}", flush=True))
    print(f"detection took {time.perf_counter() - start:.1f}s", flush=True)

    if not people:
        print("NO PEOPLE FOUND", flush=True)
        return 1

    settings = config.TryOnSettings(category=category)
    annotated = image.copy()
    combined = np.zeros(image.shape[:2], dtype=np.uint8)

    for person in people:
        colour = PALETTE[person.index % len(PALETTE)]
        x1, y1, x2, y2 = person.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(annotated, person.label, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

        has_mask = person.mask is not None
        mask = pipeline.clothing_mask(person, image.shape, settings)
        combined = cv2.bitwise_or(combined, mask)

        kp_seen = int((person.keypoints[:, 2] >= 0.35).sum())
        print(
            f"  {person.label}: box={person.box} score={person.score:.2f} "
            f"sam_mask={'yes' if has_mask else 'NO'} keypoints={kp_seen}/17 "
            f"garment_px={int(cv2.countNonZero(mask))}",
            flush=True,
        )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    imwrite(out_dir / "detect_boxes.png", annotated)
    imwrite(out_dir / "detect_masks.png", clothing.preview_overlay(image, combined))

    # Show the crop that would actually be handed to the try-on model.
    biggest = max(people, key=lambda p: p.area)
    box = biggest.padded_box(image.shape)
    crop = image[box[1]:box[3], box[0]:box[2]]
    crop_mask = pipeline.clothing_mask(biggest, image.shape, settings)[box[1]:box[3], box[0]:box[2]]
    imwrite(out_dir / "detect_crop.png", clothing.preview_overlay(crop, crop_mask))
    print(f"largest person crop: {crop.shape[1]}x{crop.shape[0]} ({biggest.label})", flush=True)

    print("PROBE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
