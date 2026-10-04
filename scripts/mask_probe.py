"""Render the clothing masks for a hand-authored pose.

The garment mask replaces DensePose + SCHP, so it is worth eyeballing on a
known skeleton before wiring up the GPU models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coverswap import clothing  # noqa: E402
from coverswap.detection import Person  # noqa: E402

W, H = 600, 900

# A plausible standing pose, in COCO-17 order.
POSE = {
    0: (300, 120),    # nose
    1: (288, 110), 2: (312, 110),
    3: (272, 118), 4: (328, 118),
    5: (250, 215), 6: (350, 215),    # shoulders
    7: (228, 320), 8: (372, 320),    # elbows
    9: (218, 425), 10: (382, 425),   # wrists
    11: (268, 440), 12: (332, 440),  # hips
    13: (262, 620), 14: (338, 620),  # knees
    15: (258, 800), 16: (342, 800),  # ankles
}


def build_person() -> tuple[Person, np.ndarray]:
    kps = np.zeros((17, 3), dtype=np.float32)
    for idx, (x, y) in POSE.items():
        kps[idx] = (x, y, 0.95)

    # A rough silhouette standing in for the SAM 2 mask.
    body = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(body, (300, 120), 58, 255, -1)
    cv2.fillPoly(body, [np.array([(232, 205), (368, 205), (352, 455), (248, 455)], np.int32)], 255)
    for a, b, t in (
        ((250, 215), (228, 320), 34), ((228, 320), (218, 425), 28),
        ((350, 215), (372, 320), 34), ((372, 320), (382, 425), 28),
        ((268, 440), (262, 620), 46), ((262, 620), (258, 800), 38),
        ((332, 440), (338, 620), 46), ((338, 620), (342, 800), 38),
    ):
        cv2.line(body, a, b, 255, t)
    cv2.line(body, (250, 215), (350, 215), 255, 40)

    person = Person(
        index=0, box=(190, 60, 410, 830), score=0.99, keypoints=kps,
        mask=body.astype(bool),
    )
    return person, body


def main() -> int:
    person, body = build_person()
    base = np.full((H, W, 3), 32, dtype=np.uint8)
    base[body > 0] = (108, 116, 132)

    panels = []
    for category in ("upper_body", "lower_body", "dress"):
        mask = clothing.garment_mask(person, (H, W), category=category, grow=10)
        covered = int(cv2.countNonZero(mask))
        inside = int(cv2.countNonZero(cv2.bitwise_and(mask, body)))
        leak = covered - inside
        print(f"{category:11s} covered={covered:6d}  outside_body={leak}", flush=True)

        panel = clothing.preview_overlay(base, mask)
        cv2.putText(panel, category, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(panel)

    out = ROOT / "outputs" / "mask_probe.png"
    cv2.imwrite(str(out), np.hstack(panels))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
