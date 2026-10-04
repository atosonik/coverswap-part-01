"""Stage 4: put a swapped person back into the family photo.

The try-on model only ever sees a single-person crop, so the last job is
returning that crop to the group shot without an obvious seam or a colour
mismatch against the rest of the photo.
"""

from __future__ import annotations

import cv2
import numpy as np


def crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].copy()


def blend_region(
    base: np.ndarray,
    patch: np.ndarray,
    box: tuple[int, int, int, int],
    mask: np.ndarray,
    feather: int = 9,
) -> np.ndarray:
    """Alpha-blend `patch` into `base` at `box`, softening the mask edge.

    Feathering matters more than it sounds: a hard-edged paste makes even a
    perfect generation look like a sticker.
    """
    x1, y1, x2, y2 = box
    target_h, target_w = y2 - y1, x2 - x1
    if patch.shape[:2] != (target_h, target_w):
        patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    region_mask = mask[y1:y2, x1:x2]
    if region_mask.shape != (target_h, target_w):
        region_mask = cv2.resize(region_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    alpha = region_mask.astype(np.float32) / 255.0
    if feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    out = base.copy()
    region = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (region * (1 - alpha) + patch.astype(np.float32) * alpha).astype(np.uint8)
    return out


def seamless_region(
    base: np.ndarray,
    patch: np.ndarray,
    box: tuple[int, int, int, int],
    mask: np.ndarray,
) -> np.ndarray:
    """Poisson-blend the patch so its lighting matches the surrounding photo.

    Falls back to a feathered alpha blend, because seamlessClone throws when
    the mask touches the border of the region or is empty.
    """
    x1, y1, x2, y2 = box
    target_h, target_w = y2 - y1, x2 - x1
    if patch.shape[:2] != (target_h, target_w):
        patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    region_mask = mask[y1:y2, x1:x2]
    if region_mask.shape != (target_h, target_w):
        region_mask = cv2.resize(region_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # seamlessClone needs a margin of zeros around the mask.
    safe = np.zeros_like(region_mask)
    safe[2:-2, 2:-2] = region_mask[2:-2, 2:-2]
    if cv2.countNonZero(safe) < 32:
        return blend_region(base, patch, box, mask)

    ys, xs = np.nonzero(safe)
    centre = (
        int(x1 + (xs.min() + xs.max()) / 2),
        int(y1 + (ys.min() + ys.max()) / 2),
    )
    try:
        return cv2.seamlessClone(patch, base, safe, centre, cv2.NORMAL_CLONE)
    except cv2.error:
        return blend_region(base, patch, box, mask)


def paste_person(
    base: np.ndarray,
    patch: np.ndarray,
    box: tuple[int, int, int, int],
    person_mask: np.ndarray | None,
    garment_mask: np.ndarray,
    feather: int = 9,
    grow: int = 6,
) -> np.ndarray:
    """Composite one finished person back into the full photo.

    The paste region is the garment area plus a little slack, since a looser
    garment can legitimately extend past the original silhouette; clipping
    strictly to the old outline would shave it off.
    """
    region = garment_mask.copy()
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
        region = cv2.dilate(region, k)

    if person_mask is not None:
        # Allow spill just beyond the body, but not across the whole frame.
        body = person_mask.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 4 + 1, grow * 4 + 1))
        region = cv2.bitwise_and(region, cv2.dilate(body, k))

    return blend_region(base, patch, box, region, feather=feather)
