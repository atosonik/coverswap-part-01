"""Background replacement: keep detected people, swap everything else."""

from __future__ import annotations

import cv2
import numpy as np

from .detection import Person


def fit_cover(background: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale and center-crop a background to exactly cover the target frame."""
    bh, bw = background.shape[:2]
    scale = max(width / bw, height / bh)
    new_w, new_h = int(bw * scale), int(bh * scale)
    resized = cv2.resize(background, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    x0 = max(0, (new_w - width) // 2)
    y0 = max(0, (new_h - height) // 2)
    return resized[y0 : y0 + height, x0 : x0 + width].copy()


def foreground_mask(
    people: list[Person],
    shape: tuple[int, int, int],
    grow: int = 8,
) -> np.ndarray:
    """Union of all person silhouettes, slightly expanded for hair and edges."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for person in people:
        if person.mask is not None:
            mask = np.maximum(mask, person.mask.astype(np.uint8) * 255)
    if grow > 0 and cv2.countNonZero(mask) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
        mask = cv2.dilate(mask, k)
    return mask


def preview_overlay(base: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Tint the area that will be replaced so the user can check the cutout."""
    out = base.copy()
    replace = fg_mask < 32
    tint = out.copy()
    tint[replace] = (
        tint[replace].astype(np.float32) * 0.55 + np.array([40, 90, 200], dtype=np.float32) * 0.45
    ).astype(np.uint8)
    return tint


def replace_background(
    image: np.ndarray,
    people: list[Person],
    background: np.ndarray,
    feather: int = 15,
    edge_grow: int = 8,
) -> np.ndarray:
    """Paste all detected people onto a new background."""
    h, w = image.shape[:2]
    if not people:
        raise ValueError("No people detected — cannot separate foreground from background.")

    fg = foreground_mask(people, image.shape, grow=edge_grow)
    if cv2.countNonZero(fg) < 64:
        raise ValueError("Could not outline anyone clearly enough to swap the background.")

    bg = fit_cover(background, w, h)
    alpha = fg.astype(np.float32) / 255.0
    if feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    out = (image.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )
    return out
