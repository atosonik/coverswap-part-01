"""Stage 2: work out which pixels on a person are clothing.

CatVTON needs a binary mask marking the region to repaint. Upstream that mask
comes from DensePose + SCHP, which are painful to build on Windows. Body
keypoints already describe the torso and limbs well enough to draw the same
region, and intersecting the result with the SAM 2 silhouette keeps it from
bleeding onto the background or a neighbouring family member.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detection import (
    KP_L_ANKLE,
    KP_L_ELBOW,
    KP_L_HIP,
    KP_L_KNEE,
    KP_L_SHOULDER,
    KP_L_WRIST,
    KP_NOSE,
    KP_R_ANKLE,
    KP_R_ELBOW,
    KP_R_HIP,
    KP_R_KNEE,
    KP_R_SHOULDER,
    KP_R_WRIST,
    Person,
)

CONF = 0.35


def _pt(kps: np.ndarray, idx: int) -> tuple[int, int] | None:
    if kps[idx, 2] < CONF:
        return None
    return int(round(kps[idx, 0])), int(round(kps[idx, 1]))


def _mid(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def _lerp(a, b, t: float):
    """Point `t` of the way from a to b."""
    if a is None or b is None:
        return None
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t))


def _limb(canvas: np.ndarray, a, b, thickness: int) -> None:
    if a is None or b is None:
        return
    cv2.line(canvas, a, b, 255, thickness=max(1, thickness), lineType=cv2.LINE_AA)
    cv2.circle(canvas, b, max(1, thickness // 2), 255, -1)


def body_scale(person: Person) -> float:
    """A rough torso width used to size every stroke, so the mask adapts to
    both an adult in the foreground and a child further back."""
    kps = person.keypoints
    ls, rs = _pt(kps, KP_L_SHOULDER), _pt(kps, KP_R_SHOULDER)
    if ls and rs:
        span = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
        if span > 4:
            return span
    return person.width * 0.55


# How far past the body landmarks each garment type is allowed to reach.
# Horizontal expansion is deliberately aggressive: real clothing is wider than
# the skeleton (a bulky coat especially), and the SAM silhouette trims anything
# that would spill onto the background or a neighbour. Being timid here is what
# makes a try-on model paint onto the old garment instead of replacing it.
SPREAD = {
    #                 sideways  lengthwise  extra below the hem
    "upper_body": (1.95, 1.22, 0.06),
    "lower_body": (1.75, 1.12, 0.04),
    "dress": (1.95, 1.18, 0.10),
}


def _expand_quad(points, sx: float, sy: float, extra_down: float) -> np.ndarray:
    """Grow a torso quad outward, with separate sideways and lengthwise scales."""
    poly = np.array(points, dtype=np.float32)
    centre = poly.mean(axis=0)
    poly[:, 0] = centre[0] + (poly[:, 0] - centre[0]) * sx
    poly[:, 1] = centre[1] + (poly[:, 1] - centre[1]) * sy
    if extra_down > 0:
        span = float(poly[:, 1].max() - poly[:, 1].min())
        lower = poly[:, 1] > centre[1]
        poly[lower, 1] += span * extra_down
    return poly.astype(np.int32)


def _waist_y(person: Person) -> int | None:
    lh = _pt(person.keypoints, KP_L_HIP)
    rh = _pt(person.keypoints, KP_R_HIP)
    hips = _mid(lh, rh)
    return hips[1] if hips else None


def _silhouette_hem(sil: np.ndarray, start_y: int, min_width: float) -> int:
    """Walk down from the waist and stop where the body silhouette narrows.

    A bulky coat keeps a wide profile well below the hips; a tucked shirt
    does not. That is how we tell those two cases apart without a parser.
    """
    h = sil.shape[0]
    widths = sil.sum(axis=1)
    hem = start_y
    for y in range(start_y, h):
        if widths[y] >= min_width:
            hem = y
        elif y > start_y + 8 and widths[y] < min_width * 0.45:
            break
    return hem


def _from_silhouette(person: Person, image_shape: tuple[int, int], category: str) -> np.ndarray | None:
    """Prefer the SAM outline when it is available.

    The skeleton alone is sized to the body, so it misses coats, hoodies and
    anything else that stands away from the torso. The silhouette already
    includes those; we only have to cut the head and the part of the body
    that belongs to a different garment.
    """
    if person.mask is None:
        return None

    h, w = image_shape[:2]
    sil = person.mask.astype(np.uint8) * 255
    if sil.shape[:2] != (h, w):
        sil = cv2.resize(sil, (w, h), interpolation=cv2.INTER_NEAREST)

    waist = _waist_y(person)
    if waist is None:
        waist = person.box[1] + int(person.height * 0.52)

    canvas = sil.copy()
    if category == "upper_body":
        widths = sil.sum(axis=1)
        waist_w = float(widths[min(h - 1, max(0, waist))])
        # If the outline is still wide below the hips, that extra volume is
        # clothing (a coat hem), not the torso, and must be replaced too.
        hem = _silhouette_hem(sil, waist, max(8.0, waist_w * 0.55))
        lk = _pt(person.keypoints, KP_L_KNEE)
        rk = _pt(person.keypoints, KP_R_KNEE)
        knees = _mid(lk, rk)
        if knees is not None:
            # Mid-thigh is as far as an upper garment is allowed to reach.
            # Without this cap a bulky coat's silhouette never narrows, and
            # the mask walks down the jeans.
            hem = min(hem, int(waist + 0.55 * (knees[1] - waist)))
        canvas[hem + 1 :, :] = 0
    elif category == "lower_body":
        canvas[: max(0, waist - 4), :] = 0

    return canvas


def garment_mask(
    person: Person,
    image_shape: tuple[int, int],
    category: str = "upper_body",
    grow: int = 12,
) -> np.ndarray:
    """Build a full-image uint8 mask (255 = repaint) for one person.

    `grow` is a percentage of the person's shoulder width rather than a pixel
    count, so the same setting behaves consistently for a parent in the
    foreground and a child further back.
    """
    scale = body_scale(person)
    canvas = _from_silhouette(person, image_shape, category)
    if canvas is None:
        canvas = _from_skeleton(person, image_shape, category)

    grow_px = int(scale * grow / 100.0)
    if grow_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_px * 2 + 1, grow_px * 2 + 1))
        canvas = cv2.dilate(canvas, k)

    if person.mask is not None:
        h, w = image_shape[:2]
        sil = person.mask.astype(np.uint8) * 255
        if sil.shape[:2] != (h, w):
            sil = cv2.resize(sil, (w, h), interpolation=cv2.INTER_NEAREST)
        canvas = cv2.bitwise_and(canvas, sil)

    canvas = _protect_head(canvas, person, scale, category)
    canvas = cv2.morphologyEx(
        canvas, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    return canvas


def _from_skeleton(person: Person, image_shape: tuple[int, int], category: str) -> np.ndarray:
    """Fallback when SAM did not produce a silhouette."""
    h, w = image_shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    kps = person.keypoints
    scale = body_scale(person)
    sx, sy, extra_down = SPREAD.get(category, SPREAD["upper_body"])

    ls, rs = _pt(kps, KP_L_SHOULDER), _pt(kps, KP_R_SHOULDER)
    lh, rh = _pt(kps, KP_L_HIP), _pt(kps, KP_R_HIP)

    if category == "lower_body":
        top_l, top_r = _lerp(lh, ls, 0.34), _lerp(rh, rs, 0.34)
    else:
        top_l, top_r = ls, rs
    torso = [p for p in (top_l, top_r, rh, lh) if p is not None]

    if len(torso) >= 3:
        cv2.fillPoly(canvas, [_expand_quad(torso, sx, sy, extra_down)], 255)
    else:
        x1, y1, x2, y2 = person.box
        if category == "lower_body":
            cv2.rectangle(canvas, (x1, int(y1 + 0.45 * person.height)), (x2, y2), 255, -1)
        elif category == "dress":
            cv2.rectangle(canvas, (x1, int(y1 + 0.15 * person.height)), (x2, int(y1 + 0.85 * person.height)), 255, -1)
        else:
            cv2.rectangle(canvas, (x1, int(y1 + 0.13 * person.height)), (x2, int(y1 + 0.58 * person.height)), 255, -1)

    if category in ("upper_body", "dress"):
        arm = max(2, int(scale * 0.46))
        _limb(canvas, ls, _pt(kps, KP_L_ELBOW), arm)
        _limb(canvas, rs, _pt(kps, KP_R_ELBOW), arm)
        _limb(canvas, _pt(kps, KP_L_ELBOW), _pt(kps, KP_L_WRIST), int(arm * 0.8))
        _limb(canvas, _pt(kps, KP_R_ELBOW), _pt(kps, KP_R_WRIST), int(arm * 0.8))
        _limb(canvas, ls, rs, max(2, int(scale * 0.42)))

    if category in ("lower_body", "dress"):
        leg = max(2, int(scale * 0.52))
        _limb(canvas, lh, _pt(kps, KP_L_KNEE), leg)
        _limb(canvas, rh, _pt(kps, KP_R_KNEE), leg)
        _limb(canvas, lh, rh, leg)
        if category == "lower_body":
            _limb(canvas, _pt(kps, KP_L_KNEE), _pt(kps, KP_L_ANKLE), int(leg * 0.72))
            _limb(canvas, _pt(kps, KP_R_KNEE), _pt(kps, KP_R_ANKLE), int(leg * 0.72))
        else:
            _limb(canvas, _pt(kps, KP_L_KNEE), _pt(kps, KP_L_ANKLE), int(leg * 0.55))
            _limb(canvas, _pt(kps, KP_R_KNEE), _pt(kps, KP_R_ANKLE), int(leg * 0.55))
    return canvas


def _protect_head(mask: np.ndarray, person: Person, scale: float, category: str) -> np.ndarray:
    """Carve the face and hair back out.

    Identity preservation is the thing people notice first, so it is worth
    guaranteeing rather than hoping the diffusion model leaves the face alone.
    """
    kps = person.keypoints
    nose = _pt(kps, KP_NOSE)
    ls, rs = _pt(kps, KP_L_SHOULDER), _pt(kps, KP_R_SHOULDER)
    neck = _mid(ls, rs)

    if nose is None:
        x1, y1, _, _ = person.box
        head_bottom = int(y1 + person.height * 0.16)
        mask[: max(0, head_bottom), :] = np.where(
            mask[: max(0, head_bottom), :] > 0, 0, mask[: max(0, head_bottom), :]
        )
        return mask

    # Big enough to guarantee the face survives, tight enough to leave the
    # collar repaintable - if the neckline stays masked off, the old garment's
    # collar shows through the new one.
    radius = int(max(scale * 0.54, 10))
    centre_y = nose[1]
    if neck is not None:
        centre_y = int(nose[1] + (neck[1] - nose[1]) * 0.12)
    cv2.circle(mask, (nose[0], centre_y), radius, 0, -1)
    # Hats and hair sit above the face circle. Leaving them in the mask makes
    # the model invent a new cap or smear the forehead.
    hat_cut = max(0, centre_y - int(radius * 0.55))
    x1, _, x2, _ = person.box
    pad = int(scale * 0.25)
    mask[:hat_cut, max(0, x1 - pad) : x2 + pad] = 0
    return mask


def preview_overlay(image: np.ndarray, mask: np.ndarray, colour=(80, 190, 255), alpha=0.45) -> np.ndarray:
    """Tint the masked region for the UI preview."""
    out = image.copy()
    tint = np.zeros_like(out)
    tint[:] = colour
    sel = mask > 0
    out[sel] = (out[sel] * (1 - alpha) + tint[sel] * alpha).astype(np.uint8)
    return out
