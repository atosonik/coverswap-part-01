"""Face swap via InsightFace + inswapper_128 (ported from the swapping project).

Models live under weights/faceswap/. Nothing is fetched at runtime.
"""

from __future__ import annotations

import contextlib
import io
import os
import threading
import warnings
from dataclasses import dataclass

import cv2
import numpy as np

from . import config

# Must import onnxruntime before PyQt5 on Windows (see main.preload_onnxruntime).
import onnxruntime as _ort  # noqa: E402

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

_analyser = None
_swapper = None
_load_lock = threading.Lock()

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", message=r".*\bestimate\b.*deprecated.*")


@dataclass
class Face:
    """One detected face, canvas-compatible with Person (box + label)."""

    index: int
    box: tuple[int, int, int, int]
    raw: object
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"Face {self.index + 1}"


def missing_models() -> list[str]:
    gone = [
        f
        for f in config.FACESWAP_BUFFALO_FILES
        if not (config.FACESWAP_BUFFALO_DIR / f).is_file()
    ]
    if not config.FACESWAP_INSWAPPER.is_file():
        gone.append("inswapper_128.onnx")
    return gone


def ready() -> bool:
    return not missing_models()


def providers() -> list[str]:
    have = _ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in have]


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_analyser(det_size: int = 640):
    global _analyser
    if _analyser is not None:
        return _analyser
    gone = missing_models()
    if any(f.endswith(".onnx") and "inswapper" not in f for f in gone):
        raise FileNotFoundError(
            "Face models are missing. Run scripts/download_models.py while online."
        )

    from insightface.app import FaceAnalysis

    with _load_lock:
        if _analyser is not None:
            return _analyser
        with _quiet():
            _analyser = FaceAnalysis(
                name="buffalo_l",
                root=str(config.FACESWAP_IF_ROOT),
                allowed_modules=["detection", "recognition", "landmark_2d_106"],
                providers=providers(),
            )
            _analyser.prepare(ctx_id=0, det_size=(det_size, det_size))
    return _analyser


def load_swapper():
    global _swapper
    if _swapper is not None:
        return _swapper
    if not config.FACESWAP_INSWAPPER.is_file():
        raise FileNotFoundError(
            "inswapper_128.onnx is missing. Run scripts/download_models.py while online."
        )

    from insightface.model_zoo import get_model

    with _load_lock:
        if _swapper is not None:
            return _swapper
        with _quiet():
            _swapper = get_model(str(config.FACESWAP_INSWAPPER), providers=providers())
    return _swapper


def load(det_size: int = 640):
    return load_analyser(det_size), load_swapper()


def detect(img_bgr: np.ndarray) -> list[Face]:
    analyser = load_analyser()
    faces = sorted(analyser.get(img_bgr), key=lambda f: f.bbox[0])
    return [
        Face(i, tuple(int(v) for v in f.bbox), f)
        for i, f in enumerate(faces)
    ]


def face_at_point(faces: list[Face], x: int, y: int) -> Face | None:
    hits = [
        f
        for f in faces
        if f.box[0] <= x <= f.box[2] and f.box[1] <= y <= f.box[3]
    ]
    if hits:
        return min(hits, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
    return None


def pick_source_face(source_bgr: np.ndarray) -> Face:
    faces = detect(source_bgr)
    if not faces:
        raise ValueError("No face found in the replacement photo.")
    return max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))


def _face_mask(shape, face: Face, feather: float = 0.10) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    pts = getattr(face.raw, "landmark_2d_106", None)
    x1, y1, x2, y2 = face.box
    if pts is None:
        cv2.ellipse(
            mask,
            ((x1 + x2) // 2, (y1 + y2) // 2),
            ((x2 - x1) // 2, (y2 - y1) // 2),
            0,
            0,
            360,
            255,
            -1,
        )
    else:
        cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 255)
    k = int(max(3, (x2 - x1) * feather)) | 1
    return cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0


def _match_tone(
    swapped: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    if strength <= 0:
        return swapped
    sel = mask > 0.5
    if sel.sum() < 50:
        return swapped

    src = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src.copy()
    for c in range(3):
        s_mu, s_sd = src[..., c][sel].mean(), src[..., c][sel].std()
        r_mu, r_sd = ref[..., c][sel].mean(), ref[..., c][sel].std()
        if s_sd < 1e-4:
            continue
        out[..., c] = (src[..., c] - s_mu) * (r_sd / s_sd) + r_mu

    corrected = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    a = (mask * float(strength))[..., None]
    return (corrected * a + swapped * (1 - a)).astype(np.uint8)


def _bounds(flags: np.ndarray, pad: int = 8):
    rows, cols = np.any(flags, axis=1), np.any(flags, axis=0)
    if not rows.any():
        return None
    h, w = flags.shape
    y0, y1 = int(np.argmax(rows)), h - int(np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), w - int(np.argmax(cols[::-1]))
    return (
        max(0, y0 - pad),
        min(h, y1 + pad),
        max(0, x0 - pad),
        min(w, x1 + pad),
    )


def swap_identity(target_bgr: np.ndarray, target_face: Face, source_bgr: np.ndarray):
    model = load_swapper()
    src_face = pick_source_face(source_bgr)
    raw = model.get(target_bgr, target_face.raw, src_face.raw, paste_back=True)
    mask = _face_mask(target_bgr.shape, target_face)
    region = _bounds(np.any(raw != target_bgr, axis=2) | (mask > 0))
    return raw, mask, region


def finish(
    raw: np.ndarray,
    target_bgr: np.ndarray,
    mask: np.ndarray,
    region=None,
    tone_match: float = 0.7,
    blend: float = 1.0,
) -> np.ndarray:
    if region is None:
        region = _bounds(np.any(raw != target_bgr, axis=2) | (mask > 0))
    if region is None:
        return raw.copy()

    y0, y1, x0, x1 = region
    out = raw.copy()
    sub_raw, sub_tgt = raw[y0:y1, x0:x1], target_bgr[y0:y1, x0:x1]
    sub_mask = mask[y0:y1, x0:x1]

    sub = _match_tone(sub_raw, sub_tgt, sub_mask, tone_match)
    if blend < 1.0:
        a = float(blend)
        sub = (sub * a + sub_tgt * (1 - a)).astype(np.uint8)
    out[y0:y1, x0:x1] = sub
    return out


def swap(
    target_bgr: np.ndarray,
    target_face: Face,
    source_bgr: np.ndarray,
    tone_match: float = 0.7,
    blend: float = 1.0,
) -> np.ndarray:
    raw, mask, region = swap_identity(target_bgr, target_face, source_bgr)
    return finish(raw, target_bgr, mask, region, tone_match, blend)
