"""Ties the four stages together.

detect -> mask -> try-on -> composite

Everything here is synchronous and UI-agnostic; the Qt layer runs it on a
worker thread and passes a `progress` callback through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import clothing, compositing, config
from . import background as bg
from . import faceswap
from .detection import Person, PersonDetector
from .tryon import CatVTONBackend, to_bgr, to_pil

Progress = Optional[Callable[[str], None]]


def imread(path: str | Path) -> np.ndarray:
    """Read an image as BGR, tolerating non-ASCII paths that break cv2.imread."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Not a readable image: {path}")
    return image


def imwrite(path: str | Path, image: np.ndarray) -> None:
    suffix = Path(path).suffix or ".png"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image as {suffix}")
    buffer.tofile(str(path))


class CoverSwapPipeline:
    def __init__(self, device: Optional[str] = None, low_vram: bool = True) -> None:
        self.device = device or config.resolve_device()
        self.detector = PersonDetector(device=self.device)
        self.backend = CatVTONBackend(device=self.device, low_vram=low_vram)

    # -- stage 1 ---------------------------------------------------------
    def detect_people(self, image: np.ndarray, progress: Progress = None) -> list[Person]:
        if progress:
            progress("Looking for people in the photo...")
        people = self.detector.detect(image)
        if progress:
            n = len(people)
            progress("Found 1 person." if n == 1 else f"Found {n} people. Click one to dress them.")
        return people

    # -- stage 2 ---------------------------------------------------------
    def clothing_mask(
        self, person: Person, image_shape: tuple[int, int], settings: config.TryOnSettings
    ) -> np.ndarray:
        return clothing.garment_mask(
            person, image_shape, category=settings.category, grow=settings.mask_grow
        )

    # -- stages 3 + 4 ----------------------------------------------------
    def swap_person(
        self,
        image: np.ndarray,
        person: Person,
        garment: np.ndarray,
        settings: config.TryOnSettings,
        progress: Progress = None,
    ) -> np.ndarray:
        """Return a full-size copy of `image` with one person re-dressed."""
        box = person.padded_box(image.shape)
        mask = self.clothing_mask(person, image.shape, settings)

        if cv2.countNonZero(mask) < 64:
            raise ValueError(
                f"{person.label} is too small or hidden. Try a clearer photo, or pick someone closer to the camera."
            )

        person_crop = compositing.crop(image, box)
        mask_crop = compositing.crop(mask, box)

        if progress:
            progress(f"Changing clothes for {person.label}… about 30 seconds.")

        result = self.backend.run(
            person=to_pil(person_crop),
            garment=to_pil(garment),
            mask=to_pil(cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)),
            settings=settings,
            progress=progress,
        )

        if progress:
            progress("Putting them back into the photo...")
        return compositing.paste_person(
            base=image,
            patch=to_bgr(result),
            box=box,
            person_mask=person.mask,
            garment_mask=mask,
            feather=settings.feather,
        )

    def warm_up(self, progress: Progress = None) -> None:
        """Pre-load the try-on model so the first swap is not the slow one."""
        self.backend.load(progress)

    # -- face swap -------------------------------------------------------
    def detect_faces(self, image: np.ndarray, progress: Progress = None):
        if progress:
            progress("Looking for faces in the photo...")
        faces = faceswap.detect(image)
        if progress:
            n = len(faces)
            progress(
                "Found 1 face." if n == 1 else f"Found {n} faces. Click one to replace."
            )
        return faces

    def swap_face(
        self,
        image: np.ndarray,
        face,
        source_face_image: np.ndarray,
        settings: config.FaceSwapSettings,
        progress: Progress = None,
    ) -> np.ndarray:
        if progress:
            progress("Swapping the face… a few seconds.")
        raw, mask, region = faceswap.swap_identity(image, face, source_face_image)
        return faceswap.finish(
            raw,
            image,
            mask,
            region,
            tone_match=settings.tone_match,
            blend=settings.blend,
        )

    def face_preview_mask(self, face, image_shape: tuple[int, int]) -> np.ndarray:
        mask = faceswap._face_mask(image_shape, face)
        return (mask * 255).astype(np.uint8)

    # -- background swap -------------------------------------------------
    def background_mask(self, people: list[Person], image_shape: tuple[int, int]) -> np.ndarray:
        fg = bg.foreground_mask(people, image_shape)
        return 255 - fg

    def swap_background(
        self,
        image: np.ndarray,
        people: list[Person],
        background_image: np.ndarray,
        settings: config.BackgroundSwapSettings,
        progress: Progress = None,
    ) -> np.ndarray:
        if progress:
            progress("Replacing the background…")
        return bg.replace_background(
            image,
            people,
            background_image,
            feather=settings.feather,
            edge_grow=settings.edge_grow,
        )
