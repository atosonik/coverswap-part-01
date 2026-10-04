"""Background threads.

Detection takes a few seconds and a swap takes tens of seconds, so neither can
run on the GUI thread without freezing the window.

The result signal is named `completed` rather than `finished`, because QThread
already defines `finished` and shadowing it breaks the thread's own lifecycle
notifications.
"""

from __future__ import annotations

import traceback

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class _Worker(QThread):
    progress = pyqtSignal(str)
    failed = pyqtSignal(str)

    def _fail(self, exc: Exception) -> None:
        traceback.print_exc()
        self.failed.emit(str(exc) or exc.__class__.__name__)


class DetectWorker(_Worker):
    completed = pyqtSignal(object)

    def __init__(self, pipeline, image: np.ndarray, mode: str) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.image = image
        self.mode = mode

    def run(self) -> None:
        try:
            if self.mode == "face":
                found = self.pipeline.detect_faces(self.image, progress=self.progress.emit)
            else:
                found = self.pipeline.detect_people(self.image, progress=self.progress.emit)
            self.completed.emit(found)
        except Exception as exc:
            self._fail(exc)


class ClothesSwapWorker(_Worker):
    completed = pyqtSignal(object)

    def __init__(self, pipeline, image, person, garment, settings) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.image = image
        self.person = person
        self.garment = garment
        self.settings = settings

    def run(self) -> None:
        try:
            result = self.pipeline.swap_person(
                image=self.image,
                person=self.person,
                garment=self.garment,
                settings=self.settings,
                progress=self.progress.emit,
            )
            self.completed.emit(result)
        except Exception as exc:
            self._fail(exc)


class FaceSwapWorker(_Worker):
    completed = pyqtSignal(object)

    def __init__(self, pipeline, image, face, source_face, settings) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.image = image
        self.face = face
        self.source_face = source_face
        self.settings = settings

    def run(self) -> None:
        try:
            from .. import faceswap

            raw, mask, region = faceswap.swap_identity(
                self.image, self.face, self.source_face
            )
            result = faceswap.finish(
                raw,
                self.image,
                mask,
                region,
                tone_match=self.settings.tone_match,
                blend=self.settings.blend,
            )
            self.completed.emit(
                {"result": result, "raw": raw, "mask": mask, "region": region}
            )
        except Exception as exc:
            self._fail(exc)


class BackgroundSwapWorker(_Worker):
    completed = pyqtSignal(object)

    def __init__(self, pipeline, image, people, background, settings) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.image = image
        self.people = people
        self.background = background
        self.settings = settings

    def run(self) -> None:
        try:
            result = self.pipeline.swap_background(
                image=self.image,
                people=self.people,
                background_image=self.background,
                settings=self.settings,
                progress=self.progress.emit,
            )
            self.completed.emit(result)
        except Exception as exc:
            self._fail(exc)


# Backwards-compatible alias
SwapWorker = ClothesSwapWorker
