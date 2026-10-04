"""Stage 1: find every person in a photo and cut them out cleanly.

YOLO-pose gives boxes and body keypoints in a single pass, and those boxes are
then used as prompts for SAM 2 to get pixel-accurate silhouettes. Both models
come from the `ultralytics` package, which avoids compiling any CUDA extensions
- the usual blocker for DensePose/Detectron2 and the official SAM 2 repo on
Windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config

# COCO-17 keypoint ordering used by every ultralytics pose model.
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16


@dataclass
class Person:
    """One detected person, in original-image coordinates."""

    index: int
    box: tuple[int, int, int, int]
    score: float
    keypoints: np.ndarray  # (17, 3) of x, y, confidence
    mask: Optional[np.ndarray] = None  # bool array, full image size
    label: str = field(default="")

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"Person {self.index + 1}"

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    def visible(self, *indices: int, threshold: float = 0.35) -> bool:
        return all(self.keypoints[i, 2] >= threshold for i in indices)

    def quality_warning(self, image_shape: tuple[int, int]) -> str:
        """Flag a person the try-on model will struggle with.

        CatVTON works at 768x1024. A crop much smaller than that gets upscaled,
        and upscaling cannot invent detail that was never captured - the result
        comes back smeared. Saying so up front beats letting someone wait a
        minute for mush.
        """
        x1, y1, x2, y2 = self.padded_box(image_shape)
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 170 or crop_h < 240:
            return "This person is very small in the photo. The new clothes may look blurry."
        if crop_w < 280 or crop_h < 380:
            return "This person is a bit far away. Fine details on the clothes may be soft."

        visible = int((self.keypoints[:, 2] >= 0.35).sum())
        if visible < 8:
            return "Part of this person is hidden, so the result may look a little off."
        return ""

    def padded_box(self, image_shape: tuple[int, int], pad: float = 0.12) -> tuple[int, int, int, int]:
        """Expand the box outward so the crop keeps context around the body.

        Try-on models expect a person to sit in the frame the way a catalogue
        photo does, so a tight box tends to produce cropped-looking garments.
        """
        h, w = image_shape[:2]
        x1, y1, x2, y2 = self.box
        dx, dy = int(self.width * pad), int(self.height * pad)
        return (
            max(0, x1 - dx),
            max(0, y1 - dy),
            min(w, x2 + dx),
            min(h, y2 + dy),
        )


class PersonDetector:
    """Lazily-loaded YOLO-pose + SAM 2 pair.

    Models stay unloaded until the first call so the UI starts instantly.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or config.resolve_device()
        self._pose = None
        self._sam = None

    def _load_pose(self):
        if self._pose is None:
            from . import models

            models.require_ready(config.MODE_BACKGROUND)
            from ultralytics import YOLO

            self._pose = YOLO(config.YOLO_POSE_MODEL)
        return self._pose

    def _load_sam(self):
        if self._sam is None:
            from . import models

            models.require_ready(config.MODE_BACKGROUND)
            from ultralytics import SAM

            self._sam = SAM(config.SAM_MODEL)
        return self._sam

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.35,
        min_area_ratio: float = 0.004,
        with_masks: bool = True,
    ) -> list[Person]:
        """Return people sorted left-to-right, largest first on ties."""
        pose = self._load_pose()
        results = pose.predict(image, conf=conf, classes=[0], device=self.device, verbose=False)
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        img_area = image.shape[0] * image.shape[1]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        if result.keypoints is not None and result.keypoints.data is not None:
            keypoints = result.keypoints.data.cpu().numpy()
        else:
            keypoints = np.zeros((len(boxes), 17, 3), dtype=np.float32)

        people: list[Person] = []
        for box, score, kps in zip(boxes, scores, keypoints):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            # Drop bystanders far in the background; they carry too few pixels
            # for a try-on model to produce anything usable.
            if (x2 - x1) * (y2 - y1) < img_area * min_area_ratio:
                continue
            people.append(
                Person(index=len(people), box=(x1, y1, x2, y2), score=float(score), keypoints=kps)
            )

        people.sort(key=lambda p: (p.box[0], -p.area))
        for i, person in enumerate(people):
            person.index = i
            person.label = f"Person {i + 1}"

        if with_masks and people:
            self.add_masks(image, people)
        return people

    def add_masks(self, image: np.ndarray, people: list[Person]) -> None:
        """Fill in each person's silhouette using SAM 2, prompted by the boxes."""
        sam = self._load_sam()
        prompts = [list(p.box) for p in people]
        results = sam.predict(image, bboxes=prompts, device=self.device, verbose=False)
        if not results or results[0].masks is None:
            return

        masks = results[0].masks.data.cpu().numpy()
        h, w = image.shape[:2]
        for person, mask in zip(people, masks):
            if mask.shape != (h, w):
                import cv2

                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            person.mask = mask.astype(bool)
