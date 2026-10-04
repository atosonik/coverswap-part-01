"""Render the main window to a PNG so the layout can be checked without a
display session. Run: python scripts/ui_preview.py [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(msg, flush=True)


def synthetic_photo(w: int = 1200, h: int = 800) -> np.ndarray:
    """A stand-in image; detection is not exercised here."""
    import cv2

    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        img[y, :] = (int(60 + 40 * t), int(48 + 30 * t), int(44 + 26 * t))
    cv2.rectangle(img, (0, int(h * 0.72)), (w, h), (70, 92, 110), -1)
    for i, cx in enumerate((260, 500, 720, 940)):
        top = int(h * 0.30) + (i % 2) * 40
        cv2.rectangle(img, (cx - 60, top), (cx + 60, int(h * 0.80)), (120, 130, 150), -1)
        cv2.circle(img, (cx, top - 45), 42, (150, 160, 178), -1)
    return img


def fake_people(image: np.ndarray):
    from coverswap.detection import Person

    people = []
    for i, cx in enumerate((260, 500, 720, 940)):
        top = int(image.shape[0] * 0.30) + (i % 2) * 40 - 90
        box = (cx - 75, top, cx + 75, int(image.shape[0] * 0.82))
        kps = np.zeros((17, 3), dtype=np.float32)
        people.append(Person(index=i, box=box, score=0.95, keypoints=kps))
    return people


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "ui_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    log("importing Qt")
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QListWidgetItem

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv[:1])
    log("QApplication ready")

    from coverswap.ui import style
    from coverswap.ui.main_window import MainWindow

    app.setStyleSheet(style.STYLESHEET)
    log("stylesheet applied")

    window = MainWindow()
    window.resize(1360, 880)
    log("window built")

    image = synthetic_photo()
    window.image = image
    window.image_info.setText("family_photo.jpg  -  1200x800")
    window.canvas.set_image(image)
    log("image set")

    people = fake_people(image)
    window.people = people
    for person in people:
        item = QListWidgetItem(person.label)
        item.setIcon(window._dot(style.PERSON_COLOURS[person.index % len(style.PERSON_COLOURS)]))
        window.people_list.addItem(item)
    log("people list filled")

    window.canvas.set_people(people)
    window.people_list.setCurrentRow(1)
    window.people_hint.setText("Click a person in the photo, or pick from the list above.")
    window.set_status("Found 4 people.")
    window._update_actions()
    log("state applied")

    window.show()
    app.processEvents()
    log("shown")

    pixmap = window.grab()
    log("grabbed")
    pixmap.save(str(out))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
