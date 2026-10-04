"""Drive the real window end to end, worker threads included.

Bypasses only the file dialogs; everything else runs the same code the user
does. Catches wiring problems that a direct pipeline call would miss, such as
signals crossing threads incorrectly.

Usage: python scripts/app_smoke.py [photo] [garment] [category]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CATVTON = ROOT / "third_party" / "CatVTON" / "resource" / "demo" / "example"
DEFAULT_PHOTO = CATVTON / "person" / "men" / "model_5.png"
DEFAULT_GARMENT = CATVTON / "condition" / "upper" / "24083449_54173465_2048.jpg"


def log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    photo = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PHOTO
    garment = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GARMENT
    category = sys.argv[3] if len(sys.argv) > 3 else "upper_body"

    # Mirror main.py: torch's DLLs must load on the main thread.
    from main import preload_onnxruntime, preload_torch

    preload_onnxruntime()
    preload_torch()

    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv[:1])

    from coverswap.pipeline import imread
    from coverswap.ui.canvas import bgr_to_qimage
    from coverswap.ui.main_window import MainWindow
    from coverswap.ui.style import STYLESHEET

    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()

    def pump(predicate, timeout: float, what: str) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            app.processEvents()
            if predicate():
                return True
            time.sleep(0.05)
        log(f"TIMEOUT waiting for {what}")
        return False

    # --- stand in for "Open photo..."
    window.image = imread(photo)
    window.image_path = photo
    window.canvas.set_image(window.image)
    h, w = window.image.shape[:2]
    window.image_info.setText(f"{photo.name}  -  {w}x{h}")
    log(f"loaded {photo.name} ({w}x{h})")

    window.run_detect()
    if not pump(lambda: window._worker is None, 600, "detection"):
        return 1
    if not window.people:
        log("FAIL no people detected")
        return 1
    log(f"detected {len(window.people)} people")

    # --- stand in for "Browse garment..."
    window.garment = imread(garment)
    window.garment_path = garment
    window.garment_preview.setPixmap(
        QPixmap.fromImage(bgr_to_qimage(window.garment)).scaled(78, 102, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    )
    window.category_combo.setCurrentIndex(
        max(0, window.category_combo.findData(category))
    )

    target = max(window.people, key=lambda p: p.area)
    window.people_list.setCurrentRow(target.index)
    log(f"selected {target.label}; warning={target.quality_warning(window.image.shape)!r}")

    if not window._ready_to_swap():
        log("FAIL swap button would be disabled")
        return 1

    window.run_swap()
    if not pump(lambda: window._worker is None, 900, "swap"):
        return 1
    if window.result is None:
        log(f"FAIL swap produced nothing; status={window.status.text()!r}")
        return 1
    log(f"swap done; status={window.status.text()!r}")

    window.refresh_view()
    app.processEvents()

    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(out / "app_smoke.png"))
    from coverswap.pipeline import imwrite

    imwrite(out / "app_smoke_result.png", window.result)
    log(f"wrote {out / 'app_smoke.png'}")
    log("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
