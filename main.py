"""Entry point: python main.py"""

from __future__ import annotations

import sys


def preload_onnxruntime() -> None:
    """Import onnxruntime before PyQt5 (Windows DLL ordering).

    The face-swap engine uses InsightFace/onnxruntime. Loading Qt first breaks
    onnxruntime's pybind extension on some Windows builds.
    """
    import onnxruntime  # noqa: F401


def preload_torch() -> None:
    """Import torch on the main thread, before any worker exists.

    On Windows, torch's DLL initialisation only succeeds on the thread that
    loads it first. If the first `import torch` happens inside a QThread - which
    is where detection and try-on run - it dies with:

        OSError: [WinError 1114] A dynamic link library (DLL) initialization
        routine failed. Error loading ...\\torch\\lib\\c10.dll

    Importing here costs a couple of seconds at startup and makes the later
    imports inside worker threads free, since the module is already loaded.
    """
    import torch  # noqa: F401


def main() -> int:
    preload_onnxruntime()
    preload_torch()

    try:
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"sync": False})
    except Exception:
        pass

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    # Must be set before the QApplication exists.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("CoverSwap")
    app.setApplicationDisplayName("CoverSwap")

    from coverswap.ui.main_window import MainWindow
    from coverswap.ui.style import STYLESHEET

    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
