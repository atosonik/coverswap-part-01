"""First-run screen: download models once, then the app stays offline."""

from __future__ import annotations

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from .. import models
from . import style


class DownloadWorker(QThread):
    progress = pyqtSignal(str)
    failed = pyqtSignal(str)
    completed = pyqtSignal()

    def run(self) -> None:
        try:
            models.download_all(progress=self.progress.emit)
            self.completed.emit()
        except Exception as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)


class SetupDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CoverSwap setup")
        self.setModal(True)
        self.setMinimumWidth(460)
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(12)

        title = QLabel("One-time setup")
        title.setObjectName("Heading")
        layout.addWidget(title)

        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setObjectName("Hint")
        layout.addWidget(self.body)

        self.status = QLabel(models.summary())
        self.status.setObjectName("StatusText")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QPushButton("Later")
        self.cancel_button.clicked.connect(self.reject)
        self.download_button = QPushButton("Download models")
        self.download_button.setObjectName("Primary")
        self.download_button.clicked.connect(self._start)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.download_button)
        layout.addLayout(buttons)

        self._refresh()

    def _refresh(self) -> None:
        if models.ready():
            self.body.setText(
                "All AI models are on this computer. CoverSwap works without the internet."
            )
            self.download_button.setText("Continue")
            self.download_button.clicked.disconnect()
            self.download_button.clicked.connect(self.accept)
            self.cancel_button.hide()
            return

        missing = models.missing_models()
        names = ", ".join(item.label for item in missing)
        self.body.setText(
            "CoverSwap needs to download its AI models once (about 3 GB). "
            "After that you can unplug the internet and use it normally.\n\n"
            f"Still needed: {names}."
        )

    def _start(self) -> None:
        # Downloads must be allowed to reach the network. The rest of the app
        # sets HF_HUB_OFFLINE so runtime cannot sneak a request through.
        import os

        os.environ["COVERSWAP_ALLOW_DOWNLOAD"] = "1"
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        self.download_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()

        self._worker = DownloadWorker()
        self._worker.progress.connect(self.status.setText)
        self._worker.completed.connect(self._done)
        self._worker.failed.connect(self._fail)
        self._worker.start()

    def _done(self) -> None:
        self.progress.hide()
        self.status.setText(models.summary())
        self._refresh()
        if models.ready():
            self.accept()

    def _fail(self, message: str) -> None:
        self.progress.hide()
        self.download_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.status.setText(message)
        self.status.setStyleSheet(f"color: {style.DANGER};")
