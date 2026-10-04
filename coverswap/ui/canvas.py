"""Image canvas with clickable person overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QSizePolicy, QWidget

from . import style

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _first_image_url(mime) -> Optional[str]:
    if not mime.hasUrls():
        return None
    for url in mime.urls():
        path = url.toLocalFile()
        if path and Path(path).suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def bgr_to_qimage(image: np.ndarray) -> QImage:
    h, w = image.shape[:2]
    rgb = np.ascontiguousarray(image[..., ::-1])
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageCanvas(QWidget):
    """Shows a photo scaled to fit, draws detected people, emits clicks."""

    personClicked = pyqtSignal(int)
    fileDropped = pyqtSignal(str)

    def __init__(self, placeholder: str = "Drop a family photo here\nor click Open photo", parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setCursor(Qt.ArrowCursor)

        self._pixmap: Optional[QPixmap] = None
        self._people: list = []
        self._selected = -1
        self._hovered = -1
        self._show_boxes = True
        self._placeholder = placeholder
        self._draw_rect = QRect()

    # -- data ------------------------------------------------------------
    def set_image(self, image: Optional[np.ndarray]) -> None:
        self._pixmap = QPixmap.fromImage(bgr_to_qimage(image)) if image is not None else None
        self.update()

    def set_people(self, people: list) -> None:
        self._people = people
        self._selected = -1
        self._hovered = -1
        self.update()

    def set_selected(self, index: int) -> None:
        self._selected = index
        self.update()

    def set_show_boxes(self, show: bool) -> None:
        self._show_boxes = show
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    # -- geometry --------------------------------------------------------
    def _compute_draw_rect(self) -> QRect:
        if self._pixmap is None or self._pixmap.isNull():
            return QRect()
        scaled = self._pixmap.size()
        scaled.scale(self.size(), Qt.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QRect(QPoint(x, y), scaled)

    def _image_to_widget(self, box) -> QRect:
        """Map original-image coordinates onto the on-screen rectangle."""
        if self._pixmap is None or self._draw_rect.isEmpty():
            return QRect()
        sx = self._draw_rect.width() / self._pixmap.width()
        sy = self._draw_rect.height() / self._pixmap.height()
        x1, y1, x2, y2 = box
        return QRect(
            int(self._draw_rect.x() + x1 * sx),
            int(self._draw_rect.y() + y1 * sy),
            int((x2 - x1) * sx),
            int((y2 - y1) * sy),
        )

    def _person_at(self, pos) -> int:
        """Topmost person under the cursor; smallest box wins so a child in
        front of an adult stays clickable."""
        best, best_area = -1, None
        for i, person in enumerate(self._people):
            rect = self._image_to_widget(person.box)
            if rect.contains(pos):
                area = rect.width() * rect.height()
                if best_area is None or area < best_area:
                    best, best_area = i, area
        return best

    # -- events ----------------------------------------------------------
    def mouseMoveEvent(self, event) -> None:
        hovered = self._person_at(event.pos()) if self._show_boxes else -1
        if hovered != self._hovered:
            self._hovered = hovered
            self.setCursor(Qt.PointingHandCursor if hovered >= 0 else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, event) -> None:
        if self._hovered != -1:
            self._hovered = -1
            self.setCursor(Qt.ArrowCursor)
            self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or not self._show_boxes:
            return
        index = self._person_at(event.pos())
        if index >= 0:
            self.personClicked.emit(index)

    def dragEnterEvent(self, event) -> None:
        if _first_image_url(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        path = _first_image_url(event.mimeData())
        if path:
            self.fileDropped.emit(path)

    # -- painting --------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.fillRect(self.rect(), QColor(style.BG))

        if self._pixmap is None or self._pixmap.isNull():
            self._draw_placeholder(painter)
            return

        self._draw_rect = self._compute_draw_rect()
        painter.drawPixmap(self._draw_rect, self._pixmap)

        if self._show_boxes:
            for i, person in enumerate(self._people):
                self._draw_person(painter, i, person)

    def _draw_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(style.BORDER), 1, Qt.DashLine))
        inset = self.rect().adjusted(24, 24, -24, -24)
        painter.drawRoundedRect(inset, 12, 12)
        painter.setPen(QColor(style.TEXT_DIM))
        font = QFont(self.font())
        font.setPointSize(11)
        painter.setFont(font)
        painter.drawText(inset, Qt.AlignCenter, self._placeholder)

    def _draw_person(self, painter: QPainter, index: int, person) -> None:
        rect = self._image_to_widget(person.box)
        if rect.isEmpty():
            return

        colour = QColor(style.PERSON_COLOURS[index % len(style.PERSON_COLOURS)])
        selected = index == self._selected
        hovered = index == self._hovered

        if selected:
            # Dim everything else so the active person stands out.
            shade = QColor(0, 0, 0, 90)
            for band in _surrounding(self._draw_rect, rect):
                if band.isValid():
                    painter.fillRect(band, shade)

        width = 3 if selected else (2 if hovered else 2)
        pen = QPen(colour, width)
        if not selected and not hovered:
            colour.setAlpha(170)
            pen = QPen(colour, 2)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 6, 6)

        label = person.label
        font = QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(label) + 14
        th = metrics.height() + 6
        tag = QRect(rect.x(), max(0, rect.y() - th - 3), tw, th)

        badge = QColor(style.PERSON_COLOURS[index % len(style.PERSON_COLOURS)])
        if not selected and not hovered:
            badge.setAlpha(190)
        painter.setPen(Qt.NoPen)
        painter.setBrush(badge)
        painter.drawRoundedRect(tag, 5, 5)
        painter.setPen(QColor("#0b0d12"))
        painter.drawText(tag, Qt.AlignCenter, label)
        painter.setBrush(Qt.NoBrush)


def _surrounding(outer: QRect, inner: QRect) -> list[QRect]:
    """The four bands of `outer` left over once `inner` is removed."""
    return [
        QRect(outer.left(), outer.top(), outer.width(), max(0, inner.top() - outer.top())),
        QRect(outer.left(), inner.bottom(), outer.width(), max(0, outer.bottom() - inner.bottom())),
        QRect(outer.left(), inner.top(), max(0, inner.left() - outer.left()), inner.height()),
        QRect(inner.right(), inner.top(), max(0, outer.right() - inner.right()), inner.height()),
    ]
