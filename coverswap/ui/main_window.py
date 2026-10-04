"""CoverSwap main window — clothes, face, and background swap."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import background as bg
from .. import clothing, config, models, pipeline as pipe
from . import style
from .canvas import IMAGE_SUFFIXES, ImageCanvas, bgr_to_qimage
from .setup import SetupDialog
from .workers import BackgroundSwapWorker, ClothesSwapWorker, DetectWorker, FaceSwapWorker

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"

CATEGORY_LABELS = {
    "upper_body": "Top / shirt",
    "lower_body": "Trousers / skirt",
    "dress": "Dress / full outfit",
}


def card(*, spacing: int = 10, margins: tuple[int, int, int, int] = (14, 14, 14, 14)):
    box = QWidget()
    box.setObjectName("Card")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    return box, layout


def section_label(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("SubHeading")
    return label


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CoverSwap")
        self.resize(1360, 880)
        self.setMinimumSize(1040, 700)

        self.pipeline = pipe.CoverSwapPipeline()
        self.mode = config.MODE_CLOTHES
        self.clothes_settings = config.TryOnSettings()
        self.face_settings = config.FaceSwapSettings()
        self.bg_settings = config.BackgroundSwapSettings()

        self.image: Optional[np.ndarray] = None
        self.image_path: Optional[Path] = None
        self.result: Optional[np.ndarray] = None
        self.garment: Optional[np.ndarray] = None
        self.garment_path: Optional[Path] = None
        self.source_face: Optional[np.ndarray] = None
        self.source_face_path: Optional[Path] = None
        self.background: Optional[np.ndarray] = None
        self.background_path: Optional[Path] = None
        self.people = []
        self.faces = []
        self.selected = -1
        self._worker = None
        self._face_raw = None
        self._face_mask = None
        self._face_region = None

        self._build_ui()
        self._update_actions()
        self._apply_mode_ui()
        self.set_status(self._idle_status())
        QTimer.singleShot(80, self._after_show)

    def _idle_status(self) -> str:
        return f"Ready · works offline · {config.describe_device()}"

    def _after_show(self) -> None:
        try:
            import ultralytics  # noqa: F401
        except Exception as exc:
            self.set_status(f"Could not start the people finder ({exc}).")
        if not models.ready():
            self._offer_setup()

    def _offer_setup(self) -> None:
        dialog = SetupDialog(self)
        dialog.exec_()
        self._refresh_setup_banner()
        self.set_status(self._idle_status())

    def _refresh_setup_banner(self) -> None:
        ready = models.ready_for_mode(self.mode)
        self.setup_banner.setVisible(not ready)
        self.swap_button.setEnabled(ready and self._ready_to_swap())

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        outer.addWidget(self._build_sidebar(), 0)
        outer.addWidget(self._build_viewer(), 1)
        self.setCentralWidget(root)

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(14)

        title = QLabel("CoverSwap")
        title.setObjectName("Heading")
        subtitle = QLabel("Swap clothes, faces, or backgrounds in a photo.")
        subtitle.setObjectName("Hint")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.mode_buttons = QButtonGroup(self)
        for i, key in enumerate(config.SWAP_MODES):
            button = QPushButton(config.MODE_LABELS[key])
            button.setCheckable(True)
            button.setChecked(key == self.mode)
            self.mode_buttons.addButton(button, i)
            mode_row.addWidget(button)
        self.mode_buttons.idClicked.connect(self._mode_changed)
        layout.addLayout(mode_row)

        self.setup_banner = QPushButton("Download AI models (one time)")
        self.setup_banner.setObjectName("Primary")
        self.setup_banner.clicked.connect(self._offer_setup)
        self.setup_banner.setVisible(not models.ready())
        layout.addWidget(self.setup_banner)

        # --- step 1: photo
        photo_card, photo_layout = card()
        photo_layout.addWidget(section_label("1  Open a photo"))
        self.open_button = QPushButton("Open photo")
        self.open_button.clicked.connect(self.open_photo)
        photo_layout.addWidget(self.open_button)

        self.people_list = QListWidget()
        self.people_list.setMinimumHeight(132)
        self.people_list.setMaximumHeight(220)
        self.people_list.currentRowChanged.connect(self.select_target)
        photo_layout.addWidget(self.people_list)

        self.people_hint = QLabel("Drop a photo on the right, or click Open photo.")
        self.people_hint.setObjectName("Hint")
        self.people_hint.setWordWrap(True)
        photo_layout.addWidget(self.people_hint)
        layout.addWidget(photo_card)

        # --- step 2: mode-specific input
        step2_card, step2_layout = card()
        self.step2_heading = section_label("2  Choose clothes")
        step2_layout.addWidget(self.step2_heading)

        self.step2_stack = QStackedWidget()
        self.step2_stack.addWidget(self._build_clothes_panel())
        self.step2_stack.addWidget(self._build_face_panel())
        self.step2_stack.addWidget(self._build_background_panel())
        step2_layout.addWidget(self.step2_stack)
        layout.addWidget(step2_card)

        # --- optional clothes knobs
        self.tune_card, tune_layout = card()
        self.advanced_toggle = QCheckBox("Show extra options")
        self.advanced_toggle.stateChanged.connect(self._toggle_advanced)
        tune_layout.addWidget(self.advanced_toggle)

        self.advanced_box = QWidget()
        advanced = QVBoxLayout(self.advanced_box)
        advanced.setContentsMargins(0, 6, 0, 0)
        advanced.setSpacing(8)
        self.steps_slider, steps_row = self._slider_row(
            "Quality (higher = slower)", 15, 50, self.clothes_settings.steps, self._steps_changed
        )
        advanced.addLayout(steps_row)
        self.guidance_slider, guidance_row = self._slider_row(
            "How closely to follow the clothes",
            10,
            50,
            int(self.clothes_settings.guidance_scale * 10),
            self._guidance_changed,
            fmt=lambda v: f"{v / 10:.1f}",
        )
        advanced.addLayout(guidance_row)
        self.mask_slider, mask_row = self._slider_row(
            "How much of the outfit to replace",
            0,
            40,
            self.clothes_settings.mask_grow,
            self._mask_changed,
        )
        advanced.addLayout(mask_row)
        hires = QCheckBox("Sharper result (uses more memory)")
        hires.setChecked(False)
        hires.stateChanged.connect(self._resolution_changed)
        advanced.addWidget(hires)
        self.advanced_box.hide()
        tune_layout.addWidget(self.advanced_box)
        layout.addWidget(self.tune_card)

        layout.addStretch(1)

        self.swap_button = QPushButton("Try these clothes on")
        self.swap_button.setObjectName("Primary")
        self.swap_button.clicked.connect(self.run_swap)
        layout.addWidget(self.swap_button)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.save_button = QPushButton("Save photo")
        self.save_button.clicked.connect(self.save_result)
        self.reset_button = QPushButton("Start over")
        self.reset_button.clicked.connect(self.reset_result)
        actions.addWidget(self.save_button)
        actions.addWidget(self.reset_button)
        layout.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.status = QLabel(self._idle_status())
        self.status.setObjectName("StatusText")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        scroller = QScrollArea()
        scroller.setWidget(panel)
        scroller.setWidgetResizable(True)
        scroller.setFixedWidth(360)
        scroller.setFrameShape(QFrame.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroller.setStyleSheet(
            "QScrollArea { background: transparent; } "
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        return scroller

    def _build_clothes_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.garment_preview = QLabel()
        self.garment_preview.setFixedSize(84, 108)
        self.garment_preview.setAlignment(Qt.AlignCenter)
        self.garment_preview.setStyleSheet(
            f"background:{style.SURFACE_2};border:1px solid {style.BORDER};"
            f"border-radius:8px;color:{style.TEXT_DIM};font-size:11px;"
        )
        self.garment_preview.setText("Drop\nclothes\nhere")
        self.garment_preview.setAcceptDrops(True)
        self.garment_preview.dragEnterEvent = self._garment_drag
        self.garment_preview.dropEvent = self._garment_drop
        row.addWidget(self.garment_preview)

        column = QVBoxLayout()
        column.setSpacing(8)
        self.garment_button = QPushButton("Choose clothes")
        self.garment_button.clicked.connect(self.open_garment)
        column.addWidget(self.garment_button)

        self.category_combo = QComboBox()
        for key in config.GARMENT_CATEGORIES:
            self.category_combo.addItem(CATEGORY_LABELS[key], key)
        self.category_combo.currentIndexChanged.connect(self._category_changed)
        column.addWidget(self.category_combo)
        column.addStretch(1)
        row.addLayout(column, 1)
        layout.addLayout(row)
        return widget

    def _build_face_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.face_preview = QLabel()
        self.face_preview.setFixedSize(84, 108)
        self.face_preview.setAlignment(Qt.AlignCenter)
        self.face_preview.setStyleSheet(
            f"background:{style.SURFACE_2};border:1px solid {style.BORDER};"
            f"border-radius:8px;color:{style.TEXT_DIM};font-size:11px;"
        )
        self.face_preview.setText("Drop\nface\nhere")
        self.face_preview.setAcceptDrops(True)
        self.face_preview.dragEnterEvent = self._face_drag
        self.face_preview.dropEvent = self._face_drop
        row.addWidget(self.face_preview)

        column = QVBoxLayout()
        column.setSpacing(8)
        self.face_button = QPushButton("Choose new face")
        self.face_button.clicked.connect(self.open_source_face)
        column.addWidget(self.face_button)
        column.addStretch(1)
        row.addLayout(column, 1)
        layout.addLayout(row)

        self.tone_slider, tone_row = self._slider_row(
            "Skin tone match",
            0,
            100,
            int(self.face_settings.tone_match * 100),
            self._tone_changed,
            fmt=lambda v: f"{v}%",
        )
        layout.addLayout(tone_row)
        self.blend_slider, blend_row = self._slider_row(
            "Swap strength",
            0,
            100,
            int(self.face_settings.blend * 100),
            self._blend_changed,
            fmt=lambda v: f"{v}%",
        )
        layout.addLayout(blend_row)
        return widget

    def _build_background_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.bg_preview = QLabel()
        self.bg_preview.setFixedSize(84, 108)
        self.bg_preview.setAlignment(Qt.AlignCenter)
        self.bg_preview.setStyleSheet(
            f"background:{style.SURFACE_2};border:1px solid {style.BORDER};"
            f"border-radius:8px;color:{style.TEXT_DIM};font-size:11px;"
        )
        self.bg_preview.setText("Drop\nbackground\nhere")
        self.bg_preview.setAcceptDrops(True)
        self.bg_preview.dragEnterEvent = self._background_drag
        self.bg_preview.dropEvent = self._background_drop
        row.addWidget(self.bg_preview)

        column = QVBoxLayout()
        column.setSpacing(8)
        self.bg_button = QPushButton("Choose background")
        self.bg_button.clicked.connect(self.open_background)
        column.addWidget(self.bg_button)
        column.addStretch(1)
        row.addLayout(column, 1)
        layout.addLayout(row)

        self.bg_feather_slider, bg_row = self._slider_row(
            "Edge softness",
            0,
            30,
            self.bg_settings.feather,
            self._bg_feather_changed,
        )
        layout.addLayout(bg_row)
        return widget

    def _slider_row(self, label, minimum, maximum, value, handler, fmt=str):
        row = QVBoxLayout()
        row.setSpacing(4)
        head = QHBoxLayout()
        name = QLabel(label)
        name.setObjectName("Hint")
        readout = QLabel(fmt(value))
        readout.setObjectName("Hint")
        readout.setAlignment(Qt.AlignRight)
        head.addWidget(name)
        head.addWidget(readout)
        row.addLayout(head)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.valueChanged.connect(lambda v: readout.setText(fmt(v)))
        slider.valueChanged.connect(handler)
        row.addWidget(slider)
        return slider, row

    def _build_viewer(self) -> QWidget:
        container, layout = card(spacing=12)
        header = QHBoxLayout()
        header.setSpacing(8)

        self.view_buttons = QButtonGroup(self)
        for i, name in enumerate(("Photo", "Highlight", "Result")):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setChecked(i == 0)
            self.view_buttons.addButton(button, i)
            header.addWidget(button)
        self.view_buttons.buttonClicked.connect(lambda _: self.refresh_view())
        header.addStretch(1)

        self.image_info = QLabel("")
        self.image_info.setObjectName("Hint")
        header.addWidget(self.image_info)
        layout.addLayout(header)

        self.canvas = ImageCanvas()
        self.canvas.personClicked.connect(self.select_target_from_canvas)
        self.canvas.fileDropped.connect(self._load_photo)
        layout.addWidget(self.canvas, 1)

        self.footer_hint = QLabel("Tip: drop a photo here, then click the person or face to edit.")
        self.footer_hint.setObjectName("Hint")
        layout.addWidget(self.footer_hint)
        return container

    # ---------------------------------------------------------- mode switch
    def _mode_changed(self, index: int) -> None:
        self.mode = config.SWAP_MODES[index]
        self._apply_mode_ui()
        self._refresh_setup_banner()
        self.result = None
        self._face_raw = None
        if self.image is not None:
            self.run_detect()
        else:
            self.refresh_view()
        self._update_actions()

    def _apply_mode_ui(self) -> None:
        index = config.SWAP_MODES.index(self.mode)
        self.step2_stack.setCurrentIndex(index)
        self.tune_card.setVisible(self.mode == config.MODE_CLOTHES)

        headings = {
            config.MODE_CLOTHES: "2  Choose clothes",
            config.MODE_FACE: "2  Choose new face",
            config.MODE_BACKGROUND: "2  Choose background",
        }
        buttons = {
            config.MODE_CLOTHES: "Try these clothes on",
            config.MODE_FACE: "Swap face",
            config.MODE_BACKGROUND: "Replace background",
        }
        hints = {
            config.MODE_CLOTHES: "Tip: drop a photo, click a person, then choose clothes.",
            config.MODE_FACE: "Tip: drop a photo, click a face, then choose the replacement face.",
            config.MODE_BACKGROUND: "Tip: everyone detected stays; the scenery behind them changes.",
        }
        self.step2_heading.setText(headings[self.mode])
        self.swap_button.setText(buttons[self.mode])
        self.footer_hint.setText(hints[self.mode])

    @property
    def _targets(self) -> list:
        return self.faces if self.mode == config.MODE_FACE else self.people

    # ------------------------------------------------------------- actions
    def open_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a photo", "", IMAGE_FILTER)
        if path:
            self._load_photo(path)

    def _load_photo(self, path: str) -> None:
        try:
            image = pipe.imread(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open that photo", str(exc))
            return

        self.image = image
        self.image_path = Path(path)
        self.result = None
        self.people = []
        self.faces = []
        self.selected = -1
        self._face_raw = None
        self.people_list.clear()
        self.view_buttons.button(0).setChecked(True)
        self.canvas.set_people([])
        self.canvas.set_image(image)
        h, w = image.shape[:2]
        self.image_info.setText(f"{self.image_path.name}  -  {w}x{h}")
        self._update_actions()
        self.run_detect()

    def run_detect(self) -> None:
        if self.image is None:
            return
        if not models.ready_for_mode(self.mode):
            self._offer_setup()
            if not models.ready_for_mode(self.mode):
                return
        self._start(DetectWorker(self.pipeline, self.image, self.mode), self._detect_ready)

    def _detect_ready(self, found: list) -> None:
        if self.mode == config.MODE_FACE:
            self.faces = found
            self.people = []
        else:
            self.people = found
            self.faces = []

        self.people_list.clear()
        for item in found:
            row = QListWidgetItem(item.label)
            colour = style.PERSON_COLOURS[item.index % len(style.PERSON_COLOURS)]
            row.setIcon(self._dot(colour))
            self.people_list.addItem(row)

        self.canvas.set_people(found)
        if found:
            if self.mode == config.MODE_FACE:
                self.people_hint.setText("Click the face you want to replace.")
            elif self.mode == config.MODE_BACKGROUND:
                self.people_hint.setText(
                    f"Found {len(found)} person(s). Everyone stays; only the background changes."
                )
            else:
                self.people_hint.setText("Click the person you want to dress.")
            self.people_list.setCurrentRow(0)
        else:
            empty = {
                config.MODE_FACE: "No faces found. Try a clearer, front-facing photo.",
                config.MODE_BACKGROUND: "No people found. Background swap needs at least one person.",
                config.MODE_CLOTHES: "No people found. Try a brighter photo with clearer poses.",
            }
            self.people_hint.setText(empty[self.mode])
        self._update_actions()

    def _dot(self, colour: str) -> QIcon:
        pixmap = QPixmap(12, 12)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(colour))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(1, 1, 10, 10)
        painter.end()
        return QIcon(pixmap)

    def select_target_from_canvas(self, index: int) -> None:
        self.people_list.setCurrentRow(index)

    def select_target(self, index: int) -> None:
        self.selected = index
        self.canvas.set_selected(index)
        self.refresh_view()
        self._update_actions()

        targets = self._targets
        if self.image is None or not (0 <= index < len(targets)):
            return

        if self.mode == config.MODE_CLOTHES:
            warning = targets[index].quality_warning(self.image.shape)
            if warning:
                self.people_hint.setText(warning)
                self.people_hint.setStyleSheet(f"color: {style.WARNING};")
            else:
                self.people_hint.setText("Now choose the clothes, then press the blue button.")
                self.people_hint.setStyleSheet("")
        elif self.mode == config.MODE_FACE:
            self.people_hint.setText("Choose the replacement face photo, then press Swap face.")
            self.people_hint.setStyleSheet("")

    def _image_drag(self, event, suffixes=IMAGE_SUFFIXES):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in suffixes:
                    event.acceptProposedAction()
                    return

    def _garment_drag(self, event) -> None:
        self._image_drag(event)

    def _garment_drop(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and Path(path).suffix.lower() in IMAGE_SUFFIXES:
                self._load_garment(path)
                return

    def _face_drag(self, event) -> None:
        self._image_drag(event)

    def _face_drop(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and Path(path).suffix.lower() in IMAGE_SUFFIXES:
                self._load_source_face(path)
                return

    def _background_drag(self, event) -> None:
        self._image_drag(event)

    def _background_drop(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and Path(path).suffix.lower() in IMAGE_SUFFIXES:
                self._load_background(path)
                return

    def _toggle_advanced(self, state: int) -> None:
        self.advanced_box.setVisible(state == Qt.Checked)

    def open_garment(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a clothes photo", "", IMAGE_FILTER)
        if path:
            self._load_garment(path)

    def _load_garment(self, path: str) -> None:
        try:
            garment = pipe.imread(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open those clothes", str(exc))
            return
        self.garment = garment
        self.garment_path = Path(path)
        self._set_preview(self.garment_preview, garment, self.garment_path.name)
        self._update_actions()

    def open_source_face(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a face photo", "", IMAGE_FILTER)
        if path:
            self._load_source_face(path)

    def _load_source_face(self, path: str) -> None:
        try:
            face_img = pipe.imread(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open that face photo", str(exc))
            return
        self.source_face = face_img
        self.source_face_path = Path(path)
        self._set_preview(self.face_preview, face_img, self.source_face_path.name)
        self._update_actions()

    def open_background(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a background photo", "", IMAGE_FILTER)
        if path:
            self._load_background(path)

    def _load_background(self, path: str) -> None:
        try:
            bg_img = pipe.imread(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open that background", str(exc))
            return
        self.background = bg_img
        self.background_path = Path(path)
        self._set_preview(self.bg_preview, bg_img, self.background_path.name)
        self.refresh_view()
        self._update_actions()

    def _set_preview(self, label: QLabel, image: np.ndarray, tooltip: str) -> None:
        pixmap = QPixmap.fromImage(bgr_to_qimage(image)).scaled(
            label.width() - 6,
            label.height() - 6,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)
        label.setToolTip(tooltip)

    def run_swap(self) -> None:
        if not models.ready_for_mode(self.mode):
            self._offer_setup()
            if not models.ready_for_mode(self.mode):
                return
        if not self._ready_to_swap():
            return

        base = self.result if self.result is not None else self.image

        if self.mode == config.MODE_CLOTHES:
            worker = ClothesSwapWorker(
                self.pipeline,
                base,
                self.people[self.selected],
                self.garment,
                self.clothes_settings,
            )
            self._start(worker, self._swap_ready)
        elif self.mode == config.MODE_FACE:
            worker = FaceSwapWorker(
                self.pipeline,
                base,
                self.faces[self.selected],
                self.source_face,
                self.face_settings,
            )
            self._start(worker, self._face_swap_ready)
        else:
            worker = BackgroundSwapWorker(
                self.pipeline,
                base,
                self.people,
                self.background,
                self.bg_settings,
            )
            self._start(worker, self._swap_ready)

    def _face_swap_ready(self, payload) -> None:
        self.result = payload["result"]
        self._face_raw = payload["raw"]
        self._face_mask = payload["mask"]
        self._face_region = payload["region"]
        self.view_buttons.button(2).setChecked(True)
        self.refresh_view()
        self.set_status("Done. Adjust the sliders to fine-tune, or save the photo.")
        self._update_actions()

    def _swap_ready(self, result: np.ndarray) -> None:
        self.result = result
        self.view_buttons.button(2).setChecked(True)
        self.refresh_view()
        self.set_status("Done. Save the photo or try another change.")
        self._update_actions()

    def save_result(self) -> None:
        if self.result is None:
            return
        stem = self.image_path.stem if self.image_path else "coverswap"
        default = config.OUTPUT_DIR / f"{stem}_swapped_{datetime.now():%H%M%S}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", str(default), "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return
        try:
            pipe.imwrite(path, self.result)
            self.set_status(f"Saved to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Could not save the photo", str(exc))

    def reset_result(self) -> None:
        self.result = None
        self._face_raw = None
        self.view_buttons.button(0).setChecked(True)
        self.refresh_view()
        self.set_status("Back to the original photo.")
        self._update_actions()

    # --------------------------------------------------------------- views
    def refresh_view(self) -> None:
        if self.image is None:
            return
        mode = self.view_buttons.checkedId()

        if mode == 2 and self.result is not None:
            self.canvas.set_show_boxes(False)
            self.canvas.set_image(self.result)
            return

        base = self.result if self.result is not None else self.image

        if mode == 1:
            self.canvas.set_show_boxes(True)
            if self.mode == config.MODE_CLOTHES and 0 <= self.selected < len(self.people):
                mask = self.pipeline.clothing_mask(
                    self.people[self.selected], base.shape, self.clothes_settings
                )
                self.canvas.set_image(clothing.preview_overlay(base, mask))
                return
            if self.mode == config.MODE_FACE and 0 <= self.selected < len(self.faces):
                mask = self.pipeline.face_preview_mask(self.faces[self.selected], base.shape[:2])
                overlay = clothing.preview_overlay(base, mask)
                self.canvas.set_image(overlay)
                return
            if self.mode == config.MODE_BACKGROUND and self.people:
                fg = bg.foreground_mask(self.people, base.shape, self.bg_settings.edge_grow)
                self.canvas.set_image(bg.preview_overlay(base, fg))
                return

        self.canvas.set_show_boxes(True)
        self.canvas.set_image(self.image)

    # ------------------------------------------------------------ settings
    def _category_changed(self, index: int) -> None:
        self.clothes_settings.category = self.category_combo.itemData(index)
        self.refresh_view()

    def _steps_changed(self, value: int) -> None:
        self.clothes_settings.steps = value

    def _guidance_changed(self, value: int) -> None:
        self.clothes_settings.guidance_scale = value / 10.0

    def _mask_changed(self, value: int) -> None:
        self.clothes_settings.mask_grow = value
        self.refresh_view()

    def _resolution_changed(self, state: int) -> None:
        if state == Qt.Checked:
            self.clothes_settings.width, self.clothes_settings.height = 768, 1024
        else:
            self.clothes_settings.width, self.clothes_settings.height = 576, 768

    def _tone_changed(self, value: int) -> None:
        self.face_settings.tone_match = value / 100.0
        self._refresh_face_tune()

    def _blend_changed(self, value: int) -> None:
        self.face_settings.blend = value / 100.0
        self._refresh_face_tune()

    def _bg_feather_changed(self, value: int) -> None:
        self.bg_settings.feather = value
        self.refresh_view()

    def _refresh_face_tune(self) -> None:
        if (
            self.mode != config.MODE_FACE
            or self._face_raw is None
            or self._face_mask is None
            or self.image is None
        ):
            return
        from .. import faceswap

        self.result = faceswap.finish(
            self._face_raw,
            self.image,
            self._face_mask,
            self._face_region,
            self.face_settings.tone_match,
            self.face_settings.blend,
        )
        if self.view_buttons.checkedId() == 2:
            self.canvas.set_show_boxes(False)
            self.canvas.set_image(self.result)

    # ------------------------------------------------------------- plumbing
    def _ready_to_swap(self) -> bool:
        if self.image is None or self._worker is not None:
            return False
        if self.mode == config.MODE_CLOTHES:
            return (
                self.garment is not None
                and 0 <= self.selected < len(self.people)
            )
        if self.mode == config.MODE_FACE:
            return self.source_face is not None and 0 <= self.selected < len(self.faces)
        return self.background is not None and len(self.people) > 0

    def _update_actions(self) -> None:
        busy = self._worker is not None
        ready_mode = models.ready_for_mode(self.mode)
        self.swap_button.setEnabled(ready_mode and self._ready_to_swap())
        self.save_button.setEnabled(self.result is not None and not busy)
        self.reset_button.setEnabled(self.result is not None and not busy)
        self.open_button.setEnabled(not busy)
        self.garment_button.setEnabled(not busy)
        self.face_button.setEnabled(not busy)
        self.bg_button.setEnabled(not busy)
        self.view_buttons.button(2).setEnabled(self.result is not None)

        self.setup_banner.setVisible(not models.ready())
        if not ready_mode:
            self.swap_button.setEnabled(False)
            self.swap_button.setToolTip("Download the AI models for this tab first.")
        else:
            self.swap_button.setToolTip("")

    def _start(self, worker, on_finished) -> None:
        self._worker = worker
        self.progress.setRange(0, 0)
        self.progress.show()
        self._update_actions()

        worker.progress.connect(self.set_status)
        worker.completed.connect(on_finished)
        worker.failed.connect(self._worker_failed)
        worker.completed.connect(self._worker_done)
        worker.failed.connect(self._worker_done)
        worker.start()

    def _worker_done(self, *_) -> None:
        self._worker = None
        self.progress.hide()
        self.progress.setRange(0, 1)
        self._update_actions()

    def _worker_failed(self, message: str) -> None:
        self.set_status(f"Failed: {message}")
        QMessageBox.warning(self, "That did not work", message)

    def set_status(self, message: str) -> None:
        self.status.setText(message)
