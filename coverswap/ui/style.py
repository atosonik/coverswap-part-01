"""Dark theme for the app."""

from __future__ import annotations

BG = "#0f1116"
SURFACE = "#171a21"
SURFACE_2 = "#1e222b"
BORDER = "#2a2f3a"
TEXT = "#e6e9ef"
TEXT_DIM = "#8b93a7"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#6b9dff"
ACCENT_DOWN = "#3d76e0"
SUCCESS = "#3ecf8e"
WARNING = "#ffb454"
DANGER = "#ff5f6b"

PERSON_COLOURS = [
    "#4f8cff",
    "#3ecf8e",
    "#ffb454",
    "#ff6ec7",
    "#a78bfa",
    "#4dd0e1",
    "#ff8a65",
    "#c3e88d",
]

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
    color: {TEXT};
}}

QMainWindow, QWidget#Root {{
    background: {BG};
}}

QWidget#Card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}

QLabel#Heading {{
    font-size: 19px;
    font-weight: 600;
    color: {TEXT};
}}

QLabel#SubHeading {{
    font-size: 11px;
    font-weight: 600;
    color: {TEXT_DIM};
    letter-spacing: 1px;
}}

QLabel#Hint {{
    color: {TEXT_DIM};
    font-size: 12px;
}}

QLabel#StatusText {{
    color: {TEXT_DIM};
    font-size: 12px;
    padding: 0 4px;
}}

QPushButton {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 9px 14px;
    color: {TEXT};
}}
QPushButton:hover {{ background: #262b36; border-color: #39404e; }}
QPushButton:pressed {{ background: #12151c; }}
QPushButton:disabled {{ color: #545b6b; background: #14171e; border-color: #22262f; }}

QPushButton#Primary {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: #ffffff;
    font-weight: 600;
    padding: 11px 18px;
}}
QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#Primary:pressed {{ background: {ACCENT_DOWN}; }}
QPushButton#Primary:disabled {{ background: #24304a; border-color: #24304a; color: #6b768f; }}

QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 10px;
    min-height: 18px;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: #39404e; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {ACCENT};
    outline: none;
    padding: 4px;
}}

QListWidget {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 5px;
    outline: none;
}}
QListWidget::item {{
    padding: 9px 10px;
    border-radius: 6px;
    margin: 2px 0;
}}
QListWidget::item:hover {{ background: #262b36; }}
QListWidget::item:selected {{ background: {ACCENT}; color: #ffffff; }}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #ffffff;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_HOVER}; }}

QProgressBar {{
    background: {SURFACE_2};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2f3542; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #3c4353; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QToolTip {{
    background: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
}}

QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {SURFACE_2};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSplitter::handle {{ background: transparent; width: 8px; }}
"""
