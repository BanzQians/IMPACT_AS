from pathlib import Path
from typing import List, Tuple

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def quick_start_assets_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "assets" / "quick_start"


class _GuideImageLabel(QLabel):
    def __init__(self, image_path: Path, parent=None):
        super().__init__(parent)
        self._image_path = Path(image_path)
        self._source = QPixmap(str(self._image_path))
        self._last_target_width = -1
        self.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setMinimumHeight(120)
        self._refresh_scaled()

    def _available_width(self) -> int:
        widget = self.parentWidget()
        while widget is not None:
            try:
                width = int(widget.contentsRect().width())
            except Exception:
                width = 0
            if width > 80:
                return width
            widget = widget.parentWidget()
        try:
            width = int(self.contentsRect().width())
        except Exception:
            width = 0
        return width

    def _refresh_scaled(self) -> None:
        if self._source.isNull():
            self.setText(f"Missing guide image:\n{self._image_path}")
            self.setAlignment(Qt.AlignCenter)
            return
        available_width = max(0, self._available_width())
        if available_width <= 0:
            available_width = min(int(self._source.width()), 920)
        width = max(320, min(int(self._source.width()), available_width))
        if width == self._last_target_width and self.pixmap() is not None:
            return
        scaled = self._source.scaledToWidth(width, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.setFixedHeight(scaled.height())
        self._last_target_width = width

    def resizeEvent(self, event) -> None:
        self._refresh_scaled()
        super().resizeEvent(event)


class _GuideStepCard(QFrame):
    def __init__(self, title: str, image_path: Path, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setObjectName("quickStartCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title_label = QLabel(title, self)
        title_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #17212b;")
        layout.addWidget(title_label)

        self.image = _GuideImageLabel(image_path, self)
        self.image.setStyleSheet(
            "border: 1px solid #d7dce2; border-radius: 10px; background: white;"
        )
        layout.addWidget(self.image, 1)


class QuickStartDialog(QDialog):
    """Modeless in-app quick start viewer based on annotated screenshots."""

    _STEP_IMAGES: List[Tuple[str, str]] = [
        ("Step 1. Open help from the information icon", "step_01_open_help.png"),
        ("Step 2. Start from a loaded coarse timeline", "step_02_loaded_timeline.png"),
        ("Step 3. Draw one uncertain scribble, then drag if needed", "step_03_draw_and_refine.png"),
        ("Step 4. Accept the split, then keep moving", "step_04_accept_and_cleanup.png"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IMPACT-Scribe Quick Start")
        self.resize(1080, 820)
        self._guide_images: List[_GuideImageLabel] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        intro = QLabel(
            "Follow the guide from top to bottom. Each screenshot is real and the instructions are drawn directly on the image.",
            self,
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #475467; font-size: 13px;")
        root.addWidget(intro)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(self.scroll, 1)

        body = QWidget(self.scroll)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(18)
        self.scroll.setWidget(body)

        assets_dir = quick_start_assets_dir()
        for title, filename in self._STEP_IMAGES:
            card = _GuideStepCard(title, assets_dir / filename, body)
            body_layout.addWidget(card)
            self._guide_images.append(card.image)
        body_layout.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QPushButton("Close", self)
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)
        QTimer.singleShot(0, self._refresh_images)

    def _refresh_images(self) -> None:
        for image in list(getattr(self, "_guide_images", []) or []):
            try:
                image._refresh_scaled()
            except Exception:
                continue

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_images()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._refresh_images)
