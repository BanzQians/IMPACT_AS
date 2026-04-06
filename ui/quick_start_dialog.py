from dataclasses import dataclass
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


@dataclass(frozen=True)
class QuickStartStep:
    title: str
    image_filename: str
    summary: str
    notes: Tuple[str, ...]


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
    def __init__(self, step: QuickStartStep, image_path: Path, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setObjectName("quickStartCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title_label = QLabel(step.title, self)
        title_label.setWordWrap(True)
        title_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #17212b;")
        layout.addWidget(title_label)

        summary_label = QLabel(step.summary, self)
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet("font-size: 14px; color: #475467; line-height: 1.4;")
        layout.addWidget(summary_label)

        if step.notes:
            notes_text = "\n".join(f"- {note}" for note in step.notes)
            notes_label = QLabel(notes_text, self)
            notes_label.setWordWrap(True)
            notes_label.setStyleSheet(
                "font-size: 14px; color: #344054; background: #f8fafc; "
                "border: 1px solid #e4e7ec; border-radius: 10px; padding: 10px 12px;"
            )
            layout.addWidget(notes_label)

        self.image = _GuideImageLabel(image_path, self)
        self.image.setStyleSheet(
            "border: 1px solid #d7dce2; border-radius: 10px; background: white;"
        )
        layout.addWidget(self.image, 1)


class QuickStartDialog(QDialog):
    """Modeless in-app quick start viewer with separate text and screenshots."""

    _STEPS: List[QuickStartStep] = [
        QuickStartStep(
            title="Step 1. Load a video or baseline",
            image_filename="step_01_load_baseline.png",
            summary="Start from the top action bar. Open a session, load a video, or import an existing segmentation before review begins.",
            notes=(
                "Use the action menu for Open Session, Load Video, or Import JSON.",
            ),
        ),
        QuickStartStep(
            title="Step 2. Inspect the loaded workspace",
            image_filename="step_02_loaded_workspace.png",
            summary="Once the baseline is loaded, review the current video frame together with the action timeline before making edits.",
            notes=(
                "The video panel gives visual context for the current frame.",
                "The timeline is the baseline segmentation you will review and refine.",
            ),
        ),
        QuickStartStep(
            title="Step 3. Click Suggest Query",
            image_filename="step_03_suggest_query.png",
            summary="When the baseline is ready, ask the planner which lightweight boundary or label question to review next.",
            notes=(
                "Click Suggest Query to generate the next focused review target.",
            ),
        ),
        QuickStartStep(
            title="Step 4. Review the suggestion",
            image_filename="step_04_review_suggestion.png",
            summary="After you click Suggest Query, the footer summarizes the next lightweight boundary or label question.",
            notes=(
                "Read the suggested boundary or label target before editing.",
                "Accept and Reject live in the bottom-right action area.",
            ),
        ),
        QuickStartStep(
            title="Step 5. Refine the boundary and accept",
            image_filename="step_05_refine_and_accept.png",
            summary="Draw one uncertain stroke across the suspicious split. If the proposal is close, drag the red line and accept the suggestion.",
            notes=(
                "Start with one uncertain stroke across the suspicious boundary region.",
                "If the split is slightly off, drag the red proposal line before accepting it.",
                "Use Start Scribble for local refinement, then Accept in the bottom-right to write it back.",
            ),
        ),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IMPACT-Scribe Quick Start")
        self.resize(1220, 900)
        self._guide_images: List[_GuideImageLabel] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        intro = QLabel(
            "Follow the guide from top to bottom. Each screenshot is real, and the instructions are listed as normal text above the image.",
            self,
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #475467; font-size: 14px;")
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
        for step in self._STEPS:
            card = _GuideStepCard(step, assets_dir / step.image_filename, body)
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
