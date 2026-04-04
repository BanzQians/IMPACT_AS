import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import _bootstrap_qt_runtime

_bootstrap_qt_runtime()

from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget
from PIL import Image, ImageDraw, ImageFont

from ui.main_window import MainWindow
from utils.op_logger import OperationLogger


@dataclass
class GuideCallout:
    number: int
    text: str
    target: QRect
    bubble: QRect
    color: QColor = field(default_factory=lambda: QColor(23, 92, 211))


def _suppress_dialogs() -> None:
    QMessageBox.information = staticmethod(lambda *args, **kwargs: QMessageBox.Ok)
    QMessageBox.warning = staticmethod(lambda *args, **kwargs: QMessageBox.Ok)
    QMessageBox.question = staticmethod(lambda *args, **kwargs: QMessageBox.Yes)


def _annotation_max_end(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 7000
    segs = list(data.get("segments") or [])
    ends = []
    for row in segs:
        if not isinstance(row, dict):
            continue
        try:
            ends.append(int(row.get("end_frame", row.get("start_frame", 0)) or 0))
        except Exception:
            continue
    return max(7000, (max(ends) + 64) if ends else 7000)


def _sample_scribble_payload(boundary_frame: int) -> dict:
    start = max(0, int(boundary_frame) - 8)
    end = max(start + 2, int(boundary_frame) + 9)
    dense_counts = {}
    for frame_i in range(start, end + 1):
        dist = abs(frame_i - int(boundary_frame))
        dense_counts[frame_i] = 3 if dist <= 1 else (2 if dist <= 3 else 1)
    return {
        "start_frame": int(start),
        "end_frame": int(end),
        "kind": "uncertain",
        "meta": {
            "path_points": [
                [float(start), 0.34],
                [float(start + 3), 0.43],
                [float(boundary_frame), 0.58],
                [float(boundary_frame + 3), 0.47],
                [float(end), 0.39],
            ],
            "frame_counts": dense_counts,
        },
    }


def _widget_rect_in(widget: QWidget, ancestor: QWidget) -> QRect:
    pos = widget.mapTo(ancestor, QPoint(0, 0))
    return QRect(int(pos.x()), int(pos.y()), int(widget.width()), int(widget.height()))


def _clip_rect(rect: QRect, width: int, height: int) -> QRect:
    return rect.intersected(QRect(0, 0, int(width), int(height)))


def _crop_image(image: QImage, rect: QRect) -> QImage:
    clipped = _clip_rect(rect, image.width(), image.height())
    return image.copy(clipped)


def _stack_images(top: QImage, bottom: QImage, *, gap: int = 22) -> tuple[QImage, int]:
    width = max(top.width(), bottom.width())
    height = int(top.height()) + int(gap) + int(bottom.height())
    canvas = QImage(width, height, QImage.Format_ARGB32)
    canvas.fill(QColor(247, 248, 250))

    painter = QPainter(canvas)
    top_x = int((width - top.width()) / 2)
    bottom_x = int((width - bottom.width()) / 2)
    painter.drawImage(top_x, 0, top)
    painter.drawImage(bottom_x, top.height() + gap, bottom)
    painter.end()
    return canvas, int(top.height() + gap)


def _qimage_to_pil(image: QImage) -> Image.Image:
    normalized = image.convertToFormat(QImage.Format_RGBA8888)
    ptr = normalized.bits()
    ptr.setsize(normalized.byteCount())
    return Image.frombuffer(
        "RGBA",
        (normalized.width(), normalized.height()),
        bytes(ptr),
        "raw",
        "RGBA",
        0,
        1,
    ).copy()


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            if Path(path).is_file():
                return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    words = str(text or "").split()
    if not words:
        return ""
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def _draw_callout(draw: ImageDraw.ImageDraw, callout: GuideCallout) -> None:
    target = QRect(callout.target)
    bubble = QRect(callout.bubble)
    color = tuple(int(v) for v in callout.color.getRgb()[:3])

    draw.rounded_rectangle(
        [
            target.left() - 2,
            target.top() - 2,
            target.right() + 2,
            target.bottom() + 2,
        ],
        radius=12,
        outline=(color[0], color[1], color[2], 255),
        width=4,
    )

    target_center = (target.center().x(), target.center().y())
    bubble_center = (bubble.center().x(), bubble.center().y())
    start = (
        bubble.right() if bubble_center[0] < target_center[0] else bubble.left(),
        bubble_center[1],
    )
    end = (
        target_center[0],
        target.top() if bubble_center[1] < target_center[1] else target.bottom(),
    )
    draw.line([start, end], fill=(color[0], color[1], color[2], 255), width=3)

    draw.rounded_rectangle(
        [bubble.left(), bubble.top(), bubble.right(), bubble.bottom()],
        radius=18,
        fill=(255, 255, 255, 248),
        outline=(24, 34, 44, 32),
        width=1,
    )

    badge_bounds = [bubble.left() + 16, bubble.top() + 14, bubble.left() + 52, bubble.top() + 50]
    draw.ellipse(badge_bounds, fill=(color[0], color[1], color[2], 255))
    badge_font = _load_font(18, bold=True)
    number_text = str(callout.number)
    badge_bbox = draw.textbbox((0, 0), number_text, font=badge_font)
    badge_x = int((badge_bounds[0] + badge_bounds[2] - (badge_bbox[2] - badge_bbox[0])) / 2)
    badge_y = int((badge_bounds[1] + badge_bounds[3] - (badge_bbox[3] - badge_bbox[1])) / 2) - 1
    draw.text((badge_x, badge_y), number_text, font=badge_font, fill=(255, 255, 255, 255))

    body_font = _load_font(17, bold=True)
    body_text = _wrap_text(draw, callout.text, body_font, max(120, bubble.width() - 84))
    draw.multiline_text(
        (bubble.left() + 64, bubble.top() + 12),
        body_text,
        font=body_font,
        fill=(23, 33, 43, 255),
        spacing=4,
    )


def _annotate_image(
    base: QImage,
    *,
    out_path: Path,
    title: str,
    subtitle: str,
    callouts: list[GuideCallout],
) -> None:
    header_h = 96
    base_pil = _qimage_to_pil(base)
    canvas = Image.new("RGBA", (base_pil.width, base_pil.height + header_h), (247, 248, 250, 255))
    canvas.paste(base_pil, (0, header_h))
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(28, bold=True)
    subtitle_font = _load_font(13, bold=False)
    draw.text((18, 14), str(title), font=title_font, fill=(23, 33, 43, 255))
    subtitle_text = _wrap_text(draw, subtitle, subtitle_font, canvas.width - 36)
    draw.multiline_text(
        (18, 52),
        subtitle_text,
        font=subtitle_font,
        fill=(91, 100, 112, 255),
        spacing=4,
    )

    for item in callouts:
        shifted = GuideCallout(
            number=int(item.number),
            text=str(item.text),
            target=item.target.translated(0, header_h),
            bubble=item.bubble.translated(0, header_h),
            color=QColor(item.color),
        )
        _draw_callout(draw, shifted)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture annotated real UI screenshots for the in-app Quick Start guide."
    )
    parser.add_argument(
        "--sample-json",
        default="test_data/20250507_1144_color_native_test.json",
        help="Sample annotation json used to populate the timeline.",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/assets/quick_start",
        help="Output directory for annotated png assets.",
    )
    args = parser.parse_args()

    sample_json = Path(args.sample_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not sample_json.is_file():
        raise SystemExit(f"Sample json not found: {sample_json}")

    _suppress_dialogs()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)

    win = MainWindow(logger=OperationLogger(False))
    win.resize(1500, 950)
    win.show()
    app.processEvents()

    action = win.action_window
    if action.views:
        max_end = _annotation_max_end(sample_json)
        action.views[0]["end"] = int(max_end)
        action.player.crop_end = int(max_end)

    main_shot = win.grab().toImage()
    quick_rect = _widget_rect_in(win.btn_quick_start, win)
    quick_crop = _clip_rect(
        QRect(
            max(0, int(quick_rect.x()) - 320),
            max(0, int(quick_rect.y()) - 14),
            430,
            64,
        ),
        main_shot.width(),
        main_shot.height(),
    )
    quick_image = _crop_image(main_shot, quick_crop)
    quick_target = quick_rect.translated(-quick_crop.x(), -quick_crop.y())
    _annotate_image(
        quick_image,
        out_path=out_dir / "step_01_open_help.png",
        title="Step 1. Open Quick Start any time",
        subtitle="The guide stays available while you work, so new users do not have to remember hidden commands.",
        callouts=[
            GuideCallout(
                1,
                "Click this information icon to reopen the guide whenever you need it.",
                quick_target,
                QRect(18, 4, 280, 56),
            )
        ],
    )

    ok = action._load_json_annotations(path=str(sample_json))
    if not ok:
        raise SystemExit(f"Failed to load sample annotations: {sample_json}")
    app.processEvents()
    segments = action._segments_from_store_for_interaction()
    if not segments or len(segments) < 2:
        raise SystemExit("Need at least two loaded segments to capture the scribble workflow.")

    boundary_frame = int(segments[0]["end"]) + 1
    action.timeline.set_current_frame(boundary_frame, follow=True)
    app.processEvents()

    timeline_image = action.timeline.grab().toImage()
    row = action._active_scribble_row()
    if row is None:
        raise SystemExit("Could not resolve the active timeline row.")
    row_rect = _widget_rect_in(row, action.timeline)
    row_rect = row_rect.adjusted(-6, -6, 6, 6)
    boundary_x_timeline = int(
        _widget_rect_in(row, action.timeline).x() + row.frame_to_x_float(float(boundary_frame))
    )
    _annotate_image(
        timeline_image,
        out_path=out_dir / "step_02_loaded_timeline.png",
        title="Step 2. Start from a loaded timeline",
        subtitle="Boundary scribble is a repair workflow. Load coarse annotations or prelabels first, then refine the suspicious splits.",
        callouts=[
            GuideCallout(
                2,
                "Work inside the timeline after you have imported a baseline segmentation.",
                row_rect,
                QRect(18, 14, 330, 62),
            ),
            GuideCallout(
                3,
                "The current playhead gives you a local frame reference while you inspect the boundary.",
                QRect(boundary_x_timeline - 10, row_rect.y(), 20, row_rect.height()),
                QRect(max(390, boundary_x_timeline + 24), 26, 330, 62),
            ),
        ],
    )

    action.enter_scribble_mode()
    payload = _sample_scribble_payload(boundary_frame)
    action._on_timeline_scribble_edited_payload(payload)
    action.timeline.set_current_frame(boundary_frame, follow=True)
    app.processEvents()

    proposal = dict(action._last_scribble_result or {})
    row = action._active_scribble_row()
    if row is None:
        raise SystemExit("Active timeline row disappeared during capture.")
    row_image = row.grab().toImage()
    stroke_start = int(payload["start_frame"])
    stroke_end = int(payload["end_frame"])
    stroke_x1 = int(row.frame_to_x_float(float(stroke_start)))
    stroke_x2 = int(row.frame_to_x_float(float(stroke_end)))
    boundary_x = int(row.frame_to_x_float(float(proposal.get("boundary_frame", boundary_frame))))
    row_h = row.height()
    stroke_target = QRect(max(0, stroke_x1 - 18), 10, max(44, stroke_x2 - stroke_x1 + 36), max(28, row_h - 20))
    boundary_target = QRect(max(0, boundary_x - 9), 0, 18, row_h)
    _annotate_image(
        row_image,
        out_path=out_dir / "step_03_draw_and_refine.png",
        title="Step 3. Draw one uncertain scribble",
        subtitle="The first stroke proposes a split. If the split is slightly off, drag the red line directly instead of starting over.",
        callouts=[
            GuideCallout(
                4,
                "Draw one freehand uncertain stroke across the suspicious split.",
                stroke_target,
                QRect(18, 6, 330, 58),
            ),
            GuideCallout(
                5,
                "If the split is close but not exact, drag the red proposal line to the better frame.",
                boundary_target,
                QRect(max(380, boundary_x + 24), 8, 360, 58),
            ),
        ],
    )

    ctrl_parent = action.btn_scribble_accept.parentWidget()
    if ctrl_parent is None:
        raise SystemExit("Could not locate the interaction controls container.")
    ctrl_image_full = ctrl_parent.grab().toImage()
    interaction_rect = action.combo_interaction.geometry()
    status_rect = action.lbl_interaction_status.geometry()
    ctrl_crop = _clip_rect(
        QRect(
            max(0, interaction_rect.x() - 48),
            0,
            min(ctrl_image_full.width(), status_rect.right() + 24) - max(0, interaction_rect.x() - 48),
            ctrl_image_full.height(),
        ),
        ctrl_image_full.width(),
        ctrl_image_full.height(),
    )
    ctrl_image = _crop_image(ctrl_image_full, ctrl_crop)
    accept_rect = action.btn_scribble_accept.geometry().translated(-ctrl_crop.x(), -ctrl_crop.y())
    row_again = row.grab().toImage()
    composite, row_offset = _stack_images(ctrl_image, row_again, gap=18)
    delete_target = QRect(max(0, stroke_x1 - 16), row_offset + 10, max(40, stroke_x2 - stroke_x1 + 32), max(28, row_h - 20))
    accept_target = accept_rect.adjusted(-4, -2, 4, 2)
    _annotate_image(
        composite,
        out_path=out_dir / "step_04_accept_and_cleanup.png",
        title="Step 4. Accept, then keep moving",
        subtitle="Accept writes the split back to the sequence. Right-click still removes a stroke or marker if you want to undo the local idea.",
        callouts=[
            GuideCallout(
                6,
                "Click Accept Proposal to commit the current split and continue to the next boundary.",
                accept_target,
                QRect(20, 44, 360, 58),
            ),
            GuideCallout(
                7,
                "Right-click a stroke or marker to delete it without leaving scribble mode.",
                delete_target,
                QRect(max(450, stroke_x2 + 40), row_offset + 10, 330, 58),
            ),
        ],
    )

    print(f"[quick_start] wrote annotated screenshots to {out_dir}")
    win.close()
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
