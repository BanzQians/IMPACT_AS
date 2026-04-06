import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault(
    "IMPACT_SCRIBE_SETTINGS_DIR",
    str(REPO_ROOT / ".runtime" / "impact_scribe_settings"),
)
Path(os.environ["IMPACT_SCRIBE_SETTINGS_DIR"]).mkdir(parents=True, exist_ok=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import _bootstrap_qt_runtime

_bootstrap_qt_runtime()

from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox, QWidget
from PIL import Image, ImageDraw, ImageFont

from core.query_planner import QueryCandidate, QueryDecision, QueryType
from ui.main_window import MainWindow
from utils.op_logger import OperationLogger

LEFT_GUTTER = 340
RIGHT_GUTTER = 340


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
    QInputDialog.getInt = staticmethod(
        lambda *args, **kwargs: (int(kwargs.get("value", 0) or 0), True)
    )
    QInputDialog.getText = staticmethod(
        lambda *args, **kwargs: (str(kwargs.get("text", "") or ""), True)
    )


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


def _union_rect(rects: list[QRect]) -> QRect:
    valid = [QRect(item) for item in rects if item is not None and not item.isNull()]
    if not valid:
        return QRect()
    out = QRect(valid[0])
    for item in valid[1:]:
        out = out.united(item)
    return out


def _fit_rect(rect: QRect, width: int, height: int, *, margin: int = 10) -> QRect:
    max_w = max(32, int(width) - 2 * int(margin))
    max_h = max(24, int(height) - 2 * int(margin))
    fitted_w = min(int(rect.width()), max_w)
    fitted_h = min(int(rect.height()), max_h)
    max_x = max(int(margin), int(width) - fitted_w - int(margin))
    max_y = max(int(margin), int(height) - fitted_h - int(margin))
    fitted_x = min(max(int(rect.x()), int(margin)), max_x)
    fitted_y = min(max(int(rect.y()), int(margin)), max_y)
    return QRect(fitted_x, fitted_y, fitted_w, fitted_h)


def _stack_images(
    top: QImage, bottom: QImage, *, gap: int = 22
) -> tuple[QImage, QPoint, QPoint]:
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
    return (
        canvas,
        QPoint(top_x, 0),
        QPoint(bottom_x, int(top.height() + gap)),
    )


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
    embed_text: bool = True,
) -> None:
    if not embed_text:
        canvas = _qimage_to_pil(base)
        draw = ImageDraw.Draw(canvas)
        for item in callouts:
            target = QRect(item.target)
            color = tuple(int(v) for v in item.color.getRgb()[:3])
            draw.rounded_rectangle(
                [
                    target.left() - 3,
                    target.top() - 3,
                    target.right() + 3,
                    target.bottom() + 3,
                ],
                radius=12,
                outline=(color[0], color[1], color[2], 255),
                width=5,
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(out_path))
        return

    header_h = 96
    base_pil = _qimage_to_pil(base)
    canvas = Image.new(
        "RGBA",
        (
            base_pil.width + LEFT_GUTTER + RIGHT_GUTTER,
            base_pil.height + header_h,
        ),
        (247, 248, 250, 255),
    )
    canvas.paste(base_pil, (LEFT_GUTTER, header_h))
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
        bubble = _fit_rect(
            item.bubble.translated(0, header_h), canvas.width, canvas.height
        )
        shifted = GuideCallout(
            number=int(item.number),
            text=str(item.text),
            target=item.target.translated(LEFT_GUTTER, header_h),
            bubble=bubble,
            color=QColor(item.color),
        )
        _draw_callout(draw, shifted)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))


def _left_bubble(y: int, *, width: int = 300, height: int = 62) -> QRect:
    return QRect(18, int(y), int(width), int(height))


def _right_bubble(
    base_width: int, y: int, *, width: int = 320, height: int = 62
) -> QRect:
    return QRect(
        LEFT_GUTTER + int(base_width) + 18,
        int(y),
        int(width),
        int(height),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture annotated real UI screenshots for the in-app Quick Start guide."
    )
    parser.add_argument(
        "--sample-json",
        default="test_data/20250410_1418_color_front_clipped.json",
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

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)

    win = MainWindow(logger=OperationLogger(False))
    win.resize(1500, 950)
    win.show()
    app.processEvents()

    action = win.action_window
    action_shot = action.grab().toImage()
    controls_rect = _widget_rect_in(action.ctrl_scroll, action)
    step1_crop = _clip_rect(
        controls_rect.adjusted(-14, -10, 14, 10),
        action_shot.width(),
        action_shot.height(),
    )
    step1_image = _crop_image(action_shot, step1_crop)
    actions_target = _widget_rect_in(action.combo_actions, action).translated(
        -step1_crop.x(), -step1_crop.y()
    )
    _annotate_image(
        step1_image,
        out_path=out_dir / "step_01_load_baseline.png",
        title="Step 1. Load a video or baseline",
        subtitle="Start from the top action bar. Open a session, load a video, or import an existing segmentation before review begins.",
        callouts=[
            GuideCallout(
                1,
                "Use this action menu for Open Session, Load Video, or Import JSON.",
                actions_target,
                _left_bubble(10),
            ),
        ],
        embed_text=False,
    )

    sample_video = sample_json.with_suffix(".mp4")
    if sample_video.is_file():
        ok_video = action._load_primary_video(str(sample_video))
        if not ok_video:
            raise SystemExit(f"Failed to load sample video: {sample_video}")
        app.processEvents()
    ok = action._load_json_annotations(path=str(sample_json))
    if not ok:
        raise SystemExit(f"Failed to load sample annotations: {sample_json}")
    app.processEvents()
    if action.views:
        max_end = _annotation_max_end(sample_json)
        action.views[0]["end"] = int(max_end)
        action.player.crop_end = int(max_end)
    segments = action._segments_from_store_for_interaction()
    if not segments or len(segments) < 2:
        raise SystemExit("Need at least two loaded segments to capture the scribble workflow.")

    boundary_frame = int(segments[0]["end"]) + 1
    action.timeline.set_current_frame(boundary_frame, follow=True)
    app.processEvents()

    workspace_shot = action.grab().toImage()
    player_rect = _widget_rect_in(action.player, action).adjusted(-6, -6, 6, 6)
    timeline_rect = _widget_rect_in(action.timeline, action).adjusted(-6, -6, 6, 6)
    step2_crop = _clip_rect(
        _union_rect([player_rect, timeline_rect]).adjusted(-12, -12, 12, 12),
        workspace_shot.width(),
        workspace_shot.height(),
    )
    workspace_image = _crop_image(workspace_shot, step2_crop)
    _annotate_image(
        workspace_image,
        out_path=out_dir / "step_02_loaded_workspace.png",
        title="Step 2. Inspect the loaded workspace",
        subtitle="Once the baseline is loaded, review the current video frame together with the action timeline before making edits.",
        callouts=[
            GuideCallout(
                2,
                "The video panel gives you the visual context for the current frame.",
                player_rect.translated(-step2_crop.x(), -step2_crop.y()),
                _left_bubble(18),
            ),
            GuideCallout(
                3,
                "The timeline is the baseline segmentation you will review and refine.",
                timeline_rect.translated(-step2_crop.x(), -step2_crop.y()),
                _right_bubble(workspace_image.width(), 22),
            ),
        ],
        embed_text=False,
    )

    controls_host = action.ctrl_scroll.widget()
    suggest_rect_content = _widget_rect_in(action.btn_query_suggest, controls_host)
    try:
        hbar = action.ctrl_scroll.horizontalScrollBar()
        if hbar is not None:
            hbar.setValue(max(0, int(suggest_rect_content.x()) - 180))
            app.processEvents()
    except Exception:
        pass
    controls_loaded = action.ctrl_scroll.grab().toImage()
    suggest_rect = _widget_rect_in(action.btn_query_suggest, action.ctrl_scroll)
    interaction_rect = _widget_rect_in(action.combo_interaction, action.ctrl_scroll)
    clear_rect = _widget_rect_in(action.btn_scribble_clear, action.ctrl_scroll)
    step3_crop = _clip_rect(
        _union_rect([interaction_rect, clear_rect, suggest_rect]).adjusted(-90, -18, 90, 18),
        controls_loaded.width(),
        controls_loaded.height(),
    )
    suggest_image = _crop_image(controls_loaded, step3_crop)
    _annotate_image(
        suggest_image,
        out_path=out_dir / "step_03_suggest_query.png",
        title="Step 3. Click Suggest Query",
        subtitle="When the baseline is ready, ask the planner which lightweight boundary or label question to review next.",
        callouts=[
            GuideCallout(
                4,
                "Click Suggest Query to generate the next focused review target.",
                suggest_rect.translated(-step3_crop.x(), -step3_crop.y()),
                _right_bubble(suggest_image.width(), 14),
            ),
        ],
        embed_text=False,
    )

    if not action._suggest_next_query():
        left = dict(segments[0] or {})
        right = dict(segments[1] or {})
        fallback_boundary = int(right.get("start", left.get("end", 0)) or 0)
        fallback_decision = QueryDecision(
            query_type=QueryType.BOUNDARY_SCRIBBLE,
            candidate=QueryCandidate(
                query_id=f"quickstart:boundary:{fallback_boundary}",
                query_type=QueryType.BOUNDARY_SCRIBBLE,
                start_frame=max(0, int(fallback_boundary) - 15),
                end_frame=int(fallback_boundary) + 15,
                score_terms={
                    "uncertainty": 0.85,
                    "disagreement": 0.0,
                    "multiview": 0.0,
                    "state_conflict": 0.0,
                    "propagation_gain": 0.6,
                    "history": 1.0,
                },
                estimated_cost=0.55,
                payload={
                    "boundary_frame": int(fallback_boundary),
                    "left_label": str(left.get("label", "") or "?"),
                    "right_label": str(right.get("label", "") or "?"),
                    "query_score": 0.85,
                },
            ),
            utility=0.85,
        )
        if not action._focus_query_decision(fallback_decision, status_prefix="Suggested"):
            raise SystemExit("Could not generate a query suggestion for the Quick Start guide.")
    app.processEvents()

    query_image = action.query_footer_card.grab().toImage()
    hint_target = _widget_rect_in(action.lbl_query_hint, action.query_footer_card)
    button_target = _union_rect(
        [
            _widget_rect_in(action.btn_query_refine, action.query_footer_card),
            _widget_rect_in(action.btn_scribble_accept, action.query_footer_card),
            _widget_rect_in(action.btn_query_reject, action.query_footer_card),
        ]
    )
    _annotate_image(
        query_image,
        out_path=out_dir / "step_04_review_suggestion.png",
        title="Step 4. Review the suggestion",
        subtitle="After you click Suggest Query, the footer summarizes the next lightweight boundary or label question.",
        callouts=[
            GuideCallout(
                5,
                "Read the suggested boundary or label target here before you edit anything.",
                hint_target,
                _left_bubble(14, width=320),
            ),
            GuideCallout(
                6,
                "Accept and Reject are here in the bottom-right action area.",
                button_target,
                _right_bubble(query_image.width(), 18),
            ),
        ],
        embed_text=False,
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
    footer_image = action.query_footer_card.grab().toImage()
    composite, footer_origin, row_origin = _stack_images(footer_image, row_image, gap=18)
    accept_target = _widget_rect_in(
        action.btn_scribble_accept, action.query_footer_card
    ).adjusted(-4, -2, 4, 2)
    refine_target = _widget_rect_in(
        action.btn_query_refine, action.query_footer_card
    ).adjusted(-4, -2, 4, 2)
    accept_target = accept_target.translated(footer_origin.x(), footer_origin.y())
    refine_target = refine_target.translated(footer_origin.x(), footer_origin.y())
    stroke_target = stroke_target.translated(row_origin.x(), row_origin.y())
    boundary_target = boundary_target.translated(row_origin.x(), row_origin.y())
    _annotate_image(
        composite,
        out_path=out_dir / "step_05_refine_and_accept.png",
        title="Step 5. Refine the boundary and accept",
        subtitle="Draw one uncertain stroke across the suspicious split. If the proposal is close, drag the red line and accept the suggestion.",
        callouts=[
            GuideCallout(
                7,
                "Start with one uncertain stroke across the suspicious boundary region.",
                stroke_target,
                _left_bubble(row_origin.y() + 8, width=320),
            ),
            GuideCallout(
                8,
                "If the split is slightly off, drag the red proposal line before accepting it.",
                boundary_target,
                _right_bubble(composite.width(), row_origin.y() + 10, width=320),
            ),
            GuideCallout(
                9,
                "Use Start Scribble here for local refinement, then Accept in the bottom-right to write it back.",
                _union_rect([accept_target, refine_target]),
                _right_bubble(composite.width(), 14, width=320),
            ),
        ],
        embed_text=False,
    )

    print(f"[quick_start] wrote annotated screenshots to {out_dir}")
    win.close()
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
