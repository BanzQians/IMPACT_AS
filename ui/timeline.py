from typing import Any, Dict, List, Callable, Optional, Tuple
import bisect
import math
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QSize, QTimer, QPointF
from PyQt5.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QFontMetrics,
    QPainterPath,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QWidget,
    QScrollArea,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QCheckBox,
    QSizePolicy,
    QToolButton,
    QApplication,
    QToolTip,
    QMenu,
)
from core.models import AnnotationStore, LabelDef
from utils.constants import (
    PRESET_COLORS,
    color_from_key,
    ROW_HEIGHT,
    EDGE_TOLERANCE_PX,
    DEFAULT_VIEW_SPAN,
    SNAP_RADIUS_FRAMES,
    CURRENT_FRAME_SNAP_RADIUS_FRAMES,
    PREFER_FORWARD,
    MIN_VIEW_SPAN,
    EDGE_SNAP_FRAMES,
    EXTRA_LABEL_NAME,
    EXTRA_ALIASES,
    is_extra_label,
)
import weakref

try:
    import sip  # type: ignore
except Exception:
    try:
        from PyQt5 import sip  # type: ignore
    except Exception:
        sip = None


# ---- helper: contiguous runs ----
def frames_to_runs(frames: List[int]) -> List[Tuple[int, int]]:
    if not frames:
        return []
    frames = sorted(frames)
    runs, s, e = [], frames[0], frames[0]
    for f in frames[1:]:
        if f == e + 1:
            e = f
        else:
            runs.append((s, e))
            s = e = f
    runs.append((s, e))
    return runs


def _safe_qt_call(ref, method: str, *args):
    obj = ref()
    if obj is None:
        return
    if sip is not None:
        try:
            if sip.isdeleted(obj):
                return
        except Exception:
            pass
    try:
        getattr(obj, method)(*args)
    except Exception:
        pass


class BaseTimelineRow(QWidget):
    """Shared geometry + grid/marker drawing for timeline rows."""

    def __init__(
        self,
        get_frame_count: Callable[[], int],
        get_view_start: Callable[[], int],
        get_view_span: Callable[[], int],
        get_fps: Callable[[], int],
        get_gutter: Callable[[], int],
        parent=None,
    ):
        super().__init__(parent)
        self.get_fc = get_frame_count
        self.get_vs = get_view_start
        self.get_span = get_view_span
        self.get_fps = get_fps
        self.get_gutter = get_gutter
        self._hover_frame: Optional[int] = None
        self.current_frame: Optional[int] = None
        self._flash_frame: Optional[int] = None
        self._row_dragging = False
        self._row_drag_active = False
        self._row_drag_start = None
        self._timeline_ref = None
        self._edit_mask_spans: Optional[List[Tuple[int, int]]] = None

    def frame_to_x(self, f: int) -> int:
        g = self.get_gutter()
        span = max(1, self.get_span())
        avail = max(1, self.width() - g)
        return g + int((f - self.get_vs()) * avail / span)

    def x_to_frame(self, x: int) -> int:
        g = self.get_gutter()
        span = max(1, self.get_span())
        avail = max(1, self.width() - g)
        x_adj = max(0, min(avail, x - g))
        return int(round(self.get_vs() + x_adj * span / avail))

    def x_to_frame_float(self, x: int) -> float:
        g = self.get_gutter()
        span = max(1, self.get_span())
        avail = max(1, self.width() - g)
        x_adj = max(0, min(avail, x - g))
        return float(self.get_vs()) + float(x_adj) * float(span) / float(avail)

    def frame_to_x_float(self, frame_value: float) -> float:
        g = self.get_gutter()
        span = max(1, self.get_span())
        avail = max(1, self.width() - g)
        return float(g) + (float(frame_value) - float(self.get_vs())) * float(avail) / float(span)

    def set_current_frame(self, f: Optional[int]):
        self.current_frame = f
        self.update()

    def set_boundary_flash(self, frame: Optional[int]) -> None:
        self._flash_frame = None if frame is None else int(frame)
        if frame is not None:
            ref = weakref.ref(self)
            QTimer.singleShot(800, lambda: _safe_qt_call(ref, "set_boundary_flash", None))
        self.update()

    @staticmethod
    def _normalize_spans(
        spans: Optional[List[Tuple[int, int]]],
    ) -> Optional[List[Tuple[int, int]]]:
        if spans is None:
            return None
        cleaned = []
        for seg in spans:
            try:
                s = int(seg[0])
                e = int(seg[1])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            cleaned.append((s, e))
        if not cleaned:
            return []
        cleaned.sort(key=lambda x: x[0])
        merged = []
        cur_s, cur_e = cleaned[0]
        for s, e in cleaned[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def set_edit_mask_spans(self, spans: Optional[List[Tuple[int, int]]]) -> None:
        self._edit_mask_spans = self._normalize_spans(spans)
        self.update()

    def _frame_in_edit_mask(self, frame: int) -> bool:
        spans = self._edit_mask_spans
        if spans is None:
            return True
        for s, e in spans:
            if int(s) <= int(frame) <= int(e):
                return True
            if int(frame) < int(s):
                break
        return False

    def _interval_in_edit_mask(self, start: int, end: int) -> bool:
        spans = self._edit_mask_spans
        if spans is None:
            return True
        if end < start:
            start, end = end, start
        for s, e in spans:
            if int(s) <= int(start) <= int(e):
                return int(end) <= int(e)
            if int(start) < int(s):
                break
        return False

    def _non_editable_visible_runs(self, start: int, end: int) -> List[Tuple[int, int]]:
        spans = self._edit_mask_spans
        if spans is None:
            return []
        if end < start:
            start, end = end, start
        blocked = []
        cursor = int(start)
        for s, e in spans:
            s = int(s)
            e = int(e)
            if e < cursor:
                continue
            if s > int(end):
                break
            if cursor < s:
                blocked.append((cursor, min(int(end), s - 1)))
            cursor = max(cursor, e + 1)
            if cursor > int(end):
                break
        if cursor <= int(end):
            blocked.append((cursor, int(end)))
        return blocked

    def _draw_non_editable_overlay(self, p: QPainter, start: int, end: int) -> None:
        blocked = self._non_editable_visible_runs(start, end)
        if not blocked:
            return
        base_fill = QColor(110, 110, 110, 70)
        hatch_a = QBrush(QColor(75, 75, 75, 150), Qt.BDiagPattern)
        hatch_b = QBrush(QColor(75, 75, 75, 120), Qt.FDiagPattern)
        edge_pen = QPen(QColor(95, 95, 95, 165), 1)
        for s, e in blocked:
            x1 = self.frame_to_x(int(s))
            x2 = self.frame_to_x(int(e) + 1)
            rect = QRect(x1, 0, max(1, x2 - x1), self.height())
            p.fillRect(rect, base_fill)
            p.setBrush(hatch_a)
            p.setPen(Qt.NoPen)
            p.drawRect(rect)
            p.setBrush(hatch_b)
            p.setPen(Qt.NoPen)
            p.drawRect(rect)
            p.setBrush(Qt.NoBrush)
            p.setPen(edge_pen)
            p.drawRect(rect)

    def _draw_time_grid(self, p: QPainter, start: int, end: int, fps: int):
        g = self.get_gutter()
        span = self.get_span()
        try:
            fps_f = float(fps)
        except Exception:
            fps_f = 1.0
        if not math.isfinite(fps_f) or fps_f <= 0:
            fps_f = 1.0

        # gutter separator line
        p.setPen(QPen(QColor(200, 200, 200)))
        p.drawLine(g, 0, g, self.height())

        def step_px(step_frames: int) -> float:
            avail = max(1, self.width() - g)
            return step_frames * avail / max(1, span)

        s_minor = max(1, int(round(fps_f * 0.1)))
        if step_px(s_minor) >= 4:
            p.setPen(QPen(QColor(210, 210, 210)))
            first = (start // s_minor) * s_minor
            for f in range(first, end + s_minor, s_minor):
                x = self.frame_to_x(f)
                p.drawLine(x, 12, x, self.height() - 6)

        s_mid = max(1, int(round(fps_f * 0.5)))
        if step_px(s_mid) >= 6:
            p.setPen(QPen(QColor(190, 190, 190)))
            first = (start // s_mid) * s_mid
            for f in range(first, end + s_mid, s_mid):
                x = self.frame_to_x(f)
                p.drawLine(x, 8, x, self.height() - 4)

        s_major = max(1, int(round(fps_f)))
        p.setPen(QPen(QColor(160, 160, 160)))
        first = (start // s_major) * s_major
        for f in range(first, end + s_major, s_major):
            x = self.frame_to_x(f)
            p.drawLine(x, 4, x, self.height() - 2)
            if step_px(s_major) >= 60:
                sec = int(round(float(f) / fps_f))
                p.setPen(QPen(QColor(90, 90, 90)))
                p.setFont(QFont("Arial", 8))
                txt = f"{sec}s"
                w = p.fontMetrics().width(txt)
                p.drawText(x - w // 2, 12, txt)
                p.setPen(QPen(QColor(160, 160, 160)))

    def _draw_gutter_title(self, p: QPainter, text: str):
        p.setPen(QPen(QColor(60, 60, 60)))
        p.setFont(QFont("Arial", 9))
        p.drawText(6, self.height() - 8, text)

    def _draw_current_frame_marker(self, p: QPainter, start: int, end: int):
        if self._flash_frame is not None and start <= self._flash_frame <= end:
            fx = self.frame_to_x(self._flash_frame)
            p.fillRect(QRect(fx - 3, 0, 6, self.height()), QColor(220, 0, 0, 36))
            p.setPen(QPen(QColor(220, 0, 0, 190), 4))
            p.drawLine(fx, 0, fx, self.height())
            p.setPen(QPen(QColor(220, 0, 0), 6))
            p.drawLine(fx, 0, fx, min(self.height(), 22))
        if self.current_frame is None or not (start <= self.current_frame <= end):
            return
        x = self.frame_to_x(self.current_frame)
        p.setPen(QPen(QColor(220, 0, 0), 3))
        p.drawLine(x, 0, x, self.height())

    def _draw_hover_marker(
        self, p: QPainter, start: int, end: int, fps: int, text: str
    ):
        if self._hover_frame is None or not (start <= self._hover_frame <= end):
            return
        x = self.frame_to_x(self._hover_frame)
        p.setPen(QPen(QColor(50, 120, 255, 180), 1, Qt.DashLine))
        p.drawLine(x, 0, x, self.height())
        p.setFont(QFont("Arial", 8))
        w = p.fontMetrics().width(text) + 8
        h = p.fontMetrics().height() + 6
        rx = x + 6
        if rx + w > self.width():
            rx = x - w - 6
        bubble = QRect(rx, 2, w, h)
        p.setBrush(QColor(255, 255, 255, 220))
        p.setPen(QPen(QColor(80, 80, 80)))
        p.drawRoundedRect(bubble, 3, 3)
        p.drawText(bubble.adjusted(4, 2, -4, -2), Qt.AlignLeft | Qt.AlignVCenter, text)


class TimelineRow(BaseTimelineRow):
    hoverFrame = pyqtSignal(int)
    changed = pyqtSignal()

    def __init__(
        self,
        label: LabelDef,
        store: AnnotationStore,
        get_frame_count: Callable[[], int],
        get_view_start: Callable[[], int],
        get_view_span: Callable[[], int],
        get_fps: Callable[[], int],
        get_gutter: Callable[[], int],
        title_prefix: str = "",
        parent=None,
    ):
        super().__init__(
            get_frame_count, get_view_start, get_view_span, get_fps, get_gutter, parent
        )
        self.label = label
        self.store = store
        self.title_prefix = title_prefix

        self.setMouseTracking(True)
        self.setMinimumHeight(ROW_HEIGHT)

        # drag state
        self._dragging = False
        self._mode: Optional[str] = None  # "create"|"move"|"resize_left"|"resize_right"
        self._preview_interval: Optional[Tuple[int, int]] = None
        self._active_interval: Optional[Tuple[int, int]] = None
        self._grab_offset_frames: int = 0

        # hover overlay
        self.current_hit: bool = False  # highlight when current frame falls in this row

        # create-mode anchor (fix start; only drag end)
        self._create_anchor: Optional[int] = None

        # optional snap-to-segment boundaries
        self._snap_segments: List[Tuple[int, int]] = []
        self._snap_starts: List[int] = []
        self._snap_ends: List[int] = []
        self._snap_end_set: set = set()
        self._snap_soft = False
        self._snap_radius = SNAP_RADIUS_FRAMES
        self._current_snap_radius = CURRENT_FRAME_SNAP_RADIUS_FRAMES
        self._frame_snap_radius = SNAP_RADIUS_FRAMES
        self._edge_snap_frames = EDGE_SNAP_FRAMES
        self._row_dragging = False
        self._row_drag_active = False
        self._row_drag_start = None
        self._timeline_ref = None

        # search highlight
        self.highlighted: bool = False
        self.delete_handler = None
        self.split_handler = None
        self._segment_cuts: List[int] = []

    def _gutter_px(self) -> int:
        """Left text gutter so bars don't cover the label name."""
        fm = self.fontMetrics()
        try:
            text_w = fm.horizontalAdvance(self.label.name)  # PyQt5 newer
        except AttributeError:
            text_w = fm.width(self.label.name)  # fallback
        return max(80, text_w + 16)  # minimum 80px; text width + padding

    def sizeHint(self):
        return QSize(800, ROW_HEIGHT)

    # painting
    def paintEvent(self, e):
        p = QPainter(self)
        bg = QColor(240, 240, 240)
        if self.highlighted:
            bg = QColor(255, 250, 230)
        if self.current_hit:
            bg = QColor(245, 250, 255)
        p.fillRect(self.rect(), bg)

        start = self.get_vs()
        span = self.get_span()
        end = start + span
        fps = max(1, self.get_fps())

        self._draw_time_grid(p, start, end, fps)
        self._draw_gutter_title(p, f"{self.title_prefix}{self.label.name}")

        # committed intervals
        color = color_from_key(self.label.color_name)
        fill = QBrush(color.lighter(100))
        border = QPen(color.darker(130), 2)
        for s, e_ in self._label_runs():
            if e_ < start or s > end:
                continue
            s_vis = max(s, start)
            e_vis = min(e_, end)
            x1 = self.frame_to_x(s_vis)
            x2 = self.frame_to_x(e_vis + 1)
            rect = QRect(x1, 6, max(4, x2 - x1), self.height() - 12)
            p.setBrush(fill)
            p.setPen(border)
            p.drawRoundedRect(rect, 4, 4)
            h = rect.height()
            handle_w = 6
            p.fillRect(
                QRect(rect.left() - handle_w // 2, rect.top(), handle_w, h),
                color.darker(120),
            )
            p.fillRect(
                QRect(rect.right() - handle_w // 2, rect.top(), handle_w, h),
                color.darker(120),
            )

        # preview interval
        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            if not (e_ < start or s > end):
                s_vis = max(s, start)
                e_vis = min(e_, end)
                x1 = self.frame_to_x(s_vis)
                x2 = self.frame_to_x(e_vis + 1)
                rect = QRect(x1, 8, max(4, x2 - x1), self.height() - 16)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(color.darker(150), 2, Qt.DashLine))
                p.drawRoundedRect(rect, 4, 4)
        if self._segment_cuts:
            cuts = [c for c in self._segment_cuts if start <= int(c) <= end]
            if cuts:
                p.setPen(QPen(QColor(80, 80, 80, 140), 1, Qt.DotLine))
                for c in cuts:
                    x = self.frame_to_x(int(c))
                    p.drawLine(x, 0, x, self.height())

        self._draw_non_editable_overlay(p, start, end)
        self._draw_current_frame_marker(p, start, end)
        label = (
            f"F {self._hover_frame} | {self._hover_frame / fps:.2f}s"
            if self._hover_frame is not None
            else ""
        )
        self._draw_hover_marker(p, start, end, fps, label)

    # hit tests / utils
    def _label_runs(self) -> List[Tuple[int, int]]:
        runs = frames_to_runs(self.store.frames_of(self.label.name))
        if not runs or not self._segment_cuts:
            return runs
        cut_set = {int(c) for c in self._segment_cuts if c is not None}
        split_runs = []
        for s, e in runs:
            split_points = sorted(c for c in cut_set if int(s) < c <= int(e))
            seg_start = int(s)
            for cut in split_points:
                split_runs.append((seg_start, int(cut) - 1))
                seg_start = int(cut)
            split_runs.append((seg_start, int(e)))
        return split_runs

    def _hit_interval(self, x: int) -> Tuple[Optional[Tuple[int, int]], str]:
        for s, e in self._label_runs():
            x1 = self.frame_to_x(s)
            x2 = self.frame_to_x(e + 1)
            if x1 - EDGE_TOLERANCE_PX <= x <= x2 + EDGE_TOLERANCE_PX:
                if abs(x - x1) <= EDGE_TOLERANCE_PX:
                    return (s, e), "left"
                if abs(x - x2) <= EDGE_TOLERANCE_PX:
                    return (s, e), "right"
                if x1 < x < x2:
                    return (s, e), "center"
        return None, "none"

    def set_highlighted(self, on: bool):
        if self.highlighted != on:
            self.highlighted = on
            self.update()

    def set_delete_handler(self, handler):
        self.delete_handler = handler

    def set_split_handler(self, handler):
        self.split_handler = handler

    def set_segment_cuts(self, cuts: List[int]) -> None:
        cleaned = []
        for c in cuts or []:
            try:
                cleaned.append(int(c))
            except Exception:
                continue
        self._segment_cuts = sorted(set(cleaned))

    def set_current_snap_radius(self, radius: int) -> None:
        try:
            self._current_snap_radius = max(0, int(radius))
        except Exception:
            self._current_snap_radius = CURRENT_FRAME_SNAP_RADIUS_FRAMES

    def set_frame_snap_radius(self, radius: int) -> None:
        try:
            self._frame_snap_radius = max(0, int(radius))
        except Exception:
            self._frame_snap_radius = SNAP_RADIUS_FRAMES

    def set_edge_snap_frames(self, radius: int) -> None:
        try:
            self._edge_snap_frames = max(0, int(radius))
        except Exception:
            self._edge_snap_frames = EDGE_SNAP_FRAMES

    def set_segment_snap_radius(self, radius: int) -> None:
        try:
            self._snap_radius = max(0, int(radius))
        except Exception:
            self._snap_radius = SNAP_RADIUS_FRAMES

    def set_current_hit(self, on: bool):
        if self.current_hit != on:
            self.current_hit = on
            self.update()

    def _snap_unlabeled(self, target: int) -> Optional[int]:
        fc = self.get_fc()
        target = max(0, min(target, fc - 1))
        f = self.store.nearest_unlabeled(
            target, self._frame_snap_radius, prefer_forward=PREFER_FORWARD
        )
        if f is None:
            return None
        return max(0, min(f, fc - 1))

    def _snap_to_current(self, start: int, end: int) -> Tuple[int, int]:
        """Snap endpoints to current frame within a smaller playhead radius."""
        cf = self.current_frame
        if cf is None:
            return start, end
        if abs(start - cf) <= self._current_snap_radius:
            start = cf
        if abs(end - cf) <= self._current_snap_radius:
            end = cf
        return start, end

    def set_snap_segments(self, segments: List[Tuple[int, int]]) -> None:
        cleaned = []
        for seg in segments or []:
            try:
                s = int(seg[0])
                e = int(seg[1])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            cleaned.append((s, e))
        cleaned.sort(key=lambda x: x[0])
        self._snap_segments = cleaned
        self._snap_starts = [s for s, _ in cleaned]
        self._snap_ends = sorted({e for _s, e in cleaned})
        self._snap_end_set = set(self._snap_ends)

    def _segment_bounds_for_frame(self, frame: int) -> Optional[Tuple[int, int]]:
        if not self._snap_segments:
            return None
        idx = bisect.bisect_right(self._snap_starts, int(frame)) - 1
        if idx < 0 or idx >= len(self._snap_segments):
            return None
        s, e = self._snap_segments[idx]
        if s <= frame <= e:
            return s, e
        return None

    def _nearest_in_list(self, values: List[int], frame: int) -> Optional[int]:
        if not values:
            return None
        frame = int(frame)
        idx = bisect.bisect_left(values, frame)
        candidates = []
        if idx < len(values):
            candidates.append(values[idx])
        if idx > 0:
            candidates.append(values[idx - 1])
        if not candidates:
            return None
        return min(candidates, key=lambda v: abs(v - frame))

    def _snap_to_segment_start(self, frame: int) -> int:
        if not self._snap_starts:
            return frame
        seg = self._segment_bounds_for_frame(frame)
        if seg:
            if not self._snap_soft:
                return seg[0]
            if abs(frame - seg[0]) <= self._snap_radius:
                return seg[0]
        nearest = self._nearest_in_list(self._snap_starts, frame)
        if nearest is None:
            return frame
        if self._snap_soft and abs(frame - nearest) > self._snap_radius:
            return frame
        return nearest

    def _snap_to_segment_end(self, frame: int) -> int:
        if not self._snap_ends:
            return frame
        seg = self._segment_bounds_for_frame(frame)
        if seg:
            if not self._snap_soft:
                return seg[1]
            if abs(frame - seg[1]) <= self._snap_radius:
                return seg[1]
        nearest = self._nearest_in_list(self._snap_ends, frame)
        if nearest is None:
            return frame
        if self._snap_soft and abs(frame - nearest) > self._snap_radius:
            return frame
        return nearest

    def _snap_move_start(self, cand_start: int, length: int) -> Optional[int]:
        if not self._snap_starts:
            return cand_start
        if length < 0:
            return None
        best = None
        best_dist = None
        for s in self._snap_starts:
            if (s + length) not in self._snap_end_set:
                continue
            dist = abs(s - cand_start)
            if best is None or dist < best_dist:
                best = s
                best_dist = dist
        if best is None:
            return None
        if self._snap_soft and best_dist is not None and best_dist > self._snap_radius:
            return cand_start
        return best

    # --- Snap helpers: only-left boundary snap to e+1 ---
    def _is_occ_here(self, f: int) -> bool:
        """Optional: if store.is_occupied supports row/entity queries, pass the current row; otherwise fall back to global."""
        try:
            return self.store.is_occupied(f, row=getattr(self, "row_key", None))
        except TypeError:
            return self.store.is_occupied(f)

    def _snap_edge_after_label_left(self, target: int) -> int:
        """
        Search left within EDGE_SNAP_FRAMES for the occupied->free boundary and return the next frame (e+1).
        Return -1 when no boundary is found (no snap).
        """
        fc = max(1, self.get_fc())
        t = max(0, min(target, fc - 1))
        for d in range(0, self._edge_snap_frames + 1):
            cand = t - d
            if (
                cand >= 1
                and (not self._is_occ_here(cand))
                and self._is_occ_here(cand - 1)
            ):
                return cand
        return -1

    def _interval_clamped_free(self, a: int, b: int) -> Optional[Tuple[int, int]]:
        fc = self.get_fc()
        a = max(0, min(a, fc - 1))
        b = max(0, min(b, fc - 1))
        if a > b:
            a, b = b, a
        # Manual trim cuts are virtual split markers and should not hard-block
        # later drag edits; only real occupied frames from other labels do.
        end = b
        for f in range(a, b + 1):
            if self.store.is_occupied(f) and self.store.label_at(f) != self.label.name:
                end = f - 1
                break
        if end < a:
            return None
        return (a, end)

    # mouse
    def mouseMoveEvent(self, e):
        g = self.get_gutter()
        if not self._dragging and e.x() < g:
            # Only update cursor style inside gutter; skip preview/seek
            self._hover_frame = None
            self.hoverFrame.emit(-1)
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return
        f = self.x_to_frame(e.x())
        self._hover_frame = f
        self.hoverFrame.emit(f)
        self.setToolTip(f"Frame {f}")

        if not self._dragging:
            interval, where = self._hit_interval(e.x())
            if where in ("left", "right"):
                self.setCursor(Qt.SizeHorCursor)
            elif where == "center":
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        # --- Dragging modes ---
        if self._mode == "create":
            # Start fixed at _create_anchor; drag end only; disable edge snap to avoid sticking at e+1
            start = (
                self._create_anchor
                if self._create_anchor is not None
                else (self._preview_interval[0] if self._preview_interval else f)
            )
            if self._snap_segments:
                start = self._snap_to_segment_start(start)
                end_cand = self._snap_to_segment_end(f)
            else:
                end_cand = self._snap_unlabeled(f) or f
            cand = (min(start, end_cand), max(start, end_cand))
            if not self._snap_segments:
                cand = self._snap_to_current(cand[0], cand[1])
            self._preview_interval = self._interval_clamped_free(*cand)
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.changed.emit()
            self.update()

        elif self._mode == "resize_left" and self._active_interval:
            old_s, old_e = self._active_interval
            if self._snap_segments:
                cand = min(f, old_e) if f >= old_s else f
                new_s = self._snap_to_segment_start(cand)
                if new_s > old_e:
                    new_s = old_e
                self._preview_interval = self._interval_clamped_free(
                    min(new_s, old_e), old_e
                )
            elif f >= old_s:
                # Move right => shorten; clamp to min(f, old_e)
                new_s = min(f, old_e)
                new_s, old_e = self._snap_to_current(new_s, old_e)
                self._preview_interval = self._interval_clamped_free(new_s, old_e)
            else:
                # Extend left => allow edge snap (search left for e+1); otherwise fallback to nearest unlabeled
                new_s = self._snap_edge_after_label_left(f)
                if new_s < 0:
                    new_s = self._snap_unlabeled(f)
                if new_s is not None:
                    new_s, old_e = self._snap_to_current(new_s, old_e)
                self._preview_interval = (
                    None
                    if new_s is None
                    else self._interval_clamped_free(min(new_s, old_e), old_e)
                )
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()

        elif self._mode == "resize_right" and self._active_interval:
            old_s, old_e = self._active_interval
            if self._snap_segments:
                cand = max(old_s, f) if f <= old_e else f
                new_e = self._snap_to_segment_end(cand)
                if new_e < old_s:
                    new_e = old_s
                new_s = old_s
                self._preview_interval = self._interval_clamped_free(new_s, new_e)
            elif f <= old_e:
                # Shorten
                new_e = max(old_s, f)
                new_s, new_e = self._snap_to_current(old_s, new_e)
                self._preview_interval = self._interval_clamped_free(new_s, new_e)
            else:
                # Extend right => disable edge snap; only use nearest unlabeled
                new_e = self._snap_unlabeled(f) or f
                new_s, new_e = self._snap_to_current(old_s, max(old_s, new_e))
                self._preview_interval = self._interval_clamped_free(new_s, new_e)
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()

        elif self._mode == "move" and self._active_interval:
            old_s, old_e = self._active_interval
            length = old_e - old_s
            target_s = f - self._grab_offset_frames
            cand_s = max(0, min(target_s, self.get_fc() - 1 - length))
            if self._snap_segments:
                snapped_s = self._snap_move_start(cand_s, length)
                if snapped_s is None:
                    self._preview_interval = None
                else:
                    cand_s = max(0, min(snapped_s, self.get_fc() - 1 - length))
                    cand_e = cand_s + length
                    self._preview_interval = self._interval_clamped_free(cand_s, cand_e)
            else:
                cand_e = cand_s + length
                self._preview_interval = self._interval_clamped_free(cand_s, cand_e)
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()

    def leaveEvent(self, e):
        self._hover_frame = None
        self.hoverFrame.emit(-1)
        self.update()
        return super().leaveEvent(e)

    def mousePressEvent(self, e):
        if (
            getattr(self, "on_extra_boundary", None)
            and getattr(self, "is_extra_mode", lambda: False)()
            and is_extra_label(self.label.name)
        ):
            try:
                self.on_extra_boundary(self.x_to_frame(e.x()))
            except Exception:
                pass
            return
        if e.button() != Qt.LeftButton:
            return
        g = self.get_gutter()
        if e.x() < g:
            return
        if (e.modifiers() & Qt.ControlModifier) and callable(self.split_handler):
            frame = self.x_to_frame(e.x())
            if not self._frame_in_edit_mask(frame):
                return
            try:
                handled = bool(self.split_handler(frame, self))
            except Exception:
                handled = False
            if handled:
                return
        f = self.x_to_frame(e.x())
        interval, where = self._hit_interval(e.x())
        if interval:
            if not self._interval_in_edit_mask(interval[0], interval[1]):
                return
            # start transaction
            if hasattr(self.store, "begin_txn"):
                self.store.begin_txn()

            self._dragging = True
            self._active_interval = interval
            if where == "left":
                self._mode = "resize_left"
                self.setCursor(Qt.SizeHorCursor)
            elif where == "right":
                self._mode = "resize_right"
                self.setCursor(Qt.SizeHorCursor)
            else:
                self._mode = "move"
                self.setCursor(Qt.ClosedHandCursor)
                self._grab_offset_frames = max(0, f - interval[0])
            self._preview_interval = interval
            self.update()
            return

        if not self._frame_in_edit_mask(f):
            return
        # Create in empty space: snap start to segment boundary when requested
        if self._snap_segments:
            s = self._snap_to_segment_start(f)
        else:
            s = self._snap_edge_after_label_left(f)
            if s < 0:
                s = self._snap_unlabeled(f)
        if s is None:
            return
        if not self._frame_in_edit_mask(s):
            return

        # start transaction
        if hasattr(self.store, "begin_txn"):
            self.store.begin_txn()

        self._dragging = True
        self._mode = "create"
        self._active_interval = None
        self._create_anchor = s  # lock start
        self._preview_interval = (s, s)
        self.setCursor(Qt.CrossCursor)
        self.update()

    def mouseReleaseEvent(self, e):
        if not self._dragging:
            return
        self.setCursor(Qt.ArrowCursor)

        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            if not self._interval_in_edit_mask(s, e_):
                self._preview_interval = None
        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            # 1) add frames needed for the new interval
            for f in range(s, e_ + 1):
                if (not self.store.is_occupied(f)) or (
                    self.store.label_at(f) == self.label.name
                ):
                    self.store.add(self.label.name, f)

            # 2) remove frames from the old interval that are outside the new interval
            if self._active_interval is not None:
                old_s, old_e = self._active_interval
                # left part trimmed away
                for f in range(old_s, min(s, old_e + 1)):
                    if self.store.label_at(f) == self.label.name:
                        self.store.remove_at(f)
                # right part trimmed away
                for f in range(max(e_ + 1, old_s), old_e + 1):
                    if self.store.label_at(f) == self.label.name:
                        self.store.remove_at(f)

        if hasattr(self.store, "end_txn"):
            self.store.end_txn()

        # reset drag state
        self._dragging = False
        self._mode = None
        self._active_interval = None
        self._preview_interval = None
        self._create_anchor = None
        self.changed.emit()
        self.update()

    def contextMenuEvent(self, e):
        e.accept()
        return


class CombinedTimelineRow(BaseTimelineRow):
    """
    Single-row, read-only view that paints the active label for every frame using the label's color.
    """

    hoverFrame = pyqtSignal(int)
    labelClicked = pyqtSignal(str, int)
    segmentSelected = pyqtSignal(int, int, object)
    scribbleEditedDetailed = pyqtSignal(object)
    scribbleActivated = pyqtSignal(object)
    scribbleProposalAdjusted = pyqtSignal(object)
    scribbleRemoved = pyqtSignal(object)
    changed = pyqtSignal()  # unused for now (view-only)

    def __init__(
        self,
        labels: List[LabelDef],
        row_sources: list,
        get_frame_count: Callable[[], int],
        get_view_start: Callable[[], int],
        get_view_span: Callable[[], int],
        get_fps: Callable[[], int],
        get_gutter: Callable[[], int],
        title: str = "Timeline",
        show_label_text: bool = True,
        extra_cuts: Optional[List[int]] = None,
        segment_cuts: Optional[List[int]] = None,
        editable: bool = False,
        split_on_extra_cuts: bool = False,
        parent=None,
    ):
        super().__init__(
            get_frame_count, get_view_start, get_view_span, get_fps, get_gutter, parent
        )
        self.labels = labels
        self.row_sources = row_sources or []
        self.title = title
        self.show_label_text = bool(show_label_text)
        self.editable = bool(editable)
        self.split_on_extra_cuts = bool(split_on_extra_cuts)

        self.setMouseTracking(True)
        self._row_height_normal = 44
        self._row_height_scribble = 96
        self._pre_scribble_row_height = 44
        self.setMinimumHeight(self._row_height_normal)
        self.setMaximumHeight(self._row_height_normal)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.highlight_labels: set = set()
        self._current_hits: set = set()
        self._current_hit: bool = False
        self._flash_frame: Optional[int] = None
        self._dragging = False
        self._mode: Optional[str] = None  # "create"|"resize_left"|"resize_right"
        self._preview_interval: Optional[Tuple[int, int]] = None
        self._active_interval: Optional[Tuple[int, int]] = None
        self._active_label: Optional[str] = None
        self._create_anchor: Optional[int] = None
        self._selected_interval: Optional[Tuple[int, int]] = None
        self._selected_label: Optional[str] = None
        self.delete_handler = None
        self.split_handler = None
        self._selection_scope = "segment"
        self._snap_soft = False
        self._snap_radius = SNAP_RADIUS_FRAMES
        self._current_snap_radius = CURRENT_FRAME_SNAP_RADIUS_FRAMES
        self._frame_snap_radius = SNAP_RADIUS_FRAMES
        self._edge_snap_frames = EDGE_SNAP_FRAMES
        self._snap_segments: List[Tuple[int, int]] = []
        self._snap_starts: List[int] = []
        self._snap_ends: List[int] = []
        self._snap_end_set: set = set()
        self._scribble_mode = False
        self._scribble_items: List[Dict[str, Any]] = []
        self._draft_scribble: Optional[Dict[str, Any]] = None
        self._scribble_last_frame: Optional[float] = None
        self._scribble_last_y_norm: Optional[float] = None
        self._scribble_seq = 0
        self._scribble_proposal: Optional[dict] = None
        self._scribble_press_pos: Optional[QPoint] = None
        self._scribble_press_frame: Optional[float] = None
        self._scribble_press_y_norm: Optional[float] = None
        self._scribble_press_segment: Optional[Tuple[int, int, Optional[str]]] = None
        self._scribble_cache_pixmap: Optional[QPixmap] = None
        self._scribble_cache_key: Optional[Tuple[int, int, int, int, int]] = None
        self._scribble_cache_rev = 0
        self._hover_scribble_marker_key: Optional[str] = None
        self._active_scribble_marker_key: Optional[str] = None
        self._proposal_drag_boundary_frame: Optional[int] = None

        # build color map once; rebuilt when the widget is reconstructed
        self._color_map = {lb.name: color_from_key(lb.color_name) for lb in labels}
        # share color across alias names for interaction/Extra
        for alias in EXTRA_ALIASES:
            for name, col in list(self._color_map.items()):
                if is_extra_label(name):
                    self._color_map[alias] = col
        # prefer non-interaction labels over interaction when multiple stores carry the same frame
        self._label_sources = []
        extras = []
        self._extra_cuts = list(extra_cuts or [])
        self._segment_cuts = list(segment_cuts or [])
        seen = set()
        for lb, st, _ in self.row_sources:
            if lb.name in seen:
                continue
            seen.add(lb.name)
            (extras if is_extra_label(lb.name) else self._label_sources).append(
                (lb.name, st)
            )
        self._label_sources.extend(extras)
        self._label_to_store = {}
        for lb, st, _prefix in self.row_sources:
            if lb.name not in self._label_to_store:
                self._label_to_store[lb.name] = st

    def sizeHint(self):
        return QSize(800, max(int(self.minimumHeight()), int(self.height()), 44))

    def set_base_row_height(self, height: int) -> None:
        try:
            target_h = max(44, int(height))
        except Exception:
            target_h = 44
        self._row_height_normal = int(target_h)
        if self._scribble_mode:
            self._pre_scribble_row_height = max(
                int(self._pre_scribble_row_height or 0), int(target_h)
            )
            target_h = max(int(target_h), int(self._row_height_scribble))
        else:
            self._pre_scribble_row_height = int(target_h)
        self.setMinimumHeight(int(target_h))
        self.setMaximumHeight(int(target_h))
        self.updateGeometry()
        self.update()

    def set_current_frame(self, f: Optional[int]):
        super().set_current_frame(f)
        self._update_active_scribble_marker()
        self.update()

    def resizeEvent(self, e):
        self._invalidate_scribble_cache()
        return super().resizeEvent(e)

    def set_scribble_mode(self, enabled: bool) -> None:
        self._scribble_mode = bool(enabled)
        if self._scribble_mode:
            current_h = max(
                int(self.height() or 0),
                int(self.minimumHeight() or 0),
                int(self._row_height_normal),
            )
            self._pre_scribble_row_height = max(self._pre_scribble_row_height, current_h)
            target_h = max(current_h, int(self._row_height_scribble))
            self.setMinimumHeight(target_h)
            self.setMaximumHeight(target_h)
        else:
            restore_h = max(44, int(self._pre_scribble_row_height or self._row_height_normal))
            self.setMinimumHeight(restore_h)
            self.setMaximumHeight(restore_h)
        if not self._scribble_mode and self._mode in ("scribble", "scribble_proposal_boundary"):
            self._dragging = False
            self._mode = None
            self._draft_scribble = None
            self._scribble_last_frame = None
            self._scribble_last_y_norm = None
            self._scribble_press_pos = None
            self._scribble_press_frame = None
            self._scribble_press_y_norm = None
            self._scribble_press_segment = None
            self._proposal_drag_boundary_frame = None
            self.setCursor(Qt.ArrowCursor)
        self._invalidate_scribble_cache()
        self.updateGeometry()
        self.update()

    def set_scribble_items(self, items) -> None:
        normalized: List[Dict[str, Any]] = []
        for item in list(items or []):
            payload = self._normalize_scribble_item(item)
            if payload is not None:
                normalized.append(payload)
        self._scribble_items = normalized
        self._update_active_scribble_marker()
        self._sync_hover_scribble_marker(None)
        self._invalidate_scribble_cache()
        self.update()

    def clear_scribble_items(self) -> None:
        self._scribble_items = []
        self._draft_scribble = None
        self._scribble_last_frame = None
        self._scribble_last_y_norm = None
        self._active_scribble_marker_key = None
        self._sync_hover_scribble_marker(None)
        self._invalidate_scribble_cache()
        self.update()

    def set_scribble_proposal(self, proposal) -> None:
        if not isinstance(proposal, dict):
            self._scribble_proposal = None
            self.update()
            return
        normalized = dict(proposal)
        try:
            boundary_frame = normalized.get("boundary_frame")
            if boundary_frame is not None:
                normalized["boundary_frame"] = int(boundary_frame)
        except Exception:
            normalized["boundary_frame"] = None
        for key in ("window_start", "window_end"):
            try:
                if normalized.get(key) is not None:
                    normalized[key] = int(normalized.get(key))
            except Exception:
                normalized[key] = None
        self._scribble_proposal = normalized
        self.update()

    def clear_scribble_proposal(self) -> None:
        self._scribble_proposal = None
        self._proposal_drag_boundary_frame = None
        self.update()

    def _invalidate_scribble_cache(self) -> None:
        self._scribble_cache_pixmap = None
        self._scribble_cache_key = None
        self._scribble_cache_rev = int(self._scribble_cache_rev) + 1

    def _scribble_style(self, kind: Optional[str]) -> Tuple[QColor, QColor]:
        key = str(kind or "uncertain").strip().lower()
        if key == "left":
            return QColor(46, 144, 250, 70), QColor(23, 92, 211)
        if key == "right":
            return QColor(18, 183, 106, 70), QColor(3, 152, 85)
        return QColor(247, 144, 9, 80), QColor(180, 35, 24)

    def _scribble_item_meta(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return dict(item.get("meta") or {})

    def _is_scribble_marker(self, item: Dict[str, Any]) -> bool:
        meta = self._scribble_item_meta(item)
        if bool(meta.get("accepted", False)) or bool(meta.get("archived_proposal", False)):
            return True
        if meta.get("boundary_frame") is not None and not meta.get("path_points"):
            return True
        return False

    def _scribble_marker_visual(self, item: Dict[str, Any]) -> Dict[str, Any]:
        meta = self._scribble_item_meta(item)
        accepted = bool(meta.get("accepted", False))
        confidence = None
        try:
            if meta.get("confidence", None) is not None:
                confidence = max(0.0, min(1.0, float(meta.get("confidence", 0.0))))
        except Exception:
            confidence = None
        low_conf = (not accepted) and confidence is not None and confidence < 0.45
        if accepted:
            state = "accepted"
            line_col = QColor(5, 150, 105)
            band_fill = QColor(16, 185, 129, 30)
            band_outline = QColor(5, 150, 105, 90)
            badge_fill = QColor(236, 253, 245, 245)
            badge_text = QColor(4, 120, 87)
            caption = "ACC"
            dash = False
        elif low_conf:
            state = "low"
            line_col = QColor(217, 119, 6)
            band_fill = QColor(245, 158, 11, 28)
            band_outline = QColor(217, 119, 6, 95)
            badge_fill = QColor(255, 247, 237, 245)
            badge_text = QColor(180, 83, 9)
            caption = "LOW"
            dash = True
        else:
            state = "proposal"
            line_col = QColor(23, 92, 211)
            band_fill = QColor(23, 92, 211, 22)
            band_outline = QColor(23, 92, 211, 80)
            badge_fill = QColor(239, 246, 255, 245)
            badge_text = QColor(30, 64, 175)
            caption = "PROP"
            dash = False
        conf_text = ""
        if confidence is not None:
            conf_text = f"{int(round(confidence * 100.0)):d}%"
        return {
            "state": state,
            "accepted": accepted,
            "confidence": confidence,
            "line_col": line_col,
            "band_fill": band_fill,
            "band_outline": band_outline,
            "badge_fill": badge_fill,
            "badge_text": badge_text,
            "caption": caption,
            "conf_text": conf_text,
            "dash": dash,
        }

    def _draw_scribble_marker_badge(
        self,
        p: QPainter,
        *,
        x_center: int,
        y_top: int,
        text: str,
        fill: QColor,
        text_col: QColor,
        outline: QColor,
        font_size: int = 7,
        bold: bool = True,
        padding_x: int = 5,
        height: int = 14,
    ) -> None:
        if not str(text or "").strip():
            return
        font = QFont("Arial", font_size)
        font.setBold(bool(bold))
        fm = QFontMetrics(font)
        badge_w = max(18, fm.horizontalAdvance(str(text)) + padding_x * 2)
        left = int(
            max(
                self.get_gutter() + 2,
                min(self.width() - badge_w - 2, int(x_center - badge_w // 2)),
            )
        )
        rect = QRect(int(left), int(y_top), int(badge_w), int(height))
        p.save()
        p.setFont(font)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(outline, 1))
        p.drawRoundedRect(rect, 6, 6)
        p.setPen(text_col)
        p.drawText(rect, Qt.AlignCenter, str(text))
        p.restore()

    def _scribble_marker_key(self, item: Dict[str, Any]) -> str:
        meta = self._scribble_item_meta(item)
        stroke_id = str(meta.get("stroke_id") or "").strip()
        if stroke_id:
            return f"stroke:{stroke_id}"
        session_id = str(meta.get("session_id") or "").strip()
        if session_id:
            return f"session:{session_id}"
        start_i, end_i = self._scribble_item_bounds(item)
        boundary = meta.get("boundary_frame")
        try:
            boundary_i = int(boundary) if boundary is not None else int(round((start_i + end_i) / 2.0))
        except Exception:
            boundary_i = int(round((start_i + end_i) / 2.0))
        return f"span:{item.get('kind')}:{start_i}:{end_i}:{boundary_i}"

    def _find_scribble_marker_by_key(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        marker_key = str(key or "").strip()
        if not marker_key:
            return None
        for item in list(self._scribble_items or []):
            if not self._is_scribble_marker(item):
                continue
            if self._scribble_marker_key(item) == marker_key:
                return dict(item)
        return None

    def _marker_boundary_frame(self, item: Dict[str, Any]) -> int:
        start_i, end_i = self._scribble_item_bounds(item)
        meta = self._scribble_item_meta(item)
        boundary = meta.get("boundary_frame")
        try:
            return int(boundary) if boundary is not None else int(round((start_i + end_i) / 2.0))
        except Exception:
            return int(round((start_i + end_i) / 2.0))

    def _update_active_scribble_marker(self) -> None:
        frame_i = self.current_frame
        if frame_i is None:
            self._active_scribble_marker_key = None
            return
        best_item = None
        best_score = None
        for item in list(self._scribble_items or []):
            if not self._is_scribble_marker(item):
                continue
            start_i, end_i = self._scribble_item_bounds(item)
            boundary_i = self._marker_boundary_frame(item)
            in_span = start_i <= int(frame_i) <= end_i
            near_boundary = abs(int(frame_i) - int(boundary_i)) <= 2
            if not in_span and not near_boundary:
                continue
            span_penalty = 0 if in_span else 1
            score = (span_penalty, abs(int(frame_i) - int(boundary_i)), end_i - start_i)
            if best_score is None or score < best_score:
                best_item = item
                best_score = score
        self._active_scribble_marker_key = (
            self._scribble_marker_key(best_item) if best_item is not None else None
        )

    def _marker_tooltip_text(self, item: Dict[str, Any]) -> str:
        if not self._is_scribble_marker(item):
            return ""
        visual = self._scribble_marker_visual(item)
        meta = self._scribble_item_meta(item)
        boundary_i = self._marker_boundary_frame(item)
        fps = max(1, int(self.get_fps() or 1))
        start_i, end_i = self._scribble_item_bounds(item)
        left_label = str(meta.get("left_label") or "").strip()
        right_label = str(meta.get("right_label") or "").strip()
        lines = [
            f"{visual['caption']} marker",
            f"Boundary: F {boundary_i} | {boundary_i / fps:.2f}s",
            f"Window: F {start_i}-{end_i}",
        ]
        if left_label or right_label:
            lines.append(f"Labels: {left_label or '?'} -> {right_label or '?'}")
        if str(visual.get("conf_text") or "").strip():
            lines.append(f"Confidence: {visual['conf_text']}")
        lines.append("Click to reopen and edit this boundary.")
        return "\n".join(lines)

    def _sync_hover_scribble_marker(
        self,
        item: Optional[Dict[str, Any]],
        *,
        global_pos=None,
    ) -> None:
        if item is None or not self._is_scribble_marker(item):
            self._hover_scribble_marker_key = None
            self.setToolTip("")
            try:
                QToolTip.hideText()
            except Exception:
                pass
            return
        self._hover_scribble_marker_key = self._scribble_marker_key(item)
        text = self._marker_tooltip_text(item)
        self.setToolTip(text)
        if global_pos is not None and text:
            try:
                QToolTip.showText(global_pos, text, self)
            except Exception:
                pass

    def _draw_scribble_marker_focus_overlay(
        self,
        p: QPainter,
        item: Dict[str, Any],
        start: int,
        end: int,
        *,
        hover: bool = False,
        active: bool = False,
    ) -> None:
        if not self._is_scribble_marker(item):
            return
        start_i, end_i = self._scribble_item_bounds(item)
        if end_i < start or start_i > end:
            return
        visual = self._scribble_marker_visual(item)
        boundary_i = self._marker_boundary_frame(item)
        if boundary_i < start or boundary_i > end:
            return
        line_col = QColor(visual["line_col"])
        x1 = self.frame_to_x(max(start_i, start))
        x2 = self.frame_to_x(min(end_i, end) + 1)
        x = self.frame_to_x(boundary_i)
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        if x2 > x1:
            rect = QRect(x1, 8, max(2, x2 - x1), self.height() - 16)
            outline = QColor(line_col)
            outline.setAlpha(155 if hover else 115)
            outline_pen = QPen(outline, 3 if hover else 2)
            if bool(visual["dash"]):
                outline_pen.setStyle(Qt.DashLine)
            p.setPen(outline_pen)
            fill = QColor(line_col)
            fill.setAlpha(22 if hover else 14)
            p.setBrush(QBrush(fill))
            p.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), 7, 7)
        glow = QColor(line_col)
        glow.setAlpha(80 if hover else 55)
        p.setPen(QPen(glow, 10 if hover else 8))
        p.drawLine(x, 8, x, self.height() - 8)
        core = QColor(line_col)
        core.setAlpha(220 if hover else 185)
        p.setPen(QPen(core, 3 if hover else 2))
        p.drawLine(x, 6, x, self.height() - 6)
        p.restore()

    def _proposal_boundary_hit(self, x: int, y: int) -> Optional[int]:
        if not self._scribble_mode or not isinstance(self._scribble_proposal, dict):
            return None
        if str(self._scribble_proposal.get("proposal_action", "") or "").strip() in ("remove_boundaries", "remove_segment"):
            return None
        try:
            boundary_i = self._scribble_proposal.get("boundary_frame")
            if boundary_i is None:
                return None
            boundary_i = int(boundary_i)
        except Exception:
            return None
        if not self._frame_in_edit_mask(int(boundary_i)):
            return None
        px = self.frame_to_x_float(float(boundary_i))
        if abs(float(px) - float(x)) > 8.0:
            return None
        if y < 0 or y > self.height():
            return None
        return int(boundary_i)

    def _pending_scribble_gesture(self, pos) -> Optional[str]:
        press_pos = self._scribble_press_pos
        if press_pos is None or pos is None:
            return None
        try:
            dx = int(pos.x()) - int(press_pos.x())
            dy = int(pos.y()) - int(press_pos.y())
        except Exception:
            return None
        adx = abs(int(dx))
        ady = abs(int(dy))
        base = max(6, int(QApplication.startDragDistance()))
        horizontal_threshold = max(14, base + 6)
        vertical_threshold = max(12, base + 3)
        if adx < horizontal_threshold and ady < vertical_threshold:
            return None
        if adx >= horizontal_threshold and adx >= int(round(float(ady) * 1.35)):
            return "horizontal"
        if ady >= vertical_threshold and ady >= int(round(float(adx) * 1.1)):
            return "vertical"
        # For ambiguous diagonal gestures, bias toward vertical so delete
        # intent is easier to trigger than a horizontal split/merge.
        if ady >= vertical_threshold:
            return "vertical"
        if adx >= horizontal_threshold:
            return "horizontal"
        return None

    def _clamp_proposal_boundary_frame(self, frame_i: int) -> int:
        try:
            frame_val = int(frame_i)
        except Exception:
            frame_val = 0
        fc = max(1, int(self.get_fc()))
        frame_val = max(0, min(fc - 1, frame_val))
        proposal = dict(self._scribble_proposal or {})
        try:
            start = proposal.get("window_start")
            end = proposal.get("window_end")
            if start is not None:
                frame_val = max(frame_val, int(start))
            if end is not None:
                frame_val = min(frame_val, int(end))
        except Exception:
            pass
        return int(frame_val)

    def _emit_scribble_proposal_adjusted(self, *, frame_i: int, finalized: bool) -> None:
        if not isinstance(self._scribble_proposal, dict):
            return
        frame_val = self._clamp_proposal_boundary_frame(int(frame_i))
        self._scribble_proposal["boundary_frame"] = int(frame_val)
        payload = dict(self._scribble_proposal)
        payload["boundary_frame"] = int(frame_val)
        payload["finalized"] = bool(finalized)
        self.scribbleProposalAdjusted.emit(payload)

    def _next_scribble_id(self) -> str:
        self._scribble_seq = int(self._scribble_seq) + 1
        return f"stroke-{self._scribble_seq}"

    def _scribble_lane_bounds(self) -> Tuple[int, int]:
        top = 8
        bottom = max(top + 1, self.height() - 8)
        return top, bottom

    def _scribble_y_norm_from_pos(self, y: int) -> float:
        top, bottom = self._scribble_lane_bounds()
        if bottom <= top:
            return 0.5
        y_i = max(top, min(bottom, int(y)))
        return float(y_i - top) / float(bottom - top)

    def _scribble_y_px(self, y_norm: object) -> int:
        top, bottom = self._scribble_lane_bounds()
        try:
            frac = float(y_norm)
        except Exception:
            frac = 0.5
        frac = max(0.0, min(1.0, frac))
        return int(round(top + frac * float(bottom - top)))

    def _normalize_path_points(self, points) -> List[List[float]]:
        normalized: List[List[float]] = []
        fc = max(1, self.get_fc())
        for row in list(points or []):
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            try:
                frame_i = max(0.0, min(float(fc - 1), float(row[0])))
                y_norm = max(0.0, min(1.0, float(row[1])))
            except Exception:
                continue
            normalized.append([float(frame_i), float(y_norm)])
        return normalized

    def _normalize_frame_counts(self, counts) -> Dict[int, float]:
        normalized: Dict[int, float] = {}
        fc = max(1, self.get_fc())
        if isinstance(counts, dict):
            iterator = counts.items()
        else:
            iterator = []
        for key, value in iterator:
            try:
                frame_i = max(0, min(fc - 1, int(key)))
                weight = float(value)
            except Exception:
                continue
            if not math.isfinite(weight) or weight <= 0.0:
                continue
            normalized[int(frame_i)] = normalized.get(int(frame_i), 0.0) + float(weight)
        return normalized

    def _frame_counts_from_path_points(self, points: List[List[float]]) -> Dict[int, float]:
        counts: Dict[int, float] = {}
        for row in points:
            try:
                frame_i = int(round(float(row[0])))
            except Exception:
                continue
            counts[frame_i] = counts.get(frame_i, 0.0) + 1.0
        return counts

    def _compress_scribble_path_points(
        self, points: List[List[float]], max_points: int = 48
    ) -> List[List[float]]:
        if len(points) <= 2:
            return list(points)
        filtered: List[List[float]] = []
        prev_frame = None
        prev_y = None
        for frame_f, y_norm in points:
            try:
                frame_val = float(frame_f)
                y_val = float(y_norm)
            except Exception:
                continue
            if prev_frame is not None:
                if abs(frame_val - prev_frame) < 0.35 and abs(y_val - prev_y) < 0.02:
                    continue
            filtered.append([frame_val, y_val])
            prev_frame = frame_val
            prev_y = y_val
        if len(filtered) <= max_points:
            return filtered
        last_idx = len(filtered) - 1
        keep_indices = {0, last_idx}
        for idx in range(1, max_points - 1):
            pick = int(round(idx * last_idx / float(max_points - 1)))
            keep_indices.add(max(0, min(last_idx, pick)))
        return [filtered[idx] for idx in sorted(keep_indices)]

    def _normalize_scribble_item(self, item) -> Optional[Dict[str, Any]]:
        fc = max(1, self.get_fc())
        meta: Dict[str, Any] = {}
        try:
            if isinstance(item, dict):
                s = item.get("start_frame", item.get("start", 0))
                e = item.get("end_frame", item.get("end", s))
                kind = item.get("kind", "uncertain")
                meta = dict(item.get("meta") or {})
            elif isinstance(item, (list, tuple)):
                s = item[0]
                e = item[1] if len(item) > 1 else item[0]
                kind = item[2] if len(item) > 2 else "uncertain"
            else:
                s = getattr(item, "start_frame")
                e = getattr(item, "end_frame")
                kind = getattr(item, "kind", "uncertain")
                meta = dict(getattr(item, "meta", {}) or {})
        except Exception:
            return None
        try:
            s_i = int(s)
            e_i = int(e)
        except Exception:
            return None
        if e_i < s_i:
            s_i, e_i = e_i, s_i
        path_points = self._normalize_path_points(meta.get("path_points"))
        frame_counts = self._normalize_frame_counts(meta.get("frame_counts"))
        if path_points:
            path_points = self._compress_scribble_path_points(path_points)
        if path_points:
            path_frames = [float(row[0]) for row in path_points]
            s_i = int(math.floor(min(path_frames)))
            e_i = int(math.ceil(max(path_frames)))
        s_i = max(0, min(fc - 1, s_i))
        e_i = max(0, min(fc - 1, e_i))
        kind_i = str(getattr(kind, "value", kind) or "uncertain").strip().lower()
        if kind_i not in {"uncertain", "left", "right"}:
            kind_i = "uncertain"
        if not frame_counts and path_points:
            frame_counts = self._frame_counts_from_path_points(path_points)
        stroke_id = str(meta.get("stroke_id") or "").strip() or self._next_scribble_id()
        try:
            if stroke_id.startswith("stroke-"):
                self._scribble_seq = max(
                    int(self._scribble_seq),
                    int(stroke_id.split("-", 1)[1]),
                )
        except Exception:
            pass
        meta_norm = dict(meta or {})
        meta_norm["stroke_id"] = stroke_id
        if path_points:
            meta_norm["path_points"] = path_points
        else:
            meta_norm.pop("path_points", None)
        if path_points:
            try:
                x_vals = [
                    float(self.frame_to_x_float(float(row[0]))) for row in path_points
                ]
                y_vals = [float(self._scribble_y_px(row[1])) for row in path_points]
                if x_vals and y_vals:
                    meta_norm["gesture_metrics"] = {
                        "x_span_px": float(max(x_vals) - min(x_vals)),
                        "y_span_px": float(max(y_vals) - min(y_vals)),
                    }
                else:
                    meta_norm.pop("gesture_metrics", None)
            except Exception:
                meta_norm.pop("gesture_metrics", None)
        else:
            meta_norm.pop("gesture_metrics", None)
        if frame_counts:
            meta_norm["frame_counts"] = {
                int(frame_i): int(max(1, min(255, round(float(weight)))))
                for frame_i, weight in sorted(frame_counts.items())
            }
        else:
            meta_norm.pop("frame_counts", None)
        return {
            "start_frame": int(s_i),
            "end_frame": int(e_i),
            "kind": kind_i,
            "meta": meta_norm,
        }

    def _scribble_item_bounds(self, item: Dict[str, Any]) -> Tuple[int, int]:
        try:
            return int(item.get("start_frame", 0)), int(item.get("end_frame", 0))
        except Exception:
            return 0, 0

    def _scribble_frame_counts(self, item: Dict[str, Any]) -> Dict[int, float]:
        meta = dict(item.get("meta") or {})
        counts = self._normalize_frame_counts(meta.get("frame_counts"))
        if counts:
            return counts
        start_i, end_i = self._scribble_item_bounds(item)
        return {frame_i: 1.0 for frame_i in range(int(start_i), int(end_i) + 1)}

    def _scribble_path_points(self, item: Dict[str, Any]) -> List[List[float]]:
        meta = dict(item.get("meta") or {})
        points = self._normalize_path_points(meta.get("path_points"))
        if points:
            return points
        start_i, end_i = self._scribble_item_bounds(item)
        return [[int(start_i), 0.5], [int(end_i), 0.5]]

    def _begin_draft_scribble(
        self,
        frame_i: float,
        y_norm: float,
        *,
        gesture_intent: str = "",
    ) -> Dict[str, Any]:
        payload = {
            "start_frame": int(round(frame_i)),
            "end_frame": int(round(frame_i)),
            "kind": "uncertain",
            "meta": {
                "stroke_id": self._next_scribble_id(),
                "path_points": [],
                "frame_counts": {},
            },
        }
        if str(gesture_intent or "").strip():
            payload["meta"]["gesture_intent"] = str(gesture_intent).strip().lower()
        self._draft_scribble = payload
        self._scribble_last_frame = None
        self._scribble_last_y_norm = None
        self._extend_draft_scribble(frame_i, y_norm, force_point=True)
        return payload

    def _append_scribble_point(
        self, payload: Dict[str, Any], frame_i: float, y_norm: float
    ) -> None:
        meta = payload.setdefault("meta", {})
        points = list(meta.get("path_points") or [])
        points.append([float(frame_i), float(y_norm)])
        meta["path_points"] = points
        counts = dict(meta.get("frame_counts") or {})
        frame_key = int(round(float(frame_i)))
        counts[frame_key] = float(counts.get(frame_key, 0.0)) + 1.0
        meta["frame_counts"] = counts
        payload["start_frame"] = min(
            int(payload.get("start_frame", frame_key)),
            int(math.floor(float(frame_i))),
        )
        payload["end_frame"] = max(
            int(payload.get("end_frame", frame_key)),
            int(math.ceil(float(frame_i))),
        )

    def _extend_draft_scribble(
        self, frame_i: float, y_norm: float, force_point: bool = False
    ) -> None:
        if not isinstance(self._draft_scribble, dict):
            return
        frame_i = max(0.0, min(float(max(0, self.get_fc() - 1)), float(frame_i)))
        y_norm = max(0.0, min(1.0, float(y_norm)))
        if not self._frame_in_edit_mask(int(round(frame_i))):
            return
        prev_frame = self._scribble_last_frame
        prev_y = self._scribble_last_y_norm
        if prev_frame is None or prev_y is None:
            self._append_scribble_point(self._draft_scribble, frame_i, y_norm)
            self._scribble_last_frame = float(frame_i)
            self._scribble_last_y_norm = float(y_norm)
            return
        if abs(float(frame_i) - float(prev_frame)) < 0.05:
            if force_point or abs(float(y_norm) - float(prev_y)) >= 0.02:
                self._append_scribble_point(self._draft_scribble, frame_i, y_norm)
            self._scribble_last_frame = float(frame_i)
            self._scribble_last_y_norm = float(y_norm)
            return
        frame_steps = int(math.ceil(abs(float(frame_i) - float(prev_frame)) * 2.0))
        try:
            px_delta = abs(
                float(self.frame_to_x_float(float(frame_i)))
                - float(self.frame_to_x_float(float(prev_frame)))
            )
        except Exception:
            px_delta = 0.0
        pixel_steps = int(math.ceil(px_delta / 4.0))
        steps = max(1, min(frame_steps, pixel_steps if pixel_steps > 0 else frame_steps))
        for step in range(1, steps + 1):
            t = float(step) / float(steps)
            interp_frame = float(prev_frame) + (float(frame_i) - float(prev_frame)) * t
            interp_y = float(prev_y) + (float(y_norm) - float(prev_y)) * t
            if not self._frame_in_edit_mask(int(round(interp_frame))):
                continue
            self._append_scribble_point(self._draft_scribble, interp_frame, interp_y)
        self._scribble_last_frame = float(frame_i)
        self._scribble_last_y_norm = float(y_norm)

    def _draw_scribble_item(
        self, p: QPainter, item: Dict[str, Any], start: int, end: int, draft: bool = False
    ) -> None:
        if not draft and self._is_scribble_marker(item):
            self._draw_scribble_marker(p, item, start, end)
            return
        start_i, end_i = self._scribble_item_bounds(item)
        if end_i < start or start_i > end:
            return
        kind = str(item.get("kind") or "uncertain")
        fill_col, line_col = self._scribble_style(kind)
        points = self._scribble_path_points(item)
        if len(points) < 2:
            points = [[float(start_i), 0.5], [float(end_i), 0.5]]

        visible_points = [
            row for row in points if float(start) - 1.0 <= float(row[0]) <= float(end) + 1.0
        ]
        if len(visible_points) < 2:
            visible_points = points[:2]

        path = QPainterPath()
        first = visible_points[0]
        path.moveTo(
            QPointF(self.frame_to_x_float(float(first[0])), float(self._scribble_y_px(first[1])))
        )
        if len(visible_points) == 2:
            second = visible_points[1]
            path.lineTo(
                QPointF(
                    self.frame_to_x_float(float(second[0])),
                    float(self._scribble_y_px(second[1])),
                )
            )
        else:
            for idx in range(1, len(visible_points) - 1):
                cur = visible_points[idx]
                nxt = visible_points[idx + 1]
                cx = self.frame_to_x_float(float(cur[0]))
                cy = float(self._scribble_y_px(cur[1]))
                mx = (cx + self.frame_to_x_float(float(nxt[0]))) / 2.0
                my = (cy + float(self._scribble_y_px(nxt[1]))) / 2.0
                path.quadTo(QPointF(cx, cy), QPointF(mx, my))
            last = visible_points[-1]
            path.lineTo(
                QPointF(
                    self.frame_to_x_float(float(last[0])),
                    float(self._scribble_y_px(last[1])),
                )
            )

        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)

        halo = QColor(fill_col)
        halo.setAlpha(55 if draft else 75)
        halo_pen = QPen(halo, 18 if draft else 16)
        halo_pen.setCapStyle(Qt.RoundCap)
        halo_pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(halo_pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

        main = QColor(line_col)
        main.setAlpha(135 if draft else 190)
        main_pen = QPen(main, 8 if draft else 6)
        main_pen.setCapStyle(Qt.RoundCap)
        main_pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(main_pen)
        p.drawPath(path)

        core = QColor(line_col)
        core.setAlpha(180 if draft else 240)
        core_pen = QPen(core, 3 if draft else 2)
        core_pen.setCapStyle(Qt.RoundCap)
        core_pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(core_pen)
        p.drawPath(path)
        p.restore()

    def _draw_scribble_marker(
        self, p: QPainter, item: Dict[str, Any], start: int, end: int
    ) -> None:
        start_i, end_i = self._scribble_item_bounds(item)
        if end_i < start or start_i > end:
            return
        meta = self._scribble_item_meta(item)
        visual = self._scribble_marker_visual(item)
        boundary = meta.get("boundary_frame")
        show_conf_badge = bool(str(visual.get("conf_text") or "").strip())
        try:
            boundary_i = int(boundary) if boundary is not None else int(round((start_i + end_i) / 2.0))
        except Exception:
            boundary_i = int(round((start_i + end_i) / 2.0))
        if boundary_i < start or boundary_i > end:
            return
        band_fill = QColor(visual["band_fill"])
        band_outline = QColor(visual["band_outline"])
        line_col = QColor(visual["line_col"])
        x1 = self.frame_to_x(max(start_i, start))
        x2 = self.frame_to_x(min(end_i, end) + 1)
        center_y = int(self.height() / 2)
        if x2 > x1:
            band_rect = QRect(x1, 8, max(2, x2 - x1), self.height() - 16)
            p.save()
            p.fillRect(band_rect, band_fill)
            outline_pen = QPen(band_outline, 1)
            if bool(visual["dash"]):
                outline_pen.setStyle(Qt.DashLine)
            p.setPen(outline_pen)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(band_rect.adjusted(0, 0, -1, -1), 5, 5)
            p.restore()
        x = self.frame_to_x(boundary_i)
        p.save()
        line_pen = QPen(QColor(line_col.red(), line_col.green(), line_col.blue(), 195), 2)
        if bool(visual["dash"]):
            line_pen.setStyle(Qt.DashLine)
            line_pen.setDashPattern([5.0, 3.0])
        p.setPen(line_pen)
        p.drawLine(x, 6, x, self.height() - 6)
        cap_pen = QPen(line_col, 4)
        if bool(visual["dash"]):
            cap_pen.setStyle(Qt.SolidLine)
        p.setPen(cap_pen)
        p.drawLine(x, 6, x, 18)
        p.drawLine(x, self.height() - 18, x, self.height() - 6)
        p.setBrush(QBrush(QColor(255, 255, 255, 235)))
        p.setPen(QPen(line_col, 1))
        p.drawEllipse(QPointF(float(x), float(center_y)), 4.5, 4.5)
        p.restore()
        self._draw_scribble_marker_badge(
            p,
            x_center=int(x),
            y_top=18,
            text=str(visual["caption"]),
            fill=QColor(visual["badge_fill"]),
            text_col=QColor(visual["badge_text"]),
            outline=QColor(line_col.red(), line_col.green(), line_col.blue(), 90),
            font_size=7,
            bold=True,
            padding_x=5,
            height=14,
        )
        if show_conf_badge:
            self._draw_scribble_marker_badge(
                p,
                x_center=int(x),
                y_top=36,
                text=str(visual["conf_text"]),
                fill=QColor(255, 255, 255, 235),
                text_col=QColor(line_col),
                outline=QColor(line_col.red(), line_col.green(), line_col.blue(), 70),
                font_size=7,
                bold=False,
                padding_x=4,
                height=13,
            )

    def _draw_committed_scribbles_cached(self, p: QPainter, start: int, end: int) -> None:
        if not self._scribble_items:
            return
        cache_key = (
            int(start),
            int(end),
            int(self.width()),
            int(self.height()),
            int(self._scribble_cache_rev),
        )
        if (
            self._scribble_cache_pixmap is None
            or self._scribble_cache_key != cache_key
            or self._scribble_cache_pixmap.size() != self.size()
        ):
            pix = QPixmap(self.size())
            pix.fill(Qt.transparent)
            qp = QPainter(pix)
            qp.setRenderHint(QPainter.Antialiasing, True)
            for item in self._scribble_items:
                self._draw_scribble_item(qp, item, start, end, draft=False)
            qp.end()
            self._scribble_cache_pixmap = pix
            self._scribble_cache_key = cache_key
        if self._scribble_cache_pixmap is not None:
            p.drawPixmap(0, 0, self._scribble_cache_pixmap)

    def _scribble_hit_distance(self, item: Dict[str, Any], x: int, y: int) -> float:
        if self._is_scribble_marker(item):
            meta = self._scribble_item_meta(item)
            start_i, end_i = self._scribble_item_bounds(item)
            try:
                boundary_i = int(
                    meta.get("boundary_frame")
                    if meta.get("boundary_frame") is not None
                    else round((start_i + end_i) / 2.0)
                )
            except Exception:
                boundary_i = int(round((start_i + end_i) / 2.0))
            px = self.frame_to_x_float(float(boundary_i))
            dx = float(px - float(x))
            # Use actual vertical distance from the boundary line (center of
            # the row) so that clicks far above/below do not match the marker,
            # allowing the user to start a new scribble near an existing marker.
            center_y = float(self.height()) / 2.0
            dy = max(0.0, abs(float(y) - center_y) - 20.0)
            return dx * dx + dy * dy
        best = float("inf")
        points = self._scribble_path_points(item)
        for frame_i, y_norm in points:
            px = self.frame_to_x_float(float(frame_i))
            py = self._scribble_y_px(y_norm)
            dx = float(px - float(x))
            dy = float(py - y)
            dist = dx * dx + dy * dy
            if dist < best:
                best = dist
        if best < float("inf"):
            return best
        start_i, end_i = self._scribble_item_bounds(item)
        if start_i <= self.x_to_frame(x) <= end_i:
            return 0.0
        return float("inf")

    def _find_scribble_index_at(self, x: int, y: int) -> Optional[int]:
        if not self._scribble_items:
            return None
        target_frame = self.x_to_frame(x)
        best_idx = None
        best_dist = float("inf")
        for idx, item in enumerate(self._scribble_items):
            start_i, end_i = self._scribble_item_bounds(item)
            if target_frame < start_i - 2 or target_frame > end_i + 2:
                continue
            dist = self._scribble_hit_distance(item, x, y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            return None
        if best_dist > float(28 * 28):
            return None
        return int(best_idx)

    def _pop_scribble_at(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        idx = self._find_scribble_index_at(x, y)
        if idx is None:
            return None
        return self._scribble_items.pop(int(idx))

    def _scribble_at(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        idx = self._find_scribble_index_at(x, y)
        if idx is None:
            return None
        try:
            return dict(self._scribble_items[int(idx)])
        except Exception:
            return None

    def set_delete_handler(self, handler):
        self.delete_handler = handler

    def set_split_handler(self, handler):
        self.split_handler = handler

    def set_current_snap_radius(self, radius: int) -> None:
        try:
            self._current_snap_radius = max(0, int(radius))
        except Exception:
            self._current_snap_radius = CURRENT_FRAME_SNAP_RADIUS_FRAMES

    def set_frame_snap_radius(self, radius: int) -> None:
        try:
            self._frame_snap_radius = max(0, int(radius))
        except Exception:
            self._frame_snap_radius = SNAP_RADIUS_FRAMES

    def set_edge_snap_frames(self, radius: int) -> None:
        try:
            self._edge_snap_frames = max(0, int(radius))
        except Exception:
            self._edge_snap_frames = EDGE_SNAP_FRAMES

    def set_segment_snap_radius(self, radius: int) -> None:
        try:
            self._snap_radius = max(0, int(radius))
        except Exception:
            self._snap_radius = SNAP_RADIUS_FRAMES

    def _row_drag_enabled(self) -> bool:
        tl = getattr(self, "_timeline_ref", None)
        if tl is None:
            return False
        if not callable(getattr(tl, "_combined_reorder_handler", None)):
            return False
        rows = getattr(tl, "_combined_rows", []) or []
        if len(rows) <= 1:
            return False
        return getattr(tl, "layout_mode", "") == "combined"

    def _finish_row_drag(self, global_pos):
        tl = getattr(self, "_timeline_ref", None)
        if tl is None:
            return
        try:
            tl._handle_combined_row_drop(self, global_pos)
        except Exception:
            pass

    def _color_for_label(self, name: Optional[str]) -> QColor:
        if name is None:
            return QColor(190, 190, 190)
        cached = self._color_map.get(name)
        if cached is not None:
            return cached
        normalized = str(name or "").strip()
        if normalized:
            folded = normalized.casefold()
            for existing_name, color in list(self._color_map.items()):
                existing = str(existing_name or "").strip()
                if not existing:
                    continue
                if existing == normalized or existing.casefold() == folded:
                    self._color_map[str(name)] = QColor(color)
                    return self._color_map[str(name)]
        fallback = self._fallback_color_for_name(str(name or ""))
        self._color_map[str(name)] = fallback
        return fallback

    def _fallback_color_for_name(self, name: str) -> QColor:
        palette = [key for key in PRESET_COLORS.keys() if key.lower() != "gray"]
        if not palette:
            return QColor(120, 120, 120)
        text = str(name or "").strip()
        if not text:
            return QColor(120, 120, 120)
        score = 0
        for idx, ch in enumerate(text):
            score += (idx + 1) * ord(ch)
        return color_from_key(palette[score % len(palette)])

    def _label_at(self, frame: int) -> Optional[str]:
        for name, st in self._label_sources:
            try:
                if st.label_at(frame) == name:
                    return name
            except Exception:
                continue
        return None

    def _status_color(self, status: str) -> QColor:
        mapping = {
            "PENDING": QColor("#f79009"),
            "ACTIVE": QColor("#175cd3"),
            "RESOLVED": QColor("#12b76a"),
        }
        return mapping.get(status, QColor("#667085"))

    def _label_runs(self, start: int, end: int) -> List[Tuple[int, int, Optional[str]]]:
        fc = max(1, self.get_fc()) if callable(self.get_fc) else None
        if fc is not None:
            start = max(0, min(start, fc - 1))
            end = max(0, min(end, fc - 1))
        runs = []
        if end < start:
            return runs
        cut_set = {
            int(c)
            for c in (self._segment_cuts or [])
            if c is not None and start < int(c) <= end
        }
        if self.split_on_extra_cuts:
            cut_set.update(
                int(c)
                for c in (self._extra_cuts or [])
                if c is not None and start < int(c) <= end
            )
        cur = self._label_at(start)
        s = start
        for f in range(start + 1, end + 1):
            lb = self._label_at(f)
            if f in cut_set or lb != cur:
                runs.append((s, f - 1, cur))
                s, cur = f, lb
        runs.append((s, end, cur))
        return runs

    def _segment_at(self, frame: int) -> Tuple[int, int, Optional[str]]:
        fc = max(1, self.get_fc())
        f = max(0, min(frame, fc - 1))
        cut_set = {int(c) for c in (self._segment_cuts or []) if c is not None}
        if self.split_on_extra_cuts:
            cut_set.update(int(c) for c in (self._extra_cuts or []) if c is not None)
        lb = self._label_at(f)
        s = f
        while s > 0 and self._label_at(s - 1) == lb and s not in cut_set:
            s -= 1
        e = f
        while e < fc - 1 and self._label_at(e + 1) == lb and (e + 1) not in cut_set:
            e += 1
        return s, e, lb

    def _store_for_label(self, name: Optional[str]):
        if not name:
            return None
        return self._label_to_store.get(name)

    def _is_occupied(self, frame: int) -> bool:
        return self._label_at(frame) is not None

    def _snap_unlabeled(self, target: int) -> Optional[int]:
        fc = max(1, self.get_fc())
        target = max(0, min(target, fc - 1))
        for d in range(0, self._frame_snap_radius + 1):
            f = target + d
            if f < fc and not self._is_occupied(f):
                return f
        for d in range(1, self._frame_snap_radius + 1):
            f = target - d
            if f >= 0 and not self._is_occupied(f):
                return f
        return None

    def _snap_to_current(self, start: int, end: int) -> Tuple[int, int]:
        cf = self.current_frame
        if cf is None:
            return start, end
        if abs(start - cf) <= self._current_snap_radius:
            start = cf
        if abs(end - cf) <= self._current_snap_radius:
            end = cf
        return start, end

    def set_snap_segments(self, segments: List[Tuple[int, int]]) -> None:
        cleaned = []
        for seg in segments or []:
            try:
                s = int(seg[0])
                e = int(seg[1])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            cleaned.append((s, e))
        cleaned.sort(key=lambda x: x[0])
        self._snap_segments = cleaned
        self._snap_starts = [s for s, _ in cleaned]
        self._snap_ends = sorted({e for _s, e in cleaned})
        self._snap_end_set = set(self._snap_ends)

    def _segment_bounds_for_frame(self, frame: int) -> Optional[Tuple[int, int]]:
        if not self._snap_segments:
            return None
        idx = bisect.bisect_right(self._snap_starts, int(frame)) - 1
        if idx < 0 or idx >= len(self._snap_segments):
            return None
        s, e = self._snap_segments[idx]
        if s <= frame <= e:
            return s, e
        return None

    def _nearest_in_list(self, values: List[int], frame: int) -> Optional[int]:
        if not values:
            return None
        frame = int(frame)
        idx = bisect.bisect_left(values, frame)
        candidates = []
        if idx < len(values):
            candidates.append(values[idx])
        if idx > 0:
            candidates.append(values[idx - 1])
        if not candidates:
            return None
        return min(candidates, key=lambda v: abs(v - frame))

    def _snap_to_segment_start(self, frame: int) -> int:
        if not self._snap_starts:
            return frame
        seg = self._segment_bounds_for_frame(frame)
        if seg:
            if not self._snap_soft:
                return seg[0]
            if abs(frame - seg[0]) <= self._snap_radius:
                return seg[0]
        nearest = self._nearest_in_list(self._snap_starts, frame)
        if nearest is None:
            return frame
        if self._snap_soft and abs(frame - nearest) > self._snap_radius:
            return frame
        return nearest

    def _snap_to_segment_end(self, frame: int) -> int:
        if not self._snap_ends:
            return frame
        seg = self._segment_bounds_for_frame(frame)
        if seg:
            if not self._snap_soft:
                return seg[1]
            if abs(frame - seg[1]) <= self._snap_radius:
                return seg[1]
        nearest = self._nearest_in_list(self._snap_ends, frame)
        if nearest is None:
            return frame
        if self._snap_soft and abs(frame - nearest) > self._snap_radius:
            return frame
        return nearest

    def _snap_move_start(self, cand_start: int, length: int) -> Optional[int]:
        if not self._snap_starts:
            return cand_start
        if length < 0:
            return None
        best = None
        best_dist = None
        for s in self._snap_starts:
            if (s + length) not in self._snap_end_set:
                continue
            dist = abs(s - cand_start)
            if best is None or dist < best_dist:
                best = s
                best_dist = dist
        if best is None:
            return None
        if self._snap_soft and best_dist is not None and best_dist > self._snap_radius:
            return cand_start
        return best

    def _snap_edge_after_label_left(self, target: int) -> int:
        fc = max(1, self.get_fc())
        t = max(0, min(target, fc - 1))
        for d in range(0, self._edge_snap_frames + 1):
            cand = t - d
            if (
                cand >= 1
                and (not self._is_occupied(cand))
                and self._is_occupied(cand - 1)
            ):
                return cand
        return -1

    def _interval_clamped_free(
        self, a: int, b: int, allow_label: Optional[str]
    ) -> Optional[Tuple[int, int]]:
        fc = max(1, self.get_fc())
        a = max(0, min(a, fc - 1))
        b = max(0, min(b, fc - 1))
        if a > b:
            a, b = b, a
        # Keep trim cuts as visual/selection boundaries only.
        # If a labeled segment is being edited, allow continuous dragging.
        if self.editable and allow_label:
            return (a, b)
        end = b
        for f in range(a, b + 1):
            cur = self._label_at(f)
            if cur is None:
                continue
            if allow_label and cur == allow_label:
                continue
            end = f - 1
            break
        if end < a:
            return None
        return (a, end)

    def set_editable(self, on: bool):
        self.editable = bool(on)
        self.setCursor(Qt.ArrowCursor)

    def apply_label_to_selection(self, new_label: str) -> bool:
        if not self._selected_interval:
            return False
        if not self._interval_in_edit_mask(
            self._selected_interval[0], self._selected_interval[1]
        ):
            return False
        if not new_label:
            return False
        if self._selected_label == new_label:
            return False
        new_store = self._store_for_label(new_label)
        if new_store is None:
            return False
        s, e = self._selected_interval
        touched_stores = []
        seen_ids = set()
        for f in range(s, e + 1):
            cur = self._label_at(f)
            if cur:
                st = self._store_for_label(cur)
                if st and id(st) not in seen_ids:
                    seen_ids.add(id(st))
                    touched_stores.append(st)
        if id(new_store) not in seen_ids:
            seen_ids.add(id(new_store))
            touched_stores.append(new_store)
        for st in touched_stores:
            try:
                st.begin_txn()
            except Exception:
                pass
        for f in range(s, e + 1):
            cur = self._label_at(f)
            if cur == new_label:
                continue
            if cur:
                st = self._store_for_label(cur)
                if st:
                    st.remove_at(f)
            new_store.add(new_label, f)
        for st in touched_stores:
            try:
                st.end_txn()
            except Exception:
                pass
        self._selected_label = new_label
        self.changed.emit()
        self.update()
        return True

    def _hit_edge(self, x: int) -> Optional[Tuple[Tuple[int, int], Optional[str], str]]:
        def check(interval: Tuple[int, int], label: Optional[str]):
            x1 = self.frame_to_x(interval[0])
            x2 = self.frame_to_x(interval[1] + 1)
            if abs(x - x1) <= EDGE_TOLERANCE_PX:
                return interval, label, "left"
            if abs(x - x2) <= EDGE_TOLERANCE_PX:
                return interval, label, "right"
            return None

        if self._selected_interval is not None:
            hit = check(self._selected_interval, self._selected_label)
            if hit:
                return hit
        f = self.x_to_frame(x)
        seg = self._segment_at(f)
        if seg and seg[2] is not None:
            return check((seg[0], seg[1]), seg[2])
        return None

    def set_current_frame(self, f: Optional[int]):
        self.current_frame = f
        self._update_current_hit()
        self.update()

    def set_current_hits(self, names):
        self._current_hits = set(names or [])
        self._update_current_hit()
        self.update()

    def set_highlight_labels(self, names):
        self.highlight_labels = set(names or [])
        self.update()

    def flash_labels(self, names):
        base = set(self.highlight_labels)
        flash = set(names or [])
        self.set_highlight_labels(base | flash)
        ref = weakref.ref(self)
        QTimer.singleShot(220, lambda: _safe_qt_call(ref, "set_highlight_labels", base))

    def _update_current_hit(self):
        if self.current_frame is None or not self._current_hits:
            self._current_hit = False
            return
        lb = self._label_at(self.current_frame)
        self._current_hit = lb in self._current_hits

    def _active_segment(self) -> Optional[Tuple[int, int, Optional[str]]]:
        if self.current_frame is None:
            return None
        try:
            frame_i = int(self.current_frame)
        except Exception:
            return None
        start = self.get_vs()
        end = start + self.get_span()
        if not (start <= frame_i <= end):
            return None
        try:
            return self._segment_at(frame_i)
        except Exception:
            return None

    def paintEvent(self, e):
        p = QPainter(self)
        meta = getattr(self, "_group_meta", None)
        is_phase_row = bool(isinstance(meta, dict) and meta.get("row_type") == "phase")
        bg = QColor(240, 240, 240)
        if is_phase_row:
            bg = QColor(232, 238, 248)
        if self._current_hit:
            bg = QColor(245, 250, 255) if not is_phase_row else QColor(225, 240, 252)
        p.fillRect(self.rect(), bg)

        start = self.get_vs()
        span = self.get_span()
        end = start + span
        fps = max(1, self.get_fps())

        self._draw_time_grid(p, start, end, fps)
        self._draw_gutter_title(p, self.title)
        active_segment = self._active_segment()

        # draw label runs
        runs = self._label_runs(start, end)
        for s, e_, lb in runs:
            if lb is None:
                continue
            s_vis = max(s, start)
            e_vis = min(e_, end)
            x1 = self.frame_to_x(s_vis)
            x2 = self.frame_to_x(e_vis + 1)
            rect = QRect(x1, 6, max(4, x2 - x1), self.height() - 12)
            is_active_segment = bool(
                active_segment
                and int(active_segment[0]) == int(s)
                and int(active_segment[1]) == int(e_)
                and active_segment[2] == lb
            )
            base_col = self._color_for_label(lb)
            fill_col = QColor(base_col)
            if is_phase_row:
                fill_col = fill_col.lighter(115)
                fill_col.setAlpha(170)
            if lb in self.highlight_labels:
                fill_col = fill_col.lighter(130)
            if is_active_segment:
                accent_fill = QColor(fill_col)
                accent_fill.setAlpha(38 if is_phase_row else 34)
                p.setBrush(QBrush(accent_fill))
                p.setPen(QPen(base_col.darker(150), 3))
                p.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), 6, 6)
            p.setBrush(QBrush(fill_col.lighter(100)))
            border_col = base_col.darker(170) if is_phase_row else base_col.darker(140)
            border_w = 3 if is_active_segment else (2 if lb in self.highlight_labels else 1)
            p.setPen(QPen(border_col, border_w))
            p.drawRoundedRect(rect, 4, 4)
            if lb and self.show_label_text:
                font = QFont("Arial", 9 if is_active_segment or rect.height() >= 56 else 8)
                font.setBold(bool(is_active_segment and rect.width() >= 64))
                p.setFont(font)
                p.setPen(QPen(QColor(24, 24, 27) if is_active_segment else QColor(40, 40, 40)))
                text = str(lb)
                text_w = max(0, rect.width() - 10)
                if text_w >= 28:
                    elided = p.fontMetrics().elidedText(text, Qt.ElideRight, text_w)
                    p.drawText(
                        rect.adjusted(5, 2, -5, -2),
                        Qt.AlignLeft | Qt.AlignVCenter,
                        elided,
                    )

        # selection highlight
        if self._selected_interval is not None:
            s, e_ = self._selected_interval
            s_vis = max(s, start)
            e_vis = min(e_, end)
            if s_vis <= e_vis:
                x1 = self.frame_to_x(s_vis)
                x2 = self.frame_to_x(e_vis + 1)
                rect = QRect(x1, 4, max(4, x2 - x1), self.height() - 8)
                sel_col = self._color_for_label(self._selected_label)
                pen = QPen(sel_col.darker(150), 3)
                if self._selected_label is None:
                    pen.setStyle(Qt.DashLine)
                p.setBrush(Qt.NoBrush)
                p.setPen(pen)
                p.drawRoundedRect(rect, 6, 6)

        # preview interval while dragging
        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            s_vis = max(s, start)
            e_vis = min(e_, end)
            if s_vis <= e_vis:
                x1 = self.frame_to_x(s_vis)
                x2 = self.frame_to_x(e_vis + 1)
                rect = QRect(x1, 6, max(4, x2 - x1), self.height() - 12)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(70, 90, 120), 2, Qt.DashLine))
                p.drawRoundedRect(rect, 4, 4)

        # temporal scribble overlays
        if self._scribble_items:
            self._draw_committed_scribbles_cached(p, start, end)
            hover_item = self._find_scribble_marker_by_key(self._hover_scribble_marker_key)
            active_item = self._find_scribble_marker_by_key(self._active_scribble_marker_key)
            if active_item is not None:
                self._draw_scribble_marker_focus_overlay(
                    p, active_item, start, end, hover=False, active=True
                )
            if hover_item is not None:
                self._draw_scribble_marker_focus_overlay(
                    p, hover_item, start, end, hover=True, active=(
                        active_item is not None
                        and self._scribble_marker_key(active_item)
                        == self._scribble_marker_key(hover_item)
                    )
                )

        if self._draft_scribble is not None:
            self._draw_scribble_item(p, self._draft_scribble, start, end, draft=True)

        if self._scribble_proposal:
            proposal = dict(self._scribble_proposal)
            boundary_frame = proposal.get("boundary_frame")
            window_start = proposal.get("window_start")
            window_end = proposal.get("window_end")
            proposal_action = str(proposal.get("proposal_action", "") or "").strip()
            if boundary_frame is not None:
                try:
                    boundary_frame = int(boundary_frame)
                except Exception:
                    boundary_frame = None
            if window_start is not None and window_end is not None:
                try:
                    window_start = int(window_start)
                    window_end = int(window_end)
                except Exception:
                    window_start = window_end = None
            if (
                window_start is not None
                and window_end is not None
                and not (window_end < start or window_start > end)
            ):
                s_vis = max(window_start, start)
                e_vis = min(window_end, end)
                x1 = self.frame_to_x(s_vis)
                x2 = self.frame_to_x(e_vis + 1)
                rect = QRect(x1, 10, max(4, x2 - x1), self.height() - 20)
                if proposal_action == "merge_then_split":
                    fill_col = QColor(176, 96, 20, 28)
                    line_col = QColor(176, 96, 20, 190)
                elif proposal_action in ("remove_boundary", "remove_boundaries", "remove_segment"):
                    fill_col = QColor(185, 28, 28, 24)
                    line_col = QColor(185, 28, 28, 185)
                else:
                    fill_col = QColor(23, 92, 211, 28)
                    line_col = QColor(23, 92, 211, 180)
                p.fillRect(rect, fill_col)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(line_col, 2, Qt.DashLine))
                p.drawRoundedRect(rect, 4, 4)
            if boundary_frame is not None and start <= boundary_frame <= end:
                x = self.frame_to_x(boundary_frame)
                if proposal_action == "merge_then_split":
                    line_pen = QPen(QColor(23, 92, 211, 180), 2, Qt.DashLine)
                    tick_pen = QPen(QColor(23, 92, 211), 4)
                elif proposal_action in ("remove_boundary", "remove_boundaries", "remove_segment"):
                    line_pen = QPen(QColor(185, 28, 28, 180), 2, Qt.DashLine)
                    tick_pen = QPen(QColor(185, 28, 28), 4)
                else:
                    line_pen = QPen(QColor(23, 92, 211, 180), 2, Qt.DashLine)
                    tick_pen = QPen(QColor(23, 92, 211), 4)
                p.setPen(line_pen)
                p.drawLine(x, 0, x, self.height())
                p.setPen(tick_pen)
                p.drawLine(x, 0, x, 16)
                conf = proposal.get("confidence")
                left_label = str(proposal.get("left_label", "") or "").strip()
                right_label = str(proposal.get("right_label", "") or "").strip()
                caption = "Proposal"
                try:
                    if conf is not None:
                        caption = f"P {float(conf):.2f}"
                except Exception:
                    pass
                if proposal_action == "merge_then_split":
                    caption = "Merge+Split"
                    if left_label or right_label:
                        caption = f"{caption} {left_label or '?'}|{right_label or '?'}"
                elif proposal_action == "remove_boundaries":
                    merged_label = str(proposal.get("merged_label", "") or left_label or right_label or "").strip()
                    removed_count = int(proposal.get("removed_boundary_count", 0) or 0)
                    caption = (
                        f"Remove {removed_count:d}" if removed_count > 0 else "Remove span"
                    )
                    if merged_label:
                        caption = f"{caption} {merged_label}"
                elif proposal_action == "remove_segment":
                    seg_label = str(proposal.get("segment_label", "") or "").strip()
                    caption = "Delete segment"
                    if seg_label:
                        caption = f"{caption} {seg_label}"
                elif proposal_action == "remove_boundary":
                    merged_label = str(proposal.get("merged_label", "") or left_label or right_label or "").strip()
                    caption = f"Remove {float(conf):.2f}" if conf is not None else "Remove"
                    if merged_label:
                        caption = f"{caption} {merged_label}"
                elif left_label or right_label:
                    caption = f"{caption} {left_label or '?'}|{right_label or '?'}"
                p.setFont(QFont("Arial", 7))
                fm = p.fontMetrics()
                text_w = max(24, fm.horizontalAdvance(caption) + 8)
                tx = max(self.get_gutter() + 2, min(x + 6, self.width() - text_w - 2))
                badge = QRect(tx, 2, text_w, 12)
                p.fillRect(badge, QColor(255, 255, 255, 220))
                if proposal_action == "merge_then_split":
                    badge_col = QColor(176, 96, 20)
                elif proposal_action in ("remove_boundary", "remove_boundaries"):
                    badge_col = QColor(185, 28, 28)
                else:
                    badge_col = QColor(23, 92, 211)
                p.setPen(QPen(badge_col, 1))
                p.drawRect(badge)
                p.drawText(badge.adjusted(3, 0, -3, 0), Qt.AlignLeft | Qt.AlignVCenter, caption)

        # manual segment cuts overlay (split markers)
        if self._segment_cuts:
            cuts = [c for c in self._segment_cuts if start <= int(c) <= end]
            if cuts:
                p.setPen(QPen(QColor(80, 80, 80, 140), 1, Qt.DotLine))
                for c in cuts:
                    x = self.frame_to_x(int(c))
                    p.drawLine(x, 0, x, self.height())

        self._draw_non_editable_overlay(p, start, end)
        self._draw_current_frame_marker(p, start, end)
        if self._hover_frame is not None:
            label = self._label_at(self._hover_frame) or "Unlabeled"
            txt = f"{label} | F {self._hover_frame} | {self._hover_frame / fps:.2f}s"
        else:
            txt = ""
        self._draw_hover_marker(p, start, end, fps, txt)

    def mouseMoveEvent(self, e):
        if self._row_dragging:
            if self._row_drag_start is None:
                self._row_drag_start = e.pos()
            if not self._row_drag_active:
                dist = (e.pos() - self._row_drag_start).manhattanLength()
                if dist >= QApplication.startDragDistance():
                    self._row_drag_active = True
                    self.setCursor(Qt.ClosedHandCursor)
            return
        g = self.get_gutter()
        if e.x() < g:
            self._hover_frame = None
            self.hoverFrame.emit(-1)
            self._sync_hover_scribble_marker(None)
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return
        f = self.x_to_frame(e.x())
        self._hover_frame = f
        hover_marker = self._scribble_at(e.x(), e.y())
        if hover_marker is not None and self._is_scribble_marker(hover_marker):
            self._sync_hover_scribble_marker(hover_marker, global_pos=e.globalPos())
        else:
            self._sync_hover_scribble_marker(None)
        self.setCursor(Qt.ArrowCursor)
        self.hoverFrame.emit(f)
        self.update()
        if self._scribble_mode:
            if self._mode == "scribble_pending":
                press_pos = self._scribble_press_pos
                if press_pos is None:
                    self._proposal_hover_boundary_frame = None
                    self.setToolTip("")
                    self.setCursor(
                        Qt.CrossCursor if self._frame_in_edit_mask(f) else Qt.ForbiddenCursor
                    )
                    return
                gesture_commit = self._pending_scribble_gesture(e.pos())
                if gesture_commit is None:
                    self._proposal_hover_boundary_frame = None
                    self.setToolTip(
                        "Click a segment to relabel it. Drag horizontally to draw a boundary gesture, or drag vertically across a boundary to remove it."
                    )
                    self.setCursor(
                        Qt.CrossCursor if self._frame_in_edit_mask(f) else Qt.ForbiddenCursor
                    )
                    return
                start_frame = (
                    float(self._scribble_press_frame)
                    if self._scribble_press_frame is not None
                    else self.x_to_frame_float(e.x())
                )
                start_y_norm = (
                    float(self._scribble_press_y_norm)
                    if self._scribble_press_y_norm is not None
                    else float(self._scribble_y_norm_from_pos(e.y()))
                )
                self._dragging = True
                self._mode = "scribble"
                self._begin_draft_scribble(
                    start_frame,
                    start_y_norm,
                    gesture_intent=str(gesture_commit or ""),
                )
                self._extend_draft_scribble(
                    self.x_to_frame_float(e.x()),
                    self._scribble_y_norm_from_pos(e.y()),
                )
                self.setCursor(Qt.CrossCursor)
                self.setToolTip("Release to finish the boundary gesture.")
                self.update()
                return
            if not self._dragging:
                proposal_hit = self._proposal_boundary_hit(e.x(), e.y())
                edge_hit = self._hit_edge(e.x()) if self.editable else None
                if hover_marker is not None and self._is_scribble_marker(hover_marker):
                    self.setCursor(Qt.PointingHandCursor)
                elif proposal_hit is not None:
                    self.setCursor(Qt.SizeHorCursor)
                    self.setToolTip("Drag to adjust the proposed boundary.")
                elif edge_hit:
                    self.setCursor(Qt.SizeHorCursor)
                    self.setToolTip("Drag to adjust the existing segment boundary.")
                else:
                    self.setToolTip("")
                    self.setCursor(
                        Qt.CrossCursor
                        if self._frame_in_edit_mask(f)
                        else Qt.ForbiddenCursor
                    )
                return
            if self._mode == "scribble_proposal_boundary":
                boundary_i = self._clamp_proposal_boundary_frame(self.x_to_frame(e.x()))
                self._proposal_drag_boundary_frame = int(boundary_i)
                if isinstance(self._scribble_proposal, dict):
                    self._scribble_proposal["boundary_frame"] = int(boundary_i)
                self._emit_scribble_proposal_adjusted(
                    frame_i=int(boundary_i), finalized=False
                )
                self.setCursor(Qt.SizeHorCursor)
                self.update()
                return
            if self._mode == "scribble":
                self._extend_draft_scribble(
                    self.x_to_frame_float(e.x()),
                    self._scribble_y_norm_from_pos(e.y()),
                )
                self.update()
                return
        if not self._dragging and hover_marker is not None and self._is_scribble_marker(hover_marker):
            self.setCursor(Qt.PointingHandCursor)
            return
        if not self.editable:
            return

        if not self._dragging:
            hit = self._hit_edge(e.x())
            if hit:
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        if self._mode == "create":
            start = (
                self._create_anchor
                if self._create_anchor is not None
                else (self._preview_interval[0] if self._preview_interval else f)
            )
            if self._snap_segments:
                start = self._snap_to_segment_start(start)
                end_cand = self._snap_to_segment_end(f)
            else:
                end_cand = self._snap_unlabeled(f) or f
            cand = (min(start, end_cand), max(start, end_cand))
            if not self._snap_segments:
                cand = self._snap_to_current(cand[0], cand[1])
            self._preview_interval = self._interval_clamped_free(cand[0], cand[1], None)
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()
            return

        if self._mode == "resize_left" and self._active_interval:
            old_s, old_e = self._active_interval
            if self._snap_segments:
                cand = min(f, old_e) if f >= old_s else f
                new_s = self._snap_to_segment_start(cand)
                if new_s > old_e:
                    new_s = old_e
                self._preview_interval = self._interval_clamped_free(
                    min(new_s, old_e), old_e, self._active_label
                )
            elif f >= old_s:
                new_s = min(f, old_e)
                if not (self.editable and self._active_label):
                    new_s, old_e = self._snap_to_current(new_s, old_e)
                self._preview_interval = self._interval_clamped_free(
                    new_s, old_e, self._active_label
                )
            else:
                new_s = self._snap_edge_after_label_left(f)
                if new_s < 0:
                    new_s = self._snap_unlabeled(f)
                if new_s is None and self.editable and self._active_label:
                    new_s = f
                if new_s is not None:
                    if not (self.editable and self._active_label):
                        new_s, old_e = self._snap_to_current(new_s, old_e)
                self._preview_interval = (
                    None
                    if new_s is None
                    else self._interval_clamped_free(
                        min(new_s, old_e), old_e, self._active_label
                    )
                )
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()
            return

        if self._mode == "resize_right" and self._active_interval:
            old_s, old_e = self._active_interval
            if self._snap_segments:
                cand = max(old_s, f) if f <= old_e else f
                new_e = self._snap_to_segment_end(cand)
                if new_e < old_s:
                    new_e = old_s
                self._preview_interval = self._interval_clamped_free(
                    old_s, new_e, self._active_label
                )
            elif f <= old_e:
                new_e = max(old_s, f)
                if self.editable and self._active_label:
                    new_s = old_s
                else:
                    new_s, new_e = self._snap_to_current(old_s, new_e)
                self._preview_interval = self._interval_clamped_free(
                    new_s, new_e, self._active_label
                )
            else:
                # While resizing a labeled segment, allow dragging through
                # neighboring labels to move the boundary continuously.
                if self.editable and self._active_label:
                    new_e = f
                else:
                    new_e = self._snap_unlabeled(f) or f
                cand_e = max(old_s, new_e)
                if self.editable and self._active_label:
                    new_s, new_e = old_s, cand_e
                else:
                    new_s, new_e = self._snap_to_current(old_s, cand_e)
                self._preview_interval = self._interval_clamped_free(
                    new_s, new_e, self._active_label
                )
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        g = self.get_gutter()
        if self._row_drag_enabled() and e.x() < g:
            self._row_dragging = True
            self._row_drag_active = False
            self._row_drag_start = e.pos()
            self.setCursor(Qt.OpenHandCursor)
            return
        if e.x() < g:
            return
        if (e.modifiers() & Qt.ControlModifier) and callable(self.split_handler):
            frame = self.x_to_frame(e.x())
            if not self._frame_in_edit_mask(frame):
                return
            try:
                handled = bool(self.split_handler(frame, self))
            except Exception:
                handled = False
            if handled:
                return
        marker = self._scribble_at(e.x(), e.y())
        if marker is not None and self._is_scribble_marker(marker):
            self.scribbleActivated.emit(dict(marker))
            return
        if self._scribble_mode:
            proposal_hit = self._proposal_boundary_hit(e.x(), e.y())
            if proposal_hit is not None:
                self._dragging = True
                self._mode = "scribble_proposal_boundary"
                self._proposal_drag_boundary_frame = int(proposal_hit)
                self.setCursor(Qt.SizeHorCursor)
                self.update()
                return
            if self.editable:
                hit = self._hit_edge(e.x())
                if hit:
                    interval, label, where = hit
                    if not self._interval_in_edit_mask(interval[0], interval[1]):
                        return
                    self._dragging = True
                    self._active_interval = interval
                    self._active_label = label
                    self._mode = "resize_left" if where == "left" else "resize_right"
                    self._preview_interval = interval
                    st = self._store_for_label(label)
                    if st is not None:
                        try:
                            st.begin_txn()
                        except Exception:
                            pass
                    self.setCursor(Qt.SizeHorCursor)
                    self.update()
                    return
            frame_f = self.x_to_frame_float(e.x())
            if not self._frame_in_edit_mask(int(round(frame_f))):
                return
            self._dragging = False
            self._mode = "scribble_pending"
            self._scribble_press_pos = e.pos()
            self._scribble_press_frame = float(frame_f)
            self._scribble_press_y_norm = float(self._scribble_y_norm_from_pos(e.y()))
            try:
                self._scribble_press_segment = self._segment_at(int(round(frame_f)))
            except Exception:
                self._scribble_press_segment = None
            self.setCursor(Qt.CrossCursor)
            self.update()
            return
        if not self.editable:
            f = self.x_to_frame(e.x())
            lb = self._label_at(f)
            if lb:
                self.labelClicked.emit(lb, f)
            return
        hit = self._hit_edge(e.x())
        if hit:
            interval, label, where = hit
            if not self._interval_in_edit_mask(interval[0], interval[1]):
                return
            self._dragging = True
            self._active_interval = interval
            self._active_label = label
            self._mode = "resize_left" if where == "left" else "resize_right"
            self._preview_interval = interval
            st = self._store_for_label(label)
            if st is not None:
                try:
                    st.begin_txn()
                except Exception:
                    pass
            self.setCursor(Qt.SizeHorCursor)
            self.update()
            return

        f = self.x_to_frame(e.x())
        if not self._frame_in_edit_mask(f):
            return
        if self._label_at(f) is not None:
            return
        if self._snap_segments:
            s = self._snap_to_segment_start(f)
        else:
            s = self._snap_edge_after_label_left(f)
            if s < 0:
                s = self._snap_unlabeled(f)
        if s is None:
            return
        if not self._frame_in_edit_mask(s):
            return
        self._dragging = True
        self._mode = "create"
        self._active_interval = None
        self._active_label = None
        self._create_anchor = s
        self._preview_interval = (s, s)
        self.setCursor(Qt.CrossCursor)
        self.update()

    def mouseDoubleClickEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        if self._scribble_mode:
            return
        g = self.get_gutter()
        if e.x() < g:
            self._row_dragging = False
            self._row_drag_active = False
            self._row_drag_start = None
            self.setCursor(Qt.ArrowCursor)
            fc = max(1, self.get_fc())
            self._selected_interval = (0, fc - 1)
            self._selected_label = None
            self._selection_scope = "all"
            self.segmentSelected.emit(0, fc - 1, None)
            self.update()
            return
        f = self.x_to_frame(e.x())
        s, e_, lb = self._segment_at(f)
        self._selected_interval = (s, e_)
        self._selected_label = lb
        self._selection_scope = "segment"
        self.segmentSelected.emit(s, e_, lb)
        if lb:
            self.labelClicked.emit(lb, f)
        self.update()

    def leaveEvent(self, e):
        self._hover_frame = None
        self.hoverFrame.emit(-1)
        self._sync_hover_scribble_marker(None)
        self.setToolTip("")
        self.update()
        return super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._row_dragging:
            if self._row_drag_active:
                self._finish_row_drag(e.globalPos())
            self._row_dragging = False
            self._row_drag_active = False
            self._row_drag_start = None
            self.setCursor(Qt.ArrowCursor)
            return
        if self._dragging and self._mode == "scribble":
            self.setCursor(Qt.ArrowCursor)
            payload = self._normalize_scribble_item(self._draft_scribble)
            if payload is not None:
                s, e_ = self._scribble_item_bounds(payload)
                if self._interval_in_edit_mask(s, e_):
                    self._scribble_items.append(payload)
                    self._invalidate_scribble_cache()
                    self.scribbleEditedDetailed.emit(dict(payload))
            self._dragging = False
            self._mode = None
            self._draft_scribble = None
            self._scribble_last_frame = None
            self._scribble_last_y_norm = None
            self._scribble_press_pos = None
            self._scribble_press_frame = None
            self._scribble_press_y_norm = None
            self._scribble_press_segment = None
            self.update()
            return
        if self._mode == "scribble_pending":
            self.setCursor(Qt.ArrowCursor)
            seg = self._scribble_press_segment
            if seg is not None:
                s, e_, lb = seg
                self._selected_interval = (int(s), int(e_))
                self._selected_label = lb
                self._selection_scope = "segment"
                self.segmentSelected.emit(int(s), int(e_), lb)
                if lb:
                    self.labelClicked.emit(lb, self.x_to_frame(e.x()))
            self._dragging = False
            self._mode = None
            self._scribble_press_pos = None
            self._scribble_press_frame = None
            self._scribble_press_y_norm = None
            self._scribble_press_segment = None
            self.update()
            return
        if self._dragging and self._mode == "scribble_proposal_boundary":
            self.setCursor(Qt.ArrowCursor)
            frame_i = (
                self._proposal_drag_boundary_frame
                if self._proposal_drag_boundary_frame is not None
                else self.x_to_frame(e.x())
            )
            frame_i = self._clamp_proposal_boundary_frame(int(frame_i))
            if isinstance(self._scribble_proposal, dict):
                self._scribble_proposal["boundary_frame"] = int(frame_i)
            self._emit_scribble_proposal_adjusted(frame_i=int(frame_i), finalized=True)
            self._dragging = False
            self._mode = None
            self._proposal_drag_boundary_frame = None
            self.update()
            return
        if not self._dragging:
            return
        self.setCursor(Qt.ArrowCursor)
        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            if not self._interval_in_edit_mask(s, e_):
                self._preview_interval = None
        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            if self._active_label is None:
                self._selected_interval = (s, e_)
                self._selected_label = None
                self.segmentSelected.emit(s, e_, None)
            else:
                st = self._store_for_label(self._active_label)
                if st is not None:
                    old_s = old_e = None
                    left_fill_label = None
                    right_fill_label = None
                    no_gap_fill = False
                    for f in range(s, e_ + 1):
                        cur = self._label_at(f)
                        if cur is None:
                            st.add(self._active_label, f)
                        elif cur != self._active_label:
                            st.remove_at(f)
                            st.add(self._active_label, f)
                    if self._active_interval is not None:
                        old_s, old_e = self._active_interval
                        meta = getattr(self, "_group_meta", None)
                        if isinstance(meta, dict) and bool(meta.get("psr_no_gap_fill")):
                            no_gap_fill = True
                            try:
                                old_s_i = int(old_s)
                                old_e_i = int(old_e)
                            except Exception:
                                old_s_i = old_e_i = None
                            if old_s_i is not None and old_s_i > 0:
                                left_fill_label = self._label_at(old_s_i - 1)
                            if old_e_i is not None:
                                fc = max(1, self.get_fc())
                                if old_e_i < fc - 1:
                                    right_fill_label = self._label_at(old_e_i + 1)
                            default_fill = meta.get("psr_default_label")
                            if left_fill_label is None and default_fill:
                                left_fill_label = default_fill
                            if right_fill_label is None and default_fill:
                                right_fill_label = default_fill

                        # Left trimmed span (start moved right): optionally fill to
                        # avoid gaps in PSR no-gap mode.
                        for f in range(old_s, min(s, old_e + 1)):
                            if self._label_at(f) != self._active_label:
                                continue
                            fill = (
                                left_fill_label
                                if no_gap_fill and self._mode == "resize_left"
                                else None
                            )
                            if fill and fill != self._active_label:
                                st.remove_at(f)
                                st.add(fill, f)
                            else:
                                st.remove_at(f)

                        # Right trimmed span (end moved left): optionally fill to
                        # shift the adjacent boundary in PSR no-gap mode.
                        for f in range(max(e_ + 1, old_s), old_e + 1):
                            if self._label_at(f) != self._active_label:
                                continue
                            fill = (
                                right_fill_label
                                if no_gap_fill and self._mode == "resize_right"
                                else None
                            )
                            if fill and fill != self._active_label:
                                st.remove_at(f)
                                st.add(fill, f)
                            else:
                                st.remove_at(f)
                    try:
                        st.end_txn()
                    except Exception:
                        pass
                self._selected_interval = (s, e_)
                self._selected_label = self._active_label
                self.segmentSelected.emit(s, e_, self._active_label)
                self.changed.emit()
        else:
            if self._active_label:
                st = self._store_for_label(self._active_label)
                if st is not None:
                    try:
                        st.end_txn()
                    except Exception:
                        pass

        self._dragging = False
        self._mode = None
        self._active_interval = None
        self._preview_interval = None
        self._create_anchor = None
        self._active_label = None
        self._scribble_press_pos = None
        self._scribble_press_frame = None
        self._scribble_press_y_norm = None
        self._scribble_press_segment = None
        self.update()

    def contextMenuEvent(self, e):
        try:
            pos = e.pos()
            marker = self._scribble_at(pos.x(), pos.y())
        except Exception:
            marker = None
        if marker is None or not self._is_scribble_marker(marker):
            e.accept()
            return
        menu = QMenu(self)
        meta = self._scribble_item_meta(marker)
        bf = meta.get("boundary_frame")
        label = "Delete marker" if bf is None else f"Delete marker @ F{int(bf)}"
        act_del = menu.addAction(label)
        chosen = menu.exec_(e.globalPos())
        if chosen is act_del:
            removed = self._pop_scribble_at(pos.x(), pos.y())
            if removed is not None:
                self._invalidate_scribble_cache()
                self.update()
                self.scribbleRemoved.emit(dict(removed))
        e.accept()


class TimelineArea(QWidget):
    hoverFrame = pyqtSignal(int)
    changed = pyqtSignal()
    viewPanned = pyqtSignal()  # emitted when user moves view/zoom
    labelClicked = pyqtSignal(str, int)
    segmentSelected = pyqtSignal(int, int, object)
    scribbleEditedDetailed = pyqtSignal(object)
    scribbleActivated = pyqtSignal(object)
    scribbleProposalAdjusted = pyqtSignal(object)
    scribbleRemoved = pyqtSignal(object)
    gapPrevRequested = pyqtSignal()
    gapNextRequested = pyqtSignal()
    frameSeekRequested = pyqtSignal(int)

    def __init__(
        self,
        labels: List[LabelDef],
        store: AnnotationStore,
        get_frame_count: Callable[[], int],
        get_fps: Callable[[], int],
        on_extra_boundary: Optional[Callable[[int], None]] = None,
        is_extra_mode: Optional[Callable[[], bool]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.labels = labels
        self.store = store
        self.get_fc = get_frame_count
        self.get_fps = get_fps
        self._extra_boundary_cb = on_extra_boundary
        self._is_extra_mode = is_extra_mode or (lambda: False)

        self.view_start = 0
        self.view_span = DEFAULT_VIEW_SPAN
        self._gutter_px = 80
        self._row_sources = None  # List[Tuple[LabelDef, AnnotationStore, str]]
        self.highlight_labels = set()  # label names to highlight
        self.current_frame: Optional[int] = None
        self._current_hits = set()
        self._block_view_signal = False
        self.layout_mode = "combined"  # "combined" | "per_label"
        self._extra_cuts: List[int] = []
        self._segment_cuts: List[int] = []
        self._snap_segments: List[Tuple[int, int]] = []
        self._current_frame_snap_radius = CURRENT_FRAME_SNAP_RADIUS_FRAMES
        self._frame_snap_radius = SNAP_RADIUS_FRAMES
        self._edge_snap_frames = EDGE_SNAP_FRAMES
        self._segment_snap_radius = SNAP_RADIUS_FRAMES
        self._center_single_row = False
        self._combined_show_text = True
        self._combined_editable = False
        self._combined_groups = None
        self._tail_combined_groups = None
        self._combined_rows = []
        self._active_combined_row = None
        self._row_delete_handler = None
        self._row_split_handler = None
        self._row_segment_cuts_provider = None
        self._row_edit_mask_provider = None
        self._combined_delete_handler = None
        self._combined_split_handler = None
        self._combined_reorder_handler = None
        self._scribble_mode = False
        self._scribble_items = []
        self._scribble_proposal = None

        root = QVBoxLayout(self)

        # scrollable rows
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vbox = QVBoxLayout(self.container)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.vbox.setSpacing(4)
        self._default_vbox_spacing = 4
        self.scroll.setWidget(self.container)
        # set a modest minimum height so single-row mode stays visible
        self.scroll.setMinimumHeight(64)
        root.addWidget(self.scroll, 1)

        # view controls: start + span
        row = QHBoxLayout()
        self.chk_layout = QCheckBox("Single timeline", self)
        self.chk_layout.setChecked(True)
        self.chk_layout.setToolTip(
            "Show all labels on one track (toggle off for per-label editing)"
        )
        self.chk_layout.toggled.connect(self._on_layout_mode_toggled)
        row.addWidget(self.chk_layout, 0)
        self.chk_action_lock = QCheckBox("Lock to segment", self)
        self.chk_action_lock.setToolTip("Snap state boundaries to segments")
        self.chk_action_lock.setVisible(False)
        row.addWidget(self.chk_action_lock, 0)
        row.addSpacing(8)
        row.addWidget(QLabel("View start:"))
        self.slider_view = QSlider(Qt.Horizontal, self)
        self.slider_view.valueChanged.connect(self._on_view_start_changed)
        row.addWidget(self.slider_view, 2)

        row.addSpacing(8)
        row.addWidget(QLabel("View span:"))
        self.slider_span = QSlider(Qt.Horizontal, self)
        self.slider_span.valueChanged.connect(self._on_view_span_changed)
        row.addWidget(self.slider_span, 3)

        row.addSpacing(8)
        row.addStretch(1)
        self.lbl_gap = QLabel("Gaps: n/a", self)
        self.lbl_gap.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_gap.setStyleSheet("color: #667085;")
        row.addWidget(self.lbl_gap, 0)
        self.btn_gap_prev = QToolButton(self)
        self.btn_gap_prev.setText("<")
        self.btn_gap_prev.setToolTip("Previous gap")
        self.btn_gap_prev.clicked.connect(self.gapPrevRequested.emit)
        row.addWidget(self.btn_gap_prev, 0)
        self.btn_gap_next = QToolButton(self)
        self.btn_gap_next.setText(">")
        self.btn_gap_next.setToolTip("Next gap")
        self.btn_gap_next.clicked.connect(self.gapNextRequested.emit)
        row.addWidget(self.btn_gap_next, 0)

        root.addLayout(row)

        self.rows: List[QWidget] = []
        self.rebuild_rows()

    def get_view_start(self):
        return self.view_start

    def get_view_span(self):
        return self.view_span

    def _compute_gutter_px(self):
        # Compute max label text width across all labels, then add padding
        fm = QFontMetrics(QFont("Arial", 9))
        max_w = 0
        for lb in self.labels:
            try:
                w = fm.horizontalAdvance(lb.name)
            except AttributeError:
                w = fm.width(lb.name)
            if w > max_w:
                max_w = w
        self._gutter_px = max(80, max_w + 16)  # min 80px, or longest text + padding

    def _compute_gutter_px_from_sources(self):
        from PyQt5.QtGui import QFont, QFontMetrics

        fm = QFontMetrics(QFont("Arial", 9))
        max_w = 0
        if getattr(self, "layout_mode", "") == "combined":
            if self._combined_groups:
                titles = []
                for g in self._combined_groups:
                    if isinstance(g, (list, tuple)) and len(g) >= 1:
                        title = g[0] if g[0] else "Timeline"
                    else:
                        title = "Timeline"
                    titles.append(str(title))
            else:
                titles = ["Timeline"]
        else:
            if self._row_sources:
                titles = [
                    f"{prefix}{lb.name}" for (lb, _store, prefix) in self._row_sources
                ]
            else:
                titles = [lb.name for lb in self.labels]
            if self._tail_combined_groups:
                for g in self._tail_combined_groups:
                    if isinstance(g, (list, tuple)) and len(g) >= 1:
                        title = g[0] if g[0] else "Timeline"
                    else:
                        title = "Timeline"
                    titles.append(str(title))
        for t in titles:
            try:
                w = fm.horizontalAdvance(t)
            except AttributeError:
                w = fm.width(t)
            max_w = max(max_w, w)
        self._gutter_px = max(80, max_w + 16)

    def get_gutter(self) -> int:
        return self._gutter_px

    def set_row_sources(self, row_sources):
        """row_sources: list of (label_def, store, title_prefix)"""
        self._row_sources = row_sources
        self.rebuild_rows()

    def set_center_single_row(self, on: bool):
        self._center_single_row = bool(on)
        self.rebuild_rows()

    def set_combined_label_text(self, show: bool):
        self._combined_show_text = bool(show)
        self.rebuild_rows()

    def set_combined_groups(self, groups):
        """Define grouped combined rows: list of (title, row_sources)."""
        self._combined_groups = groups or None
        self.rebuild_rows()

    def set_tail_combined_groups(self, groups):
        """Define combined rows appended after per-label rows."""
        self._tail_combined_groups = groups or None
        if self.layout_mode != "combined":
            self.rebuild_rows()

    def coverage_gaps(self, start: int, end: int) -> List[Tuple[int, int]]:
        """Return unlabeled spans in [start, end] using the exact same label
        sources the rows paint from — guaranteeing the indicator agrees with
        what the user sees on the timeline."""
        if end < start:
            return []
        rows: list = []
        rows.extend(getattr(self, "rows", []) or [])
        rows.extend(getattr(self, "_combined_rows", []) or [])
        if not rows:
            return [(int(start), int(end))]
        gaps: List[Tuple[int, int]] = []
        run_start: Optional[int] = None
        for f in range(int(start), int(end) + 1):
            covered = False
            for row in rows:
                try:
                    if row._label_at(f) is not None:
                        covered = True
                        break
                except Exception:
                    continue
            if not covered:
                if run_start is None:
                    run_start = int(f)
            else:
                if run_start is not None:
                    gaps.append((int(run_start), int(f) - 1))
                    run_start = None
        if run_start is not None:
            gaps.append((int(run_start), int(end)))
        return gaps

    def set_gap_summary(self, text: str, tooltip: str = "", has_gaps: bool = False):
        if not getattr(self, "lbl_gap", None):
            return
        self.lbl_gap.setText(text)
        self.lbl_gap.setToolTip(tooltip or "")
        if has_gaps:
            self.lbl_gap.setStyleSheet("color: #b42318; font-weight: 600;")
        else:
            self.lbl_gap.setStyleSheet("color: #667085;")

    def set_combined_editable(self, on: bool):
        self._combined_editable = bool(on)
        if self.layout_mode == "combined" and self._combined_rows:
            for row in self._combined_rows:
                try:
                    meta = getattr(row, "_group_meta", None)
                    if isinstance(meta, dict) and "editable" in meta:
                        row.set_editable(bool(meta["editable"]))
                    else:
                        row.set_editable(self._combined_editable)
                except Exception:
                    pass
        self.rebuild_rows()

    def set_row_delete_handler(self, handler):
        self._row_delete_handler = handler
        if self.layout_mode != "combined":
            for row in self.rows:
                try:
                    row.set_delete_handler(handler)
                except Exception:
                    pass

    def set_row_split_handler(self, handler):
        self._row_split_handler = handler
        if self.layout_mode != "combined":
            for row in self.rows:
                try:
                    row.set_split_handler(handler)
                except Exception:
                    pass

    def _apply_snap_tuning_to_row(self, row) -> None:
        if row is None:
            return
        try:
            row.set_current_snap_radius(self._current_frame_snap_radius)
        except Exception:
            pass
        try:
            row.set_frame_snap_radius(self._frame_snap_radius)
        except Exception:
            pass
        try:
            row.set_edge_snap_frames(self._edge_snap_frames)
        except Exception:
            pass
        try:
            row.set_segment_snap_radius(self._segment_snap_radius)
        except Exception:
            pass

    def set_snap_tuning(
        self,
        current_frame_radius: Optional[int] = None,
        frame_snap_radius: Optional[int] = None,
        edge_snap_frames: Optional[int] = None,
        segment_snap_radius: Optional[int] = None,
        refresh: bool = True,
    ) -> None:
        changed = False
        if current_frame_radius is not None:
            try:
                val = max(0, int(current_frame_radius))
            except Exception:
                val = CURRENT_FRAME_SNAP_RADIUS_FRAMES
            if val != self._current_frame_snap_radius:
                self._current_frame_snap_radius = val
                changed = True
        if frame_snap_radius is not None:
            try:
                val = max(0, int(frame_snap_radius))
            except Exception:
                val = SNAP_RADIUS_FRAMES
            if val != self._frame_snap_radius:
                self._frame_snap_radius = val
                changed = True
        if edge_snap_frames is not None:
            try:
                val = max(0, int(edge_snap_frames))
            except Exception:
                val = EDGE_SNAP_FRAMES
            if val != self._edge_snap_frames:
                self._edge_snap_frames = val
                changed = True
        if segment_snap_radius is not None:
            try:
                val = max(0, int(segment_snap_radius))
            except Exception:
                val = SNAP_RADIUS_FRAMES
            if val != self._segment_snap_radius:
                self._segment_snap_radius = val
                changed = True
        if not changed and not refresh:
            return
        for row in self.rows:
            self._apply_snap_tuning_to_row(row)
        if refresh:
            self.refresh_all_rows()

    def set_row_segment_cuts_provider(self, provider):
        self._row_segment_cuts_provider = provider if callable(provider) else None
        self.apply_row_segment_cuts()

    def apply_row_segment_cuts(self):
        provider = self._row_segment_cuts_provider
        for row in self.rows:
            if not hasattr(row, "set_segment_cuts"):
                continue
            cuts = []
            if callable(provider):
                try:
                    cuts = list(provider(row) or [])
                except Exception:
                    cuts = []
            try:
                row.set_segment_cuts(cuts)
            except Exception:
                pass

    def set_row_edit_mask_provider(self, provider):
        self._row_edit_mask_provider = provider if callable(provider) else None
        self.apply_row_edit_masks()

    def apply_row_edit_masks(self):
        provider = self._row_edit_mask_provider
        for row in self.rows:
            if not hasattr(row, "set_edit_mask_spans"):
                continue
            spans = None
            if callable(provider):
                try:
                    spans = provider(row)
                except Exception:
                    spans = None
            try:
                row.set_edit_mask_spans(spans)
            except Exception:
                pass

    def set_combined_delete_handler(self, handler):
        self._combined_delete_handler = handler
        if self.layout_mode == "combined":
            for row in self._combined_rows:
                try:
                    row.set_delete_handler(handler)
                except Exception:
                    pass

    def set_combined_split_handler(self, handler):
        self._combined_split_handler = handler
        if self.layout_mode == "combined":
            for row in self._combined_rows:
                try:
                    row.set_split_handler(handler)
                except Exception:
                    pass

    def set_combined_reorder_handler(self, handler):
        self._combined_reorder_handler = handler

    def _combined_row_at_global(self, global_pos):
        for row in self._combined_rows:
            try:
                top_left = row.mapToGlobal(row.rect().topLeft())
                rect = QRect(top_left, row.size())
                if rect.contains(global_pos):
                    return row
            except Exception:
                continue
        return None

    def _handle_combined_row_drop(self, src_row, global_pos):
        if not callable(self._combined_reorder_handler):
            return
        target = self._combined_row_at_global(global_pos)
        if target is None or target is src_row:
            return
        try:
            self._combined_reorder_handler(
                getattr(src_row, "title", None), getattr(target, "title", None)
            )
        except Exception:
            pass

    def apply_combined_label(self, name: str) -> bool:
        if self.layout_mode != "combined":
            return False
        row = self._active_combined_row or (
            self._combined_rows[0] if self._combined_rows else None
        )
        if row is None or not getattr(row, "editable", False):
            return False
        try:
            return bool(row.apply_label_to_selection(name))
        except Exception:
            return False

    def set_layout_mode(self, mode: str):
        target = "combined" if mode == "combined" else "per_label"
        self.layout_mode = target
        try:
            self.chk_layout.blockSignals(True)
            self.chk_layout.setChecked(target == "combined")
            self.chk_layout.blockSignals(False)
        except Exception:
            pass
        if target == "combined":
            self.fit_full_view()
        self.rebuild_rows()

    def _on_layout_mode_toggled(self, on: bool):
        self.layout_mode = "combined" if on else "per_label"
        self.rebuild_rows()

    def set_highlight_labels(self, names):
        self.highlight_labels = set(names or [])
        for row in self.rows:
            if hasattr(row, "set_highlight_labels"):
                try:
                    row.set_highlight_labels(self.highlight_labels)
                except Exception:
                    pass
                continue
            if hasattr(row, "label") and hasattr(row, "set_highlighted"):
                try:
                    row.set_highlighted(row.label.name in self.highlight_labels)
                except Exception:
                    pass
        self.refresh_all_rows()

    def flash_boundary_marker(self, frame: int):
        for row in self.rows:
            if hasattr(row, "set_boundary_flash"):
                try:
                    row.set_boundary_flash(frame)
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_extra_cuts(self, cuts: List[int]):
        self._extra_cuts = list(cuts or [])
        for row in self.rows:
            if hasattr(row, "_extra_cuts"):
                try:
                    row._extra_cuts = list(self._extra_cuts)
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_segment_cuts(self, cuts: List[int]):
        self._segment_cuts = list(cuts or [])
        for row in self.rows:
            if hasattr(row, "_segment_cuts"):
                try:
                    meta = getattr(row, "_group_meta", None)
                    if (
                        isinstance(meta, dict)
                        and meta.get("show_segment_cuts") is False
                    ):
                        row._segment_cuts = []
                    elif (
                        isinstance(meta, dict) and meta.get("segment_cuts") is not None
                    ):
                        row._segment_cuts = list(meta.get("segment_cuts") or [])
                    else:
                        row._segment_cuts = list(self._segment_cuts)
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_snap_segments(self, segments: List[Tuple[int, int]]):
        self._snap_segments = list(segments or [])
        for row in self.rows:
            if hasattr(row, "set_snap_segments"):
                try:
                    row.set_snap_segments(self._snap_segments)
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_scribble_mode(self, enabled: bool) -> None:
        self._scribble_mode = bool(enabled)
        try:
            self.scroll.setMinimumHeight(124 if self._scribble_mode else 64)
        except Exception:
            pass
        for row in self.rows:
            if hasattr(row, "set_scribble_mode"):
                try:
                    row.set_scribble_mode(self._scribble_mode)
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_scribble_items(self, items) -> None:
        self._scribble_items = list(items or [])
        for row in self.rows:
            if hasattr(row, "set_scribble_items"):
                try:
                    row.set_scribble_items(self._scribble_items)
                except Exception:
                    pass
        self.refresh_all_rows()

    def clear_scribble_items(self) -> None:
        self._scribble_items = []
        for row in self.rows:
            if hasattr(row, "clear_scribble_items"):
                try:
                    row.clear_scribble_items()
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_scribble_proposal(self, proposal) -> None:
        self._scribble_proposal = None if proposal is None else dict(proposal)
        for row in self.rows:
            if hasattr(row, "set_scribble_proposal"):
                try:
                    row.set_scribble_proposal(self._scribble_proposal)
                except Exception:
                    pass
        self.refresh_all_rows()

    def clear_scribble_proposal(self) -> None:
        self._scribble_proposal = None
        for row in self.rows:
            if hasattr(row, "clear_scribble_proposal"):
                try:
                    row.clear_scribble_proposal()
                except Exception:
                    pass
        self.refresh_all_rows()

    def _scribble_item_key(self, item) -> str:
        if isinstance(item, dict):
            meta = item.get("meta") or {}
            stroke_id = str(meta.get("stroke_id") or "").strip()
            if stroke_id:
                return stroke_id
            try:
                return (
                    f"{int(item.get('start_frame', 0))}:"
                    f"{int(item.get('end_frame', 0))}:"
                    f"{str(item.get('kind') or 'uncertain')}"
                )
            except Exception:
                return str(id(item))
        meta = dict(getattr(item, "meta", {}) or {})
        stroke_id = str(meta.get("stroke_id") or "").strip()
        if stroke_id:
            return stroke_id
        try:
            kind = getattr(getattr(item, "kind", None), "value", getattr(item, "kind", "uncertain"))
            return (
                f"{int(getattr(item, 'start_frame', 0))}:"
                f"{int(getattr(item, 'end_frame', 0))}:"
                f"{str(kind or 'uncertain')}"
            )
        except Exception:
            pass
        return str(id(item))

    def _on_row_scribble_edited_detailed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        item = dict(payload)
        self._scribble_items.append(item)
        self.scribbleEditedDetailed.emit(item)

    def _on_row_scribble_activated(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self.scribbleActivated.emit(dict(payload))

    def _on_row_scribble_proposal_adjusted(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._scribble_proposal = dict(payload)
        self.scribbleProposalAdjusted.emit(dict(payload))

    def _on_row_scribble_removed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        item = dict(payload)
        target_key = self._scribble_item_key(item)
        removed = False
        kept = []
        for row in self._scribble_items:
            if not removed and self._scribble_item_key(row) == target_key:
                removed = True
                continue
            kept.append(row)
        self._scribble_items = kept
        self.scribbleRemoved.emit(item)

    def set_current_hits(self, names):
        hits = set(names or [])
        self._current_hits = hits
        for row in self.rows:
            if hasattr(row, "set_current_hits"):
                try:
                    row.set_current_hits(hits)
                except Exception:
                    pass
                continue
            if hasattr(row, "label") and hasattr(row, "set_current_hit"):
                try:
                    row.set_current_hit(row.label.name in hits)
                except Exception:
                    pass

    def _ensure_visible(self, frame: int):
        """Ensure frame is within view; recenters if outside."""
        fc = max(1, self.get_fc())
        span = max(1, self.view_span)
        start = self.view_start
        end = start + span - 1
        if frame < start or frame > end:
            new_start = max(0, min(fc - span, frame - span // 2))
            self._block_view_signal = True
            self.slider_view.blockSignals(True)
            self.slider_view.setValue(new_start)
            self.slider_view.blockSignals(False)
            self._block_view_signal = False
            self.view_start = new_start
            for r in self.rows:
                r.update()

    def center_on_frame(self, frame: int):
        fc = max(1, self.get_fc())
        span = max(1, self.view_span)
        target = max(0, min(fc - span, int(frame) - span // 2))
        self._block_view_signal = True
        self.slider_view.blockSignals(True)
        self.slider_view.setValue(target)
        self.slider_view.blockSignals(False)
        self._block_view_signal = False
        self.view_start = target
        for r in self.rows:
            r.update()

    def set_current_frame(self, frame: int, follow: bool = False):
        self.current_frame = max(0, int(frame))
        if follow:
            self._ensure_visible(self.current_frame)
        for row in self.rows:
            if hasattr(row, "set_current_frame"):
                try:
                    row.set_current_frame(self.current_frame)
                except Exception:
                    pass

    def wheelEvent(self, e):
        mods = e.modifiers()
        if mods & (Qt.ControlModifier | Qt.ShiftModifier):
            super().wheelEvent(e)
            return
        delta = e.angleDelta()
        notches = float(delta.y() or delta.x()) / 120.0
        if not notches:
            e.ignore()
            return
        try:
            fc = max(1, int(self.get_fc()))
        except Exception:
            fc = max(1, int(self.view_span or 1))
        span = max(1, int(self.view_span or fc))
        step = max(1, int(round(span / 10.0)))
        offset = int(round(notches * step))
        if offset == 0:
            offset = 1 if notches > 0 else -1
        max_start = max(0, fc - span)
        new_start = int(self.view_start) - offset
        new_start = max(0, min(max_start, new_start))
        if new_start != int(self.view_start):
            self.view_start = new_start
            try:
                self.slider_view.blockSignals(True)
                self.slider_view.setValue(new_start)
                self.slider_view.blockSignals(False)
            except Exception:
                pass
            for r in self.rows:
                try:
                    r.update()
                except Exception:
                    pass
            self.viewPanned.emit()
        e.accept()

    def _on_combined_segment_selected(self, row):
        self._active_combined_row = row

    def _emit_segment_selected(self, start: int, end: int, label, row):
        self._on_combined_segment_selected(row)
        self.segmentSelected.emit(start, end, label)

    def rebuild_rows(self):
        # clear old layout items (rows, subtitles, spacers)
        while self.vbox.count():
            item = self.vbox.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                try:
                    w.hide()
                except Exception:
                    pass
                try:
                    w.deleteLater()
                except Exception:
                    pass
            elif item is not None and item.layout() is not None:
                try:
                    item.layout().setParent(None)
                except Exception:
                    pass
        self.rows.clear()
        self._combined_rows = []
        self._active_combined_row = None
        self._compute_gutter_px_from_sources()

        # row sources: default to global store for each label when not provided
        sources = self._row_sources or [(lb, self.store, "") for lb in self.labels]

        # build rows
        def _add_combined_row(group):
            if isinstance(group, (list, tuple)) and len(group) >= 2:
                title = group[0]
                group_sources = group[1]
                meta = group[2] if len(group) >= 3 else None
            else:
                title = "Timeline"
                group_sources = sources
                meta = None
            row_height = None
            labels_for_row = self.labels
            show_label_text = getattr(self, "_combined_show_text", True)
            editable = getattr(self, "_combined_editable", False)
            split_on_extra_cuts = False
            segment_cuts = getattr(self, "_segment_cuts", [])
            if isinstance(meta, dict):
                row_height = meta.get("row_height")
                if meta.get("labels"):
                    labels_for_row = meta.get("labels") or self.labels
                if "show_label_text" in meta:
                    show_label_text = bool(meta["show_label_text"])
                if "editable" in meta:
                    editable = bool(meta["editable"])
                if "split_on_extra_cuts" in meta:
                    split_on_extra_cuts = bool(meta["split_on_extra_cuts"])
                if "segment_cuts" in meta:
                    segment_cuts = list(meta["segment_cuts"] or [])
                if "show_segment_cuts" in meta and not meta["show_segment_cuts"]:
                    segment_cuts = []
            row = CombinedTimelineRow(
                labels_for_row,
                group_sources,
                self.get_fc,
                self.get_view_start,
                self.get_view_span,
                self.get_fps,
                self.get_gutter,
                title=title,
                show_label_text=show_label_text,
                extra_cuts=getattr(self, "_extra_cuts", []),
                segment_cuts=segment_cuts,
                editable=editable,
                split_on_extra_cuts=split_on_extra_cuts,
            )
            if row_height is not None:
                try:
                    rh = int(row_height)
                    if rh > 0:
                        row.set_base_row_height(rh)
                except Exception:
                    pass
            self._apply_snap_tuning_to_row(row)
            row._timeline_ref = self
            if meta is not None:
                try:
                    row._group_meta = meta
                except Exception:
                    pass
                try:
                    if isinstance(meta, dict):
                        snap_mode = meta.get("snap_mode")
                        if snap_mode == "soft":
                            row._snap_soft = True
                        if "snap_radius" in meta:
                            row.set_segment_snap_radius(
                                int(meta.get("snap_radius", SNAP_RADIUS_FRAMES))
                            )
                except Exception:
                    pass
            try:
                row.set_delete_handler(self._combined_delete_handler)
            except Exception:
                pass
            try:
                row.set_split_handler(self._combined_split_handler)
            except Exception:
                pass
            row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.hoverFrame.connect(self.hoverFrame.emit)
            row.labelClicked.connect(self.labelClicked.emit)
            row.segmentSelected.connect(
                lambda _s, _e, _lb, r=row: self._emit_segment_selected(_s, _e, _lb, r)
            )
            row.changed.connect(self.changed.emit)
            row.scribbleEditedDetailed.connect(self._on_row_scribble_edited_detailed)
            row.scribbleActivated.connect(self._on_row_scribble_activated)
            row.scribbleProposalAdjusted.connect(
                self._on_row_scribble_proposal_adjusted
            )
            row.scribbleRemoved.connect(self._on_row_scribble_removed)
            row.set_highlight_labels(self.highlight_labels)
            row.set_current_frame(self.current_frame)
            row.set_current_hits(self._current_hits)
            try:
                if isinstance(meta, dict) and meta.get("snap_segments") is not None:
                    row.set_snap_segments(meta.get("snap_segments") or [])
                else:
                    row.set_snap_segments(getattr(self, "_snap_segments", []))
            except Exception:
                pass
            try:
                row.set_scribble_mode(getattr(self, "_scribble_mode", False))
                row.set_scribble_items(getattr(self, "_scribble_items", []))
                row.set_scribble_proposal(getattr(self, "_scribble_proposal", None))
            except Exception:
                pass
            self.vbox.addWidget(row)
            self.rows.append(row)
            self._combined_rows.append(row)

        if self.layout_mode == "combined":
            groups = self._combined_groups or [("Timeline", sources)]
            center_block = bool(self._center_single_row)
            if center_block:
                self.vbox.setSpacing(0)
            else:
                self.vbox.setSpacing(self._default_vbox_spacing)
            if center_block:
                self.vbox.addStretch(1)
            for group in groups:
                _add_combined_row(group)
                if center_block:
                    self.vbox.addStretch(1)
        else:
            self.vbox.setSpacing(self._default_vbox_spacing)
            for lb, st, prefix in sources:
                row = TimelineRow(
                    lb,  # LabelDef
                    st,
                    self.get_fc,
                    self.get_view_start,
                    self.get_view_span,
                    self.get_fps,
                    self.get_gutter,
                    title_prefix=prefix,  # prefix shown on the left
                )
                self._apply_snap_tuning_to_row(row)
                row.hoverFrame.connect(self.hoverFrame.emit)
                row.changed.connect(self.changed.emit)
                row.set_highlighted(lb.name in self.highlight_labels)
                row.set_current_frame(self.current_frame)
                row.set_current_hit(lb.name in self._current_hits)
                try:
                    row.set_snap_segments(getattr(self, "_snap_segments", []))
                except Exception:
                    pass
                try:
                    row.set_delete_handler(self._row_delete_handler)
                except Exception:
                    pass
                try:
                    row.set_split_handler(self._row_split_handler)
                except Exception:
                    pass
                try:
                    provider = self._row_segment_cuts_provider
                    cuts = list(provider(row) or []) if callable(provider) else []
                    row.set_segment_cuts(cuts)
                except Exception:
                    pass
                self.vbox.addWidget(row)
                self.rows.append(row)
            if self._tail_combined_groups:
                for group in self._tail_combined_groups:
                    _add_combined_row(group)

        # avoid auto-scrolling down in combined layout
        if self.layout_mode == "combined":
            if not center_block:
                self.vbox.addSpacing(0)
        else:
            self.vbox.addStretch(1)
        self._init_sliders()
        try:
            sb = self.scroll.verticalScrollBar()
            sb.blockSignals(True)
            sb.setValue(0)
            sb.blockSignals(False)
        except Exception:
            pass
        self.apply_row_segment_cuts()
        self.apply_row_edit_masks()
        self.update()

    def _row_by_name(self, name: str):
        if self.layout_mode == "combined":
            return self.rows[0] if self.rows else None
        for r in self.rows:
            if getattr(r, "label", None) and r.label.name == name:
                return r
        return None

    def flash_label(self, name: str):
        """Scroll to the label row, make it visible, and blink highlight twice."""
        if self.layout_mode == "combined":
            row = self.rows[0] if self.rows else None
            if row is None:
                return
            row_ref = weakref.ref(row)
            try:
                sb = self.scroll.verticalScrollBar()
                sb.setValue(0)
            except Exception:
                pass
            try:
                _safe_qt_call(row_ref, "flash_labels", [name])
            except Exception:
                pass
            return
        row = self._row_by_name(name)
        if row is None:
            return
        row_ref = weakref.ref(row)
        # scroll into view (roughly center it)
        try:
            sb = self.scroll.verticalScrollBar()
            y = row.pos().y()
            h = row.height()
            target = max(0, y + h // 2 - self.scroll.viewport().height() // 2)
            sb.setValue(target)
        except Exception:
            pass

        base_on = row.label.name in self.highlight_labels

        def set_state(on: bool):
            _safe_qt_call(row_ref, "set_highlighted", base_on or on)

        # blink twice
        set_state(True)
        QTimer.singleShot(220, row, lambda: set_state(False))
        QTimer.singleShot(440, row, lambda: set_state(True))
        QTimer.singleShot(660, row, lambda: set_state(base_on))

    def refresh_all_rows(self):
        for r in getattr(self, "rows", []):
            try:
                r.update()
            except Exception:
                pass
        try:
            self.container.update()
        except Exception:
            pass
        self.update()

    def focus_combined_title(self, title: str):
        if self.layout_mode != "combined" or not title:
            return
        for row in self._combined_rows:
            if getattr(row, "title", "") == title:
                self._active_combined_row = row
                try:
                    sb = self.scroll.verticalScrollBar()
                    y = row.pos().y()
                    h = row.height()
                    target = max(0, y + h // 2 - self.scroll.viewport().height() // 2)
                    sb.setValue(target)
                except Exception:
                    pass
                break

    def fit_full_view(self):
        """Fit view to full frame count (helpful for single-timeline scale)."""
        fc = max(1, self.get_fc())
        self.view_start = 0
        self.view_span = fc
        self._init_sliders()
        self.refresh_all_rows()

    # ----- sliders -----
    def _init_sliders(self):
        fc = max(1, self.get_fc())
        # span slider: 0..100 -> MIN_VIEW_SPAN..fc
        self._block_view_signal = True
        self.slider_span.blockSignals(True)
        self.slider_span.setMinimum(0)
        self.slider_span.setMaximum(100)

        # linear mapping: val=0 -> min, val=100 -> full
        def span_to_val(span):
            span = max(MIN_VIEW_SPAN, min(span, fc))
            return int(round(100 * (span - MIN_VIEW_SPAN) / max(1, fc - MIN_VIEW_SPAN)))

        # default span if unset: medium range
        if self.view_span is None:
            self.view_span = min(fc, max(MIN_VIEW_SPAN, fc // 5))
        self.slider_span.setValue(span_to_val(self.view_span))
        self.slider_span.blockSignals(False)

        self._refresh_view_slider()
        self._block_view_signal = False

    def _refresh_view_slider(self):
        fc = max(1, self.get_fc())
        max_start = max(0, fc - self.view_span)
        self.slider_view.blockSignals(True)
        self.slider_view.setMinimum(0)
        self.slider_view.setMaximum(max_start)
        self.view_start = min(self.view_start, max_start)
        self.slider_view.setValue(self.view_start)
        self.slider_view.blockSignals(False)

        for r in self.rows:
            r.update()

    def _on_view_start_changed(self, v: int):
        self.view_start = v
        for r in self.rows:
            r.update()
        if not self._block_view_signal:
            self.viewPanned.emit()

    def _on_view_span_changed(self, val: int):
        # val 0..100 -> span MIN_VIEW_SPAN..full
        fc = max(1, self.get_fc())
        new_span = int(round(MIN_VIEW_SPAN + (fc - MIN_VIEW_SPAN) * val / 100.0))
        new_span = max(MIN_VIEW_SPAN, min(new_span, fc))
        # keep view_start within bounds
        if self.view_start + new_span > fc:
            self.view_start = max(0, fc - new_span)
        self.view_span = new_span
        self._refresh_view_slider()
        if not self._block_view_signal:
            self.viewPanned.emit()
