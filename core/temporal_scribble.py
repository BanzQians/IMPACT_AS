from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class ScribbleKind(str, Enum):
    UNCERTAIN = "uncertain"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class TemporalScribble:
    start_frame: int
    end_frame: int
    kind: ScribbleKind = ScribbleKind.UNCERTAIN
    view_id: str = ""
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class TemporalScribbleSet:
    items: List[TemporalScribble] = field(default_factory=list)

    def add(self, scribble: TemporalScribble) -> None:
        self.items.append(normalize_scribble(scribble))

    def clear(self) -> None:
        self.items.clear()

    def by_kind(self, kind: ScribbleKind) -> List[TemporalScribble]:
        return [item for item in self.items if item.kind == kind]


def normalize_interval(start_frame: int, end_frame: int) -> Tuple[int, int]:
    start_i = int(start_frame)
    end_i = int(end_frame)
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    return start_i, end_i


def resolve_scribble_kind(value: object) -> ScribbleKind:
    if isinstance(value, ScribbleKind):
        return value
    text = str(value or ScribbleKind.UNCERTAIN.value).strip()
    if not text:
        return ScribbleKind.UNCERTAIN
    try:
        return ScribbleKind(text.lower())
    except Exception:
        pass
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = text.strip().lower()
    alias_map = {
        "uncertain": ScribbleKind.UNCERTAIN,
        "left": ScribbleKind.LEFT,
        "right": ScribbleKind.RIGHT,
    }
    return alias_map.get(text, ScribbleKind.UNCERTAIN)


def normalize_scribble(scribble: TemporalScribble) -> TemporalScribble:
    start_i, end_i = normalize_interval(scribble.start_frame, scribble.end_frame)
    return TemporalScribble(
        start_frame=start_i,
        end_frame=end_i,
        kind=resolve_scribble_kind(scribble.kind),
        view_id=str(scribble.view_id or ""),
        meta=dict(scribble.meta or {}),
    )


def merge_intervals(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    normalized = [normalize_interval(s, e) for s, e in spans]
    if not normalized:
        return []
    normalized.sort(key=lambda row: (row[0], row[1]))
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = normalized[0]
    for s, e in normalized[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
            continue
        merged.append((cur_s, cur_e))
        cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def build_scribble_channels(
    window_start: int,
    window_end: int,
    scribbles: Sequence[TemporalScribble],
) -> Dict[str, np.ndarray]:
    start_i, end_i = normalize_interval(window_start, window_end)
    length = max(0, end_i - start_i + 1)
    channels = {
        ScribbleKind.UNCERTAIN.value: np.zeros(length, dtype=np.float32),
        ScribbleKind.LEFT.value: np.zeros(length, dtype=np.float32),
        ScribbleKind.RIGHT.value: np.zeros(length, dtype=np.float32),
    }
    if length <= 0:
        return channels
    for item in scribbles or []:
        s, e = normalize_interval(item.start_frame, item.end_frame)
        if e < start_i or s > end_i:
            continue
        clip_s = max(start_i, s)
        clip_e = min(end_i, e)
        arr = channels.get(str(item.kind.value))
        if arr is None:
            continue
        frame_counts = _normalized_frame_counts(item, start_i, end_i)
        if frame_counts:
            for frame_i, weight in frame_counts.items():
                if frame_i < start_i or frame_i > end_i:
                    continue
                arr[int(frame_i - start_i)] += float(max(0.0, weight))
            continue
        arr[(clip_s - start_i) : (clip_e - start_i + 1)] += 1.0
    for arr in channels.values():
        peak = float(np.max(arr)) if arr.size else 0.0
        if peak > 0.0:
            arr /= peak
    return channels


def _normalized_frame_counts(
    scribble: TemporalScribble,
    window_start: int,
    window_end: int,
) -> Dict[int, float]:
    meta = dict(getattr(scribble, "meta", {}) or {})
    raw_counts = meta.get("frame_counts")
    if not isinstance(raw_counts, dict):
        return {}
    normalized: Dict[int, float] = {}
    for key, value in raw_counts.items():
        try:
            frame_i = int(key)
            count_f = float(value)
        except Exception:
            continue
        if not np.isfinite(count_f) or count_f <= 0.0:
            continue
        if frame_i < int(window_start) or frame_i > int(window_end):
            continue
        normalized[int(frame_i)] = normalized.get(int(frame_i), 0.0) + count_f
    return normalized
