from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

_RESERVED_ESCAPE_LABELS = ("Unknown", "Other", "Background")
_ESCAPE_LABELS = frozenset(label.lower() for label in _RESERVED_ESCAPE_LABELS)


@dataclass
class ConfirmedWindow:
    start_frame: int
    end_frame: int
    boundary_frame: Optional[int] = None
    left_label: str = ""
    right_label: str = ""
    hard: bool = True
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class SoftConstraint:
    kind: str
    start_frame: int
    end_frame: int
    weight: float = 1.0
    payload: Dict[str, object] = field(default_factory=dict)


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = 0.0
    if out < float(lo):
        return float(lo)
    if out > float(hi):
        return float(hi)
    return float(out)


def _norm_window_bounds(item: ConfirmedWindow) -> Tuple[int, int]:
    s = int(item.start_frame)
    e = int(item.end_frame)
    if e < s:
        s, e = e, s
    return int(s), int(e)


def _hard_label_for_frame(item: ConfirmedWindow, frame: int) -> Optional[str]:
    s, e = _norm_window_bounds(item)
    if int(frame) < int(s) or int(frame) > int(e):
        return None
    boundary = item.boundary_frame
    left = str(item.left_label or "").strip()
    right = str(item.right_label or "").strip()
    if boundary is None:
        fill = right or left
        return str(fill or "") or None
    b = int(boundary)
    if int(frame) < b:
        return left or None
    return right or None


def apply_confirmed_windows_to_frame_labels(
    frame_labels: MutableMapping[int, str],
    windows: Sequence[ConfirmedWindow],
) -> Dict[int, str]:
    out = {int(k): str(v) for k, v in dict(frame_labels or {}).items()}
    for item in windows or []:
        s, e = _norm_window_bounds(item)
        boundary = item.boundary_frame
        if boundary is None:
            fill = str(item.right_label or item.left_label or "").strip()
            if not fill:
                continue
            for frame in range(int(s), int(e) + 1):
                out[int(frame)] = fill
            continue
        b = int(boundary)
        if item.left_label:
            for frame in range(int(s), min(int(e), b - 1) + 1):
                out[int(frame)] = str(item.left_label)
        if item.right_label:
            for frame in range(max(int(s), b), int(e) + 1):
                out[int(frame)] = str(item.right_label)
    return out


def count_anchor_violations(
    frame_labels: MutableMapping[int, str],
    windows: Sequence[ConfirmedWindow],
) -> int:
    labels = {int(k): str(v) for k, v in dict(frame_labels or {}).items()}
    violations = 0
    for item in windows or []:
        s, e = _norm_window_bounds(item)
        for frame in range(int(s), int(e) + 1):
            expected = _hard_label_for_frame(item, frame)
            if not expected:
                continue
            actual = str(labels.get(int(frame), "") or "").strip()
            if actual and actual != expected:
                violations += 1
    return int(violations)


def _dedup_labels(labels: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    seen_escape = set()
    for raw in labels or []:
        label = str(raw or "").strip()
        if not label:
            continue
        lower = label.lower()
        if lower in _ESCAPE_LABELS:
            seen_escape.add(lower)
            continue
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    for label in _RESERVED_ESCAPE_LABELS:
        lower = label.lower()
        if lower not in seen_escape:
            seen_escape.add(lower)
        out.append(label)
    return out


def _merge_spans(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    for raw_s, raw_e in spans or []:
        s = int(raw_s)
        e = int(raw_e)
        if e < s:
            s, e = e, s
        if not rows or s > rows[-1][1] + 1:
            rows.append((s, e))
        else:
            prev_s, prev_e = rows[-1]
            rows[-1] = (prev_s, max(prev_e, e))
    return rows


def _changed_spans(
    before: Mapping[int, str],
    after: Mapping[int, str],
    frames: Sequence[int],
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    run_start: Optional[int] = None
    prev_frame: Optional[int] = None
    for frame in sorted(int(f) for f in frames):
        old = str(before.get(int(frame), "") or "")
        new = str(after.get(int(frame), "") or "")
        changed = old != new
        if changed:
            if run_start is None:
                run_start = int(frame)
            elif prev_frame is not None and int(frame) != int(prev_frame) + 1:
                spans.append((int(run_start), int(prev_frame)))
                run_start = int(frame)
        elif run_start is not None and prev_frame is not None:
            spans.append((int(run_start), int(prev_frame)))
            run_start = None
        prev_frame = int(frame)
    if run_start is not None and prev_frame is not None:
        spans.append((int(run_start), int(prev_frame)))
    return spans


def decode_frame_labels_with_constraints(
    frame_labels: Mapping[int, str],
    windows: Sequence[ConfirmedWindow],
    soft_constraints: Sequence[SoftConstraint],
    label_vocabulary: Optional[Sequence[str]] = None,
    *,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    transition_penalty: float = 0.55,
    stay_bonus: float = 0.05,
    current_label_score: float = 1.0,
    alternate_label_score: float = 0.16,
    anchor_boost: float = 0.55,
) -> Tuple[Dict[int, str], Dict[str, object]]:
    current = {
        int(frame): str(label)
        for frame, label in dict(frame_labels or {}).items()
        if str(label or "").strip()
    }
    if not current and not windows:
        return {}, {
            "decoded_frame_count": 0,
            "changed_frame_count": 0,
            "changed_spans": [],
            "hard_violations_before": 0,
            "hard_violations_after": 0,
            "soft_penalty_total": 0.0,
            "switch_count": 0,
        }

    frame_set = set(current.keys())
    for item in windows or []:
        s, e = _norm_window_bounds(item)
        if frame_start is not None:
            s = max(int(frame_start), int(s))
        if frame_end is not None:
            e = min(int(frame_end), int(e))
        for frame in range(int(s), int(e) + 1):
            frame_set.add(int(frame))
    if frame_start is not None or frame_end is not None:
        lo = int(frame_start) if frame_start is not None else min(frame_set or [0])
        hi = int(frame_end) if frame_end is not None else max(frame_set or [0])
        frame_set = {int(f) for f in frame_set if lo <= int(f) <= hi}
    frames = sorted(frame_set)
    if not frames:
        return {}, {
            "decoded_frame_count": 0,
            "changed_frame_count": 0,
            "changed_spans": [],
            "hard_violations_before": 0,
            "hard_violations_after": 0,
            "soft_penalty_total": 0.0,
            "switch_count": 0,
        }

    label_names = _dedup_labels(
        list(label_vocabulary or [])
        + list(current.values())
        + [str(item.left_label or "") for item in windows or []]
        + [str(item.right_label or "") for item in windows or []]
        + [
            str((constraint.payload or {}).get("preferred_label") or "")
            for constraint in soft_constraints or []
        ]
    )
    if not label_names:
        return dict(current), {
            "decoded_frame_count": int(len(frames)),
            "changed_frame_count": 0,
            "changed_spans": [],
            "hard_violations_before": count_anchor_violations(current, windows),
            "hard_violations_after": count_anchor_violations(current, windows),
            "soft_penalty_total": 0.0,
            "switch_count": 0,
        }
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    num_frames = len(frames)
    num_labels = len(label_names)

    emissions: List[List[float]] = [
        [float(alternate_label_score) for _ in range(num_labels)]
        for _ in range(num_frames)
    ]
    switch_penalties = [float(max(0.0, transition_penalty)) for _ in range(num_frames)]

    for frame_pos, frame in enumerate(frames):
        cur_label = str(current.get(int(frame), "") or "").strip()
        if cur_label in label_to_idx:
            emissions[frame_pos][label_to_idx[cur_label]] = float(current_label_score)

    anchor_regions: List[Tuple[int, int]] = []
    for item in windows or []:
        s, e = _norm_window_bounds(item)
        boundary = item.boundary_frame
        if boundary is not None:
            b = int(boundary)
            anchor_regions.append((max(int(s), b - 3), min(int(e), b + 3)))
        left = str(item.left_label or "").strip()
        right = str(item.right_label or "").strip()
        ctx = max(4, int(max(1, e - s + 1) * 0.35))
        for frame_pos, frame in enumerate(frames):
            if int(frame) < int(s) - ctx or int(frame) > int(e) + ctx:
                continue
            expected = _hard_label_for_frame(item, frame)
            if expected and expected in label_to_idx:
                emissions[frame_pos][label_to_idx[expected]] = 4.0
            elif boundary is not None:
                if int(frame) < int(boundary) and left in label_to_idx:
                    dist = max(0, int(s) - int(frame)) if int(frame) < int(s) else 0
                    bonus = anchor_boost * max(0.0, 1.0 - float(dist) / float(ctx + 1))
                    emissions[frame_pos][label_to_idx[left]] += float(bonus)
                if int(frame) >= int(boundary) and right in label_to_idx:
                    dist = max(0, int(frame) - int(e)) if int(frame) > int(e) else 0
                    bonus = anchor_boost * max(0.0, 1.0 - float(dist) / float(ctx + 1))
                    emissions[frame_pos][label_to_idx[right]] += float(bonus)
        if boundary is not None:
            for frame_pos, frame in enumerate(frames):
                if abs(int(frame) - int(boundary)) <= ctx:
                    switch_penalties[frame_pos] = min(
                        switch_penalties[frame_pos], 0.08
                    )

    for constraint in soft_constraints or []:
        kind = str(constraint.kind or "").strip().lower()
        s = int(constraint.start_frame)
        e = int(constraint.end_frame)
        if e < s:
            s, e = e, s
        weight = _clamp(constraint.weight, 0.0, 4.0)
        preferred = str((constraint.payload or {}).get("preferred_label") or "").strip()
        for frame_pos, frame in enumerate(frames):
            if int(frame) < int(s) or int(frame) > int(e):
                continue
            current_label = str(current.get(int(frame), "") or "").strip()
            if preferred and preferred in label_to_idx:
                emissions[frame_pos][label_to_idx[preferred]] += float(0.22 * weight)
            if kind == "multiview_preferred_label":
                switch_penalties[frame_pos] = max(
                    0.02, switch_penalties[frame_pos] - 0.08 * float(weight)
                )
            elif kind == "state_conflict_region":
                switch_penalties[frame_pos] = max(
                    0.02, switch_penalties[frame_pos] - 0.12 * float(weight)
                )
                if current_label and current_label in label_to_idx:
                    emissions[frame_pos][label_to_idx[current_label]] -= float(
                        0.10 * weight
                    )
            elif kind == "label_preference":
                switch_penalties[frame_pos] = max(
                    0.02, switch_penalties[frame_pos] - 0.04 * float(weight)
                )
            elif kind == "discourage_transition":
                switch_penalties[frame_pos] += float(0.15 * weight)
            elif kind == "encourage_transition":
                switch_penalties[frame_pos] = max(
                    0.02, switch_penalties[frame_pos] - 0.15 * float(weight)
                )

    for frame_pos, frame in enumerate(frames):
        current_label = str(current.get(int(frame), "") or "").strip()
        if not current_label or current_label not in label_to_idx:
            continue
        in_anchor_context = any(int(s) <= int(frame) <= int(e) for s, e in anchor_regions)
        if in_anchor_context:
            continue
        emissions[frame_pos][label_to_idx[current_label]] += float(stay_bonus)

    dp: List[List[float]] = [[-1e12 for _ in range(num_labels)] for _ in range(num_frames)]
    back: List[List[int]] = [[0 for _ in range(num_labels)] for _ in range(num_frames)]
    for label_idx in range(num_labels):
        dp[0][label_idx] = float(emissions[0][label_idx])
    for frame_pos in range(1, num_frames):
        penalty = float(max(0.0, switch_penalties[frame_pos]))
        for label_idx in range(num_labels):
            best_prev = 0
            best_score = -1e12
            for prev_idx in range(num_labels):
                trans = 0.0 if prev_idx == label_idx else penalty
                cand = float(dp[frame_pos - 1][prev_idx]) - float(trans)
                if cand > best_score:
                    best_score = cand
                    best_prev = int(prev_idx)
            dp[frame_pos][label_idx] = float(best_score + emissions[frame_pos][label_idx])
            back[frame_pos][label_idx] = int(best_prev)

    last_idx = max(range(num_labels), key=lambda idx: dp[-1][idx])
    label_path = [int(last_idx)]
    for frame_pos in range(num_frames - 1, 0, -1):
        label_path.append(int(back[frame_pos][label_path[-1]]))
    label_path.reverse()

    decoded = dict(current)
    switch_count = 0
    soft_penalty_total = 0.0
    for frame_pos, frame in enumerate(frames):
        label_idx = int(label_path[frame_pos])
        decoded[int(frame)] = str(label_names[label_idx])
        if frame_pos > 0 and int(label_path[frame_pos - 1]) != label_idx:
            switch_count += 1
            soft_penalty_total += float(switch_penalties[frame_pos])

    hard_before = count_anchor_violations(current, windows)
    hard_after = count_anchor_violations(decoded, windows)
    changed_spans = _changed_spans(current, decoded, frames)
    changed_frames = sum(max(0, int(e) - int(s) + 1) for s, e in changed_spans)
    diagnostics = {
        "decoded_frame_count": int(len(frames)),
        "changed_frame_count": int(changed_frames),
        "changed_spans": list(changed_spans),
        "hard_violations_before": int(hard_before),
        "hard_violations_after": int(hard_after),
        "soft_penalty_total": float(soft_penalty_total),
        "switch_count": int(switch_count),
        "label_vocabulary": list(label_names),
    }
    return decoded, diagnostics
