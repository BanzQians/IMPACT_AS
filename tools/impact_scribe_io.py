from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.query_planner import QueryType, summarize_correction_observation


ROOT_TRACK = "__root__"


@dataclass(frozen=True)
class CasePaths:
    annotation_path: Optional[Path]
    sidecar_path: Optional[Path]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_span(row: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    candidates = [
        ("start_frame", "end_frame"),
        ("f_start", "f_end"),
        ("start", "end"),
    ]
    for start_key, end_key in candidates:
        start_raw = row.get(start_key)
        end_raw = row.get(end_key)
        if start_raw is None and end_raw is None:
            continue
        try:
            start_val = int(start_raw if start_raw is not None else end_raw)
            end_val = int(end_raw if end_raw is not None else start_raw)
        except Exception:
            continue
        if end_val < start_val:
            start_val, end_val = end_val, start_val
        return int(start_val), int(end_val)
    return None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return data


def is_sidecar_path(path: Path) -> bool:
    return str(path.name).lower().endswith("_scribble.json")


def infer_annotation_path(path: Path) -> Optional[Path]:
    if not is_sidecar_path(path):
        return path if path.is_file() else None
    stem = path.name[: -len("_scribble.json")]
    candidate = path.with_name(stem + ".json")
    return candidate if candidate.is_file() else None


def infer_sidecar_path(path: Path) -> Optional[Path]:
    if is_sidecar_path(path):
        return path if path.is_file() else None
    if path.suffix.lower() != ".json":
        return None
    candidate = path.with_name(path.stem + "_scribble.json")
    return candidate if candidate.is_file() else None


def collect_case_paths(input_path: Path) -> List[CasePaths]:
    pairs: List[CasePaths] = []
    seen = set()

    def _add(annotation_path: Optional[Path], sidecar_path: Optional[Path]) -> None:
        ann = annotation_path.resolve() if annotation_path and annotation_path.exists() else None
        side = sidecar_path.resolve() if sidecar_path and sidecar_path.exists() else None
        if ann is None and side is None:
            return
        key = (str(ann or ""), str(side or ""))
        if key in seen:
            return
        seen.add(key)
        pairs.append(CasePaths(annotation_path=ann, sidecar_path=side))

    if input_path.is_file():
        if is_sidecar_path(input_path):
            _add(infer_annotation_path(input_path), input_path)
        elif input_path.suffix.lower() == ".json":
            _add(input_path, infer_sidecar_path(input_path))
        return sorted(
            pairs,
            key=lambda item: (
                str(item.annotation_path or ""),
                str(item.sidecar_path or ""),
            ),
        )

    for sidecar in sorted(input_path.rglob("*_scribble.json")):
        _add(infer_annotation_path(sidecar), sidecar)
    for annotation in sorted(input_path.rglob("*.json")):
        if is_sidecar_path(annotation):
            continue
        _add(annotation, infer_sidecar_path(annotation))
    return sorted(
        pairs,
        key=lambda item: (
            str(item.annotation_path or ""),
            str(item.sidecar_path or ""),
        ),
    )


def load_annotation_bundle(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    labels_raw = payload.get("labels") or payload.get("action_labels") or []
    if not isinstance(labels_raw, list):
        labels_raw = []
    id_to_label: Dict[int, str] = {}
    for row in labels_raw:
        if not isinstance(row, dict):
            continue
        try:
            lid = int(row.get("id"))
        except Exception:
            continue
        name = str(row.get("name", f"Label_{lid}")).strip() or f"Label_{lid}"
        id_to_label[lid] = name

    view_start = _safe_int(payload.get("view_start", 0), 0)
    view_end = _safe_int(payload.get("view_end", view_start), view_start)
    segments_raw = payload.get("segments") or []
    if not isinstance(segments_raw, list):
        segments_raw = []

    segments_abs: List[Dict[str, Any]] = []
    track_labels: Dict[str, Dict[int, str]] = {}
    track_segments: Dict[str, List[Dict[str, Any]]] = {}

    for row in segments_raw:
        if not isinstance(row, dict):
            continue
        span = _safe_span(row)
        if span is None:
            continue
        start_rel, end_rel = span
        if end_rel < start_rel:
            start_rel, end_rel = end_rel, start_rel
        start_abs = int(view_start + start_rel)
        end_abs = int(view_start + end_rel)
        label = ""
        try:
            label = id_to_label.get(
                int(row.get("action_label")),
                "",
            )
        except Exception:
            label = ""
        if not label:
            label = str(
                row.get("label")
                or row.get("label_name")
                or row.get("name")
                or ""
            ).strip()
        entity = str(row.get("entity", "") or "").strip() or None
        record = {
            "start_frame": int(start_abs),
            "end_frame": int(end_abs),
            "start_frame_rel": int(start_rel),
            "end_frame_rel": int(end_rel),
            "label": str(label),
            "entity": entity,
        }
        segments_abs.append(record)
        track_key = entity or ROOT_TRACK
        track_segments.setdefault(track_key, []).append(record)
        track_map = track_labels.setdefault(track_key, {})
        for frame in range(int(start_abs), int(end_abs) + 1):
            track_map[int(frame)] = str(label)

    for rows in track_segments.values():
        rows.sort(
            key=lambda item: (
                int(item.get("start_frame", 0)),
                int(item.get("end_frame", 0)),
                str(item.get("label", "")),
            )
        )

    boundary_set = set()
    for rows in track_segments.values():
        for idx, row in enumerate(rows):
            if idx == 0:
                continue
            boundary_set.add(int(row.get("start_frame", 0)))

    track_keys = sorted(track_segments.keys())
    primary_track: Optional[str] = None
    if ROOT_TRACK in track_segments:
        primary_track = ROOT_TRACK
    elif len(track_keys) == 1:
        primary_track = track_keys[0]

    return {
        "path": str(path),
        "video_id": str(payload.get("video_id", path.stem)),
        "view": str(payload.get("view", "")),
        "view_start": int(view_start),
        "view_end": int(view_end),
        "labels": dict(id_to_label),
        "segments_abs": list(
            sorted(
                segments_abs,
                key=lambda item: (
                    str(item.get("entity") or ""),
                    int(item.get("start_frame", 0)),
                    int(item.get("end_frame", 0)),
                ),
            )
        ),
        "track_segments": {str(k): list(v) for k, v in track_segments.items()},
        "track_labels": {str(k): dict(v) for k, v in track_labels.items()},
        "track_keys": list(track_keys),
        "primary_track": primary_track,
        "boundaries_abs": sorted(int(x) for x in boundary_set),
    }


def load_sidecar_payload(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return load_json(path)


def _history_sort_key(summary: Dict[str, Any]) -> Tuple[str, int]:
    stamp = (
        str(summary.get("committed_at") or "")
        or str(summary.get("discarded_at") or "")
        or str(summary.get("started_at") or "")
    )
    return (stamp, 0)


def sorted_history(history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(item) for item in history if isinstance(item, dict)]
    return sorted(rows, key=_history_sort_key)


def _query_type_value(raw: Any) -> Optional[str]:
    if isinstance(raw, QueryType):
        return str(raw.value)
    text = str(raw or "").strip().lower()
    if not text:
        return None
    if text in {
        QueryType.BOUNDARY_SCRIBBLE.value,
        QueryType.LABEL_REVIEW.value,
        QueryType.STATE_REPAIR.value,
    }:
        return text
    return None


def extract_correction_events(
    history: Sequence[Dict[str, Any]],
    *,
    query_types: Optional[Iterable[str]] = None,
    include_feedback: bool = False,
) -> List[Dict[str, Any]]:
    allowed = {
        value
        for value in (_query_type_value(item) for item in (query_types or []))
        if value
    }
    rows: List[Dict[str, Any]] = []
    for index, summary in enumerate(sorted_history(history)):
        obs = summarize_correction_observation(summary)
        if not obs:
            continue
        if not include_feedback and bool(obs.get("proposal_feedback")):
            continue
        qtype = _query_type_value(obs.get("query_type"))
        if allowed and qtype not in allowed:
            continue
        meta = summary.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        rows.append(
            {
                "history_index": int(index),
                "query_type": str(qtype or ""),
                "point_type": str(meta.get("point_type", "") or "").strip().lower(),
                "proposal_feedback": bool(obs.get("proposal_feedback")),
                "accepted": bool(obs.get("accepted")),
                "changed": bool(obs.get("changed")),
                "steps": int(obs.get("steps", 0) or 0),
                "start_frame": int(obs.get("start_frame", 0) or 0),
                "end_frame": int(obs.get("end_frame", obs.get("start_frame", 0)) or 0),
                "span_len": int(obs.get("span_len", 1) or 1),
                "boundary_frame": (
                    None
                    if meta.get("boundary_frame") is None
                    else int(meta.get("boundary_frame"))
                ),
                "left_label": str(meta.get("left_label", "") or ""),
                "right_label": str(meta.get("right_label", "") or ""),
                "raw_confidence": (
                    None
                    if meta.get("raw_confidence") is None
                    else _safe_float(meta.get("raw_confidence"), 0.0)
                ),
                "calibrated_confidence": (
                    None
                    if meta.get("calibrated_confidence") is None
                    else _safe_float(meta.get("calibrated_confidence"), 0.0)
                ),
                "changed_frame_count": int(obs.get("changed_frame_count", 0) or 0),
                "anchor_before": int(obs.get("anchor_before", 0) or 0),
                "anchor_after": int(obs.get("anchor_after", 0) or 0),
                "state_before": int(obs.get("state_before", 0) or 0),
                "state_after": int(obs.get("state_after", 0) or 0),
                "second_correction": bool(obs.get("second_correction")),
                "started_at": str(summary.get("started_at") or ""),
                "committed_at": str(
                    summary.get("committed_at") or summary.get("discarded_at") or ""
                ),
                "meta": dict(meta),
                "summary": dict(summary),
            }
        )
    return rows


def extract_boundary_feedback_events(
    history: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        row
        for row in extract_correction_events(
            history,
            query_types=[QueryType.BOUNDARY_SCRIBBLE.value],
            include_feedback=True,
        )
        if bool(row.get("proposal_feedback"))
    ]


def extract_accepted_boundary_events(
    history: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for row in extract_correction_events(
        history,
        query_types=[QueryType.BOUNDARY_SCRIBBLE.value],
        include_feedback=False,
    ):
        if bool(row.get("accepted")):
            rows.append(row)
    return rows


def dense_window_payload(
    bundle: Dict[str, Any],
    start_frame: int,
    end_frame: int,
) -> Dict[str, Any]:
    start = int(start_frame)
    end = int(end_frame)
    if end < start:
        start, end = end, start
    primary_track = bundle.get("primary_track")
    track_labels = bundle.get("track_labels") or {}
    frame_map = (
        dict(track_labels.get(primary_track, {}))
        if primary_track is not None
        else {}
    )
    dense_labels = [str(frame_map.get(frame, "") or "") for frame in range(start, end + 1)]

    segments_abs: List[Dict[str, Any]] = []
    for row in bundle.get("segments_abs", []) or []:
        if int(row.get("end_frame", 0)) < start or int(row.get("start_frame", 0)) > end:
            continue
        seg_start = max(int(row.get("start_frame", start)), int(start))
        seg_end = min(int(row.get("end_frame", end)), int(end))
        segments_abs.append(
            {
                "start_frame": int(seg_start),
                "end_frame": int(seg_end),
                "label": str(row.get("label", "") or ""),
                "entity": row.get("entity"),
            }
        )
    segments_abs.sort(
        key=lambda item: (
            str(item.get("entity") or ""),
            int(item.get("start_frame", 0)),
            int(item.get("end_frame", 0)),
        )
    )

    segments_rel = [
        {
            "start_frame": int(item["start_frame"]) - int(start),
            "end_frame": int(item["end_frame"]) - int(start),
            "label": str(item.get("label", "") or ""),
            "entity": item.get("entity"),
        }
        for item in segments_abs
    ]
    return {
        "start_frame": int(start),
        "end_frame": int(end),
        "span_len": int(end - start + 1),
        "dense_labels": list(dense_labels),
        "segments_abs": list(segments_abs),
        "segments_rel": list(segments_rel),
    }


def boundaries_from_frame_labels(labels: Sequence[Any]) -> List[int]:
    if not labels:
        return []
    out: List[int] = []
    prev = labels[0]
    for index in range(1, len(labels)):
        if labels[index] != prev:
            out.append(int(index))
            prev = labels[index]
    return out


def load_gt_boundaries(path: Path) -> Tuple[List[int], Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        bundle = load_annotation_bundle(path)
        return list(bundle.get("boundaries_abs", [])), {
            "path": str(path),
            "format": "json_segments",
            "video_id": str(bundle.get("video_id", "")),
        }
    if suffix == ".txt":
        labels: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    labels.append(text)
        return boundaries_from_frame_labels(labels), {
            "path": str(path),
            "format": "txt_frame_labels",
            "frame_count": int(len(labels)),
        }
    if suffix == ".npy":
        arr = np.load(str(path), allow_pickle=True)
        arr = np.asarray(arr)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            return [], {"path": str(path), "format": "npy_empty"}
        try:
            unique = set(int(x) for x in np.unique(arr))
        except Exception:
            unique = set()
        if unique.issubset({0, 1}) and int(np.count_nonzero(arr)) <= int(arr.size // 2):
            bounds = [int(idx) for idx, value in enumerate(arr.tolist()) if int(value) != 0]
            return bounds, {
                "path": str(path),
                "format": "npy_boundary_mask",
                "frame_count": int(arr.size),
            }
        labels = [str(item) for item in arr.tolist()]
        return boundaries_from_frame_labels(labels), {
            "path": str(path),
            "format": "npy_frame_labels",
            "frame_count": int(arr.size),
        }
    raise ValueError(f"Unsupported GT format: {path}")
