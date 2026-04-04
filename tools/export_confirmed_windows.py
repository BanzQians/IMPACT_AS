#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from impact_scribe_io import (
    collect_case_paths,
    dense_window_payload,
    extract_accepted_boundary_events,
    extract_correction_events,
    load_annotation_bundle,
    load_sidecar_payload,
)

from core.query_planner import QueryType


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))


def _history_export_events(
    history: Sequence[Dict[str, Any]],
    *,
    include_label_accepts: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = list(extract_accepted_boundary_events(history))
    if include_label_accepts:
        rows.extend(
            [
                row
                for row in extract_correction_events(
                    history,
                    query_types=[QueryType.LABEL_REVIEW.value],
                    include_feedback=False,
                )
                if bool(row.get("accepted"))
            ]
        )
    rows.sort(
        key=lambda item: (
            str(item.get("committed_at", "")),
            int(item.get("history_index", 0)),
        )
    )
    return rows


def _fallback_accept_events(
    sidecar_payload: Dict[str, Any],
    *,
    include_label_accepts: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in list(sidecar_payload.get("confirmed_accept_records") or []):
        if not isinstance(record, dict):
            continue
        point_type = str(record.get("point_type", "") or "").strip().lower()
        if point_type == "boundary":
            try:
                frame = int(record.get("boundary_frame", record.get("feedback_start", 0)))
            except Exception:
                continue
            rows.append(
                {
                    "query_type": QueryType.BOUNDARY_SCRIBBLE.value,
                    "point_type": "boundary_accept_record",
                    "start_frame": int(frame),
                    "end_frame": int(frame),
                    "boundary_frame": int(frame),
                    "left_label": str(record.get("left_label", "") or ""),
                    "right_label": str(record.get("right_label", "") or ""),
                    "accepted": True,
                    "changed": True,
                    "steps": 0,
                    "started_at": "",
                    "committed_at": "",
                    "second_correction": False,
                }
            )
        elif include_label_accepts and point_type == "label":
            try:
                start = int(record.get("feedback_start", 0))
                end = int(record.get("feedback_end", start))
            except Exception:
                continue
            if end < start:
                start, end = end, start
            rows.append(
                {
                    "query_type": QueryType.LABEL_REVIEW.value,
                    "point_type": "label_accept_record",
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "boundary_frame": None,
                    "left_label": "",
                    "right_label": str(record.get("label", "") or ""),
                    "accepted": True,
                    "changed": True,
                    "steps": 0,
                    "started_at": "",
                    "committed_at": "",
                    "second_correction": False,
                }
            )
    return rows


def _build_window_sample(
    bundle: Dict[str, Any],
    annotation_path: Path,
    sidecar_path: Path | None,
    event: Dict[str, Any],
    *,
    interaction_index: int,
    context_pad: int,
    min_span: int,
) -> Dict[str, Any]:
    view_start = int(bundle.get("view_start", 0) or 0)
    view_end = int(bundle.get("view_end", view_start) or view_start)
    start = int(event.get("start_frame", 0) or 0)
    end = int(event.get("end_frame", start) or start)
    if end < start:
        start, end = end, start
    span_len = max(1, int(end - start + 1))
    if span_len < int(min_span):
        center = int((start + end) * 0.5)
        half = int(max(0, min_span - 1) // 2)
        start = int(center - half)
        end = int(start + min_span - 1)
    if int(context_pad) > 0:
        start -= int(context_pad)
        end += int(context_pad)
    start = _clamp(start, view_start, view_end)
    end = _clamp(end, view_start, view_end)
    if end < start:
        end = start

    boundary_frame = event.get("boundary_frame")
    if boundary_frame is not None:
        boundary_frame = _clamp(int(boundary_frame), start, end)

    window = dense_window_payload(bundle, start, end)
    return {
        "schema_version": 1,
        "source": "correction_history"
        if str(event.get("point_type", "") or "").startswith("boundary_scribble")
        or str(event.get("point_type", "") or "").startswith("label")
        else "confirmed_accept_records",
        "annotation_path": str(annotation_path),
        "sidecar_path": str(sidecar_path) if sidecar_path else "",
        "video_id": str(bundle.get("video_id", "")),
        "view": str(bundle.get("view", "")),
        "view_start": int(view_start),
        "view_end": int(view_end),
        "interaction_index": int(interaction_index),
        "query_type": str(event.get("query_type", "") or ""),
        "point_type": str(event.get("point_type", "") or ""),
        "started_at": str(event.get("started_at", "") or ""),
        "committed_at": str(event.get("committed_at", "") or ""),
        "start_frame": int(window["start_frame"]),
        "end_frame": int(window["end_frame"]),
        "start_frame_rel": int(window["start_frame"]) - int(view_start),
        "end_frame_rel": int(window["end_frame"]) - int(view_start),
        "span_len": int(window["span_len"]),
        "boundary_frame": (None if boundary_frame is None else int(boundary_frame)),
        "boundary_frame_rel": (
            None if boundary_frame is None else int(boundary_frame) - int(view_start)
        ),
        "left_label": str(event.get("left_label", "") or ""),
        "right_label": str(event.get("right_label", "") or ""),
        "accepted": bool(event.get("accepted")),
        "changed": bool(event.get("changed")),
        "steps": int(event.get("steps", 0) or 0),
        "second_correction": bool(event.get("second_correction")),
        "raw_confidence": event.get("raw_confidence"),
        "calibrated_confidence": event.get("calibrated_confidence"),
        "changed_frame_count": int(event.get("changed_frame_count", 0) or 0),
        "segments_abs": list(window["segments_abs"]),
        "segments_rel": list(window["segments_rel"]),
        "dense_labels": list(window["dense_labels"]),
    }


def _summary(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    boundary_count = sum(
        1
        for row in samples
        if str(row.get("query_type", "") or "") == QueryType.BOUNDARY_SCRIBBLE.value
    )
    label_count = sum(
        1
        for row in samples
        if str(row.get("query_type", "") or "") == QueryType.LABEL_REVIEW.value
    )
    avg_span = (
        float(sum(int(row.get("span_len", 0) or 0) for row in samples) / len(samples))
        if samples
        else 0.0
    )
    return {
        "window_count": int(len(samples)),
        "boundary_window_count": int(boundary_count),
        "label_window_count": int(label_count),
        "avg_span_len": float(avg_span),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export accepted IMPACT-Scribe corrections as dense local supervision windows."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Annotation json, `_scribble.json`, or a directory containing them.",
    )
    parser.add_argument("--out", required=True, help="Output .json or .jsonl path.")
    parser.add_argument(
        "--context-pad",
        type=int,
        default=0,
        help="Extra frame context added on both sides of each confirmed window.",
    )
    parser.add_argument(
        "--min-span",
        type=int,
        default=1,
        help="Minimum exported span length in frames.",
    )
    parser.add_argument(
        "--include-label-accepts",
        action="store_true",
        help="Also export accepted label-review windows.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")

    cases = collect_case_paths(input_path)
    cases = [
        case
        for case in cases
        if case.annotation_path is not None and case.sidecar_path is not None
    ]
    if not cases:
        raise SystemExit(
            f"No annotation+sidecar cases found under: {input_path}"
        )

    samples: List[Dict[str, Any]] = []
    for case in cases:
        if case.annotation_path is None or not case.annotation_path.is_file():
            print(f"[WARN] Skip case without annotation json: {case.sidecar_path}")
            continue
        bundle = load_annotation_bundle(case.annotation_path)
        sidecar_payload = load_sidecar_payload(case.sidecar_path)
        history = list(sidecar_payload.get("correction_history") or [])
        events = _history_export_events(
            history,
            include_label_accepts=bool(args.include_label_accepts),
        )
        if not events:
            events = _fallback_accept_events(
                sidecar_payload,
                include_label_accepts=bool(args.include_label_accepts),
            )
        if not events:
            print(f"[WARN] No confirmed windows found for {case.annotation_path}")
            continue
        for index, event in enumerate(events, start=1):
            samples.append(
                _build_window_sample(
                    bundle,
                    case.annotation_path,
                    case.sidecar_path,
                    event,
                    interaction_index=index,
                    context_pad=max(0, int(args.context_pad)),
                    min_span=max(1, int(args.min_span)),
                )
            )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for row in samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        payload = {
            "schema_version": 1,
            "summary": _summary(samples),
            "windows": list(samples),
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    report = _summary(samples)
    print(
        "[EXPORT] windows={window_count} boundary={boundary_window_count} "
        "label={label_window_count} avg_span={avg_span_len:.2f}".format(**report)
    )
    print(f"[EXPORT] wrote {out_path}")


if __name__ == "__main__":
    main()
