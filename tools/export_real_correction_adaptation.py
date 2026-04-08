#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impact_scribe_io import (
    collect_case_paths,
    extract_accepted_boundary_events,
    load_annotation_bundle,
    load_sidecar_payload,
)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))


def _build_side_scribbles(
    start_frame: int,
    end_frame: int,
    boundary_frame: int,
    *,
    width: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    left_end = min(int(end_frame), int(boundary_frame) - 1)
    if left_end >= int(start_frame):
        left_start = max(int(start_frame), int(left_end) - int(width) + 1)
        rows.append(
            {
                "start_frame": int(left_start),
                "end_frame": int(left_end),
                "kind": "left",
            }
        )
    right_start = max(int(start_frame), int(boundary_frame))
    if int(end_frame) >= right_start:
        right_end = min(int(end_frame), int(right_start) + int(width) - 1)
        rows.append(
            {
                "start_frame": int(right_start),
                "end_frame": int(right_end),
                "kind": "right",
            }
        )
    return rows


def _build_example(
    bundle: Dict[str, Any],
    annotation_path: Path,
    event: Dict[str, Any],
    *,
    window_radius: int,
    include_side_scribbles: bool,
    side_width: int,
) -> Dict[str, Any]:
    view_start = int(bundle.get("view_start", 0) or 0)
    view_end = int(bundle.get("view_end", view_start) or view_start)
    boundary_frame = int(
        event.get(
            "boundary_frame",
            (int(event.get("start_frame", 0) or 0) + int(event.get("end_frame", 0) or 0)) * 0.5,
        )
        or 0
    )
    boundary_frame = _clamp(boundary_frame, view_start, view_end)
    scribble_start = _clamp(
        int(event.get("start_frame", boundary_frame) or boundary_frame),
        view_start,
        view_end,
    )
    scribble_end = _clamp(
        int(event.get("end_frame", boundary_frame) or boundary_frame),
        view_start,
        view_end,
    )
    if scribble_end < scribble_start:
        scribble_start, scribble_end = scribble_end, scribble_start
    win_s = _clamp(int(boundary_frame) - int(window_radius), view_start, view_end)
    win_e = _clamp(int(boundary_frame) + int(window_radius), view_start, view_end)
    if win_e < win_s:
        win_e = win_s
    scribbles: List[Dict[str, Any]] = [
        {
            "start_frame": int(scribble_start),
            "end_frame": int(scribble_end),
            "kind": "uncertain",
        }
    ]
    if include_side_scribbles:
        scribbles.extend(
            _build_side_scribbles(
                scribble_start,
                scribble_end,
                int(boundary_frame),
                width=max(1, int(side_width)),
            )
        )
    return {
        "annotation_path": str(annotation_path),
        "video_id": str(
            bundle.get("video_id")
            or annotation_path.stem.replace("_native", "").replace("_annotations", "")
        ),
        "view": str(bundle.get("view", "") or ""),
        "view_start": int(view_start),
        "view_end": int(view_end),
        "window_start": int(win_s),
        "window_end": int(win_e),
        "boundary_frame": int(boundary_frame),
        "left_label": str(event.get("left_label", "") or ""),
        "right_label": str(event.get("right_label", "") or ""),
        "scribbles": scribbles,
        "metadata": {
            "source": "real_correction_v1",
            "history_index": int(event.get("history_index", 0) or 0),
            "started_at": str(event.get("started_at", "") or ""),
            "committed_at": str(event.get("committed_at", "") or ""),
            "steps": int(event.get("steps", 0) or 0),
            "raw_confidence": event.get("raw_confidence"),
            "calibrated_confidence": event.get("calibrated_confidence"),
            "second_correction": bool(event.get("second_correction")),
        },
    }


def _write_examples(output_path: Path, examples: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for item in examples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return
    payload = {
        "dataset_version": 1,
        "dataset_kind": "real_correction_adaptation",
        "num_examples": int(len(examples)),
        "examples": list(examples),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export accepted boundary corrections as a local-refiner fine-tuning "
            "dataset with real correction-derived scribble supervision."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Annotation json, `_scribble.json`, or a directory containing them.",
    )
    parser.add_argument("--output", required=True, help="Output .json or .jsonl path.")
    parser.add_argument("--window_radius", type=int, default=24)
    parser.add_argument("--side_width", type=int, default=6)
    parser.add_argument(
        "--no_side_scribbles",
        action="store_true",
        help="Do not synthesize side-support scribbles around the accepted boundary.",
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
        raise SystemExit(f"No annotation+sidecar cases found under: {input_path}")

    examples: List[Dict[str, Any]] = []
    for case in cases:
        if case.annotation_path is None or case.sidecar_path is None:
            continue
        bundle = load_annotation_bundle(case.annotation_path)
        sidecar_payload = load_sidecar_payload(case.sidecar_path)
        history = list(sidecar_payload.get("correction_history") or [])
        for event in extract_accepted_boundary_events(history):
            left_label = str(event.get("left_label", "") or "").strip()
            right_label = str(event.get("right_label", "") or "").strip()
            if not left_label or not right_label:
                continue
            examples.append(
                _build_example(
                    bundle,
                    case.annotation_path,
                    event,
                    window_radius=max(1, int(args.window_radius)),
                    include_side_scribbles=not bool(args.no_side_scribbles),
                    side_width=max(1, int(args.side_width)),
                )
            )

    if not examples:
        raise SystemExit("No accepted boundary corrections were found.")

    out_path = Path(args.output).expanduser().resolve()
    _write_examples(out_path, examples)
    print(
        f"[export_real_correction_adaptation] wrote {len(examples)} examples to {out_path}"
    )


if __name__ == "__main__":
    main()
