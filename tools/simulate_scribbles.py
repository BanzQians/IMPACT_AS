#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return data


def _parse_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _label_table(data: Dict[str, Any]) -> Dict[int, str]:
    rows = data.get("labels")
    if not isinstance(rows, list):
        rows = data.get("action_labels")
    if not isinstance(rows, list):
        rows = []
    out: Dict[int, str] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        lid = _parse_int(item.get("id"))
        if lid is None:
            continue
        out[int(lid)] = str(item.get("name", f"Label_{lid}") or f"Label_{lid}")
    return out


def _normalize_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    label_names = _label_table(data)
    rows = data.get("segments")
    if not isinstance(rows, list):
        raise ValueError("Expected 'segments' list in native annotation JSON")
    segments: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        start = _parse_int(item.get("start_frame"))
        end = _parse_int(item.get("end_frame"))
        label_id = _parse_int(item.get("action_label"))
        if start is None or end is None or label_id is None:
            continue
        if end < start:
            start, end = end, start
        label_name = str(label_names.get(int(label_id), f"Label_{label_id}"))
        segments.append(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "action_label": int(label_id),
                "label_name": label_name,
            }
        )
    segments.sort(key=lambda row: (int(row["start_frame"]), int(row["end_frame"])))
    return segments


def _iter_inputs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    rows: List[Path] = []
    for item in sorted(path.rglob("*.json")):
        name = item.name.lower()
        if name.endswith("_extra.json") or name.endswith("_scribble.json"):
            continue
        rows.append(item)
    return rows


def _clip_interval(start: int, end: int, lo: int, hi: int) -> Tuple[int, int]:
    s = max(int(lo), min(int(start), int(hi)))
    e = max(int(lo), min(int(end), int(hi)))
    if e < s:
        s, e = e, s
    return s, e


def _rand_span(
    center: int,
    *,
    width_range: Tuple[int, int],
    offset_range: Tuple[int, int],
    lo: int,
    hi: int,
    rng: random.Random,
) -> Tuple[int, int]:
    width = rng.randint(int(width_range[0]), int(width_range[1]))
    offset = rng.randint(int(offset_range[0]), int(offset_range[1]))
    span_center = int(center) + int(offset)
    start = int(span_center - width // 2)
    end = int(start + max(0, width - 1))
    return _clip_interval(start, end, lo, hi)


def _simulate_examples_for_file(
    path: Path,
    *,
    window_radius: int,
    samples_per_boundary: int,
    uncertain_width: Tuple[int, int],
    uncertain_jitter: Tuple[int, int],
    side_width: Tuple[int, int],
    include_side_scribbles: bool,
    include_same_label_boundaries: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    data = _load_json(path)
    segments = _normalize_segments(data)
    if len(segments) < 2:
        return []

    view_start = _parse_int(
        data.get("view_start", (data.get("meta_data") or {}).get("view_start", 0)), 0
    )
    view_end = _parse_int(
        data.get("view_end", (data.get("meta_data") or {}).get("view_end")),
        max(int(seg["end_frame"]) for seg in segments),
    )
    view_start = int(view_start or 0)
    view_end = int(view_end or view_start)
    if view_end < view_start:
        view_end = max(view_start, max(int(seg["end_frame"]) for seg in segments))

    rng = random.Random(f"{seed}:{path.as_posix()}")
    examples: List[Dict[str, Any]] = []
    for idx in range(len(segments) - 1):
        left = dict(segments[idx])
        right = dict(segments[idx + 1])
        left_label = str(left.get("label_name", "") or "")
        right_label = str(right.get("label_name", "") or "")
        if not include_same_label_boundaries and left_label == right_label:
            continue
        boundary_frame = int(right["start_frame"])
        boundary_lo = max(view_start, int(left["start_frame"]))
        boundary_hi = min(view_end, int(right["end_frame"]))
        if boundary_hi < boundary_lo:
            continue
        for sample_idx in range(max(1, int(samples_per_boundary))):
            uncertain_s, uncertain_e = _rand_span(
                boundary_frame,
                width_range=uncertain_width,
                offset_range=uncertain_jitter,
                lo=boundary_lo,
                hi=boundary_hi,
                rng=rng,
            )
            scribbles = [
                {
                    "start_frame": int(uncertain_s),
                    "end_frame": int(uncertain_e),
                    "kind": "uncertain",
                }
            ]
            if include_side_scribbles:
                left_center = max(int(left["start_frame"]), boundary_frame - 1)
                left_s, left_e = _rand_span(
                    left_center,
                    width_range=side_width,
                    offset_range=(-max(1, side_width[1]), 0),
                    lo=int(left["start_frame"]),
                    hi=max(int(left["start_frame"]), boundary_frame - 1),
                    rng=rng,
                )
                if left_e >= left_s:
                    scribbles.append(
                        {
                            "start_frame": int(left_s),
                            "end_frame": int(left_e),
                            "kind": "left",
                        }
                    )
                right_center = min(int(right["end_frame"]), boundary_frame)
                right_s, right_e = _rand_span(
                    right_center,
                    width_range=side_width,
                    offset_range=(0, max(1, side_width[1])),
                    lo=boundary_frame,
                    hi=int(right["end_frame"]),
                    rng=rng,
                )
                if right_e >= right_s:
                    scribbles.append(
                        {
                            "start_frame": int(right_s),
                            "end_frame": int(right_e),
                            "kind": "right",
                        }
                    )

            examples.append(
                {
                    "annotation_path": str(path),
                    "video_id": str(
                        data.get("video_id")
                        or path.stem.replace("_native", "").replace("_annotations", "")
                    ),
                    "view": str(data.get("view", "") or ""),
                    "view_start": int(view_start),
                    "view_end": int(view_end),
                    "window_start": int(max(view_start, boundary_frame - window_radius)),
                    "window_end": int(min(view_end, boundary_frame + window_radius)),
                    "boundary_frame": int(boundary_frame),
                    "left_label": left_label,
                    "right_label": right_label,
                    "left_segment": {
                        "start_frame": int(left["start_frame"]),
                        "end_frame": int(left["end_frame"]),
                        "label": left_label,
                    },
                    "right_segment": {
                        "start_frame": int(right["start_frame"]),
                        "end_frame": int(right["end_frame"]),
                        "label": right_label,
                    },
                    "scribbles": scribbles,
                    "metadata": {
                        "boundary_index": int(idx),
                        "sample_index": int(sample_idx),
                        "source": "synthetic_scribble_v1",
                    },
                }
            )
    return examples


def _write_examples(
    output_path: Path,
    examples: List[Dict[str, Any]],
    *,
    source_inputs: Iterable[Path],
    jsonl: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with output_path.open("w", encoding="utf-8") as f:
            for item in examples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return
    payload = {
        "dataset_version": 1,
        "num_examples": int(len(examples)),
        "source_inputs": [str(path) for path in source_inputs],
        "examples": examples,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate synthetic temporal scribble supervision from native annotation JSON."
    )
    ap.add_argument("--input", required=True, help="Annotation json file or directory")
    ap.add_argument("--output", default="", help="Output json/jsonl path")
    ap.add_argument("--window_radius", type=int, default=24)
    ap.add_argument("--samples_per_boundary", type=int, default=3)
    ap.add_argument("--uncertain_min_width", type=int, default=5)
    ap.add_argument("--uncertain_max_width", type=int, default=13)
    ap.add_argument("--uncertain_min_jitter", type=int, default=-3)
    ap.add_argument("--uncertain_max_jitter", type=int, default=3)
    ap.add_argument("--side_min_width", type=int, default=2)
    ap.add_argument("--side_max_width", type=int, default=6)
    ap.add_argument("--no_side_scribbles", action="store_true")
    ap.add_argument("--include_same_label_boundaries", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input does not exist: {src}")
    inputs = _iter_inputs(src)
    if not inputs:
        raise SystemExit("No annotation json files found.")

    all_examples: List[Dict[str, Any]] = []
    for path in inputs:
        all_examples.extend(
            _simulate_examples_for_file(
                path,
                window_radius=max(1, int(args.window_radius)),
                samples_per_boundary=max(1, int(args.samples_per_boundary)),
                uncertain_width=(
                    max(1, int(args.uncertain_min_width)),
                    max(int(args.uncertain_min_width), int(args.uncertain_max_width)),
                ),
                uncertain_jitter=(
                    int(args.uncertain_min_jitter),
                    int(args.uncertain_max_jitter),
                ),
                side_width=(
                    max(1, int(args.side_min_width)),
                    max(int(args.side_min_width), int(args.side_max_width)),
                ),
                include_side_scribbles=not bool(args.no_side_scribbles),
                include_same_label_boundaries=bool(args.include_same_label_boundaries),
                seed=int(args.seed),
            )
        )

    if not all_examples:
        raise SystemExit("No synthetic scribble examples were generated.")

    if args.output:
        out_path = Path(args.output)
    else:
        stem = src.stem if src.is_file() else src.name
        suffix = ".jsonl" if len(inputs) > 1 else ".json"
        out_path = src.parent / f"{stem}_synthetic_scribbles{suffix}"
    jsonl = out_path.suffix.lower() == ".jsonl" or len(inputs) > 1
    _write_examples(out_path, all_examples, source_inputs=inputs, jsonl=jsonl)
    print(
        f"[simulate_scribbles] wrote {len(all_examples)} examples from {len(inputs)} input file(s) to {out_path}"
    )


if __name__ == "__main__":
    main()
