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
    for row_idx, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        start = _parse_int(item.get("start_frame"))
        if start is None:
            start = _parse_int(item.get("f_start"))
        end = _parse_int(item.get("end_frame"))
        if end is None:
            end = _parse_int(item.get("f_end"))
        label_id = _parse_int(item.get("action_label"))
        if label_id is None:
            label_id = _parse_int(item.get("label_id"))
        if label_id is None:
            label_id = _parse_int(item.get("id"), int(row_idx))
        if start is None or end is None or label_id is None:
            continue
        if end < start:
            start, end = end, start
        raw_label = item.get("label")
        label_name = str(raw_label or label_names.get(int(label_id), f"Label_{label_id}"))
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


def _sample_signed_offset(rng: random.Random, min_abs: int, max_abs: int) -> int:
    lo = max(0, int(min_abs))
    hi = max(lo, int(max_abs))
    if hi <= 0:
        return 0
    mag = rng.randint(lo, hi)
    if mag == 0:
        return 0
    return mag if rng.random() < 0.5 else -mag


def _rand_span_between(
    a: int,
    b: int,
    *,
    pad_range: Tuple[int, int],
    lo: int,
    hi: int,
    rng: random.Random,
) -> Tuple[int, int]:
    pad_lo = max(0, int(pad_range[0]))
    pad_hi = max(pad_lo, int(pad_range[1]))
    pad = rng.randint(pad_lo, pad_hi)
    return _clip_interval(
        min(int(a), int(b)) - int(pad),
        max(int(a), int(b)) + int(pad),
        lo,
        hi,
    )


def _simulate_examples_for_file(
    path: Path,
    *,
    window_radius: int,
    samples_per_boundary: int,
    uncertain_width: Tuple[int, int],
    uncertain_jitter: Tuple[int, int],
    scribble_profile: str,
    proposal_jitter: Tuple[int, int],
    span_probability: float,
    span_pad: Tuple[int, int],
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
            profile = str(scribble_profile or "legacy").strip().lower()
            if profile == "mixed":
                mode = "span" if rng.random() < float(span_probability) else "noisy"
            elif profile in {"legacy", "clean", "noisy", "span"}:
                mode = profile
            else:
                mode = "legacy"

            proposal_boundary = int(boundary_frame)
            window_center = int(boundary_frame)
            if mode != "legacy":
                proposal_boundary = int(
                    max(
                        view_start,
                        min(
                            view_end,
                            boundary_frame
                            + _sample_signed_offset(
                                rng,
                                int(proposal_jitter[0]),
                                int(proposal_jitter[1]),
                            ),
                        ),
                    )
                )
                window_center = int(proposal_boundary)

            if mode == "clean":
                uncertain_s, uncertain_e = _rand_span(
                    boundary_frame,
                    width_range=uncertain_width,
                    offset_range=(0, 0),
                    lo=view_start,
                    hi=view_end,
                    rng=rng,
                )
            elif mode == "span":
                uncertain_s, uncertain_e = _rand_span_between(
                    boundary_frame,
                    proposal_boundary,
                    pad_range=span_pad,
                    lo=view_start,
                    hi=view_end,
                    rng=rng,
                )
            else:
                uncertain_s, uncertain_e = _rand_span(
                    boundary_frame,
                    width_range=uncertain_width,
                    offset_range=uncertain_jitter,
                    lo=view_start if mode != "legacy" else boundary_lo,
                    hi=view_end if mode != "legacy" else boundary_hi,
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
                left_hi = max(int(left["start_frame"]), min(boundary_frame - 1, int(uncertain_s) - 1))
                left_center = left_hi if left_hi >= int(left["start_frame"]) else max(int(left["start_frame"]), boundary_frame - 1)
                left_s, left_e = _rand_span(
                    left_center,
                    width_range=side_width,
                    offset_range=(-max(1, side_width[1]), 0),
                    lo=int(left["start_frame"]),
                    hi=max(int(left["start_frame"]), left_center),
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
                right_lo = min(int(right["end_frame"]), max(boundary_frame, int(uncertain_e) + 1))
                right_center = right_lo if right_lo <= int(right["end_frame"]) else min(int(right["end_frame"]), boundary_frame)
                right_s, right_e = _rand_span(
                    right_center,
                    width_range=side_width,
                    offset_range=(0, max(1, side_width[1])),
                    lo=min(int(right["end_frame"]), right_center),
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
                    "window_start": int(window_center - window_radius),
                    "window_end": int(window_center + window_radius),
                    "window_center_frame": int(window_center),
                    "proposal_boundary_frame": int(proposal_boundary),
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
                        "source": "synthetic_scribble_v2" if mode != "legacy" else "synthetic_scribble_v1",
                        "scribble_profile": str(profile),
                        "scribble_mode": str(mode),
                        "window_center_frame": int(window_center),
                        "proposal_boundary_frame": int(proposal_boundary),
                        "uncertain_start": int(uncertain_s),
                        "uncertain_end": int(uncertain_e),
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
    ap.add_argument(
        "--scribble_profile",
        choices=("legacy", "clean", "noisy", "span", "mixed"),
        default="legacy",
        help=(
            "legacy preserves the original GT-centered generator; noisy offsets the "
            "uncertain interval; span covers a synthetic proposed boundary and GT; "
            "mixed samples noisy/span examples."
        ),
    )
    ap.add_argument("--proposal_min_jitter", type=int, default=4)
    ap.add_argument("--proposal_max_jitter", type=int, default=18)
    ap.add_argument("--span_probability", type=float, default=0.4)
    ap.add_argument("--span_min_pad", type=int, default=1)
    ap.add_argument("--span_max_pad", type=int, default=5)
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
                scribble_profile=str(args.scribble_profile),
                proposal_jitter=(
                    max(0, int(args.proposal_min_jitter)),
                    max(int(args.proposal_min_jitter), int(args.proposal_max_jitter)),
                ),
                span_probability=max(0.0, min(1.0, float(args.span_probability))),
                span_pad=(
                    max(0, int(args.span_min_pad)),
                    max(int(args.span_min_pad), int(args.span_max_pad)),
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
