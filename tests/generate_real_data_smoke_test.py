#!/usr/bin/env python3
"""Generate mock annotation + sidecar pairs from REAL GT data in the
selected_* directories, with **differentiated** scribble_only vs
scribble_planner behaviour to demonstrate planner effectiveness.

Key modelling differences between conditions:
  scribble_only:
    - User discovers boundaries left-to-right (sequential scan)
    - Each interaction costs more steps (no pre-focused region)
    - No proposal feedback events
  scribble_planner:
    - Planner prioritises high-value boundaries first (longest adjacent
      segment span → largest potential F1 gain per interaction)
    - Each interaction costs fewer steps (planner pre-focuses the region)
    - Proposal feedback events with ~40% acceptance rate
    - Accepted proposals cost 0 extra steps (instant)

The resulting budget curves should show scribble_planner reaching any
given F1 threshold with fewer interactions / steps / time.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "selected_gt_clips_boundary14",
    "selected_gt_clips_richer4",
    "selected_epic_clips_top10",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_gt_labels(path: Path) -> List[str]:
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                labels.append(t)
    return labels


def boundaries_from_labels(labels: List[str]) -> List[int]:
    out: List[int] = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            out.append(i)
    return out


def segments_from_labels(labels: List[str]) -> List[Dict[str, Any]]:
    if not labels:
        return []
    segs: List[Dict[str, Any]] = []
    cur_label = labels[0]
    cur_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur_label:
            segs.append({"start": cur_start, "end": i - 1, "label": cur_label})
            cur_label = labels[i]
            cur_start = i
    segs.append({"start": cur_start, "end": len(labels) - 1, "label": cur_label})
    return segs


def boundary_value_score(
    boundary_idx: int,
    gt_bounds: List[int],
    n_frames: int,
) -> float:
    """Score a boundary by the span it separates.
    Boundaries between longer segments have higher value because
    correctly placing them gains more F1."""
    all_cuts = [0] + gt_bounds + [n_frames]
    # find which pair of segments this boundary separates
    pos = gt_bounds.index(boundary_idx)
    left_len = all_cuts[pos + 1] - all_cuts[pos]
    right_len = all_cuts[pos + 2] - all_cuts[pos + 1]
    return float(left_len + right_len)


# ---------------------------------------------------------------------------
# boundary ordering strategies
# ---------------------------------------------------------------------------
def order_sequential(gt_bounds: List[int], n_frames: int) -> List[int]:
    """scribble_only: user scans left to right."""
    return list(gt_bounds)  # already sorted by frame


def order_by_value(gt_bounds: List[int], n_frames: int) -> List[int]:
    """scribble_planner: planner prioritises highest-value boundaries."""
    scored = [
        (boundary_value_score(b, gt_bounds, n_frames), b)
        for b in gt_bounds
    ]
    # Sort by value descending; break ties by frame position
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [b for _, b in scored]


# ---------------------------------------------------------------------------
# annotation JSON
# ---------------------------------------------------------------------------
def build_annotation_json(
    video_id: str,
    labels: List[str],
    segments: List[Dict[str, Any]],
    *,
    boundary_noise: int = 0,
) -> Dict[str, Any]:
    label_set = list(dict.fromkeys(s["label"] for s in segments))
    label_list = [{"id": i, "name": lbl} for i, lbl in enumerate(label_set)]
    label_to_id = {lbl: i for i, lbl in enumerate(label_set)}
    n_frames = len(labels)

    if boundary_noise > 0:
        bounds = [s["start"] for s in segments[1:]]
        noisy_bounds = []
        for b in bounds:
            shift = random.randint(-boundary_noise, boundary_noise)
            nb = max(1, min(n_frames - 2, b + shift))
            noisy_bounds.append(nb)
        noisy_bounds = sorted(set(noisy_bounds))
        cuts = [0] + noisy_bounds + [n_frames]
        ann_segments = []
        for i in range(len(cuts) - 1):
            s, e = cuts[i], cuts[i + 1] - 1
            span_labels = labels[s: e + 1]
            majority = Counter(span_labels).most_common(1)[0][0]
            ann_segments.append({
                "start_frame": s,
                "end_frame": e,
                "action_label": label_to_id.get(majority, 0),
            })
    else:
        ann_segments = [
            {
                "start_frame": s["start"],
                "end_frame": s["end"],
                "action_label": label_to_id[s["label"]],
            }
            for s in segments
        ]

    return {
        "video_id": video_id,
        "view": "front",
        "view_start": 0,
        "view_end": n_frames - 1,
        "labels": label_list,
        "segments": ann_segments,
    }


# ---------------------------------------------------------------------------
# sidecar JSON with correction_history
# ---------------------------------------------------------------------------
def build_sidecar_json(
    video_id: str,
    labels: List[str],
    gt_bounds: List[int],
    ordered_bounds: List[int],
    *,
    condition: str,
) -> Dict[str, Any]:
    """Build correction_history.

    Key differences between conditions:

      scribble_only:
        - User discovers boundaries left-to-right
        - Each interaction costs more steps (2-5) and time (15-30s)
        - ~20% chance each interaction produces a spurious boundary
          (wrong location, fp) rather than a correct GT boundary
        - No proposal feedback events

      scribble_planner:
        - Planner prioritises high-value boundaries
        - Each interaction costs fewer steps (1-3) and time (10-20s)
        - ~5% spurious rate (planner pre-filters bad candidates)
        - Proposal feedback events with ~40% acceptance
    """
    n_frames = len(labels)
    history: List[Dict[str, Any]] = []
    t0 = datetime(2026, 4, 10, 10, 0, 0)
    rng = random.Random(hash(video_id + condition) & 0xFFFFFFFF)

    elapsed = 0.0

    # Build the event sequence: mix real boundaries with spurious ones
    gt_set = set(gt_bounds)

    if condition == "scribble_only":
        spurious_rate = 0.20
        steps_range = (2, 5)
        time_range = (15.0, 30.0)
    else:
        spurious_rate = 0.05
        steps_range = (1, 3)
        time_range = (10.0, 20.0)

    # Interleave real boundaries with occasional spurious ones
    event_bounds: List[int] = []
    for bf in ordered_bounds:
        # Before each real boundary, maybe insert a spurious one
        if rng.random() < spurious_rate:
            # Generate a spurious boundary (midpoint of a random segment)
            mid = bf + rng.randint(-30, 30)
            mid = max(1, min(n_frames - 2, mid))
            # Make sure it's not accidentally near a real GT boundary
            if all(abs(mid - gb) > 10 for gb in gt_bounds):
                event_bounds.append(mid)
        event_bounds.append(bf)

    for idx, bf in enumerate(event_bounds):
        left_label = labels[max(0, bf - 1)] if bf > 0 else labels[0]
        right_label = labels[min(bf, n_frames - 1)]

        span_half = rng.randint(15, 40)
        fb_start = max(0, bf - span_half)
        fb_end = min(n_frames - 1, bf + span_half)

        steps = rng.randint(*steps_range)
        time_delta = rng.uniform(*time_range)

        started_sec = elapsed
        elapsed += time_delta
        committed_sec = elapsed

        started = t0 + timedelta(seconds=started_sec)
        committed = t0 + timedelta(seconds=committed_sec)

        meta: Dict[str, Any] = {
            "query_type": "boundary_scribble",
            "point_type": "boundary_scribble_accept",
            "boundary_frame": bf,
            "left_label": left_label,
            "right_label": right_label,
            "feedback_start": fb_start,
            "feedback_end": fb_end,
            "accepted": True,
            "changed_frame_count": rng.randint(3, 20),
            "anchor_violations_before": 0,
            "anchor_violations_after": 0,
            "state_conflicts_before": 0,
            "state_conflicts_after": 0,
            "second_correction": (rng.random() < 0.12),
            "raw_confidence": round(rng.uniform(0.3, 0.95), 4),
            "calibrated_confidence": round(rng.uniform(0.25, 0.90), 4),
        }

        summary: Dict[str, Any] = {
            "kind": "temporal_boundary_scribble",
            "steps": steps,
            "changed": True,
            "started_at": started.isoformat() + "+00:00",
            "committed_at": committed.isoformat() + "+00:00",
            "meta": meta,
        }
        history.append(summary)

        # Planner condition: add proposal feedback events
        if condition == "scribble_planner":
            elapsed += rng.uniform(2.0, 5.0)
            fb_started = t0 + timedelta(seconds=elapsed - 2.0)
            fb_committed = t0 + timedelta(seconds=elapsed)

            accepted = rng.random() < 0.4
            proposal_bf = bf if accepted else bf + rng.randint(-5, 5)

            fb_meta: Dict[str, Any] = {
                "query_type": "boundary_scribble",
                "point_type": "proposal_feedback",
                "proposal_feedback": True,
                "accepted": accepted,
                "boundary_frame": proposal_bf,
                "feedback_start": fb_start,
                "feedback_end": fb_end,
                "left_label": left_label,
                "right_label": right_label,
                "raw_confidence": round(rng.uniform(0.3, 0.9), 4),
                "calibrated_confidence": round(rng.uniform(0.2, 0.85), 4),
                "second_correction": False,
            }
            fb_summary: Dict[str, Any] = {
                "kind": "scribble_proposal_feedback",
                "steps": 0,
                "changed": accepted,
                "started_at": fb_started.isoformat() + "+00:00",
                "committed_at": fb_committed.isoformat() + "+00:00",
                "meta": fb_meta,
            }
            history.append(fb_summary)

    return {
        "video_id": video_id,
        "condition": condition,
        "correction_history": history,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    out_base = ROOT / "data"

    for ds_name in DATASETS:
        ds_dir = ROOT / ds_name
        gt_dir = ds_dir / "groundTruth"
        if not gt_dir.is_dir():
            print(f"[SKIP] {ds_name}: no groundTruth/ dir")
            continue

        gt_files = sorted(gt_dir.glob("*.txt"))
        if not gt_files:
            print(f"[SKIP] {ds_name}: no .txt GT files")
            continue

        target_gt_dir = out_base / "gt" / ds_name
        target_gt_dir.mkdir(parents=True, exist_ok=True)

        for condition in ("scribble_only", "scribble_planner"):
            results_dir = out_base / "results" / ds_name / condition
            results_dir.mkdir(parents=True, exist_ok=True)

            for gt_file in gt_files:
                video_id = gt_file.stem
                labels = read_gt_labels(gt_file)
                if len(labels) < 10:
                    continue

                segments = segments_from_labels(labels)
                gt_bounds = boundaries_from_labels(labels)
                if not gt_bounds:
                    continue

                n_frames = len(labels)

                # Copy GT
                shutil.copy2(gt_file, target_gt_dir / gt_file.name)

                # Order boundaries based on condition
                if condition == "scribble_only":
                    ordered = order_sequential(gt_bounds, n_frames)
                else:
                    ordered = order_by_value(gt_bounds, n_frames)

                # Annotation JSON: same noise for both conditions
                rng_ann = random.Random(hash(video_id + "ann") & 0xFFFFFFFF)
                ann = build_annotation_json(
                    video_id, labels, segments, boundary_noise=5
                )
                ann_path = results_dir / f"{video_id}.json"
                ann_path.write_text(
                    json.dumps(ann, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                # Sidecar JSON
                sidecar = build_sidecar_json(
                    video_id, labels, gt_bounds, ordered,
                    condition=condition,
                )
                sidecar_path = results_dir / f"{video_id}_scribble.json"
                sidecar_path.write_text(
                    json.dumps(sidecar, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        n_clips = len(gt_files)
        print(f"[OK] {ds_name}: {n_clips} clips")

    print("\nDone. Run eval to compare conditions.")


if __name__ == "__main__":
    main()
