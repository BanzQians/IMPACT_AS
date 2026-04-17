#!/usr/bin/env python3
"""Generate mock data for Phase 0 smoke test of eval_interactive_scribble.py.

Creates:
  data/gt/video_01.txt, video_02.txt, video_03.txt   — GT frame labels
  data/results/scribble_only/video_0X.json            — annotation JSONs
  data/results/scribble_only/video_0X_scribble.json   — sidecar JSONs
  data/results/scribble_planner/video_0X.json
  data/results/scribble_planner/video_0X_scribble.json
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# 1. Ground truth: 3 videos, each 200 frames, 4 action segments
# ---------------------------------------------------------------------------
GT_SPECS = {
    "video_01": [
        ("cut_tomato", 0, 49),
        ("pour_oil", 50, 99),
        ("stir_pot", 100, 149),
        ("serve_plate", 150, 199),
    ],
    "video_02": [
        ("wash_hands", 0, 39),
        ("chop_onion", 40, 109),
        ("fry_egg", 110, 159),
        ("clean_pan", 160, 199),
    ],
    "video_03": [
        ("open_fridge", 0, 29),
        ("take_butter", 30, 79),
        ("spread_bread", 80, 139),
        ("close_fridge", 140, 199),
    ],
}


def _write_gt(name: str, segments):
    gt_dir = DATA / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    labels = [""] * 200
    for label, s, e in segments:
        for f in range(s, e + 1):
            labels[f] = label
    (gt_dir / f"{name}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


def _gt_boundaries(segments):
    """Return GT boundary frame indices (start of each segment except the first)."""
    return [s for _, s, _ in segments[1:]]


# ---------------------------------------------------------------------------
# 2. Build annotation JSON (mimics tool output)
# ---------------------------------------------------------------------------
def _build_annotation(name: str, segments, *, offset_boundaries=None):
    """
    offset_boundaries: list of (boundary_frame_offset, label_left, label_right)
    to simulate imperfect annotation (boundaries shifted by a few frames).
    If None, use GT boundaries exactly.
    """
    label_set = list(dict.fromkeys(lbl for lbl, _, _ in segments))
    labels = [{"id": i, "name": lbl} for i, lbl in enumerate(label_set)]
    label_to_id = {lbl: i for i, lbl in enumerate(label_set)}

    if offset_boundaries is not None:
        # Build annotation segments from shifted boundaries
        ann_segments = []
        cuts = [0] + [b for b, _, _ in offset_boundaries] + [200]
        for i in range(len(cuts) - 1):
            s, e = cuts[i], cuts[i + 1] - 1
            # find which GT segment this overlaps most
            best_label = segments[min(i, len(segments) - 1)][0]
            ann_segments.append({
                "start_frame": s,
                "end_frame": e,
                "action_label": label_to_id.get(best_label, 0),
            })
    else:
        ann_segments = []
        for lbl, s, e in segments:
            ann_segments.append({
                "start_frame": s,
                "end_frame": e,
                "action_label": label_to_id[lbl],
            })

    return {
        "video_id": name,
        "view": "front",
        "view_start": 0,
        "view_end": 199,
        "labels": labels,
        "segments": ann_segments,
    }


# ---------------------------------------------------------------------------
# 3. Build sidecar JSON with correction_history
# ---------------------------------------------------------------------------
def _build_sidecar(name: str, gt_segments, *, condition: str):
    """
    Build a realistic correction_history.
    For scribble_only: user does boundary scribbles without planner proposals.
    For scribble_planner: user gets planner proposals with accept/reject feedback.
    """
    gt_bounds = _gt_boundaries(gt_segments)
    history = []
    t0 = datetime(2026, 4, 10, 10, 0, 0)

    for idx, bf in enumerate(gt_bounds):
        started = t0 + timedelta(seconds=idx * 25)
        committed = started + timedelta(seconds=8 + idx * 3)

        meta = {
            "query_type": "boundary_scribble",
            "point_type": "boundary_accept",
            "boundary_frame": bf,
            "left_label": gt_segments[idx][0],
            "right_label": gt_segments[idx + 1][0],
            "feedback_start": max(0, bf - 20),
            "feedback_end": min(199, bf + 20),
            "accepted": True,
            "changed_frame_count": 5 + idx * 2,
            "anchor_violations_before": 0,
            "anchor_violations_after": 0,
            "state_conflicts_before": 0,
            "state_conflicts_after": 0,
            "second_correction": False,
        }

        summary = {
            "kind": "boundary_scribble",
            "steps": 2 + idx,
            "changed": True,
            "started_at": started.isoformat(),
            "committed_at": committed.isoformat(),
            "meta": meta,
        }
        history.append(summary)

        # For planner condition, add proposal feedback events
        if condition == "scribble_planner":
            # Simulate: first boundary proposal accepted, others have a
            # reject-then-accept pattern
            fb_started = committed + timedelta(seconds=2)
            fb_committed = fb_started + timedelta(seconds=3)
            if idx == 0:
                # Accept proposal
                fb_meta = {
                    "query_type": "boundary_scribble",
                    "point_type": "boundary_accept",
                    "boundary_frame": bf,
                    "proposal_feedback": True,
                    "accepted": True,
                    "raw_confidence": 0.85,
                    "calibrated_confidence": 0.82,
                }
            else:
                # Reject proposal (user overrides)
                fb_meta = {
                    "query_type": "boundary_scribble",
                    "point_type": "boundary_accept",
                    "boundary_frame": bf + 3,  # slightly off
                    "proposal_feedback": True,
                    "accepted": False,
                    "raw_confidence": 0.65,
                    "calibrated_confidence": 0.60,
                }
            fb_summary = {
                "kind": "boundary_scribble",
                "steps": 0,
                "changed": False,
                "started_at": fb_started.isoformat(),
                "committed_at": fb_committed.isoformat(),
                "meta": fb_meta,
            }
            history.append(fb_summary)

    sidecar = {
        "video_id": name,
        "condition": condition,
        "correction_history": history,
    }
    return sidecar


# ---------------------------------------------------------------------------
# 4. Write everything
# ---------------------------------------------------------------------------
def main():
    for name, gt_segs in GT_SPECS.items():
        _write_gt(name, gt_segs)

    for condition in ("scribble_only", "scribble_planner"):
        out_dir = DATA / "results" / condition
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, gt_segs in GT_SPECS.items():
            # Annotation: use slightly shifted boundaries to make F1 < 1.0
            gt_bounds = _gt_boundaries(gt_segs)
            shifted = []
            for i, bf in enumerate(gt_bounds):
                offset = (-2 if i % 2 == 0 else 3)
                shifted.append((bf + offset, gt_segs[i][0], gt_segs[i + 1][0]))

            ann = _build_annotation(name, gt_segs, offset_boundaries=shifted)
            ann_path = out_dir / f"{name}.json"
            ann_path.write_text(
                json.dumps(ann, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            sidecar = _build_sidecar(name, gt_segs, condition=condition)
            sidecar_path = out_dir / f"{name}_scribble.json"
            sidecar_path.write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # Summary
    print("Mock data generated:")
    for d in sorted(DATA.rglob("*")):
        if d.is_file():
            print(f"  {d.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
