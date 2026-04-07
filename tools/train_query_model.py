#!/usr/bin/env python3
"""Train the TrainableQueryUtilityModel from a correction history JSON file.

Usage:
    python tools/train_query_model.py --history corrections.json --output query_model.json

The history file should contain a JSON object with a "history" list of correction
summaries (as produced by CorrectionBuffer.commit / record_event).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.query_planner import QueryUtilityModel


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "accepted", "accept"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _summary_from_flat_row(row: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    query_type = str(
        row.get("query_type") or row.get("point_type") or row.get("kind") or ""
    ).strip()
    point_type = str(row.get("point_type") or "").strip()
    if query_type:
        meta["query_type"] = query_type
    if point_type:
        meta["point_type"] = point_type
    alias_fields = {
        "boundary_frame": "boundary_frame",
        "left_label": "left_label",
        "right_label": "right_label",
        "changed_frame_count": "changed_frame_count",
        "anchor_violations_before": "anchor_violations_before",
        "anchor_violations_after": "anchor_violations_after",
        "anchor_before": "anchor_violations_before",
        "anchor_after": "anchor_violations_after",
        "state_conflicts_before": "state_conflicts_before",
        "state_conflicts_after": "state_conflicts_after",
        "state_before": "state_conflicts_before",
        "state_after": "state_conflicts_after",
        "second_correction": "second_correction",
        "proposal_feedback": "proposal_feedback",
        "raw_confidence": "raw_confidence",
        "calibrated_confidence": "calibrated_confidence",
        "estimated_cost": "estimated_cost",
    }
    for key in (
        "boundary_frame",
        "left_label",
        "right_label",
        "changed_frame_count",
        "anchor_violations_before",
        "anchor_violations_after",
        "anchor_before",
        "anchor_after",
        "state_conflicts_before",
        "state_conflicts_after",
        "state_before",
        "state_after",
        "second_correction",
        "proposal_feedback",
        "raw_confidence",
        "calibrated_confidence",
        "estimated_cost",
    ):
        raw = row.get(key)
        if raw in (None, ""):
            continue
        target_key = alias_fields[key]
        if key in {"second_correction", "proposal_feedback"}:
            meta[target_key] = _as_bool(raw)
        elif key in {"raw_confidence", "calibrated_confidence", "estimated_cost"}:
            meta[target_key] = _as_float(raw)
        elif key in {"left_label", "right_label"}:
            meta[target_key] = str(raw)
        else:
            meta[target_key] = _as_int(raw)
    score_terms: Dict[str, float] = {}
    for key in (
        "uncertainty",
        "energy",
        "state_conflict",
        "disagreement",
        "multiview",
        "propagation_gain",
        "history",
    ):
        raw = row.get(key)
        if raw not in (None, ""):
            score_terms[key] = _as_float(raw)
    if score_terms:
        meta["score_terms"] = score_terms
    start_frame = _as_int(row.get("start_frame", row.get("feedback_start", 0)))
    end_frame = _as_int(row.get("end_frame", row.get("feedback_end", start_frame)))
    meta.setdefault("feedback_start", int(start_frame))
    meta.setdefault("feedback_end", int(end_frame))
    accepted = _as_bool(row.get("accepted"))
    if row.get("accepted") not in (None, ""):
        meta["accepted"] = bool(accepted)
    changed = _as_bool(row.get("changed"))
    summary: Dict[str, Any] = {
        "kind": str(row.get("kind") or row.get("event") or query_type or "event"),
        "meta": meta,
        "steps": _as_int(row.get("steps", 0)),
        "changed": bool(changed),
    }
    if accepted or changed:
        summary["committed_at"] = str(row.get("committed_at") or row.get("timestamp") or "")
    else:
        summary["discarded_at"] = str(row.get("discarded_at") or row.get("timestamp") or "")
    return summary


def _load_history(path: str) -> List[Dict[str, Any]]:
    in_path = Path(path)
    if in_path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                if isinstance(item, dict):
                    rows.append(item)
        return rows
    if in_path.suffix.lower() == ".csv":
        rows: List[Dict[str, Any]] = []
        with in_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                summary_blob = row.get("summary") or row.get("summary_json")
                if summary_blob:
                    try:
                        item = json.loads(summary_blob)
                        if isinstance(item, dict):
                            rows.append(item)
                            continue
                    except Exception:
                        pass
                rows.append(_summary_from_flat_row(row))
        return rows
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        rows = data.get("history") or data.get("corrections") or data.get("summaries") or []
        return [item for item in rows if isinstance(item, dict)]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Train query utility model from correction history.")
    ap.add_argument("--history", required=True, help="Correction history JSON file")
    ap.add_argument("--output", default="", help="Output JSON path for trained model snapshot")
    ap.add_argument("--lr", type=float, default=0.01, help="SGD learning rate")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = ap.parse_args()

    history = _load_history(args.history)
    if not history:
        print("[train_query_model] No correction summaries found.")
        return

    model = QueryUtilityModel()
    loaded = 0
    for summary in history:
        model.update_from_summary(summary)
        loaded += 1

    obs_count = len(getattr(model._linear_model, "_observations", []))
    print(f"[train_query_model] loaded {loaded} summaries, {obs_count} valid observations")
    if obs_count < 3:
        print("[train_query_model] Not enough observations to train (need >= 3).")
        return

    mse = model.fit(lr=float(args.lr), epochs=int(args.epochs))
    print(f"[train_query_model] trained, final MSE={mse:.6f}")
    print(f"[train_query_model] weights={model._linear_model.weights.tolist()}")
    print(f"[train_query_model] bias={model._linear_model.bias:.6f}")

    out_path = args.output or str(Path(args.history).with_suffix(".query_model.json"))
    snapshot = model.snapshot()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[train_query_model] saved to {out_path}")


if __name__ == "__main__":
    main()
