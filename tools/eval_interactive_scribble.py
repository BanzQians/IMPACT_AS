#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from impact_scribe_io import (
    CasePaths,
    collect_case_paths,
    extract_accepted_boundary_events,
    extract_boundary_feedback_events,
    extract_correction_events,
    load_annotation_bundle,
    load_gt_boundaries,
    load_sidecar_payload,
)

from core.query_planner import QueryCandidate, QueryCostModel, QueryType


def _parse_deltas(text: str) -> List[int]:
    rows: List[int] = []
    for part in str(text or "").split(","):
        chunk = part.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except Exception:
            continue
        if value > 0:
            rows.append(value)
    return rows or [5, 10, 20]


def _mean(values: Sequence[float]) -> Optional[float]:
    rows = [float(v) for v in values if v is not None]
    if not rows:
        return None
    return float(sum(rows) / len(rows))


def _resolve_gt_path(
    case: CasePaths,
    *,
    gt_path: Optional[str],
    gt_dir: Optional[str],
) -> Optional[Path]:
    if gt_path:
        path = Path(gt_path).expanduser().resolve()
        return path if path.is_file() else None
    if not gt_dir:
        return None
    root = Path(gt_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    base_stem = ""
    if case.annotation_path is not None:
        base_stem = case.annotation_path.stem
    elif case.sidecar_path is not None:
        name = case.sidecar_path.name
        if name.lower().endswith("_scribble.json"):
            base_stem = name[: -len("_scribble.json")]
        else:
            base_stem = case.sidecar_path.stem
    if not base_stem:
        return None
    for suffix in (".json", ".txt", ".npy"):
        candidate = root / f"{base_stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _match_boundary_lists(
    pred: Sequence[int],
    gt: Sequence[int],
    delta: int,
) -> Tuple[int, int, int]:
    pred_sorted = sorted({int(x) for x in pred})
    gt_sorted = sorted({int(x) for x in gt})
    used_gt = set()
    tp = 0
    for frame in pred_sorted:
        left = bisect.bisect_left(gt_sorted, int(frame) - int(delta))
        right = bisect.bisect_right(gt_sorted, int(frame) + int(delta))
        best_idx = None
        best_dist = None
        for idx in range(int(left), int(right)):
            if idx in used_gt:
                continue
            dist = abs(int(gt_sorted[idx]) - int(frame))
            if best_idx is None or dist < int(best_dist):
                best_idx = idx
                best_dist = dist
        if best_idx is None:
            continue
        used_gt.add(int(best_idx))
        tp += 1
    fp = max(0, len(pred_sorted) - tp)
    fn = max(0, len(gt_sorted) - tp)
    return int(tp), int(fp), int(fn)


def _boundary_f1(pred: Sequence[int], gt: Sequence[int], delta: int) -> Dict[str, Optional[float]]:
    tp, fp, fn = _match_boundary_lists(pred, gt, delta)
    precision = (float(tp) / float(tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (float(tp) / float(tp + fn)) if (tp + fn) > 0 else 0.0
    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _evaluate_case(
    case: CasePaths,
    *,
    deltas: Sequence[int],
    gt_path: Optional[Path],
) -> Dict[str, Any]:
    sidecar_payload = load_sidecar_payload(case.sidecar_path)
    history = list(sidecar_payload.get("correction_history") or [])
    feedback_events = extract_boundary_feedback_events(history)
    boundary_events = extract_accepted_boundary_events(history)
    correction_events = [
        row
        for row in extract_correction_events(history, include_feedback=False)
        if bool(row.get("accepted"))
    ]

    bundle = load_annotation_bundle(case.annotation_path) if case.annotation_path else None
    gt_boundaries: Optional[List[int]] = None
    gt_meta: Optional[Dict[str, Any]] = None
    if gt_path is not None and gt_path.is_file():
        gt_boundaries, gt_meta = load_gt_boundaries(gt_path)

    boundary_curve: List[Dict[str, Any]] = []
    seen_boundaries = set()
    ordered_boundaries: List[int] = []
    for index, event in enumerate(boundary_events, start=1):
        boundary_frame = event.get("boundary_frame")
        if boundary_frame is None:
            boundary_frame = int(
                (int(event.get("start_frame", 0)) + int(event.get("end_frame", 0))) * 0.5
            )
        frame_i = int(boundary_frame)
        if frame_i not in seen_boundaries:
            seen_boundaries.add(frame_i)
            ordered_boundaries.append(frame_i)
        row: Dict[str, Any] = {
            "interaction_count": int(index),
            "boundary_count": int(len(ordered_boundaries)),
        }
        if gt_boundaries is not None:
            for delta in deltas:
                stats = _boundary_f1(ordered_boundaries, gt_boundaries, int(delta))
                row[f"precision@{delta}"] = stats["precision"]
                row[f"recall@{delta}"] = stats["recall"]
                row[f"f1@{delta}"] = stats["f1"]
        boundary_curve.append(row)

    final_annotation_stats: Dict[str, Any] = {}
    final_accepted_stats: Dict[str, Any] = {}
    if gt_boundaries is not None:
        final_pred = ordered_boundaries
        final_ann = list(bundle.get("boundaries_abs", [])) if bundle else []
        for delta in deltas:
            final_accepted_stats[f"f1@{delta}"] = _boundary_f1(
                final_pred, gt_boundaries, int(delta)
            )["f1"]
            final_annotation_stats[f"f1@{delta}"] = _boundary_f1(
                final_ann, gt_boundaries, int(delta)
            )["f1"]

    accepted_feedback = sum(1 for row in feedback_events if bool(row.get("accepted")))
    rejected_feedback = sum(1 for row in feedback_events if not bool(row.get("accepted")))
    total_feedback = int(len(feedback_events))

    cost_model = QueryCostModel()
    estimated_costs: List[float] = []
    actual_steps: List[float] = []
    query_type_counts: Dict[str, int] = {}
    for row in extract_correction_events(history, include_feedback=False):
        qtype_text = str(row.get("query_type") or "").strip().lower()
        if not qtype_text:
            continue
        try:
            qtype = QueryType(qtype_text)
        except Exception:
            continue
        candidate = QueryCandidate(
            query_id=f"history:{len(estimated_costs)}",
            query_type=qtype,
            start_frame=int(row.get("start_frame", 0) or 0),
            end_frame=int(row.get("end_frame", row.get("start_frame", 0)) or 0),
        )
        predicted_cost = float(cost_model.predict(candidate))
        estimated_costs.append(predicted_cost)
        actual_steps.append(float(int(row.get("steps", 0) or 0)))
        query_type_counts[qtype.value] = int(query_type_counts.get(qtype.value, 0) + 1)
        summary = row.get("summary")
        if isinstance(summary, dict):
            cost_model.update_from_summary(summary)

    second_correction_rate = None
    if correction_events:
        second_correction_rate = float(
            sum(1 for row in correction_events if bool(row.get("second_correction")))
            / float(len(correction_events))
        )

    cost_gap_mean = None
    if estimated_costs and actual_steps:
        gap_values = [
            abs(float(est) - float(step))
            for est, step in zip(estimated_costs, actual_steps)
        ]
        cost_gap_mean = _mean(gap_values)

    return {
        "annotation_path": str(case.annotation_path) if case.annotation_path else "",
        "sidecar_path": str(case.sidecar_path) if case.sidecar_path else "",
        "gt_path": str(gt_path) if gt_path else "",
        "video_id": str(bundle.get("video_id", "")) if bundle else "",
        "view": str(bundle.get("view", "")) if bundle else "",
        "gt_meta": dict(gt_meta or {}),
        "history_count": int(len(history)),
        "accepted_boundary_count": int(len(boundary_events)),
        "proposal_feedback_count": int(total_feedback),
        "proposal_accept_count": int(accepted_feedback),
        "proposal_reject_count": int(rejected_feedback),
        "acceptance_rate": (
            float(accepted_feedback / float(total_feedback))
            if total_feedback > 0
            else None
        ),
        "override_rate": (
            float(rejected_feedback / float(total_feedback))
            if total_feedback > 0
            else None
        ),
        "second_correction_rate": second_correction_rate,
        "avg_estimated_cost": _mean(estimated_costs),
        "avg_actual_steps": _mean(actual_steps),
        "avg_cost_abs_gap": cost_gap_mean,
        "query_type_counts": dict(sorted(query_type_counts.items())),
        "boundary_curve": list(boundary_curve),
        "final_accepted_boundary_metrics": dict(final_accepted_stats),
        "final_annotation_boundary_metrics": dict(final_annotation_stats),
    }


def _aggregate_curve(cases: Sequence[Dict[str, Any]], deltas: Sequence[int]) -> List[Dict[str, Any]]:
    max_len = 0
    for case in cases:
        max_len = max(max_len, len(case.get("boundary_curve") or []))
    rows: List[Dict[str, Any]] = []
    for index in range(max_len):
        samples = [
            case.get("boundary_curve", [])[index]
            for case in cases
            if index < len(case.get("boundary_curve") or [])
        ]
        if not samples:
            continue
        row: Dict[str, Any] = {
            "interaction_count": int(index + 1),
            "case_count": int(len(samples)),
        }
        for delta in deltas:
            row[f"f1@{delta}"] = _mean(
                [
                    float(item.get(f"f1@{delta}", 0.0) or 0.0)
                    for item in samples
                    if item.get(f"f1@{delta}") is not None
                ]
            )
        rows.append(row)
    return rows


def _build_summary(cases: Sequence[Dict[str, Any]], deltas: Sequence[int]) -> Dict[str, Any]:
    feedback_total = sum(int(case.get("proposal_feedback_count", 0) or 0) for case in cases)
    accept_total = sum(int(case.get("proposal_accept_count", 0) or 0) for case in cases)
    reject_total = sum(int(case.get("proposal_reject_count", 0) or 0) for case in cases)
    summary: Dict[str, Any] = {
        "case_count": int(len(cases)),
        "proposal_feedback_count": int(feedback_total),
        "proposal_accept_count": int(accept_total),
        "proposal_reject_count": int(reject_total),
        "acceptance_rate": (
            float(accept_total / float(feedback_total)) if feedback_total > 0 else None
        ),
        "override_rate": (
            float(reject_total / float(feedback_total)) if feedback_total > 0 else None
        ),
        "avg_second_correction_rate": _mean(
            [case.get("second_correction_rate") for case in cases]
        ),
        "avg_estimated_cost": _mean([case.get("avg_estimated_cost") for case in cases]),
        "avg_actual_steps": _mean([case.get("avg_actual_steps") for case in cases]),
        "avg_cost_abs_gap": _mean([case.get("avg_cost_abs_gap") for case in cases]),
        "max_interactions": max(
            [int(case.get("accepted_boundary_count", 0) or 0) for case in cases] or [0]
        ),
    }
    for delta in deltas:
        summary[f"final_accepted_boundary_f1@{delta}"] = _mean(
            [
                (case.get("final_accepted_boundary_metrics") or {}).get(f"f1@{delta}")
                for case in cases
            ]
        )
        summary[f"final_annotation_boundary_f1@{delta}"] = _mean(
            [
                (case.get("final_annotation_boundary_metrics") or {}).get(f"f1@{delta}")
                for case in cases
            ]
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate IMPACT-Scribe correction histories under interaction budgets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Annotation json, `_scribble.json`, or a directory containing them.",
    )
    parser.add_argument(
        "--gt",
        default="",
        help="Single GT path for a single-case evaluation.",
    )
    parser.add_argument(
        "--gt_dir",
        default="",
        help="Directory with GT files matched by basename (.json/.txt/.npy).",
    )
    parser.add_argument(
        "--deltas",
        default="5,10,20",
        help="Comma-separated boundary tolerances.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output JSON report path.",
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
    if args.gt and len(cases) != 1:
        raise SystemExit("--gt can only be used when exactly one case is evaluated.")

    deltas = _parse_deltas(args.deltas)
    reports: List[Dict[str, Any]] = []
    for case in cases:
        gt_path = _resolve_gt_path(case, gt_path=args.gt, gt_dir=args.gt_dir)
        if gt_path is None:
            print(
                f"[WARN] No GT found for "
                f"{case.annotation_path or case.sidecar_path}; boundary F1 will be omitted."
            )
        reports.append(_evaluate_case(case, deltas=deltas, gt_path=gt_path))

    curve = _aggregate_curve(reports, deltas)
    summary = _build_summary(reports, deltas)
    payload = {
        "schema_version": 1,
        "deltas": list(deltas),
        "summary": dict(summary),
        "macro_boundary_curve": list(curve),
        "cases": list(reports),
    }

    print(
        "[EVAL] cases={case_count} accept_rate={acceptance_rate} "
        "override_rate={override_rate} avg_steps={avg_actual_steps}".format(
            case_count=summary.get("case_count"),
            acceptance_rate=(
                "n/a"
                if summary.get("acceptance_rate") is None
                else f"{float(summary['acceptance_rate']):.3f}"
            ),
            override_rate=(
                "n/a"
                if summary.get("override_rate") is None
                else f"{float(summary['override_rate']):.3f}"
            ),
            avg_actual_steps=(
                "n/a"
                if summary.get("avg_actual_steps") is None
                else f"{float(summary['avg_actual_steps']):.3f}"
            ),
        )
    )
    for delta in deltas:
        key = f"final_accepted_boundary_f1@{delta}"
        value = summary.get(key)
        text = "n/a" if value is None else f"{float(value):.3f}"
        print(f"[EVAL] {key}={text}")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[EVAL] wrote {out_path}")


if __name__ == "__main__":
    main()
