from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["timestamp", "event"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_user_study_sessions(tmp_path: Path) -> None:
    annotation_path = tmp_path / "clip01.json"
    sidecar_path = tmp_path / "clip01_scribble.json"
    ops_log_path = tmp_path / "clip01.json.ops.log.csv"
    output_prefix = tmp_path / "study_metrics"

    _write_json(
        annotation_path,
        {
            "video_id": "clip01",
            "view": "Top",
            "view_start": 0,
            "view_end": 99,
            "labels": [
                {"id": 0, "name": "null"},
                {"id": 1, "name": "retrieve"},
                {"id": 2, "name": "detach"},
            ],
            "segments": [
                {"action_label": 1, "start_frame": 0, "end_frame": 49},
                {"action_label": 2, "start_frame": 50, "end_frame": 79},
                {"action_label": 1, "start_frame": 80, "end_frame": 99},
            ],
        },
    )

    _write_json(
        sidecar_path,
        {
            "version": 1,
            "saved_at": "2026-04-09T10:00:10",
            "video_id": "clip01",
            "study_condition": "scribble_planner",
            "view": {"index": 0, "name": "Top", "start": 0, "end": 99},
            "confirmed_accept_records": [
                {
                    "schema_version": 2,
                    "point_type": "boundary",
                    "action_kind": "boundary_accept",
                    "feedback_start": 15,
                    "feedback_end": 15,
                    "boundary_frame": 15,
                    "left_label": "retrieve",
                    "right_label": "detach",
                }
            ],
            "correction_history": [
                {
                    "kind": "scribble_proposal_feedback",
                    "started_at": "2026-04-09T10:00:05",
                    "committed_at": "2026-04-09T10:00:05",
                    "steps": 0,
                    "changed": True,
                    "meta": {
                        "query_type": "boundary_scribble",
                        "point_type": "proposal_feedback",
                        "proposal_feedback": True,
                        "accepted": True,
                        "feedback_start": 10,
                        "feedback_end": 20,
                        "boundary_frame": 15,
                        "left_label": "retrieve",
                        "right_label": "detach",
                        "raw_confidence": 0.61,
                        "calibrated_confidence": 0.65,
                        "study_condition": "scribble_planner",
                    },
                },
                {
                    "kind": "scribble_accept",
                    "started_at": "2026-04-09T10:00:00",
                    "committed_at": "2026-04-09T10:00:06",
                    "steps": 2,
                    "changed": True,
                    "records": [],
                    "meta": {
                        "query_type": "boundary_scribble",
                        "point_type": "boundary_scribble_accept",
                        "feedback_start": 10,
                        "feedback_end": 20,
                        "boundary_frame": 15,
                        "left_label": "retrieve",
                        "right_label": "detach",
                        "second_correction": True,
                        "changed_frame_count": 11,
                        "study_condition": "scribble_planner",
                    },
                },
                {
                    "kind": "label_review_feedback",
                    "started_at": "2026-04-09T10:00:07",
                    "committed_at": "2026-04-09T10:00:07",
                    "steps": 0,
                    "changed": False,
                    "reason": "rejected",
                    "meta": {
                        "query_type": "label_review",
                        "point_type": "proposal_feedback",
                        "proposal_feedback": True,
                        "accepted": False,
                        "feedback_start": 30,
                        "feedback_end": 40,
                        "current_label": "retrieve",
                        "suggested_label": "detach",
                        "raw_confidence": 0.40,
                        "calibrated_confidence": 0.35,
                        "study_condition": "scribble_planner",
                    },
                },
                {
                    "kind": "label_edit",
                    "started_at": "2026-04-09T10:00:08",
                    "committed_at": "2026-04-09T10:00:09",
                    "steps": 1,
                    "changed": True,
                    "records": [],
                    "meta": {
                        "query_type": "label_review",
                        "point_type": "label_review_accept",
                        "feedback_start": 30,
                        "feedback_end": 40,
                        "changed_frame_count": 11,
                        "study_condition": "scribble_planner",
                    },
                },
            ],
        },
    )

    _write_csv(
        ops_log_path,
        [
            {"timestamp": "2026-04-09T10:00:00", "event": "validation_on"},
            {"timestamp": "2026-04-09T10:00:01", "event": "step_frame"},
            {"timestamp": "2026-04-09T10:00:02", "event": "label_select"},
            {"timestamp": "2026-04-09T10:00:03", "event": "scribble_accept"},
            {"timestamp": "2026-04-09T10:00:10", "event": "save_annotations"},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "summarize_user_study_sessions.py"),
            "--input",
            str(tmp_path),
            "--participant-id",
            "P01",
            "--output-prefix",
            str(output_prefix),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    summary_path = Path(str(output_prefix) + ".summary.json")
    sessions_path = Path(str(output_prefix) + ".sessions.csv")
    assert summary_path.exists()
    assert sessions_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["session_count"] == 1
    session = summary["sessions"][0]
    assert session["participant_id"] == "P01"
    assert session["study_condition"] == "scribble_planner"
    assert session["video_id"] == "clip01"
    assert session["interaction_count"] == 3
    assert session["interaction_count_source"] == "ops_log"
    assert session["accept_count"] == 1
    assert session["reject_count"] == 1
    assert session["boundary_feedback_count"] == 1
    assert session["label_feedback_count"] == 1
    assert session["second_correction_count"] == 1
    assert session["boundary_second_correction_count"] == 1
    assert session["final_segment_count"] == 3
    assert session["final_unique_label_count"] == 2
    assert session["scribble_accept_event_count"] == 1
    assert session["navigation_event_count"] == 1
    assert session["label_selection_count"] == 1

