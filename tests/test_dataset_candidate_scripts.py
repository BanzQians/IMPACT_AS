from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_assembly101_candidates(tmp_path: Path) -> None:
    annotations_root = tmp_path / "assembly_coarse"
    output_dir = tmp_path / "assembly_out"
    rows = [
        {"recording_name": "recording_good", "start_frame": 0, "end_frame": 120, "coarse_label": "step_a"},
        {"recording_name": "recording_good", "start_frame": 120, "end_frame": 240, "coarse_label": "step_b"},
        {"recording_name": "recording_good", "start_frame": 240, "end_frame": 360, "coarse_label": "step_c"},
        {"recording_name": "recording_good", "start_frame": 360, "end_frame": 480, "coarse_label": "step_d"},
        {"recording_name": "recording_good", "start_frame": 480, "end_frame": 600, "coarse_label": "step_e"},
        {"recording_name": "recording_good", "start_frame": 600, "end_frame": 720, "coarse_label": "step_f"},
        {"recording_name": "recording_good", "start_frame": 720, "end_frame": 900, "coarse_label": "step_g"},
        {"recording_name": "recording_bad", "start_frame": 0, "end_frame": 450, "coarse_label": "long_a"},
        {"recording_name": "recording_bad", "start_frame": 450, "end_frame": 900, "coarse_label": "long_b"},
    ]
    _write_csv(annotations_root / "coarse.csv", rows)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_assembly101_candidates.py"),
            "--annotations-root",
            str(annotations_root),
            "--output-dir",
            str(output_dir),
            "--top-k-recordings",
            "1",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    candidate_file = output_dir / "assembly101_candidate_recordings.txt"
    assert candidate_file.exists()
    assert candidate_file.read_text(encoding="utf-8").strip() == "recording_good"


def test_build_epic_candidates(tmp_path: Path) -> None:
    annotations_root = tmp_path / "epic_ann"
    output_dir = tmp_path / "epic_out"
    rows = [
        {"video_id": "P01_01", "start_timestamp": "00:00:00.000", "stop_timestamp": "00:00:04.000", "verb": "take", "noun": "knife"},
        {"video_id": "P01_01", "start_timestamp": "00:00:04.000", "stop_timestamp": "00:00:08.000", "verb": "cut", "noun": "onion"},
        {"video_id": "P01_01", "start_timestamp": "00:00:08.000", "stop_timestamp": "00:00:12.000", "verb": "move", "noun": "onion"},
        {"video_id": "P01_01", "start_timestamp": "00:00:12.000", "stop_timestamp": "00:00:16.000", "verb": "take", "noun": "pan"},
        {"video_id": "P01_01", "start_timestamp": "00:00:16.000", "stop_timestamp": "00:00:20.000", "verb": "put", "noun": "onion"},
        {"video_id": "P01_01", "start_timestamp": "00:00:20.000", "stop_timestamp": "00:00:24.000", "verb": "stir", "noun": "onion"},
        {"video_id": "P01_01", "start_timestamp": "00:00:24.000", "stop_timestamp": "00:00:30.000", "verb": "serve", "noun": "dish"},
        {"video_id": "P02_01", "start_timestamp": "00:00:00.000", "stop_timestamp": "00:00:14.000", "verb": "walk", "noun": "kitchen"},
        {"video_id": "P02_01", "start_timestamp": "00:00:14.000", "stop_timestamp": "00:00:30.000", "verb": "look", "noun": "fridge"},
    ]
    _write_csv(annotations_root / "EPIC_100_train.csv", rows)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_epic_candidates.py"),
            "--annotations-root",
            str(annotations_root),
            "--output-dir",
            str(output_dir),
            "--top-k-videos",
            "1",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    candidate_file = output_dir / "epic_candidate_videos.txt"
    assert candidate_file.exists()
    assert candidate_file.read_text(encoding="utf-8").strip() == "P01_01"
