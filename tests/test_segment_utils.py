from __future__ import annotations

from pathlib import Path

from tools.segment_utils import load_annotation_txt_segments


def test_load_annotation_txt_segments_triplets(tmp_path: Path) -> None:
    path = tmp_path / "segments.txt"
    path.write_text("0\n2\nalpha\n\n3\n5\nbeta\n", encoding="utf-8")

    segs, mode = load_annotation_txt_segments(str(path))

    assert mode == "segment_ranges"
    assert segs == [
        {"start": 0, "end": 2, "label": "alpha"},
        {"start": 3, "end": 5, "label": "beta"},
    ]


def test_load_annotation_txt_segments_framewise_labels(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.txt"
    path.write_text("alpha\nalpha\nbeta\nbeta\nbeta\ngamma\n", encoding="utf-8")

    segs, mode = load_annotation_txt_segments(str(path))

    assert mode == "framewise_labels"
    assert segs == [
        {"start": 0, "end": 1, "label": "alpha"},
        {"start": 2, "end": 4, "label": "beta"},
        {"start": 5, "end": 5, "label": "gamma"},
    ]
