from __future__ import annotations

from core.action_corrections import CorrectionBuffer
from core.models import LabelDef
from ui.action_window import ActionWindow
from utils.constants import EXTRA_LABEL_NAME


def _make_window() -> ActionWindow:
    win = ActionWindow.__new__(ActionWindow)
    win._study_condition = "scribble_planner"
    win.video_path = ""
    win.current_video_id = ""
    win.active_view_idx = 0
    win.views = [{"name": "Main"}]
    win._correction_buffer = CorrectionBuffer()
    win._study_planner_history_cache = {}
    win._root_dir = ""
    win._action_label_bank_source = ""
    win.labels = []
    win.current_label_idx = -1
    win.panel = None
    win.timeline = None
    win._rebuild_timeline_sources = lambda: None
    win._refresh_fine_label_decomposition = lambda refresh_panel=False: None
    win._psr_mark_dirty = lambda: None
    win._psr_update_component_panel = lambda *args, **kwargs: None
    win.combo_task = type(
        "_ComboStub",
        (),
        {"currentText": staticmethod(lambda: "Action Segmentation")},
    )()
    win._effective_view_name = lambda view, idx=None: str(view.get("name") or "view")
    win._rebuilt = 0
    win._rebuild_query_learning_models_from_history = lambda: setattr(
        win, "_rebuilt", int(getattr(win, "_rebuilt", 0)) + 1
    )
    return win


def _touch(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_label_map(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{name} {idx}" for name, idx in rows) + "\n",
        encoding="utf-8",
    )


def test_dataset_key_groups_videos_under_same_manifest(tmp_path) -> None:
    dataset_a = tmp_path / "dataset_a"
    dataset_b = tmp_path / "dataset_b"
    _touch(dataset_a / "manifest.csv")
    _touch(dataset_b / "manifest.csv")
    video_a1 = dataset_a / "videos" / "clip_01.mp4"
    video_a2 = dataset_a / "videos" / "clip_02.mp4"
    video_b1 = dataset_b / "videos" / "clip_03.mp4"
    _touch(video_a1)
    _touch(video_a2)
    _touch(video_b1)

    win = _make_window()
    key_a1 = win._study_planner_dataset_key_for_path(str(video_a1))
    key_a2 = win._study_planner_dataset_key_for_path(str(video_a2))
    key_b1 = win._study_planner_dataset_key_for_path(str(video_b1))

    assert key_a1 == key_a2
    assert key_a1 != key_b1


def test_study_planner_history_restores_only_for_same_dataset(tmp_path) -> None:
    dataset_a = tmp_path / "dataset_a"
    dataset_b = tmp_path / "dataset_b"
    _touch(dataset_a / "manifest.csv")
    _touch(dataset_b / "manifest.csv")
    video_a1 = dataset_a / "videos" / "clip_01.mp4"
    video_a2 = dataset_a / "videos" / "clip_02.mp4"
    video_b1 = dataset_b / "videos" / "clip_03.mp4"
    _touch(video_a1)
    _touch(video_a2)
    _touch(video_b1)

    win = _make_window()
    dataset_key = win._study_planner_dataset_key_for_path(str(video_a1))
    win.video_path = str(video_a1)
    win.current_video_id = "clip_01"
    win._correction_buffer.history = [
        {
            "kind": "boundary_accept",
            "meta": {
                "query_type": "boundary_scribble",
                "view_idx": 0,
                "video_id": "clip_01",
                "dataset_key": dataset_key,
            },
            "started_at": "2026-04-12T10:00:00Z",
            "committed_at": "2026-04-12T10:00:01Z",
            "steps": 1,
            "changed": True,
        }
    ]
    win._refresh_study_planner_history_cache()

    win._correction_buffer = CorrectionBuffer()
    win._restore_study_planner_history_for_path(str(video_a2))
    assert len(win._correction_buffer.history) == 1
    assert win._correction_buffer.history[0]["meta"]["video_id"] == "clip_01"
    assert win._rebuilt == 1

    win._correction_buffer = CorrectionBuffer()
    win._rebuilt = 0
    win._restore_study_planner_history_for_path(str(video_b1))
    assert win._correction_buffer.history == []
    assert win._rebuilt == 0


def test_query_summary_matches_active_view_ignores_other_video_history(tmp_path) -> None:
    dataset_a = tmp_path / "dataset_a"
    _touch(dataset_a / "manifest.csv")
    video_a1 = dataset_a / "videos" / "clip_01.mp4"
    video_a2 = dataset_a / "videos" / "clip_02.mp4"
    _touch(video_a1)
    _touch(video_a2)

    win = _make_window()
    dataset_key = win._study_planner_dataset_key_for_path(str(video_a2))
    win.video_path = str(video_a2)
    win.current_video_id = "clip_02"

    other_video_summary = {
        "meta": {
            "view_idx": 0,
            "view": "Main",
            "video_id": "clip_01",
            "dataset_key": dataset_key,
        }
    }
    same_video_summary = {
        "meta": {
            "view_idx": 0,
            "view": "Main",
            "video_id": "clip_02",
            "dataset_key": dataset_key,
        }
    }

    assert win._query_summary_matches_active_view(other_video_summary) is False
    assert win._query_summary_matches_active_view(same_video_summary) is True


def test_auto_switch_label_bank_reuses_previous_mapping_within_same_dataset(tmp_path) -> None:
    dataset_a = tmp_path / "dataset_a"
    _touch(dataset_a / "manifest.csv")
    video_a1 = dataset_a / "videos" / "clip_01.mp4"
    video_a2 = dataset_a / "videos" / "clip_02.mp4"
    _touch(video_a1)
    _touch(video_a2)

    win = _make_window()
    previous_mapping = [("pour", 0), ("stir", 1)]

    changed = win._auto_switch_label_bank_for_video(
        str(video_a2),
        previous_video_path=str(video_a1),
        previous_label_mapping=previous_mapping,
        previous_label_source_path="",
    )

    assert changed is True
    assert [lb.name for lb in win.labels] == ["pour", "stir"]
    assert [lb.id for lb in win.labels] == [0, 1]


def test_current_action_label_mapping_excludes_runtime_labels() -> None:
    win = _make_window()
    win.labels = [
        LabelDef(name="pour", color_name="Red", id=0),
        LabelDef(name="Unknown", color_name="Gray", id=8),
        LabelDef(name=EXTRA_LABEL_NAME, color_name="Pink", id=9),
        LabelDef(name="stir", color_name="Blue", id=1),
    ]

    assert win._current_action_label_mapping() == [("pour", 0), ("stir", 1)]


def test_current_label_bank_source_path_ignores_runtime_labels(tmp_path) -> None:
    label_map = tmp_path / "label.txt"
    _write_label_map(label_map, [("pour", 0), ("stir", 1)])

    win = _make_window()
    win._action_label_bank_source = str(label_map)
    win.labels = [
        LabelDef(name="pour", color_name="Red", id=0),
        LabelDef(name="stir", color_name="Blue", id=1),
        LabelDef(name="Unknown", color_name="Gray", id=8),
        LabelDef(name=EXTRA_LABEL_NAME, color_name="Pink", id=9),
    ]

    assert win._current_label_bank_source_path() == str(label_map)

    win.labels.append(LabelDef(name="mix", color_name="Green", id=2))
    assert win._current_label_bank_source_path() == ""


def test_auto_switch_label_bank_notifies_for_empty_explicit_mapping(tmp_path) -> None:
    dataset_a = tmp_path / "dataset_a"
    _touch(dataset_a / "manifest.csv")
    video_a1 = dataset_a / "videos" / "clip_01.mp4"
    _touch(video_a1)
    empty_label_map = dataset_a / "empty_labels.txt"
    empty_label_map.write_text("\n", encoding="utf-8")

    win = _make_window()
    issues = []
    win._notify_label_map_load_issue = (
        lambda path, *, title, reason="empty": issues.append((path, title, reason))
    )

    changed = win._auto_switch_label_bank_for_video(
        str(video_a1),
        explicit_label_map_path=str(empty_label_map),
    )

    assert changed is False
    assert issues == [(str(empty_label_map), "Open Session", "empty")]


def test_auto_switch_label_bank_loads_new_dataset_labels(tmp_path) -> None:
    dataset_a = tmp_path / "dataset_a"
    dataset_b = tmp_path / "dataset_b"
    _touch(dataset_a / "manifest.csv")
    _touch(dataset_b / "manifest.csv")
    video_a1 = dataset_a / "videos" / "clip_01.mp4"
    video_b1 = dataset_b / "videos" / "clip_03.mp4"
    _touch(video_a1)
    _touch(video_b1)
    _write_label_map(dataset_b / "label.txt", [("cut", 3), ("mix", 7)])

    win = _make_window()
    win.labels = [LabelDef(name="pour", color_name="Red", id=0)]

    changed = win._auto_switch_label_bank_for_video(
        str(video_b1),
        previous_video_path=str(video_a1),
        previous_label_mapping=[("pour", 0)],
        previous_label_source_path="",
    )

    assert changed is True
    assert [lb.name for lb in win.labels] == ["cut", "mix"]
    assert [lb.id for lb in win.labels] == [3, 7]
