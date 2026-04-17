from __future__ import annotations

from core.models import AnnotationStore, LabelDef
from core.temporal_scribble import ScribbleKind, TemporalScribble, TemporalScribbleSet
from ui.action_window import ActionWindow


def _build_window(store: AnnotationStore, cuts: set[int]) -> ActionWindow:
    win = ActionWindow.__new__(ActionWindow)
    win.views = [
        {
            "confirmed_accept_records": [
                {
                    "point_type": "boundary",
                    "action_kind": "boundary_accept",
                    "boundary_frame": 5,
                }
            ]
        }
    ]
    win.active_view_idx = 0
    win._merge_spans = ActionWindow._merge_spans
    win._get_frame_count = lambda: 16
    win._syncable_descriptor = lambda descriptor: True
    win._store_for_view_descriptor = lambda view, descriptor: store
    win._trim_cut_set_for_view = lambda view, descriptor, create=False: cuts
    win._replace_cut_set_excluding_positions = (
        lambda cut_set, start, end, protected_positions=None, new_cuts=None: (
            ActionWindow._replace_cut_set_excluding_positions(
                win,
                cut_set,
                start,
                end,
                protected_positions=protected_positions,
                new_cuts=new_cuts,
            )
        )
    )
    return win


def test_apply_decoded_labels_to_store_preserves_confirmed_boundary_cut() -> None:
    store = AnnotationStore()
    for frame in range(0, 5):
        store.add("alpha", frame)
    for frame in range(5, 10):
        store.add("beta", frame)
    cuts = {5}
    win = _build_window(store, cuts)

    decoded = {frame: "alpha" for frame in range(0, 10)}
    win._apply_decoded_labels_to_store(
        store,
        win.views[0],
        {"kind": "store"},
        decoded,
        [(5, 9)],
    )

    assert cuts == {5}
    assert all(store.label_at(frame) == "alpha" for frame in range(0, 10))


def test_apply_decoded_labels_to_store_preserves_existing_cuts_when_requested() -> None:
    store = AnnotationStore()
    for frame in range(0, 5):
        store.add("alpha", frame)
    for frame in range(5, 7):
        store.add("beta", frame)
    for frame in range(7, 10):
        store.add("gamma", frame)
    cuts = {5, 7}
    win = _build_window(store, cuts)

    decoded = {frame: "alpha" for frame in range(0, 10)}
    win._apply_decoded_labels_to_store(
        store,
        win.views[0],
        {"kind": "store"},
        decoded,
        [(5, 9)],
        preserve_existing_cuts=True,
    )

    assert cuts == {5, 7}
    assert all(store.label_at(frame) == "alpha" for frame in range(0, 10))


def test_cleanup_trim_cuts_keeps_confirmed_boundary_by_default() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {5}
    win = _build_window(store, cuts)

    ops = win._cleanup_trim_cuts_for_spans(
        0,
        {"kind": "store"},
        [(0, 9)],
    )

    assert ops == []
    assert cuts == {5}


def test_cleanup_trim_cuts_does_not_auto_merge_same_label_segments_anymore() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {5}
    win = _build_window(store, cuts)
    win.views[0]["confirmed_accept_records"] = []

    ops = win._cleanup_trim_cuts_for_spans(
        0,
        {"kind": "store"},
        [(0, 9)],
    )

    assert ops == []
    assert cuts == {5}


def test_cleanup_trim_cuts_can_explicitly_override_confirmed_boundary_lock() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {5}
    win = _build_window(store, cuts)

    ops = win._cleanup_trim_cuts_for_spans(
        0,
        {"kind": "store"},
        [(0, 9)],
        respect_confirmed_boundary_locks=False,
        allow_same_label_auto_merge=True,
    )

    assert ops == [
        {
            "view_idx": 0,
            "descriptor": {"kind": "store"},
            "frame": 5,
            "op": "remove",
        }
    ]
    assert cuts == set()


def test_remove_trim_cuts_in_spans_still_supports_explicit_merge() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {3, 5, 7}
    win = _build_window(store, cuts)
    win.views[0]["confirmed_accept_records"] = []

    ops = win._remove_trim_cuts_in_spans(
        0,
        {"kind": "store"},
        [(2, 8)],
        protected_cuts=[3, 7],
    )

    assert ops == [
        {
            "view_idx": 0,
            "descriptor": {"kind": "store"},
            "frame": 5,
            "op": "remove",
        }
    ]
    assert cuts == {3, 7}


def test_confirmed_boundary_positions_include_natural_boundary_without_cut() -> None:
    store = AnnotationStore()
    for frame in range(0, 5):
        store.add("left", frame)
    for frame in range(5, 10):
        store.add("right", frame)
    win = _build_window(store, set())

    protected = win._confirmed_accept_boundary_cut_positions(
        view_idx=0,
        descriptor={"kind": "store"},
    )

    assert protected == {5}


def test_existing_natural_boundary_does_not_count_as_new_split_cut() -> None:
    assert (
        ActionWindow._boundary_would_add_new_cut(
            start_frame=5,
            end_frame=12,
            boundary_frame=5,
            cuts=set(),
        )
        is False
    )
    assert (
        ActionWindow._boundary_would_add_new_cut(
            start_frame=5,
            end_frame=12,
            boundary_frame=8,
            cuts=set(),
        )
        is True
    )


def test_natural_boundary_with_matching_labels_is_already_satisfied() -> None:
    store = AnnotationStore()
    for frame in range(0, 5):
        store.add("left", frame)
    for frame in range(5, 10):
        store.add("right", frame)
    win = ActionWindow.__new__(ActionWindow)
    win._get_frame_count = lambda: 16
    win._resolve_action_label_name = lambda name: str(name or "").strip()

    satisfied = win._proposal_boundary_already_satisfied(
        store,
        5,
        set(),
        {
            "left_label": "left",
            "right_label": "right",
        },
    )

    assert satisfied is True


def test_remove_trim_cuts_in_spans_respects_confirmed_boundary_locks() -> None:
    """Confirmed boundary at frame 5 must survive _remove_trim_cuts_in_spans."""
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {3, 5, 7}
    win = _build_window(store, cuts)
    # Frame 5 is confirmed via the fixture — should be protected

    ops = win._remove_trim_cuts_in_spans(
        0,
        {"kind": "store"},
        [(2, 8)],
        protected_cuts=[3],
    )

    # Frame 3 is explicitly protected, frame 5 is confirmed-lock protected
    # Only frame 7 should be removed
    assert ops == [
        {
            "view_idx": 0,
            "descriptor": {"kind": "store"},
            "frame": 7,
            "op": "remove",
        }
    ]
    assert cuts == {3, 5}


def test_coalesce_same_label_internal_cuts_in_span_removes_locked_cut_for_explicit_merge() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {5}
    win = _build_window(store, cuts)

    ops = win._coalesce_same_label_internal_cuts_in_span(
        0,
        {"kind": "store"},
        0,
        9,
        preferred_label="alpha",
        respect_confirmed_boundary_locks=False,
    )

    assert ops == [
        {
            "view_idx": 0,
            "descriptor": {"kind": "store"},
            "frame": 5,
            "op": "remove",
        }
    ]
    assert cuts == set()


def test_coalesce_same_label_internal_cuts_in_span_keeps_nonmatching_boundary() -> None:
    store = AnnotationStore()
    for frame in range(0, 5):
        store.add("alpha", frame)
    for frame in range(5, 10):
        store.add("beta", frame)
    for frame in range(10, 15):
        store.add("alpha", frame)
    cuts = {5, 10}
    win = _build_window(store, cuts)
    win.views[0]["confirmed_accept_records"] = []

    ops = win._coalesce_same_label_internal_cuts_in_span(
        0,
        {"kind": "store"},
        0,
        14,
        preferred_label="alpha",
        respect_confirmed_boundary_locks=False,
    )

    assert ops == []
    assert cuts == {5, 10}


def test_confirmed_boundary_locks_include_segment_endpoints() -> None:
    """Segment endpoints stored in locked_segment_start/end should be protected."""
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {3, 5, 8}
    win = _build_window(store, cuts)
    # Add segment endpoint locks to the confirmed boundary at frame 5
    win.views[0]["confirmed_accept_records"] = [
        {
            "point_type": "boundary",
            "action_kind": "boundary_accept",
            "boundary_frame": 5,
            "locked_segment_start": 3,
            "locked_segment_end": 8,
        }
    ]

    protected = win._confirmed_accept_boundary_cut_positions(
        view_idx=0,
        descriptor={"kind": "store"},
    )

    # All three should be protected: boundary + segment start + segment end
    assert protected == {3, 5, 8}


def test_label_accept_protects_segment_boundaries() -> None:
    """When a label is accepted for a segment, its boundaries should be locked."""
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {3, 7}
    win = _build_window(store, cuts)
    win.views[0]["confirmed_accept_records"] = [
        {
            "point_type": "label",
            "action_kind": "label_accept",
            "feedback_start": 3,
            "feedback_end": 6,
        }
    ]

    protected = win._confirmed_accept_boundary_cut_positions(
        view_idx=0,
        descriptor={"kind": "store"},
    )

    # Start boundary (3) and end+1 boundary (7) should be protected
    assert protected == {3, 7}


def test_explicit_user_override_bypasses_confirmed_lock() -> None:
    """When respect_confirmed_boundary_locks=False, locked boundaries can be removed."""
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {3, 5, 7}
    win = _build_window(store, cuts)

    ops = win._remove_trim_cuts_in_spans(
        0,
        {"kind": "store"},
        [(2, 8)],
        protected_cuts=[3],
        respect_confirmed_boundary_locks=False,
    )

    removed_frames = {o["frame"] for o in ops}
    assert removed_frames == {5, 7}
    assert cuts == {3}


def test_clear_boundary_lock_records_removes_targeted_boundaries() -> None:
    """_clear_boundary_lock_records_in_span removes boundary records in span."""
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    win = _build_window(store, set())
    win.views[0]["confirmed_accept_records"] = [
        {"point_type": "boundary", "action_kind": "boundary_accept", "boundary_frame": 3},
        {"point_type": "boundary", "action_kind": "boundary_accept", "boundary_frame": 5},
        {"point_type": "boundary", "action_kind": "boundary_accept", "boundary_frame": 8},
        {
            "point_type": "label",
            "action_kind": "label_accept",
            "feedback_start": 0,
            "feedback_end": 2,
            "label": "x",
        },
    ]

    removed = win._clear_boundary_lock_records_in_span(2, 6, keep={3})

    # Only boundary_frame=5 removed (3 is in keep, 8 is out of range)
    assert removed == 1
    remaining_boundaries = [
        r["boundary_frame"]
        for r in win.views[0]["confirmed_accept_records"]
        if r.get("point_type") == "boundary"
    ]
    assert sorted(remaining_boundaries) == [3, 8]
    # Label record untouched
    label_recs = [
        r for r in win.views[0]["confirmed_accept_records"] if r.get("point_type") == "label"
    ]
    assert len(label_recs) == 1


def test_cleared_lock_no_longer_in_protected_set() -> None:
    """After clearing a lock record, the boundary is no longer protected."""
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    win = _build_window(store, set())
    win.views[0]["confirmed_accept_records"] = [
        {"point_type": "boundary", "action_kind": "boundary_accept", "boundary_frame": 3},
        {"point_type": "boundary", "action_kind": "boundary_accept", "boundary_frame": 5},
        {"point_type": "boundary", "action_kind": "boundary_accept", "boundary_frame": 8},
    ]

    win._clear_boundary_lock_records_in_span(4, 6)

    protected = win._confirmed_accept_boundary_cut_positions(
        view_idx=0, descriptor={"kind": "store"}
    )
    assert 5 not in protected
    assert 3 in protected
    assert 8 in protected


def test_on_action_segment_trim_merge_clears_locked_boundary_record() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {5}
    win = _build_window(store, cuts)
    win._is_psr_task = lambda: False
    win._mask_descriptor_from_row = lambda row: {"kind": "store"}
    win._begin_correction_session = lambda *args, **kwargs: None
    win._multiview_sync_active = lambda: False
    win._push_undo_item = lambda item: None
    win._redo_stack = []
    win._effective_view_name = lambda view, idx=0: f"view-{idx}"
    win._rebuild_timeline_sources = lambda: None
    win._log = lambda *args, **kwargs: None
    win._note_correction_step = lambda *args, **kwargs: None
    win._commit_correction_session = lambda **kwargs: {}

    result = win._on_action_segment_trim(5, object())

    assert result is True
    assert cuts == set()
    assert win.views[0]["confirmed_accept_records"] == []


def test_timeline_marker_activation_reopens_boundary_outside_scribble_mode() -> None:
    win = ActionWindow.__new__(ActionWindow)
    win.views = [{"name": "Top"}]
    win.active_view_idx = 0
    win._scribble_items = TemporalScribbleSet(
        items=[
            TemporalScribble(
                start_frame=4,
                end_frame=6,
                kind=ScribbleKind.UNCERTAIN,
                view_id="Top",
                meta={
                    "stroke_id": "marker-1",
                    "session_id": "sess-1",
                    "accepted": True,
                    "boundary_frame": 5,
                    "proposal_action": "refine_boundary",
                    "left_label": "alpha",
                    "right_label": "beta",
                    "confidence": 0.49,
                },
            )
        ]
    )
    win._active_scribble_session_id = ""
    win._active_scribble_session_items = lambda: []
    win._run_scribble_local_refiner = lambda items: None
    win._resolve_scribble_proposal_with_fallback = (
        lambda result, items, session_id, reason: {
            "boundary_frame": 5,
            "left_label": "alpha",
            "right_label": "beta",
            "confidence": 0.49,
            "session_id": str(session_id),
        }
    )
    win._store_active_view_scribble_state = lambda: None
    win._sync_timeline_scribble_items = lambda: None
    win._update_scribble_proposal_ui = lambda: None
    win._after_manual_query_relevant_edit = lambda: None
    win._log = lambda *args, **kwargs: None
    win._is_psr_task = lambda: False
    win.interaction_mode = "manual"
    win.extra_mode = False
    entered: list[str] = []
    statuses: list[str] = []
    interactions: list[str] = []

    def _enter_scribble_mode() -> None:
        entered.append("entered")
        win.interaction_mode = "scribble"

    win.enter_scribble_mode = _enter_scribble_mode
    win._set_status = lambda text: statuses.append(str(text))
    win._set_interaction_status = lambda text: interactions.append(str(text))

    win._on_timeline_scribble_activated(
        {
            "start_frame": 4,
            "end_frame": 6,
            "kind": ScribbleKind.UNCERTAIN.value,
            "meta": {"stroke_id": "marker-1", "session_id": "sess-1"},
        }
    )

    assert entered == ["entered"]
    assert win.interaction_mode == "scribble"
    assert win._active_scribble_session_id == "sess-1"
    assert isinstance(win._last_scribble_result, dict)
    assert win._last_scribble_result["boundary_frame"] == 5
    assert win._last_scribble_result["proposal_action"] == "refine_boundary"
    assert win._last_scribble_result["reopened_from_marker"] is True
    assert statuses and "Reopened boundary" in statuses[-1]
    assert interactions and "reopened for edit" in interactions[-1]


def test_reopened_marker_boundary_adjust_keeps_dragged_frame() -> None:
    win = ActionWindow.__new__(ActionWindow)
    win._scribble_interaction_active = lambda: True
    win._is_psr_task = lambda: False
    win._last_scribble_result = {
        "session_id": "sess-1",
        "proposal_action": "refine_boundary",
        "boundary_frame": 5,
        "window_start": 4,
        "window_end": 6,
        "left_label": "alpha",
        "right_label": "beta",
        "reopened_from_marker": True,
    }
    win._store_active_view_scribble_state = lambda: None
    win._update_scribble_proposal_ui = lambda: None
    win._log = lambda *args, **kwargs: None
    win._set_status = lambda text: None
    win._set_interaction_status = lambda text: None

    def _unexpected_annotate(*args, **kwargs):
        raise AssertionError("reopened marker drag should not re-annotate proposal")

    win._annotate_scribble_proposal = _unexpected_annotate

    win._on_timeline_scribble_proposal_adjusted(
        {"boundary_frame": 6, "finalized": True}
    )

    assert win._last_scribble_result["boundary_frame"] == 6
    assert win._last_scribble_result["proposal_action"] == "refine_boundary"
    assert "target_boundary_frame" not in win._last_scribble_result


def test_reject_boundary_suggestion_directly_applies_merge() -> None:
    win = ActionWindow.__new__(ActionWindow)
    accepted = {"called": False}
    feedback = []
    win._pending_label_review = None
    win._last_scribble_result = {
        "boundary_frame": 12,
        "query_hint": True,
        "left_label": "left",
        "right_label": "right",
    }
    converted = {
        "boundary_frame": 12,
        "proposal_action": "remove_boundary",
        "merged_label": "left",
        "query_hint": True,
    }
    win._planner_boundary_reject_to_merge_proposal = lambda proposal: dict(converted)
    win._record_scribble_proposal_feedback = (
        lambda proposal, accepted, reason, changed: feedback.append(
            {
                "proposal": dict(proposal),
                "accepted": bool(accepted),
                "reason": str(reason),
                "changed": bool(changed),
            }
        )
    )
    win._store_active_view_scribble_state = lambda: None
    win._accept_last_scribble_proposal = (
        lambda: accepted.__setitem__("called", True) or True
    )

    result = win._reject_current_interaction_suggestion()

    assert result is True
    assert accepted["called"] is True
    assert win._last_scribble_result == converted
    assert feedback == [
        {
            "proposal": {
                "boundary_frame": 12,
                "query_hint": True,
                "left_label": "left",
                "right_label": "right",
            },
            "accepted": False,
            "reason": "rejected_to_merge",
            "changed": False,
        }
    ]


def test_patch_frame_labels_for_boundary_accept_only_changes_local_span() -> None:
    before = {frame: "retrieve_adapter_plate" for frame in range(10, 21)}
    before.update({frame: "null" for frame in range(0, 10)})
    before.update({frame: "null" for frame in range(21, 31)})

    patched = ActionWindow._patch_frame_labels_for_boundary_accept(
        before,
        start=10,
        end=20,
        boundary_frame=15,
        left_label="null",
        right_label="retrieve_adapter_plate",
    )

    assert all(patched[frame] == "null" for frame in range(0, 15))
    assert all(patched[frame] == "retrieve_adapter_plate" for frame in range(15, 21))
    assert all(patched[frame] == "null" for frame in range(21, 31))


def test_changed_label_spans_in_range_stays_within_boundary_patch_window() -> None:
    before = {frame: "retrieve_adapter_plate" for frame in range(10, 21)}
    after = dict(before)
    for frame in range(10, 15):
        after[frame] = "null"
    for frame in range(15, 21):
        after[frame] = "retrieve_adapter_plate"

    spans = ActionWindow._changed_label_spans_in_range(
        before,
        after,
        start=10,
        end=20,
    )

    assert spans == [(10, 14)]


def test_remove_boundary_postprocessing_does_not_reintroduce_removed_cut() -> None:
    store = AnnotationStore()
    for frame in range(0, 2):
        store.add("x", frame)
    for frame in range(2, 5):
        store.add("alpha", frame)
    for frame in range(5, 8):
        store.add("beta", frame)
    for frame in range(8, 10):
        store.add("y", frame)
    cuts = {2, 5, 8}
    win = _build_window(store, cuts)
    win.views[0]["confirmed_accept_records"] = []

    cuts.discard(5)
    ops = win._ensure_trim_cuts_for_span_bounds(
        0,
        {"kind": "store"},
        cuts,
        2,
        7,
    )

    assert ops == []
    assert cuts == {2, 8}


def test_apply_label_to_segment_merges_same_label_split_back_into_one() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {5}
    win = _build_window(store, cuts)
    rebuilds: list[str] = []
    clears: list[int] = []
    store_changed_calls: list[bool] = []

    class _Timeline:
        _active_combined_row = object()

    win.timeline = _Timeline()
    win.mode = "Coarse"
    win._mask_descriptor_from_row = lambda row: {"kind": "store"}
    win._begin_correction_session = lambda *args, **kwargs: None
    win._target_entity_stores = lambda label_name: []
    win._push_undo_batch = lambda batches, meta=None: None
    win._push_undo_entry = lambda st, ds: None
    win._push_undo_item = lambda item: None
    win._push_undo_composite = lambda batches, trim_ops, meta=None: None
    win._redo_stack = []
    win._on_store_changed = lambda *args, **kwargs: store_changed_calls.append(True)
    win._note_correction_step = lambda *args, **kwargs: None
    win._commit_correction_session = lambda **kwargs: {}
    win._segment_embedding_for_span = lambda start, end: None
    win._update_label_prototype = lambda label, emb: None
    win._rebuild_timeline_sources = lambda: rebuilds.append("rebuilt")
    win._clear_boundary_lock_records_in_span = (
        lambda start, end, keep=None: clears.append(int(start)) or 0
    )

    result = win._apply_label_to_segment(0, 9, "alpha")

    assert result is True
    assert cuts == set()
    assert clears == [5]
    assert rebuilds == ["rebuilt"]
    assert store_changed_calls == [True]


def test_on_label_selected_uses_action_merge_path_for_selected_segment() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    cuts = {5}
    win = _build_window(store, cuts)
    calls: list[tuple[int, int, str]] = []
    timeline_calls: list[str] = []

    class _Timeline:
        def __init__(self) -> None:
            self._active_combined_row = type("_Row", (), {"_group_meta": {}})()

        def apply_combined_label(self, name: str) -> bool:
            timeline_calls.append(str(name))
            return True

        def flash_label(self, name: str) -> None:
            return None

    class _EntitiesPanel:
        mode = "applicability"

        def set_current_label(self, name, checked) -> None:
            return None

    win.timeline = _Timeline()
    win.entities_panel = _EntitiesPanel()
    win.labels = [LabelDef(name="alpha", color_name="c", id=0)]
    win.label_entity_map = {"alpha": set()}
    win._forced_segment = None
    win._pending_boundary_label_override = None
    win._pending_label_review = None
    win._timeline_selected_segment = {"start": 0, "end": 9, "label": "alpha"}
    win._apply_label_to_segment = (
        lambda start, end, label_name: calls.append((int(start), int(end), str(label_name))) or True
    )
    win._log = lambda *args, **kwargs: None
    win._topk_enabled = lambda: False
    win.mode = "Coarse"
    win._active_entity_name = None
    win.current_label_idx = -1

    win._on_label_selected(0)

    assert calls == [(0, 9, "alpha")]
    assert timeline_calls == []


def test_boundary_context_near_frame_respects_confirmed_keep_boundary_without_cut() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    for frame in range(10, 15):
        store.add("beta", frame)
    cuts = {10}
    win = _build_window(store, cuts)

    ctx = win._boundary_context_near_frame(
        {"kind": "store"},
        frame_i=10,
        search_radius=0,
    )

    assert ctx is not None
    assert ctx["boundary_frame"] == 10
    assert ctx["left_segment"] == {"start": 5, "end": 9, "label": "alpha"}
    assert ctx["right_segment"] == {"start": 10, "end": 14, "label": "beta"}
    assert ctx["has_trim_cut"] is True
    assert ctx["preserved_boundaries"] == [5]


def test_boundary_removal_span_context_respects_confirmed_keep_boundary_without_cut() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    for frame in range(10, 15):
        store.add("beta", frame)
    cuts = {10}
    win = _build_window(store, cuts)

    ctx = win._boundary_removal_span_context(
        {"kind": "store"},
        start_frame=8,
        end_frame=12,
    )

    assert ctx is not None
    assert ctx["start"] == 5
    assert ctx["end"] == 14
    assert ctx["boundary_frames"] == [10]
    assert ctx["boundary_count"] == 1


def test_remove_boundary_only_affects_target_boundary_when_neighbor_is_confirmed() -> None:
    store = AnnotationStore()
    for frame in range(0, 10):
        store.add("alpha", frame)
    for frame in range(10, 15):
        store.add("beta", frame)
    cuts = {10}
    win = _build_window(store, cuts)

    ctx = win._remove_boundary_context_for_proposal(
        {"kind": "store"},
        {
            "boundary_frame": 10,
            "target_boundary_frame": 10,
            "proposal_action": "remove_boundary",
        },
    )

    assert ctx is not None
    merge_s = int(ctx["left_segment"]["start"])
    merge_e = int(ctx["right_segment"]["end"])
    assert (merge_s, merge_e) == (5, 14)
    assert ctx["preserved_boundaries"] == [5]

    cuts.discard(10)
    _ = win._apply_label_range(store, merge_s, merge_e, "alpha")
    ops = win._ensure_trim_cuts_for_span_bounds(
        0,
        {"kind": "store"},
        cuts,
        merge_s,
        merge_e,
    )
    segs = win._segments_for_correction_store(
        store,
        start=0,
        end=14,
        cut_frames=cuts,
    )

    assert 10 not in cuts
    assert 5 in cuts
    assert any(op["frame"] == 5 and op["op"] == "add" for op in ops)
    assert [
        (seg["start"], seg["end"], seg["label"])
        for seg in segs
    ] == [
        (0, 4, "alpha"),
        (5, 14, "alpha"),
    ]


def test_remove_boundary_preserves_unconfirmed_left_neighbor_boundary() -> None:
    store = AnnotationStore()
    for frame in range(0, 5):
        store.add("alpha", frame)
    for frame in range(5, 10):
        store.add("beta", frame)
    for frame in range(10, 15):
        store.add("alpha", frame)
    cuts: set[int] = set()
    win = _build_window(store, cuts)
    win.views[0]["confirmed_accept_records"] = []

    ctx = win._remove_boundary_context_for_proposal(
        {"kind": "store"},
        {
            "boundary_frame": 10,
            "target_boundary_frame": 10,
            "proposal_action": "remove_boundary",
        },
    )

    assert ctx is not None
    assert ctx["left_segment"] == {"start": 5, "end": 9, "label": "beta"}
    assert ctx["right_segment"] == {"start": 10, "end": 14, "label": "alpha"}
    assert ctx["preserved_boundaries"] == [5]

    merge_s = int(ctx["left_segment"]["start"])
    merge_e = int(ctx["right_segment"]["end"])
    target_boundary = int(ctx["boundary_frame"])
    frozen_boundaries = {
        int(cut)
        for cut in (ctx.get("preserved_boundaries") or [])
        if cut is not None and int(cut) != int(target_boundary)
    }

    cuts.discard(target_boundary)
    _ = win._apply_label_range(store, merge_s, merge_e, "alpha")
    ops = []
    for cut in sorted(frozen_boundaries):
        if cut in cuts:
            continue
        cuts.add(cut)
        ops.append(
            {
                "view_idx": 0,
                "descriptor": {"kind": "store"},
                "frame": int(cut),
                "op": "add",
            }
        )
    segs = win._segments_for_correction_store(
        store,
        start=0,
        end=14,
        cut_frames=cuts,
    )

    assert ops == [
        {
            "view_idx": 0,
            "descriptor": {"kind": "store"},
            "frame": 5,
            "op": "add",
        }
    ]
    assert cuts == {5}
    assert [
        (seg["start"], seg["end"], seg["label"])
        for seg in segs
    ] == [
        (0, 4, "alpha"),
        (5, 14, "alpha"),
    ]
