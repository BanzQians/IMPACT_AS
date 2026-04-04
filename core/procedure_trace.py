import bisect
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return " ".join(text.replace("_", " ").split())


def _record_signature(record: Dict[str, Any]) -> Tuple[str, str, str, str, int]:
    return (
        _norm_text(record.get("label")),
        _norm_text(record.get("entity")),
        _norm_text(record.get("phase")),
        str(record.get("component_id")),
        int(record.get("state", 0)),
    )


def _sort_key(record: Dict[str, Any]) -> Tuple[int, int, str, str]:
    return (
        int(record.get("start", 0)),
        int(record.get("end", 0)),
        str(record.get("component_id")),
        str(record.get("label", "")),
    )


def _new_episode(record: Dict[str, Any], order_pos: int) -> Dict[str, Any]:
    start = int(record.get("start", 0))
    end = int(record.get("end", start))
    fragment = {
        "start_frame": start,
        "end_frame": end,
        "label": record.get("label"),
    }
    return {
        "signature": _record_signature(record),
        "label": record.get("label"),
        "entity": record.get("entity"),
        "phase": record.get("phase"),
        "component_id": record.get("component_id"),
        "component_name": record.get("component_name"),
        "state": int(record.get("state", 0)),
        "start_frame": start,
        "end_frame": end,
        "anchor_frame": end,
        "first_fragment_end": end,
        "fragments": [fragment],
        "resume_gaps": [],
        "interrupted": False,
        "last_order_pos": int(order_pos),
        "last_record": dict(record),
    }


def _has_blocking_transition(
    ordered: Sequence[Dict[str, Any]],
    from_pos: int,
    to_pos: int,
    component_id: Any,
    state: int,
) -> bool:
    comp_key = str(component_id)
    target_state = int(state)
    for idx in range(int(from_pos) + 1, int(to_pos)):
        other = ordered[idx]
        if str(other.get("component_id")) != comp_key:
            continue
        try:
            other_state = int(other.get("state", 0))
        except Exception:
            other_state = 0
        if other_state != target_state:
            return True
    return False


def build_trace_from_records(
    records: Sequence[Dict[str, Any]],
    *,
    resume_gap_frames: int = 0,
    block_on_opposite_state: bool = True,
) -> Dict[str, Any]:
    ordered: List[Dict[str, Any]] = []
    for idx, raw in enumerate(records or []):
        if not isinstance(raw, dict):
            continue
        try:
            start = int(raw.get("start", 0))
            end = int(raw.get("end", start))
        except Exception:
            continue
        if end < start:
            start, end = end, start
        component_id = raw.get("component_id")
        if component_id is None:
            continue
        try:
            state = int(raw.get("state", 0))
        except Exception:
            state = 0
        item = dict(raw)
        item["start"] = start
        item["end"] = end
        item["state"] = state
        item["_record_idx"] = int(idx)
        ordered.append(item)
    ordered.sort(key=_sort_key)
    if not ordered:
        return {
            "episodes": [],
            "events": [],
            "summary": {
                "episode_count": 0,
                "interrupted_count": 0,
                "multi_fragment_count": 0,
                "deferred_count": 0,
                "max_resume_gap_frames": 0,
            },
            "resume_gap_frames": int(max(0, resume_gap_frames)),
        }

    resume_gap_frames = max(0, int(resume_gap_frames))
    open_by_signature: Dict[Tuple[str, str, str, str, int], Dict[str, Any]] = {}
    episodes: List[Dict[str, Any]] = []

    def close_episode(signature: Tuple[str, str, str, str, int]) -> None:
        episode = open_by_signature.pop(signature, None)
        if episode is None:
            return
        episode["fragment_count"] = int(len(episode.get("fragments") or []))
        episode["deferred_effect"] = bool(
            int(episode.get("anchor_frame", 0))
            > int(episode.get("first_fragment_end", 0))
        )
        episodes.append(episode)

    for order_pos, record in enumerate(ordered):
        signature = _record_signature(record)
        episode = open_by_signature.get(signature)
        if episode is None:
            open_by_signature[signature] = _new_episode(record, order_pos)
            continue
        prev_order_pos = int(episode.get("last_order_pos", order_pos))
        gap = max(0, int(record.get("start", 0)) - int(episode.get("end_frame", 0)) - 1)
        blocked = False
        if block_on_opposite_state:
            blocked = _has_blocking_transition(
                ordered,
                prev_order_pos,
                int(order_pos),
                episode.get("component_id"),
                int(episode.get("state", 0)),
            )
        if gap > resume_gap_frames or blocked:
            close_episode(signature)
            open_by_signature[signature] = _new_episode(record, order_pos)
            continue
        episode["end_frame"] = int(record.get("end", episode.get("end_frame", 0)))
        episode["anchor_frame"] = int(record.get("end", episode.get("anchor_frame", 0)))
        episode["last_order_pos"] = int(order_pos)
        episode["last_record"] = dict(record)
        if gap > 0:
            episode["resume_gaps"].append(int(gap))
        if gap > 0 or int(order_pos) > prev_order_pos + 1:
            episode["interrupted"] = True
        episode["fragments"].append(
            {
                "start_frame": int(record.get("start", 0)),
                "end_frame": int(record.get("end", 0)),
                "label": record.get("label"),
            }
        )
    for signature in sorted(
        list(open_by_signature.keys()),
        key=lambda key: (
            int(open_by_signature[key].get("start_frame", 0)),
            int(open_by_signature[key].get("end_frame", 0)),
            str(open_by_signature[key].get("component_id")),
        ),
    ):
        close_episode(signature)
    episodes.sort(
        key=lambda item: (
            int(item.get("start_frame", 0)),
            int(item.get("end_frame", 0)),
            str(item.get("component_id")),
        )
    )

    events: List[Dict[str, Any]] = []
    max_gap = 0
    for episode_id, episode in enumerate(episodes, start=1):
        last_record = dict(episode.get("last_record") or {})
        event = {
            "frame": int(episode.get("anchor_frame", 0)),
            "label": last_record.get("label"),
            "component_id": episode.get("component_id"),
            "component_name": episode.get("component_name"),
            "state": int(episode.get("state", 0)),
            "trace_episode_id": int(episode_id),
            "trace_anchor_kind": "completion",
            "trace_episode_start": int(episode.get("start_frame", 0)),
            "trace_episode_end": int(episode.get("end_frame", 0)),
            "trace_fragment_count": int(episode.get("fragment_count", 1)),
            "trace_interrupted": bool(episode.get("interrupted")),
            "trace_deferred": bool(episode.get("deferred_effect")),
            "trace_resume_gaps": list(episode.get("resume_gaps") or []),
        }
        events.append(event)
        for gap in episode.get("resume_gaps") or []:
            try:
                max_gap = max(max_gap, int(gap))
            except Exception:
                continue

    summary = {
        "episode_count": int(len(episodes)),
        "interrupted_count": int(sum(1 for item in episodes if item.get("interrupted"))),
        "multi_fragment_count": int(
            sum(1 for item in episodes if int(item.get("fragment_count", 1)) > 1)
        ),
        "deferred_count": int(sum(1 for item in episodes if item.get("deferred_effect"))),
        "max_resume_gap_frames": int(max_gap),
    }
    return {
        "episodes": episodes,
        "events": events,
        "summary": summary,
        "resume_gap_frames": int(resume_gap_frames),
    }


def _initial_state_vector(
    components: Sequence[Dict[str, Any]], initial_state: Optional[Sequence[Any]]
) -> List[int]:
    if isinstance(initial_state, list) and len(initial_state) == len(components):
        out = []
        for raw in initial_state:
            try:
                val = int(raw)
            except Exception:
                val = 0
            if val not in (-1, 0, 1):
                val = 0
            out.append(val)
        return out
    return [0] * len(components)


def _state_at_frame(
    state_sequence: Sequence[Dict[str, Any]],
    state_frames: Sequence[int],
    initial_state: Sequence[int],
    frame: int,
    component_idx: int,
) -> int:
    if component_idx < 0:
        return 0
    idx = bisect.bisect_right(list(state_frames), int(frame)) - 1
    if idx < 0:
        return int(initial_state[component_idx]) if component_idx < len(initial_state) else 0
    try:
        state_vec = list(state_sequence[idx].get("state") or [])
    except Exception:
        state_vec = []
    if component_idx < len(state_vec):
        try:
            return int(state_vec[component_idx])
        except Exception:
            return 0
    return int(initial_state[component_idx]) if component_idx < len(initial_state) else 0


def _next_component_change_frame(
    state_sequence: Sequence[Dict[str, Any]],
    start_frame: int,
    component_idx: int,
) -> Optional[int]:
    for item in state_sequence or []:
        try:
            frame = int(item.get("frame", 0))
        except Exception:
            continue
        if frame <= int(start_frame):
            continue
        state_vec = list(item.get("state") or [])
        if component_idx < len(state_vec):
            return int(frame)
    return None


def analyze_trace_conflicts(
    episodes: Sequence[Dict[str, Any]],
    state_sequence: Sequence[Dict[str, Any]],
    components: Sequence[Dict[str, Any]],
    *,
    initial_state: Optional[Sequence[Any]] = None,
    view_end: Optional[int] = None,
) -> List[Dict[str, Any]]:
    component_idx_by_id = {
        str(comp.get("id")): idx for idx, comp in enumerate(components or [])
    }
    init_vec = _initial_state_vector(components, initial_state)
    state_frames = []
    for item in state_sequence or []:
        try:
            state_frames.append(int(item.get("frame", 0)))
        except Exception:
            state_frames.append(0)
    conflicts: List[Dict[str, Any]] = []
    for episode in episodes or []:
        comp_key = str(episode.get("component_id"))
        comp_idx = component_idx_by_id.get(comp_key, -1)
        if comp_idx < 0:
            continue
        try:
            episode_start = int(episode.get("start_frame", 0))
            anchor_frame = int(episode.get("anchor_frame", episode.get("end_frame", 0)))
        except Exception:
            continue
        if anchor_frame < episode_start:
            continue
        target_state = int(episode.get("state", 0))
        before_state = _state_at_frame(
            state_sequence, state_frames, init_vec, episode_start - 1, comp_idx
        )
        if before_state != target_state and anchor_frame > episode_start:
            early_frame = None
            for item in state_sequence or []:
                try:
                    frame = int(item.get("frame", 0))
                except Exception:
                    continue
                if frame < int(episode_start):
                    continue
                if frame >= int(anchor_frame):
                    break
                state_vec = list(item.get("state") or [])
                if comp_idx < len(state_vec):
                    try:
                        if int(state_vec[comp_idx]) == target_state:
                            early_frame = int(frame)
                            break
                    except Exception:
                        continue
            if early_frame is not None:
                conflicts.append(
                    {
                        "conflict_type": "early_commit",
                        "severity": 3,
                        "component_id": episode.get("component_id"),
                        "component_name": episode.get("component_name"),
                        "label": episode.get("label"),
                        "episode_start": int(episode_start),
                        "episode_end": int(episode.get("end_frame", anchor_frame)),
                        "anchor_frame": int(anchor_frame),
                        "old_state": int(target_state),
                        "new_state": int(before_state),
                        "start": int(early_frame),
                        "end": int(anchor_frame - 1),
                        "reason": (
                            f"State reached target before completion anchor. "
                            f"Keep pre-state until frame {int(anchor_frame)}."
                        ),
                        "repair": {
                            "kind": "state_patch",
                            "start": int(early_frame),
                            "end": int(anchor_frame - 1),
                            "component_id": episode.get("component_id"),
                            "new_state": int(before_state),
                        },
                    }
                )
        anchor_state = _state_at_frame(
            state_sequence, state_frames, init_vec, anchor_frame, comp_idx
        )
        if anchor_state != target_state:
            next_change = _next_component_change_frame(
                state_sequence, int(anchor_frame), comp_idx
            )
            repair_end = int(view_end) if view_end is not None else int(anchor_frame)
            if next_change is not None:
                repair_end = min(repair_end, int(next_change) - 1)
            if repair_end >= int(anchor_frame):
                conflicts.append(
                    {
                        "conflict_type": "missing_commit",
                        "severity": 2,
                        "component_id": episode.get("component_id"),
                        "component_name": episode.get("component_name"),
                        "label": episode.get("label"),
                        "episode_start": int(episode_start),
                        "episode_end": int(episode.get("end_frame", anchor_frame)),
                        "anchor_frame": int(anchor_frame),
                        "old_state": int(anchor_state),
                        "new_state": int(target_state),
                        "start": int(anchor_frame),
                        "end": int(repair_end),
                        "reason": (
                            f"Completion anchor has no matching committed state. "
                            f"Apply target state from frame {int(anchor_frame)}."
                        ),
                        "repair": {
                            "kind": "state_patch",
                            "start": int(anchor_frame),
                            "end": int(repair_end),
                            "component_id": episode.get("component_id"),
                            "new_state": int(target_state),
                        },
                    }
                )
    conflicts.sort(
        key=lambda item: (
            -int(item.get("severity", 0)),
            int(item.get("start", 0)),
            str(item.get("component_id")),
        )
    )
    return conflicts
