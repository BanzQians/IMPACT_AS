ALGO_CONFIG = {
    "timeline_snap": {
        "playhead_radius": 6,
        "empty_space_radius": 10,
        "edge_search_radius": 5,
        "segment_soft_radius": 10,
        "phase_soft_radius": 8,
        "hover_preview_multi": True,
        "hover_preview_align": "absolute",
    },
    "boundary_snap": {
        "enabled": True,
        "window_size": 15,
    },
    "segment_embedding": {
        "trim_ratio": 0.1,
    },
    "topk": {
        "enabled": True,
        "k": 5,
        "uncertainty_margin": 0.25,
    },
    "assisted": {
        "boundary_min_gap": 15,
    },
    "scribble_local_refiner": {
        "checkpoint": "",
        "device": "auto",
        "prefer_checkpoint": True,
    },
    "structured_decode": {
        "transition_penalty": 0.55,
        "stay_bonus": 0.05,
        "current_label_score": 1.0,
        "alternate_label_score": 0.16,
        "anchor_boost": 0.55,
    },
    "query_learning": {
        "enabled": True,
        "second_correction_radius": 20,
        "proposal_feedback_enabled": True,
    },
    "psr": {
        "initial_state_policy": "auto",
        "no_gap_timeline": True,
        "auto_carry_next_on_edit": True,
        "interruptible_episode": True,
        "episode_resume_gap_sec": 4.0,
    },
}
