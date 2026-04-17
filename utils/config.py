from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_algo_override() -> Dict[str, Any]:
    raw_path = str(os.environ.get("IMPACT_SCRIBE_ALGO_CONFIG_PATH", "") or "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"IMPACT_SCRIBE_ALGO_CONFIG_PATH not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Algorithm config override must be a JSON object: {path}")
    return payload


_DEFAULT_ALGO_CONFIG = {
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
        "search_auto_candidates": True,
        "hidden_dim": 128,
        "dropout": 0.1,
        "state_loss_weight": 0.5,
        "consistency_loss_weight": 0.3,
        "action_loss_weight": 0.0,
        "struct_loss_weight": 0.0,
        "query_loss_weight": 0.0,
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
        "utility_model_path": "",
        "online_refit_interval": 8,
        "fit_lr": 0.01,
        "fit_epochs": 20,
    },
    "scribble_adaptation": {
        "enabled": True,
        "min_samples": 8,
        "max_buffer": 200,
        "epochs": 6,
        "lr": 5e-4,
        "lora_rank": 4,
        "lora_alpha": 1.0,
        "max_train_samples": 192,
        "min_sample_confidence": 0.35,
        "validation_fraction": 0.15,
        "min_validation_samples": 8,
        "max_validation_regression": 0.02,
        "output_dir": "",
    },
    "psr": {
        "initial_state_policy": "auto",
        "no_gap_timeline": True,
        "auto_carry_next_on_edit": True,
        "interruptible_episode": True,
        "episode_resume_gap_sec": 4.0,
    },
}


ALGO_CONFIG = _deep_merge_dict(_DEFAULT_ALGO_CONFIG, _load_algo_override())
