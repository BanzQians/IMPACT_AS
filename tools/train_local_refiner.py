#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
except Exception as ex:  # pragma: no cover
    raise SystemExit(
        "torch is required for tools/train_local_refiner.py. "
        f"Import failed: {ex}"
    )

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.procedure_trace import analyze_trace_conflicts
from core.query_planner import estimate_observed_query_utility
from core.temporal_scribble import ScribbleKind, TemporalScribble, build_scribble_channels


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return data


def _load_examples(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                if isinstance(item, dict):
                    rows.append(item)
        return rows
    payload = _load_json(path)
    rows = payload.get("examples")
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain an 'examples' list")
    return [item for item in rows if isinstance(item, dict)]


def _load_mapping(path: str) -> Dict[str, str]:
    if not path:
        return {}
    data = _load_json(Path(path))
    out: Dict[str, str] = {}
    for key, value in data.items():
        if not key or not value:
            continue
        out[str(key)] = str(value)
    return out


def _infer_feat_layout(feat: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if feat.ndim != 2:
        raise ValueError(f"features.npy must be 2D, got {feat.shape}")
    if meta and isinstance(meta.get("feature_dim"), (int, float)):
        dim = int(meta.get("feature_dim"))
        if feat.shape[0] == dim and feat.shape[1] != dim:
            return feat.T
        if feat.shape[1] == dim and feat.shape[0] != dim:
            return feat
    typical_dims = {128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096}
    if feat.shape[0] in typical_dims and feat.shape[1] not in typical_dims:
        return feat.T
    if feat.shape[1] in typical_dims and feat.shape[0] not in typical_dims:
        return feat
    return feat


def _load_meta(features_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(features_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _build_frame_map(seq_len: int, meta: Optional[Dict[str, Any]] = None) -> List[int]:
    if meta:
        picked = meta.get("picked_indices")
        if isinstance(picked, list) and len(picked) == seq_len:
            try:
                return [int(x) for x in picked]
            except Exception:
                pass
        stride = meta.get("frame_stride")
        if stride is not None:
            try:
                stride_i = max(1, int(stride))
                return [int(i * stride_i) for i in range(seq_len)]
            except Exception:
                pass
    return list(range(seq_len))


@dataclass
class FeatureBundle:
    features_dir: str
    features: np.ndarray
    frame_map: np.ndarray
    feature_dim: int


def _load_feature_bundle(features_dir: str) -> FeatureBundle:
    feat_path = os.path.join(features_dir, "features.npy")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(f"features.npy not found in {features_dir}")
    meta = _load_meta(features_dir)
    feat = np.load(feat_path, mmap_mode="r")
    feat = _infer_feat_layout(np.asarray(feat), meta)
    frame_map = np.asarray(_build_frame_map(int(feat.shape[0]), meta), dtype=np.int64)
    return FeatureBundle(
        features_dir=str(features_dir),
        features=np.asarray(feat, dtype=np.float32),
        frame_map=frame_map,
        feature_dim=int(feat.shape[1]),
    )


def _resolve_features_dir(
    example: Dict[str, Any],
    *,
    default_dir: str,
    mapping: Dict[str, str],
) -> str:
    if default_dir:
        return str(default_dir)
    keys = [
        str(example.get("annotation_path") or ""),
        str(example.get("video_id") or ""),
    ]
    ann_path = str(example.get("annotation_path") or "")
    if ann_path:
        ann = Path(ann_path)
        keys.extend(
            [
                ann.stem,
                ann.stem.replace("_native", ""),
                ann.stem.replace("_annotations", ""),
            ]
        )
    for key in keys:
        if key and key in mapping:
            return str(mapping[key])
    raise KeyError(
        "Unable to resolve features_dir for example. "
        "Provide --features_dir or --features_map_json."
    )


def _nearest_feature_indices(frame_map: np.ndarray, frames: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(frame_map, frames, side="left")
    idx = np.clip(idx, 0, len(frame_map) - 1)
    prev_idx = np.clip(idx - 1, 0, len(frame_map) - 1)
    prev_dist = np.abs(frame_map[prev_idx] - frames)
    next_dist = np.abs(frame_map[idx] - frames)
    return np.where(prev_dist <= next_dist, prev_idx, idx).astype(np.int64)


def _boundary_energy_from_features(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] <= 1:
        return np.zeros((max(1, features.shape[0]),), dtype=np.float32)
    denom = np.linalg.norm(features, axis=1, keepdims=False)
    denom = np.maximum(denom, 1e-6)
    norm = features / denom[:, None]
    sim = np.sum(norm[1:] * norm[:-1], axis=1)
    dist = 1.0 - np.clip(sim, -1.0, 1.0)
    out = np.zeros((features.shape[0],), dtype=np.float32)
    out[1:] = dist.astype(np.float32)
    return out


def _state_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return str(value).strip()
    if isinstance(value, dict):
        parts: List[str] = []
        for key in sorted(value.keys()):
            text = str(value.get(key, "")).strip()
            if text:
                parts.append(f"{key}:{text}")
        return "|".join(parts)
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = [str(item).strip() for item in list(value) if str(item).strip()]
        return "|".join(parts)
    return str(value).strip()


def _query_utility_from_example(example: Dict[str, Any]) -> float:
    for key in ("query_utility", "observed_utility", "utility_target"):
        raw = example.get(key)
        if raw not in (None, ""):
            try:
                return float(raw)
            except Exception:
                continue
    if any(
        example.get(key) not in (None, "")
        for key in (
            "changed_frame_count",
            "anchor_violations_before",
            "anchor_violations_after",
            "state_conflicts_before",
            "state_conflicts_after",
        )
    ):
        boundary_frame = int(example.get("boundary_frame", 0) or 0)
        win_start = int(example.get("window_start", boundary_frame) or boundary_frame)
        win_end = int(example.get("window_end", boundary_frame) or boundary_frame)
        if win_end < win_start:
            win_start, win_end = win_end, win_start
        span_len = int(max(1, win_end - win_start + 1))
        return float(
            estimate_observed_query_utility(
                accepted=example.get("accepted", True),
                span_len=span_len,
                changed_frame_count=example.get("changed_frame_count", 0),
                anchor_before=example.get(
                    "anchor_violations_before", example.get("anchor_before", 0)
                ),
                anchor_after=example.get(
                    "anchor_violations_after", example.get("anchor_after", 0)
                ),
                state_before=example.get(
                    "state_conflicts_before", example.get("state_before_count", 0)
                ),
                state_after=example.get(
                    "state_conflicts_after", example.get("state_after_count", 0)
                ),
                second_correction=example.get("second_correction", False),
            )
        )
    return -1.0


def _coerce_float_vector(value: Any, length: int, *, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if int(arr.shape[0]) != int(length):
        raise ValueError(
            f"{field_name} length mismatch: expected {length}, got {arr.shape[0]}"
        )
    return arr.astype(np.float32, copy=False)


def _inline_window_payload(
    example: Dict[str, Any],
    *,
    window_radius: int,
) -> Optional[Dict[str, Any]]:
    raw_features = example.get("window_features")
    if raw_features is None:
        return None
    win_feat = np.asarray(raw_features, dtype=np.float32)
    if win_feat.ndim != 2 or win_feat.shape[0] <= 0:
        raise ValueError("window_features must be a non-empty 2D array")
    boundary_frame = int(example.get("boundary_frame", 0) or 0)
    win_s = int(example.get("window_start", boundary_frame - window_radius) or 0)
    raw_end = example.get("window_end", None)
    if raw_end is None:
        win_e = int(win_s + win_feat.shape[0] - 1)
    else:
        win_e = int(raw_end)
    if win_e < win_s:
        win_s, win_e = win_e, win_s
    expected_len = int(win_e - win_s + 1)
    if expected_len != int(win_feat.shape[0]):
        if raw_end is None:
            win_e = int(win_s + win_feat.shape[0] - 1)
        else:
            raise ValueError(
                f"window_features length mismatch: [{win_s}, {win_e}] implies {expected_len} "
                f"frames, got {win_feat.shape[0]}"
            )
    length = int(win_feat.shape[0])
    boundary_index = int(np.clip(boundary_frame - win_s, 0, length - 1))
    raw_channels = dict(example.get("scribble_channels") or {})
    if raw_channels:
        channels = {
            "uncertain": _coerce_float_vector(
                raw_channels.get("uncertain", np.zeros(length, dtype=np.float32)),
                length,
                field_name="scribble_channels.uncertain",
            ),
            "left": _coerce_float_vector(
                raw_channels.get("left", np.zeros(length, dtype=np.float32)),
                length,
                field_name="scribble_channels.left",
            ),
            "right": _coerce_float_vector(
                raw_channels.get("right", np.zeros(length, dtype=np.float32)),
                length,
                field_name="scribble_channels.right",
            ),
        }
    else:
        channels = build_scribble_channels(win_s, win_e, _example_scribbles(example))
    raw_energy = example.get("boundary_energy")
    if raw_energy is None:
        energy = _boundary_energy_from_features(win_feat)
    else:
        energy = _coerce_float_vector(raw_energy, length, field_name="boundary_energy")
    return {
        "win_start": int(win_s),
        "win_end": int(win_e),
        "win_feat": win_feat.astype(np.float32, copy=False),
        "channels": {
            "uncertain": np.asarray(
                channels.get("uncertain", np.zeros(length, dtype=np.float32)),
                dtype=np.float32,
            ),
            "left": np.asarray(
                channels.get("left", np.zeros(length, dtype=np.float32)),
                dtype=np.float32,
            ),
            "right": np.asarray(
                channels.get("right", np.zeros(length, dtype=np.float32)),
                dtype=np.float32,
            ),
        },
        "energy": np.asarray(energy, dtype=np.float32),
        "boundary_index": int(boundary_index),
    }


def _normalize_window_arrays(
    win_feat: np.ndarray,
    channels: Dict[str, Any],
    boundary_index: int,
    *,
    window_radius: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, int]:
    feat = np.asarray(win_feat, dtype=np.float32)
    if feat.ndim != 2 or feat.shape[0] <= 0:
        raise ValueError("window features must be a non-empty 2D array")
    radius = max(0, int(window_radius))
    target_len = max(1, 2 * radius + 1)
    src_len = int(feat.shape[0])
    center = int(np.clip(int(boundary_index), 0, src_len - 1))

    def _copy_slice_2d(arr: np.ndarray) -> np.ndarray:
        out = np.zeros((target_len, int(arr.shape[1])), dtype=np.float32)
        left = int(center - radius)
        right = int(center + radius + 1)
        src_start = max(0, left)
        src_end = min(src_len, right)
        dst_start = max(0, -left)
        dst_end = dst_start + max(0, src_end - src_start)
        if src_end > src_start:
            out[dst_start:dst_end] = arr[src_start:src_end]
            if dst_start > 0:
                out[:dst_start] = arr[0]
            if dst_end < target_len:
                out[dst_end:] = arr[-1]
        else:
            out[:] = arr[center]
        return out

    def _copy_slice_1d(arr_like: Any) -> np.ndarray:
        arr = np.asarray(arr_like, dtype=np.float32).reshape(-1)
        if arr.shape[0] != src_len:
            arr = np.zeros((src_len,), dtype=np.float32)
        out = np.zeros((target_len,), dtype=np.float32)
        left = int(center - radius)
        right = int(center + radius + 1)
        src_start = max(0, left)
        src_end = min(src_len, right)
        dst_start = max(0, -left)
        dst_end = dst_start + max(0, src_end - src_start)
        if src_end > src_start:
            out[dst_start:dst_end] = arr[src_start:src_end]
        return out

    norm_feat = _copy_slice_2d(feat)
    norm_channels = {
        "uncertain": _copy_slice_1d(channels.get("uncertain")),
        "left": _copy_slice_1d(channels.get("left")),
        "right": _copy_slice_1d(channels.get("right")),
    }
    norm_energy = _boundary_energy_from_features(norm_feat)
    return norm_feat, norm_channels, norm_energy, int(radius)


def _example_scribbles(example: Dict[str, Any]) -> List[TemporalScribble]:
    rows = list(example.get("scribbles") or [])
    out: List[TemporalScribble] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        try:
            kind = ScribbleKind(str(item.get("kind") or ScribbleKind.UNCERTAIN.value))
        except Exception:
            kind = ScribbleKind.UNCERTAIN
        out.append(
            TemporalScribble(
                start_frame=int(item.get("start_frame", 0) or 0),
                end_frame=int(item.get("end_frame", 0) or 0),
                kind=kind,
                view_id=str(example.get("view", "") or ""),
            )
        )
    if not out:
        boundary = int(example.get("boundary_frame", 0) or 0)
        out.append(
            TemporalScribble(
                start_frame=boundary,
                end_frame=boundary,
                kind=ScribbleKind.UNCERTAIN,
                view_id=str(example.get("view", "") or ""),
            )
        )
    return out


class ScribbleTrainingDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        *,
        features_dir: str,
        features_map: Dict[str, str],
        window_radius: int,
        label_to_idx: Dict[str, int],
        state_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.samples: List[Dict[str, Any]] = []
        self._feature_cache: Dict[str, FeatureBundle] = {}
        self.window_radius = int(window_radius)
        self.label_to_idx = dict(label_to_idx)
        self.state_to_idx: Dict[str, int] = dict(state_to_idx or {})
        self.has_states = bool(self.state_to_idx)
        if not self.label_to_idx:
            raise ValueError("label vocabulary must not be empty")
        default_label_idx = int(min(self.label_to_idx.values()))
        expected_dim: Optional[int] = None
        for example in examples:
            inline_payload = _inline_window_payload(
                example,
                window_radius=self.window_radius,
            )
            if inline_payload is not None:
                win_s = int(inline_payload["win_start"])
                win_e = int(inline_payload["win_end"])
                win_feat = np.asarray(inline_payload["win_feat"], dtype=np.float32)
                channels = dict(inline_payload["channels"] or {})
                energy = np.asarray(inline_payload["energy"], dtype=np.float32)
                bi = int(inline_payload["boundary_index"])
                win_feat, channels, energy, bi = _normalize_window_arrays(
                    win_feat,
                    channels,
                    bi,
                    window_radius=self.window_radius,
                )
                feature_dim = int(win_feat.shape[1])
            else:
                resolved = _resolve_features_dir(
                    example,
                    default_dir=features_dir,
                    mapping=features_map,
                )
                bundle = self._feature_cache.get(resolved)
                if bundle is None:
                    bundle = _load_feature_bundle(resolved)
                    self._feature_cache[resolved] = bundle
                boundary_frame = int(example.get("boundary_frame", 0) or 0)
                win_s = int(boundary_frame - self.window_radius)
                win_e = int(boundary_frame + self.window_radius)
                frames = np.arange(win_s, win_e + 1, dtype=np.int64)
                nearest = _nearest_feature_indices(bundle.frame_map, frames)
                win_feat = np.asarray(bundle.features[nearest], dtype=np.float32)
                energy = _boundary_energy_from_features(win_feat)
                channels = build_scribble_channels(win_s, win_e, _example_scribbles(example))
                bi = int(self.window_radius)
                feature_dim = int(bundle.feature_dim)
            if expected_dim is None:
                expected_dim = int(feature_dim)
            elif int(feature_dim) != int(expected_dim):
                raise ValueError(
                    f"Mixed feature dimensions are not supported: {feature_dim} vs {expected_dim}"
                )
            frames = np.arange(int(win_feat.shape[0]), dtype=np.int64)
            x = np.concatenate(
                [
                    win_feat,
                    channels.get("uncertain", np.zeros(len(frames), dtype=np.float32))[:, None],
                    channels.get("left", np.zeros(len(frames), dtype=np.float32))[:, None],
                    channels.get("right", np.zeros(len(frames), dtype=np.float32))[:, None],
                    energy[:, None],
                ],
                axis=1,
            )
            left_label = str(example.get("left_label", "") or "")
            right_label = str(example.get("right_label", "") or "")
            boundary_valid = bool(example.get("boundary_valid", True))
            side_valid = bool(example.get("side_valid", True))
            action_valid = bool(example.get("action_valid", True))
            left_label_idx = self.label_to_idx.get(left_label)
            right_label_idx = self.label_to_idx.get(right_label)
            allow_label_fallback = (not side_valid) and (not action_valid)
            if left_label_idx is None:
                if allow_label_fallback:
                    left_label_idx = int(right_label_idx if right_label_idx is not None else default_label_idx)
                else:
                    raise KeyError("Example left_label missing from label vocabulary")
            if right_label_idx is None:
                if allow_label_fallback:
                    right_label_idx = int(left_label_idx if left_label_idx is not None else default_label_idx)
                else:
                    raise KeyError("Example right_label missing from label vocabulary")
            # Dense per-frame action labels: left_label before boundary, right_label at/after
            action_labels = np.full(len(frames), -1, dtype=np.int64)
            if action_valid:
                action_labels[:bi] = int(left_label_idx)
                action_labels[bi:] = int(right_label_idx)

            sample: Dict[str, Any] = {
                "x": x.astype(np.float32),
                "uncertain": channels.get("uncertain", np.zeros(len(frames), dtype=np.float32)).astype(np.float32),
                "left": channels.get("left", np.zeros(len(frames), dtype=np.float32)).astype(np.float32),
                "right": channels.get("right", np.zeros(len(frames), dtype=np.float32)).astype(np.float32),
                "boundary_index": bi,
                "left_label": int(left_label_idx),
                "right_label": int(right_label_idx),
                "action_labels": action_labels,
                "sample_kind": str(example.get("sample_kind", "boundary_accept") or "boundary_accept"),
                "boundary_valid": bool(boundary_valid),
                "side_valid": bool(side_valid),
                "action_valid": bool(action_valid),
                "query_utility": float(_query_utility_from_example(example)),
                "multiview_conflict": float(
                    example.get("multiview_conflict", example.get("view_conflict", 0.0))
                    or 0.0
                ),
            }
            if self.has_states:
                left_state = _state_key(example.get("left_state", example.get("state_before", "")))
                right_state = _state_key(example.get("right_state", example.get("state_after", "")))
                state_before = _state_key(example.get("state_before", left_state))
                state_after = _state_key(example.get("state_after", right_state))
                sample["left_state"] = int(self.state_to_idx.get(left_state, -1))
                sample["right_state"] = int(self.state_to_idx.get(right_state, -1))
                sample["state_before"] = int(self.state_to_idx.get(state_before, -1))
                sample["state_after"] = int(self.state_to_idx.get(state_after, -1))
            self.samples.append(sample)
        self.input_dim = int(expected_dim or 0) + 4

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.samples[int(index)]
        out = {
            "x": torch.from_numpy(item["x"]),
            "uncertain": torch.from_numpy(item["uncertain"]),
            "left": torch.from_numpy(item["left"]),
            "right": torch.from_numpy(item["right"]),
            "boundary_index": torch.tensor(int(item["boundary_index"]), dtype=torch.long),
            "left_label": torch.tensor(int(item["left_label"]), dtype=torch.long),
            "right_label": torch.tensor(int(item["right_label"]), dtype=torch.long),
            "action_labels": torch.from_numpy(item["action_labels"]),
            "sample_kind": str(item.get("sample_kind", "boundary_accept") or "boundary_accept"),
            "boundary_valid": torch.tensor(bool(item.get("boundary_valid", True)), dtype=torch.bool),
            "side_valid": torch.tensor(bool(item.get("side_valid", True)), dtype=torch.bool),
            "action_valid": torch.tensor(bool(item.get("action_valid", True)), dtype=torch.bool),
            "query_utility": torch.tensor(float(item.get("query_utility", -1.0)), dtype=torch.float32),
            "multiview_conflict": torch.tensor(
                float(item.get("multiview_conflict", 0.0)), dtype=torch.float32
            ),
        }
        if self.has_states:
            out["left_state"] = torch.tensor(int(item.get("left_state", -1)), dtype=torch.long)
            out["right_state"] = torch.tensor(int(item.get("right_state", -1)), dtype=torch.long)
            out["state_before"] = torch.tensor(int(item.get("state_before", -1)), dtype=torch.long)
            out["state_after"] = torch.tensor(int(item.get("state_after", -1)), dtype=torch.long)
        return out


class TinyLocalBoundaryModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        num_states: int = 0,
        dense_action_head: bool = False,
        query_head: bool = False,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.encoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.boundary_head = nn.Linear(int(hidden_dim), 1)
        ctx_dim = int(hidden_dim) * 4
        self.left_head = nn.Sequential(
            nn.Linear(ctx_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )
        self.right_head = nn.Sequential(
            nn.Linear(ctx_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )
        self.num_states = int(num_states)
        if self.num_states > 0:
            self.left_state_head = nn.Sequential(
                nn.Linear(ctx_dim, int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), int(num_states)),
            )
            self.right_state_head = nn.Sequential(
                nn.Linear(ctx_dim, int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), int(num_states)),
            )
        self.has_action_head = bool(dense_action_head)
        if self.has_action_head:
            self.action_head = nn.Linear(int(hidden_dim), int(num_classes))
        self.has_query_head = bool(query_head)
        if self.has_query_head:
            self.query_head = nn.Sequential(
                nn.Linear(ctx_dim, int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), 1),
            )

    @staticmethod
    def _masked_mean(
        hidden: torch.Tensor,
        mask: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        pooled = (hidden * weights).sum(dim=1) / denom
        valid = (mask.sum(dim=1, keepdim=True) > 0).expand_as(pooled)
        return torch.where(valid, pooled, fallback)

    def forward(
        self,
        x: torch.Tensor,
        uncertain: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(self.input_proj(x))
        boundary_logits = self.boundary_head(hidden).squeeze(-1)
        boundary_probs = torch.softmax(boundary_logits, dim=1)
        cdf = torch.cumsum(boundary_probs, dim=1)
        left_soft = (1.0 - cdf).clamp_min(0.0)
        right_soft = (1.0 - left_soft).clamp_min(0.0)

        global_pool = hidden.mean(dim=1)
        uncertain_pool = self._masked_mean(hidden, uncertain, global_pool)
        left_pool = self._masked_mean(
            hidden,
            (left_soft + 0.5 * left + 0.15 * uncertain).clamp_min(0.0),
            global_pool,
        )
        right_pool = self._masked_mean(
            hidden,
            (right_soft + 0.5 * right + 0.15 * uncertain).clamp_min(0.0),
            global_pool,
        )
        context = torch.cat([left_pool, right_pool, uncertain_pool, global_pool], dim=1)
        out = {
            "boundary_logits": boundary_logits,
            "left_logits": self.left_head(context),
            "right_logits": self.right_head(context),
        }
        if self.num_states > 0:
            out["left_state_logits"] = self.left_state_head(context)
            out["right_state_logits"] = self.right_state_head(context)
        if self.has_action_head:
            out["action_logits"] = self.action_head(hidden)  # (B, T, num_classes)
        if self.has_query_head:
            out["query_utility_logits"] = self.query_head(context).squeeze(-1)
        return out


def _split_indices(
    count: int, val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    indices = list(range(int(count)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    if count <= 1 or val_fraction <= 0.0:
        return indices, []
    val_count = max(1, int(round(count * float(val_fraction))))
    if val_count >= count:
        val_count = max(0, count - 1)
    return indices[val_count:], indices[:val_count]


def _compute_state_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Compute L_state from left/right state predictions. Ignores samples with label=-1."""
    loss = torch.tensor(0.0, device=device)
    count = 0
    for side in ("left", "right"):
        key_logits = f"{side}_state_logits"
        key_target = f"{side}_state"
        if key_logits not in out or key_target not in batch:
            continue
        logits = out[key_logits]
        target = batch[key_target].to(device)
        valid = target >= 0
        if valid.any():
            loss = loss + criterion(logits[valid], target[valid])
            count += 1
    return loss / max(1, count)


def _compute_boundary_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    logits = out["boundary_logits"]
    target_boundary = batch["boundary_index"].to(device)
    valid = batch.get("boundary_valid")
    if valid is None:
        valid_mask = torch.ones_like(target_boundary, dtype=torch.bool, device=device)
    else:
        valid_mask = valid.to(device).bool()
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)

    boundary_probs = torch.softmax(logits, dim=1)
    uncertain = batch.get("uncertain")
    if uncertain is None:
        return criterion(logits[valid_mask], target_boundary[valid_mask])
    uncertain = uncertain.to(device)
    losses: List[torch.Tensor] = []
    for idx in torch.nonzero(valid_mask, as_tuple=False).view(-1):
        interval_mask = uncertain[idx] > 0.0
        if bool(interval_mask.any()):
            interval_mass = boundary_probs[idx][interval_mask].sum().clamp_min(1e-6)
            losses.append(-torch.log(interval_mass))
        else:
            losses.append(
                torch.nn.functional.cross_entropy(
                    logits[idx : idx + 1],
                    target_boundary[idx : idx + 1],
                    reduction="mean",
                )
            )
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


def _compute_side_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    valid = batch.get("side_valid")
    target_left = batch["left_label"].to(device)
    target_right = batch["right_label"].to(device)
    if valid is None:
        valid_mask = torch.ones_like(target_left, dtype=torch.bool, device=device)
    else:
        valid_mask = valid.to(device).bool()
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)
    loss_left = criterion(out["left_logits"][valid_mask], target_left[valid_mask])
    loss_right = criterion(out["right_logits"][valid_mask], target_right[valid_mask])
    return 0.5 * (loss_left + loss_right)


def _compute_trace_conflict_penalty(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    state_before = batch.get("state_before", batch.get("left_state"))
    state_after = batch.get("state_after", batch.get("right_state"))
    if state_before is None or state_after is None:
        return torch.tensor(0.0, device=device)
    boundary_target = batch.get("boundary_index")
    if boundary_target is None:
        return torch.tensor(0.0, device=device)
    state_before = state_before.to(device)
    state_after = state_after.to(device)
    boundary_target = boundary_target.to(device)
    valid = (state_before >= 0) & (state_after >= 0) & (boundary_target >= 0)
    boundary_valid = batch.get("boundary_valid")
    if boundary_valid is not None:
        valid = valid & boundary_valid.to(device).bool()
    if not valid.any():
        return torch.tensor(0.0, device=device)

    boundary_pred = out["boundary_logits"].argmax(dim=1)
    left_state_pred = (
        out["left_state_logits"].argmax(dim=-1)
        if "left_state_logits" in out
        else state_before
    )
    right_state_pred = (
        out["right_state_logits"].argmax(dim=-1)
        if "right_state_logits" in out
        else state_after
    )
    seq_len = int(out["boundary_logits"].shape[1])
    penalties: List[float] = []
    right_label = batch.get("right_label")

    for idx in torch.nonzero(valid, as_tuple=False).view(-1).tolist():
        s_before = int(state_before[idx].item())
        s_after = int(state_after[idx].item())
        b_target = int(boundary_target[idx].item())
        b_pred = int(boundary_pred[idx].item())
        right_state_i = int(right_state_pred[idx].item())
        state_sequence: List[Dict[str, Any]] = []
        if right_state_i != s_before:
            state_sequence.append({"frame": int(b_pred), "state": [int(right_state_i)]})
        components = [{"id": "0", "name": "window_state"}]
        episode = {
            "component_id": "0",
            "component_name": "window_state",
            "state": int(s_after),
            "start_frame": 0,
            "end_frame": int(max(0, seq_len - 1)),
            "anchor_frame": int(b_target),
            "label": (
                str(int(right_label[idx].item()))
                if right_label is not None
                else "right"
            ),
        }
        conflicts = analyze_trace_conflicts(
            [episode],
            state_sequence,
            components,
            initial_state=[int(s_before)],
            view_end=int(max(0, seq_len - 1)),
        )
        severity = 0.0
        for conflict in conflicts:
            try:
                severity += float(conflict.get("severity", 0) or 0.0) / 3.0
            except Exception:
                continue
        if int(left_state_pred[idx].item()) != int(s_before):
            severity += 0.35
        if right_state_i != int(s_after):
            severity += 0.35
        penalties.append(float(min(2.0, severity)))
    if not penalties:
        return torch.tensor(0.0, device=device)
    return torch.tensor(float(sum(penalties) / len(penalties)), device=device)


def _compute_consistency_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute L_consistency: non-differentiable-style soft penalties.

    Rules:
    1. A strong boundary inside explicit left/right anchor scribbles is inconsistent.
    2. If target state_before/state_after show no transition, strong boundary confidence is penalized.
    3. If a transition is expected, left/right state heads should not collapse to the same state.
    4. Left/right label heads should respect the confirmed anchor labels.
    """
    loss = torch.tensor(0.0, device=device)
    count = 0

    boundary_probs = torch.softmax(out["boundary_logits"], dim=1)
    boundary_peak = boundary_probs.max(dim=1).values
    boundary_valid = batch.get("boundary_valid")
    if boundary_valid is None:
        boundary_valid_mask = torch.ones_like(boundary_peak, dtype=torch.bool, device=device)
    else:
        boundary_valid_mask = boundary_valid.to(device).bool()
    side_valid = batch.get("side_valid")
    if side_valid is None:
        side_valid_mask = torch.ones_like(boundary_peak, dtype=torch.bool, device=device)
    else:
        side_valid_mask = side_valid.to(device).bool()

    left_anchor = batch.get("left")
    right_anchor = batch.get("right")
    if (left_anchor is not None or right_anchor is not None) and boundary_valid_mask.any():
        anchor_mask = torch.zeros_like(boundary_probs)
        if left_anchor is not None:
            anchor_mask = anchor_mask + left_anchor.to(device).clamp_min(0.0)
        if right_anchor is not None:
            anchor_mask = anchor_mask + right_anchor.to(device).clamp_min(0.0)
        if float(anchor_mask.sum().item()) > 0.0:
            anchor_mass = (boundary_probs * anchor_mask.clamp(max=1.0)).sum(dim=1)
            loss = loss + anchor_mass[boundary_valid_mask].mean()
            count += 1

    state_before = batch.get("state_before", batch.get("left_state"))
    state_after = batch.get("state_after", batch.get("right_state"))
    if state_before is not None and state_after is not None:
        state_before = state_before.to(device)
        state_after = state_after.to(device)
        valid_state = (state_before >= 0) & (state_after >= 0)
        if valid_state.any():
            same_state = valid_state & (state_before == state_after) & boundary_valid_mask
            if same_state.any():
                loss = loss + boundary_peak[same_state].mean()
                count += 1
            if "left_state_logits" in out and "right_state_logits" in out:
                diff_state = valid_state & (state_before != state_after)
                if diff_state.any():
                    left_state_probs = torch.softmax(out["left_state_logits"], dim=-1)
                    right_state_probs = torch.softmax(out["right_state_logits"], dim=-1)
                    overlap = torch.min(left_state_probs, right_state_probs).sum(dim=-1)
                    loss = loss + overlap[diff_state].mean()
                    count += 1
        trace_penalty = _compute_trace_conflict_penalty(out, batch, device)
        if float(trace_penalty.item()) > 0.0:
            loss = loss + trace_penalty
            count += 1

    multiview_conflict = batch.get("multiview_conflict")
    if multiview_conflict is not None:
        mv = multiview_conflict.to(device).clamp_min(0.0)
        if float(mv.max().item()) > 0.0 and boundary_valid_mask.any():
            loss = loss + (boundary_peak[boundary_valid_mask] * mv[boundary_valid_mask]).mean()
            count += 1

    for side in ("left", "right"):
        logits_key = f"{side}_logits"
        target_key = f"{side}_label"
        anchor_key = side
        if logits_key not in out or target_key not in batch:
            continue
        targets = batch[target_key].to(device)
        probs = torch.softmax(out[logits_key], dim=-1)
        target_prob = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        anchor_strength = batch.get(anchor_key)
        if anchor_strength is None:
            penalty = (1.0 - target_prob)[side_valid_mask]
        else:
            weights = anchor_strength.to(device).clamp_min(0.0).mean(dim=1)
            weights = weights * side_valid_mask.to(dtype=weights.dtype)
            if not (weights > 0).any():
                continue
            penalty = (1.0 - target_prob) * weights
            penalty = penalty[weights > 0]
        if penalty.numel() > 0:
            loss = loss + penalty.mean()
            count += 1

    return loss / max(1, count)


def _compute_struct_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute L_struct as dense sequence alignment to the constrained soft decode."""
    if "action_logits" not in out:
        return torch.tensor(0.0, device=device)
    action_logits = out["action_logits"]
    action_valid = batch.get("action_valid")
    if action_valid is not None:
        valid_mask = action_valid.to(device).bool()
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)
        action_logits = action_logits[valid_mask]
        boundary_logits = out["boundary_logits"][valid_mask]
        left_logits = out["left_logits"][valid_mask]
        right_logits = out["right_logits"][valid_mask]
        batch_action_labels = batch.get("action_labels")
        if batch_action_labels is not None:
            batch_action_labels = batch_action_labels.to(device)[valid_mask]
        batch_boundary_index = batch.get("boundary_index")
        if batch_boundary_index is not None:
            batch_boundary_index = batch_boundary_index.to(device)[valid_mask]
        batch_left_label = batch["left_label"].to(device)[valid_mask]
        batch_right_label = batch["right_label"].to(device)[valid_mask]
    else:
        boundary_logits = out["boundary_logits"]
        left_logits = out["left_logits"]
        right_logits = out["right_logits"]
        batch_action_labels = (
            batch.get("action_labels").to(device) if batch.get("action_labels") is not None else None
        )
        batch_boundary_index = (
            batch.get("boundary_index").to(device)
            if batch.get("boundary_index") is not None
            else None
        )
        batch_left_label = batch["left_label"].to(device)
        batch_right_label = batch["right_label"].to(device)
    boundary_probs = torch.softmax(boundary_logits, dim=1)
    cdf = torch.cumsum(boundary_probs, dim=1)
    left_mass = (1.0 - cdf).clamp_min(0.0)
    right_mass = (1.0 - left_mass).clamp_min(0.0)
    left_probs = torch.softmax(left_logits, dim=-1).unsqueeze(1)
    right_probs = torch.softmax(right_logits, dim=-1).unsqueeze(1)
    target_soft = left_mass.unsqueeze(-1) * left_probs + right_mass.unsqueeze(-1) * right_probs

    if batch_action_labels is not None:
        dense = batch_action_labels
        valid = dense >= 0
        if valid.any():
            one_hot = torch.nn.functional.one_hot(
                dense.clamp_min(0), num_classes=int(action_logits.shape[-1])
            ).to(dtype=target_soft.dtype, device=device)
            blend = valid.unsqueeze(-1).to(dtype=target_soft.dtype)
            target_soft = (1.0 - 0.5 * blend) * target_soft + 0.5 * blend * one_hot

    action_log_probs = torch.log_softmax(action_logits, dim=-1)
    ce_soft = -(target_soft * action_log_probs).sum(dim=-1).mean()

    if batch_boundary_index is None:
        return ce_soft
    boundary_index = batch_boundary_index
    T = int(action_logits.shape[1])
    frames = torch.arange(T, device=device).view(1, T)
    before_mask = frames < boundary_index.view(-1, 1)
    after_mask = ~before_mask
    probs = torch.softmax(action_logits, dim=-1)
    left_idx = batch_left_label.view(-1, 1, 1)
    right_idx = batch_right_label.view(-1, 1, 1)
    left_target_prob = probs.gather(2, left_idx.expand(-1, T, 1)).squeeze(-1)
    right_target_prob = probs.gather(2, right_idx.expand(-1, T, 1)).squeeze(-1)
    leakage = torch.tensor(0.0, device=device)
    count = 0
    if before_mask.any():
        leakage = leakage + (1.0 - left_target_prob[before_mask]).mean()
        count += 1
    if after_mask.any():
        leakage = leakage + (1.0 - right_target_prob[after_mask]).mean()
        count += 1
    if count > 0:
        leakage = leakage / float(count)
    return ce_soft + leakage


def _compute_action_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute L_action: dense per-frame action label supervision.

    For each window, frames before the boundary get left_label,
    frames at/after the boundary get right_label.
    Requires 'action_logits' in model output and 'action_labels' in batch.
    """
    if "action_logits" not in out or "action_labels" not in batch:
        return torch.tensor(0.0, device=device)
    logits = out["action_logits"]  # (B, T, C)
    targets = batch["action_labels"].to(device)  # (B, T)
    action_valid = batch.get("action_valid")
    if action_valid is not None:
        valid_sample_mask = action_valid.to(device).bool()
        if not valid_sample_mask.any():
            return torch.tensor(0.0, device=device)
        logits = logits[valid_sample_mask]
        targets = targets[valid_sample_mask]
    valid = targets >= 0
    if not valid.any():
        return torch.tensor(0.0, device=device)
    loss = torch.nn.functional.cross_entropy(
        logits[valid], targets[valid], reduction="mean"
    )
    return loss


def _compute_query_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if "query_utility_logits" not in out or "query_utility" not in batch:
        return torch.tensor(0.0, device=device)
    targets = batch["query_utility"].to(device)
    valid = targets >= 0.0
    if not valid.any():
        return torch.tensor(0.0, device=device)
    pred = torch.sigmoid(out["query_utility_logits"][valid])
    return torch.nn.functional.mse_loss(pred, targets[valid], reduction="mean")


def _evaluate(
    model: TinyLocalBoundaryModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    boundary_weight: float,
    label_weight: float,
    state_weight: float = 0.0,
    consistency_weight: float = 0.0,
    action_weight: float = 0.0,
    struct_weight: float = 0.0,
    query_weight: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    total = 0
    total_loss = 0.0
    boundary_ok = 0
    boundary_total = 0
    left_ok = 0
    left_total = 0
    right_ok = 0
    right_total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            uncertain = batch["uncertain"].to(device)
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            target_boundary = batch["boundary_index"].to(device)
            target_left = batch["left_label"].to(device)
            target_right = batch["right_label"].to(device)
            out = model(x, uncertain, left, right)
            loss_boundary = _compute_boundary_loss(out, batch, criterion, device)
            loss_side = _compute_side_loss(out, batch, criterion, device)
            loss = float(boundary_weight) * loss_boundary + float(label_weight) * loss_side
            if state_weight > 0.0 and "left_state_logits" in out:
                loss = loss + float(state_weight) * _compute_state_loss(
                    out, batch, criterion, device
                )
            if consistency_weight > 0.0:
                loss = loss + float(consistency_weight) * _compute_consistency_loss(
                    out, batch, device
                )
            if action_weight > 0.0 and "action_logits" in out:
                loss = loss + float(action_weight) * _compute_action_loss(
                    out, batch, device
                )
            if struct_weight > 0.0 and "action_logits" in out:
                loss = loss + float(struct_weight) * _compute_struct_loss(
                    out, batch, device
                )
            if query_weight > 0.0 and "query_utility_logits" in out:
                loss = loss + float(query_weight) * _compute_query_loss(
                    out, batch, device
                )
            batch_size = int(x.shape[0])
            total += batch_size
            total_loss += float(loss.item()) * batch_size
            boundary_valid = batch.get("boundary_valid")
            if boundary_valid is None:
                boundary_valid_mask = torch.ones_like(target_boundary, dtype=torch.bool, device=device)
            else:
                boundary_valid_mask = boundary_valid.to(device).bool()
            if boundary_valid_mask.any():
                pred_boundary = out["boundary_logits"].argmax(dim=1)
                boundary_match = pred_boundary == target_boundary
                if "uncertain" in batch:
                    uncertain_mask = uncertain > 0.0
                    interval_has_mass = uncertain_mask.any(dim=1)
                    interval_match = torch.zeros_like(boundary_match, dtype=torch.bool)
                    interval_rows = boundary_valid_mask & interval_has_mass
                    if interval_rows.any():
                        interval_match[interval_rows] = uncertain_mask[interval_rows].gather(
                            1, pred_boundary[interval_rows].view(-1, 1)
                        ).squeeze(1).bool()
                    boundary_match = torch.where(interval_has_mass, interval_match, boundary_match)
                boundary_ok += int(boundary_match[boundary_valid_mask].sum().item())
                boundary_total += int(boundary_valid_mask.sum().item())
            side_valid = batch.get("side_valid")
            if side_valid is None:
                side_valid_mask = torch.ones_like(target_left, dtype=torch.bool, device=device)
            else:
                side_valid_mask = side_valid.to(device).bool()
            if side_valid_mask.any():
                pred_left = out["left_logits"].argmax(dim=1)
                pred_right = out["right_logits"].argmax(dim=1)
                left_ok += int((pred_left[side_valid_mask] == target_left[side_valid_mask]).sum().item())
                right_ok += int((pred_right[side_valid_mask] == target_right[side_valid_mask]).sum().item())
                left_total += int(side_valid_mask.sum().item())
                right_total += int(side_valid_mask.sum().item())
    if total <= 0:
        return {"loss": 0.0, "boundary_acc": 0.0, "left_acc": 0.0, "right_acc": 0.0}
    return {
        "loss": float(total_loss / total),
        "boundary_acc": float(boundary_ok / max(1, boundary_total)),
        "left_acc": float(left_ok / max(1, left_total)),
        "right_acc": float(right_ok / max(1, right_total)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a tiny local boundary refiner on synthetic temporal scribble examples."
    )
    ap.add_argument("--dataset", required=True, help="Synthetic scribble json/jsonl path")
    ap.add_argument("--features_dir", default="", help="Single features dir for all examples")
    ap.add_argument(
        "--features_map_json",
        default="",
        help="Optional json mapping video_id / annotation path / basename to features_dir",
    )
    ap.add_argument("--output", default="", help="Checkpoint output path (.pt)")
    ap.add_argument("--window_radius", type=int, default=24)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--boundary_loss_weight", type=float, default=1.0)
    ap.add_argument("--label_loss_weight", type=float, default=1.0)
    ap.add_argument("--state_loss_weight", type=float, default=0.5)
    ap.add_argument("--consistency_loss_weight", type=float, default=0.3)
    ap.add_argument("--action_loss_weight", type=float, default=0.0,
                     help="Weight for dense per-frame action loss (0 to disable)")
    ap.add_argument(
        "--struct_loss_weight",
        type=float,
        default=0.0,
        help="Weight for structured dense action/side consistency loss",
    )
    ap.add_argument(
        "--query_loss_weight",
        type=float,
        default=0.0,
        help="Weight for learned query-utility regression loss",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument(
        "--lora_rank", type=int, default=0,
        help="If >0, freeze base weights and train only LoRA adapters of this rank",
    )
    ap.add_argument("--lora_alpha", type=float, default=1.0)
    ap.add_argument(
        "--base_checkpoint", default="",
        help="Pre-trained checkpoint to load before LoRA fine-tuning",
    )
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    dataset_path = Path(args.dataset)
    examples = _load_examples(dataset_path)
    if not examples:
        raise SystemExit("Dataset is empty.")

    labels = sorted(
        {
            str(item.get("left_label", "") or "")
            for item in examples
            if str(item.get("left_label", "") or "")
        }
        | {
            str(item.get("right_label", "") or "")
            for item in examples
            if str(item.get("right_label", "") or "")
        }
    )
    if not labels:
        raise SystemExit("No left/right labels found in dataset examples.")
    label_to_idx = {name: idx for idx, name in enumerate(labels)}

    # Build state vocabulary from examples (if state labels present)
    states = sorted(
        {
            _state_key(item.get("left_state", item.get("state_before", "")))
            for item in examples
            if _state_key(item.get("left_state", item.get("state_before", "")))
        }
        | {
            _state_key(item.get("right_state", item.get("state_after", "")))
            for item in examples
            if _state_key(item.get("right_state", item.get("state_after", "")))
        }
        | {
            _state_key(item.get("state_before", ""))
            for item in examples
            if _state_key(item.get("state_before", ""))
        }
        | {
            _state_key(item.get("state_after", ""))
            for item in examples
            if _state_key(item.get("state_after", ""))
        }
    )
    state_to_idx = {name: idx for idx, name in enumerate(states)} if states else {}

    features_map = _load_mapping(args.features_map_json)
    dataset = ScribbleTrainingDataset(
        examples,
        features_dir=str(args.features_dir or ""),
        features_map=features_map,
        window_radius=max(1, int(args.window_radius)),
        label_to_idx=label_to_idx,
        state_to_idx=state_to_idx,
    )
    train_idx, val_idx = _split_indices(
        len(dataset), float(args.val_fraction), int(args.seed)
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if val_idx else None
    train_loader = DataLoader(
        train_set, batch_size=max(1, int(args.batch_size)), shuffle=True
    )
    val_loader = (
        DataLoader(val_set, batch_size=max(1, int(args.batch_size)), shuffle=False)
        if val_set is not None
        else None
    )

    if str(args.device).lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))

    num_states = len(state_to_idx)
    model = TinyLocalBoundaryModel(
        input_dim=int(dataset.input_dim),
        num_classes=len(label_to_idx),
        hidden_dim=max(16, int(args.hidden_dim)),
        dropout=max(0.0, float(args.dropout)),
        num_states=num_states,
        dense_action_head=(
            float(args.action_loss_weight) > 0.0
            or float(args.struct_loss_weight) > 0.0
        ),
        query_head=float(args.query_loss_weight) > 0.0,
    ).to(device)

    # Load base checkpoint for fine-tuning (LoRA or full)
    lora_rank = max(0, int(args.lora_rank))
    if args.base_checkpoint and os.path.isfile(args.base_checkpoint):
        base_ckpt = torch.load(args.base_checkpoint, map_location="cpu")
        model.load_state_dict(base_ckpt["state_dict"], strict=False)
        print(f"[train_local_refiner] loaded base checkpoint from {args.base_checkpoint}")

    # Inject LoRA if requested
    if lora_rank > 0:
        from core.local_boundary_refiner import inject_lora, collect_lora_state_dict
        inject_lora(model, rank=lora_rank, alpha=float(args.lora_alpha))
        print(f"[train_local_refiner] injected LoRA rank={lora_rank} alpha={args.lora_alpha}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    criterion = nn.CrossEntropyLoss()

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric = math.inf
    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        total = 0
        total_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            uncertain = batch["uncertain"].to(device)
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            target_boundary = batch["boundary_index"].to(device)
            target_left = batch["left_label"].to(device)
            target_right = batch["right_label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x, uncertain, left, right)
            loss_boundary = _compute_boundary_loss(out, batch, criterion, device)
            loss_side = _compute_side_loss(out, batch, criterion, device)
            loss = float(args.boundary_loss_weight) * loss_boundary + float(
                args.label_loss_weight
            ) * loss_side
            if float(args.state_loss_weight) > 0.0 and "left_state_logits" in out:
                loss = loss + float(args.state_loss_weight) * _compute_state_loss(
                    out, batch, criterion, device
                )
            if float(args.consistency_loss_weight) > 0.0:
                loss = loss + float(args.consistency_loss_weight) * _compute_consistency_loss(
                    out, batch, device
                )
            if float(args.action_loss_weight) > 0.0 and "action_logits" in out:
                loss = loss + float(args.action_loss_weight) * _compute_action_loss(
                    out, batch, device
                )
            if float(args.struct_loss_weight) > 0.0 and "action_logits" in out:
                loss = loss + float(args.struct_loss_weight) * _compute_struct_loss(
                    out, batch, device
                )
            if float(args.query_loss_weight) > 0.0 and "query_utility_logits" in out:
                loss = loss + float(args.query_loss_weight) * _compute_query_loss(
                    out, batch, device
                )
            loss.backward()
            optimizer.step()

            batch_size = int(x.shape[0])
            total += batch_size
            total_loss += float(loss.item()) * batch_size

        train_loss = float(total_loss / max(1, total))
        if val_loader is not None:
            metrics = _evaluate(
                model,
                val_loader,
                criterion,
                device,
                float(args.boundary_loss_weight),
                float(args.label_loss_weight),
                state_weight=float(args.state_loss_weight),
                consistency_weight=float(args.consistency_loss_weight),
                action_weight=float(args.action_loss_weight),
                struct_weight=float(args.struct_loss_weight),
                query_weight=float(args.query_loss_weight),
            )
            score = float(metrics["loss"])
            if score < best_metric:
                best_metric = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"[train_local_refiner] epoch={epoch:02d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={metrics['loss']:.4f} "
                f"val_boundary_acc={metrics['boundary_acc']:.3f} "
                f"val_left_acc={metrics['left_acc']:.3f} "
                f"val_right_acc={metrics['right_acc']:.3f}"
            )
        else:
            if train_loss < best_metric:
                best_metric = train_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(f"[train_local_refiner] epoch={epoch:02d} train_loss={train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = (
        Path(args.output)
        if args.output
        else dataset_path.with_name(dataset_path.stem + "_tiny_local_refiner.pt")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # For LoRA mode: save LoRA weights separately, then merge into base for inference compat
    lora_sd = {}
    if lora_rank > 0:
        from core.local_boundary_refiner import collect_lora_state_dict, merge_lora
        lora_sd = collect_lora_state_dict(model)
        merge_lora(model)

    checkpoint = {
        "state_dict": model.state_dict(),
        "label_to_idx": label_to_idx,
        "idx_to_label": {idx: label for label, idx in label_to_idx.items()},
        "window_radius": int(args.window_radius),
        "input_dim": int(dataset.input_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "best_metric": float(best_metric),
        "num_examples": int(len(dataset)),
        "dataset_path": str(dataset_path),
        "features_dir": str(args.features_dir or ""),
        "features_map_json": str(args.features_map_json or ""),
        "dense_action_head": (
            float(args.action_loss_weight) > 0.0
            or float(args.struct_loss_weight) > 0.0
        ),
        "query_head": float(args.query_loss_weight) > 0.0,
        "state_loss_weight": float(args.state_loss_weight),
        "consistency_loss_weight": float(args.consistency_loss_weight),
        "action_loss_weight": float(args.action_loss_weight),
        "struct_loss_weight": float(args.struct_loss_weight),
        "query_loss_weight": float(args.query_loss_weight),
    }
    if state_to_idx:
        checkpoint["state_to_idx"] = state_to_idx
        checkpoint["idx_to_state"] = {idx: s for s, idx in state_to_idx.items()}
        checkpoint["num_states"] = num_states
    if lora_rank > 0 and lora_sd:
        checkpoint["lora_state_dict"] = lora_sd
        checkpoint["lora_rank"] = lora_rank
        checkpoint["lora_alpha"] = float(args.lora_alpha)
        checkpoint["base_checkpoint"] = str(args.base_checkpoint or "")
    torch.save(checkpoint, out_path)
    print(f"[train_local_refiner] saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
