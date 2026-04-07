#!/usr/bin/env python3
"""Stage C: Distill confirmed local corrections into a global segmentation model.

This script takes:
  1. A pseudo-label file (JSON) exported from the IMPACT-Scribe GUI
     containing per-frame action labels for confirmed windows.
  2. A features directory with features.npy.
  3. An existing ASOT checkpoint.

It fine-tunes the ASOT model on the pseudo-labels using sequence-level
cross-entropy on confirmed frames plus a structured window loss over confirmed
segments, producing an updated checkpoint that incorporates the human
corrections.

Usage:
    python tools/distill_global_model.py \
        --pseudo_labels pseudo_labels.json \
        --features_dir /path/to/features \
        --ckpt /path/to/asot_checkpoint.ckpt \
        --output distilled_model.ckpt \
        --epochs 10 --lr 1e-4
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:
    raise SystemExit(f"PyTorch is required. Import failed: {e}") from e

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.structured_decode import ConfirmedWindow, decode_frame_labels_with_constraints


def _load_pseudo_labels(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    return data


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


def _build_frame_map(seq_len: int, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if meta:
        picked = meta.get("picked_indices")
        if isinstance(picked, list) and len(picked) == seq_len:
            try:
                return np.asarray([int(x) for x in picked], dtype=np.int64)
            except Exception:
                pass
        stride = meta.get("frame_stride")
        if stride is not None:
            try:
                stride_i = max(1, int(stride))
                return np.asarray([int(i * stride_i) for i in range(seq_len)], dtype=np.int64)
            except Exception:
                pass
    return np.arange(int(seq_len), dtype=np.int64)


def _offset_frame_map_for_view(
    frame_map: np.ndarray,
    pseudo_labels: Dict[str, Any],
) -> np.ndarray:
    fmap = np.asarray(frame_map, dtype=np.int64).reshape(-1)
    if fmap.size <= 0:
        return fmap
    view_start = int(pseudo_labels.get("view_start", 0) or 0)
    view_end = int(pseudo_labels.get("view_end", view_start) or view_start)
    per_frame = pseudo_labels.get("per_frame_labels")
    view_len = len(per_frame) if isinstance(per_frame, list) else max(0, view_end - view_start + 1)
    if view_len <= 0 or fmap.size != int(view_len):
        return fmap
    if int(fmap[0]) == int(view_start) and int(fmap[-1]) == int(view_end):
        return fmap
    if int(fmap[0]) == 0 and int(fmap[-1]) == int(view_len - 1) and int(view_start) != 0:
        return fmap + int(view_start)
    return fmap


def _nearest_feature_indices(frame_map: np.ndarray, frames: np.ndarray) -> np.ndarray:
    fmap = np.asarray(frame_map, dtype=np.int64).reshape(-1)
    pts = np.asarray(frames, dtype=np.int64).reshape(-1)
    if fmap.size <= 0 or pts.size <= 0:
        return np.zeros((pts.size,), dtype=np.int64)
    idx = np.searchsorted(fmap, pts, side="left")
    idx = np.clip(idx, 0, fmap.size - 1)
    prev_idx = np.clip(idx - 1, 0, fmap.size - 1)
    prev_dist = np.abs(fmap[prev_idx] - pts)
    next_dist = np.abs(fmap[idx] - pts)
    return np.where(prev_dist <= next_dist, prev_idx, idx).astype(np.int64)


@dataclass
class FeatureBundle:
    features: np.ndarray
    frame_map: np.ndarray


def _load_feature_bundle(features_dir: str, pseudo_labels: Optional[Dict[str, Any]] = None) -> FeatureBundle:
    feat_path = os.path.join(features_dir, "features.npy")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(f"features.npy not found in {features_dir}")
    meta = _load_meta(features_dir)
    feat = np.load(feat_path)
    feat = _infer_feat_layout(feat, meta).astype(np.float32)
    frame_map = _build_frame_map(int(feat.shape[0]), meta)
    if isinstance(pseudo_labels, dict):
        frame_map = _offset_frame_map_for_view(frame_map, pseudo_labels)
    return FeatureBundle(features=feat, frame_map=frame_map)


def _build_training_data(
    features: np.ndarray,
    pseudo_labels: Dict[str, Any],
    label_to_idx: Dict[str, int],
    frame_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (features, targets, mask) arrays from pseudo-labels.

    Returns:
        features: (T, D) float32
        targets: (T,) int64 with -1 for unlabelled frames
        mask: (T,) bool — True for frames with confirmed labels
    """
    T = features.shape[0]
    targets = np.full(T, -1, dtype=np.int64)
    fmap = np.asarray(frame_map if frame_map is not None else np.arange(T), dtype=np.int64).reshape(-1)
    if fmap.size != T:
        fmap = np.arange(T, dtype=np.int64)
    view_start = int(pseudo_labels.get("view_start", 0) or 0)
    per_frame = pseudo_labels.get("per_frame_labels")
    confirmed_mask = pseudo_labels.get("confirmed_frame_mask")

    windows = pseudo_labels.get("confirmed_windows") or []
    for win in windows:
        if not isinstance(win, dict):
            continue
        start = int(win.get("start_frame", 0))
        end = int(win.get("end_frame", 0))
        boundary = win.get("boundary_frame")
        left_label = str(win.get("left_label", ""))
        right_label = str(win.get("right_label", ""))
        fill_label = left_label or right_label
        if boundary is None:
            if fill_label not in label_to_idx:
                continue
        elif left_label not in label_to_idx or right_label not in label_to_idx:
            continue
        start_idx, end_idx = _nearest_feature_indices(
            fmap,
            np.asarray([int(start), int(end)], dtype=np.int64),
        ).tolist()
        start = max(0, min(int(start_idx), T - 1))
        end = max(0, min(int(end_idx), T - 1))
        if end < start:
            start, end = end, start
        if boundary is not None:
            bf = int(
                _nearest_feature_indices(
                    fmap,
                    np.asarray([int(boundary)], dtype=np.int64),
                )[0]
            )
            bf = max(start, min(bf, end))
            targets[start:bf] = label_to_idx[left_label]
            targets[bf:end + 1] = label_to_idx[right_label]
        else:
            # Single-label window
            targets[start:end + 1] = label_to_idx[fill_label]

    # Also handle per-frame labels if present
    if isinstance(per_frame, list):
        for target_idx, raw_frame in enumerate(fmap.tolist()):
            rel = int(raw_frame) - int(view_start)
            if rel < 0 or rel >= len(per_frame):
                continue
            if isinstance(confirmed_mask, list):
                if rel >= len(confirmed_mask) or not bool(confirmed_mask[rel]):
                    continue
            lbl = per_frame[rel]
            if isinstance(lbl, str) and lbl in label_to_idx:
                targets[int(target_idx)] = label_to_idx[lbl]
            elif isinstance(lbl, int) and 0 <= lbl < len(label_to_idx):
                targets[int(target_idx)] = int(lbl)

    mask = targets >= 0
    return features, targets, mask


def _match_model_feature_dim(features: np.ndarray, model: nn.Module) -> np.ndarray:
    feat = np.asarray(features, dtype=np.float32)
    expected_dim = None
    try:
        expected_dim = int(model.layer_sizes[0])
    except Exception:
        expected_dim = None
    if not expected_dim or int(feat.shape[1]) == int(expected_dim):
        return feat
    if int(feat.shape[1]) > int(expected_dim):
        return feat[:, : int(expected_dim)].astype(np.float32, copy=False)
    pad = np.zeros((int(feat.shape[0]), int(expected_dim) - int(feat.shape[1])), dtype=np.float32)
    return np.concatenate([feat, pad], axis=1).astype(np.float32, copy=False)


def _prepare_sequence_input(features: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(device)


def _forward_backbone_repr(
    model: nn.Module,
    seq_x: torch.Tensor,
    *,
    mode: str = "auto",
) -> Tuple[torch.Tensor, str]:
    requested = str(mode or "auto").strip().lower() or "auto"
    if requested not in {"auto", "full", "mlp"}:
        requested = "auto"
    full_error: Optional[Exception] = None
    if requested in {"auto", "full"}:
        try:
            reprs = model(seq_x)
            while reprs.dim() > 3:
                reprs = reprs.squeeze(0)
            if reprs.dim() == 2:
                reprs = reprs.unsqueeze(0)
            if reprs.dim() != 3:
                raise ValueError(f"unexpected ASOT forward output shape: {tuple(reprs.shape)}")
            return reprs, "full"
        except Exception as ex:
            full_error = ex
            if requested == "full":
                raise
    try:
        base = seq_x.squeeze(0)
        if hasattr(model, "mlp"):
            reprs = model.mlp(base)
        else:
            reprs = base
        if reprs.dim() == 2:
            reprs = reprs.unsqueeze(0)
        if reprs.dim() != 3:
            raise ValueError(f"unexpected MLP output shape: {tuple(reprs.shape)}")
        return reprs, "mlp"
    except Exception as ex:
        if full_error is not None:
            raise RuntimeError(
                f"ASOT full forward failed ({full_error}) and MLP fallback failed ({ex})"
            ) from ex
        raise


def _compute_sequence_struct_loss(
    logits: torch.Tensor,
    confirmed_windows: List[Dict[str, Any]],
    label_to_idx: Dict[str, int],
    frame_map: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if not confirmed_windows:
        return torch.tensor(0.0, device=device)
    seq_logits = logits.squeeze(0) if logits.dim() == 3 else logits
    if seq_logits.dim() != 2 or seq_logits.shape[0] <= 0:
        return torch.tensor(0.0, device=device)
    fmap = np.asarray(frame_map, dtype=np.int64).reshape(-1)
    if fmap.size <= 0:
        return torch.tensor(0.0, device=device)
    idx_to_label = {int(idx): str(name) for name, idx in label_to_idx.items()}
    pred_idx = seq_logits.argmax(dim=-1)
    current_labels: Dict[int, str] = {}
    for pos in range(min(int(seq_logits.shape[0]), int(fmap.size))):
        label = idx_to_label.get(int(pred_idx[pos].item()))
        if label:
            current_labels[int(fmap[pos])] = label
    windows = [
        ConfirmedWindow(
            start_frame=int(item.get("start_frame", 0) or 0),
            end_frame=int(item.get("end_frame", item.get("start_frame", 0)) or 0),
            boundary_frame=(
                None
                if item.get("boundary_frame") is None
                else int(item.get("boundary_frame"))
            ),
            left_label=str(item.get("left_label", "") or ""),
            right_label=str(item.get("right_label", "") or ""),
            hard=True,
        )
        for item in confirmed_windows
        if isinstance(item, dict)
    ]
    if not windows:
        return torch.tensor(0.0, device=device)
    decoded, _diag = decode_frame_labels_with_constraints(
        current_labels,
        windows,
        [],
        label_vocabulary=list(label_to_idx.keys()),
        frame_start=int(fmap.min()),
        frame_end=int(fmap.max()),
    )
    targets = torch.full((int(seq_logits.shape[0]),), -1, dtype=torch.long, device=device)
    for pos in range(min(int(seq_logits.shape[0]), int(fmap.size))):
        label = str(decoded.get(int(fmap[pos]), "") or "").strip()
        if label in label_to_idx:
            targets[pos] = int(label_to_idx[label])
    valid = targets >= 0
    if not valid.any():
        return torch.tensor(0.0, device=device)
    return torch.nn.functional.cross_entropy(
        seq_logits[valid],
        targets[valid],
        reduction="mean",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage C: Distill confirmed pseudo-labels into global segmentation model."
    )
    ap.add_argument("--pseudo_labels", required=True, help="Pseudo-label JSON from GUI export")
    ap.add_argument("--features_dir", required=True, help="Directory with features.npy")
    ap.add_argument("--ckpt", required=True, help="ASOT checkpoint (.ckpt or .pth)")
    ap.add_argument("--output", default="", help="Output checkpoint path")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Legacy argument; current Stage C distillation runs on the full sequence.",
    )
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--backbone_mode",
        default="auto",
        choices=["auto", "full", "mlp"],
        help="Prefer full ASOT forward or fallback to MLP features",
    )
    ap.add_argument(
        "--struct_loss_weight",
        type=float,
        default=0.3,
        help="Weight for confirmed-window structured consistency loss",
    )
    ap.add_argument("--label_map_json", default="",
                     help="JSON mapping label names to indices (auto-built from pseudo_labels if absent)")
    args = ap.parse_args()

    # Load data
    pseudo = _load_pseudo_labels(args.pseudo_labels)
    feature_bundle = _load_feature_bundle(args.features_dir, pseudo)
    features = feature_bundle.features
    T, D = features.shape
    print(f"[distill] features shape: ({T}, {D})")

    # Build label vocabulary
    if args.label_map_json and os.path.isfile(args.label_map_json):
        with open(args.label_map_json, "r", encoding="utf-8") as f:
            raw_label_map = json.load(f)
        label_to_idx = {
            str(name): int(idx)
            for name, idx in dict(raw_label_map or {}).items()
            if str(name).strip()
        }
    elif isinstance(pseudo.get("label_to_idx"), dict):
        label_to_idx = {
            str(name): int(idx)
            for name, idx in dict(pseudo.get("label_to_idx") or {}).items()
            if str(name).strip()
        }
    else:
        labels = set()
        for win in pseudo.get("confirmed_windows") or []:
            if isinstance(win, dict):
                for key in ("left_label", "right_label"):
                    lbl = str(win.get(key, "") or "")
                    if lbl:
                        labels.add(lbl)
        per_frame = pseudo.get("per_frame_labels")
        if isinstance(per_frame, list):
            for lbl in per_frame:
                if isinstance(lbl, str) and lbl:
                    labels.add(lbl)
        label_to_idx = {name: idx for idx, name in enumerate(sorted(labels))}
    num_classes = len(label_to_idx)
    print(f"[distill] {num_classes} classes: {list(label_to_idx.keys())[:10]}...")

    # Build training data
    _, targets, mask = _build_training_data(
        features,
        pseudo,
        label_to_idx,
        frame_map=feature_bundle.frame_map,
    )
    labelled_count = int(mask.sum())
    print(f"[distill] {labelled_count}/{T} frames have pseudo-labels ({100*labelled_count/max(1,T):.1f}%)")
    if labelled_count == 0:
        print("[distill] No labelled frames. Exiting.")
        return

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    ASOT_ROOT = os.path.join(str(ROOT), "external", "action_seg_ot")
    ASOT_SRC = os.path.join(ASOT_ROOT, "src")
    if ASOT_SRC not in sys.path:
        sys.path.insert(0, ASOT_SRC)

    try:
        from tools.asot_full_infer_adapter import build_model
    except ImportError:
        from asot_full_infer_adapter import build_model

    model, hp = build_model(args.ckpt)
    model.to(device)
    print(f"[distill] loaded ASOT model from {args.ckpt}")

    features = _match_model_feature_dim(features, model)
    seq_x = _prepare_sequence_input(features, device)
    target_t = torch.from_numpy(np.asarray(targets, dtype=np.int64)).to(device)
    mask_t = torch.from_numpy(np.asarray(mask, dtype=np.bool_)).to(device)
    confirmed_windows = [
        dict(item) for item in list(pseudo.get("confirmed_windows") or []) if isinstance(item, dict)
    ]

    model.eval()
    with torch.no_grad():
        probe_repr, actual_backbone_mode = _forward_backbone_repr(
            model,
            seq_x,
            mode=str(args.backbone_mode),
        )
    repr_dim = int(probe_repr.shape[-1])
    distill_head = nn.Sequential(
        nn.Linear(repr_dim, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes),
    ).to(device)

    params = list(distill_head.parameters())
    if actual_backbone_mode == "full":
        params += [p for p in model.parameters() if p.requires_grad]
    else:
        try:
            params += [p for p in model.mlp.parameters() if p.requires_grad]
        except Exception:
            pass
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_model_state = None
    best_head_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        distill_head.train()
        optimizer.zero_grad(set_to_none=True)
        backbone_repr, _ = _forward_backbone_repr(
            model,
            seq_x,
            mode=actual_backbone_mode,
        )
        seq_logits = distill_head(backbone_repr.squeeze(0))
        labelled_logits = seq_logits[mask_t]
        labelled_targets = target_t[mask_t]
        loss = criterion(labelled_logits, labelled_targets)
        if float(args.struct_loss_weight) > 0.0:
            loss = loss + float(args.struct_loss_weight) * _compute_sequence_struct_loss(
                seq_logits,
                confirmed_windows,
                label_to_idx,
                feature_bundle.frame_map,
                device,
            )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = seq_logits.argmax(dim=-1)
            acc = float((pred[mask_t] == labelled_targets).float().mean().item())
            avg_loss = float(loss.item())
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                best_head_state = {
                    k: v.detach().cpu().clone() for k, v in distill_head.state_dict().items()
                }
        print(f"[distill] epoch {epoch}/{args.epochs} loss={avg_loss:.4f} acc={acc:.3f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state, strict=False)
    if best_head_state is not None:
        distill_head.load_state_dict(best_head_state, strict=False)

    # Save checkpoint
    out_path = args.output or str(Path(args.ckpt).with_name("distilled_global.ckpt"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_dict = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "distill_head_state_dict": {k: v.detach().cpu() for k, v in distill_head.state_dict().items()},
        "label_to_idx": label_to_idx,
        "idx_to_label": {idx: name for name, idx in label_to_idx.items()},
        "num_classes": num_classes,
        "distill_input_dim": int(repr_dim),
        "pseudo_labels_path": str(args.pseudo_labels),
        "features_dir": str(args.features_dir),
        "base_ckpt": str(args.ckpt),
        "epochs": args.epochs,
        "best_loss": float(best_loss),
        "backbone_mode": str(actual_backbone_mode),
        "struct_loss_weight": float(args.struct_loss_weight),
    }
    if isinstance(hp, dict) and hp:
        save_dict["hyper_parameters"] = dict(hp)
        save_dict["base_hparams"] = dict(hp)
    torch.save(save_dict, out_path)
    print(f"[distill] saved distilled checkpoint to {out_path}")


if __name__ == "__main__":
    main()
