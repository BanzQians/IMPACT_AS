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
    ) -> None:
        self.samples: List[Dict[str, Any]] = []
        self._feature_cache: Dict[str, FeatureBundle] = {}
        self.window_radius = int(window_radius)
        self.label_to_idx = dict(label_to_idx)
        expected_dim: Optional[int] = None
        for example in examples:
            resolved = _resolve_features_dir(
                example,
                default_dir=features_dir,
                mapping=features_map,
            )
            bundle = self._feature_cache.get(resolved)
            if bundle is None:
                bundle = _load_feature_bundle(resolved)
                self._feature_cache[resolved] = bundle
            if expected_dim is None:
                expected_dim = int(bundle.feature_dim)
            elif int(bundle.feature_dim) != int(expected_dim):
                raise ValueError(
                    f"Mixed feature dimensions are not supported: {bundle.feature_dim} vs {expected_dim}"
                )
            boundary_frame = int(example.get("boundary_frame", 0) or 0)
            win_s = int(boundary_frame - self.window_radius)
            win_e = int(boundary_frame + self.window_radius)
            frames = np.arange(win_s, win_e + 1, dtype=np.int64)
            nearest = _nearest_feature_indices(bundle.frame_map, frames)
            win_feat = np.asarray(bundle.features[nearest], dtype=np.float32)
            energy = _boundary_energy_from_features(win_feat)
            channels = build_scribble_channels(win_s, win_e, _example_scribbles(example))
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
            if left_label not in self.label_to_idx or right_label not in self.label_to_idx:
                raise KeyError("Example label missing from label vocabulary")
            self.samples.append(
                {
                    "x": x.astype(np.float32),
                    "uncertain": channels.get("uncertain", np.zeros(len(frames), dtype=np.float32)).astype(np.float32),
                    "left": channels.get("left", np.zeros(len(frames), dtype=np.float32)).astype(np.float32),
                    "right": channels.get("right", np.zeros(len(frames), dtype=np.float32)).astype(np.float32),
                    "boundary_index": int(self.window_radius),
                    "left_label": int(self.label_to_idx[left_label]),
                    "right_label": int(self.label_to_idx[right_label]),
                }
            )
        self.input_dim = int(expected_dim or 0) + 4

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.samples[int(index)]
        return {
            "x": torch.from_numpy(item["x"]),
            "uncertain": torch.from_numpy(item["uncertain"]),
            "left": torch.from_numpy(item["left"]),
            "right": torch.from_numpy(item["right"]),
            "boundary_index": torch.tensor(int(item["boundary_index"]), dtype=torch.long),
            "left_label": torch.tensor(int(item["left_label"]), dtype=torch.long),
            "right_label": torch.tensor(int(item["right_label"]), dtype=torch.long),
        }


class TinyLocalBoundaryModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float) -> None:
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
        return {
            "boundary_logits": boundary_logits,
            "left_logits": self.left_head(context),
            "right_logits": self.right_head(context),
        }


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


def _evaluate(
    model: TinyLocalBoundaryModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    boundary_weight: float,
    label_weight: float,
) -> Dict[str, float]:
    model.eval()
    total = 0
    total_loss = 0.0
    boundary_ok = 0
    left_ok = 0
    right_ok = 0
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
            loss_boundary = criterion(out["boundary_logits"], target_boundary)
            loss_left = criterion(out["left_logits"], target_left)
            loss_right = criterion(out["right_logits"], target_right)
            loss = float(boundary_weight) * loss_boundary + float(label_weight) * (
                0.5 * (loss_left + loss_right)
            )
            batch_size = int(x.shape[0])
            total += batch_size
            total_loss += float(loss.item()) * batch_size
            boundary_ok += int(
                (out["boundary_logits"].argmax(dim=1) == target_boundary).sum().item()
            )
            left_ok += int((out["left_logits"].argmax(dim=1) == target_left).sum().item())
            right_ok += int((out["right_logits"].argmax(dim=1) == target_right).sum().item())
    if total <= 0:
        return {"loss": 0.0, "boundary_acc": 0.0, "left_acc": 0.0, "right_acc": 0.0}
    return {
        "loss": float(total_loss / total),
        "boundary_acc": float(boundary_ok / total),
        "left_acc": float(left_ok / total),
        "right_acc": float(right_ok / total),
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
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
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

    features_map = _load_mapping(args.features_map_json)
    dataset = ScribbleTrainingDataset(
        examples,
        features_dir=str(args.features_dir or ""),
        features_map=features_map,
        window_radius=max(1, int(args.window_radius)),
        label_to_idx=label_to_idx,
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

    model = TinyLocalBoundaryModel(
        input_dim=int(dataset.input_dim),
        num_classes=len(label_to_idx),
        hidden_dim=max(16, int(args.hidden_dim)),
        dropout=max(0.0, float(args.dropout)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            loss_boundary = criterion(out["boundary_logits"], target_boundary)
            loss_left = criterion(out["left_logits"], target_left)
            loss_right = criterion(out["right_logits"], target_right)
            loss = float(args.boundary_loss_weight) * loss_boundary + float(
                args.label_loss_weight
            ) * (0.5 * (loss_left + loss_right))
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
    }
    torch.save(checkpoint, out_path)
    print(f"[train_local_refiner] saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
