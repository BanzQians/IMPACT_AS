#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
except Exception as ex:  # pragma: no cover
    raise SystemExit(
        "torch is required for tools/train_global_model_from_windows.py. "
        f"Import failed: {ex}"
    )

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_local_refiner import (  # noqa: E402
    _load_feature_bundle,
    _load_mapping,
    _nearest_feature_indices,
    _resolve_features_dir,
    _split_indices,
)


def _load_windows(path: Path) -> List[Dict[str, Any]]:
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
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("windows"), list):
        return [item for item in payload.get("windows") or [] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError(f"{path} must contain a windows list")


class GlobalWindowDataset(Dataset):
    def __init__(
        self,
        windows: Sequence[Dict[str, Any]],
        *,
        features_dir: str,
        features_map: Dict[str, str],
        label_to_idx: Dict[str, int],
    ) -> None:
        self.samples: List[Dict[str, Any]] = []
        self._feature_cache: Dict[str, Any] = {}
        expected_dim: Optional[int] = None
        self.label_to_idx = dict(label_to_idx)

        for window in windows:
            dense_labels = list(window.get("dense_labels") or [])
            if not dense_labels:
                continue
            resolved = _resolve_features_dir(
                window,
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
            start_frame = int(window.get("start_frame", 0) or 0)
            frames = np.arange(
                start_frame,
                start_frame + len(dense_labels),
                dtype=np.int64,
            )
            valid_positions = [
                idx
                for idx, label in enumerate(dense_labels)
                if str(label or "").strip() in self.label_to_idx
            ]
            if not valid_positions:
                continue
            sel_frames = frames[np.asarray(valid_positions, dtype=np.int64)]
            feature_indices = _nearest_feature_indices(bundle.frame_map, sel_frames)
            x = np.asarray(bundle.features[feature_indices], dtype=np.float32)
            y = np.asarray(
                [
                    self.label_to_idx[str(dense_labels[idx]).strip()]
                    for idx in valid_positions
                ],
                dtype=np.int64,
            )
            self.samples.append({"x": x, "y": y})
        self.input_dim = int(expected_dim or 0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.samples[int(index)]
        return {
            "x": torch.from_numpy(item["x"]),
            "y": torch.from_numpy(item["y"]),
        }


def _collate_batch(rows: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x = torch.cat([row["x"] for row in rows], dim=0)
    y = torch.cat([row["y"] for row in rows], dim=0)
    return {"x": x, "y": y}


class TinyGlobalSegmentationModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _evaluate(
    model: TinyGlobalSegmentationModel,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    if loader is None:
        return {"loss": 0.0, "frame_acc": 0.0}
    model.eval()
    total_frames = 0
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_frames += int(y.shape[0])
            total_loss += float(loss.item()) * int(y.shape[0])
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
    if total_frames <= 0:
        return {"loss": 0.0, "frame_acc": 0.0}
    return {
        "loss": float(total_loss / total_frames),
        "frame_acc": float(total_correct / total_frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight background global segmentation model from "
            "exported IMPACT-Scribe confirmed windows."
        )
    )
    parser.add_argument("--windows", required=True, help="export_confirmed_windows json/jsonl path")
    parser.add_argument("--features_dir", default="", help="Single features dir for all windows")
    parser.add_argument(
        "--features_map_json",
        default="",
        help="Optional json mapping video_id / annotation path / basename to features_dir",
    )
    parser.add_argument("--output", default="", help="Checkpoint output path (.pt)")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    windows_path = Path(args.windows).expanduser().resolve()
    windows = _load_windows(windows_path)
    if not windows:
        raise SystemExit("Window dataset is empty.")

    labels = sorted(
        {
            str(label or "").strip()
            for row in windows
            for label in list(row.get("dense_labels") or [])
            if str(label or "").strip()
        }
    )
    if not labels:
        raise SystemExit("No dense labels found in exported windows.")
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    features_map = _load_mapping(args.features_map_json)
    dataset = GlobalWindowDataset(
        windows,
        features_dir=str(args.features_dir or ""),
        features_map=features_map,
        label_to_idx=label_to_idx,
    )
    if len(dataset) <= 0 or int(dataset.input_dim) <= 0:
        raise SystemExit("No trainable frame samples could be built from the windows dataset.")

    train_idx, val_idx = _split_indices(len(dataset), float(args.val_fraction), int(args.seed))
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if val_idx else None
    train_loader = DataLoader(
        train_set,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        collate_fn=_collate_batch,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            collate_fn=_collate_batch,
        )
        if val_set is not None
        else None
    )

    if str(args.device).lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))

    model = TinyGlobalSegmentationModel(
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

    best_metric = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        total_frames = 0
        total_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_frames += int(y.shape[0])
            total_loss += float(loss.item()) * int(y.shape[0])
        train_loss = float(total_loss / max(1, total_frames))
        metrics = _evaluate(model, val_loader, criterion, device)
        score = float(metrics["loss"] if val_loader is not None else train_loss)
        if score < best_metric:
            best_metric = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if val_loader is not None:
            print(
                f"[train_global_model_from_windows] epoch={epoch:02d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={metrics['loss']:.4f} "
                f"val_frame_acc={metrics['frame_acc']:.3f}"
            )
        else:
            print(
                f"[train_global_model_from_windows] epoch={epoch:02d} "
                f"train_loss={train_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else windows_path.with_name(windows_path.stem + "_tiny_global_model.pt")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "label_to_idx": label_to_idx,
        "idx_to_label": {idx: label for label, idx in label_to_idx.items()},
        "input_dim": int(dataset.input_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "best_metric": float(best_metric),
        "num_windows": int(len(dataset)),
        "windows_path": str(windows_path),
        "features_dir": str(args.features_dir or ""),
        "features_map_json": str(args.features_map_json or ""),
    }
    torch.save(checkpoint, out_path)
    print(f"[train_global_model_from_windows] saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
