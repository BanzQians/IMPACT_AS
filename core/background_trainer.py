"""Background QThread for periodic LoRA fine-tuning of the local boundary refiner."""
from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from PyQt5.QtCore import QThread, pyqtSignal
except ImportError:  # headless / test
    from unittest.mock import MagicMock
    QThread = type("QThread", (), {"__init__": lambda *a, **k: None, "start": lambda s: None})
    pyqtSignal = MagicMock


class BackgroundRefinerTrainer(QThread):
    """Runs a short LoRA fine-tune on accumulated adaptation samples in a background thread.

    Signals:
        training_finished(str): emitted with the path to the new checkpoint on
        success, or an empty string on failure.
        progress(str): human-readable status messages.
    """

    training_finished = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        base_checkpoint: str,
        output_dir: str,
        *,
        label_to_idx: Dict[str, int],
        state_to_idx: Optional[Dict[str, int]] = None,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        epochs: int = 6,
        lr: float = 5e-4,
        window_radius: int = 24,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        boundary_loss_weight: float = 1.0,
        label_loss_weight: float = 1.0,
        state_loss_weight: float = 0.5,
        consistency_loss_weight: float = 0.3,
        action_loss_weight: float = 0.0,
        struct_loss_weight: float = 0.0,
        query_loss_weight: float = 0.0,
        device: str = "auto",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.samples = list(samples)
        self.base_checkpoint = str(base_checkpoint)
        self.output_dir = str(output_dir)
        self.label_to_idx = dict(label_to_idx)
        self.state_to_idx = dict(state_to_idx or {})
        self.lora_rank = max(1, int(lora_rank))
        self.lora_alpha = float(lora_alpha)
        self.epochs = max(1, int(epochs))
        self.lr = float(lr)
        self.window_radius = max(1, int(window_radius))
        self.hidden_dim = max(16, int(hidden_dim))
        self.dropout = max(0.0, float(dropout))
        self.boundary_loss_weight = max(0.0, float(boundary_loss_weight))
        self.label_loss_weight = max(0.0, float(label_loss_weight))
        self.state_loss_weight = max(0.0, float(state_loss_weight))
        self.consistency_loss_weight = max(0.0, float(consistency_loss_weight))
        self.action_loss_weight = max(0.0, float(action_loss_weight))
        self.struct_loss_weight = max(0.0, float(struct_loss_weight))
        self.query_loss_weight = max(0.0, float(query_loss_weight))
        self.device_request = str(device)
        self._result_path = ""

    @property
    def result_path(self) -> str:
        return self._result_path

    def run(self) -> None:
        try:
            self._train()
        except Exception as ex:
            self.progress.emit(f"[BackgroundRefinerTrainer] error: {ex}")
            self.training_finished.emit("")

    def _train(self) -> None:
        import torch
        from torch import nn

        from core.local_boundary_refiner import (
            inject_lora,
            collect_lora_state_dict,
            merge_lora,
            _import_torch,
            _resolve_torch_device,
        )
        self.progress.emit(f"[BackgroundRefinerTrainer] starting with {len(self.samples)} samples")

        if not self.samples:
            self.training_finished.emit("")
            return

        # Resolve device
        device = _resolve_torch_device(torch, self.device_request)

        # Build model (use the train script's class for consistency)
        from tools.train_local_refiner import (
            TinyLocalBoundaryModel,
            _compute_action_loss,
            _compute_boundary_loss,
            _compute_consistency_loss,
            _compute_query_loss,
            _compute_side_loss,
            _compute_state_loss,
            _compute_struct_loss,
            _normalize_window_arrays,
        )

        has_base_checkpoint = bool(self.base_checkpoint and os.path.isfile(self.base_checkpoint))
        ckpt: Dict[str, Any] = {}
        num_classes = len(self.label_to_idx)
        num_states = len(self.state_to_idx)
        input_dim = 0
        hidden_dim = int(self.hidden_dim)
        dropout = float(self.dropout)
        boundary_weight = float(self.boundary_loss_weight)
        label_weight = float(self.label_loss_weight)
        action_weight = float(self.action_loss_weight)
        struct_weight = float(self.struct_loss_weight)
        query_weight = float(self.query_loss_weight)
        state_weight = float(self.state_loss_weight if num_states > 0 else 0.0)
        consistency_weight = float(self.consistency_loss_weight if num_states > 0 else 0.0)
        target_radius = int(self.window_radius)
        if has_base_checkpoint:
            ckpt = torch.load(self.base_checkpoint, map_location="cpu")
            input_dim = int(ckpt.get("input_dim", 0))
            hidden_dim = int(ckpt.get("hidden_dim", hidden_dim) or hidden_dim)
            dropout = float(ckpt.get("dropout", dropout) or dropout)
            boundary_weight = float(ckpt.get("boundary_loss_weight", boundary_weight) or boundary_weight)
            label_weight = float(ckpt.get("label_loss_weight", label_weight) or label_weight)
            action_weight = float(ckpt.get("action_loss_weight", action_weight) or 0.0)
            struct_weight = float(ckpt.get("struct_loss_weight", struct_weight) or 0.0)
            query_weight = float(ckpt.get("query_loss_weight", query_weight) or 0.0)
            state_weight = float(ckpt.get("state_loss_weight", state_weight) or 0.0)
            consistency_weight = float(
                ckpt.get("consistency_loss_weight", consistency_weight) or 0.0
            )
            target_radius = max(
                1,
                int(ckpt.get("window_radius", self.window_radius) or self.window_radius),
            )
            self.progress.emit(
                f"[BackgroundRefinerTrainer] fine-tuning from base checkpoint {self.base_checkpoint}"
            )
        else:
            self.progress.emit("[BackgroundRefinerTrainer] no base checkpoint found; bootstrapping tiny local refiner from scratch")

        # Build tensors from samples
        xs, uncertains, lefts, rights = [], [], [], []
        boundary_indices, left_labels, right_labels = [], [], []
        left_states, right_states, state_before, state_after = [], [], [], []
        action_labels, query_utilities = [], []
        boundary_valids, side_valids, action_valids = [], [], []
        default_label_idx = int(min(self.label_to_idx.values())) if self.label_to_idx else 0
        inferred_input_dim: Optional[int] = None

        for sample in self.samples:
            wf = np.asarray(sample.get("window_features"), dtype=np.float32)
            sc = sample.get("scribble_channels", {})
            ws = int(sample.get("window_start", 0))
            we = int(sample.get("window_end", 0))
            length = max(0, we - ws + 1)
            if wf.ndim != 2 or wf.shape[0] != length:
                continue

            unc = np.asarray(sc.get("uncertain", np.zeros(length)), dtype=np.float32)
            lft = np.asarray(sc.get("left", np.zeros(length)), dtype=np.float32)
            rgt = np.asarray(sc.get("right", np.zeros(length)), dtype=np.float32)

            bf = int(sample.get("boundary_frame", 0))
            bi = bf - ws
            if bi < 0 or bi >= length:
                continue
            try:
                norm_feat, norm_channels, norm_energy, bi = _normalize_window_arrays(
                    wf,
                    {
                        "uncertain": unc,
                        "left": lft,
                        "right": rgt,
                    },
                    bi,
                    window_radius=target_radius,
                )
            except Exception:
                continue
            x = np.concatenate(
                [
                    norm_feat,
                    norm_channels["uncertain"][:, None],
                    norm_channels["left"][:, None],
                    norm_channels["right"][:, None],
                    norm_energy[:, None],
                ],
                axis=1,
            )
            if has_base_checkpoint:
                if x.shape[1] != input_dim:
                    continue
            else:
                if inferred_input_dim is None:
                    inferred_input_dim = int(x.shape[1])
                elif int(x.shape[1]) != int(inferred_input_dim):
                    continue

            ll = str(sample.get("left_label", ""))
            rl = str(sample.get("right_label", ""))
            boundary_valid = bool(sample.get("boundary_valid", True))
            side_valid = bool(sample.get("side_valid", True))
            action_valid = bool(sample.get("action_valid", True))
            ll_idx = self.label_to_idx.get(ll)
            rl_idx = self.label_to_idx.get(rl)
            allow_label_fallback = (not side_valid) and (not action_valid)
            if ll_idx is None:
                if allow_label_fallback:
                    ll_idx = int(rl_idx if rl_idx is not None else default_label_idx)
                else:
                    continue
            if rl_idx is None:
                if allow_label_fallback:
                    rl_idx = int(ll_idx if ll_idx is not None else default_label_idx)
                else:
                    continue

            xs.append(x)
            uncertains.append(norm_channels["uncertain"])
            lefts.append(norm_channels["left"])
            rights.append(norm_channels["right"])
            boundary_indices.append(bi)
            left_labels.append(int(ll_idx))
            right_labels.append(int(rl_idx))
            dense = np.full(int(x.shape[0]), -1, dtype=np.int64)
            if action_valid:
                dense[:bi] = int(ll_idx)
                dense[bi:] = int(rl_idx)
            action_labels.append(dense)
            boundary_valids.append(bool(boundary_valid))
            side_valids.append(bool(side_valid))
            action_valids.append(bool(action_valid))
            try:
                query_utilities.append(float(sample.get("query_utility", -1.0) or -1.0))
            except Exception:
                query_utilities.append(-1.0)

            if self.state_to_idx:
                ls = str(sample.get("left_state", ""))
                rs = str(sample.get("right_state", ""))
                ls_idx = self.state_to_idx.get(ls, -1)
                rs_idx = self.state_to_idx.get(rs, -1)
                left_states.append(ls_idx)
                right_states.append(rs_idx)
                state_before.append(ls_idx)
                state_after.append(rs_idx)

        if not xs:
            self.progress.emit("[BackgroundRefinerTrainer] no valid samples after filtering")
            self.training_finished.emit("")
            return

        if not has_base_checkpoint:
            input_dim = int(inferred_input_dim or xs[0].shape[1])

        model = TinyLocalBoundaryModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_states=num_states,
            dense_action_head=(action_weight > 0.0 or struct_weight > 0.0),
            query_head=(query_weight > 0.0),
        )
        if has_base_checkpoint:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        model.to(device)

        if has_base_checkpoint:
            inject_lora(model, rank=self.lora_rank, alpha=self.lora_alpha)
            self.progress.emit(f"[BackgroundRefinerTrainer] LoRA injected rank={self.lora_rank}")
        else:
            self.progress.emit(
                f"[BackgroundRefinerTrainer] bootstrap model initialized hidden_dim={hidden_dim} radius={target_radius}"
            )

        t_x = torch.from_numpy(np.stack(xs)).to(device)
        t_unc = torch.from_numpy(np.stack(uncertains)).to(device)
        t_lft = torch.from_numpy(np.stack(lefts)).to(device)
        t_rgt = torch.from_numpy(np.stack(rights)).to(device)
        t_bi = torch.tensor(boundary_indices, dtype=torch.long, device=device)
        t_ll = torch.tensor(left_labels, dtype=torch.long, device=device)
        t_rl = torch.tensor(right_labels, dtype=torch.long, device=device)
        t_action = torch.from_numpy(np.stack(action_labels)).to(device)
        t_query = torch.tensor(query_utilities, dtype=torch.float32, device=device)
        t_boundary_valid = torch.tensor(boundary_valids, dtype=torch.bool, device=device)
        t_side_valid = torch.tensor(side_valids, dtype=torch.bool, device=device)
        t_action_valid = torch.tensor(action_valids, dtype=torch.bool, device=device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        n = t_x.shape[0]
        model.train()
        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(n, device=device)
            total_loss = 0.0
            for start in range(0, n, max(1, min(32, n))):
                idx = perm[start:start + 32]
                optimizer.zero_grad(set_to_none=True)
                out = model(t_x[idx], t_unc[idx], t_lft[idx], t_rgt[idx])
                batch = {
                    "uncertain": t_unc[idx],
                    "left": t_lft[idx],
                    "right": t_rgt[idx],
                    "boundary_index": t_bi[idx],
                    "left_label": t_ll[idx],
                    "right_label": t_rl[idx],
                    "action_labels": t_action[idx],
                    "query_utility": t_query[idx],
                    "boundary_valid": t_boundary_valid[idx],
                    "side_valid": t_side_valid[idx],
                    "action_valid": t_action_valid[idx],
                }
                loss = float(boundary_weight) * _compute_boundary_loss(
                    out, batch, criterion, device
                ) + float(label_weight) * _compute_side_loss(
                    out, batch, criterion, device
                )
                if num_states > 0 and left_states:
                    t_ls = torch.tensor(
                        [left_states[i] for i in idx.cpu().tolist()],
                        dtype=torch.long,
                        device=device,
                    )
                    t_rs = torch.tensor(
                        [right_states[i] for i in idx.cpu().tolist()],
                        dtype=torch.long,
                        device=device,
                    )
                    t_sb = torch.tensor(
                        [state_before[i] for i in idx.cpu().tolist()],
                        dtype=torch.long,
                        device=device,
                    )
                    t_sa = torch.tensor(
                        [state_after[i] for i in idx.cpu().tolist()],
                        dtype=torch.long,
                        device=device,
                    )
                    batch["left_state"] = t_ls
                    batch["right_state"] = t_rs
                    batch["state_before"] = t_sb
                    batch["state_after"] = t_sa
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

                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(idx)
            avg = total_loss / max(1, n)
            self.progress.emit(f"[BackgroundRefinerTrainer] epoch {epoch}/{self.epochs} loss={avg:.4f}")

        # Save checkpoint
        lora_sd: Dict[str, Any] = {}
        if has_base_checkpoint:
            lora_sd = collect_lora_state_dict(model)
            merge_lora(model)

        os.makedirs(self.output_dir, exist_ok=True)
        out_name = "adapted_local_refiner.pt" if has_base_checkpoint else "bootstrapped_local_refiner.pt"
        out_path = os.path.join(self.output_dir, out_name)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=("adapted_local_refiner_" if has_base_checkpoint else "bootstrapped_local_refiner_"),
            suffix=".pt",
            dir=self.output_dir,
        )
        os.close(tmp_fd)

        new_ckpt = dict(ckpt) if has_base_checkpoint else {}
        new_ckpt["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        new_ckpt["label_to_idx"] = dict(self.label_to_idx)
        new_ckpt["idx_to_label"] = {
            int(idx): label for label, idx in self.label_to_idx.items()
        }
        new_ckpt["window_radius"] = int(target_radius)
        new_ckpt["input_dim"] = int(input_dim)
        new_ckpt["hidden_dim"] = int(hidden_dim)
        new_ckpt["dropout"] = float(dropout)
        new_ckpt["boundary_loss_weight"] = float(boundary_weight)
        new_ckpt["label_loss_weight"] = float(label_weight)
        new_ckpt["action_loss_weight"] = float(action_weight)
        new_ckpt["struct_loss_weight"] = float(struct_weight)
        new_ckpt["query_loss_weight"] = float(query_weight)
        new_ckpt["state_loss_weight"] = float(state_weight)
        new_ckpt["consistency_loss_weight"] = float(consistency_weight)
        new_ckpt["dense_action_head"] = bool(action_weight > 0.0 or struct_weight > 0.0)
        new_ckpt["query_head"] = bool(query_weight > 0.0)
        new_ckpt["num_examples"] = int(len(xs))
        new_ckpt["adaptation_samples"] = int(len(xs))
        new_ckpt["bootstrap_from_scratch"] = bool(not has_base_checkpoint)
        new_ckpt["base_checkpoint"] = str(self.base_checkpoint or "")
        if lora_sd:
            new_ckpt["lora_state_dict"] = lora_sd
            new_ckpt["lora_rank"] = self.lora_rank
            new_ckpt["lora_alpha"] = self.lora_alpha
        if self.state_to_idx:
            new_ckpt["state_to_idx"] = self.state_to_idx
            new_ckpt["num_states"] = num_states
            new_ckpt["idx_to_state"] = {
                int(idx): state for state, idx in self.state_to_idx.items()
            }
        torch.save(new_ckpt, tmp_path)
        os.replace(tmp_path, out_path)
        self._result_path = out_path
        if has_base_checkpoint:
            self.progress.emit(f"[BackgroundRefinerTrainer] saved adapted checkpoint to {out_path}")
        else:
            self.progress.emit(f"[BackgroundRefinerTrainer] saved bootstrapped checkpoint to {out_path}")
        self.training_finished.emit(out_path)
