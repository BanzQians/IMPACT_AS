from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LocalRefinerInput:
    window_start: int
    window_end: int
    boundary_energy: Optional[np.ndarray] = None
    scribble_channels: Dict[str, np.ndarray] = field(default_factory=dict)
    left_candidates: Sequence[Tuple[str, Optional[float]]] = field(default_factory=list)
    right_candidates: Sequence[Tuple[str, Optional[float]]] = field(default_factory=list)
    window_features: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class LocalRefinerOutput:
    boundary_probs: np.ndarray
    boundary_frame: Optional[int]
    left_label: str = ""
    right_label: str = ""
    confidence: float = 0.0
    extras: Dict[str, object] = field(default_factory=dict)


class BaseLocalBoundaryRefiner:
    def refine(self, item: LocalRefinerInput) -> LocalRefinerOutput:
        raise NotImplementedError


class HeuristicLocalBoundaryRefiner(BaseLocalBoundaryRefiner):
    def refine(self, item: LocalRefinerInput) -> LocalRefinerOutput:
        start = int(item.window_start)
        end = int(item.window_end)
        length = max(0, end - start + 1)
        probs = np.zeros(length, dtype=np.float32)
        if length <= 0:
            return LocalRefinerOutput(boundary_probs=probs, boundary_frame=None)

        uncertain, left, right = _channel_triplet(item, length)

        search_mask = uncertain.copy()
        if float(np.max(search_mask)) <= 0.0:
            search_mask = np.maximum(left, right)
        if float(np.max(search_mask)) <= 0.0:
            search_mask = np.ones(length, dtype=np.float32)

        energy_arr = _normalized_energy(item.boundary_energy, length)

        location_prior = np.maximum(search_mask, 0.0)
        if float(np.max(location_prior)) > 0.0:
            location_prior /= float(np.max(location_prior))

        transition = np.zeros(length, dtype=np.float32)
        if float(np.max(left)) > 0.0 or float(np.max(right)) > 0.0:
            for idx in range(length):
                left_support = float(np.mean(left[:idx])) if idx > 0 else 0.0
                right_support = (
                    float(np.mean(right[idx:])) if idx < length else 0.0
                )
                transition[idx] = left_support * right_support
            if float(np.max(transition)) > 0.0:
                transition /= float(np.max(transition))

        score = (
            0.55 * energy_arr * np.maximum(search_mask, 0.15)
            + 0.25 * location_prior
            + 0.35 * transition
        )
        if float(np.max(score)) > 0.0:
            idx = int(np.argmax(score))
            probs = score / max(1e-6, float(np.sum(score)))
        else:
            fallback = (
                location_prior
                if float(np.max(location_prior)) > 0.0
                else np.ones(length, dtype=np.float32)
            )
            idx = int(np.argmax(fallback)) if fallback.size else length // 2
            probs[idx] = 1.0
        boundary_frame = start + idx
        left_label = _top_label(item.left_candidates)
        right_label = _top_label(item.right_candidates)
        confidence = float(
            min(
                1.0,
                max(
                    0.0,
                    float(np.max(probs)) * 1.8
                    + float(transition[idx] if 0 <= idx < len(transition) else 0.0)
                    * 0.2
                    + (0.1 if float(np.max(uncertain)) > 0.0 else 0.0),
                ),
            )
        )
        return LocalRefinerOutput(
            boundary_probs=probs,
            boundary_frame=boundary_frame,
            left_label=left_label,
            right_label=right_label,
            confidence=confidence,
            extras={
                "mode": "heuristic",
                "left_coverage": float(np.mean(left)) if left.size else 0.0,
                "right_coverage": float(np.mean(right)) if right.size else 0.0,
                "uncertain_coverage": float(np.mean(uncertain)) if uncertain.size else 0.0,
                "transition_peak": float(np.max(transition)) if transition.size else 0.0,
            },
        )


class CheckpointLocalBoundaryRefiner(BaseLocalBoundaryRefiner):
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "auto",
        fallback: Optional[BaseLocalBoundaryRefiner] = None,
    ) -> None:
        self.checkpoint_path = os.path.abspath(
            os.path.expanduser(str(checkpoint_path or "").strip())
        )
        self.device_request = str(device or "auto").strip().lower() or "auto"
        self.fallback = fallback or HeuristicLocalBoundaryRefiner()
        self._model = None
        self._torch = None
        self._device = None
        self._input_dim = 0
        self._window_radius = 0
        self._num_states = 0
        self._idx_to_label: Dict[int, str] = {}
        self._load_error = ""
        self._load_checkpoint()

    @property
    def ready(self) -> bool:
        return self._model is not None and self._torch is not None

    @property
    def load_error(self) -> str:
        return str(self._load_error or "")

    def refine(self, item: LocalRefinerInput) -> LocalRefinerOutput:
        if not self.ready:
            return self._refine_with_fallback(item, self.load_error or "checkpoint_unavailable")
        try:
            x, uncertain, left, right = self._build_model_inputs(item)
        except Exception as ex:
            return self._refine_with_fallback(item, f"input_error:{ex}")
        try:
            torch = self._torch
            model = self._model
            device = self._device
            with torch.no_grad():
                batch_x = torch.from_numpy(x[None, ...]).to(device)
                batch_uncertain = torch.from_numpy(uncertain[None, ...]).to(device)
                batch_left = torch.from_numpy(left[None, ...]).to(device)
                batch_right = torch.from_numpy(right[None, ...]).to(device)
                out = model(batch_x, batch_uncertain, batch_left, batch_right)
                boundary_probs = (
                    torch.softmax(out["boundary_logits"], dim=1)[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                left_probs = (
                    torch.softmax(out["left_logits"], dim=1)[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                right_probs = (
                    torch.softmax(out["right_logits"], dim=1)[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
        except Exception as ex:
            return self._refine_with_fallback(item, f"inference_error:{ex}")

        if boundary_probs.size <= 0:
            return self._refine_with_fallback(item, "empty_boundary_probs")

        boundary_idx = int(np.argmax(boundary_probs))
        boundary_frame = int(item.window_start) + boundary_idx
        left_idx, left_score = self._choose_label_index(left_probs, item.left_candidates)
        right_idx, right_score = self._choose_label_index(right_probs, item.right_candidates)
        left_label = self._idx_to_label.get(int(left_idx), "")
        right_label = self._idx_to_label.get(int(right_idx), "")
        boundary_score = float(boundary_probs[boundary_idx]) if boundary_probs.size else 0.0
        confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    0.55 * boundary_score + 0.225 * float(left_score) + 0.225 * float(right_score),
                ),
            )
        )
        return LocalRefinerOutput(
            boundary_probs=boundary_probs,
            boundary_frame=boundary_frame,
            left_label=str(left_label or ""),
            right_label=str(right_label or ""),
            confidence=confidence,
            extras={
                "mode": "checkpoint",
                "checkpoint_path": self.checkpoint_path,
                "device": str(self._device),
                "boundary_score": boundary_score,
                "left_score": float(left_score),
                "right_score": float(right_score),
                "window_radius": int(self._window_radius),
            },
        )

    def reload_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """Hot-reload a checkpoint (or LoRA overlay). Returns True on success."""
        prev_state = {
            "checkpoint_path": self.checkpoint_path,
            "model": self._model,
            "torch": self._torch,
            "device": self._device,
            "input_dim": self._input_dim,
            "window_radius": self._window_radius,
            "num_states": self._num_states,
            "idx_to_label": dict(self._idx_to_label),
            "load_error": self._load_error,
        }
        if checkpoint_path:
            self.checkpoint_path = os.path.abspath(
                os.path.expanduser(str(checkpoint_path).strip())
            )
        self._load_checkpoint()
        if self.ready:
            return True
        self.checkpoint_path = prev_state["checkpoint_path"]
        self._model = prev_state["model"]
        self._torch = prev_state["torch"]
        self._device = prev_state["device"]
        self._input_dim = int(prev_state["input_dim"])
        self._window_radius = int(prev_state["window_radius"])
        self._num_states = int(prev_state["num_states"])
        self._idx_to_label = dict(prev_state["idx_to_label"])
        self._load_error = str(prev_state["load_error"] or "")
        return False

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            self._load_error = f"checkpoint_not_found:{self.checkpoint_path}"
            return
        try:
            torch = _import_torch()
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            if not isinstance(ckpt, dict):
                raise ValueError("checkpoint root must be a dict")
            state_dict = ckpt.get("state_dict")
            if not isinstance(state_dict, dict):
                raise KeyError("checkpoint missing state_dict")
            input_dim = int(ckpt.get("input_dim", 0) or 0)
            hidden_dim = int(ckpt.get("hidden_dim", 0) or 0)
            dropout = float(ckpt.get("dropout", 0.1) or 0.0)
            temporal_kernel_size = int(ckpt.get("temporal_kernel_size", 1) or 1)
            num_states = int(ckpt.get("num_states", 0) or 0)
            if num_states <= 0:
                state_to_idx = ckpt.get("state_to_idx") or {}
                if isinstance(state_to_idx, dict):
                    num_states = len(state_to_idx)
            idx_to_label_raw = ckpt.get("idx_to_label") or {}
            if not isinstance(idx_to_label_raw, dict) or not idx_to_label_raw:
                raise KeyError("checkpoint missing idx_to_label")
            idx_to_label = {
                int(idx): str(label or "")
                for idx, label in idx_to_label_raw.items()
                if str(label or "").strip()
            }
            if input_dim <= 0 or hidden_dim <= 0 or not idx_to_label:
                raise ValueError("checkpoint metadata is incomplete")
            model = _build_tiny_local_boundary_model(
                int(input_dim),
                len(idx_to_label),
                int(hidden_dim),
                float(dropout),
                num_states=int(num_states),
                temporal_kernel_size=int(max(1, int(temporal_kernel_size))),
            )
            model.load_state_dict(state_dict, strict=False)
            # Apply LoRA overlay if present in checkpoint
            lora_sd = ckpt.get("lora_state_dict")
            lora_rank = int(ckpt.get("lora_rank", 0) or 0)
            if isinstance(lora_sd, dict) and lora_sd and lora_rank > 0:
                lora_alpha = float(ckpt.get("lora_alpha", 1.0) or 1.0)
                inject_lora(model, rank=lora_rank, alpha=lora_alpha)
                load_lora_state_dict(model, lora_sd)
                merge_lora(model)
            device = _resolve_torch_device(torch, self.device_request)
            model.to(device)
            model.eval()
            self._torch = torch
            self._device = device
            self._model = model
            self._input_dim = int(input_dim)
            self._window_radius = int(ckpt.get("window_radius", 0) or 0)
            self._num_states = int(num_states)
            self._idx_to_label = dict(idx_to_label)
            self._load_error = ""
        except Exception as ex:
            self._model = None
            self._torch = None
            self._device = None
            self._num_states = 0
            self._load_error = f"{type(ex).__name__}:{ex}"

    def _build_model_inputs(
        self, item: LocalRefinerInput
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start = int(item.window_start)
        end = int(item.window_end)
        length = max(0, end - start + 1)
        if length <= 0:
            raise ValueError("empty_window")
        features = np.asarray(item.window_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != length:
            raise ValueError("missing_or_misaligned_window_features")
        uncertain, left, right = _channel_triplet(item, length)
        energy = _normalized_energy(item.boundary_energy, length)
        x = np.concatenate(
            [
                features.astype(np.float32),
                uncertain[:, None],
                left[:, None],
                right[:, None],
                energy[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        if int(x.shape[1]) != int(self._input_dim):
            raise ValueError(f"input_dim_mismatch:{x.shape[1]}!=expected{self._input_dim}")
        return x, uncertain.astype(np.float32), left.astype(np.float32), right.astype(np.float32)

    def _choose_label_index(
        self,
        probs: np.ndarray,
        candidates: Sequence[Tuple[str, Optional[float]]],
    ) -> Tuple[int, float]:
        if probs.size <= 0:
            return 0, 0.0
        allowed = []
        for row in candidates or []:
            if not row:
                continue
            label = str(row[0] or "").strip()
            if not label:
                continue
            for idx, name in self._idx_to_label.items():
                if str(name or "").strip() == label:
                    allowed.append(int(idx))
                    break
        if allowed:
            best_idx = max(allowed, key=lambda idx: float(probs[int(idx)]))
            return int(best_idx), float(probs[int(best_idx)])
        best_idx = int(np.argmax(probs))
        return best_idx, float(probs[best_idx])

    def _refine_with_fallback(
        self, item: LocalRefinerInput, reason: str
    ) -> LocalRefinerOutput:
        fallback_out = self.fallback.refine(item)
        extras = dict(fallback_out.extras or {})
        extras["fallback_reason"] = str(reason or "unknown")
        extras["requested_checkpoint"] = self.checkpoint_path
        extras.setdefault("mode", "heuristic")
        return LocalRefinerOutput(
            boundary_probs=np.asarray(fallback_out.boundary_probs, dtype=np.float32),
            boundary_frame=fallback_out.boundary_frame,
            left_label=str(fallback_out.left_label or ""),
            right_label=str(fallback_out.right_label or ""),
            confidence=float(fallback_out.confidence),
            extras=extras,
        )


def _channel_triplet(
    item: LocalRefinerInput, length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uncertain = np.asarray(
        item.scribble_channels.get("uncertain"), dtype=np.float32
    ).reshape(-1)
    left = np.asarray(item.scribble_channels.get("left"), dtype=np.float32).reshape(-1)
    right = np.asarray(item.scribble_channels.get("right"), dtype=np.float32).reshape(-1)
    if uncertain.shape[0] != length:
        uncertain = np.zeros(length, dtype=np.float32)
    if left.shape[0] != length:
        left = np.zeros(length, dtype=np.float32)
    if right.shape[0] != length:
        right = np.zeros(length, dtype=np.float32)
    return uncertain, left, right


def _normalized_energy(energy: Optional[np.ndarray], length: int) -> np.ndarray:
    if energy is None or np.asarray(energy).shape[0] != length:
        return np.zeros(length, dtype=np.float32)
    energy_arr = np.maximum(np.asarray(energy, dtype=np.float32).reshape(-1), 0.0)
    if float(np.max(energy_arr)) > 0.0:
        energy_arr = energy_arr / float(np.max(energy_arr))
    return energy_arr.astype(np.float32)


def _top_label(rows: Sequence[Tuple[str, Optional[float]]]) -> str:
    if not rows:
        return ""
    return str(rows[0][0] or "")


def _import_torch():
    try:
        import torch
    except Exception as ex:  # pragma: no cover - environment dependent
        raise RuntimeError(f"torch import failed: {ex}") from ex
    return torch


def _resolve_torch_device(torch: Any, requested: str):
    device_name = str(requested or "auto").strip().lower()
    if device_name in ("", "auto"):
        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            device_name = "cuda"
        else:
            device_name = "cpu"
    return torch.device(device_name)


class LoRALinear:
    """Low-Rank Adaptation wrapper for nn.Linear layers.

    Injects trainable low-rank matrices A and B alongside a frozen base Linear,
    so that the effective weight becomes  W + (B @ A) * scaling.
    """

    @staticmethod
    def wrap(linear, rank: int = 4, alpha: float = 1.0):
        """Replace *linear* (nn.Linear) in-place attributes, returning it."""
        torch = _import_torch()
        nn = torch.nn
        in_f, out_f = linear.in_features, linear.out_features
        rank = max(1, min(rank, min(in_f, out_f)))
        scaling = alpha / rank

        linear.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        linear.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(linear.lora_A, a=5 ** 0.5)
        # B starts at zero so initial output is unchanged
        linear.lora_scaling = scaling
        linear.lora_rank = rank

        # freeze base weight
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

        # monkey-patch forward
        _orig_forward = linear.forward

        def _lora_forward(x, _base=_orig_forward, _lin=linear):
            base_out = _base(x)
            lora_out = (x @ _lin.lora_A.T) @ _lin.lora_B.T
            return base_out + lora_out * _lin.lora_scaling

        linear.forward = _lora_forward
        linear._lora_orig_forward = _orig_forward
        return linear

    @staticmethod
    def merge(linear):
        """Fold LoRA weights into the base weight and remove LoRA params."""
        if not hasattr(linear, "lora_A"):
            return linear
        torch = _import_torch()
        with torch.no_grad():
            delta = (linear.lora_B @ linear.lora_A) * linear.lora_scaling
            linear.weight.add_(delta)
        # restore original forward
        if hasattr(linear, "_lora_orig_forward"):
            linear.forward = linear._lora_orig_forward
            del linear._lora_orig_forward
        for attr in ("lora_A", "lora_B", "lora_scaling", "lora_rank"):
            if hasattr(linear, attr):
                delattr(linear, attr)
        linear.weight.requires_grad_(True)
        if linear.bias is not None:
            linear.bias.requires_grad_(True)
        return linear

    @staticmethod
    def has_lora(linear) -> bool:
        return hasattr(linear, "lora_A") and hasattr(linear, "lora_B")


def inject_lora(model, rank: int = 4, alpha: float = 1.0, target_modules: Optional[Sequence[str]] = None):
    """Inject LoRA into all nn.Linear layers of *model* (or only named targets).

    Returns list of (name, module) pairs that were wrapped.
    """
    torch = _import_torch()
    nn = torch.nn
    wrapped = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if LoRALinear.has_lora(module):
            continue
        if target_modules is not None:
            leaf = name.rsplit(".", 1)[-1] if "." in name else name
            if leaf not in target_modules and name not in target_modules:
                continue
        LoRALinear.wrap(module, rank=rank, alpha=alpha)
        wrapped.append((name, module))
    return wrapped


def merge_lora(model):
    """Merge all LoRA weights back into base weights across *model*."""
    torch = _import_torch()
    nn = torch.nn
    merged = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and LoRALinear.has_lora(module):
            LoRALinear.merge(module)
            merged.append(name)
    return merged


def collect_lora_state_dict(model) -> Dict[str, Any]:
    """Return a state dict containing only LoRA parameters."""
    torch = _import_torch()
    lora_sd = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            prefix = name + "." if name else ""
            lora_sd[prefix + "lora_A"] = module.lora_A.detach().cpu().clone()
            lora_sd[prefix + "lora_B"] = module.lora_B.detach().cpu().clone()
    return lora_sd


def load_lora_state_dict(model, lora_sd: Dict[str, Any]) -> int:
    """Load LoRA parameters from a state dict into an already-injected model."""
    torch = _import_torch()
    loaded = 0
    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        key_a = prefix + "lora_A"
        key_b = prefix + "lora_B"
        if key_a in lora_sd and key_b in lora_sd and hasattr(module, "lora_A"):
            module.lora_A.data.copy_(lora_sd[key_a])
            module.lora_B.data.copy_(lora_sd[key_b])
            loaded += 1
    return loaded


def _build_tiny_local_boundary_model(
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    dropout: float,
    num_states: int = 0,
    temporal_kernel_size: int = 1,
):
    torch = _import_torch()
    nn = torch.nn

    class TinyLocalBoundaryModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            kernel = max(1, int(temporal_kernel_size))
            if kernel % 2 == 0:
                kernel += 1
            self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
            self.temporal_conv = nn.Conv1d(
                int(hidden_dim),
                int(hidden_dim),
                kernel_size=int(kernel),
                padding=int(kernel // 2),
                groups=int(hidden_dim),
                bias=False,
            )
            nn.init.zeros_(self.temporal_conv.weight)
            self.encoder = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(inplace=False),
                nn.Dropout(float(dropout)),
            )
            self.boundary_head = nn.Linear(int(hidden_dim), 1)
            ctx_dim = int(hidden_dim) * 4
            self.left_head = nn.Sequential(
                nn.Linear(ctx_dim, int(hidden_dim)),
                nn.ReLU(inplace=False),
                nn.Linear(int(hidden_dim), int(num_classes)),
            )
            self.right_head = nn.Sequential(
                nn.Linear(ctx_dim, int(hidden_dim)),
                nn.ReLU(inplace=False),
                nn.Linear(int(hidden_dim), int(num_classes)),
            )
            self.num_states = int(num_states)
            if self.num_states > 0:
                self.left_state_head = nn.Sequential(
                    nn.Linear(ctx_dim, int(hidden_dim)),
                    nn.ReLU(inplace=False),
                    nn.Linear(int(hidden_dim), int(self.num_states)),
                )
                self.right_state_head = nn.Sequential(
                    nn.Linear(ctx_dim, int(hidden_dim)),
                    nn.ReLU(inplace=False),
                    nn.Linear(int(hidden_dim), int(self.num_states)),
                )

        @staticmethod
        def _masked_mean(
            hidden: Any,
            mask: Any,
            fallback: Any,
        ) -> Any:
            weights = mask.unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1e-6)
            pooled = (hidden * weights).sum(dim=1) / denom
            valid = (mask.sum(dim=1, keepdim=True) > 0).expand_as(pooled)
            return torch.where(valid, pooled, fallback)

        def forward(
            self,
            x: Any,
            uncertain: Any,
            left: Any,
            right: Any,
        ) -> Dict[str, Any]:
            hidden = self.input_proj(x)
            conv = self.temporal_conv(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = self.encoder(torch.relu(hidden + conv))
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
            context = torch.cat(
                [left_pool, right_pool, uncertain_pool, global_pool], dim=1
            )
            out = {
                "boundary_logits": boundary_logits,
                "left_logits": self.left_head(context),
                "right_logits": self.right_head(context),
            }
            if self.num_states > 0:
                out["left_state_logits"] = self.left_state_head(context)
                out["right_state_logits"] = self.right_state_head(context)
            return out

    return TinyLocalBoundaryModel()
