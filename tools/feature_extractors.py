# -*- coding: utf-8 -*-
"""
Backbone-agnostic feature extraction helpers for per-frame video features.
"""
from __future__ import annotations

import importlib.util
import json
import os
import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".m4v")
COMMON_FEATURE_DIMS = {
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1408,
    1536,
    2048,
    4096,
}
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.RLock()
_LOCAL_MODULE_CACHE = {}

try:
    _autocast = torch.amp.autocast
    _autocast_kwargs = {"device_type": "cuda"}
except AttributeError:
    _autocast = torch.cuda.amp.autocast
    _autocast_kwargs = {}


def _resize_shorter(batch: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    if batch.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got {batch.shape}")
    _, _, h, w = batch.shape
    if min(h, w) == size:
        return batch
    if h < w:
        new_h = size
        new_w = int(round(w * size / h))
    else:
        new_w = size
        new_h = int(round(h * size / w))
    return F.interpolate(batch, size=(new_h, new_w), mode=mode, align_corners=False)


def _center_crop(batch: torch.Tensor, size: int) -> torch.Tensor:
    if batch.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got {batch.shape}")
    _, _, h, w = batch.shape
    if h == size and w == size:
        return batch
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return batch[:, :, top : top + size, left : left + size]


def default_feature_target_dim(backbone: Optional[str]) -> Optional[int]:
    key = str(backbone or "").strip().lower().replace("-", "_")
    if key in {
        "i3d",
        "i3d_inception",
        "i3d_inception_rgb",
        "i3d_rgb",
        "i3d_r50",
        "i3d_r50_legacy",
        "i3d_legacy_r50",
    }:
        try:
            return max(1, int(os.environ.get("I3D_TARGET_DIM", "1024") or 1024))
        except Exception:
            return 1024
    return None


def _build_fixed_feature_projector(
    input_dim: int,
    output_dim: int,
    *,
    source: str,
) -> Optional[Dict[str, object]]:
    try:
        in_dim = int(input_dim)
        out_dim = int(output_dim)
    except Exception:
        return None
    if in_dim <= 0 or out_dim <= 0 or in_dim == out_dim:
        return None
    seed = ((in_dim * 73856093) ^ (out_dim * 19349663) ^ 0x5F3759DF) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((in_dim, out_dim), dtype=np.float32)
    weight = np.asarray(weight / max(np.sqrt(float(out_dim)), 1.0), dtype=np.float32)
    bias = np.zeros((out_dim,), dtype=np.float32)
    return {
        "version": 1,
        "source": str(source or "feature_projection"),
        "mode": "fixed_random",
        "input_dim": int(in_dim),
        "output_dim": int(out_dim),
        "weight": weight,
        "bias": bias,
    }


def _project_feature_dim(
    features: np.ndarray,
    output_dim: int,
    *,
    source: str,
) -> Tuple[np.ndarray, dict]:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {tuple(arr.shape)}")
    in_dim = int(arr.shape[1])
    out_dim = int(output_dim)
    if in_dim == out_dim:
        return arr, {
            "projected": False,
            "projected_from_dim": in_dim,
            "feature_dim": out_dim,
        }
    projector = _build_fixed_feature_projector(in_dim, out_dim, source=source)
    if not projector:
        raise RuntimeError(f"Failed to build projector for feature dims {in_dim}->{out_dim}.")
    weight = np.asarray(projector.get("weight"), dtype=np.float32)
    bias = np.asarray(projector.get("bias"), dtype=np.float32)
    projected = np.asarray(arr @ weight + bias, dtype=np.float32)
    return projected, {
        "projected": True,
        "projected_from_dim": in_dim,
        "feature_dim": out_dim,
        "projection_source": str(projector.get("source") or source),
        "projection_mode": str(projector.get("mode") or "fixed_random"),
    }


def _should_transpose_feature_array(features: np.ndarray) -> bool:
    if features.ndim != 2:
        return False
    rows, cols = int(features.shape[0]), int(features.shape[1])
    if cols in COMMON_FEATURE_DIMS and rows not in COMMON_FEATURE_DIMS:
        return False
    if rows in COMMON_FEATURE_DIMS and cols not in COMMON_FEATURE_DIMS:
        return True
    if rows in COMMON_FEATURE_DIMS and cols > max(rows * 2, rows + 32):
        return True
    return False


def normalize_external_feature_array(
    features: np.ndarray,
    *,
    source: str = "external",
    target_dim: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got shape {tuple(arr.shape)}")
    original_shape = tuple(int(x) for x in arr.shape)
    transposed = False
    if _should_transpose_feature_array(arr):
        arr = np.asarray(arr.T, dtype=np.float32)
        transposed = True
    info = {
        "original_shape": original_shape,
        "normalized_shape": tuple(int(x) for x in arr.shape),
        "transposed": bool(transposed),
        "layout": "BTD",
        "feature_dim": int(arr.shape[1]),
        "projected": False,
        "projected_from_dim": int(arr.shape[1]),
    }
    if target_dim is not None and int(target_dim) > 0 and int(arr.shape[1]) != int(target_dim):
        arr, proj_meta = _project_feature_dim(
            arr,
            int(target_dim),
            source=f"{source}_dim_align",
        )
        info.update(proj_meta)
        info["normalized_shape"] = tuple(int(x) for x in arr.shape)
    return np.asarray(arr, dtype=np.float32), info


def load_external_features(
    source_path: str,
    *,
    target_dim: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    path = os.path.abspath(os.path.expanduser(str(source_path or "").strip()))
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"External feature file not found: {source_path}")
    arr = np.load(path, allow_pickle=False)
    features, info = normalize_external_feature_array(
        arr,
        source=os.path.splitext(os.path.basename(path))[0] or "external",
        target_dim=target_dim,
    )
    meta_path = os.path.join(os.path.dirname(path), "meta.json")
    meta: Dict[str, object] = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                meta.update(loaded)
        except Exception:
            pass
    meta.update(
        {
            "imported_from": path,
            "feature_dim": int(features.shape[1]),
            "frame_stride": int(meta.get("frame_stride", 1) or 1),
            "num_frames": int(meta.get("num_frames", features.shape[0]) or features.shape[0]),
            "layout": "BTD",
        }
    )
    if info.get("transposed"):
        meta["source_layout"] = "DBT"
    if info.get("projected"):
        meta["projected_from_dim"] = int(info.get("projected_from_dim", 0) or 0)
        meta["projection_mode"] = str(info.get("projection_mode") or "fixed_random")
        meta["projection_source"] = str(info.get("projection_source") or "")
    return features, meta


def _to_tensor(
    frames: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
) -> torch.Tensor:
    if isinstance(frames, torch.Tensor):
        tensor = frames
    elif isinstance(frames, np.ndarray):
        tensor = torch.from_numpy(frames)
    elif isinstance(frames, list):
        if not frames:
            raise ValueError("Empty frame list.")
        tensor = torch.from_numpy(np.stack(frames, axis=0))
    else:
        raise TypeError(f"Unsupported frame type: {type(frames)}")

    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (T,C,H,W), got {tensor.shape}")

    if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    return tensor


class BaseFeatureExtractor(ABC):
    name = "base"
    feature_dim: int = 0
    resize_shorter: int = 224
    crop_size: int = 224
    interpolate_mode: str = "bilinear"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self, device: Optional[str] = None, batch_size: int = 128, use_fp16: bool = True
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = max(1, int(batch_size))
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        self.model = self._get_cached_model()
        self.model.eval()
        self.model.requires_grad_(False)

    def _model_cache_key(self) -> Tuple[str, str]:
        return (str(getattr(self, "name", self.__class__.__name__)), str(self.device))

    def _get_cached_model(self) -> torch.nn.Module:
        key = self._model_cache_key()
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                self._reused_model = True
                return cached
            model = self._load_model().to(self.device)
            model.eval()
            model.requires_grad_(False)
            _MODEL_CACHE[key] = model
            self._reused_model = False
            return model

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _preprocess(self, batch: torch.Tensor) -> torch.Tensor:
        batch = _resize_shorter(batch, self.resize_shorter, mode=self.interpolate_mode)
        batch = _center_crop(batch, self.crop_size)
        mean = torch.as_tensor(self.mean, device=batch.device).view(1, 3, 1, 1)
        std = torch.as_tensor(self.std, device=batch.device).view(1, 3, 1, 1)
        return (batch - mean) / std

    def _encode_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dtype != torch.float32:
            batch = batch.float()
        if batch.max() > 1.5:
            batch = batch / 255.0
        batch = batch.to(self.device, non_blocking=True)
        batch = self._preprocess(batch)
        with torch.no_grad():
            if self.use_fp16 and self.device.type == "cuda":
                with _autocast(**_autocast_kwargs):
                    out = self._forward(batch)
            else:
                out = self._forward(batch)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):
            out = next(iter(out.values()))
        if out.dim() == 3:
            out = out.mean(dim=1)
        return out.detach().cpu()

    def extract_frames(
        self, frames: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> np.ndarray:
        tensor = _to_tensor(frames)
        if tensor.numel() == 0:
            return np.zeros((0, int(self.feature_dim)), dtype=np.float32)
        feats = []
        for i in range(0, tensor.shape[0], self.batch_size):
            batch = tensor[i : i + self.batch_size]
            feats.append(self._encode_batch(batch))
        if not feats:
            return np.zeros((0, int(self.feature_dim)), dtype=np.float32)
        feat_tensor = torch.cat(feats, dim=0)
        return feat_tensor.numpy().astype(np.float32, copy=False)

    def extract_from_video(
        self,
        video_path: str,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[np.ndarray, List[int], int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        use_total = total if (max_frames is None) else min(total, int(max_frames))
        if use_total <= 0:
            use_total = None

        stride = max(1, int(frame_stride))
        expected = 0
        if use_total is not None:
            expected = (use_total + stride - 1) // stride
        feats = []
        picked_indices: List[int] = []
        batch_frames: List[np.ndarray] = []

        if progress_cb:
            try:
                progress_cb(0, expected)
            except Exception:
                pass

        idx = 0
        while True:
            if use_total is not None and idx >= use_total:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if stride > 1 and (idx % stride) != 0:
                idx += 1
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame)
            picked_indices.append(idx)
            if len(batch_frames) >= self.batch_size:
                batch = _to_tensor(batch_frames)
                feats.append(self._encode_batch(batch))
                batch_frames = []
                if progress_cb:
                    try:
                        progress_cb(len(picked_indices), expected)
                    except Exception:
                        pass
            idx += 1

        if batch_frames:
            batch = _to_tensor(batch_frames)
            feats.append(self._encode_batch(batch))
            if progress_cb:
                try:
                    progress_cb(len(picked_indices), expected)
                except Exception:
                    pass

        cap.release()
        total_frames = total if total > 0 else idx

        if not feats:
            empty = np.zeros((0, int(self.feature_dim)), dtype=np.float32)
            return empty, picked_indices, total_frames

        feat_tensor = torch.cat(feats, dim=0)
        if progress_cb:
            try:
                progress_cb(len(picked_indices), expected or len(picked_indices))
            except Exception:
                pass
        return (
            feat_tensor.numpy().astype(np.float32, copy=False),
            picked_indices,
            total_frames,
        )


class DinoV2FeatureExtractor(BaseFeatureExtractor):
    name = "dinov2_vitb14"
    feature_dim = 768
    resize_shorter = 224
    crop_size = 224
    interpolate_mode = "bicubic"

    def _load_model(self) -> torch.nn.Module:
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.model(batch)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            if "x_norm_patchtokens" in out:
                return out["x_norm_patchtokens"].mean(dim=1)
            if "x_norm" in out:
                return out["x_norm"]
            return next(iter(out.values()))
        return out


class ResNet50FeatureExtractor(BaseFeatureExtractor):
    name = "resnet50"
    feature_dim = 2048
    resize_shorter = 256
    crop_size = 224
    interpolate_mode = "bilinear"

    def _load_model(self) -> torch.nn.Module:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        return model

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)


class I3DFeatureExtractor(BaseFeatureExtractor):
    name = "i3d_inception_rgb"
    feature_dim = 1024
    native_feature_dim = 1024
    resize_shorter = 256
    crop_size = 224
    interpolate_mode = "bilinear"

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 8,
        use_fp16: bool = True,
    ):
        self.clip_len = max(4, int(os.environ.get("I3D_CLIP_LEN", "16") or 16))
        self.target_feature_dim = int(default_feature_target_dim("i3d") or 1024)
        self.native_feature_dim = 1024
        ckpt = str(os.environ.get("I3D_CKPT", "") or "").strip()
        if ckpt:
            self.ckpt_path = os.path.abspath(os.path.expanduser(ckpt))
        else:
            self.ckpt_path = os.path.join(
                self._repo_root(),
                "external",
                "pytorch-i3d",
                "models",
                "rgb_imagenet.pt",
            )
        super().__init__(
            device=device,
            batch_size=max(1, int(batch_size)),
            use_fp16=use_fp16,
        )
        self.feature_dim = int(self.target_feature_dim)

    @staticmethod
    def _repo_root() -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    @classmethod
    def _load_local_module(cls, module_name: str, file_path: str):
        cache_key = f"{module_name}:{os.path.abspath(file_path)}"
        cached = _LOCAL_MODULE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOCAL_MODULE_CACHE[cache_key] = module
        return module

    def _model_cache_key(self) -> Tuple[str, str, str]:
        return (
            str(getattr(self, "name", self.__class__.__name__)),
            str(self.device),
            str(self.ckpt_path),
        )

    def _load_model(self) -> torch.nn.Module:
        repo_dir = os.path.join(self._repo_root(), "external", "pytorch-i3d")
        module_path = os.path.join(repo_dir, "pytorch_i3d.py")
        if not os.path.isfile(module_path):
            raise RuntimeError(
                "Missing local pytorch-i3d vendor repo. Expected:\n"
                f"- {os.path.join(self._repo_root(), 'external', 'pytorch-i3d')}"
            )
        mod = self._load_local_module("vendor_pytorch_i3d", module_path)
        InceptionI3d = getattr(mod, "InceptionI3d", None)
        if InceptionI3d is None:
            raise RuntimeError("Failed to import InceptionI3d from pytorch-i3d.")
        if not os.path.isfile(self.ckpt_path):
            raise RuntimeError(f"Missing I3D checkpoint: {self.ckpt_path}")
        model = InceptionI3d(400, in_channels=3)
        state = torch.load(self.ckpt_path, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported I3D checkpoint format: {type(state)}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Failed to load Inception I3D checkpoint cleanly. "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        return model

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.extract_features(batch)

    def _preprocess_video_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B,C,T,H,W), got {batch.shape}")
        if batch.dtype != torch.float32:
            batch = batch.float()
        if batch.max() > 1.5:
            batch = batch / 255.0
        b, c, t, h, w = batch.shape
        flat = batch.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        flat = flat.to(self.device, non_blocking=True)
        flat = _resize_shorter(flat, self.resize_shorter, mode=self.interpolate_mode)
        flat = _center_crop(flat, self.crop_size)
        flat = flat.mul(2.0).sub(1.0)
        h2, w2 = flat.shape[-2:]
        return flat.view(b, t, c, h2, w2).permute(0, 2, 1, 3, 4).contiguous()

    def _encode_clip_batch(self, batch: torch.Tensor) -> np.ndarray:
        batch = self._preprocess_video_batch(batch)
        with torch.no_grad():
            if self.use_fp16 and self.device.type == "cuda":
                with _autocast(**_autocast_kwargs):
                    out = self._forward(batch)
            else:
                out = self._forward(batch)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):
            out = next(iter(out.values()))
        out = out.reshape(out.shape[0], -1)
        out_np = out.detach().cpu().numpy().astype(np.float32, copy=False)
        projected, _ = _project_feature_dim(
            out_np,
            int(self.target_feature_dim),
            source="i3d_inception_feature_projection",
        )
        return projected.astype(np.float32, copy=False)

    def _resize_frame_for_clip_buffer(self, frame: np.ndarray) -> np.ndarray:
        h, w = int(frame.shape[0]), int(frame.shape[1])
        size = int(self.resize_shorter)
        if min(h, w) == size:
            return frame
        if h < w:
            new_h = size
            new_w = int(round(w * size / h))
        else:
            new_w = size
            new_h = int(round(h * size / w))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def extract_from_video(
        self,
        video_path: str,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[np.ndarray, List[int], int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        use_total = total if (max_frames is None) else min(total, int(max_frames))
        if use_total <= 0:
            use_total = None
        stride = max(1, int(frame_stride))
        expected = 0
        if use_total is not None:
            expected = (use_total + stride - 1) // stride

        picked_indices: List[int] = []
        buffered_frames = deque()
        pending_clips: List[np.ndarray] = []
        feats: List[np.ndarray] = []
        encoded = 0
        sampled = 0
        left_ctx = self.clip_len // 2
        right_ctx = self.clip_len - left_ctx - 1
        last_frame: Optional[np.ndarray] = None
        idx = 0

        def _flush_pending_clips() -> None:
            nonlocal encoded, pending_clips
            if not pending_clips:
                return
            batch_np = np.stack(pending_clips, axis=0)
            batch = torch.from_numpy(batch_np).permute(0, 4, 1, 2, 3)
            out = self._encode_clip_batch(batch)
            feats.append(out)
            encoded += int(out.shape[0])
            pending_clips = []
            if progress_cb:
                try:
                    progress_cb(encoded, expected or max(encoded, sampled))
                except Exception:
                    pass

        if progress_cb:
            try:
                progress_cb(0, expected)
            except Exception:
                pass
        while True:
            if use_total is not None and idx >= use_total:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if stride > 1 and (idx % stride) != 0:
                idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = self._resize_frame_for_clip_buffer(rgb)
            if sampled == 0:
                for _ in range(left_ctx):
                    buffered_frames.append(rgb)
            buffered_frames.append(rgb)
            picked_indices.append(idx)
            sampled += 1
            last_frame = rgb
            if len(buffered_frames) >= self.clip_len and (sampled - encoded) > right_ctx:
                pending_clips.append(np.stack(tuple(buffered_frames)[: self.clip_len], axis=0))
                buffered_frames.popleft()
                if len(pending_clips) >= self.batch_size:
                    _flush_pending_clips()
            idx += 1
        cap.release()
        total_frames = total if total > 0 else idx
        if not picked_indices:
            return np.zeros((0, int(self.feature_dim)), dtype=np.float32), picked_indices, total_frames
        if last_frame is None:
            raise RuntimeError("I3D sampling produced no frames.")
        while encoded + len(pending_clips) < len(picked_indices):
            while len(buffered_frames) < self.clip_len:
                buffered_frames.append(last_frame)
            pending_clips.append(np.stack(tuple(buffered_frames)[: self.clip_len], axis=0))
            buffered_frames.popleft()
            if len(pending_clips) >= self.batch_size:
                _flush_pending_clips()
        _flush_pending_clips()
        if not feats:
            return np.zeros((0, int(self.feature_dim)), dtype=np.float32), picked_indices, total_frames
        all_feats = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
        if progress_cb:
            try:
                progress_cb(len(picked_indices), expected or len(picked_indices))
            except Exception:
                pass
        return all_feats, picked_indices, total_frames


def build_feature_extractor(
    backbone: Optional[str],
    device: Optional[str] = None,
    batch_size: int = 128,
    use_fp16: bool = True,
) -> BaseFeatureExtractor:
    key = str(backbone or os.environ.get("FEATURE_BACKBONE", "i3d")).strip().lower().replace("-", "_")
    if key in {
        "i3d",
        "i3d_inception",
        "i3d_inception_rgb",
        "i3d_rgb",
        "i3d_r50",
        "i3d_r50_legacy",
        "i3d_legacy_r50",
    }:
        clip_batch = os.environ.get("I3D_BATCH_SIZE")
        try:
            clip_batch_size = max(1, int(clip_batch)) if clip_batch else min(max(1, int(batch_size)), 8)
        except Exception:
            clip_batch_size = min(max(1, int(batch_size)), 8)
        return I3DFeatureExtractor(
            device=device,
            batch_size=clip_batch_size,
            use_fp16=use_fp16,
        )
    if key in ("dinov2_vitb14", "dinov2", "dino", "dino_v2"):
        return DinoV2FeatureExtractor(
            device=device, batch_size=batch_size, use_fp16=use_fp16
        )
    if key in ("resnet50", "resnet"):
        return ResNet50FeatureExtractor(
            device=device, batch_size=batch_size, use_fp16=use_fp16
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def clear_feature_extractor_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def extract_video_features(
    video_path: str,
    backbone: Optional[str] = None,
    batch_size: int = 128,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    device: Optional[str] = None,
    use_fp16: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, dict]:
    extractor = build_feature_extractor(
        backbone=backbone,
        device=device,
        batch_size=batch_size,
        use_fp16=use_fp16,
    )
    features, picked, total = extractor.extract_from_video(
        video_path,
        frame_stride=frame_stride,
        max_frames=max_frames,
        progress_cb=progress_cb,
    )
    meta = {
        "backbone": extractor.name,
        "feature_dim": int(extractor.feature_dim),
        "frame_stride": int(frame_stride),
        "picked_indices": picked,
        "num_frames": int(total),
        "input_size": int(extractor.crop_size),
        "layout": "BTD",
        "model_cached": bool(getattr(extractor, "_reused_model", False)),
    }
    native_dim = int(getattr(extractor, "native_feature_dim", 0) or 0)
    if native_dim > 0:
        meta["native_feature_dim"] = native_dim
        if int(extractor.feature_dim) != native_dim:
            meta["projected_from_dim"] = native_dim
            meta["projection_mode"] = "fixed_random"
            meta["projection_source"] = f"{extractor.name}_projection"
    return features, meta


def save_features(
    features_dir: str, features: np.ndarray, meta: Optional[dict] = None
) -> str:
    os.makedirs(features_dir, exist_ok=True)
    feat_path = os.path.join(features_dir, "features.npy")
    np.save(feat_path, features.astype(np.float32, copy=False))
    if isinstance(meta, dict):
        meta_path = os.path.join(features_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
    return feat_path
