from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class CorrectionSession:
    kind: str
    meta: Dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_utc_now_iso)
    steps: int = 0


class CorrectionBuffer:
    def __init__(self) -> None:
        self.active: Optional[CorrectionSession] = None
        self.history: List[Dict[str, Any]] = []

    def begin(
        self,
        kind: str,
        *,
        meta: Optional[Dict[str, Any]] = None,
        replace: bool = True,
    ) -> CorrectionSession:
        if self.active is not None and not replace:
            return self.active
        self.active = CorrectionSession(
            kind=str(kind or "unknown").strip() or "unknown",
            meta=dict(meta or {}),
        )
        return self.active

    def note_step(self, count: int = 1) -> None:
        if self.active is None:
            return
        try:
            delta = int(count)
        except Exception:
            delta = 1
        if delta <= 0:
            delta = 1
        self.active.steps += delta

    def commit(
        self,
        *,
        records: Optional[List[Dict[str, Any]]] = None,
        meta_update: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.active or CorrectionSession(kind="implicit")
        if meta_update:
            session.meta.update(dict(meta_update))
        summary = {
            "kind": session.kind,
            "meta": dict(session.meta),
            "started_at": session.started_at,
            "committed_at": _utc_now_iso(),
            "steps": int(session.steps),
            "records": list(records or []),
            "changed": bool(records),
        }
        self.history.append(summary)
        self.active = None
        return summary

    def discard(self, *, reason: str = "") -> Optional[Dict[str, Any]]:
        if self.active is None:
            return None
        summary = {
            "kind": self.active.kind,
            "meta": dict(self.active.meta),
            "started_at": self.active.started_at,
            "discarded_at": _utc_now_iso(),
            "steps": int(self.active.steps),
            "reason": str(reason or "").strip(),
            "changed": False,
        }
        self.history.append(summary)
        self.active = None
        return summary

    def record_event(
        self,
        kind: str,
        *,
        meta: Optional[Dict[str, Any]] = None,
        changed: bool = False,
        steps: int = 0,
        reason: str = "",
    ) -> Dict[str, Any]:
        summary = {
            "kind": str(kind or "event").strip() or "event",
            "meta": dict(meta or {}),
            "started_at": _utc_now_iso(),
            "committed_at": _utc_now_iso(),
            "steps": max(0, int(steps or 0)),
            "changed": bool(changed),
        }
        if reason:
            summary["reason"] = str(reason or "").strip()
        self.history.append(summary)
        return summary


@dataclass
class AdaptationSample:
    """One training sample collected when the user accepts a scribble proposal."""
    window_features: np.ndarray          # (T, D) float32
    scribble_channels: Dict[str, np.ndarray]  # uncertain/left/right  (T,)
    boundary_energy: np.ndarray          # (T,) float32
    boundary_frame: int
    window_start: int
    window_end: int
    left_label: str
    right_label: str
    left_state: str = ""
    right_state: str = ""
    sample_kind: str = "boundary_accept"
    boundary_valid: bool = True
    side_valid: bool = True
    action_valid: bool = True
    query_utility: float = -1.0
    calibrated_confidence: float = -1.0
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "window_features": np.asarray(self.window_features, dtype=np.float32).tolist(),
            "scribble_channels": {
                str(name): np.asarray(values, dtype=np.float32).tolist()
                for name, values in dict(self.scribble_channels or {}).items()
            },
            "boundary_energy": np.asarray(self.boundary_energy, dtype=np.float32).tolist(),
            "boundary_frame": int(self.boundary_frame),
            "window_start": int(self.window_start),
            "window_end": int(self.window_end),
            "left_label": str(self.left_label or ""),
            "right_label": str(self.right_label or ""),
            "left_state": str(self.left_state or ""),
            "right_state": str(self.right_state or ""),
            "state_before": str(self.left_state or ""),
            "state_after": str(self.right_state or ""),
            "sample_kind": str(self.sample_kind or "boundary_accept"),
            "boundary_valid": bool(self.boundary_valid),
            "side_valid": bool(self.side_valid),
            "action_valid": bool(self.action_valid),
            "query_utility": float(self.query_utility),
            "calibrated_confidence": float(self.calibrated_confidence),
            "timestamp": str(self.timestamp or _utc_now_iso()),
        }

    @classmethod
    def from_jsonable(cls, payload: Dict[str, Any]) -> Optional["AdaptationSample"]:
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                window_features=np.asarray(
                    payload.get("window_features", []), dtype=np.float32
                ),
                scribble_channels={
                    str(name): np.asarray(values, dtype=np.float32)
                    for name, values in dict(payload.get("scribble_channels") or {}).items()
                },
                boundary_energy=np.asarray(
                    payload.get("boundary_energy", []), dtype=np.float32
                ),
                boundary_frame=int(payload.get("boundary_frame", 0) or 0),
                window_start=int(payload.get("window_start", 0) or 0),
                window_end=int(payload.get("window_end", 0) or 0),
                left_label=str(payload.get("left_label", "") or ""),
                right_label=str(payload.get("right_label", "") or ""),
                left_state=str(
                    payload.get("left_state", payload.get("state_before", "")) or ""
                ),
                right_state=str(
                    payload.get("right_state", payload.get("state_after", "")) or ""
                ),
                sample_kind=str(payload.get("sample_kind", "boundary_accept") or "boundary_accept"),
                boundary_valid=bool(payload.get("boundary_valid", True)),
                side_valid=bool(payload.get("side_valid", True)),
                action_valid=bool(payload.get("action_valid", True)),
                query_utility=float(payload.get("query_utility", -1.0) or -1.0),
                calibrated_confidence=float(
                    payload.get("calibrated_confidence", -1.0) or -1.0
                ),
                timestamp=str(payload.get("timestamp", "") or _utc_now_iso()),
            )
        except Exception:
            return None


class AdaptationSampleBuffer:
    """Collects AdaptationSample objects for periodic local-head fine-tuning."""

    def __init__(self, max_size: int = 200) -> None:
        self.samples: List[AdaptationSample] = []
        self.max_size = max(1, int(max_size))

    def add(self, sample: AdaptationSample) -> None:
        self.samples.append(sample)
        if len(self.samples) > self.max_size:
            self.samples = self.samples[-self.max_size:]

    def __len__(self) -> int:
        return len(self.samples)

    def ready(self, min_samples: int = 5) -> bool:
        return len(self.samples) >= min_samples

    def export_examples(self) -> List[Dict[str, Any]]:
        """Convert samples to the dict format expected by ScribbleTrainingDataset."""
        out: List[Dict[str, Any]] = []
        for s in self.samples:
            out.append({
                "boundary_frame": s.boundary_frame,
                "left_label": s.left_label,
                "right_label": s.right_label,
                "left_state": s.left_state,
                "right_state": s.right_state,
                "state_before": s.left_state,
                "state_after": s.right_state,
                "sample_kind": str(s.sample_kind or "boundary_accept"),
                "boundary_valid": bool(s.boundary_valid),
                "side_valid": bool(s.side_valid),
                "action_valid": bool(s.action_valid),
                "query_utility": float(s.query_utility),
                "calibrated_confidence": float(s.calibrated_confidence),
                "window_features": s.window_features,
                "scribble_channels": s.scribble_channels,
                "boundary_energy": s.boundary_energy,
                "window_start": s.window_start,
                "window_end": s.window_end,
            })
        return out

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "max_size": int(self.max_size),
            "samples": [sample.to_jsonable() for sample in self.samples],
        }

    def load_jsonable(self, payload: Dict[str, Any]) -> int:
        if not isinstance(payload, dict):
            return 0
        try:
            self.max_size = max(1, int(payload.get("max_size", self.max_size) or self.max_size))
        except Exception:
            pass
        loaded: List[AdaptationSample] = []
        for row in list(payload.get("samples") or []):
            sample = AdaptationSample.from_jsonable(row)
            if sample is not None:
                loaded.append(sample)
        self.samples = loaded[-self.max_size :]
        return len(self.samples)

    def discard_prefix(self, count: int) -> int:
        try:
            limit = max(0, int(count))
        except Exception:
            limit = 0
        if limit <= 0 or not self.samples:
            return 0
        removed = min(limit, len(self.samples))
        del self.samples[:removed]
        return removed

    def clear(self) -> None:
        self.samples.clear()
