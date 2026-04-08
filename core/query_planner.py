from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_log = logging.getLogger(__name__)

_ESCAPE_LABELS = frozenset({"unknown", "other", "background"})


class QueryType(str, Enum):
    BOUNDARY_SCRIBBLE = "boundary_scribble"
    LABEL_REVIEW = "label_review"
    STATE_REPAIR = "state_repair"


@dataclass
class QueryCandidate:
    query_id: str
    query_type: QueryType
    start_frame: int
    end_frame: int
    score_terms: Dict[str, float] = field(default_factory=dict)
    estimated_cost: float = 1.0
    payload: Dict[str, object] = field(default_factory=dict)


@dataclass
class QueryPlannerWeights:
    uncertainty: float = 1.0
    disagreement: float = 1.0
    multiview: float = 1.0
    state_conflict: float = 1.0
    propagation_gain: float = 1.0
    history: float = 1.0
    learned_utility: float = 0.0
    bias_boundary: float = 0.0
    bias_label: float = 0.0
    bias_state: float = 0.0
    tau: float = 0.25


@dataclass
class QueryDecision:
    query_type: QueryType
    candidate: QueryCandidate
    utility: float


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = 0.0
    if out < float(lo):
        return float(lo)
    if out > float(hi):
        return float(hi)
    return float(out)


def _bucket_span(span_len: int) -> str:
    try:
        val = int(span_len)
    except Exception:
        val = 1
    if val <= 16:
        return "xs"
    if val <= 48:
        return "s"
    if val <= 128:
        return "m"
    if val <= 320:
        return "l"
    return "xl"


def _dominant_signal_name(terms: Dict[str, float]) -> str:
    scored = []
    for key in (
        "uncertainty",
        "disagreement",
        "multiview",
        "state_conflict",
        "propagation_gain",
        "history",
        "learned_utility",
    ):
        try:
            val = float(terms.get(key, 0.0) or 0.0)
        except Exception:
            val = 0.0
        scored.append((val, key))
    if not scored:
        return "none"
    scored.sort()
    best_val, best_key = scored[-1]
    return str(best_key if best_val > 0.0 else "none")


def extract_candidate_features(candidate: QueryCandidate) -> Dict[str, Any]:
    span_len = max(1, int(candidate.end_frame) - int(candidate.start_frame) + 1)
    terms = dict(candidate.score_terms or {})
    return {
        "query_type": str(candidate.query_type.value),
        "span_len": int(span_len),
        "span_bucket": _bucket_span(span_len),
        "dominant_signal": _dominant_signal_name(terms),
        "estimated_cost": float(candidate.estimated_cost or 0.0),
        "score_terms": terms,
    }


def _candidate_stat_keys(candidate: QueryCandidate) -> List[Tuple[str, ...]]:
    feat = extract_candidate_features(candidate)
    qtype = str(feat.get("query_type") or "")
    span_bucket = str(feat.get("span_bucket") or "m")
    dominant = str(feat.get("dominant_signal") or "none")
    return [
        ("type_span_dom", qtype, span_bucket, dominant),
        ("type_span", qtype, span_bucket),
        ("type_dom", qtype, dominant),
        ("type", qtype),
    ]


def _candidate_is_queryable(candidate: QueryCandidate) -> bool:
    if not isinstance(candidate, QueryCandidate):
        return False
    payload = dict(candidate.payload or {})
    if bool(payload.get("ignore_for_query")):
        return False
    current_label = str(payload.get("current_label", "") or "").strip().lower()
    if current_label in _ESCAPE_LABELS:
        return False
    if candidate.query_type == QueryType.BOUNDARY_SCRIBBLE:
        if bool(payload.get("contains_escape_label")):
            return False
        left_label = str(payload.get("left_label", "") or "").strip().lower()
        right_label = str(payload.get("right_label", "") or "").strip().lower()
        if left_label in _ESCAPE_LABELS or right_label in _ESCAPE_LABELS:
            return False
    return True


def score_candidate(candidate: QueryCandidate, weights: QueryPlannerWeights) -> float:
    if not _candidate_is_queryable(candidate):
        return 0.0
    terms = dict(candidate.score_terms or {})
    score = 0.0
    score += float(terms.get("uncertainty", 0.0) or 0.0) * float(weights.uncertainty)
    score += float(terms.get("disagreement", 0.0) or 0.0) * float(weights.disagreement)
    score += float(terms.get("multiview", 0.0) or 0.0) * float(weights.multiview)
    score += float(terms.get("state_conflict", 0.0) or 0.0) * float(weights.state_conflict)
    score += float(terms.get("propagation_gain", 0.0) or 0.0) * float(weights.propagation_gain)
    score += float(terms.get("history", 0.0) or 0.0) * float(weights.history)
    score += float(terms.get("learned_utility", 0.0) or 0.0) * float(
        weights.learned_utility
    )
    if candidate.query_type == QueryType.BOUNDARY_SCRIBBLE:
        score += float(weights.bias_boundary)
    elif candidate.query_type == QueryType.LABEL_REVIEW:
        score += float(weights.bias_label)
    elif candidate.query_type == QueryType.STATE_REPAIR:
        score += float(weights.bias_state)
    cost = max(0.0, float(candidate.estimated_cost or 0.0))
    return float(score / (cost + max(1e-6, float(weights.tau))))


def choose_query_type(
    candidates: Sequence[QueryCandidate],
    weights: Optional[QueryPlannerWeights] = None,
) -> Optional[QueryType]:
    active = [item for item in candidates if _candidate_is_queryable(item)]
    if not active:
        return None
    cfg = weights or QueryPlannerWeights()
    bucket: Dict[QueryType, float] = {}
    for item in active:
        score = score_candidate(item, cfg)
        bucket[item.query_type] = max(score, bucket.get(item.query_type, float("-inf")))
    if not bucket:
        return None
    return max(bucket.items(), key=lambda row: (row[1], row[0].value))[0]


def choose_query(
    candidates: Sequence[QueryCandidate],
    weights: Optional[QueryPlannerWeights] = None,
) -> Optional[QueryDecision]:
    active = [item for item in candidates if _candidate_is_queryable(item)]
    if not active:
        return None
    cfg = weights or QueryPlannerWeights()
    chosen_type = choose_query_type(active, cfg)
    if chosen_type is None:
        return None
    typed = [item for item in active if item.query_type == chosen_type]
    if not typed:
        return None
    best = max(
        typed,
        key=lambda item: (
            score_candidate(item, cfg),
            -int(item.start_frame),
            str(item.query_id),
        ),
    )
    return QueryDecision(
        query_type=chosen_type,
        candidate=best,
        utility=score_candidate(best, cfg),
    )


def group_candidates_by_type(
    candidates: Iterable[QueryCandidate],
) -> Dict[QueryType, List[QueryCandidate]]:
    grouped: Dict[QueryType, List[QueryCandidate]] = {}
    for item in candidates:
        grouped.setdefault(item.query_type, []).append(item)
    return grouped


def query_type_from_text(raw: Any) -> Optional[QueryType]:
    text = str(raw or "").strip().lower()
    if not text:
        return None
    if "state" in text or "trace" in text:
        return QueryType.STATE_REPAIR
    if "label" in text:
        return QueryType.LABEL_REVIEW
    if "boundary" in text or "scribble" in text or "trim" in text:
        return QueryType.BOUNDARY_SCRIBBLE
    return None


def summarize_correction_observation(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(summary, dict):
        return None
    meta = summary.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    qtype = query_type_from_text(
        meta.get("query_type")
        or meta.get("point_type")
        or summary.get("kind")
        or meta.get("kind")
    )
    if qtype is None:
        return None

    start_raw = meta.get("feedback_start", meta.get("start", meta.get("frame", 0)))
    if meta.get("boundary_frame") is not None and start_raw in (None, "", 0):
        start_raw = meta.get("boundary_frame")
    end_raw = meta.get("feedback_end", meta.get("end", start_raw))
    if end_raw in (None, ""):
        end_raw = start_raw
    try:
        start = int(start_raw or 0)
    except Exception:
        start = 0
    try:
        end = int(end_raw or start)
    except Exception:
        end = start
    if end < start:
        start, end = end, start

    accepted_flag = meta.get("accepted")
    if accepted_flag is None:
        if "discarded_at" in summary:
            accepted = False
        elif bool(meta.get("proposal_feedback")):
            accepted = bool(meta.get("accepted"))
        elif str(meta.get("mode") or "").strip().lower() in ("accept", "accepted", "apply"):
            accepted = True
        elif "accept" in str(meta.get("point_type") or "").strip().lower():
            accepted = True
        else:
            accepted = bool(summary.get("changed", False))
    else:
        accepted = bool(accepted_flag)

    try:
        steps = max(0, int(summary.get("steps", 0) or 0))
    except Exception:
        steps = 0

    return {
        "query_type": qtype,
        "kind": str(summary.get("kind") or "").strip().lower(),
        "point_type": str(meta.get("point_type") or "").strip().lower(),
        "start_frame": int(start),
        "end_frame": int(end),
        "span_len": int(max(1, end - start + 1)),
        "steps": int(steps),
        "accepted": bool(accepted),
        "changed": bool(summary.get("changed", False)),
        "changed_frame_count": int(max(0, int(meta.get("changed_frame_count", 0) or 0))),
        "anchor_before": int(max(0, int(meta.get("anchor_violations_before", 0) or 0))),
        "anchor_after": int(max(0, int(meta.get("anchor_violations_after", 0) or 0))),
        "state_before": int(max(0, int(meta.get("state_conflicts_before", 0) or 0))),
        "state_after": int(max(0, int(meta.get("state_conflicts_after", 0) or 0))),
        "second_correction": bool(meta.get("second_correction", False)),
        "proposal_feedback": bool(meta.get("proposal_feedback", False)),
        "raw_confidence": meta.get("raw_confidence"),
        "calibrated_confidence": meta.get("calibrated_confidence"),
        "summary": summary,
    }


def estimate_observed_query_utility(
    *,
    accepted: Any,
    span_len: Any,
    changed_frame_count: Any = 0,
    anchor_before: Any = 0,
    anchor_after: Any = 0,
    state_before: Any = 0,
    state_after: Any = 0,
    second_correction: Any = False,
) -> float:
    accepted_val = 1.0 if bool(accepted) else 0.0
    span = max(1, int(span_len or 1))
    changed_ratio = _clamp(float(changed_frame_count or 0) / max(1.0, float(span)))
    anchor_b = max(0, int(anchor_before or 0))
    anchor_a = max(0, int(anchor_after or 0))
    anchor_gain = 0.0
    if anchor_b > 0:
        anchor_gain = _clamp(float(max(0, anchor_b - anchor_a)) / float(anchor_b))
    state_b = max(0, int(state_before or 0))
    state_a = max(0, int(state_after or 0))
    state_gain = 0.0
    if state_b > 0:
        state_gain = _clamp(float(max(0, state_b - state_a)) / float(state_b))
    second_penalty = 0.35 if bool(second_correction) else 0.0
    return _clamp(
        0.45 * accepted_val
        + 0.20 * changed_ratio
        + 0.20 * anchor_gain
        + 0.15 * state_gain
        - second_penalty,
        0.0,
        1.0,
    )


def observed_query_utility_from_summary(summary: Dict[str, Any]) -> Optional[float]:
    obs = summarize_correction_observation(summary)
    if not obs or bool(obs.get("proposal_feedback")):
        return None
    return float(
        estimate_observed_query_utility(
            accepted=obs.get("accepted"),
            span_len=obs.get("span_len", 1),
            changed_frame_count=obs.get("changed_frame_count", 0),
            anchor_before=obs.get("anchor_before", 0),
            anchor_after=obs.get("anchor_after", 0),
            state_before=obs.get("state_before", 0),
            state_after=obs.get("state_after", 0),
            second_correction=obs.get("second_correction", False),
        )
    )


class _RunningAverageTable:
    def __init__(self) -> None:
        self._stats: Dict[Tuple[str, ...], List[float]] = {}

    def clear(self) -> None:
        self._stats.clear()

    def update(self, key: Tuple[str, ...], value: float) -> None:
        if not key:
            return
        try:
            val = float(value)
        except Exception:
            return
        count, total = self._stats.get(tuple(key), [0.0, 0.0])
        self._stats[tuple(key)] = [float(count) + 1.0, float(total) + float(val)]

    def mean(self, key: Tuple[str, ...]) -> Optional[float]:
        row = self._stats.get(tuple(key))
        if not row or float(row[0]) <= 0.0:
            return None
        return float(row[1]) / max(1.0, float(row[0]))

    def count(self, key: Tuple[str, ...]) -> int:
        row = self._stats.get(tuple(key))
        if not row:
            return 0
        try:
            return int(row[0])
        except Exception:
            return 0

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for key, row in self._stats.items():
            out["|".join(key)] = {
                "count": float(row[0]),
                "mean": 0.0 if float(row[0]) <= 0.0 else float(row[1]) / float(row[0]),
            }
        return out

    def load_snapshot(self, data: Dict[str, Any]) -> None:
        self.clear()
        if not isinstance(data, dict):
            return
        for raw_key, raw_row in data.items():
            if not isinstance(raw_key, str) or not raw_key:
                continue
            row = raw_row if isinstance(raw_row, dict) else {}
            try:
                count = max(0.0, float(row.get("count", 0.0) or 0.0))
                mean = float(row.get("mean", 0.0) or 0.0)
            except Exception:
                continue
            if count <= 0.0:
                continue
            key = tuple(part for part in raw_key.split("|") if part)
            if not key:
                continue
            self._stats[key] = [float(count), float(count * mean)]


class QueryCostModel:
    def __init__(self, default_costs: Optional[Dict[QueryType, float]] = None) -> None:
        self.default_costs = {
            QueryType.BOUNDARY_SCRIBBLE: 0.55,
            QueryType.LABEL_REVIEW: 0.45,
            QueryType.STATE_REPAIR: 0.85,
        }
        if isinstance(default_costs, dict):
            for key, value in default_costs.items():
                try:
                    qtype = key if isinstance(key, QueryType) else QueryType(str(key))
                    self.default_costs[qtype] = float(value)
                except Exception:
                    continue
        self._table = _RunningAverageTable()

    def clear(self) -> None:
        self._table.clear()

    def update(self, query_type: QueryType, span_len: int, observed_cost: float) -> None:
        try:
            cost = max(0.05, float(observed_cost))
        except Exception:
            return
        candidate = QueryCandidate(
            query_id="",
            query_type=query_type,
            start_frame=0,
            end_frame=max(0, int(span_len) - 1),
        )
        for key in _candidate_stat_keys(candidate):
            self._table.update(key, cost)

    def update_from_summary(self, summary: Dict[str, Any]) -> None:
        obs = summarize_correction_observation(summary)
        if not obs or bool(obs.get("proposal_feedback")):
            return
        steps = max(1, int(obs.get("steps", 0) or 0))
        cost = min(3.0, 0.25 + 0.18 * float(steps))
        if bool(obs.get("second_correction")):
            cost = min(3.5, cost + 0.35)
        self.update(obs["query_type"], int(obs.get("span_len", 1) or 1), cost)

    def predict(self, candidate: QueryCandidate, fallback: Optional[float] = None) -> float:
        for key in _candidate_stat_keys(candidate):
            mean = self._table.mean(key)
            if mean is not None:
                return float(max(0.05, mean))
        default = (
            float(fallback)
            if fallback is not None
            else float(self.default_costs.get(candidate.query_type, 0.6))
        )
        return float(max(0.05, default))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "defaults": {str(k.value): float(v) for k, v in self.default_costs.items()},
            "stats": self._table.snapshot(),
        }


class QueryUtilityModel:
    def __init__(self) -> None:
        self._table = _RunningAverageTable()
        self._linear_model = TrainableQueryUtilityModel()

    def clear(self) -> None:
        self._table.clear()
        self._linear_model.clear()

    def update(self, query_type: QueryType, span_len: int, utility: float) -> None:
        value = _clamp(utility, 0.0, 1.0)
        candidate = QueryCandidate(
            query_id="",
            query_type=query_type,
            start_frame=0,
            end_frame=max(0, int(span_len) - 1),
        )
        for key in _candidate_stat_keys(candidate):
            self._table.update(key, value)

    def update_from_summary(self, summary: Dict[str, Any]) -> None:
        obs = summarize_correction_observation(summary)
        if not obs or bool(obs.get("proposal_feedback")):
            return
        if (
            str(obs.get("point_type") or "") == "boundary_scribble"
            and str(obs.get("kind") or "") == "temporal_boundary_scribble"
        ):
            return
        span_len = int(max(1, obs.get("span_len", 1) or 1))
        utility = estimate_observed_query_utility(
            accepted=obs.get("accepted"),
            span_len=span_len,
            changed_frame_count=obs.get("changed_frame_count", 0),
            anchor_before=obs.get("anchor_before", 0),
            anchor_after=obs.get("anchor_after", 0),
            state_before=obs.get("state_before", 0),
            state_after=obs.get("state_after", 0),
            second_correction=obs.get("second_correction", False),
        )
        self.update(obs["query_type"], span_len, utility)
        meta = obs.get("summary", {}).get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        candidate = QueryCandidate(
            query_id="",
            query_type=obs["query_type"],
            start_frame=int(obs.get("start_frame", 0)),
            end_frame=int(obs.get("end_frame", 0)),
            score_terms=dict(meta.get("score_terms") or {}),
            estimated_cost=float(meta.get("estimated_cost", 0.6) or 0.6),
            payload=dict(meta.get("query_candidate_payload") or {}),
        )
        self._linear_model.add_observation(candidate, utility)

    def predict(self, candidate: QueryCandidate) -> float:
        if not _candidate_is_queryable(candidate):
            return 0.0
        learned = None
        if self._linear_model.ready:
            learned = self._linear_model.predict(candidate)
        for key in _candidate_stat_keys(candidate):
            mean = self._table.mean(key)
            if mean is not None:
                tabular = _clamp(mean, 0.0, 1.0)
                if learned is None:
                    return tabular
                return _clamp(0.35 * tabular + 0.65 * learned, 0.0, 1.0)
        if learned is not None:
            return learned
        return 0.0

    def fit(self, *, lr: float = 0.01, epochs: int = 20) -> float:
        return self._linear_model.fit(lr=lr, epochs=epochs)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "stats": self._table.snapshot(),
            "linear_model": self._linear_model.snapshot(),
        }

    def load_snapshot(self, data: Dict[str, Any]) -> None:
        self.clear()
        if not isinstance(data, dict):
            return
        stats = data.get("stats")
        if isinstance(stats, dict):
            self._table.load_snapshot(stats)
        linear = data.get("linear_model")
        if isinstance(linear, dict):
            self._linear_model.load_snapshot(linear)
            return
        self._linear_model.load_snapshot(data)


# Feature dimension layout for TrainableQueryUtilityModel:
#   [0] span_len_norm (log-scaled)
#   [1..3] query_type one-hot (boundary_scribble, label_review, state_repair)
#   [4] uncertainty score
#   [5] disagreement score
#   [6] multiview score
#   [7] state_conflict score
#   [8] propagation_gain score
#   [9] history prior
#   [10] boundary energy score
#   [11] normalized estimated_cost
_QUERY_FEAT_DIM = 12
_QUERY_TYPE_INDEX = {
    QueryType.BOUNDARY_SCRIBBLE: 0,
    QueryType.LABEL_REVIEW: 1,
    QueryType.STATE_REPAIR: 2,
}


def _candidate_to_feature_vec(candidate: QueryCandidate) -> np.ndarray:
    """Convert a QueryCandidate into a fixed-dim feature vector for the learned model."""
    vec = np.zeros(_QUERY_FEAT_DIM, dtype=np.float32)
    span_len = max(1, int(candidate.end_frame) - int(candidate.start_frame) + 1)
    vec[0] = float(np.log1p(span_len))
    idx = _QUERY_TYPE_INDEX.get(candidate.query_type, 0)
    vec[1 + idx] = 1.0
    terms = dict(candidate.score_terms or {})
    vec[4] = _clamp(terms.get("uncertainty", 0.0))
    vec[5] = _clamp(terms.get("disagreement", 0.0))
    vec[6] = _clamp(terms.get("multiview", 0.0))
    vec[7] = _clamp(terms.get("state_conflict", 0.0))
    vec[8] = _clamp(terms.get("propagation_gain", 0.0))
    vec[9] = _clamp(terms.get("history", 0.0))
    vec[10] = _clamp(terms.get("energy", 0.0))
    vec[11] = _clamp(float(candidate.estimated_cost or 0.0) / 2.0)
    return vec


class TrainableQueryUtilityModel:
    """Lightweight linear regression model for query utility prediction (L_query).

    Uses numpy-only weights so the query planner stays free of a torch dependency.
    Trained via simple SGD on MSE loss over observed (candidate, utility) pairs.
    """

    def __init__(self) -> None:
        self.weights = np.zeros(_QUERY_FEAT_DIM, dtype=np.float64)
        self.bias = 0.0
        self._observations: List[Dict[str, Any]] = []
        self._max_observations = 500
        self.ready = False

    def clear(self) -> None:
        self.weights = np.zeros(_QUERY_FEAT_DIM, dtype=np.float64)
        self.bias = 0.0
        self._observations = []
        self.ready = False

    def _predict_raw(self, feat: np.ndarray) -> float:
        return float(np.dot(self.weights, feat.astype(np.float64)) + self.bias)

    def predict(self, candidate: QueryCandidate) -> float:
        if not _candidate_is_queryable(candidate):
            return 0.0
        feat = _candidate_to_feature_vec(candidate)
        return _clamp(self._predict_raw(feat), 0.0, 1.0)

    def add_observation(self, candidate: QueryCandidate, utility: float) -> None:
        feat = _candidate_to_feature_vec(candidate)
        self._observations.append({
            "feat": feat,
            "utility": _clamp(utility, 0.0, 1.0),
        })
        if len(self._observations) > self._max_observations:
            self._observations = self._observations[-self._max_observations:]

    def add_observation_from_summary(self, summary: Dict[str, Any]) -> None:
        obs = summarize_correction_observation(summary)
        if not obs or bool(obs.get("proposal_feedback")):
            return
        span_len = int(max(1, obs.get("span_len", 1) or 1))
        utility = estimate_observed_query_utility(
            accepted=obs.get("accepted"),
            span_len=span_len,
            changed_frame_count=obs.get("changed_frame_count", 0),
            anchor_before=obs.get("anchor_before", 0),
            anchor_after=obs.get("anchor_after", 0),
            state_before=obs.get("state_before", 0),
            state_after=obs.get("state_after", 0),
            second_correction=obs.get("second_correction", False),
        )
        candidate = QueryCandidate(
            query_id="",
            query_type=obs["query_type"],
            start_frame=int(obs.get("start_frame", 0)),
            end_frame=int(obs.get("end_frame", 0)),
            score_terms=dict(obs.get("summary", {}).get("meta", {}).get("score_terms", {}) or {}),
            estimated_cost=float(obs.get("summary", {}).get("meta", {}).get("estimated_cost", 0.6) or 0.6),
        )
        self.add_observation(candidate, utility)

    def fit(self, *, lr: float = 0.01, epochs: int = 20) -> float:
        """Train weights on accumulated observations via weighted SGD. Returns final MSE."""
        if len(self._observations) < 3:
            self.ready = False
            return 0.0
        feats = np.stack([o["feat"] for o in self._observations]).astype(np.float64)
        targets = np.array([o["utility"] for o in self._observations], dtype=np.float64)
        # Favor recent corrections slightly so the planner can adapt to the current
        # user/session without forgetting the older signal entirely.
        sample_weights = np.linspace(0.7, 1.0, num=len(targets), dtype=np.float64)
        reg = 1e-3
        w = self.weights.copy()
        b = float(self.bias)
        n = len(targets)
        for _ in range(max(1, int(epochs))):
            preds = feats @ w + b
            residuals = preds - targets
            weighted = residuals * sample_weights
            grad_w = (2.0 / n) * (feats.T @ weighted) + 2.0 * reg * w
            grad_b = (2.0 / n) * weighted.sum()
            w -= lr * grad_w
            b -= lr * grad_b
        self.weights = w
        self.bias = b
        was_ready = self.ready
        self.ready = True
        preds = feats @ w + b
        mse = float(np.mean(sample_weights * ((preds - targets) ** 2)))
        if not was_ready:
            _log.info(
                "TrainableQueryUtilityModel now ready (n=%d, mse=%.4f)",
                n, mse,
            )
        else:
            _log.debug(
                "TrainableQueryUtilityModel re-trained (n=%d, mse=%.4f)",
                n, mse,
            )
        return mse

    def snapshot(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "num_observations": len(self._observations),
            "ready": bool(self.ready),
        }

    def load_snapshot(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        w = data.get("weights")
        loaded_weights = False
        if isinstance(w, list) and w:
            arr = np.asarray(w, dtype=np.float64).reshape(-1)
            if int(arr.size) >= _QUERY_FEAT_DIM:
                self.weights = arr[:_QUERY_FEAT_DIM].astype(np.float64)
                loaded_weights = True
            else:
                padded = np.zeros(_QUERY_FEAT_DIM, dtype=np.float64)
                padded[: int(arr.size)] = arr
                self.weights = padded
                loaded_weights = True
        b = data.get("bias")
        if b is not None:
            self.bias = float(b)
        self.ready = bool(data.get("ready", True) and loaded_weights)


class ProposalConfidenceCalibrator:
    def __init__(self, bucket_count: int = 10) -> None:
        self.bucket_count = max(4, int(bucket_count))
        self._totals: Dict[int, int] = {}
        self._accepts: Dict[int, int] = {}

    def clear(self) -> None:
        self._totals.clear()
        self._accepts.clear()

    def _bucket(self, confidence: Any) -> int:
        conf = _clamp(confidence, 0.0, 1.0)
        idx = int(conf * self.bucket_count)
        if idx >= self.bucket_count:
            idx = self.bucket_count - 1
        return max(0, idx)

    def update(self, raw_confidence: Any, accepted: bool) -> None:
        if raw_confidence is None:
            return
        bucket = self._bucket(raw_confidence)
        self._totals[bucket] = int(self._totals.get(bucket, 0) + 1)
        if bool(accepted):
            self._accepts[bucket] = int(self._accepts.get(bucket, 0) + 1)

    def update_from_summary(self, summary: Dict[str, Any]) -> None:
        obs = summarize_correction_observation(summary)
        if not obs or not bool(obs.get("proposal_feedback")):
            return
        self.update(obs.get("raw_confidence"), bool(obs.get("accepted")))

    def calibrate(self, raw_confidence: Any) -> float:
        raw = _clamp(raw_confidence, 0.0, 1.0)
        bucket = self._bucket(raw)
        total = int(self._totals.get(bucket, 0))
        accept = int(self._accepts.get(bucket, 0))
        global_total = max(0, sum(int(v) for v in self._totals.values()))
        global_accept = max(0, sum(int(v) for v in self._accepts.values()))
        global_rate = float(global_accept + 1.0) / float(global_total + 2.0)
        if total <= 0:
            return float(0.6 * raw + 0.4 * global_rate)
        bucket_rate = float(accept + 1.0) / float(total + 2.0)
        blend = min(1.0, float(total) / 6.0)
        return float((1.0 - blend) * (0.6 * raw + 0.4 * global_rate) + blend * bucket_rate)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "bucket_count": int(self.bucket_count),
            "buckets": {
                str(idx): {
                    "total": int(self._totals.get(idx, 0)),
                    "accepted": int(self._accepts.get(idx, 0)),
                }
                for idx in range(int(self.bucket_count))
            },
        }
