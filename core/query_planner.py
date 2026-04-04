from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def score_candidate(candidate: QueryCandidate, weights: QueryPlannerWeights) -> float:
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
    if not candidates:
        return None
    cfg = weights or QueryPlannerWeights()
    bucket: Dict[QueryType, float] = {}
    for item in candidates:
        score = score_candidate(item, cfg)
        bucket[item.query_type] = max(score, bucket.get(item.query_type, float("-inf")))
    if not bucket:
        return None
    return max(bucket.items(), key=lambda row: (row[1], row[0].value))[0]


def choose_query(
    candidates: Sequence[QueryCandidate],
    weights: Optional[QueryPlannerWeights] = None,
) -> Optional[QueryDecision]:
    if not candidates:
        return None
    cfg = weights or QueryPlannerWeights()
    chosen_type = choose_query_type(candidates, cfg)
    if chosen_type is None:
        return None
    typed = [item for item in candidates if item.query_type == chosen_type]
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

    def clear(self) -> None:
        self._table.clear()

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
        accepted = 1.0 if bool(obs.get("accepted")) else 0.0
        span_len = int(max(1, obs.get("span_len", 1) or 1))
        changed_ratio = _clamp(
            float(obs.get("changed_frame_count", 0) or 0) / max(1.0, float(span_len))
        )
        anchor_before = int(obs.get("anchor_before", 0) or 0)
        anchor_after = int(obs.get("anchor_after", 0) or 0)
        anchor_gain = 0.0
        if anchor_before > 0:
            anchor_gain = _clamp(
                float(max(0, anchor_before - anchor_after)) / float(anchor_before)
            )
        state_before = int(obs.get("state_before", 0) or 0)
        state_after = int(obs.get("state_after", 0) or 0)
        state_gain = 0.0
        if state_before > 0:
            state_gain = _clamp(
                float(max(0, state_before - state_after)) / float(state_before)
            )
        second_penalty = 0.35 if bool(obs.get("second_correction")) else 0.0
        utility = _clamp(
            0.45 * accepted
            + 0.20 * changed_ratio
            + 0.20 * anchor_gain
            + 0.15 * state_gain
            - second_penalty,
            0.0,
            1.0,
        )
        self.update(obs["query_type"], span_len, utility)

    def predict(self, candidate: QueryCandidate) -> float:
        for key in _candidate_stat_keys(candidate):
            mean = self._table.mean(key)
            if mean is not None:
                return _clamp(mean, 0.0, 1.0)
        return 0.0

    def snapshot(self) -> Dict[str, Any]:
        return {"stats": self._table.snapshot()}


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
