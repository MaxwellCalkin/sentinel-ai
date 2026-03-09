"""Ensemble safety decisions from multiple checkers.

Combines results from multiple safety checkers using configurable
ensemble methods (majority vote, weighted average, unanimous, threshold)
for more robust, reliable safety decisions.

Usage:
    from sentinel.safety_ensemble import SafetyEnsemble, CheckerVote, EnsembleConfig

    ensemble = SafetyEnsemble()
    votes = [
        CheckerVote("toxicity", is_safe=True, confidence=0.9),
        CheckerVote("injection", is_safe=False, confidence=0.95),
        CheckerVote("pii", is_safe=True, confidence=0.8),
    ]
    decision = ensemble.decide(votes)
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_METHODS = {"majority", "weighted", "unanimous", "threshold"}


@dataclass
class CheckerVote:
    """A single safety checker's vote."""
    checker_name: str
    is_safe: bool
    confidence: float
    category: str = ""


@dataclass
class EnsembleDecision:
    """The combined decision from an ensemble of checkers."""
    is_safe: bool
    method: str
    confidence: float
    votes_safe: int
    votes_unsafe: int
    details: list[CheckerVote] = field(default_factory=list)


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble decision method."""
    method: str = "majority"
    threshold: float = 0.5
    min_voters: int = 1


@dataclass
class EnsembleStats:
    """Cumulative statistics across ensemble decisions."""
    total_decisions: int = 0
    safe_count: int = 0
    unsafe_count: int = 0
    avg_confidence: float = 0.0
    disagreement_count: int = 0


class SafetyEnsemble:
    """Combines multiple safety checker results using ensemble methods."""

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self._config = config or EnsembleConfig()
        self._validate_method(self._config.method)
        self._stats = EnsembleStats()
        self._total_confidence = 0.0

    def decide(self, votes: list[CheckerVote]) -> EnsembleDecision:
        """Make an ensemble decision from individual checker votes."""
        self._validate_votes(votes)
        safe_count, unsafe_count = _count_votes(votes)
        is_safe = self._apply_method(votes, safe_count, unsafe_count)
        confidence = _compute_confidence(votes, is_safe)
        self._update_stats(is_safe, confidence, votes)
        return EnsembleDecision(
            is_safe=is_safe,
            method=self._config.method,
            confidence=confidence,
            votes_safe=safe_count,
            votes_unsafe=unsafe_count,
            details=list(votes),
        )

    def decide_batch(self, vote_sets: list[list[CheckerVote]]) -> list[EnsembleDecision]:
        """Make ensemble decisions for multiple vote sets."""
        return [self.decide(votes) for votes in vote_sets]

    def stats(self) -> EnsembleStats:
        """Return cumulative statistics."""
        return EnsembleStats(
            total_decisions=self._stats.total_decisions,
            safe_count=self._stats.safe_count,
            unsafe_count=self._stats.unsafe_count,
            avg_confidence=self._stats.avg_confidence,
            disagreement_count=self._stats.disagreement_count,
        )

    def _validate_votes(self, votes: list[CheckerVote]) -> None:
        if len(votes) < self._config.min_voters:
            raise ValueError(
                f"Need at least {self._config.min_voters} voter(s), got {len(votes)}"
            )

    def _apply_method(self, votes: list[CheckerVote], safe_count: int, unsafe_count: int) -> bool:
        method = self._config.method
        if method == "majority":
            return _decide_majority(safe_count, unsafe_count)
        if method == "weighted":
            return _decide_weighted(votes)
        if method == "unanimous":
            return _decide_unanimous(unsafe_count)
        if method == "threshold":
            return _decide_threshold(safe_count, len(votes), self._config.threshold)
        raise ValueError(f"Unknown method: {method}")

    def _update_stats(self, is_safe: bool, confidence: float, votes: list[CheckerVote]) -> None:
        self._stats.total_decisions += 1
        if is_safe:
            self._stats.safe_count += 1
        else:
            self._stats.unsafe_count += 1
        if _has_disagreement(votes):
            self._stats.disagreement_count += 1
        self._total_confidence += confidence
        self._stats.avg_confidence = self._total_confidence / self._stats.total_decisions

    @staticmethod
    def _validate_method(method: str) -> None:
        if method not in VALID_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {sorted(VALID_METHODS)}"
            )


def _count_votes(votes: list[CheckerVote]) -> tuple[int, int]:
    safe_count = sum(1 for v in votes if v.is_safe)
    return safe_count, len(votes) - safe_count


def _decide_majority(safe_count: int, unsafe_count: int) -> bool:
    # Ties favor unsafe (conservative for safety)
    return safe_count > unsafe_count


def _decide_weighted(votes: list[CheckerVote]) -> bool:
    weighted_sum = sum(
        v.confidence * (1.0 if v.is_safe else -1.0) for v in votes
    )
    # Zero or negative weighted sum favors unsafe (conservative)
    return weighted_sum > 0


def _decide_unanimous(unsafe_count: int) -> bool:
    return unsafe_count == 0


def _decide_threshold(safe_count: int, total: int, threshold: float) -> bool:
    fraction_safe = safe_count / total
    return fraction_safe >= threshold


def _compute_confidence(votes: list[CheckerVote], is_safe: bool) -> float:
    winning_votes = [v for v in votes if v.is_safe == is_safe]
    if not winning_votes:
        return 0.0
    return sum(v.confidence for v in winning_votes) / len(winning_votes)


def _has_disagreement(votes: list[CheckerVote]) -> bool:
    if len(votes) <= 1:
        return False
    first = votes[0].is_safe
    return any(v.is_safe != first for v in votes)
