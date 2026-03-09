"""Embedding drift monitoring for LLM applications.

Monitors semantic drift in embeddings over time to detect model
degradation, data distribution shifts, or adversarial manipulation.
Uses cosine similarity on raw embedding vectors with zero dependencies.

Usage:
    from sentinel.embedding_drift import EmbeddingDrift

    monitor = EmbeddingDrift(threshold=0.8)
    monitor.set_baseline([0.1, 0.9, 0.3])
    monitor.record([0.1, 0.85, 0.32], label="batch_1")
    report = monitor.report()
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class EmbeddingRecord:
    """A single recorded embedding with metadata."""
    vector: list[float]
    timestamp: float
    label: str


@dataclass
class DriftCheck:
    """Result of checking one embedding for drift."""
    similarity: float
    drifted: bool
    baseline_similarity: float | None


@dataclass
class DriftStats:
    """Rolling window statistics over recent drift checks."""
    mean_similarity: float
    min_similarity: float
    max_similarity: float
    drift_count: int
    total_checks: int


@dataclass
class DriftReport:
    """Full drift report with statistics and recent checks."""
    stats: DriftStats
    recent_checks: list[DriftCheck]
    baseline_set: bool


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


class EmbeddingDrift:
    """Monitor semantic drift in embedding vectors over time.

    Tracks consecutive embeddings and compares via cosine similarity.
    Flags drift when similarity drops below a configurable threshold.
    Supports baseline comparison and rolling-window statistics.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        window_size: int = 50,
    ) -> None:
        """
        Args:
            threshold: Cosine similarity below which drift is flagged.
            window_size: Number of records to retain for rolling stats.
        """
        self._threshold = threshold
        self._window_size = window_size
        self._history: list[EmbeddingRecord] = []
        self._checks: list[DriftCheck] = []
        self._baseline: list[float] | None = None

    def set_baseline(self, vector: list[float]) -> None:
        """Set the baseline embedding for future comparisons.

        Args:
            vector: The reference embedding vector.
        """
        self._baseline = list(vector)

    @property
    def baseline(self) -> list[float] | None:
        """Current baseline vector, or None if unset."""
        return self._baseline

    @property
    def history_count(self) -> int:
        """Number of embedding records currently stored."""
        return len(self._history)

    @property
    def check_count(self) -> int:
        """Number of drift checks performed."""
        return len(self._checks)

    def record(
        self,
        vector: list[float],
        label: str = "",
        timestamp: float | None = None,
    ) -> DriftCheck:
        """Record an embedding and check for drift.

        Compares the new embedding against the previous one
        (consecutive similarity) and against the baseline if set.

        Args:
            vector: The embedding vector.
            label: Optional descriptive label.
            timestamp: Unix timestamp; defaults to current time.

        Returns:
            DriftCheck with similarity scores and drift flag.
        """
        ts = timestamp if timestamp is not None else time.time()
        record = EmbeddingRecord(vector=list(vector), timestamp=ts, label=label)

        consecutive_similarity = self._compute_consecutive_similarity(vector)
        baseline_similarity = self._compute_baseline_similarity(vector)
        drifted = self._is_drifted(consecutive_similarity, baseline_similarity)

        self._history.append(record)
        self._trim_history()

        check = DriftCheck(
            similarity=consecutive_similarity,
            drifted=drifted,
            baseline_similarity=baseline_similarity,
        )
        self._checks.append(check)

        return check

    def rolling_stats(self, window: int | None = None) -> DriftStats:
        """Compute rolling statistics over recent drift checks.

        Args:
            window: Number of recent checks to consider.
                    Defaults to the configured window_size.

        Returns:
            DriftStats with mean, min, max similarity and counts.
        """
        count = window if window is not None else self._window_size
        recent = self._checks[-count:] if self._checks else []
        return self._build_stats(recent)

    def report(self, window: int | None = None) -> DriftReport:
        """Generate a full drift report.

        Args:
            window: Number of recent checks to include.

        Returns:
            DriftReport with stats, recent checks, and baseline status.
        """
        count = window if window is not None else self._window_size
        recent = self._checks[-count:] if self._checks else []
        stats = self._build_stats(recent)
        return DriftReport(
            stats=stats,
            recent_checks=list(recent),
            baseline_set=self._baseline is not None,
        )

    def clear(self) -> None:
        """Clear all recorded history, checks, and baseline."""
        self._history.clear()
        self._checks.clear()
        self._baseline = None

    def clear_history(self) -> None:
        """Clear recorded embeddings and checks, but keep the baseline."""
        self._history.clear()
        self._checks.clear()

    def _compute_consecutive_similarity(self, vector: list[float]) -> float:
        if not self._history:
            return 1.0
        previous = self._history[-1].vector
        return _cosine_similarity(previous, vector)

    def _compute_baseline_similarity(self, vector: list[float]) -> float | None:
        if self._baseline is None:
            return None
        return _cosine_similarity(self._baseline, vector)

    def _is_drifted(
        self,
        consecutive_similarity: float,
        baseline_similarity: float | None,
    ) -> bool:
        if consecutive_similarity < self._threshold:
            return True
        if baseline_similarity is not None and baseline_similarity < self._threshold:
            return True
        return False

    def _trim_history(self) -> None:
        while len(self._history) > self._window_size:
            self._history.pop(0)

    @staticmethod
    def _build_stats(checks: list[DriftCheck]) -> DriftStats:
        if not checks:
            return DriftStats(
                mean_similarity=0.0,
                min_similarity=0.0,
                max_similarity=0.0,
                drift_count=0,
                total_checks=0,
            )
        similarities = [c.similarity for c in checks]
        return DriftStats(
            mean_similarity=sum(similarities) / len(similarities),
            min_similarity=min(similarities),
            max_similarity=max(similarities),
            drift_count=sum(1 for c in checks if c.drifted),
            total_checks=len(checks),
        )
