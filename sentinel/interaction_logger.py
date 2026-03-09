"""Complete LLM interaction logger for replay analysis and compliance auditing.

Log input/output pairs with safety metadata, query by model or score
range, export for compliance, and generate aggregate summaries.

Usage:
    from sentinel.interaction_logger import InteractionLogger

    logger = InteractionLogger(max_size=10000)
    logger.log("What is AI?", "AI is...", model="claude-3", safety_score=0.95, latency_ms=120.0)
    summary = logger.summary()
    print(summary.avg_safety_score)
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Interaction:
    """A single logged LLM interaction."""
    id: str
    input_text: str
    output_text: str
    model: str
    safety_score: float
    latency_ms: float
    timestamp: float
    metadata: dict[str, Any]


@dataclass
class InteractionSummary:
    """Aggregate statistics across logged interactions."""
    total: int
    avg_safety_score: float
    avg_latency_ms: float
    by_model: dict[str, int]
    low_safety_count: int


class InteractionLogger:
    """Thread-safe logger for complete LLM interactions.

    Records input/output pairs with safety scores, latency,
    and metadata. Supports querying, export, and FIFO eviction.
    """

    LOW_SAFETY_THRESHOLD = 0.5

    def __init__(self, max_size: int = 10000) -> None:
        self._max_size = max_size
        self._interactions: list[Interaction] = []
        self._lock = threading.Lock()

    def log(
        self,
        input_text: str,
        output_text: str,
        model: str = "unknown",
        safety_score: float = 1.0,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Interaction:
        """Log a complete LLM interaction.

        Returns the recorded Interaction with a unique ID and timestamp.
        """
        interaction = Interaction(
            id=uuid.uuid4().hex,
            input_text=input_text,
            output_text=output_text,
            model=model,
            safety_score=safety_score,
            latency_ms=latency_ms,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        with self._lock:
            self._interactions.append(interaction)
            self._evict_if_needed()
        return interaction

    def query_by_model(self, model: str) -> list[Interaction]:
        """Return all interactions for a given model."""
        with self._lock:
            return [i for i in self._interactions if i.model == model]

    def query_by_safety_score(
        self, min_score: float = 0.0, max_score: float = 1.0
    ) -> list[Interaction]:
        """Return interactions with safety scores in the given range (inclusive)."""
        with self._lock:
            return [
                i for i in self._interactions
                if min_score <= i.safety_score <= max_score
            ]

    def query_by_time_range(
        self, start: float, end: float
    ) -> list[Interaction]:
        """Return interactions with timestamps in [start, end] (inclusive)."""
        with self._lock:
            return [
                i for i in self._interactions
                if start <= i.timestamp <= end
            ]

    def export(self) -> list[dict[str, Any]]:
        """Export all interactions as a list of JSON-serializable dicts."""
        with self._lock:
            return [self._interaction_to_dict(i) for i in self._interactions]

    def summary(self) -> InteractionSummary:
        """Generate aggregate summary of all logged interactions."""
        with self._lock:
            total = len(self._interactions)
            if total == 0:
                return InteractionSummary(
                    total=0,
                    avg_safety_score=0.0,
                    avg_latency_ms=0.0,
                    by_model={},
                    low_safety_count=0,
                )

            avg_safety = sum(i.safety_score for i in self._interactions) / total
            avg_latency = sum(i.latency_ms for i in self._interactions) / total
            by_model = self._count_by_model()
            low_safety = sum(
                1 for i in self._interactions
                if i.safety_score < self.LOW_SAFETY_THRESHOLD
            )

            return InteractionSummary(
                total=total,
                avg_safety_score=round(avg_safety, 4),
                avg_latency_ms=round(avg_latency, 2),
                by_model=by_model,
                low_safety_count=low_safety,
            )

    def count(self) -> int:
        """Return the number of logged interactions."""
        with self._lock:
            return len(self._interactions)

    def clear(self) -> int:
        """Clear all interactions. Returns the number cleared."""
        with self._lock:
            cleared = len(self._interactions)
            self._interactions.clear()
            return cleared

    def _evict_if_needed(self) -> None:
        """Remove oldest interactions when max_size is exceeded."""
        if len(self._interactions) > self._max_size:
            overflow = len(self._interactions) - self._max_size
            self._interactions = self._interactions[overflow:]

    def _count_by_model(self) -> dict[str, int]:
        """Count interactions grouped by model name."""
        counts: dict[str, int] = {}
        for interaction in self._interactions:
            counts[interaction.model] = counts.get(interaction.model, 0) + 1
        return counts

    @staticmethod
    def _interaction_to_dict(interaction: Interaction) -> dict[str, Any]:
        """Convert an Interaction dataclass to a plain dict."""
        return {
            "id": interaction.id,
            "input_text": interaction.input_text,
            "output_text": interaction.output_text,
            "model": interaction.model,
            "safety_score": interaction.safety_score,
            "latency_ms": interaction.latency_ms,
            "timestamp": interaction.timestamp,
            "metadata": interaction.metadata,
        }
