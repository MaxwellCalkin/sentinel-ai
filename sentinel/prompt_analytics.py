"""Prompt usage analytics and safety monitoring.

Track prompt patterns, token consumption, safety flag rates, and trends
over time. Useful for monitoring LLM application health and identifying
patterns without storing raw prompt content.

Usage:
    from sentinel.prompt_analytics import PromptAnalytics

    analytics = PromptAnalytics()
    analytics.record("What is the capital of France?", tokens=25)
    analytics.record("Ignore previous instructions", tokens=30, flagged=True)

    print(analytics.flag_rate())   # 0.5
    print(analytics.avg_tokens())  # 27.5
    print(analytics.summary())     # AnalyticsSummary(...)
"""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PromptRecord:
    """A single recorded prompt event with privacy-preserving hash."""
    timestamp: datetime
    prompt_hash: str
    tokens: int
    flagged: bool
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class AnalyticsSummary:
    """Aggregate analytics across all recorded prompts."""
    total_prompts: int
    total_tokens: int
    avg_tokens: float
    flag_rate: float
    top_labels: list[tuple[str, int]]
    unique_prompts: int


def _compute_prompt_hash(prompt: str) -> str:
    """Return a truncated SHA-256 hash of the prompt for privacy."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


class PromptAnalytics:
    """Track prompt usage patterns and safety metrics.

    Thread-safe analytics collector that stores prompt hashes (never raw
    prompts) and computes aggregate statistics including flag rates,
    token averages, label frequencies, and trend direction.
    """

    def __init__(self) -> None:
        self._records: list[PromptRecord] = []
        self._unique_hashes: set[str] = set()
        self._label_counter: Counter[str] = Counter()
        self._lock = threading.Lock()

    def record(
        self,
        prompt: str,
        tokens: int = 0,
        flagged: bool = False,
        labels: dict[str, str] | None = None,
    ) -> PromptRecord:
        """Record a prompt event.

        Args:
            prompt: The raw prompt text (stored only as a hash).
            tokens: Token count for this prompt.
            flagged: Whether this prompt was flagged by a safety scanner.
            labels: Optional key-value labels for categorization.

        Returns:
            The PromptRecord that was created.
        """
        prompt_hash = _compute_prompt_hash(prompt)
        resolved_labels = labels or {}
        record = PromptRecord(
            timestamp=datetime.now(timezone.utc),
            prompt_hash=prompt_hash,
            tokens=tokens,
            flagged=flagged,
            labels=resolved_labels,
        )

        with self._lock:
            self._records.append(record)
            self._unique_hashes.add(prompt_hash)
            for key in resolved_labels:
                self._label_counter[key] += 1

        return record

    def summary(self) -> AnalyticsSummary:
        """Return aggregate statistics across all recorded prompts."""
        with self._lock:
            return AnalyticsSummary(
                total_prompts=len(self._records),
                total_tokens=self._total_tokens(),
                avg_tokens=self._avg_tokens(),
                flag_rate=self._flag_rate(),
                top_labels=self._top_labels(5),
                unique_prompts=len(self._unique_hashes),
            )

    def top_labels(self, n: int = 5) -> list[tuple[str, int]]:
        """Return the most common label keys with their counts.

        Args:
            n: Maximum number of labels to return.

        Returns:
            List of (label_key, count) tuples sorted by frequency.
        """
        with self._lock:
            return self._top_labels(n)

    def flag_rate(self) -> float:
        """Return the fraction of prompts that were flagged (0.0 to 1.0)."""
        with self._lock:
            return self._flag_rate()

    def avg_tokens(self) -> float:
        """Return the average token count across all recorded prompts."""
        with self._lock:
            return self._avg_tokens()

    def trend(self, window: int = 10) -> str:
        """Determine flag rate trend from recent records.

        Compares flag rate in the first half vs second half of the most
        recent `window` records. Returns "increasing" if the second half
        has a higher flag rate, "decreasing" if lower, or "stable" if equal.

        Args:
            window: Number of recent records to analyze.

        Returns:
            One of "increasing", "decreasing", or "stable".
        """
        with self._lock:
            return self._compute_trend(window)

    def clear(self) -> None:
        """Reset all analytics data."""
        with self._lock:
            self._records.clear()
            self._unique_hashes.clear()
            self._label_counter.clear()

    # -- Private helpers (must be called while holding self._lock) --

    def _total_tokens(self) -> int:
        return sum(record.tokens for record in self._records)

    def _avg_tokens(self) -> float:
        if not self._records:
            return 0.0
        return self._total_tokens() / len(self._records)

    def _flag_rate(self) -> float:
        if not self._records:
            return 0.0
        flagged_count = sum(1 for record in self._records if record.flagged)
        return flagged_count / len(self._records)

    def _top_labels(self, n: int) -> list[tuple[str, int]]:
        return self._label_counter.most_common(n)

    def _compute_trend(self, window: int) -> str:
        recent = self._records[-window:]
        if len(recent) < 2:
            return "stable"

        midpoint = len(recent) // 2
        first_half = recent[:midpoint]
        second_half = recent[midpoint:]

        first_rate = _flag_rate_for_slice(first_half)
        second_rate = _flag_rate_for_slice(second_half)

        if second_rate > first_rate:
            return "increasing"
        if second_rate < first_rate:
            return "decreasing"
        return "stable"


def _flag_rate_for_slice(records: list[PromptRecord]) -> float:
    """Compute flag rate for a list of records."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.flagged) / len(records)
