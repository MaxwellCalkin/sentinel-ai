"""User feedback loop for LLM output quality tracking.

Capture thumbs-up/down feedback, ratings, and corrections
on LLM outputs. Track quality trends and identify degradation.

Usage:
    from sentinel.feedback_loop import FeedbackLoop

    loop = FeedbackLoop()
    loop.record("output-1", rating=5, comment="Great response")
    report = loop.report()
    print(report.avg_rating)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeedbackEntry:
    """A single feedback record."""
    output_id: str
    rating: int  # 1-5
    positive: bool
    comment: str = ""
    correction: str = ""
    labels: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackReport:
    """Aggregate feedback statistics."""
    total: int
    positive_count: int
    negative_count: int
    avg_rating: float
    positive_rate: float  # 0.0 to 1.0
    by_label: dict[str, dict[str, Any]]
    recent_trend: str  # "improving", "declining", "stable"
    corrections_count: int


class FeedbackLoop:
    """Track user feedback on LLM outputs.

    Capture ratings, thumbs-up/down, corrections, and labels.
    Analyze trends and identify quality issues.
    """

    def __init__(self, trend_window: int = 20) -> None:
        """
        Args:
            trend_window: Number of recent entries for trend analysis.
        """
        self._entries: list[FeedbackEntry] = []
        self._trend_window = trend_window

    def record(
        self,
        output_id: str,
        rating: int = 3,
        positive: bool | None = None,
        comment: str = "",
        correction: str = "",
        labels: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEntry:
        """Record feedback for an output.

        Args:
            output_id: Identifier for the LLM output.
            rating: 1-5 rating (clamped).
            positive: Explicit thumbs up/down (inferred from rating if None).
            comment: Optional text feedback.
            correction: Optional corrected output.
            labels: Optional category labels.
            metadata: Optional metadata.

        Returns:
            The recorded FeedbackEntry.
        """
        rating = max(1, min(5, rating))
        if positive is None:
            positive = rating >= 4

        entry = FeedbackEntry(
            output_id=output_id,
            rating=rating,
            positive=positive,
            comment=comment,
            correction=correction,
            labels=labels or [],
            metadata=metadata or {},
        )
        self._entries.append(entry)
        return entry

    def thumbs_up(self, output_id: str, comment: str = "", **kwargs: Any) -> FeedbackEntry:
        """Record positive feedback."""
        return self.record(output_id, rating=5, positive=True, comment=comment, **kwargs)

    def thumbs_down(self, output_id: str, comment: str = "", correction: str = "", **kwargs: Any) -> FeedbackEntry:
        """Record negative feedback."""
        return self.record(output_id, rating=1, positive=False, comment=comment, correction=correction, **kwargs)

    def report(self) -> FeedbackReport:
        """Generate aggregate feedback report."""
        total = len(self._entries)
        if total == 0:
            return FeedbackReport(
                total=0,
                positive_count=0,
                negative_count=0,
                avg_rating=0.0,
                positive_rate=0.0,
                by_label={},
                recent_trend="stable",
                corrections_count=0,
            )

        positive_count = sum(1 for e in self._entries if e.positive)
        negative_count = total - positive_count
        avg_rating = sum(e.rating for e in self._entries) / total
        positive_rate = positive_count / total
        corrections_count = sum(1 for e in self._entries if e.correction)

        # By-label stats
        by_label: dict[str, dict[str, Any]] = {}
        for entry in self._entries:
            for label in entry.labels:
                if label not in by_label:
                    by_label[label] = {"count": 0, "total_rating": 0, "positive": 0}
                by_label[label]["count"] += 1
                by_label[label]["total_rating"] += entry.rating
                if entry.positive:
                    by_label[label]["positive"] += 1
        for label, stats in by_label.items():
            stats["avg_rating"] = round(stats["total_rating"] / stats["count"], 2)
            stats["positive_rate"] = round(stats["positive"] / stats["count"], 2)

        # Trend analysis
        trend = self._compute_trend()

        return FeedbackReport(
            total=total,
            positive_count=positive_count,
            negative_count=negative_count,
            avg_rating=round(avg_rating, 2),
            positive_rate=round(positive_rate, 4),
            by_label=by_label,
            recent_trend=trend,
            corrections_count=corrections_count,
        )

    def get_corrections(self) -> list[FeedbackEntry]:
        """Get all entries that include corrections."""
        return [e for e in self._entries if e.correction]

    def get_by_label(self, label: str) -> list[FeedbackEntry]:
        """Get entries with a specific label."""
        return [e for e in self._entries if label in e.labels]

    def clear(self) -> int:
        """Clear all feedback. Returns count cleared."""
        count = len(self._entries)
        self._entries.clear()
        return count

    def _compute_trend(self) -> str:
        """Compute recent trend from entries."""
        if len(self._entries) < 4:
            return "stable"

        window = self._entries[-self._trend_window:]
        mid = len(window) // 2
        first_half = window[:mid]
        second_half = window[mid:]

        avg_first = sum(e.rating for e in first_half) / len(first_half)
        avg_second = sum(e.rating for e in second_half) / len(second_half)

        diff = avg_second - avg_first
        if diff > 0.5:
            return "improving"
        elif diff < -0.5:
            return "declining"
        return "stable"
