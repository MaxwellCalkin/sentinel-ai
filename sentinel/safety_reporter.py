"""Aggregate safety event reporting with trend analysis.

Collect safety events over time and produce human-readable and
machine-readable reports with trend analysis and export capabilities.

Usage:
    from sentinel.safety_reporter import SafetyReporter

    reporter = SafetyReporter()
    reporter.record("warning", "injection", "Possible prompt injection detected")
    summary = reporter.summary()
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any

VALID_SEVERITIES = ("info", "warning", "critical")

SEVERITY_WEIGHTS = {
    "info": 1.0,
    "warning": 3.0,
    "critical": 5.0,
}


@dataclass
class SafetyEvent:
    """A single safety event."""
    timestamp: float
    severity: str
    category: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Direction and magnitude of safety trend over a period."""
    direction: str  # "improving", "stable", "degrading"
    score_change: float
    period_seconds: float


@dataclass
class ReporterSummary:
    """Aggregated reporter statistics."""
    total_events: int
    by_severity: dict[str, int]
    by_category: dict[str, int]
    trend: TrendAnalysis
    escalation_needed: bool


class SafetyReporter:
    """Aggregate safety event reporting with trend analysis.

    Records safety events with severity levels, performs
    time-windowed aggregation, analyzes trends, and exports
    reports in dict format.
    """

    def __init__(
        self,
        max_events: int = 10000,
        escalation_threshold: int = 3,
    ) -> None:
        """
        Args:
            max_events: Maximum events to retain in memory.
            escalation_threshold: Number of critical events that triggers escalation.
        """
        self._max_events = max_events
        self._escalation_threshold = escalation_threshold
        self._events: list[SafetyEvent] = []
        self._lock = threading.Lock()

    def record(
        self,
        severity: str,
        category: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> SafetyEvent:
        """Record a safety event.

        Args:
            severity: One of "info", "warning", "critical".
            category: Event category (e.g., "injection", "pii", "toxicity").
            description: Human-readable description.
            metadata: Optional additional data.

        Returns:
            The recorded SafetyEvent.

        Raises:
            ValueError: If severity is not valid.
        """
        if severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{severity}'. Must be one of {VALID_SEVERITIES}"
            )

        event = SafetyEvent(
            timestamp=time.time(),
            severity=severity,
            category=category,
            description=description,
            metadata=metadata or {},
        )

        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events.pop(0)

        return event

    def events_in_window(self, seconds: float) -> list[SafetyEvent]:
        """Get events within a time window.

        Args:
            seconds: Window size in seconds from now.

        Returns:
            List of events within the window.
        """
        cutoff = time.time() - seconds
        with self._lock:
            return [e for e in self._events if e.timestamp >= cutoff]

    def count_by_severity(
        self,
        events: list[SafetyEvent] | None = None,
    ) -> dict[str, int]:
        """Count events grouped by severity.

        Args:
            events: Events to count. Uses all events if None.
        """
        target = events if events is not None else self._all_events()
        counts: dict[str, int] = {}
        for event in target:
            counts[event.severity] = counts.get(event.severity, 0) + 1
        return counts

    def count_by_category(
        self,
        events: list[SafetyEvent] | None = None,
    ) -> dict[str, int]:
        """Count events grouped by category.

        Args:
            events: Events to count. Uses all events if None.
        """
        target = events if events is not None else self._all_events()
        counts: dict[str, int] = {}
        for event in target:
            counts[event.category] = counts.get(event.category, 0) + 1
        return counts

    def category_percentages(
        self,
        events: list[SafetyEvent] | None = None,
    ) -> dict[str, float]:
        """Get category breakdown as percentages.

        Args:
            events: Events to analyze. Uses all events if None.

        Returns:
            Dict mapping category to percentage (0-100).
        """
        counts = self.count_by_category(events)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {
            category: round(count / total * 100, 2)
            for category, count in counts.items()
        }

    def analyze_trend(self, window_seconds: float = 3600.0) -> TrendAnalysis:
        """Analyze safety trend by comparing two halves of a time window.

        Splits the window into an earlier half and a later half,
        computes weighted severity scores for each, and determines
        whether safety is improving, degrading, or stable.

        Args:
            window_seconds: Total window to analyze (default 1 hour).

        Returns:
            TrendAnalysis with direction and score change.
        """
        now = time.time()
        window_start = now - window_seconds
        midpoint = now - window_seconds / 2

        with self._lock:
            events = [e for e in self._events if e.timestamp >= window_start]

        earlier = [e for e in events if e.timestamp < midpoint]
        later = [e for e in events if e.timestamp >= midpoint]

        earlier_score = _compute_severity_score(earlier)
        later_score = _compute_severity_score(later)
        score_change = later_score - earlier_score

        direction = _classify_direction(score_change)

        return TrendAnalysis(
            direction=direction,
            score_change=round(score_change, 4),
            period_seconds=window_seconds,
        )

    def summary(self, window_seconds: float | None = None) -> ReporterSummary:
        """Generate a full reporter summary.

        Args:
            window_seconds: If provided, only include events in this window.
                           If None, include all events.

        Returns:
            ReporterSummary with counts, breakdowns, trend, and escalation status.
        """
        if window_seconds is not None:
            events = self.events_in_window(window_seconds)
        else:
            events = self._all_events()

        by_severity = self.count_by_severity(events)
        by_category = self.count_by_category(events)
        trend = self.analyze_trend()
        escalation_needed = self._check_escalation(events)

        return ReporterSummary(
            total_events=len(events),
            by_severity=by_severity,
            by_category=by_category,
            trend=trend,
            escalation_needed=escalation_needed,
        )

    def export(self, window_seconds: float | None = None) -> dict[str, Any]:
        """Export report as a dictionary.

        Args:
            window_seconds: If provided, only include events in this window.

        Returns:
            Dict with all report data.
        """
        if window_seconds is not None:
            events = self.events_in_window(window_seconds)
        else:
            events = self._all_events()

        return {
            "total_events": len(events),
            "by_severity": self.count_by_severity(events),
            "by_category": self.count_by_category(events),
            "category_percentages": self.category_percentages(events),
            "trend": {
                "direction": self.analyze_trend().direction,
                "score_change": self.analyze_trend().score_change,
            },
            "escalation_needed": self._check_escalation(events),
            "events": [_event_to_dict(e) for e in events],
        }

    def clear(self) -> None:
        """Remove all recorded events."""
        with self._lock:
            self._events.clear()

    @property
    def total_events(self) -> int:
        """Total number of stored events."""
        with self._lock:
            return len(self._events)

    def _all_events(self) -> list[SafetyEvent]:
        """Return a snapshot of all events."""
        with self._lock:
            return list(self._events)

    def _check_escalation(self, events: list[SafetyEvent]) -> bool:
        """Determine if escalation is needed based on critical event count."""
        critical_count = sum(1 for e in events if e.severity == "critical")
        return critical_count >= self._escalation_threshold


def _compute_severity_score(events: list[SafetyEvent]) -> float:
    """Compute a weighted severity score for a list of events."""
    return sum(SEVERITY_WEIGHTS.get(e.severity, 0.0) for e in events)


def _classify_direction(score_change: float) -> str:
    """Classify trend direction from score change."""
    if score_change > 1.0:
        return "degrading"
    if score_change < -1.0:
        return "improving"
    return "stable"


def _event_to_dict(event: SafetyEvent) -> dict[str, Any]:
    """Convert a SafetyEvent to a plain dict."""
    return {
        "timestamp": event.timestamp,
        "severity": event.severity,
        "category": event.category,
        "description": event.description,
        "metadata": event.metadata,
    }
