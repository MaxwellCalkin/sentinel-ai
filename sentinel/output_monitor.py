"""Real-time LLM output stream monitoring.

Continuously monitors LLM outputs for safety violations,
quality degradation, and anomalous patterns. Uses a rolling
window of recent outputs to detect drift and outliers.

Usage:
    from sentinel.output_monitor import OutputMonitor

    monitor = OutputMonitor(window_size=100)
    event = monitor.record("The capital of France is Paris.")
    report = monitor.report()
    drift = monitor.check_drift()
    anomalies = monitor.check_anomaly()
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


_HARMFUL_KEYWORDS = frozenset({
    "kill", "hack", "exploit", "bomb",
    "weapon", "attack", "destroy", "illegal",
})


@dataclass
class MonitorEvent:
    """A single monitored output event."""
    output: str
    length: int
    safe: bool
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Result of drift analysis between window halves."""
    drifting: bool
    safety_trend: str
    avg_length_trend: str
    first_half_safety: float
    second_half_safety: float


@dataclass
class AnomalyEvent:
    """A detected output anomaly."""
    output: str
    reason: str
    deviation: float


@dataclass
class MonitorReport:
    """Full monitoring report."""
    total_outputs: int
    avg_length: float
    safety_rate: float
    anomaly_count: int
    drift: DriftReport


class OutputMonitor:
    """Monitor LLM output streams for safety and quality.

    Tracks outputs in a rolling window and provides drift detection,
    anomaly detection, and aggregate safety metrics.
    """

    def __init__(self, window_size: int = 100) -> None:
        """
        Args:
            window_size: Maximum number of outputs to retain in the rolling window.
        """
        self._window_size = window_size
        self._events: deque[MonitorEvent] = deque(maxlen=window_size)

    def record(self, output: str, metadata: dict[str, Any] | None = None) -> MonitorEvent:
        """Record an output and assess its safety.

        Args:
            output: The LLM output text.
            metadata: Optional traceability metadata.

        Returns:
            MonitorEvent with safety assessment.
        """
        event = MonitorEvent(
            output=output,
            length=len(output),
            safe=_is_safe(output),
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._events.append(event)
        return event

    def check_drift(self, threshold: float = 0.3) -> DriftReport:
        """Detect quality and safety drift across the window.

        Splits the window into first and second halves, then compares
        safety rates and average lengths. A difference exceeding the
        threshold triggers a drift flag.

        Args:
            threshold: Minimum difference to consider as drift.

        Returns:
            DriftReport with trend analysis.
        """
        events = list(self._events)
        if len(events) < 2:
            return DriftReport(
                drifting=False,
                safety_trend="stable",
                avg_length_trend="stable",
                first_half_safety=0.0,
                second_half_safety=0.0,
            )

        midpoint = len(events) // 2
        first_half = events[:midpoint]
        second_half = events[midpoint:]

        first_safety = _safety_rate_for(first_half)
        second_safety = _safety_rate_for(second_half)
        safety_trend = _classify_trend(first_safety, second_safety, threshold)

        first_avg_length = _avg_length_for(first_half)
        second_avg_length = _avg_length_for(second_half)
        length_trend = _classify_trend(first_avg_length, second_avg_length, threshold * first_avg_length if first_avg_length > 0 else threshold)

        safety_drifting = abs(second_safety - first_safety) >= threshold
        length_drifting = first_avg_length > 0 and abs(second_avg_length - first_avg_length) / first_avg_length >= threshold

        return DriftReport(
            drifting=safety_drifting or length_drifting,
            safety_trend=safety_trend,
            avg_length_trend=length_trend,
            first_half_safety=first_safety,
            second_half_safety=second_safety,
        )

    def check_anomaly(self) -> list[AnomalyEvent]:
        """Detect anomalous outputs based on length outliers.

        An output is anomalous if its length is more than 2 standard
        deviations from the mean length across the window.

        Returns:
            List of detected anomaly events.
        """
        events = list(self._events)
        if len(events) < 2:
            return []

        lengths = [e.length for e in events]
        mean = sum(lengths) / len(lengths)
        stddev = _stddev(lengths)

        if stddev == 0.0:
            return []

        anomalies: list[AnomalyEvent] = []
        for event in events:
            deviation = abs(event.length - mean) / stddev
            if deviation > 2.0:
                direction = "above" if event.length > mean else "below"
                anomalies.append(AnomalyEvent(
                    output=event.output,
                    reason=f"length {direction} mean by {deviation:.1f} standard deviations",
                    deviation=round(deviation, 4),
                ))

        return anomalies

    def avg_length(self) -> float:
        """Average output length across the window."""
        events = list(self._events)
        return _avg_length_for(events)

    def safety_rate(self) -> float:
        """Fraction of outputs that passed the safety check."""
        events = list(self._events)
        return _safety_rate_for(events)

    def get_recent(self, n: int = 10) -> list[MonitorEvent]:
        """Get the most recent monitor events.

        Args:
            n: Number of recent events to return.

        Returns:
            List of recent MonitorEvent objects (newest last).
        """
        events = list(self._events)
        return events[-n:]

    def report(self) -> MonitorReport:
        """Generate a full monitoring report."""
        drift = self.check_drift()
        anomalies = self.check_anomaly()

        return MonitorReport(
            total_outputs=len(self._events),
            avg_length=self.avg_length(),
            safety_rate=self.safety_rate(),
            anomaly_count=len(anomalies),
            drift=drift,
        )

    def clear(self) -> None:
        """Reset the monitor, discarding all recorded events."""
        self._events.clear()


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _is_safe(text: str) -> bool:
    """Check whether text is free of harmful keywords."""
    words = text.lower().split()
    return not any(word.strip(".,!?;:\"'()[]") in _HARMFUL_KEYWORDS for word in words)


def _safety_rate_for(events: list[MonitorEvent]) -> float:
    """Compute safety rate for a list of events."""
    if not events:
        return 0.0
    safe_count = sum(1 for e in events if e.safe)
    return safe_count / len(events)


def _avg_length_for(events: list[MonitorEvent]) -> float:
    """Compute average output length for a list of events."""
    if not events:
        return 0.0
    return sum(e.length for e in events) / len(events)


def _stddev(values: list[float | int]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _classify_trend(first: float, second: float, threshold: float) -> str:
    """Classify the trend between two values."""
    diff = second - first
    if abs(diff) < threshold:
        return "stable"
    return "improving" if diff > 0 else "declining"
