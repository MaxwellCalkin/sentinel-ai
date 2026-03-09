"""Aggregate safety metrics dashboard.

Collect and aggregate safety metrics across scans, incidents,
and guardrail actions. Provides real-time visibility into
system safety posture.

Usage:
    from sentinel.safety_dashboard import SafetyDashboard

    dash = SafetyDashboard()
    dash.record_scan(blocked=False, risk_score=0.2, scanner="injection")
    dash.record_incident(severity="high", category="injection")
    summary = dash.summary()
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardSummary:
    """Dashboard summary statistics."""
    total_scans: int
    total_blocked: int
    block_rate: float
    total_incidents: int
    incidents_by_severity: dict[str, int]
    avg_risk_score: float
    scans_by_scanner: dict[str, int]
    uptime_seconds: float


@dataclass
class TimeWindow:
    """Metrics for a time window."""
    scans: int
    blocked: int
    block_rate: float
    incidents: int
    avg_risk: float


class SafetyDashboard:
    """Aggregate safety metrics for monitoring.

    Track scans, blocks, incidents, and risk scores
    with time-windowed analysis.
    """

    def __init__(self, max_points: int = 10000) -> None:
        """
        Args:
            max_points: Maximum metric points to retain.
        """
        self._max_points = max_points
        self._scans: list[MetricPoint] = []
        self._incidents: list[MetricPoint] = []
        self._start_time = time.time()
        self._lock = threading.Lock()

        # Counters
        self._total_scans = 0
        self._total_blocked = 0
        self._total_incidents = 0
        self._risk_sum = 0.0
        self._scanner_counts: dict[str, int] = {}
        self._severity_counts: dict[str, int] = {}

    def record_scan(
        self,
        blocked: bool = False,
        risk_score: float = 0.0,
        scanner: str = "default",
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a scan result.

        Args:
            blocked: Whether the scan resulted in a block.
            risk_score: Risk score (0.0 to 1.0).
            scanner: Scanner name.
            labels: Additional labels.
        """
        with self._lock:
            self._total_scans += 1
            if blocked:
                self._total_blocked += 1
            self._risk_sum += risk_score
            self._scanner_counts[scanner] = self._scanner_counts.get(scanner, 0) + 1

            point = MetricPoint(
                timestamp=time.time(),
                value=risk_score,
                labels={"blocked": str(blocked), "scanner": scanner, **(labels or {})},
            )
            self._scans.append(point)
            if len(self._scans) > self._max_points:
                self._scans.pop(0)

    def record_incident(
        self,
        severity: str = "medium",
        category: str = "unknown",
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a safety incident.

        Args:
            severity: Incident severity (critical, high, medium, low).
            category: Incident category.
            labels: Additional labels.
        """
        with self._lock:
            self._total_incidents += 1
            self._severity_counts[severity] = self._severity_counts.get(severity, 0) + 1

            point = MetricPoint(
                timestamp=time.time(),
                value=1.0,
                labels={"severity": severity, "category": category, **(labels or {})},
            )
            self._incidents.append(point)
            if len(self._incidents) > self._max_points:
                self._incidents.pop(0)

    def summary(self) -> DashboardSummary:
        """Get current dashboard summary."""
        with self._lock:
            return DashboardSummary(
                total_scans=self._total_scans,
                total_blocked=self._total_blocked,
                block_rate=(
                    self._total_blocked / self._total_scans
                    if self._total_scans > 0 else 0.0
                ),
                total_incidents=self._total_incidents,
                incidents_by_severity=dict(self._severity_counts),
                avg_risk_score=(
                    round(self._risk_sum / self._total_scans, 4)
                    if self._total_scans > 0 else 0.0
                ),
                scans_by_scanner=dict(self._scanner_counts),
                uptime_seconds=round(time.time() - self._start_time, 2),
            )

    def window(self, seconds: float = 300) -> TimeWindow:
        """Get metrics for a time window.

        Args:
            seconds: Window size in seconds (default 5 minutes).
        """
        cutoff = time.time() - seconds

        with self._lock:
            recent_scans = [s for s in self._scans if s.timestamp >= cutoff]
            recent_incidents = [i for i in self._incidents if i.timestamp >= cutoff]

        scan_count = len(recent_scans)
        blocked = sum(1 for s in recent_scans if s.labels.get("blocked") == "True")
        risk_scores = [s.value for s in recent_scans]

        return TimeWindow(
            scans=scan_count,
            blocked=blocked,
            block_rate=blocked / scan_count if scan_count > 0 else 0.0,
            incidents=len(recent_incidents),
            avg_risk=sum(risk_scores) / len(risk_scores) if risk_scores else 0.0,
        )

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._scans.clear()
            self._incidents.clear()
            self._total_scans = 0
            self._total_blocked = 0
            self._total_incidents = 0
            self._risk_sum = 0.0
            self._scanner_counts.clear()
            self._severity_counts.clear()
            self._start_time = time.time()
