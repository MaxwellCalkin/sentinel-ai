"""Latency tracker for LLM operations.

Track, analyze, and alert on latency for API calls, safety
scans, and pipeline operations. Compute percentiles, detect
regressions, and enforce SLA thresholds.

Usage:
    from sentinel.latency_tracker import LatencyTracker

    tracker = LatencyTracker(sla_p99_ms=500)
    tracker.record("scan", 12.5)
    tracker.record("scan", 15.3)
    report = tracker.report("scan")
    print(report.p99)
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyRecord:
    """A single latency measurement."""
    operation: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyReport:
    """Latency statistics for an operation."""
    operation: str
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95: float
    p99: float
    std_dev: float
    sla_met: bool
    sla_violations: int


class LatencyTracker:
    """Track and analyze operation latencies."""

    def __init__(
        self,
        sla_p99_ms: float = 1000.0,
        max_records: int = 10000,
    ) -> None:
        self._sla_p99 = sla_p99_ms
        self._max_records = max_records
        self._records: dict[str, list[float]] = {}
        self._timestamps: dict[str, list[float]] = {}
        self._total_records = 0

    def record(self, operation: str, latency_ms: float, metadata: dict[str, Any] | None = None) -> None:
        """Record a latency measurement."""
        if operation not in self._records:
            self._records[operation] = []
            self._timestamps[operation] = []

        self._records[operation].append(latency_ms)
        self._timestamps[operation].append(time.time())
        self._total_records += 1

        # Trim if over limit
        if len(self._records[operation]) > self._max_records:
            self._records[operation] = self._records[operation][-self._max_records:]
            self._timestamps[operation] = self._timestamps[operation][-self._max_records:]

    def report(self, operation: str) -> LatencyReport | None:
        """Generate latency report for an operation."""
        values = self._records.get(operation)
        if not values:
            return None

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean = sum(sorted_vals) / n

        # Median
        if n % 2 == 0:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        else:
            median = sorted_vals[n // 2]

        # Percentiles
        p95 = self._percentile(sorted_vals, 95)
        p99 = self._percentile(sorted_vals, 99)

        # Std dev
        variance = sum((x - mean) ** 2 for x in sorted_vals) / n
        std_dev = math.sqrt(variance)

        # SLA check
        sla_violations = sum(1 for v in sorted_vals if v > self._sla_p99)

        return LatencyReport(
            operation=operation,
            count=n,
            min_ms=round(sorted_vals[0], 2),
            max_ms=round(sorted_vals[-1], 2),
            mean_ms=round(mean, 2),
            median_ms=round(median, 2),
            p95=round(p95, 2),
            p99=round(p99, 2),
            std_dev=round(std_dev, 2),
            sla_met=p99 <= self._sla_p99,
            sla_violations=sla_violations,
        )

    def report_all(self) -> dict[str, LatencyReport]:
        """Generate reports for all tracked operations."""
        reports = {}
        for op in self._records:
            report = self.report(op)
            if report:
                reports[op] = report
        return reports

    def list_operations(self) -> list[str]:
        """List tracked operations."""
        return list(self._records.keys())

    def clear(self, operation: str | None = None) -> int:
        """Clear records. If operation specified, clear only that one."""
        if operation:
            count = len(self._records.get(operation, []))
            self._records.pop(operation, None)
            self._timestamps.pop(operation, None)
            return count
        count = sum(len(v) for v in self._records.values())
        self._records.clear()
        self._timestamps.clear()
        return count

    def _percentile(self, sorted_vals: list[float], pct: float) -> float:
        """Compute percentile from sorted values."""
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        k = (pct / 100) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)
