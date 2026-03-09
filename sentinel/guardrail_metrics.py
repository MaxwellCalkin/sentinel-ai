"""Prometheus-style metrics for guardrail performance.

Collect counters, gauges, and histograms for safety guardrail
operations. Export in standard formats for monitoring systems.

Usage:
    from sentinel.guardrail_metrics import GuardrailMetrics

    metrics = GuardrailMetrics()
    metrics.counter("scans_total").inc()
    metrics.counter("blocks_total").inc()
    metrics.histogram("scan_latency_ms").observe(12.5)
    output = metrics.export()
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CounterValue:
    """A monotonically increasing counter."""
    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount


@dataclass
class GaugeValue:
    """A value that can go up and down."""
    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def set(self, value: float) -> None:
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


class HistogramValue:
    """Distribution of observed values."""

    def __init__(self, name: str, buckets: list[float] | None = None) -> None:
        self.name = name
        self._buckets = buckets or [1, 5, 10, 25, 50, 100, 250, 500, 1000]
        self._counts: dict[float, int] = {b: 0 for b in self._buckets}
        self._counts[float('inf')] = 0
        self._sum = 0.0
        self._count = 0
        self._values: list[float] = []

    def observe(self, value: float) -> None:
        """Record an observation."""
        self._sum += value
        self._count += 1
        self._values.append(value)
        for bucket in self._buckets:
            if value <= bucket:
                self._counts[bucket] += 1
        self._counts[float('inf')] += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    def percentile(self, pct: float) -> float:
        """Compute percentile from observations."""
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        k = (pct / 100) * (len(sorted_vals) - 1)
        f = math.floor(k)
        c = min(math.ceil(k), len(sorted_vals) - 1)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


class GuardrailMetrics:
    """Collect and export guardrail metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, CounterValue] = {}
        self._gauges: dict[str, GaugeValue] = {}
        self._histograms: dict[str, HistogramValue] = {}

    def counter(self, name: str, labels: dict[str, str] | None = None) -> CounterValue:
        """Get or create a counter."""
        key = self._key(name, labels)
        if key not in self._counters:
            self._counters[key] = CounterValue(name=name, labels=labels or {})
        return self._counters[key]

    def gauge(self, name: str, labels: dict[str, str] | None = None) -> GaugeValue:
        """Get or create a gauge."""
        key = self._key(name, labels)
        if key not in self._gauges:
            self._gauges[key] = GaugeValue(name=name, labels=labels or {})
        return self._gauges[key]

    def histogram(self, name: str, buckets: list[float] | None = None) -> HistogramValue:
        """Get or create a histogram."""
        if name not in self._histograms:
            self._histograms[name] = HistogramValue(name=name, buckets=buckets)
        return self._histograms[name]

    def export(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        for key, counter in sorted(self._counters.items()):
            label_str = self._format_labels(counter.labels)
            lines.append(f"{counter.name}{label_str} {counter.value}")
        for key, gauge in sorted(self._gauges.items()):
            label_str = self._format_labels(gauge.labels)
            lines.append(f"{gauge.name}{label_str} {gauge.value}")
        for name, hist in sorted(self._histograms.items()):
            lines.append(f"{name}_count {hist.count}")
            lines.append(f"{name}_sum {hist.sum}")
            lines.append(f"{name}_mean {hist.mean:.4f}")
        return "\n".join(lines)

    def export_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        result: dict[str, Any] = {"counters": {}, "gauges": {}, "histograms": {}}
        for key, c in self._counters.items():
            result["counters"][key] = {"value": c.value, "labels": c.labels}
        for key, g in self._gauges.items():
            result["gauges"][key] = {"value": g.value, "labels": g.labels}
        for name, h in self._histograms.items():
            result["histograms"][name] = {"count": h.count, "sum": h.sum, "mean": h.mean}
        return result

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()

    def _key(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        label_parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_parts}}}"

    def _format_labels(self, labels: dict[str, str]) -> str:
        if not labels:
            return ""
        parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{{{parts}}}"
