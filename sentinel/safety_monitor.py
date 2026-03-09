"""Real-time safety metric monitoring with alerting.

Track safety metrics over time and trigger alerts when thresholds
are exceeded. Supports configurable alert rules with time windows,
minimum sample counts, and severity levels.

Usage:
    from sentinel.safety_monitor import SafetyMonitor, AlertRule

    monitor = SafetyMonitor()
    monitor.add_rule(AlertRule(
        name="high_toxicity",
        metric="toxicity_score",
        condition="above",
        threshold=0.8,
        window_seconds=60.0,
        severity="critical",
    ))
    monitor.record("toxicity_score", 0.9)
    alerts = monitor.check_alerts()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

VALID_CONDITIONS = ("above", "below", "equals")
VALID_SEVERITIES = ("info", "warning", "error", "critical")


@dataclass
class MetricPoint:
    """A single metric measurement."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: dict = field(default_factory=dict)


@dataclass
class AlertRule:
    """A rule that triggers alerts when metric conditions are met."""

    name: str
    metric: str
    condition: str
    threshold: float
    window_seconds: float = 60.0
    min_samples: int = 1
    severity: str = "warning"


@dataclass
class Alert:
    """A triggered alert."""

    rule_name: str
    metric: str
    current_value: float
    threshold: float
    severity: str
    triggered_at: float
    message: str


@dataclass
class MonitorReport:
    """Health report for the monitoring system."""

    metrics_tracked: int
    total_points: int
    active_alerts: list[Alert]
    rules_count: int
    health: str


@dataclass
class MonitorStats:
    """Cumulative monitoring statistics."""

    total_points: int = 0
    total_alerts: int = 0
    by_metric: dict[str, int] = field(default_factory=dict)


class SafetyMonitor:
    """Real-time safety metric monitoring with threshold-based alerting."""

    def __init__(self, max_points: int = 10000) -> None:
        self._max_points = max_points
        self._points: list[MetricPoint] = []
        self._rules: dict[str, AlertRule] = {}
        self._stats = MonitorStats()

    def record(self, name: str, value: float, tags: dict | None = None) -> None:
        """Record a metric data point."""
        point = MetricPoint(name=name, value=value, tags=tags or {})
        self._points.append(point)
        self._stats.total_points += 1
        self._stats.by_metric[name] = self._stats.by_metric.get(name, 0) + 1
        self._enforce_max_points()

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule by name. Raises KeyError if not found."""
        if name not in self._rules:
            raise KeyError(f"Rule not found: {name}")
        del self._rules[name]

    def check_alerts(self) -> list[Alert]:
        """Evaluate all rules against recent metrics and return triggered alerts."""
        now = time.time()
        alerts: list[Alert] = []
        for rule in self._rules.values():
            alert = self._evaluate_rule(rule, now)
            if alert is not None:
                alerts.append(alert)
        self._stats.total_alerts += len(alerts)
        return alerts

    def get_metric(self, name: str, last_n: int | None = None) -> list[MetricPoint]:
        """Return recorded points for a metric, optionally limited to last N."""
        matching = [p for p in self._points if p.name == name]
        if last_n is not None:
            return matching[-last_n:]
        return matching

    def report(self) -> MonitorReport:
        """Generate a health report based on current alert state."""
        active_alerts = self.check_alerts()
        health = _determine_health(active_alerts)
        metric_names = {p.name for p in self._points}
        return MonitorReport(
            metrics_tracked=len(metric_names),
            total_points=len(self._points),
            active_alerts=active_alerts,
            rules_count=len(self._rules),
            health=health,
        )

    def clear_metric(self, name: str) -> None:
        """Remove all recorded points for a metric."""
        self._points = [p for p in self._points if p.name != name]

    def stats(self) -> MonitorStats:
        """Return cumulative monitoring statistics."""
        return MonitorStats(
            total_points=self._stats.total_points,
            total_alerts=self._stats.total_alerts,
            by_metric=dict(self._stats.by_metric),
        )

    def _evaluate_rule(self, rule: AlertRule, now: float) -> Alert | None:
        """Evaluate a single rule and return an Alert if triggered."""
        window_start = now - rule.window_seconds
        windowed_points = [
            p for p in self._points
            if p.name == rule.metric and p.timestamp >= window_start
        ]
        if len(windowed_points) < rule.min_samples:
            return None
        average = sum(p.value for p in windowed_points) / len(windowed_points)
        if not _condition_met(rule.condition, average, rule.threshold):
            return None
        message = (
            f"{rule.name}: {rule.metric} average {average:.4f} "
            f"is {rule.condition} threshold {rule.threshold}"
        )
        return Alert(
            rule_name=rule.name,
            metric=rule.metric,
            current_value=average,
            threshold=rule.threshold,
            severity=rule.severity,
            triggered_at=now,
            message=message,
        )

    def _enforce_max_points(self) -> None:
        """Evict oldest points when storage exceeds max capacity."""
        if len(self._points) > self._max_points:
            self._points = self._points[-self._max_points:]


def _condition_met(condition: str, value: float, threshold: float) -> bool:
    """Check whether a value satisfies a condition against a threshold."""
    if condition == "above":
        return value > threshold
    if condition == "below":
        return value < threshold
    if condition == "equals":
        return abs(value - threshold) < 1e-9
    return False


def _determine_health(alerts: list[Alert]) -> str:
    """Derive overall health status from active alerts."""
    severities = {a.severity for a in alerts}
    if "critical" in severities:
        return "critical"
    if severities:
        return "degraded"
    return "healthy"
