"""Safety alert management with severity, routing, and escalation.

Create, acknowledge, resolve, and escalate safety alerts.
Supports notification channels, source suppression, and statistics.

Usage:
    from sentinel.alert_manager import AlertManager

    manager = AlertManager()
    manager.add_channel("log", lambda alert: print(alert.title))
    alert = manager.create_alert("Injection detected", "critical", source="scanner")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

SEVERITY_LEVELS = ("info", "warning", "error", "critical")


@dataclass
class Alert:
    """A single safety alert."""

    id: str
    title: str
    severity: str
    source: str
    details: str
    status: str
    created_at: float
    acknowledged_at: float | None = None
    resolved_at: float | None = None
    acknowledged_by: str = ""
    resolution: str = ""


@dataclass
class AlertStats:
    """Aggregate alert statistics."""

    total: int
    active: int
    acknowledged: int
    resolved: int
    by_severity: dict[str, int] = field(default_factory=dict)
    avg_resolution_time: float | None = None


class AlertManager:
    """Manage safety alerts with severity levels, routing, and escalation."""

    def __init__(self, auto_escalate_minutes: int = 30) -> None:
        self._auto_escalate_minutes = auto_escalate_minutes
        self._alerts: dict[str, Alert] = {}
        self._channels: dict[str, Callable[[Alert], None]] = {}
        self._suppressions: dict[str, float] = {}

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _validate_severity(self, severity: str) -> None:
        if severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"Invalid severity '{severity}'. Must be one of: {', '.join(SEVERITY_LEVELS)}"
            )

    def _is_source_suppressed(self, source: str) -> bool:
        if not source or source not in self._suppressions:
            return False
        expiry = self._suppressions[source]
        if time.time() < expiry:
            return True
        del self._suppressions[source]
        return False

    def _notify_channels(self, alert: Alert) -> None:
        for handler in self._channels.values():
            try:
                handler(alert)
            except Exception:
                pass

    def _require_alert(self, alert_id: str) -> Alert:
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise KeyError(f"Alert '{alert_id}' not found")
        return alert

    def create_alert(
        self,
        title: str,
        severity: str,
        source: str = "",
        details: str = "",
    ) -> Alert:
        """Create a new alert and notify channels unless the source is suppressed."""
        self._validate_severity(severity)

        alert = Alert(
            id=self._generate_id(),
            title=title,
            severity=severity,
            source=source,
            details=details,
            status="active",
            created_at=time.time(),
        )
        self._alerts[alert.id] = alert

        if not self._is_source_suppressed(source):
            self._notify_channels(alert)

        return alert

    def acknowledge(self, alert_id: str, by: str = "") -> None:
        """Mark an alert as acknowledged."""
        alert = self._require_alert(alert_id)
        alert.status = "acknowledged"
        alert.acknowledged_at = time.time()
        alert.acknowledged_by = by

    def resolve(self, alert_id: str, resolution: str = "") -> None:
        """Mark an alert as resolved."""
        alert = self._require_alert(alert_id)
        alert.status = "resolved"
        alert.resolved_at = time.time()
        alert.resolution = resolution

    def escalate(self, alert_id: str) -> None:
        """Escalate an alert's severity by one level."""
        alert = self._require_alert(alert_id)
        current_index = SEVERITY_LEVELS.index(alert.severity)
        max_index = len(SEVERITY_LEVELS) - 1
        alert.severity = SEVERITY_LEVELS[min(current_index + 1, max_index)]

    def add_channel(self, name: str, handler: Callable[[Alert], None]) -> None:
        """Register a notification channel."""
        self._channels[name] = handler

    def get_active(self) -> list[Alert]:
        """Return all unresolved alerts."""
        return [a for a in self._alerts.values() if a.status != "resolved"]

    def get_by_severity(self, severity: str) -> list[Alert]:
        """Return alerts matching the given severity."""
        self._validate_severity(severity)
        return [a for a in self._alerts.values() if a.severity == severity]

    def suppress(self, source: str, duration_seconds: float = 3600) -> None:
        """Suppress alerts from a source for the given duration."""
        self._suppressions[source] = time.time() + duration_seconds

    def stats(self) -> AlertStats:
        """Compute aggregate alert statistics."""
        alerts = list(self._alerts.values())
        by_severity = _count_by_severity(alerts)
        active, acknowledged, resolved = _count_by_status(alerts)
        avg_resolution_time = _compute_avg_resolution_time(alerts)

        return AlertStats(
            total=len(alerts),
            active=active,
            acknowledged=acknowledged,
            resolved=resolved,
            by_severity=by_severity,
            avg_resolution_time=avg_resolution_time,
        )

    def clear_resolved(self) -> None:
        """Remove all resolved alerts from history."""
        resolved_ids = [
            alert_id
            for alert_id, alert in self._alerts.items()
            if alert.status == "resolved"
        ]
        for alert_id in resolved_ids:
            del self._alerts[alert_id]


def _count_by_severity(alerts: list[Alert]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for alert in alerts:
        counts[alert.severity] = counts.get(alert.severity, 0) + 1
    return counts


def _count_by_status(alerts: list[Alert]) -> tuple[int, int, int]:
    active = 0
    acknowledged = 0
    resolved = 0
    for alert in alerts:
        if alert.status == "active":
            active += 1
        elif alert.status == "acknowledged":
            acknowledged += 1
        elif alert.status == "resolved":
            resolved += 1
    return active, acknowledged, resolved


def _compute_avg_resolution_time(alerts: list[Alert]) -> float | None:
    resolution_times = [
        alert.resolved_at - alert.created_at
        for alert in alerts
        if alert.resolved_at is not None
    ]
    if not resolution_times:
        return None
    return sum(resolution_times) / len(resolution_times)
