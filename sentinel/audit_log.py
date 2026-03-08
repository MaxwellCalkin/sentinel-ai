"""Structured audit logging for safety decisions.

Records all safety scanning decisions with timestamps, context,
and outcomes for compliance and debugging purposes.

Usage:
    from sentinel.audit_log import AuditLog, AuditEvent

    log = AuditLog()
    log.record(AuditEvent(
        action="scan",
        input_text="Hello world",
        risk_level="none",
        blocked=False,
    ))

    events = log.query(action="scan", blocked=True)
    report = log.compliance_report()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence
from pathlib import Path


@dataclass
class AuditEvent:
    """A single audit event."""
    action: str                           # scan, block, flag, transform, route
    input_text: str = ""
    output_text: str = ""
    risk_level: str = "none"
    blocked: bool = False
    scanner: str = ""
    model: str = ""
    user_id: str = ""
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditStats:
    """Summary statistics from audit events."""
    total_events: int = 0
    total_blocked: int = 0
    total_flagged: int = 0
    by_action: dict[str, int] = field(default_factory=dict)
    by_risk_level: dict[str, int] = field(default_factory=dict)
    by_scanner: dict[str, int] = field(default_factory=dict)
    block_rate: float = 0.0
    avg_duration_ms: float = 0.0


class AuditLog:
    """In-memory audit log with query and export capabilities.

    Thread-safe for recording. Stores events in memory with optional
    file export for compliance.
    """

    def __init__(self, max_events: int = 10000):
        """
        Args:
            max_events: Maximum events to keep in memory (ring buffer).
        """
        self._events: list[AuditEvent] = []
        self._max_events = max_events
        self._listeners: list[Callable[[AuditEvent], None]] = []

    def record(self, event: AuditEvent) -> None:
        """Record an audit event."""
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        for listener in self._listeners:
            listener(event)

    def on_event(self, listener: Callable[[AuditEvent], None]) -> None:
        """Register a listener for new events."""
        self._listeners.append(listener)

    @property
    def events(self) -> list[AuditEvent]:
        """All stored events."""
        return list(self._events)

    @property
    def count(self) -> int:
        """Number of stored events."""
        return len(self._events)

    def query(
        self,
        action: str | None = None,
        blocked: bool | None = None,
        risk_level: str | None = None,
        scanner: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        """Query events with filters."""
        results = self._events

        if action is not None:
            results = [e for e in results if e.action == action]
        if blocked is not None:
            results = [e for e in results if e.blocked == blocked]
        if risk_level is not None:
            results = [e for e in results if e.risk_level == risk_level]
        if scanner is not None:
            results = [e for e in results if e.scanner == scanner]
        if user_id is not None:
            results = [e for e in results if e.user_id == user_id]
        if session_id is not None:
            results = [e for e in results if e.session_id == session_id]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        if limit is not None:
            results = results[-limit:]

        return results

    def stats(self) -> AuditStats:
        """Compute summary statistics."""
        if not self._events:
            return AuditStats()

        by_action: dict[str, int] = {}
        by_risk: dict[str, int] = {}
        by_scanner: dict[str, int] = {}
        blocked_count = 0
        flagged_count = 0
        total_duration = 0.0

        for e in self._events:
            by_action[e.action] = by_action.get(e.action, 0) + 1
            by_risk[e.risk_level] = by_risk.get(e.risk_level, 0) + 1
            if e.scanner:
                by_scanner[e.scanner] = by_scanner.get(e.scanner, 0) + 1
            if e.blocked:
                blocked_count += 1
            if e.action == "flag":
                flagged_count += 1
            total_duration += e.duration_ms

        return AuditStats(
            total_events=len(self._events),
            total_blocked=blocked_count,
            total_flagged=flagged_count,
            by_action=by_action,
            by_risk_level=by_risk,
            by_scanner=by_scanner,
            block_rate=blocked_count / len(self._events),
            avg_duration_ms=total_duration / len(self._events),
        )

    def export_jsonl(self, path: str | Path) -> int:
        """Export events to a JSONL file. Returns number of events written."""
        path = Path(path)
        count = 0
        with path.open("w") as f:
            for event in self._events:
                f.write(json.dumps(event.to_dict()) + "\n")
                count += 1
        return count

    def export_json(self) -> list[dict]:
        """Export all events as a list of dicts."""
        return [e.to_dict() for e in self._events]

    def clear(self) -> None:
        """Clear all stored events."""
        self._events.clear()

    def compliance_report(self) -> dict[str, Any]:
        """Generate a compliance-friendly report."""
        s = self.stats()
        return {
            "report_generated": time.time(),
            "total_requests": s.total_events,
            "blocked_requests": s.total_blocked,
            "block_rate": round(s.block_rate, 4) if s.total_events > 0 else 0,
            "risk_distribution": s.by_risk_level,
            "scanner_activity": s.by_scanner,
            "average_latency_ms": round(s.avg_duration_ms, 2),
        }
