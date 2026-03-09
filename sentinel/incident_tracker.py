"""Security incident tracking and response.

Log, categorize, and manage safety incidents for compliance
reporting and continuous improvement.

Usage:
    from sentinel.incident_tracker import IncidentTracker

    tracker = IncidentTracker()
    tracker.report("injection_attempt", prompt="Ignore all rules", severity="high")
"""

from __future__ import annotations

import time
import hashlib
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class IncidentSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IncidentState(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


@dataclass
class Incident:
    """A single security incident."""
    id: str
    category: str
    severity: IncidentSeverity
    state: IncidentState
    description: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved_at: float | None = None
    resolution: str | None = None


@dataclass
class IncidentSummary:
    """Summary of incidents."""
    total: int
    by_severity: dict[str, int]
    by_category: dict[str, int]
    by_state: dict[str, int]
    open_count: int
    avg_resolution_time: float | None


class IncidentTracker:
    """Track and manage security incidents.

    Provides incident reporting, state management, callbacks,
    and summary reporting for compliance and audit.
    """

    def __init__(self) -> None:
        self._incidents: dict[str, Incident] = {}
        self._callbacks: list[Callable[[Incident], None]] = []
        self._lock = threading.Lock()
        self._counter = 0

    def _generate_id(self) -> str:
        self._counter += 1
        raw = f"{time.time()}-{self._counter}"
        return "INC-" + hashlib.md5(raw.encode()).hexdigest()[:8].upper()

    def on_incident(self, callback: Callable[[Incident], None]) -> None:
        """Register a callback for new incidents."""
        self._callbacks.append(callback)

    def report(
        self,
        category: str,
        description: str = "",
        severity: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> Incident:
        """Report a new incident.

        Args:
            category: Incident category (e.g., "injection_attempt").
            description: Human-readable description.
            severity: One of critical, high, medium, low, info.
            metadata: Additional data (prompt, response, user_id, etc.).

        Returns:
            The created Incident.
        """
        sev = IncidentSeverity(severity)
        incident = Incident(
            id=self._generate_id(),
            category=category,
            severity=sev,
            state=IncidentState.OPEN,
            description=description,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        with self._lock:
            self._incidents[incident.id] = incident

        for cb in self._callbacks:
            try:
                cb(incident)
            except Exception:
                pass

        return incident

    def get(self, incident_id: str) -> Incident | None:
        """Get an incident by ID."""
        return self._incidents.get(incident_id)

    def update_state(
        self,
        incident_id: str,
        state: str,
        resolution: str | None = None,
    ) -> bool:
        """Update incident state.

        Args:
            incident_id: Incident ID.
            state: New state (open, investigating, resolved, dismissed).
            resolution: Resolution description (for resolved/dismissed).

        Returns:
            True if incident was found and updated.
        """
        with self._lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                return False
            inc.state = IncidentState(state)
            if state in ("resolved", "dismissed"):
                inc.resolved_at = time.time()
                inc.resolution = resolution
            return True

    def list_incidents(
        self,
        category: str | None = None,
        severity: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> list[Incident]:
        """List incidents with optional filters.

        Args:
            category: Filter by category.
            severity: Filter by severity.
            state: Filter by state.
            limit: Max results.
        """
        results = list(self._incidents.values())

        if category:
            results = [i for i in results if i.category == category]
        if severity:
            sev = IncidentSeverity(severity)
            results = [i for i in results if i.severity == sev]
        if state:
            st = IncidentState(state)
            results = [i for i in results if i.state == st]

        results.sort(key=lambda i: i.timestamp, reverse=True)
        return results[:limit]

    def open_incidents(self) -> list[Incident]:
        """Get all open incidents."""
        return self.list_incidents(state="open")

    @property
    def total_count(self) -> int:
        return len(self._incidents)

    def summary(self) -> IncidentSummary:
        """Generate incident summary."""
        incidents = list(self._incidents.values())

        by_severity: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_state: dict[str, int] = {}
        resolution_times: list[float] = []

        for inc in incidents:
            by_severity[inc.severity.value] = by_severity.get(inc.severity.value, 0) + 1
            by_category[inc.category] = by_category.get(inc.category, 0) + 1
            by_state[inc.state.value] = by_state.get(inc.state.value, 0) + 1
            if inc.resolved_at:
                resolution_times.append(inc.resolved_at - inc.timestamp)

        open_count = by_state.get("open", 0)
        avg_resolution = (
            sum(resolution_times) / len(resolution_times)
            if resolution_times else None
        )

        return IncidentSummary(
            total=len(incidents),
            by_severity=by_severity,
            by_category=by_category,
            by_state=by_state,
            open_count=open_count,
            avg_resolution_time=avg_resolution,
        )

    def clear_resolved(self) -> int:
        """Remove resolved/dismissed incidents. Returns count removed."""
        with self._lock:
            to_remove = [
                k for k, v in self._incidents.items()
                if v.state in (IncidentState.RESOLVED, IncidentState.DISMISSED)
            ]
            for k in to_remove:
                del self._incidents[k]
            return len(to_remove)
