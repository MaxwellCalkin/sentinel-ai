"""Structured logging of safety decisions with queryable history.

Records safety decisions (allow, block, flag, redact) with timestamps,
reasons, and metadata for audit trails and compliance reporting.

Usage:
    from sentinel.decision_logger import DecisionLogger, Decision

    logger = DecisionLogger(max_size=5000)
    logger.log("block", reason="Injection detected", component="PromptInjectionScanner")
    logger.log("allow", reason="Clean input", component="PIIScanner", metadata={"score": 0.1})

    blocked = logger.query(action="block")
    summary = logger.summary()
    export = logger.export()
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Decision:
    """A single safety decision record."""
    action: str
    reason: str
    component: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionSummary:
    """Aggregate statistics over logged decisions."""
    total: int
    by_action: dict[str, int]
    by_component: dict[str, int]
    block_rate: float


def _compute_input_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class DecisionLogger:
    """Thread-safe structured logger for safety decisions.

    Stores decisions in memory with oldest-first eviction when max_size
    is exceeded. Supports querying by action, component, and time range.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._decisions: list[Decision] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def log(
        self,
        action: str,
        reason: str = "",
        component: str = "",
        metadata: dict[str, Any] | None = None,
        input_text: str = "",
    ) -> Decision:
        """Record a safety decision.

        Args:
            action: Decision type (allow, block, flag, redact).
            reason: Human-readable explanation.
            component: Name of the component that made the decision.
            metadata: Arbitrary key-value context.
            input_text: Original input (hashed, not stored raw).

        Returns:
            The recorded Decision.
        """
        input_hash = _compute_input_hash(input_text) if input_text else ""
        decision = Decision(
            action=action,
            reason=reason,
            component=component,
            timestamp=time.time(),
            metadata=metadata or {},
            input_hash=input_hash,
        )

        with self._lock:
            self._decisions.append(decision)
            if len(self._decisions) > self._max_size:
                self._decisions = self._decisions[-self._max_size:]

        return decision

    @property
    def count(self) -> int:
        """Number of decisions currently stored."""
        return len(self._decisions)

    @property
    def decisions(self) -> list[Decision]:
        """Snapshot of all stored decisions (defensive copy)."""
        with self._lock:
            return list(self._decisions)

    def query(
        self,
        action: str | None = None,
        component: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[Decision]:
        """Query decisions with optional filters.

        Args:
            action: Filter by action type.
            component: Filter by component name.
            since: Include decisions at or after this timestamp.
            until: Include decisions at or before this timestamp.

        Returns:
            List of matching decisions.
        """
        with self._lock:
            results = list(self._decisions)

        if action is not None:
            results = [d for d in results if d.action == action]
        if component is not None:
            results = [d for d in results if d.component == component]
        if since is not None:
            results = [d for d in results if d.timestamp >= since]
        if until is not None:
            results = [d for d in results if d.timestamp <= until]

        return results

    def export(self) -> list[dict[str, Any]]:
        """Export all decisions as a list of dicts."""
        with self._lock:
            return [d.to_dict() for d in self._decisions]

    def summary(self) -> DecisionSummary:
        """Compute aggregate statistics over all stored decisions."""
        with self._lock:
            snapshot = list(self._decisions)

        if not snapshot:
            return DecisionSummary(
                total=0,
                by_action={},
                by_component={},
                block_rate=0.0,
            )

        by_action: dict[str, int] = {}
        by_component: dict[str, int] = {}

        for decision in snapshot:
            by_action[decision.action] = by_action.get(decision.action, 0) + 1
            if decision.component:
                by_component[decision.component] = by_component.get(decision.component, 0) + 1

        block_count = by_action.get("block", 0)
        block_rate = block_count / len(snapshot)

        return DecisionSummary(
            total=len(snapshot),
            by_action=by_action,
            by_component=by_component,
            block_rate=block_rate,
        )

    def clear(self) -> None:
        """Remove all stored decisions."""
        with self._lock:
            self._decisions.clear()
