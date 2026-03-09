"""Explainability tracer for safety decisions.

Trace and explain why content was blocked, flagged, or allowed.
Generate human-readable audit trails for compliance and debugging.

Usage:
    from sentinel.explainability import ExplainabilityTracer

    tracer = ExplainabilityTracer()
    with tracer.trace("scan-123") as t:
        t.record_check("pii_scan", passed=True, detail="No PII found")
        t.record_check("injection_scan", passed=False, detail="SQL injection detected", severity="high")
    explanation = tracer.explain("scan-123")
    print(explanation.summary)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from contextlib import contextmanager


@dataclass
class CheckRecord:
    """A single safety check record."""
    name: str
    passed: bool
    detail: str = ""
    severity: str = "medium"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    """Complete trace of a safety decision."""
    trace_id: str
    checks: list[CheckRecord] = field(default_factory=list)
    decision: str = ""  # "allow", "block", "flag"
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[CheckRecord]:
        return [c for c in self.checks if not c.passed]


@dataclass
class Explanation:
    """Human-readable explanation of a safety decision."""
    trace_id: str
    decision: str
    summary: str
    checks_run: int
    checks_passed: int
    checks_failed: int
    failed_details: list[str]
    duration_ms: float
    severity: str  # highest severity among failed checks


class TraceContext:
    """Context manager for recording checks within a trace."""

    def __init__(self, record: TraceRecord) -> None:
        self._record = record

    def record_check(
        self,
        name: str,
        passed: bool,
        detail: str = "",
        severity: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> CheckRecord:
        check = CheckRecord(
            name=name, passed=passed, detail=detail,
            severity=severity, metadata=metadata or {},
        )
        self._record.checks.append(check)
        return check

    def set_decision(self, decision: str) -> None:
        self._record.decision = decision

    def set_metadata(self, key: str, value: Any) -> None:
        self._record.metadata[key] = value


class ExplainabilityTracer:
    """Trace and explain safety decisions."""

    def __init__(self, max_traces: int = 1000) -> None:
        self._traces: dict[str, TraceRecord] = {}
        self._max_traces = max_traces

    @contextmanager
    def trace(self, trace_id: str, metadata: dict[str, Any] | None = None):
        """Context manager for tracing a safety decision."""
        record = TraceRecord(trace_id=trace_id, metadata=metadata or {})
        self._traces[trace_id] = record

        # Evict old traces if over limit
        while len(self._traces) > self._max_traces:
            oldest = next(iter(self._traces))
            del self._traces[oldest]

        ctx = TraceContext(record)
        try:
            yield ctx
        finally:
            record.end_time = time.time()
            # Auto-determine decision if not set
            if not record.decision:
                if record.all_passed:
                    record.decision = "allow"
                elif any(c.severity in ("critical", "high") for c in record.failed_checks):
                    record.decision = "block"
                else:
                    record.decision = "flag"

    def explain(self, trace_id: str) -> Explanation | None:
        """Generate human-readable explanation for a trace."""
        record = self._traces.get(trace_id)
        if not record:
            return None

        checks_run = len(record.checks)
        checks_passed = sum(1 for c in record.checks if c.passed)
        checks_failed = checks_run - checks_passed
        failed_details = [f"{c.name}: {c.detail}" for c in record.failed_checks]

        # Determine highest severity
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        max_severity = "low"
        for c in record.failed_checks:
            if severity_order.get(c.severity, 0) > severity_order.get(max_severity, 0):
                max_severity = c.severity

        # Generate summary
        if record.decision == "allow":
            summary = f"Content allowed: all {checks_run} safety checks passed."
        elif record.decision == "block":
            summary = f"Content blocked: {checks_failed} of {checks_run} checks failed ({max_severity} severity)."
        else:
            summary = f"Content flagged: {checks_failed} of {checks_run} checks raised concerns."

        return Explanation(
            trace_id=trace_id,
            decision=record.decision,
            summary=summary,
            checks_run=checks_run,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            failed_details=failed_details,
            duration_ms=round(record.duration_ms, 2),
            severity=max_severity if checks_failed > 0 else "none",
        )

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        """Get raw trace record."""
        return self._traces.get(trace_id)

    def list_traces(self, decision: str | None = None) -> list[str]:
        """List trace IDs, optionally filtered by decision."""
        if decision:
            return [tid for tid, t in self._traces.items() if t.decision == decision]
        return list(self._traces.keys())

    def clear(self) -> int:
        count = len(self._traces)
        self._traces.clear()
        return count
