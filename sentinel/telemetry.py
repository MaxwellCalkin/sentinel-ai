"""OpenTelemetry integration for Sentinel AI.

Emits traces and metrics for every scan operation, enabling
observability dashboards, alerting, and performance monitoring.

Usage:
    from sentinel.telemetry import InstrumentedGuard

    guard = InstrumentedGuard.default()
    result = guard.scan("text")  # automatically emits traces + metrics

Or wrap an existing guard:
    from sentinel import SentinelGuard
    from sentinel.telemetry import instrument

    guard = SentinelGuard.default()
    instrumented = instrument(guard)
"""

from __future__ import annotations

import time
import logging
from functools import wraps
from typing import Any, Callable

from sentinel.core import SentinelGuard, ScanResult, RiskLevel, Finding

logger = logging.getLogger("sentinel.telemetry")

# Metrics counters (in-process, no dependency required)
_metrics: dict[str, int | float] = {
    "scans_total": 0,
    "scans_blocked": 0,
    "scans_safe": 0,
    "findings_total": 0,
    "latency_sum_ms": 0.0,
    "latency_max_ms": 0.0,
}

_risk_counts: dict[str, int] = {r.value: 0 for r in RiskLevel}
_scanner_counts: dict[str, int] = {}
_category_counts: dict[str, int] = {}


def get_metrics() -> dict[str, Any]:
    """Return current metrics snapshot."""
    avg_latency = (
        _metrics["latency_sum_ms"] / _metrics["scans_total"]
        if _metrics["scans_total"] > 0
        else 0.0
    )
    return {
        **_metrics,
        "latency_avg_ms": round(avg_latency, 2),
        "risk_distribution": dict(_risk_counts),
        "scanner_counts": dict(_scanner_counts),
        "category_counts": dict(_category_counts),
    }


def reset_metrics() -> None:
    """Reset all metrics counters."""
    for k in _metrics:
        _metrics[k] = 0 if isinstance(_metrics[k], int) else 0.0
    for k in _risk_counts:
        _risk_counts[k] = 0
    _scanner_counts.clear()
    _category_counts.clear()


def _record(result: ScanResult) -> None:
    """Record metrics from a scan result."""
    _metrics["scans_total"] += 1
    _metrics["latency_sum_ms"] += result.latency_ms
    _metrics["latency_max_ms"] = max(_metrics["latency_max_ms"], result.latency_ms)

    if result.blocked:
        _metrics["scans_blocked"] += 1
    if result.safe:
        _metrics["scans_safe"] += 1

    _metrics["findings_total"] += len(result.findings)
    _risk_counts[result.risk.value] = _risk_counts.get(result.risk.value, 0) + 1

    for f in result.findings:
        _scanner_counts[f.scanner] = _scanner_counts.get(f.scanner, 0) + 1
        _category_counts[f.category] = _category_counts.get(f.category, 0) + 1

    # Emit structured log for SIEM/log aggregation
    logger.info(
        "sentinel.scan",
        extra={
            "sentinel_safe": result.safe,
            "sentinel_blocked": result.blocked,
            "sentinel_risk": result.risk.value,
            "sentinel_findings_count": len(result.findings),
            "sentinel_latency_ms": result.latency_ms,
            "sentinel_categories": list({f.category for f in result.findings}),
        },
    )

    # Try OpenTelemetry if available
    try:
        _emit_otel_span(result)
    except ImportError:
        pass


def _emit_otel_span(result: ScanResult) -> None:
    """Emit OpenTelemetry span if the SDK is installed."""
    from opentelemetry import trace

    tracer = trace.get_tracer("sentinel-ai")
    with tracer.start_as_current_span("sentinel.scan") as span:
        span.set_attribute("sentinel.safe", result.safe)
        span.set_attribute("sentinel.blocked", result.blocked)
        span.set_attribute("sentinel.risk", result.risk.value)
        span.set_attribute("sentinel.findings_count", len(result.findings))
        span.set_attribute("sentinel.latency_ms", result.latency_ms)
        if result.findings:
            span.set_attribute(
                "sentinel.categories",
                list({f.category for f in result.findings}),
            )


class InstrumentedGuard:
    """SentinelGuard wrapper that automatically records telemetry."""

    def __init__(self, guard: SentinelGuard):
        self._guard = guard

    @classmethod
    def default(cls) -> InstrumentedGuard:
        return cls(SentinelGuard.default())

    def scan(self, text: str, context: dict | None = None) -> ScanResult:
        result = self._guard.scan(text, context)
        _record(result)
        return result

    async def scan_async(self, text: str, context: dict | None = None) -> ScanResult:
        result = await self._guard.scan_async(text, context)
        _record(result)
        return result

    @property
    def metrics(self) -> dict[str, Any]:
        return get_metrics()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._guard, name)


def instrument(guard: SentinelGuard) -> InstrumentedGuard:
    """Wrap a SentinelGuard with telemetry instrumentation."""
    return InstrumentedGuard(guard)
