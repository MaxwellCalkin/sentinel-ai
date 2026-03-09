"""Health checker for LLM safety infrastructure.

Monitor the health of safety scanners, guardrails, and
infrastructure components. Report latency, availability,
and component status.

Usage:
    from sentinel.health_check import HealthCheck

    health = HealthCheck()
    health.register("pii_scanner", check_fn=lambda: True)
    health.register("injection_scanner", check_fn=lambda: True)
    report = health.check_all()
    print(report.healthy)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    healthy: bool
    latency_ms: float
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Overall health report."""
    healthy: bool
    components: list[ComponentHealth]
    total: int
    healthy_count: int
    unhealthy_count: int
    avg_latency_ms: float
    max_latency_ms: float
    timestamp: float = field(default_factory=time.time)


class HealthCheck:
    """Health checker for safety infrastructure."""

    def __init__(self, timeout: float = 5.0) -> None:
        """
        Args:
            timeout: Max seconds for a health check.
        """
        self._components: dict[str, dict[str, Any]] = {}
        self._timeout = timeout

    def register(
        self,
        name: str,
        check_fn: Callable[[], bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a component for health checking.

        Args:
            name: Component name.
            check_fn: Function that returns True if healthy.
            metadata: Optional metadata.
        """
        self._components[name] = {
            "check_fn": check_fn or (lambda: True),
            "metadata": metadata or {},
        }

    def check(self, name: str) -> ComponentHealth:
        """Check health of a single component."""
        if name not in self._components:
            return ComponentHealth(
                name=name, healthy=False, latency_ms=0.0,
                message="Component not registered",
            )

        comp = self._components[name]
        check_fn = comp["check_fn"]

        start = time.time()
        try:
            result = check_fn()
            latency = (time.time() - start) * 1000
            healthy = bool(result)
            msg = "" if healthy else "Check returned False"
        except Exception as e:
            latency = (time.time() - start) * 1000
            healthy = False
            msg = f"Check failed: {type(e).__name__}: {e}"

        return ComponentHealth(
            name=name, healthy=healthy, latency_ms=round(latency, 2),
            message=msg, metadata=comp["metadata"],
        )

    def check_all(self) -> HealthReport:
        """Check health of all registered components."""
        results = []
        for name in self._components:
            results.append(self.check(name))

        total = len(results)
        healthy_count = sum(1 for r in results if r.healthy)
        unhealthy_count = total - healthy_count
        latencies = [r.latency_ms for r in results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0

        return HealthReport(
            healthy=unhealthy_count == 0,
            components=results,
            total=total,
            healthy_count=healthy_count,
            unhealthy_count=unhealthy_count,
            avg_latency_ms=round(avg_latency, 2),
            max_latency_ms=round(max_latency, 2),
        )

    def unregister(self, name: str) -> bool:
        """Remove a component from health checking."""
        if name in self._components:
            del self._components[name]
            return True
        return False

    def list_components(self) -> list[str]:
        """List registered component names."""
        return list(self._components.keys())
