"""Unified safety gateway for orchestrating multiple safety checks.

Register input checks (run before LLM call) and output checks
(run after), then process requests through a single gateway API
that runs all enabled checks in priority order.

Usage:
    from sentinel.safety_gateway import SafetyGateway

    gw = SafetyGateway()
    gw.add_input_check("length", lambda text: (len(text) < 1000, "Too long"))
    gw.add_output_check("toxicity", lambda text: ("hate" not in text, "Toxic"))

    result = gw.process(input_text="Hello", output_text="World")
    assert result.passed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

INPUT_PHASE = "input"
OUTPUT_PHASE = "output"

CheckFunction = Callable[[str], tuple[bool, str]]


@dataclass
class GatewayCheck:
    """A registered safety check."""
    name: str
    check_fn: CheckFunction
    priority: int
    phase: str
    enabled: bool = True


@dataclass
class GatewayCheckResult:
    """Result of a single check execution."""
    name: str
    passed: bool
    message: str
    phase: str


@dataclass
class GatewayResult:
    """Result of processing through the gateway."""
    passed: bool
    input_results: list[GatewayCheckResult] = field(default_factory=list)
    output_results: list[GatewayCheckResult] = field(default_factory=list)
    short_circuited: bool = False


@dataclass
class GatewayStats:
    """Aggregate gateway statistics."""
    total_requests: int
    input_pass_rate: float
    output_pass_rate: float


class SafetyGateway:
    """Unified entry point that orchestrates multiple safety checks.

    Checks are organized into two phases:
      - input: run before the LLM call
      - output: run after the LLM call

    Within each phase, checks execute in ascending priority order
    (lower number = higher priority = runs first). When short_circuit
    is enabled, the first failing check in a phase stops further
    checks in that phase.
    """

    def __init__(self, *, short_circuit: bool = False) -> None:
        self._checks: list[GatewayCheck] = []
        self._short_circuit = short_circuit
        self._total_requests = 0
        self._input_passes = 0
        self._input_totals = 0
        self._output_passes = 0
        self._output_totals = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_input_check(
        self,
        name: str,
        check_fn: CheckFunction,
        priority: int = 0,
        enabled: bool = True,
    ) -> None:
        """Register a check that runs during the input phase."""
        self._checks.append(GatewayCheck(
            name=name,
            check_fn=check_fn,
            priority=priority,
            phase=INPUT_PHASE,
            enabled=enabled,
        ))

    def add_output_check(
        self,
        name: str,
        check_fn: CheckFunction,
        priority: int = 0,
        enabled: bool = True,
    ) -> None:
        """Register a check that runs during the output phase."""
        self._checks.append(GatewayCheck(
            name=name,
            check_fn=check_fn,
            priority=priority,
            phase=OUTPUT_PHASE,
            enabled=enabled,
        ))

    # ------------------------------------------------------------------
    # Runtime enable / disable
    # ------------------------------------------------------------------

    def enable_check(self, name: str) -> bool:
        """Enable a check by name. Returns True if the check was found."""
        return self._set_enabled(name, True)

    def disable_check(self, name: str) -> bool:
        """Disable a check by name. Returns True if the check was found."""
        return self._set_enabled(name, False)

    def _set_enabled(self, name: str, enabled: bool) -> bool:
        found = False
        for check in self._checks:
            if check.name == name:
                check.enabled = enabled
                found = True
        return found

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self,
        input_text: str = "",
        output_text: str = "",
    ) -> GatewayResult:
        """Run all enabled checks against the provided texts.

        Input checks receive *input_text*; output checks receive
        *output_text*. Returns a GatewayResult summarizing all
        individual check outcomes and overall pass/fail.
        """
        self._total_requests += 1

        input_results, input_short_circuited = self._run_phase(
            INPUT_PHASE, input_text,
        )
        output_results, output_short_circuited = self._run_phase(
            OUTPUT_PHASE, output_text,
        )

        self._record_stats(input_results, output_results)

        all_passed = (
            all(r.passed for r in input_results)
            and all(r.passed for r in output_results)
        )
        short_circuited = input_short_circuited or output_short_circuited

        return GatewayResult(
            passed=all_passed,
            input_results=input_results,
            output_results=output_results,
            short_circuited=short_circuited,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> GatewayStats:
        """Return aggregate pass-rate statistics."""
        return GatewayStats(
            total_requests=self._total_requests,
            input_pass_rate=self._safe_rate(
                self._input_passes, self._input_totals,
            ),
            output_pass_rate=self._safe_rate(
                self._output_passes, self._output_totals,
            ),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def check_count(self) -> int:
        """Total number of registered checks."""
        return len(self._checks)

    @property
    def enabled_check_count(self) -> int:
        """Number of currently enabled checks."""
        return sum(1 for c in self._checks if c.enabled)

    def get_checks(self, phase: str | None = None) -> list[GatewayCheck]:
        """Return a copy of registered checks, optionally filtered by phase."""
        if phase is None:
            return list(self._checks)
        return [c for c in self._checks if c.phase == phase]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_phase(
        self, phase: str, text: str,
    ) -> tuple[list[GatewayCheckResult], bool]:
        """Run all enabled checks for a given phase in priority order."""
        phase_checks = sorted(
            (c for c in self._checks if c.phase == phase and c.enabled),
            key=lambda c: c.priority,
        )

        results: list[GatewayCheckResult] = []
        short_circuited = False

        for check in phase_checks:
            passed, message = check.check_fn(text)
            results.append(GatewayCheckResult(
                name=check.name,
                passed=passed,
                message=message,
                phase=phase,
            ))
            if not passed and self._short_circuit:
                short_circuited = True
                break

        return results, short_circuited

    def _record_stats(
        self,
        input_results: list[GatewayCheckResult],
        output_results: list[GatewayCheckResult],
    ) -> None:
        """Update running pass-rate counters."""
        for result in input_results:
            self._input_totals += 1
            if result.passed:
                self._input_passes += 1

        for result in output_results:
            self._output_totals += 1
            if result.passed:
                self._output_passes += 1

    @staticmethod
    def _safe_rate(passes: int, total: int) -> float:
        if total == 0:
            return 1.0
        return passes / total
