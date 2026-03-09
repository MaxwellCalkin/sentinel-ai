"""Pre-deployment compliance gating for LLM applications.

Blocks deployments unless all required compliance checks pass.
Recommended checks that fail produce warnings but do not block.
Tracks evaluation history and provides aggregate statistics.

Usage:
    from sentinel.compliance_gate import ComplianceGate, ComplianceCheck

    gate = ComplianceGate()
    gate.add_check(ComplianceCheck(
        name="model_versioned",
        check_fn=lambda: True,
        severity="required",
        description="Model must have a version tag",
    ))
    decision = gate.evaluate()
    print(decision.passed)  # True
"""

from __future__ import annotations

import time
from dataclasses import dataclass

_VALID_SEVERITIES = frozenset({"required", "recommended"})


@dataclass
class ComplianceCheck:
    """A single compliance check definition."""

    name: str
    check_fn: object  # Callable[[], bool] — stored as object to avoid runtime typing import
    severity: str
    description: str = ""


@dataclass
class CheckOutcome:
    """Result of executing a single compliance check."""

    name: str
    passed: bool
    message: str
    duration_ms: float
    severity: str


@dataclass
class GateDecision:
    """Aggregate result of a gate evaluation."""

    passed: bool
    outcomes: list[CheckOutcome]
    required_passed: int
    required_failed: int
    warnings: list[str]
    timestamp: float


@dataclass
class GateStats:
    """Aggregate statistics across all gate evaluations."""

    total_evaluations: int
    pass_count: int
    pass_rate: float
    failed_checks: dict[str, int]


def _validate_severity(severity: str) -> None:
    """Raise ValueError if severity is not recognized."""
    if severity not in _VALID_SEVERITIES:
        raise ValueError(
            f"Invalid severity '{severity}'. "
            f"Must be one of: {', '.join(sorted(_VALID_SEVERITIES))}"
        )


def _run_single_check(check: ComplianceCheck) -> CheckOutcome:
    """Execute one check and capture its outcome with timing."""
    start = time.monotonic()
    try:
        passed = bool(check.check_fn())
    except Exception as exc:
        passed = False
        elapsed_ms = (time.monotonic() - start) * 1000
        return CheckOutcome(
            name=check.name,
            passed=False,
            message=f"Exception: {exc}",
            duration_ms=round(elapsed_ms, 3),
            severity=check.severity,
        )
    elapsed_ms = (time.monotonic() - start) * 1000
    message = "passed" if passed else "failed"
    return CheckOutcome(
        name=check.name,
        passed=passed,
        message=message,
        duration_ms=round(elapsed_ms, 3),
        severity=check.severity,
    )


def _build_warnings(outcomes: list[CheckOutcome]) -> list[str]:
    """Collect warning messages for failed recommended checks."""
    return [
        f"Recommended check '{outcome.name}' failed"
        for outcome in outcomes
        if outcome.severity == "recommended" and not outcome.passed
    ]


def _build_decision(outcomes: list[CheckOutcome]) -> GateDecision:
    """Determine gate pass/fail from individual outcomes."""
    required_outcomes = [o for o in outcomes if o.severity == "required"]
    required_passed = sum(1 for o in required_outcomes if o.passed)
    required_failed = sum(1 for o in required_outcomes if not o.passed)
    warnings = _build_warnings(outcomes)
    gate_passed = required_failed == 0

    return GateDecision(
        passed=gate_passed,
        outcomes=outcomes,
        required_passed=required_passed,
        required_failed=required_failed,
        warnings=warnings,
        timestamp=time.time(),
    )


class ComplianceGate:
    """Pre-deployment compliance gate that blocks unless all required checks pass.

    Register checks with ``add_check``, then call ``evaluate`` to run them
    all and receive a ``GateDecision``.  Failed recommended checks generate
    warnings but do not block the gate.  Every evaluation is recorded in
    history for later analysis via ``stats``.
    """

    def __init__(self) -> None:
        self._checks: list[ComplianceCheck] = []
        self._history: list[GateDecision] = []

    def add_check(self, check: ComplianceCheck) -> None:
        """Register a compliance check.

        Args:
            check: The check definition to add.

        Raises:
            ValueError: If check severity is not 'required' or 'recommended'.
            ValueError: If a check with the same name already exists.
        """
        _validate_severity(check.severity)
        if any(existing.name == check.name for existing in self._checks):
            raise ValueError(f"Duplicate check name '{check.name}'")
        self._checks.append(check)

    def evaluate(self) -> GateDecision:
        """Run all registered checks and produce a gate decision.

        Returns:
            GateDecision indicating pass/fail, individual outcomes, and warnings.
        """
        outcomes = [_run_single_check(check) for check in self._checks]
        decision = _build_decision(outcomes)
        self._history.append(decision)
        return decision

    @property
    def history(self) -> list[GateDecision]:
        """Return the list of past gate evaluations."""
        return list(self._history)

    @property
    def checks(self) -> list[ComplianceCheck]:
        """Return the list of registered checks."""
        return list(self._checks)

    def stats(self) -> GateStats:
        """Compute aggregate statistics across all evaluations.

        Returns:
            GateStats with total evaluations, pass count, pass rate,
            and a mapping of check names to failure counts.
        """
        total = len(self._history)
        pass_count = sum(1 for d in self._history if d.passed)
        pass_rate = pass_count / total if total > 0 else 0.0
        failed_checks = _aggregate_failed_checks(self._history)

        return GateStats(
            total_evaluations=total,
            pass_count=pass_count,
            pass_rate=round(pass_rate, 4),
            failed_checks=failed_checks,
        )

    def export_dict(self) -> dict:
        """Export the most recent gate decision as a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.

        Raises:
            ValueError: If no evaluations have been run yet.
        """
        if not self._history:
            raise ValueError("No evaluations to export. Call evaluate() first.")
        decision = self._history[-1]
        return _decision_to_dict(decision)

    def remove_check(self, name: str) -> None:
        """Remove a registered check by name.

        Args:
            name: The name of the check to remove.

        Raises:
            KeyError: If no check with the given name exists.
        """
        for i, check in enumerate(self._checks):
            if check.name == name:
                self._checks.pop(i)
                return
        raise KeyError(f"No check named '{name}'")


def _aggregate_failed_checks(history: list[GateDecision]) -> dict[str, int]:
    """Count failures per check name across all evaluations."""
    counts: dict[str, int] = {}
    for decision in history:
        for outcome in decision.outcomes:
            if not outcome.passed:
                counts[outcome.name] = counts.get(outcome.name, 0) + 1
    return counts


def _decision_to_dict(decision: GateDecision) -> dict:
    """Convert a GateDecision to a plain dictionary."""
    return {
        "passed": decision.passed,
        "required_passed": decision.required_passed,
        "required_failed": decision.required_failed,
        "warnings": decision.warnings,
        "timestamp": decision.timestamp,
        "outcomes": [
            {
                "name": o.name,
                "passed": o.passed,
                "message": o.message,
                "duration_ms": o.duration_ms,
                "severity": o.severity,
            }
            for o in decision.outcomes
        ],
    }
