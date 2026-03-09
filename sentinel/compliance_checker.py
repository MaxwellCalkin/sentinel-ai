"""Automated compliance rule checking with evidence collection.

Define compliance rules, check systems against them, and collect
evidence for audit trails. Supports multiple frameworks (GDPR, SOC2,
HIPAA, NIST) with severity levels and remediation guidance.

Usage:
    from sentinel.compliance_checker import ComplianceChecker, ComplianceRule

    checker = ComplianceChecker()
    checker.add_rule(ComplianceRule(
        rule_id="GDPR-1",
        name="Data Encryption",
        description="All PII must be encrypted at rest",
        framework="GDPR",
        severity="critical",
    ))
    result = checker.check("GDPR-1", passed=True, details="AES-256 verified")
    overview = checker.overview()
    print(overview.pass_rate)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

_VALID_SEVERITIES = frozenset({"critical", "high", "medium", "low"})


@dataclass
class ComplianceRule:
    """A single compliance rule to check against."""

    rule_id: str
    name: str
    description: str
    framework: str
    severity: str
    check_fn: str = ""

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{self.severity}', "
                f"must be one of {sorted(_VALID_SEVERITIES)}"
            )


@dataclass
class Evidence:
    """A single piece of evidence collected during a compliance check."""

    rule_id: str
    collected_at: float
    passed: bool
    details: str
    artifacts: dict = field(default_factory=dict)


@dataclass
class ComplianceResult:
    """The outcome of checking a single compliance rule."""

    rule: ComplianceRule
    passed: bool
    evidence: list[Evidence]
    remediation: str = ""


@dataclass
class ComplianceOverview:
    """High-level summary across all compliance rules."""

    total_rules: int
    passed: int
    failed: int
    pass_rate: float
    by_framework: dict[str, dict[str, int]]
    critical_failures: list[str]


@dataclass
class CheckerStats:
    """Cumulative statistics for all checks performed."""

    total_checks: int = 0
    total_passed: int = 0
    total_failed: int = 0
    by_framework: dict[str, int] = field(default_factory=dict)


class ComplianceChecker:
    """Check systems against compliance rules and collect evidence.

    Register rules, record pass/fail evidence, and generate
    overviews and exports for audit trails.
    """

    def __init__(self) -> None:
        self._rules: dict[str, ComplianceRule] = {}
        self._evidence: dict[str, list[Evidence]] = {}
        self._stats = CheckerStats()

    def add_rule(self, rule: ComplianceRule) -> None:
        """Register a single compliance rule."""
        self._rules[rule.rule_id] = rule
        if rule.rule_id not in self._evidence:
            self._evidence[rule.rule_id] = []

    def add_rules(self, rules: list[ComplianceRule]) -> None:
        """Register multiple compliance rules at once."""
        for rule in rules:
            self.add_rule(rule)

    def check(
        self,
        rule_id: str,
        passed: bool,
        details: str = "",
        artifacts: dict | None = None,
        remediation: str = "",
    ) -> ComplianceResult:
        """Record evidence for a rule check and return the result.

        Raises:
            KeyError: If rule_id is not registered.
        """
        if rule_id not in self._rules:
            raise KeyError(f"Unknown rule_id: '{rule_id}'")

        rule = self._rules[rule_id]
        evidence = Evidence(
            rule_id=rule_id,
            collected_at=time.time(),
            passed=passed,
            details=details,
            artifacts=artifacts or {},
        )
        self._evidence[rule_id].append(evidence)
        self._update_stats(rule, passed)

        return ComplianceResult(
            rule=rule,
            passed=passed,
            evidence=self._evidence[rule_id],
            remediation=remediation,
        )

    def check_all(self, results: dict[str, bool]) -> list[ComplianceResult]:
        """Check multiple rules at once from a rule_id-to-pass/fail mapping."""
        return [
            self.check(rule_id, passed)
            for rule_id, passed in results.items()
        ]

    def overview(self) -> ComplianceOverview:
        """Generate a high-level overview based on latest evidence per rule."""
        passed_count = 0
        failed_count = 0
        by_framework: dict[str, dict[str, int]] = {}
        critical_failures: list[str] = []

        for rule_id, rule in self._rules.items():
            framework = rule.framework
            if framework not in by_framework:
                by_framework[framework] = {"passed": 0, "failed": 0}

            rule_passed = self._latest_passed(rule_id)
            if rule_passed:
                passed_count += 1
                by_framework[framework]["passed"] += 1
            else:
                failed_count += 1
                by_framework[framework]["failed"] += 1
                if rule.severity == "critical":
                    critical_failures.append(rule_id)

        total = len(self._rules)
        pass_rate = (passed_count / total * 100.0) if total > 0 else 0.0

        return ComplianceOverview(
            total_rules=total,
            passed=passed_count,
            failed=failed_count,
            pass_rate=pass_rate,
            by_framework=by_framework,
            critical_failures=critical_failures,
        )

    def get_evidence(self, rule_id: str) -> list[Evidence]:
        """Return all evidence entries for a given rule."""
        if rule_id not in self._rules:
            raise KeyError(f"Unknown rule_id: '{rule_id}'")
        return list(self._evidence.get(rule_id, []))

    def export(self, framework: str | None = None) -> list[dict]:
        """Export results as dicts, optionally filtered by framework."""
        exported: list[dict] = []
        for rule_id, rule in self._rules.items():
            if framework is not None and rule.framework != framework:
                continue

            evidence_list = self._evidence.get(rule_id, [])
            exported.append({
                "rule_id": rule.rule_id,
                "name": rule.name,
                "framework": rule.framework,
                "severity": rule.severity,
                "passed": self._latest_passed(rule_id),
                "evidence_count": len(evidence_list),
                "latest_details": evidence_list[-1].details if evidence_list else "",
            })

        return exported

    def list_rules(self, framework: str | None = None) -> list[ComplianceRule]:
        """List all registered rules, optionally filtered by framework."""
        if framework is None:
            return list(self._rules.values())
        return [
            rule for rule in self._rules.values()
            if rule.framework == framework
        ]

    def stats(self) -> CheckerStats:
        """Return cumulative check statistics."""
        return CheckerStats(
            total_checks=self._stats.total_checks,
            total_passed=self._stats.total_passed,
            total_failed=self._stats.total_failed,
            by_framework=dict(self._stats.by_framework),
        )

    def _latest_passed(self, rule_id: str) -> bool:
        """Return pass/fail based on the most recent evidence entry."""
        evidence_list = self._evidence.get(rule_id, [])
        if not evidence_list:
            return False
        return evidence_list[-1].passed

    def _update_stats(self, rule: ComplianceRule, passed: bool) -> None:
        """Increment cumulative stats counters."""
        self._stats.total_checks += 1
        if passed:
            self._stats.total_passed += 1
        else:
            self._stats.total_failed += 1
        framework = rule.framework
        self._stats.by_framework[framework] = (
            self._stats.by_framework.get(framework, 0) + 1
        )
