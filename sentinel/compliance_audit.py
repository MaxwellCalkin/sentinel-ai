"""Automated compliance auditing against safety frameworks.

Check LLM deployments against GDPR, CCPA, NIST AI RMF, EU AI Act,
and custom compliance frameworks. Generate audit reports with
pass/fail per requirement.

Usage:
    from sentinel.compliance_audit import ComplianceAuditor

    auditor = ComplianceAuditor()
    auditor.add_check("data_retention", lambda ctx: ctx.get("retention_days", 0) <= 90)
    report = auditor.audit({"retention_days": 30, "encryption": True})
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class ComplianceCheck:
    """A single compliance check."""
    name: str
    description: str
    framework: str
    check_fn: Callable[[dict[str, Any]], bool]
    severity: str = "high"
    remediation: str = ""


@dataclass
class CheckResult:
    """Result of a single compliance check."""
    name: str
    framework: str
    status: CheckStatus
    description: str
    severity: str
    remediation: str
    error: str | None = None


@dataclass
class AuditReport:
    """Full compliance audit report."""
    checks: list[CheckResult]
    passed: int
    failed: int
    warnings: int
    skipped: int
    errors: int
    score: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)

    @property
    def compliant(self) -> bool:
        """Compliant if no high-severity failures."""
        return all(
            c.status != CheckStatus.FAIL or c.severity != "high"
            for c in self.checks
        )

    @property
    def frameworks(self) -> list[str]:
        return list(set(c.framework for c in self.checks))


# Built-in framework checks
_BUILTIN_FRAMEWORKS: dict[str, list[dict[str, Any]]] = {
    "gdpr": [
        {
            "name": "data_minimization",
            "description": "Only collect data necessary for stated purpose",
            "check": lambda ctx: ctx.get("data_minimization", False),
            "remediation": "Review data collection to ensure only necessary data is collected",
        },
        {
            "name": "consent_mechanism",
            "description": "Explicit consent obtained before processing",
            "check": lambda ctx: ctx.get("consent_tracking", False),
            "remediation": "Implement consent tracking before data processing",
        },
        {
            "name": "right_to_erasure",
            "description": "Users can request deletion of their data",
            "check": lambda ctx: ctx.get("deletion_support", False),
            "remediation": "Implement user data deletion capability",
        },
        {
            "name": "data_encryption",
            "description": "Personal data encrypted at rest and in transit",
            "check": lambda ctx: ctx.get("encryption", False),
            "remediation": "Enable encryption for all personal data storage and transmission",
        },
    ],
    "nist_ai_rmf": [
        {
            "name": "risk_assessment",
            "description": "AI system risks are identified and assessed",
            "check": lambda ctx: ctx.get("risk_assessment", False),
            "remediation": "Conduct formal AI risk assessment",
        },
        {
            "name": "bias_testing",
            "description": "System tested for demographic bias",
            "check": lambda ctx: ctx.get("bias_testing", False),
            "remediation": "Run bias detection tests across demographic groups",
        },
        {
            "name": "explainability",
            "description": "AI decisions are explainable to stakeholders",
            "check": lambda ctx: ctx.get("explainability", False),
            "remediation": "Add explanation capabilities to AI outputs",
        },
        {
            "name": "monitoring",
            "description": "Continuous monitoring of AI system performance",
            "check": lambda ctx: ctx.get("monitoring", False),
            "remediation": "Deploy performance monitoring and alerting",
        },
    ],
    "eu_ai_act": [
        {
            "name": "risk_classification",
            "description": "System classified by risk level (minimal/limited/high/unacceptable)",
            "check": lambda ctx: ctx.get("risk_classification") is not None,
            "remediation": "Classify your AI system according to EU AI Act risk categories",
        },
        {
            "name": "human_oversight",
            "description": "Human oversight mechanisms in place for high-risk systems",
            "check": lambda ctx: ctx.get("human_oversight", False),
            "remediation": "Implement human-in-the-loop for high-risk decisions",
        },
        {
            "name": "transparency",
            "description": "Users informed they are interacting with AI",
            "check": lambda ctx: ctx.get("ai_disclosure", False),
            "remediation": "Add clear AI disclosure to user interfaces",
        },
    ],
}


class ComplianceAuditor:
    """Automated compliance auditing.

    Check deployments against regulatory frameworks with
    built-in and custom checks.
    """

    def __init__(self) -> None:
        self._checks: list[ComplianceCheck] = []

    def add_check(
        self,
        name: str,
        check_fn: Callable[[dict[str, Any]], bool],
        description: str = "",
        framework: str = "custom",
        severity: str = "high",
        remediation: str = "",
    ) -> None:
        """Add a compliance check.

        Args:
            name: Check identifier.
            check_fn: Function(context) -> True if check passes.
            description: Human-readable description.
            framework: Framework this check belongs to.
            severity: high, medium, or low.
            remediation: How to fix if check fails.
        """
        self._checks.append(ComplianceCheck(
            name=name,
            description=description or name,
            framework=framework,
            check_fn=check_fn,
            severity=severity,
            remediation=remediation,
        ))

    def load_framework(self, framework: str) -> int:
        """Load built-in framework checks.

        Args:
            framework: Framework name (gdpr, nist_ai_rmf, eu_ai_act).

        Returns:
            Number of checks loaded.
        """
        checks = _BUILTIN_FRAMEWORKS.get(framework, [])
        for c in checks:
            self.add_check(
                name=c["name"],
                check_fn=c["check"],
                description=c["description"],
                framework=framework,
                remediation=c.get("remediation", ""),
            )
        return len(checks)

    def load_all_frameworks(self) -> int:
        """Load all built-in frameworks. Returns total checks loaded."""
        total = 0
        for fw in _BUILTIN_FRAMEWORKS:
            total += self.load_framework(fw)
        return total

    def audit(
        self,
        context: dict[str, Any],
        frameworks: list[str] | None = None,
    ) -> AuditReport:
        """Run compliance audit.

        Args:
            context: Dictionary of deployment properties to check.
            frameworks: Filter to specific frameworks. None = all.

        Returns:
            AuditReport with per-check results and compliance score.
        """
        results: list[CheckResult] = []
        passed = failed = warnings = skipped = errors = 0

        for check in self._checks:
            if frameworks and check.framework not in frameworks:
                skipped += 1
                results.append(CheckResult(
                    name=check.name,
                    framework=check.framework,
                    status=CheckStatus.SKIP,
                    description=check.description,
                    severity=check.severity,
                    remediation=check.remediation,
                ))
                continue

            try:
                result = check.check_fn(context)
                if result:
                    status = CheckStatus.PASS
                    passed += 1
                else:
                    status = CheckStatus.FAIL
                    failed += 1
            except Exception as e:
                status = CheckStatus.ERROR
                errors += 1
                results.append(CheckResult(
                    name=check.name,
                    framework=check.framework,
                    status=status,
                    description=check.description,
                    severity=check.severity,
                    remediation=check.remediation,
                    error=str(e),
                ))
                continue

            results.append(CheckResult(
                name=check.name,
                framework=check.framework,
                status=status,
                description=check.description,
                severity=check.severity,
                remediation=check.remediation,
            ))

        total_evaluated = passed + failed
        score = passed / total_evaluated if total_evaluated > 0 else 1.0

        return AuditReport(
            checks=results,
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            errors=errors,
            score=round(score, 4),
        )

    @property
    def check_count(self) -> int:
        return len(self._checks)

    @property
    def available_frameworks(self) -> list[str]:
        return list(_BUILTIN_FRAMEWORKS.keys())
