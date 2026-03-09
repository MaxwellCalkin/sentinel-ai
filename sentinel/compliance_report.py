"""Structured compliance report generation for AI deployments.

Generates compliance reports covering frameworks like EU AI Act,
NIST AI RMF, ISO 42001, and SOC 2. Tracks control assessments,
risk levels, and recommendations with compliance rate calculation.

Usage:
    from sentinel.compliance_report import ComplianceReport

    report = ComplianceReport("Acme Corp", "ChatBot v2")
    report.add_control("EU AI Act", "AIA-1", "Risk Classification", "compliant")
    report.set_risk_level("medium")
    report.add_recommendation("high", "Implement human oversight")
    output = report.generate()
    print(output.overall_compliance_rate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


_VALID_STATUSES = frozenset({"compliant", "partial", "non_compliant", "not_applicable"})
_VALID_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})
_PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class Control:
    """A single control assessment within a compliance framework."""

    framework: str
    control_id: str
    title: str
    status: str
    evidence: str = ""
    notes: str = ""


@dataclass
class Recommendation:
    """A prioritized recommendation for improving compliance."""

    priority: str
    description: str
    framework: str = ""


@dataclass
class FrameworkSummary:
    """Aggregated compliance statistics for a single framework."""

    framework: str
    total_controls: int
    compliant: int
    partial: int
    non_compliant: int
    not_applicable: int
    compliance_rate: float


@dataclass
class ReportOutput:
    """Complete generated compliance report."""

    organization: str
    system_name: str
    risk_level: str
    frameworks: list[FrameworkSummary]
    controls: list[Control]
    recommendations: list[Recommendation]
    overall_compliance_rate: float
    generated_at: str


def _validate_status(status: str) -> None:
    """Raise ValueError if status is not a recognized control status."""
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. "
            f"Must be one of: {', '.join(sorted(_VALID_STATUSES))}"
        )


def _validate_risk_level(level: str) -> None:
    """Raise ValueError if level is not a recognized risk level."""
    if level not in _VALID_RISK_LEVELS:
        raise ValueError(
            f"Invalid risk level '{level}'. "
            f"Must be one of: {', '.join(sorted(_VALID_RISK_LEVELS))}"
        )


def _calculate_compliance_rate(controls: list[Control]) -> float:
    """Compute compliance rate with partial credit.

    Formula: (compliant + 0.5 * partial) / (total - not_applicable)
    Returns 1.0 when all controls are not_applicable.
    """
    applicable_count = sum(
        1 for c in controls if c.status != "not_applicable"
    )
    if applicable_count == 0:
        return 1.0
    compliant_count = sum(1 for c in controls if c.status == "compliant")
    partial_count = sum(1 for c in controls if c.status == "partial")
    return (compliant_count + 0.5 * partial_count) / applicable_count


def _sort_recommendations(recommendations: list[Recommendation]) -> list[Recommendation]:
    """Sort recommendations by priority: critical > high > medium > low."""
    return sorted(
        recommendations,
        key=lambda r: _PRIORITY_ORDER.get(r.priority, 999),
    )


class ComplianceReport:
    """Generate structured compliance reports for AI deployments.

    Tracks control assessments across multiple frameworks, computes
    compliance rates with partial credit, and produces a structured
    report output.
    """

    def __init__(self, organization: str, system_name: str) -> None:
        self._organization = organization
        self._system_name = system_name
        self._risk_level = "medium"
        self._controls: list[Control] = []
        self._recommendations: list[Recommendation] = []

    def add_control(
        self,
        framework: str,
        control_id: str,
        title: str,
        status: str,
        evidence: str = "",
        notes: str = "",
    ) -> None:
        """Add a control assessment to the report.

        Args:
            framework: Framework name (e.g. "EU AI Act", "NIST AI RMF").
            control_id: Control identifier within the framework.
            title: Human-readable control title.
            status: One of "compliant", "partial", "non_compliant", "not_applicable".
            evidence: Supporting evidence for the assessment.
            notes: Additional notes.

        Raises:
            ValueError: If status is not a recognized value.
        """
        _validate_status(status)
        self._controls.append(Control(
            framework=framework,
            control_id=control_id,
            title=title,
            status=status,
            evidence=evidence,
            notes=notes,
        ))

    def set_risk_level(self, level: str) -> None:
        """Set the overall risk level for the deployment.

        Args:
            level: One of "low", "medium", "high", "critical".

        Raises:
            ValueError: If level is not a recognized value.
        """
        _validate_risk_level(level)
        self._risk_level = level

    def add_recommendation(
        self,
        priority: str,
        description: str,
        framework: str = "",
    ) -> None:
        """Add a recommendation to the report.

        Args:
            priority: One of "low", "medium", "high", "critical".
            description: What should be done.
            framework: Optional framework this recommendation relates to.
        """
        self._recommendations.append(Recommendation(
            priority=priority,
            description=description,
            framework=framework,
        ))

    def generate(self) -> ReportOutput:
        """Generate the full compliance report.

        Returns:
            ReportOutput with framework summaries, controls,
            sorted recommendations, and overall compliance rate.
        """
        framework_names = self.list_frameworks()
        framework_summaries = [
            self.framework_summary(name) for name in framework_names
        ]
        overall_rate = _calculate_compliance_rate(self._controls)
        sorted_recommendations = _sort_recommendations(self._recommendations)

        return ReportOutput(
            organization=self._organization,
            system_name=self._system_name,
            risk_level=self._risk_level,
            frameworks=framework_summaries,
            controls=list(self._controls),
            recommendations=sorted_recommendations,
            overall_compliance_rate=round(overall_rate, 4),
            generated_at=datetime.now().isoformat(),
        )

    def framework_summary(self, framework: str) -> FrameworkSummary:
        """Get compliance statistics for a specific framework.

        Args:
            framework: The framework name to summarize.

        Returns:
            FrameworkSummary with counts and compliance rate.
        """
        framework_controls = [
            c for c in self._controls if c.framework == framework
        ]
        total = len(framework_controls)
        compliant = sum(1 for c in framework_controls if c.status == "compliant")
        partial = sum(1 for c in framework_controls if c.status == "partial")
        non_compliant = sum(1 for c in framework_controls if c.status == "non_compliant")
        not_applicable = sum(1 for c in framework_controls if c.status == "not_applicable")
        rate = _calculate_compliance_rate(framework_controls)

        return FrameworkSummary(
            framework=framework,
            total_controls=total,
            compliant=compliant,
            partial=partial,
            non_compliant=non_compliant,
            not_applicable=not_applicable,
            compliance_rate=round(rate, 4),
        )

    def export_dict(self) -> dict:
        """Export the report as a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        output = self.generate()
        return {
            "organization": output.organization,
            "system_name": output.system_name,
            "risk_level": output.risk_level,
            "overall_compliance_rate": output.overall_compliance_rate,
            "generated_at": output.generated_at,
            "frameworks": [
                {
                    "framework": fs.framework,
                    "total_controls": fs.total_controls,
                    "compliant": fs.compliant,
                    "partial": fs.partial,
                    "non_compliant": fs.non_compliant,
                    "not_applicable": fs.not_applicable,
                    "compliance_rate": fs.compliance_rate,
                }
                for fs in output.frameworks
            ],
            "controls": [
                {
                    "framework": c.framework,
                    "control_id": c.control_id,
                    "title": c.title,
                    "status": c.status,
                    "evidence": c.evidence,
                    "notes": c.notes,
                }
                for c in output.controls
            ],
            "recommendations": [
                {
                    "priority": r.priority,
                    "description": r.description,
                    "framework": r.framework,
                }
                for r in output.recommendations
            ],
        }

    def list_frameworks(self) -> list[str]:
        """List all frameworks that have controls added.

        Returns:
            Sorted list of unique framework names.
        """
        seen: dict[str, None] = {}
        for control in self._controls:
            seen[control.framework] = None
        return list(seen.keys())
