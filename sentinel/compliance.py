"""Regulatory compliance mapping for AI safety scan results.

Maps Sentinel scan findings to regulatory frameworks including:
- EU AI Act (2024) — risk classification and transparency obligations
- NIST AI RMF 1.0 — governance, mapping, measurement, management functions
- ISO/IEC 42001 — AI management system requirements

Usage:
    from sentinel.compliance import ComplianceMapper, Framework

    mapper = ComplianceMapper()
    report = mapper.evaluate(scan_results, frameworks=[Framework.EU_AI_ACT, Framework.NIST_RMF])
    print(report.to_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

from sentinel.core import RiskLevel, ScanResult


class Framework(Enum):
    """Supported regulatory frameworks."""

    EU_AI_ACT = "eu_ai_act"
    NIST_RMF = "nist_ai_rmf"
    ISO_42001 = "iso_42001"


class ComplianceStatus(Enum):
    """Compliance assessment status."""

    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


@dataclass
class ControlAssessment:
    """Assessment of a single regulatory control."""

    control_id: str
    control_name: str
    framework: Framework
    status: ComplianceStatus
    findings_count: int
    description: str
    remediation: str | None = None


@dataclass
class FrameworkReport:
    """Compliance report for a single framework."""

    framework: Framework
    status: ComplianceStatus
    controls_assessed: int
    controls_compliant: int
    controls_partial: int
    controls_non_compliant: int
    controls: list[ControlAssessment]
    risk_classification: str | None = None


@dataclass
class ComplianceReport:
    """Full compliance assessment across frameworks."""

    generated_at: str
    total_scans: int
    overall_risk: RiskLevel
    frameworks: list[FrameworkReport]

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total_scans": self.total_scans,
            "overall_risk": self.overall_risk.value,
            "frameworks": [
                {
                    "framework": fr.framework.value,
                    "status": fr.status.value,
                    "controls_assessed": fr.controls_assessed,
                    "controls_compliant": fr.controls_compliant,
                    "controls_partial": fr.controls_partial,
                    "controls_non_compliant": fr.controls_non_compliant,
                    "risk_classification": fr.risk_classification,
                    "controls": [
                        {
                            "control_id": c.control_id,
                            "control_name": c.control_name,
                            "status": c.status.value,
                            "findings_count": c.findings_count,
                            "description": c.description,
                            "remediation": c.remediation,
                        }
                        for c in fr.controls
                    ],
                }
                for fr in self.frameworks
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# AI Compliance Assessment Report",
            f"*Generated: {self.generated_at}*",
            "",
            "## Summary",
            "",
            f"- **Total Scans Analyzed**: {self.total_scans}",
            f"- **Overall Risk Level**: {self.overall_risk.value.upper()}",
            f"- **Frameworks Assessed**: {len(self.frameworks)}",
            "",
        ]

        for fr in self.frameworks:
            label = _FRAMEWORK_LABELS[fr.framework]
            lines.extend([
                f"## {label}",
                "",
                f"**Overall Status**: {_STATUS_LABELS[fr.status]}",
                "",
            ])
            if fr.risk_classification:
                lines.append(f"**Risk Classification**: {fr.risk_classification}")
                lines.append("")

            lines.extend([
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Controls Assessed | {fr.controls_assessed} |",
                f"| Compliant | {fr.controls_compliant} |",
                f"| Partial | {fr.controls_partial} |",
                f"| Non-Compliant | {fr.controls_non_compliant} |",
                "",
                "### Control Details",
                "",
                "| ID | Control | Status | Findings |",
                "|----|---------|--------|----------|",
            ])
            for c in fr.controls:
                status_icon = _STATUS_ICONS[c.status]
                lines.append(
                    f"| {c.control_id} | {c.control_name} | "
                    f"{status_icon} {c.status.value} | {c.findings_count} |"
                )

            # Remediations for non-compliant controls
            non_compliant = [
                c for c in fr.controls
                if c.status in (ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIAL)
                and c.remediation
            ]
            if non_compliant:
                lines.extend(["", "### Recommended Actions", ""])
                for c in non_compliant:
                    lines.append(f"- **{c.control_id}** ({c.control_name}): {c.remediation}")

            lines.append("")

        lines.extend([
            "---",
            "*Report generated by Sentinel AI — automated compliance assessment "
            "for AI safety.*",
        ])
        return "\n".join(lines)


_FRAMEWORK_LABELS = {
    Framework.EU_AI_ACT: "EU AI Act (2024)",
    Framework.NIST_RMF: "NIST AI Risk Management Framework 1.0",
    Framework.ISO_42001: "ISO/IEC 42001 — AI Management System",
}

_STATUS_LABELS = {
    ComplianceStatus.COMPLIANT: "COMPLIANT",
    ComplianceStatus.PARTIAL: "PARTIALLY COMPLIANT",
    ComplianceStatus.NON_COMPLIANT: "NON-COMPLIANT",
    ComplianceStatus.NOT_ASSESSED: "NOT ASSESSED",
}

_STATUS_ICONS = {
    ComplianceStatus.COMPLIANT: "[PASS]",
    ComplianceStatus.PARTIAL: "[WARN]",
    ComplianceStatus.NON_COMPLIANT: "[FAIL]",
    ComplianceStatus.NOT_ASSESSED: "[--]",
}


# ---------- EU AI Act Controls ----------

_EU_AI_ACT_CONTROLS = [
    {
        "id": "EU-AIA-6",
        "name": "Risk Classification",
        "desc": "AI systems must be classified by risk level (unacceptable, high, limited, minimal).",
        "categories": [],  # assessed by overall risk
        "assess_fn": "_assess_eu_risk_classification",
    },
    {
        "id": "EU-AIA-9",
        "name": "Risk Management System",
        "desc": "High-risk AI must have a risk management system identifying and mitigating risks.",
        "categories": ["prompt_injection", "harmful_content", "tool_use"],
        "threshold": RiskLevel.HIGH,
    },
    {
        "id": "EU-AIA-10",
        "name": "Data Governance",
        "desc": "Training and input data must be subject to appropriate governance practices.",
        "categories": ["pii"],
        "threshold": RiskLevel.MEDIUM,
    },
    {
        "id": "EU-AIA-13",
        "name": "Transparency",
        "desc": "AI systems must provide transparent information about capabilities and limitations.",
        "categories": ["hallucination"],
        "threshold": RiskLevel.MEDIUM,
    },
    {
        "id": "EU-AIA-14",
        "name": "Human Oversight",
        "desc": "High-risk AI must enable effective human oversight of system operation.",
        "categories": ["tool_use", "prompt_injection"],
        "threshold": RiskLevel.HIGH,
    },
    {
        "id": "EU-AIA-15",
        "name": "Accuracy & Robustness",
        "desc": "AI systems must achieve appropriate levels of accuracy and robustness.",
        "categories": ["hallucination", "prompt_injection"],
        "threshold": RiskLevel.MEDIUM,
    },
    {
        "id": "EU-AIA-52",
        "name": "Transparency for Users",
        "desc": "Providers must ensure users are informed they are interacting with an AI system.",
        "categories": [],
        "threshold": RiskLevel.NONE,
    },
]

# ---------- NIST AI RMF Controls ----------

_NIST_RMF_CONTROLS = [
    {
        "id": "GOVERN-1",
        "name": "Risk Management Policies",
        "desc": "Policies for AI risk management are established and organizational roles defined.",
        "categories": ["prompt_injection", "harmful_content"],
        "threshold": RiskLevel.HIGH,
    },
    {
        "id": "MAP-1",
        "name": "Context Identification",
        "desc": "Intended context of use and potential negative impacts are identified.",
        "categories": ["harmful_content", "toxicity"],
        "threshold": RiskLevel.MEDIUM,
    },
    {
        "id": "MAP-3",
        "name": "Benefits and Costs",
        "desc": "AI risks and benefits are assessed with domain expertise and affected communities.",
        "categories": ["harmful_content"],
        "threshold": RiskLevel.HIGH,
    },
    {
        "id": "MEASURE-1",
        "name": "Risk Measurement",
        "desc": "Appropriate methods and metrics are identified to measure AI risks.",
        "categories": ["prompt_injection", "hallucination", "toxicity", "pii"],
        "threshold": RiskLevel.LOW,
    },
    {
        "id": "MEASURE-2",
        "name": "Trustworthiness Evaluation",
        "desc": "AI systems are evaluated for trustworthiness characteristics.",
        "categories": ["hallucination", "prompt_injection"],
        "threshold": RiskLevel.MEDIUM,
    },
    {
        "id": "MANAGE-1",
        "name": "Risk Prioritization",
        "desc": "AI risks are prioritized and resources allocated based on impact.",
        "categories": ["prompt_injection", "harmful_content", "tool_use"],
        "threshold": RiskLevel.HIGH,
    },
    {
        "id": "MANAGE-2",
        "name": "Risk Response",
        "desc": "Strategies to respond to identified AI risks are planned and implemented.",
        "categories": ["prompt_injection", "pii", "harmful_content"],
        "threshold": RiskLevel.MEDIUM,
    },
    {
        "id": "MANAGE-4",
        "name": "Incident Management",
        "desc": "Mechanisms are in place for AI risk and incident management.",
        "categories": ["tool_use", "prompt_injection"],
        "threshold": RiskLevel.HIGH,
    },
]

# ---------- ISO 42001 Controls ----------

_ISO_42001_CONTROLS = [
    {
        "id": "ISO-6.1",
        "name": "Risk Assessment",
        "desc": "Organization shall determine AI risks and opportunities requiring action.",
        "categories": ["prompt_injection", "harmful_content", "tool_use"],
        "threshold": RiskLevel.MEDIUM,
    },
    {
        "id": "ISO-8.4",
        "name": "AI System Impact Assessment",
        "desc": "Impact assessments shall be conducted for AI systems.",
        "categories": ["harmful_content", "toxicity", "pii"],
        "threshold": RiskLevel.MEDIUM,
    },
    {
        "id": "ISO-A.4",
        "name": "Data Quality for AI",
        "desc": "Data used by AI systems meets quality and governance requirements.",
        "categories": ["pii", "hallucination"],
        "threshold": RiskLevel.MEDIUM,
    },
    {
        "id": "ISO-A.6",
        "name": "AI System Security",
        "desc": "Security controls for AI systems against adversarial attacks and data poisoning.",
        "categories": ["prompt_injection", "tool_use"],
        "threshold": RiskLevel.HIGH,
    },
    {
        "id": "ISO-A.8",
        "name": "Transparency and Explainability",
        "desc": "AI systems provide appropriate transparency about decisions and limitations.",
        "categories": ["hallucination"],
        "threshold": RiskLevel.MEDIUM,
    },
    {
        "id": "ISO-A.10",
        "name": "Third-Party AI Risk",
        "desc": "Risks from third-party AI components are assessed and managed.",
        "categories": ["tool_use", "prompt_injection"],
        "threshold": RiskLevel.HIGH,
    },
]

_REMEDIATIONS = {
    "prompt_injection": (
        "Deploy prompt injection scanning on all user inputs and enable "
        "prompt hardening (XML tagging, sandwich defense, role lock)."
    ),
    "pii": (
        "Enable automatic PII redaction before data reaches AI models. "
        "Audit data pipelines for unredacted personal information."
    ),
    "harmful_content": (
        "Configure harmful content blocking at appropriate risk thresholds. "
        "Maintain incident response procedures for safety-critical failures."
    ),
    "hallucination": (
        "Implement citation verification and factual grounding mechanisms. "
        "Add hallucination scanning to output pipelines."
    ),
    "toxicity": (
        "Enable toxicity scanning on model outputs. Configure content "
        "filtering thresholds appropriate for your user base."
    ),
    "tool_use": (
        "Restrict tool-use permissions to allowlisted operations. Scan "
        "all tool calls for dangerous commands before execution."
    ),
}

_FRAMEWORK_CONTROLS = {
    Framework.EU_AI_ACT: _EU_AI_ACT_CONTROLS,
    Framework.NIST_RMF: _NIST_RMF_CONTROLS,
    Framework.ISO_42001: _ISO_42001_CONTROLS,
}


class ComplianceMapper:
    """Maps AI safety scan results to regulatory compliance frameworks.

    Evaluates scan results against controls from EU AI Act, NIST AI RMF,
    and ISO/IEC 42001, producing compliance assessment reports.
    """

    def evaluate(
        self,
        scan_results: Sequence[ScanResult],
        frameworks: Sequence[Framework] | None = None,
    ) -> ComplianceReport:
        """Evaluate scan results against compliance frameworks.

        Args:
            scan_results: Results from SentinelGuard.scan().
            frameworks: Which frameworks to assess (default: all).

        Returns:
            ComplianceReport with per-framework control assessments.
        """
        if frameworks is None:
            frameworks = list(Framework)

        results = list(scan_results)

        # Aggregate category findings and max risk per category
        category_counts: dict[str, int] = {}
        category_max_risk: dict[str, RiskLevel] = {}
        for r in results:
            for f in r.findings:
                category_counts[f.category] = category_counts.get(f.category, 0) + 1
                current = category_max_risk.get(f.category, RiskLevel.NONE)
                if f.risk > current:
                    category_max_risk[f.category] = f.risk

        overall_risk = RiskLevel.NONE
        for r in results:
            if r.risk > overall_risk:
                overall_risk = r.risk

        framework_reports = []
        for fw in frameworks:
            report = self._assess_framework(
                fw, category_counts, category_max_risk, overall_risk
            )
            framework_reports.append(report)

        return ComplianceReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            total_scans=len(results),
            overall_risk=overall_risk,
            frameworks=framework_reports,
        )

    def _assess_framework(
        self,
        framework: Framework,
        category_counts: dict[str, int],
        category_max_risk: dict[str, RiskLevel],
        overall_risk: RiskLevel,
    ) -> FrameworkReport:
        controls_defs = _FRAMEWORK_CONTROLS[framework]
        assessments: list[ControlAssessment] = []

        for ctrl in controls_defs:
            assessment = self._assess_control(
                ctrl, framework, category_counts, category_max_risk, overall_risk
            )
            assessments.append(assessment)

        compliant = sum(1 for a in assessments if a.status == ComplianceStatus.COMPLIANT)
        partial = sum(1 for a in assessments if a.status == ComplianceStatus.PARTIAL)
        non_compliant = sum(
            1 for a in assessments if a.status == ComplianceStatus.NON_COMPLIANT
        )

        if non_compliant > 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif partial > 0:
            overall_status = ComplianceStatus.PARTIAL
        else:
            overall_status = ComplianceStatus.COMPLIANT

        risk_class = None
        if framework == Framework.EU_AI_ACT:
            risk_class = _classify_eu_risk(overall_risk)

        return FrameworkReport(
            framework=framework,
            status=overall_status,
            controls_assessed=len(assessments),
            controls_compliant=compliant,
            controls_partial=partial,
            controls_non_compliant=non_compliant,
            controls=assessments,
            risk_classification=risk_class,
        )

    def _assess_control(
        self,
        ctrl: dict,
        framework: Framework,
        category_counts: dict[str, int],
        category_max_risk: dict[str, RiskLevel],
        overall_risk: RiskLevel,
    ) -> ControlAssessment:
        categories = ctrl.get("categories", [])
        threshold = ctrl.get("threshold", RiskLevel.MEDIUM)

        if not categories:
            # Controls without specific categories are assessed by overall risk
            findings = sum(category_counts.values())
            if overall_risk >= RiskLevel.HIGH:
                status = ComplianceStatus.NON_COMPLIANT
            elif overall_risk >= RiskLevel.MEDIUM:
                status = ComplianceStatus.PARTIAL
            else:
                status = ComplianceStatus.COMPLIANT
            return ControlAssessment(
                control_id=ctrl["id"],
                control_name=ctrl["name"],
                framework=framework,
                status=status,
                findings_count=findings,
                description=ctrl["desc"],
            )

        # Count relevant findings
        relevant_count = sum(category_counts.get(c, 0) for c in categories)
        max_risk = RiskLevel.NONE
        for c in categories:
            r = category_max_risk.get(c, RiskLevel.NONE)
            if r > max_risk:
                max_risk = r

        # Determine status
        if relevant_count == 0:
            status = ComplianceStatus.COMPLIANT
        elif max_risk >= threshold:
            status = ComplianceStatus.NON_COMPLIANT
        else:
            status = ComplianceStatus.PARTIAL

        # Build remediation from failing categories
        remediation = None
        if status != ComplianceStatus.COMPLIANT:
            failing_cats = [
                c for c in categories
                if category_counts.get(c, 0) > 0
            ]
            parts = [_REMEDIATIONS[c] for c in failing_cats if c in _REMEDIATIONS]
            if parts:
                remediation = " ".join(parts)

        return ControlAssessment(
            control_id=ctrl["id"],
            control_name=ctrl["name"],
            framework=framework,
            status=status,
            findings_count=relevant_count,
            description=ctrl["desc"],
            remediation=remediation,
        )


def _classify_eu_risk(risk: RiskLevel) -> str:
    """Map overall risk level to EU AI Act risk classification."""
    if risk >= RiskLevel.CRITICAL:
        return "Unacceptable Risk (Article 5)"
    elif risk >= RiskLevel.HIGH:
        return "High Risk (Article 6)"
    elif risk >= RiskLevel.MEDIUM:
        return "Limited Risk (Article 52)"
    else:
        return "Minimal Risk"
