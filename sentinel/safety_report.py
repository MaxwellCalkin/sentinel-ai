"""Formatted safety audit report generation from scan results.

Produces structured reports for different audiences (technical, executive,
compliance) with auto-generated summaries and risk ratings.

Usage:
    from sentinel.safety_report import SafetyReport, Finding

    report = SafetyReport()
    rid = report.create("Q1 Safety Audit", audience="executive")
    report.add_section(rid, "Injection Tests", summary="Prompt injection scan results")
    report.add_finding(rid, "Injection Tests", Finding(
        id="INJ-001",
        title="SQL injection detected",
        severity="critical",
        description="User input contained SQL injection payload.",
        recommendation="Enable input sanitization.",
    ))
    result = report.generate(rid)
    print(result.risk_rating)       # "critical"
    print(result.executive_summary) # auto-generated summary
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


VALID_SEVERITIES = ("critical", "high", "medium", "low", "info")
VALID_AUDIENCES = ("technical", "executive", "compliance")


@dataclass
class Finding:
    """A single finding within a report section."""
    id: str
    title: str
    severity: str
    description: str
    recommendation: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{self.severity}'. "
                f"Must be one of: {', '.join(VALID_SEVERITIES)}"
            )


@dataclass
class ReportSection:
    """A logical section grouping related findings."""
    title: str
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""


@dataclass
class ReportMetadata:
    """Metadata describing a generated report."""
    title: str
    author: str = ""
    generated_at: float = field(default_factory=time.time)
    version: str = "1.0"
    audience: str = "technical"


@dataclass
class GeneratedReport:
    """The final generated report with computed fields."""
    metadata: ReportMetadata
    sections: list[ReportSection]
    executive_summary: str
    total_findings: int
    critical_count: int
    high_count: int
    risk_rating: str


@dataclass
class ReportStats:
    """Cumulative statistics across all generated reports."""
    total_generated: int = 0
    by_audience: dict[str, int] = field(default_factory=dict)
    avg_findings: float = 0.0


def _compute_severity_counts(sections: list[ReportSection]) -> dict[str, int]:
    counts: dict[str, int] = {s: 0 for s in VALID_SEVERITIES}
    for section in sections:
        for finding in section.findings:
            counts[finding.severity] += 1
    return counts


def _compute_risk_rating(counts: dict[str, int]) -> str:
    if counts["critical"] > 0:
        return "critical"
    if counts["high"] > 0:
        return "high"
    if counts["medium"] > 0:
        return "medium"
    if counts["low"] > 0:
        return "low"
    return "minimal"


def _build_executive_summary(counts: dict[str, int], total: int) -> str:
    if total == 0:
        return "No findings were identified during this assessment."

    parts = []
    parts.append(f"Assessment identified {total} finding(s).")

    severity_descriptions = []
    if counts["critical"] > 0:
        severity_descriptions.append(f"{counts['critical']} critical")
    if counts["high"] > 0:
        severity_descriptions.append(f"{counts['high']} high")
    if counts["medium"] > 0:
        severity_descriptions.append(f"{counts['medium']} medium")
    if counts["low"] > 0:
        severity_descriptions.append(f"{counts['low']} low")
    if counts["info"] > 0:
        severity_descriptions.append(f"{counts['info']} informational")

    if severity_descriptions:
        parts.append(f"Breakdown: {', '.join(severity_descriptions)}.")

    rating = _compute_risk_rating(counts)
    parts.append(f"Overall risk rating: {rating}.")

    return " ".join(parts)


def _format_text_report(report: GeneratedReport) -> str:
    lines: list[str] = []
    separator = "=" * 60

    lines.append(separator)
    lines.append(f"  {report.metadata.title}")
    lines.append(separator)
    lines.append("")
    lines.append(f"Audience: {report.metadata.audience}")
    if report.metadata.author:
        lines.append(f"Author: {report.metadata.author}")
    lines.append(f"Version: {report.metadata.version}")
    lines.append(f"Risk Rating: {report.risk_rating.upper()}")
    lines.append("")

    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    lines.append(report.executive_summary)
    lines.append("")

    for section in report.sections:
        lines.append(f"SECTION: {section.title}")
        lines.append("-" * 40)
        if section.summary:
            lines.append(section.summary)
            lines.append("")

        if not section.findings:
            lines.append("  No findings in this section.")
        for finding in section.findings:
            lines.append(f"  [{finding.severity.upper()}] {finding.id}: {finding.title}")
            lines.append(f"    {finding.description}")
            if finding.recommendation:
                lines.append(f"    Recommendation: {finding.recommendation}")
            if finding.evidence:
                lines.append(f"    Evidence: {finding.evidence}")
        lines.append("")

    lines.append(separator)
    lines.append(
        f"Total findings: {report.total_findings} | "
        f"Critical: {report.critical_count} | "
        f"High: {report.high_count}"
    )
    lines.append(separator)

    return "\n".join(lines)


class SafetyReport:
    """Generate formatted safety audit reports.

    Manages report lifecycle: creation, section/finding assembly,
    generation, and export in multiple formats.
    """

    def __init__(self) -> None:
        self._sections: dict[str, list[ReportSection]] = {}
        self._metadata: dict[str, ReportMetadata] = {}
        self._generated: dict[str, GeneratedReport] = {}
        self._generation_count: int = 0
        self._total_findings_generated: int = 0
        self._audience_counts: dict[str, int] = {}

    def create(
        self,
        title: str,
        audience: str = "technical",
        author: str = "",
    ) -> str:
        """Start a new report and return its unique ID."""
        report_id = uuid.uuid4().hex[:12]
        self._metadata[report_id] = ReportMetadata(
            title=title,
            audience=audience,
            author=author,
        )
        self._sections[report_id] = []
        return report_id

    def add_section(
        self,
        report_id: str,
        title: str,
        summary: str = "",
    ) -> None:
        """Add a named section to a report."""
        self._require_report_exists(report_id)
        self._sections[report_id].append(
            ReportSection(title=title, summary=summary)
        )

    def add_finding(
        self,
        report_id: str,
        section_title: str,
        finding: Finding,
    ) -> None:
        """Add a finding to an existing section.

        Raises:
            KeyError: If report_id or section_title not found.
        """
        self._require_report_exists(report_id)
        section = self._find_section(report_id, section_title)
        section.findings.append(finding)

    def generate(self, report_id: str) -> GeneratedReport:
        """Generate the final report with computed summaries and ratings."""
        self._require_report_exists(report_id)
        metadata = self._metadata[report_id]
        sections = self._sections[report_id]

        counts = _compute_severity_counts(sections)
        total = sum(counts.values())
        executive_summary = _build_executive_summary(counts, total)
        risk_rating = _compute_risk_rating(counts)

        generated = GeneratedReport(
            metadata=metadata,
            sections=sections,
            executive_summary=executive_summary,
            total_findings=total,
            critical_count=counts["critical"],
            high_count=counts["high"],
            risk_rating=risk_rating,
        )
        self._generated[report_id] = generated
        self._generation_count += 1
        self._total_findings_generated += total
        audience = metadata.audience
        self._audience_counts[audience] = self._audience_counts.get(audience, 0) + 1

        return generated

    def export_text(self, report_id: str) -> str:
        """Export a report as formatted plain text."""
        report = self._require_generated(report_id)
        return _format_text_report(report)

    def export_dict(self, report_id: str) -> dict:
        """Export a report as a plain dictionary."""
        report = self._require_generated(report_id)
        return {
            "metadata": {
                "title": report.metadata.title,
                "author": report.metadata.author,
                "generated_at": report.metadata.generated_at,
                "version": report.metadata.version,
                "audience": report.metadata.audience,
            },
            "executive_summary": report.executive_summary,
            "total_findings": report.total_findings,
            "critical_count": report.critical_count,
            "high_count": report.high_count,
            "risk_rating": report.risk_rating,
            "sections": [
                {
                    "title": section.title,
                    "summary": section.summary,
                    "findings": [
                        {
                            "id": f.id,
                            "title": f.title,
                            "severity": f.severity,
                            "description": f.description,
                            "recommendation": f.recommendation,
                            "evidence": f.evidence,
                        }
                        for f in section.findings
                    ],
                }
                for section in report.sections
            ],
        }

    def list_reports(self) -> list[str]:
        """Return all report IDs."""
        return list(self._metadata.keys())

    def stats(self) -> ReportStats:
        """Return cumulative generation statistics."""
        avg = 0.0
        if self._generation_count > 0:
            avg = self._total_findings_generated / self._generation_count
        return ReportStats(
            total_generated=self._generation_count,
            by_audience=dict(self._audience_counts),
            avg_findings=avg,
        )

    def _require_report_exists(self, report_id: str) -> None:
        if report_id not in self._metadata:
            raise KeyError(f"Report '{report_id}' not found.")

    def _require_generated(self, report_id: str) -> GeneratedReport:
        if report_id not in self._generated:
            raise KeyError(
                f"Report '{report_id}' has not been generated yet. "
                "Call generate() first."
            )
        return self._generated[report_id]

    def _find_section(self, report_id: str, section_title: str) -> ReportSection:
        for section in self._sections[report_id]:
            if section.title == section_title:
                return section
        raise KeyError(
            f"Section '{section_title}' not found in report '{report_id}'."
        )
