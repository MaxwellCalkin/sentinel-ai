"""Tests for SafetyReport — formatted safety audit report generation."""

import pytest

from sentinel.safety_report import (
    Finding,
    GeneratedReport,
    ReportMetadata,
    ReportSection,
    ReportStats,
    SafetyReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_finding(
    finding_id: str = "F-001",
    severity: str = "medium",
    title: str = "Test finding",
) -> Finding:
    return Finding(
        id=finding_id,
        title=title,
        severity=severity,
        description="A test finding description.",
        recommendation="Fix it.",
        evidence="Evidence here.",
    )


def _populated_report(reporter: SafetyReport) -> str:
    """Create a report with two sections and multiple findings."""
    rid = reporter.create("Audit Report", audience="technical", author="tester")
    reporter.add_section(rid, "Injection", summary="Injection scan results")
    reporter.add_section(rid, "PII", summary="PII scan results")
    reporter.add_finding(rid, "Injection", _make_finding("INJ-001", "critical", "SQL injection"))
    reporter.add_finding(rid, "Injection", _make_finding("INJ-002", "high", "XSS attempt"))
    reporter.add_finding(rid, "PII", _make_finding("PII-001", "medium", "Email exposed"))
    reporter.add_finding(rid, "PII", _make_finding("PII-002", "low", "Name in output"))
    return rid


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

class TestFinding:
    def test_valid_finding_creation(self) -> None:
        finding = _make_finding()
        assert finding.id == "F-001"
        assert finding.severity == "medium"
        assert finding.recommendation == "Fix it."

    def test_invalid_severity_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid severity"):
            Finding(id="X", title="Bad", severity="extreme", description="desc")

    def test_finding_defaults(self) -> None:
        finding = Finding(id="F-002", title="Minimal", severity="info", description="d")
        assert finding.recommendation == ""
        assert finding.evidence == ""


# ---------------------------------------------------------------------------
# Create report
# ---------------------------------------------------------------------------

class TestCreateReport:
    def test_create_returns_report_id(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Test Report")
        assert isinstance(rid, str)
        assert len(rid) == 12

    def test_create_with_audience_and_author(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Exec Report", audience="executive", author="Alice")
        report = reporter.generate(rid)
        assert report.metadata.audience == "executive"
        assert report.metadata.author == "Alice"

    def test_create_default_audience_is_technical(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Default Report")
        report = reporter.generate(rid)
        assert report.metadata.audience == "technical"


# ---------------------------------------------------------------------------
# Add sections and findings
# ---------------------------------------------------------------------------

class TestAddSectionsAndFindings:
    def test_add_section(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Report")
        reporter.add_section(rid, "Security", summary="Security findings")
        report = reporter.generate(rid)
        assert len(report.sections) == 1
        assert report.sections[0].title == "Security"
        assert report.sections[0].summary == "Security findings"

    def test_add_finding_to_section(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Report")
        reporter.add_section(rid, "Security")
        reporter.add_finding(rid, "Security", _make_finding())
        report = reporter.generate(rid)
        assert len(report.sections[0].findings) == 1

    def test_add_finding_to_nonexistent_section_raises(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Report")
        with pytest.raises(KeyError, match="not found"):
            reporter.add_finding(rid, "NoSuchSection", _make_finding())

    def test_add_finding_to_nonexistent_report_raises(self) -> None:
        reporter = SafetyReport()
        with pytest.raises(KeyError, match="not found"):
            reporter.add_finding("fake-id", "Section", _make_finding())

    def test_add_section_to_nonexistent_report_raises(self) -> None:
        reporter = SafetyReport()
        with pytest.raises(KeyError, match="not found"):
            reporter.add_section("fake-id", "Section")

    def test_multiple_sections(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Report")
        reporter.add_section(rid, "Section A")
        reporter.add_section(rid, "Section B")
        reporter.add_section(rid, "Section C")
        report = reporter.generate(rid)
        assert len(report.sections) == 3
        titles = [s.title for s in report.sections]
        assert titles == ["Section A", "Section B", "Section C"]


# ---------------------------------------------------------------------------
# Generate report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_generate_returns_generated_report(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        report = reporter.generate(rid)
        assert isinstance(report, GeneratedReport)

    def test_generate_total_findings(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        report = reporter.generate(rid)
        assert report.total_findings == 4

    def test_generate_severity_counts(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        report = reporter.generate(rid)
        assert report.critical_count == 1
        assert report.high_count == 1

    def test_generate_nonexistent_report_raises(self) -> None:
        reporter = SafetyReport()
        with pytest.raises(KeyError, match="not found"):
            reporter.generate("nonexistent")

    def test_generate_empty_report(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Empty Report")
        report = reporter.generate(rid)
        assert report.total_findings == 0
        assert report.critical_count == 0
        assert report.high_count == 0
        assert report.risk_rating == "minimal"

    def test_metadata_fields_populated(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Meta Test", audience="compliance", author="Bob")
        report = reporter.generate(rid)
        assert report.metadata.title == "Meta Test"
        assert report.metadata.author == "Bob"
        assert report.metadata.audience == "compliance"
        assert report.metadata.version == "1.0"
        assert isinstance(report.metadata.generated_at, float)


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

class TestExecutiveSummary:
    def test_summary_mentions_finding_count(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        report = reporter.generate(rid)
        assert "4 finding(s)" in report.executive_summary

    def test_summary_mentions_critical(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        report = reporter.generate(rid)
        assert "1 critical" in report.executive_summary

    def test_empty_report_summary(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Empty")
        report = reporter.generate(rid)
        assert "No findings" in report.executive_summary


# ---------------------------------------------------------------------------
# Risk rating
# ---------------------------------------------------------------------------

class TestRiskRating:
    def test_critical_finding_yields_critical_rating(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("R")
        reporter.add_section(rid, "S")
        reporter.add_finding(rid, "S", _make_finding(severity="critical"))
        assert reporter.generate(rid).risk_rating == "critical"

    def test_high_finding_yields_high_rating(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("R")
        reporter.add_section(rid, "S")
        reporter.add_finding(rid, "S", _make_finding(severity="high"))
        assert reporter.generate(rid).risk_rating == "high"

    def test_medium_finding_yields_medium_rating(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("R")
        reporter.add_section(rid, "S")
        reporter.add_finding(rid, "S", _make_finding(severity="medium"))
        assert reporter.generate(rid).risk_rating == "medium"

    def test_low_finding_yields_low_rating(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("R")
        reporter.add_section(rid, "S")
        reporter.add_finding(rid, "S", _make_finding(severity="low"))
        assert reporter.generate(rid).risk_rating == "low"

    def test_info_only_yields_minimal_rating(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("R")
        reporter.add_section(rid, "S")
        reporter.add_finding(rid, "S", _make_finding(severity="info"))
        assert reporter.generate(rid).risk_rating == "minimal"

    def test_no_findings_yields_minimal_rating(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("R")
        assert reporter.generate(rid).risk_rating == "minimal"


# ---------------------------------------------------------------------------
# Export text
# ---------------------------------------------------------------------------

class TestExportText:
    def test_export_text_contains_title(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        reporter.generate(rid)
        text = reporter.export_text(rid)
        assert "Audit Report" in text

    def test_export_text_contains_findings(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        reporter.generate(rid)
        text = reporter.export_text(rid)
        assert "INJ-001" in text
        assert "SQL injection" in text
        assert "[CRITICAL]" in text

    def test_export_text_contains_executive_summary(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        reporter.generate(rid)
        text = reporter.export_text(rid)
        assert "EXECUTIVE SUMMARY" in text

    def test_export_text_before_generate_raises(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Report")
        with pytest.raises(KeyError, match="not been generated"):
            reporter.export_text(rid)


# ---------------------------------------------------------------------------
# Export dict
# ---------------------------------------------------------------------------

class TestExportDict:
    def test_export_dict_structure(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        reporter.generate(rid)
        data = reporter.export_dict(rid)
        assert "metadata" in data
        assert "sections" in data
        assert "executive_summary" in data
        assert "risk_rating" in data
        assert data["total_findings"] == 4

    def test_export_dict_sections_contain_findings(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        reporter.generate(rid)
        data = reporter.export_dict(rid)
        injection_section = data["sections"][0]
        assert injection_section["title"] == "Injection"
        assert len(injection_section["findings"]) == 2

    def test_export_dict_before_generate_raises(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Report")
        with pytest.raises(KeyError, match="not been generated"):
            reporter.export_dict(rid)


# ---------------------------------------------------------------------------
# List reports
# ---------------------------------------------------------------------------

class TestListReports:
    def test_list_reports_empty(self) -> None:
        reporter = SafetyReport()
        assert reporter.list_reports() == []

    def test_list_reports_returns_all_ids(self) -> None:
        reporter = SafetyReport()
        id1 = reporter.create("A")
        id2 = reporter.create("B")
        ids = reporter.list_reports()
        assert id1 in ids
        assert id2 in ids
        assert len(ids) == 2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_initial(self) -> None:
        reporter = SafetyReport()
        s = reporter.stats()
        assert s.total_generated == 0
        assert s.by_audience == {}
        assert s.avg_findings == 0.0

    def test_stats_after_generation(self) -> None:
        reporter = SafetyReport()
        rid = _populated_report(reporter)
        reporter.generate(rid)
        s = reporter.stats()
        assert s.total_generated == 1
        assert s.by_audience["technical"] == 1
        assert s.avg_findings == 4.0

    def test_stats_multiple_reports(self) -> None:
        reporter = SafetyReport()
        rid1 = _populated_report(reporter)
        reporter.generate(rid1)

        rid2 = reporter.create("Executive Report", audience="executive")
        reporter.add_section(rid2, "Overview")
        reporter.add_finding(rid2, "Overview", _make_finding(severity="high"))
        reporter.add_finding(rid2, "Overview", _make_finding(finding_id="F-002", severity="low"))
        reporter.generate(rid2)

        s = reporter.stats()
        assert s.total_generated == 2
        assert s.by_audience["technical"] == 1
        assert s.by_audience["executive"] == 1
        assert s.avg_findings == 3.0  # (4 + 2) / 2


# ---------------------------------------------------------------------------
# Different audiences
# ---------------------------------------------------------------------------

class TestDifferentAudiences:
    def test_compliance_audience(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Compliance Audit", audience="compliance")
        report = reporter.generate(rid)
        assert report.metadata.audience == "compliance"

    def test_executive_audience_in_text_export(self) -> None:
        reporter = SafetyReport()
        rid = reporter.create("Board Report", audience="executive", author="CISO")
        reporter.generate(rid)
        text = reporter.export_text(rid)
        assert "Audience: executive" in text
        assert "Author: CISO" in text
