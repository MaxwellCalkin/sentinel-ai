"""Tests for structured compliance report generation."""

import pytest
from sentinel.compliance_report import (
    ComplianceReport,
    Control,
    Recommendation,
    FrameworkSummary,
    ReportOutput,
)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

class TestControls:
    def test_add_control(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("EU AI Act", "AIA-1", "Risk Classification", "compliant")
        output = report.generate()
        assert len(output.controls) == 1
        assert output.controls[0].framework == "EU AI Act"
        assert output.controls[0].control_id == "AIA-1"
        assert output.controls[0].status == "compliant"

    def test_invalid_status(self):
        report = ComplianceReport("Acme", "ChatBot")
        with pytest.raises(ValueError, match="Invalid status"):
            report.add_control("EU AI Act", "AIA-1", "Risk Classification", "unknown")

    def test_multiple_frameworks(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("EU AI Act", "AIA-1", "Risk Classification", "compliant")
        report.add_control("NIST AI RMF", "GOV-1", "Governance", "partial")
        report.add_control("ISO 42001", "ISO-1", "AI Policy", "non_compliant")
        output = report.generate()
        assert len(output.controls) == 3
        frameworks = [c.framework for c in output.controls]
        assert "EU AI Act" in frameworks
        assert "NIST AI RMF" in frameworks
        assert "ISO 42001" in frameworks


# ---------------------------------------------------------------------------
# Risk Level
# ---------------------------------------------------------------------------

class TestRisk:
    def test_set_risk_level(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.set_risk_level("high")
        output = report.generate()
        assert output.risk_level == "high"

    def test_invalid_risk(self):
        report = ComplianceReport("Acme", "ChatBot")
        with pytest.raises(ValueError, match="Invalid risk level"):
            report.set_risk_level("extreme")


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_add_recommendation(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_recommendation("high", "Implement human oversight", "EU AI Act")
        output = report.generate()
        assert len(output.recommendations) == 1
        assert output.recommendations[0].priority == "high"
        assert output.recommendations[0].description == "Implement human oversight"
        assert output.recommendations[0].framework == "EU AI Act"

    def test_recommendation_sorting(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_recommendation("low", "Minor improvement")
        report.add_recommendation("critical", "Fix immediately")
        report.add_recommendation("medium", "Plan for next quarter")
        report.add_recommendation("high", "Address soon")
        output = report.generate()
        priorities = [r.priority for r in output.recommendations]
        assert priorities == ["critical", "high", "medium", "low"]


# ---------------------------------------------------------------------------
# Framework Summary
# ---------------------------------------------------------------------------

class TestFramework:
    def test_framework_summary(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("EU AI Act", "AIA-1", "Risk Classification", "compliant")
        report.add_control("EU AI Act", "AIA-2", "Transparency", "non_compliant")
        report.add_control("EU AI Act", "AIA-3", "Human Oversight", "partial")
        summary = report.framework_summary("EU AI Act")
        assert summary.framework == "EU AI Act"
        assert summary.total_controls == 3
        assert summary.compliant == 1
        assert summary.partial == 1
        assert summary.non_compliant == 1
        assert summary.not_applicable == 0

    def test_compliance_rate_calculation(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("SOC 2", "SOC-1", "Access Control", "compliant")
        report.add_control("SOC 2", "SOC-2", "Encryption", "compliant")
        report.add_control("SOC 2", "SOC-3", "Logging", "non_compliant")
        report.add_control("SOC 2", "SOC-4", "Backup", "non_compliant")
        summary = report.framework_summary("SOC 2")
        # rate = (2 + 0) / 4 = 0.5
        assert summary.compliance_rate == 0.5

    def test_partial_credit(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("NIST", "N-1", "Govern", "partial")
        report.add_control("NIST", "N-2", "Map", "partial")
        summary = report.framework_summary("NIST")
        # rate = (0 + 0.5 * 2) / 2 = 0.5
        assert summary.compliance_rate == 0.5

    def test_not_applicable_excluded_from_rate(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("ISO", "I-1", "Policy", "compliant")
        report.add_control("ISO", "I-2", "Scope", "not_applicable")
        summary = report.framework_summary("ISO")
        # rate = 1 / (2 - 1) = 1.0
        assert summary.compliance_rate == 1.0
        assert summary.not_applicable == 1


# ---------------------------------------------------------------------------
# Generate Report
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_generate_report(self):
        report = ComplianceReport("Acme Corp", "LLM Gateway")
        report.set_risk_level("low")
        report.add_control("EU AI Act", "AIA-1", "Risk Classification", "compliant")
        report.add_recommendation("medium", "Continue monitoring")
        output = report.generate()
        assert output.organization == "Acme Corp"
        assert output.system_name == "LLM Gateway"
        assert output.risk_level == "low"

    def test_report_structure(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("EU AI Act", "AIA-1", "Risk Classification", "compliant")
        output = report.generate()
        assert isinstance(output, ReportOutput)
        assert isinstance(output.frameworks, list)
        assert isinstance(output.controls, list)
        assert isinstance(output.recommendations, list)
        assert isinstance(output.generated_at, str)
        # ISO timestamp has 'T' separator
        assert "T" in output.generated_at

    def test_overall_compliance(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("EU AI Act", "AIA-1", "Risk", "compliant")
        report.add_control("EU AI Act", "AIA-2", "Transparency", "compliant")
        report.add_control("NIST", "N-1", "Govern", "non_compliant")
        report.add_control("NIST", "N-2", "Map", "non_compliant")
        output = report.generate()
        # overall = (2 + 0) / 4 = 0.5
        assert output.overall_compliance_rate == 0.5

    def test_overall_compliance_with_partial(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("FW", "C-1", "Control A", "compliant")
        report.add_control("FW", "C-2", "Control B", "partial")
        output = report.generate()
        # overall = (1 + 0.5) / 2 = 0.75
        assert output.overall_compliance_rate == 0.75


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_dict(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.set_risk_level("high")
        report.add_control("EU AI Act", "AIA-1", "Risk Classification", "compliant",
                           evidence="Documented risk assessment", notes="Reviewed Q1")
        report.add_recommendation("high", "Add monitoring", "EU AI Act")
        exported = report.export_dict()
        assert exported["organization"] == "Acme"
        assert exported["system_name"] == "ChatBot"
        assert exported["risk_level"] == "high"
        assert len(exported["controls"]) == 1
        assert exported["controls"][0]["evidence"] == "Documented risk assessment"
        assert len(exported["recommendations"]) == 1
        assert "overall_compliance_rate" in exported
        assert "generated_at" in exported
        assert "frameworks" in exported

    def test_list_frameworks(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("EU AI Act", "AIA-1", "Risk", "compliant")
        report.add_control("NIST AI RMF", "GOV-1", "Govern", "partial")
        report.add_control("EU AI Act", "AIA-2", "Transparency", "compliant")
        frameworks = report.list_frameworks()
        assert len(frameworks) == 2
        assert "EU AI Act" in frameworks
        assert "NIST AI RMF" in frameworks


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdge:
    def test_empty_report(self):
        report = ComplianceReport("Acme", "ChatBot")
        output = report.generate()
        assert output.overall_compliance_rate == 1.0
        assert output.controls == []
        assert output.recommendations == []
        assert output.frameworks == []
        assert output.organization == "Acme"

    def test_all_not_applicable(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("EU AI Act", "AIA-1", "Risk", "not_applicable")
        report.add_control("EU AI Act", "AIA-2", "Transparency", "not_applicable")
        output = report.generate()
        assert output.overall_compliance_rate == 1.0
        summary = report.framework_summary("EU AI Act")
        assert summary.compliance_rate == 1.0
        assert summary.not_applicable == 2

    def test_control_evidence_and_notes_default_empty(self):
        report = ComplianceReport("Acme", "ChatBot")
        report.add_control("FW", "C-1", "Title", "compliant")
        output = report.generate()
        assert output.controls[0].evidence == ""
        assert output.controls[0].notes == ""

    def test_default_risk_level_is_medium(self):
        report = ComplianceReport("Acme", "ChatBot")
        output = report.generate()
        assert output.risk_level == "medium"

    def test_framework_summary_empty_framework(self):
        report = ComplianceReport("Acme", "ChatBot")
        summary = report.framework_summary("nonexistent")
        assert summary.total_controls == 0
        assert summary.compliance_rate == 1.0
