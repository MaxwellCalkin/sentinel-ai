"""Tests for regulatory compliance mapping."""

from sentinel.compliance import (
    ComplianceMapper,
    ComplianceReport,
    ComplianceStatus,
    Framework,
    FrameworkReport,
)
from sentinel.core import Finding, RiskLevel, ScanResult


def _clean_result() -> ScanResult:
    return ScanResult(text="safe text", risk=RiskLevel.NONE, findings=[])


def _risky_result(
    category: str = "prompt_injection",
    risk: RiskLevel = RiskLevel.HIGH,
) -> ScanResult:
    return ScanResult(
        text="risky text",
        risk=risk,
        findings=[
            Finding(
                scanner="test",
                category=category,
                description="test finding",
                risk=risk,
            )
        ],
        blocked=risk >= RiskLevel.HIGH,
    )


class TestComplianceMapperBasic:
    def test_clean_results_all_compliant(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate([_clean_result()])
        assert report.overall_risk == RiskLevel.NONE
        for fw in report.frameworks:
            assert fw.status == ComplianceStatus.COMPLIANT

    def test_returns_all_frameworks_by_default(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate([_clean_result()])
        assert len(report.frameworks) == 3
        fw_names = {fr.framework for fr in report.frameworks}
        assert fw_names == {Framework.EU_AI_ACT, Framework.NIST_RMF, Framework.ISO_42001}

    def test_single_framework_selection(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_clean_result()], frameworks=[Framework.EU_AI_ACT]
        )
        assert len(report.frameworks) == 1
        assert report.frameworks[0].framework == Framework.EU_AI_ACT

    def test_report_has_timestamp(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate([_clean_result()])
        assert "UTC" in report.generated_at

    def test_report_total_scans(self):
        mapper = ComplianceMapper()
        results = [_clean_result(), _clean_result(), _risky_result()]
        report = mapper.evaluate(results)
        assert report.total_scans == 3


class TestEUAIAct:
    def test_clean_classified_minimal_risk(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_clean_result()], frameworks=[Framework.EU_AI_ACT]
        )
        fr = report.frameworks[0]
        assert fr.risk_classification == "Minimal Risk"

    def test_high_risk_classification(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("prompt_injection", RiskLevel.HIGH)],
            frameworks=[Framework.EU_AI_ACT],
        )
        fr = report.frameworks[0]
        assert fr.risk_classification == "High Risk (Article 6)"

    def test_critical_risk_unacceptable(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("harmful_content", RiskLevel.CRITICAL)],
            frameworks=[Framework.EU_AI_ACT],
        )
        fr = report.frameworks[0]
        assert fr.risk_classification == "Unacceptable Risk (Article 5)"

    def test_medium_risk_limited(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("pii", RiskLevel.MEDIUM)],
            frameworks=[Framework.EU_AI_ACT],
        )
        fr = report.frameworks[0]
        assert fr.risk_classification == "Limited Risk (Article 52)"

    def test_pii_findings_affect_data_governance(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("pii", RiskLevel.MEDIUM)],
            frameworks=[Framework.EU_AI_ACT],
        )
        fr = report.frameworks[0]
        data_gov = next(c for c in fr.controls if c.control_id == "EU-AIA-10")
        assert data_gov.findings_count > 0
        assert data_gov.status != ComplianceStatus.COMPLIANT

    def test_injection_affects_risk_management(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("prompt_injection", RiskLevel.HIGH)],
            frameworks=[Framework.EU_AI_ACT],
        )
        fr = report.frameworks[0]
        risk_mgmt = next(c for c in fr.controls if c.control_id == "EU-AIA-9")
        assert risk_mgmt.status == ComplianceStatus.NON_COMPLIANT

    def test_has_all_eu_controls(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_clean_result()], frameworks=[Framework.EU_AI_ACT]
        )
        fr = report.frameworks[0]
        assert fr.controls_assessed == 7


class TestNISTRMF:
    def test_clean_all_compliant(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_clean_result()], frameworks=[Framework.NIST_RMF]
        )
        fr = report.frameworks[0]
        assert fr.status == ComplianceStatus.COMPLIANT

    def test_injection_affects_governance(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("prompt_injection", RiskLevel.HIGH)],
            frameworks=[Framework.NIST_RMF],
        )
        fr = report.frameworks[0]
        govern = next(c for c in fr.controls if c.control_id == "GOVERN-1")
        assert govern.status == ComplianceStatus.NON_COMPLIANT

    def test_measurement_controls(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("hallucination", RiskLevel.LOW)],
            frameworks=[Framework.NIST_RMF],
        )
        fr = report.frameworks[0]
        measure = next(c for c in fr.controls if c.control_id == "MEASURE-1")
        # Low risk hallucination is at threshold, so non-compliant
        assert measure.findings_count > 0

    def test_has_all_nist_controls(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_clean_result()], frameworks=[Framework.NIST_RMF]
        )
        fr = report.frameworks[0]
        assert fr.controls_assessed == 8

    def test_no_risk_classification_for_nist(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_clean_result()], frameworks=[Framework.NIST_RMF]
        )
        fr = report.frameworks[0]
        assert fr.risk_classification is None


class TestISO42001:
    def test_clean_all_compliant(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_clean_result()], frameworks=[Framework.ISO_42001]
        )
        fr = report.frameworks[0]
        assert fr.status == ComplianceStatus.COMPLIANT

    def test_security_control_with_injection(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("prompt_injection", RiskLevel.HIGH)],
            frameworks=[Framework.ISO_42001],
        )
        fr = report.frameworks[0]
        sec = next(c for c in fr.controls if c.control_id == "ISO-A.6")
        assert sec.status == ComplianceStatus.NON_COMPLIANT
        assert sec.remediation is not None

    def test_has_all_iso_controls(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_clean_result()], frameworks=[Framework.ISO_42001]
        )
        fr = report.frameworks[0]
        assert fr.controls_assessed == 6


class TestRemediation:
    def test_non_compliant_has_remediation(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("pii", RiskLevel.HIGH)],
            frameworks=[Framework.EU_AI_ACT],
        )
        fr = report.frameworks[0]
        # EU-AIA-10 (Data Governance) should be non-compliant with remediation
        data_gov = next(c for c in fr.controls if c.control_id == "EU-AIA-10")
        assert data_gov.status == ComplianceStatus.NON_COMPLIANT
        assert data_gov.remediation is not None
        assert "PII" in data_gov.remediation

    def test_compliant_no_remediation(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_clean_result()], frameworks=[Framework.EU_AI_ACT]
        )
        fr = report.frameworks[0]
        for ctrl in fr.controls:
            if ctrl.status == ComplianceStatus.COMPLIANT:
                assert ctrl.remediation is None


class TestReportOutput:
    def test_to_dict_structure(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate([_clean_result()])
        d = report.to_dict()
        assert "generated_at" in d
        assert "overall_risk" in d
        assert "frameworks" in d
        assert len(d["frameworks"]) == 3
        for fw in d["frameworks"]:
            assert "controls" in fw
            assert "status" in fw

    def test_to_markdown_contains_headers(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("prompt_injection", RiskLevel.HIGH)]
        )
        md = report.to_markdown()
        assert "# AI Compliance Assessment Report" in md
        assert "EU AI Act" in md
        assert "NIST" in md
        assert "ISO" in md

    def test_markdown_has_remediation_section(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate(
            [_risky_result("prompt_injection", RiskLevel.HIGH)]
        )
        md = report.to_markdown()
        assert "Recommended Actions" in md

    def test_markdown_clean_no_remediation_section(self):
        mapper = ComplianceMapper()
        report = mapper.evaluate([_clean_result()])
        md = report.to_markdown()
        assert "Recommended Actions" not in md


class TestMultipleFindings:
    def test_mixed_findings(self):
        mapper = ComplianceMapper()
        results = [
            _risky_result("prompt_injection", RiskLevel.HIGH),
            _risky_result("pii", RiskLevel.MEDIUM),
            _clean_result(),
        ]
        report = mapper.evaluate(results, frameworks=[Framework.EU_AI_ACT])
        fr = report.frameworks[0]
        # Should be non-compliant due to high-risk injection
        assert fr.status == ComplianceStatus.NON_COMPLIANT
        assert fr.controls_non_compliant > 0

    def test_overall_risk_is_max(self):
        mapper = ComplianceMapper()
        results = [
            _risky_result("pii", RiskLevel.LOW),
            _risky_result("prompt_injection", RiskLevel.CRITICAL),
        ]
        report = mapper.evaluate(results)
        assert report.overall_risk == RiskLevel.CRITICAL

    def test_partial_compliance(self):
        """Low-risk findings create partial compliance for controls with medium threshold."""
        mapper = ComplianceMapper()
        results = [_risky_result("hallucination", RiskLevel.LOW)]
        report = mapper.evaluate(results, frameworks=[Framework.EU_AI_ACT])
        fr = report.frameworks[0]
        transparency = next(c for c in fr.controls if c.control_id == "EU-AIA-13")
        assert transparency.status == ComplianceStatus.PARTIAL
