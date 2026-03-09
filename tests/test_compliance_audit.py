"""Tests for compliance auditing."""

import pytest
from sentinel.compliance_audit import (
    ComplianceAuditor, AuditReport, CheckResult, CheckStatus,
)


# ---------------------------------------------------------------------------
# Custom checks
# ---------------------------------------------------------------------------

class TestCustomChecks:
    def test_passing_check(self):
        a = ComplianceAuditor()
        a.add_check("encryption", lambda ctx: ctx.get("encryption", False))
        report = a.audit({"encryption": True})
        assert report.passed == 1
        assert report.failed == 0
        assert report.score == 1.0

    def test_failing_check(self):
        a = ComplianceAuditor()
        a.add_check("encryption", lambda ctx: ctx.get("encryption", False))
        report = a.audit({"encryption": False})
        assert report.passed == 0
        assert report.failed == 1
        assert report.score == 0.0

    def test_multiple_checks(self):
        a = ComplianceAuditor()
        a.add_check("enc", lambda ctx: ctx.get("encryption"))
        a.add_check("log", lambda ctx: ctx.get("logging"))
        report = a.audit({"encryption": True, "logging": False})
        assert report.passed == 1
        assert report.failed == 1
        assert report.score == 0.5

    def test_check_with_remediation(self):
        a = ComplianceAuditor()
        a.add_check("enc", lambda ctx: False, remediation="Enable TLS")
        report = a.audit({})
        assert report.checks[0].remediation == "Enable TLS"


# ---------------------------------------------------------------------------
# Built-in frameworks
# ---------------------------------------------------------------------------

class TestFrameworks:
    def test_load_gdpr(self):
        a = ComplianceAuditor()
        count = a.load_framework("gdpr")
        assert count == 4
        assert a.check_count == 4

    def test_load_nist(self):
        a = ComplianceAuditor()
        count = a.load_framework("nist_ai_rmf")
        assert count == 4

    def test_load_eu_ai_act(self):
        a = ComplianceAuditor()
        count = a.load_framework("eu_ai_act")
        assert count == 3

    def test_load_all_frameworks(self):
        a = ComplianceAuditor()
        total = a.load_all_frameworks()
        assert total == 11  # 4 + 4 + 3

    def test_load_unknown_framework(self):
        a = ComplianceAuditor()
        assert a.load_framework("unknown") == 0

    def test_available_frameworks(self):
        a = ComplianceAuditor()
        assert "gdpr" in a.available_frameworks
        assert "nist_ai_rmf" in a.available_frameworks
        assert "eu_ai_act" in a.available_frameworks


# ---------------------------------------------------------------------------
# GDPR compliance
# ---------------------------------------------------------------------------

class TestGDPR:
    def test_fully_compliant(self):
        a = ComplianceAuditor()
        a.load_framework("gdpr")
        report = a.audit({
            "data_minimization": True,
            "consent_tracking": True,
            "deletion_support": True,
            "encryption": True,
        })
        assert report.compliant
        assert report.score == 1.0

    def test_non_compliant(self):
        a = ComplianceAuditor()
        a.load_framework("gdpr")
        report = a.audit({
            "data_minimization": False,
            "consent_tracking": False,
            "deletion_support": False,
            "encryption": False,
        })
        assert not report.compliant
        assert report.score == 0.0
        assert report.failed == 4

    def test_partial_compliance(self):
        a = ComplianceAuditor()
        a.load_framework("gdpr")
        report = a.audit({
            "data_minimization": True,
            "consent_tracking": True,
            "deletion_support": False,
            "encryption": False,
        })
        assert report.score == 0.5


# ---------------------------------------------------------------------------
# Framework filtering
# ---------------------------------------------------------------------------

class TestFrameworkFilter:
    def test_filter_to_framework(self):
        a = ComplianceAuditor()
        a.load_all_frameworks()
        report = a.audit(
            {"data_minimization": True, "consent_tracking": True,
             "deletion_support": True, "encryption": True},
            frameworks=["gdpr"],
        )
        evaluated = [c for c in report.checks if c.status != CheckStatus.SKIP]
        assert len(evaluated) == 4
        assert all(c.framework == "gdpr" for c in evaluated)


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure(self):
        a = ComplianceAuditor()
        a.add_check("a", lambda ctx: True, framework="fw1")
        a.add_check("b", lambda ctx: False, framework="fw2")
        report = a.audit({})
        assert isinstance(report, AuditReport)
        assert report.timestamp > 0
        assert set(report.frameworks) == {"fw1", "fw2"}

    def test_compliant_property(self):
        a = ComplianceAuditor()
        a.add_check("low", lambda ctx: False, severity="low")
        report = a.audit({})
        # Low severity failure doesn't break compliance
        assert report.compliant

    def test_high_severity_breaks_compliance(self):
        a = ComplianceAuditor()
        a.add_check("critical", lambda ctx: False, severity="high")
        report = a.audit({})
        assert not report.compliant


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_check_error(self):
        a = ComplianceAuditor()
        a.add_check("bad", lambda ctx: 1 / 0)
        report = a.audit({})
        assert report.errors == 1
        assert report.checks[0].status == CheckStatus.ERROR
        assert report.checks[0].error is not None

    def test_empty_audit(self):
        a = ComplianceAuditor()
        report = a.audit({})
        assert report.score == 1.0
        assert report.compliant
