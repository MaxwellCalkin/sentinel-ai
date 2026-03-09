"""Tests for automated compliance rule checking with evidence collection."""

import pytest
from sentinel.compliance_checker import (
    ComplianceChecker,
    ComplianceRule,
    ComplianceResult,
    ComplianceOverview,
    CheckerStats,
    Evidence,
)


def _make_rule(
    rule_id: str = "GDPR-1",
    name: str = "Data Encryption",
    description: str = "All PII must be encrypted at rest",
    framework: str = "GDPR",
    severity: str = "critical",
) -> ComplianceRule:
    return ComplianceRule(
        rule_id=rule_id,
        name=name,
        description=description,
        framework=framework,
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Rule registration
# ---------------------------------------------------------------------------


class TestAddAndListRules:
    def test_add_single_rule(self):
        checker = ComplianceChecker()
        rule = _make_rule()
        checker.add_rule(rule)

        rules = checker.list_rules()
        assert len(rules) == 1
        assert rules[0].rule_id == "GDPR-1"

    def test_add_multiple_rules(self):
        checker = ComplianceChecker()
        rules = [
            _make_rule(rule_id="GDPR-1", framework="GDPR"),
            _make_rule(rule_id="SOC2-1", framework="SOC2", severity="high"),
            _make_rule(rule_id="HIPAA-1", framework="HIPAA", severity="medium"),
        ]
        checker.add_rules(rules)

        assert len(checker.list_rules()) == 3

    def test_list_rules_filtered_by_framework(self):
        checker = ComplianceChecker()
        checker.add_rules([
            _make_rule(rule_id="GDPR-1", framework="GDPR"),
            _make_rule(rule_id="GDPR-2", framework="GDPR", severity="high"),
            _make_rule(rule_id="SOC2-1", framework="SOC2", severity="high"),
        ])

        gdpr_rules = checker.list_rules(framework="GDPR")
        assert len(gdpr_rules) == 2
        assert all(r.framework == "GDPR" for r in gdpr_rules)

    def test_list_rules_no_match_returns_empty(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule(framework="GDPR"))

        assert checker.list_rules(framework="NIST") == []


# ---------------------------------------------------------------------------
# Checking rules
# ---------------------------------------------------------------------------


class TestCheck:
    def test_check_passing_rule(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())

        result = checker.check("GDPR-1", passed=True, details="AES-256 verified")

        assert result.passed is True
        assert result.rule.rule_id == "GDPR-1"
        assert len(result.evidence) == 1
        assert result.evidence[0].passed is True
        assert result.evidence[0].details == "AES-256 verified"

    def test_check_failing_rule(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())

        result = checker.check("GDPR-1", passed=False, details="No encryption found")

        assert result.passed is False
        assert result.evidence[0].passed is False

    def test_check_unknown_rule_raises_key_error(self):
        checker = ComplianceChecker()

        with pytest.raises(KeyError, match="Unknown rule_id"):
            checker.check("NONEXISTENT", passed=True)

    def test_check_with_artifacts(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())

        artifacts = {"scan_tool": "openssl", "version": "3.0"}
        result = checker.check(
            "GDPR-1", passed=True, details="Verified", artifacts=artifacts
        )

        assert result.evidence[0].artifacts == artifacts

    def test_check_with_remediation(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())

        result = checker.check(
            "GDPR-1",
            passed=False,
            details="Missing encryption",
            remediation="Enable AES-256 on storage volumes",
        )

        assert result.remediation == "Enable AES-256 on storage volumes"


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------


class TestEvidence:
    def test_get_evidence_for_rule(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())
        checker.check("GDPR-1", passed=True, details="First check")

        evidence = checker.get_evidence("GDPR-1")
        assert len(evidence) == 1
        assert evidence[0].details == "First check"

    def test_multiple_evidence_entries_per_rule(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())
        checker.check("GDPR-1", passed=True, details="First check")
        checker.check("GDPR-1", passed=False, details="Second check failed")
        checker.check("GDPR-1", passed=True, details="Third check passed")

        evidence = checker.get_evidence("GDPR-1")
        assert len(evidence) == 3
        assert evidence[0].details == "First check"
        assert evidence[1].details == "Second check failed"
        assert evidence[2].details == "Third check passed"

    def test_evidence_has_timestamp(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())
        checker.check("GDPR-1", passed=True)

        evidence = checker.get_evidence("GDPR-1")
        assert evidence[0].collected_at > 0

    def test_get_evidence_unknown_rule_raises_key_error(self):
        checker = ComplianceChecker()

        with pytest.raises(KeyError, match="Unknown rule_id"):
            checker.get_evidence("NONEXISTENT")

    def test_evidence_returns_copy(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())
        checker.check("GDPR-1", passed=True)

        evidence_a = checker.get_evidence("GDPR-1")
        evidence_b = checker.get_evidence("GDPR-1")
        assert evidence_a is not evidence_b


# ---------------------------------------------------------------------------
# Check all
# ---------------------------------------------------------------------------


class TestCheckAll:
    def test_check_all_multiple_rules(self):
        checker = ComplianceChecker()
        checker.add_rules([
            _make_rule(rule_id="GDPR-1"),
            _make_rule(rule_id="SOC2-1", framework="SOC2", severity="high"),
        ])

        results = checker.check_all({"GDPR-1": True, "SOC2-1": False})

        assert len(results) == 2
        passed_ids = {r.rule.rule_id for r in results if r.passed}
        failed_ids = {r.rule.rule_id for r in results if not r.passed}
        assert passed_ids == {"GDPR-1"}
        assert failed_ids == {"SOC2-1"}

    def test_check_all_unknown_rule_raises_key_error(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())

        with pytest.raises(KeyError):
            checker.check_all({"GDPR-1": True, "MISSING": False})


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


class TestOverview:
    def test_overview_all_passing(self):
        checker = ComplianceChecker()
        checker.add_rules([
            _make_rule(rule_id="GDPR-1"),
            _make_rule(rule_id="GDPR-2", severity="high"),
        ])
        checker.check("GDPR-1", passed=True)
        checker.check("GDPR-2", passed=True)

        overview = checker.overview()

        assert overview.total_rules == 2
        assert overview.passed == 2
        assert overview.failed == 0
        assert overview.pass_rate == 100.0
        assert overview.critical_failures == []

    def test_overview_with_failures(self):
        checker = ComplianceChecker()
        checker.add_rules([
            _make_rule(rule_id="GDPR-1", severity="critical"),
            _make_rule(rule_id="SOC2-1", framework="SOC2", severity="high"),
        ])
        checker.check("GDPR-1", passed=False)
        checker.check("SOC2-1", passed=True)

        overview = checker.overview()

        assert overview.total_rules == 2
        assert overview.passed == 1
        assert overview.failed == 1
        assert overview.pass_rate == 50.0

    def test_overview_pass_rate_calculation(self):
        checker = ComplianceChecker()
        checker.add_rules([
            _make_rule(rule_id=f"R-{i}", severity="high") for i in range(3)
        ])
        checker.check("R-0", passed=True)
        checker.check("R-1", passed=True)
        checker.check("R-2", passed=False)

        overview = checker.overview()
        assert abs(overview.pass_rate - 66.66666666666667) < 0.01

    def test_overview_critical_failures_listed(self):
        checker = ComplianceChecker()
        checker.add_rules([
            _make_rule(rule_id="CRIT-1", severity="critical"),
            _make_rule(rule_id="HIGH-1", severity="high"),
            _make_rule(rule_id="CRIT-2", severity="critical"),
        ])
        checker.check("CRIT-1", passed=False)
        checker.check("HIGH-1", passed=False)
        checker.check("CRIT-2", passed=False)

        overview = checker.overview()

        assert "CRIT-1" in overview.critical_failures
        assert "CRIT-2" in overview.critical_failures
        assert "HIGH-1" not in overview.critical_failures

    def test_overview_by_framework_breakdown(self):
        checker = ComplianceChecker()
        checker.add_rules([
            _make_rule(rule_id="GDPR-1", framework="GDPR"),
            _make_rule(rule_id="GDPR-2", framework="GDPR", severity="high"),
            _make_rule(rule_id="SOC2-1", framework="SOC2", severity="high"),
        ])
        checker.check("GDPR-1", passed=True)
        checker.check("GDPR-2", passed=False)
        checker.check("SOC2-1", passed=True)

        overview = checker.overview()

        assert overview.by_framework["GDPR"]["passed"] == 1
        assert overview.by_framework["GDPR"]["failed"] == 1
        assert overview.by_framework["SOC2"]["passed"] == 1
        assert overview.by_framework["SOC2"]["failed"] == 0

    def test_empty_checker_overview(self):
        checker = ComplianceChecker()

        overview = checker.overview()

        assert overview.total_rules == 0
        assert overview.passed == 0
        assert overview.failed == 0
        assert overview.pass_rate == 0.0
        assert overview.by_framework == {}
        assert overview.critical_failures == []

    def test_overview_uses_latest_evidence(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule(rule_id="R-1", severity="high"))

        checker.check("R-1", passed=False)
        checker.check("R-1", passed=True)

        overview = checker.overview()
        assert overview.passed == 1
        assert overview.failed == 0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_all_rules(self):
        checker = ComplianceChecker()
        checker.add_rules([
            _make_rule(rule_id="GDPR-1", framework="GDPR"),
            _make_rule(rule_id="SOC2-1", framework="SOC2", severity="high"),
        ])
        checker.check("GDPR-1", passed=True, details="Encrypted")
        checker.check("SOC2-1", passed=False, details="Access control missing")

        exported = checker.export()

        assert len(exported) == 2
        gdpr_entry = next(e for e in exported if e["rule_id"] == "GDPR-1")
        assert gdpr_entry["passed"] is True
        assert gdpr_entry["framework"] == "GDPR"
        assert gdpr_entry["evidence_count"] == 1
        assert gdpr_entry["latest_details"] == "Encrypted"

    def test_export_filtered_by_framework(self):
        checker = ComplianceChecker()
        checker.add_rules([
            _make_rule(rule_id="GDPR-1", framework="GDPR"),
            _make_rule(rule_id="SOC2-1", framework="SOC2", severity="high"),
        ])
        checker.check("GDPR-1", passed=True)
        checker.check("SOC2-1", passed=True)

        exported = checker.export(framework="GDPR")

        assert len(exported) == 1
        assert exported[0]["rule_id"] == "GDPR-1"

    def test_export_with_no_evidence(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())

        exported = checker.export()

        assert len(exported) == 1
        assert exported[0]["passed"] is False
        assert exported[0]["evidence_count"] == 0
        assert exported[0]["latest_details"] == ""


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_initial(self):
        checker = ComplianceChecker()

        stats = checker.stats()

        assert stats.total_checks == 0
        assert stats.total_passed == 0
        assert stats.total_failed == 0
        assert stats.by_framework == {}

    def test_stats_after_checks(self):
        checker = ComplianceChecker()
        checker.add_rules([
            _make_rule(rule_id="GDPR-1", framework="GDPR"),
            _make_rule(rule_id="SOC2-1", framework="SOC2", severity="high"),
        ])
        checker.check("GDPR-1", passed=True)
        checker.check("SOC2-1", passed=False)
        checker.check("GDPR-1", passed=True)

        stats = checker.stats()

        assert stats.total_checks == 3
        assert stats.total_passed == 2
        assert stats.total_failed == 1
        assert stats.by_framework["GDPR"] == 2
        assert stats.by_framework["SOC2"] == 1

    def test_stats_returns_copy(self):
        checker = ComplianceChecker()
        checker.add_rule(_make_rule())
        checker.check("GDPR-1", passed=True)

        stats_a = checker.stats()
        stats_b = checker.stats()
        assert stats_a is not stats_b
        assert stats_a.by_framework is not stats_b.by_framework


# ---------------------------------------------------------------------------
# Dataclass validation
# ---------------------------------------------------------------------------


class TestRuleValidation:
    def test_invalid_severity_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid severity"):
            _make_rule(severity="urgent")
