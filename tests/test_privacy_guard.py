"""Tests for privacy guard."""

import time
import pytest
from sentinel.privacy_guard import (
    PrivacyGuard,
    PrivacyCheck,
    MinimizationResult,
    PrivacyReport,
)


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

class TestInput:
    def test_clean_input(self):
        guard = PrivacyGuard()
        result = guard.check_input("Hello, how can I help you?")
        assert result.compliant
        assert result.issues == []
        assert result.sensitive_categories == []
        assert result.purpose_allowed

    def test_input_with_email(self):
        guard = PrivacyGuard()
        result = guard.check_input("Contact me at user@example.com")
        assert not result.compliant
        assert "email" in result.sensitive_categories
        assert len(result.issues) > 0

    def test_purpose_blocked(self):
        guard = PrivacyGuard(purposes=["support"])
        result = guard.check_input("Hello", purpose="marketing")
        assert not result.compliant
        assert not result.purpose_allowed
        assert any("marketing" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# Output checks
# ---------------------------------------------------------------------------

class TestOutput:
    def test_clean_output(self):
        guard = PrivacyGuard()
        result = guard.check_output("Here is your answer.")
        assert result.compliant
        assert result.issues == []
        assert result.purpose_allowed

    def test_output_with_pii(self):
        guard = PrivacyGuard()
        result = guard.check_output("Your SSN is 123-45-6789")
        assert not result.compliant
        assert "ssn" in result.sensitive_categories


# ---------------------------------------------------------------------------
# Minimization
# ---------------------------------------------------------------------------

class TestMinimize:
    def test_minimize_email(self):
        guard = PrivacyGuard()
        result = guard.minimize("Email me at alice@example.com please")
        assert "alice@example.com" not in result.minimized
        assert "[REDACTED_EMAIL]" in result.minimized
        assert result.items_removed == 1
        assert "email" in result.categories_removed

    def test_minimize_phone(self):
        guard = PrivacyGuard()
        result = guard.minimize("Call 555-123-4567")
        assert "555-123-4567" not in result.minimized
        assert "[REDACTED_PHONE]" in result.minimized
        assert result.items_removed == 1
        assert "phone" in result.categories_removed

    def test_minimize_mixed(self):
        guard = PrivacyGuard()
        text = "Email bob@test.org or call 555.111.2222"
        result = guard.minimize(text)
        assert result.items_removed == 2
        assert "email" in result.categories_removed
        assert "phone" in result.categories_removed
        assert "bob@test.org" not in result.minimized
        assert "555.111.2222" not in result.minimized

    def test_minimize_clean(self):
        guard = PrivacyGuard()
        result = guard.minimize("Nothing sensitive here")
        assert result.minimized == result.original
        assert result.items_removed == 0
        assert result.categories_removed == []


# ---------------------------------------------------------------------------
# Purpose management
# ---------------------------------------------------------------------------

class TestPurpose:
    def test_add_purpose(self):
        guard = PrivacyGuard(purposes=["support"])
        assert not guard.is_purpose_allowed("analytics")
        guard.add_purpose("analytics")
        assert guard.is_purpose_allowed("analytics")

    def test_is_allowed(self):
        guard = PrivacyGuard(purposes=["support", "billing"])
        assert guard.is_purpose_allowed("support")
        assert guard.is_purpose_allowed("billing")
        assert not guard.is_purpose_allowed("marketing")

    def test_no_purposes_all_allowed(self):
        guard = PrivacyGuard()
        assert guard.is_purpose_allowed("anything")
        assert guard.is_purpose_allowed("random_purpose")


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

class TestRetention:
    def test_within_retention(self):
        guard = PrivacyGuard(retention_days=30)
        recent = time.time() - 86400  # 1 day ago
        assert guard.retention_check(recent)

    def test_expired_retention(self):
        guard = PrivacyGuard(retention_days=30)
        old = time.time() - (31 * 86400)  # 31 days ago
        assert not guard.retention_check(old)


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------

class TestCustom:
    def test_custom_pattern(self):
        guard = PrivacyGuard()
        guard.add_sensitive_pattern(r'ACCT-\d{6}', "account_id")
        result = guard.check_input("My account is ACCT-123456")
        assert not result.compliant
        assert "account_id" in result.sensitive_categories


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure(self):
        guard = PrivacyGuard()
        guard.check_input("Hello")
        guard.check_input("Email: a@b.com")
        guard.check_output("SSN: 111-22-3333")
        report = guard.report()
        assert report.checks_performed == 3
        assert report.violations_found == 2
        assert isinstance(report.common_categories, list)
        assert isinstance(report.purposes_used, dict)

    def test_compliance_rate(self):
        guard = PrivacyGuard()
        guard.check_input("clean text")
        guard.check_input("clean again")
        guard.check_input("has user@test.com PII")
        guard.check_input("also has 555-123-4567")
        report = guard.report()
        assert report.compliance_rate == pytest.approx(0.5)

    def test_report_empty(self):
        guard = PrivacyGuard()
        report = guard.report()
        assert report.checks_performed == 0
        assert report.compliance_rate == 1.0

    def test_clear_stats(self):
        guard = PrivacyGuard()
        guard.check_input("user@test.com")
        guard.clear_stats()
        report = guard.report()
        assert report.checks_performed == 0
        assert report.violations_found == 0
        assert report.compliance_rate == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdge:
    def test_empty_text(self):
        guard = PrivacyGuard()
        input_result = guard.check_input("")
        assert input_result.compliant
        output_result = guard.check_output("")
        assert output_result.compliant
        min_result = guard.minimize("")
        assert min_result.minimized == ""
        assert min_result.items_removed == 0
