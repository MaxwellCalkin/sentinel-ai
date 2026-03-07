"""Tests for the blocked terms scanner."""

from sentinel.scanners.blocked_terms import BlockedTermsScanner
from sentinel.core import RiskLevel


class TestBlockedTermsScanner:
    def test_basic_detection(self):
        scanner = BlockedTermsScanner(terms=["ProjectX", "secret_code"])
        findings = scanner.scan("Let's discuss ProjectX tomorrow")
        assert len(findings) == 1
        assert findings[0].metadata["term"] == "ProjectX"

    def test_case_insensitive(self):
        scanner = BlockedTermsScanner(terms=["secret"])
        findings = scanner.scan("This is a SECRET document")
        assert len(findings) == 1

    def test_case_sensitive(self):
        scanner = BlockedTermsScanner(terms=["Secret"], case_sensitive=True)
        findings = scanner.scan("This is a secret document")
        assert len(findings) == 0
        findings = scanner.scan("This is a Secret document")
        assert len(findings) == 1

    def test_multiple_matches(self):
        scanner = BlockedTermsScanner(terms=["foo"])
        findings = scanner.scan("foo bar foo baz foo")
        assert len(findings) == 3

    def test_no_terms_no_findings(self):
        scanner = BlockedTermsScanner(terms=[])
        findings = scanner.scan("anything here")
        assert len(findings) == 0

    def test_custom_risk_level(self):
        scanner = BlockedTermsScanner(terms=["test"], risk=RiskLevel.CRITICAL)
        findings = scanner.scan("this is a test")
        assert findings[0].risk == RiskLevel.CRITICAL
