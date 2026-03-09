"""Tests for dependency security guard."""

import pytest
from sentinel.dependency_guard import (
    DependencyGuard,
    DependencyReport,
    PackageCheck,
    TyposquatMatch,
)


# ---------------------------------------------------------------------------
# Package checks
# ---------------------------------------------------------------------------


class TestPackageCheck:
    def test_safe_package(self):
        guard = DependencyGuard()
        result = guard.check_package("requests", "2.28.0")
        assert result.risk_level == "low"
        assert result.blocked is False
        assert result.issues == []

    def test_blocked_package(self):
        guard = DependencyGuard()
        result = guard.check_package("keylogger-pro")
        assert result.blocked is True
        assert result.risk_level == "critical"
        assert any("malicious" in issue.lower() for issue in result.issues)

    def test_short_name_warning(self):
        guard = DependencyGuard()
        result = guard.check_package("ab")
        assert result.risk_level == "medium"
        assert any("short" in issue.lower() for issue in result.issues)

    def test_suspicious_prefix(self):
        guard = DependencyGuard()
        result = guard.check_package("python-requests")
        assert result.risk_level == "medium"
        assert any("prefix" in issue.lower() for issue in result.issues)


# ---------------------------------------------------------------------------
# Requirements parsing and checking
# ---------------------------------------------------------------------------


class TestRequirements:
    def test_check_requirements(self):
        guard = DependencyGuard()
        report = guard.check_requirements(["flask>=2.0", "requests==2.28.0"])
        assert report.total == 2
        assert report.safe is True

    def test_parse_version(self):
        guard = DependencyGuard()
        report = guard.check_requirements(["numpy>=1.21.0"])
        assert report.packages[0].name == "numpy"
        assert report.packages[0].version == ">=1.21.0"

    def test_mixed_safety(self):
        guard = DependencyGuard()
        report = guard.check_requirements([
            "flask>=2.0",
            "exploit-toolkit==1.0",
        ])
        assert report.total == 2
        assert report.safe is False
        assert report.blocked_count == 1
        assert report.high_risk_count >= 1

    def test_no_version_specifier(self):
        guard = DependencyGuard()
        report = guard.check_requirements(["flask"])
        assert report.packages[0].name == "flask"
        assert report.packages[0].version == ""


# ---------------------------------------------------------------------------
# Typosquat detection
# ---------------------------------------------------------------------------


class TestTyposquat:
    def test_similar_names(self):
        guard = DependencyGuard()
        known = ["requests", "flask", "numpy"]
        matches = guard.detect_typosquat("requets", known)
        assert len(matches) >= 1
        assert matches[0].original == "requests"
        assert matches[0].similarity > 0.7

    def test_exact_match_no_typosquat(self):
        guard = DependencyGuard()
        known = ["requests", "flask"]
        matches = guard.detect_typosquat("requests", known)
        assert len(matches) == 0

    def test_different_names(self):
        guard = DependencyGuard()
        known = ["requests", "flask", "numpy"]
        matches = guard.detect_typosquat("zzzzzzzzz", known)
        assert len(matches) == 0

    def test_typosquat_sorted_by_similarity(self):
        guard = DependencyGuard()
        known = ["requests", "requester"]
        matches = guard.detect_typosquat("requets", known)
        if len(matches) >= 2:
            assert matches[0].similarity >= matches[1].similarity

    def test_single_char_swap(self):
        guard = DependencyGuard()
        known = ["requests"]
        matches = guard.detect_typosquat("reqeusts", known)
        assert len(matches) >= 1
        assert "swap" in matches[0].reason.lower() or "character" in matches[0].reason.lower()


# ---------------------------------------------------------------------------
# Blocklist management
# ---------------------------------------------------------------------------


class TestBlocklist:
    def test_default_blocklist(self):
        guard = DependencyGuard()
        assert guard.is_blocked("keylogger-lib")
        assert guard.is_blocked("backdoor-tool")
        assert not guard.is_blocked("flask")

    def test_add_blocklist(self):
        guard = DependencyGuard()
        assert not guard.is_blocked("bad-package")
        guard.add_blocklist(["bad-package"])
        assert guard.is_blocked("bad-package")

    def test_is_blocked(self):
        guard = DependencyGuard()
        guard.add_blocklist(["evil-lib", "dangerous-pkg"])
        assert guard.is_blocked("evil-lib")
        assert guard.is_blocked("dangerous-pkg")
        assert not guard.is_blocked("safe-package")

    def test_blocklist_case_insensitive(self):
        guard = DependencyGuard()
        guard.add_blocklist(["Evil-Lib"])
        assert guard.is_blocked("evil-lib")
        assert guard.is_blocked("EVIL-LIB")


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_structure(self):
        guard = DependencyGuard()
        report = guard.check_requirements(["flask>=2.0"])
        assert isinstance(report, DependencyReport)
        assert isinstance(report.packages, list)
        assert isinstance(report.packages[0], PackageCheck)
        assert report.total == 1

    def test_all_safe(self):
        guard = DependencyGuard()
        report = guard.check_requirements(["flask", "django", "requests"])
        assert report.safe is True
        assert report.high_risk_count == 0
        assert report.blocked_count == 0
        assert report.total == 3

    def test_has_blocked(self):
        guard = DependencyGuard()
        guard.add_blocklist(["bad-pkg"])
        report = guard.check_requirements(["flask", "bad-pkg"])
        assert report.safe is False
        assert report.blocked_count == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCase:
    def test_empty_requirements(self):
        guard = DependencyGuard()
        report = guard.check_requirements([])
        assert report.total == 0
        assert report.safe is True
        assert report.packages == []

    def test_empty_name(self):
        guard = DependencyGuard()
        result = guard.check_package("")
        assert result.risk_level == "critical"
        assert result.blocked is True

    def test_whitespace_only_name(self):
        guard = DependencyGuard()
        result = guard.check_package("   ")
        assert result.risk_level == "critical"
        assert result.blocked is True

    def test_version_with_extras(self):
        guard = DependencyGuard()
        report = guard.check_requirements(["requests>=2.0"])
        assert report.packages[0].name == "requests"
        assert report.packages[0].version == ">=2.0"
