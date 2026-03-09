"""Tests for environment security checker."""

import pytest
from sentinel.env_security import EnvSecurityChecker, EnvSecurityReport, SecurityIssue


# ---------------------------------------------------------------------------
# Exposed secrets
# ---------------------------------------------------------------------------

class TestExposedSecrets:
    def test_detect_api_key(self):
        c = EnvSecurityChecker()
        report = c.check_env({"API_KEY": "sk-123456"})
        assert any(i.category == "exposed_secret" for i in report.issues)
        assert any(i.severity == "critical" for i in report.issues)

    def test_detect_aws_secret(self):
        c = EnvSecurityChecker()
        report = c.check_env({"AWS_SECRET_ACCESS_KEY": "abc123"})
        assert any("AWS" in i.message for i in report.issues)

    def test_detect_database_password(self):
        c = EnvSecurityChecker()
        report = c.check_env({"DATABASE_PASSWORD": "mypass"})
        assert any("Database" in i.message for i in report.issues)

    def test_detect_anthropic_key(self):
        c = EnvSecurityChecker()
        report = c.check_env({"ANTHROPIC_API_KEY": "sk-ant-123"})
        assert any("AI provider" in i.message for i in report.issues)

    def test_empty_value_ignored(self):
        c = EnvSecurityChecker()
        report = c.check_env({"API_KEY": ""})
        secret_issues = [i for i in report.issues if i.category == "exposed_secret"]
        assert len(secret_issues) == 0

    def test_safe_env(self):
        c = EnvSecurityChecker()
        report = c.check_env({"PATH": "/usr/bin", "HOME": "/home/user"})
        secret_issues = [i for i in report.issues if i.category == "exposed_secret"]
        assert len(secret_issues) == 0


# ---------------------------------------------------------------------------
# Insecure defaults
# ---------------------------------------------------------------------------

class TestInsecureDefaults:
    def test_password_default(self):
        c = EnvSecurityChecker()
        report = c.check_env({"MY_VAR": "password"})
        assert any(i.category == "insecure_default" for i in report.issues)

    def test_changeme_default(self):
        c = EnvSecurityChecker()
        report = c.check_env({"SECRET": "changeme"})
        assert any(i.category == "insecure_default" for i in report.issues)

    def test_strong_value_ok(self):
        c = EnvSecurityChecker()
        report = c.check_env({"MY_VAR": "x9f2k4m8n1p3q7r5"})
        default_issues = [i for i in report.issues if i.category == "insecure_default"]
        assert len(default_issues) == 0


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------

class TestDebugMode:
    def test_debug_true(self):
        c = EnvSecurityChecker()
        report = c.check_env({"DEBUG": "true"})
        assert any(i.category == "debug_mode" for i in report.issues)

    def test_node_env_development(self):
        c = EnvSecurityChecker()
        report = c.check_env({"NODE_ENV": "development"})
        assert any(i.category == "debug_mode" for i in report.issues)

    def test_production_ok(self):
        c = EnvSecurityChecker()
        report = c.check_env({"NODE_ENV": "production"})
        debug_issues = [i for i in report.issues if i.category == "debug_mode"]
        assert len(debug_issues) == 0


# ---------------------------------------------------------------------------
# CORS misconfiguration
# ---------------------------------------------------------------------------

class TestCORS:
    def test_cors_wildcard(self):
        c = EnvSecurityChecker()
        report = c.check_env({"CORS_ORIGIN": "*"})
        assert any(i.category == "misconfiguration" for i in report.issues)

    def test_cors_specific_origin(self):
        c = EnvSecurityChecker()
        report = c.check_env({"CORS_ORIGIN": "https://example.com"})
        misconfig_issues = [i for i in report.issues if i.category == "misconfiguration"]
        assert len(misconfig_issues) == 0


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------

class TestCustomPatterns:
    def test_custom_pattern(self):
        c = EnvSecurityChecker(custom_patterns=[
            (r"(?i)^LEGACY_", "Legacy variable detected"),
        ])
        report = c.check_env({"LEGACY_DB_HOST": "localhost"})
        assert any(i.category == "custom" for i in report.issues)


# ---------------------------------------------------------------------------
# Check value
# ---------------------------------------------------------------------------

class TestCheckValue:
    def test_check_value_secret(self):
        c = EnvSecurityChecker()
        issues = c.check_value("API_KEY", "sk-123")
        assert len(issues) > 0
        assert issues[0].severity == "critical"

    def test_check_value_safe(self):
        c = EnvSecurityChecker()
        issues = c.check_value("APP_NAME", "my-app")
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure(self):
        c = EnvSecurityChecker()
        report = c.check_env({"PATH": "/usr/bin"})
        assert isinstance(report, EnvSecurityReport)
        assert report.checks_run >= 4
        assert report.score >= 0.0
        assert report.score <= 1.0

    def test_secure_report(self):
        c = EnvSecurityChecker()
        report = c.check_env({"PATH": "/usr/bin", "HOME": "/home"})
        assert report.secure
        assert report.score > 0.5

    def test_insecure_report(self):
        c = EnvSecurityChecker()
        report = c.check_env({"API_KEY": "sk-123", "DEBUG": "true"})
        assert not report.secure

    def test_empty_env(self):
        c = EnvSecurityChecker()
        report = c.check_env({})
        assert report.secure
        assert report.score == 1.0
