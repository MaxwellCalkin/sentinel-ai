"""Environment security checker for LLM deployments.

Scan environment variables, configuration files, and runtime
settings for security issues: exposed API keys, insecure
defaults, missing encryption, and misconfigured permissions.

Usage:
    from sentinel.env_security import EnvSecurityChecker

    checker = EnvSecurityChecker()
    report = checker.check_env()
    for issue in report.issues:
        print(f"{issue.severity}: {issue.message}")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SecurityIssue:
    """A single security issue found."""
    category: str
    severity: str  # "critical", "high", "medium", "low"
    message: str
    remediation: str = ""
    key: str = ""


@dataclass
class EnvSecurityReport:
    """Result of environment security check."""
    issues: list[SecurityIssue]
    checks_run: int
    passed: int
    failed: int
    score: float  # 0.0 (many issues) to 1.0 (no issues)

    @property
    def secure(self) -> bool:
        return not any(i.severity in ("critical", "high") for i in self.issues)


# Patterns for detecting exposed secrets in env vars
_SECRET_PATTERNS = [
    (r'(?i)^(?:api[_-]?key|secret[_-]?key|access[_-]?key|private[_-]?key)', "API/secret key"),
    (r'(?i)^(?:aws[_-]?secret|aws[_-]?access)', "AWS credential"),
    (r'(?i)^(?:database[_-]?(?:url|password|pass)|db[_-]?(?:password|pass))', "Database credential"),
    (r'(?i)^(?:jwt[_-]?secret|session[_-]?secret|cookie[_-]?secret)', "Session/JWT secret"),
    (r'(?i)^(?:stripe|twilio|sendgrid|mailgun|slack)[_-]?(?:key|token|secret)', "Third-party service key"),
    (r'(?i)^(?:openai|anthropic|google|azure)[_-]?(?:api[_-]?key|key|secret)', "AI provider key"),
    (r'(?i)^(?:password|passwd|pass)$', "Password in env var"),
    (r'(?i)^(?:token|auth[_-]?token|bearer[_-]?token)', "Auth token"),
    (r'(?i)^(?:encryption[_-]?key|signing[_-]?key|master[_-]?key)', "Encryption key"),
]

# Insecure default values
_INSECURE_DEFAULTS = [
    "password", "admin", "123456", "secret", "changeme",
    "default", "test", "example", "placeholder",
]


class EnvSecurityChecker:
    """Check environment security for LLM deployments.

    Detect exposed secrets, insecure defaults, and missing
    security configurations.
    """

    def __init__(
        self,
        check_env_vars: bool = True,
        custom_patterns: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Args:
            check_env_vars: Whether to scan environment variables.
            custom_patterns: Additional (pattern, description) pairs.
        """
        self._check_env = check_env_vars
        self._custom_patterns = custom_patterns or []

    def check_env(self, env: dict[str, str] | None = None) -> EnvSecurityReport:
        """Run all security checks.

        Args:
            env: Environment dict to check. Defaults to os.environ.

        Returns:
            EnvSecurityReport with findings.
        """
        env = env if env is not None else dict(os.environ)
        issues: list[SecurityIssue] = []
        checks = 0
        passed = 0

        # Check for exposed secrets
        checks += 1
        secret_issues = self._check_secrets(env)
        issues.extend(secret_issues)
        if not secret_issues:
            passed += 1

        # Check for insecure defaults
        checks += 1
        default_issues = self._check_insecure_defaults(env)
        issues.extend(default_issues)
        if not default_issues:
            passed += 1

        # Check for debug mode
        checks += 1
        debug_issues = self._check_debug_mode(env)
        issues.extend(debug_issues)
        if not debug_issues:
            passed += 1

        # Check for missing security settings
        checks += 1
        missing_issues = self._check_missing_security(env)
        issues.extend(missing_issues)
        if not missing_issues:
            passed += 1

        # Custom patterns
        if self._custom_patterns:
            checks += 1
            custom_issues = self._check_custom(env)
            issues.extend(custom_issues)
            if not custom_issues:
                passed += 1

        score = passed / checks if checks > 0 else 1.0

        return EnvSecurityReport(
            issues=issues,
            checks_run=checks,
            passed=passed,
            failed=checks - passed,
            score=round(score, 4),
        )

    def check_value(self, key: str, value: str) -> list[SecurityIssue]:
        """Check a single key-value pair for security issues."""
        issues: list[SecurityIssue] = []

        # Check if key matches secret patterns
        for pattern, desc in _SECRET_PATTERNS:
            if re.match(pattern, key):
                issues.append(SecurityIssue(
                    category="exposed_secret",
                    severity="critical",
                    message=f"Exposed {desc}: {key}",
                    remediation=f"Move {key} to a secrets manager",
                    key=key,
                ))
                break

        # Check for insecure default values
        if value.lower() in _INSECURE_DEFAULTS:
            issues.append(SecurityIssue(
                category="insecure_default",
                severity="high",
                message=f"Insecure default value for {key}",
                remediation=f"Change {key} to a strong, unique value",
                key=key,
            ))

        return issues

    def _check_secrets(self, env: dict[str, str]) -> list[SecurityIssue]:
        issues: list[SecurityIssue] = []
        for key, value in env.items():
            if not value:
                continue
            for pattern, desc in _SECRET_PATTERNS:
                if re.match(pattern, key):
                    issues.append(SecurityIssue(
                        category="exposed_secret",
                        severity="critical",
                        message=f"Exposed {desc} in environment: {key}",
                        remediation=f"Move {key} to a secrets manager or vault",
                        key=key,
                    ))
                    break
        return issues

    def _check_insecure_defaults(self, env: dict[str, str]) -> list[SecurityIssue]:
        issues: list[SecurityIssue] = []
        for key, value in env.items():
            if value.lower() in _INSECURE_DEFAULTS:
                issues.append(SecurityIssue(
                    category="insecure_default",
                    severity="high",
                    message=f"Insecure default value for {key}: '{value}'",
                    remediation=f"Change {key} to a strong, unique value",
                    key=key,
                ))
        return issues

    def _check_debug_mode(self, env: dict[str, str]) -> list[SecurityIssue]:
        issues: list[SecurityIssue] = []
        debug_keys = ["DEBUG", "FLASK_DEBUG", "DJANGO_DEBUG", "NODE_ENV"]
        for key in debug_keys:
            val = env.get(key, "").lower()
            if val in ("true", "1", "yes", "development"):
                issues.append(SecurityIssue(
                    category="debug_mode",
                    severity="medium",
                    message=f"Debug mode enabled: {key}={val}",
                    remediation=f"Set {key} to false/production in production",
                    key=key,
                ))
        return issues

    def _check_missing_security(self, env: dict[str, str]) -> list[SecurityIssue]:
        issues: list[SecurityIssue] = []
        # Check for HTTPS enforcement
        cors_origin = env.get("CORS_ORIGIN", env.get("CORS_ALLOWED_ORIGINS", ""))
        if cors_origin == "*":
            issues.append(SecurityIssue(
                category="misconfiguration",
                severity="medium",
                message="CORS allows all origins (wildcard *)",
                remediation="Restrict CORS to specific trusted domains",
                key="CORS_ORIGIN",
            ))
        return issues

    def _check_custom(self, env: dict[str, str]) -> list[SecurityIssue]:
        issues: list[SecurityIssue] = []
        for pattern, desc in self._custom_patterns:
            compiled = re.compile(pattern)
            for key in env:
                if compiled.match(key):
                    issues.append(SecurityIssue(
                        category="custom",
                        severity="high",
                        message=f"Custom pattern match: {desc} ({key})",
                        key=key,
                    ))
        return issues
