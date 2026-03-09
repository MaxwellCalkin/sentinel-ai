"""Configuration validator for LLM deployments.

Validate LLM API settings and deployment parameters for
security best practices and common misconfigurations.

Usage:
    from sentinel.config_validator import ConfigValidator

    validator = ConfigValidator()
    result = validator.validate({"model": "claude-sonnet", "temperature": 0.7})
    print(result.valid)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConfigIssue:
    """A single configuration issue found."""

    field: str
    severity: str  # "error", "warning", "info"
    message: str
    suggestion: str = ""


@dataclass
class ConfigResult:
    """Result of configuration validation."""

    valid: bool
    issues: list[ConfigIssue]
    warnings: int
    errors: int
    fields_checked: int


class ConfigValidator:
    """Validate LLM deployment configurations.

    Check temperature, max_tokens, API key handling, and other
    parameters for security and correctness issues.
    """

    def __init__(
        self,
        strict: bool = False,
        custom_rules: list[tuple[str, Any, str]] | None = None,
    ) -> None:
        """
        Args:
            strict: Promote all warnings to errors.
            custom_rules: List of (field, expected_value, message) tuples.
        """
        self._strict = strict
        self._custom_rules = custom_rules or []

    def validate(self, config: dict[str, Any]) -> ConfigResult:
        """Validate a single configuration dict.

        Args:
            config: Configuration parameters to validate.

        Returns:
            ConfigResult with findings.
        """
        issues: list[ConfigIssue] = []
        fields_checked = 0

        if "temperature" in config:
            fields_checked += 1
            temp = config["temperature"]
            if isinstance(temp, (int, float)):
                if temp < 0 or temp > 2:
                    issues.append(ConfigIssue(
                        field="temperature",
                        severity="error",
                        message=f"Temperature {temp} outside valid range [0, 2]",
                        suggestion="Set temperature between 0 and 2",
                    ))
                elif temp > 1.5:
                    issues.append(ConfigIssue(
                        field="temperature",
                        severity="warning",
                        message=f"High temperature ({temp}) may produce unreliable outputs",
                        suggestion="Use temperature <= 1.0 for production",
                    ))

        if "max_tokens" in config:
            fields_checked += 1
            mt = config["max_tokens"]
            if isinstance(mt, int):
                if mt <= 0:
                    issues.append(ConfigIssue(
                        field="max_tokens",
                        severity="error",
                        message="max_tokens must be positive",
                    ))
                elif mt > 100000:
                    issues.append(ConfigIssue(
                        field="max_tokens",
                        severity="warning",
                        message=f"Very high max_tokens ({mt}) increases cost and latency",
                        suggestion="Use a lower max_tokens unless needed",
                    ))

        if "model" in config:
            fields_checked += 1
            model = config["model"]
            if isinstance(model, str) and not model.strip():
                issues.append(ConfigIssue(
                    field="model",
                    severity="error",
                    message="Model name is empty",
                ))

        if "api_key" in config:
            fields_checked += 1
            key = config["api_key"]
            if isinstance(key, str) and key and not key.startswith("${") and not key.startswith("env:"):
                issues.append(ConfigIssue(
                    field="api_key",
                    severity="error",
                    message="API key appears hardcoded in config",
                    suggestion="Use environment variables or a secrets manager",
                ))

        if "system_prompt" in config:
            fields_checked += 1
            sp = config["system_prompt"]
            if isinstance(sp, str) and len(sp) > 10000:
                issues.append(ConfigIssue(
                    field="system_prompt",
                    severity="warning",
                    message=f"Very long system prompt ({len(sp)} chars) wastes tokens",
                    suggestion="Keep system prompts concise",
                ))

        if "top_p" in config:
            fields_checked += 1
            tp = config["top_p"]
            if isinstance(tp, (int, float)) and (tp < 0 or tp > 1):
                issues.append(ConfigIssue(
                    field="top_p",
                    severity="error",
                    message=f"top_p {tp} outside valid range [0, 1]",
                ))

        if "timeout" in config:
            fields_checked += 1
            timeout = config["timeout"]
            if isinstance(timeout, (int, float)) and timeout < 1:
                issues.append(ConfigIssue(
                    field="timeout",
                    severity="warning",
                    message=f"Very short timeout ({timeout}s) may cause failures",
                    suggestion="Set timeout to at least 10 seconds",
                ))

        for field_name, expected, message in self._custom_rules:
            fields_checked += 1
            actual = config.get(field_name)
            if actual != expected:
                issues.append(ConfigIssue(
                    field=field_name,
                    severity="error" if self._strict else "warning",
                    message=message,
                ))

        if self._strict:
            for issue in issues:
                if issue.severity == "warning":
                    issue.severity = "error"

        errors = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")

        return ConfigResult(
            valid=errors == 0,
            issues=issues,
            warnings=warnings,
            errors=errors,
            fields_checked=fields_checked,
        )

    def validate_batch(self, configs: list[dict[str, Any]]) -> list[ConfigResult]:
        """Validate multiple configurations.

        Args:
            configs: List of configuration dicts.

        Returns:
            List of ConfigResult, one per config.
        """
        return [self.validate(c) for c in configs]
