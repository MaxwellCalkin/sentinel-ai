"""Comprehensive prompt validation engine for LLM applications.

Validates prompts before they are sent to LLMs: structure, length,
encoding, content safety, format compliance, and custom rules.

Usage:
    from sentinel.prompt_validator import PromptValidator, ValidationRule

    validator = PromptValidator(max_length=5000, blocked_words=["hack"])
    report = validator.validate("Tell me about cybersecurity")
    print(report.is_valid, report.score)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


VALID_CHECK_TYPES = frozenset({
    "length",
    "encoding",
    "pattern",
    "blocked_word",
    "required_field",
    "format",
})

SEVERITY_SCORE_PENALTIES = {
    "error": 0.2,
    "warning": 0.1,
    "info": 0.05,
}


@dataclass
class ValidationRule:
    """A single validation rule to apply to prompts."""

    name: str
    check: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.check not in VALID_CHECK_TYPES:
            raise ValueError(
                f"Invalid check type '{self.check}'. "
                f"Must be one of: {', '.join(sorted(VALID_CHECK_TYPES))}"
            )


@dataclass
class ValidationIssue:
    """A single issue found during prompt validation."""

    rule_name: str
    severity: str
    message: str


@dataclass
class ValidationReport:
    """Complete result of validating a prompt."""

    prompt: str
    is_valid: bool
    issues: list[ValidationIssue]
    score: float
    checked_at: float


@dataclass
class ValidatorStats:
    """Cumulative statistics across all validations."""

    total_validated: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    issues_by_severity: dict[str, int] = field(default_factory=dict)


class PromptValidator:
    """Validates prompts before they are sent to LLMs.

    Runs built-in checks (length, encoding, blocked words, repetition)
    plus any custom ValidationRules added via add_rule().
    """

    def __init__(
        self,
        max_length: int = 10000,
        min_length: int = 1,
        blocked_words: list[str] | None = None,
    ) -> None:
        self._max_length = max_length
        self._min_length = min_length
        self._blocked_words = [w.lower() for w in (blocked_words or [])]
        self._custom_rules: list[ValidationRule] = []
        self._stats = ValidatorStats()

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self._custom_rules.append(rule)

    def validate(self, prompt: str) -> ValidationReport:
        """Run all built-in and custom rules against a prompt."""
        issues: list[ValidationIssue] = []

        self._check_empty(prompt, issues)
        self._check_length(prompt, issues)
        self._check_encoding(prompt, issues)
        self._check_blocked_words(prompt, issues)
        self._check_excessive_repetition(prompt, issues)
        self._run_custom_rules(prompt, issues)

        report = self._build_report(prompt, issues)
        self._update_stats(report)
        return report

    def validate_batch(self, prompts: list[str]) -> list[ValidationReport]:
        """Validate multiple prompts."""
        return [self.validate(p) for p in prompts]

    def stats(self) -> ValidatorStats:
        """Return cumulative validation statistics."""
        return ValidatorStats(
            total_validated=self._stats.total_validated,
            passed=self._stats.passed,
            failed=self._stats.failed,
            pass_rate=self._stats.pass_rate,
            issues_by_severity=dict(self._stats.issues_by_severity),
        )

    # ------------------------------------------------------------------
    # Built-in checks
    # ------------------------------------------------------------------

    def _check_empty(self, prompt: str, issues: list[ValidationIssue]) -> None:
        if not prompt or not prompt.strip():
            issues.append(ValidationIssue(
                rule_name="empty_check",
                severity="error",
                message="Prompt is empty or contains only whitespace",
            ))

    def _check_length(self, prompt: str, issues: list[ValidationIssue]) -> None:
        if len(prompt) < self._min_length:
            issues.append(ValidationIssue(
                rule_name="min_length",
                severity="error",
                message=f"Prompt length {len(prompt)} is below minimum {self._min_length}",
            ))
        if len(prompt) > self._max_length:
            issues.append(ValidationIssue(
                rule_name="max_length",
                severity="error",
                message=f"Prompt length {len(prompt)} exceeds maximum {self._max_length}",
            ))

    def _check_encoding(self, prompt: str, issues: list[ValidationIssue]) -> None:
        for char in prompt:
            code_point = ord(char)
            if _is_control_character(code_point):
                issues.append(ValidationIssue(
                    rule_name="encoding_check",
                    severity="error",
                    message=f"Control character U+{code_point:04X} detected",
                ))
                return
            if _is_zero_width_character(code_point):
                issues.append(ValidationIssue(
                    rule_name="encoding_check",
                    severity="warning",
                    message="Zero-width or invisible character detected",
                ))
                return

    def _check_blocked_words(self, prompt: str, issues: list[ValidationIssue]) -> None:
        prompt_lower = prompt.lower()
        for word in self._blocked_words:
            if word in prompt_lower:
                issues.append(ValidationIssue(
                    rule_name="blocked_word",
                    severity="error",
                    message=f"Blocked word detected: '{word}'",
                ))

    def _check_excessive_repetition(
        self, prompt: str, issues: list[ValidationIssue]
    ) -> None:
        word_counts: dict[str, int] = {}
        for word in prompt.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            if count >= 10:
                issues.append(ValidationIssue(
                    rule_name="repetition_check",
                    severity="warning",
                    message=f"Word '{word}' repeated {count} times",
                ))
                return

    # ------------------------------------------------------------------
    # Custom rule execution
    # ------------------------------------------------------------------

    def _run_custom_rules(
        self, prompt: str, issues: list[ValidationIssue]
    ) -> None:
        for rule in self._custom_rules:
            if rule.check == "pattern":
                self._apply_pattern_rule(prompt, rule, issues)
            elif rule.check == "length":
                self._apply_length_rule(prompt, rule, issues)
            elif rule.check == "encoding":
                self._apply_encoding_rule(prompt, rule, issues)
            elif rule.check == "blocked_word":
                self._apply_blocked_word_rule(prompt, rule, issues)
            elif rule.check == "required_field":
                self._apply_required_field_rule(prompt, rule, issues)
            elif rule.check == "format":
                self._apply_format_rule(prompt, rule, issues)

    def _apply_pattern_rule(
        self, prompt: str, rule: ValidationRule, issues: list[ValidationIssue]
    ) -> None:
        pattern = rule.params.get("regex", "")
        if not pattern:
            return
        severity = rule.params.get("severity", "warning")
        message = rule.params.get("message", f"Pattern '{pattern}' matched")
        if re.search(pattern, prompt, re.IGNORECASE):
            issues.append(ValidationIssue(
                rule_name=rule.name,
                severity=severity,
                message=message,
            ))

    def _apply_length_rule(
        self, prompt: str, rule: ValidationRule, issues: list[ValidationIssue]
    ) -> None:
        min_len = rule.params.get("min", 0)
        max_len = rule.params.get("max", float("inf"))
        if len(prompt) < min_len:
            issues.append(ValidationIssue(
                rule_name=rule.name,
                severity="error",
                message=f"Prompt length {len(prompt)} below rule minimum {min_len}",
            ))
        if len(prompt) > max_len:
            issues.append(ValidationIssue(
                rule_name=rule.name,
                severity="error",
                message=f"Prompt length {len(prompt)} exceeds rule maximum {max_len}",
            ))

    def _apply_encoding_rule(
        self, prompt: str, rule: ValidationRule, issues: list[ValidationIssue]
    ) -> None:
        allowed_charset = rule.params.get("charset", "utf-8")
        try:
            prompt.encode(allowed_charset)
        except (UnicodeEncodeError, LookupError):
            issues.append(ValidationIssue(
                rule_name=rule.name,
                severity="error",
                message=f"Prompt contains characters not valid in {allowed_charset}",
            ))

    def _apply_blocked_word_rule(
        self, prompt: str, rule: ValidationRule, issues: list[ValidationIssue]
    ) -> None:
        words = rule.params.get("words", [])
        prompt_lower = prompt.lower()
        for word in words:
            if word.lower() in prompt_lower:
                issues.append(ValidationIssue(
                    rule_name=rule.name,
                    severity="error",
                    message=f"Blocked word detected: '{word}'",
                ))

    def _apply_required_field_rule(
        self, prompt: str, rule: ValidationRule, issues: list[ValidationIssue]
    ) -> None:
        fields = rule.params.get("fields", [])
        for field_name in fields:
            if field_name not in prompt:
                issues.append(ValidationIssue(
                    rule_name=rule.name,
                    severity="error",
                    message=f"Required field '{field_name}' not found in prompt",
                ))

    def _apply_format_rule(
        self, prompt: str, rule: ValidationRule, issues: list[ValidationIssue]
    ) -> None:
        expected_format = rule.params.get("format", "")
        if expected_format == "json" and not _looks_like_json(prompt):
            issues.append(ValidationIssue(
                rule_name=rule.name,
                severity="error",
                message="Prompt does not appear to be valid JSON",
            ))
        elif expected_format == "markdown" and not _looks_like_markdown(prompt):
            issues.append(ValidationIssue(
                rule_name=rule.name,
                severity="info",
                message="Prompt does not appear to contain markdown formatting",
            ))

    # ------------------------------------------------------------------
    # Reporting and stats
    # ------------------------------------------------------------------

    def _build_report(
        self, prompt: str, issues: list[ValidationIssue]
    ) -> ValidationReport:
        has_errors = any(issue.severity == "error" for issue in issues)
        score = _calculate_score(issues)
        return ValidationReport(
            prompt=prompt,
            is_valid=not has_errors,
            issues=issues,
            score=score,
            checked_at=time.time(),
        )

    def _update_stats(self, report: ValidationReport) -> None:
        self._stats.total_validated += 1
        if report.is_valid:
            self._stats.passed += 1
        else:
            self._stats.failed += 1
        self._stats.pass_rate = self._stats.passed / self._stats.total_validated

        for issue in report.issues:
            severity = issue.severity
            self._stats.issues_by_severity[severity] = (
                self._stats.issues_by_severity.get(severity, 0) + 1
            )


# ----------------------------------------------------------------------
# Pure helper functions
# ----------------------------------------------------------------------

def _is_control_character(code_point: int) -> bool:
    """Return True for ASCII control chars except tab, newline, carriage return."""
    return code_point < 32 and code_point not in (9, 10, 13)


def _is_zero_width_character(code_point: int) -> bool:
    """Return True for zero-width and invisible Unicode characters."""
    return code_point in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060)


def _calculate_score(issues: list[ValidationIssue]) -> float:
    """Compute a 0.0-1.0 score: 1.0 minus penalties per issue severity."""
    penalty = sum(
        SEVERITY_SCORE_PENALTIES.get(issue.severity, 0.0) for issue in issues
    )
    return max(0.0, round(1.0 - penalty, 2))


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    )


def _looks_like_markdown(text: str) -> bool:
    markdown_indicators = ["# ", "## ", "**", "- ", "* ", "```", "[", "]("]
    return any(indicator in text for indicator in markdown_indicators)
