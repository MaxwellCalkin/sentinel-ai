"""Output constraint enforcement for LLM responses.

Enforce output requirements: max length, format constraints,
banned phrases, required sections, and more. Ensures LLM
outputs meet application-specific requirements.

Usage:
    from sentinel.output_guard import OutputGuard

    guard = OutputGuard()
    guard.max_length(500)
    guard.ban_phrases(["I cannot", "As an AI"])
    guard.require_format("json")

    result = guard.check("Some LLM output...")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class OutputViolation:
    """A single output constraint violation."""
    rule: str
    message: str
    severity: str = "error"  # error, warning


@dataclass
class OutputCheckResult:
    """Result of output constraint checking."""
    text: str
    passed: bool
    violations: list[OutputViolation] = field(default_factory=list)

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def errors(self) -> list[OutputViolation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[OutputViolation]:
        return [v for v in self.violations if v.severity == "warning"]


class OutputGuard:
    """Enforce output constraints on LLM responses.

    Configure rules for max/min length, banned phrases,
    required content, format validation, and custom checks.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[str, Callable[[str], OutputViolation | None]]] = []

    def max_length(self, chars: int) -> OutputGuard:
        """Enforce maximum character length."""
        def check(text: str) -> OutputViolation | None:
            if len(text) > chars:
                return OutputViolation(
                    rule="max_length",
                    message=f"Output length {len(text)} exceeds max {chars}",
                )
            return None
        self._rules.append(("max_length", check))
        return self

    def min_length(self, chars: int) -> OutputGuard:
        """Enforce minimum character length."""
        def check(text: str) -> OutputViolation | None:
            if len(text.strip()) < chars:
                return OutputViolation(
                    rule="min_length",
                    message=f"Output length {len(text.strip())} below min {chars}",
                )
            return None
        self._rules.append(("min_length", check))
        return self

    def max_words(self, count: int) -> OutputGuard:
        """Enforce maximum word count."""
        def check(text: str) -> OutputViolation | None:
            words = len(text.split())
            if words > count:
                return OutputViolation(
                    rule="max_words",
                    message=f"Word count {words} exceeds max {count}",
                )
            return None
        self._rules.append(("max_words", check))
        return self

    def ban_phrases(self, phrases: list[str], case_sensitive: bool = False) -> OutputGuard:
        """Ban specific phrases from output."""
        def check(text: str) -> OutputViolation | None:
            check_text = text if case_sensitive else text.lower()
            for phrase in phrases:
                check_phrase = phrase if case_sensitive else phrase.lower()
                if check_phrase in check_text:
                    return OutputViolation(
                        rule="banned_phrase",
                        message=f"Contains banned phrase: '{phrase}'",
                    )
            return None
        self._rules.append(("ban_phrases", check))
        return self

    def require_phrases(self, phrases: list[str], case_sensitive: bool = False) -> OutputGuard:
        """Require specific phrases in output."""
        def check(text: str) -> OutputViolation | None:
            check_text = text if case_sensitive else text.lower()
            for phrase in phrases:
                check_phrase = phrase if case_sensitive else phrase.lower()
                if check_phrase not in check_text:
                    return OutputViolation(
                        rule="required_phrase",
                        message=f"Missing required phrase: '{phrase}'",
                    )
            return None
        self._rules.append(("require_phrases", check))
        return self

    def require_format(self, fmt: str) -> OutputGuard:
        """Require output to be in a specific format (json, xml, markdown)."""
        def check(text: str) -> OutputViolation | None:
            stripped = text.strip()
            if fmt == "json":
                try:
                    json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    # Try extracting from code block
                    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.S)
                    if match:
                        try:
                            json.loads(match.group(1).strip())
                            return None
                        except (json.JSONDecodeError, ValueError):
                            pass
                    return OutputViolation(
                        rule="format",
                        message="Output is not valid JSON",
                    )
            elif fmt == "xml":
                if not re.search(r"<\w+[^>]*>.*</\w+>", stripped, re.S):
                    return OutputViolation(
                        rule="format",
                        message="Output does not contain valid XML elements",
                    )
            elif fmt == "markdown":
                if not re.search(r"(?:^#{1,6}\s|^\*|\^-\s|\d+\.\s|```)", stripped, re.M):
                    return OutputViolation(
                        rule="format",
                        message="Output does not appear to be Markdown",
                        severity="warning",
                    )
            return None
        self._rules.append(("format", check))
        return self

    def no_empty(self) -> OutputGuard:
        """Disallow empty or whitespace-only output."""
        def check(text: str) -> OutputViolation | None:
            if not text.strip():
                return OutputViolation(
                    rule="no_empty",
                    message="Output is empty or whitespace-only",
                )
            return None
        self._rules.append(("no_empty", check))
        return self

    def regex_match(self, pattern: str, description: str = "") -> OutputGuard:
        """Require output to match a regex pattern."""
        compiled = re.compile(pattern)
        def check(text: str) -> OutputViolation | None:
            if not compiled.search(text):
                return OutputViolation(
                    rule="regex_match",
                    message=f"Output does not match pattern: {description or pattern}",
                )
            return None
        self._rules.append(("regex_match", check))
        return self

    def regex_deny(self, pattern: str, description: str = "") -> OutputGuard:
        """Deny output matching a regex pattern."""
        compiled = re.compile(pattern)
        def check(text: str) -> OutputViolation | None:
            match = compiled.search(text)
            if match:
                return OutputViolation(
                    rule="regex_deny",
                    message=f"Output matches denied pattern: {description or pattern}",
                )
            return None
        self._rules.append(("regex_deny", check))
        return self

    def custom(self, name: str, fn: Callable[[str], str | None]) -> OutputGuard:
        """Add a custom rule. fn(text) -> error message or None."""
        def check(text: str) -> OutputViolation | None:
            result = fn(text)
            if result:
                return OutputViolation(rule=name, message=result)
            return None
        self._rules.append((name, check))
        return self

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def check(self, text: str) -> OutputCheckResult:
        """Check output against all rules.

        Returns:
            OutputCheckResult with pass/fail status and violations.
        """
        violations: list[OutputViolation] = []
        for _, rule_fn in self._rules:
            violation = rule_fn(text)
            if violation:
                violations.append(violation)

        return OutputCheckResult(
            text=text,
            passed=len([v for v in violations if v.severity == "error"]) == 0,
            violations=violations,
        )

    def check_batch(self, texts: list[str]) -> list[OutputCheckResult]:
        """Check multiple outputs."""
        return [self.check(t) for t in texts]
