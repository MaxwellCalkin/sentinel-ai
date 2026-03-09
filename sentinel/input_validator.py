"""Structured input validation for LLM prompts.

Validate user inputs before sending to LLMs: type checking,
format validation, length limits, encoding safety, and
injection pattern detection.

Usage:
    from sentinel.input_validator import InputValidator

    v = InputValidator()
    v.max_length(1000).no_code_injection().safe_encoding()
    result = v.validate("User input here")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class InputIssue:
    """A single validation issue."""
    rule: str
    message: str
    severity: str  # "error" or "warning"


@dataclass
class InputValidationResult:
    """Result of input validation."""
    valid: bool
    issues: list[InputIssue]
    sanitized: str | None = None  # Cleaned input if available
    checks_run: int = 0

    @property
    def errors(self) -> list[InputIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[InputIssue]:
        return [i for i in self.issues if i.severity == "warning"]


class InputValidator:
    """Fluent input validation for LLM prompts.

    Chain validation rules and run them against user input.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[str, Callable[[str], InputIssue | None]]] = []

    def max_length(self, chars: int) -> "InputValidator":
        """Reject input exceeding max characters."""
        def check(text: str) -> InputIssue | None:
            if len(text) > chars:
                return InputIssue("max_length", f"Input exceeds {chars} characters ({len(text)})", "error")
            return None
        self._rules.append(("max_length", check))
        return self

    def min_length(self, chars: int) -> "InputValidator":
        """Reject input below min characters."""
        def check(text: str) -> InputIssue | None:
            if len(text) < chars:
                return InputIssue("min_length", f"Input below {chars} characters ({len(text)})", "error")
            return None
        self._rules.append(("min_length", check))
        return self

    def no_empty(self) -> "InputValidator":
        """Reject empty or whitespace-only input."""
        def check(text: str) -> InputIssue | None:
            if not text.strip():
                return InputIssue("no_empty", "Input is empty or whitespace-only", "error")
            return None
        self._rules.append(("no_empty", check))
        return self

    def no_code_injection(self) -> "InputValidator":
        """Detect common code injection patterns."""
        patterns = [
            (r'<script\b', "Script tag detected"),
            (r'javascript:', "JavaScript URI detected"),
            (r'on\w+\s*=', "Event handler detected"),
            (r'(?:eval|exec|import)\s*\(', "Code execution pattern detected"),
            (r'(?:DROP|DELETE|INSERT|UPDATE)\s+(?:TABLE|FROM|INTO)', "SQL injection pattern"),
        ]
        compiled = [(re.compile(p, re.IGNORECASE), msg) for p, msg in patterns]

        def check(text: str) -> InputIssue | None:
            for pattern, msg in compiled:
                if pattern.search(text):
                    return InputIssue("no_code_injection", msg, "error")
            return None
        self._rules.append(("no_code_injection", check))
        return self

    def safe_encoding(self) -> "InputValidator":
        """Detect unsafe Unicode characters."""
        def check(text: str) -> InputIssue | None:
            # Check for zero-width characters
            for char in text:
                cp = ord(char)
                if cp in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060):
                    return InputIssue("safe_encoding", "Zero-width character detected", "error")
                # Check for control characters (except common whitespace)
                if cp < 32 and cp not in (9, 10, 13):  # tab, newline, carriage return
                    return InputIssue("safe_encoding", f"Control character U+{cp:04X} detected", "error")
            return None
        self._rules.append(("safe_encoding", check))
        return self

    def allowed_chars(self, pattern: str) -> "InputValidator":
        """Only allow characters matching pattern."""
        compiled = re.compile(f'^[{pattern}]*$')

        def check(text: str) -> InputIssue | None:
            if not compiled.match(text):
                return InputIssue("allowed_chars", "Input contains disallowed characters", "error")
            return None
        self._rules.append(("allowed_chars", check))
        return self

    def blocked_words(self, words: list[str], case_sensitive: bool = False) -> "InputValidator":
        """Block specific words or phrases."""
        if not case_sensitive:
            words = [w.lower() for w in words]

        def check(text: str) -> InputIssue | None:
            t = text if case_sensitive else text.lower()
            for word in words:
                if word in t:
                    return InputIssue("blocked_words", f"Blocked word detected", "error")
            return None
        self._rules.append(("blocked_words", check))
        return self

    def max_lines(self, lines: int) -> "InputValidator":
        """Limit number of lines."""
        def check(text: str) -> InputIssue | None:
            count = text.count('\n') + 1
            if count > lines:
                return InputIssue("max_lines", f"Input exceeds {lines} lines ({count})", "error")
            return None
        self._rules.append(("max_lines", check))
        return self

    def regex_match(self, pattern: str, message: str = "") -> "InputValidator":
        """Require input to match a regex pattern."""
        compiled = re.compile(pattern)

        def check(text: str) -> InputIssue | None:
            if not compiled.search(text):
                return InputIssue("regex_match", message or f"Input doesn't match required pattern", "error")
            return None
        self._rules.append(("regex_match", check))
        return self

    def regex_deny(self, pattern: str, message: str = "") -> "InputValidator":
        """Block input matching a regex pattern."""
        compiled = re.compile(pattern, re.IGNORECASE)

        def check(text: str) -> InputIssue | None:
            if compiled.search(text):
                return InputIssue("regex_deny", message or "Input matches blocked pattern", "error")
            return None
        self._rules.append(("regex_deny", check))
        return self

    def custom(self, name: str, check_fn: Callable[[str], bool], message: str = "") -> "InputValidator":
        """Add a custom validation rule.

        Args:
            name: Rule name.
            check_fn: Function(text) -> True if valid.
            message: Error message if invalid.
        """
        def check(text: str) -> InputIssue | None:
            if not check_fn(text):
                return InputIssue(name, message or f"Custom check '{name}' failed", "error")
            return None
        self._rules.append((name, check))
        return self

    def validate(self, text: str) -> InputValidationResult:
        """Run all validation rules.

        Args:
            text: User input to validate.

        Returns:
            InputValidationResult with issues and validity.
        """
        issues: list[InputIssue] = []
        checks_run = 0

        for name, check_fn in self._rules:
            checks_run += 1
            try:
                issue = check_fn(text)
                if issue:
                    issues.append(issue)
            except Exception:
                continue

        return InputValidationResult(
            valid=len([i for i in issues if i.severity == "error"]) == 0,
            issues=issues,
            checks_run=checks_run,
        )

    def validate_batch(self, texts: list[str]) -> list[InputValidationResult]:
        """Validate multiple inputs."""
        return [self.validate(t) for t in texts]

    @property
    def rule_count(self) -> int:
        return len(self._rules)
