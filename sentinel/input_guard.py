"""Composable input validation pipeline with declarative rules.

Build multi-step input validation chains using a fluent API.
Each rule has a configurable action: block, warn, or sanitize.
Supports batch validation and aggregate reporting.

Usage:
    from sentinel.input_guard import InputGuard

    guard = (InputGuard()
        .max_length(1000)
        .block_patterns([r"<script", r"DROP TABLE"])
        .require_language("en"))

    result = guard.validate("Hello, how are you?")
    assert result.passed
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GuardRule:
    """A single validation rule in the pipeline."""

    name: str
    check_fn: Callable[[str], str | None]
    action: str = "block"  # "block", "warn", or "sanitize"
    description: str = ""


@dataclass
class GuardResult:
    """Outcome of validating a single text input."""

    text: str
    passed: bool
    blocked: bool
    warnings: list[str] = field(default_factory=list)
    sanitized_text: str = ""
    rules_triggered: list[str] = field(default_factory=list)


@dataclass
class GuardReport:
    """Aggregate statistics from batch validation."""

    total_checked: int
    blocked_count: int
    warned_count: int
    pass_rate: float


# -- Language detection helpers (zero-dep, Latin-script only) ----------------

_LANG_WORDS: dict[str, set[str]] = {
    "en": {
        "the", "is", "and", "to", "of", "in", "that", "it", "for", "was",
        "are", "with", "this", "have", "from", "not", "but", "they", "his",
        "she", "will", "been", "would", "could", "their", "which", "about",
    },
    "fr": {
        "le", "la", "les", "de", "des", "du", "et", "est", "un", "une",
        "en", "que", "pas", "pour", "sur", "avec", "dans", "plus", "nous",
    },
    "es": {
        "el", "la", "los", "las", "de", "en", "que", "es", "por", "con",
        "una", "para", "como", "pero", "sus", "del", "este", "todo",
    },
    "de": {
        "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem",
        "nicht", "sich", "mit", "auf", "auch", "nach", "wie", "noch",
    },
}


def _detect_language(text: str) -> str:
    """Return ISO 639-1 code of the best-matching language."""
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if not words:
        return "unknown"
    best_lang = "unknown"
    best_score = 0
    for lang, signature in _LANG_WORDS.items():
        score = len(words & signature)
        if score > best_score:
            best_score = score
            best_lang = lang
    return best_lang


class InputGuard:
    """Composable input validation pipeline with fluent builder API.

    Chain validation rules and run them against user input.
    Each rule specifies an action: block, warn, or sanitize.
    """

    def __init__(self) -> None:
        self._rules: list[GuardRule] = []
        self._total_checked: int = 0
        self._blocked_count: int = 0
        self._warned_count: int = 0

    # -- Builder methods -----------------------------------------------------

    def max_length(self, chars: int, action: str = "block") -> InputGuard:
        """Reject or warn on input exceeding *chars* characters."""
        def check(text: str) -> str | None:
            if len(text) > chars:
                return f"Input exceeds {chars} characters ({len(text)})"
            return None

        self._rules.append(GuardRule(
            name="max_length",
            check_fn=check,
            action=action,
            description=f"Max length {chars}",
        ))
        return self

    def min_length(self, chars: int, action: str = "block") -> InputGuard:
        """Reject or warn on input shorter than *chars* characters."""
        def check(text: str) -> str | None:
            if len(text) < chars:
                return f"Input below {chars} characters ({len(text)})"
            return None

        self._rules.append(GuardRule(
            name="min_length",
            check_fn=check,
            action=action,
            description=f"Min length {chars}",
        ))
        return self

    def block_patterns(
        self,
        patterns: list[str],
        action: str = "block",
    ) -> InputGuard:
        """Block or warn on input matching any regex pattern."""
        compiled = [(re.compile(p, re.IGNORECASE), p) for p in patterns]

        def check(text: str) -> str | None:
            for regex, raw in compiled:
                if regex.search(text):
                    return f"Blocked pattern matched: {raw}"
            return None

        self._rules.append(GuardRule(
            name="block_patterns",
            check_fn=check,
            action=action,
            description=f"Block {len(patterns)} patterns",
        ))
        return self

    def require_language(
        self,
        lang: str,
        action: str = "block",
    ) -> InputGuard:
        """Require the input to be in a specific language (ISO 639-1)."""
        def check(text: str) -> str | None:
            detected = _detect_language(text)
            if detected != lang:
                return f"Expected language '{lang}', detected '{detected}'"
            return None

        self._rules.append(GuardRule(
            name="require_language",
            check_fn=check,
            action=action,
            description=f"Require language {lang}",
        ))
        return self

    def safe_encoding(self, action: str = "block") -> InputGuard:
        """Reject input containing zero-width or control characters."""
        _ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060}

        def check(text: str) -> str | None:
            for char in text:
                code_point = ord(char)
                if code_point in _ZERO_WIDTH:
                    return "Zero-width character detected"
                if code_point < 32 and code_point not in (9, 10, 13):
                    return f"Control character U+{code_point:04X} detected"
            return None

        self._rules.append(GuardRule(
            name="safe_encoding",
            check_fn=check,
            action=action,
            description="No zero-width or control characters",
        ))
        return self

    def custom(
        self,
        name: str,
        check_fn: Callable[[str], str | None],
        action: str = "block",
        description: str = "",
    ) -> InputGuard:
        """Add a custom validation rule.

        *check_fn* receives the text and returns an error message string
        if the check fails, or ``None`` if the text is acceptable.
        """
        self._rules.append(GuardRule(
            name=name,
            check_fn=check_fn,
            action=action,
            description=description or f"Custom rule: {name}",
        ))
        return self

    # -- Execution -----------------------------------------------------------

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def validate(self, text: str) -> GuardResult:
        """Run *text* through all configured rules.

        Returns a ``GuardResult`` summarising which rules fired,
        whether the input was blocked, and any warnings produced.
        """
        self._total_checked += 1
        blocked = False
        warnings: list[str] = []
        rules_triggered: list[str] = []
        sanitized_text = text

        for rule in self._rules:
            message = rule.check_fn(sanitized_text)
            if message is None:
                continue

            rules_triggered.append(rule.name)

            if rule.action == "block":
                blocked = True
            elif rule.action == "warn":
                warnings.append(f"[{rule.name}] {message}")
            elif rule.action == "sanitize":
                sanitized_text = _apply_sanitization(rule.name, sanitized_text)

        if blocked:
            self._blocked_count += 1
        if warnings:
            self._warned_count += 1

        passed = not blocked
        return GuardResult(
            text=text,
            passed=passed,
            blocked=blocked,
            warnings=warnings,
            sanitized_text=sanitized_text,
            rules_triggered=rules_triggered,
        )

    def validate_batch(self, texts: list[str]) -> list[GuardResult]:
        """Validate multiple inputs and return a list of results."""
        return [self.validate(t) for t in texts]

    def report(self) -> GuardReport:
        """Return aggregate statistics from all validations so far."""
        total = self._total_checked
        pass_rate = (total - self._blocked_count) / total if total > 0 else 0.0
        return GuardReport(
            total_checked=total,
            blocked_count=self._blocked_count,
            warned_count=self._warned_count,
            pass_rate=pass_rate,
        )

    def reset_stats(self) -> None:
        """Reset internal counters to zero."""
        self._total_checked = 0
        self._blocked_count = 0
        self._warned_count = 0


# -- Sanitization helpers ----------------------------------------------------

def _apply_sanitization(rule_name: str, text: str) -> str:
    """Best-effort sanitization for known rule types."""
    if rule_name == "max_length":
        return text  # cannot shorten without knowing the limit
    if rule_name == "block_patterns":
        return text  # patterns already stripped would lose context
    if rule_name == "safe_encoding":
        return _strip_unsafe_chars(text)
    return text


def _strip_unsafe_chars(text: str) -> str:
    """Remove zero-width and control characters."""
    zero_width = {0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060}
    out: list[str] = []
    for char in text:
        code_point = ord(char)
        if code_point in zero_width:
            continue
        if code_point < 32 and code_point not in (9, 10, 13):
            continue
        out.append(char)
    return "".join(out)
