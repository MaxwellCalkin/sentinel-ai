"""Configurable output censoring for LLM responses.

Redact, replace, or block sensitive content patterns in LLM
responses before they reach users. Rules are applied in
registration order (priority order).

Usage:
    from sentinel.output_censor import OutputCensor, CensorRule

    censor = OutputCensor()
    censor.add_rule(CensorRule(
        name="ssn",
        pattern=r"\\d{3}-\\d{2}-\\d{4}",
        action="redact",
    ))
    result = censor.censor("My SSN is 123-45-6789")
    # result.censored: "My SSN is [REDACTED]"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CensorRule:
    """A single censoring rule."""

    name: str
    pattern: str
    action: str  # "redact", "replace", or "block"
    replacement: str = ""


@dataclass
class CensorResult:
    """Result of censoring a single text."""

    original: str
    censored: str
    blocked: bool
    rules_applied: list[str] = field(default_factory=list)
    redaction_count: int = 0


@dataclass
class CensorStats:
    """Aggregate censoring statistics."""

    total_processed: int = 0
    total_redactions: int = 0
    total_blocks: int = 0
    rules_triggered: dict[str, int] = field(default_factory=dict)


class OutputCensor:
    """Configurable output censoring engine.

    Register rules with regex patterns and actions (redact, replace,
    block), then apply them to text in priority order. Block rules
    cause the entire output to be suppressed.
    """

    def __init__(self) -> None:
        self._rules: list[CensorRule] = []
        self._stats = CensorStats()

    @property
    def stats(self) -> CensorStats:
        """Return current censoring statistics."""
        return self._stats

    @property
    def rules(self) -> list[CensorRule]:
        """Return registered rules in priority order."""
        return list(self._rules)

    def add_rule(self, rule: CensorRule) -> None:
        """Register a censor rule.

        Rules are applied in the order they are added (first added
        has highest priority).
        """
        _validate_rule(rule)
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name. Returns True if found and removed."""
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                return True
        return False

    def censor(self, text: str) -> CensorResult:
        """Apply all rules to text and return the censored result."""
        self._stats.total_processed += 1

        rules_applied: list[str] = []
        redaction_count = 0
        censored = text

        for rule in self._rules:
            compiled = re.compile(rule.pattern)
            matches = compiled.findall(censored)

            if not matches:
                continue

            rules_applied.append(rule.name)
            self._stats.rules_triggered[rule.name] = (
                self._stats.rules_triggered.get(rule.name, 0) + 1
            )

            if rule.action == "block":
                self._stats.total_blocks += 1
                return CensorResult(
                    original=text,
                    censored="",
                    blocked=True,
                    rules_applied=rules_applied,
                    redaction_count=0,
                )

            match_count = len(matches)

            if rule.action == "redact":
                censored = compiled.sub("[REDACTED]", censored)
                redaction_count += match_count
            elif rule.action == "replace":
                censored = compiled.sub(rule.replacement, censored)
                redaction_count += match_count

        self._stats.total_redactions += redaction_count

        return CensorResult(
            original=text,
            censored=censored,
            blocked=False,
            rules_applied=rules_applied,
            redaction_count=redaction_count,
        )

    def censor_batch(self, texts: list[str]) -> list[CensorResult]:
        """Apply censoring to multiple texts."""
        return [self.censor(t) for t in texts]

    def reset_stats(self) -> None:
        """Reset all statistics to zero."""
        self._stats = CensorStats()


def _validate_rule(rule: CensorRule) -> None:
    """Validate a censor rule, raising ValueError on problems."""
    if not rule.name:
        raise ValueError("Rule name must not be empty")
    if not rule.pattern:
        raise ValueError("Rule pattern must not be empty")
    valid_actions = {"redact", "replace", "block"}
    if rule.action not in valid_actions:
        raise ValueError(
            f"Invalid action '{rule.action}'; must be one of {sorted(valid_actions)}"
        )
    # Verify the pattern compiles
    try:
        re.compile(rule.pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc
