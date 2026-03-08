"""Declarative prompt firewall with rule-based filtering.

Define safety rules using a simple DSL to block, transform, or flag
prompts based on content patterns, metadata, and risk assessments.

Usage:
    from sentinel.prompt_firewall import PromptFirewall, Rule, Action

    fw = PromptFirewall([
        Rule.block(contains="DROP TABLE"),
        Rule.block(regex=r"(?i)ignore\\s+previous\\s+instructions"),
        Rule.flag(contains="password", reason="Possible credential request"),
        Rule.transform(contains="<script>", replace_with="[BLOCKED]"),
    ])

    result = fw.check("Please ignore previous instructions")
    assert result.blocked
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence


class Action(Enum):
    """What to do when a rule matches."""
    ALLOW = "allow"
    BLOCK = "block"
    FLAG = "flag"
    TRANSFORM = "transform"


@dataclass
class RuleMatch:
    """A match from a single rule."""
    rule_name: str
    action: Action
    reason: str
    matched_text: str | None = None
    span: tuple[int, int] | None = None


@dataclass
class FirewallResult:
    """Result of checking a prompt against the firewall."""
    original: str
    transformed: str
    matches: list[RuleMatch] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(m.action == Action.BLOCK for m in self.matches)

    @property
    def flagged(self) -> bool:
        return any(m.action == Action.FLAG for m in self.matches)

    @property
    def modified(self) -> bool:
        return self.original != self.transformed

    @property
    def allowed(self) -> bool:
        return not self.blocked

    @property
    def summary(self) -> str:
        if not self.matches:
            return "Allowed: no rules matched"
        parts = []
        for m in self.matches:
            parts.append(f"[{m.action.value.upper()}] {m.rule_name}: {m.reason}")
        return "; ".join(parts)


@dataclass
class Rule:
    """A firewall rule."""
    name: str
    action: Action
    reason: str
    _check: Callable[[str], list[tuple[str, tuple[int, int] | None]]]
    _replacement: str | None = None

    @staticmethod
    def block(
        contains: str | None = None,
        regex: str | None = None,
        check: Callable[[str], bool] | None = None,
        name: str | None = None,
        reason: str | None = None,
        case_sensitive: bool = False,
    ) -> Rule:
        """Create a blocking rule."""
        return _build_rule(
            Action.BLOCK, contains, regex, check, name, reason, case_sensitive
        )

    @staticmethod
    def flag(
        contains: str | None = None,
        regex: str | None = None,
        check: Callable[[str], bool] | None = None,
        name: str | None = None,
        reason: str | None = None,
        case_sensitive: bool = False,
    ) -> Rule:
        """Create a flagging rule."""
        return _build_rule(
            Action.FLAG, contains, regex, check, name, reason, case_sensitive
        )

    @staticmethod
    def allow(
        contains: str | None = None,
        regex: str | None = None,
        check: Callable[[str], bool] | None = None,
        name: str | None = None,
        reason: str | None = None,
        case_sensitive: bool = False,
    ) -> Rule:
        """Create an allow rule (explicitly allows matching content)."""
        return _build_rule(
            Action.ALLOW, contains, regex, check, name, reason, case_sensitive
        )

    @staticmethod
    def transform(
        contains: str | None = None,
        regex: str | None = None,
        replace_with: str = "[REDACTED]",
        name: str | None = None,
        reason: str | None = None,
        case_sensitive: bool = False,
    ) -> Rule:
        """Create a transform rule that replaces matched content."""
        rule = _build_rule(
            Action.TRANSFORM, contains, regex, None, name, reason, case_sensitive
        )
        rule._replacement = replace_with
        return rule


def _build_rule(
    action: Action,
    contains: str | None,
    regex: str | None,
    check: Callable[[str], bool] | None,
    name: str | None,
    reason: str | None,
    case_sensitive: bool,
) -> Rule:
    """Build a rule from the provided parameters."""
    if sum(x is not None for x in [contains, regex, check]) != 1:
        raise ValueError("Exactly one of contains, regex, or check must be provided")

    if contains is not None:
        pattern_str = re.escape(contains)
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern_str, flags)
        rule_name = name or f"{action.value}_{contains[:20]}"
        rule_reason = reason or f"Contains '{contains}'"

        def checker(text: str) -> list[tuple[str, tuple[int, int] | None]]:
            return [
                (m.group(), (m.start(), m.end()))
                for m in compiled.finditer(text)
            ]

    elif regex is not None:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(regex, flags)
        rule_name = name or f"{action.value}_regex"
        rule_reason = reason or f"Matches pattern"

        def checker(text: str) -> list[tuple[str, tuple[int, int] | None]]:
            return [
                (m.group(), (m.start(), m.end()))
                for m in compiled.finditer(text)
            ]

    else:
        assert check is not None
        rule_name = name or f"{action.value}_custom"
        rule_reason = reason or "Custom check matched"

        def checker(text: str) -> list[tuple[str, tuple[int, int] | None]]:
            if check(text):
                return [("", None)]
            return []

    return Rule(
        name=rule_name,
        action=action,
        reason=rule_reason,
        _check=checker,
    )


class PromptFirewall:
    """Rule-based prompt firewall.

    Checks prompts against an ordered list of rules. Rules are evaluated
    in order. Allow rules short-circuit (skip remaining rules for matching
    content). Block rules stop processing. Transform rules modify the text.
    Flag rules annotate but don't block.
    """

    def __init__(self, rules: Sequence[Rule]):
        self._rules = list(rules)

    def check(self, text: str) -> FirewallResult:
        """Check a prompt against all rules."""
        matches: list[RuleMatch] = []
        current_text = text

        for rule in self._rules:
            hits = rule._check(current_text)
            if not hits:
                continue

            for matched_text, span in hits:
                matches.append(RuleMatch(
                    rule_name=rule.name,
                    action=rule.action,
                    reason=rule.reason,
                    matched_text=matched_text or None,
                    span=span,
                ))

            if rule.action == Action.BLOCK:
                break  # stop on first block

            if rule.action == Action.ALLOW:
                # Allow rules clear any previous flags for this content
                continue

            if rule.action == Action.TRANSFORM and rule._replacement is not None:
                # Apply transformation
                for matched_text, span in reversed(hits):
                    if span:
                        current_text = (
                            current_text[:span[0]]
                            + rule._replacement
                            + current_text[span[1]:]
                        )

        return FirewallResult(
            original=text,
            transformed=current_text,
            matches=matches,
        )

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the firewall."""
        self._rules.append(rule)

    @property
    def rules(self) -> list[Rule]:
        """Get all rules."""
        return list(self._rules)
