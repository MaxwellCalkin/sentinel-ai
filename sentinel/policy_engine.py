"""Declarative safety policy engine.

Define safety policies as composable rules with conditions,
actions, and priorities. Evaluate text against policies
to determine allow/deny/modify decisions.

Usage:
    from sentinel.policy_engine import PolicyEngine

    engine = PolicyEngine()
    engine.add_rule("no_pii", condition=lambda t: "ssn" in t.lower(), action="block")
    result = engine.evaluate("My SSN is 123-45-6789")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class PolicyAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"
    ESCALATE = "escalate"


@dataclass
class PolicyRule:
    """A single policy rule."""
    name: str
    condition: Callable[[str], bool]
    action: PolicyAction
    priority: int = 0
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class PolicyViolation:
    """A single policy violation found."""
    rule_name: str
    action: PolicyAction
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    allowed: bool
    violations: list[PolicyViolation]
    warnings: list[PolicyViolation]
    applied_action: PolicyAction
    rules_evaluated: int
    rules_matched: int

    @property
    def blocked(self) -> bool:
        return not self.allowed

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class PolicyEngine:
    """Declarative policy engine for safety rules.

    Evaluate text against a set of rules to determine
    the appropriate action (allow, block, warn, etc.).
    Rules are evaluated in priority order.
    """

    def __init__(self, default_action: str = "allow") -> None:
        """
        Args:
            default_action: Action when no rules match.
        """
        self._default = PolicyAction(default_action)
        self._rules: list[PolicyRule] = []

    def add_rule(
        self,
        name: str,
        condition: Callable[[str], bool],
        action: str = "block",
        priority: int = 0,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a policy rule.

        Args:
            name: Rule identifier.
            condition: Function(text) -> True if rule triggers.
            action: Action to take (allow, block, warn, redact, escalate).
            priority: Higher = evaluated first.
            message: Human-readable violation message.
            metadata: Additional rule data.
        """
        self._rules.append(PolicyRule(
            name=name,
            condition=condition,
            action=PolicyAction(action),
            priority=priority,
            message=message or f"Policy violation: {name}",
            metadata=metadata or {},
        ))
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add_keyword_rule(
        self,
        name: str,
        keywords: list[str],
        action: str = "block",
        case_sensitive: bool = False,
        message: str = "",
    ) -> None:
        """Add a keyword-based rule."""
        if not case_sensitive:
            keywords = [k.lower() for k in keywords]

        def condition(text: str) -> bool:
            t = text if case_sensitive else text.lower()
            return any(k in t for k in keywords)

        self.add_rule(name, condition, action, message=message)

    def add_pattern_rule(
        self,
        name: str,
        patterns: list[str],
        action: str = "block",
        message: str = "",
    ) -> None:
        """Add a regex pattern-based rule."""
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

        def condition(text: str) -> bool:
            return any(r.search(text) for r in compiled)

        self.add_rule(name, condition, action, message=message)

    def add_length_rule(
        self,
        name: str,
        max_length: int,
        action: str = "block",
        message: str = "",
    ) -> None:
        """Block text exceeding max length."""
        self.add_rule(
            name,
            condition=lambda t: len(t) > max_length,
            action=action,
            message=message or f"Text exceeds {max_length} characters",
        )

    def disable_rule(self, name: str) -> bool:
        """Disable a rule by name."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = False
                return True
        return False

    def enable_rule(self, name: str) -> bool:
        """Enable a rule by name."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = True
                return True
        return False

    def evaluate(self, text: str) -> PolicyResult:
        """Evaluate text against all policies.

        Args:
            text: Text to evaluate.

        Returns:
            PolicyResult with violations, warnings, and final action.
        """
        violations: list[PolicyViolation] = []
        warnings: list[PolicyViolation] = []
        rules_evaluated = 0
        rules_matched = 0
        strongest_action = self._default

        for rule in self._rules:
            if not rule.enabled:
                continue
            rules_evaluated += 1

            try:
                if rule.condition(text):
                    rules_matched += 1
                    violation = PolicyViolation(
                        rule_name=rule.name,
                        action=rule.action,
                        message=rule.message,
                        metadata=rule.metadata,
                    )

                    if rule.action == PolicyAction.WARN:
                        warnings.append(violation)
                    else:
                        violations.append(violation)

                    # Track strongest action
                    if self._action_severity(rule.action) > self._action_severity(strongest_action):
                        strongest_action = rule.action
            except Exception:
                continue

        allowed = strongest_action in (PolicyAction.ALLOW, PolicyAction.WARN)

        return PolicyResult(
            allowed=allowed,
            violations=violations,
            warnings=warnings,
            applied_action=strongest_action,
            rules_evaluated=rules_evaluated,
            rules_matched=rules_matched,
        )

    def evaluate_batch(self, texts: list[str]) -> list[PolicyResult]:
        """Evaluate multiple texts."""
        return [self.evaluate(t) for t in texts]

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def active_rule_count(self) -> int:
        return sum(1 for r in self._rules if r.enabled)

    @staticmethod
    def _action_severity(action: PolicyAction) -> int:
        return {
            PolicyAction.ALLOW: 0,
            PolicyAction.WARN: 1,
            PolicyAction.REDACT: 2,
            PolicyAction.ESCALATE: 3,
            PolicyAction.BLOCK: 4,
        }.get(action, 0)
