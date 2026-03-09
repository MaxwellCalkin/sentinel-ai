"""SafetyPolicy -- declarative safety policy definition language.

Specify what LLM behaviours are allowed, restricted, or blocked using
string-based conditions and priority-ordered rules.

Usage:
    from sentinel.safety_policy import SafetyPolicy

    policy = SafetyPolicy("strict", mode="enforce")
    policy.add_rule("no_secrets", condition="contains", pattern="password",
                     action="block", priority=10, message="Secrets not allowed")
    policy.add_rule("short_only", condition="length_gt", pattern="500",
                     action="warn", priority=5)

    result = policy.evaluate("my password is hunter2")
    assert result.blocked
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VALID_CONDITIONS = {"contains", "regex", "length_gt", "length_lt"}
VALID_ACTIONS = {"block", "warn", "allow", "redact"}


@dataclass
class PolicyRule:
    """A single declarative safety rule."""

    name: str
    condition: str
    pattern: str
    action: str
    priority: int = 0
    message: str = ""
    enabled: bool = True


@dataclass
class RuleMatch:
    """A rule that matched during evaluation."""

    rule_name: str
    action: str
    message: str
    matched_text: str


@dataclass
class PolicyEvaluation:
    """Result of evaluating text against a SafetyPolicy."""

    text: str
    passed: bool
    matches: list[RuleMatch]
    action_taken: str
    blocked: bool


class SafetyPolicy:
    """Declarative safety policy with string-based conditions.

    Modes:
        enforce   -- block violations (default)
        audit     -- log matches but never block
        permissive -- downgrade "block" actions to "warn"
    """

    _VALID_MODES = {"enforce", "audit", "permissive"}

    def __init__(self, name: str, mode: str = "enforce") -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {sorted(self._VALID_MODES)}"
            )
        self.name = name
        self.mode = mode
        self._rules: dict[str, PolicyRule] = {}

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(
        self,
        name: str,
        condition: str,
        pattern: str = "",
        action: str = "block",
        priority: int = 0,
        message: str = "",
    ) -> None:
        """Add a rule to the policy.

        Args:
            name:      Unique rule identifier.
            condition: One of "contains", "regex", "length_gt", "length_lt".
            pattern:   The value used by the condition (substring, regex, or int string).
            action:    One of "block", "warn", "allow", "redact".
            priority:  Higher values are evaluated first.
            message:   Human-readable description of the violation.
        """
        _validate_condition(condition)
        _validate_action(action)
        self._rules[name] = PolicyRule(
            name=name,
            condition=condition,
            pattern=pattern,
            action=action,
            priority=priority,
            message=message,
            enabled=True,
        )

    def remove_rule(self, name: str) -> None:
        """Remove a rule by name. Raises KeyError if not found."""
        del self._rules[name]

    def list_rules(self) -> list[PolicyRule]:
        """Return all rules sorted by priority descending."""
        return sorted(self._rules.values(), key=lambda r: r.priority, reverse=True)

    def enable_rule(self, name: str) -> None:
        """Enable a previously disabled rule. Raises KeyError if not found."""
        self._rules[name].enabled = True

    def disable_rule(self, name: str) -> None:
        """Disable a rule so it is skipped during evaluation. Raises KeyError if not found."""
        self._rules[name].enabled = False

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, text: str) -> PolicyEvaluation:
        """Evaluate *text* against all enabled rules in priority order."""
        matches: list[RuleMatch] = []
        strongest_action = "allow"

        for rule in self.list_rules():
            if not rule.enabled:
                continue
            matched_text = _check_condition(rule.condition, rule.pattern, text)
            if matched_text is None:
                continue

            effective_action = _effective_action(rule.action, self.mode)
            matches.append(RuleMatch(
                rule_name=rule.name,
                action=effective_action,
                message=rule.message,
                matched_text=matched_text,
            ))

            if _action_severity(effective_action) > _action_severity(strongest_action):
                strongest_action = effective_action

        blocked = strongest_action == "block" and self.mode != "audit"
        passed = not blocked

        return PolicyEvaluation(
            text=text,
            passed=passed,
            matches=matches,
            action_taken=strongest_action,
            blocked=blocked,
        )

    def evaluate_batch(self, texts: list[str]) -> list[PolicyEvaluation]:
        """Evaluate multiple texts and return a list of results."""
        return [self.evaluate(t) for t in texts]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def export(self) -> dict:
        """Export the policy as a plain dict (JSON-serialisable)."""
        return {
            "name": self.name,
            "mode": self.mode,
            "rules": [
                {
                    "name": r.name,
                    "condition": r.condition,
                    "pattern": r.pattern,
                    "action": r.action,
                    "priority": r.priority,
                    "message": r.message,
                    "enabled": r.enabled,
                }
                for r in self._rules.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SafetyPolicy:
        """Re-create a SafetyPolicy from a dict produced by :meth:`export`."""
        policy = cls(name=data["name"], mode=data.get("mode", "enforce"))
        for rule_data in data.get("rules", []):
            policy.add_rule(
                name=rule_data["name"],
                condition=rule_data["condition"],
                pattern=rule_data.get("pattern", ""),
                action=rule_data.get("action", "block"),
                priority=rule_data.get("priority", 0),
                message=rule_data.get("message", ""),
            )
            if not rule_data.get("enabled", True):
                policy.disable_rule(rule_data["name"])
        return policy

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge(self, other: SafetyPolicy) -> SafetyPolicy:
        """Merge *other* into a new policy. Higher priority wins on name conflict."""
        merged = SafetyPolicy(
            name=f"{self.name}+{other.name}",
            mode=self.mode,
        )
        combined: dict[str, PolicyRule] = {}
        for rule in list(self._rules.values()) + list(other._rules.values()):
            existing = combined.get(rule.name)
            if existing is None or rule.priority > existing.priority:
                combined[rule.name] = rule

        for rule in combined.values():
            merged.add_rule(
                name=rule.name,
                condition=rule.condition,
                pattern=rule.pattern,
                action=rule.action,
                priority=rule.priority,
                message=rule.message,
            )
            if not rule.enabled:
                merged.disable_rule(rule.name)

        return merged


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _validate_condition(condition: str) -> None:
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Invalid condition '{condition}'. Must be one of {sorted(VALID_CONDITIONS)}"
        )


def _validate_action(action: str) -> None:
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"Invalid action '{action}'. Must be one of {sorted(VALID_ACTIONS)}"
        )


def _check_condition(condition: str, pattern: str, text: str) -> str | None:
    """Return the matched substring, or None if the condition does not match."""
    if condition == "contains":
        lower_text = text.lower()
        lower_pattern = pattern.lower()
        idx = lower_text.find(lower_pattern)
        if idx == -1:
            return None
        return text[idx : idx + len(pattern)]

    if condition == "regex":
        match = re.search(pattern, text)
        if match is None:
            return None
        return match.group(0)

    if condition == "length_gt":
        threshold = int(pattern)
        if len(text) > threshold:
            return text
        return None

    if condition == "length_lt":
        threshold = int(pattern)
        if len(text) < threshold:
            return text
        return None

    return None


def _effective_action(action: str, mode: str) -> str:
    """Downgrade block to warn in permissive mode."""
    if mode == "permissive" and action == "block":
        return "warn"
    return action


def _action_severity(action: str) -> int:
    return {"allow": 0, "warn": 1, "redact": 2, "block": 3}.get(action, 0)
