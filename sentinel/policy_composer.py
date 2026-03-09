"""Compose and merge multiple safety policies into a unified policy.

Merge rules from independent policies, resolve conflicts by priority,
detect contradictions, apply the composed policy to text, and diff
two policies for change analysis.

Usage:
    from sentinel.policy_composer import PolicyComposer, PolicyRule

    composer = PolicyComposer()
    composer.add_policy("security", [
        PolicyRule("block_secrets", lambda t: "secret" in t.lower(), "block", 10, "security"),
    ])
    composer.add_policy("ux", [
        PolicyRule("allow_greetings", lambda t: "hello" in t.lower(), "allow", 5, "ux"),
    ])
    composed = composer.compose()
    result = composer.apply("This is a secret hello")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

VALID_ACTIONS = frozenset({"allow", "block", "flag", "redact"})

_ACTION_SEVERITY: dict[str, int] = {
    "allow": 0,
    "flag": 1,
    "redact": 2,
    "block": 3,
}


@dataclass
class PolicyRule:
    """A single named rule within a policy."""

    name: str
    condition_fn: Callable[[str], bool]
    action: str
    priority: int
    source_policy: str

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{self.action}'. "
                f"Must be one of: {', '.join(sorted(VALID_ACTIONS))}"
            )


@dataclass
class RuleMatch:
    """A rule that triggered during policy application."""

    rule_name: str
    action: str
    priority: int
    source: str


@dataclass
class ComposedPolicy:
    """The result of composing multiple policies."""

    rules: list[PolicyRule]
    conflicts: list[str]
    total_rules: int


@dataclass
class PolicyApplyResult:
    """Result of applying a composed policy to text."""

    text: str
    matches: list[RuleMatch]
    final_action: str
    conflict_detected: bool


@dataclass
class PolicyDiff:
    """Differences between two composed policies."""

    added_rules: list[str]
    removed_rules: list[str]
    changed_rules: list[str]
    action_changes: dict[str, tuple[str, str]]


@dataclass
class PolicyStats:
    """Statistics about the composed policy."""

    total_rules: int
    conflicts_resolved: int
    policies_merged: int
    rules_per_policy: dict[str, int]


def _action_severity(action: str) -> int:
    return _ACTION_SEVERITY.get(action, 0)


def _actions_contradict(action_a: str, action_b: str) -> bool:
    """Two actions contradict if one allows and the other blocks/redacts."""
    permissive = {"allow", "flag"}
    restrictive = {"block", "redact"}
    return (
        (action_a in permissive and action_b in restrictive)
        or (action_a in restrictive and action_b in permissive)
    )


class PolicyComposer:
    """Compose multiple safety policies into a unified policy.

    Merge rules from independent policies, detect conflicts where
    rules from different policies contradict, and resolve them by
    priority. Apply the composed policy to text for safety evaluation.
    """

    def __init__(self) -> None:
        self._policies: dict[str, list[PolicyRule]] = {}
        self._composed: ComposedPolicy | None = None
        self._conflicts_resolved: int = 0

    def add_policy(self, name: str, rules: list[PolicyRule]) -> None:
        """Register a named policy with its rules.

        Args:
            name: Unique policy identifier.
            rules: List of PolicyRule instances belonging to this policy.
        """
        tagged_rules = []
        for rule in rules:
            tagged = PolicyRule(
                name=rule.name,
                condition_fn=rule.condition_fn,
                action=rule.action,
                priority=rule.priority,
                source_policy=name,
            )
            tagged_rules.append(tagged)
        self._policies[name] = tagged_rules
        self._composed = None

    def compose(self) -> ComposedPolicy:
        """Merge all registered policies into a single composed policy.

        Detects conflicts between rules with the same name from
        different policies and resolves them by keeping the
        higher-priority rule.

        Returns:
            ComposedPolicy with merged rules and conflict descriptions.
        """
        rules_by_name: dict[str, list[PolicyRule]] = {}
        for policy_rules in self._policies.values():
            for rule in policy_rules:
                rules_by_name.setdefault(rule.name, []).append(rule)

        merged_rules: list[PolicyRule] = []
        conflicts: list[str] = []
        self._conflicts_resolved = 0

        for name, candidates in rules_by_name.items():
            if len(candidates) == 1:
                merged_rules.append(candidates[0])
                continue

            conflict_found = _detect_conflict_among(candidates)
            if conflict_found:
                conflict_description = _build_conflict_description(name, candidates)
                conflicts.append(conflict_description)
                self._conflicts_resolved += 1

            winner = _resolve_by_priority(candidates)
            merged_rules.append(winner)

        merged_rules.sort(key=lambda r: r.priority, reverse=True)

        self._composed = ComposedPolicy(
            rules=merged_rules,
            conflicts=conflicts,
            total_rules=len(merged_rules),
        )
        return self._composed

    def apply(self, text: str) -> PolicyApplyResult:
        """Apply the composed policy to text.

        Evaluates all rules against the text and returns triggered
        matches with a final action determined by the highest-severity
        triggered rule.

        Args:
            text: Input text to evaluate.

        Returns:
            PolicyApplyResult with matches and final action.
        """
        if self._composed is None:
            self.compose()
        assert self._composed is not None

        matches: list[RuleMatch] = []
        for rule in self._composed.rules:
            try:
                if rule.condition_fn(text):
                    matches.append(RuleMatch(
                        rule_name=rule.name,
                        action=rule.action,
                        priority=rule.priority,
                        source=rule.source_policy,
                    ))
            except Exception:
                continue

        final_action = _determine_final_action(matches)
        conflict_detected = _matches_have_conflict(matches)

        return PolicyApplyResult(
            text=text,
            matches=matches,
            final_action=final_action,
            conflict_detected=conflict_detected,
        )

    def diff(self, other: PolicyComposer) -> PolicyDiff:
        """Compare this composer's composed policy against another.

        Args:
            other: Another PolicyComposer to compare against.

        Returns:
            PolicyDiff showing added, removed, and changed rules.
        """
        self_composed = self._ensure_composed()
        other_composed = other._ensure_composed()

        self_rules = {r.name: r for r in self_composed.rules}
        other_rules = {r.name: r for r in other_composed.rules}

        self_names = set(self_rules.keys())
        other_names = set(other_rules.keys())

        added = sorted(self_names - other_names)
        removed = sorted(other_names - self_names)
        changed: list[str] = []
        action_changes: dict[str, tuple[str, str]] = {}

        for name in sorted(self_names & other_names):
            self_rule = self_rules[name]
            other_rule = other_rules[name]
            if self_rule.action != other_rule.action:
                changed.append(name)
                action_changes[name] = (other_rule.action, self_rule.action)

        return PolicyDiff(
            added_rules=added,
            removed_rules=removed,
            changed_rules=changed,
            action_changes=action_changes,
        )

    def export(self) -> dict:
        """Export the composed policy as a serializable dictionary.

        Returns:
            Dictionary representation of the composed policy.
        """
        composed = self._ensure_composed()
        return {
            "total_rules": composed.total_rules,
            "conflicts": composed.conflicts,
            "rules": [
                {
                    "name": rule.name,
                    "action": rule.action,
                    "priority": rule.priority,
                    "source_policy": rule.source_policy,
                }
                for rule in composed.rules
            ],
        }

    def stats(self) -> PolicyStats:
        """Compute statistics about the composed policy.

        Returns:
            PolicyStats with rule counts and conflict resolution info.
        """
        composed = self._ensure_composed()
        rules_per_policy: dict[str, int] = {}
        for rule in composed.rules:
            rules_per_policy[rule.source_policy] = (
                rules_per_policy.get(rule.source_policy, 0) + 1
            )

        return PolicyStats(
            total_rules=composed.total_rules,
            conflicts_resolved=self._conflicts_resolved,
            policies_merged=len(self._policies),
            rules_per_policy=rules_per_policy,
        )

    @property
    def policy_names(self) -> list[str]:
        """List of registered policy names."""
        return list(self._policies.keys())

    def _ensure_composed(self) -> ComposedPolicy:
        if self._composed is None:
            self.compose()
        assert self._composed is not None
        return self._composed


def _detect_conflict_among(candidates: list[PolicyRule]) -> bool:
    """Check if any pair of candidates have contradicting actions."""
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if _actions_contradict(candidates[i].action, candidates[j].action):
                return True
    return False


def _build_conflict_description(name: str, candidates: list[PolicyRule]) -> str:
    sources = [
        f"{c.source_policy}({c.action}, pri={c.priority})"
        for c in candidates
    ]
    return f"Rule '{name}' conflicts: {' vs '.join(sources)}"


def _resolve_by_priority(candidates: list[PolicyRule]) -> PolicyRule:
    """Return the candidate with the highest priority. Break ties by action severity."""
    return max(
        candidates,
        key=lambda r: (r.priority, _action_severity(r.action)),
    )


def _determine_final_action(matches: list[RuleMatch]) -> str:
    """Determine the final action from a list of matches by highest severity."""
    if not matches:
        return "allow"
    return max(matches, key=lambda m: _action_severity(m.action)).action


def _matches_have_conflict(matches: list[RuleMatch]) -> bool:
    """Check if triggered matches include contradicting actions."""
    if len(matches) < 2:
        return False
    actions = {m.action for m in matches}
    permissive = actions & {"allow", "flag"}
    restrictive = actions & {"block", "redact"}
    return bool(permissive and restrictive)
