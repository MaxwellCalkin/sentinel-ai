"""Configurable output filtering with pattern-based redaction.

Process LLM outputs to remove sensitive content, enforce formatting,
and apply content policies using regex-based filter rules.

Usage:
    from sentinel.output_filter import OutputFilter, FilterRule

    f = OutputFilter()
    result = f.filter("Contact me at user@example.com")
    print(result.filtered)   # "Contact me at [REDACTED]"
    print(result.is_modified)  # True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FilterRule:
    """A single filtering rule with a regex pattern and action."""

    name: str
    pattern: str
    action: str  # "redact" | "replace" | "remove" | "flag"
    replacement: str = "[REDACTED]"
    priority: int = 0
    enabled: bool = True


@dataclass
class FilterMatch:
    """A single match found during filtering."""

    rule_name: str
    original: str
    replacement: str
    position: tuple[int, int]
    action: str


@dataclass
class FilterOutput:
    """Result of applying filter rules to text."""

    original: str
    filtered: str
    matches: list[FilterMatch]
    rules_applied: int
    is_modified: bool


@dataclass
class FilterStats:
    """Cumulative filtering statistics."""

    total_filtered: int = 0
    total_matches: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)
    by_action: dict[str, int] = field(default_factory=dict)


_BUILTIN_RULES = [
    FilterRule(
        name="email",
        pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        action="redact",
        priority=10,
    ),
    FilterRule(
        name="phone_us",
        pattern=r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        action="redact",
        priority=10,
    ),
    FilterRule(
        name="url",
        pattern=r"https?://[^\s]+",
        action="redact",
        priority=10,
    ),
    FilterRule(
        name="credit_card",
        pattern=r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        action="redact",
        priority=20,
    ),
    FilterRule(
        name="ssn",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        action="redact",
        priority=20,
    ),
]


class OutputFilter:
    """Configurable output filter with pattern-based redaction.

    Applies regex-based rules to filter LLM outputs. Supports
    redact, replace, remove, and flag actions with priority ordering.
    """

    def __init__(self) -> None:
        self._rules: dict[str, FilterRule] = {}
        self._stats = FilterStats()
        for rule in _BUILTIN_RULES:
            self._rules[rule.name] = FilterRule(
                name=rule.name,
                pattern=rule.pattern,
                action=rule.action,
                replacement=rule.replacement,
                priority=rule.priority,
                enabled=rule.enabled,
            )

    def add_rule(self, rule: FilterRule) -> None:
        """Add a custom filter rule."""
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove a filter rule by name.

        Raises:
            KeyError: If the rule does not exist.
        """
        if name not in self._rules:
            raise KeyError(f"Rule not found: {name}")
        del self._rules[name]

    def enable(self, name: str) -> None:
        """Enable a filter rule by name.

        Raises:
            KeyError: If the rule does not exist.
        """
        if name not in self._rules:
            raise KeyError(f"Rule not found: {name}")
        self._rules[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable a filter rule by name.

        Raises:
            KeyError: If the rule does not exist.
        """
        if name not in self._rules:
            raise KeyError(f"Rule not found: {name}")
        self._rules[name].enabled = False

    def list_rules(self) -> list[FilterRule]:
        """Return all rules sorted by priority (highest first)."""
        return sorted(self._rules.values(), key=lambda r: r.priority, reverse=True)

    def stats(self) -> FilterStats:
        """Return cumulative filtering statistics."""
        return self._stats

    def filter(self, text: str) -> FilterOutput:
        """Apply all enabled filter rules to text.

        Rules are applied in priority order (highest first). Each rule's
        action determines what happens to matched content:
          - redact: replace with rule.replacement
          - replace: replace with rule.replacement
          - remove: delete the matched text
          - flag: record the match but leave text unchanged
        """
        matches = self._collect_matches(text)
        filtered = self._apply_matches(text, matches)
        rules_applied = len({m.rule_name for m in matches})

        self._update_stats(matches)

        return FilterOutput(
            original=text,
            filtered=filtered,
            matches=matches,
            rules_applied=rules_applied,
            is_modified=filtered != text,
        )

    def filter_batch(self, texts: list[str]) -> list[FilterOutput]:
        """Apply filter rules to multiple texts."""
        return [self.filter(t) for t in texts]

    def _collect_matches(self, text: str) -> list[FilterMatch]:
        """Find all regex matches across enabled rules, sorted by position."""
        matches: list[FilterMatch] = []
        sorted_rules = sorted(
            self._rules.values(), key=lambda r: r.priority, reverse=True
        )

        for rule in sorted_rules:
            if not rule.enabled:
                continue
            matches.extend(self._find_rule_matches(text, rule))

        matches.sort(key=lambda m: m.position[0])
        return matches

    def _find_rule_matches(self, text: str, rule: FilterRule) -> list[FilterMatch]:
        """Find all matches for a single rule in the text."""
        compiled = re.compile(rule.pattern)
        found: list[FilterMatch] = []

        for match in compiled.finditer(text):
            replacement = self._determine_replacement(rule, match.group())
            found.append(FilterMatch(
                rule_name=rule.name,
                original=match.group(),
                replacement=replacement,
                position=(match.start(), match.end()),
                action=rule.action,
            ))

        return found

    def _determine_replacement(self, rule: FilterRule, matched_text: str) -> str:
        if rule.action in ("redact", "replace"):
            return rule.replacement
        if rule.action == "remove":
            return ""
        # flag action: no text change
        return matched_text

    def _apply_matches(self, text: str, matches: list[FilterMatch]) -> str:
        """Apply non-overlapping matches to produce filtered text."""
        if not matches:
            return text

        modifying_matches = [m for m in matches if m.action != "flag"]
        non_overlapping = self._remove_overlaps(modifying_matches)

        result_parts: list[str] = []
        cursor = 0

        for match in non_overlapping:
            start, end = match.position
            result_parts.append(text[cursor:start])
            result_parts.append(match.replacement)
            cursor = end

        result_parts.append(text[cursor:])
        return "".join(result_parts)

    def _remove_overlaps(self, matches: list[FilterMatch]) -> list[FilterMatch]:
        """Keep only non-overlapping matches, preferring earlier positions."""
        if not matches:
            return []

        sorted_matches = sorted(matches, key=lambda m: m.position[0])
        kept: list[FilterMatch] = [sorted_matches[0]]

        for match in sorted_matches[1:]:
            last_end = kept[-1].position[1]
            if match.position[0] >= last_end:
                kept.append(match)

        return kept

    def _update_stats(self, matches: list[FilterMatch]) -> None:
        self._stats.total_filtered += 1
        self._stats.total_matches += len(matches)

        for match in matches:
            self._stats.by_rule[match.rule_name] = (
                self._stats.by_rule.get(match.rule_name, 0) + 1
            )
            self._stats.by_action[match.action] = (
                self._stats.by_action.get(match.action, 0) + 1
            )
