"""Tests for policy composition and merging."""

import pytest
from sentinel.policy_composer import (
    PolicyComposer,
    PolicyRule,
    RuleMatch,
    ComposedPolicy,
    PolicyApplyResult,
    PolicyDiff,
    PolicyStats,
)


# ---------------------------------------------------------------------------
# PolicyRule dataclass
# ---------------------------------------------------------------------------

class TestPolicyRule:
    def test_create_valid_rule(self):
        rule = PolicyRule("test", lambda t: True, "block", 10, "security")
        assert rule.name == "test"
        assert rule.action == "block"
        assert rule.priority == 10
        assert rule.source_policy == "security"

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="Invalid action"):
            PolicyRule("bad", lambda t: True, "destroy", 1, "src")


# ---------------------------------------------------------------------------
# Compose policies
# ---------------------------------------------------------------------------

class TestCompose:
    def test_compose_single_policy(self):
        composer = PolicyComposer()
        composer.add_policy("sec", [
            PolicyRule("block_secrets", lambda t: "secret" in t, "block", 10, "sec"),
        ])
        composed = composer.compose()
        assert isinstance(composed, ComposedPolicy)
        assert composed.total_rules == 1
        assert len(composed.conflicts) == 0

    def test_compose_merges_disjoint_rules(self):
        composer = PolicyComposer()
        composer.add_policy("sec", [
            PolicyRule("block_secrets", lambda t: "secret" in t, "block", 10, "sec"),
        ])
        composer.add_policy("ux", [
            PolicyRule("flag_long", lambda t: len(t) > 100, "flag", 5, "ux"),
        ])
        composed = composer.compose()
        assert composed.total_rules == 2
        assert len(composed.conflicts) == 0

    def test_compose_resolves_conflict_by_priority(self):
        composer = PolicyComposer()
        composer.add_policy("strict", [
            PolicyRule("greetings", lambda t: "hello" in t.lower(), "block", 10, "strict"),
        ])
        composer.add_policy("lenient", [
            PolicyRule("greetings", lambda t: "hello" in t.lower(), "allow", 5, "lenient"),
        ])
        composed = composer.compose()
        assert composed.total_rules == 1
        winning_rule = composed.rules[0]
        assert winning_rule.action == "block"
        assert winning_rule.source_policy == "strict"
        assert len(composed.conflicts) == 1

    def test_compose_rules_sorted_by_priority(self):
        composer = PolicyComposer()
        composer.add_policy("a", [
            PolicyRule("low", lambda t: True, "flag", 1, "a"),
            PolicyRule("high", lambda t: True, "block", 100, "a"),
            PolicyRule("mid", lambda t: True, "redact", 50, "a"),
        ])
        composed = composer.compose()
        priorities = [r.priority for r in composed.rules]
        assert priorities == sorted(priorities, reverse=True)


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

class TestConflictDetection:
    def test_no_conflict_same_action(self):
        composer = PolicyComposer()
        composer.add_policy("a", [
            PolicyRule("shared", lambda t: True, "block", 10, "a"),
        ])
        composer.add_policy("b", [
            PolicyRule("shared", lambda t: True, "block", 5, "b"),
        ])
        composed = composer.compose()
        assert len(composed.conflicts) == 0

    def test_conflict_allow_vs_block(self):
        composer = PolicyComposer()
        composer.add_policy("a", [
            PolicyRule("check", lambda t: True, "allow", 10, "a"),
        ])
        composer.add_policy("b", [
            PolicyRule("check", lambda t: True, "block", 5, "b"),
        ])
        composed = composer.compose()
        assert len(composed.conflicts) == 1
        assert "check" in composed.conflicts[0]


# ---------------------------------------------------------------------------
# Apply composed policy
# ---------------------------------------------------------------------------

class TestApply:
    def test_apply_no_matches_allows(self):
        composer = PolicyComposer()
        composer.add_policy("sec", [
            PolicyRule("block_bad", lambda t: "bad" in t, "block", 10, "sec"),
        ])
        result = composer.apply("good text")
        assert isinstance(result, PolicyApplyResult)
        assert result.final_action == "allow"
        assert len(result.matches) == 0
        assert result.text == "good text"

    def test_apply_triggers_matching_rules(self):
        composer = PolicyComposer()
        composer.add_policy("sec", [
            PolicyRule("block_bad", lambda t: "bad" in t, "block", 10, "sec"),
            PolicyRule("flag_maybe", lambda t: "maybe" in t, "flag", 5, "sec"),
        ])
        result = composer.apply("this is bad")
        assert result.final_action == "block"
        assert len(result.matches) == 1
        assert result.matches[0].rule_name == "block_bad"

    def test_apply_multiple_matches_highest_severity_wins(self):
        composer = PolicyComposer()
        composer.add_policy("mixed", [
            PolicyRule("flag_it", lambda t: True, "flag", 1, "mixed"),
            PolicyRule("block_it", lambda t: True, "block", 2, "mixed"),
        ])
        result = composer.apply("anything")
        assert result.final_action == "block"
        assert len(result.matches) == 2

    def test_apply_conflict_detected_in_matches(self):
        composer = PolicyComposer()
        composer.add_policy("dual", [
            PolicyRule("allow_all", lambda t: True, "allow", 1, "dual"),
            PolicyRule("block_all", lambda t: True, "block", 2, "dual"),
        ])
        result = composer.apply("test")
        assert result.conflict_detected is True

    def test_apply_condition_error_skipped(self):
        composer = PolicyComposer()
        composer.add_policy("broken", [
            PolicyRule("crash", lambda t: 1 / 0, "block", 10, "broken"),
        ])
        result = composer.apply("test")
        assert result.final_action == "allow"
        assert len(result.matches) == 0


# ---------------------------------------------------------------------------
# Policy diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_diff_identical_policies(self):
        composer_a = PolicyComposer()
        composer_a.add_policy("x", [
            PolicyRule("r1", lambda t: True, "block", 10, "x"),
        ])
        composer_b = PolicyComposer()
        composer_b.add_policy("x", [
            PolicyRule("r1", lambda t: True, "block", 10, "x"),
        ])
        diff = composer_a.diff(composer_b)
        assert isinstance(diff, PolicyDiff)
        assert diff.added_rules == []
        assert diff.removed_rules == []
        assert diff.changed_rules == []

    def test_diff_added_and_removed_rules(self):
        composer_a = PolicyComposer()
        composer_a.add_policy("x", [
            PolicyRule("r1", lambda t: True, "block", 10, "x"),
            PolicyRule("r_new", lambda t: True, "flag", 5, "x"),
        ])
        composer_b = PolicyComposer()
        composer_b.add_policy("x", [
            PolicyRule("r1", lambda t: True, "block", 10, "x"),
            PolicyRule("r_old", lambda t: True, "redact", 3, "x"),
        ])
        diff = composer_a.diff(composer_b)
        assert "r_new" in diff.added_rules
        assert "r_old" in diff.removed_rules

    def test_diff_action_change(self):
        composer_a = PolicyComposer()
        composer_a.add_policy("x", [
            PolicyRule("r1", lambda t: True, "block", 10, "x"),
        ])
        composer_b = PolicyComposer()
        composer_b.add_policy("x", [
            PolicyRule("r1", lambda t: True, "flag", 10, "x"),
        ])
        diff = composer_a.diff(composer_b)
        assert "r1" in diff.changed_rules
        assert diff.action_changes["r1"] == ("flag", "block")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_as_dict(self):
        composer = PolicyComposer()
        composer.add_policy("sec", [
            PolicyRule("rule_a", lambda t: True, "block", 10, "sec"),
        ])
        exported = composer.export()
        assert isinstance(exported, dict)
        assert exported["total_rules"] == 1
        assert len(exported["rules"]) == 1
        assert exported["rules"][0]["name"] == "rule_a"
        assert exported["rules"][0]["action"] == "block"
        assert exported["rules"][0]["priority"] == 10
        assert exported["rules"][0]["source_policy"] == "sec"
        assert isinstance(exported["conflicts"], list)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_basic(self):
        composer = PolicyComposer()
        composer.add_policy("sec", [
            PolicyRule("r1", lambda t: True, "block", 10, "sec"),
            PolicyRule("r2", lambda t: True, "flag", 5, "sec"),
        ])
        composer.add_policy("ux", [
            PolicyRule("r3", lambda t: True, "allow", 3, "ux"),
        ])
        stats = composer.stats()
        assert isinstance(stats, PolicyStats)
        assert stats.total_rules == 3
        assert stats.policies_merged == 2
        assert stats.rules_per_policy["sec"] == 2
        assert stats.rules_per_policy["ux"] == 1
        assert set(composer.policy_names) == {"sec", "ux"}

    def test_stats_conflicts_resolved(self):
        composer = PolicyComposer()
        composer.add_policy("a", [
            PolicyRule("shared", lambda t: True, "allow", 1, "a"),
        ])
        composer.add_policy("b", [
            PolicyRule("shared", lambda t: True, "block", 10, "b"),
        ])
        stats = composer.stats()
        assert stats.conflicts_resolved == 1
        assert stats.total_rules == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_composer(self):
        composer = PolicyComposer()
        composed = composer.compose()
        assert composed.total_rules == 0
        assert composed.conflicts == []

    def test_apply_auto_composes_if_needed(self):
        composer = PolicyComposer()
        composer.add_policy("x", [
            PolicyRule("r1", lambda t: "trigger" in t, "flag", 5, "x"),
        ])
        result = composer.apply("trigger this")
        assert result.final_action == "flag"
        assert len(result.matches) == 1

    def test_recompose_after_adding_policy(self):
        composer = PolicyComposer()
        composer.add_policy("a", [
            PolicyRule("r1", lambda t: "x" in t, "block", 10, "a"),
        ])
        composed1 = composer.compose()
        assert composed1.total_rules == 1
        composer.add_policy("b", [
            PolicyRule("r2", lambda t: "y" in t, "flag", 5, "b"),
        ])
        composed2 = composer.compose()
        assert composed2.total_rules == 2
