"""Tests for SafetyPolicy declarative safety policy language."""

import pytest

from sentinel.safety_policy import (
    SafetyPolicy,
    PolicyRule,
    RuleMatch,
    PolicyEvaluation,
)


# ---------------------------------------------------------------------------
# TestRules -- add, remove, list, validation
# ---------------------------------------------------------------------------


class TestRules:
    def test_add_rule(self):
        policy = SafetyPolicy("test")
        policy.add_rule("r1", condition="contains", pattern="secret", action="block")
        rules = policy.list_rules()
        assert len(rules) == 1
        assert rules[0].name == "r1"
        assert rules[0].condition == "contains"
        assert rules[0].pattern == "secret"
        assert rules[0].action == "block"

    def test_remove_rule(self):
        policy = SafetyPolicy("test")
        policy.add_rule("r1", condition="contains", pattern="x", action="block")
        policy.remove_rule("r1")
        assert policy.list_rules() == []

    def test_remove_nonexistent_raises(self):
        policy = SafetyPolicy("test")
        with pytest.raises(KeyError):
            policy.remove_rule("nope")

    def test_list_rules_sorted(self):
        policy = SafetyPolicy("test")
        policy.add_rule("low", condition="contains", pattern="a", action="warn", priority=1)
        policy.add_rule("high", condition="contains", pattern="b", action="block", priority=10)
        policy.add_rule("mid", condition="contains", pattern="c", action="warn", priority=5)
        names = [r.name for r in policy.list_rules()]
        assert names == ["high", "mid", "low"]

    def test_invalid_condition(self):
        policy = SafetyPolicy("test")
        with pytest.raises(ValueError, match="Invalid condition"):
            policy.add_rule("bad", condition="startswith", pattern="x", action="block")

    def test_invalid_action(self):
        policy = SafetyPolicy("test")
        with pytest.raises(ValueError, match="Invalid action"):
            policy.add_rule("bad", condition="contains", pattern="x", action="delete")


# ---------------------------------------------------------------------------
# TestEvaluate -- condition matching and actions
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_contains_match(self):
        policy = SafetyPolicy("test")
        policy.add_rule("no_password", condition="contains", pattern="password",
                        action="block", message="No passwords")
        result = policy.evaluate("my password is 1234")
        assert len(result.matches) == 1
        assert result.matches[0].rule_name == "no_password"
        assert result.matches[0].matched_text == "password"

    def test_contains_case_insensitive(self):
        policy = SafetyPolicy("test")
        policy.add_rule("r1", condition="contains", pattern="SECRET", action="block")
        result = policy.evaluate("this is a secret message")
        assert len(result.matches) == 1

    def test_regex_match(self):
        policy = SafetyPolicy("test")
        policy.add_rule("ssn", condition="regex", pattern=r"\d{3}-\d{2}-\d{4}",
                        action="block", message="SSN detected")
        result = policy.evaluate("SSN: 123-45-6789")
        assert len(result.matches) == 1
        assert result.matches[0].matched_text == "123-45-6789"

    def test_length_gt(self):
        policy = SafetyPolicy("test")
        policy.add_rule("too_long", condition="length_gt", pattern="10", action="warn")
        result = policy.evaluate("short")
        assert result.passed
        assert len(result.matches) == 0

        long_result = policy.evaluate("this is a very long string")
        assert len(long_result.matches) == 1
        assert long_result.matches[0].action == "warn"

    def test_length_lt(self):
        policy = SafetyPolicy("test")
        policy.add_rule("too_short", condition="length_lt", pattern="5", action="warn")
        result = policy.evaluate("hi")
        assert len(result.matches) == 1

        ok_result = policy.evaluate("this is fine")
        assert len(ok_result.matches) == 0

    def test_no_match(self):
        policy = SafetyPolicy("test")
        policy.add_rule("r1", condition="contains", pattern="forbidden", action="block")
        result = policy.evaluate("perfectly safe text")
        assert result.passed
        assert not result.blocked
        assert result.matches == []
        assert result.action_taken == "allow"

    def test_block_action(self):
        policy = SafetyPolicy("test")
        policy.add_rule("r1", condition="contains", pattern="evil", action="block")
        result = policy.evaluate("evil plan")
        assert result.blocked
        assert not result.passed
        assert result.action_taken == "block"


# ---------------------------------------------------------------------------
# TestModes -- enforce, audit, permissive
# ---------------------------------------------------------------------------


class TestModes:
    def test_enforce_blocks(self):
        policy = SafetyPolicy("strict", mode="enforce")
        policy.add_rule("r1", condition="contains", pattern="bad", action="block")
        result = policy.evaluate("bad stuff")
        assert result.blocked
        assert not result.passed

    def test_audit_never_blocks(self):
        policy = SafetyPolicy("audit", mode="audit")
        policy.add_rule("r1", condition="contains", pattern="bad", action="block")
        result = policy.evaluate("bad stuff")
        assert not result.blocked
        assert result.passed
        assert len(result.matches) == 1
        assert result.action_taken == "block"

    def test_permissive_downgrades(self):
        policy = SafetyPolicy("soft", mode="permissive")
        policy.add_rule("r1", condition="contains", pattern="bad", action="block")
        result = policy.evaluate("bad stuff")
        assert not result.blocked
        assert result.passed
        assert result.matches[0].action == "warn"
        assert result.action_taken == "warn"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            SafetyPolicy("bad", mode="strict_mode")


# ---------------------------------------------------------------------------
# TestToggle -- enable / disable rules
# ---------------------------------------------------------------------------


class TestToggle:
    def test_disable_rule(self):
        policy = SafetyPolicy("test")
        policy.add_rule("r1", condition="contains", pattern="bad", action="block")
        policy.disable_rule("r1")
        result = policy.evaluate("bad stuff")
        assert result.passed
        assert len(result.matches) == 0

    def test_enable_rule(self):
        policy = SafetyPolicy("test")
        policy.add_rule("r1", condition="contains", pattern="bad", action="block")
        policy.disable_rule("r1")
        policy.enable_rule("r1")
        result = policy.evaluate("bad stuff")
        assert result.blocked

    def test_toggle_nonexistent_raises(self):
        policy = SafetyPolicy("test")
        with pytest.raises(KeyError):
            policy.enable_rule("nope")
        with pytest.raises(KeyError):
            policy.disable_rule("nope")


# ---------------------------------------------------------------------------
# TestBatch -- batch evaluation
# ---------------------------------------------------------------------------


class TestBatch:
    def test_evaluate_batch(self):
        policy = SafetyPolicy("test")
        policy.add_rule("r1", condition="contains", pattern="bad", action="block")
        results = policy.evaluate_batch(["good text", "bad text", "fine"])
        assert len(results) == 3
        assert results[0].passed
        assert results[1].blocked
        assert results[2].passed


# ---------------------------------------------------------------------------
# TestExport -- export / import roundtrip
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_import_roundtrip(self):
        policy = SafetyPolicy("roundtrip", mode="audit")
        policy.add_rule("r1", condition="contains", pattern="secret",
                        action="block", priority=10, message="no secrets")
        policy.add_rule("r2", condition="regex", pattern=r"\d{3}",
                        action="warn", priority=5)
        policy.disable_rule("r2")

        exported = policy.export()
        restored = SafetyPolicy.from_dict(exported)

        assert restored.name == "roundtrip"
        assert restored.mode == "audit"
        rules = restored.list_rules()
        assert len(rules) == 2
        assert rules[0].name == "r1"
        assert rules[0].priority == 10
        assert rules[1].name == "r2"
        assert not rules[1].enabled

        result = restored.evaluate("the secret code is 999")
        assert not result.blocked  # audit mode
        assert len(result.matches) == 1  # r2 disabled, only r1 matches


# ---------------------------------------------------------------------------
# TestMerge -- merging two policies
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_policies(self):
        policy_a = SafetyPolicy("a")
        policy_a.add_rule("shared", condition="contains", pattern="x",
                          action="warn", priority=1)
        policy_a.add_rule("only_a", condition="contains", pattern="a",
                          action="block", priority=5)

        policy_b = SafetyPolicy("b")
        policy_b.add_rule("shared", condition="contains", pattern="x",
                          action="block", priority=10)
        policy_b.add_rule("only_b", condition="regex", pattern=r"\d+",
                          action="warn", priority=3)

        merged = policy_a.merge(policy_b)
        assert merged.name == "a+b"
        rules = {r.name: r for r in merged.list_rules()}
        assert len(rules) == 3
        # "shared" should come from policy_b (higher priority)
        assert rules["shared"].priority == 10
        assert rules["shared"].action == "block"
        assert "only_a" in rules
        assert "only_b" in rules

    def test_merge_preserves_mode(self):
        policy_a = SafetyPolicy("a", mode="audit")
        policy_b = SafetyPolicy("b", mode="enforce")
        merged = policy_a.merge(policy_b)
        assert merged.mode == "audit"  # inherits from self


# ---------------------------------------------------------------------------
# TestEdge -- edge cases
# ---------------------------------------------------------------------------


class TestEdge:
    def test_empty_policy(self):
        policy = SafetyPolicy("empty")
        result = policy.evaluate("anything goes")
        assert result.passed
        assert not result.blocked
        assert result.matches == []
        assert result.action_taken == "allow"

    def test_empty_text(self):
        policy = SafetyPolicy("test")
        policy.add_rule("r1", condition="contains", pattern="bad", action="block")
        result = policy.evaluate("")
        assert result.passed

    def test_multiple_matches(self):
        policy = SafetyPolicy("test")
        policy.add_rule("r1", condition="contains", pattern="bad", action="warn", priority=2)
        policy.add_rule("r2", condition="contains", pattern="bad", action="block", priority=1)
        result = policy.evaluate("bad input")
        assert len(result.matches) == 2
        assert result.blocked
        assert result.action_taken == "block"

    def test_dataclass_fields(self):
        rule = PolicyRule(name="r", condition="contains", pattern="x",
                          action="block", priority=0, message="msg", enabled=True)
        assert rule.name == "r"

        match = RuleMatch(rule_name="r", action="block", message="msg", matched_text="x")
        assert match.rule_name == "r"

        evaluation = PolicyEvaluation(text="t", passed=True, matches=[], action_taken="allow", blocked=False)
        assert evaluation.text == "t"
