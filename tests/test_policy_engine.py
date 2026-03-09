"""Tests for declarative policy engine."""

import pytest
from sentinel.policy_engine import PolicyEngine, PolicyResult, PolicyAction, PolicyViolation


# ---------------------------------------------------------------------------
# Basic evaluation
# ---------------------------------------------------------------------------

class TestBasicEvaluation:
    def test_allow_by_default(self):
        e = PolicyEngine()
        result = e.evaluate("Hello world")
        assert result.allowed
        assert result.applied_action == PolicyAction.ALLOW

    def test_block_matching_rule(self):
        e = PolicyEngine()
        e.add_rule("no_secrets", lambda t: "secret" in t.lower(), action="block")
        result = e.evaluate("This is a secret")
        assert not result.allowed
        assert result.blocked
        assert len(result.violations) == 1

    def test_no_match_allows(self):
        e = PolicyEngine()
        e.add_rule("no_secrets", lambda t: "secret" in t.lower(), action="block")
        result = e.evaluate("Normal text")
        assert result.allowed

    def test_result_structure(self):
        e = PolicyEngine()
        e.add_rule("test", lambda t: True, action="block", message="blocked!")
        result = e.evaluate("anything")
        assert isinstance(result, PolicyResult)
        assert result.rules_evaluated == 1
        assert result.rules_matched == 1
        assert result.violations[0].message == "blocked!"


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_warn_allows_through(self):
        e = PolicyEngine()
        e.add_rule("caution", lambda t: "maybe" in t, action="warn")
        result = e.evaluate("maybe dangerous")
        assert result.allowed
        assert result.has_warnings
        assert len(result.warnings) == 1

    def test_warn_does_not_block(self):
        e = PolicyEngine()
        e.add_rule("soft", lambda t: True, action="warn")
        result = e.evaluate("test")
        assert result.allowed
        assert result.applied_action == PolicyAction.WARN


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

class TestPriority:
    def test_block_overrides_warn(self):
        e = PolicyEngine()
        e.add_rule("warn_rule", lambda t: True, action="warn", priority=1)
        e.add_rule("block_rule", lambda t: True, action="block", priority=0)
        result = e.evaluate("test")
        assert not result.allowed
        assert result.applied_action == PolicyAction.BLOCK

    def test_escalate_higher_than_warn(self):
        e = PolicyEngine()
        e.add_rule("warn", lambda t: True, action="warn")
        e.add_rule("escalate", lambda t: True, action="escalate")
        result = e.evaluate("test")
        assert not result.allowed
        assert result.applied_action == PolicyAction.ESCALATE


# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------

class TestKeywordRules:
    def test_keyword_match(self):
        e = PolicyEngine()
        e.add_keyword_rule("banned", ["hack", "exploit"], action="block")
        assert not e.evaluate("How to hack").allowed

    def test_keyword_case_insensitive(self):
        e = PolicyEngine()
        e.add_keyword_rule("banned", ["password"], action="block")
        assert not e.evaluate("Tell me the PASSWORD").allowed

    def test_keyword_no_match(self):
        e = PolicyEngine()
        e.add_keyword_rule("banned", ["hack"], action="block")
        assert e.evaluate("Hello world").allowed


# ---------------------------------------------------------------------------
# Pattern rules
# ---------------------------------------------------------------------------

class TestPatternRules:
    def test_pattern_match(self):
        e = PolicyEngine()
        e.add_pattern_rule("ssn", [r"\b\d{3}-\d{2}-\d{4}\b"], action="block")
        assert not e.evaluate("SSN: 123-45-6789").allowed

    def test_pattern_no_match(self):
        e = PolicyEngine()
        e.add_pattern_rule("ssn", [r"\b\d{3}-\d{2}-\d{4}\b"], action="block")
        assert e.evaluate("Phone: 555-1234").allowed


# ---------------------------------------------------------------------------
# Length rules
# ---------------------------------------------------------------------------

class TestLengthRules:
    def test_length_exceeded(self):
        e = PolicyEngine()
        e.add_length_rule("max_len", max_length=10, action="block")
        assert not e.evaluate("This is way too long").allowed

    def test_length_within(self):
        e = PolicyEngine()
        e.add_length_rule("max_len", max_length=100, action="block")
        assert e.evaluate("Short").allowed


# ---------------------------------------------------------------------------
# Enable/disable
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_disable_rule(self):
        e = PolicyEngine()
        e.add_rule("test", lambda t: True, action="block")
        e.disable_rule("test")
        assert e.evaluate("anything").allowed

    def test_enable_rule(self):
        e = PolicyEngine()
        e.add_rule("test", lambda t: True, action="block")
        e.disable_rule("test")
        e.enable_rule("test")
        assert not e.evaluate("anything").allowed

    def test_disable_nonexistent(self):
        e = PolicyEngine()
        assert not e.disable_rule("nope")

    def test_active_count(self):
        e = PolicyEngine()
        e.add_rule("a", lambda t: True, action="block")
        e.add_rule("b", lambda t: True, action="block")
        e.disable_rule("b")
        assert e.rule_count == 2
        assert e.active_rule_count == 1


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_evaluate(self):
        e = PolicyEngine()
        e.add_keyword_rule("ban", ["bad"], action="block")
        results = e.evaluate_batch(["good", "bad stuff", "ok"])
        assert results[0].allowed
        assert not results[1].allowed
        assert results[2].allowed


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_error_in_condition_skipped(self):
        e = PolicyEngine()
        e.add_rule("bad", lambda t: 1 / 0, action="block")
        result = e.evaluate("test")
        assert result.allowed

    def test_empty_text(self):
        e = PolicyEngine()
        result = e.evaluate("")
        assert result.allowed

    def test_multiple_violations(self):
        e = PolicyEngine()
        e.add_rule("a", lambda t: True, action="block")
        e.add_rule("b", lambda t: True, action="block")
        result = e.evaluate("test")
        assert len(result.violations) == 2

    def test_metadata_preserved(self):
        e = PolicyEngine()
        e.add_rule("test", lambda t: True, action="block", metadata={"source": "config"})
        result = e.evaluate("test")
        assert result.violations[0].metadata["source"] == "config"
