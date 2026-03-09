"""Tests for configurable output censoring."""

import pytest
from sentinel.output_censor import OutputCensor, CensorRule, CensorResult, CensorStats


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_redacts_matching_pattern(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="ssn", pattern=r"\d{3}-\d{2}-\d{4}", action="redact"))
        result = censor.censor("My SSN is 123-45-6789")
        assert result.censored == "My SSN is [REDACTED]"
        assert not result.blocked
        assert "ssn" in result.rules_applied
        assert result.original == "My SSN is 123-45-6789"

    def test_redacts_multiple_occurrences(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="ssn", pattern=r"\d{3}-\d{2}-\d{4}", action="redact"))
        result = censor.censor("SSNs: 123-45-6789 and 987-65-4321")
        assert result.censored == "SSNs: [REDACTED] and [REDACTED]"
        assert result.redaction_count == 2


# ---------------------------------------------------------------------------
# Replacement
# ---------------------------------------------------------------------------

class TestReplacement:
    def test_replaces_with_custom_text(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(
            name="profanity", pattern=r"\bbadword\b", action="replace", replacement="***"
        ))
        result = censor.censor("This is a badword here")
        assert result.censored == "This is a *** here"
        assert "profanity" in result.rules_applied

    def test_replaces_multiple_matches(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(
            name="curse", pattern=r"\bdarn\b", action="replace", replacement="[BLEEP]"
        ))
        result = censor.censor("darn it, oh darn")
        assert result.censored == "[BLEEP] it, oh [BLEEP]"
        assert result.redaction_count == 2


# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------

class TestBlocking:
    def test_block_returns_empty_string_with_flag(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="secret", pattern=r"TOP SECRET", action="block"))
        result = censor.censor("This is TOP SECRET information")
        assert result.censored == ""
        assert result.blocked is True
        assert "secret" in result.rules_applied

    def test_block_takes_precedence_over_later_rules(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="blocker", pattern=r"classified", action="block"))
        censor.add_rule(CensorRule(name="redactor", pattern=r"\d+", action="redact"))
        result = censor.censor("classified document 42")
        assert result.blocked is True
        assert result.censored == ""

    def test_no_block_when_pattern_absent(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="blocker", pattern=r"classified", action="block"))
        result = censor.censor("This is a normal message")
        assert result.blocked is False
        assert result.censored == "This is a normal message"


# ---------------------------------------------------------------------------
# Multiple rules (chaining)
# ---------------------------------------------------------------------------

class TestRuleChaining:
    def test_applies_rules_in_registration_order(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="email", pattern=r"\S+@\S+\.\S+", action="redact"))
        censor.add_rule(CensorRule(name="phone", pattern=r"\d{3}-\d{3}-\d{4}", action="redact"))
        result = censor.censor("Email user@test.com, call 555-123-4567")
        assert "user@test.com" not in result.censored
        assert "555-123-4567" not in result.censored
        assert result.rules_applied == ["email", "phone"]

    def test_only_triggered_rules_appear_in_result(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="email", pattern=r"\S+@\S+\.\S+", action="redact"))
        censor.add_rule(CensorRule(name="phone", pattern=r"\d{3}-\d{3}-\d{4}", action="redact"))
        result = censor.censor("Email user@test.com please")
        assert result.rules_applied == ["email"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_tracks_processed_and_redactions(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="digits", pattern=r"\d+", action="redact"))
        censor.censor("abc 123 def 456")
        censor.censor("no digits")
        assert censor.stats.total_processed == 2
        assert censor.stats.total_redactions == 2

    def test_tracks_blocks_and_rules_triggered(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="ban", pattern=r"forbidden", action="block"))
        censor.censor("this is forbidden")
        censor.censor("this is fine")
        assert censor.stats.total_blocks == 1
        assert censor.stats.rules_triggered["ban"] == 1

    def test_reset_stats(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="d", pattern=r"\d+", action="redact"))
        censor.censor("123")
        censor.reset_stats()
        assert censor.stats.total_processed == 0
        assert censor.stats.total_redactions == 0
        assert censor.stats.rules_triggered == {}


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_processes_all_texts(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="digits", pattern=r"\d+", action="redact"))
        results = censor.censor_batch(["abc 123", "def", "789 ghi"])
        assert len(results) == 3
        assert results[0].redaction_count == 1
        assert results[1].redaction_count == 0
        assert results[2].redaction_count == 1

    def test_empty_batch(self):
        censor = OutputCensor()
        assert censor.censor_batch([]) == []


# ---------------------------------------------------------------------------
# Rule management
# ---------------------------------------------------------------------------

class TestRuleManagement:
    def test_remove_rule(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="digits", pattern=r"\d+", action="redact"))
        assert censor.remove_rule("digits") is True
        assert censor.remove_rule("digits") is False
        result = censor.censor("abc 123")
        assert result.censored == "abc 123"

    def test_rules_property_returns_copy(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="test", pattern=r"x", action="redact"))
        rules_copy = censor.rules
        rules_copy.clear()
        assert len(censor.rules) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_rejects_invalid_action(self):
        censor = OutputCensor()
        with pytest.raises(ValueError, match="Invalid action"):
            censor.add_rule(CensorRule(name="bad", pattern=r"x", action="destroy"))

    def test_rejects_empty_name_and_invalid_regex(self):
        censor = OutputCensor()
        with pytest.raises(ValueError, match="name must not be empty"):
            censor.add_rule(CensorRule(name="", pattern=r"x", action="redact"))
        with pytest.raises(ValueError, match="Invalid regex"):
            censor.add_rule(CensorRule(name="bad", pattern=r"[invalid", action="redact"))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="digits", pattern=r"\d+", action="redact"))
        result = censor.censor("")
        assert result.censored == ""
        assert result.blocked is False
        assert result.redaction_count == 0

    def test_no_rules_returns_text_unchanged(self):
        censor = OutputCensor()
        result = censor.censor("hello world")
        assert result.censored == "hello world"
        assert result.rules_applied == []

    def test_case_insensitive_with_inline_flag(self):
        censor = OutputCensor()
        censor.add_rule(CensorRule(name="secret", pattern=r"(?i)secret", action="redact"))
        result = censor.censor("This is SECRET information")
        assert result.censored == "This is [REDACTED] information"
        assert result.redaction_count == 1
