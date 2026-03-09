"""Tests for input guard."""

import pytest
from sentinel.input_guard import InputGuard, GuardResult, GuardReport, GuardRule


# ---------------------------------------------------------------------------
# Length checks
# ---------------------------------------------------------------------------

class TestLength:
    def test_max_length_pass_and_fail(self):
        guard = InputGuard().max_length(10)
        assert guard.validate("Short").passed
        result = guard.validate("This text is way too long")
        assert result.blocked
        assert "max_length" in result.rules_triggered

    def test_min_length_pass_and_fail(self):
        guard = InputGuard().min_length(5)
        assert guard.validate("Hello world").passed
        result = guard.validate("Hi")
        assert result.blocked
        assert "min_length" in result.rules_triggered


# ---------------------------------------------------------------------------
# Block patterns
# ---------------------------------------------------------------------------

class TestBlockPatterns:
    def test_matching_pattern_blocks(self):
        guard = InputGuard().block_patterns([r"<script", r"DROP TABLE"])
        assert guard.validate("<script>alert(1)</script>").blocked
        assert guard.validate("What is the weather?").passed

    def test_case_insensitive(self):
        guard = InputGuard().block_patterns([r"drop table"])
        assert guard.validate("DROP TABLE users").blocked


# ---------------------------------------------------------------------------
# Language requirement
# ---------------------------------------------------------------------------

class TestRequireLanguage:
    def test_english_passes(self):
        guard = InputGuard().require_language("en")
        assert guard.validate("The weather is very nice and the sun is shining").passed

    def test_non_english_blocks(self):
        guard = InputGuard().require_language("en")
        result = guard.validate("Le chat est sur la table avec les enfants")
        assert result.blocked
        assert "require_language" in result.rules_triggered


# ---------------------------------------------------------------------------
# Safe encoding
# ---------------------------------------------------------------------------

class TestSafeEncoding:
    def test_normal_text_passes(self):
        guard = InputGuard().safe_encoding()
        assert guard.validate("Normal text with tabs\tand newlines\n").passed

    def test_zero_width_char_blocks(self):
        guard = InputGuard().safe_encoding()
        assert guard.validate("Hello\u200BWorld").blocked


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------

class TestCustomRules:
    def test_custom_pass_and_fail(self):
        guard = InputGuard().custom(
            "must_have_question",
            lambda t: None if "?" in t else "Must contain a question mark",
        )
        assert guard.validate("What time is it?").passed
        result = guard.validate("Tell me the time")
        assert result.blocked
        assert "must_have_question" in result.rules_triggered


# ---------------------------------------------------------------------------
# Warn action
# ---------------------------------------------------------------------------

class TestWarnAction:
    def test_warn_does_not_block_but_records(self):
        guard = InputGuard().max_length(5, action="warn")
        result = guard.validate("This is longer than five")
        assert result.passed
        assert not result.blocked
        assert len(result.warnings) == 1
        assert "max_length" in result.warnings[0]
        assert "max_length" in result.rules_triggered


# ---------------------------------------------------------------------------
# Sanitize action
# ---------------------------------------------------------------------------

class TestSanitizeAction:
    def test_sanitize_strips_unsafe_chars(self):
        guard = InputGuard().safe_encoding(action="sanitize")
        result = guard.validate("Hello\u200BWorld")
        assert result.passed
        assert result.sanitized_text == "HelloWorld"
        assert "safe_encoding" in result.rules_triggered


# ---------------------------------------------------------------------------
# Fluent chaining
# ---------------------------------------------------------------------------

class TestChaining:
    def test_multiple_rules_all_pass(self):
        guard = (InputGuard()
                 .max_length(200)
                 .min_length(3)
                 .block_patterns([r"<script"])
                 .safe_encoding())
        assert guard.validate("Hello world, this is a safe input").passed
        assert guard.rule_count == 4

    def test_multiple_rules_with_block(self):
        guard = InputGuard().max_length(5).min_length(3)
        result = guard.validate("Way too long text here")
        assert result.blocked
        assert "max_length" in result.rules_triggered


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_returns_per_input_results(self):
        guard = InputGuard().max_length(10)
        results = guard.validate_batch(["short", "this is way too long", "ok"])
        assert len(results) == 3
        assert results[0].passed
        assert not results[1].passed
        assert results[2].passed


# ---------------------------------------------------------------------------
# Report / stats
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_counts_and_pass_rate(self):
        guard = InputGuard().max_length(5)
        guard.validate("ok")
        guard.validate("way too long text")
        guard.validate("hi")
        report = guard.report()
        assert isinstance(report, GuardReport)
        assert report.total_checked == 3
        assert report.blocked_count == 1
        assert report.pass_rate == pytest.approx(2 / 3)

    def test_report_warned_count(self):
        guard = InputGuard().max_length(5, action="warn")
        guard.validate("way too long")
        report = guard.report()
        assert report.warned_count == 1
        assert report.blocked_count == 0

    def test_reset_stats(self):
        guard = InputGuard().max_length(5)
        guard.validate("too long text")
        guard.reset_stats()
        report = guard.report()
        assert report.total_checked == 0
        assert report.blocked_count == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_rules_passes_anything(self):
        guard = InputGuard()
        result = guard.validate("anything at all")
        assert result.passed
        assert result.rules_triggered == []

    def test_empty_string_blocked_by_min_length(self):
        guard = InputGuard().min_length(1)
        assert guard.validate("").blocked

    def test_result_preserves_original_text(self):
        guard = InputGuard().safe_encoding(action="sanitize")
        original = "Hello\u200BWorld"
        result = guard.validate(original)
        assert result.text == original
        assert result.sanitized_text == "HelloWorld"

    def test_empty_report(self):
        guard = InputGuard()
        report = guard.report()
        assert report.total_checked == 0
        assert report.pass_rate == 0.0

    def test_guard_rule_dataclass(self):
        rule = GuardRule(
            name="test",
            check_fn=lambda t: None,
            action="block",
            description="A test rule",
        )
        assert rule.name == "test"
        assert rule.action == "block"
