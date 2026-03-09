"""Tests for prompt_validator module."""

import time

import pytest

from sentinel.prompt_validator import (
    PromptValidator,
    ValidationRule,
    ValidationIssue,
    ValidationReport,
    ValidatorStats,
)


# ---------------------------------------------------------------------------
# Clean prompt passes
# ---------------------------------------------------------------------------


class TestCleanPrompt:
    def test_clean_prompt_passes(self):
        validator = PromptValidator()
        report = validator.validate("What is the capital of France?")
        assert report.is_valid
        assert report.issues == []
        assert report.score == 1.0

    def test_report_has_checked_at_timestamp(self):
        validator = PromptValidator()
        before = time.time()
        report = validator.validate("Hello world")
        after = time.time()
        assert before <= report.checked_at <= after

    def test_report_contains_original_prompt(self):
        validator = PromptValidator()
        prompt = "Tell me a story"
        report = validator.validate(prompt)
        assert report.prompt == prompt


# ---------------------------------------------------------------------------
# Length checks
# ---------------------------------------------------------------------------


class TestLengthValidation:
    def test_too_short_prompt_fails(self):
        validator = PromptValidator(min_length=10)
        report = validator.validate("Hi")
        assert not report.is_valid
        assert any(i.rule_name == "min_length" for i in report.issues)

    def test_too_long_prompt_fails(self):
        validator = PromptValidator(max_length=20)
        report = validator.validate("A" * 25)
        assert not report.is_valid
        assert any(i.rule_name == "max_length" for i in report.issues)

    def test_prompt_at_exact_max_passes(self):
        validator = PromptValidator(max_length=10)
        report = validator.validate("A" * 10)
        assert report.is_valid

    def test_prompt_at_exact_min_passes(self):
        validator = PromptValidator(min_length=5)
        report = validator.validate("Hello")
        assert report.is_valid


# ---------------------------------------------------------------------------
# Empty / whitespace-only
# ---------------------------------------------------------------------------


class TestEmptyPrompt:
    def test_empty_string_fails(self):
        validator = PromptValidator()
        report = validator.validate("")
        assert not report.is_valid
        assert any(i.rule_name == "empty_check" for i in report.issues)

    def test_whitespace_only_fails(self):
        validator = PromptValidator()
        report = validator.validate("   \t\n  ")
        assert not report.is_valid
        assert any(i.rule_name == "empty_check" for i in report.issues)


# ---------------------------------------------------------------------------
# Blocked words
# ---------------------------------------------------------------------------


class TestBlockedWords:
    def test_blocked_word_detected(self):
        validator = PromptValidator(blocked_words=["hack", "exploit"])
        report = validator.validate("How to hack a system")
        assert not report.is_valid
        assert any(i.rule_name == "blocked_word" for i in report.issues)

    def test_blocked_word_case_insensitive(self):
        validator = PromptValidator(blocked_words=["hack"])
        report = validator.validate("HACK the planet")
        assert not report.is_valid

    def test_no_blocked_word_passes(self):
        validator = PromptValidator(blocked_words=["hack"])
        report = validator.validate("Tell me about cybersecurity")
        assert report.is_valid

    def test_multiple_blocked_words_all_reported(self):
        validator = PromptValidator(blocked_words=["hack", "exploit"])
        report = validator.validate("hack and exploit the system")
        blocked_issues = [i for i in report.issues if i.rule_name == "blocked_word"]
        assert len(blocked_issues) == 2


# ---------------------------------------------------------------------------
# Encoding check
# ---------------------------------------------------------------------------


class TestEncodingCheck:
    def test_control_character_detected(self):
        validator = PromptValidator()
        report = validator.validate("Hello\x01World")
        assert not report.is_valid
        assert any(i.rule_name == "encoding_check" for i in report.issues)

    def test_zero_width_character_detected(self):
        validator = PromptValidator()
        report = validator.validate("Hello\u200BWorld")
        zero_width_issues = [
            i for i in report.issues if i.rule_name == "encoding_check"
        ]
        assert len(zero_width_issues) == 1
        assert zero_width_issues[0].severity == "warning"

    def test_normal_whitespace_passes(self):
        validator = PromptValidator()
        report = validator.validate("Hello\tWorld\nLine 2")
        encoding_issues = [i for i in report.issues if i.rule_name == "encoding_check"]
        assert encoding_issues == []


# ---------------------------------------------------------------------------
# Excessive repetition
# ---------------------------------------------------------------------------


class TestExcessiveRepetition:
    def test_repetition_detected(self):
        validator = PromptValidator()
        repeated = " ".join(["please"] * 15)
        report = validator.validate(repeated)
        assert any(i.rule_name == "repetition_check" for i in report.issues)

    def test_repetition_below_threshold_passes(self):
        validator = PromptValidator()
        prompt = " ".join(["word"] * 9)
        report = validator.validate(prompt)
        repetition_issues = [
            i for i in report.issues if i.rule_name == "repetition_check"
        ]
        assert repetition_issues == []

    def test_repetition_severity_is_warning(self):
        validator = PromptValidator()
        repeated = " ".join(["stop"] * 12)
        report = validator.validate(repeated)
        repetition_issues = [
            i for i in report.issues if i.rule_name == "repetition_check"
        ]
        assert repetition_issues[0].severity == "warning"


# ---------------------------------------------------------------------------
# Custom pattern rules
# ---------------------------------------------------------------------------


class TestCustomPatternRules:
    def test_pattern_rule_matches(self):
        validator = PromptValidator()
        rule = ValidationRule(
            name="no_email",
            check="pattern",
            params={
                "regex": r"[\w.+-]+@[\w-]+\.[\w.]+",
                "severity": "warning",
                "message": "Email address detected",
            },
        )
        validator.add_rule(rule)
        report = validator.validate("Contact me at user@example.com")
        assert any(i.rule_name == "no_email" for i in report.issues)

    def test_pattern_rule_no_match_passes(self):
        validator = PromptValidator()
        rule = ValidationRule(
            name="no_email",
            check="pattern",
            params={"regex": r"[\w.+-]+@[\w-]+\.[\w.]+"},
        )
        validator.add_rule(rule)
        report = validator.validate("No email here")
        assert not any(i.rule_name == "no_email" for i in report.issues)

    def test_required_field_rule(self):
        validator = PromptValidator()
        rule = ValidationRule(
            name="needs_context",
            check="required_field",
            params={"fields": ["Context:", "Question:"]},
        )
        validator.add_rule(rule)
        report = validator.validate("Context: some context\nQuestion: what?")
        required_issues = [
            i for i in report.issues if i.rule_name == "needs_context"
        ]
        assert required_issues == []

    def test_required_field_missing(self):
        validator = PromptValidator()
        rule = ValidationRule(
            name="needs_context",
            check="required_field",
            params={"fields": ["Context:", "Question:"]},
        )
        validator.add_rule(rule)
        report = validator.validate("Just a plain prompt")
        assert not report.is_valid
        required_issues = [
            i for i in report.issues if i.rule_name == "needs_context"
        ]
        assert len(required_issues) == 2

    def test_format_rule_json(self):
        validator = PromptValidator()
        rule = ValidationRule(
            name="json_format",
            check="format",
            params={"format": "json"},
        )
        validator.add_rule(rule)
        report = validator.validate('{"key": "value"}')
        format_issues = [i for i in report.issues if i.rule_name == "json_format"]
        assert format_issues == []

    def test_format_rule_json_fails_for_plain_text(self):
        validator = PromptValidator()
        rule = ValidationRule(
            name="json_format",
            check="format",
            params={"format": "json"},
        )
        validator.add_rule(rule)
        report = validator.validate("Just plain text")
        assert not report.is_valid

    def test_invalid_check_type_raises(self):
        with pytest.raises(ValueError, match="Invalid check type"):
            ValidationRule(name="bad", check="nonexistent")


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


class TestBatchValidation:
    def test_batch_returns_list(self):
        validator = PromptValidator()
        reports = validator.validate_batch(["Hello", "World"])
        assert isinstance(reports, list)
        assert len(reports) == 2

    def test_batch_mixed_results(self):
        validator = PromptValidator(max_length=10, blocked_words=["bad"])
        prompts = ["Good", "This is way too long for the validator", "bad word"]
        reports = validator.validate_batch(prompts)
        assert reports[0].is_valid
        assert not reports[1].is_valid
        assert not reports[2].is_valid


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_initial(self):
        validator = PromptValidator()
        s = validator.stats()
        assert s.total_validated == 0
        assert s.passed == 0
        assert s.failed == 0
        assert s.pass_rate == 0.0

    def test_stats_after_validations(self):
        validator = PromptValidator(blocked_words=["bad"])
        validator.validate("Good prompt")
        validator.validate("Another good one")
        validator.validate("This is bad")
        s = validator.stats()
        assert s.total_validated == 3
        assert s.passed == 2
        assert s.failed == 1
        assert s.pass_rate == pytest.approx(2 / 3)

    def test_stats_issues_by_severity(self):
        validator = PromptValidator(blocked_words=["bad"])
        repeated = " ".join(["bad"] * 15)
        validator.validate(repeated)
        s = validator.stats()
        assert s.issues_by_severity.get("error", 0) >= 1
        assert s.issues_by_severity.get("warning", 0) >= 1

    def test_stats_returns_copy(self):
        validator = PromptValidator()
        validator.validate("Hello")
        stats1 = validator.stats()
        validator.validate("World")
        stats2 = validator.stats()
        assert stats1.total_validated == 1
        assert stats2.total_validated == 2


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------


class TestScoreCalculation:
    def test_perfect_score(self):
        validator = PromptValidator()
        report = validator.validate("A perfectly normal prompt")
        assert report.score == 1.0

    def test_score_reduced_by_error(self):
        validator = PromptValidator(max_length=5)
        report = validator.validate("This prompt is too long")
        assert report.score < 1.0
        assert report.score == pytest.approx(0.8)

    def test_score_reduced_by_warning(self):
        validator = PromptValidator()
        repeated = " ".join(["word"] * 12)
        report = validator.validate(repeated)
        assert report.score == pytest.approx(0.9)

    def test_score_floors_at_zero(self):
        validator = PromptValidator(blocked_words=["a", "b", "c", "d", "e", "f"])
        report = validator.validate("a b c d e f")
        assert report.score == 0.0

    def test_score_with_info_severity(self):
        validator = PromptValidator()
        rule = ValidationRule(
            name="markdown_check",
            check="format",
            params={"format": "markdown"},
        )
        validator.add_rule(rule)
        report = validator.validate("No markdown here")
        info_issues = [i for i in report.issues if i.severity == "info"]
        assert len(info_issues) == 1
        assert report.score == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unicode_prompt_passes(self):
        validator = PromptValidator()
        report = validator.validate("Bonjour le monde! Hola mundo! \u4f60\u597d\u4e16\u754c!")
        assert report.is_valid

    def test_very_long_prompt_fails(self):
        validator = PromptValidator(max_length=10000)
        report = validator.validate("x" * 20000)
        assert not report.is_valid

    def test_newlines_and_tabs_pass(self):
        validator = PromptValidator()
        report = validator.validate("Line 1\nLine 2\tTabbed")
        assert report.is_valid

    def test_custom_blocked_word_rule(self):
        validator = PromptValidator()
        rule = ValidationRule(
            name="extra_blocks",
            check="blocked_word",
            params={"words": ["forbidden"]},
        )
        validator.add_rule(rule)
        report = validator.validate("This is forbidden content")
        assert not report.is_valid

    def test_encoding_rule_ascii_charset(self):
        validator = PromptValidator()
        rule = ValidationRule(
            name="ascii_only",
            check="encoding",
            params={"charset": "ascii"},
        )
        validator.add_rule(rule)
        report_ascii = validator.validate("Hello world")
        assert not any(i.rule_name == "ascii_only" for i in report_ascii.issues)

        report_unicode = validator.validate("Caf\u00e9")
        assert any(i.rule_name == "ascii_only" for i in report_unicode.issues)

    def test_dataclass_fields_accessible(self):
        issue = ValidationIssue(
            rule_name="test", severity="error", message="test msg"
        )
        assert issue.rule_name == "test"
        assert issue.severity == "error"
        assert issue.message == "test msg"

        stats = ValidatorStats()
        assert stats.total_validated == 0
        assert stats.issues_by_severity == {}
