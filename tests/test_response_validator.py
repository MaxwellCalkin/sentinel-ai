"""Tests for LLM response validation."""

import pytest
from sentinel.response_validator import (
    ResponseValidator,
    ResponseValidation,
    ResponseCheck,
    CheckResult,
    ResponseValidatorStats,
)


# ---------------------------------------------------------------------------
# JSON format validation
# ---------------------------------------------------------------------------

class TestJsonValidation:
    def test_valid_json_and_code_block_pass(self):
        v = ResponseValidator().require_json()
        assert v.validate('{"name": "Alice", "age": 30}').passed
        assert v.validate('```json\n{"key": "value"}\n```').passed

    def test_invalid_json_fails(self):
        v = ResponseValidator().require_json()
        result = v.validate("This is plain text, not JSON")
        assert not result.passed
        assert result.failed_required == 1


# ---------------------------------------------------------------------------
# List and code block format validation
# ---------------------------------------------------------------------------

class TestFormatValidation:
    def test_bulleted_and_numbered_lists_detected(self):
        v = ResponseValidator().require_list_format()
        assert v.validate("Items:\n- Item one\n- Item two").passed
        assert v.validate("Steps:\n1. First step\n2. Second step").passed

    def test_no_list_fails(self):
        v = ResponseValidator().require_list_format()
        assert not v.validate("Just a paragraph of plain text.").passed

    def test_code_block_present_and_absent(self):
        v = ResponseValidator().require_code_block()
        assert v.validate("Example:\n```python\nprint('hi')\n```").passed
        assert not v.validate("No code block here.").passed


# ---------------------------------------------------------------------------
# Length bounds
# ---------------------------------------------------------------------------

class TestLengthBounds:
    def test_within_bounds_passes(self):
        v = ResponseValidator().set_length_bounds(min_chars=5, max_chars=100)
        assert v.validate("This response is within bounds.").passed

    def test_below_min_fails(self):
        v = ResponseValidator().set_length_bounds(min_chars=50)
        result = v.validate("Short")
        assert not result.passed
        assert result.failed_required >= 1

    def test_above_max_fails(self):
        v = ResponseValidator().set_length_bounds(max_chars=10)
        assert not v.validate("This exceeds the maximum length.").passed


# ---------------------------------------------------------------------------
# Keyword presence and absence
# ---------------------------------------------------------------------------

class TestKeywordChecks:
    def test_required_keywords_present(self):
        v = ResponseValidator().require_keywords(["summary", "conclusion"])
        text = "In summary, growth continues. In conclusion, invest."
        assert v.validate(text).passed

    def test_required_keyword_missing(self):
        v = ResponseValidator().require_keywords(["disclaimer"])
        assert not v.validate("Here is the answer.").passed

    def test_keyword_case_insensitive(self):
        v = ResponseValidator().require_keywords(["Summary"])
        assert v.validate("in summary, here are the findings").passed

    def test_forbidden_keyword_checks(self):
        v = ResponseValidator().forbid_keywords(["password", "secret"])
        assert v.validate("Uses environment variables.").passed
        assert not v.validate("The password is stored in plain text.").passed


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    def test_all_checks_pass_full_confidence(self):
        v = ResponseValidator().set_length_bounds(min_chars=5).require_keywords(["hello"])
        result = v.validate("hello world, this is a test response")
        assert result.confidence == 1.0

    def test_partial_failure_reduces_confidence(self):
        v = ResponseValidator()
        v.add_check("always_pass", lambda t: True, required=True)
        v.add_check("always_fail", lambda t: False, required=False)
        result = v.validate("test")
        assert 0.0 < result.confidence < 1.0
        assert round(result.confidence, 4) == round(2 / 3, 4)

    def test_no_checks_returns_full_confidence(self):
        v = ResponseValidator()
        assert v.validate("anything").confidence == 1.0


# ---------------------------------------------------------------------------
# Custom validation rules
# ---------------------------------------------------------------------------

class TestCustomRules:
    def test_custom_check_pass_and_fail(self):
        v = ResponseValidator()
        v.add_check("has_greeting", lambda t: "hello" in t.lower())
        assert v.validate("Hello, how can I help?").passed
        v2 = ResponseValidator()
        v2.add_check("has_greeting", lambda t: "hello" in t.lower())
        assert not v2.validate("Goodbye.").passed

    def test_exception_in_check_treated_as_failure(self):
        v = ResponseValidator()
        v.add_check("bad_check", lambda t: 1 / 0)
        result = v.validate("test")
        assert not result.passed
        assert result.failed_required == 1


# ---------------------------------------------------------------------------
# Batch validation and statistics
# ---------------------------------------------------------------------------

class TestBatchAndStats:
    def test_batch_returns_correct_count(self):
        v = ResponseValidator().set_length_bounds(min_chars=3)
        results = v.validate_batch(["hello", "hi", "good morning"])
        assert len(results) == 3
        assert results[0].passed
        assert not results[1].passed
        assert results[2].passed

    def test_stats_after_validations(self):
        v = ResponseValidator().set_length_bounds(min_chars=5)
        v.validate("long enough text")
        v.validate("no")
        v.validate("also long enough")
        stats = v.stats()
        assert stats.total_validated == 3
        assert stats.pass_rate == pytest.approx(2 / 3, abs=0.01)
        assert 0.0 < stats.avg_confidence <= 1.0

    def test_stats_empty(self):
        stats = ResponseValidator().stats()
        assert stats.total_validated == 0
        assert stats.pass_rate == 0.0
        assert stats.avg_confidence == 0.0


# ---------------------------------------------------------------------------
# Combined / integration
# ---------------------------------------------------------------------------

class TestCombinedValidation:
    def test_fluent_chaining_check_count(self):
        v = (ResponseValidator()
             .require_json()
             .set_length_bounds(min_chars=5, max_chars=500))
        assert v.check_count == 3

    def test_required_vs_optional_failure_counts(self):
        v = ResponseValidator()
        v.add_check("required_a", lambda t: False, required=True)
        v.add_check("optional_b", lambda t: False, required=False)
        result = v.validate("test")
        assert result.failed_required == 1
        assert result.failed_optional == 1
        assert not result.passed
