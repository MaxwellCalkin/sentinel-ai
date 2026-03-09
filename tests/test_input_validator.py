"""Tests for input validator."""

import pytest
from sentinel.input_validator import InputValidator, InputValidationResult, InputIssue


# ---------------------------------------------------------------------------
# Length checks
# ---------------------------------------------------------------------------

class TestLength:
    def test_max_length_pass(self):
        v = InputValidator().max_length(100)
        result = v.validate("Short text")
        assert result.valid

    def test_max_length_fail(self):
        v = InputValidator().max_length(5)
        result = v.validate("Too long text")
        assert not result.valid
        assert result.errors[0].rule == "max_length"

    def test_min_length_pass(self):
        v = InputValidator().min_length(3)
        assert v.validate("Hello").valid

    def test_min_length_fail(self):
        v = InputValidator().min_length(10)
        assert not v.validate("Hi").valid

    def test_no_empty_pass(self):
        v = InputValidator().no_empty()
        assert v.validate("content").valid

    def test_no_empty_fail(self):
        v = InputValidator().no_empty()
        assert not v.validate("   ").valid
        assert not v.validate("").valid


# ---------------------------------------------------------------------------
# Code injection
# ---------------------------------------------------------------------------

class TestCodeInjection:
    def test_script_tag(self):
        v = InputValidator().no_code_injection()
        assert not v.validate("<script>alert('xss')</script>").valid

    def test_javascript_uri(self):
        v = InputValidator().no_code_injection()
        assert not v.validate("javascript:void(0)").valid

    def test_event_handler(self):
        v = InputValidator().no_code_injection()
        assert not v.validate('onerror="alert(1)"').valid

    def test_sql_injection(self):
        v = InputValidator().no_code_injection()
        assert not v.validate("DROP TABLE users").valid

    def test_clean_text(self):
        v = InputValidator().no_code_injection()
        assert v.validate("What is the weather today?").valid


# ---------------------------------------------------------------------------
# Encoding safety
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_zero_width_char(self):
        v = InputValidator().safe_encoding()
        assert not v.validate("Hello\u200BWorld").valid

    def test_control_char(self):
        v = InputValidator().safe_encoding()
        assert not v.validate("Hello\x01World").valid

    def test_normal_whitespace_ok(self):
        v = InputValidator().safe_encoding()
        assert v.validate("Hello\tWorld\nLine2").valid

    def test_clean_text(self):
        v = InputValidator().safe_encoding()
        assert v.validate("Normal text here").valid


# ---------------------------------------------------------------------------
# Blocked words
# ---------------------------------------------------------------------------

class TestBlockedWords:
    def test_blocked_word(self):
        v = InputValidator().blocked_words(["hack", "exploit"])
        assert not v.validate("How to hack a system").valid

    def test_case_insensitive(self):
        v = InputValidator().blocked_words(["hack"])
        assert not v.validate("HACK the planet").valid

    def test_no_match(self):
        v = InputValidator().blocked_words(["hack"])
        assert v.validate("Hello world").valid


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

class TestLines:
    def test_max_lines_pass(self):
        v = InputValidator().max_lines(5)
        assert v.validate("Line1\nLine2\nLine3").valid

    def test_max_lines_fail(self):
        v = InputValidator().max_lines(2)
        assert not v.validate("L1\nL2\nL3").valid


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

class TestRegex:
    def test_regex_match_pass(self):
        v = InputValidator().regex_match(r'\d+')
        assert v.validate("I have 42 items").valid

    def test_regex_match_fail(self):
        v = InputValidator().regex_match(r'\d+', message="Number required")
        result = v.validate("No numbers here")
        assert not result.valid
        assert result.errors[0].message == "Number required"

    def test_regex_deny(self):
        v = InputValidator().regex_deny(r'\b\d{3}-\d{2}-\d{4}\b', "SSN detected")
        assert not v.validate("SSN: 123-45-6789").valid

    def test_regex_deny_clean(self):
        v = InputValidator().regex_deny(r'\b\d{3}-\d{2}-\d{4}\b')
        assert v.validate("Phone: 555-1234").valid


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------

class TestCustom:
    def test_custom_pass(self):
        v = InputValidator().custom("has_question", lambda t: "?" in t, "Must be a question")
        assert v.validate("What time is it?").valid

    def test_custom_fail(self):
        v = InputValidator().custom("has_question", lambda t: "?" in t, "Must be a question")
        result = v.validate("Tell me the time")
        assert not result.valid
        assert result.errors[0].message == "Must be a question"


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------

class TestChaining:
    def test_multiple_rules(self):
        v = (InputValidator()
             .no_empty()
             .max_length(100)
             .no_code_injection()
             .safe_encoding())
        assert v.validate("Normal question here").valid
        assert v.rule_count == 4

    def test_multiple_failures(self):
        v = InputValidator().max_length(5).no_empty()
        result = v.validate("This is too long")
        assert not result.valid
        assert result.checks_run == 2


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_validate(self):
        v = InputValidator().max_length(10)
        results = v.validate_batch(["short", "this is way too long", "ok"])
        assert results[0].valid
        assert not results[1].valid
        assert results[2].valid


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_rules(self):
        v = InputValidator()
        result = v.validate("anything")
        assert result.valid
        assert result.checks_run == 0

    def test_result_structure(self):
        v = InputValidator().max_length(5)
        result = v.validate("toolong")
        assert isinstance(result, InputValidationResult)
        assert len(result.errors) == 1
        assert len(result.warnings) == 0
