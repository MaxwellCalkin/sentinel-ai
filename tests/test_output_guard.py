"""Tests for output constraint enforcement."""

import pytest
from sentinel.output_guard import OutputGuard, OutputCheckResult, OutputViolation


# ---------------------------------------------------------------------------
# Length constraints
# ---------------------------------------------------------------------------

class TestLengthConstraints:
    def test_max_length_pass(self):
        g = OutputGuard().max_length(100)
        result = g.check("Short text")
        assert result.passed

    def test_max_length_fail(self):
        g = OutputGuard().max_length(10)
        result = g.check("This text is too long for the constraint")
        assert not result.passed
        assert result.violations[0].rule == "max_length"

    def test_min_length_pass(self):
        g = OutputGuard().min_length(5)
        result = g.check("Hello world")
        assert result.passed

    def test_min_length_fail(self):
        g = OutputGuard().min_length(50)
        result = g.check("Short")
        assert not result.passed

    def test_max_words_pass(self):
        g = OutputGuard().max_words(10)
        result = g.check("One two three")
        assert result.passed

    def test_max_words_fail(self):
        g = OutputGuard().max_words(3)
        result = g.check("This sentence has more than three words")
        assert not result.passed


# ---------------------------------------------------------------------------
# Banned phrases
# ---------------------------------------------------------------------------

class TestBannedPhrases:
    def test_ban_detected(self):
        g = OutputGuard().ban_phrases(["I cannot", "As an AI"])
        result = g.check("I cannot help with that request.")
        assert not result.passed
        assert "banned phrase" in result.violations[0].message

    def test_ban_case_insensitive(self):
        g = OutputGuard().ban_phrases(["as an ai"])
        result = g.check("As An AI, I must decline.")
        assert not result.passed

    def test_ban_not_found(self):
        g = OutputGuard().ban_phrases(["I cannot"])
        result = g.check("Here is the information you requested.")
        assert result.passed


# ---------------------------------------------------------------------------
# Required phrases
# ---------------------------------------------------------------------------

class TestRequiredPhrases:
    def test_required_present(self):
        g = OutputGuard().require_phrases(["disclaimer"])
        result = g.check("Please note this disclaimer: results may vary.")
        assert result.passed

    def test_required_missing(self):
        g = OutputGuard().require_phrases(["disclaimer"])
        result = g.check("Here is the answer without any caveats.")
        assert not result.passed


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

class TestFormatValidation:
    def test_json_valid(self):
        g = OutputGuard().require_format("json")
        result = g.check('{"name": "test", "value": 42}')
        assert result.passed

    def test_json_in_code_block(self):
        g = OutputGuard().require_format("json")
        result = g.check('```json\n{"name": "test"}\n```')
        assert result.passed

    def test_json_invalid(self):
        g = OutputGuard().require_format("json")
        result = g.check("This is not JSON at all")
        assert not result.passed

    def test_xml_valid(self):
        g = OutputGuard().require_format("xml")
        result = g.check("<root><item>test</item></root>")
        assert result.passed

    def test_xml_invalid(self):
        g = OutputGuard().require_format("xml")
        result = g.check("Just plain text")
        assert not result.passed


# ---------------------------------------------------------------------------
# No empty
# ---------------------------------------------------------------------------

class TestNoEmpty:
    def test_non_empty_passes(self):
        g = OutputGuard().no_empty()
        result = g.check("Content here")
        assert result.passed

    def test_empty_fails(self):
        g = OutputGuard().no_empty()
        result = g.check("")
        assert not result.passed

    def test_whitespace_fails(self):
        g = OutputGuard().no_empty()
        result = g.check("   \n\n  ")
        assert not result.passed


# ---------------------------------------------------------------------------
# Regex rules
# ---------------------------------------------------------------------------

class TestRegex:
    def test_regex_match_pass(self):
        g = OutputGuard().regex_match(r"\d{3}-\d{4}", "phone number")
        result = g.check("Call us at 555-1234")
        assert result.passed

    def test_regex_match_fail(self):
        g = OutputGuard().regex_match(r"\d{3}-\d{4}", "phone number")
        result = g.check("No phone number here")
        assert not result.passed

    def test_regex_deny_pass(self):
        g = OutputGuard().regex_deny(r"password\s*[:=]", "password leak")
        result = g.check("Secure content here")
        assert result.passed

    def test_regex_deny_fail(self):
        g = OutputGuard().regex_deny(r"password\s*[:=]", "password leak")
        result = g.check("The password: secret123")
        assert not result.passed


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------

class TestCustomRules:
    def test_custom_pass(self):
        g = OutputGuard().custom("word_check", lambda t: None if "hello" in t else "missing hello")
        result = g.check("hello world")
        assert result.passed

    def test_custom_fail(self):
        g = OutputGuard().custom("word_check", lambda t: "no hello" if "hello" not in t else None)
        result = g.check("goodbye world")
        assert not result.passed


# ---------------------------------------------------------------------------
# Combined rules
# ---------------------------------------------------------------------------

class TestCombinedRules:
    def test_fluent_chaining(self):
        g = (OutputGuard()
             .max_length(1000)
             .min_length(10)
             .ban_phrases(["I cannot"])
             .no_empty())
        assert g.rule_count == 4

    def test_multiple_violations(self):
        g = (OutputGuard()
             .min_length(100)
             .require_phrases(["disclaimer"]))
        result = g.check("Short")
        assert result.violation_count == 2

    def test_all_pass(self):
        g = (OutputGuard()
             .max_length(1000)
             .no_empty()
             .ban_phrases(["forbidden"]))
        result = g.check("This is perfectly fine output.")
        assert result.passed


# ---------------------------------------------------------------------------
# Result properties
# ---------------------------------------------------------------------------

class TestResultProperties:
    def test_errors_vs_warnings(self):
        g = OutputGuard().require_format("markdown")
        result = g.check("Plain text without markdown")
        # Markdown check is a warning
        assert len(result.warnings) >= 1
        assert result.passed  # Warnings don't cause failure

    def test_batch(self):
        g = OutputGuard().no_empty()
        results = g.check_batch(["hello", "", "world"])
        assert results[0].passed
        assert not results[1].passed
        assert results[2].passed
