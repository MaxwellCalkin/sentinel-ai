"""Tests for response sanitizer."""

import pytest
from sentinel.response_sanitizer import ResponseSanitizer, SanitizeOutput


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------

class TestWhitespace:
    def test_collapse_spaces(self):
        s = ResponseSanitizer().normalize_whitespace()
        result = s.sanitize("Hello   world   here")
        assert result.sanitized == "Hello world here"
        assert result.modified

    def test_collapse_newlines(self):
        s = ResponseSanitizer().normalize_whitespace()
        result = s.sanitize("Line1\n\n\n\n\nLine2")
        assert result.sanitized == "Line1\n\nLine2"

    def test_strip_leading_trailing(self):
        s = ResponseSanitizer().normalize_whitespace()
        result = s.sanitize("  hello  ")
        assert result.sanitized == "hello"

    def test_no_change(self):
        s = ResponseSanitizer().normalize_whitespace()
        result = s.sanitize("Normal text here")
        assert not result.modified


# ---------------------------------------------------------------------------
# System leak stripping
# ---------------------------------------------------------------------------

class TestSystemLeaks:
    def test_strip_system_prompt_leak(self):
        s = ResponseSanitizer().strip_system_leaks()
        result = s.sanitize("System prompt: You are a helpful assistant\nHere is the answer.")
        assert "system prompt" not in result.sanitized.lower()

    def test_strip_identity_leak(self):
        s = ResponseSanitizer().strip_system_leaks()
        result = s.sanitize("I am a helpful AI assistant. The answer is 42.")
        assert "helpful ai assistant" not in result.sanitized.lower()

    def test_no_false_positive(self):
        s = ResponseSanitizer().strip_system_leaks()
        result = s.sanitize("The answer to your question is 42.")
        assert result.sanitized == "The answer to your question is 42."


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class TestHTML:
    def test_strip_tags(self):
        s = ResponseSanitizer().strip_html()
        result = s.sanitize("<p>Hello <b>world</b></p>")
        assert result.sanitized == "Hello world"

    def test_strip_script(self):
        s = ResponseSanitizer().strip_html()
        result = s.sanitize("Before<script>alert('xss')</script>After")
        assert "script" not in result.sanitized
        assert "alert" not in result.sanitized
        assert result.sanitized == "BeforeAfter"

    def test_strip_style(self):
        s = ResponseSanitizer().strip_html()
        result = s.sanitize("Text<style>.red{color:red}</style>More")
        assert "style" not in result.sanitized


# ---------------------------------------------------------------------------
# URL/email stripping
# ---------------------------------------------------------------------------

class TestURLEmail:
    def test_strip_urls(self):
        s = ResponseSanitizer().strip_urls()
        result = s.sanitize("Visit https://example.com for more info")
        assert "example.com" not in result.sanitized
        assert "[URL removed]" in result.sanitized

    def test_strip_emails(self):
        s = ResponseSanitizer().strip_emails()
        result = s.sanitize("Contact user@example.com for help")
        assert "user@example.com" not in result.sanitized
        assert "[email removed]" in result.sanitized


# ---------------------------------------------------------------------------
# Length limiting
# ---------------------------------------------------------------------------

class TestLength:
    def test_truncate(self):
        s = ResponseSanitizer().limit_length(10)
        result = s.sanitize("This is a very long response")
        assert len(result.sanitized) == 13  # 10 + "..."
        assert result.sanitized.endswith("...")

    def test_no_truncate(self):
        s = ResponseSanitizer().limit_length(100)
        result = s.sanitize("Short")
        assert result.sanitized == "Short"
        assert not result.modified


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------

class TestCustom:
    def test_replace_pattern(self):
        s = ResponseSanitizer().replace_pattern("no_phone", r"\b\d{3}-\d{4}\b", "[REDACTED]")
        result = s.sanitize("Call 555-1234 for info")
        assert result.sanitized == "Call [REDACTED] for info"

    def test_custom_step(self):
        s = ResponseSanitizer().add_step("upper", lambda t: t.upper())
        result = s.sanitize("hello")
        assert result.sanitized == "HELLO"


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------

class TestChaining:
    def test_multiple_steps(self):
        s = (ResponseSanitizer()
             .strip_html()
             .normalize_whitespace()
             .strip_urls())
        text = "<p>Visit  https://evil.com  now</p>"
        result = s.sanitize(text)
        assert "<p>" not in result.sanitized
        assert "evil.com" not in result.sanitized
        assert "  " not in result.sanitized

    def test_steps_applied_list(self):
        s = (ResponseSanitizer()
             .strip_html()
             .normalize_whitespace())
        result = s.sanitize("<b>Hello</b>   world")
        assert "strip_html" in result.steps_applied
        assert "normalize_whitespace" in result.steps_applied


# ---------------------------------------------------------------------------
# Enable/disable
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_disable_step(self):
        s = ResponseSanitizer().strip_html().normalize_whitespace()
        s.disable_step("strip_html")
        result = s.sanitize("<b>Hello</b>")
        assert "<b>" in result.sanitized

    def test_enable_step(self):
        s = ResponseSanitizer().strip_html()
        s.disable_step("strip_html")
        s.enable_step("strip_html")
        result = s.sanitize("<b>Hello</b>")
        assert "<b>" not in result.sanitized

    def test_step_count(self):
        s = ResponseSanitizer().strip_html().normalize_whitespace()
        assert s.step_count == 2
        s.disable_step("strip_html")
        assert s.active_step_count == 1


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_sanitize(self):
        s = ResponseSanitizer().normalize_whitespace()
        results = s.sanitize_batch(["hello  world", "  spaced  "])
        assert len(results) == 2
        assert results[0].sanitized == "hello world"
        assert results[1].sanitized == "spaced"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self):
        s = ResponseSanitizer().normalize_whitespace()
        result = s.sanitize("")
        assert result.sanitized == ""

    def test_error_in_step_skipped(self):
        s = ResponseSanitizer()
        s.add_step("bad", lambda t: 1 / 0)
        s.normalize_whitespace()
        result = s.sanitize("hello  world")
        assert result.sanitized == "hello world"

    def test_chars_removed(self):
        s = ResponseSanitizer().strip_html()
        result = s.sanitize("<b>Hello</b>")
        assert result.chars_removed == 7  # <b> and </b>

    def test_result_structure(self):
        s = ResponseSanitizer().normalize_whitespace()
        result = s.sanitize("  hi  ")
        assert isinstance(result, SanitizeOutput)
        assert result.original == "  hi  "
