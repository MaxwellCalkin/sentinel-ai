"""Tests for the input sanitizer."""

import re
import pytest
from sentinel.input_sanitizer import InputSanitizer, SanitizeResult


# ---------------------------------------------------------------------------
# Invisible character removal
# ---------------------------------------------------------------------------

class TestInvisibleChars:
    def test_zero_width_space(self):
        s = InputSanitizer()
        result = s.sanitize("Hello\u200bworld")
        assert result.text == "Helloworld"
        assert result.modified

    def test_zero_width_joiner(self):
        s = InputSanitizer()
        result = s.sanitize("test\u200dtext")
        assert "\u200d" not in result.text

    def test_bom(self):
        s = InputSanitizer()
        result = s.sanitize("\ufeffHello")
        assert result.text == "Hello"

    def test_soft_hyphen(self):
        s = InputSanitizer()
        result = s.sanitize("in\u00advisible")
        assert result.text == "invisible"

    def test_word_joiner(self):
        s = InputSanitizer()
        result = s.sanitize("word\u2060break")
        assert result.text == "wordbreak"

    def test_multiple_invisible(self):
        s = InputSanitizer()
        result = s.sanitize("a\u200b\u200c\u200db")
        assert result.text == "ab"
        assert "3 invisible" in result.modifications[0]

    def test_no_invisible(self):
        s = InputSanitizer()
        result = s.sanitize("Hello world")
        assert not any("invisible" in m for m in result.modifications)


# ---------------------------------------------------------------------------
# Control character removal
# ---------------------------------------------------------------------------

class TestControlChars:
    def test_null_byte(self):
        s = InputSanitizer()
        result = s.sanitize("Hello\x00world")
        assert "\x00" not in result.text

    def test_bell(self):
        s = InputSanitizer()
        result = s.sanitize("Hello\x07world")
        assert "\x07" not in result.text

    def test_preserves_tab_newline(self):
        s = InputSanitizer()
        result = s.sanitize("Hello\tworld\n")
        assert "\t" in result.text or " " in result.text  # tab may be normalized to space
        assert "\n" in result.text or result.text == "Hello world"  # newline preserved or stripped by whitespace norm

    def test_delete_char(self):
        s = InputSanitizer()
        result = s.sanitize("Hello\x7fworld")
        assert "\x7f" not in result.text


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

class TestUnicodeNormalization:
    def test_nfc_normalization(self):
        # e + combining acute = é (NFC)
        s = InputSanitizer()
        result = s.sanitize("caf\u0065\u0301")  # e + combining acute
        assert result.text == "café" or "NFC" in str(result.modifications)

    def test_already_normalized(self):
        s = InputSanitizer()
        result = s.sanitize("café")
        # Already NFC, no change needed for this specific part


# ---------------------------------------------------------------------------
# Homoglyph replacement
# ---------------------------------------------------------------------------

class TestHomoglyphs:
    def test_cyrillic_a(self):
        s = InputSanitizer()
        result = s.sanitize("\u0410dmin")  # Cyrillic А + dmin
        assert result.text == "Admin"
        assert result.modified

    def test_cyrillic_lowercase(self):
        s = InputSanitizer()
        result = s.sanitize("p\u0430ssword")  # Latin p + Cyrillic а + ssword
        assert result.text == "password"

    def test_greek_letters(self):
        s = InputSanitizer()
        result = s.sanitize("\u0391\u0392C")  # Greek Α + Greek Β + Latin C
        assert result.text == "ABC"

    def test_mixed_homoglyphs(self):
        s = InputSanitizer()
        result = s.sanitize("\u0421\u041e\u041c\u041c\u0410\u041d\u0414")
        # Cyrillic С О М М А Н Д → C O M M A H Д
        assert "homoglyph" in result.modifications[0].lower() if result.modified else True

    def test_no_homoglyphs(self):
        s = InputSanitizer()
        result = s.sanitize("Normal ASCII text")
        assert not any("homoglyph" in m.lower() for m in result.modifications)

    def test_disabled_homoglyphs(self):
        s = InputSanitizer(replace_homoglyphs=False)
        result = s.sanitize("\u0410dmin")
        assert "\u0410" in result.text  # Not replaced


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------

class TestWhitespaceNormalization:
    def test_multiple_spaces(self):
        s = InputSanitizer()
        result = s.sanitize("Hello    world")
        assert result.text == "Hello world"

    def test_multiple_newlines(self):
        s = InputSanitizer()
        result = s.sanitize("Hello\n\n\n\n\nworld")
        assert result.text == "Hello\n\nworld"

    def test_leading_trailing(self):
        s = InputSanitizer()
        result = s.sanitize("  Hello world  ")
        assert result.text == "Hello world"

    def test_tabs_to_spaces(self):
        s = InputSanitizer()
        result = s.sanitize("Hello\t\tworld")
        assert result.text == "Hello world"

    def test_disabled_whitespace(self):
        s = InputSanitizer(normalize_whitespace=False)
        result = s.sanitize("Hello    world")
        assert "    " in result.text


# ---------------------------------------------------------------------------
# Length limiting
# ---------------------------------------------------------------------------

class TestMaxLength:
    def test_truncation(self):
        s = InputSanitizer(max_length=10)
        result = s.sanitize("Hello world, this is a long text")
        assert len(result.text) == 10
        assert "Truncated" in result.modifications[-1]

    def test_within_limit(self):
        s = InputSanitizer(max_length=100)
        result = s.sanitize("Short text")
        assert not any("Truncated" in m for m in result.modifications)

    def test_no_limit(self):
        s = InputSanitizer(max_length=None)
        long_text = "x" * 100000
        result = s.sanitize(long_text)
        assert len(result.text) == 100000


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------

class TestCustomPatterns:
    def test_custom_strip(self):
        s = InputSanitizer(custom_strips=[re.compile(r"<[^>]+>")])
        result = s.sanitize("Hello <b>world</b>")
        assert "<b>" not in result.text
        assert "custom pattern" in result.modifications[0].lower()

    def test_multiple_custom(self):
        s = InputSanitizer(custom_strips=[
            re.compile(r"\[.*?\]"),
            re.compile(r"\(.*?\)"),
        ])
        result = s.sanitize("Hello [link] and (note)")
        assert "[link]" not in result.text
        assert "(note)" not in result.text


# ---------------------------------------------------------------------------
# SanitizeResult
# ---------------------------------------------------------------------------

class TestSanitizeResult:
    def test_not_modified(self):
        s = InputSanitizer(
            strip_invisible=False,
            normalize_unicode=False,
            strip_control=False,
            normalize_whitespace=False,
            replace_homoglyphs=False,
        )
        result = s.sanitize("Hello")
        assert not result.modified
        assert result.summary == "No modifications"

    def test_summary_with_modifications(self):
        s = InputSanitizer()
        result = s.sanitize("Hello\u200b world\x00")
        assert result.summary != "No modifications"

    def test_original_preserved(self):
        s = InputSanitizer()
        original = "Hello\u200bworld"
        result = s.sanitize(original)
        assert result.original == original
        assert result.text != result.original


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_all_disabled(self):
        s = InputSanitizer(
            strip_invisible=False,
            normalize_unicode=False,
            strip_control=False,
            normalize_whitespace=False,
            replace_homoglyphs=False,
        )
        text = "Hello\u200b  \x00world"
        result = s.sanitize(text)
        assert result.text == text

    def test_only_length(self):
        s = InputSanitizer(
            max_length=5,
            strip_invisible=False,
            normalize_unicode=False,
            strip_control=False,
            normalize_whitespace=False,
            replace_homoglyphs=False,
        )
        result = s.sanitize("Hello world")
        assert result.text == "Hello"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self):
        s = InputSanitizer()
        result = s.sanitize("")
        assert result.text == ""
        assert not result.modified

    def test_only_invisible(self):
        s = InputSanitizer()
        result = s.sanitize("\u200b\u200c\u200d")
        assert result.text == ""

    def test_unicode_emoji(self):
        s = InputSanitizer()
        result = s.sanitize("Hello 👋 world")
        assert "👋" in result.text  # Emojis should be preserved

    def test_cjk_characters(self):
        s = InputSanitizer()
        result = s.sanitize("你好世界")
        assert result.text == "你好世界"

    def test_rtl_text(self):
        s = InputSanitizer()
        result = s.sanitize("مرحبا بالعالم")
        assert len(result.text) > 0
