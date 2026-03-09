"""Tests for the input normalizer."""

import pytest
from sentinel.input_normalizer import (
    InputNormalizer,
    NormalizerConfig,
    NormalizationResult,
    NormalizationStep,
    NormalizerStats,
)


# ---------------------------------------------------------------------------
# Clean text passthrough
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_clean_ascii_unchanged(self):
        n = InputNormalizer()
        result = n.normalize("Hello world")
        assert result.normalized == "Hello world"
        assert not result.is_modified

    def test_already_normalized_text(self):
        n = InputNormalizer()
        result = n.normalize("The quick brown fox")
        assert result.normalized == "The quick brown fox"
        assert result.total_changed == 0

    def test_empty_text(self):
        n = InputNormalizer()
        result = n.normalize("")
        assert result.normalized == ""
        assert not result.is_modified
        assert result.total_changed == 0


# ---------------------------------------------------------------------------
# Unicode NFKC normalization
# ---------------------------------------------------------------------------

class TestUnicodeNfkc:
    def test_fullwidth_latin_to_ascii(self):
        n = InputNormalizer()
        result = n.normalize("\uff28\uff45\uff4c\uff4c\uff4f")  # Ｈｅｌｌｏ
        assert result.normalized == "Hello"
        assert result.is_modified

    def test_fullwidth_digits(self):
        n = InputNormalizer()
        result = n.normalize("\uff11\uff12\uff13")  # １２３
        assert result.normalized == "123"

    def test_nfkc_step_recorded(self):
        n = InputNormalizer()
        result = n.normalize("\uff28ello")
        nfkc_step = _find_step(result, "unicode_nfkc")
        assert nfkc_step.applied
        assert nfkc_step.chars_changed >= 1


# ---------------------------------------------------------------------------
# Invisible character stripping
# ---------------------------------------------------------------------------

class TestStripInvisible:
    def test_zero_width_space(self):
        n = InputNormalizer()
        result = n.normalize("Hello\u200bworld")
        assert result.normalized == "Helloworld"
        assert result.is_modified

    def test_zero_width_non_joiner(self):
        n = InputNormalizer()
        result = n.normalize("te\u200cst")
        assert result.normalized == "test"

    def test_zero_width_joiner(self):
        n = InputNormalizer()
        result = n.normalize("ab\u200dcd")
        assert result.normalized == "abcd"

    def test_bom_stripped(self):
        n = InputNormalizer()
        result = n.normalize("\ufeffHello")
        assert result.normalized == "Hello"

    def test_soft_hyphen(self):
        n = InputNormalizer()
        result = n.normalize("in\u00advisible")
        assert result.normalized == "invisible"

    def test_multiple_invisible_chars(self):
        n = InputNormalizer()
        result = n.normalize("a\u200b\u200c\u200db")
        assert result.normalized == "ab"
        step = _find_step(result, "strip_invisible")
        assert step.chars_changed == 3


# ---------------------------------------------------------------------------
# Homoglyph mapping
# ---------------------------------------------------------------------------

class TestHomoglyphMapping:
    def test_cyrillic_a_to_latin(self):
        n = InputNormalizer()
        result = n.normalize("H\u0435llo")  # Cyrillic е
        assert result.normalized == "Hello"

    def test_cyrillic_o_to_latin(self):
        n = InputNormalizer()
        result = n.normalize("w\u043erld")  # Cyrillic о
        assert result.normalized == "world"

    def test_cyrillic_p_to_latin(self):
        n = InputNormalizer()
        result = n.normalize("\u0440ython")  # Cyrillic р
        assert result.normalized == "python"

    def test_cyrillic_c_to_latin(self):
        n = InputNormalizer()
        result = n.normalize("\u0441ode")  # Cyrillic с
        assert result.normalized == "code"

    def test_cyrillic_x_to_latin(self):
        n = InputNormalizer()
        result = n.normalize("e\u0445it")  # Cyrillic х
        assert result.normalized == "exit"

    def test_multiple_homoglyphs_in_one_word(self):
        n = InputNormalizer()
        # "scope" with Cyrillic с, о, р
        result = n.normalize("\u0441\u043e\u0440e")
        assert result.normalized == "cope"

    def test_uppercase_cyrillic(self):
        n = InputNormalizer()
        result = n.normalize("\u0420\u0410\u0421\u0421")  # РАСС
        assert result.normalized == "PACC"

    def test_homoglyph_step_counts(self):
        n = InputNormalizer()
        result = n.normalize("\u0430\u0435")  # two Cyrillic chars
        step = _find_step(result, "homoglyph_map")
        assert step.applied
        assert step.chars_changed == 2


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------

class TestWhitespaceNormalization:
    def test_collapse_multiple_spaces(self):
        n = InputNormalizer()
        result = n.normalize("Hello    world")
        assert result.normalized == "Hello world"

    def test_collapse_tabs(self):
        n = InputNormalizer()
        result = n.normalize("Hello\t\tworld")
        assert result.normalized == "Hello world"

    def test_collapse_newlines(self):
        n = InputNormalizer()
        result = n.normalize("Hello\n\n\nworld")
        assert result.normalized == "Hello world"

    def test_strip_leading_trailing(self):
        n = InputNormalizer()
        result = n.normalize("  Hello world  ")
        assert result.normalized == "Hello world"

    def test_mixed_whitespace(self):
        n = InputNormalizer()
        result = n.normalize("  Hello  \t\n  world  ")
        assert result.normalized == "Hello world"


# ---------------------------------------------------------------------------
# Accent stripping
# ---------------------------------------------------------------------------

class TestAccentStripping:
    def test_accented_e(self):
        config = NormalizerConfig(strip_accents=True)
        n = InputNormalizer(config)
        result = n.normalize("caf\u00e9")
        assert result.normalized == "cafe"

    def test_accented_n(self):
        config = NormalizerConfig(strip_accents=True)
        n = InputNormalizer(config)
        result = n.normalize("ja\u00f1o")
        assert "n" in result.normalized

    def test_accents_disabled_by_default(self):
        n = InputNormalizer()
        result = n.normalize("caf\u00e9")
        step = _find_step(result, "strip_accents")
        assert not step.applied


# ---------------------------------------------------------------------------
# Lowercase
# ---------------------------------------------------------------------------

class TestLowercase:
    def test_lowercase_enabled(self):
        config = NormalizerConfig(lowercase=True)
        n = InputNormalizer(config)
        result = n.normalize("Hello WORLD")
        assert result.normalized == "hello world"

    def test_lowercase_disabled_by_default(self):
        n = InputNormalizer()
        result = n.normalize("Hello")
        assert result.normalized == "Hello"
        step = _find_step(result, "lowercase")
        assert not step.applied


# ---------------------------------------------------------------------------
# Multiple normalizations combined
# ---------------------------------------------------------------------------

class TestCombined:
    def test_invisible_and_homoglyph(self):
        n = InputNormalizer()
        # Cyrillic е + zero-width space
        result = n.normalize("H\u0435\u200bllo")
        assert result.normalized == "Hello"

    def test_fullwidth_with_extra_spaces(self):
        n = InputNormalizer()
        result = n.normalize("\uff28ello    world")
        assert result.normalized == "Hello world"

    def test_all_steps_enabled(self):
        config = NormalizerConfig(
            lowercase=True,
            strip_accents=True,
        )
        n = InputNormalizer(config)
        result = n.normalize("  Caf\u00e9  \u200b  W\u043erld  ")
        assert result.normalized == "cafe world"

    def test_steps_list_has_all_entries(self):
        n = InputNormalizer()
        result = n.normalize("test")
        assert len(result.steps) == 6


# ---------------------------------------------------------------------------
# Batch normalization
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_returns_list(self):
        n = InputNormalizer()
        results = n.normalize_batch(["Hello", "W\u043erld"])
        assert len(results) == 2
        assert results[0].normalized == "Hello"
        assert results[1].normalized == "World"

    def test_batch_empty_list(self):
        n = InputNormalizer()
        results = n.normalize_batch([])
        assert results == []

    def test_batch_updates_stats(self):
        n = InputNormalizer()
        n.normalize_batch(["a\u200bb", "\u0435"])
        s = n.stats()
        assert s.total_normalized == 2


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStats:
    def test_initial_stats(self):
        n = InputNormalizer()
        s = n.stats()
        assert s.total_normalized == 0
        assert s.total_chars_changed == 0
        assert s.modified_count == 0

    def test_stats_after_normalize(self):
        n = InputNormalizer()
        n.normalize("Hello\u200bworld")
        s = n.stats()
        assert s.total_normalized == 1
        assert s.total_chars_changed >= 1
        assert s.modified_count == 1

    def test_stats_unmodified_not_counted(self):
        n = InputNormalizer()
        n.normalize("Hello world")
        s = n.stats()
        assert s.total_normalized == 1
        assert s.modified_count == 0

    def test_stats_accumulate(self):
        n = InputNormalizer()
        n.normalize("a\u200bb")
        n.normalize("c\u200dd")
        n.normalize("Hello")
        s = n.stats()
        assert s.total_normalized == 3
        assert s.modified_count == 2

    def test_stats_returns_copy(self):
        n = InputNormalizer()
        s1 = n.stats()
        n.normalize("a\u200bb")
        s2 = n.stats()
        assert s1.total_normalized == 0
        assert s2.total_normalized == 1


# ---------------------------------------------------------------------------
# Config toggling
# ---------------------------------------------------------------------------

class TestConfigToggling:
    def test_disable_nfkc(self):
        config = NormalizerConfig(unicode_nfkc=False)
        n = InputNormalizer(config)
        result = n.normalize("\uff28ello")
        step = _find_step(result, "unicode_nfkc")
        assert not step.applied

    def test_disable_invisible_stripping(self):
        config = NormalizerConfig(strip_invisible=False)
        n = InputNormalizer(config)
        result = n.normalize("a\u200bb")
        assert "\u200b" in result.normalized

    def test_disable_whitespace(self):
        config = NormalizerConfig(normalize_whitespace=False)
        n = InputNormalizer(config)
        result = n.normalize("a  b")
        assert result.normalized == "a  b"

    def test_disable_homoglyph(self):
        config = NormalizerConfig(homoglyph_map=False)
        n = InputNormalizer(config)
        result = n.normalize("\u0430bc")
        assert "\u0430" in result.normalized

    def test_all_disabled(self):
        config = NormalizerConfig(
            unicode_nfkc=False,
            strip_invisible=False,
            normalize_whitespace=False,
            lowercase=False,
            strip_accents=False,
            homoglyph_map=False,
        )
        n = InputNormalizer(config)
        text = "H\u0435\u200bllo  W\u043erld"
        result = n.normalize(text)
        assert result.normalized == text
        assert not result.is_modified


# ---------------------------------------------------------------------------
# Dataclass contracts
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_normalization_step_fields(self):
        step = NormalizationStep(name="test", applied=True, chars_changed=5)
        assert step.name == "test"
        assert step.applied is True
        assert step.chars_changed == 5
        assert step.description == ""

    def test_normalizer_config_defaults(self):
        config = NormalizerConfig()
        assert config.unicode_nfkc is True
        assert config.strip_invisible is True
        assert config.normalize_whitespace is True
        assert config.lowercase is False
        assert config.strip_accents is False
        assert config.homoglyph_map is True

    def test_normalizer_stats_defaults(self):
        stats = NormalizerStats()
        assert stats.total_normalized == 0
        assert stats.total_chars_changed == 0
        assert stats.modified_count == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_step(result: NormalizationResult, name: str) -> NormalizationStep:
    for step in result.steps:
        if step.name == name:
            return step
    raise ValueError(f"Step '{name}' not found in result")
