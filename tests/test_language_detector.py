"""Tests for language detection."""

import pytest
from sentinel.language_detector import LanguageDetector, LanguageResult


# ---------------------------------------------------------------------------
# English detection
# ---------------------------------------------------------------------------

class TestEnglishDetection:
    def test_english_text(self):
        d = LanguageDetector()
        result = d.check("The quick brown fox jumps over the lazy dog.")
        assert result.language == "en"
        assert result.is_english

    def test_english_simple(self):
        d = LanguageDetector()
        assert d.detect("This is a simple English sentence.") == "en"


# ---------------------------------------------------------------------------
# Other Latin-script languages
# ---------------------------------------------------------------------------

class TestLatinScriptLanguages:
    def test_french(self):
        d = LanguageDetector()
        result = d.check("Le chat est sur la table dans le jardin avec les fleurs.")
        assert result.language == "fr"

    def test_german(self):
        d = LanguageDetector()
        result = d.check("Der Hund ist nicht auf dem Tisch und die Katze auch nicht.")
        assert result.language == "de"

    def test_spanish(self):
        d = LanguageDetector()
        result = d.check("El gato es muy bonito para todo el mundo como las flores.")
        assert result.language == "es"

    def test_italian(self):
        d = LanguageDetector()
        result = d.check("Il gatto non sono della casa per questo molto bello.")
        assert result.language == "it"


# ---------------------------------------------------------------------------
# Non-Latin script languages
# ---------------------------------------------------------------------------

class TestNonLatinScripts:
    def test_chinese(self):
        d = LanguageDetector()
        result = d.check("这是一个中文句子测试")
        assert result.language == "zh"

    def test_japanese_hiragana(self):
        d = LanguageDetector()
        result = d.check("これはひらがなのテストです")
        assert result.language == "ja"

    def test_korean(self):
        d = LanguageDetector()
        result = d.check("한국어 테스트 문장입니다")
        assert result.language == "ko"

    def test_arabic(self):
        d = LanguageDetector()
        result = d.check("هذا نص باللغة العربية")
        assert result.language == "ar"

    def test_russian(self):
        d = LanguageDetector()
        result = d.check("Это тестовое предложение на русском языке")
        assert result.language == "ru"


# ---------------------------------------------------------------------------
# Language enforcement
# ---------------------------------------------------------------------------

class TestEnforcement:
    def test_allowed_language(self):
        d = LanguageDetector(allowed=["en", "fr"])
        result = d.check("This is an English sentence with the words.")
        assert result.allowed

    def test_disallowed_language(self):
        d = LanguageDetector(allowed=["en"])
        result = d.check("Le chat est sur la table dans le jardin.")
        assert not result.allowed

    def test_blocked_language(self):
        d = LanguageDetector(blocked=["ru"])
        result = d.check("Это тестовое предложение")
        assert not result.allowed

    def test_no_constraints(self):
        d = LanguageDetector()
        result = d.check("Any language is fine")
        assert result.allowed


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_range(self):
        d = LanguageDetector()
        result = d.check("This is clearly English text.")
        assert 0.0 <= result.confidence <= 1.0

    def test_scores_dict(self):
        d = LanguageDetector()
        result = d.check("The cat is on the table.")
        assert "en" in result.scores


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text(self):
        d = LanguageDetector()
        result = d.check("")
        assert result.language == "und"
        assert result.allowed

    def test_numbers_only(self):
        d = LanguageDetector()
        result = d.check("12345 67890")
        # Numbers have no language signal
        assert result.language == "und"

    def test_mixed_scripts(self):
        d = LanguageDetector()
        result = d.check("Hello 你好 Bonjour")
        assert result.language in ("en", "zh", "fr")
