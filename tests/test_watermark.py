"""Tests for watermark detection."""

import pytest
from sentinel.watermark import WatermarkDetector, WatermarkResult, TextStats


# ---------------------------------------------------------------------------
# Text statistics
# ---------------------------------------------------------------------------

class TestTextStats:
    def test_analyze_basic(self):
        d = WatermarkDetector()
        stats = d.analyze("The quick brown fox jumps over the lazy dog.")
        assert stats.word_count == 9
        assert stats.unique_ratio > 0
        assert stats.avg_word_length > 0
        assert stats.entropy > 0

    def test_empty_text(self):
        d = WatermarkDetector()
        stats = d.analyze("")
        assert stats.word_count == 0
        assert stats.entropy == 0.0

    def test_entropy_low_for_repetitive(self):
        d = WatermarkDetector()
        s1 = d.analyze("the the the the the the the the")
        s2 = d.analyze("the quick brown fox jumps over lazy dog")
        assert s1.entropy < s2.entropy

    def test_unique_ratio(self):
        d = WatermarkDetector()
        s1 = d.analyze("word word word word word")
        s2 = d.analyze("alpha beta gamma delta epsilon")
        assert s1.unique_ratio < s2.unique_ratio

    def test_burstiness(self):
        d = WatermarkDetector()
        uniform = "This is short. This is short. This is short."
        varied = "Short. This sentence is considerably much longer than the others here."
        s1 = d.analyze(uniform)
        s2 = d.analyze(varied)
        assert s2.burstiness >= s1.burstiness


# ---------------------------------------------------------------------------
# Signature detection
# ---------------------------------------------------------------------------

class TestSignatureDetection:
    def test_no_signatures(self):
        d = WatermarkDetector(min_words=5)
        result = d.check("This is a test of watermark detection system here.")
        assert not result.detected

    def test_signature_match(self):
        d = WatermarkDetector(min_words=10)
        d.add_signature("test_wm", seed_words=["the", "a", "is"], bias=0.3)
        # Text with high frequency of seed words
        text = " ".join(["the cat is a dog and the bird is a fish"] * 10)
        result = d.check(text)
        assert result.detected or result.confidence > 0

    def test_insufficient_text(self):
        d = WatermarkDetector(min_words=100)
        d.add_signature("test", seed_words=["the"])
        result = d.check("Short text.")
        assert not result.detected
        assert "insufficient" in str(result.stats.get("warning", ""))


# ---------------------------------------------------------------------------
# Confidence scores
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_range(self):
        d = WatermarkDetector(min_words=5)
        d.add_signature("test", seed_words=["the"])
        text = "The cat and the dog went to the park with the ball and the toys."
        result = d.check(text)
        assert 0.0 <= result.confidence <= 1.0

    def test_stats_in_result(self):
        d = WatermarkDetector(min_words=5)
        d.add_signature("test", seed_words=["the"])
        text = "The quick brown fox jumps over the lazy dog in the park."
        result = d.check(text)
        assert "word_count" in result.stats
        assert "entropy" in result.stats


# ---------------------------------------------------------------------------
# Multiple signatures
# ---------------------------------------------------------------------------

class TestMultipleSignatures:
    def test_multiple_signatures(self):
        d = WatermarkDetector(min_words=5)
        d.add_signature("sig1", seed_words=["the", "a"])
        d.add_signature("sig2", seed_words=["xyz", "qqq"])
        text = "The cat is a nice animal and the dog is a good pet too."
        result = d.check(text)
        # sig1 might match, sig2 shouldn't
        if result.detected:
            assert "sig1" in result.signatures_matched


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_numbers_only(self):
        d = WatermarkDetector(min_words=5)
        stats = d.analyze("123 456 789")
        assert stats.word_count == 0

    def test_long_text(self):
        d = WatermarkDetector(min_words=10)
        d.add_signature("test", seed_words=["the"])
        text = "The quick brown fox. " * 50
        result = d.check(text)
        assert isinstance(result, WatermarkResult)

    def test_result_structure(self):
        d = WatermarkDetector(min_words=5)
        result = d.check("Test text with enough words for analysis here now.")
        assert hasattr(result, 'detected')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'signatures_matched')
        assert hasattr(result, 'stats')
