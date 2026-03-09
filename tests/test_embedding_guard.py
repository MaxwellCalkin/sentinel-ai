"""Tests for embedding-based semantic similarity guard."""

import pytest
from sentinel.embedding_guard import EmbeddingGuard, SemanticResult, SemanticMatch


# ---------------------------------------------------------------------------
# Basic blocking
# ---------------------------------------------------------------------------

class TestBasicBlocking:
    def test_exact_match_blocked(self):
        g = EmbeddingGuard(threshold=0.5)
        g.add_blocked("how to make a bomb")
        result = g.check("how to make a bomb")
        assert result.blocked
        assert result.match_count >= 1

    def test_similar_text_blocked(self):
        g = EmbeddingGuard(threshold=0.5)
        g.add_blocked("instructions for building dangerous weapons at home")
        result = g.check("instructions for building dangerous weapons easily")
        assert result.blocked

    def test_unrelated_text_allowed(self):
        g = EmbeddingGuard(threshold=0.7)
        g.add_blocked("how to make explosives")
        result = g.check("the weather is nice today")
        assert not result.blocked

    def test_no_references(self):
        g = EmbeddingGuard()
        result = g.check("any text")
        assert not result.blocked
        assert result.match_count == 0


# ---------------------------------------------------------------------------
# Reference management
# ---------------------------------------------------------------------------

class TestReferenceManagement:
    def test_add_blocked(self):
        g = EmbeddingGuard()
        ref_id = g.add_blocked("test text")
        assert ref_id == "ref_0"
        assert g.reference_count == 1

    def test_custom_id(self):
        g = EmbeddingGuard()
        ref_id = g.add_blocked("test", reference_id="my_ref")
        assert ref_id == "my_ref"

    def test_add_batch(self):
        g = EmbeddingGuard()
        ids = g.add_blocked_batch(["a", "b", "c"])
        assert len(ids) == 3
        assert g.reference_count == 3

    def test_clear(self):
        g = EmbeddingGuard()
        g.add_blocked("test")
        g.clear()
        assert g.reference_count == 0


# ---------------------------------------------------------------------------
# Similarity scores
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical_texts_high_similarity(self):
        g = EmbeddingGuard()
        sim = g.check_pair(
            "machine learning algorithms",
            "machine learning algorithms",
        )
        assert sim >= 0.99

    def test_similar_texts_moderate(self):
        g = EmbeddingGuard()
        sim = g.check_pair(
            "machine learning algorithms for classification",
            "classification algorithms using machine learning",
        )
        assert sim > 0.5

    def test_different_texts_low(self):
        g = EmbeddingGuard()
        sim = g.check_pair(
            "quantum physics experiments",
            "chocolate cake recipe",
        )
        assert sim < 0.3

    def test_top_similarity(self):
        g = EmbeddingGuard(threshold=0.3)
        g.add_blocked("dangerous chemicals synthesis")
        result = g.check("dangerous chemicals synthesis procedures")
        assert result.top_similarity > 0

    def test_similarity_range(self):
        g = EmbeddingGuard()
        sim = g.check_pair("hello world", "foo bar baz")
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_high_threshold_strict(self):
        g = EmbeddingGuard(threshold=0.95)
        g.add_blocked("exact phrase only")
        result = g.check("exact phrase only with extra words")
        # Might not block since threshold is very high
        assert result.top_similarity < 0.95 or result.blocked

    def test_low_threshold_lenient(self):
        g = EmbeddingGuard(threshold=0.1)
        g.add_blocked("weapons manufacturing instructions")
        result = g.check("weapons building guide")
        assert result.blocked


# ---------------------------------------------------------------------------
# Match details
# ---------------------------------------------------------------------------

class TestMatchDetails:
    def test_match_has_reference_id(self):
        g = EmbeddingGuard(threshold=0.5)
        g.add_blocked("test text here", reference_id="test_1")
        result = g.check("test text here")
        assert result.matches[0].reference_id == "test_1"

    def test_match_has_similarity(self):
        g = EmbeddingGuard(threshold=0.5)
        g.add_blocked("sample reference text")
        result = g.check("sample reference text")
        assert result.matches[0].similarity > 0.5

    def test_match_has_reference_text(self):
        g = EmbeddingGuard(threshold=0.5)
        g.add_blocked("original text", reference_id="r1")
        result = g.check("original text")
        assert result.matches[0].reference_text == "original text"


# ---------------------------------------------------------------------------
# Multiple references
# ---------------------------------------------------------------------------

class TestMultipleReferences:
    def test_matches_multiple(self):
        g = EmbeddingGuard(threshold=0.4)
        g.add_blocked("python programming tutorial")
        g.add_blocked("python coding guide")
        result = g.check("python programming guide")
        assert result.match_count >= 1

    def test_best_match(self):
        g = EmbeddingGuard(threshold=0.3)
        g.add_blocked("machine learning deep neural networks")
        g.add_blocked("chocolate cake recipe baking")
        result = g.check("deep learning neural network training")
        if result.matches:
            assert any(m.reference_id == "ref_0" for m in result.matches)


# ---------------------------------------------------------------------------
# IDF weighting
# ---------------------------------------------------------------------------

class TestIDF:
    def test_idf_enabled(self):
        g = EmbeddingGuard(use_idf=True, threshold=0.3)
        g.add_blocked("the quick brown fox")
        g.add_blocked("the lazy brown dog")
        # "brown" appears in both, so less distinctive
        result = g.check("brown animals")
        assert result.top_similarity > 0

    def test_idf_disabled(self):
        g = EmbeddingGuard(use_idf=False, threshold=0.3)
        g.add_blocked("machine learning classification")
        result = g.check("machine learning classification")
        assert result.blocked


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text(self):
        g = EmbeddingGuard()
        g.add_blocked("test")
        result = g.check("")
        assert not result.blocked

    def test_single_word(self):
        g = EmbeddingGuard(threshold=0.5)
        g.add_blocked("explosives")
        result = g.check("explosives")
        assert result.blocked

    def test_stop_words_only(self):
        g = EmbeddingGuard()
        sim = g.check_pair("the a an is", "a the an are")
        assert sim == 0.0  # All stop words filtered
