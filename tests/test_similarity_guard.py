"""Tests for the similarity guard."""

import pytest
from sentinel.similarity_guard import SimilarityGuard, SimilarityResult, SimilarityMatch


# ---------------------------------------------------------------------------
# Basic similarity detection
# ---------------------------------------------------------------------------

class TestBasicSimilarity:
    def test_identical_text(self):
        guard = SimilarityGuard(threshold=0.8)
        guard.add_reference("doc1", "The quick brown fox jumps over the lazy dog")
        result = guard.check("The quick brown fox jumps over the lazy dog")
        assert result.is_similar
        assert result.most_similar[1] == 1.0

    def test_completely_different(self):
        guard = SimilarityGuard(threshold=0.8)
        guard.add_reference("doc1", "The quick brown fox jumps over the lazy dog")
        result = guard.check("Quantum mechanics describes subatomic particles")
        assert not result.is_similar

    def test_slightly_modified(self):
        guard = SimilarityGuard(threshold=0.5)
        guard.add_reference("doc1", "The quick brown fox jumps over the lazy dog")
        result = guard.check("The quick brown fox leaps over the lazy dog")
        assert result.is_similar  # Very similar with one word changed

    def test_threshold_boundary(self):
        guard = SimilarityGuard(threshold=0.99)
        guard.add_reference("doc1", "Hello world this is a test of similarity")
        result = guard.check("Hello world this is a test of similarity!")
        # Nearly identical, but threshold is very high
        assert result.most_similar is not None
        # The similarity should be very high but not exactly 1.0


# ---------------------------------------------------------------------------
# Reference management
# ---------------------------------------------------------------------------

class TestReferenceManagement:
    def test_add_reference(self):
        guard = SimilarityGuard()
        guard.add_reference("doc1", "Hello world")
        assert guard.reference_count == 1

    def test_add_multiple(self):
        guard = SimilarityGuard()
        guard.add_references({
            "doc1": "Hello world",
            "doc2": "Goodbye world",
        })
        assert guard.reference_count == 2

    def test_remove_reference(self):
        guard = SimilarityGuard()
        guard.add_reference("doc1", "Hello world")
        guard.remove_reference("doc1")
        assert guard.reference_count == 0

    def test_remove_nonexistent(self):
        guard = SimilarityGuard()
        guard.remove_reference("nonexistent")  # should not raise
        assert guard.reference_count == 0

    def test_clear(self):
        guard = SimilarityGuard()
        guard.add_references({"a": "text a", "b": "text b", "c": "text c"})
        guard.clear()
        assert guard.reference_count == 0

    def test_overwrite_reference(self):
        guard = SimilarityGuard()
        guard.add_reference("doc1", "Original text")
        guard.add_reference("doc1", "New text")
        assert guard.reference_count == 1


# ---------------------------------------------------------------------------
# Multiple references
# ---------------------------------------------------------------------------

class TestMultipleReferences:
    def test_finds_most_similar(self):
        guard = SimilarityGuard(threshold=0.3)
        guard.add_reference("doc1", "The cat sat on the mat")
        guard.add_reference("doc2", "The quick brown fox jumps over the lazy dog")
        guard.add_reference("doc3", "The cat sat on the rug")

        result = guard.check("The cat sat on the mat")
        assert result.most_similar[0] == "doc1"
        assert result.most_similar[1] == 1.0

    def test_multiple_above_threshold(self):
        guard = SimilarityGuard(threshold=0.3)
        guard.add_reference("v1", "The quick brown fox jumps over the lazy dog")
        guard.add_reference("v2", "The quick brown fox jumps over a lazy dog")

        result = guard.check("The quick brown fox jumps over the lazy dog")
        assert len(result.above_threshold) == 2


# ---------------------------------------------------------------------------
# Pair comparison
# ---------------------------------------------------------------------------

class TestPairComparison:
    def test_identical(self):
        guard = SimilarityGuard()
        sim = guard.check_pair("Hello world", "Hello world")
        assert sim == 1.0

    def test_different(self):
        guard = SimilarityGuard()
        sim = guard.check_pair("Hello world", "Goodbye universe")
        assert sim < 0.5

    def test_empty_strings(self):
        guard = SimilarityGuard()
        sim = guard.check_pair("", "")
        assert sim == 0.0

    def test_one_empty(self):
        guard = SimilarityGuard()
        sim = guard.check_pair("Hello world", "")
        assert sim == 0.0


# ---------------------------------------------------------------------------
# Batch checking
# ---------------------------------------------------------------------------

class TestBatchChecking:
    def test_batch_check(self):
        guard = SimilarityGuard(threshold=0.8)
        guard.add_reference("doc1", "The quick brown fox")
        results = guard.check_batch(["The quick brown fox", "Hello world"])
        assert len(results) == 2
        assert results[0].is_similar
        assert not results[1].is_similar

    def test_empty_batch(self):
        guard = SimilarityGuard()
        results = guard.check_batch([])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Duplicate finding
# ---------------------------------------------------------------------------

class TestDuplicateFinding:
    def test_find_duplicates(self):
        guard = SimilarityGuard(threshold=0.8)
        texts = {
            "a": "The quick brown fox jumps over the lazy dog",
            "b": "The quick brown fox jumps over the lazy dog",
            "c": "Completely different text about something else",
        }
        dupes = guard.find_duplicates(texts)
        assert len(dupes) == 1
        assert dupes[0][2] == 1.0  # a and b are identical

    def test_no_duplicates(self):
        guard = SimilarityGuard(threshold=0.8)
        texts = {
            "a": "The quick brown fox",
            "b": "Quantum mechanics and physics",
            "c": "Cooking recipes for dinner",
        }
        dupes = guard.find_duplicates(texts)
        assert len(dupes) == 0

    def test_custom_threshold(self):
        guard = SimilarityGuard(threshold=0.8)
        texts = {
            "a": "Hello world this is a test",
            "b": "Hello world this is a test!",
        }
        dupes_strict = guard.find_duplicates(texts, threshold=1.0)
        dupes_loose = guard.find_duplicates(texts, threshold=0.5)
        assert len(dupes_strict) == 0
        assert len(dupes_loose) == 1

    def test_multiple_duplicates(self):
        guard = SimilarityGuard(threshold=0.9)
        texts = {
            "a": "The quick brown fox jumps",
            "b": "The quick brown fox jumps",
            "c": "The quick brown fox jumps",
        }
        dupes = guard.find_duplicates(texts)
        assert len(dupes) == 3  # (a,b), (a,c), (b,c)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_invalid_threshold_high(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SimilarityGuard(threshold=1.5)

    def test_invalid_threshold_low(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SimilarityGuard(threshold=-0.1)

    def test_invalid_ngram_size(self):
        with pytest.raises(ValueError, match="ngram_size"):
            SimilarityGuard(ngram_size=0)

    def test_custom_ngram_size(self):
        guard = SimilarityGuard(ngram_size=2)
        sim = guard.check_pair("Hello", "Hello")
        assert sim == 1.0

    def test_threshold_zero_catches_all(self):
        guard = SimilarityGuard(threshold=0.0)
        guard.add_reference("doc1", "The quick brown fox")
        result = guard.check("Something entirely different but sharing a few chars")
        # With threshold 0.0, even tiny overlap counts
        if result.matches:
            assert result.is_similar


# ---------------------------------------------------------------------------
# SimilarityResult properties
# ---------------------------------------------------------------------------

class TestSimilarityResult:
    def test_most_similar_none(self):
        result = SimilarityResult(text="test", matches=[], threshold=0.8)
        assert result.most_similar is None

    def test_above_threshold_empty(self):
        result = SimilarityResult(
            text="test",
            matches=[SimilarityMatch("doc1", 0.3)],
            threshold=0.8,
        )
        assert len(result.above_threshold) == 0

    def test_not_similar_below_threshold(self):
        result = SimilarityResult(
            text="test",
            matches=[SimilarityMatch("doc1", 0.5)],
            threshold=0.8,
        )
        assert not result.is_similar


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_short_text(self):
        guard = SimilarityGuard(ngram_size=4)
        guard.add_reference("doc1", "Hi")
        result = guard.check("Hi")
        assert result.most_similar is not None

    def test_no_references(self):
        guard = SimilarityGuard()
        result = guard.check("Some text to check")
        assert not result.is_similar
        assert result.most_similar is None

    def test_case_insensitive(self):
        guard = SimilarityGuard(threshold=0.8)
        guard.add_reference("doc1", "THE QUICK BROWN FOX")
        result = guard.check("the quick brown fox")
        assert result.most_similar is not None
        assert result.most_similar[1] == 1.0
