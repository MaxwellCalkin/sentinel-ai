"""Tests for the output diversity checker."""

import pytest
from sentinel.output_diversity import (
    OutputDiversityChecker,
    DiversityScore,
    ComparisonResult,
    DiversityReport,
    DiversityStats,
)


# ---------------------------------------------------------------------------
# Single text scoring — diverse text
# ---------------------------------------------------------------------------

class TestScoreDiverseText:
    def test_diverse_text_has_high_uniqueness(self):
        checker = OutputDiversityChecker()
        score = checker.score(
            "The quick brown fox jumps over the lazy dog near a stream."
        )
        assert score.uniqueness > 0.8

    def test_diverse_text_has_high_vocabulary_richness(self):
        checker = OutputDiversityChecker()
        score = checker.score(
            "Elephants migrate across vast savannas during seasonal rains."
        )
        assert score.vocabulary_richness > 0.7

    def test_diverse_text_overall_above_half(self):
        checker = OutputDiversityChecker()
        score = checker.score(
            "Stars illuminate the night sky. Planets orbit in silence. "
            "Galaxies drift apart slowly."
        )
        assert score.overall > 0.5


# ---------------------------------------------------------------------------
# Single text scoring — repetitive text
# ---------------------------------------------------------------------------

class TestScoreRepetitiveText:
    def test_repetitive_text_low_uniqueness(self):
        checker = OutputDiversityChecker()
        score = checker.score("the the the the the the the the")
        assert score.uniqueness < 0.2

    def test_repetitive_text_low_vocabulary(self):
        checker = OutputDiversityChecker()
        score = checker.score("go go go go go go go go go go")
        assert score.vocabulary_richness < 0.2

    def test_repetitive_text_low_overall(self):
        checker = OutputDiversityChecker()
        # Repeating sentences to keep structural_variety low too
        score = checker.score("bad bad bad. bad bad bad. bad bad bad. bad bad bad.")
        assert score.overall < 0.3


# ---------------------------------------------------------------------------
# Comparison — similar texts
# ---------------------------------------------------------------------------

class TestCompareSimilarTexts:
    def test_identical_texts_are_duplicates(self):
        checker = OutputDiversityChecker()
        result = checker.compare(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )
        assert result.is_duplicate
        assert result.similarity == 1.0

    def test_nearly_identical_texts_high_similarity(self):
        checker = OutputDiversityChecker()
        result = checker.compare(
            "The quick brown fox jumps over the lazy dog today",
            "The quick brown fox jumps over the lazy dog yesterday",
        )
        assert result.similarity > 0.5


# ---------------------------------------------------------------------------
# Comparison — different texts
# ---------------------------------------------------------------------------

class TestCompareDifferentTexts:
    def test_unrelated_texts_not_duplicates(self):
        checker = OutputDiversityChecker()
        result = checker.compare(
            "The weather is sunny and warm today in the park",
            "Quantum computing leverages superposition and entanglement for calculations",
        )
        assert not result.is_duplicate
        assert result.similarity < 0.3

    def test_different_topics_low_shared_ngrams(self):
        checker = OutputDiversityChecker()
        result = checker.compare(
            "Cooking pasta requires boiling water and adding salt",
            "Astrophysics studies the behavior of celestial bodies in space",
        )
        assert result.shared_ngrams == 0


# ---------------------------------------------------------------------------
# Batch analysis — with duplicates
# ---------------------------------------------------------------------------

class TestBatchWithDuplicates:
    def test_detects_duplicate_pair(self):
        checker = OutputDiversityChecker()
        texts = [
            "The quick brown fox jumps over the lazy dog near the river",
            "The quick brown fox jumps over the lazy dog near the river",
            "Quantum computing will transform the technology landscape significantly",
        ]
        report = checker.analyze_batch(texts)
        assert (0, 1) in report.duplicate_pairs

    def test_duplicate_pair_count(self):
        checker = OutputDiversityChecker()
        texts = [
            "Alpha beta gamma delta epsilon zeta eta theta",
            "Alpha beta gamma delta epsilon zeta eta theta",
            "Alpha beta gamma delta epsilon zeta eta theta",
        ]
        report = checker.analyze_batch(texts)
        assert len(report.duplicate_pairs) == 3  # (0,1), (0,2), (1,2)

    def test_overall_diversity_penalized_by_duplicates(self):
        checker = OutputDiversityChecker()
        all_same = [
            "Repeated text repeated text repeated text repeated text"
        ] * 4
        all_different = [
            "Stars illuminate the night sky brightly and clearly overhead",
            "Oceans cover most of the Earth surface with deep blue water",
            "Mountains rise above the clouds in majestic towering peaks",
            "Rivers flow through valleys carrying sediment and fresh water downstream",
        ]
        report_same = checker.analyze_batch(all_same)
        report_diff = checker.analyze_batch(all_different)
        assert report_diff.overall_diversity > report_same.overall_diversity


# ---------------------------------------------------------------------------
# Batch analysis — without duplicates
# ---------------------------------------------------------------------------

class TestBatchWithoutDuplicates:
    def test_no_duplicate_pairs(self):
        checker = OutputDiversityChecker()
        texts = [
            "Artificial intelligence is transforming modern healthcare delivery systems",
            "The stock market fluctuated wildly during the third fiscal quarter",
            "Ancient Roman aqueducts were marvels of early civil engineering design",
        ]
        report = checker.analyze_batch(texts)
        assert report.duplicate_pairs == []

    def test_high_overall_diversity(self):
        checker = OutputDiversityChecker()
        texts = [
            "Elephants migrate across vast savannas during dry seasonal periods",
            "Quantum entanglement enables faster-than-light information correlation experiments",
            "Baroque music emphasizes ornamentation and expressive harmonic complexity throughout",
        ]
        report = checker.analyze_batch(texts)
        assert report.overall_diversity > 0.7


# ---------------------------------------------------------------------------
# Empty and edge-case text handling
# ---------------------------------------------------------------------------

class TestEmptyTextHandling:
    def test_empty_string_scores_zero(self):
        checker = OutputDiversityChecker()
        score = checker.score("")
        assert score.uniqueness == 0.0
        assert score.vocabulary_richness == 0.0
        assert score.structural_variety == 0.0
        assert score.overall == 0.0

    def test_compare_empty_strings(self):
        checker = OutputDiversityChecker()
        result = checker.compare("", "")
        assert result.similarity == 0.0
        assert result.shared_ngrams == 0

    def test_compare_one_empty(self):
        checker = OutputDiversityChecker()
        result = checker.compare("Hello world foo bar baz", "")
        assert result.similarity == 0.0

    def test_empty_batch(self):
        checker = OutputDiversityChecker()
        report = checker.analyze_batch([])
        assert report.texts_analyzed == 0
        assert report.duplicate_pairs == []


# ---------------------------------------------------------------------------
# Single word text
# ---------------------------------------------------------------------------

class TestSingleWordText:
    def test_single_word_uniqueness_is_one(self):
        checker = OutputDiversityChecker()
        score = checker.score("hello")
        assert score.uniqueness == 1.0

    def test_single_word_vocabulary_is_one(self):
        checker = OutputDiversityChecker()
        score = checker.score("hello")
        assert score.vocabulary_richness == 1.0

    def test_single_word_structural_variety(self):
        checker = OutputDiversityChecker()
        score = checker.score("hello")
        # No sentence-ending punctuation, so one "sentence" with unique length
        assert score.structural_variety == 1.0


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStatsTracking:
    def test_stats_start_at_zero(self):
        checker = OutputDiversityChecker()
        s = checker.stats()
        assert s.total_checked == 0
        assert s.duplicates_found == 0
        assert s.avg_overall == 0.0

    def test_stats_increment_after_score(self):
        checker = OutputDiversityChecker()
        checker.score("Some diverse text with many different words here")
        s = checker.stats()
        assert s.total_checked == 1
        assert s.avg_overall > 0.0

    def test_stats_track_duplicates_from_compare(self):
        checker = OutputDiversityChecker()
        checker.compare(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )
        s = checker.stats()
        assert s.duplicates_found == 1

    def test_stats_avg_overall_accumulates(self):
        checker = OutputDiversityChecker()
        checker.score("word word word word word word word word")
        checker.score(
            "Stars illuminate the night sky. Planets orbit in silence. "
            "Galaxies drift apart slowly."
        )
        s = checker.stats()
        assert s.total_checked == 2
        assert 0.0 < s.avg_overall < 1.0


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

class TestThresholdConfiguration:
    def test_strict_threshold_fewer_duplicates(self):
        checker_strict = OutputDiversityChecker(duplicate_threshold=0.99)
        checker_loose = OutputDiversityChecker(duplicate_threshold=0.3)
        text_a = "The quick brown fox jumps over the lazy dog near the river bank"
        text_b = "The quick brown fox jumps over the lazy cat near the river bank"
        assert not checker_strict.compare(text_a, text_b).is_duplicate
        assert checker_loose.compare(text_a, text_b).is_duplicate

    def test_invalid_threshold_high(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            OutputDiversityChecker(duplicate_threshold=1.5)

    def test_invalid_threshold_low(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            OutputDiversityChecker(duplicate_threshold=-0.1)

    def test_threshold_zero_marks_all_as_duplicate(self):
        checker = OutputDiversityChecker(duplicate_threshold=0.0)
        result = checker.compare(
            "Alpha beta gamma delta epsilon",
            "Zeta eta theta iota kappa",
        )
        # Even with no overlap, threshold=0.0 means similarity >= 0.0 is always true
        assert result.is_duplicate


# ---------------------------------------------------------------------------
# N-gram size effect
# ---------------------------------------------------------------------------

class TestNgramSizeEffect:
    def test_invalid_ngram_size(self):
        with pytest.raises(ValueError, match="ngram_size"):
            OutputDiversityChecker(ngram_size=0)

    def test_larger_ngram_reduces_similarity(self):
        text_a = "the cat sat on the mat by the door near the wall"
        text_b = "the cat sat on the rug by the door near the fence"
        checker_small = OutputDiversityChecker(ngram_size=2, duplicate_threshold=0.5)
        checker_large = OutputDiversityChecker(ngram_size=5, duplicate_threshold=0.5)
        sim_small = checker_small.compare(text_a, text_b).similarity
        sim_large = checker_large.compare(text_a, text_b).similarity
        assert sim_small >= sim_large

    def test_ngram_size_one_uses_word_overlap(self):
        checker = OutputDiversityChecker(ngram_size=1)
        result = checker.compare("hello world", "hello world")
        assert result.similarity == 1.0


# ---------------------------------------------------------------------------
# Dataclass fields
# ---------------------------------------------------------------------------

class TestDataclassFields:
    def test_diversity_score_fields(self):
        checker = OutputDiversityChecker()
        score = checker.score("Some text with words in it for testing diversity")
        assert isinstance(score.text, str)
        assert 0.0 <= score.uniqueness <= 1.0
        assert 0.0 <= score.vocabulary_richness <= 1.0
        assert 0.0 <= score.structural_variety <= 1.0
        assert 0.0 <= score.overall <= 1.0

    def test_comparison_result_fields(self):
        checker = OutputDiversityChecker()
        result = checker.compare("alpha beta gamma", "alpha beta gamma")
        assert isinstance(result.text_a, str)
        assert isinstance(result.text_b, str)
        assert isinstance(result.similarity, float)
        assert isinstance(result.shared_ngrams, int)
        assert isinstance(result.is_duplicate, bool)

    def test_diversity_report_fields(self):
        checker = OutputDiversityChecker()
        report = checker.analyze_batch(["hello world foo bar baz"])
        assert isinstance(report.texts_analyzed, int)
        assert isinstance(report.avg_uniqueness, float)
        assert isinstance(report.avg_vocabulary, float)
        assert isinstance(report.duplicate_pairs, list)
        assert isinstance(report.overall_diversity, float)

    def test_diversity_stats_defaults(self):
        s = DiversityStats()
        assert s.total_checked == 0
        assert s.duplicates_found == 0
        assert s.avg_overall == 0.0
