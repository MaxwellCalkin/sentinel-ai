"""Output diversity checking for LLM responses.

Detects when LLM outputs become repetitive, template-like, or lack diversity
across multiple responses. Useful for detecting model collapse, lazy outputs,
and canned responses.

Uses word-level n-gram Jaccard similarity (no external dependencies).

Usage:
    from sentinel.output_diversity import OutputDiversityChecker

    checker = OutputDiversityChecker(ngram_size=3, duplicate_threshold=0.8)

    score = checker.score("The quick brown fox jumps over the lazy dog.")
    print(score.overall)  # 0.0-1.0

    result = checker.compare("Hello world", "Hello world")
    print(result.is_duplicate)  # True

    report = checker.analyze_batch(["text one", "text two", "text one again"])
    print(report.duplicate_pairs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class DiversityScore:
    """Diversity metrics for a single text."""

    text: str
    uniqueness: float
    vocabulary_richness: float
    structural_variety: float
    overall: float


@dataclass
class ComparisonResult:
    """Pairwise comparison between two texts."""

    text_a: str
    text_b: str
    similarity: float
    shared_ngrams: int
    is_duplicate: bool


@dataclass
class DiversityReport:
    """Aggregate diversity analysis across multiple texts."""

    texts_analyzed: int
    avg_uniqueness: float
    avg_vocabulary: float
    duplicate_pairs: list[tuple[int, int]]
    overall_diversity: float


@dataclass
class DiversityStats:
    """Cumulative statistics from all operations."""

    total_checked: int = 0
    duplicates_found: int = 0
    avg_overall: float = 0.0


_SENTENCE_SPLITTER = re.compile(r'[.!?]+')


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase words."""
    return text.lower().split()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation boundaries."""
    parts = _SENTENCE_SPLITTER.split(text)
    return [s.strip() for s in parts if s.strip()]


def _word_stem(word: str) -> str:
    """Crude stemming: first 4 characters of lowercased word."""
    return word.lower()[:4]


def _extract_word_ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    """Build a set of word-level n-grams."""
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


class OutputDiversityChecker:
    """Detect repetitive or template-like LLM outputs.

    Zero external dependencies -- uses word-level n-grams for fast,
    language-agnostic diversity checking.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        duplicate_threshold: float = 0.8,
    ) -> None:
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be >= 1, got {ngram_size}")
        if not 0.0 <= duplicate_threshold <= 1.0:
            raise ValueError(
                f"duplicate_threshold must be between 0.0 and 1.0, got {duplicate_threshold}"
            )
        self._ngram_size = ngram_size
        self._duplicate_threshold = duplicate_threshold
        self._stats = DiversityStats()
        self._overall_sum: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str) -> DiversityScore:
        """Score a single text's internal diversity."""
        words = _tokenize(text)
        uniqueness = self._compute_uniqueness(words)
        vocabulary_richness = self._compute_vocabulary_richness(words)
        structural_variety = self._compute_structural_variety(text)
        overall = self._weighted_overall(uniqueness, vocabulary_richness, structural_variety)

        self._record_score(overall)

        return DiversityScore(
            text=text,
            uniqueness=round(uniqueness, 4),
            vocabulary_richness=round(vocabulary_richness, 4),
            structural_variety=round(structural_variety, 4),
            overall=round(overall, 4),
        )

    def compare(self, text_a: str, text_b: str) -> ComparisonResult:
        """Compare two texts for similarity using word n-gram Jaccard."""
        words_a = _tokenize(text_a)
        words_b = _tokenize(text_b)
        ngrams_a = _extract_word_ngrams(words_a, self._ngram_size)
        ngrams_b = _extract_word_ngrams(words_b, self._ngram_size)

        shared = len(ngrams_a & ngrams_b)
        similarity = _jaccard(ngrams_a, ngrams_b)
        is_duplicate = similarity >= self._duplicate_threshold

        if is_duplicate:
            self._stats.duplicates_found += 1

        return ComparisonResult(
            text_a=text_a,
            text_b=text_b,
            similarity=round(similarity, 4),
            shared_ngrams=shared,
            is_duplicate=is_duplicate,
        )

    def analyze_batch(self, texts: list[str]) -> DiversityReport:
        """Analyze diversity across multiple outputs."""
        scores = [self.score(t) for t in texts]

        avg_uniqueness = self._safe_mean([s.uniqueness for s in scores])
        avg_vocabulary = self._safe_mean([s.vocabulary_richness for s in scores])

        duplicate_pairs = self._find_duplicate_pairs(texts)
        pair_count = len(texts) * (len(texts) - 1) / 2 if len(texts) > 1 else 1
        duplicate_ratio = len(duplicate_pairs) / pair_count if pair_count else 0.0
        overall_diversity = self._batch_overall(avg_uniqueness, avg_vocabulary, duplicate_ratio)

        return DiversityReport(
            texts_analyzed=len(texts),
            avg_uniqueness=round(avg_uniqueness, 4),
            avg_vocabulary=round(avg_vocabulary, 4),
            duplicate_pairs=duplicate_pairs,
            overall_diversity=round(overall_diversity, 4),
        )

    def stats(self) -> DiversityStats:
        """Return cumulative stats from all scoring operations."""
        return DiversityStats(
            total_checked=self._stats.total_checked,
            duplicates_found=self._stats.duplicates_found,
            avg_overall=round(
                self._overall_sum / self._stats.total_checked
                if self._stats.total_checked
                else 0.0,
                4,
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_uniqueness(words: list[str]) -> float:
        """Ratio of unique words to total words."""
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    @staticmethod
    def _compute_vocabulary_richness(words: list[str]) -> float:
        """Ratio of unique word stems (first 4 chars) to total words."""
        if not words:
            return 0.0
        stems = {_word_stem(w) for w in words}
        return len(stems) / len(words)

    @staticmethod
    def _compute_structural_variety(text: str) -> float:
        """Ratio of unique sentence lengths to total sentences."""
        sentences = _split_sentences(text)
        if not sentences:
            return 0.0
        lengths = [len(s.split()) for s in sentences]
        return len(set(lengths)) / len(lengths)

    @staticmethod
    def _weighted_overall(
        uniqueness: float,
        vocabulary_richness: float,
        structural_variety: float,
    ) -> float:
        """Weighted average: 0.4 * uniqueness + 0.3 * vocabulary + 0.3 * structural."""
        return 0.4 * uniqueness + 0.3 * vocabulary_richness + 0.3 * structural_variety

    def _find_duplicate_pairs(self, texts: list[str]) -> list[tuple[int, int]]:
        """Return index pairs whose n-gram similarity exceeds the threshold."""
        ngrams_list = [
            _extract_word_ngrams(_tokenize(t), self._ngram_size) for t in texts
        ]
        pairs: list[tuple[int, int]] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = _jaccard(ngrams_list[i], ngrams_list[j])
                if similarity >= self._duplicate_threshold:
                    pairs.append((i, j))
        return pairs

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        """Mean that returns 0.0 for empty lists."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _batch_overall(
        avg_uniqueness: float,
        avg_vocabulary: float,
        duplicate_ratio: float,
    ) -> float:
        """Overall diversity for a batch, penalizing duplicates."""
        # Higher duplicate_ratio means less diversity
        individual_quality = 0.4 * avg_uniqueness + 0.3 * avg_vocabulary
        duplication_penalty = 0.3 * (1.0 - duplicate_ratio)
        return individual_quality + duplication_penalty

    def _record_score(self, overall: float) -> None:
        """Track cumulative scoring stats."""
        self._stats.total_checked += 1
        self._overall_sum += overall
