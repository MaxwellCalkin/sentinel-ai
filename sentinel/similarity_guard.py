"""Similarity-based content deduplication and copy detection.

Detects when LLM outputs are suspiciously similar to reference texts,
previous outputs, or each other — useful for detecting memorization,
plagiarism, and output caching issues.

Uses character n-gram Jaccard similarity (no external dependencies).

Usage:
    from sentinel.similarity_guard import SimilarityGuard

    guard = SimilarityGuard(threshold=0.8)
    guard.add_reference("original_doc", "The quick brown fox...")

    result = guard.check("The quick brown fox...")
    assert result.is_similar
    print(result.most_similar)  # ("original_doc", 1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SimilarityMatch:
    """A similarity match against a reference text."""
    reference_id: str
    similarity: float


@dataclass
class SimilarityResult:
    """Result of similarity checking."""
    text: str
    matches: list[SimilarityMatch] = field(default_factory=list)
    threshold: float = 0.8

    @property
    def is_similar(self) -> bool:
        """True if any reference exceeds the threshold."""
        return any(m.similarity >= self.threshold for m in self.matches)

    @property
    def most_similar(self) -> tuple[str, float] | None:
        """Return the most similar reference (id, score) or None."""
        if not self.matches:
            return None
        best = max(self.matches, key=lambda m: m.similarity)
        return (best.reference_id, best.similarity)

    @property
    def above_threshold(self) -> list[SimilarityMatch]:
        """Return all matches above the threshold."""
        return [m for m in self.matches if m.similarity >= self.threshold]


class SimilarityGuard:
    """Detect similar/duplicate content using n-gram Jaccard similarity.

    Zero external dependencies — uses character-level n-grams for fast,
    language-agnostic similarity computation.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        ngram_size: int = 4,
    ):
        """
        Args:
            threshold: Similarity threshold (0.0-1.0) above which content
                      is considered similar. Default 0.8.
            ngram_size: Character n-gram size. Default 4.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be >= 1, got {ngram_size}")

        self._threshold = threshold
        self._ngram_size = ngram_size
        self._references: dict[str, set[str]] = {}

    def add_reference(self, ref_id: str, text: str) -> None:
        """Add a reference text to check against."""
        self._references[ref_id] = self._extract_ngrams(text)

    def add_references(self, refs: dict[str, str]) -> None:
        """Add multiple reference texts."""
        for ref_id, text in refs.items():
            self.add_reference(ref_id, text)

    def remove_reference(self, ref_id: str) -> None:
        """Remove a reference text."""
        self._references.pop(ref_id, None)

    def clear(self) -> None:
        """Remove all reference texts."""
        self._references.clear()

    @property
    def reference_count(self) -> int:
        """Number of reference texts stored."""
        return len(self._references)

    def check(self, text: str) -> SimilarityResult:
        """Check text similarity against all references."""
        text_ngrams = self._extract_ngrams(text)
        matches: list[SimilarityMatch] = []

        for ref_id, ref_ngrams in self._references.items():
            sim = self._jaccard_similarity(text_ngrams, ref_ngrams)
            if sim > 0:
                matches.append(SimilarityMatch(
                    reference_id=ref_id,
                    similarity=round(sim, 4),
                ))

        matches.sort(key=lambda m: m.similarity, reverse=True)

        return SimilarityResult(
            text=text,
            matches=matches,
            threshold=self._threshold,
        )

    def check_pair(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two texts directly."""
        ngrams_a = self._extract_ngrams(text_a)
        ngrams_b = self._extract_ngrams(text_b)
        return round(self._jaccard_similarity(ngrams_a, ngrams_b), 4)

    def check_batch(self, texts: list[str]) -> list[SimilarityResult]:
        """Check multiple texts against references."""
        return [self.check(text) for text in texts]

    def find_duplicates(
        self, texts: dict[str, str], threshold: float | None = None
    ) -> list[tuple[str, str, float]]:
        """Find all pairs of texts exceeding the similarity threshold.

        Args:
            texts: Mapping of text IDs to text content.
            threshold: Override threshold for this check.

        Returns:
            List of (id_a, id_b, similarity) tuples.
        """
        thresh = threshold if threshold is not None else self._threshold
        keys = list(texts.keys())
        ngrams = {k: self._extract_ngrams(v) for k, v in texts.items()}
        duplicates: list[tuple[str, str, float]] = []

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                sim = self._jaccard_similarity(ngrams[keys[i]], ngrams[keys[j]])
                if sim >= thresh:
                    duplicates.append((keys[i], keys[j], round(sim, 4)))

        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates

    def _extract_ngrams(self, text: str) -> set[str]:
        """Extract character n-grams from text."""
        text = text.lower().strip()
        if len(text) < self._ngram_size:
            return {text} if text else set()
        return {text[i:i + self._ngram_size] for i in range(len(text) - self._ngram_size + 1)}

    @staticmethod
    def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
