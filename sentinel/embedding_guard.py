"""Embedding-based semantic similarity guard.

Detects semantically similar inputs using TF-IDF-like bag-of-words
vectors with cosine similarity. Zero external dependencies.

Usage:
    from sentinel.embedding_guard import EmbeddingGuard

    guard = EmbeddingGuard(threshold=0.85)
    guard.add_blocked("how to make a bomb")
    result = guard.check("instructions for building an explosive device")
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SemanticMatch:
    """A semantic similarity match."""
    reference_id: str
    reference_text: str
    similarity: float
    blocked: bool


@dataclass
class SemanticResult:
    """Result of semantic similarity check."""
    text: str
    matches: list[SemanticMatch] = field(default_factory=list)
    blocked: bool = False
    top_similarity: float = 0.0

    @property
    def match_count(self) -> int:
        return len(self.matches)


# Common English stop words to ignore
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom", "how",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "up", "out", "all", "some", "any", "each",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize and filter stop words."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    """Compute normalized term frequency."""
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    # Dot product
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    # Magnitudes
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class EmbeddingGuard:
    """Semantic similarity guard using bag-of-words cosine similarity.

    No external embedding model needed — uses TF-based vectors
    with cosine similarity for zero-dependency semantic matching.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        use_idf: bool = True,
    ) -> None:
        """
        Args:
            threshold: Similarity threshold to flag/block (0.0-1.0).
            use_idf: Weight terms by inverse document frequency.
        """
        self._threshold = threshold
        self._use_idf = use_idf
        self._references: list[tuple[str, str, dict[str, float]]] = []  # (id, text, tf)
        self._doc_freq: Counter = Counter()
        self._doc_count = 0

    def add_blocked(
        self,
        text: str,
        reference_id: str | None = None,
    ) -> str:
        """Add a reference text to block against.

        Args:
            text: Text to match against.
            reference_id: Optional identifier.

        Returns:
            The reference_id.
        """
        ref_id = reference_id or f"ref_{len(self._references)}"
        tokens = _tokenize(text)
        tf = _term_freq(tokens)

        # Update IDF
        unique_tokens = set(tokens)
        self._doc_freq.update(unique_tokens)
        self._doc_count += 1

        self._references.append((ref_id, text, tf))
        return ref_id

    def add_blocked_batch(self, texts: list[str]) -> list[str]:
        """Add multiple reference texts."""
        return [self.add_blocked(t) for t in texts]

    @property
    def reference_count(self) -> int:
        return len(self._references)

    def check(self, text: str) -> SemanticResult:
        """Check text against all references.

        Returns:
            SemanticResult with matches above threshold.
        """
        tokens = _tokenize(text)
        tf = _term_freq(tokens)

        if self._use_idf and self._doc_count > 0:
            tf = self._apply_idf(tf)

        matches: list[SemanticMatch] = []
        top_sim = 0.0

        for ref_id, ref_text, ref_tf in self._references:
            ref_weighted = self._apply_idf(ref_tf) if self._use_idf and self._doc_count > 0 else ref_tf
            sim = _cosine_similarity(tf, ref_weighted)

            if sim >= self._threshold:
                matches.append(SemanticMatch(
                    reference_id=ref_id,
                    reference_text=ref_text,
                    similarity=round(sim, 4),
                    blocked=True,
                ))
            top_sim = max(top_sim, sim)

        return SemanticResult(
            text=text,
            matches=matches,
            blocked=len(matches) > 0,
            top_similarity=round(top_sim, 4),
        )

    def check_pair(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two texts."""
        tf_a = _term_freq(_tokenize(text_a))
        tf_b = _term_freq(_tokenize(text_b))
        return round(_cosine_similarity(tf_a, tf_b), 4)

    def clear(self) -> None:
        """Remove all references."""
        self._references.clear()
        self._doc_freq.clear()
        self._doc_count = 0

    def _apply_idf(self, tf: dict[str, float]) -> dict[str, float]:
        """Apply IDF weighting to term frequencies."""
        result: dict[str, float] = {}
        for term, freq in tf.items():
            df = self._doc_freq.get(term, 0)
            idf = math.log((self._doc_count + 1) / (df + 1)) + 1
            result[term] = freq * idf
        return result
