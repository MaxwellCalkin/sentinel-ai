"""Context validation for LLM response grounding.

Validates that LLM responses stay grounded in provided context passages,
detecting hallucinations and unsupported claims via word overlap scoring.

Usage:
    from sentinel.context_validator import ContextValidator

    validator = ContextValidator()
    result = validator.validate(
        response="Paris is the capital of France.",
        context=["France is a country in Europe. Its capital is Paris."],
    )
    print(result.is_grounded)       # True
    print(result.grounding_score)   # 0.75
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroundingCheck:
    """Result of checking a single sentence against context."""
    sentence: str
    supported: bool
    best_match_score: float
    best_match_passage: str


@dataclass
class GroundingResult:
    """Result of validating a full response against context."""
    response: str
    grounding_score: float
    is_grounded: bool
    checks: list[GroundingCheck] = field(default_factory=list)
    unsupported_count: int = 0


@dataclass
class GroundingStats:
    """Aggregate statistics across multiple validations."""
    total_validated: int
    avg_grounding_score: float
    fully_grounded_count: int


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "not", "no", "so", "if", "than",
})


def _extract_content_words(text: str) -> set[str]:
    """Extract meaningful words, excluding stop words and single characters."""
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def _jaccard_similarity(words_a: set[str], words_b: set[str]) -> float:
    """Compute Jaccard similarity between two word sets."""
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _find_best_passage_match(
    sentence_words: set[str],
    passages: list[str],
) -> tuple[float, str]:
    """Find the passage with the highest Jaccard similarity to the sentence.

    Each passage is also split into its own sentences for finer matching.
    Returns (best_score, best_passage_text).
    """
    best_score = 0.0
    best_passage = ""

    for passage in passages:
        passage_sentences = _split_sentences(passage)
        candidates = [passage] + passage_sentences

        for candidate in candidates:
            candidate_words = _extract_content_words(candidate)
            score = _jaccard_similarity(sentence_words, candidate_words)
            if score > best_score:
                best_score = score
                best_passage = candidate

    return best_score, best_passage


class ContextValidator:
    """Validate that LLM responses are grounded in provided context.

    Uses Jaccard word overlap to score how well each response
    sentence is supported by the given context passages.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        min_grounding_score: float = 0.5,
    ) -> None:
        """
        Args:
            similarity_threshold: Minimum Jaccard score for a sentence
                to be considered supported by a passage.
            min_grounding_score: Minimum overall grounding score for the
                response to be flagged as grounded.
        """
        self._similarity_threshold = similarity_threshold
        self._min_grounding_score = min_grounding_score
        self._results: list[GroundingResult] = []

    def validate(
        self,
        response: str,
        context: list[str],
    ) -> GroundingResult:
        """Validate that a response is grounded in the given context passages.

        Args:
            response: The LLM response text to validate.
            context: List of context passages to check against.

        Returns:
            GroundingResult with per-sentence analysis.
        """
        sentences = _split_sentences(response)
        if not sentences:
            result = GroundingResult(
                response=response,
                grounding_score=1.0,
                is_grounded=True,
                checks=[],
                unsupported_count=0,
            )
            self._results.append(result)
            return result

        checks = self._check_sentences(sentences, context)
        unsupported_count = sum(1 for check in checks if not check.supported)
        grounding_score = self._compute_grounding_score(checks)
        is_grounded = grounding_score >= self._min_grounding_score

        result = GroundingResult(
            response=response,
            grounding_score=round(grounding_score, 4),
            is_grounded=is_grounded,
            checks=checks,
            unsupported_count=unsupported_count,
        )
        self._results.append(result)
        return result

    def validate_batch(
        self,
        items: list[tuple[str, list[str]]],
    ) -> list[GroundingResult]:
        """Validate multiple (response, context) pairs.

        Args:
            items: List of (response, context_passages) tuples.

        Returns:
            List of GroundingResult, one per input pair.
        """
        return [self.validate(response, context) for response, context in items]

    def stats(self) -> GroundingStats:
        """Return aggregate statistics across all validations performed.

        Returns:
            GroundingStats with counts and averages.
        """
        total = len(self._results)
        if total == 0:
            return GroundingStats(
                total_validated=0,
                avg_grounding_score=0.0,
                fully_grounded_count=0,
            )

        avg_score = sum(r.grounding_score for r in self._results) / total
        fully_grounded = sum(
            1 for r in self._results if r.unsupported_count == 0
        )

        return GroundingStats(
            total_validated=total,
            avg_grounding_score=round(avg_score, 4),
            fully_grounded_count=fully_grounded,
        )

    def _check_sentences(
        self,
        sentences: list[str],
        context: list[str],
    ) -> list[GroundingCheck]:
        """Check each sentence against all context passages."""
        checks: list[GroundingCheck] = []
        for sentence in sentences:
            sentence_words = _extract_content_words(sentence)
            if not sentence_words:
                checks.append(GroundingCheck(
                    sentence=sentence,
                    supported=True,
                    best_match_score=1.0,
                    best_match_passage="",
                ))
                continue

            best_score, best_passage = _find_best_passage_match(
                sentence_words, context,
            )
            supported = best_score >= self._similarity_threshold

            checks.append(GroundingCheck(
                sentence=sentence,
                supported=supported,
                best_match_score=round(best_score, 4),
                best_match_passage=best_passage,
            ))
        return checks

    def _compute_grounding_score(
        self,
        checks: list[GroundingCheck],
    ) -> float:
        """Compute fraction of sentences that are supported."""
        if not checks:
            return 1.0
        supported_count = sum(1 for check in checks if check.supported)
        return supported_count / len(checks)
