"""Groundedness checking for RAG applications.

Verifies that LLM outputs are grounded in the provided context,
detecting potential hallucinations by checking claim overlap.

Usage:
    from sentinel.groundedness import GroundednessChecker

    checker = GroundednessChecker()
    result = checker.check(
        response="Paris is the capital of France.",
        context="France is a country in Europe. Its capital is Paris.",
    )
    print(result.grounded)       # True
    print(result.score)          # 0.95
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroundednessResult:
    """Result of groundedness check."""
    response: str
    context: str
    score: float          # 0.0 (ungrounded) to 1.0 (fully grounded)
    grounded: bool
    claims: list[Claim] = field(default_factory=list)
    ungrounded_claims: list[Claim] = field(default_factory=list)

    @property
    def claim_count(self) -> int:
        return len(self.claims)

    @property
    def grounded_ratio(self) -> float:
        if not self.claims:
            return 1.0
        grounded = sum(1 for c in self.claims if c.grounded)
        return grounded / len(self.claims)


@dataclass
class Claim:
    """A claim extracted from the response."""
    text: str
    grounded: bool
    support_score: float  # How well supported by context
    evidence: str = ""    # Matching context snippet


_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Words to ignore when computing overlap
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
    """Extract content words (excluding stop words)."""
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def _extract_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    """Extract word n-grams."""
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS and len(w) > 1]
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


def _sentence_overlap(sentence: str, context: str) -> tuple[float, str]:
    """Compute overlap between a sentence and context.

    Returns (score, best matching context snippet).
    """
    sent_words = _extract_content_words(sentence)
    if not sent_words:
        return 1.0, ""  # Trivial sentence, consider grounded

    ctx_sentences = [s.strip() for s in _SENTENCE_RE.findall(context) if s.strip()]
    if not ctx_sentences:
        return 0.0, ""

    best_score = 0.0
    best_evidence = ""

    for ctx_sent in ctx_sentences:
        ctx_words = _extract_content_words(ctx_sent)
        if not ctx_words:
            continue

        # Word overlap
        overlap = sent_words & ctx_words
        word_score = len(overlap) / len(sent_words) if sent_words else 0

        # N-gram overlap bonus
        sent_ngrams = _extract_ngrams(sentence, 2)
        ctx_ngrams = _extract_ngrams(ctx_sent, 2)
        if sent_ngrams:
            ngram_overlap = len(sent_ngrams & ctx_ngrams) / len(sent_ngrams)
            score = 0.6 * word_score + 0.4 * ngram_overlap
        else:
            score = word_score

        if score > best_score:
            best_score = score
            best_evidence = ctx_sent

    return min(1.0, best_score), best_evidence


class GroundednessChecker:
    """Check if LLM responses are grounded in provided context.

    Uses word overlap and n-gram matching to verify claims
    are supported by the source context.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_claim_words: int = 3,
    ) -> None:
        """
        Args:
            threshold: Minimum score for a claim to be considered grounded.
            min_claim_words: Minimum content words for a sentence to be a claim.
        """
        self._threshold = threshold
        self._min_claim_words = min_claim_words

    def check(
        self,
        response: str,
        context: str,
    ) -> GroundednessResult:
        """Check if response is grounded in context.

        Args:
            response: The LLM response to verify.
            context: The source context/documents.

        Returns:
            GroundednessResult with per-claim analysis.
        """
        sentences = [s.strip() for s in _SENTENCE_RE.findall(response) if s.strip()]

        claims: list[Claim] = []
        for sent in sentences:
            content_words = _extract_content_words(sent)
            if len(content_words) < self._min_claim_words:
                continue

            score, evidence = _sentence_overlap(sent, context)
            grounded = score >= self._threshold
            claims.append(Claim(
                text=sent,
                grounded=grounded,
                support_score=round(score, 4),
                evidence=evidence,
            ))

        ungrounded = [c for c in claims if not c.grounded]

        if not claims:
            overall_score = 1.0
        else:
            overall_score = sum(c.support_score for c in claims) / len(claims)

        return GroundednessResult(
            response=response,
            context=context,
            score=round(overall_score, 4),
            grounded=len(ungrounded) == 0,
            claims=claims,
            ungrounded_claims=ungrounded,
        )

    def check_batch(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[GroundednessResult]:
        """Check multiple (response, context) pairs."""
        return [self.check(resp, ctx) for resp, ctx in pairs]
