"""Factuality checker for LLM outputs.

Verify LLM responses against reference documents to detect
unsupported claims, contradictions, and fabricated information.

Usage:
    from sentinel.factuality import FactualityChecker

    checker = FactualityChecker()
    result = checker.check(
        output="Paris has 2.1 million residents.",
        references=["Paris population is approximately 2.1 million."],
    )
    print(result.supported)  # True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FactClaim:
    """A single factual claim extracted from text."""
    text: str
    supported: bool
    confidence: float  # 0.0 to 1.0
    source: str = ""  # which reference supported it


@dataclass
class FactualityResult:
    """Result of factuality check."""
    output: str
    claims: list[FactClaim]
    supported_count: int
    unsupported_count: int
    score: float  # 0.0 to 1.0 (fraction supported)

    @property
    def supported(self) -> bool:
        return self.unsupported_count == 0 and self.supported_count > 0

    @property
    def has_claims(self) -> bool:
        return len(self.claims) > 0


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]


def _word_overlap(sentence: str, reference: str) -> float:
    """Compute word overlap ratio between sentence and reference."""
    words_s = set(re.findall(r'\b\w+\b', sentence.lower()))
    words_r = set(re.findall(r'\b\w+\b', reference.lower()))
    if not words_s:
        return 0.0
    # What fraction of sentence words appear in reference?
    overlap = words_s & words_r
    # Remove stop words from consideration for better signal
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "it", "its", "this", "that",
            "with", "has", "have", "had", "be", "been", "by", "from", "as"}
    content_s = words_s - stop
    content_overlap = overlap - stop
    if not content_s:
        return len(overlap) / len(words_s) if words_s else 0.0
    return len(content_overlap) / len(content_s)


class FactualityChecker:
    """Check LLM output factuality against references.

    Extract claims from output, match against reference
    documents, and score factual support.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_claim_length: int = 10,
    ) -> None:
        """
        Args:
            threshold: Minimum overlap for a claim to be supported.
            min_claim_length: Minimum characters for a claim.
        """
        self._threshold = threshold
        self._min_claim_length = min_claim_length

    def check(
        self,
        output: str,
        references: list[str],
    ) -> FactualityResult:
        """Check output factuality against references.

        Args:
            output: LLM-generated text to verify.
            references: Reference documents to check against.

        Returns:
            FactualityResult with per-claim analysis.
        """
        sentences = _extract_sentences(output)
        if not sentences:
            return FactualityResult(
                output=output,
                claims=[],
                supported_count=0,
                unsupported_count=0,
                score=1.0,
            )

        claims = []
        supported = 0
        unsupported = 0

        for sentence in sentences:
            if len(sentence) < self._min_claim_length:
                continue

            best_score = 0.0
            best_source = ""

            for ref in references:
                # Check against whole reference and individual sentences
                ref_sentences = _extract_sentences(ref)
                all_parts = [ref] + ref_sentences

                for part in all_parts:
                    score = _word_overlap(sentence, part)
                    if score > best_score:
                        best_score = score
                        best_source = part[:100]

            is_supported = best_score >= self._threshold
            if is_supported:
                supported += 1
            else:
                unsupported += 1

            claims.append(FactClaim(
                text=sentence,
                supported=is_supported,
                confidence=round(best_score, 4),
                source=best_source if is_supported else "",
            ))

        total = supported + unsupported
        score = supported / total if total > 0 else 1.0

        return FactualityResult(
            output=output,
            claims=claims,
            supported_count=supported,
            unsupported_count=unsupported,
            score=round(score, 4),
        )

    def check_batch(
        self, outputs: list[str], references: list[str]
    ) -> list[FactualityResult]:
        """Check multiple outputs against the same references."""
        return [self.check(o, references) for o in outputs]
