"""Response classification for LLM outputs.

Classifies LLM responses into safety categories: safe, refusal,
hedging, harmful, hallucination, off-topic, etc. Useful for monitoring
model behavior and detecting safety issues in production.

Usage:
    from sentinel.response_classifier import ResponseClassifier, ResponseCategory

    classifier = ResponseClassifier()
    result = classifier.classify("I cannot help with that request.")
    print(result.category)      # ResponseCategory.REFUSAL
    print(result.confidence)    # 0.95
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResponseCategory(Enum):
    """Categories of LLM responses."""
    SAFE = "safe"                    # Normal, helpful response
    REFUSAL = "refusal"              # Model refused to answer
    HEDGING = "hedging"              # Uncertain or overly cautious
    HARMFUL = "harmful"              # Contains harmful content
    OFF_TOPIC = "off_topic"          # Doesn't address the question
    HALLUCINATION = "hallucination"  # Likely fabricated content
    ERROR = "error"                  # Error or malformed response
    EMPTY = "empty"                  # No meaningful content


@dataclass
class ClassifyResult:
    """Result of response classification."""
    category: ResponseCategory
    confidence: float          # 0.0 to 1.0
    signals: list[str] = field(default_factory=list)
    subcategory: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.category == ResponseCategory.SAFE

    @property
    def needs_review(self) -> bool:
        return self.category in (
            ResponseCategory.HARMFUL,
            ResponseCategory.HALLUCINATION,
        )


# Pattern definitions for each category
_REFUSAL_PATTERNS = [
    re.compile(r"\bI (?:can't|cannot|won't|will not|am (?:not able|unable)) (?:help|assist|provide|do|create|generate)\b", re.IGNORECASE),
    re.compile(r"\bI'm (?:not able|unable) to\b", re.IGNORECASE),
    re.compile(r"\bI (?:don't|do not) (?:think (?:I should|it's appropriate)|feel comfortable)\b", re.IGNORECASE),
    re.compile(r"\bI (?:must|have to) (?:decline|refuse)\b", re.IGNORECASE),
    re.compile(r"\bagainst my (?:guidelines|policies|programming)\b", re.IGNORECASE),
    re.compile(r"\bAs an AI(?:,| ) I (?:can't|cannot|shouldn't)\b", re.IGNORECASE),
    re.compile(r"\bI'?m sorry,? (?:but )?I (?:can't|cannot)\b", re.IGNORECASE),
    re.compile(r"\bthat (?:request |)(?:goes against|violates)\b", re.IGNORECASE),
]

_HEDGING_PATTERNS = [
    re.compile(r"\bI'?m not (?:entirely |completely )?(?:sure|certain)\b", re.IGNORECASE),
    re.compile(r"\bI (?:don't|do not) have (?:access to|information about)\b", re.IGNORECASE),
    re.compile(r"\bI (?:should|would) note that\b", re.IGNORECASE),
    re.compile(r"\bplease (?:note|be aware|keep in mind)\b", re.IGNORECASE),
    re.compile(r"\bit(?:'s| is) important to (?:note|mention|remember)\b", re.IGNORECASE),
    re.compile(r"\bI (?:might|may|could) be wrong\b", re.IGNORECASE),
    re.compile(r"\bwith (?:the |that )?caveat\b", re.IGNORECASE),
]

_HARMFUL_PATTERNS = [
    re.compile(r"\bhow to (?:make|build|create) (?:a )?(?:bomb|weapon|explosive)\b", re.IGNORECASE),
    re.compile(r"\bstep[- ]by[- ]step (?:guide|instructions?) (?:to|for) (?:hack|attack|exploit)\b", re.IGNORECASE),
    re.compile(r"\bhere(?:'s| is) how (?:to|you can) (?:bypass|circumvent|evade)\b", re.IGNORECASE),
    re.compile(r"\b(?:kill|murder|harm|hurt) (?:yourself|someone|people)\b", re.IGNORECASE),
]

_HALLUCINATION_PATTERNS = [
    re.compile(r"\baccording to (?:a |my )(?:recent )?(?:study|research|report) (?:by|from|in) \w+\b", re.IGNORECASE),
    re.compile(r"\b(?:published|released) (?:on|in) (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? 202[5-9]\b", re.IGNORECASE),
    re.compile(r"\bDr\.\s+[A-Z][a-z]+ [A-Z][a-z]+ (?:at|from|of) (?:the )?(?:University|Institute|Center)\b"),
]

_ERROR_PATTERNS = [
    re.compile(r"\b(?:error|exception|traceback|stack trace)\b", re.IGNORECASE),
    re.compile(r"\bsomething went wrong\b", re.IGNORECASE),
    re.compile(r"\binternal (?:server )?error\b", re.IGNORECASE),
    re.compile(r"\b(?:rate limit|quota) (?:exceeded|reached)\b", re.IGNORECASE),
]


class ResponseClassifier:
    """Classify LLM responses into safety categories.

    Uses pattern matching to detect refusals, hedging, harmful content,
    hallucinations, errors, and other response types.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        check_hallucination: bool = True,
    ):
        """
        Args:
            min_confidence: Minimum confidence to assign a non-SAFE category.
            check_hallucination: Enable hallucination pattern checking.
        """
        self._min_confidence = min_confidence
        self._check_hallucination = check_hallucination

    def classify(self, text: str) -> ClassifyResult:
        """Classify a response text."""
        text = text.strip()

        # Empty check
        if not text or len(text.split()) < 2:
            return ClassifyResult(
                category=ResponseCategory.EMPTY,
                confidence=1.0,
                signals=["Empty or near-empty response"],
            )

        # Check each category and score
        scores: list[tuple[ResponseCategory, float, list[str], str]] = []

        # Refusal detection
        refusal_signals = self._check_patterns(text, _REFUSAL_PATTERNS)
        if refusal_signals:
            confidence = min(1.0, 0.5 + 0.2 * len(refusal_signals))
            scores.append((
                ResponseCategory.REFUSAL, confidence, refusal_signals, ""
            ))

        # Harmful content
        harmful_signals = self._check_patterns(text, _HARMFUL_PATTERNS)
        if harmful_signals:
            confidence = min(1.0, 0.6 + 0.2 * len(harmful_signals))
            scores.append((
                ResponseCategory.HARMFUL, confidence, harmful_signals, ""
            ))

        # Error detection
        error_signals = self._check_patterns(text, _ERROR_PATTERNS)
        if error_signals:
            confidence = min(1.0, 0.4 + 0.2 * len(error_signals))
            scores.append((
                ResponseCategory.ERROR, confidence, error_signals, ""
            ))

        # Hedging detection
        hedging_signals = self._check_patterns(text, _HEDGING_PATTERNS)
        if hedging_signals:
            # Hedging only if it's a significant portion of response
            ratio = len(hedging_signals) / max(1, len(text.split()) / 50)
            confidence = min(0.9, 0.3 + 0.15 * len(hedging_signals))
            if confidence >= self._min_confidence:
                scores.append((
                    ResponseCategory.HEDGING, confidence, hedging_signals, ""
                ))

        # Hallucination detection
        if self._check_hallucination:
            hall_signals = self._check_patterns(text, _HALLUCINATION_PATTERNS)
            if hall_signals:
                confidence = min(0.8, 0.3 + 0.2 * len(hall_signals))
                if confidence >= self._min_confidence:
                    scores.append((
                        ResponseCategory.HALLUCINATION, confidence,
                        hall_signals, ""
                    ))

        # Pick highest confidence category
        if not scores:
            return ClassifyResult(
                category=ResponseCategory.SAFE,
                confidence=0.8,
                signals=["No concerning patterns detected"],
            )

        # Filter by minimum confidence
        valid = [s for s in scores if s[1] >= self._min_confidence]
        if not valid:
            return ClassifyResult(
                category=ResponseCategory.SAFE,
                confidence=0.6,
                signals=["Weak signals below confidence threshold"],
            )

        # Return highest confidence match
        best = max(valid, key=lambda s: s[1])
        return ClassifyResult(
            category=best[0],
            confidence=round(best[1], 2),
            signals=best[2],
            subcategory=best[3],
        )

    def classify_batch(self, texts: list[str]) -> list[ClassifyResult]:
        """Classify multiple responses."""
        return [self.classify(text) for text in texts]

    def _check_patterns(
        self, text: str, patterns: list[re.Pattern]
    ) -> list[str]:
        """Check text against patterns, return matched signal descriptions."""
        signals = []
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                signals.append(match.group())
        return signals
