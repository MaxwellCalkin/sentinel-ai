"""LLM model fingerprinting via text analysis.

Detect which LLM likely generated a piece of text based on
stylistic patterns, vocabulary choices, and structural markers.
Useful for attribution, compliance, and fraud detection.

Usage:
    from sentinel.model_fingerprint import ModelFingerprint

    fp = ModelFingerprint()
    result = fp.analyze("I'd be happy to help! Let me break this down...")
    print(result.likely_model)  # "claude"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FingerprintResult:
    """Result of model fingerprinting."""
    likely_model: str
    confidence: float
    scores: dict[str, float]  # model -> score
    markers_found: list[str]
    is_ai_generated: bool


# Model-specific linguistic markers
_MODEL_MARKERS: dict[str, list[tuple[str, float]]] = {
    "claude": [
        (r"\bi'd be happy to\b", 0.3),
        (r"\blet me\b.*\bbreak\b.*\bdown\b", 0.2),
        (r"\bthat's a (?:great|excellent|thoughtful) question\b", 0.2),
        (r"\bi should note\b", 0.15),
        (r"\bi want to be (?:upfront|transparent|honest)\b", 0.2),
        (r"\bhowever,? i (?:should|must|need to) (?:note|mention|point out)\b", 0.2),
        (r"\bi don't have (?:access to|the ability)\b", 0.1),
        (r"\bhere's (?:a|my|the) (?:breakdown|summary|analysis)\b", 0.15),
        (r"\bi appreciate you\b", 0.15),
        (r"\blet me know if you'd like\b", 0.15),
    ],
    "gpt": [
        (r"\bcertainly[!.]", 0.2),
        (r"\babsolutely[!.]", 0.15),
        (r"\bgreat question[!.]", 0.15),
        (r"\bas an ai\b", 0.3),
        (r"\bi'm just an ai\b", 0.3),
        (r"\bmy training data\b", 0.2),
        (r"\bmy knowledge cutoff\b", 0.3),
        (r"\bdelve\b", 0.15),
        (r"\btapestry\b", 0.1),
        (r"\blandscape\b.*\bnavigate\b", 0.1),
    ],
    "gemini": [
        (r"\bas a large language model\b", 0.3),
        (r"\bi'm a large language model\b", 0.3),
        (r"\btrained by google\b", 0.3),
        (r"\bhere are some\b.*\bpoints\b", 0.1),
        (r"\blet me provide\b", 0.1),
    ],
    "llama": [
        (r"\bi'm meta ai\b", 0.3),
        (r"\bi cannot provide\b.*\billegal\b", 0.15),
        (r"\bnote:\b", 0.05),
    ],
}

# Generic AI markers (model-agnostic)
_AI_MARKERS = [
    (r"\bas an ai\b", 0.3),
    (r"\blanguage model\b", 0.3),
    (r"\bi don't have (?:feelings|emotions|consciousness)\b", 0.2),
    (r"\bi was trained\b", 0.2),
    (r"\bhere (?:are|is) (?:a|the) (?:summary|breakdown|list|overview)\b", 0.1),
    (r"\bhope (?:this|that) helps\b", 0.1),
    (r"(?:^|\n)\d+\.\s+\*\*", 0.1),  # Numbered bold lists
    (r"(?:^|\n)[-*]\s+\*\*", 0.1),   # Bulleted bold lists
]


class ModelFingerprint:
    """Detect which LLM generated text.

    Uses linguistic markers, structural patterns, and
    vocabulary analysis for model attribution.
    """

    def __init__(
        self,
        custom_markers: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        """
        Args:
            custom_markers: Additional model markers {model: [(pattern, weight)]}.
        """
        self._markers = dict(_MODEL_MARKERS)
        if custom_markers:
            for model, patterns in custom_markers.items():
                if model in self._markers:
                    self._markers[model].extend(patterns)
                else:
                    self._markers[model] = patterns

    def analyze(self, text: str) -> FingerprintResult:
        """Analyze text to determine likely source model.

        Args:
            text: Text to analyze.

        Returns:
            FingerprintResult with model attribution.
        """
        text_lower = text.lower()
        scores: dict[str, float] = {}
        markers_found: list[str] = []

        # Score against each model
        for model, patterns in self._markers.items():
            score = 0.0
            for pattern, weight in patterns:
                if re.search(pattern, text_lower):
                    score += weight
                    markers_found.append(f"{model}:{pattern}")
            scores[model] = round(min(score, 1.0), 4)

        # Check AI generation markers
        ai_score = 0.0
        for pattern, weight in _AI_MARKERS:
            if re.search(pattern, text_lower):
                ai_score += weight

        is_ai = ai_score >= 0.2 or max(scores.values(), default=0) >= 0.2

        # Determine most likely model
        if scores:
            best_model = max(scores, key=scores.get)
            best_score = scores[best_model]
        else:
            best_model = "unknown"
            best_score = 0.0

        if best_score < 0.1:
            best_model = "unknown"

        return FingerprintResult(
            likely_model=best_model,
            confidence=best_score,
            scores=scores,
            markers_found=markers_found,
            is_ai_generated=is_ai,
        )

    def analyze_batch(self, texts: list[str]) -> list[FingerprintResult]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]

    @property
    def supported_models(self) -> list[str]:
        return list(self._markers.keys())
