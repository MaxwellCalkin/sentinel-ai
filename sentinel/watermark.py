"""AI text watermark detection.

Detect statistical watermarks in AI-generated text using
token distribution analysis. Useful for verifying provenance
of LLM outputs and detecting AI-generated content.

Usage:
    from sentinel.watermark import WatermarkDetector

    detector = WatermarkDetector()
    detector.add_signature("model_v1", seed_words=["the", "a"], bias=0.7)
    result = detector.check("Some generated text...")
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WatermarkResult:
    """Result of watermark detection."""
    text: str
    detected: bool
    confidence: float        # 0.0 to 1.0
    signatures_matched: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextStats:
    """Statistical properties of text."""
    word_count: int = 0
    unique_ratio: float = 0.0     # Unique words / total words
    avg_word_length: float = 0.0
    entropy: float = 0.0          # Shannon entropy of word distribution
    repetition_score: float = 0.0  # How repetitive the text is
    burstiness: float = 0.0       # Variance in sentence length


_WORD_RE = re.compile(r"[a-z]+")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


def _compute_entropy(tokens: list[str]) -> float:
    """Shannon entropy of token distribution."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_burstiness(text: str) -> float:
    """Variance in sentence lengths (burstiness)."""
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return math.sqrt(variance) / (mean + 1e-6)


class WatermarkDetector:
    """Detect watermarks in AI-generated text.

    Uses statistical analysis of token distributions to detect
    potential watermarking patterns. Also provides general text
    statistics useful for AI-vs-human detection.
    """

    def __init__(
        self,
        min_words: int = 50,
    ) -> None:
        """
        Args:
            min_words: Minimum words for reliable detection.
        """
        self._min_words = min_words
        self._signatures: dict[str, dict[str, Any]] = {}

    def add_signature(
        self,
        name: str,
        seed_words: list[str] | None = None,
        bias: float = 0.7,
        expected_entropy_range: tuple[float, float] = (3.5, 5.5),
    ) -> None:
        """Register a watermark signature to detect.

        Args:
            name: Signature identifier.
            seed_words: Words expected to be biased in watermarked text.
            bias: Expected frequency bias for seed words.
            expected_entropy_range: Expected entropy range for this watermark.
        """
        self._signatures[name] = {
            "seed_words": set(w.lower() for w in (seed_words or [])),
            "bias": bias,
            "entropy_range": expected_entropy_range,
        }

    def analyze(self, text: str) -> TextStats:
        """Compute statistical properties of text."""
        words = _WORD_RE.findall(text.lower())
        if not words:
            return TextStats()

        unique = set(words)
        entropy = _compute_entropy(words)
        burstiness = _compute_burstiness(text)

        # Repetition: ratio of bigrams that repeat
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        repeated = sum(1 for c in bigram_counts.values() if c > 1)
        repetition = repeated / len(bigram_counts) if bigram_counts else 0

        return TextStats(
            word_count=len(words),
            unique_ratio=len(unique) / len(words),
            avg_word_length=sum(len(w) for w in words) / len(words),
            entropy=round(entropy, 4),
            repetition_score=round(repetition, 4),
            burstiness=round(burstiness, 4),
        )

    def check(self, text: str) -> WatermarkResult:
        """Check text for watermark signatures.

        Returns:
            WatermarkResult with detection status and confidence.
        """
        words = _WORD_RE.findall(text.lower())
        stats = self.analyze(text)

        if len(words) < self._min_words:
            return WatermarkResult(
                text=text,
                detected=False,
                confidence=0.0,
                stats={"warning": "insufficient text", "word_count": len(words)},
            )

        matched_sigs: list[str] = []
        max_confidence = 0.0
        word_counts = Counter(words)
        total = len(words)

        for name, sig in self._signatures.items():
            confidence = 0.0
            seed_words = sig["seed_words"]
            bias = sig["bias"]
            ent_lo, ent_hi = sig["entropy_range"]

            # Check seed word frequency bias
            if seed_words:
                seed_freq = sum(word_counts.get(w, 0) for w in seed_words) / total
                expected_freq = len(seed_words) / 1000  # Rough baseline
                if seed_freq > expected_freq * bias:
                    confidence += 0.4

            # Check entropy range
            if ent_lo <= stats.entropy <= ent_hi:
                confidence += 0.2

            # Low burstiness (AI text is typically more uniform)
            if stats.burstiness < 0.5:
                confidence += 0.1

            # Moderate unique ratio (not too high, not too low)
            if 0.3 <= stats.unique_ratio <= 0.7:
                confidence += 0.1

            confidence = min(1.0, confidence)
            if confidence > 0.3:
                matched_sigs.append(name)
                max_confidence = max(max_confidence, confidence)

        return WatermarkResult(
            text=text,
            detected=len(matched_sigs) > 0,
            confidence=round(max_confidence, 2),
            signatures_matched=matched_sigs,
            stats={
                "word_count": stats.word_count,
                "entropy": stats.entropy,
                "unique_ratio": stats.unique_ratio,
                "burstiness": stats.burstiness,
                "repetition": stats.repetition_score,
            },
        )
