"""Prompt leak detection — flags LLM outputs that expose system prompt content.

When an attacker tricks a model into revealing its system prompt, that's a
security incident. This module detects when a model's output contains
significant fragments of the system prompt, enabling real-time alerting
and blocking.

Usage:
    from sentinel.prompt_leak import PromptLeakDetector

    detector = PromptLeakDetector(system_prompt="You are a helpful assistant...")
    result = detector.check("The system prompt says: You are a helpful assistant")
    if result.leaked:
        print(f"Leak detected! Overlap: {result.overlap_ratio:.0%}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class LeakResult:
    """Result of a prompt leak check."""
    leaked: bool
    overlap_ratio: float
    matched_fragments: list[str]
    risk_score: float  # 0.0–1.0

    @property
    def summary(self) -> str:
        if not self.leaked:
            return "No prompt leak detected"
        n = len(self.matched_fragments)
        return (
            f"Prompt leak: {n} fragment(s) matched, "
            f"{self.overlap_ratio:.0%} overlap, "
            f"risk={self.risk_score:.2f}"
        )


class PromptLeakDetector:
    """Detect when model output contains system prompt fragments.

    Uses n-gram matching to find overlapping text between the system prompt
    and the model output. Short common phrases are filtered out to reduce
    false positives.
    """

    # Common phrases that appear in many outputs but aren't leaks
    _COMMON_PHRASES = frozenset([
        "i am", "you are", "i can", "you can", "please", "thank you",
        "i will", "you will", "i would", "how can i help",
        "is there anything", "let me know", "i'm happy to",
        "feel free", "here is", "here are", "for example",
        "in order to", "as well as", "such as", "based on",
        "make sure", "keep in mind", "note that", "however",
        "therefore", "furthermore", "additionally", "also",
    ])

    def __init__(
        self,
        system_prompt: str,
        min_ngram: int = 4,
        max_ngram: int = 20,
        leak_threshold: float = 0.15,
    ):
        """
        Args:
            system_prompt: The system prompt to protect.
            min_ngram: Minimum n-gram size (in words) to consider a match.
            max_ngram: Maximum n-gram size for matching.
            leak_threshold: Overlap ratio above which a leak is flagged.
        """
        self._system_prompt = system_prompt
        self._min_ngram = min_ngram
        self._max_ngram = max_ngram
        self._leak_threshold = leak_threshold
        self._prompt_words = self._tokenize(system_prompt)
        self._prompt_ngrams = self._build_ngrams(self._prompt_words)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize and split text into lowercase word tokens."""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return text.split()

    def _build_ngrams(self, words: list[str]) -> set[tuple[str, ...]]:
        """Build a set of n-grams from a word list."""
        ngrams: set[tuple[str, ...]] = set()
        for n in range(self._min_ngram, min(self._max_ngram + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                gram = tuple(words[i : i + n])
                # Filter common phrases
                text = " ".join(gram)
                if text not in self._COMMON_PHRASES:
                    ngrams.add(gram)
        return ngrams

    def check(self, output: str) -> LeakResult:
        """Check if the output contains system prompt fragments."""
        if not self._system_prompt or not output:
            return LeakResult(
                leaked=False, overlap_ratio=0.0, matched_fragments=[], risk_score=0.0
            )

        output_words = self._tokenize(output)
        if len(output_words) < self._min_ngram:
            return LeakResult(
                leaked=False, overlap_ratio=0.0, matched_fragments=[], risk_score=0.0
            )

        output_ngrams = self._build_ngrams(output_words)

        # Find matching n-grams
        matches = self._prompt_ngrams & output_ngrams

        if not matches:
            return LeakResult(
                leaked=False, overlap_ratio=0.0, matched_fragments=[], risk_score=0.0
            )

        # Calculate overlap based on matched word positions in prompt
        matched_prompt_positions: set[int] = set()
        for gram in matches:
            gram_len = len(gram)
            for i in range(len(self._prompt_words) - gram_len + 1):
                if tuple(self._prompt_words[i : i + gram_len]) == gram:
                    matched_prompt_positions.update(range(i, i + gram_len))

        overlap_ratio = (
            len(matched_prompt_positions) / len(self._prompt_words)
            if self._prompt_words
            else 0.0
        )

        # Reconstruct matched fragments (merge overlapping)
        fragments = self._merge_fragments(matches)

        # Risk score considers both overlap and longest match
        longest_match = max(len(g) for g in matches) if matches else 0
        length_factor = min(longest_match / max(len(self._prompt_words), 1), 1.0)
        risk_score = min((overlap_ratio * 0.6 + length_factor * 0.4) * 2, 1.0)

        leaked = overlap_ratio >= self._leak_threshold

        return LeakResult(
            leaked=leaked,
            overlap_ratio=round(overlap_ratio, 4),
            matched_fragments=fragments,
            risk_score=round(risk_score, 4),
        )

    @staticmethod
    def _merge_fragments(ngrams: set[tuple[str, ...]]) -> list[str]:
        """Merge overlapping n-grams into readable fragments."""
        if not ngrams:
            return []

        # Sort by length descending to prefer longer matches
        sorted_grams = sorted(ngrams, key=len, reverse=True)

        # Keep only non-substring matches
        unique: list[tuple[str, ...]] = []
        for gram in sorted_grams:
            text = " ".join(gram)
            if not any(" ".join(u) != text and text in " ".join(u) for u in unique):
                unique.append(gram)

        return [" ".join(g) for g in unique[:10]]  # Cap at 10 fragments
