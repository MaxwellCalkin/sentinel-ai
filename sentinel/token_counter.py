"""Zero-dependency token estimation for LLM text.

Estimate token counts for text without requiring tiktoken or
other tokenizer libraries. Uses character/word heuristics
calibrated against common LLM tokenizers.

Usage:
    from sentinel.token_counter import TokenCounter

    counter = TokenCounter()
    estimate = counter.count("Hello, world!")
    print(estimate.tokens)  # ~4
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenEstimate:
    """Estimated token count."""
    tokens: int
    chars: int
    words: int
    method: str
    confidence: str  # "high", "medium", "low"


@dataclass
class TruncateResult:
    """Result of truncation to token limit."""
    text: str
    estimated_tokens: int
    truncated: bool
    chars_removed: int


# Heuristic: ~4 chars per token for English, ~1.5 for CJK
_CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0xAC00, 0xD7AF),    # Hangul
]

_WORD_RE = re.compile(r'\S+')
_SPECIAL_RE = re.compile(r'[^\w\s]')


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return any(start <= cp <= end for start, end in _CJK_RANGES)


class TokenCounter:
    """Estimate token counts using heuristics.

    Calibrated against common LLM tokenizers (cl100k_base).
    No external dependencies required.
    """

    def __init__(
        self,
        chars_per_token: float = 4.0,
        cjk_chars_per_token: float = 1.5,
    ) -> None:
        """
        Args:
            chars_per_token: Estimated chars per token for Latin text.
            cjk_chars_per_token: Estimated chars per token for CJK text.
        """
        self._cpt = chars_per_token
        self._cjk_cpt = cjk_chars_per_token

    def count(self, text: str) -> TokenEstimate:
        """Estimate token count for text.

        Args:
            text: Input text.

        Returns:
            TokenEstimate with token count and metadata.
        """
        if not text:
            return TokenEstimate(tokens=0, chars=0, words=0, method="heuristic", confidence="high")

        chars = len(text)
        words = len(_WORD_RE.findall(text))

        # Count CJK vs non-CJK characters
        cjk_chars = sum(1 for c in text if _is_cjk(c))
        latin_chars = chars - cjk_chars

        # Estimate tokens
        cjk_tokens = cjk_chars / self._cjk_cpt if cjk_chars > 0 else 0
        latin_tokens = latin_chars / self._cpt if latin_chars > 0 else 0

        # Special characters often become their own tokens
        special_count = len(_SPECIAL_RE.findall(text))
        special_adjustment = special_count * 0.3  # ~30% of specials are own tokens

        total = cjk_tokens + latin_tokens + special_adjustment
        tokens = max(1, round(total))

        # Confidence based on text characteristics
        if cjk_chars > chars * 0.5:
            confidence = "medium"  # CJK estimation is less precise
        elif chars < 10:
            confidence = "medium"
        else:
            confidence = "high"

        return TokenEstimate(
            tokens=tokens,
            chars=chars,
            words=words,
            method="heuristic",
            confidence=confidence,
        )

    def count_messages(
        self,
        messages: list[dict[str, str]],
        overhead_per_message: int = 4,
    ) -> TokenEstimate:
        """Estimate tokens for a chat message list.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            overhead_per_message: Token overhead per message (role, delimiters).

        Returns:
            Combined TokenEstimate.
        """
        total_tokens = 0
        total_chars = 0
        total_words = 0

        for msg in messages:
            content = msg.get("content", "")
            est = self.count(content)
            total_tokens += est.tokens + overhead_per_message
            total_chars += est.chars
            total_words += est.words

        # Base overhead (system delimiters)
        total_tokens += 3

        return TokenEstimate(
            tokens=total_tokens,
            chars=total_chars,
            words=total_words,
            method="heuristic",
            confidence="medium",
        )

    def fits_context(
        self,
        text: str,
        max_tokens: int,
        reserved: int = 0,
    ) -> bool:
        """Check if text fits within a token budget.

        Args:
            text: Input text.
            max_tokens: Context window size.
            reserved: Tokens reserved for response.
        """
        est = self.count(text)
        return est.tokens <= (max_tokens - reserved)

    def truncate(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "...",
    ) -> TruncateResult:
        """Truncate text to fit within token limit.

        Args:
            text: Input text.
            max_tokens: Maximum tokens.
            suffix: Suffix to add if truncated.

        Returns:
            TruncateResult with truncated text.
        """
        est = self.count(text)
        if est.tokens <= max_tokens:
            return TruncateResult(
                text=text,
                estimated_tokens=est.tokens,
                truncated=False,
                chars_removed=0,
            )

        # Binary search for the right character cutoff
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = text[:mid] + suffix
            if self.count(candidate).tokens <= max_tokens:
                lo = mid
            else:
                hi = mid - 1

        truncated_text = text[:lo] + suffix if lo < len(text) else text
        final_est = self.count(truncated_text)

        return TruncateResult(
            text=truncated_text,
            estimated_tokens=final_est.tokens,
            truncated=True,
            chars_removed=len(text) - lo,
        )

    def estimate_cost(
        self,
        text: str,
        price_per_1k: float = 0.003,
    ) -> float:
        """Estimate cost for processing text.

        Args:
            text: Input text.
            price_per_1k: Price per 1000 tokens.

        Returns:
            Estimated cost in dollars.
        """
        est = self.count(text)
        return round(est.tokens * price_per_1k / 1000, 8)
