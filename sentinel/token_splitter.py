"""Token-aware text splitter for LLM context windows.

Split long documents into chunks that fit within token limits
while preserving paragraph, sentence, and word boundaries.

Usage:
    from sentinel.token_splitter import TokenSplitter

    splitter = TokenSplitter(max_tokens=1000)
    chunks = splitter.split("Very long document text...")
    for chunk in chunks:
        print(len(chunk.text), chunk.token_estimate)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextChunk:
    """A chunk of split text."""
    text: str
    token_estimate: int
    index: int
    total_chunks: int
    overlap_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenSplitter:
    """Split text into token-limited chunks."""

    def __init__(
        self,
        max_tokens: int = 1000,
        overlap_tokens: int = 0,
        chars_per_token: float = 4.0,
        separator: str = "paragraph",  # "paragraph", "sentence", "word", "char"
    ) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._cpt = chars_per_token
        self._separator = separator

    def split(self, text: str) -> list[TextChunk]:
        """Split text into chunks."""
        if not text.strip():
            return []

        max_chars = int(self._max_tokens * self._cpt)
        overlap_chars = int(self._overlap_tokens * self._cpt)

        # Get segments based on separator
        segments = self._segment(text)

        # Build chunks from segments
        chunks_text: list[str] = []
        current = ""

        for seg in segments:
            if len(seg) > max_chars:
                # Segment too large — flush current and split segment
                if current.strip():
                    chunks_text.append(current.strip())
                    current = ""
                # Hard-split the oversized segment
                for i in range(0, len(seg), max_chars):
                    chunks_text.append(seg[i:i + max_chars].strip())
            elif len(current) + len(seg) > max_chars:
                if current.strip():
                    chunks_text.append(current.strip())
                current = seg
            else:
                current += seg

        if current.strip():
            chunks_text.append(current.strip())

        # Apply overlap
        if overlap_chars > 0 and len(chunks_text) > 1:
            overlapped = [chunks_text[0]]
            for i in range(1, len(chunks_text)):
                prev = chunks_text[i - 1]
                overlap = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
                overlapped.append(overlap + " " + chunks_text[i])
            chunks_text = overlapped

        total = len(chunks_text)
        result = []
        for i, chunk_text in enumerate(chunks_text):
            token_est = max(1, int(len(chunk_text) / self._cpt))
            result.append(TextChunk(
                text=chunk_text,
                token_estimate=token_est,
                index=i,
                total_chunks=total,
                overlap_tokens=self._overlap_tokens if i > 0 else 0,
            ))

        return result

    def count_chunks(self, text: str) -> int:
        """Estimate number of chunks without splitting."""
        if not text.strip():
            return 0
        total_tokens = len(text) / self._cpt
        if total_tokens <= self._max_tokens:
            return 1
        effective = self._max_tokens - self._overlap_tokens
        if effective <= 0:
            effective = self._max_tokens
        return max(1, int((total_tokens + effective - 1) / effective))

    def _segment(self, text: str) -> list[str]:
        """Break text into segments based on separator type."""
        if self._separator == "paragraph":
            parts = re.split(r'(\n\s*\n)', text)
            return parts if parts else [text]
        elif self._separator == "sentence":
            parts = re.split(r'(?<=[.!?])\s+', text)
            return [p + " " for p in parts]
        elif self._separator == "word":
            parts = text.split(" ")
            return [p + " " for p in parts]
        else:  # char
            return [text]
