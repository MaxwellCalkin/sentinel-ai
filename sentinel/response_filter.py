"""Post-processing filters for LLM outputs.

Apply safety, quality, and formatting filters to LLM responses
before they reach users. Supports custom filters, blocklists,
regex replacements, PII stripping, and length truncation.

Usage:
    from sentinel.response_filter import ResponseFilter

    rf = ResponseFilter(max_length=500, strip_pii=True)
    rf.add_blocklist(["badword"])
    result = rf.apply("Some LLM output with badword")
    print(result.filtered)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class FilterResult:
    """Result of applying filters to text."""

    original: str
    filtered: str
    modifications: int
    filters_applied: list[str]
    truncated: bool
    blocked_words_found: int


@dataclass
class FilterStats:
    """Cumulative filtering statistics."""

    total_processed: int
    total_modifications: int
    total_blocked_words: int
    total_truncated: int
    avg_modifications: float


@dataclass
class _FilterEntry:
    """Internal representation of a named filter."""

    name: str
    filter_fn: Callable[[str], str]
    priority: int


_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
_PII_REPLACEMENT = "[PII_REDACTED]"


class ResponseFilter:
    """Apply post-processing filters to LLM outputs.

    Filters are applied in priority order (highest first).
    Built-in capabilities include PII stripping, word blocklists,
    regex replacements, and length truncation.
    """

    def __init__(
        self,
        max_length: int = 0,
        strip_pii: bool = False,
    ) -> None:
        self._max_length = max_length
        self._strip_pii = strip_pii
        self._filters: list[_FilterEntry] = []
        self._blocklist: list[str] = []
        self._replacements: list[tuple[re.Pattern[str], str]] = []
        self._total_processed = 0
        self._total_modifications = 0
        self._total_blocked_words = 0
        self._total_truncated = 0

    def add_filter(
        self,
        name: str,
        filter_fn: Callable[[str], str],
        priority: int = 0,
    ) -> None:
        """Add a named custom filter function.

        Filters are applied in descending priority order.
        """
        self._filters.append(_FilterEntry(
            name=name,
            filter_fn=filter_fn,
            priority=priority,
        ))

    def remove_filter(self, name: str) -> None:
        """Remove a named filter."""
        self._filters = [f for f in self._filters if f.name != name]

    def list_filters(self) -> list[str]:
        """List filter names in priority order (highest first)."""
        sorted_filters = sorted(
            self._filters, key=lambda f: f.priority, reverse=True
        )
        return [f.name for f in sorted_filters]

    def add_blocklist(self, words: list[str]) -> None:
        """Add words to the blocklist (case-insensitive replacement)."""
        self._blocklist.extend(words)

    def add_replacement(self, pattern: str, replacement: str) -> None:
        """Add a regex replacement rule."""
        self._replacements.append((re.compile(pattern), replacement))

    def apply(self, text: str) -> FilterResult:
        """Apply all filters to text and return the result."""
        original = text
        filters_applied: list[str] = []
        blocked_words_found = 0

        text = self._apply_custom_filters(text, filters_applied)
        text, blocked_words_found = self._apply_blocklist(
            text, filters_applied
        )
        text = self._apply_replacements(text, filters_applied)
        text = self._apply_pii_stripping(text, filters_applied)
        truncated = False
        text, truncated = self._apply_truncation(text, filters_applied)

        modifications = _count_modifications(original, text)

        self._update_stats(modifications, blocked_words_found, truncated)

        return FilterResult(
            original=original,
            filtered=text,
            modifications=modifications,
            filters_applied=filters_applied,
            truncated=truncated,
            blocked_words_found=blocked_words_found,
        )

    def apply_batch(self, texts: list[str]) -> list[FilterResult]:
        """Apply filters to multiple texts."""
        return [self.apply(text) for text in texts]

    def stats(self) -> FilterStats:
        """Get cumulative filtering statistics."""
        avg = (
            self._total_modifications / self._total_processed
            if self._total_processed > 0
            else 0.0
        )
        return FilterStats(
            total_processed=self._total_processed,
            total_modifications=self._total_modifications,
            total_blocked_words=self._total_blocked_words,
            total_truncated=self._total_truncated,
            avg_modifications=avg,
        )

    def clear(self) -> None:
        """Reset all filters and statistics."""
        self._filters.clear()
        self._blocklist.clear()
        self._replacements.clear()
        self._total_processed = 0
        self._total_modifications = 0
        self._total_blocked_words = 0
        self._total_truncated = 0

    def _apply_custom_filters(
        self, text: str, filters_applied: list[str]
    ) -> str:
        sorted_filters = sorted(
            self._filters, key=lambda f: f.priority, reverse=True
        )
        for entry in sorted_filters:
            new_text = entry.filter_fn(text)
            if new_text != text:
                filters_applied.append(entry.name)
            text = new_text
        return text

    def _apply_blocklist(
        self, text: str, filters_applied: list[str]
    ) -> tuple[str, int]:
        if not self._blocklist:
            return text, 0

        blocked_count = 0
        for word in self._blocklist:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            matches = pattern.findall(text)
            if matches:
                blocked_count += len(matches)
                text = pattern.sub("[BLOCKED]", text)

        if blocked_count > 0:
            filters_applied.append("blocklist")
        return text, blocked_count

    def _apply_replacements(
        self, text: str, filters_applied: list[str]
    ) -> str:
        for pattern, replacement in self._replacements:
            new_text = pattern.sub(replacement, text)
            if new_text != text:
                filters_applied.append("replacement")
            text = new_text
        return text

    def _apply_pii_stripping(
        self, text: str, filters_applied: list[str]
    ) -> str:
        if not self._strip_pii:
            return text

        new_text = _EMAIL_PATTERN.sub(_PII_REPLACEMENT, text)
        new_text = _PHONE_PATTERN.sub(_PII_REPLACEMENT, new_text)
        if new_text != text:
            filters_applied.append("pii_strip")
        return new_text

    def _apply_truncation(
        self, text: str, filters_applied: list[str]
    ) -> tuple[str, bool]:
        if self._max_length <= 0 or len(text) <= self._max_length:
            return text, False

        truncated_text = text[: self._max_length] + "..."
        filters_applied.append("truncation")
        return truncated_text, True

    def _update_stats(
        self,
        modifications: int,
        blocked_words_found: int,
        truncated: bool,
    ) -> None:
        self._total_processed += 1
        self._total_modifications += modifications
        self._total_blocked_words += blocked_words_found
        if truncated:
            self._total_truncated += 1


def _count_modifications(original: str, filtered: str) -> int:
    """Count character-level differences between original and filtered."""
    if original == filtered:
        return 0
    return abs(len(original) - len(filtered)) + sum(
        1
        for a, b in zip(original, filtered)
        if a != b
    )
