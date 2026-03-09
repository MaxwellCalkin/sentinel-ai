"""Output sanitization for LLM responses.

Clean and sanitize LLM outputs to remove unwanted artifacts,
strip system prompt leakage, normalize whitespace, and enforce
safe output formatting.

Usage:
    from sentinel.response_sanitizer import ResponseSanitizer

    sanitizer = ResponseSanitizer()
    sanitizer.strip_system_leaks()
    sanitizer.normalize_whitespace()
    clean = sanitizer.sanitize("Response with  extra   spaces")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SanitizeStep:
    """A single sanitization step."""
    name: str
    transform: Callable[[str], str]
    enabled: bool = True


@dataclass
class SanitizeOutput:
    """Result of sanitization."""
    original: str
    sanitized: str
    steps_applied: list[str]
    chars_removed: int
    modified: bool


class ResponseSanitizer:
    """Sanitize LLM output text.

    Chain sanitization steps to clean responses:
    whitespace normalization, system prompt leak stripping,
    HTML/script removal, encoding fixes, etc.
    """

    def __init__(self) -> None:
        self._steps: list[SanitizeStep] = []

    def add_step(
        self,
        name: str,
        transform: Callable[[str], str],
    ) -> "ResponseSanitizer":
        """Add a custom sanitization step."""
        self._steps.append(SanitizeStep(name=name, transform=transform))
        return self

    def normalize_whitespace(self) -> "ResponseSanitizer":
        """Collapse multiple spaces/newlines."""
        def _normalize(text: str) -> str:
            text = re.sub(r' {2,}', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()
        self._steps.append(SanitizeStep(name="normalize_whitespace", transform=_normalize))
        return self

    def strip_system_leaks(self) -> "ResponseSanitizer":
        """Remove common system prompt leakage patterns."""
        patterns = [
            r'(?i)(?:system\s*prompt|system\s*message)\s*[:=]\s*.*?(?:\n|$)',
            r'(?i)(?:you\s+are\s+a|i\s+am\s+a)\s+(?:helpful|friendly|safe)\s+(?:ai|assistant|chatbot).*?(?:\.\s|\n|$)',
            r'(?i)my\s+(?:instructions|guidelines|rules)\s+(?:are|say|tell).*?(?:\.\s|\n|$)',
        ]
        compiled = [re.compile(p) for p in patterns]

        def _strip(text: str) -> str:
            for pattern in compiled:
                text = pattern.sub('', text)
            return text.strip()

        self._steps.append(SanitizeStep(name="strip_system_leaks", transform=_strip))
        return self

    def strip_html(self) -> "ResponseSanitizer":
        """Remove HTML tags and script content."""
        def _strip_html(text: str) -> str:
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', '', text)
            return text

        self._steps.append(SanitizeStep(name="strip_html", transform=_strip_html))
        return self

    def strip_urls(self) -> "ResponseSanitizer":
        """Remove URLs from text."""
        def _strip(text: str) -> str:
            return re.sub(r'https?://\S+', '[URL removed]', text)

        self._steps.append(SanitizeStep(name="strip_urls", transform=_strip))
        return self

    def strip_emails(self) -> "ResponseSanitizer":
        """Remove email addresses."""
        def _strip(text: str) -> str:
            return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[email removed]', text)

        self._steps.append(SanitizeStep(name="strip_emails", transform=_strip))
        return self

    def limit_length(self, max_chars: int) -> "ResponseSanitizer":
        """Truncate to max character length."""
        def _limit(text: str) -> str:
            if len(text) > max_chars:
                return text[:max_chars] + "..."
            return text

        self._steps.append(SanitizeStep(name="limit_length", transform=_limit))
        return self

    def replace_pattern(
        self,
        name: str,
        pattern: str,
        replacement: str = "",
    ) -> "ResponseSanitizer":
        """Add a regex replacement step."""
        compiled = re.compile(pattern, re.IGNORECASE)

        def _replace(text: str) -> str:
            return compiled.sub(replacement, text)

        self._steps.append(SanitizeStep(name=name, transform=_replace))
        return self

    def disable_step(self, name: str) -> bool:
        for step in self._steps:
            if step.name == name:
                step.enabled = False
                return True
        return False

    def enable_step(self, name: str) -> bool:
        for step in self._steps:
            if step.name == name:
                step.enabled = True
                return True
        return False

    def sanitize(self, text: str) -> SanitizeOutput:
        """Apply all sanitization steps.

        Args:
            text: Raw LLM output.

        Returns:
            SanitizeOutput with cleaned text and applied steps.
        """
        original = text
        applied: list[str] = []

        for step in self._steps:
            if not step.enabled:
                continue
            try:
                new_text = step.transform(text)
                if new_text != text:
                    applied.append(step.name)
                text = new_text
            except Exception:
                continue

        return SanitizeOutput(
            original=original,
            sanitized=text,
            steps_applied=applied,
            chars_removed=len(original) - len(text),
            modified=text != original,
        )

    def sanitize_batch(self, texts: list[str]) -> list[SanitizeOutput]:
        """Sanitize multiple texts."""
        return [self.sanitize(t) for t in texts]

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def active_step_count(self) -> int:
        return sum(1 for s in self._steps if s.enabled)
