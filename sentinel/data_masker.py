"""Data masking for sensitive information in LLM context.

Mask PII, credentials, and other sensitive data before
sending to LLMs, with reversible masking to restore
original values after processing.

Usage:
    from sentinel.data_masker import DataMasker

    masker = DataMasker()
    masked, mapping = masker.mask("Call John at 555-1234")
    # masked: "Call [NAME_1] at [PHONE_1]"
    restored = masker.unmask(masked, mapping)
    # restored: "Call John at 555-1234"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MaskResult:
    """Result of masking operation."""
    original: str
    masked: str
    mapping: dict[str, str]  # placeholder -> original value
    items_masked: int
    categories: list[str]


# Built-in masking patterns
_PATTERNS: list[tuple[str, str, str]] = [
    # (pattern, category, replacement_prefix)
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "EMAIL", "EMAIL"),
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "PHONE", "PHONE"),
    (r'\b\d{3}-\d{2}-\d{4}\b', "SSN", "SSN"),
    (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', "CREDIT_CARD", "CC"),
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "IP_ADDRESS", "IP"),
    (r'(?i)\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|xoxb-[a-zA-Z0-9-]+)\b', "API_KEY", "KEY"),
]

# Common first names for name detection
_COMMON_NAMES = frozenset([
    "james", "john", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "charles", "mary", "patricia", "jennifer", "linda",
    "elizabeth", "barbara", "susan", "jessica", "sarah", "karen",
    "alice", "bob", "charlie", "diana", "emma", "frank", "george",
    "hannah", "ivan", "julia", "kevin", "laura", "max", "nancy",
    "oliver", "peter", "rachel", "steve", "tina", "victor",
])


class DataMasker:
    """Mask sensitive data with reversible placeholders.

    Replaces PII and sensitive data with placeholders like
    [EMAIL_1], [PHONE_1], etc. Supports unmasking to restore
    original values.
    """

    def __init__(
        self,
        mask_emails: bool = True,
        mask_phones: bool = True,
        mask_ssns: bool = True,
        mask_credit_cards: bool = True,
        mask_ips: bool = True,
        mask_api_keys: bool = True,
        mask_names: bool = False,
        custom_patterns: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Args:
            mask_emails: Mask email addresses.
            mask_phones: Mask phone numbers.
            mask_ssns: Mask SSNs.
            mask_credit_cards: Mask credit card numbers.
            mask_ips: Mask IP addresses.
            mask_api_keys: Mask API keys.
            mask_names: Mask common first names.
            custom_patterns: Additional (pattern, category) pairs.
        """
        self._active_patterns: list[tuple[str, str, str]] = []
        category_flags = {
            "EMAIL": mask_emails,
            "PHONE": mask_phones,
            "SSN": mask_ssns,
            "CREDIT_CARD": mask_credit_cards,
            "IP_ADDRESS": mask_ips,
            "API_KEY": mask_api_keys,
        }

        for pattern, category, prefix in _PATTERNS:
            if category_flags.get(category, True):
                self._active_patterns.append((pattern, category, prefix))

        self._mask_names = mask_names
        self._custom_patterns = custom_patterns or []

    def mask(self, text: str) -> MaskResult:
        """Mask sensitive data in text.

        Args:
            text: Input text with potential sensitive data.

        Returns:
            MaskResult with masked text and reverse mapping.
        """
        mapping: dict[str, str] = {}
        masked = text
        counters: dict[str, int] = {}
        categories: list[str] = []

        # Apply built-in patterns (order matters: SSN before phone)
        # Sort by pattern specificity (longer patterns first)
        sorted_patterns = sorted(self._active_patterns, key=lambda p: len(p[0]), reverse=True)

        for pattern, category, prefix in sorted_patterns:
            for match in re.finditer(pattern, masked):
                value = match.group()
                # Check if already masked
                if value.startswith("[") and value.endswith("]"):
                    continue
                counter = counters.get(prefix, 0) + 1
                counters[prefix] = counter
                placeholder = f"[{prefix}_{counter}]"
                mapping[placeholder] = value
                masked = masked.replace(value, placeholder, 1)
                if category not in categories:
                    categories.append(category)

        # Name masking
        if self._mask_names:
            words = masked.split()
            for i, word in enumerate(words):
                clean = word.strip(".,!?;:\"'()[]")
                if clean.lower() in _COMMON_NAMES and not clean.startswith("["):
                    counter = counters.get("NAME", 0) + 1
                    counters["NAME"] = counter
                    placeholder = f"[NAME_{counter}]"
                    mapping[placeholder] = clean
                    words[i] = word.replace(clean, placeholder)
                    if "NAME" not in categories:
                        categories.append("NAME")
            masked = " ".join(words)

        # Custom patterns
        for pattern, category in self._custom_patterns:
            for match in re.finditer(pattern, masked):
                value = match.group()
                if value.startswith("[") and value.endswith("]"):
                    continue
                counter = counters.get(category, 0) + 1
                counters[category] = counter
                placeholder = f"[{category}_{counter}]"
                mapping[placeholder] = value
                masked = masked.replace(value, placeholder, 1)
                if category not in categories:
                    categories.append(category)

        return MaskResult(
            original=text,
            masked=masked,
            mapping=mapping,
            items_masked=len(mapping),
            categories=categories,
        )

    def unmask(self, text: str, mapping: dict[str, str]) -> str:
        """Restore original values from masked text.

        Args:
            text: Masked text.
            mapping: Placeholder -> original value mapping.

        Returns:
            Text with original values restored.
        """
        result = text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result

    def mask_batch(self, texts: list[str]) -> list[MaskResult]:
        """Mask multiple texts."""
        return [self.mask(t) for t in texts]
