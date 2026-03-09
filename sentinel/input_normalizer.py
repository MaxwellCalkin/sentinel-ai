"""Input normalization to defeat obfuscation attacks.

Canonicalizes text by converting homoglyphs, stripping invisible characters,
normalizing Unicode, and standardizing whitespace before safety scanning.

Usage:
    from sentinel.input_normalizer import InputNormalizer

    normalizer = InputNormalizer()
    result = normalizer.normalize("H\u0435llo w\u043erld")
    print(result.normalized)   # "Hello world"
    print(result.is_modified)  # True
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class NormalizationStep:
    """Record of a single normalization step applied to text."""
    name: str
    applied: bool
    chars_changed: int
    description: str = ""


@dataclass
class NormalizationResult:
    """Result of normalizing input text."""
    original: str
    normalized: str
    steps: list[NormalizationStep]
    total_changed: int
    is_modified: bool


@dataclass
class NormalizerConfig:
    """Configuration for which normalization steps to apply."""
    unicode_nfkc: bool = True
    strip_invisible: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False
    strip_accents: bool = False
    homoglyph_map: bool = True


@dataclass
class NormalizerStats:
    """Cumulative statistics across all normalize calls."""
    total_normalized: int = 0
    total_chars_changed: int = 0
    modified_count: int = 0


_INVISIBLE_CHARS = frozenset("\u200b\u200c\u200d\ufeff\u00ad")

_HOMOGLYPH_MAP: dict[str, str] = {
    "\u0430": "a",  # Cyrillic а → Latin a
    "\u0435": "e",  # Cyrillic е → Latin e
    "\u043e": "o",  # Cyrillic о → Latin o
    "\u0440": "p",  # Cyrillic р → Latin p
    "\u0441": "c",  # Cyrillic с → Latin c
    "\u0443": "y",  # Cyrillic у → Latin y
    "\u0445": "x",  # Cyrillic х → Latin x
    "\u0456": "i",  # Cyrillic і → Latin i
    "\u0410": "A",  # Cyrillic А → Latin A
    "\u0412": "B",  # Cyrillic В → Latin B
    "\u0415": "E",  # Cyrillic Е → Latin E
    "\u041a": "K",  # Cyrillic К → Latin K
    "\u041c": "M",  # Cyrillic М → Latin M
    "\u041d": "H",  # Cyrillic Н → Latin H
    "\u041e": "O",  # Cyrillic О → Latin O
    "\u0420": "P",  # Cyrillic Р → Latin P
    "\u0421": "C",  # Cyrillic С → Latin C
    "\u0422": "T",  # Cyrillic Т → Latin T
    "\u0425": "X",  # Cyrillic Х → Latin X
}

_WHITESPACE_PATTERN = re.compile(r"[ \t\n\r]+")


class InputNormalizer:
    """Normalize input text to defeat obfuscation attacks.

    Applies a configurable pipeline of canonicalization steps
    including Unicode NFKC, invisible char removal, homoglyph
    mapping, whitespace normalization, accent stripping, and
    lowercasing.
    """

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self._config = config or NormalizerConfig()
        self._stats = NormalizerStats()

    def normalize(self, text: str) -> NormalizationResult:
        """Apply all enabled normalization steps in order."""
        current = text
        steps: list[NormalizationStep] = []

        current = self._apply_unicode_nfkc(current, steps)
        current = self._apply_strip_invisible(current, steps)
        current = self._apply_homoglyph_map(current, steps)
        current = self._apply_normalize_whitespace(current, steps)
        current = self._apply_strip_accents(current, steps)
        current = self._apply_lowercase(current, steps)

        total_changed = sum(step.chars_changed for step in steps)
        is_modified = current != text

        self._stats.total_normalized += 1
        self._stats.total_chars_changed += total_changed
        if is_modified:
            self._stats.modified_count += 1

        return NormalizationResult(
            original=text,
            normalized=current,
            steps=steps,
            total_changed=total_changed,
            is_modified=is_modified,
        )

    def normalize_batch(self, texts: list[str]) -> list[NormalizationResult]:
        """Normalize multiple texts."""
        return [self.normalize(text) for text in texts]

    def stats(self) -> NormalizerStats:
        """Return cumulative normalization statistics."""
        return NormalizerStats(
            total_normalized=self._stats.total_normalized,
            total_chars_changed=self._stats.total_chars_changed,
            modified_count=self._stats.modified_count,
        )

    def _apply_unicode_nfkc(
        self, text: str, steps: list[NormalizationStep]
    ) -> str:
        if not self._config.unicode_nfkc:
            steps.append(NormalizationStep(
                name="unicode_nfkc", applied=False, chars_changed=0,
                description="Skipped (disabled)",
            ))
            return text

        normalized = unicodedata.normalize("NFKC", text)
        chars_changed = _count_char_differences(text, normalized)
        steps.append(NormalizationStep(
            name="unicode_nfkc", applied=True, chars_changed=chars_changed,
            description="Applied NFKC normalization",
        ))
        return normalized

    def _apply_strip_invisible(
        self, text: str, steps: list[NormalizationStep]
    ) -> str:
        if not self._config.strip_invisible:
            steps.append(NormalizationStep(
                name="strip_invisible", applied=False, chars_changed=0,
                description="Skipped (disabled)",
            ))
            return text

        cleaned = _remove_invisible_chars(text)
        chars_changed = len(text) - len(cleaned)
        steps.append(NormalizationStep(
            name="strip_invisible", applied=True, chars_changed=chars_changed,
            description="Stripped invisible characters",
        ))
        return cleaned

    def _apply_homoglyph_map(
        self, text: str, steps: list[NormalizationStep]
    ) -> str:
        if not self._config.homoglyph_map:
            steps.append(NormalizationStep(
                name="homoglyph_map", applied=False, chars_changed=0,
                description="Skipped (disabled)",
            ))
            return text

        mapped = _replace_homoglyphs(text)
        chars_changed = _count_char_differences(text, mapped)
        steps.append(NormalizationStep(
            name="homoglyph_map", applied=True, chars_changed=chars_changed,
            description="Mapped homoglyphs to Latin equivalents",
        ))
        return mapped

    def _apply_normalize_whitespace(
        self, text: str, steps: list[NormalizationStep]
    ) -> str:
        if not self._config.normalize_whitespace:
            steps.append(NormalizationStep(
                name="normalize_whitespace", applied=False, chars_changed=0,
                description="Skipped (disabled)",
            ))
            return text

        collapsed = _collapse_whitespace(text)
        chars_changed = abs(len(text) - len(collapsed))
        steps.append(NormalizationStep(
            name="normalize_whitespace", applied=True,
            chars_changed=chars_changed,
            description="Collapsed whitespace to single spaces",
        ))
        return collapsed

    def _apply_strip_accents(
        self, text: str, steps: list[NormalizationStep]
    ) -> str:
        if not self._config.strip_accents:
            steps.append(NormalizationStep(
                name="strip_accents", applied=False, chars_changed=0,
                description="Skipped (disabled)",
            ))
            return text

        stripped = _remove_accents(text)
        chars_changed = _count_char_differences(text, stripped)
        steps.append(NormalizationStep(
            name="strip_accents", applied=True, chars_changed=chars_changed,
            description="Removed combining accent marks",
        ))
        return stripped

    def _apply_lowercase(
        self, text: str, steps: list[NormalizationStep]
    ) -> str:
        if not self._config.lowercase:
            steps.append(NormalizationStep(
                name="lowercase", applied=False, chars_changed=0,
                description="Skipped (disabled)",
            ))
            return text

        lowered = text.lower()
        chars_changed = _count_char_differences(text, lowered)
        steps.append(NormalizationStep(
            name="lowercase", applied=True, chars_changed=chars_changed,
            description="Converted to lowercase",
        ))
        return lowered


def _count_char_differences(original: str, transformed: str) -> int:
    """Count character-level differences between two strings."""
    changes = 0
    for i in range(min(len(original), len(transformed))):
        if original[i] != transformed[i]:
            changes += 1
    changes += abs(len(original) - len(transformed))
    return changes


def _remove_invisible_chars(text: str) -> str:
    """Remove zero-width and invisible Unicode characters."""
    return "".join(ch for ch in text if ch not in _INVISIBLE_CHARS)


def _replace_homoglyphs(text: str) -> str:
    """Replace Cyrillic homoglyphs with Latin equivalents."""
    return "".join(_HOMOGLYPH_MAP.get(ch, ch) for ch in text)


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space, then strip."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _remove_accents(text: str) -> str:
    """Strip combining marks (accents) via NFD decomposition."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(
        ch for ch in decomposed
        if unicodedata.category(ch) != "Mn"
    )
