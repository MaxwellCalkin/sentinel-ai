"""Input sanitization for LLM applications.

Normalizes and cleans user input before sending to LLMs, removing
invisible characters, normalizing Unicode, limiting length, and
stripping potentially dangerous content.

Usage:
    from sentinel.input_sanitizer import InputSanitizer

    sanitizer = InputSanitizer(max_length=10000)
    clean = sanitizer.sanitize("Hello\\u200bworld")
    print(clean.text)       # "Helloworld"
    print(clean.modified)   # True
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SanitizeResult:
    """Result of input sanitization."""
    text: str
    original: str
    modifications: list[str] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return self.text != self.original

    @property
    def summary(self) -> str:
        if not self.modifications:
            return "No modifications"
        return "; ".join(self.modifications)


class InputSanitizer:
    """Sanitize user input for safe LLM consumption.

    Applies a configurable pipeline of sanitization steps:
    - Remove zero-width and invisible Unicode characters
    - Normalize Unicode (NFC form)
    - Strip control characters
    - Limit input length
    - Remove homoglyph attacks
    - Normalize whitespace
    """

    # Zero-width and invisible characters used in smuggling attacks
    _INVISIBLE_CHARS = re.compile(
        r"[\u200b\u200c\u200d\u200e\u200f"  # zero-width space, joiners, marks
        r"\u2060\u2061\u2062\u2063\u2064"    # word joiner, invisible operators
        r"\ufeff"                             # BOM / zero-width no-break space
        r"\u00ad"                             # soft hyphen
        r"\u034f"                             # combining grapheme joiner
        r"\u180e"                             # Mongolian vowel separator
        r"\ufff9\ufffa\ufffb"                # interlinear annotations
        r"]"
    )

    # Control characters (except tab, newline, carriage return)
    _CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    # Common homoglyphs (Cyrillic/Greek that look like Latin)
    _HOMOGLYPH_MAP: dict[str, str] = {
        "\u0410": "A",  # Cyrillic А → Latin A
        "\u0412": "B",  # Cyrillic В → Latin B
        "\u0421": "C",  # Cyrillic С → Latin C
        "\u0415": "E",  # Cyrillic Е → Latin E
        "\u041d": "H",  # Cyrillic Н → Latin H
        "\u041a": "K",  # Cyrillic К → Latin K
        "\u041c": "M",  # Cyrillic М → Latin M
        "\u041e": "O",  # Cyrillic О → Latin O
        "\u0420": "P",  # Cyrillic Р → Latin P
        "\u0422": "T",  # Cyrillic Т → Latin T
        "\u0425": "X",  # Cyrillic Х → Latin X
        "\u0430": "a",  # Cyrillic а → Latin a
        "\u0435": "e",  # Cyrillic е → Latin e
        "\u043e": "o",  # Cyrillic о → Latin o
        "\u0440": "p",  # Cyrillic р → Latin p
        "\u0441": "c",  # Cyrillic с → Latin c
        "\u0443": "y",  # Cyrillic у → Latin y
        "\u0445": "x",  # Cyrillic х → Latin x
        "\u0391": "A",  # Greek Α → Latin A
        "\u0392": "B",  # Greek Β → Latin B
        "\u0395": "E",  # Greek Ε → Latin E
        "\u0397": "H",  # Greek Η → Latin H
        "\u0399": "I",  # Greek Ι → Latin I
        "\u039a": "K",  # Greek Κ → Latin K
        "\u039c": "M",  # Greek Μ → Latin M
        "\u039d": "N",  # Greek Ν → Latin N
        "\u039f": "O",  # Greek Ο → Latin O
        "\u03a1": "P",  # Greek Ρ → Latin P
        "\u03a4": "T",  # Greek Τ → Latin T
        "\u03a7": "X",  # Greek Χ → Latin X
        "\u0370": "A",  # Greek A → Latin A (rare variant)
    }

    def __init__(
        self,
        max_length: int | None = None,
        strip_invisible: bool = True,
        normalize_unicode: bool = True,
        strip_control: bool = True,
        normalize_whitespace: bool = True,
        replace_homoglyphs: bool = True,
        custom_strips: Sequence[re.Pattern] | None = None,
    ):
        """
        Args:
            max_length: Maximum allowed character count (None = no limit).
            strip_invisible: Remove zero-width and invisible characters.
            normalize_unicode: Apply NFC normalization.
            strip_control: Remove control characters (except \\t, \\n, \\r).
            normalize_whitespace: Collapse multiple spaces/newlines.
            replace_homoglyphs: Replace common Cyrillic/Greek homoglyphs.
            custom_strips: Additional regex patterns to remove.
        """
        self._max_length = max_length
        self._strip_invisible = strip_invisible
        self._normalize_unicode = normalize_unicode
        self._strip_control = strip_control
        self._normalize_whitespace = normalize_whitespace
        self._replace_homoglyphs = replace_homoglyphs
        self._custom_strips = list(custom_strips) if custom_strips else []

    def sanitize(self, text: str) -> SanitizeResult:
        """Sanitize input text and return result with modification log."""
        original = text
        modifications: list[str] = []

        if self._strip_invisible:
            new_text = self._INVISIBLE_CHARS.sub("", text)
            if new_text != text:
                count = len(text) - len(new_text)
                modifications.append(f"Removed {count} invisible characters")
                text = new_text

        if self._strip_control:
            new_text = self._CONTROL_CHARS.sub("", text)
            if new_text != text:
                count = len(text) - len(new_text)
                modifications.append(f"Removed {count} control characters")
                text = new_text

        if self._normalize_unicode:
            new_text = unicodedata.normalize("NFC", text)
            if new_text != text:
                modifications.append("Applied NFC normalization")
                text = new_text

        if self._replace_homoglyphs:
            new_text = self._apply_homoglyph_replacement(text)
            if new_text != text:
                modifications.append("Replaced homoglyph characters")
                text = new_text

        if self._normalize_whitespace:
            # Collapse multiple spaces to single
            new_text = re.sub(r"[ \t]+", " ", text)
            # Collapse multiple newlines to double
            new_text = re.sub(r"\n{3,}", "\n\n", new_text)
            # Strip leading/trailing whitespace per line
            new_text = "\n".join(line.strip() for line in new_text.split("\n"))
            new_text = new_text.strip()
            if new_text != text:
                modifications.append("Normalized whitespace")
                text = new_text

        for pattern in self._custom_strips:
            new_text = pattern.sub("", text)
            if new_text != text:
                modifications.append(f"Applied custom pattern: {pattern.pattern}")
                text = new_text

        if self._max_length is not None and len(text) > self._max_length:
            text = text[:self._max_length]
            modifications.append(f"Truncated to {self._max_length} characters")

        return SanitizeResult(
            text=text,
            original=original,
            modifications=modifications,
        )

    def _apply_homoglyph_replacement(self, text: str) -> str:
        """Replace homoglyph characters with their Latin equivalents."""
        result = []
        for ch in text:
            result.append(self._HOMOGLYPH_MAP.get(ch, ch))
        return "".join(result)
