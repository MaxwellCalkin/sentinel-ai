"""Prompt sanitization for LLM applications.

Sanitizes and normalizes user prompts before sending to LLMs,
removing potential attack vectors while preserving legitimate content.

Usage:
    from sentinel.prompt_sanitizer import PromptSanitizer, SanitizeConfig

    config = SanitizeConfig(max_length=5000)
    sanitizer = PromptSanitizer(config)
    report = sanitizer.sanitize("Hello\\u200bworld")
    print(report.sanitized)        # "Helloworld"
    print(report.chars_removed)    # 1
    print(report.modifications)    # ["Stripped 1 invisible character(s)"]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SanitizeConfig:
    """Configuration for prompt sanitization steps."""

    strip_invisible: bool = True
    normalize_whitespace: bool = True
    escape_delimiters: bool = True
    strip_homoglyphs: bool = True
    max_length: int | None = None
    truncation_strategy: str = "hard"  # "hard" or "word_boundary"


@dataclass
class SanitizeReport:
    """Report of sanitization actions taken on a prompt."""

    original: str
    sanitized: str
    modifications: list[str] = field(default_factory=list)
    chars_removed: int = 0
    was_truncated: bool = False


_INVISIBLE_CHARS = re.compile(
    r"[\u200b\u200c\u200d"  # zero-width space, non-joiner, joiner
    r"\u200e\u200f"  # left-to-right mark, right-to-left mark
    r"\u2060\u2061\u2062\u2063\u2064"  # word joiner, invisible operators
    r"\ufeff"  # BOM / zero-width no-break space
    r"\u00ad"  # soft hyphen
    r"\u034f"  # combining grapheme joiner
    r"\u180e"  # Mongolian vowel separator
    r"\ufff9\ufffa\ufffb"  # interlinear annotations
    r"\u2028\u2029"  # line/paragraph separator
    r"\u202a-\u202e"  # bidi embedding/override
    r"\u2066-\u2069"  # bidi isolate
    r"]"
)

_SYSTEM_DELIMITERS = [
    (re.compile(r"\[INST\]", re.IGNORECASE), "[INST]"),
    (re.compile(r"\[/INST\]", re.IGNORECASE), "[/INST]"),
    (re.compile(r"</s>", re.IGNORECASE), "</s>"),
    (re.compile(r"<<SYS>>", re.IGNORECASE), "<<SYS>>"),
    (re.compile(r"<</SYS>>", re.IGNORECASE), "<</SYS>>"),
    (re.compile(r"<\|im_start\|>", re.IGNORECASE), "<|im_start|>"),
    (re.compile(r"<\|im_end\|>", re.IGNORECASE), "<|im_end|>"),
    (re.compile(r"<\|system\|>", re.IGNORECASE), "<|system|>"),
    (re.compile(r"<\|user\|>", re.IGNORECASE), "<|user|>"),
    (re.compile(r"<\|assistant\|>", re.IGNORECASE), "<|assistant|>"),
    (re.compile(r"<\|endoftext\|>", re.IGNORECASE), "<|endoftext|>"),
]

_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic uppercase
    "\u0410": "A",
    "\u0412": "B",
    "\u0421": "C",
    "\u0415": "E",
    "\u041d": "H",
    "\u041a": "K",
    "\u041c": "M",
    "\u041e": "O",
    "\u0420": "P",
    "\u0422": "T",
    "\u0425": "X",
    # Cyrillic lowercase
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",
    "\u0445": "x",
    # Greek uppercase
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a7": "X",
}


def _strip_invisible_chars(text: str) -> tuple[str, int]:
    """Remove invisible Unicode characters and return cleaned text with removal count."""
    cleaned = _INVISIBLE_CHARS.sub("", text)
    removed_count = len(text) - len(cleaned)
    return cleaned, removed_count


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and newlines into single instances."""
    result = re.sub(r"[ \t]+", " ", text)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = "\n".join(line.strip() for line in result.split("\n"))
    return result.strip()


def _escape_delimiters(text: str) -> tuple[str, list[str]]:
    """Remove system token delimiters and return cleaned text with list of escaped tokens."""
    escaped_tokens: list[str] = []
    for pattern, token_name in _SYSTEM_DELIMITERS:
        if pattern.search(text):
            text = pattern.sub("", text)
            escaped_tokens.append(token_name)
    return text, escaped_tokens


def _replace_homoglyphs(text: str) -> tuple[str, int]:
    """Replace homoglyph characters with Latin equivalents."""
    replacements = 0
    result_chars: list[str] = []
    for char in text:
        latin_equivalent = _HOMOGLYPH_MAP.get(char)
        if latin_equivalent is not None:
            result_chars.append(latin_equivalent)
            replacements += 1
        else:
            result_chars.append(char)
    return "".join(result_chars), replacements


def _truncate_hard(text: str, max_length: int) -> str:
    """Truncate text at exact character position."""
    return text[:max_length]


def _truncate_word_boundary(text: str, max_length: int) -> str:
    """Truncate text at the last word boundary within max_length."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space <= 0:
        return truncated
    return truncated[:last_space]


class PromptSanitizer:
    """Sanitize user prompts before sending to LLMs.

    Applies configurable sanitization steps to remove attack vectors
    while preserving legitimate content. Each step can be independently
    enabled or disabled via SanitizeConfig.
    """

    def __init__(self, config: SanitizeConfig | None = None):
        self._config = config or SanitizeConfig()

    @property
    def config(self) -> SanitizeConfig:
        return self._config

    def sanitize(self, prompt: str) -> SanitizeReport:
        """Sanitize a prompt and return a detailed report."""
        original = prompt
        modifications: list[str] = []
        total_chars_removed = 0
        text = prompt

        text, total_chars_removed, modifications = self._apply_invisible_strip(
            text, total_chars_removed, modifications
        )
        text, total_chars_removed, modifications = self._apply_homoglyph_strip(
            text, total_chars_removed, modifications
        )
        text, total_chars_removed, modifications = self._apply_delimiter_escape(
            text, total_chars_removed, modifications
        )
        text, modifications = self._apply_whitespace_normalization(
            text, modifications
        )
        text, was_truncated, modifications = self._apply_truncation(
            text, modifications
        )

        return SanitizeReport(
            original=original,
            sanitized=text,
            modifications=modifications,
            chars_removed=total_chars_removed,
            was_truncated=was_truncated,
        )

    def _apply_invisible_strip(
        self,
        text: str,
        chars_removed: int,
        modifications: list[str],
    ) -> tuple[str, int, list[str]]:
        if not self._config.strip_invisible:
            return text, chars_removed, modifications
        cleaned, removed = _strip_invisible_chars(text)
        if removed > 0:
            modifications.append(f"Stripped {removed} invisible character(s)")
            chars_removed += removed
            text = cleaned
        return text, chars_removed, modifications

    def _apply_homoglyph_strip(
        self,
        text: str,
        chars_removed: int,
        modifications: list[str],
    ) -> tuple[str, int, list[str]]:
        if not self._config.strip_homoglyphs:
            return text, chars_removed, modifications
        cleaned, replacement_count = _replace_homoglyphs(text)
        if replacement_count > 0:
            modifications.append(
                f"Replaced {replacement_count} homoglyph character(s)"
            )
            text = cleaned
        return text, chars_removed, modifications

    def _apply_delimiter_escape(
        self,
        text: str,
        chars_removed: int,
        modifications: list[str],
    ) -> tuple[str, int, list[str]]:
        if not self._config.escape_delimiters:
            return text, chars_removed, modifications
        length_before = len(text)
        cleaned, escaped_tokens = _escape_delimiters(text)
        if escaped_tokens:
            removed = length_before - len(cleaned)
            chars_removed += removed
            modifications.append(
                f"Escaped delimiters: {', '.join(escaped_tokens)}"
            )
            text = cleaned
        return text, chars_removed, modifications

    def _apply_whitespace_normalization(
        self,
        text: str,
        modifications: list[str],
    ) -> tuple[str, list[str]]:
        if not self._config.normalize_whitespace:
            return text, modifications
        cleaned = _normalize_whitespace(text)
        if cleaned != text:
            modifications.append("Normalized whitespace")
            text = cleaned
        return text, modifications

    def _apply_truncation(
        self,
        text: str,
        modifications: list[str],
    ) -> tuple[str, bool, list[str]]:
        max_length = self._config.max_length
        if max_length is None or len(text) <= max_length:
            return text, False, modifications

        if self._config.truncation_strategy == "word_boundary":
            text = _truncate_word_boundary(text, max_length)
        else:
            text = _truncate_hard(text, max_length)

        modifications.append(
            f"Truncated to {len(text)} characters ({self._config.truncation_strategy})"
        )
        return text, True, modifications
