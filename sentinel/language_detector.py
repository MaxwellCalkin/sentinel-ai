"""Language detection and enforcement.

Detect the language of text and enforce language constraints
for LLM inputs and outputs. Useful for multilingual safety
and compliance.

Usage:
    from sentinel.language_detector import LanguageDetector

    detector = LanguageDetector(allowed=["en", "fr", "de"])
    result = detector.check("Hello, how are you?")
    assert result.language == "en"
    assert result.allowed
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LanguageResult:
    """Result of language detection."""
    text: str
    language: str      # ISO 639-1 code
    confidence: float  # 0.0 to 1.0
    allowed: bool = True
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def is_english(self) -> bool:
        return self.language == "en"


# Character ranges for script detection
_SCRIPT_RANGES: list[tuple[str, str, str]] = [
    ("zh", "\u4e00", "\u9fff"),      # CJK Unified Ideographs
    ("ja", "\u3040", "\u309f"),      # Hiragana
    ("ja", "\u30a0", "\u30ff"),      # Katakana
    ("ko", "\uac00", "\ud7af"),      # Hangul
    ("ar", "\u0600", "\u06ff"),      # Arabic
    ("hi", "\u0900", "\u097f"),      # Devanagari
    ("th", "\u0e00", "\u0e7f"),      # Thai
    ("ru", "\u0400", "\u04ff"),      # Cyrillic
    ("el", "\u0370", "\u03ff"),      # Greek
    ("he", "\u0590", "\u05ff"),      # Hebrew
]

# Common words by language for Latin-script languages
_WORD_SIGNATURES: dict[str, set[str]] = {
    "en": {"the", "is", "and", "to", "of", "in", "that", "it", "for", "was",
           "are", "with", "this", "have", "from", "not", "but", "they", "his",
           "she", "will", "been", "would", "could", "their", "which", "about"},
    "fr": {"le", "la", "les", "de", "des", "du", "et", "est", "un", "une",
           "en", "que", "pas", "pour", "sur", "avec", "dans", "plus", "nous",
           "sont", "mais", "ont", "aux", "cette", "tout"},
    "de": {"der", "die", "das", "und", "ist", "ein", "eine", "den", "dem",
           "nicht", "sich", "mit", "auf", "auch", "noch", "wie", "aber",
           "werden", "haben", "oder", "nach", "sind", "kann"},
    "es": {"el", "la", "los", "las", "de", "del", "en", "que", "por", "con",
           "una", "para", "como", "pero", "sus", "mas", "este", "entre",
           "cuando", "muy", "sin", "sobre", "todo", "desde"},
    "pt": {"de", "que", "em", "um", "uma", "para", "com", "por", "mais",
           "como", "mas", "dos", "das", "foi", "tem", "seu", "sua",
           "quando", "muito", "nos", "entre", "depois", "sem"},
    "it": {"di", "che", "il", "la", "per", "non", "una", "con", "del",
           "della", "sono", "anche", "come", "dal", "questo", "essere",
           "stato", "hanno", "suo", "sua", "loro", "molto"},
    "nl": {"de", "het", "een", "van", "en", "dat", "die", "niet", "zijn",
           "voor", "met", "ook", "maar", "nog", "wel", "naar", "bij",
           "kan", "meer", "dan", "werd", "heeft"},
}

_WORD_RE = re.compile(r"[a-zA-Z\u00C0-\u024F]+")


class LanguageDetector:
    """Detect and enforce language constraints.

    Uses script detection and word frequency analysis
    for zero-dependency language identification.
    """

    def __init__(
        self,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
    ) -> None:
        """
        Args:
            allowed: List of allowed ISO 639-1 codes (None = all allowed).
            blocked: List of blocked language codes.
        """
        self._allowed = set(allowed) if allowed else None
        self._blocked = set(blocked) if blocked else set()

    def detect(self, text: str) -> str:
        """Detect the primary language of text.

        Returns:
            ISO 639-1 language code.
        """
        result = self.check(text)
        return result.language

    def check(self, text: str) -> LanguageResult:
        """Check text language and whether it's allowed.

        Returns:
            LanguageResult with detected language and allowed status.
        """
        if not text.strip():
            return LanguageResult(
                text=text, language="und", confidence=0.0,
                allowed=True, scores={},
            )

        scores: dict[str, float] = {}

        # Script-based detection
        script_scores = self._detect_scripts(text)
        for lang, score in script_scores.items():
            scores[lang] = scores.get(lang, 0) + score

        # Word-based detection for Latin scripts
        word_scores = self._detect_words(text)
        for lang, score in word_scores.items():
            scores[lang] = scores.get(lang, 0) + score

        if not scores:
            language = "und"
            confidence = 0.0
        else:
            language = max(scores, key=lambda k: scores[k])
            total = sum(scores.values())
            confidence = scores[language] / total if total > 0 else 0.0
            confidence = min(1.0, confidence)

        # Check if allowed
        is_allowed = True
        if language in self._blocked:
            is_allowed = False
        elif self._allowed is not None and language not in self._allowed:
            is_allowed = False

        return LanguageResult(
            text=text,
            language=language,
            confidence=round(confidence, 4),
            allowed=is_allowed,
            scores={k: round(v, 4) for k, v in scores.items()},
        )

    def _detect_scripts(self, text: str) -> dict[str, float]:
        """Detect languages by Unicode script ranges."""
        counts: Counter = Counter()
        total = 0
        for char in text:
            for lang, start, end in _SCRIPT_RANGES:
                if start <= char <= end:
                    counts[lang] += 1
                    total += 1
                    break

        if total == 0:
            return {}
        return {lang: count / total for lang, count in counts.items()}

    def _detect_words(self, text: str) -> dict[str, float]:
        """Detect language by common word frequency."""
        words = [w.lower() for w in _WORD_RE.findall(text)]
        if not words:
            return {}

        scores: dict[str, float] = {}
        for lang, signature in _WORD_SIGNATURES.items():
            matched = sum(1 for w in words if w in signature)
            if matched > 0:
                scores[lang] = matched / len(words)

        return scores
