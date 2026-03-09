"""Multi-category toxicity detection and filtering for LLM inputs/outputs.

Detects toxicity across categories (hate speech, harassment, threats, etc.)
using keyword/pattern matching with severity scoring.

Usage:
    from sentinel.toxicity_filter import ToxicityFilter

    f = ToxicityFilter()
    result = f.scan("I will destroy you completely")
    assert result.is_toxic
    assert result.score > 0.0
    assert "threats" in result.categories_triggered
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ToxicityMatch:
    """A single toxicity match found in text."""
    text: str
    category: str
    severity: float
    pattern: str


@dataclass
class ToxicityResult:
    """Result of scanning text for toxicity."""
    text: str
    is_toxic: bool
    score: float
    matches: list[ToxicityMatch] = field(default_factory=list)
    categories_triggered: list[str] = field(default_factory=list)


@dataclass
class ToxicityStats:
    """Aggregate statistics from toxicity scanning."""
    total_scanned: int
    flagged_count: int
    flag_rate: float
    by_category: dict[str, int] = field(default_factory=dict)


_DEFAULT_PATTERNS: dict[str, list[tuple[re.Pattern, float]]] = {
    "hate_speech": [
        (re.compile(r"\b(?:supremac(?:y|ist))\b", re.I), 0.9),
        (re.compile(r"\b(?:ethnic cleansing)\b", re.I), 0.95),
        (re.compile(r"\b(?:inferior race|master race)\b", re.I), 0.9),
        (re.compile(r"\b(?:subhuman|untermensch)\b", re.I), 0.85),
        (re.compile(r"\b(?:degenerate|vermin|parasit\w*)\b", re.I), 0.6),
        (re.compile(r"\bgo back to (?:your|where)\b", re.I), 0.7),
    ],
    "harassment": [
        (re.compile(r"\byou(?:'re| are) (?:worthless|pathetic|disgusting|useless)\b", re.I), 0.7),
        (re.compile(r"\b(?:nobody (?:likes|loves|cares about) you)\b", re.I), 0.7),
        (re.compile(r"\b(?:shut up|get lost|go away)\b", re.I), 0.3),
        (re.compile(r"\b(?:loser|moron|idiot|stupid)\b", re.I), 0.4),
        (re.compile(r"\b(?:stalking|harassing|bullying)\b", re.I), 0.6),
        (re.compile(r"\byou deserve (?:to suffer|pain|nothing)\b", re.I), 0.8),
    ],
    "threats": [
        (re.compile(r"\bi(?:'ll| will) (?:kill|destroy|hurt|end) you\b", re.I), 0.9),
        (re.compile(r"\b(?:death threat|bomb threat)\b", re.I), 0.95),
        (re.compile(r"\byou(?:'re| are) (?:dead|done for)\b", re.I), 0.7),
        (re.compile(r"\bi(?:'ll| will) (?:find|hunt|track) you\b", re.I), 0.8),
        (re.compile(r"\b(?:watch your back|sleep with one eye)\b", re.I), 0.7),
        (re.compile(r"\b(?:burn down|blow up)\b", re.I), 0.8),
    ],
    "profanity": [
        (re.compile(r"\b(?:damn|hell|crap)\b", re.I), 0.2),
        (re.compile(r"\b(?:ass|asshole|bastard)\b", re.I), 0.4),
        (re.compile(r"\bf+u+c+k+\w*\b", re.I), 0.6),
        (re.compile(r"\bs+h+i+t+\w*\b", re.I), 0.5),
        (re.compile(r"\bbitch\w*\b", re.I), 0.5),
    ],
    "sexual_content": [
        (re.compile(r"\b(?:explicit sexual|sexual acts?)\b", re.I), 0.8),
        (re.compile(r"\b(?:pornograph\w*|obscen\w*)\b", re.I), 0.7),
        (re.compile(r"\b(?:nude|naked|strip)\b", re.I), 0.4),
        (re.compile(r"\b(?:sexually explicit)\b", re.I), 0.8),
        (re.compile(r"\b(?:erotic|lewd|indecent)\b", re.I), 0.5),
    ],
    "self_harm": [
        (re.compile(r"\b(?:kill myself|end my life)\b", re.I), 0.95),
        (re.compile(r"\b(?:suicide|suicidal)\b", re.I), 0.9),
        (re.compile(r"\b(?:self[- ]harm|cut myself|hurt myself)\b", re.I), 0.85),
        (re.compile(r"\b(?:want to die|better off dead)\b", re.I), 0.9),
        (re.compile(r"\b(?:no reason to live|not worth living)\b", re.I), 0.85),
    ],
}


def _compute_category_score(match_severities: list[float]) -> float:
    """Combine individual match severities into a single category score.

    Uses the maximum severity, boosted slightly by additional matches.
    """
    if not match_severities:
        return 0.0
    base = max(match_severities)
    extra_matches = len(match_severities) - 1
    boost = min(0.1, extra_matches * 0.02)
    return min(1.0, base + boost)


def _compute_overall_score(category_scores: list[float]) -> float:
    """Combine per-category scores into an overall toxicity score.

    Uses the maximum category score, boosted slightly by additional categories.
    """
    if not category_scores:
        return 0.0
    base = max(category_scores)
    extra_categories = len(category_scores) - 1
    boost = min(0.15, extra_categories * 0.05)
    return round(min(1.0, base + boost), 4)


class ToxicityFilter:
    """Multi-category toxicity detection and filtering.

    Scans text for toxic content across configurable categories,
    returning severity scores and optional content redaction.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._patterns: dict[str, list[tuple[re.Pattern, float]]] = {
            category: list(patterns)
            for category, patterns in _DEFAULT_PATTERNS.items()
        }
        self._total_scanned = 0
        self._flagged_count = 0
        self._flagged_by_category: dict[str, int] = {}

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def categories(self) -> list[str]:
        return sorted(self._patterns.keys())

    def register_category(
        self,
        name: str,
        patterns: list[tuple[str, float]],
    ) -> None:
        """Register a custom toxicity category with pattern/severity pairs."""
        compiled = [
            (re.compile(pat, re.I), severity)
            for pat, severity in patterns
        ]
        self._patterns[name] = compiled

    def scan(self, text: str) -> ToxicityResult:
        """Scan text for toxicity across all registered categories."""
        self._total_scanned += 1
        matches = self._find_matches(text)
        category_scores = self._aggregate_by_category(matches)
        overall_score = _compute_overall_score(list(category_scores.values()))
        is_toxic = overall_score >= self._threshold
        categories_triggered = sorted(category_scores.keys())

        if is_toxic:
            self._flagged_count += 1
            for category in categories_triggered:
                self._flagged_by_category[category] = (
                    self._flagged_by_category.get(category, 0) + 1
                )

        return ToxicityResult(
            text=text,
            is_toxic=is_toxic,
            score=overall_score,
            matches=matches,
            categories_triggered=categories_triggered,
        )

    def filter(self, text: str) -> str:
        """Scan text and redact toxic matches with [FILTERED]."""
        result = self.scan(text)
        if not result.is_toxic:
            return text
        return self._redact_matches(text, result.matches)

    def scan_batch(self, texts: list[str]) -> list[ToxicityResult]:
        """Scan multiple texts for toxicity."""
        return [self.scan(text) for text in texts]

    def stats(self) -> ToxicityStats:
        """Return aggregate scanning statistics."""
        flag_rate = (
            self._flagged_count / self._total_scanned
            if self._total_scanned > 0
            else 0.0
        )
        return ToxicityStats(
            total_scanned=self._total_scanned,
            flagged_count=self._flagged_count,
            flag_rate=round(flag_rate, 4),
            by_category=dict(self._flagged_by_category),
        )

    def _find_matches(self, text: str) -> list[ToxicityMatch]:
        """Find all toxicity pattern matches in text."""
        matches: list[ToxicityMatch] = []
        for category, patterns in self._patterns.items():
            for compiled_pattern, severity in patterns:
                for match in compiled_pattern.finditer(text):
                    matches.append(ToxicityMatch(
                        text=match.group(),
                        category=category,
                        severity=severity,
                        pattern=compiled_pattern.pattern,
                    ))
        return matches

    def _aggregate_by_category(
        self, matches: list[ToxicityMatch],
    ) -> dict[str, float]:
        """Group matches by category and compute per-category scores."""
        severities_by_category: dict[str, list[float]] = {}
        for match in matches:
            severities_by_category.setdefault(match.category, []).append(
                match.severity
            )
        return {
            category: _compute_category_score(severities)
            for category, severities in severities_by_category.items()
        }

    @staticmethod
    def _redact_matches(text: str, matches: list[ToxicityMatch]) -> str:
        """Replace matched toxic spans with [FILTERED]."""
        sorted_matches = sorted(
            matches,
            key=lambda m: text.find(m.text),
            reverse=True,
        )
        redacted = text
        seen_positions: set[int] = set()
        for match in sorted_matches:
            position = redacted.find(match.text)
            if position != -1 and position not in seen_positions:
                seen_positions.add(position)
                redacted = (
                    redacted[:position]
                    + "[FILTERED]"
                    + redacted[position + len(match.text):]
                )
        return redacted
