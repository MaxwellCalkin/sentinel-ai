"""Configurable content filter with category-based rules.

Filter content by categories (violence, sexual, hate speech, etc.)
with configurable severity thresholds per category.

Usage:
    from sentinel.content_filter import ContentFilter, Category

    f = ContentFilter()
    f.set_threshold(Category.VIOLENCE, 0.3)
    result = f.filter("Some violent text about fighting")
    print(result.blocked)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Category(Enum):
    """Content categories for filtering."""
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    HATE_SPEECH = "hate_speech"
    SELF_HARM = "self_harm"
    DRUGS = "drugs"
    PROFANITY = "profanity"
    WEAPONS = "weapons"
    PERSONAL_INFO = "personal_info"
    MISINFORMATION = "misinformation"


@dataclass
class CategoryScore:
    """Score for a content category."""
    category: Category
    score: float  # 0.0 to 1.0
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class FilterResult:
    """Result of content filtering."""
    text: str
    blocked: bool
    scores: list[CategoryScore] = field(default_factory=list)
    blocked_categories: list[Category] = field(default_factory=list)
    filtered_text: str = ""

    @property
    def is_clean(self) -> bool:
        return not self.blocked

    @property
    def top_category(self) -> Category | None:
        if not self.scores:
            return None
        return max(self.scores, key=lambda s: s.score).category

    @property
    def top_score(self) -> float:
        if not self.scores:
            return 0.0
        return max(s.score for s in self.scores)


# Pattern definitions for each category
_CATEGORY_PATTERNS: dict[Category, list[tuple[re.Pattern, float]]] = {
    Category.VIOLENCE: [
        (re.compile(r"\b(?:kill|murder|slaughter|assassinat|massacr)\w*\b", re.I), 0.8),
        (re.compile(r"\b(?:stab|shoot|beat|attack|assault|fight)\w*\b", re.I), 0.5),
        (re.compile(r"\b(?:blood|wound|gore|brutal|violent)\w*\b", re.I), 0.4),
        (re.compile(r"\b(?:tortur|mutilat|dismember)\w*\b", re.I), 0.9),
    ],
    Category.SEXUAL: [
        (re.compile(r"\b(?:explicit|pornograph|obscene)\b", re.I), 0.7),
        (re.compile(r"\b(?:nude|naked|strip)\b", re.I), 0.5),
    ],
    Category.HATE_SPEECH: [
        (re.compile(r"\b(?:supremac|inferior race|ethnic cleansing)\b", re.I), 0.9),
        (re.compile(r"\b(?:bigot|racist|xenophob|discriminat)\b", re.I), 0.4),
    ],
    Category.SELF_HARM: [
        (re.compile(r"\b(?:suicide|kill myself|end my life)\b", re.I), 0.9),
        (re.compile(r"\b(?:self[- ]harm|cut myself|hurt myself)\b", re.I), 0.8),
    ],
    Category.DRUGS: [
        (re.compile(r"\b(?:cocaine|heroin|methamphetamine|fentanyl)\b", re.I), 0.7),
        (re.compile(r"\b(?:drug deal|drug traffick|illegal substance)\b", re.I), 0.6),
    ],
    Category.PROFANITY: [
        (re.compile(r"\b(?:damn|hell|crap)\b", re.I), 0.3),
    ],
    Category.WEAPONS: [
        (re.compile(r"\b(?:bomb|explosive|grenade|landmine)\b", re.I), 0.7),
        (re.compile(r"\b(?:firearm|rifle|shotgun|ammunition)\b", re.I), 0.5),
        (re.compile(r"\b(?:weapon of mass destruction|chemical weapon|biological weapon)\b", re.I), 0.9),
    ],
    Category.PERSONAL_INFO: [
        (re.compile(r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b"), 0.8),  # SSN-like
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), 0.6),  # Email
        (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), 0.5),  # Phone
    ],
}


class ContentFilter:
    """Configurable content filter with category-based rules.

    Uses pattern matching to score content across multiple
    categories and block based on configurable thresholds.
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        redact: bool = False,
    ) -> None:
        """
        Args:
            default_threshold: Default blocking threshold per category.
            redact: Whether to produce redacted text in results.
        """
        self._thresholds: dict[Category, float] = {
            cat: default_threshold for cat in Category
        }
        self._redact = redact
        self._disabled: set[Category] = set()

    def set_threshold(self, category: Category, threshold: float) -> None:
        """Set blocking threshold for a category (0.0-1.0)."""
        self._thresholds[category] = threshold

    def disable_category(self, category: Category) -> None:
        """Disable checking for a category."""
        self._disabled.add(category)

    def enable_category(self, category: Category) -> None:
        """Re-enable checking for a category."""
        self._disabled.discard(category)

    def filter(self, text: str) -> FilterResult:
        """Filter text against all enabled categories.

        Returns:
            FilterResult with scores, blocked status, and optionally redacted text.
        """
        scores: list[CategoryScore] = []
        blocked_cats: list[Category] = []
        redacted = text

        for cat, patterns in _CATEGORY_PATTERNS.items():
            if cat in self._disabled:
                continue

            matched_terms: list[str] = []
            max_score = 0.0

            for pattern, weight in patterns:
                matches = pattern.findall(text)
                if matches:
                    matched_terms.extend(matches)
                    # Score increases with weight and number of matches
                    score = min(1.0, weight + 0.1 * (len(matches) - 1))
                    max_score = max(max_score, score)

            if matched_terms:
                scores.append(CategoryScore(
                    category=cat,
                    score=round(max_score, 2),
                    matched_terms=matched_terms,
                ))

                threshold = self._thresholds.get(cat, 0.5)
                if max_score >= threshold:
                    blocked_cats.append(cat)

                # Redact matched terms
                if self._redact:
                    for term in set(matched_terms):
                        redacted = redacted.replace(term, "[REDACTED]")

        return FilterResult(
            text=text,
            blocked=len(blocked_cats) > 0,
            scores=scores,
            blocked_categories=blocked_cats,
            filtered_text=redacted if self._redact else text,
        )

    def filter_batch(self, texts: list[str]) -> list[FilterResult]:
        """Filter multiple texts."""
        return [self.filter(t) for t in texts]
