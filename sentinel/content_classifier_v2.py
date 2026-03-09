"""Multi-label content classification with confidence calibration.

Classifies text into multiple categories simultaneously with calibrated
confidence scores. A text can match several labels (e.g., both "technical"
and "educational") depending on keyword overlap.

Usage:
    from sentinel.content_classifier_v2 import ContentClassifierV2

    classifier = ContentClassifierV2()
    result = classifier.classify("Learn how to build an API with database queries")
    print(result.top_label)       # "technical"
    print(result.is_multi_label)  # True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class LabelDefinition:
    """Definition of a classification label with associated keywords."""

    name: str
    keywords: list[str]
    weight: float = 1.0
    description: str = ""


@dataclass
class ClassificationLabel:
    """A single label result with confidence and evidence."""

    name: str
    confidence: float
    matched_keywords: list[str]


@dataclass
class ClassificationResult:
    """Full classification result for a piece of text."""

    text: str
    labels: list[ClassificationLabel]
    top_label: str
    is_multi_label: bool
    total_confidence: float


@dataclass
class ClassifierConfig:
    """Configuration for the content classifier."""

    min_confidence: float = 0.1
    max_labels: int = 5
    calibration_factor: float = 1.0


@dataclass
class ClassifierStats:
    """Cumulative statistics from classification runs."""

    total_classified: int = 0
    by_label: dict[str, int] = field(default_factory=dict)
    multi_label_rate: float = 0.0


_BUILTIN_LABELS = [
    LabelDefinition(
        name="technical",
        keywords=[
            "algorithm", "api", "code", "database", "function",
            "server", "deploy", "debug", "compile", "runtime",
        ],
        description="Technical and programming content",
    ),
    LabelDefinition(
        name="educational",
        keywords=[
            "learn", "tutorial", "explain", "teach", "study",
            "course", "lesson", "understand", "knowledge", "practice",
        ],
        description="Educational and instructional content",
    ),
    LabelDefinition(
        name="creative",
        keywords=[
            "story", "poem", "imagine", "fiction", "novel",
            "character", "narrative", "metaphor", "artistic", "creative",
        ],
        description="Creative and artistic content",
    ),
    LabelDefinition(
        name="business",
        keywords=[
            "revenue", "strategy", "market", "roi", "profit",
            "growth", "investor", "stakeholder", "budget", "quarterly",
        ],
        description="Business and financial content",
    ),
    LabelDefinition(
        name="safety",
        keywords=[
            "harmful", "unsafe", "injection", "attack", "exploit",
            "vulnerability", "malicious", "threat", "breach", "risk",
        ],
        description="Safety and security content",
    ),
    LabelDefinition(
        name="personal",
        keywords=[
            "my", "i", "me", "feel", "opinion",
            "personally", "believe", "myself", "emotion", "experience",
        ],
        description="Personal and subjective content",
    ),
]


class ContentClassifierV2:
    """Multi-label content classifier with confidence calibration.

    Classifies text against registered labels using keyword matching,
    producing calibrated confidence scores. Supports custom labels,
    batch classification, and cumulative statistics tracking.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self._config = config or ClassifierConfig()
        self._labels: dict[str, LabelDefinition] = {}
        self._total_classified = 0
        self._multi_label_count = 0
        self._label_hit_counts: dict[str, int] = {}
        self._load_builtin_labels()

    def _load_builtin_labels(self) -> None:
        for label in _BUILTIN_LABELS:
            self._labels[label.name] = label

    def classify(self, text: str) -> ClassificationResult:
        """Classify text against all registered labels."""
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)

        scored_labels = self._score_all_labels(text_lower, total_words)
        filtered_labels = self._apply_config_filters(scored_labels)
        result = self._build_result(text, filtered_labels)
        self._update_stats(result)
        return result

    def _score_all_labels(
        self, text_lower: str, total_words: int
    ) -> list[ClassificationLabel]:
        scored: list[ClassificationLabel] = []
        for label_def in self._labels.values():
            classification = self._score_single_label(
                label_def, text_lower, total_words
            )
            if classification is not None:
                scored.append(classification)
        scored.sort(key=lambda label: label.confidence, reverse=True)
        return scored

    def _score_single_label(
        self, label_def: LabelDefinition, text_lower: str, total_words: int
    ) -> ClassificationLabel | None:
        matched = [
            kw for kw in label_def.keywords
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
        ]
        if not matched:
            return None

        divisor = max(total_words * 0.1, 1)
        raw_confidence = (len(matched) * label_def.weight) / divisor
        calibrated = raw_confidence * self._config.calibration_factor
        capped = min(calibrated, 1.0)

        if capped < self._config.min_confidence:
            return None

        return ClassificationLabel(
            name=label_def.name,
            confidence=round(capped, 4),
            matched_keywords=matched,
        )

    def _apply_config_filters(
        self, labels: list[ClassificationLabel]
    ) -> list[ClassificationLabel]:
        return labels[: self._config.max_labels]

    def _build_result(
        self, text: str, labels: list[ClassificationLabel]
    ) -> ClassificationResult:
        top_label = labels[0].name if labels else "unclassified"
        total_confidence = sum(label.confidence for label in labels)
        return ClassificationResult(
            text=text,
            labels=labels,
            top_label=top_label,
            is_multi_label=len(labels) > 1,
            total_confidence=round(total_confidence, 4),
        )

    def _update_stats(self, result: ClassificationResult) -> None:
        self._total_classified += 1
        if result.is_multi_label:
            self._multi_label_count += 1
        for label in result.labels:
            self._label_hit_counts[label.name] = (
                self._label_hit_counts.get(label.name, 0) + 1
            )

    def add_label(self, label: LabelDefinition) -> None:
        """Register a custom label definition."""
        self._labels[label.name] = label

    def remove_label(self, name: str) -> None:
        """Remove a registered label by name.

        Raises:
            KeyError: If the label name is not registered.
        """
        if name not in self._labels:
            raise KeyError(f"Label '{name}' not found")
        del self._labels[name]

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify multiple texts in sequence."""
        return [self.classify(text) for text in texts]

    def list_labels(self) -> list[str]:
        """Return all registered label names."""
        return list(self._labels.keys())

    def stats(self) -> ClassifierStats:
        """Return cumulative classification statistics."""
        total = self._total_classified
        multi_label_rate = (
            self._multi_label_count / total if total > 0 else 0.0
        )
        return ClassifierStats(
            total_classified=total,
            by_label=dict(self._label_hit_counts),
            multi_label_rate=round(multi_label_rate, 4),
        )
