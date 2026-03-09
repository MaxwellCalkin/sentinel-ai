"""Intent classifier for LLM prompts.

Classifies user prompts by intent to enable intent-aware safety
routing and policy enforcement. Supports keyword and regex pattern
matching with configurable risk levels per intent.

Usage:
    from sentinel.intent_classifier import IntentClassifier

    classifier = IntentClassifier()
    classifier.add_intent("code_gen", keywords=["write", "code", "function"], risk_level="medium")
    classifier.add_intent("data_query", keywords=["select", "database", "query"], risk_level="low")

    result = classifier.classify("Write a Python function to sort a list")
    print(result.intent)      # "code_gen"
    print(result.risk_level)  # "medium"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_VALID_RISK_LEVELS = {"low", "medium", "high", "critical"}


@dataclass
class IntentDef:
    """Definition of an intent."""
    name: str
    keywords: list[str]
    patterns: list[str]
    risk_level: str


@dataclass
class ClassificationOutput:
    """Result of intent classification."""
    text: str
    intent: str
    confidence: float
    risk_level: str
    matched_keywords: list[str]
    matched_patterns: list[str]


@dataclass
class ClassifierStats:
    """Accumulated classification statistics."""
    total_classified: int
    by_intent: dict[str, int]
    by_risk: dict[str, int]
    avg_confidence: float


class IntentClassifier:
    """Classify prompts by intent for safety routing.

    Register intents with keywords and regex patterns, then classify
    text to determine the best-matching intent with a confidence score
    and associated risk level.
    """

    def __init__(self, default_intent: str = "general") -> None:
        self._default_intent = default_intent
        self._intents: dict[str, IntentDef] = {}
        self._total_classified: int = 0
        self._by_intent: dict[str, int] = {}
        self._by_risk: dict[str, int] = {}
        self._confidence_sum: float = 0.0

    def add_intent(
        self,
        name: str,
        keywords: list[str],
        patterns: list[str] | None = None,
        risk_level: str = "low",
    ) -> None:
        """Register an intent with keywords and optional regex patterns."""
        if risk_level not in _VALID_RISK_LEVELS:
            raise ValueError(
                f"Invalid risk_level '{risk_level}'. "
                f"Must be one of: {', '.join(sorted(_VALID_RISK_LEVELS))}"
            )
        self._intents[name] = IntentDef(
            name=name,
            keywords=[k.lower() for k in keywords],
            patterns=list(patterns) if patterns else [],
            risk_level=risk_level,
        )

    def remove_intent(self, name: str) -> None:
        """Remove a registered intent by name."""
        self._intents.pop(name, None)

    def list_intents(self) -> list[str]:
        """Return sorted list of registered intent names."""
        return sorted(self._intents.keys())

    def get_risk(self, intent: str) -> str:
        """Return the risk level for a registered intent."""
        if intent not in self._intents:
            raise KeyError(f"Intent '{intent}' not found")
        return self._intents[intent].risk_level

    def classify(self, text: str) -> ClassificationOutput:
        """Classify text into the best-matching intent."""
        best_output = self._score_all_intents(text)
        self._record_stats(best_output)
        return best_output

    def classify_batch(self, texts: list[str]) -> list[ClassificationOutput]:
        """Classify multiple texts."""
        return [self.classify(t) for t in texts]

    def stats(self) -> ClassifierStats:
        """Return accumulated classification statistics."""
        avg_confidence = (
            self._confidence_sum / self._total_classified
            if self._total_classified > 0
            else 0.0
        )
        return ClassifierStats(
            total_classified=self._total_classified,
            by_intent=dict(self._by_intent),
            by_risk=dict(self._by_risk),
            avg_confidence=round(avg_confidence, 4),
        )

    def _score_all_intents(self, text: str) -> ClassificationOutput:
        """Score text against all intents and return the best match."""
        text_lower = text.lower()
        best_score = 0.0
        best_intent: IntentDef | None = None
        best_matched_keywords: list[str] = []
        best_matched_patterns: list[str] = []

        for intent_def in self._intents.values():
            matched_keywords = self._find_keyword_matches(text_lower, intent_def.keywords)
            matched_patterns = self._find_pattern_matches(text, intent_def.patterns)
            score = len(matched_keywords) * 1.0 + len(matched_patterns) * 2.0

            if score > best_score:
                best_score = score
                best_intent = intent_def
                best_matched_keywords = matched_keywords
                best_matched_patterns = matched_patterns

        if best_intent is None:
            return self._build_default_output(text)

        confidence = self._compute_confidence(best_score, best_intent)
        return ClassificationOutput(
            text=text,
            intent=best_intent.name,
            confidence=confidence,
            risk_level=best_intent.risk_level,
            matched_keywords=best_matched_keywords,
            matched_patterns=best_matched_patterns,
        )

    def _find_keyword_matches(self, text_lower: str, keywords: list[str]) -> list[str]:
        """Return keywords that appear as substrings in the lowered text."""
        return [k for k in keywords if k in text_lower]

    def _find_pattern_matches(self, text: str, patterns: list[str]) -> list[str]:
        """Return patterns that match via re.search."""
        return [p for p in patterns if re.search(p, text)]

    def _compute_confidence(self, score: float, intent_def: IntentDef) -> float:
        """Compute confidence as score / max_possible_score, clamped to 1.0."""
        max_possible = len(intent_def.keywords) * 1.0 + len(intent_def.patterns) * 2.0
        if max_possible == 0.0:
            return 0.0
        return min(1.0, round(score / max_possible, 4))

    def _build_default_output(self, text: str) -> ClassificationOutput:
        """Build output for the default intent (no match)."""
        return ClassificationOutput(
            text=text,
            intent=self._default_intent,
            confidence=0.0,
            risk_level="low",
            matched_keywords=[],
            matched_patterns=[],
        )

    def _record_stats(self, output: ClassificationOutput) -> None:
        """Record classification result into running statistics."""
        self._total_classified += 1
        self._by_intent[output.intent] = self._by_intent.get(output.intent, 0) + 1
        self._by_risk[output.risk_level] = self._by_risk.get(output.risk_level, 0) + 1
        self._confidence_sum += output.confidence
