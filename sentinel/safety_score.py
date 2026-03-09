"""Composite safety scoring for LLM interactions.

Aggregates multiple safety signals into a single 0-100 score
with letter grades. Useful for dashboards and monitoring.

Usage:
    from sentinel.safety_score import SafetyScorer

    scorer = SafetyScorer()
    score = scorer.score(text="Some LLM output", context={
        "injection_score": 0.1,
        "toxicity_score": 0.0,
        "pii_detected": False,
        "hallucination_risk": 0.2,
    })
    print(score.value)   # 92
    print(score.grade)   # "A"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SafetyScore:
    """A composite safety score."""
    value: int           # 0-100
    grade: str           # A, B, C, D, F
    components: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return self.value >= 70

    @property
    def is_critical(self) -> bool:
        return self.value < 30


def _grade(score: int) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    return "F"


class SafetyScorer:
    """Compute composite safety scores.

    Combine multiple safety signals with configurable weights
    into a single 0-100 score.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            weights: Custom weights for components. Higher = more important.
                Default weights used if not specified.
        """
        self._weights = weights or {
            "injection": 3.0,
            "toxicity": 2.5,
            "pii": 2.0,
            "hallucination": 1.5,
            "harmful": 3.0,
            "bias": 1.5,
            "off_topic": 0.5,
            "profanity": 1.0,
        }
        self._custom_components: list[tuple[str, Callable[[str, dict], float], float]] = []

    def add_component(
        self,
        name: str,
        scorer: Callable[[str, dict[str, Any]], float],
        weight: float = 1.0,
    ) -> None:
        """Add a custom scoring component.

        Args:
            name: Component name.
            scorer: Function(text, context) -> risk score (0.0 = safe, 1.0 = dangerous).
            weight: Weight for this component.
        """
        self._custom_components.append((name, scorer, weight))

    def score(
        self,
        text: str = "",
        context: dict[str, Any] | None = None,
    ) -> SafetyScore:
        """Compute composite safety score.

        Args:
            text: The text to score.
            context: Pre-computed safety signals. Keys:
                - injection_score: 0.0-1.0
                - toxicity_score: 0.0-1.0
                - pii_detected: bool
                - hallucination_risk: 0.0-1.0
                - harmful_score: 0.0-1.0
                - bias_score: 0.0-1.0
                - off_topic_score: 0.0-1.0
                - profanity_score: 0.0-1.0

        Returns:
            SafetyScore with 0-100 value and letter grade.
        """
        ctx = context or {}
        components: dict[str, float] = {}
        flags: list[str] = []

        # Map context keys to component scores
        signal_map = {
            "injection": ctx.get("injection_score", 0.0),
            "toxicity": ctx.get("toxicity_score", 0.0),
            "pii": 1.0 if ctx.get("pii_detected", False) else ctx.get("pii_score", 0.0),
            "hallucination": ctx.get("hallucination_risk", 0.0),
            "harmful": ctx.get("harmful_score", 0.0),
            "bias": ctx.get("bias_score", 0.0),
            "off_topic": ctx.get("off_topic_score", 0.0),
            "profanity": ctx.get("profanity_score", 0.0),
        }

        total_weight = 0.0
        weighted_risk = 0.0

        for name, risk in signal_map.items():
            weight = self._weights.get(name, 1.0)
            components[name] = round(risk, 4)
            weighted_risk += risk * weight
            total_weight += weight

            if risk > 0.5:
                flags.append(f"{name}: high risk ({risk:.2f})")

        # Custom components
        for name, scorer, weight in self._custom_components:
            try:
                risk = scorer(text, ctx)
                risk = max(0.0, min(1.0, risk))
            except Exception:
                risk = 0.0
            components[name] = round(risk, 4)
            weighted_risk += risk * weight
            total_weight += weight
            if risk > 0.5:
                flags.append(f"{name}: high risk ({risk:.2f})")

        if total_weight > 0:
            avg_risk = weighted_risk / total_weight
        else:
            avg_risk = 0.0

        # Convert risk (0-1) to score (100-0)
        value = max(0, min(100, int(100 * (1 - avg_risk))))

        return SafetyScore(
            value=value,
            grade=_grade(value),
            components=components,
            flags=flags,
        )

    def score_batch(
        self,
        items: list[tuple[str, dict[str, Any]]],
    ) -> list[SafetyScore]:
        """Score multiple items."""
        return [self.score(text, ctx) for text, ctx in items]
