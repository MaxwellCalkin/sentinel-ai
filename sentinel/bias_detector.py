"""Bias detection in LLM outputs.

Detects potential biases in LLM responses including gender bias,
racial bias, age bias, and other forms of discrimination.

Usage:
    from sentinel.bias_detector import BiasDetector

    detector = BiasDetector()
    result = detector.check("Women are not good at math.")
    assert result.biased
    assert "gender" in [b.category for b in result.biases]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BiasInstance:
    """A detected bias instance."""
    category: str        # gender, racial, age, disability, religion, etc.
    text: str            # The biased text
    severity: float      # 0.0 to 1.0
    explanation: str = ""


@dataclass
class BiasResult:
    """Result of bias detection."""
    text: str
    biased: bool
    biases: list[BiasInstance] = field(default_factory=list)
    overall_score: float = 0.0  # 0.0 = unbiased, 1.0 = highly biased

    @property
    def bias_count(self) -> int:
        return len(self.biases)

    @property
    def categories(self) -> list[str]:
        return list(set(b.category for b in self.biases))


# Bias patterns organized by category
_BIAS_PATTERNS: dict[str, list[tuple[re.Pattern, float, str]]] = {
    "gender": [
        (re.compile(r"\b(?:women|girls|females?) (?:are|can't|cannot|shouldn't|don't|lack|fail)\b", re.I), 0.8,
         "Gender-based capability generalization"),
        (re.compile(r"\b(?:men|boys|males?) (?:are|should) (?:always|naturally|inherently)\b", re.I), 0.7,
         "Gender essentialism"),
        (re.compile(r"\b(?:women|girls) (?:belong|should stay|should be)\b", re.I), 0.9,
         "Prescriptive gender role"),
        (re.compile(r"\b(?:man|men|male)(?:'s)? (?:job|work|role|domain)\b", re.I), 0.7,
         "Gendered role assignment"),
        (re.compile(r"\b(?:typical|like a) (?:woman|girl|man|boy)\b", re.I), 0.5,
         "Gender stereotyping"),
    ],
    "racial": [
        (re.compile(r"\b(?:all|most|typical)\s+(?:black|white|asian|hispanic|latino|arab)\s+(?:people|men|women|folks)\b", re.I), 0.8,
         "Racial generalization"),
        (re.compile(r"\b(?:race|ethnic)\s+(?:is|are)\s+(?:superior|inferior|better|worse)\b", re.I), 0.95,
         "Racial supremacy claim"),
        (re.compile(r"\b(?:those|these|their)\s+(?:people|kind|type|sort)\b", re.I), 0.4,
         "Othering language"),
    ],
    "age": [
        (re.compile(r"\b(?:old|elderly) people (?:are|can't|cannot|shouldn't|don't)\b", re.I), 0.7,
         "Age-based capability generalization"),
        (re.compile(r"\b(?:young|millennials?|boomers?|gen[- ]?z) (?:are|always|never)\b", re.I), 0.6,
         "Generational stereotyping"),
        (re.compile(r"\btoo old (?:to|for)\b", re.I), 0.5,
         "Age discrimination"),
    ],
    "disability": [
        (re.compile(r"\b(?:disabled|handicapped) people (?:can't|cannot|shouldn't|are unable)\b", re.I), 0.7,
         "Disability-based capability generalization"),
        (re.compile(r"\b(?:suffer from|afflicted|victim of)\s+(?:a |)\w+(?:ism|ity|ness|ia)\b", re.I), 0.4,
         "Victimizing disability language"),
        (re.compile(r"\b(?:crazy|insane|psycho|retard|lame|dumb)\b", re.I), 0.5,
         "Ableist language"),
    ],
    "religion": [
        (re.compile(r"\b(?:all|most|typical)\s+(?:muslims?|christians?|jews?|hindus?|buddhists?)\s+(?:are|believe|want)\b", re.I), 0.7,
         "Religious generalization"),
    ],
    "socioeconomic": [
        (re.compile(r"\b(?:poor|lower class|uneducated) people (?:are|can't|don't|lack)\b", re.I), 0.7,
         "Class-based generalization"),
        (re.compile(r"\b(?:welfare|homeless) (?:people|folks) (?:are|should|deserve)\b", re.I), 0.6,
         "Socioeconomic stereotyping"),
    ],
}


class BiasDetector:
    """Detect bias in LLM outputs.

    Uses pattern matching to identify gender, racial, age,
    disability, religious, and socioeconomic biases.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        categories: list[str] | None = None,
    ) -> None:
        """
        Args:
            threshold: Minimum severity to report a bias.
            categories: Which categories to check (None = all).
        """
        self._threshold = threshold
        self._categories = set(categories) if categories else None

    def check(self, text: str) -> BiasResult:
        """Check text for bias.

        Returns:
            BiasResult with detected biases and overall score.
        """
        biases: list[BiasInstance] = []

        for category, patterns in _BIAS_PATTERNS.items():
            if self._categories and category not in self._categories:
                continue

            for pattern, severity, explanation in patterns:
                match = pattern.search(text)
                if match and severity >= self._threshold:
                    biases.append(BiasInstance(
                        category=category,
                        text=match.group(),
                        severity=severity,
                        explanation=explanation,
                    ))

        overall = max((b.severity for b in biases), default=0.0)

        return BiasResult(
            text=text,
            biased=len(biases) > 0,
            biases=biases,
            overall_score=round(overall, 2),
        )

    def check_batch(self, texts: list[str]) -> list[BiasResult]:
        """Check multiple texts for bias."""
        return [self.check(t) for t in texts]
