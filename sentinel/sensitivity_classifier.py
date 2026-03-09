"""Sensitivity classifier for LLM prompts.

Classify prompts into sensitivity levels to determine
appropriate handling: public, internal, confidential, or
restricted. Used for access control and routing decisions.

Usage:
    from sentinel.sensitivity_classifier import SensitivityClassifier

    classifier = SensitivityClassifier()
    result = classifier.classify("What is the CEO's salary?")
    print(result.level)  # "confidential"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SensitivityResult:
    """Result of sensitivity classification."""
    text: str
    level: str  # "public", "internal", "confidential", "restricted"
    confidence: float
    triggers: list[str]
    recommended_actions: list[str]


# Sensitivity patterns by level
_RESTRICTED_PATTERNS = [
    (r'(?i)\b(?:social\s+security|ssn|tax\s+id)\b', "SSN/Tax ID reference"),
    (r'(?i)\b(?:credit\s+card|card\s+number|cvv|expir(?:y|ation)\s+date)\b', "Payment card data"),
    (r'(?i)\b(?:medical\s+record|diagnosis|prescription|patient)\b', "Medical/health data"),
    (r'(?i)\b(?:classified|top\s+secret|security\s+clearance)\b', "Classified information"),
    (r'(?i)\b(?:encryption\s+key|private\s+key|master\s+key)\b', "Cryptographic material"),
]

_CONFIDENTIAL_PATTERNS = [
    (r'(?i)\b(?:salary|compensation|bonus|equity|stock\s+option)\b', "Compensation data"),
    (r'(?i)\b(?:revenue|profit|financial\s+(?:report|statement))\b', "Financial data"),
    (r'(?i)\b(?:password|credential|auth(?:entication)?\s+token)\b', "Authentication data"),
    (r'(?i)\b(?:trade\s+secret|proprietary|intellectual\s+property)\b', "IP/trade secrets"),
    (r'(?i)\b(?:merger|acquisition|due\s+diligence|term\s+sheet)\b', "M&A data"),
    (r'(?i)\b(?:employee\s+(?:record|review|performance))\b', "HR data"),
]

_INTERNAL_PATTERNS = [
    (r'(?i)\b(?:internal\s+(?:memo|document|policy|meeting))\b', "Internal document"),
    (r'(?i)\b(?:roadmap|sprint|backlog|release\s+plan)\b', "Product planning"),
    (r'(?i)\b(?:org\s+chart|team\s+structure|headcount)\b', "Organizational info"),
    (r'(?i)\b(?:vendor|supplier|contract|agreement)\b', "Business relationship"),
]


class SensitivityClassifier:
    """Classify prompt sensitivity for access control."""

    def __init__(
        self,
        custom_patterns: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        self._patterns: dict[str, list[tuple[str, str]]] = {
            "restricted": list(_RESTRICTED_PATTERNS),
            "confidential": list(_CONFIDENTIAL_PATTERNS),
            "internal": list(_INTERNAL_PATTERNS),
        }
        if custom_patterns:
            for level, patterns in custom_patterns.items():
                if level not in self._patterns:
                    self._patterns[level] = []
                self._patterns[level].extend(patterns)

    def classify(self, text: str) -> SensitivityResult:
        """Classify text sensitivity level."""
        triggers: list[str] = []
        level_hits: dict[str, int] = {"restricted": 0, "confidential": 0, "internal": 0}

        for level in ["restricted", "confidential", "internal"]:
            for pattern, desc in self._patterns.get(level, []):
                if re.search(pattern, text):
                    level_hits[level] += 1
                    triggers.append(f"{level}: {desc}")

        # Determine level (highest sensitivity wins)
        if level_hits["restricted"] > 0:
            level = "restricted"
        elif level_hits["confidential"] > 0:
            level = "confidential"
        elif level_hits["internal"] > 0:
            level = "internal"
        else:
            level = "public"

        # Confidence based on number of triggers
        total_triggers = sum(level_hits.values())
        confidence = min(1.0, total_triggers * 0.3) if total_triggers > 0 else 1.0

        # Recommended actions
        actions = self._get_actions(level)

        return SensitivityResult(
            text=text, level=level, confidence=round(confidence, 4),
            triggers=triggers, recommended_actions=actions,
        )

    def classify_batch(self, texts: list[str]) -> list[SensitivityResult]:
        """Classify multiple texts."""
        return [self.classify(t) for t in texts]

    def _get_actions(self, level: str) -> list[str]:
        actions = {
            "restricted": ["Require explicit authorization", "Log access attempt", "Encrypt in transit", "Apply data masking"],
            "confidential": ["Verify user permissions", "Log access", "Limit response detail"],
            "internal": ["Verify internal user", "Standard logging"],
            "public": ["No restrictions"],
        }
        return actions.get(level, [])
