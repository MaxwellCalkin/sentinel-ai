"""Hallucination signal detection scanner.

Detects linguistic markers correlated with LLM hallucinations:
hedging language, false confidence, fabricated citations,
and self-contradictions.
"""

from __future__ import annotations

import re
from sentinel.core import Finding, RiskLevel


_HEDGING_PHRASES = re.compile(
    r"(?i)\b(I\s+think|I\s+believe|I'm\s+not\s+(entirely\s+)?sure|"
    r"if\s+I\s+recall\s+correctly|to\s+the\s+best\s+of\s+my\s+knowledge|"
    r"I\s+may\s+be\s+wrong|don'?t\s+quote\s+me\s+on\s+this|"
    r"approximately|roughly|around\s+\d|sometime\s+around|"
    r"it'?s\s+possible\s+that|there'?s\s+a\s+chance)\b"
)

_FALSE_CONFIDENCE = re.compile(
    r"(?i)\b(it\s+is\s+(a\s+)?well[\s-]known\s+fact|"
    r"as\s+everyone\s+knows|it\s+is\s+widely\s+accepted|"
    r"studies\s+have\s+(conclusively\s+)?shown|"
    r"research\s+proves|experts\s+agree|"
    r"it\s+has\s+been\s+proven|science\s+has\s+shown)\b"
)

_FABRICATED_CITATION = re.compile(
    r"(?i)(according\s+to\s+a\s+(\d{4}\s+)?(study|paper|report|survey|analysis)\s+"
    r"(published\s+)?in\s+|"
    r"(Dr\.?|Professor)\s+[A-Z][a-z]+\s+(et\s+al\.?\s+)?"
    r"\(\d{4}\)\s+(found|showed|demonstrated|concluded)|"
    r"as\s+reported\s+by\s+[A-Z][a-z]+\s+\(\d{4}\))"
)

_CONTRADICTION_MARKERS = re.compile(
    r"(?i)(however,?\s+(contrary\s+to\s+)?what\s+I\s+(just\s+)?said|"
    r"actually,?\s+(that'?s|I\s+was)\s+(?:not\s+(?:quite\s+)?right|wrong|incorrect)|"
    r"wait,?\s+(?:let\s+me\s+correct|I\s+made\s+an?\s+error)|"
    r"correction:|I\s+need\s+to\s+correct\s+myself)"
)


class HallucinationScanner:
    name = "hallucination"

    def __init__(self, hedging_threshold: int = 3):
        self._hedging_threshold = hedging_threshold

    def scan(self, text: str, context: dict | None = None) -> list[Finding]:
        findings: list[Finding] = []

        # Detect fabricated citations (highest signal)
        for match in _FABRICATED_CITATION.finditer(text):
            findings.append(
                Finding(
                    scanner=self.name,
                    category="hallucination",
                    description="Possible fabricated citation or reference",
                    risk=RiskLevel.MEDIUM,
                    span=(match.start(), match.end()),
                    metadata={"signal": "fabricated_citation"},
                )
            )

        # Detect self-contradictions
        for match in _CONTRADICTION_MARKERS.finditer(text):
            findings.append(
                Finding(
                    scanner=self.name,
                    category="hallucination",
                    description="Self-contradiction detected, possible hallucination correction",
                    risk=RiskLevel.MEDIUM,
                    span=(match.start(), match.end()),
                    metadata={"signal": "self_contradiction"},
                )
            )

        # Detect false confidence markers
        for match in _FALSE_CONFIDENCE.finditer(text):
            findings.append(
                Finding(
                    scanner=self.name,
                    category="hallucination",
                    description="False confidence marker (may indicate confabulation)",
                    risk=RiskLevel.LOW,
                    span=(match.start(), match.end()),
                    metadata={"signal": "false_confidence"},
                )
            )

        # Excessive hedging = uncertainty = higher hallucination risk
        hedges = list(_HEDGING_PHRASES.finditer(text))
        if len(hedges) >= self._hedging_threshold:
            findings.append(
                Finding(
                    scanner=self.name,
                    category="hallucination",
                    description=f"High hedging density ({len(hedges)} instances) suggests low confidence",
                    risk=RiskLevel.LOW,
                    metadata={
                        "signal": "excessive_hedging",
                        "hedge_count": len(hedges),
                    },
                )
            )

        return findings
