"""Toxicity and tone detection scanner.

Detects toxic language patterns: insults, threats, profanity,
and aggressive tone in LLM outputs. Uses pattern matching with
contextual scoring. For higher accuracy, use the ML-based
classifier (requires sentinel-ai[ml]).
"""

from __future__ import annotations

import re
from sentinel.core import Finding, RiskLevel


_THREAT_PATTERNS = re.compile(
    r"(?i)\b(i\s+will\s+(kill|hurt|destroy|end|ruin|murder)\s+you|"
    r"you('re|\s+are)\s+(dead|finished|done\s+for)|"
    r"(i'?ll|gonna|going\s+to)\s+(beat|attack|stab|shoot|strangle)\s+you|"
    r"watch\s+your\s+back|you\s+won'?t\s+survive|"
    r"(death|kill)\s+threat)\b"
)

_SEVERE_INSULTS = re.compile(
    r"(?i)\b(you\s+(stupid|worthless|pathetic|disgusting|piece\s+of\s+shit)|"
    r"(retard|retarded)\b|"
    r"(kill\s+yourself|kys)\b|"
    r"go\s+(die|fuck\s+yourself))\b"
)

_PROFANITY_HEAVY = re.compile(
    r"(?i)\b(f+u+c+k+|sh+i+t+|a+ss+h+o+le+|"
    r"motherf+|bullsh+i+t+|damn+i+t+|"
    r"b+i+t+c+h+|c+u+n+t+|d+i+c+k+h+e+a+d+)\b"
)

_AGGRESSIVE_TONE = re.compile(
    r"(?i)(!!+|[A-Z\s]{20,})"  # Multiple exclamation marks or ALL CAPS blocks
)


class ToxicityScanner:
    name = "toxicity"

    def __init__(self, profanity_risk: RiskLevel = RiskLevel.LOW):
        self._profanity_risk = profanity_risk

    def scan(self, text: str, context: dict | None = None) -> list[Finding]:
        findings: list[Finding] = []

        for match in _THREAT_PATTERNS.finditer(text):
            findings.append(
                Finding(
                    scanner=self.name,
                    category="toxicity",
                    description="Threat or violent language detected",
                    risk=RiskLevel.CRITICAL,
                    span=(match.start(), match.end()),
                    metadata={"toxicity_type": "threat"},
                )
            )

        for match in _SEVERE_INSULTS.finditer(text):
            findings.append(
                Finding(
                    scanner=self.name,
                    category="toxicity",
                    description="Severe insult or derogatory language",
                    risk=RiskLevel.HIGH,
                    span=(match.start(), match.end()),
                    metadata={"toxicity_type": "severe_insult"},
                )
            )

        for match in _PROFANITY_HEAVY.finditer(text):
            findings.append(
                Finding(
                    scanner=self.name,
                    category="toxicity",
                    description="Profanity detected",
                    risk=self._profanity_risk,
                    span=(match.start(), match.end()),
                    metadata={"toxicity_type": "profanity"},
                )
            )

        # Aggressive tone (ALL CAPS blocks)
        caps_matches = list(_AGGRESSIVE_TONE.finditer(text))
        if len(caps_matches) >= 2:
            findings.append(
                Finding(
                    scanner=self.name,
                    category="toxicity",
                    description="Aggressive tone detected (excessive caps/exclamation)",
                    risk=RiskLevel.LOW,
                    metadata={
                        "toxicity_type": "aggressive_tone",
                        "indicator_count": len(caps_matches),
                    },
                )
            )

        return findings
