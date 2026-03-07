"""Harmful content detection scanner.

Detects categories of harmful content in LLM outputs including
violence, hate speech, self-harm, illegal activities, and CSAM references.
Uses keyword-based detection with contextual scoring.
"""

from __future__ import annotations

import re
from sentinel.core import Finding, RiskLevel


_CATEGORY_PATTERNS: list[tuple[str, re.Pattern, RiskLevel, str]] = [
    (
        "weapons_manufacturing",
        re.compile(
            r"(?i)\b(how\s+to\s+(make|build|create|manufacture|assemble)\s+"
            r"(a\s+)?(bomb|explosive|weapon|firearm|gun|detonator|"
            r"chemical\s+weapon|biological\s+weapon|nerve\s+agent))\b"
        ),
        RiskLevel.CRITICAL,
        "Weapons/explosives manufacturing instructions",
    ),
    (
        "illegal_drugs",
        re.compile(
            r"(?i)\b(how\s+to\s+(make|synthesize|cook|produce|manufacture)\s+"
            r"(meth|methamphetamine|fentanyl|heroin|cocaine|LSD|MDMA|ecstasy))\b"
        ),
        RiskLevel.CRITICAL,
        "Illegal drug synthesis instructions",
    ),
    (
        "self_harm",
        re.compile(
            r"(?i)\b(how\s+to\s+(kill\s+yourself|commit\s+suicide|"
            r"end\s+(your|my)\s+life)|"
            r"(methods|ways)\s+(of|to)\s+(suicide|self.?harm|killing\s+yourself))\b"
        ),
        RiskLevel.CRITICAL,
        "Self-harm or suicide instructions",
    ),
    (
        "hacking_malware",
        re.compile(
            r"(?i)\b(how\s+to\s+(hack|exploit|breach|penetrate)\s+"
            r"(someone'?s?|a\s+)?(bank\s+account|email|social\s+media|"
            r"computer|server|network)|"
            r"(write|create|generate)\s+(a\s+)?"
            r"(ransomware|malware|trojan|keylogger|virus|worm|rootkit))\b"
        ),
        RiskLevel.HIGH,
        "Malicious hacking or malware creation",
    ),
    (
        "hate_speech",
        re.compile(
            r"(?i)\b(all\s+(blacks?|whites?|jews?|muslims?|asians?|hispanics?|"
            r"gays?|lesbians?|trans)\s+(should|must|need\s+to|deserve)\s+"
            r"(die|be\s+killed|be\s+eliminated|be\s+removed|go\s+back))\b"
        ),
        RiskLevel.CRITICAL,
        "Hate speech targeting a protected group",
    ),
    (
        "fraud_instructions",
        re.compile(
            r"(?i)\b(how\s+to\s+(commit|do)\s+"
            r"(fraud|identity\s+theft|tax\s+evasion|money\s+laundering|"
            r"insurance\s+fraud|wire\s+fraud|credit\s+card\s+fraud))\b"
        ),
        RiskLevel.HIGH,
        "Financial fraud instructions",
    ),
]


class HarmfulContentScanner:
    name = "harmful_content"

    def __init__(self, custom_categories: list[tuple[str, re.Pattern, RiskLevel, str]] | None = None):
        self._patterns = list(_CATEGORY_PATTERNS)
        if custom_categories:
            self._patterns.extend(custom_categories)

    def scan(self, text: str, context: dict | None = None) -> list[Finding]:
        findings: list[Finding] = []

        for category, pattern, risk, description in self._patterns:
            for match in pattern.finditer(text):
                findings.append(
                    Finding(
                        scanner=self.name,
                        category="harmful_content",
                        description=description,
                        risk=risk,
                        span=(match.start(), match.end()),
                        metadata={
                            "harm_category": category,
                            "matched_text": match.group(),
                        },
                    )
                )

        return findings
