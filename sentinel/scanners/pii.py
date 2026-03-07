"""PII (Personally Identifiable Information) detection and redaction scanner.

Detects emails, phone numbers, SSNs, credit cards, IP addresses, and other
PII in LLM inputs and outputs.
"""

from __future__ import annotations

import re
from sentinel.core import Finding, RiskLevel


_PII_PATTERNS: list[tuple[str, re.Pattern, RiskLevel, str]] = [
    (
        "EMAIL",
        re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        RiskLevel.MEDIUM,
        "Email address detected",
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        RiskLevel.CRITICAL,
        "US Social Security Number detected",
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        RiskLevel.CRITICAL,
        "Possible credit card number detected",
    ),
    (
        "PHONE_US",
        re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        RiskLevel.MEDIUM,
        "US phone number detected",
    ),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        RiskLevel.LOW,
        "IP address detected",
    ),
    (
        "DATE_OF_BIRTH",
        re.compile(
            r"(?i)\b(?:date\s+of\s+birth|dob|born\s+on)\s*:?\s*"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        ),
        RiskLevel.HIGH,
        "Date of birth detected",
    ),
    (
        "PASSPORT",
        re.compile(r"(?i)\bpassport\s*(?:#|number|no\.?)\s*:?\s*[A-Z0-9]{6,9}\b"),
        RiskLevel.CRITICAL,
        "Passport number detected",
    ),
    (
        "API_KEY",
        re.compile(
            r"\b(?:sk-[a-zA-Z0-9]{20,}|"
            r"AIza[a-zA-Z0-9_-]{35}|"
            r"AKIA[A-Z0-9]{16}|"
            r"ghp_[a-zA-Z0-9]{36}|"
            r"xox[bpras]-[a-zA-Z0-9-]+)\b"
        ),
        RiskLevel.CRITICAL,
        "API key or secret token detected",
    ),
]

# Luhn algorithm for credit card validation
def _luhn_check(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


class PIIScanner:
    name = "pii"

    def __init__(self, extra_patterns: list[tuple[str, re.Pattern, RiskLevel, str]] | None = None):
        self._patterns = list(_PII_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def scan(self, text: str, context: dict | None = None) -> list[Finding]:
        findings: list[Finding] = []

        for pii_type, pattern, risk, description in self._patterns:
            for match in pattern.finditer(text):
                # Extra validation for credit cards
                if pii_type == "CREDIT_CARD":
                    if not _luhn_check(match.group()):
                        continue

                findings.append(
                    Finding(
                        scanner=self.name,
                        category="pii",
                        description=description,
                        risk=risk,
                        span=(match.start(), match.end()),
                        metadata={"pii_type": pii_type},
                    )
                )

        return findings
