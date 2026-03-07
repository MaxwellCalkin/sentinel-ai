"""Custom blocked terms scanner.

Detects enterprise-defined blocked terms, brand names, codenames,
or any custom strings that should not appear in LLM I/O.
"""

from __future__ import annotations

import re
from sentinel.core import Finding, RiskLevel


class BlockedTermsScanner:
    name = "blocked_terms"

    def __init__(
        self,
        terms: list[str] | None = None,
        risk: RiskLevel = RiskLevel.HIGH,
        case_sensitive: bool = False,
    ):
        self._terms = terms or []
        self._risk = risk
        self._case_sensitive = case_sensitive
        self._patterns = []
        for term in self._terms:
            flags = 0 if case_sensitive else re.IGNORECASE
            self._patterns.append(
                (term, re.compile(re.escape(term), flags))
            )

    def scan(self, text: str, context: dict | None = None) -> list[Finding]:
        findings: list[Finding] = []
        for term, pattern in self._patterns:
            for match in pattern.finditer(text):
                findings.append(
                    Finding(
                        scanner=self.name,
                        category="blocked_term",
                        description=f"Blocked term detected: '{term}'",
                        risk=self._risk,
                        span=(match.start(), match.end()),
                        metadata={"term": term},
                    )
                )
        return findings
