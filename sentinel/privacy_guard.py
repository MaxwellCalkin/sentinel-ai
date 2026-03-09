"""Privacy policy enforcement for LLM interactions.

Enforce GDPR/CCPA compliance through data minimization,
purpose limitation, and retention controls on LLM inputs
and outputs.

Usage:
    from sentinel.privacy_guard import PrivacyGuard

    guard = PrivacyGuard(retention_days=90, purposes=["support", "analytics"])
    result = guard.check_input("Contact me at user@example.com", purpose="support")
    assert not result.compliant  # contains PII
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass


@dataclass
class PrivacyCheck:
    """Result of a privacy compliance check."""
    compliant: bool
    issues: list[str]
    sensitive_categories: list[str]
    purpose_allowed: bool


@dataclass
class MinimizationResult:
    """Result of data minimization."""
    original: str
    minimized: str
    items_removed: int
    categories_removed: list[str]


@dataclass
class PrivacyReport:
    """Privacy compliance summary."""
    checks_performed: int
    violations_found: int
    common_categories: list[tuple[str, int]]
    purposes_used: dict[str, int]
    compliance_rate: float


_BUILTIN_PATTERNS: list[tuple[str, str]] = [
    (r'\b\d{3}-\d{2}-\d{4}\b', "ssn"),
    (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', "credit_card"),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email"),
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "phone"),
]

_SECONDS_PER_DAY = 86400


class PrivacyGuard:
    """Enforce privacy policies on LLM interactions.

    Checks inputs and outputs for sensitive data, enforces
    purpose limitation, and tracks data retention windows.
    """

    def __init__(
        self,
        retention_days: int = 30,
        purposes: list[str] | None = None,
    ) -> None:
        self._retention_days = retention_days
        self._purposes: set[str] | None = set(purposes) if purposes is not None else None
        self._patterns: list[tuple[str, str]] = list(_BUILTIN_PATTERNS)
        self._checks_performed = 0
        self._violations_found = 0
        self._category_counter: Counter[str] = Counter()
        self._purpose_counter: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Purpose management
    # ------------------------------------------------------------------

    def add_purpose(self, purpose: str) -> None:
        """Add an allowed purpose."""
        if self._purposes is None:
            self._purposes = set()
        self._purposes.add(purpose)

    def is_purpose_allowed(self, purpose: str) -> bool:
        """Check if a purpose is permitted."""
        if self._purposes is None:
            return True
        return purpose in self._purposes

    # ------------------------------------------------------------------
    # Sensitive data detection
    # ------------------------------------------------------------------

    def _detect_sensitive(self, text: str) -> list[tuple[str, str]]:
        """Return list of (matched_text, category) for all sensitive hits."""
        hits: list[tuple[str, str]] = []
        for pattern, category in self._patterns:
            for match in re.finditer(pattern, text):
                hits.append((match.group(), category))
        return hits

    def _unique_categories(self, hits: list[tuple[str, str]]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for _, category in hits:
            if category not in seen:
                seen.add(category)
                result.append(category)
        return result

    # ------------------------------------------------------------------
    # Input / output checks
    # ------------------------------------------------------------------

    def check_input(self, text: str, purpose: str = "general") -> PrivacyCheck:
        """Check if input complies with privacy policy."""
        self._checks_performed += 1
        self._purpose_counter[purpose] += 1

        purpose_ok = self.is_purpose_allowed(purpose)
        hits = self._detect_sensitive(text)
        categories = self._unique_categories(hits)

        issues: list[str] = []
        if not purpose_ok:
            issues.append(f"Purpose '{purpose}' is not allowed")
        for matched_text, category in hits:
            issues.append(f"Sensitive data detected: {category}")

        compliant = len(issues) == 0
        if not compliant:
            self._violations_found += 1
        for category in categories:
            self._category_counter[category] += 1

        return PrivacyCheck(
            compliant=compliant,
            issues=issues,
            sensitive_categories=categories,
            purpose_allowed=purpose_ok,
        )

    def check_output(self, text: str) -> PrivacyCheck:
        """Check if output contains privacy-sensitive data."""
        self._checks_performed += 1

        hits = self._detect_sensitive(text)
        categories = self._unique_categories(hits)

        issues: list[str] = []
        for matched_text, category in hits:
            issues.append(f"Sensitive data in output: {category}")

        compliant = len(issues) == 0
        if not compliant:
            self._violations_found += 1
        for category in categories:
            self._category_counter[category] += 1

        return PrivacyCheck(
            compliant=compliant,
            issues=issues,
            sensitive_categories=categories,
            purpose_allowed=True,
        )

    # ------------------------------------------------------------------
    # Data minimization
    # ------------------------------------------------------------------

    def minimize(self, text: str) -> MinimizationResult:
        """Apply data minimization by replacing PII with placeholders."""
        minimized = text
        items_removed = 0
        categories_removed: list[str] = []

        for pattern, category in self._patterns:
            placeholder = f"[REDACTED_{category.upper()}]"
            new_text, count = re.subn(pattern, placeholder, minimized)
            if count > 0:
                minimized = new_text
                items_removed += count
                if category not in categories_removed:
                    categories_removed.append(category)

        return MinimizationResult(
            original=text,
            minimized=minimized,
            items_removed=items_removed,
            categories_removed=categories_removed,
        )

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def retention_check(self, created_at: float) -> bool:
        """Check if data is within retention period.

        Returns True if data should be kept, False if it should be deleted.
        """
        elapsed = time.time() - created_at
        return elapsed < self._retention_days * _SECONDS_PER_DAY

    # ------------------------------------------------------------------
    # Custom patterns
    # ------------------------------------------------------------------

    def add_sensitive_pattern(self, pattern: str, category: str) -> None:
        """Add a custom sensitive data pattern."""
        self._patterns.append((pattern, category))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> PrivacyReport:
        """Get privacy compliance summary."""
        if self._checks_performed > 0:
            compliance_rate = 1.0 - (self._violations_found / self._checks_performed)
        else:
            compliance_rate = 1.0

        common_categories = self._category_counter.most_common()

        return PrivacyReport(
            checks_performed=self._checks_performed,
            violations_found=self._violations_found,
            common_categories=common_categories,
            purposes_used=dict(self._purpose_counter),
            compliance_rate=compliance_rate,
        )

    def clear_stats(self) -> None:
        """Reset statistics."""
        self._checks_performed = 0
        self._violations_found = 0
        self._category_counter.clear()
        self._purpose_counter.clear()
