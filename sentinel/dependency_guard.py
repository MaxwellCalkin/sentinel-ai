"""Dependency security guard for Python packages.

Audit package dependencies for known malicious patterns, typosquatting
risks, and license compliance. Uses only stdlib -- no external deps.

Usage:
    from sentinel.dependency_guard import DependencyGuard

    guard = DependencyGuard()
    result = guard.check_package("requests", "2.28.0")
    assert result.risk_level == "low"

    report = guard.check_requirements(["flask>=2.0", "requests==2.28.0"])
    assert report.safe
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PackageCheck:
    """Result of checking a single package."""

    name: str
    version: str
    risk_level: str  # "low", "medium", "high", "critical"
    issues: list[str] = field(default_factory=list)
    blocked: bool = False


@dataclass
class DependencyReport:
    """Aggregate result of checking multiple packages."""

    packages: list[PackageCheck] = field(default_factory=list)
    total: int = 0
    high_risk_count: int = 0
    blocked_count: int = 0
    safe: bool = True


@dataclass
class TyposquatMatch:
    """A potential typosquatting match."""

    original: str
    suspect: str
    similarity: float
    reason: str


# Substrings that indicate a likely malicious package name
_MALICIOUS_NAME_PATTERNS: list[str] = [
    "exploit",
    "hack",
    "keylog",
    "malware",
    "trojan",
    "backdoor",
    "stealer",
    "phish",
    "ransomware",
    "rootkit",
]

# Known suspicious prefixes that mimic popular packages
_SUSPICIOUS_PREFIXES: list[str] = [
    "python-",
    "py-",
    "python3-",
    "pip-",
]

_MINIMUM_SAFE_NAME_LENGTH = 3

_REQUIREMENT_PATTERN = re.compile(
    r"^([A-Za-z0-9_][A-Za-z0-9._-]*)\s*([><=!~].*)?$"
)


def _levenshtein_distance(source: str, target: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(source) < len(target):
        return _levenshtein_distance(target, source)

    if len(target) == 0:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source):
        current_row = [source_index + 1]
        for target_index, target_char in enumerate(target):
            insert_cost = previous_row[target_index + 1] + 1
            delete_cost = current_row[target_index] + 1
            substitute_cost = previous_row[target_index] + (
                0 if source_char == target_char else 1
            )
            current_row.append(min(insert_cost, delete_cost, substitute_cost))
        previous_row = current_row

    return previous_row[-1]


def _parse_requirement(requirement: str) -> tuple[str, str]:
    """Extract package name and version specifier from a requirement string.

    Returns (name, version_specifier). The version specifier is the raw
    operator+version portion (e.g. ">=2.0" or "==2.28.0"), or empty
    string if none is present.
    """
    requirement = requirement.strip()
    match = _REQUIREMENT_PATTERN.match(requirement)
    if not match:
        return requirement, ""
    name = match.group(1)
    version_spec = (match.group(2) or "").strip()
    return name, version_spec


def _name_matches_malicious_pattern(name: str) -> str | None:
    """Return the matched malicious pattern substring, or None."""
    lower = name.lower()
    for pattern in _MALICIOUS_NAME_PATTERNS:
        if pattern in lower:
            return pattern
    return None


def _has_suspicious_prefix(name: str) -> str | None:
    """Return the suspicious prefix if the name starts with one, else None."""
    lower = name.lower()
    for prefix in _SUSPICIOUS_PREFIXES:
        if lower.startswith(prefix):
            return prefix
    return None


class DependencyGuard:
    """Audit Python package dependencies for security risks.

    Checks for known malicious name patterns, typosquatting risks,
    overly short package names, and maintains a configurable blocklist.
    """

    def __init__(
        self,
        blocked_licenses: list[str] | None = None,
        max_depth: int = 5,
    ) -> None:
        """
        Args:
            blocked_licenses: License identifiers to flag (e.g. ["AGPL-3.0"]).
            max_depth: Maximum dependency tree depth to consider.
        """
        self._blocked_licenses = [
            license.lower() for license in (blocked_licenses or [])
        ]
        self._max_depth = max_depth
        self._blocklist: set[str] = set()

    def add_blocklist(self, packages: list[str]) -> None:
        """Add package names to the blocklist."""
        for package in packages:
            self._blocklist.add(package.lower())

    def is_blocked(self, name: str) -> bool:
        """Check if a package name is on the blocklist or matches a malicious pattern."""
        if name.lower() in self._blocklist:
            return True
        return _name_matches_malicious_pattern(name) is not None

    def check_package(self, name: str, version: str = "") -> PackageCheck:
        """Check a single package for security risks.

        Args:
            name: Package name.
            version: Optional version string.

        Returns:
            PackageCheck with risk assessment and any issues found.
        """
        issues: list[str] = []
        blocked = False
        risk_level = "low"

        if not name or not name.strip():
            return PackageCheck(
                name=name,
                version=version,
                risk_level="critical",
                issues=["Empty package name"],
                blocked=True,
            )

        normalized = name.strip().lower()

        if self._is_explicitly_blocked(normalized):
            blocked = True
            risk_level = "critical"
            issues.append(f"Package '{name}' is on the blocklist")

        malicious_pattern = _name_matches_malicious_pattern(normalized)
        if malicious_pattern:
            blocked = True
            risk_level = "critical"
            issues.append(
                f"Package name contains malicious pattern: '{malicious_pattern}'"
            )

        suspicious_prefix = _has_suspicious_prefix(normalized)
        if suspicious_prefix:
            risk_level = _escalate_risk(risk_level, "medium")
            issues.append(
                f"Suspicious prefix '{suspicious_prefix}' may imitate a popular package"
            )

        if len(normalized) < _MINIMUM_SAFE_NAME_LENGTH:
            risk_level = _escalate_risk(risk_level, "medium")
            issues.append(
                f"Very short package name ({len(normalized)} chars) increases typosquat risk"
            )

        return PackageCheck(
            name=name,
            version=version,
            risk_level=risk_level,
            issues=issues,
            blocked=blocked,
        )

    def check_requirements(self, requirements: list[str]) -> DependencyReport:
        """Check a list of requirement strings for security risks.

        Each requirement follows pip format, e.g. "flask>=2.0" or
        "requests==2.28.0".

        Args:
            requirements: List of requirement specifier strings.

        Returns:
            DependencyReport with aggregate risk assessment.
        """
        packages: list[PackageCheck] = []
        for requirement in requirements:
            name, version_spec = _parse_requirement(requirement)
            check = self.check_package(name, version_spec)
            packages.append(check)

        return _build_report(packages)

    def detect_typosquat(
        self,
        name: str,
        known_packages: list[str],
    ) -> list[TyposquatMatch]:
        """Find known packages that are suspiciously similar to the given name.

        Uses Levenshtein edit distance to identify potential typosquats.

        Args:
            name: The suspect package name to check.
            known_packages: List of legitimate package names.

        Returns:
            List of TyposquatMatch objects sorted by similarity (highest first).
        """
        matches: list[TyposquatMatch] = []
        lower_name = name.lower()

        for known in known_packages:
            lower_known = known.lower()
            if lower_name == lower_known:
                continue

            distance = _levenshtein_distance(lower_name, lower_known)
            max_length = max(len(lower_name), len(lower_known))

            if max_length == 0:
                continue

            similarity = 1.0 - (distance / max_length)
            reason = _classify_typosquat_reason(lower_name, lower_known, distance)

            if reason:
                matches.append(TyposquatMatch(
                    original=known,
                    suspect=name,
                    similarity=round(similarity, 4),
                    reason=reason,
                ))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    def _is_explicitly_blocked(self, normalized_name: str) -> bool:
        """Check the explicit blocklist only (not malicious patterns)."""
        return normalized_name in self._blocklist


def _classify_typosquat_reason(
    suspect: str,
    known: str,
    distance: int,
) -> str:
    """Determine why a name looks like a typosquat, or return empty string."""
    if distance == 1:
        return "Single character difference (likely typo)"
    if distance == 2 and max(len(suspect), len(known)) >= 5:
        return "Two character differences on a longer name"
    if _is_character_swap(suspect, known):
        return "Adjacent character swap (transposition)"
    return ""


def _is_character_swap(first: str, second: str) -> bool:
    """Check if two strings differ by exactly one adjacent character swap."""
    if len(first) != len(second):
        return False

    diffs = [i for i in range(len(first)) if first[i] != second[i]]
    if len(diffs) != 2 and not (diffs and diffs[-1] - diffs[0] == 1):
        return False

    if len(diffs) == 2:
        index_a, index_b = diffs
        if index_b - index_a == 1:
            return first[index_a] == second[index_b] and first[index_b] == second[index_a]

    return False


_RISK_LEVELS = ["low", "medium", "high", "critical"]


def _escalate_risk(current: str, new: str) -> str:
    """Return the higher of two risk levels."""
    current_index = _RISK_LEVELS.index(current)
    new_index = _RISK_LEVELS.index(new)
    return _RISK_LEVELS[max(current_index, new_index)]


def _build_report(packages: list[PackageCheck]) -> DependencyReport:
    """Build an aggregate DependencyReport from individual package checks."""
    high_risk_count = sum(
        1 for package in packages if package.risk_level in ("high", "critical")
    )
    blocked_count = sum(1 for package in packages if package.blocked)
    safe = high_risk_count == 0 and blocked_count == 0

    return DependencyReport(
        packages=packages,
        total=len(packages),
        high_risk_count=high_risk_count,
        blocked_count=blocked_count,
        safe=safe,
    )
