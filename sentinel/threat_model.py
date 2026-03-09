"""STRIDE-based threat modeling for AI/LLM systems.

Helps teams identify, categorize, and mitigate security threats
using the STRIDE framework (Spoofing, Tampering, Repudiation,
Information Disclosure, Denial of Service, Elevation of Privilege).

Usage:
    from sentinel.threat_model import ThreatModel, ThreatAnalysis

    model = ThreatModel("chatbot-api", description="Customer support LLM")
    model.add_component("llm", component_type="model", trust_level="internal")
    model.add_component("web-ui", component_type="user_interface", trust_level="external")
    model.add_threat("llm", "tampering", "Prompt injection via user input", severity="high")
    model.add_threat("web-ui", "spoofing", "Session hijacking", mitigation="Use secure cookies")
    analysis = model.analyze()
    print(analysis.risk_score)
"""

from __future__ import annotations

from dataclasses import dataclass, field


STRIDE_CATEGORIES: frozenset[str] = frozenset({
    "spoofing",
    "tampering",
    "repudiation",
    "information_disclosure",
    "denial_of_service",
    "elevation_of_privilege",
})

VALID_SEVERITIES: frozenset[str] = frozenset({
    "low",
    "medium",
    "high",
    "critical",
})

VALID_COMPONENT_TYPES: frozenset[str] = frozenset({
    "service",
    "database",
    "api",
    "model",
    "user_interface",
})

VALID_TRUST_LEVELS: frozenset[str] = frozenset({
    "external",
    "boundary",
    "internal",
})

_SEVERITY_WEIGHTS: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


@dataclass
class Component:
    """A system component within the threat model."""

    name: str
    component_type: str
    trust_level: str


@dataclass
class Threat:
    """A specific security threat against a component."""

    component: str
    category: str
    description: str
    severity: str
    mitigation: str
    mitigated: bool


@dataclass
class ThreatAnalysis:
    """Result of analyzing a complete threat model."""

    system_name: str
    components: list[Component]
    threats: list[Threat]
    total_threats: int
    mitigated_count: int
    unmitigated_count: int
    by_category: dict[str, int]
    by_severity: dict[str, int]
    risk_score: float


def _validate_category(category: str) -> None:
    if category not in STRIDE_CATEGORIES:
        raise ValueError(
            f"Invalid STRIDE category '{category}'. "
            f"Must be one of: {', '.join(sorted(STRIDE_CATEGORIES))}"
        )


def _validate_severity(severity: str) -> None:
    if severity not in VALID_SEVERITIES:
        raise ValueError(
            f"Invalid severity '{severity}'. "
            f"Must be one of: {', '.join(sorted(VALID_SEVERITIES))}"
        )


def _validate_component_type(component_type: str) -> None:
    if component_type not in VALID_COMPONENT_TYPES:
        raise ValueError(
            f"Invalid component type '{component_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_COMPONENT_TYPES))}"
        )


def _validate_trust_level(trust_level: str) -> None:
    if trust_level not in VALID_TRUST_LEVELS:
        raise ValueError(
            f"Invalid trust level '{trust_level}'. "
            f"Must be one of: {', '.join(sorted(VALID_TRUST_LEVELS))}"
        )


def _compute_risk_score(threats: list[Threat]) -> float:
    """Weighted sum of unmitigated threats, normalized by max possible score."""
    if not threats:
        return 0.0
    max_possible = len(threats) * 4
    unmitigated_weight = sum(
        _SEVERITY_WEIGHTS[threat.severity]
        for threat in threats
        if not threat.mitigated
    )
    return round(unmitigated_weight / max_possible, 4)


def _count_by_category(threats: list[Threat]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for threat in threats:
        counts[threat.category] = counts.get(threat.category, 0) + 1
    return counts


def _count_by_severity(threats: list[Threat]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for threat in threats:
        counts[threat.severity] = counts.get(threat.severity, 0) + 1
    return counts


class ThreatModel:
    """STRIDE-based threat modeling for AI/LLM systems.

    Provides structured threat identification, categorization,
    mitigation tracking, and risk analysis using the STRIDE
    framework adapted for AI security concerns.
    """

    def __init__(self, system_name: str, description: str = "") -> None:
        self._system_name = system_name
        self._description = description
        self._components: dict[str, Component] = {}
        self._threats: list[Threat] = []

    @property
    def system_name(self) -> str:
        return self._system_name

    @property
    def description(self) -> str:
        return self._description

    def add_component(
        self,
        name: str,
        component_type: str = "service",
        trust_level: str = "internal",
    ) -> None:
        """Register a system component.

        Args:
            name: Unique component identifier.
            component_type: One of service, database, api, model, user_interface.
            trust_level: One of external, boundary, internal.

        Raises:
            ValueError: If component_type or trust_level is invalid.
        """
        _validate_component_type(component_type)
        _validate_trust_level(trust_level)
        self._components[name] = Component(
            name=name,
            component_type=component_type,
            trust_level=trust_level,
        )

    def add_threat(
        self,
        component: str,
        category: str,
        description: str,
        severity: str = "medium",
        mitigation: str = "",
    ) -> None:
        """Record a threat against a component.

        Args:
            component: Name of the target component (must already exist).
            category: STRIDE category for this threat.
            description: Human-readable threat description.
            severity: One of low, medium, high, critical.
            mitigation: Optional mitigation strategy.

        Raises:
            KeyError: If the component has not been added.
            ValueError: If category or severity is invalid.
        """
        if component not in self._components:
            raise KeyError(
                f"Component '{component}' not found. "
                f"Add it with add_component() first."
            )
        _validate_category(category)
        _validate_severity(severity)
        self._threats.append(Threat(
            component=component,
            category=category,
            description=description,
            severity=severity,
            mitigation=mitigation,
            mitigated=bool(mitigation),
        ))

    def add_mitigation(self, threat_index: int, mitigation: str) -> None:
        """Apply a mitigation to an existing threat.

        Args:
            threat_index: Zero-based index into the threats list.
            mitigation: Mitigation strategy text.

        Raises:
            IndexError: If threat_index is out of range.
        """
        threat = self._threats[threat_index]
        self._threats[threat_index] = Threat(
            component=threat.component,
            category=threat.category,
            description=threat.description,
            severity=threat.severity,
            mitigation=mitigation,
            mitigated=bool(mitigation),
        )

    def get_threats(
        self,
        component: str | None = None,
        category: str | None = None,
    ) -> list[Threat]:
        """Filter threats by component and/or category.

        Args:
            component: If provided, only return threats for this component.
            category: If provided, only return threats in this STRIDE category.

        Returns:
            List of matching Threat objects.
        """
        results = self._threats
        if component is not None:
            results = [t for t in results if t.component == component]
        if category is not None:
            results = [t for t in results if t.category == category]
        return results

    def analyze(self) -> ThreatAnalysis:
        """Produce a full threat analysis of the system.

        Returns:
            ThreatAnalysis with counts, breakdowns, and risk score.
        """
        mitigated_count = sum(1 for t in self._threats if t.mitigated)
        unmitigated_count = len(self._threats) - mitigated_count
        return ThreatAnalysis(
            system_name=self._system_name,
            components=list(self._components.values()),
            threats=list(self._threats),
            total_threats=len(self._threats),
            mitigated_count=mitigated_count,
            unmitigated_count=unmitigated_count,
            by_category=_count_by_category(self._threats),
            by_severity=_count_by_severity(self._threats),
            risk_score=_compute_risk_score(self._threats),
        )

    def risk_matrix(self) -> dict[str, dict[str, int]]:
        """Build a severity-by-category count matrix.

        Returns:
            Nested dict mapping severity -> category -> count.
        """
        matrix: dict[str, dict[str, int]] = {
            severity: {category: 0 for category in sorted(STRIDE_CATEGORIES)}
            for severity in sorted(VALID_SEVERITIES)
        }
        for threat in self._threats:
            matrix[threat.severity][threat.category] += 1
        return matrix

    def export(self) -> dict:
        """Export the threat model as a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "system_name": self._system_name,
            "description": self._description,
            "components": [
                {
                    "name": comp.name,
                    "component_type": comp.component_type,
                    "trust_level": comp.trust_level,
                }
                for comp in self._components.values()
            ],
            "threats": [
                {
                    "component": threat.component,
                    "category": threat.category,
                    "description": threat.description,
                    "severity": threat.severity,
                    "mitigation": threat.mitigation,
                    "mitigated": threat.mitigated,
                }
                for threat in self._threats
            ],
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the threat model.

        Returns:
            Multi-line summary string.
        """
        analysis = self.analyze()
        lines = [
            f"Threat Model: {self._system_name}",
        ]
        if self._description:
            lines.append(f"Description: {self._description}")
        lines.append(f"Components: {len(analysis.components)}")
        lines.append(f"Total threats: {analysis.total_threats}")
        lines.append(f"Mitigated: {analysis.mitigated_count}")
        lines.append(f"Unmitigated: {analysis.unmitigated_count}")
        lines.append(f"Risk score: {analysis.risk_score}")
        if analysis.by_category:
            category_parts = [
                f"{cat}={count}"
                for cat, count in sorted(analysis.by_category.items())
            ]
            lines.append(f"By category: {', '.join(category_parts)}")
        if analysis.by_severity:
            severity_parts = [
                f"{sev}={count}"
                for sev, count in sorted(analysis.by_severity.items())
            ]
            lines.append(f"By severity: {', '.join(severity_parts)}")
        return "\n".join(lines)
