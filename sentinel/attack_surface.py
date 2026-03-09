"""Attack surface mapping and analysis for AI/LLM applications.

Identifies exposed endpoints, data flows, and potential entry points
to help teams understand and reduce their application's attack surface.

Usage:
    from sentinel.attack_surface import AttackSurface, SurfaceAnalysis

    surface = AttackSurface("chatbot-api")
    surface.add_endpoint("llm-api", endpoint_type="api", exposure="public")
    surface.add_endpoint("db", endpoint_type="api", exposure="internal")
    surface.add_data_flow("llm-api", "db", data_type="user_input")
    surface.add_vulnerability("llm-api", "high", "No rate limiting", cwe="CWE-770")
    analysis = surface.analyze()
    print(analysis.risk_score)
"""

from __future__ import annotations

from dataclasses import dataclass, field


VALID_ENDPOINT_TYPES: frozenset[str] = frozenset({
    "api",
    "websocket",
    "webhook",
    "grpc",
    "ui",
})

VALID_EXPOSURES: frozenset[str] = frozenset({
    "public",
    "partner",
    "internal",
})

VALID_SEVERITIES: frozenset[str] = frozenset({
    "low",
    "medium",
    "high",
    "critical",
})

_SEVERITY_RISK_WEIGHTS: dict[str, float] = {
    "low": 0.05,
    "medium": 0.1,
    "high": 0.2,
    "critical": 0.3,
}

_EXPOSED_LEVELS: frozenset[str] = frozenset({"public", "partner"})


@dataclass
class Endpoint:
    """An application endpoint in the attack surface."""

    name: str
    endpoint_type: str
    exposure: str
    auth_required: bool
    description: str


@dataclass
class DataFlow:
    """A data flow between two endpoints."""

    source: str
    destination: str
    data_type: str
    encrypted: bool


@dataclass
class Vulnerability:
    """A known vulnerability associated with an endpoint."""

    endpoint: str
    severity: str
    description: str
    cwe: str


@dataclass
class SurfaceAnalysis:
    """Result of analyzing the full attack surface."""

    application: str
    total_endpoints: int
    public_count: int
    unauthenticated_count: int
    vulnerability_count: int
    critical_vulns: int
    unencrypted_flows: int
    risk_score: float
    risk_level: str
    recommendations: list[str]


def _validate_endpoint_type(endpoint_type: str) -> None:
    if endpoint_type not in VALID_ENDPOINT_TYPES:
        raise ValueError(
            f"Invalid endpoint type '{endpoint_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_ENDPOINT_TYPES))}"
        )


def _validate_exposure(exposure: str) -> None:
    if exposure not in VALID_EXPOSURES:
        raise ValueError(
            f"Invalid exposure '{exposure}'. "
            f"Must be one of: {', '.join(sorted(VALID_EXPOSURES))}"
        )


def _validate_severity(severity: str) -> None:
    if severity not in VALID_SEVERITIES:
        raise ValueError(
            f"Invalid severity '{severity}'. "
            f"Must be one of: {', '.join(sorted(VALID_SEVERITIES))}"
        )


def _require_endpoint_exists(
    endpoints: dict[str, Endpoint],
    name: str,
) -> None:
    if name not in endpoints:
        raise KeyError(
            f"Endpoint '{name}' not found. "
            f"Add it with add_endpoint() first."
        )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _compute_risk_score(
    endpoints: list[Endpoint],
    flows: list[DataFlow],
    vulnerabilities: list[Vulnerability],
) -> float:
    score = 0.0
    for endpoint in endpoints:
        if endpoint.exposure in _EXPOSED_LEVELS:
            score += 0.1
        if not endpoint.auth_required:
            score += 0.15
    for flow in flows:
        if not flow.encrypted:
            score += 0.1
    for vulnerability in vulnerabilities:
        score += _SEVERITY_RISK_WEIGHTS[vulnerability.severity]
    return _clamp(score)


def _classify_risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.6:
        return "medium"
    if score < 0.8:
        return "high"
    return "critical"


def _generate_recommendations(
    unauthenticated: list[Endpoint],
    unencrypted: list[DataFlow],
    vulnerabilities: list[Vulnerability],
    exposed: list[Endpoint],
) -> list[str]:
    recommendations: list[str] = []
    if unauthenticated:
        count = len(unauthenticated)
        recommendations.append(
            f"Enable authentication on {count} endpoint"
            f"{'s' if count != 1 else ''}"
        )
    if unencrypted:
        count = len(unencrypted)
        recommendations.append(
            f"Encrypt {count} data flow"
            f"{'s' if count != 1 else ''}"
        )
    critical_vulns = [v for v in vulnerabilities if v.severity == "critical"]
    if critical_vulns:
        count = len(critical_vulns)
        recommendations.append(
            f"Address {count} critical vulnerabilit"
            f"{'ies' if count != 1 else 'y'}"
        )
    high_vulns = [v for v in vulnerabilities if v.severity == "high"]
    if high_vulns:
        count = len(high_vulns)
        recommendations.append(
            f"Address {count} high-severity vulnerabilit"
            f"{'ies' if count != 1 else 'y'}"
        )
    if exposed:
        count = len(exposed)
        recommendations.append(
            f"Review {count} publicly exposed endpoint"
            f"{'s' if count != 1 else ''}"
        )
    return recommendations


class AttackSurface:
    """Map and analyze the attack surface of an AI/LLM application.

    Tracks endpoints, data flows, and known vulnerabilities to produce
    a risk assessment with actionable recommendations.
    """

    def __init__(self, application_name: str) -> None:
        self._application_name = application_name
        self._endpoints: dict[str, Endpoint] = {}
        self._flows: list[DataFlow] = []
        self._vulnerabilities: list[Vulnerability] = []

    @property
    def application_name(self) -> str:
        return self._application_name

    def add_endpoint(
        self,
        name: str,
        endpoint_type: str,
        exposure: str = "internal",
        auth_required: bool = True,
        description: str = "",
    ) -> None:
        """Register an endpoint in the attack surface.

        Args:
            name: Unique endpoint identifier.
            endpoint_type: One of api, websocket, webhook, grpc, ui.
            exposure: One of public, partner, internal.
            auth_required: Whether authentication is required.
            description: Human-readable description.

        Raises:
            ValueError: If endpoint_type or exposure is invalid.
        """
        _validate_endpoint_type(endpoint_type)
        _validate_exposure(exposure)
        self._endpoints[name] = Endpoint(
            name=name,
            endpoint_type=endpoint_type,
            exposure=exposure,
            auth_required=auth_required,
            description=description,
        )

    def add_data_flow(
        self,
        source: str,
        destination: str,
        data_type: str,
        encrypted: bool = True,
    ) -> None:
        """Register a data flow between two endpoints.

        Args:
            source: Name of the source endpoint (must exist).
            destination: Name of the destination endpoint (must exist).
            data_type: Category of data being transferred.
            encrypted: Whether the flow is encrypted in transit.

        Raises:
            KeyError: If source or destination endpoint does not exist.
        """
        _require_endpoint_exists(self._endpoints, source)
        _require_endpoint_exists(self._endpoints, destination)
        self._flows.append(DataFlow(
            source=source,
            destination=destination,
            data_type=data_type,
            encrypted=encrypted,
        ))

    def add_vulnerability(
        self,
        endpoint: str,
        severity: str,
        description: str,
        cwe: str = "",
    ) -> None:
        """Record a known vulnerability on an endpoint.

        Args:
            endpoint: Name of the affected endpoint (must exist).
            severity: One of low, medium, high, critical.
            description: Human-readable vulnerability description.
            cwe: Optional CWE identifier (e.g. "CWE-770").

        Raises:
            KeyError: If the endpoint does not exist.
            ValueError: If severity is invalid.
        """
        _require_endpoint_exists(self._endpoints, endpoint)
        _validate_severity(severity)
        self._vulnerabilities.append(Vulnerability(
            endpoint=endpoint,
            severity=severity,
            description=description,
            cwe=cwe,
        ))

    def analyze(self) -> SurfaceAnalysis:
        """Analyze the attack surface and produce a full report.

        Returns:
            SurfaceAnalysis with counts, risk score, and recommendations.
        """
        endpoints_list = list(self._endpoints.values())
        exposed = self.get_exposed_endpoints()
        unauthenticated = self.get_unauthenticated()
        unencrypted = self.get_unencrypted_flows()
        critical_vulns = sum(
            1 for v in self._vulnerabilities if v.severity == "critical"
        )
        score = _compute_risk_score(
            endpoints_list, self._flows, self._vulnerabilities,
        )
        recommendations = _generate_recommendations(
            unauthenticated, unencrypted, self._vulnerabilities, exposed,
        )
        return SurfaceAnalysis(
            application=self._application_name,
            total_endpoints=len(self._endpoints),
            public_count=len(exposed),
            unauthenticated_count=len(unauthenticated),
            vulnerability_count=len(self._vulnerabilities),
            critical_vulns=critical_vulns,
            unencrypted_flows=len(unencrypted),
            risk_score=round(score, 4),
            risk_level=_classify_risk_level(score),
            recommendations=recommendations,
        )

    def get_exposed_endpoints(self) -> list[Endpoint]:
        """Return all public or partner endpoints."""
        return [
            ep for ep in self._endpoints.values()
            if ep.exposure in _EXPOSED_LEVELS
        ]

    def get_unencrypted_flows(self) -> list[DataFlow]:
        """Return all data flows that are not encrypted."""
        return [f for f in self._flows if not f.encrypted]

    def get_unauthenticated(self) -> list[Endpoint]:
        """Return all endpoints that do not require authentication."""
        return [
            ep for ep in self._endpoints.values()
            if not ep.auth_required
        ]

    def risk_score(self) -> float:
        """Compute the overall risk score from 0.0 (safe) to 1.0 (critical).

        Returns:
            Clamped risk score based on endpoints, flows, and vulnerabilities.
        """
        return round(
            _compute_risk_score(
                list(self._endpoints.values()),
                self._flows,
                self._vulnerabilities,
            ),
            4,
        )

    def export(self) -> dict:
        """Export the attack surface as a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "application": self._application_name,
            "endpoints": [
                {
                    "name": ep.name,
                    "endpoint_type": ep.endpoint_type,
                    "exposure": ep.exposure,
                    "auth_required": ep.auth_required,
                    "description": ep.description,
                }
                for ep in self._endpoints.values()
            ],
            "data_flows": [
                {
                    "source": f.source,
                    "destination": f.destination,
                    "data_type": f.data_type,
                    "encrypted": f.encrypted,
                }
                for f in self._flows
            ],
            "vulnerabilities": [
                {
                    "endpoint": v.endpoint,
                    "severity": v.severity,
                    "description": v.description,
                    "cwe": v.cwe,
                }
                for v in self._vulnerabilities
            ],
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the attack surface.

        Returns:
            Multi-line summary string.
        """
        analysis = self.analyze()
        lines = [
            f"Attack Surface: {self._application_name}",
            f"Endpoints: {analysis.total_endpoints}",
            f"Public/Partner: {analysis.public_count}",
            f"Unauthenticated: {analysis.unauthenticated_count}",
            f"Vulnerabilities: {analysis.vulnerability_count}",
            f"Critical: {analysis.critical_vulns}",
            f"Unencrypted flows: {analysis.unencrypted_flows}",
            f"Risk score: {analysis.risk_score}",
            f"Risk level: {analysis.risk_level}",
        ]
        if analysis.recommendations:
            lines.append("Recommendations:")
            for recommendation in analysis.recommendations:
                lines.append(f"  - {recommendation}")
        return "\n".join(lines)
