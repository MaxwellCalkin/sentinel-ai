"""Comprehensive risk assessment scorecards for LLM deployments.

Aggregates multiple safety dimensions into a single report with
weighted scoring, findings tracking, grade assignment, and
pass/fail determination.

Usage:
    from sentinel.risk_scorecard import RiskScorecard, ScorecardResult

    card = RiskScorecard("production-v2")
    card.add_dimension("injection", score=0.95, weight=3.0)
    card.add_dimension("pii", score=0.85, weight=2.0)
    card.add_finding("injection", "medium", "SQL-like pattern detected")
    result = card.calculate()
    print(result.overall_score)  # 0.81
    print(result.grade)          # "B"
"""

from __future__ import annotations

from dataclasses import dataclass, field


_SEVERITY_WEIGHTS: dict[str, float] = {
    "info": 0.0,
    "low": 0.05,
    "medium": 0.1,
    "high": 0.2,
    "critical": 0.4,
}

_VALID_SEVERITIES = frozenset(_SEVERITY_WEIGHTS.keys())


@dataclass
class Finding:
    """A specific safety finding within a dimension."""

    dimension: str
    severity: str
    description: str


@dataclass
class DimensionScore:
    """A scored risk dimension with optional findings."""

    name: str
    score: float
    weight: float
    details: str
    findings: list[Finding] = field(default_factory=list)


@dataclass
class ScorecardResult:
    """Computed scorecard with aggregate metrics."""

    name: str
    overall_score: float
    grade: str
    dimensions: list[DimensionScore]
    findings_count: int
    critical_count: int
    passed: bool


@dataclass
class ComparisonResult:
    """Result of comparing two scorecards."""

    improved: bool
    score_delta: float
    new_findings: int
    resolved_findings: int


def _assign_grade(score: float) -> str:
    if score >= 0.9:
        return "A"
    if score >= 0.8:
        return "B"
    if score >= 0.7:
        return "C"
    if score >= 0.6:
        return "D"
    return "F"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class RiskScorecard:
    """Generate comprehensive risk assessment scorecards.

    Aggregates multiple safety dimensions into a weighted composite
    score with grade assignment, findings tracking, and pass/fail
    determination.
    """

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._dimensions: dict[str, DimensionScore] = {}
        self._findings: list[Finding] = []

    @property
    def name(self) -> str:
        return self._name

    def add_dimension(
        self,
        name: str,
        score: float,
        weight: float = 1.0,
        details: str = "",
    ) -> None:
        """Add a risk dimension to the scorecard.

        Args:
            name: Dimension identifier (e.g. "injection", "pii").
            score: Safety score from 0.0 (dangerous) to 1.0 (safest).
            weight: Relative importance weight for aggregation.
            details: Optional description of the dimension assessment.
        """
        self._dimensions[name] = DimensionScore(
            name=name,
            score=_clamp(score),
            weight=weight,
            details=details,
        )

    def add_finding(
        self,
        dimension: str,
        severity: str,
        description: str,
    ) -> None:
        """Record a finding against a dimension.

        The finding's severity penalty is applied to the dimension
        score during calculation.

        Args:
            dimension: Name of the dimension this finding belongs to.
            severity: One of "info", "low", "medium", "high", "critical".
            description: Human-readable description of the finding.

        Raises:
            ValueError: If severity is not a recognized level.
            KeyError: If the dimension has not been added yet.
        """
        severity_lower = severity.lower()
        if severity_lower not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{severity}'. "
                f"Must be one of: {', '.join(sorted(_VALID_SEVERITIES))}"
            )
        if dimension not in self._dimensions:
            raise KeyError(
                f"Dimension '{dimension}' not found. "
                f"Add it with add_dimension() first."
            )
        finding = Finding(
            dimension=dimension,
            severity=severity_lower,
            description=description,
        )
        self._findings.append(finding)
        self._dimensions[dimension].findings.append(finding)

    def calculate(self) -> ScorecardResult:
        """Compute aggregate score across all dimensions.

        Returns:
            ScorecardResult with overall score, grade, and pass/fail status.
        """
        if not self._dimensions:
            return ScorecardResult(
                name=self._name,
                overall_score=0.0,
                grade="F",
                dimensions=[],
                findings_count=0,
                critical_count=0,
                passed=False,
            )

        total_weight = 0.0
        weighted_sum = 0.0
        critical_count = 0
        dimensions_output: list[DimensionScore] = []

        for dim in self._dimensions.values():
            adjusted_score = self._apply_finding_penalties(dim)
            adjusted_dimension = DimensionScore(
                name=dim.name,
                score=adjusted_score,
                weight=dim.weight,
                details=dim.details,
                findings=list(dim.findings),
            )
            dimensions_output.append(adjusted_dimension)
            weighted_sum += adjusted_score * dim.weight
            total_weight += dim.weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall_score = _clamp(overall_score)

        for finding in self._findings:
            if finding.severity == "critical":
                critical_count += 1

        passed = overall_score >= 0.7 and critical_count == 0

        return ScorecardResult(
            name=self._name,
            overall_score=round(overall_score, 4),
            grade=_assign_grade(overall_score),
            dimensions=dimensions_output,
            findings_count=len(self._findings),
            critical_count=critical_count,
            passed=passed,
        )

    def compare(self, other: RiskScorecard) -> ComparisonResult:
        """Compare this scorecard against another.

        Args:
            other: The baseline scorecard to compare against.

        Returns:
            ComparisonResult indicating improvement or decline.
        """
        self_result = self.calculate()
        other_result = other.calculate()
        score_delta = round(self_result.overall_score - other_result.overall_score, 4)

        self_finding_set = _finding_keys(self._findings)
        other_finding_set = _finding_keys(other._findings)

        new_findings = len(self_finding_set - other_finding_set)
        resolved_findings = len(other_finding_set - self_finding_set)

        return ComparisonResult(
            improved=score_delta > 0,
            score_delta=score_delta,
            new_findings=new_findings,
            resolved_findings=resolved_findings,
        )

    def export(self) -> dict:
        """Export scorecard configuration as a dictionary.

        Returns:
            Dictionary representation suitable for serialization.
        """
        return {
            "name": self._name,
            "dimensions": [
                {
                    "name": dim.name,
                    "score": dim.score,
                    "weight": dim.weight,
                    "details": dim.details,
                }
                for dim in self._dimensions.values()
            ],
            "findings": [
                {
                    "dimension": f.dimension,
                    "severity": f.severity,
                    "description": f.description,
                }
                for f in self._findings
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> RiskScorecard:
        """Create a RiskScorecard from a dictionary.

        Args:
            data: Dictionary with "name", "dimensions", and "findings" keys.

        Returns:
            Reconstructed RiskScorecard instance.
        """
        card = cls(name=data.get("name", "default"))
        for dim in data.get("dimensions", []):
            card.add_dimension(
                name=dim["name"],
                score=dim["score"],
                weight=dim.get("weight", 1.0),
                details=dim.get("details", ""),
            )
        for finding in data.get("findings", []):
            card.add_finding(
                dimension=finding["dimension"],
                severity=finding["severity"],
                description=finding["description"],
            )
        return card

    def reset(self) -> None:
        """Clear all dimensions and findings."""
        self._dimensions.clear()
        self._findings.clear()

    @staticmethod
    def _apply_finding_penalties(dimension: DimensionScore) -> float:
        """Reduce a dimension's score based on its findings' severities."""
        penalty = sum(
            _SEVERITY_WEIGHTS.get(f.severity, 0.0)
            for f in dimension.findings
        )
        return _clamp(dimension.score - penalty)


def _finding_keys(findings: list[Finding]) -> set[tuple[str, str, str]]:
    """Create a set of identifying tuples for finding comparison."""
    return {
        (f.dimension, f.severity, f.description)
        for f in findings
    }
