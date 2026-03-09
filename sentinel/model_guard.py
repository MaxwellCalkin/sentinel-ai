"""Model deployment configuration guard.

Validates model configs against safety policies, checks for unsafe
parameter combinations, and enforces deployment constraints.

Usage:
    from sentinel.model_guard import ModelGuard, ModelConfig

    guard = ModelGuard()
    report = guard.check(ModelConfig(model_name="claude-sonnet", temperature=0.7))
    print(report.is_safe)    # True
    print(report.risk_level) # "low"
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration parameters for an LLM deployment."""

    model_name: str
    temperature: float = 1.0
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ConfigIssue:
    """A single issue found during config validation."""

    field_name: str
    severity: str  # "error" | "warning" | "info"
    message: str
    current_value: str
    recommended_value: str = ""


@dataclass
class GuardReport:
    """Result of validating a model config against a policy."""

    config: ModelConfig
    is_safe: bool
    issues: list[ConfigIssue]
    risk_level: str  # "low" | "medium" | "high" | "critical"
    score: float  # 0-1, where 1 = perfectly safe


@dataclass
class GuardPolicy:
    """Constraints that model configs must satisfy."""

    max_temperature: float = 2.0
    max_max_tokens: int = 100_000
    require_stop_sequences: bool = False
    blocked_models: list[str] = field(default_factory=list)
    max_frequency_penalty: float = 2.0
    max_presence_penalty: float = 2.0


@dataclass
class ModelGuardStats:
    """Cumulative statistics from guard checks."""

    total_checked: int = 0
    passed: int = 0
    failed: int = 0
    by_risk_level: dict[str, int] = field(default_factory=dict)


_TEMPERATURE_WARNING_THRESHOLD = 1.5
_MAX_TOKENS_WARNING_THRESHOLD = 50_000


def _risk_level_from_score(score: float) -> str:
    if score < 0.25:
        return "critical"
    if score < 0.5:
        return "high"
    if score < 0.75:
        return "medium"
    return "low"


def _compute_score(issues: list[ConfigIssue]) -> float:
    penalty = 0.0
    for issue in issues:
        if issue.severity == "error":
            penalty += 0.25
        elif issue.severity == "warning":
            penalty += 0.1
    return max(0.0, 1.0 - penalty)


class ModelGuard:
    """Validate model deployment configurations against a safety policy.

    Checks temperature, token limits, penalties, blocked models,
    stop sequences, and combined-risk patterns. Tracks cumulative
    statistics across all checks.
    """

    def __init__(self, policy: GuardPolicy | None = None) -> None:
        self._policy = policy or GuardPolicy()
        self._stats = ModelGuardStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, config: ModelConfig) -> GuardReport:
        """Validate a model config against the current policy.

        Returns a GuardReport describing all issues found, an overall
        safety score (1.0 = safe), and a risk level.
        """
        issues: list[ConfigIssue] = []

        self._check_temperature(config, issues)
        self._check_max_tokens(config, issues)
        self._check_top_p(config, issues)
        self._check_frequency_penalty(config, issues)
        self._check_presence_penalty(config, issues)
        self._check_blocked_model(config, issues)
        self._check_stop_sequences(config, issues)
        self._check_combined_risk(config, issues)

        score = _compute_score(issues)
        risk_level = _risk_level_from_score(score)
        has_errors = any(issue.severity == "error" for issue in issues)
        is_safe = not has_errors

        report = GuardReport(
            config=config,
            is_safe=is_safe,
            issues=issues,
            risk_level=risk_level,
            score=score,
        )
        self._record_stats(report)
        return report

    def check_batch(self, configs: list[ModelConfig]) -> list[GuardReport]:
        """Validate multiple configs and return reports in order."""
        return [self.check(config) for config in configs]

    def compare(self, config_a: ModelConfig, config_b: ModelConfig) -> dict:
        """Return a dict of fields that differ between two configs."""
        differences: dict[str, dict[str, object]] = {}
        fields_to_compare = [
            "model_name", "temperature", "max_tokens", "top_p",
            "frequency_penalty", "presence_penalty", "stop_sequences",
        ]
        for field_name in fields_to_compare:
            value_a = getattr(config_a, field_name)
            value_b = getattr(config_b, field_name)
            if value_a != value_b:
                differences[field_name] = {"a": value_a, "b": value_b}
        return differences

    def stats(self) -> ModelGuardStats:
        """Return cumulative check statistics."""
        return self._stats

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_temperature(
        self, config: ModelConfig, issues: list[ConfigIssue]
    ) -> None:
        temp = config.temperature
        if temp < 0 or temp > self._policy.max_temperature:
            issues.append(ConfigIssue(
                field_name="temperature",
                severity="error",
                message=f"Temperature {temp} is outside allowed range [0, {self._policy.max_temperature}]",
                current_value=str(temp),
                recommended_value="0.7",
            ))
        elif temp > _TEMPERATURE_WARNING_THRESHOLD:
            issues.append(ConfigIssue(
                field_name="temperature",
                severity="warning",
                message=f"Temperature {temp} is high; outputs may be unpredictable",
                current_value=str(temp),
                recommended_value="1.0",
            ))

    def _check_max_tokens(
        self, config: ModelConfig, issues: list[ConfigIssue]
    ) -> None:
        tokens = config.max_tokens
        if tokens <= 0:
            issues.append(ConfigIssue(
                field_name="max_tokens",
                severity="error",
                message="max_tokens must be positive",
                current_value=str(tokens),
                recommended_value="4096",
            ))
        elif tokens > self._policy.max_max_tokens:
            issues.append(ConfigIssue(
                field_name="max_tokens",
                severity="error",
                message=f"max_tokens {tokens} exceeds policy limit of {self._policy.max_max_tokens}",
                current_value=str(tokens),
                recommended_value=str(self._policy.max_max_tokens),
            ))
        elif tokens > _MAX_TOKENS_WARNING_THRESHOLD:
            issues.append(ConfigIssue(
                field_name="max_tokens",
                severity="warning",
                message=f"max_tokens {tokens} is very large; consider lowering for cost control",
                current_value=str(tokens),
                recommended_value="4096",
            ))

    def _check_top_p(
        self, config: ModelConfig, issues: list[ConfigIssue]
    ) -> None:
        if config.top_p < 0 or config.top_p > 1:
            issues.append(ConfigIssue(
                field_name="top_p",
                severity="error",
                message=f"top_p {config.top_p} is outside allowed range [0, 1]",
                current_value=str(config.top_p),
                recommended_value="1.0",
            ))

    def _check_frequency_penalty(
        self, config: ModelConfig, issues: list[ConfigIssue]
    ) -> None:
        penalty = config.frequency_penalty
        limit = self._policy.max_frequency_penalty
        if penalty < -2 or penalty > limit:
            issues.append(ConfigIssue(
                field_name="frequency_penalty",
                severity="error",
                message=f"frequency_penalty {penalty} is outside allowed range [-2, {limit}]",
                current_value=str(penalty),
                recommended_value="0.0",
            ))

    def _check_presence_penalty(
        self, config: ModelConfig, issues: list[ConfigIssue]
    ) -> None:
        penalty = config.presence_penalty
        limit = self._policy.max_presence_penalty
        if penalty < -2 or penalty > limit:
            issues.append(ConfigIssue(
                field_name="presence_penalty",
                severity="error",
                message=f"presence_penalty {penalty} is outside allowed range [-2, {limit}]",
                current_value=str(penalty),
                recommended_value="0.0",
            ))

    def _check_blocked_model(
        self, config: ModelConfig, issues: list[ConfigIssue]
    ) -> None:
        lowered = config.model_name.lower()
        for blocked in self._policy.blocked_models:
            if blocked.lower() == lowered:
                issues.append(ConfigIssue(
                    field_name="model_name",
                    severity="error",
                    message=f"Model '{config.model_name}' is blocked by policy",
                    current_value=config.model_name,
                ))
                return

    def _check_stop_sequences(
        self, config: ModelConfig, issues: list[ConfigIssue]
    ) -> None:
        if self._policy.require_stop_sequences and not config.stop_sequences:
            issues.append(ConfigIssue(
                field_name="stop_sequences",
                severity="error",
                message="Policy requires at least one stop sequence",
                current_value="[]",
                recommended_value='["\\n"]',
            ))

    def _check_combined_risk(
        self, config: ModelConfig, issues: list[ConfigIssue]
    ) -> None:
        """Flag when high temperature is paired with high max_tokens."""
        high_temp = config.temperature > _TEMPERATURE_WARNING_THRESHOLD
        high_tokens = config.max_tokens > _MAX_TOKENS_WARNING_THRESHOLD
        if high_temp and high_tokens:
            issues.append(ConfigIssue(
                field_name="temperature+max_tokens",
                severity="warning",
                message="High temperature combined with high max_tokens increases risk of unpredictable, costly outputs",
                current_value=f"temperature={config.temperature}, max_tokens={config.max_tokens}",
                recommended_value="Lower temperature or max_tokens",
            ))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _record_stats(self, report: GuardReport) -> None:
        self._stats.total_checked += 1
        if report.is_safe:
            self._stats.passed += 1
        else:
            self._stats.failed += 1
        level = report.risk_level
        self._stats.by_risk_level[level] = self._stats.by_risk_level.get(level, 0) + 1
