"""Model configuration validator for LLM deployments.

Validate LLM model configurations against safety requirements
before production use. Supports built-in checks, custom rules,
batch validation, and failure tracking.

Usage:
    from sentinel.model_validator import ModelValidator

    validator = ModelValidator()
    result = validator.validate({"model": "claude-sonnet", "temperature": 0.7})
    print(result.passed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ValidationRule:
    """A single validation rule for a model config field."""

    field: str
    check_fn: Callable[[Any], bool]
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class FieldResult:
    """Result of validating a single field."""

    field: str
    passed: bool
    message: str
    severity: str


@dataclass
class ModelValidation:
    """Result of validating a complete model config."""

    config: dict[str, Any]
    passed: bool
    results: list[FieldResult]
    errors: int
    warnings: int


@dataclass
class ValidatorStats:
    """Aggregate statistics across validations."""

    total_validations: int
    pass_rate: float
    common_failures: dict[str, int]


def _is_temperature_in_range(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0 <= value <= 2


def _is_max_tokens_positive(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _is_model_name_not_empty(value: Any) -> bool:
    return isinstance(value, str) and len(value.strip()) > 0


def _is_top_p_in_range(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0 <= value <= 1


_BUILTIN_RULES = [
    ValidationRule(
        field="temperature",
        check_fn=_is_temperature_in_range,
        message="temperature must be a number between 0 and 2",
        severity="error",
    ),
    ValidationRule(
        field="max_tokens",
        check_fn=_is_max_tokens_positive,
        message="max_tokens must be a positive integer",
        severity="error",
    ),
    ValidationRule(
        field="model",
        check_fn=_is_model_name_not_empty,
        message="model name must be a non-empty string",
        severity="error",
    ),
    ValidationRule(
        field="top_p",
        check_fn=_is_top_p_in_range,
        message="top_p must be a number between 0 and 1",
        severity="error",
    ),
]


class ModelValidator:
    """Validate LLM model configurations against safety requirements.

    Combines built-in checks (temperature, max_tokens, model, top_p)
    with user-defined custom rules. Tracks validation history for
    aggregate statistics.
    """

    def __init__(self, include_builtins: bool = True) -> None:
        self._rules: list[ValidationRule] = []
        if include_builtins:
            self._rules.extend(_BUILTIN_RULES)
        self._total_validations = 0
        self._passed_validations = 0
        self._failure_counts: dict[str, int] = {}

    def add_rule(self, rule: ValidationRule) -> None:
        """Register a custom validation rule."""
        self._rules.append(rule)

    @property
    def rules(self) -> list[ValidationRule]:
        """Return a copy of the current rule set."""
        return list(self._rules)

    def validate(self, config: dict[str, Any]) -> ModelValidation:
        """Validate a model config dict against all registered rules.

        Only rules whose field is present in the config are evaluated.
        """
        field_results = self._evaluate_rules(config)
        errors = _count_by_severity(field_results, "error")
        warnings = _count_by_severity(field_results, "warning")
        passed = errors == 0

        self._record_stats(passed, field_results)

        return ModelValidation(
            config=config,
            passed=passed,
            results=field_results,
            errors=errors,
            warnings=warnings,
        )

    def validate_batch(
        self, configs: list[dict[str, Any]]
    ) -> list[ModelValidation]:
        """Validate multiple model configs."""
        return [self.validate(config) for config in configs]

    def stats(self) -> ValidatorStats:
        """Return aggregate validation statistics."""
        pass_rate = self._compute_pass_rate()
        return ValidatorStats(
            total_validations=self._total_validations,
            pass_rate=pass_rate,
            common_failures=dict(self._failure_counts),
        )

    def reset_stats(self) -> None:
        """Clear all recorded statistics."""
        self._total_validations = 0
        self._passed_validations = 0
        self._failure_counts = {}

    def _evaluate_rules(
        self, config: dict[str, Any]
    ) -> list[FieldResult]:
        results: list[FieldResult] = []
        for rule in self._rules:
            if rule.field not in config:
                continue
            value = config[rule.field]
            passed = rule.check_fn(value)
            results.append(FieldResult(
                field=rule.field,
                passed=passed,
                message="" if passed else rule.message,
                severity=rule.severity if not passed else "ok",
            ))
        return results

    def _record_stats(
        self, passed: bool, results: list[FieldResult]
    ) -> None:
        self._total_validations += 1
        if passed:
            self._passed_validations += 1
        for result in results:
            if not result.passed:
                self._failure_counts[result.field] = (
                    self._failure_counts.get(result.field, 0) + 1
                )

    def _compute_pass_rate(self) -> float:
        if self._total_validations == 0:
            return 0.0
        return self._passed_validations / self._total_validations


def _count_by_severity(results: list[FieldResult], severity: str) -> int:
    return sum(
        1 for r in results if not r.passed and r.severity == severity
    )
