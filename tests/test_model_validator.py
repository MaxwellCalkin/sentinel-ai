"""Tests for model validator."""

import pytest

from sentinel.model_validator import (
    ModelValidator,
    ModelValidation,
    FieldResult,
    ValidationRule,
    ValidatorStats,
)


class TestBuiltinTemperature:
    def test_valid_temperature_including_bounds(self):
        v = ModelValidator()
        assert v.validate({"temperature": 0.7}).passed
        assert v.validate({"temperature": 0}).passed
        assert v.validate({"temperature": 2}).passed

    def test_temperature_out_of_range(self):
        v = ModelValidator()
        result = v.validate({"temperature": 2.5})
        assert not result.passed
        assert result.errors == 1

    def test_negative_temperature(self):
        v = ModelValidator()
        result = v.validate({"temperature": -0.1})
        assert not result.passed


class TestBuiltinMaxTokens:
    def test_valid_max_tokens(self):
        v = ModelValidator()
        result = v.validate({"max_tokens": 4096})
        assert result.passed

    def test_zero_and_negative_max_tokens(self):
        v = ModelValidator()
        assert not v.validate({"max_tokens": 0}).passed
        assert not v.validate({"max_tokens": -10}).passed


class TestBuiltinModelName:
    def test_valid_model_name(self):
        v = ModelValidator()
        result = v.validate({"model": "claude-sonnet"})
        assert result.passed

    def test_empty_and_whitespace_model_name(self):
        v = ModelValidator()
        assert not v.validate({"model": ""}).passed
        assert not v.validate({"model": "   "}).passed


class TestBuiltinTopP:
    def test_valid_top_p(self):
        v = ModelValidator()
        result = v.validate({"top_p": 0.9})
        assert result.passed

    def test_top_p_out_of_range(self):
        v = ModelValidator()
        result = v.validate({"top_p": 1.5})
        assert not result.passed


class TestCustomRules:
    def test_custom_rule_passes(self):
        v = ModelValidator()
        v.add_rule(ValidationRule(
            field="env",
            check_fn=lambda val: val == "production",
            message="env must be production",
        ))
        result = v.validate({"env": "production"})
        assert result.passed

    def test_custom_rule_fails(self):
        v = ModelValidator()
        v.add_rule(ValidationRule(
            field="env",
            check_fn=lambda val: val == "production",
            message="env must be production",
        ))
        result = v.validate({"env": "staging"})
        assert not result.passed
        failed = [r for r in result.results if not r.passed]
        assert any("env must be production" in r.message for r in failed)

    def test_custom_warning_severity(self):
        v = ModelValidator()
        v.add_rule(ValidationRule(
            field="log_level",
            check_fn=lambda val: val != "debug",
            message="debug logging is not recommended",
            severity="warning",
        ))
        result = v.validate({"log_level": "debug"})
        assert result.passed  # warnings do not cause failure
        assert result.warnings == 1


class TestValidationReport:
    def test_report_structure(self):
        v = ModelValidator()
        result = v.validate({"model": "claude", "temperature": 5.0})
        assert isinstance(result, ModelValidation)
        assert isinstance(result.results, list)
        assert all(isinstance(r, FieldResult) for r in result.results)
        assert result.errors == 1
        assert result.config == {"model": "claude", "temperature": 5.0}

    def test_fields_not_in_config_are_skipped(self):
        v = ModelValidator()
        result = v.validate({"model": "claude"})
        checked_fields = [r.field for r in result.results]
        assert "model" in checked_fields
        assert "temperature" not in checked_fields


class TestBatchValidation:
    def test_batch_returns_one_result_per_config(self):
        v = ModelValidator()
        configs = [
            {"model": "claude", "temperature": 0.5},
            {"model": "", "temperature": 3.0},
        ]
        results = v.validate_batch(configs)
        assert len(results) == 2
        assert results[0].passed
        assert not results[1].passed


class TestStats:
    def test_stats_after_validations(self):
        v = ModelValidator()
        v.validate({"temperature": 0.7})
        v.validate({"temperature": 5.0})
        v.validate({"temperature": 0.3})

        stats = v.stats()
        assert isinstance(stats, ValidatorStats)
        assert stats.total_validations == 3
        assert stats.pass_rate == pytest.approx(2 / 3)
        assert "temperature" in stats.common_failures
        assert stats.common_failures["temperature"] == 1

    def test_stats_with_no_validations(self):
        v = ModelValidator()
        stats = v.stats()
        assert stats.total_validations == 0
        assert stats.pass_rate == 0.0
        assert stats.common_failures == {}

    def test_reset_stats(self):
        v = ModelValidator()
        v.validate({"temperature": 5.0})
        v.reset_stats()
        stats = v.stats()
        assert stats.total_validations == 0


class TestEmptyAndEdge:
    def test_empty_config(self):
        v = ModelValidator()
        result = v.validate({})
        assert result.passed
        assert result.results == []

    def test_no_builtins(self):
        v = ModelValidator(include_builtins=False)
        result = v.validate({"temperature": 999})
        assert result.passed
        assert result.results == []

    def test_rules_property_returns_copy(self):
        v = ModelValidator()
        rules = v.rules
        original_len = len(rules)
        rules.append(ValidationRule(
            field="x", check_fn=lambda _: True, message="x"
        ))
        assert len(v.rules) == original_len
