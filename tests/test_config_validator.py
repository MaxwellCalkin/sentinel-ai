"""Tests for config validator."""

import pytest

from sentinel.config_validator import ConfigValidator, ConfigResult, ConfigIssue


class TestTemperature:
    def test_valid_temperature(self):
        v = ConfigValidator()
        result = v.validate({"temperature": 0.7})
        assert result.valid

    def test_high_temperature_warning(self):
        v = ConfigValidator()
        result = v.validate({"temperature": 1.8})
        assert any(i.field == "temperature" and i.severity == "warning" for i in result.issues)

    def test_invalid_temperature(self):
        v = ConfigValidator()
        result = v.validate({"temperature": 3.0})
        assert not result.valid


class TestMaxTokens:
    def test_valid_max_tokens(self):
        v = ConfigValidator()
        result = v.validate({"max_tokens": 4096})
        assert result.valid

    def test_zero_max_tokens(self):
        v = ConfigValidator()
        result = v.validate({"max_tokens": 0})
        assert not result.valid

    def test_very_high_max_tokens(self):
        v = ConfigValidator()
        result = v.validate({"max_tokens": 200000})
        assert any(i.severity == "warning" for i in result.issues)


class TestApiKey:
    def test_hardcoded_key(self):
        v = ConfigValidator()
        result = v.validate({"api_key": "sk-1234567890"})
        assert not result.valid

    def test_env_reference_ok(self):
        v = ConfigValidator()
        result = v.validate({"api_key": "${ANTHROPIC_API_KEY}"})
        assert result.valid

    def test_env_prefix_ok(self):
        v = ConfigValidator()
        result = v.validate({"api_key": "env:API_KEY"})
        assert result.valid


class TestTopP:
    def test_valid_top_p(self):
        v = ConfigValidator()
        result = v.validate({"top_p": 0.9})
        assert result.valid

    def test_invalid_top_p(self):
        v = ConfigValidator()
        result = v.validate({"top_p": 1.5})
        assert not result.valid


class TestStrictMode:
    def test_strict_promotes_warnings(self):
        v = ConfigValidator(strict=True)
        result = v.validate({"temperature": 1.8})
        assert not result.valid

    def test_non_strict_allows_warnings(self):
        v = ConfigValidator(strict=False)
        result = v.validate({"temperature": 1.8})
        assert result.valid


class TestCustomRules:
    def test_custom_rule_pass(self):
        v = ConfigValidator(custom_rules=[("env", "production", "Must be production")])
        result = v.validate({"env": "production"})
        assert result.valid

    def test_custom_rule_fail(self):
        v = ConfigValidator(custom_rules=[("env", "production", "Must be production")])
        result = v.validate({"env": "development"})
        assert any("Must be production" in i.message for i in result.issues)


class TestBatch:
    def test_batch_validate(self):
        v = ConfigValidator()
        results = v.validate_batch([{"temperature": 0.7}, {"temperature": 5.0}])
        assert len(results) == 2
        assert results[0].valid
        assert not results[1].valid


class TestResult:
    def test_result_structure(self):
        v = ConfigValidator()
        result = v.validate({"temperature": 0.7, "max_tokens": 100})
        assert isinstance(result, ConfigResult)
        assert result.fields_checked >= 2

    def test_empty_config(self):
        v = ConfigValidator()
        result = v.validate({})
        assert result.valid
        assert result.fields_checked == 0
