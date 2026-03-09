"""Tests for model deployment configuration guard."""

import pytest

from sentinel.model_guard import (
    ConfigIssue,
    GuardPolicy,
    GuardReport,
    ModelConfig,
    ModelGuard,
    ModelGuardStats,
)


class TestSafeConfig:
    def test_default_config_is_safe(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="claude-sonnet"))
        assert report.is_safe
        assert report.risk_level == "low"
        assert report.score == 1.0
        assert report.issues == []

    def test_explicit_safe_config(self):
        config = ModelConfig(
            model_name="claude-sonnet",
            temperature=0.7,
            max_tokens=4096,
            top_p=0.9,
            stop_sequences=["\n"],
        )
        guard = ModelGuard()
        report = guard.check(config)
        assert report.is_safe
        assert report.score == 1.0

    def test_report_references_original_config(self):
        config = ModelConfig(model_name="gpt-4")
        report = ModelGuard().check(config)
        assert report.config is config


class TestTemperature:
    def test_high_temperature_warning(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", temperature=1.8))
        assert report.is_safe  # warning does not fail
        issues = [i for i in report.issues if i.field_name == "temperature"]
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_temperature_at_warning_boundary_no_warning(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", temperature=1.5))
        temp_issues = [i for i in report.issues if i.field_name == "temperature"]
        assert temp_issues == []

    def test_temperature_above_policy_max_error(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", temperature=2.5))
        assert not report.is_safe
        issues = [i for i in report.issues if i.field_name == "temperature"]
        assert issues[0].severity == "error"

    def test_negative_temperature_error(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", temperature=-0.1))
        assert not report.is_safe
        issues = [i for i in report.issues if i.field_name == "temperature"]
        assert issues[0].severity == "error"


class TestMaxTokens:
    def test_negative_max_tokens_error(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", max_tokens=-1))
        assert not report.is_safe
        issues = [i for i in report.issues if i.field_name == "max_tokens"]
        assert issues[0].severity == "error"

    def test_zero_max_tokens_error(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", max_tokens=0))
        assert not report.is_safe

    def test_exceeds_policy_max_tokens_error(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", max_tokens=200_000))
        assert not report.is_safe
        issues = [i for i in report.issues if i.field_name == "max_tokens"]
        assert issues[0].severity == "error"

    def test_large_max_tokens_warning(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", max_tokens=60_000))
        assert report.is_safe  # warning, not error
        issues = [i for i in report.issues if i.field_name == "max_tokens"]
        assert len(issues) == 1
        assert issues[0].severity == "warning"


class TestTopP:
    def test_top_p_above_one_error(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", top_p=1.5))
        assert not report.is_safe
        issues = [i for i in report.issues if i.field_name == "top_p"]
        assert issues[0].severity == "error"

    def test_top_p_negative_error(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", top_p=-0.1))
        assert not report.is_safe


class TestPenalties:
    def test_frequency_penalty_out_of_range(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", frequency_penalty=3.0))
        assert not report.is_safe
        issues = [i for i in report.issues if i.field_name == "frequency_penalty"]
        assert len(issues) == 1

    def test_presence_penalty_out_of_range(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", presence_penalty=-3.0))
        assert not report.is_safe
        issues = [i for i in report.issues if i.field_name == "presence_penalty"]
        assert len(issues) == 1

    def test_valid_penalties_pass(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(
            model_name="m",
            frequency_penalty=1.5,
            presence_penalty=-1.5,
        ))
        assert report.is_safe


class TestBlockedModel:
    def test_blocked_model_error(self):
        policy = GuardPolicy(blocked_models=["dangerous-model"])
        guard = ModelGuard(policy=policy)
        report = guard.check(ModelConfig(model_name="dangerous-model"))
        assert not report.is_safe
        issues = [i for i in report.issues if i.field_name == "model_name"]
        assert issues[0].severity == "error"

    def test_blocked_model_case_insensitive(self):
        policy = GuardPolicy(blocked_models=["Dangerous-Model"])
        guard = ModelGuard(policy=policy)
        report = guard.check(ModelConfig(model_name="dangerous-model"))
        assert not report.is_safe


class TestStopSequences:
    def test_missing_stop_sequences_when_required(self):
        policy = GuardPolicy(require_stop_sequences=True)
        guard = ModelGuard(policy=policy)
        report = guard.check(ModelConfig(model_name="m"))
        assert not report.is_safe
        issues = [i for i in report.issues if i.field_name == "stop_sequences"]
        assert len(issues) == 1

    def test_stop_sequences_present_when_required(self):
        policy = GuardPolicy(require_stop_sequences=True)
        guard = ModelGuard(policy=policy)
        report = guard.check(ModelConfig(model_name="m", stop_sequences=["\n"]))
        assert report.is_safe


class TestCombinedRisk:
    def test_high_temp_and_high_tokens_combined_warning(self):
        guard = ModelGuard()
        config = ModelConfig(model_name="m", temperature=1.8, max_tokens=60_000)
        report = guard.check(config)
        combined = [i for i in report.issues if i.field_name == "temperature+max_tokens"]
        assert len(combined) == 1
        assert combined[0].severity == "warning"

    def test_high_temp_low_tokens_no_combined_warning(self):
        guard = ModelGuard()
        config = ModelConfig(model_name="m", temperature=1.8, max_tokens=1000)
        report = guard.check(config)
        combined = [i for i in report.issues if i.field_name == "temperature+max_tokens"]
        assert combined == []


class TestScoreCalculation:
    def test_single_error_reduces_score(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", temperature=-1))
        assert report.score == pytest.approx(0.75)

    def test_single_warning_reduces_score(self):
        guard = ModelGuard()
        report = guard.check(ModelConfig(model_name="m", temperature=1.8))
        assert report.score == pytest.approx(0.9)

    def test_multiple_errors_stack(self):
        guard = ModelGuard()
        config = ModelConfig(
            model_name="m",
            temperature=-1,
            max_tokens=-1,
            top_p=5.0,
            frequency_penalty=10.0,
        )
        report = guard.check(config)
        assert report.score == 0.0

    def test_score_floor_at_zero(self):
        policy = GuardPolicy(
            blocked_models=["m"],
            require_stop_sequences=True,
        )
        guard = ModelGuard(policy=policy)
        config = ModelConfig(
            model_name="m",
            temperature=-1,
            max_tokens=-1,
            top_p=5.0,
        )
        report = guard.check(config)
        assert report.score == 0.0


class TestRiskLevel:
    def test_low_risk(self):
        report = ModelGuard().check(ModelConfig(model_name="m"))
        assert report.risk_level == "low"

    def test_medium_risk(self):
        guard = ModelGuard()
        # 3 warnings: temp warning + max_tokens warning + combined warning = 0.3 penalty -> score 0.7
        config = ModelConfig(model_name="m", temperature=1.8, max_tokens=60_000)
        report = guard.check(config)
        assert report.risk_level == "medium"

    def test_high_risk(self):
        guard = ModelGuard()
        # 3 errors = 0.75 penalty -> score 0.25 -> "high" (0.25 <= score < 0.5)
        config = ModelConfig(model_name="m", temperature=-1, max_tokens=-1, top_p=5.0)
        report = guard.check(config)
        assert report.risk_level == "high"

    def test_critical_risk(self):
        guard = ModelGuard()
        config = ModelConfig(
            model_name="m",
            temperature=-1,
            max_tokens=-1,
            top_p=5.0,
            frequency_penalty=10.0,
        )
        report = guard.check(config)
        assert report.risk_level == "critical"


class TestBatchCheck:
    def test_batch_returns_correct_count(self):
        guard = ModelGuard()
        configs = [
            ModelConfig(model_name="a"),
            ModelConfig(model_name="b", temperature=5.0),
        ]
        reports = guard.check_batch(configs)
        assert len(reports) == 2
        assert reports[0].is_safe
        assert not reports[1].is_safe


class TestCompare:
    def test_identical_configs_no_differences(self):
        guard = ModelGuard()
        config = ModelConfig(model_name="m")
        assert guard.compare(config, config) == {}

    def test_differing_fields_listed(self):
        guard = ModelGuard()
        config_a = ModelConfig(model_name="a", temperature=0.5)
        config_b = ModelConfig(model_name="b", temperature=0.9)
        diff = guard.compare(config_a, config_b)
        assert "model_name" in diff
        assert diff["model_name"]["a"] == "a"
        assert diff["model_name"]["b"] == "b"
        assert "temperature" in diff


class TestStats:
    def test_initial_stats_are_zero(self):
        guard = ModelGuard()
        s = guard.stats()
        assert s.total_checked == 0
        assert s.passed == 0
        assert s.failed == 0

    def test_stats_track_checks(self):
        guard = ModelGuard()
        guard.check(ModelConfig(model_name="m"))
        guard.check(ModelConfig(model_name="m", temperature=5.0))
        s = guard.stats()
        assert s.total_checked == 2
        assert s.passed == 1
        assert s.failed == 1

    def test_stats_track_risk_levels(self):
        guard = ModelGuard()
        guard.check(ModelConfig(model_name="m"))
        guard.check(ModelConfig(model_name="m"))
        s = guard.stats()
        assert s.by_risk_level.get("low") == 2


class TestCustomPolicy:
    def test_custom_max_temperature(self):
        policy = GuardPolicy(max_temperature=1.0)
        guard = ModelGuard(policy=policy)
        report = guard.check(ModelConfig(model_name="m", temperature=1.5))
        assert not report.is_safe

    def test_custom_max_max_tokens(self):
        policy = GuardPolicy(max_max_tokens=1000)
        guard = ModelGuard(policy=policy)
        report = guard.check(ModelConfig(model_name="m", max_tokens=2000))
        assert not report.is_safe

    def test_custom_penalty_limits(self):
        policy = GuardPolicy(max_frequency_penalty=1.0, max_presence_penalty=1.0)
        guard = ModelGuard(policy=policy)
        report = guard.check(ModelConfig(model_name="m", frequency_penalty=1.5))
        assert not report.is_safe
