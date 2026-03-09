"""Tests for SafetyOrchestrator pipeline coordination."""

import pytest
from sentinel.safety_orchestrator import (
    SafetyOrchestrator,
    PipelineComponent,
    ComponentOutcome,
    PipelineResult,
    OrchestratorStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def always_pass(text: str) -> bool:
    return True


def always_fail(text: str) -> bool:
    return False


_flaky_counter = 0


def flaky_check(text: str) -> bool:
    """Fails on first call, passes on second."""
    global _flaky_counter
    _flaky_counter += 1
    if _flaky_counter % 2 == 1:
        raise RuntimeError("transient failure")
    return True


def exploding_check(text: str) -> bool:
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_and_list_components(self):
        orch = SafetyOrchestrator()
        orch.register("check_a", always_pass, phase="pre", priority=1)
        assert len(orch.components) == 1
        assert orch.components[0].name == "check_a"

    def test_register_duplicate_raises(self):
        orch = SafetyOrchestrator()
        orch.register("check_a", always_pass)
        with pytest.raises(ValueError, match="already registered"):
            orch.register("check_a", always_fail)

    def test_register_invalid_phase_raises(self):
        orch = SafetyOrchestrator()
        with pytest.raises(ValueError, match="Invalid phase"):
            orch.register("check_a", always_pass, phase="middle")


# ---------------------------------------------------------------------------
# Sequential strategy
# ---------------------------------------------------------------------------

class TestSequentialStrategy:
    def test_all_pass(self):
        orch = SafetyOrchestrator(strategy="sequential")
        orch.register("a", always_pass, priority=1)
        orch.register("b", always_pass, priority=2)
        result = orch.execute("hello")
        assert result.passed
        assert len(result.outcomes) == 2
        assert result.strategy == "sequential"

    def test_stops_on_first_failure(self):
        orch = SafetyOrchestrator(strategy="sequential")
        orch.register("a", always_fail, priority=1)
        orch.register("b", always_pass, priority=2)
        result = orch.execute("hello")
        assert not result.passed
        assert len(result.outcomes) == 1
        assert result.outcomes[0].name == "a"


# ---------------------------------------------------------------------------
# Parallel strategy
# ---------------------------------------------------------------------------

class TestParallelStrategy:
    def test_runs_all_and_collects_results(self):
        orch = SafetyOrchestrator(strategy="parallel")
        orch.register("a", always_fail, priority=1)
        orch.register("b", always_pass, priority=2)
        result = orch.execute("hello")
        assert not result.passed
        assert len(result.outcomes) == 2
        assert not result.outcomes[0].passed
        assert result.outcomes[1].passed


# ---------------------------------------------------------------------------
# Best-effort strategy
# ---------------------------------------------------------------------------

class TestBestEffortStrategy:
    def test_ignores_exceptions_and_continues(self):
        orch = SafetyOrchestrator(strategy="best_effort")
        orch.register("boom", exploding_check, priority=1)
        orch.register("ok", always_pass, priority=2)
        result = orch.execute("hello")
        assert len(result.outcomes) == 2
        assert result.outcomes[1].passed


# ---------------------------------------------------------------------------
# Enable / Disable
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_disable_and_reenable_component(self):
        orch = SafetyOrchestrator(strategy="sequential")
        orch.register("blocker", always_fail, priority=1)
        orch.register("ok", always_pass, priority=2)

        orch.disable("blocker")
        result = orch.execute("hello")
        assert result.passed
        assert len(result.outcomes) == 1

        orch.enable("blocker")
        result = orch.execute("hello")
        assert not result.passed

    def test_disable_unknown_raises(self):
        orch = SafetyOrchestrator()
        with pytest.raises(KeyError, match="No component"):
            orch.disable("nonexistent")


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_lower_priority_runs_first(self):
        execution_order = []

        def track_a(text: str) -> bool:
            execution_order.append("a")
            return True

        def track_b(text: str) -> bool:
            execution_order.append("b")
            return True

        orch = SafetyOrchestrator(strategy="sequential")
        orch.register("b", track_b, priority=10)
        orch.register("a", track_a, priority=1)
        orch.execute("hello")
        assert execution_order == ["a", "b"]


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retry_recovers_from_transient_failure(self):
        global _flaky_counter
        _flaky_counter = 0
        orch = SafetyOrchestrator(strategy="sequential")
        orch.register("flaky", flaky_check, max_retries=1)
        result = orch.execute("hello")
        assert result.passed
        assert result.outcomes[0].retries_used == 1

    def test_no_retry_reports_error(self):
        orch = SafetyOrchestrator(strategy="sequential")
        orch.register("boom", exploding_check, max_retries=0)
        result = orch.execute("hello")
        assert not result.passed
        assert "error" in result.outcomes[0].message


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_reports_failure_without_enforcing(self):
        orch = SafetyOrchestrator(strategy="sequential", dry_run=True)
        orch.register("a", always_fail)
        result = orch.execute("hello")
        assert not result.passed
        assert not result.enforced

    def test_normal_mode_enforces(self):
        orch = SafetyOrchestrator(strategy="sequential", dry_run=False)
        orch.register("a", always_pass)
        result = orch.execute("hello")
        assert result.enforced


# ---------------------------------------------------------------------------
# Phase filtering
# ---------------------------------------------------------------------------

class TestPhaseFiltering:
    def test_execute_filters_by_phase(self):
        orch = SafetyOrchestrator(strategy="parallel")
        orch.register("pre_check", always_pass, phase="pre")
        orch.register("post_check", always_fail, phase="post")
        result = orch.execute("hello", phase="pre")
        assert result.passed
        assert len(result.outcomes) == 1


# ---------------------------------------------------------------------------
# Export config
# ---------------------------------------------------------------------------

class TestExportConfig:
    def test_export_roundtrip(self):
        orch = SafetyOrchestrator(strategy="parallel", dry_run=True)
        orch.register("a", always_pass, phase="pre", priority=5, max_retries=2)
        config = orch.export_config()
        assert config["strategy"] == "parallel"
        assert config["dry_run"] is True
        assert len(config["components"]) == 1
        comp = config["components"][0]
        assert comp["name"] == "a"
        assert comp["phase"] == "pre"
        assert comp["priority"] == 5
        assert comp["max_retries"] == 2
        assert comp["enabled"] is True


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_pass_rate_tracks_executions(self):
        orch = SafetyOrchestrator(strategy="sequential")
        orch.register("a", always_pass)
        orch.execute("hello")
        orch.execute("world")
        stats = orch.stats()
        assert stats.total_executions == 2
        assert stats.pass_rate == 1.0
        assert stats.component_pass_rates["a"] == 1.0

    def test_failure_rate_tracked(self):
        orch = SafetyOrchestrator(strategy="parallel")
        orch.register("a", always_fail)
        orch.execute("hello")
        stats = orch.stats()
        assert stats.total_executions == 1
        assert stats.pass_rate == 0.0
        assert stats.component_pass_rates["a"] == 0.0

    def test_stats_empty_orchestrator(self):
        stats = SafetyOrchestrator().stats()
        assert stats.total_executions == 0
        assert stats.pass_rate == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_strategy_in_constructor(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            SafetyOrchestrator(strategy="random")

    def test_invalid_strategy_via_setter(self):
        orch = SafetyOrchestrator()
        with pytest.raises(ValueError, match="Invalid strategy"):
            orch.strategy = "random"


# ---------------------------------------------------------------------------
# Duration tracking
# ---------------------------------------------------------------------------

class TestDuration:
    def test_pipeline_and_component_durations(self):
        orch = SafetyOrchestrator()
        orch.register("a", always_pass)
        result = orch.execute("hello")
        assert result.duration_ms >= 0
        assert result.outcomes[0].duration_ms >= 0
