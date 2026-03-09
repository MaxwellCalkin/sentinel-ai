"""Tests for unified safety gateway."""

import pytest
from sentinel.safety_gateway import (
    SafetyGateway,
    GatewayCheck,
    GatewayCheckResult,
    GatewayResult,
    GatewayStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def always_pass(text):
    return (True, "OK")


def always_fail(text):
    return (False, "Blocked")


def length_check(text):
    if len(text) > 100:
        return (False, "Input too long")
    return (True, "Length OK")


def toxicity_check(text):
    if "hate" in text.lower():
        return (False, "Toxic content detected")
    return (True, "Clean")


def injection_check(text):
    if "ignore previous" in text.lower():
        return (False, "Injection attempt")
    return (True, "No injection")


# ---------------------------------------------------------------------------
# Basic processing
# ---------------------------------------------------------------------------

class TestBasicProcessing:
    def test_all_checks_pass(self):
        gw = SafetyGateway()
        gw.add_input_check("length", length_check)
        gw.add_output_check("toxicity", toxicity_check)
        result = gw.process(input_text="Hello", output_text="World")
        assert result.passed
        assert len(result.input_results) == 1
        assert len(result.output_results) == 1
        assert not result.short_circuited

    def test_input_check_fails(self):
        gw = SafetyGateway()
        gw.add_input_check("length", length_check)
        result = gw.process(input_text="x" * 200)
        assert not result.passed
        assert not result.input_results[0].passed
        assert result.input_results[0].message == "Input too long"

    def test_output_check_fails(self):
        gw = SafetyGateway()
        gw.add_output_check("toxicity", toxicity_check)
        result = gw.process(output_text="I hate you")
        assert not result.passed
        assert not result.output_results[0].passed

    def test_no_checks_registered(self):
        gw = SafetyGateway()
        result = gw.process(input_text="Hello", output_text="World")
        assert result.passed
        assert result.input_results == []
        assert result.output_results == []


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_input_checks_run_in_priority_order(self):
        gw = SafetyGateway()
        gw.add_input_check("second", always_pass, priority=10)
        gw.add_input_check("first", always_pass, priority=1)
        gw.add_input_check("third", always_pass, priority=20)
        result = gw.process(input_text="test")
        names = [r.name for r in result.input_results]
        assert names == ["first", "second", "third"]

    def test_output_checks_run_in_priority_order(self):
        gw = SafetyGateway()
        gw.add_output_check("beta", always_pass, priority=5)
        gw.add_output_check("alpha", always_pass, priority=1)
        result = gw.process(output_text="test")
        names = [r.name for r in result.output_results]
        assert names == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Short-circuit behavior
# ---------------------------------------------------------------------------

class TestShortCircuit:
    def test_short_circuit_stops_on_first_failure(self):
        gw = SafetyGateway(short_circuit=True)
        gw.add_input_check("fail_first", always_fail, priority=1)
        gw.add_input_check("never_runs", always_pass, priority=2)
        result = gw.process(input_text="test")
        assert not result.passed
        assert result.short_circuited
        assert len(result.input_results) == 1
        assert result.input_results[0].name == "fail_first"

    def test_no_short_circuit_runs_all_checks(self):
        gw = SafetyGateway(short_circuit=False)
        gw.add_input_check("fail_first", always_fail, priority=1)
        gw.add_input_check("still_runs", always_pass, priority=2)
        result = gw.process(input_text="test")
        assert not result.passed
        assert not result.short_circuited
        assert len(result.input_results) == 2

    def test_short_circuit_output_phase(self):
        gw = SafetyGateway(short_circuit=True)
        gw.add_output_check("fail_output", always_fail, priority=1)
        gw.add_output_check("skipped", always_pass, priority=2)
        result = gw.process(output_text="test")
        assert result.short_circuited
        assert len(result.output_results) == 1


# ---------------------------------------------------------------------------
# Enable / disable checks at runtime
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_disable_check(self):
        gw = SafetyGateway()
        gw.add_input_check("blocker", always_fail)
        assert not gw.process(input_text="test").passed

        gw.disable_check("blocker")
        assert gw.process(input_text="test").passed

    def test_enable_check(self):
        gw = SafetyGateway()
        gw.add_input_check("blocker", always_fail, enabled=False)
        assert gw.process(input_text="test").passed

        gw.enable_check("blocker")
        assert not gw.process(input_text="test").passed

    def test_toggle_returns_false_for_unknown(self):
        gw = SafetyGateway()
        assert not gw.disable_check("nonexistent")
        assert not gw.enable_check("nonexistent")


# ---------------------------------------------------------------------------
# Gateway statistics
# ---------------------------------------------------------------------------

class TestGatewayStats:
    def test_total_requests_increments(self):
        gw = SafetyGateway()
        gw.add_input_check("check", always_pass)
        gw.process(input_text="a")
        gw.process(input_text="b")
        gw.process(input_text="c")
        assert gw.get_stats().total_requests == 3

    def test_input_pass_rate(self):
        gw = SafetyGateway()
        gw.add_input_check("check", length_check)
        gw.process(input_text="short")          # pass
        gw.process(input_text="x" * 200)        # fail
        stats = gw.get_stats()
        assert stats.input_pass_rate == 0.5

    def test_output_pass_rate(self):
        gw = SafetyGateway()
        gw.add_output_check("tox", toxicity_check)
        gw.process(output_text="fine")           # pass
        gw.process(output_text="hate speech")    # fail
        gw.process(output_text="also fine")      # pass
        stats = gw.get_stats()
        assert abs(stats.output_pass_rate - 2.0 / 3.0) < 1e-9

    def test_stats_default_rates_with_no_requests(self):
        gw = SafetyGateway()
        stats = gw.get_stats()
        assert stats.total_requests == 0
        assert stats.input_pass_rate == 1.0
        assert stats.output_pass_rate == 1.0


# ---------------------------------------------------------------------------
# Check queries
# ---------------------------------------------------------------------------

class TestCheckQueries:
    def test_check_count(self):
        gw = SafetyGateway()
        gw.add_input_check("a", always_pass)
        gw.add_output_check("b", always_pass)
        assert gw.check_count == 2

    def test_enabled_check_count(self):
        gw = SafetyGateway()
        gw.add_input_check("a", always_pass, enabled=True)
        gw.add_input_check("b", always_pass, enabled=False)
        assert gw.enabled_check_count == 1

    def test_get_checks_by_phase(self):
        gw = SafetyGateway()
        gw.add_input_check("in1", always_pass)
        gw.add_output_check("out1", always_pass)
        assert len(gw.get_checks("input")) == 1
        assert len(gw.get_checks("output")) == 1
        assert len(gw.get_checks()) == 2


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_gateway_check_result_fields(self):
        result = GatewayCheckResult(
            name="test", passed=True, message="OK", phase="input",
        )
        assert result.name == "test"
        assert result.passed is True
        assert result.message == "OK"
        assert result.phase == "input"

    def test_gateway_result_and_stats_fields(self):
        result = GatewayResult(passed=True, short_circuited=False)
        assert result.passed is True
        assert result.input_results == []
        assert result.output_results == []
        assert result.short_circuited is False

        stats = GatewayStats(
            total_requests=10, input_pass_rate=0.9, output_pass_rate=0.8,
        )
        assert stats.total_requests == 10
        assert stats.input_pass_rate == 0.9
        assert stats.output_pass_rate == 0.8
