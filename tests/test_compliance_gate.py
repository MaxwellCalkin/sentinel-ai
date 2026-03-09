"""Tests for pre-deployment compliance gating."""

import pytest
from sentinel.compliance_gate import (
    ComplianceCheck,
    ComplianceGate,
    CheckOutcome,
    GateDecision,
    GateStats,
)


# ---------------------------------------------------------------------------
# Check Registration
# ---------------------------------------------------------------------------

class TestCheckRegistration:
    def test_add_required_check(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck(
            name="version_tag",
            check_fn=lambda: True,
            severity="required",
            description="Model must be versioned",
        ))
        assert len(gate.checks) == 1
        assert gate.checks[0].name == "version_tag"
        assert gate.checks[0].severity == "required"

    def test_add_recommended_check(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck(
            name="docs_present",
            check_fn=lambda: True,
            severity="recommended",
            description="Documentation should exist",
        ))
        assert gate.checks[0].severity == "recommended"

    def test_invalid_severity_rejected(self):
        gate = ComplianceGate()
        with pytest.raises(ValueError, match="Invalid severity"):
            gate.add_check(ComplianceCheck(
                name="bad", check_fn=lambda: True, severity="optional",
            ))

    def test_duplicate_name_rejected(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck(
            name="unique", check_fn=lambda: True, severity="required",
        ))
        with pytest.raises(ValueError, match="Duplicate check name"):
            gate.add_check(ComplianceCheck(
                name="unique", check_fn=lambda: False, severity="required",
            ))


# ---------------------------------------------------------------------------
# Gate Evaluation — Pass / Fail Logic
# ---------------------------------------------------------------------------

class TestGateDecision:
    def test_all_required_pass(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("a", lambda: True, "required"))
        gate.add_check(ComplianceCheck("b", lambda: True, "required"))
        decision = gate.evaluate()
        assert decision.passed is True
        assert decision.required_passed == 2
        assert decision.required_failed == 0

    def test_required_failure_blocks_gate(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("pass", lambda: True, "required"))
        gate.add_check(ComplianceCheck("fail", lambda: False, "required"))
        decision = gate.evaluate()
        assert decision.passed is False
        assert decision.required_failed == 1

    def test_recommended_failure_does_not_block(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("req", lambda: True, "required"))
        gate.add_check(ComplianceCheck("rec", lambda: False, "recommended"))
        decision = gate.evaluate()
        assert decision.passed is True
        assert decision.required_passed == 1
        assert decision.required_failed == 0

    def test_recommended_failure_produces_warning(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("docs", lambda: False, "recommended"))
        decision = gate.evaluate()
        assert len(decision.warnings) == 1
        assert "docs" in decision.warnings[0]

    def test_no_warnings_when_all_pass(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("a", lambda: True, "required"))
        gate.add_check(ComplianceCheck("b", lambda: True, "recommended"))
        decision = gate.evaluate()
        assert decision.warnings == []

    def test_empty_gate_passes(self):
        gate = ComplianceGate()
        decision = gate.evaluate()
        assert decision.passed is True
        assert decision.outcomes == []
        assert decision.required_passed == 0
        assert decision.required_failed == 0


# ---------------------------------------------------------------------------
# Check Outcomes
# ---------------------------------------------------------------------------

class TestCheckOutcomes:
    def test_outcome_records_timing(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("slow", lambda: True, "required"))
        decision = gate.evaluate()
        assert decision.outcomes[0].duration_ms >= 0

    def test_outcome_message_reflects_result(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("ok", lambda: True, "required"))
        gate.add_check(ComplianceCheck("nope", lambda: False, "recommended"))
        decision = gate.evaluate()
        assert decision.outcomes[0].message == "passed"
        assert decision.outcomes[1].message == "failed"

    def test_exception_in_check_treated_as_failure(self):
        def boom():
            raise RuntimeError("broken")

        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("boom", boom, "required"))
        decision = gate.evaluate()
        assert decision.passed is False
        assert decision.outcomes[0].passed is False
        assert "Exception" in decision.outcomes[0].message
        assert "broken" in decision.outcomes[0].message


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_records_evaluations(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("a", lambda: True, "required"))
        gate.evaluate()
        gate.evaluate()
        assert len(gate.history) == 2

    def test_history_returns_copy(self):
        gate = ComplianceGate()
        gate.evaluate()
        history = gate.history
        history.clear()
        assert len(gate.history) == 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_dict_structure(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("v", lambda: True, "required"))
        gate.evaluate()
        exported = gate.export_dict()
        assert exported["passed"] is True
        assert isinstance(exported["outcomes"], list)
        assert exported["outcomes"][0]["name"] == "v"
        assert "timestamp" in exported
        assert "required_passed" in exported
        assert "required_failed" in exported
        assert "warnings" in exported

    def test_export_before_evaluate_raises(self):
        gate = ComplianceGate()
        with pytest.raises(ValueError, match="No evaluations"):
            gate.export_dict()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_pass_rate(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("a", lambda: True, "required"))
        gate.evaluate()
        gate.evaluate()
        stats = gate.stats()
        assert stats.total_evaluations == 2
        assert stats.pass_count == 2
        assert stats.pass_rate == 1.0

    def test_stats_tracks_failed_checks(self):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            return call_count > 1

        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("flaky", flaky, "required"))
        gate.evaluate()  # fails (call_count == 1)
        gate.evaluate()  # passes (call_count == 2)
        stats = gate.stats()
        assert stats.total_evaluations == 2
        assert stats.pass_count == 1
        assert stats.pass_rate == 0.5
        assert stats.failed_checks == {"flaky": 1}

    def test_stats_empty_history(self):
        gate = ComplianceGate()
        stats = gate.stats()
        assert stats.total_evaluations == 0
        assert stats.pass_count == 0
        assert stats.pass_rate == 0.0
        assert stats.failed_checks == {}


# ---------------------------------------------------------------------------
# Remove Check
# ---------------------------------------------------------------------------

class TestRemoveCheck:
    def test_remove_existing_check(self):
        gate = ComplianceGate()
        gate.add_check(ComplianceCheck("a", lambda: True, "required"))
        gate.add_check(ComplianceCheck("b", lambda: True, "required"))
        gate.remove_check("a")
        assert len(gate.checks) == 1
        assert gate.checks[0].name == "b"

    def test_remove_nonexistent_raises(self):
        gate = ComplianceGate()
        with pytest.raises(KeyError, match="No check named"):
            gate.remove_check("ghost")
