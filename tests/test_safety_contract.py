"""Tests for safety contract enforcement."""

from sentinel.safety_contract import (
    SafetyContract,
    Condition,
    Violation,
    ContractResult,
    ContractReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _non_empty_condition() -> Condition:
    return Condition(
        name="non_empty",
        check_fn=lambda text: len(text.strip()) > 0,
        message="Text must not be empty",
        severity="error",
    )


def _max_length_condition(limit: int) -> Condition:
    return Condition(
        name="max_length",
        check_fn=lambda text: len(text) <= limit,
        message=f"Text must be at most {limit} characters",
        severity="warning",
    )


def _no_pii_condition() -> Condition:
    return Condition(
        name="no_pii",
        check_fn=lambda text: "SSN" not in text.upper(),
        message="Must not contain PII references",
        severity="error",
    )


def _english_only_condition() -> Condition:
    return Condition(
        name="english_only",
        check_fn=lambda text: all(ord(c) < 128 for c in text),
        message="Must contain only ASCII characters",
        severity="warning",
    )


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

class TestPreconditions:
    def test_passing_precondition(self):
        contract = SafetyContract()
        contract.add_precondition(_non_empty_condition())
        result = contract.check_preconditions("Hello")
        assert result.passed
        assert result.violations == []
        assert result.phase == "pre"

    def test_failing_precondition(self):
        contract = SafetyContract()
        contract.add_precondition(_non_empty_condition())
        result = contract.check_preconditions("   ")
        assert not result.passed
        assert len(result.violations) == 1
        assert result.violations[0].condition_name == "non_empty"
        assert result.violations[0].severity == "error"
        assert result.violations[0].phase == "pre"

    def test_multiple_preconditions_all_fail(self):
        contract = SafetyContract()
        contract.add_precondition(_non_empty_condition())
        contract.add_precondition(_max_length_condition(3))
        result = contract.check_preconditions("    ")
        assert not result.passed
        # Empty after strip AND len("    ") > 3
        assert len(result.violations) == 2


# ---------------------------------------------------------------------------
# Postconditions
# ---------------------------------------------------------------------------

class TestPostconditions:
    def test_passing_postcondition(self):
        contract = SafetyContract()
        contract.add_postcondition(_no_pii_condition())
        result = contract.check_postconditions("The answer is 42")
        assert result.passed
        assert result.phase == "post"

    def test_failing_postcondition(self):
        contract = SafetyContract()
        contract.add_postcondition(_no_pii_condition())
        result = contract.check_postconditions("Your SSN is 123-45-6789")
        assert not result.passed
        assert result.violations[0].condition_name == "no_pii"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_passing_invariant(self):
        contract = SafetyContract()
        contract.add_invariant(_english_only_condition())
        result = contract.check_invariants("Hello world")
        assert result.passed
        assert result.phase == "invariant"

    def test_failing_invariant(self):
        contract = SafetyContract()
        contract.add_invariant(_english_only_condition())
        result = contract.check_invariants("Caf\u00e9")
        assert not result.passed
        assert result.violations[0].phase == "invariant"


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------

class TestFullValidation:
    def test_all_pass(self):
        contract = SafetyContract()
        contract.add_precondition(_non_empty_condition())
        contract.add_postcondition(_no_pii_condition())
        contract.add_invariant(_english_only_condition())
        report = contract.validate("Hello", "The answer is 42")
        assert report.all_passed
        assert report.input_result.passed
        assert report.output_result.passed
        assert report.invariant_result.passed

    def test_precondition_fails(self):
        contract = SafetyContract()
        contract.add_precondition(_non_empty_condition())
        contract.add_postcondition(_no_pii_condition())
        report = contract.validate("   ", "The answer is 42")
        assert not report.all_passed
        assert not report.input_result.passed
        assert report.output_result.passed

    def test_postcondition_fails(self):
        contract = SafetyContract()
        contract.add_precondition(_non_empty_condition())
        contract.add_postcondition(_no_pii_condition())
        report = contract.validate("Hello", "Your SSN is 123-45-6789")
        assert not report.all_passed
        assert report.input_result.passed
        assert not report.output_result.passed

    def test_invariant_fails_on_output(self):
        contract = SafetyContract()
        contract.add_invariant(_english_only_condition())
        report = contract.validate("Hello", "Caf\u00e9")
        assert not report.all_passed
        assert report.input_result.passed
        assert report.invariant_result is not None
        assert not report.invariant_result.passed


# ---------------------------------------------------------------------------
# Input-only and output-only validation
# ---------------------------------------------------------------------------

class TestPartialValidation:
    def test_validate_input_passes(self):
        contract = SafetyContract()
        contract.add_precondition(_non_empty_condition())
        contract.add_invariant(_english_only_condition())
        report = contract.validate_input("Hello")
        assert report.all_passed
        assert report.output_result is None

    def test_validate_input_fails(self):
        contract = SafetyContract()
        contract.add_precondition(_non_empty_condition())
        report = contract.validate_input("   ")
        assert not report.all_passed

    def test_validate_output_passes(self):
        contract = SafetyContract()
        contract.add_postcondition(_no_pii_condition())
        report = contract.validate_output("Safe text")
        assert report.all_passed
        assert report.output_result is not None
        assert report.output_result.passed


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class TestComposition:
    def test_compose_merges_conditions_and_validates(self):
        first = SafetyContract()
        first.add_precondition(_non_empty_condition())

        second = SafetyContract()
        second.add_postcondition(_no_pii_condition())
        second.add_invariant(_english_only_condition())

        combined = first.compose(second)
        assert len(combined.preconditions) == 1
        assert len(combined.postconditions) == 1
        assert len(combined.invariants) == 1

        report = combined.validate("Hello", "Safe answer")
        assert report.all_passed

    def test_composed_contract_catches_violations(self):
        first = SafetyContract()
        first.add_precondition(_non_empty_condition())

        second = SafetyContract()
        second.add_postcondition(_no_pii_condition())

        combined = first.compose(second)
        report = combined.validate("  ", "Your SSN is leaked")
        assert not report.all_passed
        assert not report.input_result.passed
        assert not report.output_result.passed


# ---------------------------------------------------------------------------
# Fluent API (chaining)
# ---------------------------------------------------------------------------

class TestFluentAPI:
    def test_chained_add_returns_self(self):
        contract = SafetyContract()
        result = (
            contract
            .add_precondition(_non_empty_condition())
            .add_postcondition(_no_pii_condition())
            .add_invariant(_english_only_condition())
        )
        assert result is contract
        assert len(contract.preconditions) == 1
        assert len(contract.postconditions) == 1
        assert len(contract.invariants) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_contract_always_passes(self):
        contract = SafetyContract()
        report = contract.validate("anything", "anything else")
        assert report.all_passed

    def test_severity_preserved_in_violation(self):
        contract = SafetyContract()
        contract.add_precondition(_max_length_condition(2))
        result = contract.check_preconditions("too long")
        assert result.violations[0].severity == "warning"

    def test_violation_message_preserved(self):
        contract = SafetyContract()
        contract.add_precondition(_non_empty_condition())
        result = contract.check_preconditions("")
        assert result.violations[0].message == "Text must not be empty"


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_condition_and_violation_fields(self):
        cond = Condition(
            name="test",
            check_fn=lambda t: True,
            message="test message",
            severity="warning",
        )
        assert cond.name == "test"
        assert cond.message == "test message"
        assert cond.severity == "warning"
        assert cond.check_fn("anything")

        violation = Violation(
            condition_name="rule_x",
            message="broke it",
            severity="error",
            phase="post",
        )
        assert violation.condition_name == "rule_x"
        assert violation.phase == "post"

    def test_contract_result_fields(self):
        result = ContractResult(passed=False, violations=[], phase="pre")
        assert not result.passed
        assert result.phase == "pre"
