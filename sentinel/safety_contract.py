"""Design by Contract safety enforcement for LLM interactions.

Define preconditions on inputs, postconditions on outputs, and invariants
that must hold throughout — then validate LLM interactions against them.

Usage:
    from sentinel.safety_contract import SafetyContract, Condition

    contract = SafetyContract()
    contract.add_precondition(Condition(
        name="non_empty_input",
        check_fn=lambda text: len(text.strip()) > 0,
        message="Input must not be empty",
        severity="error",
    ))
    contract.add_postcondition(Condition(
        name="no_pii_in_output",
        check_fn=lambda text: "SSN" not in text,
        message="Output must not contain PII references",
        severity="error",
    ))
    report = contract.validate("Hello", "The answer is 42")
    assert report.all_passed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Condition:
    """A single contract condition (pre, post, or invariant)."""

    name: str
    check_fn: Callable[[str], bool]
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class Violation:
    """A failed contract condition."""

    condition_name: str
    message: str
    severity: str
    phase: str  # "pre", "post", or "invariant"


@dataclass
class ContractResult:
    """Result of evaluating conditions for a single phase."""

    passed: bool
    violations: list[Violation] = field(default_factory=list)
    phase: str = ""


@dataclass
class ContractReport:
    """Full report across all contract phases."""

    input_result: ContractResult
    output_result: ContractResult | None = None
    invariant_result: ContractResult | None = None
    all_passed: bool = True


class SafetyContract:
    """Define and enforce safety contracts on LLM interactions.

    Supports preconditions (checked on input), postconditions (checked
    on output), and invariants (checked on both input and output).
    """

    def __init__(self) -> None:
        self._preconditions: list[Condition] = []
        self._postconditions: list[Condition] = []
        self._invariants: list[Condition] = []

    def add_precondition(self, condition: Condition) -> SafetyContract:
        """Add a precondition that input must satisfy."""
        self._preconditions.append(condition)
        return self

    def add_postcondition(self, condition: Condition) -> SafetyContract:
        """Add a postcondition that output must satisfy."""
        self._postconditions.append(condition)
        return self

    def add_invariant(self, condition: Condition) -> SafetyContract:
        """Add an invariant that both input and output must satisfy."""
        self._invariants.append(condition)
        return self

    @property
    def preconditions(self) -> list[Condition]:
        return list(self._preconditions)

    @property
    def postconditions(self) -> list[Condition]:
        return list(self._postconditions)

    @property
    def invariants(self) -> list[Condition]:
        return list(self._invariants)

    def check_preconditions(self, input_text: str) -> ContractResult:
        """Validate input against all preconditions."""
        return self._evaluate_conditions(self._preconditions, input_text, "pre")

    def check_postconditions(self, output_text: str) -> ContractResult:
        """Validate output against all postconditions."""
        return self._evaluate_conditions(self._postconditions, output_text, "post")

    def check_invariants(self, text: str) -> ContractResult:
        """Validate text against all invariants."""
        return self._evaluate_conditions(self._invariants, text, "invariant")

    def validate_input(self, input_text: str) -> ContractReport:
        """Validate input against preconditions and invariants."""
        input_result = self.check_preconditions(input_text)
        invariant_result = self.check_invariants(input_text)
        all_passed = input_result.passed and invariant_result.passed
        return ContractReport(
            input_result=input_result,
            invariant_result=invariant_result,
            all_passed=all_passed,
        )

    def validate_output(self, output_text: str) -> ContractReport:
        """Validate output against postconditions and invariants."""
        output_result = self.check_postconditions(output_text)
        invariant_result = self.check_invariants(output_text)
        all_passed = output_result.passed and invariant_result.passed
        return ContractReport(
            input_result=ContractResult(passed=True, phase="pre"),
            output_result=output_result,
            invariant_result=invariant_result,
            all_passed=all_passed,
        )

    def validate(self, input_text: str, output_text: str) -> ContractReport:
        """Full contract validation: pre + post + invariants on both."""
        input_result = self.check_preconditions(input_text)
        output_result = self.check_postconditions(output_text)

        input_invariant = self._evaluate_conditions(
            self._invariants, input_text, "invariant",
        )
        output_invariant = self._evaluate_conditions(
            self._invariants, output_text, "invariant",
        )
        invariant_result = self._merge_results(
            input_invariant, output_invariant,
        )

        all_passed = (
            input_result.passed
            and output_result.passed
            and invariant_result.passed
        )
        return ContractReport(
            input_result=input_result,
            output_result=output_result,
            invariant_result=invariant_result,
            all_passed=all_passed,
        )

    def compose(self, other: SafetyContract) -> SafetyContract:
        """Combine this contract with another, returning a new contract."""
        combined = SafetyContract()
        combined._preconditions = self._preconditions + other._preconditions
        combined._postconditions = self._postconditions + other._postconditions
        combined._invariants = self._invariants + other._invariants
        return combined

    def _evaluate_conditions(
        self,
        conditions: list[Condition],
        text: str,
        phase: str,
    ) -> ContractResult:
        """Run a list of conditions against text and collect violations."""
        violations: list[Violation] = []
        for condition in conditions:
            if not condition.check_fn(text):
                violations.append(Violation(
                    condition_name=condition.name,
                    message=condition.message,
                    severity=condition.severity,
                    phase=phase,
                ))
        return ContractResult(
            passed=len(violations) == 0,
            violations=violations,
            phase=phase,
        )

    def _merge_results(
        self,
        first: ContractResult,
        second: ContractResult,
    ) -> ContractResult:
        """Merge two ContractResults from the same phase."""
        violations = first.violations + second.violations
        return ContractResult(
            passed=len(violations) == 0,
            violations=violations,
            phase=first.phase,
        )
