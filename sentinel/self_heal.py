"""Self-healing structured output with automatic retry.

When an LLM produces invalid structured output, this module automatically
retries with a corrective prompt that includes the validation errors,
dramatically improving reliability of structured generation.

Usage:
    from sentinel.self_heal import StructuredGenerator

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    gen = StructuredGenerator(schema, call_llm=my_llm_function)
    result = gen.generate("Extract name and age from: Alice is 30 years old")
    print(result.parsed)  # {"name": "Alice", "age": 30}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from sentinel.output_validator import OutputValidator, ValidationResult


@dataclass
class GenerationResult:
    """Result of a structured generation attempt."""
    parsed: Any
    valid: bool
    attempts: int
    validation: ValidationResult
    history: list[AttemptRecord] = field(default_factory=list)

    @property
    def healed(self) -> bool:
        """True if the output required retries to become valid."""
        return self.valid and self.attempts > 1


@dataclass
class AttemptRecord:
    """Record of a single generation attempt."""
    attempt: int
    raw_output: str
    validation: ValidationResult
    corrective_prompt: str | None = None


class StructuredGenerator:
    """Generate validated structured output with automatic retry.

    Wraps any LLM call function and ensures the output matches a JSON schema.
    On validation failure, constructs a corrective prompt with the specific
    errors and retries, up to max_retries times.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        call_llm: Callable[[str], str],
        max_retries: int = 3,
        strict: bool = False,
    ):
        """
        Args:
            schema: JSON Schema the output must conform to.
            call_llm: Function that takes a prompt string and returns LLM output.
            max_retries: Maximum number of retry attempts (default 3).
            strict: If True, reject additional properties not in schema.
        """
        self._schema = schema
        self._call_llm = call_llm
        self._max_retries = max_retries
        self._validator = OutputValidator(schema, strict=strict)

    def generate(self, prompt: str) -> GenerationResult:
        """Generate structured output, retrying on validation failure."""
        history: list[AttemptRecord] = []
        current_prompt = self._build_initial_prompt(prompt)

        for attempt in range(1, self._max_retries + 2):  # +2: 1 initial + max_retries
            raw_output = self._call_llm(current_prompt)
            validation = self._validator.validate(raw_output)

            record = AttemptRecord(
                attempt=attempt,
                raw_output=raw_output,
                validation=validation,
            )
            history.append(record)

            if validation.valid:
                return GenerationResult(
                    parsed=validation.parsed,
                    valid=True,
                    attempts=attempt,
                    validation=validation,
                    history=history,
                )

            if attempt <= self._max_retries:
                corrective = self._build_corrective_prompt(
                    prompt, raw_output, validation
                )
                record.corrective_prompt = corrective
                current_prompt = corrective

        # All retries exhausted
        last_validation = history[-1].validation
        return GenerationResult(
            parsed=last_validation.parsed,
            valid=False,
            attempts=len(history),
            validation=last_validation,
            history=history,
        )

    def _build_initial_prompt(self, user_prompt: str) -> str:
        """Add schema instructions to the user's prompt."""
        schema_str = json.dumps(self._schema, indent=2)
        return (
            f"{user_prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"```json\n{schema_str}\n```\n"
            f"Output ONLY the JSON, no other text."
        )

    def _build_corrective_prompt(
        self,
        original_prompt: str,
        failed_output: str,
        validation: ValidationResult,
    ) -> str:
        """Build a corrective prompt with specific error details."""
        error_details = "\n".join(
            f"- {e.path}: {e.message}" for e in validation.errors
        )
        schema_str = json.dumps(self._schema, indent=2)

        return (
            f"{original_prompt}\n\n"
            f"Your previous response had validation errors:\n"
            f"{error_details}\n\n"
            f"Your invalid output was:\n"
            f"```\n{failed_output}\n```\n\n"
            f"Please fix these errors and respond with valid JSON matching:\n"
            f"```json\n{schema_str}\n```\n"
            f"Output ONLY the corrected JSON, no other text."
        )
