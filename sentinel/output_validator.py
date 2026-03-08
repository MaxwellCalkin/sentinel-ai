"""Output schema validator for structured LLM responses.

Validates that LLM outputs conform to expected JSON schemas, catching
malformed responses, missing fields, type mismatches, and constraint
violations before they propagate to downstream systems.

Usage:
    from sentinel.output_validator import OutputValidator

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0},
        },
        "required": ["name", "age"],
    }
    validator = OutputValidator(schema)
    result = validator.validate('{"name": "Alice", "age": 30}')
    assert result.valid
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationError:
    """A single validation error."""
    path: str
    message: str
    expected: str | None = None
    actual: str | None = None


@dataclass
class ValidationResult:
    """Result of output validation."""
    valid: bool
    parsed: Any = None
    errors: list[ValidationError] = field(default_factory=list)
    raw_text: str = ""

    @property
    def error_summary(self) -> str:
        if self.valid:
            return "Valid"
        return "; ".join(f"{e.path}: {e.message}" for e in self.errors)


class OutputValidator:
    """Validate LLM output against a JSON schema (subset).

    Supports a practical subset of JSON Schema:
    - type: object, array, string, number, integer, boolean, null
    - required: list of required property names
    - properties: property schemas
    - items: array item schema
    - enum: allowed values
    - minLength, maxLength: string constraints
    - minimum, maximum: number constraints
    - minItems, maxItems: array length constraints
    - pattern: regex pattern for strings
    """

    def __init__(self, schema: dict[str, Any], strict: bool = False):
        """
        Args:
            schema: JSON Schema to validate against.
            strict: If True, reject additional properties not in schema.
        """
        self._schema = schema
        self._strict = strict

    def validate(self, text: str) -> ValidationResult:
        """Validate text as JSON against the schema."""
        # Try to extract JSON from the text
        parsed, parse_error = self._parse_json(text)

        if parse_error:
            return ValidationResult(
                valid=False,
                errors=[ValidationError("$", f"Invalid JSON: {parse_error}")],
                raw_text=text,
            )

        errors = self._validate_value(parsed, self._schema, "$")

        return ValidationResult(
            valid=len(errors) == 0,
            parsed=parsed,
            errors=errors,
            raw_text=text,
        )

    def _parse_json(self, text: str) -> tuple[Any, str | None]:
        """Try to parse JSON, with fallback to extract JSON from markdown."""
        text = text.strip()

        # Direct parse
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if md_match:
            try:
                return json.loads(md_match.group(1).strip()), None
            except json.JSONDecodeError as e:
                return None, str(e)

        # Try finding JSON object/array in text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            if start >= 0:
                # Find matching end
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == start_char:
                        depth += 1
                    elif text[i] == end_char:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start:i + 1]), None
                            except json.JSONDecodeError:
                                break

        return None, "Could not parse JSON from text"

    def _validate_value(
        self, value: Any, schema: dict[str, Any], path: str
    ) -> list[ValidationError]:
        """Validate a value against a schema node."""
        errors: list[ValidationError] = []

        # Type check
        if "type" in schema:
            type_errors = self._check_type(value, schema["type"], path)
            if type_errors:
                return type_errors  # Skip further validation if type is wrong

        # Enum
        if "enum" in schema:
            if value not in schema["enum"]:
                errors.append(ValidationError(
                    path, f"Value not in enum",
                    expected=str(schema["enum"]),
                    actual=str(value),
                ))

        # String constraints
        if isinstance(value, str):
            errors.extend(self._validate_string(value, schema, path))

        # Number constraints
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            errors.extend(self._validate_number(value, schema, path))

        # Object constraints
        if isinstance(value, dict):
            errors.extend(self._validate_object(value, schema, path))

        # Array constraints
        if isinstance(value, list):
            errors.extend(self._validate_array(value, schema, path))

        return errors

    def _check_type(
        self, value: Any, expected_type: str, path: str
    ) -> list[ValidationError]:
        """Check if value matches the expected JSON Schema type."""
        type_map = {
            "object": dict,
            "array": list,
            "string": str,
            "boolean": bool,
            "null": type(None),
        }

        if expected_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return [ValidationError(
                    path, f"Expected integer, got {type(value).__name__}",
                    expected="integer",
                    actual=type(value).__name__,
                )]
        elif expected_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return [ValidationError(
                    path, f"Expected number, got {type(value).__name__}",
                    expected="number",
                    actual=type(value).__name__,
                )]
        elif expected_type in type_map:
            if not isinstance(value, type_map[expected_type]):
                return [ValidationError(
                    path, f"Expected {expected_type}, got {type(value).__name__}",
                    expected=expected_type,
                    actual=type(value).__name__,
                )]

        return []

    def _validate_string(
        self, value: str, schema: dict, path: str
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []

        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(ValidationError(
                path, f"String too short: {len(value)} < {schema['minLength']}",
            ))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(ValidationError(
                path, f"String too long: {len(value)} > {schema['maxLength']}",
            ))
        if "pattern" in schema:
            if not re.search(schema["pattern"], value):
                errors.append(ValidationError(
                    path, f"String does not match pattern: {schema['pattern']}",
                ))

        return errors

    def _validate_number(
        self, value: int | float, schema: dict, path: str
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []

        if "minimum" in schema and value < schema["minimum"]:
            errors.append(ValidationError(
                path, f"Value {value} < minimum {schema['minimum']}",
            ))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(ValidationError(
                path, f"Value {value} > maximum {schema['maximum']}",
            ))

        return errors

    def _validate_object(
        self, value: dict, schema: dict, path: str
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []

        # Required fields
        for req in schema.get("required", []):
            if req not in value:
                errors.append(ValidationError(
                    f"{path}.{req}", f"Required field missing",
                ))

        # Property validation
        properties = schema.get("properties", {})
        for prop, prop_schema in properties.items():
            if prop in value:
                errors.extend(
                    self._validate_value(value[prop], prop_schema, f"{path}.{prop}")
                )

        # Strict mode: reject extra properties
        if self._strict and properties:
            for key in value:
                if key not in properties:
                    errors.append(ValidationError(
                        f"{path}.{key}", "Additional property not allowed in strict mode",
                    ))

        return errors

    def _validate_array(
        self, value: list, schema: dict, path: str
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []

        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(ValidationError(
                path, f"Array too short: {len(value)} < {schema['minItems']}",
            ))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(ValidationError(
                path, f"Array too long: {len(value)} > {schema['maxItems']}",
            ))

        # Validate items
        if "items" in schema:
            for i, item in enumerate(value):
                errors.extend(
                    self._validate_value(item, schema["items"], f"{path}[{i}]")
                )

        return errors
