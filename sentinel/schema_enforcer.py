"""Schema enforcer for structured LLM output compliance.

Validates that LLM-generated data conforms to JSON schemas and attempts
automatic coercion when types don't match exactly. Useful for ensuring
structured output from models meets application requirements before
downstream consumption.

Usage:
    from sentinel.schema_enforcer import SchemaEnforcer

    enforcer = SchemaEnforcer()
    enforcer.register_schema("user", {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0},
        },
        "required": ["name", "age"],
    })
    result = enforcer.validate_against({"name": "Alice", "age": 30}, "user")
    assert result.valid
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


@dataclass
class EnforcementResult:
    """Result of validating data against a JSON schema."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    path: str = ""


@dataclass
class CoercionResult:
    """Result of attempting to coerce data to match a schema."""

    success: bool
    original: Any = None
    coerced: Any = None
    changes: list[str] = field(default_factory=list)


class SchemaEnforcer:
    """Validates and coerces LLM outputs against JSON schemas.

    Supports a practical subset of JSON Schema: type, required, properties,
    items, enum, minLength, maxLength, minimum, maximum, and pattern.
    """

    def __init__(self, strict: bool = False) -> None:
        """Initialize the schema enforcer.

        Args:
            strict: When True, warnings are promoted to errors.
        """
        self._strict = strict
        self._schemas: dict[str, dict] = {}

    def validate(self, data: Any, schema: dict) -> EnforcementResult:
        """Validate data against a JSON schema.

        Args:
            data: The Python value to validate.
            schema: A JSON schema dict.

        Returns:
            EnforcementResult with validation outcome.
        """
        errors: list[str] = []
        warnings: list[str] = []
        self._validate_node(data, schema, "$", errors, warnings)

        if self._strict:
            errors.extend(warnings)
            warnings = []

        first_error_path = self._find_first_error_path(data, schema)
        return EnforcementResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            path=first_error_path,
        )

    def coerce(self, data: Any, schema: dict) -> CoercionResult:
        """Attempt to coerce data to match a schema.

        Supports string-to-int, string-to-float, string-to-bool,
        int-to-string, and wrapping a single item into an array.

        Args:
            data: The Python value to coerce.
            schema: A JSON schema dict.

        Returns:
            CoercionResult describing what changed.
        """
        changes: list[str] = []
        coerced = self._coerce_node(data, schema, "$", changes)

        check_result = self.validate(coerced, schema)
        return CoercionResult(
            success=check_result.valid,
            original=data,
            coerced=coerced,
            changes=changes,
        )

    def register_schema(self, name: str, schema: dict) -> None:
        """Register a named schema for later use.

        Args:
            name: Unique name for the schema.
            schema: A JSON schema dict.
        """
        self._schemas[name] = schema

    def validate_against(self, data: Any, schema_name: str) -> EnforcementResult:
        """Validate data against a previously registered schema.

        Args:
            data: The Python value to validate.
            schema_name: Name of a registered schema.

        Returns:
            EnforcementResult with validation outcome.

        Raises:
            KeyError: If schema_name is not registered.
        """
        if schema_name not in self._schemas:
            raise KeyError(f"Schema '{schema_name}' is not registered")
        return self.validate(data, self._schemas[schema_name])

    def list_schemas(self) -> list[str]:
        """Return the names of all registered schemas."""
        return list(self._schemas.keys())

    # ------------------------------------------------------------------
    # Validation internals
    # ------------------------------------------------------------------

    def _validate_node(
        self,
        value: Any,
        schema: dict,
        path: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Recursively validate a value against a schema node."""
        if "type" in schema:
            if not self._type_matches(value, schema["type"]):
                errors.append(
                    f"{path}: expected type '{schema['type']}', "
                    f"got '{type(value).__name__}'"
                )
                return

        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: value {value!r} not in enum {schema['enum']}")

        if isinstance(value, str):
            self._validate_string_constraints(value, schema, path, errors)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self._validate_number_constraints(value, schema, path, errors)

        if isinstance(value, dict):
            self._validate_object(value, schema, path, errors, warnings)

        if isinstance(value, list):
            self._validate_array(value, schema, path, errors, warnings)

    def _type_matches(self, value: Any, expected_type: str) -> bool:
        """Check whether a value matches the expected JSON Schema type."""
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        expected_python_type = _TYPE_MAP.get(expected_type)
        if expected_python_type is None:
            return True
        # bool is a subclass of int; "object" and "array" shouldn't match bool
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "boolean":
            return isinstance(value, bool)
        return isinstance(value, expected_python_type) and not isinstance(value, bool)

    def _validate_string_constraints(
        self,
        value: str,
        schema: dict,
        path: str,
        errors: list[str],
    ) -> None:
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(
                f"{path}: string length {len(value)} < minLength {schema['minLength']}"
            )
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(
                f"{path}: string length {len(value)} > maxLength {schema['maxLength']}"
            )
        if "pattern" in schema and not re.match(schema["pattern"], value):
            errors.append(
                f"{path}: string does not match pattern '{schema['pattern']}'"
            )

    def _validate_number_constraints(
        self,
        value: int | float,
        schema: dict,
        path: str,
        errors: list[str],
    ) -> None:
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value {value} > maximum {schema['maximum']}")

    def _validate_object(
        self,
        value: dict,
        schema: dict,
        path: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        for required_field in schema.get("required", []):
            if required_field not in value:
                errors.append(f"{path}.{required_field}: required field missing")

        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name in value:
                self._validate_node(
                    value[prop_name], prop_schema, f"{path}.{prop_name}", errors, warnings
                )

        extra_keys = set(value.keys()) - set(properties.keys())
        if extra_keys and properties:
            for key in sorted(extra_keys):
                warnings.append(f"{path}.{key}: additional property not in schema")

    def _validate_array(
        self,
        value: list,
        schema: dict,
        path: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        if "items" in schema:
            for index, item in enumerate(value):
                self._validate_node(
                    item, schema["items"], f"{path}[{index}]", errors, warnings
                )

    def _find_first_error_path(self, data: Any, schema: dict) -> str:
        """Return the JSON path of the first error, or empty string if valid."""
        errors: list[str] = []
        warnings: list[str] = []
        self._validate_node(data, schema, "$", errors, warnings)
        if self._strict:
            errors.extend(warnings)
        if not errors:
            return ""
        # The error message starts with the path followed by ':'
        first_error = errors[0]
        colon_index = first_error.find(":")
        if colon_index > 0:
            return first_error[:colon_index]
        return "$"

    # ------------------------------------------------------------------
    # Coercion internals
    # ------------------------------------------------------------------

    def _coerce_node(
        self, value: Any, schema: dict, path: str, changes: list[str]
    ) -> Any:
        """Attempt to coerce a value to match a schema node."""
        expected_type = schema.get("type")
        if expected_type is None:
            return value

        coerced = self._coerce_type(value, expected_type, path, changes)

        if expected_type == "object" and isinstance(coerced, dict):
            coerced = self._coerce_object(coerced, schema, path, changes)

        if expected_type == "array" and isinstance(coerced, list):
            coerced = self._coerce_array(coerced, schema, path, changes)

        return coerced

    def _coerce_type(
        self, value: Any, expected_type: str, path: str, changes: list[str]
    ) -> Any:
        """Coerce a single value to the expected type if possible."""
        if self._type_matches(value, expected_type):
            return value

        if expected_type == "integer" and isinstance(value, str):
            return self._try_coerce(value, int, path, "string", "integer", changes)

        if expected_type == "number" and isinstance(value, str):
            return self._try_coerce(value, float, path, "string", "number", changes)

        if expected_type == "boolean" and isinstance(value, str):
            return self._coerce_string_to_bool(value, path, changes)

        if expected_type == "string" and isinstance(value, int) and not isinstance(value, bool):
            changes.append(f"{path}: coerced int to string")
            return str(value)

        if expected_type == "array" and not isinstance(value, list):
            changes.append(f"{path}: wrapped single value in array")
            return [value]

        return value

    def _try_coerce(
        self,
        value: str,
        target_type: type,
        path: str,
        from_name: str,
        to_name: str,
        changes: list[str],
    ) -> Any:
        """Try converting a string to a numeric type."""
        try:
            converted = target_type(value)
            changes.append(f"{path}: coerced {from_name} to {to_name}")
            return converted
        except (ValueError, TypeError):
            return value

    def _coerce_string_to_bool(
        self, value: str, path: str, changes: list[str]
    ) -> Any:
        """Coerce 'true'/'false' strings to bool."""
        lower = value.lower()
        if lower == "true":
            changes.append(f"{path}: coerced string to boolean")
            return True
        if lower == "false":
            changes.append(f"{path}: coerced string to boolean")
            return False
        return value

    def _coerce_object(
        self, value: dict, schema: dict, path: str, changes: list[str]
    ) -> dict:
        """Coerce object properties recursively."""
        properties = schema.get("properties", {})
        result = dict(value)
        for prop_name, prop_schema in properties.items():
            if prop_name in result:
                result[prop_name] = self._coerce_node(
                    result[prop_name], prop_schema, f"{path}.{prop_name}", changes
                )
        return result

    def _coerce_array(
        self, value: list, schema: dict, path: str, changes: list[str]
    ) -> list:
        """Coerce array items recursively."""
        if "items" not in schema:
            return value
        return [
            self._coerce_node(item, schema["items"], f"{path}[{i}]", changes)
            for i, item in enumerate(value)
        ]
