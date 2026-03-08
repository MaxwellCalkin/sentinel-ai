"""Tests for the output schema validator."""

import pytest
from sentinel.output_validator import OutputValidator, ValidationResult, ValidationError


PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
    },
    "required": ["name", "age"],
}


# ---------------------------------------------------------------------------
# Valid outputs
# ---------------------------------------------------------------------------

class TestValidOutputs:
    def test_valid_json(self):
        v = OutputValidator(PERSON_SCHEMA)
        result = v.validate('{"name": "Alice", "age": 30}')
        assert result.valid
        assert result.parsed == {"name": "Alice", "age": 30}

    def test_valid_with_optional(self):
        v = OutputValidator(PERSON_SCHEMA)
        result = v.validate('{"name": "Bob", "age": 25, "email": "bob@example.com"}')
        assert result.valid

    def test_valid_with_extra_fields(self):
        v = OutputValidator(PERSON_SCHEMA)
        result = v.validate('{"name": "Alice", "age": 30, "hobby": "chess"}')
        assert result.valid  # non-strict mode allows extra fields


# ---------------------------------------------------------------------------
# Invalid outputs
# ---------------------------------------------------------------------------

class TestInvalidOutputs:
    def test_missing_required(self):
        v = OutputValidator(PERSON_SCHEMA)
        result = v.validate('{"name": "Alice"}')
        assert not result.valid
        assert any("age" in e.path for e in result.errors)

    def test_wrong_type(self):
        v = OutputValidator(PERSON_SCHEMA)
        result = v.validate('{"name": "Alice", "age": "thirty"}')
        assert not result.valid
        assert any("Expected integer" in e.message for e in result.errors)

    def test_below_minimum(self):
        v = OutputValidator(PERSON_SCHEMA)
        result = v.validate('{"name": "Alice", "age": -1}')
        assert not result.valid

    def test_above_maximum(self):
        v = OutputValidator(PERSON_SCHEMA)
        result = v.validate('{"name": "Alice", "age": 200}')
        assert not result.valid

    def test_empty_string(self):
        v = OutputValidator(PERSON_SCHEMA)
        result = v.validate('{"name": "", "age": 30}')
        assert not result.valid
        assert any("too short" in e.message for e in result.errors)

    def test_invalid_pattern(self):
        v = OutputValidator(PERSON_SCHEMA)
        result = v.validate('{"name": "Alice", "age": 30, "email": "not-an-email"}')
        assert not result.valid
        assert any("pattern" in e.message for e in result.errors)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

class TestJsonParsing:
    def test_plain_json(self):
        v = OutputValidator({"type": "object"})
        result = v.validate('{"key": "value"}')
        assert result.valid

    def test_markdown_code_block(self):
        v = OutputValidator({"type": "object"})
        result = v.validate('```json\n{"key": "value"}\n```')
        assert result.valid

    def test_markdown_no_lang(self):
        v = OutputValidator({"type": "object"})
        result = v.validate('```\n{"key": "value"}\n```')
        assert result.valid

    def test_json_embedded_in_text(self):
        v = OutputValidator(PERSON_SCHEMA)
        text = 'Here is the result: {"name": "Alice", "age": 30}'
        result = v.validate(text)
        assert result.valid

    def test_invalid_json(self):
        v = OutputValidator({"type": "object"})
        result = v.validate("This is not JSON at all")
        assert not result.valid
        assert any("JSON" in e.message for e in result.errors)

    def test_whitespace_handling(self):
        v = OutputValidator({"type": "object"})
        result = v.validate('  \n  {"key": "value"}  \n  ')
        assert result.valid


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------

class TestTypeValidation:
    def test_string_type(self):
        v = OutputValidator({"type": "string"})
        assert v.validate('"hello"').valid
        assert not v.validate("42").valid

    def test_integer_type(self):
        v = OutputValidator({"type": "integer"})
        assert v.validate("42").valid
        assert not v.validate("3.14").valid

    def test_number_type(self):
        v = OutputValidator({"type": "number"})
        assert v.validate("3.14").valid
        assert v.validate("42").valid
        assert not v.validate('"hello"').valid

    def test_boolean_type(self):
        v = OutputValidator({"type": "boolean"})
        assert v.validate("true").valid
        assert v.validate("false").valid
        assert not v.validate("1").valid

    def test_null_type(self):
        v = OutputValidator({"type": "null"})
        assert v.validate("null").valid
        assert not v.validate("0").valid

    def test_array_type(self):
        v = OutputValidator({"type": "array"})
        assert v.validate("[1, 2, 3]").valid
        assert not v.validate("{}").valid

    def test_object_type(self):
        v = OutputValidator({"type": "object"})
        assert v.validate("{}").valid
        assert not v.validate("[]").valid


# ---------------------------------------------------------------------------
# String constraints
# ---------------------------------------------------------------------------

class TestStringConstraints:
    def test_min_length(self):
        v = OutputValidator({"type": "string", "minLength": 3})
        assert v.validate('"abc"').valid
        assert not v.validate('"ab"').valid

    def test_max_length(self):
        v = OutputValidator({"type": "string", "maxLength": 5})
        assert v.validate('"hello"').valid
        assert not v.validate('"toolong"').valid

    def test_pattern(self):
        v = OutputValidator({"type": "string", "pattern": r"^\d{3}-\d{4}$"})
        assert v.validate('"123-4567"').valid
        assert not v.validate('"abc"').valid


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class TestEnum:
    def test_valid_enum(self):
        v = OutputValidator({"type": "string", "enum": ["red", "green", "blue"]})
        assert v.validate('"red"').valid

    def test_invalid_enum(self):
        v = OutputValidator({"type": "string", "enum": ["red", "green", "blue"]})
        result = v.validate('"yellow"')
        assert not result.valid
        assert any("enum" in e.message for e in result.errors)


# ---------------------------------------------------------------------------
# Array constraints
# ---------------------------------------------------------------------------

class TestArrayConstraints:
    def test_min_items(self):
        v = OutputValidator({"type": "array", "minItems": 2})
        assert v.validate("[1, 2]").valid
        assert not v.validate("[1]").valid

    def test_max_items(self):
        v = OutputValidator({"type": "array", "maxItems": 3})
        assert v.validate("[1, 2, 3]").valid
        assert not v.validate("[1, 2, 3, 4]").valid

    def test_item_schema(self):
        v = OutputValidator({
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        })
        assert v.validate("[1, 2, 3]").valid
        assert not v.validate("[1, -2, 3]").valid


# ---------------------------------------------------------------------------
# Nested schemas
# ---------------------------------------------------------------------------

class TestNestedSchemas:
    def test_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string", "pattern": r"^\d{5}$"},
                    },
                    "required": ["city"],
                },
            },
        }
        v = OutputValidator(schema)
        assert v.validate('{"address": {"city": "NYC", "zip": "10001"}}').valid
        result = v.validate('{"address": {"zip": "10001"}}')
        assert not result.valid

    def test_array_of_objects(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        }
        v = OutputValidator(schema)
        assert v.validate('[{"id": 1}, {"id": 2}]').valid
        assert not v.validate('[{"id": 1}, {"name": "no-id"}]').valid


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------

class TestStrictMode:
    def test_strict_rejects_extra(self):
        v = OutputValidator(PERSON_SCHEMA, strict=True)
        result = v.validate('{"name": "Alice", "age": 30, "hobby": "chess"}')
        assert not result.valid
        assert any("Additional property" in e.message for e in result.errors)

    def test_strict_allows_defined(self):
        v = OutputValidator(PERSON_SCHEMA, strict=True)
        result = v.validate('{"name": "Alice", "age": 30}')
        assert result.valid


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_error_summary_valid(self):
        r = ValidationResult(valid=True)
        assert r.error_summary == "Valid"

    def test_error_summary_invalid(self):
        r = ValidationResult(
            valid=False,
            errors=[ValidationError("$.name", "Required field missing")],
        )
        assert "name" in r.error_summary
        assert "Required" in r.error_summary
