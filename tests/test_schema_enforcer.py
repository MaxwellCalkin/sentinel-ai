"""Tests for the schema enforcer module."""

import pytest
from sentinel.schema_enforcer import SchemaEnforcer, EnforcementResult, CoercionResult


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------

class TestTypeValidation:
    def test_string_valid(self):
        enforcer = SchemaEnforcer()
        result = enforcer.validate("hello", {"type": "string"})
        assert result.valid
        assert result.errors == []

    def test_string_invalid(self):
        enforcer = SchemaEnforcer()
        result = enforcer.validate(42, {"type": "string"})
        assert not result.valid
        assert any("string" in e for e in result.errors)

    def test_number_valid(self):
        enforcer = SchemaEnforcer()
        assert enforcer.validate(3.14, {"type": "number"}).valid
        assert enforcer.validate(42, {"type": "number"}).valid

    def test_integer_valid(self):
        enforcer = SchemaEnforcer()
        result = enforcer.validate(42, {"type": "integer"})
        assert result.valid

    def test_boolean_valid(self):
        enforcer = SchemaEnforcer()
        assert enforcer.validate(True, {"type": "boolean"}).valid
        assert enforcer.validate(False, {"type": "boolean"}).valid

    def test_null_valid(self):
        enforcer = SchemaEnforcer()
        result = enforcer.validate(None, {"type": "null"})
        assert result.valid


# ---------------------------------------------------------------------------
# Object validation
# ---------------------------------------------------------------------------

class TestObject:
    def test_required_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        enforcer = SchemaEnforcer()
        result = enforcer.validate({"name": "Alice", "age": 30}, schema)
        assert result.valid

    def test_missing_required(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        enforcer = SchemaEnforcer()
        result = enforcer.validate({"name": "Alice"}, schema)
        assert not result.valid
        assert any("age" in e for e in result.errors)
        assert "$.age" == result.path

    def test_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
            "required": ["address"],
        }
        enforcer = SchemaEnforcer()
        valid_result = enforcer.validate(
            {"address": {"city": "NYC", "zip": "10001"}}, schema
        )
        assert valid_result.valid

        invalid_result = enforcer.validate({"address": {"zip": "10001"}}, schema)
        assert not invalid_result.valid
        assert "$.address.city" == invalid_result.path


# ---------------------------------------------------------------------------
# Array validation
# ---------------------------------------------------------------------------

class TestArray:
    def test_array_valid(self):
        schema = {"type": "array"}
        enforcer = SchemaEnforcer()
        result = enforcer.validate([1, 2, 3], schema)
        assert result.valid

    def test_array_items_validation(self):
        schema = {
            "type": "array",
            "items": {"type": "integer"},
        }
        enforcer = SchemaEnforcer()
        assert enforcer.validate([1, 2, 3], schema).valid

        result = enforcer.validate([1, "two", 3], schema)
        assert not result.valid
        assert any("[1]" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_min_max_length(self):
        schema = {"type": "string", "minLength": 2, "maxLength": 5}
        enforcer = SchemaEnforcer()
        assert enforcer.validate("ab", schema).valid
        assert enforcer.validate("abcde", schema).valid
        assert not enforcer.validate("a", schema).valid
        assert not enforcer.validate("abcdef", schema).valid

    def test_min_max_number(self):
        schema = {"type": "integer", "minimum": 0, "maximum": 100}
        enforcer = SchemaEnforcer()
        assert enforcer.validate(0, schema).valid
        assert enforcer.validate(100, schema).valid
        assert not enforcer.validate(-1, schema).valid
        assert not enforcer.validate(101, schema).valid

    def test_enum(self):
        schema = {"type": "string", "enum": ["red", "green", "blue"]}
        enforcer = SchemaEnforcer()
        assert enforcer.validate("red", schema).valid
        result = enforcer.validate("yellow", schema)
        assert not result.valid
        assert any("enum" in e for e in result.errors)

    def test_pattern(self):
        schema = {"type": "string", "pattern": r"^\d{3}-\d{4}$"}
        enforcer = SchemaEnforcer()
        assert enforcer.validate("123-4567", schema).valid
        assert not enforcer.validate("abc", schema).valid


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------

class TestCoercion:
    def test_string_to_int(self):
        enforcer = SchemaEnforcer()
        result = enforcer.coerce("42", {"type": "integer"})
        assert result.success
        assert result.original == "42"
        assert result.coerced == 42
        assert len(result.changes) == 1

    def test_string_to_bool(self):
        enforcer = SchemaEnforcer()
        result_true = enforcer.coerce("true", {"type": "boolean"})
        assert result_true.success
        assert result_true.coerced is True

        result_false = enforcer.coerce("false", {"type": "boolean"})
        assert result_false.success
        assert result_false.coerced is False

    def test_wrap_single_to_array(self):
        enforcer = SchemaEnforcer()
        result = enforcer.coerce("hello", {"type": "array", "items": {"type": "string"}})
        assert result.success
        assert result.coerced == ["hello"]
        assert any("wrapped" in c for c in result.changes)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_and_validate(self):
        enforcer = SchemaEnforcer()
        enforcer.register_schema("person", {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        })
        result = enforcer.validate_against({"name": "Alice"}, "person")
        assert result.valid

        bad_result = enforcer.validate_against({}, "person")
        assert not bad_result.valid

    def test_list_schemas(self):
        enforcer = SchemaEnforcer()
        assert enforcer.list_schemas() == []
        enforcer.register_schema("alpha", {"type": "string"})
        enforcer.register_schema("beta", {"type": "integer"})
        assert enforcer.list_schemas() == ["alpha", "beta"]

    def test_validate_against_unknown_raises(self):
        enforcer = SchemaEnforcer()
        with pytest.raises(KeyError, match="not_registered"):
            enforcer.validate_against("data", "not_registered")


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------

class TestStrict:
    def test_strict_mode(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        # Non-strict: extra properties produce warnings, not errors
        non_strict = SchemaEnforcer(strict=False)
        result = non_strict.validate({"name": "Alice", "extra": 1}, schema)
        assert result.valid
        assert len(result.warnings) > 0
        assert any("extra" in w for w in result.warnings)

        # Strict: warnings become errors
        strict = SchemaEnforcer(strict=True)
        result = strict.validate({"name": "Alice", "extra": 1}, schema)
        assert not result.valid
        assert any("extra" in e for e in result.errors)
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_boolean_not_integer(self):
        """bool is a subclass of int in Python; enforce schema distinction."""
        enforcer = SchemaEnforcer()
        assert not enforcer.validate(True, {"type": "integer"}).valid
        assert not enforcer.validate(False, {"type": "number"}).valid

    def test_empty_schema_accepts_anything(self):
        enforcer = SchemaEnforcer()
        assert enforcer.validate("hello", {}).valid
        assert enforcer.validate(42, {}).valid
        assert enforcer.validate(None, {}).valid

    def test_coercion_string_to_float(self):
        enforcer = SchemaEnforcer()
        result = enforcer.coerce("3.14", {"type": "number"})
        assert result.success
        assert result.coerced == pytest.approx(3.14)

    def test_coercion_int_to_string(self):
        enforcer = SchemaEnforcer()
        result = enforcer.coerce(42, {"type": "string"})
        assert result.success
        assert result.coerced == "42"

    def test_coercion_failure_leaves_original(self):
        enforcer = SchemaEnforcer()
        result = enforcer.coerce("not-a-number", {"type": "integer"})
        assert not result.success
        assert result.coerced == "not-a-number"

    def test_enforcement_result_path_empty_when_valid(self):
        enforcer = SchemaEnforcer()
        result = enforcer.validate("hello", {"type": "string"})
        assert result.path == ""
