"""Tests for tool call validator."""

import pytest
from sentinel.tool_call_validator import ToolCallValidator, ToolValidationResult


# ---------------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------------

class TestBasicValidation:
    def test_valid_call(self):
        v = ToolCallValidator()
        v.register_tool("search", schema={"query": {"type": "string"}})
        result = v.validate("search", {"query": "hello"})
        assert result.valid
        assert result.tool == "search"

    def test_unknown_tool_strict(self):
        v = ToolCallValidator(strict=True)
        result = v.validate("unknown_tool", {"x": 1})
        assert not result.valid

    def test_unknown_tool_lenient(self):
        v = ToolCallValidator(strict=False)
        result = v.validate("unknown_tool", {"x": 1})
        assert result.valid

    def test_no_args(self):
        v = ToolCallValidator()
        v.register_tool("ping")
        result = v.validate("ping")
        assert result.valid


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------

class TestTypeChecking:
    def test_string_type(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"name": {"type": "string"}})
        assert v.validate("t", {"name": "hello"}).valid
        assert not v.validate("t", {"name": 123}).valid

    def test_int_type(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"count": {"type": "int"}})
        assert v.validate("t", {"count": 5}).valid
        assert not v.validate("t", {"count": "five"}).valid

    def test_float_type(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"score": {"type": "float"}})
        assert v.validate("t", {"score": 3.14}).valid
        assert v.validate("t", {"score": 3}).valid  # int is valid float

    def test_bool_type(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"flag": {"type": "bool"}})
        assert v.validate("t", {"flag": True}).valid
        assert not v.validate("t", {"flag": "yes"}).valid

    def test_list_type(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"items": {"type": "list"}})
        assert v.validate("t", {"items": [1, 2]}).valid
        assert not v.validate("t", {"items": "not a list"}).valid


# ---------------------------------------------------------------------------
# String constraints
# ---------------------------------------------------------------------------

class TestStringConstraints:
    def test_max_length(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"q": {"type": "string", "max_length": 10}})
        assert v.validate("t", {"q": "short"}).valid
        assert not v.validate("t", {"q": "this is way too long"}).valid

    def test_pattern(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"email": {"type": "string", "pattern": r"@.*\."}})
        assert v.validate("t", {"email": "user@example.com"}).valid
        assert not v.validate("t", {"email": "noemail"}).valid


# ---------------------------------------------------------------------------
# Numeric constraints
# ---------------------------------------------------------------------------

class TestNumericConstraints:
    def test_min_value(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"age": {"type": "int", "min_value": 0}})
        assert v.validate("t", {"age": 5}).valid
        assert not v.validate("t", {"age": -1}).valid

    def test_max_value(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"pct": {"type": "float", "max_value": 100.0}})
        assert v.validate("t", {"pct": 50.0}).valid
        assert not v.validate("t", {"pct": 150.0}).valid

    def test_range(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"temp": {"type": "float", "min_value": -273.15, "max_value": 1000}})
        assert v.validate("t", {"temp": 20.0}).valid
        assert not v.validate("t", {"temp": -300.0}).valid


# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

class TestAllowedValues:
    def test_allowed(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"color": {"allowed": ["red", "blue", "green"]}})
        assert v.validate("t", {"color": "red"}).valid
        assert not v.validate("t", {"color": "purple"}).valid


# ---------------------------------------------------------------------------
# Required params
# ---------------------------------------------------------------------------

class TestRequired:
    def test_required_present(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"q": {"type": "string"}}, required=["q"])
        assert v.validate("t", {"q": "hello"}).valid

    def test_required_missing(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"q": {"type": "string"}}, required=["q"])
        result = v.validate("t", {})
        assert not result.valid
        assert any("Required" in i.message for i in result.issues)


# ---------------------------------------------------------------------------
# Call limits
# ---------------------------------------------------------------------------

class TestCallLimits:
    def test_within_limit(self):
        v = ToolCallValidator()
        v.register_tool("t", max_calls=3)
        assert v.validate("t").valid
        assert v.validate("t").valid
        assert v.validate("t").valid

    def test_exceed_limit(self):
        v = ToolCallValidator()
        v.register_tool("t", max_calls=2)
        v.validate("t")
        v.validate("t")
        result = v.validate("t")
        assert not result.valid
        assert any("limit" in i.message.lower() for i in result.issues)

    def test_reset_counts(self):
        v = ToolCallValidator()
        v.register_tool("t", max_calls=1)
        v.validate("t")
        v.reset_counts()
        assert v.validate("t").valid

    def test_get_call_count(self):
        v = ToolCallValidator()
        v.register_tool("t")
        v.validate("t")
        v.validate("t")
        assert v.get_call_count("t") == 2


# ---------------------------------------------------------------------------
# Custom validator
# ---------------------------------------------------------------------------

class TestCustomValidator:
    def test_custom_passes(self):
        v = ToolCallValidator()
        v.register_tool("t", custom_validator=lambda args: None)
        assert v.validate("t").valid

    def test_custom_fails(self):
        v = ToolCallValidator()
        v.register_tool("t", custom_validator=lambda args: "Custom error")
        result = v.validate("t")
        assert not result.valid
        assert any("Custom error" in i.message for i in result.issues)

    def test_custom_exception(self):
        v = ToolCallValidator()
        v.register_tool("t", custom_validator=lambda args: (_ for _ in ()).throw(RuntimeError("boom")))
        result = v.validate("t")
        assert not result.valid


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_tool_count(self):
        v = ToolCallValidator()
        v.register_tool("a")
        v.register_tool("b")
        assert v.tool_count == 2

    def test_unknown_param_warning(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"known": {"type": "string"}})
        result = v.validate("t", {"known": "ok", "extra": "surprise"})
        warnings = [i for i in result.issues if i.severity == "warning"]
        assert len(warnings) > 0
        assert result.valid  # Warnings don't block

    def test_result_structure(self):
        v = ToolCallValidator()
        v.register_tool("t", schema={"x": {"type": "int"}})
        result = v.validate("t", {"x": 5})
        assert isinstance(result, ToolValidationResult)
        assert result.args_validated == 1
