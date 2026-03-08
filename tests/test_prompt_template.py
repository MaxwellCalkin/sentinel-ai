"""Tests for safe prompt templates."""

import pytest
from sentinel.prompt_template import (
    PromptTemplate, PromptLibrary, VariableSpec, RenderResult, TemplateError,
)


# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------

class TestBasicRendering:
    def test_simple_render(self):
        t = PromptTemplate("Hello {name}!")
        result = t.render(name="Alice")
        assert result.text == "Hello Alice!"

    def test_multiple_variables(self):
        t = PromptTemplate("Hello {name}, you are {age} years old")
        result = t.render(name="Bob", age=25)
        assert result.text == "Hello Bob, you are 25 years old"

    def test_no_variables(self):
        t = PromptTemplate("Hello world!")
        result = t.render()
        assert result.text == "Hello world!"

    def test_repeated_variable(self):
        t = PromptTemplate("{x} and {x}")
        result = t.render(x="hi")
        assert result.text == "hi and hi"

    def test_variables_property(self):
        t = PromptTemplate("Hello {name}, age {age}")
        assert t.variables == {"name", "age"}

    def test_template_property(self):
        t = PromptTemplate("Hello {name}")
        assert t.template == "Hello {name}"


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------

class TestTypeValidation:
    def test_integer_type(self):
        t = PromptTemplate(
            "Count: {n}",
            variables={"n": {"type": "integer"}},
        )
        result = t.render(n=42)
        assert result.text == "Count: 42"

    def test_integer_wrong_type(self):
        t = PromptTemplate(
            "Count: {n}",
            variables={"n": {"type": "integer"}},
        )
        with pytest.raises(TemplateError, match="expected integer"):
            t.render(n="not a number")

    def test_float_type(self):
        t = PromptTemplate(
            "Score: {s}",
            variables={"s": {"type": "float"}},
        )
        result = t.render(s=3.14)
        assert "3.14" in result.text

    def test_float_accepts_int(self):
        t = PromptTemplate(
            "Score: {s}",
            variables={"s": {"type": "float"}},
        )
        result = t.render(s=42)
        assert result.text == "Score: 42"

    def test_list_type(self):
        t = PromptTemplate(
            "Items: {items}",
            variables={"items": {"type": "list"}},
        )
        result = t.render(items=["a", "b", "c"])
        assert result.text == "Items: a, b, c"

    def test_list_wrong_type(self):
        t = PromptTemplate(
            "Items: {items}",
            variables={"items": {"type": "list"}},
        )
        with pytest.raises(TemplateError, match="expected list"):
            t.render(items="not a list")


# ---------------------------------------------------------------------------
# String constraints
# ---------------------------------------------------------------------------

class TestStringConstraints:
    def test_max_length(self):
        t = PromptTemplate(
            "Input: {text}",
            variables={"text": {"max_length": 10}},
        )
        result = t.render(text="This is a very long text")
        assert "truncated" in result.warnings[0]
        assert len("Input: ") + 10 >= len(result.text) - 4  # -4 for brace escaping buffer

    def test_allowed_values(self):
        t = PromptTemplate(
            "Color: {color}",
            variables={"color": {"allowed_values": ["red", "green", "blue"]}},
        )
        result = t.render(color="red")
        assert result.text == "Color: red"

    def test_allowed_values_violation(self):
        t = PromptTemplate(
            "Color: {color}",
            variables={"color": {"allowed_values": ["red", "green", "blue"]}},
        )
        with pytest.raises(TemplateError, match="not in allowed"):
            t.render(color="yellow")

    def test_pattern_match(self):
        t = PromptTemplate(
            "Email: {email}",
            variables={"email": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"}},
        )
        result = t.render(email="user@example.com")
        assert "user@example.com" in result.text

    def test_pattern_violation(self):
        t = PromptTemplate(
            "Email: {email}",
            variables={"email": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"}},
        )
        with pytest.raises(TemplateError, match="pattern"):
            t.render(email="not-an-email")

    def test_strip_newlines(self):
        t = PromptTemplate(
            "Input: {text}",
            variables={"text": {"strip_newlines": True}},
        )
        result = t.render(text="line1\nline2\nline3")
        assert "\n" not in result.text
        assert "newlines stripped" in result.warnings[0]


# ---------------------------------------------------------------------------
# Injection protection
# ---------------------------------------------------------------------------

class TestInjectionProtection:
    def test_brace_escaping(self):
        t = PromptTemplate(
            "Input: {text}",
            variables={"text": {"escape_braces": True}},
        )
        result = t.render(text="Hello {name}")
        # Single braces are doubled to prevent template injection
        assert "{{name}}" in result.text

    def test_brace_escaping_disabled(self):
        t = PromptTemplate(
            "Input: {text}",
            variables={"text": {"escape_braces": False}},
        )
        result = t.render(text="Hello {name}")
        assert "{name}" in result.text

    def test_total_length_limit(self):
        t = PromptTemplate(
            "Input: {text}",
            max_total_length=20,
        )
        result = t.render(text="A very long input string that exceeds the limit")
        assert len(result.text) == 20
        assert "truncated" in result.warnings[0]


# ---------------------------------------------------------------------------
# Required/optional variables
# ---------------------------------------------------------------------------

class TestRequiredOptional:
    def test_missing_required(self):
        t = PromptTemplate(
            "Hello {name}",
            variables={"name": {"required": True}},
        )
        with pytest.raises(TemplateError, match="Missing required"):
            t.render()

    def test_optional_with_default(self):
        t = PromptTemplate(
            "Hello {name}",
            variables={"name": {"required": False, "default": "World"}},
        )
        result = t.render()
        assert result.text == "Hello World"

    def test_optional_without_default(self):
        t = PromptTemplate(
            "Hello {name}",
            variables={"name": {"required": False}},
        )
        result = t.render()
        assert result.text == "Hello "

    def test_default_overridden(self):
        t = PromptTemplate(
            "Hello {name}",
            variables={"name": {"required": False, "default": "World"}},
        )
        result = t.render(name="Alice")
        assert result.text == "Hello Alice"


# ---------------------------------------------------------------------------
# RenderResult
# ---------------------------------------------------------------------------

class TestRenderResult:
    def test_no_warnings(self):
        t = PromptTemplate("Hello {name}")
        result = t.render(name="Alice")
        assert not result.has_warnings

    def test_variables_used(self):
        t = PromptTemplate("Hello {name}")
        result = t.render(name="Alice")
        assert result.variables_used == {"name": "Alice"}


# ---------------------------------------------------------------------------
# VariableSpec as object
# ---------------------------------------------------------------------------

class TestVariableSpecObject:
    def test_spec_object(self):
        spec = VariableSpec(type="integer", required=True)
        t = PromptTemplate("N: {n}", variables={"n": spec})
        result = t.render(n=42)
        assert result.text == "N: 42"


# ---------------------------------------------------------------------------
# PromptLibrary
# ---------------------------------------------------------------------------

class TestPromptLibrary:
    def test_register_and_get(self):
        lib = PromptLibrary()
        t = PromptTemplate("Hello {name}")
        lib.register("greeting", t)
        assert lib.get("greeting") is t

    def test_render(self):
        lib = PromptLibrary()
        lib.register("greeting", PromptTemplate("Hello {name}"))
        result = lib.render("greeting", name="Alice")
        assert result.text == "Hello Alice"

    def test_missing_template(self):
        lib = PromptLibrary()
        with pytest.raises(KeyError, match="not found"):
            lib.get("nonexistent")

    def test_names(self):
        lib = PromptLibrary()
        lib.register("a", PromptTemplate("A"))
        lib.register("b", PromptTemplate("B"))
        assert set(lib.names) == {"a", "b"}

    def test_len(self):
        lib = PromptLibrary()
        lib.register("a", PromptTemplate("A"))
        assert len(lib) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_template(self):
        t = PromptTemplate("")
        result = t.render()
        assert result.text == ""

    def test_no_specs_auto_detect(self):
        t = PromptTemplate("Hello {name} you are {age}")
        result = t.render(name="Alice", age=30)
        assert "Alice" in result.text
        assert "30" in result.text

    def test_unspecified_variable_default(self):
        # Variable in template without spec — defaults to required string
        t = PromptTemplate("Hello {name}")
        with pytest.raises(TemplateError, match="Missing required"):
            t.render()
