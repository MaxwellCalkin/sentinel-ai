"""Safe prompt templates with variable injection protection.

Provides type-safe prompt templates that prevent injection through
template variables, enforce length limits, and validate inputs.

Usage:
    from sentinel.prompt_template import PromptTemplate

    template = PromptTemplate(
        "Summarize the following text: {text}",
        variables={"text": {"type": "string", "max_length": 5000}},
    )
    prompt = template.render(text="The quick brown fox...")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VariableSpec:
    """Specification for a template variable."""
    type: str = "string"               # string, integer, float, list
    max_length: int | None = None      # For strings
    allowed_values: list[Any] | None = None  # Enum constraint
    pattern: str | None = None         # Regex pattern
    required: bool = True
    default: Any = None
    strip_newlines: bool = False       # Remove newlines from value
    escape_braces: bool = True         # Escape { and } in value


@dataclass
class RenderResult:
    """Result of rendering a template."""
    text: str
    variables_used: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class TemplateError(Exception):
    """Error in template rendering."""
    pass


class PromptTemplate:
    """Safe prompt template with injection protection.

    Variables are validated against their specs before insertion.
    Provides protection against:
    - Injection through overly long values
    - Type mismatch attacks
    - Template escape attacks (nested braces)
    """

    _VAR_PATTERN = re.compile(r"\{(\w+)\}")

    def __init__(
        self,
        template: str,
        variables: dict[str, dict | VariableSpec] | None = None,
        max_total_length: int | None = None,
    ):
        """
        Args:
            template: Template string with {variable} placeholders.
            variables: Variable specifications (dict or VariableSpec).
            max_total_length: Maximum length of rendered output.
        """
        self._template = template
        self._max_total_length = max_total_length

        # Parse variable specs
        self._specs: dict[str, VariableSpec] = {}
        if variables:
            for name, spec in variables.items():
                if isinstance(spec, dict):
                    self._specs[name] = VariableSpec(**spec)
                else:
                    self._specs[name] = spec

        # Discover template variables
        self._template_vars = set(self._VAR_PATTERN.findall(template))

    @property
    def template(self) -> str:
        return self._template

    @property
    def variables(self) -> set[str]:
        """Variable names found in the template."""
        return set(self._template_vars)

    def render(self, **kwargs: Any) -> RenderResult:
        """Render the template with the given variables."""
        warnings: list[str] = []
        values: dict[str, Any] = {}

        # Check for missing required variables
        for var_name in self._template_vars:
            spec = self._specs.get(var_name, VariableSpec())

            if var_name not in kwargs:
                if spec.required and spec.default is None:
                    raise TemplateError(f"Missing required variable: {var_name}")
                values[var_name] = spec.default if spec.default is not None else ""
            else:
                values[var_name] = kwargs[var_name]

        # Validate and sanitize each value
        sanitized: dict[str, str] = {}
        for var_name, value in values.items():
            spec = self._specs.get(var_name, VariableSpec())
            str_value, var_warnings = self._validate_and_sanitize(
                var_name, value, spec
            )
            sanitized[var_name] = str_value
            warnings.extend(var_warnings)

        # Render template
        result = self._template
        for var_name, str_value in sanitized.items():
            result = result.replace(f"{{{var_name}}}", str_value)

        # Check total length
        if self._max_total_length and len(result) > self._max_total_length:
            result = result[:self._max_total_length]
            warnings.append(
                f"Output truncated to {self._max_total_length} characters"
            )

        return RenderResult(
            text=result,
            variables_used=values,
            warnings=warnings,
        )

    def _validate_and_sanitize(
        self, name: str, value: Any, spec: VariableSpec
    ) -> tuple[str, list[str]]:
        """Validate a value against its spec and return sanitized string."""
        warnings: list[str] = []

        # Type checking
        if spec.type == "integer":
            if not isinstance(value, int):
                raise TemplateError(
                    f"Variable '{name}' expected integer, got {type(value).__name__}"
                )
            return str(value), warnings

        if spec.type == "float":
            if not isinstance(value, (int, float)):
                raise TemplateError(
                    f"Variable '{name}' expected float, got {type(value).__name__}"
                )
            return str(value), warnings

        if spec.type == "list":
            if not isinstance(value, (list, tuple)):
                raise TemplateError(
                    f"Variable '{name}' expected list, got {type(value).__name__}"
                )
            return ", ".join(str(item) for item in value), warnings

        # String type (default)
        str_value = str(value)

        # Enum check
        if spec.allowed_values is not None:
            if value not in spec.allowed_values:
                raise TemplateError(
                    f"Variable '{name}' value '{value}' not in allowed values: "
                    f"{spec.allowed_values}"
                )

        # Pattern check
        if spec.pattern is not None:
            if not re.match(spec.pattern, str_value):
                raise TemplateError(
                    f"Variable '{name}' does not match pattern: {spec.pattern}"
                )

        # Length check
        if spec.max_length is not None and len(str_value) > spec.max_length:
            str_value = str_value[:spec.max_length]
            warnings.append(
                f"Variable '{name}' truncated to {spec.max_length} characters"
            )

        # Strip newlines
        if spec.strip_newlines:
            new_value = str_value.replace("\n", " ").replace("\r", "")
            if new_value != str_value:
                warnings.append(f"Variable '{name}': newlines stripped")
                str_value = new_value

        # Escape braces to prevent nested template injection
        if spec.escape_braces:
            str_value = str_value.replace("{", "{{").replace("}", "}}")

        return str_value, warnings


class PromptLibrary:
    """Collection of named prompt templates."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}

    def register(self, name: str, template: PromptTemplate) -> None:
        """Register a named template."""
        self._templates[name] = template

    def get(self, name: str) -> PromptTemplate:
        """Get a template by name."""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")
        return self._templates[name]

    def render(self, template_name: str, **kwargs: Any) -> RenderResult:
        """Render a named template."""
        return self.get(template_name).render(**kwargs)

    @property
    def names(self) -> list[str]:
        """All registered template names."""
        return list(self._templates.keys())

    def __len__(self) -> int:
        return len(self._templates)
