"""Validate LLM tool/function calls before execution.

Ensure tool calls from LLMs have valid arguments, don't exceed
permissions, and match expected schemas before allowing execution.
Critical for agentic AI safety.

Usage:
    from sentinel.tool_call_validator import ToolCallValidator

    v = ToolCallValidator()
    v.register_tool("search", schema={"query": {"type": "string", "max_length": 200}})
    result = v.validate("search", {"query": "python tutorials"})
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSchema:
    """Schema definition for a tool."""
    name: str
    params: dict[str, dict[str, Any]]  # param_name -> constraints
    required: list[str] = field(default_factory=list)
    max_calls_per_session: int | None = None
    allowed_values: dict[str, list[Any]] = field(default_factory=dict)
    custom_validator: Callable[[dict[str, Any]], str | None] | None = None


@dataclass
class ToolValidationIssue:
    """A single validation issue."""
    param: str
    message: str
    severity: str = "error"


@dataclass
class ToolValidationResult:
    """Result of tool call validation."""
    valid: bool
    tool: str
    issues: list[ToolValidationIssue]
    args_validated: int


class ToolCallValidator:
    """Validate LLM tool calls against schemas.

    Register tools with parameter schemas and validate
    calls before execution. Supports type checking,
    value ranges, allowed values, and custom validators.
    """

    def __init__(self, strict: bool = True) -> None:
        """
        Args:
            strict: If True, reject unknown tools. If False, allow them.
        """
        self._strict = strict
        self._tools: dict[str, ToolSchema] = {}
        self._call_counts: dict[str, int] = {}

    def register_tool(
        self,
        name: str,
        schema: dict[str, dict[str, Any]] | None = None,
        required: list[str] | None = None,
        max_calls: int | None = None,
        custom_validator: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> None:
        """Register a tool with its schema.

        Args:
            name: Tool name.
            schema: Parameter schemas. Each param has constraints:
                - type: "string", "int", "float", "bool", "list", "dict"
                - max_length: Max string length
                - min_value/max_value: Numeric range
                - pattern: Regex pattern for strings
                - allowed: List of allowed values
            required: Required parameters.
            max_calls: Max calls per session.
            custom_validator: Function(args) -> error_message or None.
        """
        self._tools[name] = ToolSchema(
            name=name,
            params=schema or {},
            required=required or [],
            max_calls_per_session=max_calls,
            custom_validator=custom_validator,
        )

    def validate(self, tool: str, args: dict[str, Any] | None = None) -> ToolValidationResult:
        """Validate a tool call.

        Args:
            tool: Tool name.
            args: Tool arguments.

        Returns:
            ToolValidationResult with any issues found.
        """
        args = args or {}
        issues: list[ToolValidationIssue] = []

        # Check if tool is registered
        schema = self._tools.get(tool)
        if schema is None:
            if self._strict:
                issues.append(ToolValidationIssue(
                    param="",
                    message=f"Unknown tool: {tool}",
                ))
                return ToolValidationResult(valid=False, tool=tool, issues=issues, args_validated=0)
            return ToolValidationResult(valid=True, tool=tool, issues=[], args_validated=0)

        # Check call limits
        if schema.max_calls_per_session is not None:
            count = self._call_counts.get(tool, 0)
            if count >= schema.max_calls_per_session:
                issues.append(ToolValidationIssue(
                    param="",
                    message=f"Call limit exceeded: {count}/{schema.max_calls_per_session}",
                ))

        # Check required params
        for req in schema.required:
            if req not in args:
                issues.append(ToolValidationIssue(
                    param=req,
                    message=f"Required parameter missing: {req}",
                ))

        # Check unknown params
        known_params = set(schema.params.keys())
        for param in args:
            if known_params and param not in known_params:
                issues.append(ToolValidationIssue(
                    param=param,
                    message=f"Unknown parameter: {param}",
                    severity="warning",
                ))

        # Validate each parameter
        args_validated = 0
        for param_name, constraints in schema.params.items():
            if param_name not in args:
                continue
            args_validated += 1
            value = args[param_name]
            self._validate_param(param_name, value, constraints, issues)

        # Custom validator
        if schema.custom_validator:
            try:
                error = schema.custom_validator(args)
                if error:
                    issues.append(ToolValidationIssue(param="", message=error))
            except Exception as e:
                issues.append(ToolValidationIssue(
                    param="",
                    message=f"Custom validator error: {e}",
                ))

        # Track call count
        self._call_counts[tool] = self._call_counts.get(tool, 0) + 1

        errors = [i for i in issues if i.severity == "error"]
        return ToolValidationResult(
            valid=len(errors) == 0,
            tool=tool,
            issues=issues,
            args_validated=args_validated,
        )

    def _validate_param(
        self,
        name: str,
        value: Any,
        constraints: dict[str, Any],
        issues: list[ToolValidationIssue],
    ) -> None:
        """Validate a single parameter against constraints."""
        expected_type = constraints.get("type")
        if expected_type:
            type_map = {
                "string": str,
                "int": int,
                "float": (int, float),
                "bool": bool,
                "list": list,
                "dict": dict,
            }
            expected = type_map.get(expected_type)
            if expected and not isinstance(value, expected):
                issues.append(ToolValidationIssue(
                    param=name,
                    message=f"Type mismatch: expected {expected_type}, got {type(value).__name__}",
                ))
                return

        # String constraints
        if isinstance(value, str):
            max_len = constraints.get("max_length")
            if max_len and len(value) > max_len:
                issues.append(ToolValidationIssue(
                    param=name,
                    message=f"String too long: {len(value)} > {max_len}",
                ))

            pattern = constraints.get("pattern")
            if pattern and not re.search(pattern, value):
                issues.append(ToolValidationIssue(
                    param=name,
                    message=f"Pattern mismatch for {name}",
                ))

        # Numeric constraints
        if isinstance(value, (int, float)):
            min_val = constraints.get("min_value")
            if min_val is not None and value < min_val:
                issues.append(ToolValidationIssue(
                    param=name,
                    message=f"Value {value} below minimum {min_val}",
                ))

            max_val = constraints.get("max_value")
            if max_val is not None and value > max_val:
                issues.append(ToolValidationIssue(
                    param=name,
                    message=f"Value {value} above maximum {max_val}",
                ))

        # Allowed values
        allowed = constraints.get("allowed")
        if allowed and value not in allowed:
            issues.append(ToolValidationIssue(
                param=name,
                message=f"Value not in allowed list for {name}",
            ))

    def reset_counts(self) -> None:
        """Reset call counts."""
        self._call_counts.clear()

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def get_call_count(self, tool: str) -> int:
        return self._call_counts.get(tool, 0)
