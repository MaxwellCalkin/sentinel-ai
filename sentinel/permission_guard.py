"""Permission guard for LLM tool use.

Controls which tools an LLM agent can use, with allowlists,
denylists, and per-tool argument validation. Critical for
safe agentic AI deployments.

Usage:
    from sentinel.permission_guard import PermissionGuard

    guard = PermissionGuard()
    guard.allow("read_file", args_schema={"path": {"pattern": r"^/safe/.*"}})
    guard.deny("delete_file")

    result = guard.check("read_file", {"path": "/safe/data.txt"})
    assert result.allowed
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PermissionResult:
    """Result of a permission check."""
    tool: str
    allowed: bool
    reason: str = ""
    violations: list[str] = field(default_factory=list)


@dataclass
class ToolPermission:
    """Permission configuration for a tool."""
    tool: str
    allowed: bool
    args_schema: dict[str, dict[str, Any]] | None = None
    max_calls: int | None = None
    custom_check: Callable[[str, dict[str, Any]], bool] | None = None
    reason: str = ""


class PermissionGuard:
    """Guard for LLM tool-use authorization.

    Control which tools an agent can call, validate arguments,
    and enforce rate limits per tool.
    """

    def __init__(self, default_allow: bool = False) -> None:
        """
        Args:
            default_allow: Whether to allow tools not explicitly configured.
        """
        self._default_allow = default_allow
        self._permissions: dict[str, ToolPermission] = {}
        self._call_counts: dict[str, int] = {}

    def allow(
        self,
        tool: str,
        args_schema: dict[str, dict[str, Any]] | None = None,
        max_calls: int | None = None,
        custom_check: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> None:
        """Allow a tool with optional constraints.

        Args:
            tool: Tool name.
            args_schema: Validation schema for arguments.
                Keys are arg names, values are dicts with:
                - "pattern": regex the value must match
                - "allowed_values": list of allowed values
                - "max_length": maximum string length
                - "required": whether arg is required (default True)
            max_calls: Maximum number of calls allowed.
            custom_check: Custom validation function(tool, args) -> bool.
        """
        self._permissions[tool] = ToolPermission(
            tool=tool,
            allowed=True,
            args_schema=args_schema,
            max_calls=max_calls,
            custom_check=custom_check,
        )

    def deny(self, tool: str, reason: str = "") -> None:
        """Explicitly deny a tool."""
        self._permissions[tool] = ToolPermission(
            tool=tool,
            allowed=False,
            reason=reason,
        )

    def check(self, tool: str, args: dict[str, Any] | None = None) -> PermissionResult:
        """Check if a tool call is permitted.

        Args:
            tool: Tool name to check.
            args: Arguments being passed to the tool.

        Returns:
            PermissionResult with allowed status and violations.
        """
        args = args or {}

        # Check if tool has explicit permissions
        perm = self._permissions.get(tool)
        if perm is None:
            return PermissionResult(
                tool=tool,
                allowed=self._default_allow,
                reason="allowed by default" if self._default_allow else "not in allowlist",
            )

        # Explicitly denied
        if not perm.allowed:
            return PermissionResult(
                tool=tool,
                allowed=False,
                reason=perm.reason or "explicitly denied",
            )

        violations: list[str] = []

        # Check call limit
        if perm.max_calls is not None:
            count = self._call_counts.get(tool, 0)
            if count >= perm.max_calls:
                return PermissionResult(
                    tool=tool,
                    allowed=False,
                    reason=f"call limit exceeded ({count}/{perm.max_calls})",
                    violations=[f"max_calls: {count} >= {perm.max_calls}"],
                )

        # Validate arguments against schema
        if perm.args_schema:
            violations.extend(self._validate_args(args, perm.args_schema))

        # Custom check
        if perm.custom_check and not perm.custom_check(tool, args):
            violations.append("custom check failed")

        if violations:
            return PermissionResult(
                tool=tool,
                allowed=False,
                reason="argument validation failed",
                violations=violations,
            )

        # Track call
        self._call_counts[tool] = self._call_counts.get(tool, 0) + 1

        return PermissionResult(
            tool=tool,
            allowed=True,
            reason="allowed",
        )

    def _validate_args(
        self,
        args: dict[str, Any],
        schema: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Validate arguments against schema."""
        violations: list[str] = []

        for arg_name, constraints in schema.items():
            required = constraints.get("required", True)
            value = args.get(arg_name)

            if value is None:
                if required:
                    violations.append(f"missing required arg: {arg_name}")
                continue

            str_value = str(value)

            # Pattern check
            pattern = constraints.get("pattern")
            if pattern and not re.match(pattern, str_value):
                violations.append(
                    f"{arg_name}: value '{str_value}' does not match pattern '{pattern}'"
                )

            # Allowed values
            allowed = constraints.get("allowed_values")
            if allowed and value not in allowed:
                violations.append(
                    f"{arg_name}: value '{value}' not in allowed values"
                )

            # Max length
            max_len = constraints.get("max_length")
            if max_len and len(str_value) > max_len:
                violations.append(
                    f"{arg_name}: length {len(str_value)} exceeds max {max_len}"
                )

        return violations

    @property
    def tool_count(self) -> int:
        """Number of configured tools."""
        return len(self._permissions)

    @property
    def allowed_tools(self) -> list[str]:
        """List of allowed tool names."""
        return [t for t, p in self._permissions.items() if p.allowed]

    @property
    def denied_tools(self) -> list[str]:
        """List of denied tool names."""
        return [t for t, p in self._permissions.items() if not p.allowed]

    def reset_counts(self) -> None:
        """Reset call counters."""
        self._call_counts.clear()

    def clear(self) -> None:
        """Remove all permissions."""
        self._permissions.clear()
        self._call_counts.clear()
