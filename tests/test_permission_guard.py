"""Tests for tool permission guard."""

import pytest
from sentinel.permission_guard import PermissionGuard, PermissionResult


# ---------------------------------------------------------------------------
# Basic allow/deny
# ---------------------------------------------------------------------------

class TestBasicPermissions:
    def test_allow_tool(self):
        g = PermissionGuard()
        g.allow("read_file")
        result = g.check("read_file")
        assert result.allowed

    def test_deny_tool(self):
        g = PermissionGuard()
        g.deny("delete_file", reason="too dangerous")
        result = g.check("delete_file")
        assert not result.allowed
        assert "too dangerous" in result.reason

    def test_default_deny(self):
        g = PermissionGuard(default_allow=False)
        result = g.check("unknown_tool")
        assert not result.allowed

    def test_default_allow(self):
        g = PermissionGuard(default_allow=True)
        result = g.check("unknown_tool")
        assert result.allowed

    def test_explicit_deny_overrides_default_allow(self):
        g = PermissionGuard(default_allow=True)
        g.deny("dangerous_tool")
        assert not g.check("dangerous_tool").allowed


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

class TestArgValidation:
    def test_pattern_match(self):
        g = PermissionGuard()
        g.allow("read_file", args_schema={
            "path": {"pattern": r"^/safe/.*"},
        })
        assert g.check("read_file", {"path": "/safe/data.txt"}).allowed
        assert not g.check("read_file", {"path": "/etc/passwd"}).allowed

    def test_allowed_values(self):
        g = PermissionGuard()
        g.allow("set_mode", args_schema={
            "mode": {"allowed_values": ["read", "write"]},
        })
        assert g.check("set_mode", {"mode": "read"}).allowed
        assert not g.check("set_mode", {"mode": "admin"}).allowed

    def test_max_length(self):
        g = PermissionGuard()
        g.allow("search", args_schema={
            "query": {"max_length": 100},
        })
        assert g.check("search", {"query": "short"}).allowed
        assert not g.check("search", {"query": "x" * 101}).allowed

    def test_required_arg(self):
        g = PermissionGuard()
        g.allow("read_file", args_schema={
            "path": {"required": True},
        })
        assert not g.check("read_file", {}).allowed
        assert g.check("read_file", {"path": "/file"}).allowed

    def test_optional_arg(self):
        g = PermissionGuard()
        g.allow("search", args_schema={
            "limit": {"required": False, "allowed_values": [10, 20, 50]},
        })
        assert g.check("search", {}).allowed
        assert g.check("search", {"limit": 10}).allowed
        assert not g.check("search", {"limit": 999}).allowed

    def test_violations_listed(self):
        g = PermissionGuard()
        g.allow("tool", args_schema={
            "a": {"pattern": r"^ok$"},
            "b": {"required": True},
        })
        result = g.check("tool", {"a": "bad"})
        assert not result.allowed
        assert len(result.violations) == 2  # pattern fail + missing b


# ---------------------------------------------------------------------------
# Call limits
# ---------------------------------------------------------------------------

class TestCallLimits:
    def test_within_limit(self):
        g = PermissionGuard()
        g.allow("api_call", max_calls=3)
        assert g.check("api_call").allowed
        assert g.check("api_call").allowed
        assert g.check("api_call").allowed

    def test_exceeds_limit(self):
        g = PermissionGuard()
        g.allow("api_call", max_calls=2)
        g.check("api_call")
        g.check("api_call")
        result = g.check("api_call")
        assert not result.allowed
        assert "limit exceeded" in result.reason

    def test_reset_counts(self):
        g = PermissionGuard()
        g.allow("api_call", max_calls=1)
        g.check("api_call")
        g.reset_counts()
        assert g.check("api_call").allowed


# ---------------------------------------------------------------------------
# Custom checks
# ---------------------------------------------------------------------------

class TestCustomChecks:
    def test_custom_check_pass(self):
        g = PermissionGuard()
        g.allow("tool", custom_check=lambda t, a: a.get("safe", False))
        assert g.check("tool", {"safe": True}).allowed
        assert not g.check("tool", {"safe": False}).allowed

    def test_custom_check_fail(self):
        g = PermissionGuard()
        g.allow("tool", custom_check=lambda t, a: False)
        result = g.check("tool")
        assert not result.allowed
        assert "custom check failed" in result.violations


# ---------------------------------------------------------------------------
# Management
# ---------------------------------------------------------------------------

class TestManagement:
    def test_tool_count(self):
        g = PermissionGuard()
        g.allow("a")
        g.deny("b")
        assert g.tool_count == 2

    def test_allowed_tools(self):
        g = PermissionGuard()
        g.allow("read")
        g.allow("write")
        g.deny("delete")
        assert set(g.allowed_tools) == {"read", "write"}

    def test_denied_tools(self):
        g = PermissionGuard()
        g.allow("read")
        g.deny("delete")
        assert g.denied_tools == ["delete"]

    def test_clear(self):
        g = PermissionGuard()
        g.allow("a")
        g.deny("b")
        g.clear()
        assert g.tool_count == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_args(self):
        g = PermissionGuard()
        g.allow("tool")
        assert g.check("tool", {}).allowed

    def test_no_schema_allows_any_args(self):
        g = PermissionGuard()
        g.allow("tool")
        assert g.check("tool", {"anything": "goes"}).allowed

    def test_result_structure(self):
        g = PermissionGuard()
        g.allow("tool")
        r = g.check("tool")
        assert isinstance(r, PermissionResult)
        assert r.tool == "tool"
