"""Tests for tool safety guard."""

import pytest
from sentinel.tool_safety_guard import (
    ToolCallDecision,
    ToolCallRequest,
    ToolDefinition,
    ToolGuardConfig,
    ToolGuardStats,
    ToolSafetyGuard,
)


# ---------------------------------------------------------------------------
# Registered tool allowed
# ---------------------------------------------------------------------------

class TestRegisteredToolAllowed:
    def test_registered_tool_passes_all_checks(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="search", allowed_params=["query"]))
        decision = guard.check(ToolCallRequest(tool_name="search", params={"query": "hello"}))
        assert decision.allowed
        assert decision.reason == "All checks passed"
        assert decision.violations == []

    def test_decision_carries_tool_risk_level(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="deploy", risk_level="high"))
        decision = guard.check(ToolCallRequest(tool_name="deploy", params={}))
        assert decision.allowed
        assert decision.risk_level == "high"


# ---------------------------------------------------------------------------
# Unknown tool blocked
# ---------------------------------------------------------------------------

class TestUnknownToolBlocked:
    def test_unknown_tool_blocked_by_default(self):
        guard = ToolSafetyGuard()
        decision = guard.check(ToolCallRequest(tool_name="hack_server", params={}))
        assert not decision.allowed
        assert "Unknown tool" in decision.reason
        assert len(decision.violations) == 1

    def test_unknown_tool_uses_default_risk(self):
        config = ToolGuardConfig(default_risk="critical")
        guard = ToolSafetyGuard(config=config)
        decision = guard.check(ToolCallRequest(tool_name="nope", params={}))
        assert decision.risk_level == "critical"


# ---------------------------------------------------------------------------
# Invalid parameter blocked
# ---------------------------------------------------------------------------

class TestInvalidParameterBlocked:
    def test_unexpected_param_rejected(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="search", allowed_params=["query"]))
        decision = guard.check(ToolCallRequest(
            tool_name="search",
            params={"query": "ok", "malicious_flag": "true"},
        ))
        assert not decision.allowed
        assert any("malicious_flag" in v for v in decision.violations)

    def test_multiple_invalid_params_all_reported(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="t", allowed_params=["a"]))
        decision = guard.check(ToolCallRequest(
            tool_name="t",
            params={"a": "ok", "b": "bad", "c": "bad"},
        ))
        assert not decision.allowed
        param_violations = [v for v in decision.violations if "Parameter not allowed" in v]
        assert len(param_violations) == 2


# ---------------------------------------------------------------------------
# Dangerous parameter value blocked
# ---------------------------------------------------------------------------

class TestDangerousValueBlocked:
    def test_dangerous_pattern_in_value(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="run"))
        decision = guard.check(ToolCallRequest(
            tool_name="run",
            params={"cmd": "rm -rf /"},
        ))
        assert not decision.allowed
        assert any("rm -rf" in v for v in decision.violations)

    def test_eval_pattern_blocked(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="calc"))
        decision = guard.check(ToolCallRequest(
            tool_name="calc",
            params={"expr": "eval(user_input)"},
        ))
        assert not decision.allowed

    def test_tool_specific_blocked_values(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(
            name="query",
            blocked_values=["UNION SELECT"],
        ))
        decision = guard.check(ToolCallRequest(
            tool_name="query",
            params={"sql": "1 UNION SELECT * FROM users"},
        ))
        assert not decision.allowed
        assert any("UNION SELECT" in v for v in decision.violations)

    def test_safe_value_allowed(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="search"))
        decision = guard.check(ToolCallRequest(
            tool_name="search",
            params={"query": "python tutorials"},
        ))
        assert decision.allowed


# ---------------------------------------------------------------------------
# Rate limit enforcement
# ---------------------------------------------------------------------------

class TestRateLimitEnforcement:
    def test_exceeds_max_calls(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="api", max_calls_per_session=2))
        request = ToolCallRequest(tool_name="api", params={})
        assert guard.check(request).allowed
        assert guard.check(request).allowed
        decision = guard.check(request)
        assert not decision.allowed
        assert any("Rate limit" in v for v in decision.violations)

    def test_unlimited_calls_when_zero(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="ping", max_calls_per_session=0))
        request = ToolCallRequest(tool_name="ping", params={})
        for _ in range(100):
            assert guard.check(request).allowed


# ---------------------------------------------------------------------------
# Batch checking
# ---------------------------------------------------------------------------

class TestBatchChecking:
    def test_batch_returns_list_of_decisions(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="a"))
        guard.register_tool(ToolDefinition(name="b"))
        requests = [
            ToolCallRequest(tool_name="a", params={}),
            ToolCallRequest(tool_name="b", params={}),
            ToolCallRequest(tool_name="unknown", params={}),
        ]
        decisions = guard.check_batch(requests)
        assert len(decisions) == 3
        assert decisions[0].allowed
        assert decisions[1].allowed
        assert not decisions[2].allowed


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------

class TestSessionReset:
    def test_reset_clears_rate_limit(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="api", max_calls_per_session=1))
        request = ToolCallRequest(tool_name="api", params={}, session_id="s1")
        assert guard.check(request).allowed
        assert not guard.check(request).allowed
        guard.reset_session("s1")
        assert guard.check(request).allowed

    def test_reset_nonexistent_session_is_safe(self):
        guard = ToolSafetyGuard()
        guard.reset_session("does_not_exist")


# ---------------------------------------------------------------------------
# List tools
# ---------------------------------------------------------------------------

class TestListTools:
    def test_list_registered_tools(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="alpha"))
        guard.register_tool(ToolDefinition(name="beta"))
        names = guard.list_tools()
        assert "alpha" in names
        assert "beta" in names
        assert len(names) == 2

    def test_empty_when_none_registered(self):
        guard = ToolSafetyGuard()
        assert guard.list_tools() == []


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStatsTracking:
    def test_stats_count_allowed_and_blocked(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="ok"))
        guard.check(ToolCallRequest(tool_name="ok", params={}))
        guard.check(ToolCallRequest(tool_name="bad", params={}))
        s = guard.stats()
        assert s.total_requests == 2
        assert s.allowed == 1
        assert s.blocked == 1

    def test_stats_by_tool(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="search"))
        guard.check(ToolCallRequest(tool_name="search", params={}))
        guard.check(ToolCallRequest(tool_name="search", params={}))
        assert guard.stats().by_tool["search"] == 2

    def test_stats_by_reason(self):
        guard = ToolSafetyGuard()
        guard.check(ToolCallRequest(tool_name="x", params={}))
        assert guard.stats().by_reason["unknown_tool"] == 1


# ---------------------------------------------------------------------------
# Config: allow unknown tools
# ---------------------------------------------------------------------------

class TestAllowUnknownTools:
    def test_unknown_tool_allowed_when_configured(self):
        config = ToolGuardConfig(block_unknown_tools=False)
        guard = ToolSafetyGuard(config=config)
        decision = guard.check(ToolCallRequest(tool_name="anything", params={}))
        assert decision.allowed
        assert "allowed by config" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Multiple sessions
# ---------------------------------------------------------------------------

class TestMultipleSessions:
    def test_rate_limits_are_per_session(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="api", max_calls_per_session=1))
        r1 = ToolCallRequest(tool_name="api", params={}, session_id="user_a")
        r2 = ToolCallRequest(tool_name="api", params={}, session_id="user_b")
        assert guard.check(r1).allowed
        assert guard.check(r2).allowed
        assert not guard.check(r1).allowed
        assert not guard.check(r2).allowed


# ---------------------------------------------------------------------------
# Tool with no param restrictions
# ---------------------------------------------------------------------------

class TestNoParamRestrictions:
    def test_any_params_allowed_when_no_allowlist(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="flexible"))
        decision = guard.check(ToolCallRequest(
            tool_name="flexible",
            params={"anything": "goes", "extra": 42},
        ))
        assert decision.allowed


# ---------------------------------------------------------------------------
# Empty params
# ---------------------------------------------------------------------------

class TestEmptyParams:
    def test_empty_params_always_valid(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="ping", allowed_params=["host"]))
        decision = guard.check(ToolCallRequest(tool_name="ping", params={}))
        assert decision.allowed

    def test_no_params_field_defaults_to_empty(self):
        guard = ToolSafetyGuard()
        guard.register_tool(ToolDefinition(name="ping"))
        decision = guard.check(ToolCallRequest(tool_name="ping"))
        assert decision.allowed


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestDataclassDefaults:
    def test_tool_definition_defaults(self):
        td = ToolDefinition(name="t")
        assert td.allowed_params == []
        assert td.blocked_values == []
        assert td.max_calls_per_session == 0
        assert td.risk_level == "low"

    def test_tool_guard_config_defaults(self):
        cfg = ToolGuardConfig()
        assert cfg.default_risk == "medium"
        assert cfg.block_unknown_tools is True
        assert cfg.block_dangerous_params is True
        assert "rm -rf" in cfg.dangerous_patterns

    def test_tool_guard_stats_defaults(self):
        s = ToolGuardStats()
        assert s.total_requests == 0
        assert s.allowed == 0
        assert s.blocked == 0
        assert s.by_tool == {}
        assert s.by_reason == {}


# ---------------------------------------------------------------------------
# Dangerous params disabled via config
# ---------------------------------------------------------------------------

class TestDangerousParamsDisabled:
    def test_dangerous_values_pass_when_disabled(self):
        config = ToolGuardConfig(block_dangerous_params=False)
        guard = ToolSafetyGuard(config=config)
        guard.register_tool(ToolDefinition(name="shell"))
        decision = guard.check(ToolCallRequest(
            tool_name="shell",
            params={"cmd": "rm -rf /"},
        ))
        assert decision.allowed
