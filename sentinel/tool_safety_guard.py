"""Guard LLM tool/function calls before execution.

Validates parameters against allowlists, detects dangerous
invocations via pattern matching, and enforces per-session
rate limits. Zero external dependencies.

Usage:
    from sentinel.tool_safety_guard import ToolSafetyGuard, ToolDefinition

    guard = ToolSafetyGuard()
    guard.register_tool(ToolDefinition(name="search", allowed_params=["query", "limit"]))
    decision = guard.check(ToolCallRequest(tool_name="search", params={"query": "hello"}))
    assert decision.allowed
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    """Definition of an allowed tool and its constraints."""

    name: str
    allowed_params: list[str] = field(default_factory=list)
    blocked_values: list[str] = field(default_factory=list)
    max_calls_per_session: int = 0
    risk_level: str = "low"


@dataclass
class ToolCallRequest:
    """A request to invoke a tool."""

    tool_name: str
    params: dict = field(default_factory=dict)
    session_id: str = "default"


@dataclass
class ToolCallDecision:
    """Result of evaluating a tool call request."""

    request: ToolCallRequest
    allowed: bool
    reason: str
    risk_level: str
    violations: list[str] = field(default_factory=list)


@dataclass
class ToolGuardConfig:
    """Configuration for the tool safety guard."""

    default_risk: str = "medium"
    block_unknown_tools: bool = True
    block_dangerous_params: bool = True
    dangerous_patterns: list[str] = field(default_factory=lambda: [
        "rm -rf",
        "DROP TABLE",
        "eval(",
        "exec(",
        "os.system",
        "__import__",
        "subprocess",
    ])


@dataclass
class ToolGuardStats:
    """Cumulative statistics for the guard."""

    total_requests: int = 0
    allowed: int = 0
    blocked: int = 0
    by_tool: dict[str, int] = field(default_factory=dict)
    by_reason: dict[str, int] = field(default_factory=dict)


class ToolSafetyGuard:
    """Guard LLM tool calls with allowlists, parameter validation,
    dangerous pattern detection, and per-session rate limiting.
    """

    def __init__(self, config: ToolGuardConfig | None = None) -> None:
        self._config = config or ToolGuardConfig()
        self._tools: dict[str, ToolDefinition] = {}
        self._session_counts: dict[str, dict[str, int]] = {}
        self._stats = ToolGuardStats()

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool as allowed."""
        self._tools[tool.name] = tool

    def check(self, request: ToolCallRequest) -> ToolCallDecision:
        """Evaluate a tool call request against all safety checks."""
        violations: list[str] = []

        self._record_tool_stat(request.tool_name)

        if not self._is_known_tool(request.tool_name):
            return self._block_unknown(request, violations)

        tool = self._tools[request.tool_name]

        self._validate_params(tool, request.params, violations)
        self._check_dangerous_values(tool, request.params, violations)
        self._check_rate_limit(tool, request, violations)

        return self._build_decision(request, tool, violations)

    def check_batch(self, requests: list[ToolCallRequest]) -> list[ToolCallDecision]:
        """Evaluate multiple tool call requests."""
        return [self.check(request) for request in requests]

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def reset_session(self, session_id: str) -> None:
        """Clear call counts for a session."""
        self._session_counts.pop(session_id, None)

    def stats(self) -> ToolGuardStats:
        """Return cumulative guard statistics."""
        return self._stats

    # -- Private helpers (one responsibility each) --

    def _is_known_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def _block_unknown(
        self, request: ToolCallRequest, violations: list[str]
    ) -> ToolCallDecision:
        if self._config.block_unknown_tools:
            reason = f"Unknown tool: {request.tool_name}"
            violations.append(reason)
            self._record_decision(blocked=True, reason="unknown_tool")
            return ToolCallDecision(
                request=request,
                allowed=False,
                reason=reason,
                risk_level=self._config.default_risk,
                violations=violations,
            )
        self._record_decision(blocked=False, reason="allowed")
        return ToolCallDecision(
            request=request,
            allowed=True,
            reason="Unknown tool allowed by config",
            risk_level=self._config.default_risk,
        )

    def _validate_params(
        self,
        tool: ToolDefinition,
        params: dict,
        violations: list[str],
    ) -> None:
        if not tool.allowed_params:
            return
        for param_name in params:
            if param_name not in tool.allowed_params:
                violations.append(f"Parameter not allowed: {param_name}")

    def _check_dangerous_values(
        self,
        tool: ToolDefinition,
        params: dict,
        violations: list[str],
    ) -> None:
        if not self._config.block_dangerous_params:
            return
        all_blocked = list(self._config.dangerous_patterns) + list(tool.blocked_values)
        for param_name, value in params.items():
            text = str(value)
            for pattern in all_blocked:
                if pattern in text:
                    violations.append(
                        f"Dangerous value in '{param_name}': contains '{pattern}'"
                    )

    def _check_rate_limit(
        self,
        tool: ToolDefinition,
        request: ToolCallRequest,
        violations: list[str],
    ) -> None:
        if tool.max_calls_per_session <= 0:
            return
        session_counts = self._session_counts.setdefault(request.session_id, {})
        current = session_counts.get(tool.name, 0)
        if current >= tool.max_calls_per_session:
            violations.append(
                f"Rate limit exceeded: {tool.name} called {current}/{tool.max_calls_per_session}"
            )
        session_counts[tool.name] = current + 1

    def _build_decision(
        self,
        request: ToolCallRequest,
        tool: ToolDefinition,
        violations: list[str],
    ) -> ToolCallDecision:
        if violations:
            reason = "; ".join(violations)
            self._record_decision(blocked=True, reason="violation")
            return ToolCallDecision(
                request=request,
                allowed=False,
                reason=reason,
                risk_level=tool.risk_level,
                violations=violations,
            )
        self._record_decision(blocked=False, reason="allowed")
        return ToolCallDecision(
            request=request,
            allowed=True,
            reason="All checks passed",
            risk_level=tool.risk_level,
        )

    def _record_tool_stat(self, tool_name: str) -> None:
        self._stats.total_requests += 1
        self._stats.by_tool[tool_name] = self._stats.by_tool.get(tool_name, 0) + 1

    def _record_decision(self, *, blocked: bool, reason: str) -> None:
        if blocked:
            self._stats.blocked += 1
        else:
            self._stats.allowed += 1
        self._stats.by_reason[reason] = self._stats.by_reason.get(reason, 0) + 1
