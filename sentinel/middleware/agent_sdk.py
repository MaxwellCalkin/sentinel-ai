"""Claude Agent SDK integration for Sentinel AI.

Provides PreToolUse hooks and tool permission callbacks that use
Sentinel AI's safety scanners to protect agentic tool execution.

Usage with hooks:
    from sentinel.middleware.agent_sdk import sentinel_pretooluse_hook
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

    options = ClaudeAgentOptions(
        hooks={
            "PreToolUse": [
                HookMatcher(matcher=".*", hooks=[sentinel_pretooluse_hook]),
            ],
        }
    )

Usage with permission callback:
    from sentinel.middleware.agent_sdk import sentinel_permission_callback

    options = ClaudeAgentOptions(
        can_use_tool=sentinel_permission_callback,
    )
"""

from __future__ import annotations

from typing import Any

from sentinel.core import SentinelGuard, RiskLevel
from sentinel.scanners.tool_use import ToolUseScanner

# Read-only tools that don't need scanning
_SAFE_TOOLS = {"Read", "Glob", "Grep", "LS"}


def _extract_tool_text(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Extract scannable text from tool input based on tool type."""
    if tool_name == "Bash":
        return tool_input.get("command", "")
    elif tool_name == "Write":
        return tool_input.get("content", "")
    elif tool_name == "Edit":
        return tool_input.get("new_string", "")
    elif tool_name == "MultiEdit":
        edits = tool_input.get("edits", [])
        return " ".join(e.get("new_string", "") for e in edits if isinstance(e, dict))
    return ""


async def sentinel_pretooluse_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """PreToolUse hook for Claude Agent SDK.

    Scans tool calls with Sentinel AI before execution.
    Returns deny decision for HIGH+ risk findings.
    """
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    if tool_name in _SAFE_TOOLS:
        return {}

    guard = SentinelGuard.default()
    text_to_scan = _extract_tool_text(tool_name, tool_input)

    if text_to_scan:
        result = guard.scan(text_to_scan)
        if result.blocked:
            desc = result.findings[0].description if result.findings else "Safety violation"
            return {
                "reason": f"Sentinel AI: {desc}",
                "systemMessage": f"Blocked by Sentinel AI safety scan (risk: {result.risk.value})",
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Sentinel AI detected {result.risk.value} risk: {desc}",
                },
            }

    scanner = ToolUseScanner()
    findings = scanner.scan_tool_call(tool_name, tool_input)
    if findings:
        max_risk = max(f.risk for f in findings)
        if max_risk >= RiskLevel.HIGH:
            return {
                "reason": f"Sentinel AI: {findings[0].description}",
                "systemMessage": f"Blocked by Sentinel AI (risk: {max_risk.value})",
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Sentinel AI: {findings[0].description}",
                },
            }

    return {}


async def sentinel_permission_callback(
    tool_name: str,
    input_data: dict[str, Any],
    context: Any,
) -> Any:
    """Tool permission callback for Claude Agent SDK.

    Uses Sentinel AI to allow/deny tool execution.
    Returns PermissionResultAllow or PermissionResultDeny.

    Requires claude-agent-sdk to be installed for return types.
    """
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    if tool_name in _SAFE_TOOLS:
        return PermissionResultAllow()

    guard = SentinelGuard.default()
    text_to_scan = _extract_tool_text(tool_name, input_data)

    if text_to_scan:
        result = guard.scan(text_to_scan)
        if result.blocked:
            desc = result.findings[0].description if result.findings else "Safety violation"
            return PermissionResultDeny(
                message=f"Sentinel AI: {desc} (risk: {result.risk.value})"
            )

    scanner = ToolUseScanner()
    findings = scanner.scan_tool_call(tool_name, input_data)
    if findings:
        max_risk = max(f.risk for f in findings)
        if max_risk >= RiskLevel.HIGH:
            return PermissionResultDeny(
                message=f"Sentinel AI: {findings[0].description} (risk: {max_risk.value})"
            )

    return PermissionResultAllow()
