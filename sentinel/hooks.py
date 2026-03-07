"""Claude Code hooks integration for Sentinel AI.

Provides a PreToolUse hook that scans tool calls for safety issues
before they execute. Blocks dangerous commands, PII exfiltration,
and prompt injection attempts in real time.

Setup — add to .claude/settings.json:
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "sentinel hook"
          }
        ]
      }
    ]
  }
}

The hook reads a JSON event from stdin and:
  - Exits 0 (allow) if the tool call is safe
  - Exits 2 (block) with a reason on stdout if unsafe
"""

from __future__ import annotations

import json
import sys

from sentinel.core import SentinelGuard, RiskLevel
from sentinel.scanners.tool_use import ToolUseScanner


def run_hook() -> int:
    """Run the sentinel hook, reading a Claude Code hook event from stdin."""
    try:
        raw = sys.stdin.read()
    except Exception:
        return 0  # Allow on stdin read failure (non-blocking)

    if not raw.strip():
        return 0

    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return 0  # Allow on malformed input

    tool_name = event.get("tool_name", "")
    tool_input = event.get("tool_input", {})

    # Skip scanning for read-only / low-risk tools
    _SAFE_TOOLS = {
        "Read", "Glob", "Grep", "WebSearch", "WebFetch",
        "AskUserQuestion", "TodoRead", "ToolSearch",
    }
    if tool_name in _SAFE_TOOLS:
        return 0

    # Scan tool call with ToolUseScanner
    scanner = ToolUseScanner()
    findings = []

    if tool_name in ("Bash", "bash", "terminal", "shell"):
        command = tool_input.get("command", "")
        if command:
            findings.extend(scanner.scan(command))
            findings.extend(scanner.scan_tool_call(tool_name, tool_input))
    elif tool_name in ("Write", "Edit", "NotebookEdit"):
        # Scan file content for injected payloads
        content = tool_input.get("content", "") or tool_input.get("new_string", "")
        if content:
            guard = SentinelGuard.default()
            result = guard.scan(content)
            findings.extend(result.findings)
    else:
        # Generic tool call scan
        findings.extend(scanner.scan_tool_call(tool_name, tool_input))

    # Also scan any string arguments for prompt injection / harmful content
    for key, value in tool_input.items():
        if isinstance(value, str) and len(value) > 10:
            guard = SentinelGuard.default()
            result = guard.scan(value)
            # Only flag high+ risk findings from content scanning
            findings.extend(f for f in result.findings if f.risk >= RiskLevel.HIGH)

    if not findings:
        return 0

    max_risk = max(f.risk for f in findings)

    if max_risk >= RiskLevel.HIGH:
        # Block the tool call
        reasons = []
        for f in findings:
            if f.risk >= RiskLevel.HIGH:
                reasons.append(f"[{f.risk.value.upper()}] {f.description}")
        message = (
            f"Sentinel AI blocked this {tool_name} call:\n"
            + "\n".join(f"  - {r}" for r in reasons[:5])
        )
        print(message, file=sys.stdout)
        return 2  # Exit code 2 = block

    return 0  # Allow medium/low risk


if __name__ == "__main__":
    sys.exit(run_hook())
