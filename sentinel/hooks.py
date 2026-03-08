"""Claude Code hooks integration for Sentinel AI.

Provides a PreToolUse hook that scans tool calls for safety issues
before they execute. Blocks dangerous commands, PII exfiltration,
prompt injection attempts, and CLAUDE.md rule violations in real time.

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

CLAUDE.md enforcement:
  If a CLAUDE.md file exists in the project root, the hook automatically
  extracts prohibition rules and checks tool calls against them. This
  converts soft model instructions into hard deterministic blocks.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from sentinel.core import SentinelGuard, RiskLevel
from sentinel.scanners.tool_use import ToolUseScanner


# Cache the ClaudeMdEnforcer instance across calls within the same process
_enforcer_cache: dict[str, object] = {}


def _find_claudemd() -> Path | None:
    """Find CLAUDE.md in the current directory or parents."""
    cwd = Path.cwd()
    for directory in [cwd, *cwd.parents]:
        candidate = directory / "CLAUDE.md"
        if candidate.is_file():
            return candidate
        # Stop at git root
        if (directory / ".git").exists():
            break
    return None


def _get_enforcer():
    """Get or create a ClaudeMdEnforcer from CLAUDE.md."""
    from sentinel.claudemd_enforcer import ClaudeMdEnforcer

    md_path = _find_claudemd()
    if md_path is None:
        return None

    # Cache by path + mtime to avoid re-parsing on every call
    key = str(md_path)
    try:
        mtime = md_path.stat().st_mtime
    except OSError:
        return None

    cached = _enforcer_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        enforcer = ClaudeMdEnforcer.from_file(md_path)
        _enforcer_cache[key] = (mtime, enforcer)
        return enforcer
    except Exception:
        return None


def _get_guard_policy():
    """Get a GuardPolicy from .sentinel/policy.json or sentinel-policy.yaml if present."""
    from sentinel.guard_policy import GuardPolicy

    cwd = Path.cwd()
    candidates = [
        cwd / ".sentinel" / "policy.json",
        cwd / ".sentinel" / "policy.yaml",
        cwd / "sentinel-policy.yaml",
        cwd / "sentinel-policy.json",
    ]

    for path in candidates:
        if path.is_file():
            try:
                return GuardPolicy.from_yaml(str(path))
            except Exception:
                continue

    return None


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

    block_reasons: list[str] = []

    # --- Layer 1: CLAUDE.md rule enforcement ---
    enforcer = _get_enforcer()
    if enforcer is not None:
        verdict = enforcer.check(tool_name, tool_input)
        if not verdict.allowed:
            for rule in verdict.violated_rules:
                block_reasons.append(f"[CLAUDE.md] {rule}")

    # --- Layer 2: Guard policy enforcement ---
    policy = _get_guard_policy()
    if policy is not None:
        guard = policy.create_guard()
        policy_verdict = guard.check(tool_name, tool_input)
        if not policy_verdict.allowed:
            for reason in policy_verdict.findings:
                block_reasons.append(f"[POLICY] {reason}")

    # --- Layer 3: Built-in safety scanning ---
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
            guard_instance = SentinelGuard.default()
            result = guard_instance.scan(value)
            # Only flag high+ risk findings from content scanning
            findings.extend(f for f in result.findings if f.risk >= RiskLevel.HIGH)

    # Add high-risk findings to block reasons
    for f in findings:
        if f.risk >= RiskLevel.HIGH:
            block_reasons.append(f"[{f.risk.value.upper()}] {f.description}")

    # --- Decision ---
    if block_reasons:
        message = (
            f"Sentinel AI blocked this {tool_name} call:\n"
            + "\n".join(f"  - {r}" for r in block_reasons[:5])
        )
        print(message, file=sys.stdout)
        return 2  # Exit code 2 = block

    return 0  # Allow


if __name__ == "__main__":
    sys.exit(run_hook())
