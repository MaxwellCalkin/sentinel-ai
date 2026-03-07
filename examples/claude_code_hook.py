#!/usr/bin/env python3
"""Sentinel AI hook for Claude Code.

Use as a pre-tool-call hook in Claude Code to scan tool arguments
for safety issues before execution.

Setup:
    Add to your Claude Code settings (~/.claude/settings.json):

    {
      "hooks": {
        "PreToolUse": [
          {
            "matcher": "Bash",
            "command": "python /path/to/claude_code_hook.py"
          }
        ]
      }
    }

    The hook reads the tool call from stdin (JSON) and outputs
    a decision to stdout. If unsafe content is detected, it
    blocks the tool call with a reason.

Protocol:
    Input (stdin):  JSON with tool_name and tool_input fields
    Output (stdout): JSON with decision ("allow", "block", or "review")
                     and optional reason
"""

import json
import sys

from sentinel.scanners.tool_use import ToolUseScanner
from sentinel.scanners.pii import PIIScanner
from sentinel.core import RiskLevel


def main():
    # Read tool call from stdin
    try:
        raw = sys.stdin.read()
        tool_call = json.loads(raw)
    except (json.JSONDecodeError, KeyError):
        # Can't parse input — allow by default
        print(json.dumps({"decision": "allow"}))
        return

    tool_name = tool_call.get("tool_name", "")
    tool_input = tool_call.get("tool_input", {})

    # Initialize scanners
    tool_scanner = ToolUseScanner()
    pii_scanner = PIIScanner()

    # Scan the tool call
    findings = tool_scanner.scan_tool_call(tool_name, tool_input)

    # Also check for PII in arguments
    text_parts = []
    _collect_strings(tool_input, text_parts)
    combined_text = " ".join(text_parts)
    pii_findings = pii_scanner.scan(combined_text)

    all_findings = findings + pii_findings

    if not all_findings:
        print(json.dumps({"decision": "allow"}))
        return

    max_risk = max(f.risk for f in all_findings)

    if max_risk >= RiskLevel.HIGH:
        reasons = [f"{f.description} ({f.risk.value})" for f in all_findings]
        print(json.dumps({
            "decision": "block",
            "reason": f"Sentinel AI blocked this tool call: {'; '.join(reasons)}",
        }))
    elif max_risk >= RiskLevel.MEDIUM:
        reasons = [f"{f.description} ({f.risk.value})" for f in all_findings]
        print(json.dumps({
            "decision": "review",
            "reason": f"Sentinel AI flagged for review: {'; '.join(reasons)}",
        }))
    else:
        print(json.dumps({"decision": "allow"}))


def _collect_strings(obj, parts):
    if isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, parts)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_strings(item, parts)


if __name__ == "__main__":
    main()
