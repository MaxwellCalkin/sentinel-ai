#!/usr/bin/env bash
# Sentinel AI — PreToolUse hook for Claude Code
# Scans tool inputs for prompt injection, PII, harmful content before execution.
#
# Install: Add to .claude/settings.json under hooks.PreToolUse
# Exit 0 = allow, Exit 2 = block (stderr fed back to Claude)

set -euo pipefail

# Read tool input from stdin (JSON with tool_name, tool_input fields)
INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null || echo "")

# Only scan tools that take user-facing text input
case "$TOOL_NAME" in
  Bash|Write|Edit|NotebookEdit|mcp__*)
    ;;
  *)
    exit 0
    ;;
esac

# Extract the text content to scan
TEXT=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
inp = data.get('tool_input', {})
# Extract text from various tool input formats
text = inp.get('command', '') or inp.get('content', '') or inp.get('new_string', '') or str(inp)
print(text[:4000])  # Limit to 4KB
" 2>/dev/null || echo "")

if [ -z "$TEXT" ]; then
  exit 0
fi

# Run Sentinel scan
RESULT=$(python3 -c "
import json, sys
from sentinel import SentinelGuard
guard = SentinelGuard.default()
result = guard.scan(sys.argv[1])
if result.blocked:
    findings = [{'category': f.category, 'risk': f.risk.value, 'description': f.description} for f in result.findings]
    print(json.dumps({'blocked': True, 'risk': result.risk.value, 'findings': findings}))
    sys.exit(2)
" "$TEXT" 2>&1)

EXIT_CODE=$?
if [ $EXIT_CODE -eq 2 ]; then
  echo "Sentinel AI: Content blocked — $RESULT" >&2
  exit 2
fi

exit 0
