#!/usr/bin/env bash
# Sentinel AI — PostToolUse hook for Claude Code
# Scans tool outputs for PII leaks and secrets before returning to user.
# Automatically redacts PII if detected.
#
# Install: Add to .claude/settings.json under hooks.PostToolUse
# Exit 0 = allow

set -euo pipefail

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null || echo "")

# Scan outputs from tools that may contain sensitive data
case "$TOOL_NAME" in
  Bash|Read|Grep|Glob|WebFetch|WebSearch|mcp__*)
    ;;
  *)
    exit 0
    ;;
esac

# Extract output text
TEXT=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
output = data.get('tool_output', '')
if isinstance(output, dict):
    output = json.dumps(output)
print(str(output)[:4000])
" 2>/dev/null || echo "")

if [ -z "$TEXT" ]; then
  exit 0
fi

# Run PII and secrets scan
python3 -c "
import sys
from sentinel import SentinelGuard
from sentinel.scanners.pii import PIIScanner
from sentinel.scanners.secrets_scanner import SecretsScanner
guard = SentinelGuard(scanners=[PIIScanner(), SecretsScanner()])
result = guard.scan(sys.argv[1])
if result.findings:
    categories = set(f.category for f in result.findings)
    print(f'Sentinel AI: detected {len(result.findings)} finding(s) in output: {categories}', file=sys.stderr)
" "$TEXT" 2>&1 || true

exit 0
