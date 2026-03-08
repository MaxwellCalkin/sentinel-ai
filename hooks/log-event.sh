#!/usr/bin/env bash
# Sentinel AI — Event logger hook for Claude Code
# Logs all tool calls to a local audit file for security review.
#
# Creates ~/.sentinel/audit.jsonl with timestamped entries.

set -euo pipefail

AUDIT_DIR="${HOME}/.sentinel"
AUDIT_FILE="${AUDIT_DIR}/audit.jsonl"

mkdir -p "$AUDIT_DIR"

INPUT=$(cat)

python3 -c "
import json, sys, datetime
data = json.load(sys.stdin)
entry = {
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'tool': data.get('tool_name', 'unknown'),
    'event': data.get('hook_event', 'unknown'),
}
# Add truncated input for audit (no secrets)
tool_input = data.get('tool_input', {})
if isinstance(tool_input, dict):
    entry['input_keys'] = list(tool_input.keys())
print(json.dumps(entry))
" <<< "$INPUT" >> "$AUDIT_FILE" 2>/dev/null || true

exit 0
