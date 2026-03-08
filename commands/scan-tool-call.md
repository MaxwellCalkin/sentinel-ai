# /scan-tool-call — Check if a tool call is safe

Scan a tool call for dangerous operations before execution.

## Usage

/scan-tool-call <tool_name> <arguments as JSON>

## Behavior

Use the Sentinel AI `scan_tool_call` MCP tool to:
1. Parse the tool name and JSON arguments
2. Check for dangerous shell commands (rm -rf, credential access, etc.)
3. Check for data exfiltration patterns (curl to external servers, etc.)
4. Check for privilege escalation attempts
5. Report risk level and whether the call should be blocked

Present findings with:
- Risk level (NONE, LOW, MEDIUM, HIGH, CRITICAL)
- Threat type (dangerous_command, exfiltration, sensitive_file, privilege_escalation)
- Description of the threat
- Whether the tool call would be blocked

## Input

$ARGUMENTS
