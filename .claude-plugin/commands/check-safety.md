# /check-safety — Generate a safety risk report

Generate a detailed safety risk report for the current conversation or a specific file.

## Usage

/check-safety [file path or text]

## Behavior

Use the Sentinel AI `get_risk_report` MCP tool to generate a comprehensive safety analysis:
1. If a file path is provided, read the file and scan its contents
2. If text is provided, scan that text directly
3. If nothing is provided, scan the most recent assistant response in the conversation

Present the report showing:
- Overall risk level
- Whether content would be blocked
- Findings grouped by category (prompt_injection, pii, harmful_content, toxicity, hallucination)
- Specific findings with risk levels
- Redacted text if PII was found

## Input

$ARGUMENTS
