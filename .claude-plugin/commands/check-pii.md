# /check-pii — Check text for PII and redact

Check text or a file for personally identifiable information and show the redacted version.

## Usage

/check-pii <text or file path>

## Behavior

Use the Sentinel AI `check_pii` MCP tool to:
1. Scan the provided text for PII (emails, SSNs, credit cards, phone numbers, API keys, tokens)
2. Report what PII types were found
3. Show the redacted version with PII replaced by labels like [EMAIL], [SSN], [CREDIT_CARD]

If a file path is provided instead of text, read the file first then scan its contents.

## Input

$ARGUMENTS
