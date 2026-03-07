# /scan — Scan text for safety issues

Scan the provided text for safety issues using all Sentinel AI scanners.

## Usage

/scan <text to scan>

## Behavior

Run all 7 Sentinel AI safety scanners against the provided text:
- Prompt injection detection
- PII detection and redaction
- Harmful content filtering
- Hallucination signal detection
- Toxicity detection
- Blocked terms checking
- Tool-use safety analysis

Report findings with risk levels (NONE, LOW, MEDIUM, HIGH, CRITICAL) and whether the content should be blocked.

If PII is detected, also show the redacted version of the text.

## Input

$ARGUMENTS
