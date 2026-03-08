# /red-team — Test safety scanning robustness

Red-team your safety scanning by generating adversarial variants of a prompt injection attempt.

## Usage

/red-team <injection attempt text>

## Behavior

Use the Sentinel AI `test_robustness` MCP tool to:
1. Take the provided text (a known injection attempt)
2. Generate adversarial variants using 10+ evasion techniques (homoglyphs, zero-width characters, leetspeak, payload splitting, synonym substitution, etc.)
3. Test whether each variant is still detected by the safety scanner
4. Report the detection rate, total variants tested, and any variants that evaded detection

Present results showing:
- Overall detection rate (e.g., 95%)
- Number of variants tested vs detected
- List of evasion techniques used
- Any variants that evaded detection (with the technique used)

This helps identify weaknesses in safety scanning before they can be exploited.

## Input

$ARGUMENTS
