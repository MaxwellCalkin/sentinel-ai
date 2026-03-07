"""Using Sentinel AI with the Anthropic Claude SDK.

Wraps Claude API calls with safety scanning on both input and output.
Requires: pip install anthropic sentinel-ai
"""

from anthropic import Anthropic
from sentinel.middleware.anthropic_wrapper import guarded_message

client = Anthropic()

# Every message is automatically scanned for safety
result = guarded_message(
    client,
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
)

if not result["blocked"]:
    print(result["response"].content[0].text)

    # Check output scan results
    if result["output_scan"] and result["output_scan"].findings:
        print(f"\nOutput findings: {len(result['output_scan'].findings)}")
else:
    print(f"Blocked: {result['block_reason']}")
