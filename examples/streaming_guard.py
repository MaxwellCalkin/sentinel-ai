"""Real-time streaming protection with Sentinel AI.

Scans LLM output token-by-token as it streams, blocking mid-stream
if dangerous content is detected.
"""

from sentinel.streaming import StreamingGuard

guard = StreamingGuard(buffer_size=200)

# Simulate a streaming LLM response
chunks = [
    "Sure, I can help you with that. ",
    "The capital of France is Paris. ",
    "It has a population of about 2.1 million people. ",
    "The city is known for the Eiffel Tower.",
]

print("Streaming output:")
for chunk in chunks:
    result = guard.feed(chunk)
    if result.blocked:
        print(f"\n[BLOCKED] Stream halted - risk: {result.risk.value}")
        break
    if result.safe_text:
        print(result.safe_text, end="", flush=True)

# Flush remaining buffer
final = guard.finalize()
print(final.safe_text)
print(f"\nTotal findings: {len(guard.all_findings)}")
