"""Using Sentinel AI with the Claude Agent SDK for safe agentic workflows.

Wraps agent tool execution with safety scanning — every tool call is scanned
before execution, and every tool result is scanned before being sent back.

Requires: pip install claude-agent-sdk sentinel-guardrails
"""

from sentinel import SentinelGuard

guard = SentinelGuard.default()


def safe_tool_wrapper(tool_fn):
    """Wrap any agent tool function with Sentinel safety scanning.

    Scans the tool input before execution and the output after execution.
    Blocks dangerous tool calls (prompt injection, data exfiltration, etc.)
    and redacts PII from tool outputs.
    """

    def wrapper(input_text: str) -> str:
        # Scan input before executing tool
        input_scan = guard.scan(input_text)
        if input_scan.blocked:
            return f"[BLOCKED] Tool input rejected: {input_scan.findings[0].description}"

        # Execute the tool
        result = tool_fn(input_text)

        # Scan output before returning
        output_scan = guard.scan(result)
        if output_scan.blocked:
            return f"[BLOCKED] Tool output rejected: {output_scan.findings[0].description}"

        # Return redacted output if PII was found
        if output_scan.redacted_text:
            return output_scan.redacted_text

        return result

    return wrapper


# --- Example: Define agent tools with safety scanning ---


@safe_tool_wrapper
def search_database(query: str) -> str:
    """Simulated database search tool."""
    return f"Results for '{query}': No records found."


@safe_tool_wrapper
def read_file(path: str) -> str:
    """Simulated file read tool."""
    return f"Contents of {path}: [file data]"


# --- Usage ---

if __name__ == "__main__":
    # Safe tool call — passes through
    print(search_database("active users"))

    # Injection attempt — blocked by Sentinel
    print(search_database("'; DROP TABLE users; --"))

    # Exfiltration attempt — blocked
    print(read_file("/etc/passwd"))

    # Tool returning PII — auto-redacted
    print()
    print("--- Pre-execution scanning catches attacks in ~0.05ms ---")
    print("--- No GPU required, no API calls, deterministic ---")
