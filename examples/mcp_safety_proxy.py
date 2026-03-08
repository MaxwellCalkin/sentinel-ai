"""MCP Safety Proxy example.

Shows how to use Sentinel AI as a transparent safety layer
between Claude and any MCP server.

Usage (CLI):
    sentinel mcp-proxy -- npx @modelcontextprotocol/server-filesystem /tmp

Usage (Claude Desktop config):
    {
      "mcpServers": {
        "safe-filesystem": {
          "command": "sentinel",
          "args": ["mcp-proxy", "--", "npx", "@modelcontextprotocol/server-filesystem", "/tmp"]
        }
      }
    }

What it does:
    - Scans all tool arguments for shell injection, data exfiltration, prompt injection
    - Auto-redacts PII (emails, SSNs, credit cards) in tool responses
    - Blocks dangerous operations before they reach the upstream server
    - Logs all actions to stderr for audit trail

Configurable:
    sentinel mcp-proxy --block-on critical -- your-server  # Only block critical risks
    sentinel mcp-proxy --block-on low -- your-server       # Block everything suspicious
"""

from sentinel.mcp_proxy import run_proxy
from sentinel.core import RiskLevel

# Programmatic usage (equivalent to CLI)
if __name__ == "__main__":
    import sys

    upstream_cmd = sys.argv[1:] or ["echo", "No upstream server specified"]
    exit(run_proxy(upstream_cmd, block_threshold=RiskLevel.HIGH))
