"""Auto-configure Sentinel AI for Claude Code and MCP clients.

Provides `sentinel init` functionality that sets up:
  - Claude Code hooks (.claude/settings.json) for PreToolUse scanning
  - MCP server config (claude_desktop_config.json) for tool-based scanning
  - Starter policy.yaml for custom policy configuration
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


_HOOKS_CONFIG = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": ".*",
                "hooks": [
                    {
                        "type": "command",
                        "command": "sentinel hook",
                    }
                ],
            }
        ],
    }
}

_MCP_CONFIG = {
    "sentinel-ai": {
        "command": "python",
        "args": ["-m", "sentinel.mcp_server"],
    }
}

_STARTER_POLICY = """\
# Sentinel AI Safety Policy
# See https://github.com/MaxwellCalkin/sentinel-ai for documentation

block_threshold: high
redact_pii: true

scanners:
  prompt_injection:
    enabled: true
  pii:
    enabled: true
  harmful_content:
    enabled: true
  toxicity:
    enabled: true
    profanity_risk: low
  hallucination:
    enabled: true
  tool_use:
    enabled: true
  structured_output:
    enabled: true
  blocked_terms:
    enabled: false
    terms: []
"""


def init_claude_code(project_dir: Path) -> list[str]:
    """Configure Claude Code hooks in the project's .claude/settings.json."""
    actions = []
    claude_dir = project_dir / ".claude"
    settings_path = claude_dir / "settings.json"

    claude_dir.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    else:
        settings = {}

    # Merge hooks config
    existing_hooks = settings.get("hooks", {})
    existing_pre = existing_hooks.get("PreToolUse", [])

    # Check if sentinel hook is already configured
    already_configured = any(
        any(
            h.get("command", "").startswith("sentinel hook")
            for h in entry.get("hooks", [])
        )
        for entry in existing_pre
    )

    if already_configured:
        actions.append("Claude Code hooks: already configured (skipped)")
    else:
        existing_pre.extend(_HOOKS_CONFIG["hooks"]["PreToolUse"])
        existing_hooks["PreToolUse"] = existing_pre
        settings["hooks"] = existing_hooks

        settings_path.write_text(
            json.dumps(settings, indent=2) + "\n", encoding="utf-8"
        )
        actions.append(f"Claude Code hooks: configured in {settings_path}")

    return actions


def init_mcp(project_dir: Path) -> list[str]:
    """Add Sentinel AI MCP server to Claude Desktop config."""
    actions = []

    # Try common locations for claude_desktop_config.json
    candidates = [
        project_dir / ".claude" / "claude_desktop_config.json",
        Path.home() / ".config" / "claude" / "claude_desktop_config.json",
        Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",
        Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
    ]

    config_path = None
    for c in candidates:
        if c.exists():
            config_path = c
            break

    if config_path is None:
        # Create in project .claude dir
        config_path = project_dir / ".claude" / "claude_desktop_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}

    mcp_servers = config.get("mcpServers", {})
    if "sentinel-ai" in mcp_servers:
        actions.append("MCP server: already configured (skipped)")
    else:
        mcp_servers.update(_MCP_CONFIG)
        config["mcpServers"] = mcp_servers
        config_path.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        actions.append(f"MCP server: configured in {config_path}")

    return actions


def init_policy(project_dir: Path) -> list[str]:
    """Create a starter policy.yaml if one doesn't exist."""
    actions = []
    policy_path = project_dir / "sentinel-policy.yaml"

    if policy_path.exists():
        actions.append("Policy file: sentinel-policy.yaml already exists (skipped)")
    else:
        policy_path.write_text(_STARTER_POLICY, encoding="utf-8")
        actions.append(f"Policy file: created {policy_path}")

    return actions


def run_init(
    project_dir: Path | None = None,
    hooks: bool = True,
    mcp: bool = True,
    policy: bool = True,
) -> list[str]:
    """Run full initialization. Returns list of actions taken."""
    if project_dir is None:
        project_dir = Path.cwd()

    actions = []

    if hooks:
        actions.extend(init_claude_code(project_dir))
    if mcp:
        actions.extend(init_mcp(project_dir))
    if policy:
        actions.extend(init_policy(project_dir))

    return actions
