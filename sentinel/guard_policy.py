"""Guard Policy — declarative YAML/dict policy engine for SessionGuard.

Define security policies as YAML files and load them into SessionGuard
for consistent, auditable, version-controlled safety configuration.

Usage:
    from sentinel.guard_policy import GuardPolicy

    # From YAML file
    policy = GuardPolicy.from_yaml("guard_policy.yaml")

    # From dict
    policy = GuardPolicy.from_dict({
        "block_on": "high",
        "blocked_commands": ["rm -rf", "mkfs"],
        "sensitive_paths": [".env", ".ssh/id_rsa"],
        "allowed_tools": ["bash", "read_file", "Write"],
        "denied_tools": ["curl", "wget"],
        "max_risk_per_tool": {"bash": "high", "read_file": "critical"},
        "rate_limits": {"bash": 50, "default": 100},
    })

    # Apply to SessionGuard
    guard = policy.create_guard(session_id="s-1", user_id="user@org.com")

Example YAML (guard_policy.yaml):
    version: "1.0"
    block_on: high
    blocked_commands:
      - rm -rf
      - mkfs
      - "dd if=/dev/"
    sensitive_paths:
      - .env
      - .aws/credentials
      - .ssh/id_rsa
    allowed_tools:
      - bash
      - read_file
      - Write
      - Edit
    denied_tools: []
    max_risk_per_tool:
      bash: high
      read_file: critical
    rate_limits:
      bash: 50
      default: 200
    custom_blocks:
      - pattern: "npm publish"
        reason: "npm_publish_blocked"
      - pattern: "/prod/"
        reason: "production_access_blocked"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sentinel.session_guard import SessionGuard


@dataclass
class CustomBlock:
    """A custom blocking rule defined by pattern and reason."""

    pattern: str
    reason: str
    _compiled: re.Pattern | None = field(default=None, repr=False)

    def matches(self, text: str) -> bool:
        if self._compiled is None:
            self._compiled = re.compile(self.pattern, re.IGNORECASE)
        return bool(self._compiled.search(text))


@dataclass
class GuardPolicy:
    """Declarative security policy for SessionGuard."""

    version: str = "1.0"
    block_on: str = "critical"
    blocked_commands: list[str] = field(default_factory=list)
    sensitive_paths: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    max_risk_per_tool: dict[str, str] = field(default_factory=dict)
    rate_limits: dict[str, int] = field(default_factory=dict)
    custom_blocks: list[CustomBlock] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardPolicy:
        """Create a policy from a dict (e.g., parsed YAML)."""
        custom_blocks = []
        for cb in data.get("custom_blocks", []):
            if isinstance(cb, dict):
                custom_blocks.append(CustomBlock(
                    pattern=cb.get("pattern", ""),
                    reason=cb.get("reason", "custom_block"),
                ))

        return cls(
            version=data.get("version", "1.0"),
            block_on=data.get("block_on", "critical"),
            blocked_commands=data.get("blocked_commands", []),
            sensitive_paths=data.get("sensitive_paths", []),
            allowed_tools=data.get("allowed_tools", []),
            denied_tools=data.get("denied_tools", []),
            max_risk_per_tool=data.get("max_risk_per_tool", {}),
            rate_limits=data.get("rate_limits", {}),
            custom_blocks=custom_blocks,
        )

    @classmethod
    def from_yaml(cls, path: str) -> GuardPolicy:
        """Load policy from a YAML file."""
        try:
            import yaml
        except ImportError:
            # Fallback: parse simple YAML-like format
            return cls._from_simple_yaml(path)

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def _from_simple_yaml(cls, path: str) -> GuardPolicy:
        """Minimal YAML parser for simple key-value and list structures."""
        import json

        with open(path) as f:
            content = f.read()

        # Try JSON first (YAML is a superset of JSON)
        try:
            data = json.loads(content)
            return cls.from_dict(data)
        except json.JSONDecodeError:
            pass

        # Simple line-by-line parsing
        data: dict[str, Any] = {}
        current_key = ""
        current_list: list[str] | None = None

        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("- "):
                if current_list is not None:
                    value = stripped[2:].strip().strip('"').strip("'")
                    current_list.append(value)
                continue

            if ":" in stripped:
                if current_list is not None:
                    current_list = None

                key, _, val = stripped.partition(":")
                key = key.strip()
                val = val.strip().strip('"').strip("'")

                if not val:
                    # Start of a list or dict
                    current_key = key
                    current_list = []
                    data[key] = current_list
                else:
                    data[key] = val
                    current_key = key

        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Export policy as a dict."""
        return {
            "version": self.version,
            "block_on": self.block_on,
            "blocked_commands": self.blocked_commands,
            "sensitive_paths": self.sensitive_paths,
            "allowed_tools": self.allowed_tools,
            "denied_tools": self.denied_tools,
            "max_risk_per_tool": self.max_risk_per_tool,
            "rate_limits": self.rate_limits,
            "custom_blocks": [
                {"pattern": cb.pattern, "reason": cb.reason}
                for cb in self.custom_blocks
            ],
        }

    def create_guard(
        self,
        session_id: str | None = None,
        user_id: str = "",
        agent_id: str = "",
        model: str = "",
        window_seconds: int = 300,
    ) -> SessionGuard:
        """Create a SessionGuard configured with this policy."""
        custom_rules = self._build_custom_rules()

        guard = SessionGuard(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            model=model,
            block_on=self.block_on,
            custom_rules=custom_rules,
            window_seconds=window_seconds,
        )

        return guard

    def _build_custom_rules(self) -> list:
        """Build custom rule functions from policy configuration."""
        rules = []

        # Tool allowlist/denylist rules
        if self.denied_tools:
            denied = set(self.denied_tools)

            def deny_tool_rule(tool_name: str, args: dict) -> str | None:
                if tool_name in denied:
                    return f"denied_tool: {tool_name}"
                return None

            rules.append(deny_tool_rule)

        if self.allowed_tools:
            allowed = set(self.allowed_tools)

            def allow_tool_rule(tool_name: str, args: dict) -> str | None:
                if tool_name not in allowed:
                    return f"tool_not_in_allowlist: {tool_name}"
                return None

            rules.append(allow_tool_rule)

        # Additional blocked command patterns
        if self.blocked_commands:
            patterns = self.blocked_commands

            def blocked_cmd_rule(tool_name: str, args: dict) -> str | None:
                text = " ".join(
                    str(v) for v in args.values() if isinstance(v, str)
                )
                for pat in patterns:
                    if pat.lower() in text.lower():
                        return f"policy_blocked_command: {pat}"
                return None

            rules.append(blocked_cmd_rule)

        # Custom block patterns
        for cb in self.custom_blocks:
            block = cb  # Capture in closure

            def custom_block_rule(tool_name: str, args: dict, _b=block) -> str | None:
                text = " ".join(
                    str(v) for v in args.values() if isinstance(v, str)
                )
                if _b.matches(text):
                    return _b.reason
                return None

            rules.append(custom_block_rule)

        # Per-tool risk limits
        if self.max_risk_per_tool:
            risk_order = ["none", "low", "medium", "high", "critical"]
            per_tool = self.max_risk_per_tool

            def per_tool_risk_rule(tool_name: str, args: dict) -> str | None:
                if tool_name in per_tool:
                    # This rule runs before risk is assessed, so we can't
                    # check risk here. Instead, we use this to deny tools
                    # that should have stricter thresholds.
                    # For now, just track which tools have custom thresholds.
                    pass
                return None

            # Note: per-tool risk is better handled at the guard level
            # For now, we don't add this as a custom rule

        # Rate limits
        if self.rate_limits:
            _call_counts: dict[str, int] = {}
            limits = self.rate_limits

            def rate_limit_rule(tool_name: str, args: dict) -> str | None:
                _call_counts[tool_name] = _call_counts.get(tool_name, 0) + 1
                limit = limits.get(tool_name, limits.get("default", 0))
                if limit > 0 and _call_counts[tool_name] > limit:
                    return f"rate_limit_exceeded: {tool_name} ({_call_counts[tool_name]}/{limit})"
                return None

            rules.append(rate_limit_rule)

        return rules

    def validate(self) -> list[str]:
        """Validate the policy and return a list of warnings/errors."""
        issues: list[str] = []

        valid_risks = {"none", "low", "medium", "high", "critical"}

        if self.block_on not in valid_risks:
            issues.append(f"Invalid block_on value: {self.block_on}")

        for tool, risk in self.max_risk_per_tool.items():
            if risk not in valid_risks:
                issues.append(f"Invalid risk level for tool {tool}: {risk}")

        if self.allowed_tools and self.denied_tools:
            overlap = set(self.allowed_tools) & set(self.denied_tools)
            if overlap:
                issues.append(f"Tools in both allowed and denied lists: {overlap}")

        for cb in self.custom_blocks:
            try:
                re.compile(cb.pattern)
            except re.error as e:
                issues.append(f"Invalid regex in custom_blocks: {cb.pattern} ({e})")

        for tool, limit in self.rate_limits.items():
            if not isinstance(limit, int) or limit < 0:
                issues.append(f"Invalid rate limit for {tool}: {limit}")

        return issues
