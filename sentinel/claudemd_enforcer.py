"""CLAUDE.md Rule Enforcer — extract enforceable rules from CLAUDE.md and check tool calls.

Parses CLAUDE.md files for rules like "never use rm -rf", "do not modify tests/",
"always use --dry-run", etc. and converts them into deterministic checks that can
be applied as Claude Code PreToolUse hooks.

Usage:
    from sentinel.claudemd_enforcer import ClaudeMdEnforcer

    enforcer = ClaudeMdEnforcer.from_file("CLAUDE.md")
    verdict = enforcer.check("bash", {"command": "rm -rf /"})
    # verdict.allowed == False
    # verdict.violated_rules == ["Never use rm -rf"]

    # Or generate a GuardPolicy from CLAUDE.md rules
    policy = enforcer.to_guard_policy()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EnforcedRule:
    """A rule extracted from CLAUDE.md that can be enforced programmatically."""

    text: str  # Original rule text from CLAUDE.md
    rule_type: str  # "blocked_command", "blocked_path", "blocked_tool", "required_flag", "custom_pattern"
    pattern: str  # Regex or substring to match
    severity: str = "high"  # none, low, medium, high, critical
    line_number: int | None = None
    _compiled: re.Pattern | None = field(default=None, repr=False)

    def matches(self, text: str) -> bool:
        if self._compiled is None:
            try:
                self._compiled = re.compile(self.pattern, re.IGNORECASE)
            except re.error:
                self._compiled = re.compile(re.escape(self.pattern), re.IGNORECASE)
        return bool(self._compiled.search(text))


@dataclass
class EnforcementVerdict:
    """Result of checking a tool call against CLAUDE.md rules."""

    allowed: bool
    tool_name: str
    violated_rules: list[str]
    warnings: list[str]
    matched_rules: list[EnforcedRule]

    @property
    def safe(self) -> bool:
        return self.allowed and len(self.warnings) == 0


# Patterns that indicate a blocking/prohibition rule
_PROHIBITION_PATTERNS = [
    # "never X", "do not X", "don't X", "must not X", "should not X"
    re.compile(
        r"(?:^|\n)[^\n]*?(?:never|do\s+not|don't|must\s+not|should\s+not|"
        r"shall\s+not|cannot|can't|forbidden|prohibited|disallowed|"
        r"avoid|refrain\s+from)\s+(.+?)(?:(?<![`\w/._-])\.|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "X is not allowed", "X is prohibited", "X is forbidden"
    re.compile(
        r"(?:^|\n)\s*[-*]?\s*(.+?)\s+(?:is|are)\s+(?:not\s+allowed|prohibited|forbidden|banned|blocked)",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "NO X" (all caps)
    re.compile(
        r"(?:^|\n)\s*[-*]?\s*NO\s+(.+?)(?:\.|$)",
        re.MULTILINE,
    ),
]

# Command patterns to extract from prohibition text
_COMMAND_EXTRACTORS = [
    # "never use rm -rf" → "rm -rf"
    (re.compile(r"use\s+[`\"']?([a-z][\w\s-]+(?:--[\w-]+)?)[`\"']?", re.IGNORECASE), "blocked_command"),
    # "never run git push --force" → "git push --force"
    (re.compile(r"(?:run|execute|call)\s+[`\"']?([a-z][\w\s.-]+(?:--[\w-]+)?)[`\"']?", re.IGNORECASE), "blocked_command"),
    # "never modify tests/" or "do not modify files in tests/" → "tests/"
    (re.compile(r"(?:modify|edit|change|delete|remove|touch|write\s+to)\s+(?:files?\s+(?:in|under|at)\s+)?[`\"']?([/\w._-]+/?)[`\"']?", re.IGNORECASE), "blocked_path"),
    # "never access .env" → ".env"
    (re.compile(r"(?:access|read|open|cat)\s+[`\"']?([/\w._-]+/?)[`\"']?", re.IGNORECASE), "blocked_path"),
    # Backtick-quoted commands: "never use `rm -rf`"
    (re.compile(r"`([^`]+)`"), "blocked_command"),
]

# Path patterns
_PATH_PATTERN = re.compile(r"[/\w._-]*(?:/[/\w._-]+)+|\.(?:env|ssh|aws|kube|npmrc|pypirc)\b")

# Tool name patterns
_TOOL_NAMES = {"bash", "read_file", "write_file", "edit", "curl", "wget", "scp", "ssh", "npm", "pip", "git"}


class ClaudeMdEnforcer:
    """Extract and enforce rules from CLAUDE.md files."""

    def __init__(self, rules: list[EnforcedRule] | None = None):
        self.rules: list[EnforcedRule] = rules or []

    @classmethod
    def from_file(cls, path: str | Path) -> ClaudeMdEnforcer:
        """Parse a CLAUDE.md file and extract enforceable rules."""
        content = Path(path).read_text(encoding="utf-8")
        return cls.from_text(content)

    @classmethod
    def from_text(cls, content: str) -> ClaudeMdEnforcer:
        """Parse CLAUDE.md content and extract enforceable rules."""
        rules: list[EnforcedRule] = []
        lines = content.split("\n")

        for prohibition_pat in _PROHIBITION_PATTERNS:
            for match in prohibition_pat.finditer(content):
                rule_text = match.group(0).strip().lstrip("-* ")
                body = match.group(1).strip()

                # Figure out line number — find the actual rule text start, not the \n before it
                match_text = match.group(0)
                actual_start = match.start()
                if match_text.startswith("\n"):
                    actual_start += 1
                line_num = content[:actual_start].count("\n") + 1

                # Try to extract specific commands/paths
                extracted = False
                for extractor, rule_type in _COMMAND_EXTRACTORS:
                    ext_match = extractor.search(body)
                    if ext_match:
                        pattern = ext_match.group(1).strip().strip("`\"'")
                        if len(pattern) >= 2:
                            rules.append(EnforcedRule(
                                text=rule_text,
                                rule_type=rule_type,
                                pattern=pattern,
                                severity="high",
                                line_number=line_num,
                            ))
                            extracted = True
                            break

                if not extracted:
                    # Try to find paths
                    path_match = _PATH_PATTERN.search(body)
                    if path_match:
                        rules.append(EnforcedRule(
                            text=rule_text,
                            rule_type="blocked_path",
                            pattern=path_match.group(0),
                            severity="high",
                            line_number=line_num,
                        ))
                        extracted = True

                if not extracted:
                    # Check if it mentions a tool name
                    body_lower = body.lower()
                    for tool in _TOOL_NAMES:
                        if tool in body_lower:
                            rules.append(EnforcedRule(
                                text=rule_text,
                                rule_type="blocked_tool",
                                pattern=tool,
                                severity="high",
                                line_number=line_num,
                            ))
                            extracted = True
                            break

                if not extracted and len(body) >= 5:
                    # Generic rule — use the body as a custom pattern
                    # Clean it up for use as a pattern
                    clean = re.sub(r"[^\w\s/._-]", "", body).strip()
                    if len(clean) >= 5:
                        rules.append(EnforcedRule(
                            text=rule_text,
                            rule_type="custom_pattern",
                            pattern=re.escape(clean),
                            severity="medium",
                            line_number=line_num,
                        ))

        # Also extract explicit blocked command lists
        # Patterns like: "Blocked commands: rm -rf, mkfs, dd if=/dev/"
        block_list_pat = re.compile(
            r"(?:blocked|forbidden|prohibited|banned)\s+(?:commands?|tools?|operations?)\s*:\s*(.+?)(?:\n|$)",
            re.IGNORECASE,
        )
        for match in block_list_pat.finditer(content):
            items = re.split(r"[,;]", match.group(1))
            line_num = content[:match.start()].count("\n") + 1
            for item in items:
                item = item.strip().strip("`\"'")
                if len(item) >= 2:
                    rules.append(EnforcedRule(
                        text=f"Blocked: {item}",
                        rule_type="blocked_command",
                        pattern=item,
                        severity="high",
                        line_number=line_num,
                    ))

        # Deduplicate by pattern
        seen = set()
        unique_rules = []
        for r in rules:
            key = (r.rule_type, r.pattern.lower())
            if key not in seen:
                seen.add(key)
                unique_rules.append(r)

        return cls(unique_rules)

    def check(self, tool_name: str, arguments: dict[str, Any]) -> EnforcementVerdict:
        """Check a tool call against extracted rules."""
        violated: list[str] = []
        warnings: list[str] = []
        matched: list[EnforcedRule] = []

        # Build searchable text from arguments
        text_parts: list[str] = [tool_name]
        for key in ("command", "cmd", "path", "file_path", "content", "new_string", "url"):
            val = arguments.get(key)
            if isinstance(val, str):
                text_parts.append(val)
        # Fallback: all string values
        if len(text_parts) == 1:
            for v in arguments.values():
                if isinstance(v, str):
                    text_parts.append(v)

        text = " ".join(text_parts)

        for rule in self.rules:
            if rule.rule_type == "blocked_tool":
                if rule.pattern.lower() == tool_name.lower():
                    violated.append(rule.text)
                    matched.append(rule)
                    continue

            if rule.rule_type == "blocked_command":
                if rule.matches(text):
                    violated.append(rule.text)
                    matched.append(rule)
                    continue

            if rule.rule_type == "blocked_path":
                if rule.matches(text):
                    violated.append(rule.text)
                    matched.append(rule)
                    continue

            if rule.rule_type == "custom_pattern":
                if rule.matches(text):
                    warnings.append(f"Possible violation: {rule.text}")
                    matched.append(rule)
                    continue

        return EnforcementVerdict(
            allowed=len(violated) == 0,
            tool_name=tool_name,
            violated_rules=violated,
            warnings=warnings,
            matched_rules=matched,
        )

    def to_guard_policy_dict(self) -> dict[str, Any]:
        """Convert extracted rules into a GuardPolicy-compatible dict."""
        blocked_commands: list[str] = []
        sensitive_paths: list[str] = []
        denied_tools: list[str] = []
        custom_blocks: list[dict[str, str]] = []

        for rule in self.rules:
            if rule.rule_type == "blocked_command":
                blocked_commands.append(rule.pattern)
            elif rule.rule_type == "blocked_path":
                sensitive_paths.append(rule.pattern)
            elif rule.rule_type == "blocked_tool":
                denied_tools.append(rule.pattern)
            elif rule.rule_type == "custom_pattern":
                custom_blocks.append({
                    "pattern": rule.pattern,
                    "reason": f"claudemd_rule: {rule.text[:80]}",
                })

        return {
            "block_on": "high",
            "blocked_commands": blocked_commands,
            "sensitive_paths": sensitive_paths,
            "denied_tools": denied_tools,
            "custom_blocks": custom_blocks,
        }

    def to_guard_policy(self):
        """Convert to a GuardPolicy object."""
        from sentinel.guard_policy import GuardPolicy
        return GuardPolicy.from_dict(self.to_guard_policy_dict())

    def summary(self) -> str:
        """Return a human-readable summary of extracted rules."""
        if not self.rules:
            return "No enforceable rules found in CLAUDE.md"

        lines = [f"Extracted {len(self.rules)} enforceable rule(s) from CLAUDE.md:\n"]
        by_type: dict[str, list[EnforcedRule]] = {}
        for r in self.rules:
            by_type.setdefault(r.rule_type, []).append(r)

        type_labels = {
            "blocked_command": "Blocked Commands",
            "blocked_path": "Blocked Paths",
            "blocked_tool": "Blocked Tools",
            "required_flag": "Required Flags",
            "custom_pattern": "Custom Rules",
        }

        for rtype, label in type_labels.items():
            rules = by_type.get(rtype, [])
            if rules:
                lines.append(f"  {label} ({len(rules)}):")
                for r in rules:
                    loc = f"L{r.line_number}" if r.line_number else ""
                    lines.append(f"    {loc} {r.pattern}  — {r.text[:60]}")

        return "\n".join(lines)
