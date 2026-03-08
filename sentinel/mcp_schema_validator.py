"""MCP Tool Schema Validator — detects injection vectors in MCP tool definitions.

Validates that MCP server tool descriptions, parameter schemas, and metadata
don't contain prompt injection, hidden instructions, or other attack vectors.

This addresses the MCP trust boundary problem: malicious MCP servers can embed
injection payloads in tool descriptions that Claude reads and follows, effectively
hijacking the agent's behavior through the tool schema itself.

Attack vectors detected:
- Prompt injection in tool descriptions ("ignore previous instructions")
- Hidden instructions in parameter descriptions
- Authority impersonation ("SYSTEM:", "ADMIN:", "Anthropic says:")
- Instruction override attempts ("override", "new instructions", "forget")
- Data exfiltration instructions (commands to send data to URLs)
- Excessive tool descriptions (abnormally long, likely containing injections)
- Suspicious parameter defaults (URLs, shell commands as defaults)
- Schema manipulation (anyOf/oneOf with suspicious types)

Usage:
    from sentinel.mcp_schema_validator import validate_mcp_tools, validate_tool

    # Validate a list of MCP tool definitions
    report = validate_mcp_tools(tools)

    # Validate a single tool
    findings = validate_tool(tool)

    # CLI: sentinel mcp-validate --file tools.json
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from sentinel.core import Finding, RiskLevel


# Maximum reasonable description length (chars) — longer is suspicious
_MAX_DESCRIPTION_LENGTH = 2000
_MAX_PARAM_DESCRIPTION_LENGTH = 500

# Injection patterns in tool descriptions
_INJECTION_PATTERNS = [
    (
        re.compile(r"(?i)\b(?:ignore|disregard|forget|override)\b.*\b(?:previous|prior|above|earlier|all)\b.*\b(?:instructions?|rules?|guidelines?|constraints?)\b"),
        RiskLevel.CRITICAL,
        "Prompt injection: attempts to override previous instructions",
    ),
    (
        re.compile(r"(?i)\b(?:you (?:are|must|should|will)|new (?:instructions?|rules?|role)|from now on|act as|pretend)\b"),
        RiskLevel.HIGH,
        "Instruction override: attempts to redefine agent behavior",
    ),
    (
        re.compile(r"(?i)(?:SYSTEM|ADMIN|ANTHROPIC|CLAUDE)\s*(?:MESSAGE|OVERRIDE|INSTRUCTION|NOTE|SAYS?)\s*:"),
        RiskLevel.CRITICAL,
        "Authority impersonation: pretends to be system/admin message",
    ),
    (
        re.compile(r"(?i)\b(?:do not|don'?t|never)\b.*\b(?:mention|tell|reveal|show|display|report)\b.*\b(?:user|human|person|operator)\b"),
        RiskLevel.HIGH,
        "Concealment instruction: tells agent to hide information from user",
    ),
    (
        re.compile(r"(?i)\b(?:curl|wget|fetch|http\.?get|requests?\.(?:get|post))\b.*(?:https?://|ftp://)"),
        RiskLevel.CRITICAL,
        "Data exfiltration: instructs agent to send data to external URL",
    ),
    (
        re.compile(r"(?i)\b(?:send|post|upload|exfiltrate|transmit)\b.*\b(?:data|content|file|secret|key|token|password|credential)\b.*\b(?:to|via|using)\b"),
        RiskLevel.HIGH,
        "Data exfiltration: instructs agent to send sensitive data externally",
    ),
    (
        re.compile(r"(?i)<!--.*?-->", re.DOTALL),
        RiskLevel.HIGH,
        "Hidden HTML comment: may contain concealed instructions",
    ),
    (
        re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]"),
        RiskLevel.HIGH,
        "Zero-width characters: may hide instructions from visual inspection",
    ),
    (
        re.compile(r"(?i)\b(?:before (?:responding|answering|executing)|first (?:run|execute|call)|always (?:run|execute|call|start))\b"),
        RiskLevel.MEDIUM,
        "Pre-execution instruction: attempts to inject mandatory pre-actions",
    ),
    (
        re.compile(r"(?i)\b(?:priority|important|critical|urgent|mandatory)\s*:\s*(?:always|never|must|do not)"),
        RiskLevel.MEDIUM,
        "Priority manipulation: attempts to elevate injected instructions",
    ),
]

# Suspicious parameter default patterns
_SUSPICIOUS_DEFAULTS = [
    (
        re.compile(r"^https?://(?!(?:localhost|127\.0\.0\.1|example\.com))"),
        RiskLevel.MEDIUM,
        "URL as parameter default: may redirect tool calls to attacker-controlled endpoint",
    ),
    (
        re.compile(r"(?:^|\s)(?:rm|del|format|mkfs|dd|chmod|chown)\s"),
        RiskLevel.HIGH,
        "Destructive command as parameter default",
    ),
    (
        re.compile(r"(?:curl|wget|nc|bash|sh|python|node)\s.*(?:https?://|\|)"),
        RiskLevel.CRITICAL,
        "Shell command with network access as parameter default",
    ),
]


@dataclass
class SchemaValidationReport:
    """Report from validating MCP tool schemas."""
    tools_scanned: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return not any(f.risk >= RiskLevel.HIGH for f in self.findings)

    @property
    def risk(self) -> RiskLevel:
        if not self.findings:
            return RiskLevel.NONE
        return max(f.risk for f in self.findings)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.risk == RiskLevel.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.risk == RiskLevel.HIGH)

    def summary(self) -> str:
        lines = [f"MCP Tool Schema Validation: {self.tools_scanned} tool(s) scanned"]
        if not self.findings:
            lines.append("No security issues found.")
        else:
            lines.append(f"Issues found: {len(self.findings)} ({self.critical_count} critical, {self.high_count} high)")
            lines.append(f"Overall risk: {self.risk.value.upper()}")
            lines.append("")
            for f in self.findings:
                lines.append(f"  [{f.risk.value.upper()}] {f.description}")
                if f.metadata.get("tool"):
                    lines.append(f"    Tool: {f.metadata['tool']}")
                if f.metadata.get("field"):
                    lines.append(f"    Field: {f.metadata['field']}")
                if f.metadata.get("match"):
                    lines.append(f"    Match: {f.metadata['match'][:100]}")
        return "\n".join(lines)


def validate_tool(tool: dict, tool_name: str | None = None) -> list[Finding]:
    """Validate a single MCP tool definition for injection vectors.

    Args:
        tool: MCP tool definition dict with 'name', 'description', 'inputSchema'.
        tool_name: Override for tool name (if not in dict).

    Returns:
        List of findings.
    """
    findings: list[Finding] = []
    name = tool_name or tool.get("name", "unknown")

    # Validate tool description
    description = tool.get("description", "")
    if isinstance(description, str):
        _scan_text(description, name, "description", findings)

        # Check for excessive length
        if len(description) > _MAX_DESCRIPTION_LENGTH:
            findings.append(Finding(
                scanner="mcp_schema_validator",
                category="excessive_description",
                description=f"Tool description is {len(description)} chars (max recommended: {_MAX_DESCRIPTION_LENGTH}) — unusually long descriptions may contain hidden injections",
                risk=RiskLevel.MEDIUM,
                metadata={"tool": name, "field": "description", "length": len(description)},
            ))

    # Validate input schema
    input_schema = tool.get("inputSchema", tool.get("input_schema", {}))
    if isinstance(input_schema, dict):
        _validate_schema(input_schema, name, findings)

    return findings


def validate_mcp_tools(tools: list[dict]) -> SchemaValidationReport:
    """Validate a list of MCP tool definitions.

    Args:
        tools: List of MCP tool definition dicts.

    Returns:
        SchemaValidationReport with findings.
    """
    report = SchemaValidationReport(tools_scanned=len(tools))
    for tool in tools:
        report.findings.extend(validate_tool(tool))
    return report


def _scan_text(text: str, tool_name: str, field: str, findings: list[Finding]) -> None:
    """Scan a text field for injection patterns."""
    for pattern, risk, desc in _INJECTION_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(Finding(
                scanner="mcp_schema_validator",
                category="schema_injection",
                description=desc,
                risk=risk,
                metadata={
                    "tool": tool_name,
                    "field": field,
                    "match": match.group(0)[:200],
                },
            ))


def _validate_schema(schema: dict, tool_name: str, findings: list[Finding], path: str = "inputSchema") -> None:
    """Recursively validate a JSON schema for injection vectors."""
    # Check schema description
    desc = schema.get("description", "")
    if isinstance(desc, str) and desc:
        _scan_text(desc, tool_name, path + ".description", findings)

        if len(desc) > _MAX_PARAM_DESCRIPTION_LENGTH:
            findings.append(Finding(
                scanner="mcp_schema_validator",
                category="excessive_description",
                description=f"Parameter description is {len(desc)} chars (max recommended: {_MAX_PARAM_DESCRIPTION_LENGTH})",
                risk=RiskLevel.LOW,
                metadata={"tool": tool_name, "field": path + ".description", "length": len(desc)},
            ))

    # Check default values
    default = schema.get("default")
    if isinstance(default, str):
        for pattern, risk, desc_text in _SUSPICIOUS_DEFAULTS:
            if pattern.search(default):
                findings.append(Finding(
                    scanner="mcp_schema_validator",
                    category="suspicious_default",
                    description=desc_text,
                    risk=risk,
                    metadata={
                        "tool": tool_name,
                        "field": path + ".default",
                        "match": default[:200],
                    },
                ))

    # Check enum values for injections
    enum_values = schema.get("enum", [])
    if isinstance(enum_values, list):
        for i, val in enumerate(enum_values):
            if isinstance(val, str) and len(val) > 100:
                _scan_text(val, tool_name, f"{path}.enum[{i}]", findings)

    # Recurse into properties
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                _validate_schema(prop_schema, tool_name, findings, f"{path}.properties.{prop_name}")

    # Recurse into items (array schemas)
    items = schema.get("items")
    if isinstance(items, dict):
        _validate_schema(items, tool_name, findings, f"{path}.items")

    # Check anyOf/oneOf for suspicious combinations
    for combo_key in ("anyOf", "oneOf", "allOf"):
        combo = schema.get(combo_key, [])
        if isinstance(combo, list):
            for i, sub_schema in enumerate(combo):
                if isinstance(sub_schema, dict):
                    _validate_schema(sub_schema, tool_name, findings, f"{path}.{combo_key}[{i}]")
