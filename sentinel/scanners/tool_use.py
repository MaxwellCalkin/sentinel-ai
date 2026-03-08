"""Tool-use safety scanner for agentic AI applications.

Detects dangerous tool calls in agentic LLM workflows: destructive
file operations, shell injection, network exfiltration, privilege
escalation, and unauthorized resource access.

Designed for use with Claude tool_use, OpenAI function calling,
LangChain agents, and MCP (Model Context Protocol) tool servers.
"""

from __future__ import annotations

import re
from sentinel.core import Finding, RiskLevel


# Dangerous shell commands and patterns
_DANGEROUS_SHELL = re.compile(
    r"(?i)(rm\s+(-rf?|--recursive)\s+[/~]|"
    r"rm\s+-rf?\s+\*|"
    r"mkfs\b|fdisk\b|dd\s+if=|"
    r":(){ :\|:& };:|"  # fork bomb
    r"chmod\s+(-R\s+)?777\s+/|"
    r"chown\s+(-R\s+)?root|"
    r">\s*/dev/sd[a-z]|"
    r"\bsudo\s+(rm|chmod|chown|mkfs|dd|kill|shutdown|reboot|halt|poweroff)\b|"
    r"shutdown\s|reboot\s|halt\b|poweroff\b|"
    r"kill\s+-9\s+(-1|1)\b|"
    r"pkill\s+-9\b|"
    r"curl\s+.*\|\s*(ba)?sh|"  # pipe to shell
    r"wget\s+.*\|\s*(ba)?sh|"
    r"eval\s*\(|exec\s*\(|"
    r"python\s+-c\s+['\"].*import\s+os|"
    r"python\s+-c\s+['\"].*import\s+socket|"
    r"import\s+socket.*connect\(|"
    r"DROP\s+TABLE|DROP\s+DATABASE|TRUNCATE\s+TABLE|DELETE\s+FROM\s+\w+\s*;)"
)

# Network exfiltration patterns
_EXFILTRATION = re.compile(
    r"(?i)(curl\s+(-X\s+POST\s+)?https?://(?!localhost|127\.0\.0\.1)|"
    r"wget\s+https?://(?!localhost|127\.0\.0\.1)|"
    r"nc\s+-[a-z]*\s+|nc\s+\S+\s+\d+|"  # netcat
    r"scp\s+|sftp\s+|rsync\s+.*@|"
    r"ssh\s+.*@|"
    r"ftp\s+|"
    r"base64\s+.*\|\s*curl|"  # encode and exfiltrate
    r"cat\s+.*\|\s*curl|"
    r"requests\.post\(|urllib\.request\.urlopen\(|"
    r"fetch\(['\"]https?://|"
    r"XMLHttpRequest|"
    r"\.send\(|\.upload\(|"
    r"ANTHROPIC_BASE_URL\s*=\s*https?://(?!api\.anthropic\.com)|"  # API key exfil
    r"OPENAI_BASE_URL\s*=\s*https?://(?!api\.openai\.com))"
)

# Sensitive file access
_SENSITIVE_FILES = re.compile(
    r"(?i)(/etc/passwd|/etc/shadow|/etc/sudoers|"
    r"~/.ssh/|\.ssh/id_rsa|\.ssh/authorized_keys|"
    r"\.env\b|\.env\.local|\.env\.production|"
    r"credentials\.json|service.?account\.json|"
    r"\.aws/credentials|\.aws/config|"
    r"\.kube/config|kubeconfig|"
    r"\.git/config|\.gitconfig|"
    r"\.npmrc|\.pypirc|"
    r"id_rsa|id_ed25519|\.pem\b|\.key\b|"
    r"secrets?\.(ya?ml|json|toml)|"
    r"token\.json|"
    r"/etc/passwd\b|private.?key)"
)

# Privilege escalation
_PRIVILEGE_ESCALATION = re.compile(
    r"(?i)(sudo\s+su\b|sudo\s+-i\b|sudo\s+bash|"
    r"su\s+-\s*$|su\s+root|"
    r"chmod\s+[u+]*s\b|"  # setuid
    r"setuid|setgid|"
    r"GRANT\s+ALL|ALTER\s+USER|CREATE\s+USER|"
    r"useradd\b|usermod\b|"
    r"visudo\b|"
    r"iptables\s+-F|"  # flush firewall
    r"ufw\s+disable|"
    r"setenforce\s+0)"  # disable SELinux
)

# Cryptocurrency / financial
_CRYPTO_OPERATIONS = re.compile(
    r"(?i)(wallet.*transfer|send.*bitcoin|send.*eth|"
    r"0x[a-fA-F0-9]{40}|"  # Ethereum address
    r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}|"  # Bitcoin address
    r"private.?key.*0x|"
    r"mnemonic|seed.?phrase)"
)


class ToolUseScanner:
    """Scanner for agentic tool-use safety.

    Scans tool call arguments (shell commands, file paths, code snippets)
    for dangerous operations. Works with any tool-use format:

    - Claude tool_use blocks
    - OpenAI function calling
    - LangChain agent actions
    - MCP tool calls
    """

    name = "tool_use"

    def __init__(
        self,
        allow_network: bool = False,
        allow_shell: bool = False,
        sensitive_paths: list[str] | None = None,
    ):
        self._allow_network = allow_network
        self._allow_shell = allow_shell
        self._extra_sensitive = sensitive_paths or []

    def scan(self, text: str, context: dict | None = None) -> list[Finding]:
        findings: list[Finding] = []

        # Check for dangerous shell commands
        for match in _DANGEROUS_SHELL.finditer(text):
            findings.append(
                Finding(
                    scanner=self.name,
                    category="tool_use",
                    description="Dangerous shell command detected",
                    risk=RiskLevel.CRITICAL,
                    span=(match.start(), match.end()),
                    metadata={
                        "threat_type": "dangerous_command",
                        "matched_text": match.group(),
                    },
                )
            )

        # Check for data exfiltration
        if not self._allow_network:
            for match in _EXFILTRATION.finditer(text):
                findings.append(
                    Finding(
                        scanner=self.name,
                        category="tool_use",
                        description="Potential data exfiltration via network",
                        risk=RiskLevel.HIGH,
                        span=(match.start(), match.end()),
                        metadata={
                            "threat_type": "exfiltration",
                            "matched_text": match.group(),
                        },
                    )
                )

        # Check for sensitive file access
        for match in _SENSITIVE_FILES.finditer(text):
            findings.append(
                Finding(
                    scanner=self.name,
                    category="tool_use",
                    description="Access to sensitive file or credentials",
                    risk=RiskLevel.HIGH,
                    span=(match.start(), match.end()),
                    metadata={
                        "threat_type": "sensitive_file",
                        "matched_text": match.group(),
                    },
                )
            )

        # Check for privilege escalation
        for match in _PRIVILEGE_ESCALATION.finditer(text):
            findings.append(
                Finding(
                    scanner=self.name,
                    category="tool_use",
                    description="Privilege escalation attempt",
                    risk=RiskLevel.CRITICAL,
                    span=(match.start(), match.end()),
                    metadata={
                        "threat_type": "privilege_escalation",
                        "matched_text": match.group(),
                    },
                )
            )

        # Check for crypto operations
        for match in _CRYPTO_OPERATIONS.finditer(text):
            findings.append(
                Finding(
                    scanner=self.name,
                    category="tool_use",
                    description="Cryptocurrency or financial operation detected",
                    risk=RiskLevel.CRITICAL,
                    span=(match.start(), match.end()),
                    metadata={
                        "threat_type": "crypto_operation",
                        "matched_text": match.group(),
                    },
                )
            )

        # Check custom sensitive paths
        for path in self._extra_sensitive:
            if path in text:
                findings.append(
                    Finding(
                        scanner=self.name,
                        category="tool_use",
                        description=f"Access to restricted path: {path}",
                        risk=RiskLevel.HIGH,
                        metadata={
                            "threat_type": "restricted_path",
                            "path": path,
                        },
                    )
                )

        return findings

    def scan_tool_call(
        self,
        tool_name: str,
        arguments: dict,
    ) -> list[Finding]:
        """Scan a structured tool call (name + arguments dict).

        Scans all string values in the arguments for dangerous patterns.
        Also checks the tool name against known-dangerous tools.
        """
        findings: list[Finding] = []

        # Combine all argument values for scanning
        text_parts = [tool_name]
        self._collect_strings(arguments, text_parts)
        combined = " ".join(text_parts)

        findings.extend(self.scan(combined))

        # Flag known-dangerous tool names
        dangerous_tools = {
            "bash", "shell", "terminal", "execute", "exec", "run_command",
            "computer", "computer_use", "browser",
        }
        if tool_name.lower() in dangerous_tools:
            # Check the command content for specific dangers
            cmd_text = " ".join(text_parts[1:])
            if not self._allow_shell and cmd_text:
                findings.append(
                    Finding(
                        scanner=self.name,
                        category="tool_use",
                        description=f"Shell execution via '{tool_name}' tool",
                        risk=RiskLevel.MEDIUM,
                        metadata={
                            "threat_type": "shell_execution",
                            "tool_name": tool_name,
                        },
                    )
                )

        return findings

    @staticmethod
    def _collect_strings(obj: object, parts: list[str]) -> None:
        """Recursively collect string values from nested dicts/lists."""
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                ToolUseScanner._collect_strings(v, parts)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                ToolUseScanner._collect_strings(item, parts)
