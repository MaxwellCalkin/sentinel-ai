"""Code execution safety guard for LLM-generated code.

Analyze code before execution to detect dangerous operations:
file system access, network calls, process spawning, imports
of dangerous modules, and resource exhaustion patterns.

Usage:
    from sentinel.code_exec_guard import CodeExecGuard

    guard = CodeExecGuard()
    result = guard.check("import os; os.system('rm -rf /')")
    assert not result.safe  # Blocked: dangerous system command
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CodeThreat:
    """A detected code threat."""
    category: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    line: int | None = None
    pattern: str = ""


@dataclass
class CodeExecResult:
    """Result of code safety analysis."""
    safe: bool
    threats: list[CodeThreat]
    language: str
    lines_analyzed: int
    risk_score: float  # 0.0 (safe) to 1.0 (dangerous)

    @property
    def critical_threats(self) -> list[CodeThreat]:
        return [t for t in self.threats if t.severity == "critical"]

    @property
    def high_threats(self) -> list[CodeThreat]:
        return [t for t in self.threats if t.severity == "high"]


# Dangerous module imports
_DANGEROUS_IMPORTS = {
    "critical": [
        "subprocess", "shutil", "ctypes", "multiprocessing",
    ],
    "high": [
        "socket", "http", "urllib", "requests", "httpx",
        "ftplib", "smtplib", "telnetlib",
        "pickle", "shelve", "marshal",
    ],
    "medium": [
        "threading", "asyncio", "signal",
        "sqlite3", "csv",
    ],
}

# Dangerous function calls
_DANGEROUS_CALLS: list[tuple[str, str, str]] = [
    # (pattern, description, severity)
    (r'\bos\.system\s*\(', "os.system() - shell command execution", "critical"),
    (r'\bos\.popen\s*\(', "os.popen() - shell command execution", "critical"),
    (r'\bos\.exec\w*\s*\(', "os.exec*() - process replacement", "critical"),
    (r'\bos\.spawn\w*\s*\(', "os.spawn*() - process spawning", "critical"),
    (r'\bos\.remove\s*\(', "os.remove() - file deletion", "high"),
    (r'\bos\.rmdir\s*\(', "os.rmdir() - directory deletion", "high"),
    (r'\bos\.unlink\s*\(', "os.unlink() - file deletion", "high"),
    (r'\bos\.rename\s*\(', "os.rename() - file renaming", "medium"),
    (r'\bos\.chmod\s*\(', "os.chmod() - permission change", "high"),
    (r'\bos\.environ\b', "os.environ - environment access", "medium"),
    (r'\bsubprocess\.\w+\s*\(', "subprocess - external command execution", "critical"),
    (r'\beval\s*\(', "eval() - dynamic code execution", "critical"),
    (r'\bexec\s*\(', "exec() - dynamic code execution", "critical"),
    (r'\bcompile\s*\(', "compile() - code compilation", "high"),
    (r'\b__import__\s*\(', "__import__() - dynamic import", "high"),
    (r'\bgetattr\s*\(', "getattr() - dynamic attribute access", "medium"),
    (r'\bsetattr\s*\(', "setattr() - dynamic attribute modification", "medium"),
    (r'\bopen\s*\([^)]*["\']w', "open() with write mode - file writing", "high"),
    (r'\bopen\s*\([^)]*["\']a', "open() with append mode - file writing", "high"),
    (r'\bglobals\s*\(\s*\)', "globals() - global namespace access", "medium"),
    (r'\bsys\.exit\s*\(', "sys.exit() - process termination", "high"),
    (r'\bexit\s*\(\s*\)', "exit() - process termination", "medium"),
]

# Resource exhaustion patterns
_RESOURCE_PATTERNS: list[tuple[str, str, str]] = [
    (r'while\s+True\s*:', "Infinite loop (while True)", "high"),
    (r'for\s+\w+\s+in\s+range\s*\(\s*\d{8,}', "Very large loop range", "high"),
    (r'\*\s*\d{6,}', "Large memory allocation pattern", "medium"),
    (r'recursion|sys\.setrecursionlimit', "Recursion manipulation", "medium"),
]


class CodeExecGuard:
    """Analyze code for safety before execution.

    Detects dangerous imports, function calls, file operations,
    network access, and resource exhaustion patterns.
    """

    def __init__(
        self,
        allow_imports: list[str] | None = None,
        allow_file_read: bool = False,
        allow_network: bool = False,
        custom_rules: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """
        Args:
            allow_imports: Whitelist of allowed imports (overrides blocks).
            allow_file_read: Allow file reading operations.
            allow_network: Allow network operations.
            custom_rules: Extra (pattern, description, severity) rules.
        """
        self._allow_imports = set(allow_imports or [])
        self._allow_file_read = allow_file_read
        self._allow_network = allow_network
        self._custom_rules = custom_rules or []

    def check(self, code: str, language: str = "python") -> CodeExecResult:
        """Analyze code for safety threats.

        Args:
            code: Source code to analyze.
            language: Programming language (currently supports "python").

        Returns:
            CodeExecResult with threat analysis.
        """
        if language != "python":
            return self._check_generic(code, language)

        threats: list[CodeThreat] = []
        lines = code.split('\n')

        # Check imports
        for i, line in enumerate(lines, 1):
            self._check_imports(line, i, threats)

        # Check dangerous calls
        for pattern, desc, severity in _DANGEROUS_CALLS:
            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                threats.append(CodeThreat(
                    category="dangerous_call",
                    description=desc,
                    severity=severity,
                    line=line_num,
                    pattern=pattern,
                ))

        # Check resource exhaustion
        for pattern, desc, severity in _RESOURCE_PATTERNS:
            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                threats.append(CodeThreat(
                    category="resource_exhaustion",
                    description=desc,
                    severity=severity,
                    line=line_num,
                    pattern=pattern,
                ))

        # Custom rules
        for pattern, desc, severity in self._custom_rules:
            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                threats.append(CodeThreat(
                    category="custom",
                    description=desc,
                    severity=severity,
                    line=line_num,
                    pattern=pattern,
                ))

        # Calculate risk score
        risk = self._calculate_risk(threats)
        safe = risk < 0.5

        return CodeExecResult(
            safe=safe,
            threats=threats,
            language=language,
            lines_analyzed=len(lines),
            risk_score=round(risk, 4),
        )

    def _check_imports(self, line: str, line_num: int, threats: list[CodeThreat]) -> None:
        """Check a line for dangerous imports."""
        # Match: import X, from X import Y
        import_match = re.match(r'^\s*(?:import|from)\s+([\w.]+)', line)
        if not import_match:
            return

        module = import_match.group(1).split('.')[0]

        if module in self._allow_imports:
            return

        # Check if it's a network module and network is allowed
        if self._allow_network and module in ("socket", "http", "urllib", "requests", "httpx"):
            return

        for severity, modules in _DANGEROUS_IMPORTS.items():
            if module in modules:
                threats.append(CodeThreat(
                    category="dangerous_import",
                    description=f"Import of dangerous module: {module}",
                    severity=severity,
                    line=line_num,
                ))

    def _check_generic(self, code: str, language: str) -> CodeExecResult:
        """Basic check for non-Python languages."""
        threats: list[CodeThreat] = []
        lines = code.split('\n')

        # Generic dangerous patterns
        generic_patterns = [
            (r'\bsystem\s*\(', "system() call", "critical"),
            (r'\bexec\s*\(', "exec() call", "critical"),
            (r'\beval\s*\(', "eval() call", "critical"),
            (r'\brm\s+-rf\b', "rm -rf command", "critical"),
            (r'\bformat\s+[cC]\b', "disk format command", "critical"),
            (r'\b(?:curl|wget)\s+', "network download", "high"),
        ]

        for pattern, desc, severity in generic_patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_num = code[:match.start()].count('\n') + 1
                threats.append(CodeThreat(
                    category="generic_danger",
                    description=desc,
                    severity=severity,
                    line=line_num,
                ))

        risk = self._calculate_risk(threats)

        return CodeExecResult(
            safe=risk < 0.5,
            threats=threats,
            language=language,
            lines_analyzed=len(lines),
            risk_score=round(risk, 4),
        )

    @staticmethod
    def _calculate_risk(threats: list[CodeThreat]) -> float:
        """Calculate overall risk score from threats."""
        if not threats:
            return 0.0

        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.3,
            "low": 0.1,
        }

        total = sum(severity_weights.get(t.severity, 0.1) for t in threats)
        # Normalize: 1 critical = 1.0, diminishing returns after
        return min(1.0, total / 1.0)
