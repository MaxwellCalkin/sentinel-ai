"""Code vulnerability scanner for LLM-generated code.

Detects OWASP Top 10 vulnerabilities in generated Python, JavaScript/TypeScript,
and shell code. Designed as a PostToolUse hook for Claude Code Write/Edit tools.

Scans for:
- SQL injection (string concatenation in queries)
- Command injection (unsanitized subprocess/exec calls)
- XSS (unescaped user input in HTML templates)
- Path traversal (unsanitized file paths)
- Insecure deserialization (pickle/yaml.load/eval)
- Hardcoded secrets (API keys, passwords, tokens in source)
- Insecure crypto (MD5/SHA1 for security, ECB mode)
- SSRF (user-controlled URLs in HTTP requests)
"""

from __future__ import annotations

import re
from sentinel.core import Finding, RiskLevel


# --- SQL Injection ---
_SQL_INJECTION = [
    # String concatenation/f-string in SQL queries (not parameterized %s/%d)
    (
        re.compile(
            r"""(?ix)
            (?:cursor\.execute|\.query|\.raw|\.execute|\.executemany)\s*\(\s*
            (?:f['""]|['""].*?\s*\+|.*?\.format\s*\()
            """,
        ),
        RiskLevel.CRITICAL,
        "SQL injection: string interpolation in SQL query — use parameterized queries",
    ),
    # Raw SQL with string concatenation
    (
        re.compile(
            r"""(?ix)
            (?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\s+.*?
            (?:['""]?\s*\+\s*(?:req|request|user|input|params|args|data)\b|
            \{\s*(?:req|request|user|input|params|args|data)\b)
            """,
        ),
        RiskLevel.HIGH,
        "SQL injection: user input concatenated into SQL statement",
    ),
]

# --- Command Injection ---
_COMMAND_INJECTION = [
    # subprocess with shell=True and user input
    (
        re.compile(
            r"""(?ix)
            subprocess\.(?:call|run|Popen|check_output|check_call)\s*\(
            .*?shell\s*=\s*True
            """,
            re.DOTALL,
        ),
        RiskLevel.HIGH,
        "Command injection risk: subprocess with shell=True — use shell=False with argument list",
    ),
    # os.system / os.popen with variable input
    (
        re.compile(
            r"""(?ix)
            os\.(?:system|popen)\s*\(\s*
            (?:f['""]|['""].*?\+|.*?\.format|.*?%\s)
            """,
        ),
        RiskLevel.CRITICAL,
        "Command injection: os.system/popen with string interpolation",
    ),
    # JS/TS exec/eval with user input
    (
        re.compile(
            r"""(?ix)
            (?:child_process\.exec|execSync|eval)\s*\(
            .*?(?:req\.|request\.|params\.|query\.|body\.|input|args)
            """,
        ),
        RiskLevel.CRITICAL,
        "Command injection: exec/eval with user-controlled input",
    ),
]

# --- XSS ---
_XSS_PATTERNS = [
    # innerHTML/outerHTML with user input
    (
        re.compile(
            r"""(?ix)
            \.innerHTML\s*=\s*(?!['""]\s*$)
            .*?(?:req|request|user|input|params|query|data|variable)\b
            """,
        ),
        RiskLevel.HIGH,
        "XSS: innerHTML set with potentially user-controlled value — use textContent or sanitize",
    ),
    # document.write with variables
    (
        re.compile(r"(?i)document\.write\s*\((?!['\"]\s*\))"),
        RiskLevel.MEDIUM,
        "XSS risk: document.write with dynamic content",
    ),
    # Jinja2/Django template |safe filter
    (
        re.compile(r"\{\{.*?\|\s*safe\s*\}\}"),
        RiskLevel.HIGH,
        "XSS: template |safe filter bypasses auto-escaping — ensure content is sanitized",
    ),
    # Flask/Django render with user data marked safe
    (
        re.compile(r"(?i)Markup\s*\(.*?(?:request|user|input|data)\b"),
        RiskLevel.HIGH,
        "XSS: Markup() with user-controlled content bypasses escaping",
    ),
]

# --- Path Traversal ---
_PATH_TRAVERSAL = [
    # open() with user-controlled path, no validation
    (
        re.compile(
            r"""(?ix)
            open\s*\(\s*
            (?:f['""]|.*?\+|.*?\.format|.*?os\.path\.join\s*\()
            .*?(?:req|request|user|input|params|filename|path|file_path|upload)\b
            """,
        ),
        RiskLevel.HIGH,
        "Path traversal: file opened with user-controlled path — validate and sanitize path",
    ),
    # send_file / send_from_directory with user input
    (
        re.compile(
            r"""(?ix)
            (?:send_file|send_from_directory)\s*\(
            .*?(?:request|user|input|params|filename)\b
            """,
        ),
        RiskLevel.MEDIUM,
        "Path traversal risk: serving file with user-controlled path",
    ),
]

# --- Insecure Deserialization ---
_DESERIALIZATION = [
    (
        re.compile(r"(?i)pickle\.loads?\s*\("),
        RiskLevel.HIGH,
        "Insecure deserialization: pickle.load can execute arbitrary code — use JSON or safe alternatives",
    ),
    (
        re.compile(r"(?i)yaml\.(?:load|unsafe_load)\s*\((?!.*Loader\s*=\s*yaml\.SafeLoader)"),
        RiskLevel.HIGH,
        "Insecure deserialization: yaml.load without SafeLoader can execute arbitrary code",
    ),
    (
        re.compile(r"(?i)\beval\s*\(\s*(?!['\"][^'\"]*['\"]\s*\))"),
        RiskLevel.HIGH,
        "Code injection: eval() with dynamic input can execute arbitrary code",
    ),
    (
        re.compile(r"(?i)marshal\.loads?\s*\("),
        RiskLevel.HIGH,
        "Insecure deserialization: marshal.load is not safe for untrusted data",
    ),
]

# --- Hardcoded Secrets ---
_HARDCODED_SECRETS = [
    # API keys, passwords, tokens in source
    (
        re.compile(
            r"""(?ix)
            (?:password|passwd|pwd|secret|api_key|apikey|api_secret|
            access_token|auth_token|private_key|secret_key)\s*
            =\s*['""][^'""]{8,}['""]\s*$
            """,
            re.MULTILINE,
        ),
        RiskLevel.CRITICAL,
        "Hardcoded secret: credential value embedded in source code — use environment variables",
    ),
    # AWS keys
    (
        re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
        RiskLevel.CRITICAL,
        "Hardcoded AWS access key detected in source code",
    ),
]

# --- Insecure Crypto ---
_INSECURE_CRYPTO = [
    (
        re.compile(r"(?i)(?:hashlib\.)?(?:md5|sha1)\s*\("),
        RiskLevel.MEDIUM,
        "Weak hash: MD5/SHA1 should not be used for security purposes — use SHA-256+",
    ),
    (
        re.compile(r"(?i)AES\.new\s*\(.*?MODE_ECB"),
        RiskLevel.HIGH,
        "Insecure crypto: AES-ECB mode does not provide semantic security — use CBC or GCM",
    ),
    (
        re.compile(r"(?i)DES\b.*?\.new\s*\("),
        RiskLevel.HIGH,
        "Insecure crypto: DES is broken — use AES-256",
    ),
]

# --- SSRF ---
_SSRF_PATTERNS = [
    (
        re.compile(
            r"""(?ix)
            (?:requests\.(?:get|post|put|delete|patch|head)|
            urllib\.request\.urlopen|
            http\.client\.HTTP|
            fetch\s*\(|
            axios\.(?:get|post))\s*\(
            .*?(?:req\.|request\.|params\.|query\.|body\.|user_?input|url_param)\b
            """,
        ),
        RiskLevel.HIGH,
        "SSRF risk: HTTP request with user-controlled URL — validate against allowlist",
    ),
]

_ALL_PATTERNS = (
    [("sql_injection", p, r, d) for p, r, d in _SQL_INJECTION]
    + [("command_injection", p, r, d) for p, r, d in _COMMAND_INJECTION]
    + [("xss", p, r, d) for p, r, d in _XSS_PATTERNS]
    + [("path_traversal", p, r, d) for p, r, d in _PATH_TRAVERSAL]
    + [("insecure_deserialization", p, r, d) for p, r, d in _DESERIALIZATION]
    + [("hardcoded_secret", p, r, d) for p, r, d in _HARDCODED_SECRETS]
    + [("insecure_crypto", p, r, d) for p, r, d in _INSECURE_CRYPTO]
    + [("ssrf", p, r, d) for p, r, d in _SSRF_PATTERNS]
)


class CodeScanner:
    """Scans generated code for OWASP Top 10 vulnerabilities.

    Usage:
        scanner = CodeScanner()

        # Scan a code string
        findings = scanner.scan(code_text)

        # Scan with file context (adjusts patterns by language)
        findings = scanner.scan(code_text, filename="app.py")

        # Use as a PostToolUse hook for Claude Code
        findings = scanner.scan_tool_output(tool_name, output)
    """

    def scan(
        self,
        code: str,
        *,
        filename: str | None = None,
    ) -> list[Finding]:
        """Scan code for security vulnerabilities.

        Args:
            code: Source code text to scan.
            filename: Optional filename for language-specific filtering.

        Returns:
            List of findings.
        """
        if not code or len(code.strip()) < 5:
            return []

        findings = []
        for category, pattern, risk, description in _ALL_PATTERNS:
            for match in pattern.finditer(code):
                # Get line number for the match
                line_num = code[:match.start()].count("\n") + 1
                findings.append(
                    Finding(
                        scanner="code_scanner",
                        category=category,
                        description=f"Line {line_num}: {description}",
                        risk=risk,
                        span=(match.start(), match.end()),
                        metadata={
                            "line": line_num,
                            "match": match.group()[:100],
                            "filename": filename or "unknown",
                        },
                    )
                )

        return findings

    def scan_tool_output(
        self,
        tool_name: str,
        output: dict,
    ) -> list[Finding]:
        """Scan a Claude Code tool output for code vulnerabilities.

        Designed for PostToolUse hooks on Write and Edit tools.

        Args:
            tool_name: Name of the tool (e.g., "Write", "Edit").
            output: Tool output/arguments dict.

        Returns:
            List of findings.
        """
        if tool_name not in ("Write", "Edit", "write", "edit"):
            return []

        filename = output.get("file_path", output.get("path", ""))
        content = output.get("content", output.get("new_string", ""))

        if not content:
            return []

        return self.scan(content, filename=filename)
