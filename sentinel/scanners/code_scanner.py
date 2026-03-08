"""Code vulnerability scanner for LLM-generated Python code.

Detects common security vulnerabilities that LLMs frequently introduce when
generating code, including OWASP Top 10 issues and Python-specific risks.

Scans for 9 vulnerability categories:
- SQL injection (f-strings, .format(), % formatting, + concatenation in queries)
- Command injection (os.system(), subprocess with shell=True, os.popen(),
  child_process.exec)
- Path traversal (open() with unsanitized paths containing ../)
- Unsafe deserialization (pickle.loads(), yaml.load() without SafeLoader, eval(), exec())
- Hardcoded credentials (passwords/secrets directly in code)
- XSS in templates (render_template_string, Markup(), |safe filter, innerHTML)
- Insecure HTTP (http:// URLs, verify=False in requests)
- Insecure crypto (MD5/SHA1 for security, AES-ECB mode, DES)
- SSRF (user-controlled URLs in HTTP requests)

Usage:
    from sentinel.scanners.code_scanner import CodeScanner

    scanner = CodeScanner()

    # Detect SQL injection
    findings = scanner.scan('cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")')
    # => [Finding(category="sql_injection", risk=CRITICAL, ...)]

    # Detect command injection
    findings = scanner.scan('os.system(f"rm -rf {user_input}")')
    # => [Finding(category="command_injection", risk=CRITICAL, ...)]

    # Detect unsafe deserialization
    findings = scanner.scan('data = pickle.loads(user_data)')
    # => [Finding(category="unsafe_deserialization", risk=HIGH, ...)]

    # Detect hardcoded credentials
    findings = scanner.scan('password = "SuperSecret123!"')
    # => [Finding(category="hardcoded_credential", risk=HIGH, ...)]

    # Detect insecure HTTP
    findings = scanner.scan('requests.get("http://api.example.com/data")')
    # => [Finding(category="insecure_http", risk=MEDIUM, ...)]

    # Use with SentinelGuard
    from sentinel.core import SentinelGuard
    guard = SentinelGuard(scanners=[CodeScanner()])
    result = guard.scan(generated_code)
    if not result.safe:
        print("Vulnerabilities found!", result.findings)
"""

from __future__ import annotations

import re
from sentinel.core import Finding, RiskLevel


# ---------------------------------------------------------------------------
# 1. SQL Injection
# ---------------------------------------------------------------------------
_SQL_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    # f-string in execute/query call
    (
        "sql_fstring",
        re.compile(
            r"""(?ix)
            (?:cursor\.execute|\.execute|\.executemany|\.query|\.raw)\s*\(\s*
            f["']
            """,
        ),
        RiskLevel.CRITICAL,
        "SQL injection: f-string used in SQL query — use parameterized queries instead",
    ),
    # .format() in execute/query call
    (
        "sql_format",
        re.compile(
            r"""(?ix)
            (?:cursor\.execute|\.execute|\.executemany|\.query|\.raw)\s*\(\s*
            ["'].*?["']\s*\.format\s*\(
            """,
        ),
        RiskLevel.CRITICAL,
        "SQL injection: .format() used in SQL query — use parameterized queries instead",
    ),
    # % formatting in execute/query call
    (
        "sql_percent",
        re.compile(
            r"""(?ix)
            (?:cursor\.execute|\.execute|\.executemany|\.query|\.raw)\s*\(\s*
            ["'].*?%s.*?["']\s*%\s*
            """,
        ),
        RiskLevel.HIGH,
        "SQL injection: %-formatting used in SQL query — use parameterized queries instead",
    ),
    # String concatenation (+) with SQL keywords
    (
        "sql_concat",
        re.compile(
            r"""(?ix)
            (?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|DROP|CREATE|ALTER)\s+
            .*?["']\s*\+\s*\w
            """,
        ),
        RiskLevel.HIGH,
        "SQL injection: string concatenation in SQL statement — use parameterized queries",
    ),
    # Raw SQL with user input variables
    (
        "sql_user_input",
        re.compile(
            r"""(?ix)
            (?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\s+.*?
            (?:["']?\s*\+\s*(?:req|request|user|input|params|args|data)\b|
            \{\s*(?:req|request|user|input|params|args|data)\b)
            """,
        ),
        RiskLevel.HIGH,
        "SQL injection: user input concatenated into SQL statement",
    ),
    # f-string containing SQL keywords (standalone, not in execute)
    (
        "sql_fstring_standalone",
        re.compile(
            r"""(?ix)
            f["'](?=.*(?:SELECT|INSERT|UPDATE|DELETE|DROP)\s)
            .*?\{.*?\}.*?["']
            """,
        ),
        RiskLevel.HIGH,
        "SQL injection: f-string contains SQL keywords with interpolated variables",
    ),
]


# ---------------------------------------------------------------------------
# 2. Command Injection
# ---------------------------------------------------------------------------
_COMMAND_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    # os.system() — always dangerous
    (
        "os_system",
        re.compile(
            r"""(?ix)
            os\.system\s*\(
            """,
        ),
        RiskLevel.CRITICAL,
        "Command injection: os.system() executes shell commands — use subprocess with shell=False",
    ),
    # os.popen() — always dangerous
    (
        "os_popen",
        re.compile(
            r"""(?ix)
            os\.popen\s*\(
            """,
        ),
        RiskLevel.CRITICAL,
        "Command injection: os.popen() executes shell commands — use subprocess with shell=False",
    ),
    # subprocess with shell=True
    (
        "subprocess_shell",
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
    # os.system/popen with string interpolation
    (
        "os_system_interpolation",
        re.compile(
            r"""(?ix)
            os\.(?:system|popen)\s*\(\s*
            (?:f["']|["'].*?\+|.*?\.format|.*?%\s)
            """,
        ),
        RiskLevel.CRITICAL,
        "Command injection: os.system/popen with string interpolation",
    ),
    # JS/TS child_process.exec, execSync, eval with user input
    (
        "js_exec_user_input",
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


# ---------------------------------------------------------------------------
# 3. Path Traversal
# ---------------------------------------------------------------------------
_PATH_TRAVERSAL_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    # open() with f-string or .format() containing path-like variables
    (
        "open_fstring",
        re.compile(
            r"""(?ix)
            open\s*\(\s*f["']
            .*?\{.*?(?:path|file|filename|name|dir|folder|upload|user|input|param).*?\}
            """,
        ),
        RiskLevel.HIGH,
        "Path traversal: open() with user-controllable f-string path — validate and sanitize path",
    ),
    # open() with string concatenation
    (
        "open_concat",
        re.compile(
            r"""(?ix)
            open\s*\(\s*
            (?:\w+\s*\+\s*["']|["'].*?["']\s*\+\s*\w)
            """,
        ),
        RiskLevel.MEDIUM,
        "Path traversal risk: open() with string concatenation — validate and sanitize path",
    ),
    # Literal ../ in file paths passed to open/Path
    (
        "dotdot_path",
        re.compile(
            r"""(?ix)
            (?:open|Path)\s*\(\s*
            .*?\.\.[\\/]
            """,
        ),
        RiskLevel.HIGH,
        "Path traversal: ../ sequence in file path — normalize and validate path",
    ),
    # open() with user-controlled path via os.path.join or format
    (
        "open_user_input",
        re.compile(
            r"""(?ix)
            open\s*\(\s*
            (?:f["']|.*?\+|.*?\.format|.*?os\.path\.join\s*\()
            .*?(?:req|request|user|input|params|filename|path|file_path|upload)\b
            """,
        ),
        RiskLevel.HIGH,
        "Path traversal: file opened with user-controlled path — validate and sanitize path",
    ),
    # send_file / send_from_directory with user input
    (
        "send_file_user_input",
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


# ---------------------------------------------------------------------------
# 4. Unsafe Deserialization
# ---------------------------------------------------------------------------
_UNSAFE_DESERIALIZATION_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    # pickle.load / pickle.loads
    (
        "pickle_load",
        re.compile(r"(?i)pickle\.loads?\s*\("),
        RiskLevel.HIGH,
        "Unsafe deserialization: pickle.load(s) can execute arbitrary code — use JSON or safe alternatives",
    ),
    # yaml.load without SafeLoader / yaml.unsafe_load
    (
        "yaml_unsafe_load",
        re.compile(
            r"(?i)yaml\.(?:load|unsafe_load)\s*\("
            r"(?!.*Loader\s*=\s*(?:yaml\.)?SafeLoader)"
        ),
        RiskLevel.HIGH,
        "Unsafe deserialization: yaml.load() without SafeLoader can execute arbitrary code",
    ),
    # eval() with dynamic input (not a string literal)
    (
        "eval_call",
        re.compile(r"(?i)\beval\s*\(\s*(?!['\"][^'\"]*['\"]?\s*\))"),
        RiskLevel.HIGH,
        "Unsafe deserialization: eval() with dynamic input can execute arbitrary code",
    ),
    # exec() with dynamic input
    (
        "exec_call",
        re.compile(r"(?i)\bexec\s*\(\s*(?!['\"][^'\"]*['\"]?\s*\))"),
        RiskLevel.HIGH,
        "Unsafe deserialization: exec() with dynamic input can execute arbitrary code",
    ),
    # marshal.load / marshal.loads
    (
        "marshal_load",
        re.compile(r"(?i)marshal\.loads?\s*\("),
        RiskLevel.HIGH,
        "Unsafe deserialization: marshal.load(s) is not safe for untrusted data",
    ),
    # shelve.open (uses pickle internally)
    (
        "shelve_open",
        re.compile(r"(?i)shelve\.open\s*\("),
        RiskLevel.MEDIUM,
        "Unsafe deserialization: shelve uses pickle internally — not safe for untrusted data",
    ),
]


# ---------------------------------------------------------------------------
# 5. Hardcoded Credentials
# ---------------------------------------------------------------------------
_HARDCODED_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    # password/secret/key = "literal value" (not env vars or placeholders)
    (
        "hardcoded_password",
        re.compile(
            r"""(?ix)
            (?:password|passwd|pwd|pass_?word)\s*=\s*
            ["'](?!(?:\s*["']|.*os\.environ|.*os\.getenv|.*ENV\[|.*\$\{))
            [^"']{8,}["']
            """,
        ),
        RiskLevel.HIGH,
        "Hardcoded credential: password embedded in source code — use environment variables",
    ),
    # secret / api_key / token = "literal value"
    (
        "hardcoded_secret",
        re.compile(
            r"""(?ix)
            (?:secret|secret_key|api_key|apikey|api_secret|api_token|
            access_key|access_token|auth_token|private_key|client_secret)\s*=\s*
            ["'](?!(?:\s*["']|.*os\.environ|.*os\.getenv|.*ENV\[|.*\$\{))
            [^"']{8,}["']
            """,
        ),
        RiskLevel.HIGH,
        "Hardcoded credential: secret/API key embedded in source code — use environment variables",
    ),
    # AWS access key pattern in source
    (
        "hardcoded_aws_key",
        re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
        RiskLevel.CRITICAL,
        "Hardcoded credential: AWS access key found in source code",
    ),
    # Hardcoded password/secret with = "value" ending at line boundary (broader)
    (
        "hardcoded_credential_eol",
        re.compile(
            r"""(?ix)
            (?:password|passwd|pwd|secret|api_key|apikey|api_secret|
            access_token|auth_token|private_key|secret_key)\s*
            =\s*[""][^""]{8,}[""]\s*$
            """,
            re.MULTILINE,
        ),
        RiskLevel.CRITICAL,
        "Hardcoded secret: credential value embedded in source code — use environment variables",
    ),
]

# Values that are clearly placeholders, not real secrets
_CREDENTIAL_FALSE_POSITIVES = {
    "password", "your_password", "your_password_here", "changeme",
    "example", "your_api_key", "your_secret", "your_token",
    "xxx", "todo", "placeholder", "test", "testing",
    "dummy", "fake", "mock", "replace_me", "insert_key_here",
    "password123", "secret123", "change_me",
}


# ---------------------------------------------------------------------------
# 6. XSS in Templates
# ---------------------------------------------------------------------------
_XSS_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    # render_template_string with user input
    (
        "render_template_string",
        re.compile(
            r"""(?ix)
            render_template_string\s*\(
            .*?(?:request|user|input|data|param|query|body)\b
            """,
        ),
        RiskLevel.HIGH,
        "XSS vulnerability: render_template_string with user input enables server-side template injection",
    ),
    # render_template_string with f-string or .format()
    (
        "render_template_string_interpolation",
        re.compile(
            r"""(?ix)
            render_template_string\s*\(\s*
            (?:f["']|["'].*?["']\s*\.format\s*\(|["'].*?%s)
            """,
        ),
        RiskLevel.HIGH,
        "XSS vulnerability: render_template_string with string interpolation",
    ),
    # Markup() with user-controlled data
    (
        "markup_user_data",
        re.compile(
            r"""(?ix)
            Markup\s*\(\s*
            (?:f["']|.*?(?:request|user|input|data|param|query|body)\b|
            ["'].*?["']\s*\.format\s*\()
            """,
        ),
        RiskLevel.HIGH,
        "XSS vulnerability: Markup() with user-controlled content bypasses auto-escaping",
    ),
    # |safe filter in Jinja2 templates
    (
        "jinja_safe_filter",
        re.compile(r"\{\{.*?\|\s*safe\s*\}\}"),
        RiskLevel.HIGH,
        "XSS vulnerability: |safe filter bypasses Jinja2 auto-escaping — ensure content is sanitized",
    ),
    # innerHTML with user input (JS generated by Python)
    (
        "innerhtml_assignment",
        re.compile(
            r"""(?ix)
            \.innerHTML\s*=\s*(?!["']\s*$)
            .*?(?:request|user|input|param|query|data|variable)\b
            """,
        ),
        RiskLevel.HIGH,
        "XSS: innerHTML set with potentially user-controlled value — use textContent or sanitize",
    ),
    # document.write with variables
    (
        "document_write",
        re.compile(r"(?i)document\.write\s*\((?!['\"][^'\"]*['\"]\s*\))"),
        RiskLevel.MEDIUM,
        "XSS risk: document.write with dynamic content",
    ),
]


# ---------------------------------------------------------------------------
# 7. Insecure HTTP
# ---------------------------------------------------------------------------
_INSECURE_HTTP_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    # requests to http:// URLs (not https://)
    (
        "http_url",
        re.compile(
            r"""(?ix)
            (?:requests\.(?:get|post|put|delete|patch|head|options)|
            urllib\.request\.urlopen|
            urlopen|
            http\.client\.HTTPConnection)\s*\(\s*
            ["']http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])
            """,
        ),
        RiskLevel.MEDIUM,
        "Insecure HTTP: request uses http:// instead of https:// — data sent in plaintext",
    ),
    # http:// URL in string assignments (broader pattern)
    (
        "http_url_assignment",
        re.compile(
            r"""(?ix)
            (?:url|endpoint|api_url|base_url|server_url|host)\s*=\s*
            ["']http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])
            [^"']+["']
            """,
        ),
        RiskLevel.MEDIUM,
        "Insecure HTTP: URL variable uses http:// instead of https://",
    ),
    # verify=False in requests (disables TLS verification)
    (
        "verify_false",
        re.compile(
            r"""(?ix)
            (?:requests\.(?:get|post|put|delete|patch|head|options|request|Session)|
            \.request|\.get|\.post)\s*\(
            .*?verify\s*=\s*False
            """,
            re.DOTALL,
        ),
        RiskLevel.HIGH,
        "Insecure HTTP: verify=False disables TLS certificate verification — vulnerable to MITM attacks",
    ),
]


# ---------------------------------------------------------------------------
# 8. Insecure Crypto
# ---------------------------------------------------------------------------
_INSECURE_CRYPTO_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    (
        "weak_hash",
        re.compile(r"(?i)(?:hashlib\.)?(?:md5|sha1)\s*\("),
        RiskLevel.MEDIUM,
        "Weak hash: MD5/SHA1 should not be used for security purposes — use SHA-256+",
    ),
    (
        "aes_ecb",
        re.compile(r"(?i)AES\.new\s*\(.*?MODE_ECB"),
        RiskLevel.HIGH,
        "Insecure crypto: AES-ECB mode does not provide semantic security — use CBC or GCM",
    ),
    (
        "des_usage",
        re.compile(r"(?i)DES\b.*?\.new\s*\("),
        RiskLevel.HIGH,
        "Insecure crypto: DES is broken — use AES-256",
    ),
]


# ---------------------------------------------------------------------------
# 9. SSRF
# ---------------------------------------------------------------------------
_SSRF_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    (
        "ssrf_user_url",
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


# ---------------------------------------------------------------------------
# Combine all patterns into a single list with category labels
# ---------------------------------------------------------------------------
_ALL_PATTERNS: list[tuple[str, str, re.Pattern[str], RiskLevel, str]] = []

for _label, _pat, _risk, _desc in _SQL_INJECTION_PATTERNS:
    _ALL_PATTERNS.append(("sql_injection", _label, _pat, _risk, _desc))

for _label, _pat, _risk, _desc in _COMMAND_INJECTION_PATTERNS:
    _ALL_PATTERNS.append(("command_injection", _label, _pat, _risk, _desc))

for _label, _pat, _risk, _desc in _PATH_TRAVERSAL_PATTERNS:
    _ALL_PATTERNS.append(("path_traversal", _label, _pat, _risk, _desc))

for _label, _pat, _risk, _desc in _UNSAFE_DESERIALIZATION_PATTERNS:
    _ALL_PATTERNS.append(("unsafe_deserialization", _label, _pat, _risk, _desc))

for _label, _pat, _risk, _desc in _HARDCODED_CREDENTIAL_PATTERNS:
    _ALL_PATTERNS.append(("hardcoded_credential", _label, _pat, _risk, _desc))

for _label, _pat, _risk, _desc in _XSS_PATTERNS:
    _ALL_PATTERNS.append(("xss_vulnerability", _label, _pat, _risk, _desc))

for _label, _pat, _risk, _desc in _INSECURE_HTTP_PATTERNS:
    _ALL_PATTERNS.append(("insecure_http", _label, _pat, _risk, _desc))

for _label, _pat, _risk, _desc in _INSECURE_CRYPTO_PATTERNS:
    _ALL_PATTERNS.append(("insecure_crypto", _label, _pat, _risk, _desc))

for _label, _pat, _risk, _desc in _SSRF_PATTERNS:
    _ALL_PATTERNS.append(("ssrf", _label, _pat, _risk, _desc))

# Category aliases: some categories have both a primary name and a legacy alias
# to maintain backward compatibility. Findings are emitted under BOTH names.
_CATEGORY_ALIASES: dict[str, str] = {
    "unsafe_deserialization": "insecure_deserialization",
    "hardcoded_credential": "hardcoded_secret",
    "xss_vulnerability": "xss",
}


class CodeScanner:
    """Scans LLM-generated code for common security vulnerabilities.

    Detects 9 categories of vulnerabilities using regex-based pattern matching
    with zero external dependencies:

    - **sql_injection**: String formatting/concatenation in SQL queries
    - **command_injection**: os.system(), subprocess with shell=True, os.popen(),
      child_process.exec
    - **path_traversal**: File operations with unsanitized user-controllable paths
    - **unsafe_deserialization**: pickle, yaml.load without SafeLoader, eval(), exec()
    - **hardcoded_credential**: Passwords/secrets embedded directly in code
    - **xss_vulnerability**: Template injection via render_template_string, Markup(),
      |safe filter, innerHTML
    - **insecure_http**: Plaintext HTTP requests, disabled TLS verification
    - **insecure_crypto**: Weak hashes (MD5/SHA1), insecure modes (AES-ECB), broken
      ciphers (DES)
    - **ssrf**: HTTP requests with user-controlled URLs

    Follows the Sentinel Scanner protocol (name attribute + scan method returning
    list[Finding]).

    Usage:
        from sentinel.scanners.code_scanner import CodeScanner
        from sentinel.core import SentinelGuard

        scanner = CodeScanner()

        # Standalone scan
        findings = scanner.scan('cursor.execute(f"SELECT * FROM users WHERE id = {uid}")')
        assert findings[0].category == "sql_injection"
        assert findings[0].risk.value == "critical"

        # With SentinelGuard
        guard = SentinelGuard(scanners=[scanner])
        result = guard.scan(generated_code)
        if result.blocked:
            print("Dangerous code detected!")
    """

    name: str = "code_scanner"

    def scan(
        self,
        text: str,
        context: dict | None = None,
        *,
        filename: str | None = None,
    ) -> list[Finding]:
        """Scan code text for security vulnerabilities.

        Args:
            text: Source code string to analyze.
            context: Optional context dict. Supports:
                - "filename": str — filename for metadata.
            filename: Optional filename for metadata (keyword-only convenience).

        Returns:
            List of Finding objects, one per detected vulnerability.
        """
        if not text or len(text.strip()) < 5:
            return []

        # Resolve filename from context or keyword arg
        _filename = filename
        if _filename is None and context and isinstance(context, dict):
            _filename = context.get("filename")

        findings: list[Finding] = []
        seen_spans: set[tuple[int, int]] = set()

        for category, pattern_name, pattern, risk, description in _ALL_PATTERNS:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())

                # Deduplicate overlapping matches
                if span in seen_spans:
                    continue
                seen_spans.add(span)

                # For hardcoded credentials, filter out obvious placeholders
                if category == "hardcoded_credential" and pattern_name in (
                    "hardcoded_password",
                    "hardcoded_secret",
                    "hardcoded_credential_eol",
                ):
                    matched_text = match.group()
                    # Extract the value between quotes
                    value_match = re.search(r"""["']([^"']+)["']""", matched_text)
                    if value_match:
                        value = value_match.group(1).lower().strip()
                        if value in _CREDENTIAL_FALSE_POSITIVES:
                            continue

                line_num = text[:match.start()].count("\n") + 1

                metadata = {
                    "pattern": pattern_name,
                    "line": line_num,
                    "match": match.group()[:100],
                    "matched_text": match.group()[:120],
                    "filename": _filename or "unknown",
                }

                findings.append(
                    Finding(
                        scanner=self.name,
                        category=category,
                        description=f"Line {line_num}: {description}",
                        risk=risk,
                        span=span,
                        metadata=metadata,
                    )
                )

                # Emit alias finding for backward compatibility
                alias = _CATEGORY_ALIASES.get(category)
                if alias:
                    findings.append(
                        Finding(
                            scanner=self.name,
                            category=alias,
                            description=f"Line {line_num}: {description}",
                            risk=risk,
                            span=span,
                            metadata=metadata,
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
            output: Tool output/arguments dict with "file_path"/"path"
                    and "content"/"new_string" keys.

        Returns:
            List of findings.
        """
        if tool_name.lower() not in ("write", "edit"):
            return []

        fn = output.get("file_path", output.get("path", ""))
        content = output.get("content", output.get("new_string", ""))

        if not content:
            return []

        return self.scan(content, filename=fn)
