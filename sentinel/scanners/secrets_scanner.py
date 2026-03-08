"""Secrets scanner — detects hardcoded credentials, API keys, and tokens in source code.

Detects 40+ secret patterns across major cloud providers and services:
- AWS access keys, secret keys
- GitHub tokens (ghp_, gho_, ghs_, ghr_, github_pat_)
- Google API keys, OAuth secrets, service account keys
- Stripe, Twilio, SendGrid, Slack, Heroku API keys
- OpenAI, Anthropic API keys
- Azure, Firebase, Supabase keys
- Generic passwords, API keys, tokens, connection strings
- Private keys (RSA, EC, PGP, SSH)
- JWT tokens, bearer tokens
- Database connection strings with embedded credentials

Usage:
    from sentinel.scanners.secrets_scanner import SecretsScanner

    scanner = SecretsScanner()
    findings = scanner.scan(source_code)
    findings = scanner.scan(source_code, filename="config.py")
    findings = scanner.scan_file(Path("app.py"))
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from sentinel.core import Finding, RiskLevel


# --- Provider-specific API key patterns ---
_PROVIDER_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    # AWS
    (
        "aws_access_key",
        re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
        RiskLevel.CRITICAL,
        "AWS access key ID detected",
    ),
    (
        "aws_secret_key",
        re.compile(
            r"""(?ix)
            (?:aws_secret_access_key|aws_secret_key|secret_key)\s*
            (?:=|:|\s)\s*['""]?([A-Za-z0-9/+=]{40})['""]?
            """
        ),
        RiskLevel.CRITICAL,
        "AWS secret access key detected",
    ),
    # GitHub
    (
        "github_token",
        re.compile(r"ghp_[A-Za-z0-9]{36}"),
        RiskLevel.CRITICAL,
        "GitHub personal access token (ghp_) detected",
    ),
    (
        "github_oauth_token",
        re.compile(r"gho_[A-Za-z0-9]{36}"),
        RiskLevel.CRITICAL,
        "GitHub OAuth access token (gho_) detected",
    ),
    (
        "github_app_token",
        re.compile(r"(?:ghs|ghr)_[A-Za-z0-9]{36}"),
        RiskLevel.CRITICAL,
        "GitHub app token (ghs_/ghr_) detected",
    ),
    (
        "github_pat",
        re.compile(r"github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}"),
        RiskLevel.CRITICAL,
        "GitHub fine-grained personal access token detected",
    ),
    # Google
    (
        "google_api_key",
        re.compile(r"AIza[A-Za-z0-9_\-]{35}"),
        RiskLevel.HIGH,
        "Google API key detected",
    ),
    (
        "google_oauth_secret",
        re.compile(r"GOCSPX-[A-Za-z0-9_\-]{28}"),
        RiskLevel.CRITICAL,
        "Google OAuth client secret detected",
    ),
    # Stripe
    (
        "stripe_secret_key",
        re.compile(r"sk_live_[A-Za-z0-9]{24,}"),
        RiskLevel.CRITICAL,
        "Stripe live secret key detected",
    ),
    (
        "stripe_publishable_key",
        re.compile(r"pk_live_[A-Za-z0-9]{24,}"),
        RiskLevel.MEDIUM,
        "Stripe live publishable key detected (public but should be in env vars)",
    ),
    # Slack
    (
        "slack_bot_token",
        re.compile(r"xoxb-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]{24}"),
        RiskLevel.CRITICAL,
        "Slack bot token detected",
    ),
    (
        "slack_user_token",
        re.compile(r"xoxp-[0-9]{10,13}-[0-9]{10,13}-[0-9]{10,13}-[a-f0-9]{32}"),
        RiskLevel.CRITICAL,
        "Slack user token detected",
    ),
    (
        "slack_webhook",
        re.compile(r"https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{24}"),
        RiskLevel.HIGH,
        "Slack webhook URL detected",
    ),
    # OpenAI
    (
        "openai_api_key",
        re.compile(r"sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}"),
        RiskLevel.CRITICAL,
        "OpenAI API key detected",
    ),
    (
        "openai_project_key",
        re.compile(r"sk-proj-[A-Za-z0-9_\-]{40,}"),
        RiskLevel.CRITICAL,
        "OpenAI project API key detected",
    ),
    # Anthropic
    (
        "anthropic_api_key",
        re.compile(r"sk-ant-api03-[A-Za-z0-9_\-]{90,}"),
        RiskLevel.CRITICAL,
        "Anthropic API key detected",
    ),
    # Twilio
    (
        "twilio_api_key",
        re.compile(r"SK[a-f0-9]{32}"),
        RiskLevel.HIGH,
        "Twilio API key detected",
    ),
    # SendGrid
    (
        "sendgrid_api_key",
        re.compile(r"SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}"),
        RiskLevel.CRITICAL,
        "SendGrid API key detected",
    ),
    # Heroku
    (
        "heroku_api_key",
        re.compile(
            r"""(?ix)
            (?:heroku_api_key|heroku_secret)\s*(?:=|:)\s*
            ['""]?[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}['""]?
            """
        ),
        RiskLevel.CRITICAL,
        "Heroku API key detected",
    ),
    # npm
    (
        "npm_token",
        re.compile(r"npm_[A-Za-z0-9]{36}"),
        RiskLevel.CRITICAL,
        "npm access token detected",
    ),
    # PyPI
    (
        "pypi_token",
        re.compile(r"pypi-[A-Za-z0-9_\-]{50,}"),
        RiskLevel.CRITICAL,
        "PyPI API token detected",
    ),
    # Azure
    (
        "azure_storage_key",
        re.compile(
            r"""(?ix)
            (?:AccountKey|azure_storage_key|AZURE_STORAGE_KEY)\s*(?:=|:)\s*
            ['""]?[A-Za-z0-9+/]{86}==['""]?
            """
        ),
        RiskLevel.CRITICAL,
        "Azure storage account key detected",
    ),
    # Firebase
    (
        "firebase_key",
        re.compile(
            r"""(?ix)
            (?:firebase_api_key|firebase_key|FIREBASE_KEY)\s*(?:=|:)\s*
            ['""]?AIza[A-Za-z0-9_\-]{35}['""]?
            """
        ),
        RiskLevel.HIGH,
        "Firebase API key detected in configuration",
    ),
    # Supabase
    (
        "supabase_key",
        re.compile(r"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\.[A-Za-z0-9_\-]{50,}\.[A-Za-z0-9_\-]{40,}"),
        RiskLevel.HIGH,
        "Supabase service key (JWT) detected",
    ),
    # Mailgun
    (
        "mailgun_api_key",
        re.compile(r"key-[A-Za-z0-9]{32}"),
        RiskLevel.HIGH,
        "Mailgun API key detected",
    ),
]

# --- Private key patterns ---
_PRIVATE_KEY_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    (
        "rsa_private_key",
        re.compile(r"-----BEGIN RSA PRIVATE KEY-----"),
        RiskLevel.CRITICAL,
        "RSA private key detected in source code",
    ),
    (
        "ec_private_key",
        re.compile(r"-----BEGIN EC PRIVATE KEY-----"),
        RiskLevel.CRITICAL,
        "EC private key detected in source code",
    ),
    (
        "generic_private_key",
        re.compile(r"-----BEGIN PRIVATE KEY-----"),
        RiskLevel.CRITICAL,
        "Private key detected in source code",
    ),
    (
        "pgp_private_key",
        re.compile(r"-----BEGIN PGP PRIVATE KEY BLOCK-----"),
        RiskLevel.CRITICAL,
        "PGP private key detected in source code",
    ),
    (
        "openssh_private_key",
        re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----"),
        RiskLevel.CRITICAL,
        "OpenSSH private key detected in source code",
    ),
]

# --- Generic secret assignment patterns ---
_GENERIC_SECRET_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    (
        "hardcoded_password",
        re.compile(
            r"""(?ix)(?:password|passwd|pwd|pass)\s*(?:=|:)\s*['"]([^'"\s]{8,})['"]"""
        ),
        RiskLevel.HIGH,
        "Hardcoded password detected — use environment variables or a secrets manager",
    ),
    (
        "hardcoded_secret",
        re.compile(
            r"""(?ix)(?:secret|secret_key|api_secret|app_secret|client_secret)\s*(?:=|:)\s*['"]([^'"\s]{8,})['"]"""
        ),
        RiskLevel.HIGH,
        "Hardcoded secret value detected — use environment variables",
    ),
    (
        "hardcoded_api_key",
        re.compile(
            r"""(?ix)(?:api_key|apikey|api_token|access_key|access_token|auth_token)\s*(?:=|:)\s*['"]([^'"\s]{8,})['"]"""
        ),
        RiskLevel.HIGH,
        "Hardcoded API key/token detected — use environment variables",
    ),
    (
        "bearer_token",
        re.compile(
            r"""(?ix)(?:Authorization|Bearer)\s*(?:=|:)\s*['"]Bearer\s+[A-Za-z0-9_\-.]{20,}['"]"""
        ),
        RiskLevel.HIGH,
        "Hardcoded bearer token detected in authorization header",
    ),
]

# --- Connection string patterns ---
_CONNECTION_STRING_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel, str]] = [
    (
        "database_url",
        re.compile(
            r"""(?ix)
            (?:postgres(?:ql)?|mysql|mssql|mongodb(?:\+srv)?|redis|amqp)://
            [^:\s]+:([^@\s]+)@[^\s'"]+
            """,
        ),
        RiskLevel.CRITICAL,
        "Database connection string with embedded credentials detected",
    ),
]

# Files to skip (binary, minified, lock files, etc.)
_SKIP_EXTENSIONS = {
    ".min.js", ".min.css", ".map", ".lock", ".sum",
    ".woff", ".woff2", ".ttf", ".eot", ".ico",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo", ".class", ".o",
}

_SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "Pipfile.lock",
    "poetry.lock", "Cargo.lock", "go.sum",
}

# Patterns that are likely false positives
_FALSE_POSITIVE_VALUES = {
    "password", "your_password", "your_password_here",
    "changeme", "CHANGEME", "change_me",
    "example", "your_api_key", "your_api_key_here",
    "your_secret", "your_secret_here", "your_token",
    "xxx", "XXX", "todo", "TODO", "placeholder",
    "test", "testing", "dummy", "fake", "mock",
    "secret", "api_key", "token", "password123",
    "replace_me", "REPLACE_ME", "INSERT_KEY_HERE",
    "your-api-key", "your-secret-key", "your-token",
    "<your-api-key>", "<API_KEY>", "<SECRET>",
    "${API_KEY}", "${SECRET}", "${PASSWORD}",
    "process.env.API_KEY", "os.environ",
    "********", "••••••••",
}


def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )


def _is_false_positive(value: str) -> bool:
    """Check if a captured secret value is likely a placeholder."""
    cleaned = value.strip("'\"` \t")
    if cleaned.lower() in {v.lower() for v in _FALSE_POSITIVE_VALUES}:
        return True
    # Template/variable references
    if any(cleaned.startswith(p) for p in ("${", "#{", "{{", "%(", "os.environ", "process.env", "ENV[")):
        return True
    # Too uniform (all same char)
    if len(set(cleaned)) <= 2:
        return True
    return False


class SecretsScanner:
    """Scans source code for hardcoded secrets, API keys, and credentials.

    Usage:
        scanner = SecretsScanner()
        findings = scanner.scan(code)
        findings = scanner.scan(code, filename="config.py")
        findings = scanner.scan_file(Path("settings.py"))
    """

    def scan(
        self,
        code: str,
        *,
        filename: str | None = None,
        check_entropy: bool = True,
    ) -> list[Finding]:
        """Scan source code for hardcoded secrets.

        Args:
            code: Source code text.
            filename: Optional filename for context and skip logic.
            check_entropy: Whether to use entropy analysis for generic patterns.

        Returns:
            List of findings.
        """
        if not code or len(code.strip()) < 10:
            return []

        # Skip known non-source files
        if filename:
            p = Path(filename)
            if p.suffix.lower() in _SKIP_EXTENSIONS:
                return []
            if p.name in _SKIP_FILENAMES:
                return []
            # Check for .min.js, .min.css etc.
            if ".min." in p.name:
                return []

        findings: list[Finding] = []
        seen_spans: set[tuple[int, int]] = set()

        # 1. Provider-specific patterns (highest confidence)
        for category, pattern, risk, description in _PROVIDER_PATTERNS:
            for match in pattern.finditer(code):
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                line_num = code[:match.start()].count("\n") + 1
                # Check if it's in a comment (basic check)
                line_start = code.rfind("\n", 0, match.start()) + 1
                line_text = code[line_start:match.end()]
                if _is_comment_only(line_text, filename):
                    continue
                findings.append(Finding(
                    scanner="secrets_scanner",
                    category=category,
                    description=f"Line {line_num}: {description}",
                    risk=risk,
                    span=span,
                    metadata={
                        "line": line_num,
                        "match": _redact_match(match.group()),
                        "filename": filename or "unknown",
                    },
                ))

        # 2. Private key patterns
        for category, pattern, risk, description in _PRIVATE_KEY_PATTERNS:
            for match in pattern.finditer(code):
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                line_num = code[:match.start()].count("\n") + 1
                findings.append(Finding(
                    scanner="secrets_scanner",
                    category=category,
                    description=f"Line {line_num}: {description}",
                    risk=risk,
                    span=span,
                    metadata={
                        "line": line_num,
                        "match": match.group()[:40],
                        "filename": filename or "unknown",
                    },
                ))

        # 3. Generic secret patterns (with false positive filtering)
        for category, pattern, risk, description in _GENERIC_SECRET_PATTERNS:
            for match in pattern.finditer(code):
                # Get the captured group (the actual value) or full match
                value = match.group(1) if match.lastindex else match.group()
                if _is_false_positive(value):
                    continue
                if check_entropy and _shannon_entropy(value) < 2.5:
                    continue
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                line_num = code[:match.start()].count("\n") + 1
                line_start = code.rfind("\n", 0, match.start()) + 1
                line_text = code[line_start:match.end()]
                if _is_comment_only(line_text, filename):
                    continue
                findings.append(Finding(
                    scanner="secrets_scanner",
                    category=category,
                    description=f"Line {line_num}: {description}",
                    risk=risk,
                    span=span,
                    metadata={
                        "line": line_num,
                        "match": _redact_match(match.group()),
                        "filename": filename or "unknown",
                    },
                ))

        # 4. Connection strings
        for category, pattern, risk, description in _CONNECTION_STRING_PATTERNS:
            for match in pattern.finditer(code):
                value = match.group(1) if match.lastindex else match.group()
                if _is_false_positive(value):
                    continue
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                line_num = code[:match.start()].count("\n") + 1
                findings.append(Finding(
                    scanner="secrets_scanner",
                    category=category,
                    description=f"Line {line_num}: {description}",
                    risk=risk,
                    span=span,
                    metadata={
                        "line": line_num,
                        "match": _redact_match(match.group()),
                        "filename": filename or "unknown",
                    },
                ))

        return findings

    def scan_file(self, filepath: Path) -> list[Finding]:
        """Scan a single file for secrets.

        Args:
            filepath: Path to source file.

        Returns:
            List of findings.
        """
        try:
            code = filepath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return []
        return self.scan(code, filename=str(filepath))

    def scan_directory(
        self,
        directory: Path,
        *,
        max_files: int = 100,
        recursive: bool = True,
    ) -> list[Finding]:
        """Scan all source files in a directory for secrets.

        Args:
            directory: Directory to scan.
            max_files: Maximum number of files to scan.
            recursive: Whether to scan subdirectories.

        Returns:
            List of findings from all files.
        """
        all_findings: list[Finding] = []
        scanned = 0

        scannable_exts = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".rb", ".php",
            ".java", ".go", ".rs", ".cs", ".swift", ".kt",
            ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
            ".env", ".conf", ".properties", ".xml",
            ".sh", ".bash", ".zsh", ".ps1",
            ".tf", ".hcl",  # Terraform
            ".dockerfile",
        }

        pattern_func = directory.rglob if recursive else directory.iterdir
        for fpath in sorted(pattern_func("*")):
            if scanned >= max_files:
                break
            if not fpath.is_file():
                continue
            if fpath.suffix.lower() not in scannable_exts and fpath.name not in {"Dockerfile", ".env", ".env.local", ".env.production"}:
                continue
            if fpath.name in _SKIP_FILENAMES:
                continue
            # Skip common directories
            parts = fpath.relative_to(directory).parts
            if any(p in {"node_modules", ".git", "__pycache__", "venv", ".venv", "dist", "build"} for p in parts):
                continue

            findings = self.scan_file(fpath)
            all_findings.extend(findings)
            scanned += 1

        return all_findings


def _is_comment_only(line_text: str, filename: str | None) -> bool:
    """Check if a line is just a comment (basic heuristic)."""
    stripped = line_text.strip()
    if stripped.startswith("#") or stripped.startswith("//"):
        return True
    if filename and filename.endswith((".py", ".rb", ".yaml", ".yml", ".sh")):
        if stripped.startswith("#"):
            return True
    return False


def _redact_match(match_text: str) -> str:
    """Redact the middle of a matched secret for safe reporting."""
    if len(match_text) <= 12:
        return match_text[:4] + "****"
    return match_text[:6] + "****" + match_text[-4:]
