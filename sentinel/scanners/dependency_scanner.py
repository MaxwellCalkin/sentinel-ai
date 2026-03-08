"""Dependency file scanner for supply chain attack detection.

Scans requirements.txt, package.json, Pipfile, pyproject.toml, and similar
files for supply chain attack patterns:
- Typosquatting (common misspellings of popular packages)
- Dependency confusion (private package names that shadow public ones)
- Suspicious install URLs (git+http, direct tarballs)
- Pinning issues (unpinned or wildcard versions)
- Known malicious packages
- Install scripts pointing to external URLs
"""

from __future__ import annotations

import re
from sentinel.core import Finding, RiskLevel


# Popular packages and their common typosquats
_TYPOSQUAT_TARGETS: dict[str, list[str]] = {
    # Python
    "requests": ["reqests", "requets", "request", "requsts", "reqeusts", "rquests"],
    "urllib3": ["urllib", "urlib3", "urllib4", "urrlib3"],
    "numpy": ["numpi", "numppy", "nunpy", "nump"],
    "pandas": ["panda", "pandsa", "pandass"],
    "flask": ["flaask", "flassk", "flaskk"],
    "django": ["djnago", "dajngo", "djago", "djngo"],
    "boto3": ["botto3", "boto33", "botoo3"],
    "cryptography": ["cyptography", "crytography", "cryptograpy"],
    "paramiko": ["parmiko", "parmaiko", "paramko"],
    "pyyaml": ["pymal", "pyaml", "pyamml"],
    "pillow": ["pilow", "pilllow", "pillw"],
    "scrapy": ["scrappy", "scapy", "scraapyy"],
    "beautifulsoup4": ["beautifulsoup", "beutifulsoup4", "beautifulsop4"],
    "setuptools": ["setuptool", "setup-tools", "setuptoolss"],
    "pip": ["piip", "pipp"],
    # JavaScript
    "express": ["expres", "expresss", "exress"],
    "lodash": ["lodassh", "lodaash", "loddash", "loadash"],
    "react": ["reac", "reactt", "raect"],
    "axios": ["axois", "axioss", "axos"],
    "webpack": ["webpak", "webppack", "wepback"],
    "next": ["nextt", "nex"],
    "typescript": ["typesript", "tyepscript", "typscript"],
    "eslint": ["eslintt", "eslit", "eslnt"],
}

# Known malicious package names (reported and removed from registries)
_KNOWN_MALICIOUS: set[str] = {
    # Python
    "colourama", "python-dateutil2", "jeIlyfish", "python3-dateutil",
    "acqusition", "apidev-coop", "baborern", "coffee-script",
    "diaborern", "easyinstall", "free-net-vpn", "graborern",
    "lemonern", "libpeshka", "mybiutiflpackage", "noblesse",
    "noblessev2", "noblesse2", "pypistats", "python-sqlite",
    "setup-tools", "testing123test", "virtualnv", "beautifulsup4",
    # JavaScript
    "event-stream-malicious", "crossenv", "cross-env.js", "d3.js",
    "fabric-js", "ffmpegs", "gruntcli", "http-proxy.js",
    "jquery.js", "mariadb", "mongose", "mssql-node", "mssql.js",
    "mysqljs", "node-fabric", "node-opencv", "node-opensl",
    "node-openssl", "node-sqlite", "node-tkinter", "nodecaffe",
    "nodefabric", "nodemailer-js", "noderequest", "nodesass",
    "nodesqlite", "opencv.js", "openssl.js", "proxy.js",
    "shadowsock", "smb", "sqlite.js", "sqliter", "sqlserver",
    "tkinter",
}

# Patterns for suspicious install sources
_SUSPICIOUS_URL_PATTERNS = [
    (
        re.compile(r"git\+https?://(?!github\.com/|gitlab\.com/|bitbucket\.org/)"),
        RiskLevel.HIGH,
        "Suspicious git URL: dependency installed from untrusted source",
    ),
    (
        re.compile(r"https?://[^/\s]+\.(?:xyz|tk|ml|ga|cf|gq|top|buzz|club)/"),
        RiskLevel.CRITICAL,
        "Dependency URL uses suspicious TLD commonly associated with malware",
    ),
    (
        re.compile(r"https?://(?:pastebin\.com|paste\.ee|hastebin\.com|ghostbin\.co)/"),
        RiskLevel.CRITICAL,
        "Dependency installed from paste site — likely malicious",
    ),
    (
        re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"),
        RiskLevel.HIGH,
        "Dependency installed from raw IP address — suspicious",
    ),
]

# Patterns for dangerous install hooks in package.json
_INSTALL_HOOK_PATTERNS = [
    (
        re.compile(r'"(?:pre|post)?install"\s*:\s*"[^"]*(?:curl|wget|nc |bash|sh |eval|exec)\b'),
        RiskLevel.CRITICAL,
        "Install script executes external commands — potential supply chain attack",
    ),
    (
        re.compile(r'"(?:pre|post)?install"\s*:\s*"[^"]*(?:https?://|ftp://)'),
        RiskLevel.HIGH,
        "Install script references external URL — review for supply chain safety",
    ),
]


class DependencyScanner:
    """Scans dependency files for supply chain attack patterns."""

    name = "dependency_scanner"

    def scan(self, content: str, filename: str = "") -> list[Finding]:
        """Scan dependency file content for supply chain risks.

        Args:
            content: File content to scan.
            filename: Name of the file being scanned (for format detection).

        Returns:
            List of findings.
        """
        findings: list[Finding] = []

        ext = filename.lower()

        if ext.endswith("requirements.txt") or ext.endswith(".txt"):
            findings.extend(self._scan_requirements(content))
        elif ext.endswith("package.json"):
            findings.extend(self._scan_package_json(content))
        elif ext.endswith("pyproject.toml") or ext.endswith("Pipfile"):
            findings.extend(self._scan_pyproject(content))
        else:
            # Try all parsers
            findings.extend(self._scan_requirements(content))
            findings.extend(self._scan_package_json(content))
            findings.extend(self._scan_pyproject(content))

        # Deduplicate findings by description + line
        seen = set()
        deduped = []
        for f in findings:
            key = (f.description, f.metadata.get("line"))
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        return deduped

    def _scan_requirements(self, content: str) -> list[Finding]:
        """Scan requirements.txt format."""
        findings: list[Finding] = []
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Extract package name (before any version specifier)
            pkg_match = re.match(r"^([a-zA-Z0-9_.-]+)", line)
            if not pkg_match:
                # Check for URL-based installs
                self._check_urls(line, line_num, findings)
                continue

            pkg_name = pkg_match.group(1).lower()

            # Check for known malicious packages
            if pkg_name in _KNOWN_MALICIOUS:
                findings.append(Finding(
                    scanner=self.name,
                    category="malicious_package",
                    description=f"Known malicious package: '{pkg_name}' — removed from registry for malware",
                    risk=RiskLevel.CRITICAL,
                    metadata={"line": line_num, "package": pkg_name},
                ))

            # Check for typosquatting
            self._check_typosquat(pkg_name, line_num, findings)

            # Check for URL-based installs
            self._check_urls(line, line_num, findings)

            # Check for unpinned versions
            if re.match(r"^[a-zA-Z0-9_.-]+\s*$", line):
                findings.append(Finding(
                    scanner=self.name,
                    category="unpinned_dependency",
                    description=f"Unpinned dependency: '{pkg_name}' — pin to specific version for reproducibility",
                    risk=RiskLevel.LOW,
                    metadata={"line": line_num, "package": pkg_name},
                ))
            elif re.search(r">=\s*\d|>\s*\d", line) and not re.search(r"[<,]", line):
                findings.append(Finding(
                    scanner=self.name,
                    category="loose_version",
                    description=f"Loosely pinned dependency: '{pkg_name}' — lower bound only, vulnerable to breaking changes",
                    risk=RiskLevel.LOW,
                    metadata={"line": line_num, "package": pkg_name},
                ))

        return findings

    def _scan_package_json(self, content: str) -> list[Finding]:
        """Scan package.json format."""
        import json as json_module

        findings: list[Finding] = []

        try:
            data = json_module.loads(content)
        except (json_module.JSONDecodeError, ValueError):
            return findings

        # Check dependencies and devDependencies
        for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            deps = data.get(section, {})
            if not isinstance(deps, dict):
                continue
            for pkg_name, version in deps.items():
                pkg_lower = pkg_name.lower()

                # Check for known malicious
                if pkg_lower in _KNOWN_MALICIOUS:
                    findings.append(Finding(
                        scanner=self.name,
                        category="malicious_package",
                        description=f"Known malicious package: '{pkg_name}' — removed from registry for malware",
                        risk=RiskLevel.CRITICAL,
                        metadata={"package": pkg_name, "section": section},
                    ))

                # Check typosquatting
                self._check_typosquat(pkg_lower, 0, findings)

                # Check URL-based versions
                if isinstance(version, str):
                    self._check_urls(version, 0, findings)

                    # Wildcard version
                    if version == "*" or version == "latest":
                        findings.append(Finding(
                            scanner=self.name,
                            category="unpinned_dependency",
                            description=f"Wildcard dependency: '{pkg_name}' version '{version}' — pin to specific version",
                            risk=RiskLevel.MEDIUM,
                            metadata={"package": pkg_name, "version": version, "section": section},
                        ))

        # Check install scripts
        scripts = data.get("scripts", {})
        if isinstance(scripts, dict):
            for script_name, script_cmd in scripts.items():
                if not isinstance(script_cmd, str):
                    continue
                if script_name in ("preinstall", "install", "postinstall"):
                    # Check for dangerous commands in install hooks
                    dangerous_cmds = re.search(
                        r'\b(?:curl|wget|nc|bash|sh|eval|exec)\b', script_cmd
                    )
                    if dangerous_cmds:
                        findings.append(Finding(
                            scanner=self.name,
                            category="dangerous_install_script",
                            description="Install script executes external commands — potential supply chain attack",
                            risk=RiskLevel.CRITICAL,
                            metadata={"script": script_name, "command": script_cmd[:200]},
                        ))
                    elif re.search(r'https?://|ftp://', script_cmd):
                        findings.append(Finding(
                            scanner=self.name,
                            category="dangerous_install_script",
                            description="Install script references external URL — review for supply chain safety",
                            risk=RiskLevel.HIGH,
                            metadata={"script": script_name, "command": script_cmd[:200]},
                        ))

        return findings

    def _scan_pyproject(self, content: str) -> list[Finding]:
        """Scan pyproject.toml/Pipfile format (basic pattern matching)."""
        findings: list[Finding] = []
        for line_num, line in enumerate(content.splitlines(), 1):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue

            # Look for package references in dependencies sections
            dep_match = re.match(r'^"?([a-zA-Z0-9_.-]+)"?\s*[=<>~!]', line_stripped)
            if dep_match:
                pkg_name = dep_match.group(1).lower()
                if pkg_name in _KNOWN_MALICIOUS:
                    findings.append(Finding(
                        scanner=self.name,
                        category="malicious_package",
                        description=f"Known malicious package: '{pkg_name}' — removed from registry for malware",
                        risk=RiskLevel.CRITICAL,
                        metadata={"line": line_num, "package": pkg_name},
                    ))
                self._check_typosquat(pkg_name, line_num, findings)

            # Check for URL-based installs
            self._check_urls(line, line_num, findings)

        return findings

    def _check_typosquat(self, pkg_name: str, line_num: int, findings: list[Finding]) -> None:
        """Check if a package name is a known typosquat."""
        for legit, squats in _TYPOSQUAT_TARGETS.items():
            if pkg_name in squats:
                findings.append(Finding(
                    scanner=self.name,
                    category="typosquat",
                    description=f"Possible typosquat: '{pkg_name}' looks like '{legit}' — verify package name",
                    risk=RiskLevel.CRITICAL,
                    metadata={"line": line_num, "package": pkg_name, "likely_intended": legit},
                ))
                return

    def _check_urls(self, line: str, line_num: int, findings: list[Finding]) -> None:
        """Check for suspicious URLs in dependency specifications."""
        for pattern, risk, desc in _SUSPICIOUS_URL_PATTERNS:
            for match in pattern.finditer(line):
                findings.append(Finding(
                    scanner=self.name,
                    category="suspicious_url",
                    description=desc,
                    risk=risk,
                    metadata={"line": line_num, "match": match.group(0)[:200]},
                ))
