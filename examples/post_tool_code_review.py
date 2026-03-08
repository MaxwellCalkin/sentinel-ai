"""Example: PostToolUse hook for real-time code review.

Scans every Write/Edit tool output for OWASP vulnerabilities and
every package manifest update for supply chain attacks.

Setup — add to .claude/settings.json:
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [{"type": "command", "command": "python post_tool_code_review.py"}]
      }
    ]
  }
}

The hook reads the tool output from stdin and:
  - Runs CodeScanner for OWASP vulnerabilities
  - Runs DependencyScanner if the file is a package manifest
  - Prints warnings for medium-risk findings
  - Exits 2 (block) for high/critical findings
"""

import json
import sys

from sentinel.scanners.code_scanner import CodeScanner
from sentinel.scanners.dependency_scanner import DependencyScanner
from sentinel.core import RiskLevel


def main() -> int:
    try:
        raw = sys.stdin.read()
    except Exception:
        return 0

    if not raw.strip():
        return 0

    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return 0

    tool_name = event.get("tool_name", "")
    tool_input = event.get("tool_input", {})

    # Only scan Write/Edit outputs
    if tool_name not in ("Write", "Edit", "write", "edit"):
        return 0

    file_path = tool_input.get("file_path", tool_input.get("path", ""))
    content = tool_input.get("content", tool_input.get("new_string", ""))

    if not content:
        return 0

    all_findings = []

    # --- Code vulnerability scan ---
    code_scanner = CodeScanner()
    code_findings = code_scanner.scan(content, filename=file_path)
    all_findings.extend(code_findings)

    # --- Dependency scan (if it's a manifest file) ---
    manifest_files = {
        "requirements.txt", "package.json", "pyproject.toml",
        "Pipfile", "setup.py", "setup.cfg",
    }
    if any(file_path.endswith(m) for m in manifest_files):
        dep_scanner = DependencyScanner()
        dep_findings = dep_scanner.scan(content, filename=file_path)
        all_findings.extend(dep_findings)

    # --- Report findings ---
    if not all_findings:
        return 0

    critical_high = [f for f in all_findings if f.risk >= RiskLevel.HIGH]
    medium = [f for f in all_findings if f.risk == RiskLevel.MEDIUM]

    # Print warnings for medium findings
    for f in medium:
        print(f"  WARNING: {f.description}")

    # Block on high/critical findings
    if critical_high:
        print(f"\nSentinel AI: blocked {tool_name} — {len(critical_high)} security issue(s) found:\n")
        for f in critical_high:
            print(f"  [{f.risk.value.upper()}] {f.description}")
        print(f"\n  File: {file_path}")
        print("  Fix the issues above before writing this code.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
