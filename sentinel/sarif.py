"""SARIF (Static Analysis Results Interchange Format) output for Sentinel AI.

Generates SARIF v2.1.0 output compatible with GitHub Code Scanning,
Azure DevOps, and other SARIF-consuming tools.
"""

from __future__ import annotations

import json
from typing import Any

from sentinel.core import Finding, RiskLevel, ScanResult

SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json"
SARIF_VERSION = "2.1.0"

_RISK_TO_SARIF_LEVEL = {
    RiskLevel.NONE: "none",
    RiskLevel.LOW: "note",
    RiskLevel.MEDIUM: "warning",
    RiskLevel.HIGH: "error",
    RiskLevel.CRITICAL: "error",
}

_SCANNER_HELP_URIS = {
    "prompt_injection": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    "pii": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    "harmful_content": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    "toxicity": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    "tool_use": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    "code_scanner": "https://owasp.org/www-project-top-10/",
    "structured_output": "https://owasp.org/www-project-top-10/",
    "obfuscation": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
}


def _make_rule_id(finding: Finding) -> str:
    """Generate a stable rule ID from scanner and category."""
    return f"sentinel/{finding.scanner}/{finding.category}"


def _make_rule(finding: Finding) -> dict[str, Any]:
    """Create a SARIF rule object from a Finding."""
    rule_id = _make_rule_id(finding)
    rule: dict[str, Any] = {
        "id": rule_id,
        "name": finding.category.replace("_", " ").title().replace(" ", ""),
        "shortDescription": {"text": f"{finding.scanner}: {finding.category}"},
        "defaultConfiguration": {
            "level": _RISK_TO_SARIF_LEVEL.get(finding.risk, "warning"),
        },
    }
    help_uri = _SCANNER_HELP_URIS.get(finding.scanner)
    if help_uri:
        rule["helpUri"] = help_uri
    return rule


def _make_result(
    finding: Finding,
    rule_index: int,
    artifact_uri: str = "input",
    line: int | None = None,
) -> dict[str, Any]:
    """Create a SARIF result object from a Finding."""
    result: dict[str, Any] = {
        "ruleId": _make_rule_id(finding),
        "ruleIndex": rule_index,
        "level": _RISK_TO_SARIF_LEVEL.get(finding.risk, "warning"),
        "message": {"text": finding.description},
    }

    location: dict[str, Any] = {
        "physicalLocation": {
            "artifactLocation": {"uri": artifact_uri},
        }
    }

    effective_line = finding.metadata.get("line", line)
    if effective_line is not None:
        region: dict[str, Any] = {"startLine": int(effective_line)}
        if finding.span:
            region["charOffset"] = finding.span[0]
            region["charLength"] = finding.span[1] - finding.span[0]
        location["physicalLocation"]["region"] = region
    elif finding.span:
        location["physicalLocation"]["region"] = {
            "charOffset": finding.span[0],
            "charLength": finding.span[1] - finding.span[0],
        }

    result["locations"] = [location]

    if finding.metadata:
        props = {}
        for k, v in finding.metadata.items():
            if k != "line":
                props[k] = v
        if props:
            result["properties"] = props

    return result


def findings_to_sarif(
    findings: list[Finding],
    artifact_uri: str = "input",
    version: str | None = None,
) -> dict[str, Any]:
    """Convert a list of Findings to a SARIF log.

    Args:
        findings: List of Finding objects.
        artifact_uri: URI of the scanned artifact.
        version: Sentinel AI version string.

    Returns:
        SARIF v2.1.0 log as a dictionary.
    """
    if version is None:
        try:
            from sentinel import __version__
            version = __version__
        except ImportError:
            version = "unknown"

    seen_rules: dict[str, int] = {}
    rules: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for finding in findings:
        rule_id = _make_rule_id(finding)
        if rule_id not in seen_rules:
            seen_rules[rule_id] = len(rules)
            rules.append(_make_rule(finding))

        rule_index = seen_rules[rule_id]
        results.append(_make_result(finding, rule_index, artifact_uri))

    return {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Sentinel AI",
                        "version": version,
                        "informationUri": "https://github.com/MaxwellCalkin/sentinel-ai",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }


def scan_result_to_sarif(
    result: ScanResult,
    artifact_uri: str = "input",
    version: str | None = None,
) -> dict[str, Any]:
    """Convert a ScanResult to a SARIF log.

    Args:
        result: ScanResult from SentinelGuard.scan().
        artifact_uri: URI of the scanned artifact.
        version: Sentinel AI version string.

    Returns:
        SARIF v2.1.0 log as a dictionary.
    """
    sarif = findings_to_sarif(result.findings, artifact_uri, version)
    run = sarif["runs"][0]
    run["properties"] = {
        "sentinel:blocked": result.blocked,
        "sentinel:risk": result.risk.value,
        "sentinel:latencyMs": result.latency_ms,
    }
    if result.redacted_text:
        run["properties"]["sentinel:redactedText"] = result.redacted_text
    return sarif


def sarif_to_json(sarif: dict[str, Any], indent: int = 2) -> str:
    """Serialize a SARIF log to JSON string."""
    return json.dumps(sarif, indent=indent)
