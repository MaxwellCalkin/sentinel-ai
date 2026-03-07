"""Structured output validation scanner.

Validates LLM-generated JSON/structured outputs for safety issues:
- Injection attacks hidden in field values
- Schema violations and unexpected fields
- Sensitive data in outputs that should be clean

Usage:
    from sentinel.scanners.structured_output import StructuredOutputScanner

    scanner = StructuredOutputScanner(schema={
        "name": {"type": "string"},
        "age": {"type": "integer", "min": 0, "max": 150},
    })
    findings = scanner.scan('{"name": "John", "age": 25}')
"""

from __future__ import annotations

import json
import re
from typing import Any

from sentinel.core import Finding, RiskLevel


# Patterns for injection hidden in JSON values
_VALUE_INJECTION_PATTERNS = [
    (
        re.compile(r"(?i)<script[^>]*>"),
        "XSS script tag in output value",
        RiskLevel.CRITICAL,
        "xss",
    ),
    (
        re.compile(r"(?i)javascript\s*:"),
        "JavaScript URI in output value",
        RiskLevel.HIGH,
        "xss",
    ),
    (
        re.compile(r"(?i)on(click|load|error|mouseover)\s*="),
        "HTML event handler in output value",
        RiskLevel.HIGH,
        "xss",
    ),
    (
        re.compile(
            r"(?i)(;\s*(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE)\s+)"
        ),
        "SQL injection in output value",
        RiskLevel.CRITICAL,
        "sql_injection",
    ),
    (
        re.compile(r"(?i)('\s*(OR|AND)\s+'\w+'\s*=\s*'\w+)"),
        "SQL injection boolean condition in output value",
        RiskLevel.HIGH,
        "sql_injection",
    ),
    (
        re.compile(r"\{\{.*\}\}|\$\{.*\}|<%.*%>"),
        "Template injection in output value",
        RiskLevel.HIGH,
        "template_injection",
    ),
    (
        re.compile(r"(?i)\\u0000|%00|\x00"),
        "Null byte injection in output value",
        RiskLevel.HIGH,
        "null_byte",
    ),
    (
        re.compile(r"\.\./\.\./|\.\.\\\.\.\\"),
        "Path traversal in output value",
        RiskLevel.HIGH,
        "path_traversal",
    ),
]


class StructuredOutputScanner:
    """Validate structured (JSON) LLM outputs for safety issues.

    Checks for:
    - Injection attacks hidden in string field values
    - Schema violations (wrong types, out-of-range values, unexpected fields)
    - Oversized outputs that could indicate data exfiltration
    """

    name = "structured_output"

    def __init__(
        self,
        schema: dict[str, Any] | None = None,
        max_string_length: int = 10000,
        max_depth: int = 20,
        allow_extra_fields: bool = True,
    ):
        self._schema = schema
        self._max_string_length = max_string_length
        self._max_depth = max_depth
        self._allow_extra_fields = allow_extra_fields

    def scan(self, text: str, context: dict | None = None) -> list[Finding]:
        """Scan text as structured output. Attempts JSON parse first."""
        findings: list[Finding] = []

        # Try to parse as JSON
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # Not JSON — scan as raw text for injection patterns
            findings.extend(self._scan_string_value(text, "raw_text"))
            return findings

        # Validate structure
        findings.extend(self._scan_value(data, "$", 0))

        # Validate against schema if provided
        if self._schema and isinstance(data, dict):
            findings.extend(self._validate_schema(data))

        return findings

    def scan_dict(self, data: dict, context: dict | None = None) -> list[Finding]:
        """Scan a pre-parsed dict/object directly."""
        findings = list(self._scan_value(data, "$", 0))
        if self._schema:
            findings.extend(self._validate_schema(data))
        return findings

    def _scan_value(
        self, value: Any, path: str, depth: int
    ) -> list[Finding]:
        """Recursively scan a parsed JSON value."""
        findings: list[Finding] = []

        if depth > self._max_depth:
            findings.append(Finding(
                scanner=self.name,
                category="structured_output",
                description=f"Excessive nesting depth ({depth}) at {path}",
                risk=RiskLevel.MEDIUM,
                metadata={"path": path, "issue": "max_depth_exceeded"},
            ))
            return findings

        if isinstance(value, str):
            if len(value) > self._max_string_length:
                findings.append(Finding(
                    scanner=self.name,
                    category="structured_output",
                    description=(
                        f"Oversized string ({len(value)} chars) at {path}"
                    ),
                    risk=RiskLevel.MEDIUM,
                    metadata={
                        "path": path,
                        "issue": "oversized_string",
                        "length": len(value),
                    },
                ))
            findings.extend(self._scan_string_value(value, path))

        elif isinstance(value, dict):
            for key, val in value.items():
                child_path = f"{path}.{key}"
                # Check key itself for injection
                findings.extend(self._scan_string_value(key, f"{path}[key]"))
                findings.extend(self._scan_value(val, child_path, depth + 1))

        elif isinstance(value, list):
            for i, item in enumerate(value):
                findings.extend(
                    self._scan_value(item, f"{path}[{i}]", depth + 1)
                )

        return findings

    def _scan_string_value(
        self, value: str, path: str
    ) -> list[Finding]:
        """Check a string value for injection patterns."""
        findings: list[Finding] = []
        for pattern, description, risk, injection_type in _VALUE_INJECTION_PATTERNS:
            match = pattern.search(value)
            if match:
                findings.append(Finding(
                    scanner=self.name,
                    category="structured_output",
                    description=f"{description} at {path}",
                    risk=risk,
                    span=(match.start(), match.end()),
                    metadata={
                        "path": path,
                        "issue": "injection",
                        "injection_type": injection_type,
                        "matched": match.group(),
                    },
                ))
        return findings

    def _validate_schema(self, data: dict) -> list[Finding]:
        """Validate data against the provided schema."""
        findings: list[Finding] = []

        if not self._allow_extra_fields:
            extra = set(data.keys()) - set(self._schema.keys())
            if extra:
                findings.append(Finding(
                    scanner=self.name,
                    category="structured_output",
                    description=(
                        f"Unexpected fields in output: {', '.join(sorted(extra))}"
                    ),
                    risk=RiskLevel.LOW,
                    metadata={
                        "issue": "unexpected_fields",
                        "fields": sorted(extra),
                    },
                ))

        for field_name, constraints in self._schema.items():
            if field_name not in data:
                if constraints.get("required", False):
                    findings.append(Finding(
                        scanner=self.name,
                        category="structured_output",
                        description=f"Required field missing: {field_name}",
                        risk=RiskLevel.MEDIUM,
                        metadata={
                            "issue": "missing_required_field",
                            "field": field_name,
                        },
                    ))
                continue

            value = data[field_name]
            expected_type = constraints.get("type")

            if expected_type:
                type_ok = self._check_type(value, expected_type)
                if not type_ok:
                    findings.append(Finding(
                        scanner=self.name,
                        category="structured_output",
                        description=(
                            f"Type mismatch for '{field_name}': "
                            f"expected {expected_type}, got {type(value).__name__}"
                        ),
                        risk=RiskLevel.MEDIUM,
                        metadata={
                            "issue": "type_mismatch",
                            "field": field_name,
                            "expected": expected_type,
                            "actual": type(value).__name__,
                        },
                    ))

            # Range checks for numbers
            if isinstance(value, (int, float)):
                min_val = constraints.get("min")
                max_val = constraints.get("max")
                if min_val is not None and value < min_val:
                    findings.append(Finding(
                        scanner=self.name,
                        category="structured_output",
                        description=(
                            f"Value out of range for '{field_name}': "
                            f"{value} < min {min_val}"
                        ),
                        risk=RiskLevel.LOW,
                        metadata={
                            "issue": "out_of_range",
                            "field": field_name,
                        },
                    ))
                if max_val is not None and value > max_val:
                    findings.append(Finding(
                        scanner=self.name,
                        category="structured_output",
                        description=(
                            f"Value out of range for '{field_name}': "
                            f"{value} > max {max_val}"
                        ),
                        risk=RiskLevel.LOW,
                        metadata={
                            "issue": "out_of_range",
                            "field": field_name,
                        },
                    ))

            # Enum checks
            allowed = constraints.get("enum")
            if allowed is not None and value not in allowed:
                findings.append(Finding(
                    scanner=self.name,
                    category="structured_output",
                    description=(
                        f"Invalid value for '{field_name}': "
                        f"'{value}' not in {allowed}"
                    ),
                    risk=RiskLevel.MEDIUM,
                    metadata={
                        "issue": "invalid_enum",
                        "field": field_name,
                        "allowed": allowed,
                    },
                ))

        return findings

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected_types = type_map.get(expected)
        if expected_types is None:
            return True
        # bool is subclass of int in Python — handle explicitly
        if expected == "integer" and isinstance(value, bool):
            return False
        return isinstance(value, expected_types)
