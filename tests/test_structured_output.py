"""Tests for structured output validation scanner."""

import json
import pytest
from sentinel.scanners.structured_output import StructuredOutputScanner
from sentinel.core import RiskLevel


@pytest.fixture
def scanner():
    return StructuredOutputScanner()


@pytest.fixture
def schema_scanner():
    return StructuredOutputScanner(schema={
        "name": {"type": "string", "required": True},
        "age": {"type": "integer", "min": 0, "max": 150},
        "role": {"type": "string", "enum": ["admin", "user", "guest"]},
    })


class TestInjectionDetection:
    def test_clean_json(self, scanner):
        text = json.dumps({"name": "Alice", "score": 95})
        findings = scanner.scan(text)
        assert len(findings) == 0

    def test_xss_in_value(self, scanner):
        text = json.dumps({"bio": "<script>alert('xss')</script>"})
        findings = scanner.scan(text)
        assert any(f.metadata.get("injection_type") == "xss" for f in findings)
        assert any(f.risk >= RiskLevel.HIGH for f in findings)

    def test_javascript_uri(self, scanner):
        text = json.dumps({"url": "javascript:alert(1)"})
        findings = scanner.scan(text)
        assert any(f.metadata.get("injection_type") == "xss" for f in findings)

    def test_event_handler_injection(self, scanner):
        text = json.dumps({"name": 'a" onclick="alert(1)'})
        findings = scanner.scan(text)
        assert any(f.metadata.get("injection_type") == "xss" for f in findings)

    def test_sql_injection(self, scanner):
        text = json.dumps({"query": "'; DROP TABLE users; --"})
        findings = scanner.scan(text)
        assert any(
            f.metadata.get("injection_type") == "sql_injection"
            for f in findings
        )

    def test_sql_boolean_injection(self, scanner):
        text = json.dumps({"filter": "' OR '1'='1"})
        findings = scanner.scan(text)
        assert any(
            f.metadata.get("injection_type") == "sql_injection"
            for f in findings
        )

    def test_template_injection_jinja(self, scanner):
        text = json.dumps({"name": "{{config.SECRET_KEY}}"})
        findings = scanner.scan(text)
        assert any(
            f.metadata.get("injection_type") == "template_injection"
            for f in findings
        )

    def test_template_injection_dollar(self, scanner):
        text = json.dumps({"cmd": "${7*7}"})
        findings = scanner.scan(text)
        assert any(
            f.metadata.get("injection_type") == "template_injection"
            for f in findings
        )

    def test_path_traversal(self, scanner):
        text = json.dumps({"file": "../../etc/passwd"})
        findings = scanner.scan(text)
        assert any(
            f.metadata.get("injection_type") == "path_traversal"
            for f in findings
        )

    def test_nested_injection(self, scanner):
        text = json.dumps({
            "data": {
                "items": [
                    {"name": "safe"},
                    {"name": "<script>evil()</script>"},
                ]
            }
        })
        findings = scanner.scan(text)
        assert any(f.metadata.get("injection_type") == "xss" for f in findings)
        xss = next(
            f for f in findings if f.metadata.get("injection_type") == "xss"
        )
        assert "items" in xss.metadata["path"]


class TestSchemaValidation:
    def test_valid_data(self, schema_scanner):
        text = json.dumps({"name": "Alice", "age": 25, "role": "user"})
        findings = schema_scanner.scan(text)
        assert len(findings) == 0

    def test_missing_required_field(self, schema_scanner):
        text = json.dumps({"age": 25})
        findings = schema_scanner.scan(text)
        assert any(
            f.metadata.get("issue") == "missing_required_field"
            for f in findings
        )

    def test_type_mismatch(self, schema_scanner):
        text = json.dumps({"name": 123, "age": 25})
        findings = schema_scanner.scan(text)
        assert any(
            f.metadata.get("issue") == "type_mismatch" for f in findings
        )

    def test_out_of_range_low(self, schema_scanner):
        text = json.dumps({"name": "Bob", "age": -5})
        findings = schema_scanner.scan(text)
        assert any(
            f.metadata.get("issue") == "out_of_range" for f in findings
        )

    def test_out_of_range_high(self, schema_scanner):
        text = json.dumps({"name": "Bob", "age": 200})
        findings = schema_scanner.scan(text)
        assert any(
            f.metadata.get("issue") == "out_of_range" for f in findings
        )

    def test_invalid_enum(self, schema_scanner):
        text = json.dumps({"name": "Eve", "role": "superadmin"})
        findings = schema_scanner.scan(text)
        assert any(
            f.metadata.get("issue") == "invalid_enum" for f in findings
        )

    def test_valid_enum(self, schema_scanner):
        text = json.dumps({"name": "Eve", "role": "admin"})
        findings = schema_scanner.scan(text)
        enum_issues = [
            f for f in findings if f.metadata.get("issue") == "invalid_enum"
        ]
        assert len(enum_issues) == 0

    def test_extra_fields_allowed(self, schema_scanner):
        text = json.dumps({"name": "Alice", "extra": "data"})
        findings = schema_scanner.scan(text)
        assert not any(
            f.metadata.get("issue") == "unexpected_fields" for f in findings
        )

    def test_extra_fields_disallowed(self):
        scanner = StructuredOutputScanner(
            schema={"name": {"type": "string"}},
            allow_extra_fields=False,
        )
        text = json.dumps({"name": "Alice", "secret": "hidden"})
        findings = scanner.scan(text)
        assert any(
            f.metadata.get("issue") == "unexpected_fields" for f in findings
        )

    def test_boolean_not_integer(self, schema_scanner):
        """Python bool is subclass of int — should still fail type check."""
        text = json.dumps({"name": "Test", "age": True})
        findings = schema_scanner.scan(text)
        assert any(
            f.metadata.get("issue") == "type_mismatch"
            and f.metadata.get("field") == "age"
            for f in findings
        )


class TestStructuralChecks:
    def test_oversized_string(self):
        scanner = StructuredOutputScanner(max_string_length=100)
        text = json.dumps({"data": "x" * 200})
        findings = scanner.scan(text)
        assert any(
            f.metadata.get("issue") == "oversized_string" for f in findings
        )

    def test_excessive_nesting(self):
        scanner = StructuredOutputScanner(max_depth=3)
        nested = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        text = json.dumps(nested)
        findings = scanner.scan(text)
        assert any(
            f.metadata.get("issue") == "max_depth_exceeded" for f in findings
        )

    def test_non_json_text(self, scanner):
        findings = scanner.scan("this is not json")
        # Should scan as raw text, no crash
        assert isinstance(findings, list)

    def test_non_json_with_injection(self, scanner):
        findings = scanner.scan("<script>alert(1)</script>")
        assert any(f.metadata.get("injection_type") == "xss" for f in findings)

    def test_scan_dict_directly(self, scanner):
        data = {"name": "<script>xss</script>"}
        findings = scanner.scan_dict(data)
        assert any(f.metadata.get("injection_type") == "xss" for f in findings)

    def test_array_root(self, scanner):
        text = json.dumps([{"name": "safe"}, {"name": "<script>x</script>"}])
        findings = scanner.scan(text)
        assert any(f.metadata.get("injection_type") == "xss" for f in findings)

    def test_empty_json(self, scanner):
        findings = scanner.scan("{}")
        assert len(findings) == 0

    def test_null_values(self, scanner):
        text = json.dumps({"name": None, "items": [None]})
        findings = scanner.scan(text)
        assert len(findings) == 0
