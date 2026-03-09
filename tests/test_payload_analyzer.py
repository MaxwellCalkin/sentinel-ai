"""Tests for payload analyzer."""

import json
import pytest
from sentinel.payload_analyzer import (
    PayloadAnalyzer,
    PayloadReport,
    PayloadThreat,
    PayloadField,
    AnalyzerConfig,
    AnalyzerStats,
)


# ---------------------------------------------------------------------------
# Clean payloads — should be safe
# ---------------------------------------------------------------------------

class TestCleanPayloads:
    def test_clean_dict_is_safe(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"name": "Alice", "age": 30})
        assert report.is_safe
        assert report.risk_score == 0.0
        assert len(report.threats) == 0
        assert report.total_fields == 2

    def test_empty_dict_is_safe(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({})
        assert report.is_safe
        assert report.total_fields == 0
        assert report.risk_score == 0.0

    def test_empty_list_is_safe(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze([])
        assert report.is_safe
        assert report.total_fields == 0

    def test_nested_clean_dict(self):
        payload = {"user": {"name": "Bob", "address": {"city": "NYC"}}}
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze(payload)
        assert report.is_safe
        assert report.max_depth == 3


# ---------------------------------------------------------------------------
# SQL injection detection
# ---------------------------------------------------------------------------

class TestSQLInjection:
    def test_drop_table_detected(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"query": "DROP TABLE users"})
        assert any(t.threat_type == "nested_injection" for t in report.threats)
        assert report.risk_score > 0

    def test_select_star_detected(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"input": "SELECT * FROM secrets"})
        threats = [t for t in report.threats if t.threat_type == "nested_injection"]
        assert len(threats) >= 1

    def test_or_one_equals_one_detected(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"field": "admin' OR 1=1 --"})
        assert any(t.threat_type == "nested_injection" for t in report.threats)

    def test_union_select_detected(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"q": "1 UNION SELECT password FROM users"})
        assert any(t.threat_type == "nested_injection" for t in report.threats)


# ---------------------------------------------------------------------------
# Command injection detection
# ---------------------------------------------------------------------------

class TestCommandInjection:
    def test_semicolon_rm_detected(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"cmd": "; rm -rf /"})
        assert any(t.threat_type == "nested_injection" for t in report.threats)

    def test_pipe_cat_detected(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"input": "| cat /etc/passwd"})
        assert any(t.threat_type == "nested_injection" for t in report.threats)

    def test_script_tag_detected(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"html": "<script>alert('xss')</script>"})
        assert any(t.threat_type == "nested_injection" for t in report.threats)


# ---------------------------------------------------------------------------
# Deep nesting detection
# ---------------------------------------------------------------------------

class TestDeepNesting:
    def test_deep_nesting_detected(self):
        payload = {"a": {}}
        current = payload["a"]
        for i in range(12):
            current["nested"] = {}
            current = current["nested"]
        current["value"] = "deep"

        analyzer = PayloadAnalyzer(AnalyzerConfig(max_depth=10))
        report = analyzer.analyze(payload)
        assert any(t.threat_type == "deep_nesting" for t in report.threats)

    def test_shallow_nesting_is_safe(self):
        payload = {"a": {"b": {"c": "hello"}}}
        analyzer = PayloadAnalyzer(AnalyzerConfig(max_depth=10))
        report = analyzer.analyze(payload)
        assert not any(t.threat_type == "deep_nesting" for t in report.threats)


# ---------------------------------------------------------------------------
# Oversized field detection
# ---------------------------------------------------------------------------

class TestOversizedField:
    def test_oversized_string_detected(self):
        analyzer = PayloadAnalyzer(AnalyzerConfig(max_field_size=100))
        report = analyzer.analyze({"data": "x" * 200})
        assert any(t.threat_type == "oversized_field" for t in report.threats)

    def test_normal_string_is_safe(self):
        analyzer = PayloadAnalyzer(AnalyzerConfig(max_field_size=100))
        report = analyzer.analyze({"data": "short"})
        assert not any(t.threat_type == "oversized_field" for t in report.threats)


# ---------------------------------------------------------------------------
# Type confusion detection
# ---------------------------------------------------------------------------

class TestTypeConfusion:
    def test_function_prefix_detected(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"callback": "function() { evil(); }"})
        assert any(t.threat_type == "type_confusion" for t in report.threats)

    def test_eval_prefix_detected(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"code": "eval(something)"})
        assert any(t.threat_type == "type_confusion" for t in report.threats)

    def test_import_prefix_detected(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"module": "__import__('os').system('ls')"})
        assert any(t.threat_type == "type_confusion" for t in report.threats)

    def test_normal_string_no_type_confusion(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"text": "just a regular sentence"})
        assert not any(t.threat_type == "type_confusion" for t in report.threats)


# ---------------------------------------------------------------------------
# JSON string input parsing
# ---------------------------------------------------------------------------

class TestJSONStringInput:
    def test_valid_json_string_parsed(self):
        payload_str = json.dumps({"name": "test", "value": 42})
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze(payload_str)
        assert report.is_safe
        assert report.total_fields == 2

    def test_invalid_json_string_analyzed_as_raw(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze("not valid json {{{")
        assert report.total_fields == 1
        field = report.fields_analyzed[0]
        assert field.value_type == "str"

    def test_json_string_with_threats(self):
        payload_str = json.dumps({"q": "DROP TABLE users"})
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze(payload_str)
        assert len(report.threats) > 0
        assert report.risk_score > 0


# ---------------------------------------------------------------------------
# List payload analysis
# ---------------------------------------------------------------------------

class TestListPayload:
    def test_list_of_strings(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze(["hello", "world"])
        assert report.is_safe
        assert report.total_fields == 2

    def test_list_with_dicts(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze([{"a": 1}, {"b": 2}])
        assert report.total_fields == 2

    def test_list_path_notation(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze(["first", "second"])
        paths = [f.path for f in report.fields_analyzed]
        assert "root[0]" in paths
        assert "root[1]" in paths


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

class TestBatchAnalysis:
    def test_batch_returns_list(self):
        analyzer = PayloadAnalyzer()
        reports = analyzer.analyze_batch([
            {"a": 1},
            {"b": "eval(DROP TABLE x)"},
        ])
        assert len(reports) == 2
        assert reports[0].is_safe
        assert not reports[1].is_safe  # 2+ threats (injection + type confusion) >= 0.3

    def test_batch_empty_list(self):
        analyzer = PayloadAnalyzer()
        reports = analyzer.analyze_batch([])
        assert reports == []


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStatsTracking:
    def test_stats_count_analyses(self):
        analyzer = PayloadAnalyzer()
        analyzer.analyze({"a": 1})
        analyzer.analyze({"b": 2})
        assert analyzer.stats().total_analyzed == 2

    def test_stats_count_threats(self):
        analyzer = PayloadAnalyzer()
        analyzer.analyze({"q": "DROP TABLE users"})
        analyzer.analyze({"ok": "fine"})
        assert analyzer.stats().threats_found >= 1

    def test_stats_by_threat_type(self):
        analyzer = PayloadAnalyzer()
        analyzer.analyze({"q": "DROP TABLE users"})
        by_type = analyzer.stats().by_threat_type
        assert "nested_injection" in by_type
        assert by_type["nested_injection"] >= 1

    def test_stats_initial_state(self):
        analyzer = PayloadAnalyzer()
        s = analyzer.stats()
        assert s.total_analyzed == 0
        assert s.threats_found == 0
        assert s.by_threat_type == {}


# ---------------------------------------------------------------------------
# Config customization
# ---------------------------------------------------------------------------

class TestConfigCustomization:
    def test_custom_max_depth(self):
        config = AnalyzerConfig(max_depth=2)
        analyzer = PayloadAnalyzer(config)
        payload = {"a": {"b": {"c": {"d": "deep"}}}}
        report = analyzer.analyze(payload)
        assert any(t.threat_type == "deep_nesting" for t in report.threats)

    def test_custom_max_field_size(self):
        config = AnalyzerConfig(max_field_size=5)
        analyzer = PayloadAnalyzer(config)
        report = analyzer.analyze({"val": "abcdefghij"})
        assert any(t.threat_type == "oversized_field" for t in report.threats)

    def test_injections_disabled(self):
        config = AnalyzerConfig(check_injections=False)
        analyzer = PayloadAnalyzer(config)
        report = analyzer.analyze({"q": "DROP TABLE users"})
        assert not any(t.threat_type == "nested_injection" for t in report.threats)


# ---------------------------------------------------------------------------
# Many fields threshold
# ---------------------------------------------------------------------------

class TestManyFields:
    def test_exceeding_max_fields_creates_threat(self):
        config = AnalyzerConfig(max_fields=5)
        analyzer = PayloadAnalyzer(config)
        payload = {f"field_{i}": i for i in range(10)}
        report = analyzer.analyze(payload)
        oversized = [t for t in report.threats if "fields" in t.description]
        assert len(oversized) >= 1


# ---------------------------------------------------------------------------
# Risk score calculation
# ---------------------------------------------------------------------------

class TestRiskScore:
    def test_zero_threats_zero_score(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"clean": "data"})
        assert report.risk_score == 0.0

    def test_one_threat_gives_015(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"q": "SELECT * FROM users"})
        assert report.risk_score == pytest.approx(0.15)

    def test_two_threats_gives_03(self):
        analyzer = PayloadAnalyzer()
        # eval( triggers type_confusion, and also check for injection
        report = analyzer.analyze({"a": "eval(DROP TABLE x)"})
        # Should have at least 2 threats (injection + type confusion)
        assert report.risk_score >= 0.3

    def test_risk_score_capped_at_one(self):
        analyzer = PayloadAnalyzer(AnalyzerConfig(max_field_size=5))
        # Many threatening fields to exceed 1.0
        payload = {f"f{i}": "DROP TABLE x; rm -rf /; eval(bad)" for i in range(10)}
        report = analyzer.analyze(payload)
        assert report.risk_score <= 1.0

    def test_is_safe_when_below_threshold(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"q": "SELECT * FROM users"})
        # 1 threat = 0.15 risk_score < 0.3 threshold
        assert report.is_safe

    def test_not_safe_when_above_threshold(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"a": "eval(DROP TABLE x)"})
        # 2+ threats >= 0.3
        assert not report.is_safe


# ---------------------------------------------------------------------------
# Path notation
# ---------------------------------------------------------------------------

class TestPathNotation:
    def test_dict_path_uses_dot_notation(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"user": {"name": "Alice"}})
        paths = [f.path for f in report.fields_analyzed]
        assert "root.user.name" in paths

    def test_list_path_uses_bracket_notation(self):
        analyzer = PayloadAnalyzer()
        report = analyzer.analyze({"items": ["a", "b"]})
        paths = [f.path for f in report.fields_analyzed]
        assert "root.items[0]" in paths
        assert "root.items[1]" in paths


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_payload_field_defaults(self):
        f = PayloadField(path="root.x", value_type="str", size=5, depth=1)
        assert not f.is_suspicious
        assert f.reason == ""

    def test_payload_threat_fields(self):
        t = PayloadThreat(
            threat_type="nested_injection",
            path="root.q",
            severity="high",
            description="SQL injection",
        )
        assert t.threat_type == "nested_injection"
        assert t.severity == "high"

    def test_analyzer_config_defaults(self):
        c = AnalyzerConfig()
        assert c.max_depth == 10
        assert c.max_field_size == 10000
        assert c.max_fields == 1000
        assert c.check_injections is True

    def test_analyzer_stats_defaults(self):
        s = AnalyzerStats()
        assert s.total_analyzed == 0
        assert s.threats_found == 0
        assert s.by_threat_type == {}
