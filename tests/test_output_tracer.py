"""Tests for output tracer."""

import time
import pytest
from sentinel.output_tracer import OutputTracer, Span, TraceTree, SpanSummary


class TestSpanContextManager:
    def test_basic_span_creates_and_completes(self):
        tracer = OutputTracer()
        with tracer.span("check") as ctx:
            span_id = ctx.span_id
        span = tracer.get_span(span_id)
        assert span.name == "check"
        assert span.is_complete
        assert span.status == "pass"

    def test_span_records_timing(self):
        tracer = OutputTracer()
        with tracer.span("slow_check") as ctx:
            time.sleep(0.01)
            span_id = ctx.span_id
        span = tracer.get_span(span_id)
        assert span.duration_ms >= 5

    def test_span_context_set_status(self):
        tracer = OutputTracer()
        with tracer.span("check") as ctx:
            ctx.set_status("fail")
            span_id = ctx.span_id
        span = tracer.get_span(span_id)
        assert span.status == "fail"

    def test_span_context_set_metadata(self):
        tracer = OutputTracer()
        with tracer.span("check") as ctx:
            ctx.set_metadata("scanner", "pii")
            ctx.set_metadata("score", 0.95)
            span_id = ctx.span_id
        span = tracer.get_span(span_id)
        assert span.metadata["scanner"] == "pii"
        assert span.metadata["score"] == 0.95


class TestNestedSpans:
    def test_parent_child_relationship(self):
        tracer = OutputTracer()
        with tracer.span("pipeline") as root:
            with tracer.span("scan_a", parent=root.span_id) as child:
                child_id = child.span_id
            root_id = root.span_id
        root_span = tracer.get_span(root_id)
        assert child_id in root_span.children

    def test_child_references_parent(self):
        tracer = OutputTracer()
        with tracer.span("pipeline") as root:
            with tracer.span("scan_a", parent=root.span_id) as child:
                child_id = child.span_id
            root_id = root.span_id
        child_span = tracer.get_span(child_id)
        assert child_span.parent == root_id

    def test_multiple_children(self):
        tracer = OutputTracer()
        with tracer.span("pipeline") as root:
            with tracer.span("scan_a", parent=root.span_id) as a:
                a_id = a.span_id
            with tracer.span("scan_b", parent=root.span_id) as b:
                b_id = b.span_id
            root_id = root.span_id
        root_span = tracer.get_span(root_id)
        assert len(root_span.children) == 2
        assert a_id in root_span.children
        assert b_id in root_span.children


class TestManualSpans:
    def test_start_and_end_span(self):
        tracer = OutputTracer()
        span_id = tracer.start_span("manual_check")
        tracer.end_span(span_id, status="pass")
        span = tracer.get_span(span_id)
        assert span.is_complete
        assert span.status == "pass"

    def test_end_unknown_span_raises(self):
        tracer = OutputTracer()
        with pytest.raises(KeyError):
            tracer.end_span("nonexistent")

    def test_attach_metadata_to_manual_span(self):
        tracer = OutputTracer()
        span_id = tracer.start_span("check")
        tracer.attach_metadata(span_id, "result", "clean")
        span = tracer.get_span(span_id)
        assert span.metadata["result"] == "clean"

    def test_attach_metadata_unknown_span_raises(self):
        tracer = OutputTracer()
        with pytest.raises(KeyError):
            tracer.attach_metadata("nonexistent", "k", "v")


class TestTraceTree:
    def test_build_tree_identifies_roots(self):
        tracer = OutputTracer()
        with tracer.span("root_a"):
            pass
        with tracer.span("root_b"):
            pass
        tree = tracer.build_tree()
        assert len(tree.root_spans) == 2

    def test_build_tree_contains_all_spans(self):
        tracer = OutputTracer()
        with tracer.span("pipeline") as root:
            with tracer.span("child", parent=root.span_id):
                pass
        tree = tracer.build_tree()
        assert len(tree.all_spans) == 2

    def test_build_tree_total_duration(self):
        tracer = OutputTracer()
        with tracer.span("pipeline"):
            time.sleep(0.01)
        tree = tracer.build_tree()
        assert tree.total_duration_ms >= 5
        assert isinstance(tree, TraceTree)


class TestExport:
    def test_export_flat_returns_summaries(self):
        tracer = OutputTracer()
        with tracer.span("check_a") as ctx:
            ctx.set_status("pass")
        with tracer.span("check_b") as ctx:
            ctx.set_status("fail")
        flat = tracer.export_flat()
        assert len(flat) == 2
        assert all(isinstance(s, SpanSummary) for s in flat)
        names = {s.name for s in flat}
        assert names == {"check_a", "check_b"}

    def test_export_nested_mirrors_tree(self):
        tracer = OutputTracer()
        with tracer.span("pipeline") as root:
            with tracer.span("scan", parent=root.span_id):
                pass
        nested = tracer.export_nested()
        assert len(nested) == 1
        assert nested[0]["name"] == "pipeline"
        assert len(nested[0]["children"]) == 1
        assert nested[0]["children"][0]["name"] == "scan"


class TestFindSpans:
    def test_find_by_name(self):
        tracer = OutputTracer()
        with tracer.span("pii_scan"):
            pass
        with tracer.span("injection_scan"):
            pass
        with tracer.span("pii_scan"):
            pass
        found = tracer.find_by_name("pii_scan")
        assert len(found) == 2

    def test_find_by_status(self):
        tracer = OutputTracer()
        with tracer.span("check_a") as ctx:
            ctx.set_status("fail")
        with tracer.span("check_b"):
            pass  # auto-pass
        failed = tracer.find_by_status("fail")
        assert len(failed) == 1
        passed = tracer.find_by_status("pass")
        assert len(passed) == 1


class TestEdgeCases:
    def test_get_unknown_span_returns_none(self):
        tracer = OutputTracer()
        assert tracer.get_span("nonexistent") is None

    def test_empty_tracer_tree(self):
        tracer = OutputTracer()
        tree = tracer.build_tree()
        assert tree.root_spans == []
        assert tree.total_duration_ms == 0.0

    def test_clear_removes_all_spans(self):
        tracer = OutputTracer()
        with tracer.span("a"):
            pass
        with tracer.span("b"):
            pass
        cleared = tracer.clear()
        assert cleared == 2
        assert tracer.span_count() == 0

    def test_incomplete_span_duration_is_zero(self):
        tracer = OutputTracer()
        span_id = tracer.start_span("pending")
        span = tracer.get_span(span_id)
        assert span.duration_ms == 0.0
        assert not span.is_complete
