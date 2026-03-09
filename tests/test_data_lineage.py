"""Tests for the data lineage tracker."""

import pytest
from sentinel.data_lineage import (
    DataLineage,
    LineageNode,
    LineageEdge,
    FlowRecord,
    LineageValidation,
)


# ---------------------------------------------------------------------------
# Helper to build a standard three-node pipeline: source -> transform -> model
# ---------------------------------------------------------------------------

def _build_simple_pipeline() -> DataLineage:
    lineage = DataLineage("test-pipeline")
    lineage.add_node("input", node_type="source")
    lineage.add_node("sanitize", node_type="transform")
    lineage.add_node("llm", node_type="model")
    lineage.add_edge("input", "sanitize")
    lineage.add_edge("sanitize", "llm")
    return lineage


# ---------------------------------------------------------------------------
# TestNodes
# ---------------------------------------------------------------------------

class TestNodes:
    def test_add_node(self):
        lineage = DataLineage()
        node = lineage.add_node("input", node_type="source")
        assert isinstance(node, LineageNode)
        assert node.name == "input"
        assert node.node_type == "source"

    def test_invalid_type(self):
        lineage = DataLineage()
        with pytest.raises(ValueError, match="Invalid node_type"):
            lineage.add_node("bad", node_type="unknown")

    def test_metadata(self):
        lineage = DataLineage()
        meta = {"version": 2, "owner": "team-safety"}
        node = lineage.add_node("src", node_type="source", metadata=meta)
        assert node.metadata == meta

    def test_duplicate_node(self):
        lineage = DataLineage()
        lineage.add_node("input", node_type="source")
        with pytest.raises(ValueError, match="already exists"):
            lineage.add_node("input", node_type="transform")


# ---------------------------------------------------------------------------
# TestEdges
# ---------------------------------------------------------------------------

class TestEdges:
    def test_add_edge(self):
        lineage = DataLineage()
        lineage.add_node("a", node_type="source")
        lineage.add_node("b", node_type="transform")
        edge = lineage.add_edge("a", "b", label="clean")
        assert isinstance(edge, LineageEdge)
        assert edge.from_node == "a"
        assert edge.to_node == "b"
        assert edge.label == "clean"

    def test_missing_node_raises(self):
        lineage = DataLineage()
        lineage.add_node("a", node_type="source")
        with pytest.raises(KeyError, match="does not exist"):
            lineage.add_edge("a", "nonexistent")
        with pytest.raises(KeyError, match="does not exist"):
            lineage.add_edge("nonexistent", "a")


# ---------------------------------------------------------------------------
# TestTrace (upstream)
# ---------------------------------------------------------------------------

class TestTrace:
    def test_trace_upstream(self):
        lineage = _build_simple_pipeline()
        ancestors = lineage.trace("llm")
        assert "sanitize" in ancestors
        assert "input" in ancestors

    def test_trace_from_source(self):
        lineage = _build_simple_pipeline()
        ancestors = lineage.trace("input")
        assert ancestors == []

    def test_trace_multi_path(self):
        lineage = DataLineage("multi")
        lineage.add_node("src1", node_type="source")
        lineage.add_node("src2", node_type="source")
        lineage.add_node("merge", node_type="transform")
        lineage.add_node("out", node_type="output")
        lineage.add_edge("src1", "merge")
        lineage.add_edge("src2", "merge")
        lineage.add_edge("merge", "out")
        ancestors = lineage.trace("out")
        assert "merge" in ancestors
        assert "src1" in ancestors
        assert "src2" in ancestors
        assert len(ancestors) == 3


# ---------------------------------------------------------------------------
# TestDownstream
# ---------------------------------------------------------------------------

class TestDownstream:
    def test_downstream(self):
        lineage = _build_simple_pipeline()
        dependents = lineage.downstream("input")
        assert "sanitize" in dependents
        assert "llm" in dependents

    def test_downstream_leaf(self):
        lineage = _build_simple_pipeline()
        dependents = lineage.downstream("llm")
        assert dependents == []


# ---------------------------------------------------------------------------
# TestFlow
# ---------------------------------------------------------------------------

class TestFlow:
    def test_record_flow(self):
        lineage = DataLineage()
        lineage.add_node("step", node_type="transform")
        record = lineage.record_flow("step", "filtered PII", record_count=10)
        assert isinstance(record, FlowRecord)
        assert record.node_name == "step"
        assert record.record_count == 10
        assert record.timestamp > 0

    def test_flow_history(self):
        lineage = DataLineage()
        lineage.add_node("step", node_type="transform")
        lineage.record_flow("step", "batch 1", record_count=5)
        lineage.record_flow("step", "batch 2", record_count=8)
        history = lineage.get_flow_history("step")
        assert len(history) == 2
        assert history[0].data_summary == "batch 1"
        assert history[1].data_summary == "batch 2"

    def test_flow_missing_node_raises(self):
        lineage = DataLineage()
        with pytest.raises(KeyError, match="does not exist"):
            lineage.record_flow("ghost", "data")


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_pipeline(self):
        lineage = _build_simple_pipeline()
        result = lineage.validate()
        assert isinstance(result, LineageValidation)
        assert result.valid is True
        assert result.orphan_nodes == []
        assert result.has_cycles is False

    def test_orphan_detection(self):
        lineage = DataLineage()
        lineage.add_node("connected_a", node_type="source")
        lineage.add_node("connected_b", node_type="transform")
        lineage.add_edge("connected_a", "connected_b")
        lineage.add_node("orphan", node_type="cache")
        result = lineage.validate()
        assert result.valid is False
        assert "orphan" in result.orphan_nodes
        assert any("Orphan" in w for w in result.warnings)

    def test_cycle_detection(self):
        lineage = DataLineage()
        lineage.add_node("a", node_type="transform")
        lineage.add_node("b", node_type="transform")
        lineage.add_node("c", node_type="transform")
        lineage.add_edge("a", "b")
        lineage.add_edge("b", "c")
        lineage.add_edge("c", "a")
        result = lineage.validate()
        assert result.valid is False
        assert result.has_cycles is True
        assert any("cycle" in w.lower() for w in result.warnings)

    def test_disconnected_subgraph_warning(self):
        lineage = DataLineage()
        lineage.add_node("a", node_type="source")
        lineage.add_node("b", node_type="transform")
        lineage.add_edge("a", "b")
        lineage.add_node("c", node_type="source")
        lineage.add_node("d", node_type="output")
        lineage.add_edge("c", "d")
        result = lineage.validate()
        assert any("disconnected" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# TestExport
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_dict(self):
        lineage = _build_simple_pipeline()
        lineage.record_flow("sanitize", "cleaned text", record_count=3)
        exported = lineage.export()
        assert exported["pipeline_name"] == "test-pipeline"
        assert len(exported["nodes"]) == 3
        assert len(exported["edges"]) == 2
        assert len(exported["flow_records"]) == 1
        assert exported["flow_records"][0]["record_count"] == 3

    def test_export_node_fields(self):
        lineage = DataLineage()
        lineage.add_node("src", node_type="source", metadata={"key": "val"})
        exported = lineage.export()
        node = exported["nodes"][0]
        assert node["name"] == "src"
        assert node["node_type"] == "source"
        assert node["metadata"] == {"key": "val"}


# ---------------------------------------------------------------------------
# TestSummary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_content(self):
        lineage = _build_simple_pipeline()
        text = lineage.summary()
        assert "test-pipeline" in text
        assert "Nodes: 3" in text
        assert "Edges: 2" in text


# ---------------------------------------------------------------------------
# TestClear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets(self):
        lineage = _build_simple_pipeline()
        lineage.record_flow("sanitize", "data", record_count=1)
        lineage.clear()
        exported = lineage.export()
        assert exported["nodes"] == []
        assert exported["edges"] == []
        assert exported["flow_records"] == []
