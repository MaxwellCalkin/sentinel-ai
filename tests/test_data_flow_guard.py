"""Tests for DataFlowGuard — data flow validation for LLM pipelines."""

from sentinel.data_flow_guard import (
    DataFlowGuard,
    FlowAnalysis,
    FlowConnection,
    FlowNode,
    FlowStats,
    FlowViolation,
)


# ------------------------------------------------------------------
# Helper factories
# ------------------------------------------------------------------


def _source(name: str, sensitivity: str = "public") -> FlowNode:
    return FlowNode(name=name, node_type="source", sensitivity=sensitivity)


def _processor(name: str, sensitivity: str = "public") -> FlowNode:
    return FlowNode(name=name, node_type="processor", sensitivity=sensitivity)


def _sink(name: str, sensitivity: str = "public") -> FlowNode:
    return FlowNode(name=name, node_type="sink", sensitivity=sensitivity)


def _storage(name: str, sensitivity: str = "public") -> FlowNode:
    return FlowNode(name=name, node_type="storage", sensitivity=sensitivity)


def _connect(source: str, dest: str, allowed: bool = True) -> FlowConnection:
    return FlowConnection(source=source, destination=dest, allowed=allowed)


# ------------------------------------------------------------------
# Tests: clean / compliant graphs
# ------------------------------------------------------------------


def test_empty_graph_is_compliant():
    guard = DataFlowGuard()
    result = guard.analyze()

    assert result.is_compliant is True
    assert result.violations == []
    assert result.nodes == 0
    assert result.connections == 0
    assert result.risk_score == 0.0


def test_clean_flow_graph_is_compliant():
    guard = DataFlowGuard()
    guard.add_node(_source("input", "public"))
    guard.add_node(_processor("sanitizer", "public"))
    guard.add_node(_sink("api", "public"))
    guard.add_connection(_connect("input", "sanitizer"))
    guard.add_connection(_connect("sanitizer", "api"))

    result = guard.analyze()

    assert result.is_compliant is True
    assert result.violations == []
    assert result.nodes == 3
    assert result.connections == 2
    assert result.risk_score == 0.0


def test_same_sensitivity_flow_is_allowed():
    guard = DataFlowGuard()
    guard.add_node(_source("a", "confidential"))
    guard.add_node(_processor("b", "confidential"))
    guard.add_connection(_connect("a", "b"))

    result = guard.analyze()

    assert result.is_compliant is True
    assert result.risk_score == 0.0


def test_upward_sensitivity_flow_is_allowed():
    """Data flowing from lower to higher sensitivity is fine."""
    guard = DataFlowGuard()
    guard.add_node(_source("public_src", "public"))
    guard.add_node(_storage("restricted_store", "restricted"))
    guard.add_connection(_connect("public_src", "restricted_store"))

    result = guard.analyze()

    assert result.is_compliant is True


# ------------------------------------------------------------------
# Tests: unauthorized flow
# ------------------------------------------------------------------


def test_unauthorized_flow_violation():
    guard = DataFlowGuard()
    guard.add_node(_source("input"))
    guard.add_node(_sink("blocked_api"))
    guard.add_connection(_connect("input", "blocked_api", allowed=False))

    result = guard.analyze()

    assert result.is_compliant is False
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.violation_type == "unauthorized_flow"
    assert violation.severity == "high"
    assert "input" in violation.description
    assert "blocked_api" in violation.description


# ------------------------------------------------------------------
# Tests: sensitivity mismatch
# ------------------------------------------------------------------


def test_sensitivity_mismatch_restricted_to_public():
    guard = DataFlowGuard()
    guard.add_node(_source("secret_data", "restricted"))
    guard.add_node(_processor("open_processor", "public"))
    guard.add_connection(_connect("secret_data", "open_processor"))

    result = guard.analyze()

    mismatches = [
        v for v in result.violations if v.violation_type == "sensitivity_mismatch"
    ]
    assert len(mismatches) == 1
    assert mismatches[0].severity == "critical"


def test_sensitivity_mismatch_confidential_to_internal():
    guard = DataFlowGuard()
    guard.add_node(_source("conf_src", "confidential"))
    guard.add_node(_processor("int_proc", "internal"))
    guard.add_connection(_connect("conf_src", "int_proc"))

    result = guard.analyze()

    mismatches = [
        v for v in result.violations if v.violation_type == "sensitivity_mismatch"
    ]
    assert len(mismatches) == 1
    assert mismatches[0].severity == "high"


# ------------------------------------------------------------------
# Tests: data leak detection
# ------------------------------------------------------------------


def test_data_leak_to_external_sink():
    guard = DataFlowGuard()
    guard.add_node(_source("pii_data", "confidential"))
    guard.add_node(_sink("external_api", "confidential"))
    guard.add_connection(_connect("pii_data", "external_api"))

    result = guard.analyze()

    leaks = [v for v in result.violations if v.violation_type == "data_leak"]
    assert len(leaks) == 1
    assert leaks[0].severity == "critical"
    assert "pii_data" in leaks[0].description


def test_data_leak_restricted_to_sink():
    guard = DataFlowGuard()
    guard.add_node(_source("top_secret", "restricted"))
    guard.add_node(_sink("logger", "restricted"))
    guard.add_connection(_connect("top_secret", "logger"))

    result = guard.analyze()

    leaks = [v for v in result.violations if v.violation_type == "data_leak"]
    assert len(leaks) == 1


def test_no_leak_when_public_data_to_sink():
    guard = DataFlowGuard()
    guard.add_node(_source("public_data", "public"))
    guard.add_node(_sink("api"))
    guard.add_connection(_connect("public_data", "api"))

    result = guard.analyze()

    leaks = [v for v in result.violations if v.violation_type == "data_leak"]
    assert len(leaks) == 0


def test_no_leak_when_sensitive_to_non_sink():
    """Confidential data to a processor (not a sink) is not a leak."""
    guard = DataFlowGuard()
    guard.add_node(_source("conf_src", "confidential"))
    guard.add_node(_processor("internal_proc", "confidential"))
    guard.add_connection(_connect("conf_src", "internal_proc"))

    result = guard.analyze()

    leaks = [v for v in result.violations if v.violation_type == "data_leak"]
    assert len(leaks) == 0


# ------------------------------------------------------------------
# Tests: circular flow
# ------------------------------------------------------------------


def test_circular_flow_detection():
    guard = DataFlowGuard()
    guard.add_node(_processor("a"))
    guard.add_node(_processor("b"))
    guard.add_node(_processor("c"))
    guard.add_connection(_connect("a", "b"))
    guard.add_connection(_connect("b", "c"))
    guard.add_connection(_connect("c", "a"))

    result = guard.analyze()

    cycles = [v for v in result.violations if v.violation_type == "circular_flow"]
    assert len(cycles) >= 1
    assert cycles[0].severity == "medium"


def test_self_loop_is_circular():
    guard = DataFlowGuard()
    guard.add_node(_processor("loop"))
    guard.add_connection(_connect("loop", "loop"))

    result = guard.analyze()

    cycles = [v for v in result.violations if v.violation_type == "circular_flow"]
    assert len(cycles) >= 1


# ------------------------------------------------------------------
# Tests: node and connection management
# ------------------------------------------------------------------


def test_list_nodes():
    guard = DataFlowGuard()
    guard.add_node(_source("a"))
    guard.add_node(_processor("b"))

    nodes = guard.list_nodes()

    assert len(nodes) == 2
    names = {n.name for n in nodes}
    assert names == {"a", "b"}


def test_list_connections():
    guard = DataFlowGuard()
    guard.add_node(_source("a"))
    guard.add_node(_sink("b"))
    conn = _connect("a", "b")
    guard.add_connection(conn)

    connections = guard.list_connections()

    assert len(connections) == 1
    assert connections[0].source == "a"
    assert connections[0].destination == "b"


def test_remove_node_removes_connections():
    guard = DataFlowGuard()
    guard.add_node(_source("a"))
    guard.add_node(_processor("b"))
    guard.add_node(_sink("c"))
    guard.add_connection(_connect("a", "b"))
    guard.add_connection(_connect("b", "c"))

    guard.remove_node("b")

    assert len(guard.list_nodes()) == 2
    assert len(guard.list_connections()) == 0


def test_remove_connection():
    guard = DataFlowGuard()
    guard.add_node(_source("a"))
    guard.add_node(_sink("b"))
    guard.add_connection(_connect("a", "b"))

    guard.remove_connection("a", "b")

    assert len(guard.list_connections()) == 0


def test_remove_nonexistent_node_is_safe():
    guard = DataFlowGuard()
    guard.remove_node("nonexistent")

    assert len(guard.list_nodes()) == 0


def test_remove_nonexistent_connection_is_safe():
    guard = DataFlowGuard()
    guard.remove_connection("x", "y")

    assert len(guard.list_connections()) == 0


# ------------------------------------------------------------------
# Tests: stats tracking
# ------------------------------------------------------------------


def test_stats_initial():
    guard = DataFlowGuard()
    s = guard.stats()

    assert s.total_analyses == 0
    assert s.violations_found == 0
    assert s.compliant_count == 0


def test_stats_after_compliant_analysis():
    guard = DataFlowGuard()
    guard.add_node(_source("a"))
    guard.analyze()

    s = guard.stats()

    assert s.total_analyses == 1
    assert s.compliant_count == 1
    assert s.violations_found == 0


def test_stats_accumulate_across_analyses():
    guard = DataFlowGuard()
    guard.add_node(_source("a", "restricted"))
    guard.add_node(_sink("b", "public"))
    guard.add_connection(_connect("a", "b"))

    guard.analyze()
    guard.analyze()

    s = guard.stats()

    assert s.total_analyses == 2
    assert s.violations_found > 0
    assert s.compliant_count == 0


# ------------------------------------------------------------------
# Tests: risk score
# ------------------------------------------------------------------


def test_risk_score_scales_with_violations():
    guard = DataFlowGuard()
    guard.add_node(_source("a"))
    guard.add_node(_sink("b"))
    guard.add_connection(_connect("a", "b", allowed=False))

    result = guard.analyze()

    assert result.risk_score == 0.2


def test_risk_score_caps_at_one():
    guard = DataFlowGuard()
    for i in range(10):
        src = f"src_{i}"
        dst = f"dst_{i}"
        guard.add_node(_source(src))
        guard.add_node(_sink(dst))
        guard.add_connection(_connect(src, dst, allowed=False))

    result = guard.analyze()

    assert result.risk_score == 1.0


# ------------------------------------------------------------------
# Tests: complex graph with multiple violations
# ------------------------------------------------------------------


def test_complex_graph_multiple_violations():
    guard = DataFlowGuard()
    guard.add_node(_source("user_pii", "restricted"))
    guard.add_node(_processor("normalizer", "public"))
    guard.add_node(_sink("third_party_api", "public"))
    guard.add_node(_processor("feedback", "public"))

    guard.add_connection(_connect("user_pii", "normalizer"))
    guard.add_connection(_connect("normalizer", "third_party_api"))
    guard.add_connection(_connect("third_party_api", "feedback", allowed=False))
    guard.add_connection(_connect("feedback", "normalizer"))

    result = guard.analyze()

    assert result.is_compliant is False
    violation_types = {v.violation_type for v in result.violations}
    assert "sensitivity_mismatch" in violation_types
    assert "unauthorized_flow" in violation_types
    assert "circular_flow" in violation_types
    assert result.risk_score > 0


# ------------------------------------------------------------------
# Tests: node types
# ------------------------------------------------------------------


def test_node_types_stored_correctly():
    guard = DataFlowGuard()
    guard.add_node(_source("a"))
    guard.add_node(_processor("b"))
    guard.add_node(_sink("c"))
    guard.add_node(_storage("d"))

    nodes = {n.name: n for n in guard.list_nodes()}

    assert nodes["a"].node_type == "source"
    assert nodes["b"].node_type == "processor"
    assert nodes["c"].node_type == "sink"
    assert nodes["d"].node_type == "storage"


# ------------------------------------------------------------------
# Tests: dataclass fields
# ------------------------------------------------------------------


def test_flow_connection_default_data_type():
    conn = FlowConnection(source="a", destination="b")
    assert conn.data_type == "text"
    assert conn.allowed is True


def test_flow_node_default_sensitivity():
    node = FlowNode(name="x", node_type="source")
    assert node.sensitivity == "public"


def test_flow_analysis_dataclass():
    analysis = FlowAnalysis(nodes=5, connections=3)
    assert analysis.is_compliant is True
    assert analysis.risk_score == 0.0
    assert analysis.violations == []


def test_flow_violation_dataclass():
    v = FlowViolation(
        source="a",
        destination="b",
        violation_type="data_leak",
        description="test",
        severity="critical",
    )
    assert v.source == "a"
    assert v.severity == "critical"


def test_flow_stats_dataclass():
    s = FlowStats()
    assert s.total_analyses == 0
    assert s.violations_found == 0
    assert s.compliant_count == 0
