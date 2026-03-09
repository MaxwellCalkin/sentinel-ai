"""Data flow guard for LLM pipelines.

Track and validate data flow between pipeline components.
Detect unauthorized flows, sensitivity mismatches, data leaks,
and circular dependencies.

Usage:
    from sentinel.data_flow_guard import DataFlowGuard, FlowNode, FlowConnection

    guard = DataFlowGuard()
    guard.add_node(FlowNode("user_input", "source", "confidential"))
    guard.add_node(FlowNode("sanitizer", "processor"))
    guard.add_node(FlowNode("llm_api", "sink", "public"))
    guard.add_connection(FlowConnection("user_input", "sanitizer"))
    guard.add_connection(FlowConnection("sanitizer", "llm_api"))
    result = guard.analyze()
    print(result.is_compliant, result.violations)
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_NODE_TYPES = frozenset({"source", "processor", "sink", "storage"})

SENSITIVITY_LEVELS = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


@dataclass
class FlowNode:
    """A node in the data flow graph."""

    name: str
    node_type: str
    sensitivity: str = "public"


@dataclass
class FlowConnection:
    """A directed connection between two flow nodes."""

    source: str
    destination: str
    data_type: str = "text"
    allowed: bool = True


@dataclass
class FlowViolation:
    """A detected violation in the data flow graph."""

    source: str
    destination: str
    violation_type: str
    description: str
    severity: str


@dataclass
class FlowAnalysis:
    """Result of analyzing a data flow graph."""

    nodes: int
    connections: int
    violations: list[FlowViolation] = field(default_factory=list)
    is_compliant: bool = True
    risk_score: float = 0.0


@dataclass
class FlowStats:
    """Cumulative statistics across analyses."""

    total_analyses: int = 0
    violations_found: int = 0
    compliant_count: int = 0


class DataFlowGuard:
    """Track and validate data flow between LLM pipeline components."""

    def __init__(self) -> None:
        self._nodes: dict[str, FlowNode] = {}
        self._connections: list[FlowConnection] = []
        self._stats = FlowStats()

    def add_node(self, node: FlowNode) -> None:
        """Register a node in the flow graph."""
        self._nodes[node.name] = node

    def add_connection(self, connection: FlowConnection) -> None:
        """Register a connection between two nodes."""
        self._connections.append(connection)

    def remove_node(self, name: str) -> None:
        """Remove a node and all its connections."""
        self._nodes.pop(name, None)
        self._connections = [
            conn
            for conn in self._connections
            if conn.source != name and conn.destination != name
        ]

    def remove_connection(self, source: str, destination: str) -> None:
        """Remove a specific connection by source and destination."""
        self._connections = [
            conn
            for conn in self._connections
            if not (conn.source == source and conn.destination == destination)
        ]

    def list_nodes(self) -> list[FlowNode]:
        """Return all registered nodes."""
        return list(self._nodes.values())

    def list_connections(self) -> list[FlowConnection]:
        """Return all registered connections."""
        return list(self._connections)

    def analyze(self) -> FlowAnalysis:
        """Analyze the flow graph for violations.

        Checks for:
        1. Unauthorized flows (connections marked allowed=False)
        2. Sensitivity mismatches (high to low without declassification)
        3. Data leaks (confidential/restricted data to external sinks)
        4. Circular flows (cycles via DFS)
        """
        violations: list[FlowViolation] = []
        violations.extend(self._check_unauthorized_flows())
        violations.extend(self._check_sensitivity_mismatches())
        violations.extend(self._check_data_leaks())
        violations.extend(self._check_circular_flows())

        risk_score = min(len(violations) * 0.2, 1.0)
        is_compliant = len(violations) == 0

        self._stats.total_analyses += 1
        self._stats.violations_found += len(violations)
        if is_compliant:
            self._stats.compliant_count += 1

        return FlowAnalysis(
            nodes=len(self._nodes),
            connections=len(self._connections),
            violations=violations,
            is_compliant=is_compliant,
            risk_score=risk_score,
        )

    def stats(self) -> FlowStats:
        """Return cumulative analysis statistics."""
        return FlowStats(
            total_analyses=self._stats.total_analyses,
            violations_found=self._stats.violations_found,
            compliant_count=self._stats.compliant_count,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_unauthorized_flows(self) -> list[FlowViolation]:
        """Detect connections explicitly marked as disallowed."""
        violations: list[FlowViolation] = []
        for conn in self._connections:
            if not conn.allowed:
                violations.append(
                    FlowViolation(
                        source=conn.source,
                        destination=conn.destination,
                        violation_type="unauthorized_flow",
                        description=(
                            f"Unauthorized flow from '{conn.source}' "
                            f"to '{conn.destination}'"
                        ),
                        severity="high",
                    )
                )
        return violations

    def _check_sensitivity_mismatches(self) -> list[FlowViolation]:
        """Detect data flowing from higher to lower sensitivity without declassification."""
        violations: list[FlowViolation] = []
        for conn in self._connections:
            source_node = self._nodes.get(conn.source)
            dest_node = self._nodes.get(conn.destination)
            if source_node is None or dest_node is None:
                continue
            source_level = SENSITIVITY_LEVELS.get(source_node.sensitivity, 0)
            dest_level = SENSITIVITY_LEVELS.get(dest_node.sensitivity, 0)
            if source_level > dest_level:
                violations.append(
                    FlowViolation(
                        source=conn.source,
                        destination=conn.destination,
                        violation_type="sensitivity_mismatch",
                        description=(
                            f"Data flows from '{source_node.sensitivity}' "
                            f"node '{conn.source}' to '{dest_node.sensitivity}' "
                            f"node '{conn.destination}'"
                        ),
                        severity="critical" if source_level >= 3 else "high",
                    )
                )
        return violations

    def _check_data_leaks(self) -> list[FlowViolation]:
        """Detect confidential/restricted data flowing to external sinks."""
        violations: list[FlowViolation] = []
        for conn in self._connections:
            source_node = self._nodes.get(conn.source)
            dest_node = self._nodes.get(conn.destination)
            if source_node is None or dest_node is None:
                continue
            source_level = SENSITIVITY_LEVELS.get(source_node.sensitivity, 0)
            is_sensitive = source_level >= SENSITIVITY_LEVELS["confidential"]
            is_external_sink = dest_node.node_type == "sink"
            if is_sensitive and is_external_sink:
                violations.append(
                    FlowViolation(
                        source=conn.source,
                        destination=conn.destination,
                        violation_type="data_leak",
                        description=(
                            f"'{source_node.sensitivity}' data from "
                            f"'{conn.source}' flows to sink '{conn.destination}'"
                        ),
                        severity="critical",
                    )
                )
        return violations

    def _check_circular_flows(self) -> list[FlowViolation]:
        """Detect cycles in the flow graph via DFS."""
        adjacency = self._build_downstream_adjacency()
        white = set(self._nodes.keys())
        gray: set[str] = set()
        black: set[str] = set()
        cycle_edges: list[tuple[str, str]] = []

        for node in list(self._nodes.keys()):
            if node in white:
                self._dfs_find_cycles(
                    node, adjacency, white, gray, black, cycle_edges
                )

        violations: list[FlowViolation] = []
        for source, destination in cycle_edges:
            violations.append(
                FlowViolation(
                    source=source,
                    destination=destination,
                    violation_type="circular_flow",
                    description=(
                        f"Circular flow detected: '{source}' -> '{destination}'"
                    ),
                    severity="medium",
                )
            )
        return violations

    def _build_downstream_adjacency(self) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = {}
        for conn in self._connections:
            adjacency.setdefault(conn.source, []).append(conn.destination)
        return adjacency

    def _dfs_find_cycles(
        self,
        node: str,
        adjacency: dict[str, list[str]],
        white: set[str],
        gray: set[str],
        black: set[str],
        cycle_edges: list[tuple[str, str]],
    ) -> None:
        """DFS helper that collects back edges indicating cycles."""
        white.discard(node)
        gray.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor in gray:
                cycle_edges.append((node, neighbor))
            elif neighbor in white:
                self._dfs_find_cycles(
                    neighbor, adjacency, white, gray, black, cycle_edges
                )

        gray.discard(node)
        black.add(node)
