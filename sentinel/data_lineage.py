"""Data lineage tracking for LLM pipelines.

Track data flow and transformations through LLM pipelines,
providing visibility into how data moves and is transformed.

Usage:
    from sentinel.data_lineage import DataLineage

    lineage = DataLineage("my-pipeline")
    lineage.add_node("raw_input", node_type="source")
    lineage.add_node("sanitizer", node_type="transform")
    lineage.add_node("llm", node_type="model")
    lineage.add_edge("raw_input", "sanitizer")
    lineage.add_edge("sanitizer", "llm")
    lineage.record_flow("sanitizer", "PII removed", record_count=42)
    print(lineage.trace("llm"))  # ['sanitizer', 'raw_input']
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

VALID_NODE_TYPES = frozenset(
    {"source", "transform", "model", "output", "filter", "cache"}
)


@dataclass
class LineageNode:
    """A node in the data lineage graph."""
    name: str
    node_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdge:
    """A directed edge connecting two lineage nodes."""
    from_node: str
    to_node: str
    label: str = ""


@dataclass
class FlowRecord:
    """A record of data flowing through a node."""
    node_name: str
    data_summary: str
    record_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class LineageValidation:
    """Result of lineage graph validation."""
    valid: bool
    orphan_nodes: list[str] = field(default_factory=list)
    has_cycles: bool = False
    warnings: list[str] = field(default_factory=list)


class DataLineage:
    """Track data flow and transformations through LLM pipelines."""

    def __init__(self, pipeline_name: str = "default") -> None:
        self.pipeline_name = pipeline_name
        self._nodes: dict[str, LineageNode] = {}
        self._edges: list[LineageEdge] = []
        self._flow_history: dict[str, list[FlowRecord]] = {}

    def add_node(
        self,
        name: str,
        node_type: str = "transform",
        metadata: dict[str, Any] | None = None,
    ) -> LineageNode:
        """Add a pipeline node.

        Args:
            name: Unique node name.
            node_type: One of source, transform, model, output, filter, cache.
            metadata: Optional metadata dict.

        Returns:
            The created LineageNode.

        Raises:
            ValueError: If node_type is invalid or name is already taken.
        """
        if node_type not in VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type '{node_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_NODE_TYPES))}"
            )
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")
        node = LineageNode(name=name, node_type=node_type, metadata=metadata or {})
        self._nodes[name] = node
        return node

    def add_edge(self, from_node: str, to_node: str, label: str = "") -> LineageEdge:
        """Connect two nodes with a directed edge.

        Raises:
            KeyError: If either node does not exist.
        """
        self._require_node(from_node)
        self._require_node(to_node)
        edge = LineageEdge(from_node=from_node, to_node=to_node, label=label)
        self._edges.append(edge)
        return edge

    def record_flow(
        self, node_name: str, data_summary: str, record_count: int = 0
    ) -> FlowRecord:
        """Record data flowing through a node.

        Raises:
            KeyError: If the node does not exist.
        """
        self._require_node(node_name)
        record = FlowRecord(
            node_name=node_name,
            data_summary=data_summary,
            record_count=record_count,
        )
        self._flow_history.setdefault(node_name, []).append(record)
        return record

    def trace(self, node_name: str) -> list[str]:
        """Trace all upstream ancestors of a node via BFS.

        Returns:
            List of ancestor node names (excluding the node itself),
            ordered by discovery (breadth-first).

        Raises:
            KeyError: If the node does not exist.
        """
        self._require_node(node_name)
        return self._bfs_walk(node_name, direction="upstream")

    def downstream(self, node_name: str) -> list[str]:
        """Find all downstream dependents of a node via BFS.

        Returns:
            List of descendant node names (excluding the node itself),
            ordered by discovery (breadth-first).

        Raises:
            KeyError: If the node does not exist.
        """
        self._require_node(node_name)
        return self._bfs_walk(node_name, direction="downstream")

    def get_flow_history(self, node_name: str) -> list[FlowRecord]:
        """Get all flow records for a node.

        Raises:
            KeyError: If the node does not exist.
        """
        self._require_node(node_name)
        return list(self._flow_history.get(node_name, []))

    def validate(self) -> LineageValidation:
        """Validate the lineage graph for structural issues."""
        orphans = self._find_orphan_nodes()
        has_cycles = self._detect_cycles()
        warnings = self._build_warnings(orphans, has_cycles)
        is_valid = len(orphans) == 0 and not has_cycles
        return LineageValidation(
            valid=is_valid,
            orphan_nodes=orphans,
            has_cycles=has_cycles,
            warnings=warnings,
        )

    def export(self) -> dict[str, Any]:
        """Export the full lineage graph as a dictionary."""
        return {
            "pipeline_name": self.pipeline_name,
            "nodes": [
                {
                    "name": node.name,
                    "node_type": node.node_type,
                    "metadata": node.metadata,
                }
                for node in self._nodes.values()
            ],
            "edges": [
                {
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "label": edge.label,
                }
                for edge in self._edges
            ],
            "flow_records": [
                {
                    "node_name": record.node_name,
                    "data_summary": record.data_summary,
                    "record_count": record.record_count,
                    "timestamp": record.timestamp,
                }
                for records in self._flow_history.values()
                for record in records
            ],
        }

    def summary(self) -> str:
        """Return a human-readable summary of the lineage graph."""
        total_flows = sum(len(r) for r in self._flow_history.values())
        lines = [
            f"Pipeline: {self.pipeline_name}",
            f"  Nodes: {len(self._nodes)}",
            f"  Edges: {len(self._edges)}",
            f"  Flow records: {total_flows}",
        ]
        if self._nodes:
            type_counts: dict[str, int] = {}
            for node in self._nodes.values():
                type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1
            breakdown = ", ".join(
                f"{count} {ntype}" for ntype, count in sorted(type_counts.items())
            )
            lines.append(f"  Node types: {breakdown}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset all nodes, edges, and flow history."""
        self._nodes.clear()
        self._edges.clear()
        self._flow_history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_node(self, name: str) -> None:
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' does not exist")

    def _bfs_walk(self, start: str, direction: str) -> list[str]:
        """Walk the graph via BFS in the given direction."""
        adjacency = self._build_adjacency(direction)
        visited: set[str] = set()
        result: list[str] = []
        queue: deque[str] = deque()

        for neighbor in adjacency.get(start, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                result.append(neighbor)

        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    result.append(neighbor)

        return result

    def _build_adjacency(self, direction: str) -> dict[str, list[str]]:
        """Build an adjacency list for the given traversal direction."""
        adjacency: dict[str, list[str]] = {}
        for edge in self._edges:
            if direction == "upstream":
                adjacency.setdefault(edge.to_node, []).append(edge.from_node)
            else:
                adjacency.setdefault(edge.from_node, []).append(edge.to_node)
        return adjacency

    def _find_orphan_nodes(self) -> list[str]:
        """Find nodes with no incoming or outgoing edges."""
        connected: set[str] = set()
        for edge in self._edges:
            connected.add(edge.from_node)
            connected.add(edge.to_node)
        return sorted(name for name in self._nodes if name not in connected)

    def _detect_cycles(self) -> bool:
        """Detect cycles using DFS with three-color marking."""
        white: set[str] = set(self._nodes.keys())
        gray: set[str] = set()
        black: set[str] = set()
        adjacency = self._build_adjacency("downstream")

        for node in list(self._nodes.keys()):
            if node in white:
                if self._dfs_has_cycle(node, adjacency, white, gray, black):
                    return True
        return False

    def _dfs_has_cycle(
        self,
        node: str,
        adjacency: dict[str, list[str]],
        white: set[str],
        gray: set[str],
        black: set[str],
    ) -> bool:
        """DFS helper that returns True if a cycle is found."""
        white.discard(node)
        gray.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor in gray:
                return True
            if neighbor in white:
                if self._dfs_has_cycle(neighbor, adjacency, white, gray, black):
                    return True

        gray.discard(node)
        black.add(node)
        return False

    def _build_warnings(self, orphans: list[str], has_cycles: bool) -> list[str]:
        warnings: list[str] = []
        if orphans:
            warnings.append(
                f"Orphan nodes (no edges): {', '.join(orphans)}"
            )
        if has_cycles:
            warnings.append("Graph contains cycles")
        disconnected_count = self._count_connected_components()
        if disconnected_count > 1:
            warnings.append(
                f"Graph has {disconnected_count} disconnected subgraphs"
            )
        return warnings

    def _count_connected_components(self) -> int:
        """Count connected components treating edges as undirected."""
        if not self._nodes:
            return 0

        undirected: dict[str, set[str]] = {name: set() for name in self._nodes}
        for edge in self._edges:
            undirected[edge.from_node].add(edge.to_node)
            undirected[edge.to_node].add(edge.from_node)

        visited: set[str] = set()
        components = 0

        for node in self._nodes:
            if node not in visited:
                components += 1
                queue: deque[str] = deque([node])
                visited.add(node)
                while queue:
                    current = queue.popleft()
                    for neighbor in undirected[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

        return components
