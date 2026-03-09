"""Span-based output tracing for LLM safety debugging.

Trace LLM output generation with hierarchical spans showing which
safety checks ran, their results, and timing. Build trace trees
for visualization and export.

Usage:
    from sentinel.output_tracer import OutputTracer

    tracer = OutputTracer()
    with tracer.span("pipeline") as root:
        with tracer.span("pii_scan", parent=root.span_id) as s:
            s.set_metadata("scanner", "pii")
            s.set_status("pass")
    tree = tracer.build_tree()
    print(tree.total_duration_ms)
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """A single traced operation with timing and metadata."""
    name: str
    start_time: float
    end_time: float | None = None
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    children: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def is_complete(self) -> bool:
        return self.end_time is not None


@dataclass
class TraceTree:
    """Hierarchical view of all spans in a trace."""
    root_spans: list[str]
    all_spans: dict[str, Span]
    total_duration_ms: float


@dataclass
class SpanSummary:
    """Compact summary of a single span."""
    name: str
    duration_ms: float
    status: str
    child_count: int


class SpanContext:
    """Context wrapper providing a fluent interface for span operations."""

    def __init__(self, span_id: str, span: Span) -> None:
        self._span_id = span_id
        self._span = span

    @property
    def span_id(self) -> str:
        return self._span_id

    def set_metadata(self, key: str, value: Any) -> None:
        self._span.metadata[key] = value

    def set_status(self, status: str) -> None:
        self._span.status = status


class OutputTracer:
    """Traces LLM output generation with span-based tracking."""

    def __init__(self) -> None:
        self._spans: dict[str, Span] = {}

    @contextmanager
    def span(self, name: str, parent: str | None = None):
        """Context manager that creates, tracks, and auto-closes a span."""
        span_id = uuid.uuid4().hex[:12]
        new_span = Span(name=name, start_time=time.time(), parent=parent)
        self._spans[span_id] = new_span

        if parent and parent in self._spans:
            self._spans[parent].children.append(span_id)

        ctx = SpanContext(span_id, new_span)
        try:
            yield ctx
        finally:
            new_span.end_time = time.time()
            if new_span.status == "pending":
                new_span.status = "pass"

    def start_span(self, name: str, parent: str | None = None) -> str:
        """Manually start a span and return its ID."""
        span_id = uuid.uuid4().hex[:12]
        new_span = Span(name=name, start_time=time.time(), parent=parent)
        self._spans[span_id] = new_span

        if parent and parent in self._spans:
            self._spans[parent].children.append(span_id)

        return span_id

    def end_span(self, span_id: str, status: str = "pass") -> None:
        """Manually end a previously started span."""
        span = self._spans.get(span_id)
        if span is None:
            raise KeyError(f"Unknown span: {span_id}")
        span.end_time = time.time()
        span.status = status

    def attach_metadata(self, span_id: str, key: str, value: Any) -> None:
        """Attach a metadata key-value pair to an existing span."""
        span = self._spans.get(span_id)
        if span is None:
            raise KeyError(f"Unknown span: {span_id}")
        span.metadata[key] = value

    def get_span(self, span_id: str) -> Span | None:
        """Retrieve a span by ID."""
        return self._spans.get(span_id)

    def build_tree(self) -> TraceTree:
        """Build a trace tree from all completed spans."""
        root_ids = [
            sid for sid, span in self._spans.items()
            if span.parent is None
        ]

        total_duration = self._compute_total_duration()

        return TraceTree(
            root_spans=root_ids,
            all_spans=dict(self._spans),
            total_duration_ms=round(total_duration, 2),
        )

    def export_flat(self) -> list[SpanSummary]:
        """Export all spans as a flat list of summaries."""
        return [
            SpanSummary(
                name=span.name,
                duration_ms=round(span.duration_ms, 2),
                status=span.status,
                child_count=len(span.children),
            )
            for span in self._spans.values()
        ]

    def export_nested(self) -> list[dict[str, Any]]:
        """Export spans as nested dicts mirroring the parent-child tree."""
        root_ids = [
            sid for sid, span in self._spans.items()
            if span.parent is None
        ]
        return [self._span_to_nested_dict(sid) for sid in root_ids]

    def find_by_name(self, name: str) -> list[str]:
        """Find all span IDs matching a given name."""
        return [
            sid for sid, span in self._spans.items()
            if span.name == name
        ]

    def find_by_status(self, status: str) -> list[str]:
        """Find all span IDs matching a given status."""
        return [
            sid for sid, span in self._spans.items()
            if span.status == status
        ]

    def total_duration_ms(self) -> float:
        """Compute the total trace duration across all root spans."""
        return round(self._compute_total_duration(), 2)

    def span_count(self) -> int:
        """Return the number of tracked spans."""
        return len(self._spans)

    def clear(self) -> int:
        """Remove all spans and return the count cleared."""
        count = len(self._spans)
        self._spans.clear()
        return count

    def _compute_total_duration(self) -> float:
        """Sum the durations of all root spans to get total trace time."""
        total = 0.0
        for span in self._spans.values():
            if span.parent is None and span.is_complete:
                total += span.duration_ms
        return total

    def _span_to_nested_dict(self, span_id: str) -> dict[str, Any]:
        """Recursively convert a span and its children to a nested dict."""
        span = self._spans[span_id]
        result: dict[str, Any] = {
            "name": span.name,
            "status": span.status,
            "duration_ms": round(span.duration_ms, 2),
            "metadata": dict(span.metadata),
        }
        if span.children:
            result["children"] = [
                self._span_to_nested_dict(child_id)
                for child_id in span.children
                if child_id in self._spans
            ]
        return result
