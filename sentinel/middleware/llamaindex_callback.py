"""LlamaIndex integration for Sentinel AI safety guardrails.

Provides a callback handler for LlamaIndex's callback system that scans
queries and responses in real-time.

Usage:
    from sentinel.middleware.llamaindex_callback import SentinelEventHandler

    handler = SentinelEventHandler()
    # With LlamaIndex settings:
    from llama_index.core import Settings
    Settings.callback_manager.add_handler(handler)

    # Or with query engine:
    query_engine = index.as_query_engine(callbacks=[handler])
    response = query_engine.query("What is the meaning of life?")
    print(handler.findings)
"""

from __future__ import annotations

from typing import Any
from sentinel.core import SentinelGuard, ScanResult, Finding, RiskLevel


class SentinelEventHandler:
    """LlamaIndex callback handler that scans queries and responses.

    Compatible with LlamaIndex's CallbackManager interface without
    requiring llama-index as a dependency.

    Attributes:
        findings: List of all findings from scans.
        blocked: Whether any query or response was blocked.
        scans: List of all ScanResult objects.
    """

    def __init__(
        self,
        guard: SentinelGuard | None = None,
        scan_queries: bool = True,
        scan_responses: bool = True,
        raise_on_block: bool = False,
    ):
        self._guard = guard or SentinelGuard.default()
        self._scan_queries = scan_queries
        self._scan_responses = scan_responses
        self._raise_on_block = raise_on_block
        self.findings: list[Finding] = []
        self.scans: list[ScanResult] = []
        self.blocked: bool = False

    def reset(self) -> None:
        """Clear findings between runs."""
        self.findings.clear()
        self.scans.clear()
        self.blocked = False

    def _handle_scan(self, scan: ScanResult, source: str) -> None:
        self.scans.append(scan)
        self.findings.extend(scan.findings)
        if scan.blocked:
            self.blocked = True
            if self._raise_on_block:
                raise SentinelBlockedError(
                    f"{source} blocked by Sentinel: "
                    f"risk={scan.risk.value}, "
                    f"findings={len(scan.findings)}"
                )

    def on_event_start(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Handle event start - scan queries before LLM call."""
        if not self._scan_queries or payload is None:
            return event_id

        # LlamaIndex query events
        if event_type in ("query", "llm"):
            text = self._extract_query_text(event_type, payload)
            if text:
                scan = self._guard.scan(text)
                self._handle_scan(scan, f"LlamaIndex {event_type} input")

        return event_id

    def on_event_end(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Handle event end - scan responses after LLM call."""
        if not self._scan_responses or payload is None:
            return

        if event_type in ("query", "llm"):
            text = self._extract_response_text(event_type, payload)
            if text:
                scan = self._guard.scan(text)
                self._handle_scan(scan, f"LlamaIndex {event_type} output")

    def start_trace(self, trace_id: str | None = None) -> None:
        """Called when a trace starts."""
        pass

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        """Called when a trace ends."""
        pass

    @staticmethod
    def _extract_query_text(event_type: str, payload: dict[str, Any]) -> str:
        """Extract query text from LlamaIndex event payload."""
        # QueryStartEvent payload
        if "query_str" in payload:
            return payload["query_str"]
        # LLM event payload
        if "messages" in payload:
            messages = payload["messages"]
            if isinstance(messages, list):
                texts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        texts.append(msg.get("content", ""))
                    elif hasattr(msg, "content"):
                        texts.append(str(msg.content))
                return " ".join(texts)
        if "prompt" in payload:
            return payload["prompt"]
        return ""

    @staticmethod
    def _extract_response_text(event_type: str, payload: dict[str, Any]) -> str:
        """Extract response text from LlamaIndex event payload."""
        if "response" in payload:
            resp = payload["response"]
            if isinstance(resp, str):
                return resp
            if hasattr(resp, "response"):
                return str(resp.response)
            if hasattr(resp, "text"):
                return str(resp.text)
        if "completion" in payload:
            comp = payload["completion"]
            if isinstance(comp, str):
                return comp
            if hasattr(comp, "text"):
                return str(comp.text)
        return ""

    def scan_query(self, query: str) -> ScanResult:
        """Manually scan a query string. Returns the ScanResult."""
        scan = self._guard.scan(query)
        self._handle_scan(scan, "Manual query scan")
        return scan

    def scan_response(self, response: str) -> ScanResult:
        """Manually scan a response string. Returns the ScanResult."""
        scan = self._guard.scan(response)
        self._handle_scan(scan, "Manual response scan")
        return scan


class SentinelBlockedError(Exception):
    """Raised when Sentinel blocks content and raise_on_block=True."""
    pass


def create_sentinel_handler(
    block_threshold: RiskLevel = RiskLevel.HIGH,
    scan_queries: bool = True,
    scan_responses: bool = True,
    raise_on_block: bool = False,
) -> SentinelEventHandler:
    """Factory function to create a configured Sentinel event handler."""
    guard = SentinelGuard.default()
    guard._block_threshold = block_threshold
    return SentinelEventHandler(
        guard=guard,
        scan_queries=scan_queries,
        scan_responses=scan_responses,
        raise_on_block=raise_on_block,
    )
