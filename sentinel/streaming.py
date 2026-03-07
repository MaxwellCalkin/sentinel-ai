"""Streaming scanner for real-time LLM output scanning.

Scans text incrementally as tokens arrive from a streaming LLM response.
Maintains a sliding buffer and emits findings as soon as they're detected.

Usage:
    from sentinel.streaming import StreamingGuard

    guard = StreamingGuard()
    async for chunk in llm_stream:
        result = guard.feed(chunk.text)
        if result.blocked:
            break  # stop the stream
        yield result.safe_text  # or result.redacted_text
    final = guard.finalize()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sentinel.core import Finding, RiskLevel, Scanner, SentinelGuard


@dataclass
class StreamChunkResult:
    chunk: str
    safe_text: str
    redacted_text: str | None = None
    findings: list[Finding] = field(default_factory=list)
    blocked: bool = False
    risk: RiskLevel = RiskLevel.NONE
    buffer_text: str = ""


class StreamingGuard:
    """Incrementally scan streaming LLM output.

    Maintains an internal buffer. On each feed(), scans the buffer
    for findings. Text that has been fully scanned (past the lookback
    window) is released as safe_text.
    """

    def __init__(
        self,
        guard: SentinelGuard | None = None,
        buffer_size: int = 500,
        block_threshold: RiskLevel = RiskLevel.HIGH,
        redact_pii: bool = True,
    ):
        if guard is None:
            guard = SentinelGuard.default()
        self._guard = guard
        self._buffer_size = buffer_size
        self._block_threshold = block_threshold
        self._redact_pii = redact_pii
        self._buffer = ""
        self._released = ""
        self._all_findings: list[Finding] = []
        self._blocked = False

    def feed(self, chunk: str) -> StreamChunkResult:
        """Feed a new chunk from the stream. Returns scan result for this chunk."""
        if self._blocked:
            return StreamChunkResult(
                chunk=chunk,
                safe_text="",
                blocked=True,
                risk=RiskLevel.CRITICAL,
            )

        self._buffer += chunk

        # Scan the full buffer
        result = self._guard.scan(self._buffer)

        # Check for blocking
        if result.risk >= self._block_threshold:
            self._blocked = True
            self._all_findings.extend(result.findings)
            return StreamChunkResult(
                chunk=chunk,
                safe_text="",
                blocked=True,
                risk=result.risk,
                findings=result.findings,
                buffer_text=self._buffer,
            )

        self._all_findings.extend(
            f for f in result.findings if f not in self._all_findings
        )

        # Release text beyond the lookback window
        safe_text = ""
        if len(self._buffer) > self._buffer_size:
            release_point = len(self._buffer) - self._buffer_size
            safe_text = self._buffer[:release_point]
            self._buffer = self._buffer[release_point:]
            self._released += safe_text

        redacted = None
        if self._redact_pii and result.redacted_text:
            redacted = result.redacted_text

        return StreamChunkResult(
            chunk=chunk,
            safe_text=safe_text,
            redacted_text=redacted,
            findings=result.findings,
            risk=result.risk,
            buffer_text=self._buffer,
        )

    def finalize(self) -> StreamChunkResult:
        """Flush the remaining buffer. Call after stream ends."""
        if self._blocked:
            return StreamChunkResult(
                chunk="",
                safe_text="",
                blocked=True,
                risk=RiskLevel.CRITICAL,
            )

        result = self._guard.scan(self._buffer)
        safe_text = self._buffer
        self._released += safe_text
        self._buffer = ""

        return StreamChunkResult(
            chunk="",
            safe_text=safe_text,
            redacted_text=result.redacted_text,
            findings=result.findings,
            risk=result.risk,
        )

    @property
    def full_text(self) -> str:
        return self._released + self._buffer

    @property
    def blocked(self) -> bool:
        return self._blocked

    @property
    def all_findings(self) -> list[Finding]:
        return list(self._all_findings)

    def reset(self) -> None:
        self._buffer = ""
        self._released = ""
        self._all_findings.clear()
        self._blocked = False
