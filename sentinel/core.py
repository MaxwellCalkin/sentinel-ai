"""Core orchestration engine for Sentinel guardrails."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence


class RiskLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __ge__(self, other: RiskLevel) -> bool:
        order = list(RiskLevel)
        return order.index(self) >= order.index(other)

    def __gt__(self, other: RiskLevel) -> bool:
        order = list(RiskLevel)
        return order.index(self) > order.index(other)

    def __le__(self, other: RiskLevel) -> bool:
        order = list(RiskLevel)
        return order.index(self) <= order.index(other)

    def __lt__(self, other: RiskLevel) -> bool:
        order = list(RiskLevel)
        return order.index(self) < order.index(other)


@dataclass
class Finding:
    scanner: str
    category: str
    description: str
    risk: RiskLevel
    span: tuple[int, int] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ScanResult:
    text: str
    findings: list[Finding] = field(default_factory=list)
    risk: RiskLevel = RiskLevel.NONE
    blocked: bool = False
    latency_ms: float = 0.0
    redacted_text: str | None = None

    @property
    def safe(self) -> bool:
        return not self.blocked and self.risk <= RiskLevel.LOW


class Scanner(Protocol):
    name: str

    def scan(self, text: str, context: dict | None = None) -> list[Finding]: ...


class SentinelGuard:
    """Main entry point. Orchestrates multiple scanners and applies policy."""

    def __init__(
        self,
        scanners: Sequence[Scanner] | None = None,
        block_threshold: RiskLevel = RiskLevel.HIGH,
        redact_pii: bool = True,
    ):
        self._scanners: list[Scanner] = list(scanners) if scanners else []
        self._block_threshold = block_threshold
        self._redact_pii = redact_pii

    def add_scanner(self, scanner: Scanner) -> SentinelGuard:
        self._scanners.append(scanner)
        return self

    @classmethod
    def default(cls) -> SentinelGuard:
        """Create a guard with all built-in scanners enabled."""
        from sentinel.scanners.prompt_injection import PromptInjectionScanner
        from sentinel.scanners.pii import PIIScanner
        from sentinel.scanners.harmful_content import HarmfulContentScanner
        from sentinel.scanners.hallucination import HallucinationScanner

        return cls(
            scanners=[
                PromptInjectionScanner(),
                PIIScanner(),
                HarmfulContentScanner(),
                HallucinationScanner(),
            ]
        )

    def scan(self, text: str, context: dict | None = None) -> ScanResult:
        start = time.perf_counter()
        all_findings: list[Finding] = []

        for scanner in self._scanners:
            findings = scanner.scan(text, context)
            all_findings.extend(findings)

        max_risk = RiskLevel.NONE
        for f in all_findings:
            if f.risk > max_risk:
                max_risk = f.risk

        blocked = max_risk >= self._block_threshold

        redacted = None
        if self._redact_pii and any(f.category == "pii" for f in all_findings):
            redacted = self._apply_redactions(text, all_findings)

        elapsed = (time.perf_counter() - start) * 1000

        return ScanResult(
            text=text,
            findings=all_findings,
            risk=max_risk,
            blocked=blocked,
            latency_ms=round(elapsed, 2),
            redacted_text=redacted,
        )

    async def scan_async(self, text: str, context: dict | None = None) -> ScanResult:
        """Run scanners concurrently."""
        start = time.perf_counter()
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(None, scanner.scan, text, context)
            for scanner in self._scanners
        ]
        results = await asyncio.gather(*tasks)

        all_findings: list[Finding] = []
        for findings in results:
            all_findings.extend(findings)

        max_risk = RiskLevel.NONE
        for f in all_findings:
            if f.risk > max_risk:
                max_risk = f.risk

        blocked = max_risk >= self._block_threshold

        redacted = None
        if self._redact_pii and any(f.category == "pii" for f in all_findings):
            redacted = self._apply_redactions(text, all_findings)

        elapsed = (time.perf_counter() - start) * 1000

        return ScanResult(
            text=text,
            findings=all_findings,
            risk=max_risk,
            blocked=blocked,
            latency_ms=round(elapsed, 2),
            redacted_text=redacted,
        )

    def _apply_redactions(self, text: str, findings: list[Finding]) -> str:
        pii_findings = sorted(
            [f for f in findings if f.category == "pii" and f.span],
            key=lambda f: f.span[0],
            reverse=True,
        )
        redacted = text
        for f in pii_findings:
            start, end = f.span
            label = f.metadata.get("pii_type", "REDACTED")
            redacted = redacted[:start] + f"[{label}]" + redacted[end:]
        return redacted
