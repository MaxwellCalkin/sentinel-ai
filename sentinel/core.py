"""Core orchestration engine for Sentinel guardrails."""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from collections import OrderedDict
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
        from sentinel.scanners.toxicity import ToxicityScanner
        from sentinel.scanners.tool_use import ToolUseScanner
        from sentinel.scanners.obfuscation import ObfuscationScanner
        from sentinel.scanners.secrets_scanner import SecretsScanner

        return cls(
            scanners=[
                PromptInjectionScanner(),
                PIIScanner(),
                HarmfulContentScanner(),
                HallucinationScanner(),
                ToxicityScanner(),
                ToolUseScanner(),
                ObfuscationScanner(),
                SecretsScanner(),
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

    def scan_batch(
        self, texts: list[str], context: dict | None = None
    ) -> list[ScanResult]:
        """Scan multiple texts sequentially. Use scan_batch_async for concurrency."""
        return [self.scan(t, context) for t in texts]

    async def scan_batch_async(
        self, texts: list[str], context: dict | None = None
    ) -> list[ScanResult]:
        """Scan multiple texts concurrently using asyncio."""
        tasks = [self.scan_async(t, context) for t in texts]
        return list(await asyncio.gather(*tasks))

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


class ScanCache:
    """Thread-safe LRU cache for scan results. Avoids re-scanning identical content.

    Usage:
        guard = SentinelGuard.default()
        cache = ScanCache(guard, maxsize=1000, ttl=300)
        result = cache.scan("text")  # cached on second call
    """

    def __init__(
        self,
        guard: SentinelGuard,
        maxsize: int = 1024,
        ttl: float = 300.0,
    ):
        self._guard = guard
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[ScanResult, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def scan(self, text: str, context: dict | None = None) -> ScanResult:
        key = self._key(text)
        now = time.monotonic()

        with self._lock:
            if key in self._cache:
                result, ts = self._cache[key]
                if now - ts < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return result
                else:
                    del self._cache[key]

        result = self._guard.scan(text, context)

        with self._lock:
            self._cache[key] = (result, now)
            self._misses += 1
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

        return result

    def invalidate(self, text: str | None = None) -> None:
        """Clear a specific entry or the entire cache."""
        with self._lock:
            if text is None:
                self._cache.clear()
            else:
                self._cache.pop(self._key(text), None)

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "size": len(self._cache),
                "maxsize": self._maxsize,
            }


@dataclass
class ScanMetrics:
    """Collects scanning metrics for monitoring and alerting.

    Usage:
        metrics = ScanMetrics()
        guard = SentinelGuard.default()
        result = guard.scan("text")
        metrics.record(result)
        print(metrics.summary())
    """

    total_scans: int = 0
    total_blocked: int = 0
    total_findings: int = 0
    total_latency_ms: float = 0.0
    risk_counts: dict = field(default_factory=lambda: {r.value: 0 for r in RiskLevel})
    category_counts: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, result: ScanResult) -> None:
        """Record metrics from a scan result."""
        with self._lock:
            self.total_scans += 1
            self.total_latency_ms += result.latency_ms
            self.total_findings += len(result.findings)
            self.risk_counts[result.risk.value] = (
                self.risk_counts.get(result.risk.value, 0) + 1
            )
            if result.blocked:
                self.total_blocked += 1
            for f in result.findings:
                self.category_counts[f.category] = (
                    self.category_counts.get(f.category, 0) + 1
                )

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_scans if self.total_scans > 0 else 0.0

    @property
    def block_rate(self) -> float:
        return self.total_blocked / self.total_scans if self.total_scans > 0 else 0.0

    def summary(self) -> dict:
        """Return metrics summary as a dict (Prometheus/JSON compatible)."""
        with self._lock:
            return {
                "total_scans": self.total_scans,
                "total_blocked": self.total_blocked,
                "block_rate": round(self.block_rate, 4),
                "total_findings": self.total_findings,
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "risk_distribution": dict(self.risk_counts),
                "category_distribution": dict(self.category_counts),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_scans = 0
            self.total_blocked = 0
            self.total_findings = 0
            self.total_latency_ms = 0.0
            self.risk_counts = {r.value: 0 for r in RiskLevel}
            self.category_counts = {}
