"""Sentinel AI API server.

Run with: uvicorn sentinel.api:app
Or:       python -m sentinel.api
"""

import time
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentinel.core import SentinelGuard


class ScanRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    context: Optional[dict[str, Any]] = None
    scanners: Optional[list[str]] = Field(
        None,
        description="Filter to specific scanners: prompt_injection, pii, harmful_content, hallucination",
    )


class FindingResponse(BaseModel):
    scanner: str
    category: str
    description: str
    risk: str
    span: Optional[list[int]] = None
    metadata: dict[str, Any] = {}


class ScanResponse(BaseModel):
    safe: bool
    blocked: bool
    risk: str
    findings: list[FindingResponse]
    redacted_text: Optional[str] = None
    latency_ms: float


class BatchScanRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchScanResponse(BaseModel):
    results: list[ScanResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    scanners: list[str]


def create_app(guard: Optional[SentinelGuard] = None) -> FastAPI:
    if guard is None:
        guard = SentinelGuard.default()

    app = FastAPI(
        title="Sentinel AI",
        description="Real-time AI safety guardrails API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            version="0.1.0",
            scanners=[s.name for s in guard._scanners],
        )

    @app.get("/metrics")
    async def metrics():
        from sentinel.telemetry import get_metrics
        return get_metrics()

    @app.post("/scan", response_model=ScanResponse)
    async def scan(req: ScanRequest):
        result = guard.scan(req.text, req.context)
        findings = []
        for f in result.findings:
            if req.scanners and f.scanner not in req.scanners:
                continue
            findings.append(
                FindingResponse(
                    scanner=f.scanner,
                    category=f.category,
                    description=f.description,
                    risk=f.risk.value,
                    span=list(f.span) if f.span else None,
                    metadata=f.metadata,
                )
            )
        return ScanResponse(
            safe=result.safe,
            blocked=result.blocked,
            risk=result.risk.value,
            findings=findings,
            redacted_text=result.redacted_text,
            latency_ms=result.latency_ms,
        )

    @app.post("/scan/batch", response_model=BatchScanResponse)
    async def scan_batch(req: BatchScanRequest):
        start = time.perf_counter()
        results = []
        for text in req.texts:
            result = guard.scan(text)
            findings = [
                FindingResponse(
                    scanner=f.scanner,
                    category=f.category,
                    description=f.description,
                    risk=f.risk.value,
                    span=list(f.span) if f.span else None,
                    metadata=f.metadata,
                )
                for f in result.findings
            ]
            results.append(
                ScanResponse(
                    safe=result.safe,
                    blocked=result.blocked,
                    risk=result.risk.value,
                    findings=findings,
                    redacted_text=result.redacted_text,
                    latency_ms=result.latency_ms,
                )
            )
        total = (time.perf_counter() - start) * 1000
        return BatchScanResponse(results=results, total_latency_ms=round(total, 2))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sentinel.api:app", host="0.0.0.0", port=8329, reload=True)
