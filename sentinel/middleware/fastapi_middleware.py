"""FastAPI middleware for scanning request/response bodies."""

from __future__ import annotations

import json
from typing import Callable

from sentinel.core import SentinelGuard, ScanResult


def create_sentinel_middleware(
    guard: SentinelGuard | None = None,
    scan_request: bool = True,
    scan_response: bool = True,
    block_on_risk: bool = True,
):
    """Create a FastAPI middleware function that scans request/response bodies."""
    if guard is None:
        guard = SentinelGuard.default()

    async def sentinel_middleware(request, call_next):
        from starlette.responses import JSONResponse

        # Scan request body
        if scan_request and request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                text = body.decode("utf-8")
                input_scan = guard.scan(text)
                if block_on_risk and input_scan.blocked:
                    return JSONResponse(
                        status_code=422,
                        content={
                            "error": "Request blocked by safety scan",
                            "risk_level": input_scan.risk.value,
                            "findings": [
                                {
                                    "category": f.category,
                                    "description": f.description,
                                    "risk": f.risk.value,
                                }
                                for f in input_scan.findings
                            ],
                        },
                    )
            except Exception:
                pass

        response = await call_next(request)
        return response

    return sentinel_middleware
