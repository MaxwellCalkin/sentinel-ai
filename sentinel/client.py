"""Python SDK client for the hosted Sentinel AI API.

Usage:
    from sentinel.client import SentinelClient

    client = SentinelClient("https://api.sentinel-ai.dev", api_key="sk-...")
    result = client.scan("Check this text for safety issues")
    if not result["safe"]:
        print(result["findings"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SentinelClient:
    """HTTP client for the Sentinel AI hosted API."""

    base_url: str = "http://localhost:8329"
    api_key: str | None = None
    timeout: float = 30.0

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def scan(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        scanners: list[str] | None = None,
    ) -> dict[str, Any]:
        import httpx

        payload: dict[str, Any] = {"text": text}
        if context:
            payload["context"] = context
        if scanners:
            payload["scanners"] = scanners

        resp = httpx.post(
            f"{self.base_url}/scan",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def scan_batch(self, texts: list[str]) -> dict[str, Any]:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/scan/batch",
            json={"texts": texts},
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict[str, Any]:
        import httpx

        resp = httpx.get(
            f"{self.base_url}/health",
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    async def scan_async(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        scanners: list[str] | None = None,
    ) -> dict[str, Any]:
        import httpx

        payload: dict[str, Any] = {"text": text}
        if context:
            payload["context"] = context
        if scanners:
            payload["scanners"] = scanners

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/scan",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

    async def scan_batch_async(self, texts: list[str]) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/scan/batch",
                json={"texts": texts},
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
