"""Webhook and callback system for enterprise alerting.

Triggers notifications when scans detect issues above a threshold.
Supports HTTP webhooks (Slack, PagerDuty, etc.) and custom callbacks.

Usage:
    from sentinel.webhooks import WebhookGuard, WebhookConfig

    config = WebhookConfig(
        url="https://hooks.slack.com/services/...",
        trigger_on=RiskLevel.HIGH,
    )
    guard = WebhookGuard(webhook=config)
    result = guard.scan("some text")  # fires webhook if HIGH+ risk
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from sentinel.core import SentinelGuard, ScanResult, RiskLevel

logger = logging.getLogger("sentinel.webhooks")


@dataclass
class WebhookConfig:
    url: str
    trigger_on: RiskLevel = RiskLevel.HIGH
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0
    include_text: bool = False  # whether to include scanned text in payload
    format: str = "default"  # "default", "slack", "pagerduty"


def _build_payload(
    result: ScanResult,
    config: WebhookConfig,
) -> dict[str, Any]:
    """Build webhook payload based on format."""
    findings_summary = [
        {
            "scanner": f.scanner,
            "category": f.category,
            "description": f.description,
            "risk": f.risk.value,
        }
        for f in result.findings
    ]

    base = {
        "event": "sentinel.scan.alert",
        "timestamp": time.time(),
        "risk": result.risk.value,
        "blocked": result.blocked,
        "findings_count": len(result.findings),
        "findings": findings_summary,
        "latency_ms": result.latency_ms,
    }

    if config.include_text:
        base["text"] = result.redacted_text or result.text

    if config.format == "slack":
        risk_emoji = {
            "critical": ":rotating_light:",
            "high": ":warning:",
            "medium": ":large_yellow_circle:",
            "low": ":information_source:",
        }
        emoji = risk_emoji.get(result.risk.value, ":question:")
        finding_lines = "\n".join(
            f"  - [{f['risk'].upper()}] {f['description']}" for f in findings_summary
        )
        return {
            "text": (
                f"{emoji} *Sentinel AI Alert* - Risk: {result.risk.value.upper()}\n"
                f"Blocked: {result.blocked} | Findings: {len(result.findings)}\n"
                f"{finding_lines}"
            )
        }

    if config.format == "pagerduty":
        return {
            "routing_key": config.headers.get("routing_key", ""),
            "event_action": "trigger",
            "payload": {
                "summary": f"Sentinel AI: {result.risk.value.upper()} risk detected ({len(result.findings)} findings)",
                "severity": "critical" if result.blocked else "warning",
                "source": "sentinel-ai",
                "custom_details": base,
            },
        }

    return base


CallbackFn = Callable[[ScanResult], None]


class WebhookGuard:
    """SentinelGuard wrapper that fires webhooks/callbacks on findings."""

    def __init__(
        self,
        guard: SentinelGuard | None = None,
        webhook: WebhookConfig | None = None,
        webhooks: list[WebhookConfig] | None = None,
        callbacks: list[CallbackFn] | None = None,
    ):
        self._guard = guard or SentinelGuard.default()
        self._webhooks: list[WebhookConfig] = []
        if webhook:
            self._webhooks.append(webhook)
        if webhooks:
            self._webhooks.extend(webhooks)
        self._callbacks: list[CallbackFn] = list(callbacks or [])

    def add_webhook(self, config: WebhookConfig) -> WebhookGuard:
        self._webhooks.append(config)
        return self

    def add_callback(self, fn: CallbackFn) -> WebhookGuard:
        self._callbacks.append(fn)
        return self

    def scan(self, text: str, context: dict | None = None) -> ScanResult:
        result = self._guard.scan(text, context)
        self._fire(result)
        return result

    def _fire(self, result: ScanResult) -> None:
        # Fire callbacks (always sync, in-process)
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.error("Callback error: %s", e)

        # Fire webhooks for qualifying results
        for wh in self._webhooks:
            if result.risk >= wh.trigger_on:
                self._send_webhook(wh, result)

    def _send_webhook(self, config: WebhookConfig, result: ScanResult) -> None:
        payload = _build_payload(result, config)
        try:
            import httpx

            headers = {"Content-Type": "application/json", **config.headers}
            resp = httpx.post(
                config.url,
                json=payload,
                headers=headers,
                timeout=config.timeout,
            )
            logger.info("Webhook sent to %s: %s", config.url, resp.status_code)
        except Exception as e:
            logger.error("Webhook failed for %s: %s", config.url, e)
