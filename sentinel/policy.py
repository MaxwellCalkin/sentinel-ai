"""Configurable safety policies via YAML or dict.

Allows enterprises to define custom guardrail policies:
- Which scanners to enable/disable
- Risk thresholds per scanner
- Custom blocked terms
- Allow-lists for known-safe patterns
- PII redaction preferences

Example policy YAML:
    version: "1"
    name: "production-strict"
    block_threshold: high
    redact_pii: true
    scanners:
      prompt_injection:
        enabled: true
        block_threshold: medium
      pii:
        enabled: true
        redact: true
        allow_patterns:
          - "support@company.com"
      harmful_content:
        enabled: true
      hallucination:
        enabled: false
    custom_blocked_terms:
      - "competitor_name"
      - "internal_project_codename"
    allow_list:
      - "known safe phrase"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentinel.core import Finding, RiskLevel, SentinelGuard


@dataclass
class ScannerPolicy:
    enabled: bool = True
    block_threshold: RiskLevel | None = None
    allow_patterns: list[str] = field(default_factory=list)
    extra_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Policy:
    name: str = "default"
    version: str = "1"
    block_threshold: RiskLevel = RiskLevel.HIGH
    redact_pii: bool = True
    scanner_policies: dict[str, ScannerPolicy] = field(default_factory=dict)
    custom_blocked_terms: list[str] = field(default_factory=list)
    allow_list: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        risk_map = {r.value: r for r in RiskLevel}
        block_threshold = risk_map.get(
            data.get("block_threshold", "high"), RiskLevel.HIGH
        )

        scanner_policies = {}
        for name, cfg in data.get("scanners", {}).items():
            sp_threshold = None
            if "block_threshold" in cfg:
                sp_threshold = risk_map.get(cfg["block_threshold"])
            scanner_policies[name] = ScannerPolicy(
                enabled=cfg.get("enabled", True),
                block_threshold=sp_threshold,
                allow_patterns=cfg.get("allow_patterns", []),
                extra_config={
                    k: v
                    for k, v in cfg.items()
                    if k not in ("enabled", "block_threshold", "allow_patterns")
                },
            )

        return cls(
            name=data.get("name", "default"),
            version=data.get("version", "1"),
            block_threshold=block_threshold,
            redact_pii=data.get("redact_pii", True),
            scanner_policies=scanner_policies,
            custom_blocked_terms=data.get("custom_blocked_terms", []),
            allow_list=data.get("allow_list", []),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Policy:
        """Load policy from a YAML file. Requires PyYAML."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def build_guard(self) -> SentinelGuard:
        """Construct a SentinelGuard configured according to this policy."""
        from sentinel.scanners.prompt_injection import PromptInjectionScanner
        from sentinel.scanners.pii import PIIScanner
        from sentinel.scanners.harmful_content import HarmfulContentScanner
        from sentinel.scanners.hallucination import HallucinationScanner
        from sentinel.scanners.blocked_terms import BlockedTermsScanner

        scanner_map = {
            "prompt_injection": PromptInjectionScanner,
            "pii": PIIScanner,
            "harmful_content": HarmfulContentScanner,
            "hallucination": HallucinationScanner,
        }

        scanners = []
        for name, scanner_cls in scanner_map.items():
            sp = self.scanner_policies.get(name, ScannerPolicy())
            if sp.enabled:
                scanners.append(scanner_cls())

        if self.custom_blocked_terms:
            scanners.append(BlockedTermsScanner(terms=self.custom_blocked_terms))

        guard = SentinelGuard(
            scanners=scanners,
            block_threshold=self.block_threshold,
            redact_pii=self.redact_pii,
        )

        if self.allow_list:
            guard._allow_list = self.allow_list

        return guard
