"""Canary token system for detecting LLM prompt leakage and manipulation.

Plant invisible canary tokens in system prompts, context, or documents.
If the canary appears in the model's output, it means the system prompt
or context has been leaked — indicating a successful prompt injection attack.

Usage:
    from sentinel.canary import CanarySystem

    canary = CanarySystem()

    # Generate a canary token and embed it in your system prompt
    token = canary.create_token("my-app-system-prompt")
    system_prompt = f"You are a helpful assistant. {token.invisible_marker}"

    # Later, scan model output for leaked canaries
    output = model.generate(...)
    leaks = canary.scan_output(output)
    if leaks:
        print("ALERT: System prompt leaked!", leaks)

    # Or use the zero-width character canary for truly invisible embedding
    token = canary.create_token("secret-context", style="zero-width")
    document = f"Confidential report: {token.invisible_marker} Revenue was $10M"
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Literal

from sentinel.core import Finding, RiskLevel


@dataclass
class CanaryToken:
    """A canary token that can be embedded in prompts or documents."""
    name: str
    token_id: str
    visible_marker: str
    invisible_marker: str
    style: str
    created_at: float = field(default_factory=time.time)

    @property
    def marker(self) -> str:
        """Returns the invisible marker for embedding."""
        return self.invisible_marker


# Zero-width characters used for encoding
_ZW_CHARS = [
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space
]


def _encode_to_zero_width(data: str) -> str:
    """Encode a string into zero-width characters (binary encoding)."""
    bits = "".join(format(byte, "08b") for byte in data.encode("utf-8"))
    return "".join(_ZW_CHARS[int(bits[i:i+2], 2)] for i in range(0, len(bits), 2))


def _decode_from_zero_width(text: str) -> str | None:
    """Decode zero-width characters back to a string."""
    zw_set = set(_ZW_CHARS)
    zw_chars = [c for c in text if c in zw_set]
    if not zw_chars:
        return None
    char_map = {c: str(i) for i, c in enumerate(_ZW_CHARS)}
    bits = ""
    for c in zw_chars:
        val = int(char_map[c])
        bits += format(val, "02b")
    # Pad to byte boundary
    remainder = len(bits) % 8
    if remainder:
        bits = bits[:len(bits) - remainder]
    if not bits:
        return None
    try:
        byte_array = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
        return byte_array.decode("utf-8", errors="replace")
    except (ValueError, UnicodeDecodeError):
        return None


class CanarySystem:
    """Create and detect canary tokens for prompt leakage detection.

    Canary tokens are invisible markers embedded in system prompts,
    context documents, or other privileged content. If they appear in
    model output, it indicates the content has been leaked.

    Supports two styles:
    - "comment": HTML comment markers (e.g., <!-- CANARY:abc123 -->)
    - "zero-width": Zero-width Unicode characters (truly invisible)
    """

    def __init__(self) -> None:
        self._tokens: dict[str, CanaryToken] = {}

    def create_token(
        self,
        name: str,
        style: Literal["comment", "zero-width"] = "comment",
    ) -> CanaryToken:
        """Create a new canary token.

        Args:
            name: Human-readable name for this canary (e.g., "system-prompt").
            style: Embedding style — "comment" for HTML comments,
                   "zero-width" for invisible Unicode.

        Returns:
            CanaryToken with markers for embedding.
        """
        token_id = secrets.token_hex(8)
        payload = f"SENTINEL_CANARY:{token_id}"

        if style == "comment":
            visible_marker = f"<!-- {payload} -->"
            invisible_marker = f"<!-- {payload} -->"
        elif style == "zero-width":
            visible_marker = f"[CANARY:{token_id}]"
            invisible_marker = _encode_to_zero_width(payload)
        else:
            raise ValueError(f"Unknown style: {style}")

        token = CanaryToken(
            name=name,
            token_id=token_id,
            visible_marker=visible_marker,
            invisible_marker=invisible_marker,
            style=style,
        )
        self._tokens[token_id] = token
        return token

    def scan_output(self, text: str) -> list[Finding]:
        """Scan model output for leaked canary tokens.

        Args:
            text: The model's output text to scan.

        Returns:
            List of findings for any detected canary leaks.
        """
        findings: list[Finding] = []

        # Check for comment-style canaries
        comment_pattern = re.compile(r"<!--\s*SENTINEL_CANARY:([a-f0-9]+)\s*-->")
        for match in comment_pattern.finditer(text):
            token_id = match.group(1)
            token = self._tokens.get(token_id)
            name = token.name if token else "unknown"
            findings.append(Finding(
                scanner="canary",
                category="prompt_leak",
                description=f"Canary token leaked in output: '{name}' (id: {token_id})",
                risk=RiskLevel.CRITICAL,
                span=(match.start(), match.end()),
                metadata={
                    "token_id": token_id,
                    "canary_name": name,
                    "style": "comment",
                    "leak_type": "system_prompt_leak",
                },
            ))

        # Check for plain text canary references
        plain_pattern = re.compile(r"SENTINEL_CANARY:([a-f0-9]+)")
        for match in plain_pattern.finditer(text):
            # Skip if already caught by comment pattern
            if any(f.metadata.get("token_id") == match.group(1) for f in findings):
                continue
            token_id = match.group(1)
            token = self._tokens.get(token_id)
            name = token.name if token else "unknown"
            findings.append(Finding(
                scanner="canary",
                category="prompt_leak",
                description=f"Canary token leaked in output: '{name}' (id: {token_id})",
                risk=RiskLevel.CRITICAL,
                span=(match.start(), match.end()),
                metadata={
                    "token_id": token_id,
                    "canary_name": name,
                    "style": "plain",
                    "leak_type": "system_prompt_leak",
                },
            ))

        # Check for zero-width encoded canaries
        zw_set = set(_ZW_CHARS)
        if any(c in zw_set for c in text):
            decoded = _decode_from_zero_width(text)
            if decoded and "SENTINEL_CANARY:" in decoded:
                zw_match = re.search(r"SENTINEL_CANARY:([a-f0-9]+)", decoded)
                if zw_match:
                    token_id = zw_match.group(1)
                    token = self._tokens.get(token_id)
                    name = token.name if token else "unknown"
                    findings.append(Finding(
                        scanner="canary",
                        category="prompt_leak",
                        description=f"Zero-width canary token leaked: '{name}' (id: {token_id})",
                        risk=RiskLevel.CRITICAL,
                        metadata={
                            "token_id": token_id,
                            "canary_name": name,
                            "style": "zero-width",
                            "leak_type": "system_prompt_leak",
                        },
                    ))

        return findings

    def scan_for_any_canary(self, text: str) -> list[Finding]:
        """Scan for ANY canary token pattern, even ones not created by this instance.

        Useful for detecting if someone else planted canaries, or for
        scanning output from a system you didn't instrument.
        """
        findings: list[Finding] = []

        # Check for comment-style
        for match in re.finditer(r"<!--\s*SENTINEL_CANARY:([a-f0-9]+)\s*-->", text):
            findings.append(Finding(
                scanner="canary",
                category="prompt_leak",
                description=f"Sentinel canary token detected (id: {match.group(1)})",
                risk=RiskLevel.CRITICAL,
                span=(match.start(), match.end()),
                metadata={"token_id": match.group(1), "style": "comment"},
            ))

        # Check for plain text
        for match in re.finditer(r"SENTINEL_CANARY:([a-f0-9]+)", text):
            if not any(f.metadata.get("token_id") == match.group(1) for f in findings):
                findings.append(Finding(
                    scanner="canary",
                    category="prompt_leak",
                    description=f"Sentinel canary token detected (id: {match.group(1)})",
                    risk=RiskLevel.CRITICAL,
                    span=(match.start(), match.end()),
                    metadata={"token_id": match.group(1), "style": "plain"},
                ))

        return findings

    @property
    def tokens(self) -> dict[str, CanaryToken]:
        return dict(self._tokens)

    def get_token(self, token_id: str) -> CanaryToken | None:
        return self._tokens.get(token_id)
