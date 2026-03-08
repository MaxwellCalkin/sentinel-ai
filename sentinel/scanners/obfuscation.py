"""Obfuscation detection scanner.

Detects encoded or obfuscated payloads that may hide malicious content
from keyword-based safety filters:
- Base64-encoded attack payloads
- Hex-encoded commands
- ROT13-encoded instructions
- Unicode escape sequences
- Leetspeak variants of dangerous terms

This is a deterministic first-pass filter that catches common obfuscation
techniques at sub-millisecond latency.
"""

from __future__ import annotations

import base64
import codecs
import re
from sentinel.core import Finding, RiskLevel


# Dangerous terms to detect in decoded content
_DANGEROUS_DECODED = re.compile(
    r"(?i)"
    r"(ignore\s+(all\s+)?instructions|system\s+prompt|"
    r"rm\s+-rf|drop\s+table|delete\s+from|exec\s*\(|eval\s*\(|"
    r"subprocess|os\.system|__import__|"
    r"curl\s+.*\|\s*(sh|bash)|wget\s+.*\|\s*(sh|bash)|"
    r"/etc/passwd|/etc/shadow|\.ssh/|"
    r"password|secret.?key|api.?key|access.?token)"
)

# Minimum length for base64 strings to avoid false positives on short words
_MIN_B64_LEN = 16

# Base64 pattern — continuous base64 chars with optional padding
_BASE64_RE = re.compile(
    r"(?<![A-Za-z0-9+/=])"  # not preceded by base64 chars
    r"([A-Za-z0-9+/]{" + str(_MIN_B64_LEN) + r",}={0,2})"
    r"(?![A-Za-z0-9+/=])"  # not followed by base64 chars
)

# Hex-encoded string pattern (\x41\x42 or 0x41 0x42 or 41 42 43...)
_HEX_ESCAPE_RE = re.compile(
    r"((?:\\x[0-9a-fA-F]{2}){4,})"  # \x41\x42\x43\x44...
)
_HEX_0X_RE = re.compile(
    r"((?:0x[0-9a-fA-F]{2}\s*){4,})"  # 0x41 0x42 0x43...
)

# ROT13 indicators — text explicitly mentioning rot13 encoding
_ROT13_INDICATOR_RE = re.compile(
    r"(?i)(rot13|rot-13|rotate\s*13)\s*[:=]?\s*([A-Za-z\s]{8,})"
)

# Unicode escape sequences
_UNICODE_ESCAPE_RE = re.compile(
    r"((?:\\u[0-9a-fA-F]{4}){4,})"  # \u0069\u0067\u006e...
)

# Leetspeak mappings for detection
_LEET_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "8": "b", "@": "a", "$": "s", "!": "i",
}

# Dangerous terms to detect in leetspeak (lowercase)
_LEET_TARGETS = [
    "ignore instructions",
    "system prompt",
    "drop table",
    "delete from",
    "exec(",
    "eval(",
    "rm -rf",
    "passwd",
    "hack",
    "exploit",
    "bypass filter",
    "bypass safety",
]


def _decode_leet(text: str) -> str:
    """Convert leetspeak to regular text."""
    result = []
    for ch in text.lower():
        result.append(_LEET_MAP.get(ch, ch))
    return "".join(result)


def _try_base64_decode(s: str) -> str | None:
    """Attempt to decode a base64 string. Returns decoded text or None."""
    try:
        # Add padding if needed
        padded = s + "=" * (4 - len(s) % 4) if len(s) % 4 else s
        decoded = base64.b64decode(padded, validate=True)
        # Check if result is valid text
        text = decoded.decode("utf-8")
        # Ensure it's mostly printable (not random binary)
        printable_ratio = sum(1 for c in text if c.isprintable() or c.isspace()) / max(len(text), 1)
        if printable_ratio >= 0.8:
            return text
    except Exception:
        pass
    return None


def _try_hex_decode(hex_str: str) -> str | None:
    """Decode hex escape sequences to text."""
    try:
        # Extract hex bytes
        hex_bytes = re.findall(r"[0-9a-fA-F]{2}", hex_str)
        if len(hex_bytes) < 4:
            return None
        raw = bytes(int(b, 16) for b in hex_bytes)
        text = raw.decode("utf-8")
        printable_ratio = sum(1 for c in text if c.isprintable() or c.isspace()) / max(len(text), 1)
        if printable_ratio >= 0.8:
            return text
    except Exception:
        pass
    return None


def _try_unicode_decode(escaped: str) -> str | None:
    """Decode unicode escape sequences."""
    try:
        # Replace \\u with \u for codec
        text = escaped.encode("utf-8").decode("unicode_escape")
        printable_ratio = sum(1 for c in text if c.isprintable() or c.isspace()) / max(len(text), 1)
        if printable_ratio >= 0.8:
            return text
    except Exception:
        pass
    return None


class ObfuscationScanner:
    """Detects encoded/obfuscated payloads hiding malicious content."""

    name = "obfuscation"

    def scan(self, text: str, context: dict | None = None) -> list[Finding]:
        findings: list[Finding] = []

        findings.extend(self._check_base64(text))
        findings.extend(self._check_hex(text))
        findings.extend(self._check_rot13(text))
        findings.extend(self._check_unicode_escapes(text))
        findings.extend(self._check_leetspeak(text))

        return findings

    def _check_base64(self, text: str) -> list[Finding]:
        findings = []
        for m in _BASE64_RE.finditer(text):
            encoded = m.group(1)
            decoded = _try_base64_decode(encoded)
            if decoded and _DANGEROUS_DECODED.search(decoded):
                findings.append(Finding(
                    scanner=self.name,
                    category="obfuscation",
                    description=f"Base64-encoded payload contains dangerous content: {decoded[:80]}",
                    risk=RiskLevel.HIGH,
                    span=m.span(),
                    metadata={"encoding": "base64", "decoded": decoded[:200]},
                ))
        return findings

    def _check_hex(self, text: str) -> list[Finding]:
        findings = []
        for pattern in (_HEX_ESCAPE_RE, _HEX_0X_RE):
            for m in pattern.finditer(text):
                decoded = _try_hex_decode(m.group(1))
                if decoded and _DANGEROUS_DECODED.search(decoded):
                    findings.append(Finding(
                        scanner=self.name,
                        category="obfuscation",
                        description=f"Hex-encoded payload contains dangerous content: {decoded[:80]}",
                        risk=RiskLevel.HIGH,
                        span=m.span(),
                        metadata={"encoding": "hex", "decoded": decoded[:200]},
                    ))
        return findings

    def _check_rot13(self, text: str) -> list[Finding]:
        findings = []
        for m in _ROT13_INDICATOR_RE.finditer(text):
            encoded_text = m.group(2)
            decoded = codecs.decode(encoded_text, "rot_13")
            if _DANGEROUS_DECODED.search(decoded):
                findings.append(Finding(
                    scanner=self.name,
                    category="obfuscation",
                    description=f"ROT13-encoded payload contains dangerous content: {decoded[:80]}",
                    risk=RiskLevel.HIGH,
                    span=m.span(),
                    metadata={"encoding": "rot13", "decoded": decoded[:200]},
                ))
        return findings

    def _check_unicode_escapes(self, text: str) -> list[Finding]:
        findings = []
        for m in _UNICODE_ESCAPE_RE.finditer(text):
            decoded = _try_unicode_decode(m.group(1))
            if decoded and _DANGEROUS_DECODED.search(decoded):
                findings.append(Finding(
                    scanner=self.name,
                    category="obfuscation",
                    description=f"Unicode-escaped payload contains dangerous content: {decoded[:80]}",
                    risk=RiskLevel.HIGH,
                    span=m.span(),
                    metadata={"encoding": "unicode_escape", "decoded": decoded[:200]},
                ))
        return findings

    def _check_leetspeak(self, text: str) -> list[Finding]:
        findings = []
        decoded = _decode_leet(text)
        for target in _LEET_TARGETS:
            if target in decoded and target not in text.lower():
                # Only flag if the leetspeak conversion revealed something
                # that wasn't visible in the original text
                idx = decoded.index(target)
                findings.append(Finding(
                    scanner=self.name,
                    category="obfuscation",
                    description=f"Leetspeak obfuscation detected: '{target}' hidden in text",
                    risk=RiskLevel.MEDIUM,
                    span=(idx, idx + len(target)),
                    metadata={"encoding": "leetspeak", "decoded_term": target},
                ))
                break  # One finding per text for leetspeak
        return findings
