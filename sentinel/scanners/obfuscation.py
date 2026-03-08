"""Obfuscation detection scanner.

Detects encoded or obfuscated payloads that may hide malicious content
from keyword-based safety filters:
- Base64-encoded attack payloads
- Hex-encoded commands
- ROT13-encoded instructions
- Unicode escape sequences
- Leetspeak variants of dangerous terms
- Zero-width character smuggling
- String concatenation building dangerous keywords
- Character code manipulation (fromCharCode / chr())
- Homoglyph attacks (Cyrillic/Greek look-alikes)

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

# Zero-width characters used for smuggling
_ZERO_WIDTH_CHARS = frozenset("\u200b\u200c\u200d\u2060\ufeff")
_ZERO_WIDTH_THRESHOLD = 5  # Minimum count to flag

# String concatenation patterns — building dangerous keywords
_CONCAT_RE = re.compile(
    r"""['"](\w{1,6})['"]\s*\+\s*['"](\w{1,6})['"]"""
    r"""(?:\s*\+\s*['"](\w{1,6})['"])?"""
    r"""(?:\s*\+\s*['"](\w{1,6})['"])?""",
)
_CONCAT_SUSPICIOUS = {
    "eval", "exec", "system", "import", "require",
    "subprocess", "password", "passwd", "secret",
}

# Character code patterns — String.fromCharCode / chr()
_FROMCHARCODE_RE = re.compile(
    r"String\.fromCharCode\s*\(\s*([\d,\s]+)\s*\)",
    re.IGNORECASE,
)
_CHR_JOIN_RE = re.compile(
    r"""(?:chr\s*\(\s*(\d+)\s*\)\s*(?:\+|,|\.)?\s*){4,}""",
)

# Homoglyph mapping — Cyrillic/Greek characters that look like Latin
_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E",
    "\u041d": "H", "\u041a": "K", "\u041c": "M", "\u041e": "O",
    "\u0420": "P", "\u0422": "T", "\u0425": "X",
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x",
    # Greek
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N",
    "\u039f": "O", "\u03a1": "P", "\u03a4": "T", "\u03a7": "X",
    "\u03b1": "a", "\u03b5": "e", "\u03bf": "o", "\u03c1": "p",
}


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
        findings.extend(self._check_zero_width(text))
        findings.extend(self._check_string_concat(text))
        findings.extend(self._check_charcode(text))
        findings.extend(self._check_homoglyphs(text))

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

    def _check_zero_width(self, text: str) -> list[Finding]:
        """Detect zero-width character smuggling."""
        count = sum(1 for c in text if c in _ZERO_WIDTH_CHARS)
        if count >= _ZERO_WIDTH_THRESHOLD:
            return [Finding(
                scanner=self.name,
                category="obfuscation",
                description=f"Zero-width character smuggling: {count} invisible characters detected",
                risk=RiskLevel.HIGH,
                span=(0, len(text)),
                metadata={"encoding": "zero_width", "count": count},
            )]
        return []

    def _check_string_concat(self, text: str) -> list[Finding]:
        """Detect string concatenation building suspicious keywords."""
        findings = []
        for m in _CONCAT_RE.finditer(text):
            parts = [g for g in m.groups() if g]
            combined = "".join(parts).lower()
            if combined in _CONCAT_SUSPICIOUS:
                findings.append(Finding(
                    scanner=self.name,
                    category="obfuscation",
                    description=f"String concatenation builds suspicious keyword: '{combined}'",
                    risk=RiskLevel.HIGH,
                    span=m.span(),
                    metadata={"encoding": "string_concat", "built_keyword": combined},
                ))
        return findings

    def _check_charcode(self, text: str) -> list[Finding]:
        """Detect String.fromCharCode / chr() building dangerous strings."""
        findings = []
        for m in _FROMCHARCODE_RE.finditer(text):
            try:
                codes = [int(c.strip()) for c in m.group(1).split(",") if c.strip()]
                decoded = "".join(chr(c) for c in codes if 0 < c < 0x10000)
                if _DANGEROUS_DECODED.search(decoded):
                    findings.append(Finding(
                        scanner=self.name,
                        category="obfuscation",
                        description=f"fromCharCode builds dangerous content: {decoded[:80]}",
                        risk=RiskLevel.HIGH,
                        span=m.span(),
                        metadata={"encoding": "charcode", "decoded": decoded[:200]},
                    ))
            except (ValueError, OverflowError):
                pass
        for m in _CHR_JOIN_RE.finditer(text):
            try:
                codes = [int(c) for c in re.findall(r"\d+", m.group())]
                decoded = "".join(chr(c) for c in codes if 0 < c < 0x10000)
                if _DANGEROUS_DECODED.search(decoded):
                    findings.append(Finding(
                        scanner=self.name,
                        category="obfuscation",
                        description=f"chr() sequence builds dangerous content: {decoded[:80]}",
                        risk=RiskLevel.HIGH,
                        span=m.span(),
                        metadata={"encoding": "charcode", "decoded": decoded[:200]},
                    ))
            except (ValueError, OverflowError):
                pass
        return findings

    def _check_homoglyphs(self, text: str) -> list[Finding]:
        """Detect Cyrillic/Greek homoglyph attacks."""
        homoglyph_count = sum(1 for c in text if c in _HOMOGLYPH_MAP)
        if homoglyph_count == 0:
            return []
        decoded = "".join(_HOMOGLYPH_MAP.get(c, c) for c in text)
        if _DANGEROUS_DECODED.search(decoded):
            return [Finding(
                scanner=self.name,
                category="obfuscation",
                description=f"Homoglyph attack: {homoglyph_count} look-alike characters hide dangerous content",
                risk=RiskLevel.HIGH,
                span=(0, len(text)),
                metadata={"encoding": "homoglyph", "count": homoglyph_count, "decoded": decoded[:200]},
            )]
        return []
