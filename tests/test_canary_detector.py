"""Tests for canary token detection."""

import pytest
from sentinel.canary_detector import CanaryDetector, CanaryResult, CanaryMatch


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------

class TestBasicDetection:
    def test_exact_match(self):
        d = CanaryDetector()
        d.add_canary("SECRET-TOKEN-123")
        result = d.check("The output contains SECRET-TOKEN-123 leaked data")
        assert result.detected
        assert result.match_count == 1
        assert result.matches[0].match_type == "exact"

    def test_no_match(self):
        d = CanaryDetector()
        d.add_canary("SECRET-TOKEN-123")
        result = d.check("This text has no canary tokens")
        assert not result.detected
        assert result.match_count == 0

    def test_multiple_canaries(self):
        d = CanaryDetector()
        d.add_canary("TOKEN-A")
        d.add_canary("TOKEN-B")
        result = d.check("Found TOKEN-A and TOKEN-B in output")
        assert result.detected
        assert result.match_count == 2

    def test_repeated_match(self):
        d = CanaryDetector()
        d.add_canary("LEAK")
        result = d.check("LEAK and another LEAK")
        assert result.match_count == 2

    def test_custom_canary_id(self):
        d = CanaryDetector()
        cid = d.add_canary("SECRET", canary_id="my_canary")
        assert cid == "my_canary"
        result = d.check("Found SECRET here")
        assert result.matches[0].canary_id == "my_canary"


# ---------------------------------------------------------------------------
# Case sensitivity
# ---------------------------------------------------------------------------

class TestCaseSensitivity:
    def test_case_sensitive_default(self):
        d = CanaryDetector()
        d.add_canary("Secret-Token")
        result = d.check("secret-token is here")
        assert not result.detected  # Case-sensitive by default

    def test_case_insensitive(self):
        d = CanaryDetector()
        d.add_canary("Secret-Token", case_sensitive=False)
        result = d.check("SECRET-TOKEN is here")
        assert result.detected
        assert result.matches[0].match_type == "case_insensitive"


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

class TestPatternMatching:
    def test_pattern_match(self):
        d = CanaryDetector()
        d.add_pattern(r"CANARY-[a-f0-9]{8}-TOKEN")
        result = d.check("Found CANARY-deadbeef-TOKEN in output")
        assert result.detected
        assert result.matches[0].match_type == "pattern"

    def test_pattern_multiple_matches(self):
        d = CanaryDetector()
        d.add_pattern(r"KEY-\d{4}")
        result = d.check("KEY-1234 and KEY-5678")
        assert result.match_count == 2

    def test_pattern_no_match(self):
        d = CanaryDetector()
        d.add_pattern(r"CANARY-[a-f0-9]{8}-TOKEN")
        result = d.check("No canary tokens here")
        assert not result.detected

    def test_pattern_custom_id(self):
        d = CanaryDetector()
        cid = d.add_pattern(r"TEST-\d+", canary_id="test_pattern")
        assert cid == "test_pattern"


# ---------------------------------------------------------------------------
# Span tracking
# ---------------------------------------------------------------------------

class TestSpanTracking:
    def test_exact_span(self):
        d = CanaryDetector()
        d.add_canary("TOKEN")
        result = d.check("Find TOKEN here")
        assert result.matches[0].span == (5, 10)

    def test_pattern_span(self):
        d = CanaryDetector()
        d.add_pattern(r"KEY-\d{4}")
        result = d.check("Value: KEY-1234")
        span = result.matches[0].span
        assert "KEY-1234" == "Value: KEY-1234"[span[0]:span[1]]


# ---------------------------------------------------------------------------
# Canary management
# ---------------------------------------------------------------------------

class TestCanaryManagement:
    def test_add_canaries_batch(self):
        d = CanaryDetector()
        ids = d.add_canaries(["A", "B", "C"])
        assert len(ids) == 3
        assert d.canary_count == 3

    def test_clear(self):
        d = CanaryDetector()
        d.add_canary("TOKEN")
        d.add_pattern(r"PATTERN")
        d.clear()
        assert d.canary_count == 0
        assert not d.check("TOKEN PATTERN").detected

    def test_canary_count(self):
        d = CanaryDetector()
        d.add_canary("A")
        d.add_canary("B")
        d.add_pattern(r"C")
        assert d.canary_count == 3


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------

class TestTokenGeneration:
    def test_generate_token(self):
        token = CanaryDetector.generate_token()
        assert token.startswith("CANARY-")
        assert token.endswith("-TOKEN")
        assert len(token) > 20

    def test_generate_custom_prefix(self):
        token = CanaryDetector.generate_token(prefix="SECRET")
        assert token.startswith("SECRET-")

    def test_generate_unique(self):
        t1 = CanaryDetector.generate_token()
        t2 = CanaryDetector.generate_token()
        assert t1 != t2


# ---------------------------------------------------------------------------
# Batch checking
# ---------------------------------------------------------------------------

class TestBatchChecking:
    def test_batch(self):
        d = CanaryDetector()
        d.add_canary("LEAK")
        results = d.check_batch(["No leak here", "Found LEAK", "Also LEAK"])
        assert len(results) == 3
        assert not results[0].detected
        assert results[1].detected
        assert results[2].detected

    def test_empty_batch(self):
        d = CanaryDetector()
        results = d.check_batch([])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# CanaryResult properties
# ---------------------------------------------------------------------------

class TestCanaryResult:
    def test_canary_ids(self):
        d = CanaryDetector()
        d.add_canary("A", canary_id="id_a")
        d.add_canary("B", canary_id="id_b")
        result = d.check("Found A and B")
        assert set(result.canary_ids) == {"id_a", "id_b"}

    def test_checked_at(self):
        import time
        before = time.time()
        d = CanaryDetector()
        result = d.check("test")
        after = time.time()
        assert before <= result.checked_at <= after


# ---------------------------------------------------------------------------
# Confidence scores
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_exact_high_confidence(self):
        d = CanaryDetector()
        d.add_canary("TOKEN")
        result = d.check("TOKEN")
        assert result.matches[0].confidence == 1.0

    def test_case_insensitive_lower_confidence(self):
        d = CanaryDetector()
        d.add_canary("Token", case_sensitive=False)
        result = d.check("token")
        ci_matches = [m for m in result.matches if m.match_type == "case_insensitive"]
        if ci_matches:
            assert ci_matches[0].confidence < 1.0

    def test_pattern_lower_confidence(self):
        d = CanaryDetector()
        d.add_pattern(r"KEY-\d+")
        result = d.check("KEY-123")
        assert result.matches[0].confidence < 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text(self):
        d = CanaryDetector()
        d.add_canary("TOKEN")
        result = d.check("")
        assert not result.detected

    def test_no_canaries(self):
        d = CanaryDetector()
        result = d.check("Any text")
        assert not result.detected

    def test_token_at_boundaries(self):
        d = CanaryDetector()
        d.add_canary("TOKEN")
        assert d.check("TOKEN").detected
        assert d.check("TOKEN at start").detected
        assert d.check("at end TOKEN").detected

    def test_unicode_canary(self):
        d = CanaryDetector()
        d.add_canary("CANARY-ñ-TOKEN")
        result = d.check("Found CANARY-ñ-TOKEN here")
        assert result.detected
