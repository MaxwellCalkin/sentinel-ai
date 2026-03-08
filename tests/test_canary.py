"""Tests for the canary token system."""

from sentinel.canary import CanarySystem, CanaryToken, _encode_to_zero_width, _decode_from_zero_width
from sentinel.core import RiskLevel


class TestCanaryTokenCreation:
    def test_create_comment_token(self):
        system = CanarySystem()
        token = system.create_token("test-prompt")
        assert token.name == "test-prompt"
        assert token.style == "comment"
        assert "SENTINEL_CANARY:" in token.visible_marker
        assert token.token_id in token.visible_marker

    def test_create_zero_width_token(self):
        system = CanarySystem()
        token = system.create_token("secret-doc", style="zero-width")
        assert token.name == "secret-doc"
        assert token.style == "zero-width"
        # Invisible marker should be zero-width chars only
        assert len(token.invisible_marker) > 0
        # Should not contain visible ASCII
        for c in token.invisible_marker:
            assert ord(c) > 127 or c in "\u200b\u200c\u200d\ufeff"

    def test_unique_token_ids(self):
        system = CanarySystem()
        t1 = system.create_token("a")
        t2 = system.create_token("b")
        assert t1.token_id != t2.token_id

    def test_token_stored_in_system(self):
        system = CanarySystem()
        token = system.create_token("my-canary")
        assert system.get_token(token.token_id) is token
        assert token.token_id in system.tokens

    def test_marker_property(self):
        system = CanarySystem()
        token = system.create_token("test")
        assert token.marker == token.invisible_marker


class TestCommentCanaryDetection:
    def test_detect_leaked_comment_canary(self):
        system = CanarySystem()
        token = system.create_token("system-prompt")
        output = f"Here is the system prompt: {token.visible_marker}"
        findings = system.scan_output(output)
        assert len(findings) == 1
        assert findings[0].category == "prompt_leak"
        assert findings[0].risk == RiskLevel.CRITICAL
        assert findings[0].metadata["canary_name"] == "system-prompt"

    def test_no_detection_on_clean_output(self):
        system = CanarySystem()
        system.create_token("unused")
        findings = system.scan_output("The capital of France is Paris.")
        assert len(findings) == 0

    def test_detect_plain_text_canary(self):
        system = CanarySystem()
        token = system.create_token("test")
        # Simulating a model that strips HTML comments but leaves the text
        output = f"Found: SENTINEL_CANARY:{token.token_id}"
        findings = system.scan_output(output)
        assert len(findings) == 1
        assert findings[0].metadata["style"] == "plain"

    def test_no_duplicate_findings(self):
        system = CanarySystem()
        token = system.create_token("test")
        # Output has both comment and plain text form
        output = f"<!-- SENTINEL_CANARY:{token.token_id} --> and SENTINEL_CANARY:{token.token_id}"
        findings = system.scan_output(output)
        # Should deduplicate by token_id
        token_ids = [f.metadata["token_id"] for f in findings]
        # The comment one is found first, plain text should be skipped
        assert token_ids.count(token.token_id) == 1

    def test_detect_multiple_different_canaries(self):
        system = CanarySystem()
        t1 = system.create_token("prompt-a")
        t2 = system.create_token("prompt-b")
        output = f"Leaked: {t1.visible_marker} and also {t2.visible_marker}"
        findings = system.scan_output(output)
        assert len(findings) == 2


class TestZeroWidthCanaryDetection:
    def test_encode_decode_roundtrip(self):
        original = "SENTINEL_CANARY:abc123"
        encoded = _encode_to_zero_width(original)
        decoded = _decode_from_zero_width(encoded)
        assert decoded is not None
        assert original in decoded or decoded.startswith(original[:10])

    def test_detect_zero_width_canary(self):
        system = CanarySystem()
        token = system.create_token("secret", style="zero-width")
        # Simulate output containing the zero-width encoded canary
        output = f"Normal text {token.invisible_marker} more text"
        findings = system.scan_output(output)
        assert len(findings) >= 1
        assert any(f.metadata.get("style") == "zero-width" for f in findings)

    def test_no_false_positive_on_few_zw_chars(self):
        system = CanarySystem()
        system.create_token("test", style="zero-width")
        # Just a couple zero-width chars shouldn't trigger
        output = "Normal\u200btext"
        findings = system.scan_output(output)
        assert len(findings) == 0


class TestScanForAnyCanary:
    def test_detect_unknown_canary(self):
        system = CanarySystem()
        # Don't create a token, just scan for any canary pattern
        output = "The system said: <!-- SENTINEL_CANARY:deadbeef12345678 -->"
        findings = system.scan_for_any_canary(output)
        assert len(findings) == 1
        assert findings[0].metadata["token_id"] == "deadbeef12345678"

    def test_detect_plain_unknown_canary(self):
        system = CanarySystem()
        output = "SENTINEL_CANARY:abcd1234abcd1234 was in the output"
        findings = system.scan_for_any_canary(output)
        assert len(findings) == 1

    def test_no_false_positive(self):
        system = CanarySystem()
        output = "This is a normal response with no canaries."
        findings = system.scan_for_any_canary(output)
        assert len(findings) == 0


class TestCanaryIntegration:
    def test_embed_in_system_prompt(self):
        system = CanarySystem()
        token = system.create_token("my-app")
        prompt = f"You are a helpful assistant. {token.invisible_marker} Answer questions clearly."
        assert "SENTINEL_CANARY" in prompt

    def test_full_workflow(self):
        """End-to-end: create token, embed in prompt, detect in output."""
        system = CanarySystem()
        token = system.create_token("production-prompt")

        # Embed in system prompt
        system_prompt = f"You are helpful. {token.marker} Be concise."

        # Simulate model leaking the prompt
        model_output = f"My system prompt says: {system_prompt}"

        # Detect the leak
        findings = system.scan_output(model_output)
        assert len(findings) >= 1
        assert findings[0].category == "prompt_leak"
        assert findings[0].metadata["canary_name"] == "production-prompt"

    def test_safe_output_no_leak(self):
        system = CanarySystem()
        system.create_token("production-prompt")
        model_output = "The capital of France is Paris. How can I help you?"
        findings = system.scan_output(model_output)
        assert len(findings) == 0
