"""Tests for model fingerprinting."""

import pytest
from sentinel.model_fingerprint import ModelFingerprint, FingerprintResult


# ---------------------------------------------------------------------------
# Claude detection
# ---------------------------------------------------------------------------

class TestClaudeDetection:
    def test_claude_markers(self):
        fp = ModelFingerprint()
        text = "I'd be happy to help! Let me break this down for you. I should note that this is complex."
        result = fp.analyze(text)
        assert result.likely_model == "claude"
        assert result.confidence > 0.3

    def test_claude_appreciation(self):
        fp = ModelFingerprint()
        text = "I appreciate you sharing this. Here's my analysis of the situation."
        result = fp.analyze(text)
        assert result.scores["claude"] > 0


# ---------------------------------------------------------------------------
# GPT detection
# ---------------------------------------------------------------------------

class TestGPTDetection:
    def test_gpt_markers(self):
        fp = ModelFingerprint()
        text = "Certainly! As an AI, my knowledge cutoff means I may not have the latest information."
        result = fp.analyze(text)
        assert result.likely_model == "gpt"
        assert result.confidence > 0.3

    def test_gpt_vocabulary(self):
        fp = ModelFingerprint()
        text = "Let us delve into this tapestry of ideas and navigate the landscape together."
        result = fp.analyze(text)
        assert result.scores["gpt"] > 0


# ---------------------------------------------------------------------------
# AI detection (generic)
# ---------------------------------------------------------------------------

class TestAIDetection:
    def test_ai_generated(self):
        fp = ModelFingerprint()
        text = "As an AI language model, I was trained on data and don't have feelings."
        result = fp.analyze(text)
        assert result.is_ai_generated

    def test_human_text(self):
        fp = ModelFingerprint()
        text = "Hey, can you grab milk on the way home? Also the kids need picking up at 3."
        result = fp.analyze(text)
        assert not result.is_ai_generated

    def test_structured_lists(self):
        fp = ModelFingerprint()
        text = "Here is the summary:\n1. **First point** about the topic\n2. **Second point** about details"
        result = fp.analyze(text)
        assert result.is_ai_generated


# ---------------------------------------------------------------------------
# Unknown model
# ---------------------------------------------------------------------------

class TestUnknown:
    def test_no_markers(self):
        fp = ModelFingerprint()
        text = "The cat sat on the mat."
        result = fp.analyze(text)
        assert result.likely_model == "unknown"
        assert result.confidence < 0.1

    def test_ambiguous(self):
        fp = ModelFingerprint()
        text = "I can help you with that request."
        result = fp.analyze(text)
        # Generic enough that no model should score high
        assert result.confidence < 0.3


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

class TestCustomMarkers:
    def test_custom_model(self):
        fp = ModelFingerprint(custom_markers={
            "custom_bot": [(r"\bcustom response\b", 0.5)],
        })
        result = fp.analyze("This is a custom response to your query.")
        assert result.likely_model == "custom_bot"
        assert result.confidence >= 0.5

    def test_extend_existing(self):
        fp = ModelFingerprint(custom_markers={
            "claude": [(r"\banthropically speaking\b", 0.5)],
        })
        result = fp.analyze("Anthropically speaking, this is interesting.")
        assert result.scores["claude"] >= 0.5


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_analyze(self):
        fp = ModelFingerprint()
        results = fp.analyze_batch([
            "I'd be happy to help!",
            "Certainly! As an AI, I can assist.",
            "Hey what's up?",
        ])
        assert len(results) == 3
        assert all(isinstance(r, FingerprintResult) for r in results)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResult:
    def test_result_structure(self):
        fp = ModelFingerprint()
        result = fp.analyze("Test text here")
        assert isinstance(result, FingerprintResult)
        assert isinstance(result.scores, dict)
        assert isinstance(result.markers_found, list)

    def test_supported_models(self):
        fp = ModelFingerprint()
        assert "claude" in fp.supported_models
        assert "gpt" in fp.supported_models
        assert "gemini" in fp.supported_models
