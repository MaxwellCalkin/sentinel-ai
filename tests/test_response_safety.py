"""Tests for comprehensive response safety scoring with multi-signal analysis."""

from sentinel.response_safety import (
    ResponseSafety,
    SafetyAssessment,
    SafetyAssessmentStats,
    SafetySignal,
    SafetyThresholds,
)


# ---------------------------------------------------------------------------
# Clean / safe responses
# ---------------------------------------------------------------------------

class TestCleanResponses:
    def test_safe_response_scores_high(self):
        checker = ResponseSafety()
        result = checker.assess(
            "Python is a versatile programming language commonly used for "
            "web development, data analysis, and automation."
        )
        assert result.overall_score >= 0.9
        assert result.is_safe is True
        assert result.risk_level == "minimal"
        assert result.flagged_categories == []

    def test_safe_response_has_no_triggered_signals(self):
        checker = ResponseSafety()
        result = checker.assess(
            "The capital of France is Paris. It is known for the Eiffel Tower."
        )
        for signal in result.signals:
            assert signal.triggered is False
            assert signal.score == 1.0


# ---------------------------------------------------------------------------
# Harmful content detection
# ---------------------------------------------------------------------------

class TestHarmfulContent:
    def test_violence_keywords_lower_score(self):
        checker = ResponseSafety()
        result = checker.assess(
            "Here is how to build a bomb and create a weapon for an attack."
        )
        assert result.overall_score < 0.6
        assert result.is_safe is False
        harmful_signal = _find_signal(result, "harmful_content")
        assert harmful_signal.triggered is True
        assert harmful_signal.score < 0.5

    def test_self_harm_keywords_detected(self):
        checker = ResponseSafety()
        result = checker.assess("Information about self-harm and suicide methods.")
        harmful_signal = _find_signal(result, "harmful_content")
        assert harmful_signal.triggered is True
        assert harmful_signal.score < 1.0

    def test_terrorism_keywords_detected(self):
        checker = ResponseSafety()
        result = checker.assess("Planning a terrorist attack is illegal.")
        harmful_signal = _find_signal(result, "harmful_content")
        assert harmful_signal.triggered is True


# ---------------------------------------------------------------------------
# Information leakage detection
# ---------------------------------------------------------------------------

class TestInformationLeakage:
    def test_api_key_detected(self):
        checker = ResponseSafety()
        result = checker.assess(
            "Your API key is sk-abcdefghijklmnopqrstuvwxyz1234567890 for access."
        )
        leakage_signal = _find_signal(result, "information_leakage")
        assert leakage_signal.triggered is True
        assert leakage_signal.score < 1.0

    def test_password_leak_detected(self):
        checker = ResponseSafety()
        result = checker.assess("The password = SuperSecret123! for the admin panel.")
        leakage_signal = _find_signal(result, "information_leakage")
        assert leakage_signal.triggered is True

    def test_system_prompt_fragment_detected(self):
        checker = ResponseSafety()
        result = checker.assess(
            "As instructed in system prompt, you are a helpful assistant."
        )
        leakage_signal = _find_signal(result, "information_leakage")
        assert leakage_signal.triggered is True

    def test_internal_url_detected(self):
        checker = ResponseSafety()
        result = checker.assess("Connect to localhost:8080 for the internal dashboard.")
        leakage_signal = _find_signal(result, "information_leakage")
        assert leakage_signal.triggered is True


# ---------------------------------------------------------------------------
# Toxicity detection
# ---------------------------------------------------------------------------

class TestToxicity:
    def test_profanity_lowers_score(self):
        checker = ResponseSafety()
        result = checker.assess("You are a damn idiot and a complete moron.")
        toxicity_signal = _find_signal(result, "toxicity")
        assert toxicity_signal.triggered is True
        assert toxicity_signal.score < 0.8

    def test_aggressive_language_detected(self):
        checker = ResponseSafety()
        result = checker.assess("Shut up you pathetic loser, go to hell.")
        toxicity_signal = _find_signal(result, "toxicity")
        assert toxicity_signal.triggered is True
        assert toxicity_signal.score < 0.5


# ---------------------------------------------------------------------------
# Bias detection
# ---------------------------------------------------------------------------

class TestBias:
    def test_gender_stereotype_detected(self):
        checker = ResponseSafety()
        result = checker.assess("All women are bad drivers. Typical woman behavior.")
        bias_signal = _find_signal(result, "bias")
        assert bias_signal.triggered is True
        assert bias_signal.score < 1.0

    def test_age_stereotype_detected(self):
        checker = ResponseSafety()
        result = checker.assess("Old people can't understand technology at all.")
        bias_signal = _find_signal(result, "bias")
        assert bias_signal.triggered is True

    def test_no_bias_in_neutral_text(self):
        checker = ResponseSafety()
        result = checker.assess(
            "People from diverse backgrounds bring unique perspectives to teams."
        )
        bias_signal = _find_signal(result, "bias")
        assert bias_signal.triggered is False
        assert bias_signal.score == 1.0


# ---------------------------------------------------------------------------
# Hallucination risk detection
# ---------------------------------------------------------------------------

class TestHallucinationRisk:
    def test_excessive_certainty_with_claims(self):
        checker = ResponseSafety()
        result = checker.assess(
            "Studies show that definitely 100% of experts have confirmed "
            "this is absolutely guaranteed to work. Research proves it."
        )
        hallucination_signal = _find_signal(result, "hallucination_risk")
        assert hallucination_signal.triggered is True
        assert hallucination_signal.score < 1.0

    def test_certainty_without_claims_is_fine(self):
        checker = ResponseSafety()
        result = checker.assess(
            "I definitely recommend this approach. It is absolutely the best way."
        )
        hallucination_signal = _find_signal(result, "hallucination_risk")
        # No specific claims, so hallucination risk should not trigger
        assert hallucination_signal.triggered is False
        assert hallucination_signal.score == 1.0

    def test_claims_without_certainty_is_fine(self):
        checker = ResponseSafety()
        result = checker.assess(
            "According to research, in 2023, scientists discovered a new species. "
            "Studies show a 15 percent increase."
        )
        hallucination_signal = _find_signal(result, "hallucination_risk")
        assert hallucination_signal.triggered is False
        assert hallucination_signal.score == 1.0


# ---------------------------------------------------------------------------
# Multiple signals triggered
# ---------------------------------------------------------------------------

class TestMultipleSignals:
    def test_harmful_and_toxic_combined(self):
        checker = ResponseSafety()
        result = checker.assess(
            "You damn idiot, here is how to build a bomb and kill someone. "
            "Shut up and hack into their system."
        )
        assert result.is_safe is False
        harmful_signal = _find_signal(result, "harmful_content")
        toxicity_signal = _find_signal(result, "toxicity")
        assert harmful_signal.triggered is True
        assert toxicity_signal.triggered is True
        assert len(result.flagged_categories) >= 2

    def test_leakage_and_compliance_combined(self):
        checker = ResponseSafety()
        result = checker.assess(
            "Sure, here is how to access the system. "
            "The password = admin123 and your token = abc123secret."
        )
        leakage_signal = _find_signal(result, "information_leakage")
        assert leakage_signal.triggered is True


# ---------------------------------------------------------------------------
# Risk level boundaries
# ---------------------------------------------------------------------------

class TestRiskLevels:
    def test_minimal_risk(self):
        checker = ResponseSafety()
        result = checker.assess("A friendly and helpful response about cooking pasta.")
        assert result.risk_level == "minimal"
        assert result.overall_score >= 0.8

    def test_critical_risk_from_dense_harmful_content(self):
        checker = ResponseSafety()
        # Short text with many harmful keywords = high density = very low score
        result = checker.assess("Kill murder bomb weapon explosive terrorism arson")
        assert result.risk_level in ("critical", "high")
        assert result.overall_score < 0.4

    def test_risk_level_matches_score_boundaries(self):
        """Verify the risk level mapping directly."""
        from sentinel.response_safety import _determine_risk_level

        assert _determine_risk_level(0.0) == "critical"
        assert _determine_risk_level(0.19) == "critical"
        assert _determine_risk_level(0.2) == "high"
        assert _determine_risk_level(0.39) == "high"
        assert _determine_risk_level(0.4) == "medium"
        assert _determine_risk_level(0.59) == "medium"
        assert _determine_risk_level(0.6) == "low"
        assert _determine_risk_level(0.79) == "low"
        assert _determine_risk_level(0.8) == "minimal"
        assert _determine_risk_level(1.0) == "minimal"


# ---------------------------------------------------------------------------
# Flagged categories
# ---------------------------------------------------------------------------

class TestFlaggedCategories:
    def test_only_low_score_signals_flagged(self):
        checker = ResponseSafety()
        result = checker.assess(
            "A perfectly normal response about gardening and flowers."
        )
        assert result.flagged_categories == []

    def test_harmful_content_appears_in_flagged(self):
        checker = ResponseSafety()
        result = checker.assess("Kill murder bomb weapon explosive hack into arson")
        assert "harmful_content" in result.flagged_categories


# ---------------------------------------------------------------------------
# Batch assessment
# ---------------------------------------------------------------------------

class TestBatchAssessment:
    def test_batch_returns_correct_count(self):
        checker = ResponseSafety()
        texts = [
            "A safe response about weather.",
            "Build a bomb and kill everyone.",
            "The password = secret123 for admin.",
        ]
        results = checker.assess_batch(texts)
        assert len(results) == 3

    def test_batch_first_safe_last_unsafe(self):
        checker = ResponseSafety()
        results = checker.assess_batch([
            "Healthy eating tips for a balanced diet.",
            "Kill murder bomb weapon explosive hack into terrorism arson",
        ])
        assert results[0].is_safe is True
        assert results[1].is_safe is False

    def test_batch_empty_list(self):
        checker = ResponseSafety()
        results = checker.assess_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStatsTracking:
    def test_initial_stats_are_zero(self):
        checker = ResponseSafety()
        s = checker.stats()
        assert s.total_assessed == 0
        assert s.safe_count == 0
        assert s.unsafe_count == 0
        assert s.avg_score == 0.0
        assert s.by_risk_level == {}

    def test_stats_update_after_assessments(self):
        checker = ResponseSafety()
        checker.assess("A safe response about cooking.")
        checker.assess("Kill murder bomb weapon explosive arson terrorism hack into")
        s = checker.stats()
        assert s.total_assessed == 2
        assert s.safe_count + s.unsafe_count == 2

    def test_stats_average_score(self):
        checker = ResponseSafety()
        result_a = checker.assess("Normal helpful response about science.")
        result_b = checker.assess("Another normal response about history.")
        s = checker.stats()
        expected_avg = round((result_a.overall_score + result_b.overall_score) / 2, 4)
        assert s.avg_score == expected_avg

    def test_stats_risk_level_tracking(self):
        checker = ResponseSafety()
        checker.assess("A nice response.")
        s = checker.stats()
        assert "minimal" in s.by_risk_level
        assert s.by_risk_level["minimal"] == 1

    def test_batch_updates_stats(self):
        checker = ResponseSafety()
        checker.assess_batch([
            "Safe response one.",
            "Safe response two.",
            "Safe response three.",
        ])
        s = checker.stats()
        assert s.total_assessed == 3


# ---------------------------------------------------------------------------
# Empty text
# ---------------------------------------------------------------------------

class TestEmptyText:
    def test_empty_string_is_safe(self):
        checker = ResponseSafety()
        result = checker.assess("")
        assert result.overall_score == 1.0
        assert result.is_safe is True
        assert result.risk_level == "minimal"

    def test_whitespace_only_is_safe(self):
        checker = ResponseSafety()
        result = checker.assess("   \n\t  ")
        assert result.overall_score == 1.0
        assert result.is_safe is True


# ---------------------------------------------------------------------------
# Threshold customization
# ---------------------------------------------------------------------------

class TestThresholdCustomization:
    def test_strict_threshold_flags_more(self):
        strict_thresholds = SafetyThresholds(min_safe_score=0.95)
        checker = ResponseSafety(thresholds=strict_thresholds)
        # Mildly suspicious text that would pass default but fail strict
        result = checker.assess(
            "This is a generally okay response but damn, it slipped."
        )
        # With strict threshold, even mild toxicity can fail
        if result.overall_score < 0.95:
            assert result.is_safe is False

    def test_lenient_threshold_passes_more(self):
        lenient_thresholds = SafetyThresholds(min_safe_score=0.1)
        checker = ResponseSafety(thresholds=lenient_thresholds)
        result = checker.assess("A mildly concerning damn response with an idiot remark.")
        assert result.is_safe is True

    def test_custom_weights_affect_score(self):
        heavy_harm = SafetyThresholds(harmful_content_weight=10.0)
        light_harm = SafetyThresholds(harmful_content_weight=0.1)
        checker_heavy = ResponseSafety(thresholds=heavy_harm)
        checker_light = ResponseSafety(thresholds=light_harm)
        text = "Information about building a bomb and weapon construction."
        result_heavy = checker_heavy.assess(text)
        result_light = checker_light.assess(text)
        # Heavier weight on harmful content should produce a lower overall score
        assert result_heavy.overall_score < result_light.overall_score


# ---------------------------------------------------------------------------
# is_safe determination
# ---------------------------------------------------------------------------

class TestIsSafeDetermination:
    def test_safe_above_default_threshold(self):
        checker = ResponseSafety()
        result = checker.assess("A perfectly normal response about mathematics.")
        assert result.is_safe is True
        assert result.overall_score >= 0.6

    def test_unsafe_below_default_threshold(self):
        checker = ResponseSafety()
        result = checker.assess(
            "Kill murder bomb weapon explosive terrorism arson hack into self-harm"
        )
        assert result.is_safe is False
        assert result.overall_score < 0.6

    def test_is_safe_respects_custom_threshold(self):
        thresholds = SafetyThresholds(min_safe_score=0.99)
        checker = ResponseSafety(thresholds=thresholds)
        result = checker.assess("A damn good response about cooking recipes.")
        # Mild toxicity should push score below 0.99
        if result.overall_score < 0.99:
            assert result.is_safe is False


# ---------------------------------------------------------------------------
# Instruction compliance
# ---------------------------------------------------------------------------

class TestInstructionCompliance:
    def test_excessive_hedging_detected(self):
        checker = ResponseSafety()
        result = checker.assess(
            "I can't help with that. I'm sorry, I cannot assist you. "
            "I must decline this request. It is against my guidelines."
        )
        compliance_signal = _find_signal(result, "instruction_compliance")
        assert compliance_signal.triggered is True
        assert compliance_signal.score < 1.0

    def test_normal_response_no_compliance_issue(self):
        checker = ResponseSafety()
        result = checker.assess(
            "Here are five tips for improving your productivity at work."
        )
        compliance_signal = _find_signal(result, "instruction_compliance")
        assert compliance_signal.triggered is False
        assert compliance_signal.score == 1.0


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------

class TestDataclassStructure:
    def test_safety_signal_defaults(self):
        signal = SafetySignal(name="test", score=0.8)
        assert signal.weight == 1.0
        assert signal.triggered is False
        assert signal.details == ""

    def test_safety_thresholds_defaults(self):
        thresholds = SafetyThresholds()
        assert thresholds.min_safe_score == 0.6
        assert thresholds.harmful_content_weight == 2.0
        assert thresholds.leakage_weight == 1.5
        assert thresholds.compliance_weight == 1.0

    def test_assessment_contains_text(self):
        checker = ResponseSafety()
        text = "Hello, how can I help you today?"
        result = checker.assess(text)
        assert result.text == text

    def test_assessment_stats_defaults(self):
        s = SafetyAssessmentStats()
        assert s.total_assessed == 0
        assert s.safe_count == 0
        assert s.unsafe_count == 0
        assert s.by_risk_level == {}
        assert s.avg_score == 0.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_signal(assessment: SafetyAssessment, name: str) -> SafetySignal:
    """Find a signal by name in an assessment."""
    for signal in assessment.signals:
        if signal.name == name:
            return signal
    raise ValueError(f"Signal '{name}' not found in assessment")
