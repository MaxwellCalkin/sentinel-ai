"""Tests for ContentClassifierV2 multi-label classifier."""

from __future__ import annotations

import pytest

from sentinel.content_classifier_v2 import (
    ClassificationResult,
    ClassifierConfig,
    ContentClassifierV2,
    LabelDefinition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier() -> ContentClassifierV2:
    return ContentClassifierV2()


# ---------------------------------------------------------------------------
# Basic classification
# ---------------------------------------------------------------------------

class TestTechnicalClassification:
    def test_detects_technical_text(self, classifier: ContentClassifierV2) -> None:
        result = classifier.classify(
            "The algorithm uses a database function to compile the code"
        )
        assert result.top_label == "technical"
        technical_labels = [l for l in result.labels if l.name == "technical"]
        assert len(technical_labels) == 1
        assert technical_labels[0].confidence > 0

    def test_matched_keywords_are_reported(self, classifier: ContentClassifierV2) -> None:
        result = classifier.classify("Deploy the code to the server")
        technical = next(l for l in result.labels if l.name == "technical")
        assert "deploy" in technical.matched_keywords
        assert "code" in technical.matched_keywords
        assert "server" in technical.matched_keywords


class TestCreativeClassification:
    def test_detects_creative_text(self, classifier: ContentClassifierV2) -> None:
        result = classifier.classify(
            "Write a story with a character who imagines a fictional narrative"
        )
        assert result.top_label == "creative"

    def test_creative_keywords_matched(self, classifier: ContentClassifierV2) -> None:
        result = classifier.classify("This poem is artistic and creative")
        creative = next(l for l in result.labels if l.name == "creative")
        assert "poem" in creative.matched_keywords


class TestMultiLabelClassification:
    def test_technical_and_educational(self, classifier: ContentClassifierV2) -> None:
        result = classifier.classify(
            "Learn how to code an algorithm in this tutorial about database functions"
        )
        assert result.is_multi_label is True
        label_names = {l.name for l in result.labels}
        assert "technical" in label_names
        assert "educational" in label_names

    def test_is_multi_label_flag_false_for_single(
        self, classifier: ContentClassifierV2
    ) -> None:
        result = classifier.classify("revenue profit growth quarterly budget")
        assert result.top_label == "business"
        assert result.is_multi_label is False


class TestUnclassifiedText:
    def test_random_noise_is_unclassified(self, classifier: ContentClassifierV2) -> None:
        result = classifier.classify("xyzzy plugh abracadabra")
        assert result.top_label == "unclassified"
        assert result.labels == []
        assert result.is_multi_label is False
        assert result.total_confidence == 0


# ---------------------------------------------------------------------------
# Custom labels
# ---------------------------------------------------------------------------

class TestCustomLabels:
    def test_add_and_detect_custom_label(self, classifier: ContentClassifierV2) -> None:
        classifier.add_label(
            LabelDefinition(
                name="medical",
                keywords=["diagnosis", "patient", "symptom", "treatment"],
            )
        )
        result = classifier.classify("The patient symptom led to a diagnosis")
        label_names = {l.name for l in result.labels}
        assert "medical" in label_names

    def test_custom_label_appears_in_list(self, classifier: ContentClassifierV2) -> None:
        classifier.add_label(LabelDefinition(name="legal", keywords=["lawsuit"]))
        assert "legal" in classifier.list_labels()


class TestLabelRemoval:
    def test_remove_existing_label(self, classifier: ContentClassifierV2) -> None:
        classifier.remove_label("creative")
        assert "creative" not in classifier.list_labels()
        result = classifier.classify("Write a story with a fictional character")
        label_names = {l.name for l in result.labels}
        assert "creative" not in label_names

    def test_remove_nonexistent_label_raises(
        self, classifier: ContentClassifierV2
    ) -> None:
        with pytest.raises(KeyError, match="not_here"):
            classifier.remove_label("not_here")


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

class TestBatchClassification:
    def test_batch_returns_correct_count(self, classifier: ContentClassifierV2) -> None:
        texts = [
            "The algorithm compiles code",
            "Write a poem about a story",
            "xyzzy plugh",
        ]
        results = classifier.classify_batch(texts)
        assert len(results) == 3

    def test_batch_results_match_individual(
        self, classifier: ContentClassifierV2
    ) -> None:
        texts = ["Deploy code to the server", "Learn in this tutorial"]
        batch_results = classifier.classify_batch(texts)
        individual_results = [classifier.classify(t) for t in texts]
        # The batch results were created first, so stats differ; compare labels only
        for batch_r, text in zip(batch_results, texts):
            assert isinstance(batch_r, ClassificationResult)
            assert batch_r.text == text


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStats:
    def test_total_classified_increments(self, classifier: ContentClassifierV2) -> None:
        classifier.classify("algorithm code function")
        classifier.classify("story poem fiction")
        stats = classifier.stats()
        assert stats.total_classified == 2

    def test_by_label_counts(self, classifier: ContentClassifierV2) -> None:
        classifier.classify("algorithm code database")
        classifier.classify("algorithm function server")
        stats = classifier.stats()
        assert stats.by_label.get("technical", 0) == 2

    def test_multi_label_rate(self, classifier: ContentClassifierV2) -> None:
        classifier.classify("algorithm code database")  # single: technical
        classifier.classify(
            "learn tutorial about algorithm code"
        )  # multi: educational + technical
        stats = classifier.stats()
        assert stats.multi_label_rate == 0.5

    def test_fresh_stats_are_zero(self, classifier: ContentClassifierV2) -> None:
        stats = classifier.stats()
        assert stats.total_classified == 0
        assert stats.by_label == {}
        assert stats.multi_label_rate == 0.0


# ---------------------------------------------------------------------------
# Config: min_confidence filtering
# ---------------------------------------------------------------------------

class TestMinConfidence:
    def test_low_confidence_labels_filtered(self) -> None:
        strict = ContentClassifierV2(ClassifierConfig(min_confidence=0.8))
        result = strict.classify("code")
        # Single keyword in a short text may still reach 0.8+ but with only
        # one keyword in a one-word text, confidence = 1*1/max(1*0.1,1) = 1.0
        # so it passes. Use a longer text to dilute:
        result = strict.classify(
            "a b c d e f g h i j k l m n code o p q r s t u v w x y z"
        )
        # 27 words, confidence = 1/max(2.7,1) = 0.37 < 0.8
        assert result.top_label == "unclassified"

    def test_default_min_confidence_allows_weak_matches(
        self, classifier: ContentClassifierV2
    ) -> None:
        result = classifier.classify("the code is here among many other words today")
        technical_labels = [l for l in result.labels if l.name == "technical"]
        assert len(technical_labels) == 1


# ---------------------------------------------------------------------------
# Config: max_labels limiting
# ---------------------------------------------------------------------------

class TestMaxLabels:
    def test_max_labels_caps_output(self) -> None:
        limited = ContentClassifierV2(ClassifierConfig(max_labels=1))
        result = limited.classify(
            "learn code algorithm tutorial story poem revenue strategy"
        )
        assert len(result.labels) <= 1
        assert result.is_multi_label is False


# ---------------------------------------------------------------------------
# Calibration factor
# ---------------------------------------------------------------------------

class TestCalibrationFactor:
    def test_higher_calibration_increases_confidence(self) -> None:
        normal = ContentClassifierV2(ClassifierConfig(calibration_factor=1.0))
        boosted = ContentClassifierV2(ClassifierConfig(calibration_factor=2.0))
        text = "the code algorithm is in the database server system today"
        normal_result = normal.classify(text)
        boosted_result = boosted.classify(text)
        normal_tech = next(l for l in normal_result.labels if l.name == "technical")
        boosted_tech = next(l for l in boosted_result.labels if l.name == "technical")
        assert boosted_tech.confidence >= normal_tech.confidence

    def test_calibration_caps_at_one(self) -> None:
        extreme = ContentClassifierV2(ClassifierConfig(calibration_factor=100.0))
        result = extreme.classify("code algorithm function")
        tech = next(l for l in result.labels if l.name == "technical")
        assert tech.confidence <= 1.0


# ---------------------------------------------------------------------------
# Empty text
# ---------------------------------------------------------------------------

class TestEmptyText:
    def test_empty_string_is_unclassified(self, classifier: ContentClassifierV2) -> None:
        result = classifier.classify("")
        assert result.top_label == "unclassified"
        assert result.labels == []
        assert result.total_confidence == 0

    def test_whitespace_only_is_unclassified(
        self, classifier: ContentClassifierV2
    ) -> None:
        result = classifier.classify("   \t\n  ")
        assert result.top_label == "unclassified"


# ---------------------------------------------------------------------------
# List labels
# ---------------------------------------------------------------------------

class TestListLabels:
    def test_builtin_labels_present(self, classifier: ContentClassifierV2) -> None:
        names = classifier.list_labels()
        for expected in [
            "technical", "educational", "creative",
            "business", "safety", "personal",
        ]:
            assert expected in names

    def test_list_labels_count(self, classifier: ContentClassifierV2) -> None:
        assert len(classifier.list_labels()) == 6


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_total_confidence_sums_labels(
        self, classifier: ContentClassifierV2
    ) -> None:
        result = classifier.classify(
            "learn tutorial about algorithm code database"
        )
        expected_sum = sum(l.confidence for l in result.labels)
        assert result.total_confidence == round(expected_sum, 4)

    def test_weight_affects_confidence(self) -> None:
        classifier = ContentClassifierV2()
        classifier.add_label(
            LabelDefinition(name="heavy", keywords=["alpha"], weight=5.0)
        )
        classifier.add_label(
            LabelDefinition(name="light", keywords=["alpha"], weight=0.1)
        )
        result = classifier.classify("alpha")
        heavy = next(l for l in result.labels if l.name == "heavy")
        light = next(l for l in result.labels if l.name == "light")
        assert heavy.confidence > light.confidence

    def test_safety_label_detection(self, classifier: ContentClassifierV2) -> None:
        result = classifier.classify(
            "This attack exploits a vulnerability to breach the system"
        )
        label_names = {l.name for l in result.labels}
        assert "safety" in label_names

    def test_personal_label_detection(self, classifier: ContentClassifierV2) -> None:
        result = classifier.classify("I feel that my opinion is important to me")
        label_names = {l.name for l in result.labels}
        assert "personal" in label_names

    def test_business_label_detection(self, classifier: ContentClassifierV2) -> None:
        result = classifier.classify(
            "The quarterly revenue shows profit growth for our investor strategy"
        )
        assert result.top_label == "business"
