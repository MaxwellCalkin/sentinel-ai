"""Tests for context validation and grounding detection."""

import pytest
from sentinel.context_validator import (
    ContextValidator,
    GroundingCheck,
    GroundingResult,
    GroundingStats,
)


# ---------------------------------------------------------------------------
# Grounded responses
# ---------------------------------------------------------------------------

class TestGroundedResponses:
    def test_response_fully_supported_by_single_passage(self):
        validator = ContextValidator()
        result = validator.validate(
            response="Paris is the capital of France.",
            context=["Paris is the capital city of France."],
        )
        assert result.is_grounded
        assert result.grounding_score > 0.5
        assert result.unsupported_count == 0

    def test_response_supported_by_one_of_multiple_passages(self):
        validator = ContextValidator()
        result = validator.validate(
            response="Python was created by Guido van Rossum.",
            context=[
                "JavaScript was designed by Brendan Eich.",
                "Python was created by Guido van Rossum in 1991.",
            ],
        )
        assert result.is_grounded
        assert result.checks[0].supported

    def test_multiple_sentences_all_grounded(self):
        validator = ContextValidator()
        result = validator.validate(
            response="The Earth orbits the Sun. Water boils at 100 degrees Celsius.",
            context=[
                "The Earth orbits the Sun once per year.",
                "Water boils at 100 degrees Celsius at sea level.",
            ],
        )
        assert result.is_grounded
        assert result.unsupported_count == 0
        assert len(result.checks) == 2


# ---------------------------------------------------------------------------
# Ungrounded responses
# ---------------------------------------------------------------------------

class TestUngroundedResponses:
    def test_fabricated_claim_detected(self):
        validator = ContextValidator()
        result = validator.validate(
            response="The company announced a merger with Google worth 50 billion dollars.",
            context=["The company released quarterly earnings showing steady growth."],
        )
        assert not result.is_grounded
        assert result.unsupported_count > 0

    def test_no_context_passages_means_ungrounded(self):
        validator = ContextValidator()
        result = validator.validate(
            response="The population is approximately 10 million people.",
            context=[],
        )
        assert not result.is_grounded
        assert result.unsupported_count > 0

    def test_completely_unrelated_response(self):
        validator = ContextValidator()
        result = validator.validate(
            response="Quantum entanglement enables faster-than-light communication.",
            context=["The recipe calls for flour, sugar, and butter."],
        )
        assert not result.is_grounded
        assert result.grounding_score == 0.0


# ---------------------------------------------------------------------------
# Similarity threshold configuration
# ---------------------------------------------------------------------------

class TestSimilarityThreshold:
    def test_strict_threshold_rejects_weak_overlap(self):
        validator = ContextValidator(similarity_threshold=0.8)
        result = validator.validate(
            response="The company launched new products for consumers.",
            context=["The company introduced items for retail customers."],
        )
        assert not result.is_grounded

    def test_lenient_threshold_accepts_partial_overlap(self):
        validator = ContextValidator(similarity_threshold=0.1)
        result = validator.validate(
            response="Technology advances rapidly in modern society.",
            context=["Modern society sees rapid technology advancement."],
        )
        assert result.is_grounded
        assert result.checks[0].supported


# ---------------------------------------------------------------------------
# Minimum grounding score configuration
# ---------------------------------------------------------------------------

class TestMinGroundingScore:
    def test_low_grounding_threshold_allows_partial_support(self):
        validator = ContextValidator(
            similarity_threshold=0.3,
            min_grounding_score=0.3,
        )
        result = validator.validate(
            response="The sky is blue. Aliens live on Jupiter.",
            context=["The sky appears blue during clear weather."],
        )
        # One of two sentences supported means 0.5 score, above 0.3
        assert result.is_grounded

    def test_high_grounding_threshold_rejects_partial_support(self):
        validator = ContextValidator(
            similarity_threshold=0.3,
            min_grounding_score=1.0,
        )
        result = validator.validate(
            response="The sky is blue. Aliens live on Jupiter.",
            context=["The sky appears blue during clear weather."],
        )
        assert not result.is_grounded


# ---------------------------------------------------------------------------
# Grounding checks (per-sentence detail)
# ---------------------------------------------------------------------------

class TestGroundingChecks:
    def test_check_has_best_match_passage(self):
        validator = ContextValidator()
        result = validator.validate(
            response="Machine learning improves with more training data.",
            context=["Machine learning algorithms improve their accuracy with more training data."],
        )
        assert len(result.checks) == 1
        check = result.checks[0]
        assert check.best_match_passage != ""
        assert check.best_match_score > 0

    def test_unsupported_check_has_low_score(self):
        validator = ContextValidator(similarity_threshold=0.3)
        result = validator.validate(
            response="The volcano erupted spectacularly last Tuesday.",
            context=["The quarterly financial report was filed yesterday."],
        )
        check = result.checks[0]
        assert not check.supported
        assert check.best_match_score < 0.3

    def test_identifies_which_sentences_are_unsupported(self):
        validator = ContextValidator(similarity_threshold=0.3)
        result = validator.validate(
            response="The river flows north. Unicorns guard the bridge.",
            context=["The river flows north through the valley."],
        )
        supported_sentences = [c.sentence for c in result.checks if c.supported]
        unsupported_sentences = [c.sentence for c in result.checks if not c.supported]
        assert any("river" in s.lower() for s in supported_sentences)
        assert any("unicorn" in s.lower() for s in unsupported_sentences)


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

class TestBatchValidation:
    def test_batch_returns_result_per_item(self):
        validator = ContextValidator()
        results = validator.validate_batch([
            ("Paris is in France.", ["France's capital is Paris."]),
            ("Mars is blue.", ["Mars is known as the red planet."]),
        ])
        assert len(results) == 2
        assert all(isinstance(r, GroundingResult) for r in results)

    def test_empty_batch_returns_empty_list(self):
        validator = ContextValidator()
        results = validator.validate_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_after_no_validations(self):
        validator = ContextValidator()
        stats = validator.stats()
        assert stats.total_validated == 0
        assert stats.avg_grounding_score == 0.0
        assert stats.fully_grounded_count == 0

    def test_stats_accumulate_across_validations(self):
        validator = ContextValidator()
        validator.validate(
            response="The sky is blue.",
            context=["The sky appears blue."],
        )
        validator.validate(
            response="Quantum teleportation invented yesterday by cats.",
            context=["The weather forecast calls for rain."],
        )
        stats = validator.stats()
        assert stats.total_validated == 2
        assert 0.0 < stats.avg_grounding_score < 1.0
        assert stats.fully_grounded_count == 1

    def test_stats_reflects_batch_validations(self):
        validator = ContextValidator()
        validator.validate_batch([
            ("Dogs are mammals.", ["Dogs belong to the mammal family."]),
            ("Cats are reptiles.", ["Cats are furry mammals."]),
        ])
        stats = validator.stats()
        assert stats.total_validated == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_response_is_grounded(self):
        validator = ContextValidator()
        result = validator.validate(response="", context=["Some context."])
        assert result.is_grounded
        assert result.grounding_score == 1.0
        assert result.unsupported_count == 0

    def test_sentence_with_only_stop_words_treated_as_supported(self):
        validator = ContextValidator()
        result = validator.validate(
            response="It is.",
            context=["Unrelated context passage here."],
        )
        # "It is." has no content words after filtering, treated as supported
        assert result.checks[0].supported

    def test_score_always_between_zero_and_one(self):
        validator = ContextValidator()
        result = validator.validate(
            response="Testing the score range for this validator module.",
            context=["Testing the score range for this validator module."],
        )
        assert 0.0 <= result.grounding_score <= 1.0
        for check in result.checks:
            assert 0.0 <= check.best_match_score <= 1.0
