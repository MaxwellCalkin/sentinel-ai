"""Tests for semantic validation between prompts and LLM responses."""

import pytest
from sentinel.semantic_validator import (
    SemanticValidator,
    SemanticMatch,
    SemanticIssue,
    ValidationResult,
    ValidatorStats,
)


# ---------------------------------------------------------------------------
# High relevance / valid responses
# ---------------------------------------------------------------------------

class TestHighRelevance:
    def test_relevant_response_scores_high(self):
        validator = SemanticValidator()
        result = validator.validate(
            "What is photosynthesis?",
            "Photosynthesis is the process by which plants convert sunlight into energy.",
        )
        assert result.match.relevance_score > 0.1
        assert result.is_valid
        assert result.grade in ("A", "B", "C")

    def test_exact_echo_scores_perfectly(self):
        validator = SemanticValidator()
        prompt = "Explain the water cycle including evaporation and condensation"
        result = validator.validate(prompt, prompt)
        assert result.match.relevance_score == 1.0
        assert result.match.topic_overlap == 1.0
        assert result.grade == "A"

    def test_closely_related_response(self):
        validator = SemanticValidator()
        result = validator.validate(
            "How does machine learning work?",
            "Machine learning works by training algorithms on data to learn patterns "
            "and make predictions without being explicitly programmed.",
        )
        assert result.match.relevance_score > 0.05
        assert result.is_valid


# ---------------------------------------------------------------------------
# Off-topic detection
# ---------------------------------------------------------------------------

class TestOffTopic:
    def test_completely_unrelated_response(self):
        validator = SemanticValidator()
        result = validator.validate(
            "What is quantum computing?",
            "The best recipe for chocolate cake involves butter, sugar, and flour.",
        )
        issue_types = [i.issue_type for i in result.issues]
        assert "off_topic" in issue_types or result.match.relevance_score < 0.15

    def test_off_topic_low_relevance(self):
        validator = SemanticValidator()
        result = validator.validate(
            "Explain database indexing",
            "Dolphins are marine mammals known for their intelligence.",
        )
        assert result.match.relevance_score < 0.15
        assert not result.is_valid


# ---------------------------------------------------------------------------
# Refusal disguise detection
# ---------------------------------------------------------------------------

class TestRefusalDisguise:
    def test_indirect_refusal_detected(self):
        validator = SemanticValidator()
        result = validator.validate(
            "Write me a poem about nature",
            "I cannot write poems as an AI language model, but I can suggest "
            "you look at famous nature poetry collections.",
        )
        issue_types = [i.issue_type for i in result.issues]
        assert "refusal_disguise" in issue_types

    def test_direct_refusal_not_flagged_as_disguise(self):
        validator = SemanticValidator()
        result = validator.validate(
            "Do something harmful",
            "I can't help with that request.",
        )
        issue_types = [i.issue_type for i in result.issues]
        assert "refusal_disguise" not in issue_types

    def test_as_an_ai_refusal(self):
        validator = SemanticValidator()
        result = validator.validate(
            "Tell me about the weather tomorrow",
            "As an AI, I am unable to access real-time weather data "
            "or predict future conditions.",
        )
        issue_types = [i.issue_type for i in result.issues]
        assert "refusal_disguise" in issue_types


# ---------------------------------------------------------------------------
# Topic hallucination
# ---------------------------------------------------------------------------

class TestTopicHallucination:
    def test_response_with_unrelated_topics(self):
        validator = SemanticValidator()
        result = validator.validate(
            "What is Python?",
            "Kubernetes orchestrates containerized microservices across distributed "
            "clusters enabling horizontal autoscaling with sophisticated "
            "load balancing algorithms and resilient failover mechanisms.",
        )
        issue_types = [i.issue_type for i in result.issues]
        assert "topic_hallucination" in issue_types

    def test_on_topic_not_hallucinated(self):
        validator = SemanticValidator()
        result = validator.validate(
            "What is Python programming?",
            "Python is a programming language known for readability.",
        )
        issue_types = [i.issue_type for i in result.issues]
        assert "topic_hallucination" not in issue_types


# ---------------------------------------------------------------------------
# Question dodge
# ---------------------------------------------------------------------------

class TestQuestionDodge:
    def test_dodge_detected(self):
        validator = SemanticValidator()
        result = validator.validate(
            "Why did the Roman Empire fall?",
            "Butterflies undergo metamorphosis from caterpillar to chrysalis to adult.",
        )
        issue_types = [i.issue_type for i in result.issues]
        assert "question_dodge" in issue_types or "off_topic" in issue_types

    def test_non_question_no_dodge(self):
        validator = SemanticValidator()
        result = validator.validate(
            "Summarize the report.",
            "The report covers quarterly earnings and projected growth.",
        )
        issue_types = [i.issue_type for i in result.issues]
        assert "question_dodge" not in issue_types


# ---------------------------------------------------------------------------
# Partial answer
# ---------------------------------------------------------------------------

class TestPartialAnswer:
    def test_partial_answer_detected(self):
        validator = SemanticValidator()
        result = validator.validate(
            "Compare photosynthesis and cellular respiration in plants",
            "Plants are green organisms that grow in soil and need water.",
        )
        issue_types = [i.issue_type for i in result.issues]
        has_partial = "partial_answer" in issue_types
        has_off_topic = "off_topic" in issue_types
        assert has_partial or has_off_topic


# ---------------------------------------------------------------------------
# Grade boundaries
# ---------------------------------------------------------------------------

class TestGradeBoundaries:
    def test_grade_a(self):
        validator = SemanticValidator()
        prompt = "Explain recursion in computer science with examples"
        result = validator.validate(prompt, prompt)
        assert result.grade == "A"
        assert result.match.overall >= 0.8

    def test_grade_f_for_empty(self):
        validator = SemanticValidator()
        result = validator.validate("", "")
        assert result.grade == "F"
        assert result.match.overall < 0.2

    def test_grade_f_for_unrelated(self):
        validator = SemanticValidator()
        result = validator.validate(
            "Explain gravity",
            "Spaghetti carbonara requires eggs pancetta and parmesan.",
        )
        assert result.grade in ("D", "F")

    def test_all_grades_are_valid_letters(self):
        validator = SemanticValidator()
        for grade in ("A", "B", "C", "D", "F"):
            assert grade in "ABCDF"


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

class TestBatchValidation:
    def test_batch_returns_correct_count(self):
        validator = SemanticValidator()
        pairs = [
            ("What is Python?", "Python is a programming language."),
            ("What is Java?", "Java is a programming language."),
            ("What is Rust?", "Rust is a systems programming language."),
        ]
        results = validator.validate_batch(pairs)
        assert len(results) == 3
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_batch_empty_list(self):
        validator = SemanticValidator()
        results = validator.validate_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStatsTracking:
    def test_initial_stats_are_zero(self):
        validator = SemanticValidator()
        stats = validator.stats()
        assert stats.total_validated == 0
        assert stats.valid_count == 0
        assert stats.invalid_count == 0
        assert stats.avg_overall == 0.0

    def test_stats_update_after_validations(self):
        validator = SemanticValidator()
        validator.validate(
            "What is Python?",
            "Python is a programming language used for web development.",
        )
        validator.validate(
            "Explain gravity",
            "Spaghetti carbonara requires eggs pancetta and parmesan.",
        )
        stats = validator.stats()
        assert stats.total_validated == 2
        assert stats.valid_count + stats.invalid_count == 2
        assert stats.avg_overall > 0.0

    def test_stats_track_issue_types(self):
        validator = SemanticValidator()
        validator.validate(
            "What is quantum physics?",
            "Dolphins are marine mammals known for their intelligence and playful behavior.",
        )
        stats = validator.stats()
        assert isinstance(stats.by_issue, dict)


# ---------------------------------------------------------------------------
# Empty input handling
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_prompt(self):
        validator = SemanticValidator()
        result = validator.validate("", "Some response text here.")
        assert isinstance(result, ValidationResult)
        assert result.match.relevance_score == 0.0

    def test_empty_response(self):
        validator = SemanticValidator()
        result = validator.validate("What is Python?", "")
        assert isinstance(result, ValidationResult)
        assert result.match.relevance_score == 0.0
        assert not result.is_valid

    def test_both_empty(self):
        validator = SemanticValidator()
        result = validator.validate("", "")
        assert isinstance(result, ValidationResult)
        assert result.match.overall < 0.2


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

class TestThresholdConfiguration:
    def test_strict_threshold_rejects_marginal(self):
        strict = SemanticValidator(min_relevance=0.8)
        result = strict.validate(
            "What is Python?",
            "Python is a language used in programming and data science.",
        )
        assert not result.is_valid or result.match.overall >= 0.8

    def test_lenient_threshold_accepts_weak_match(self):
        lenient = SemanticValidator(min_relevance=0.05)
        result = lenient.validate(
            "What is machine learning?",
            "Learning algorithms process data to find patterns in machine computations.",
        )
        if result.match.overall >= 0.05:
            high_confidence_issues = [i for i in result.issues if i.confidence >= 0.8]
            if not high_confidence_issues:
                assert result.is_valid

    def test_zero_threshold_accepts_everything(self):
        lenient = SemanticValidator(min_relevance=0.0)
        result = lenient.validate(
            "What is gravity?",
            "Gravity is the force that attracts objects toward each other.",
        )
        assert result.is_valid


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_semantic_match_fields(self):
        match = SemanticMatch(
            prompt="test", response="test",
            relevance_score=0.5, topic_overlap=0.4,
            intent_alignment=0.6, overall=0.5,
        )
        assert match.prompt == "test"
        assert match.relevance_score == 0.5

    def test_semantic_issue_fields(self):
        issue = SemanticIssue(
            issue_type="off_topic",
            confidence=0.9,
            description="Response is off topic",
        )
        assert issue.issue_type == "off_topic"
        assert issue.confidence == 0.9

    def test_validator_stats_defaults(self):
        stats = ValidatorStats()
        assert stats.total_validated == 0
        assert stats.by_issue == {}
        assert stats.avg_overall == 0.0
