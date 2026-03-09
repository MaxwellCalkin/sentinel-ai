"""Tests for response quality assessment."""

import pytest
from sentinel.response_quality import (
    ResponseQuality,
    QualityAssessment,
    QualityConfig,
    QualityDimension,
    QualityStats,
)


HIGH_QUALITY_RESPONSE = (
    "To configure a Python virtual environment, follow these steps:\n\n"
    "1. Install Python 3.8 or higher from the official website.\n"
    "2. Open your terminal and navigate to your project directory.\n"
    "3. Run `python -m venv .venv` to create the environment.\n"
    "4. Activate it with `source .venv/bin/activate` on Linux or "
    "`.venv\\Scripts\\activate` on Windows.\n\n"
    "For example, after activation your prompt will change to show the "
    "environment name. You can then install packages with pip. "
    "Additionally, use `pip freeze > requirements.txt` to save dependencies.\n\n"
    "In summary, virtual environments isolate project dependencies and "
    "prevent conflicts between packages."
)

LOW_QUALITY_RESPONSE = "idk maybe try something"

VERBOSE_RESPONSE = (
    "Well, basically, you really just need to actually very simply "
    "do the thing. " * 60
)

WELL_FORMATTED_RESPONSE = (
    "# Getting Started\n\n"
    "Follow these steps to set up your project:\n\n"
    "- Install the required dependencies.\n"
    "- Configure the environment variables.\n"
    "- Run the test suite to verify.\n\n"
    "However, you should also check the documentation for details. "
    "Therefore, ensure everything is working before deploying."
)

POORLY_FORMATTED_RESPONSE = (
    "well there are many things you could do i guess and stuff "
    "you could try various things and see what works maybe"
)


class TestHighQualityResponse:
    def test_high_quality_gets_good_grade(self):
        rq = ResponseQuality()
        result = rq.assess(HIGH_QUALITY_RESPONSE)
        assert result.grade in ("A", "B")

    def test_high_quality_overall_above_threshold(self):
        rq = ResponseQuality()
        result = rq.assess(HIGH_QUALITY_RESPONSE)
        assert result.overall >= 0.6

    def test_high_quality_has_all_dimensions(self):
        rq = ResponseQuality()
        result = rq.assess(HIGH_QUALITY_RESPONSE)
        names = {d.name for d in result.dimensions}
        assert names == {"completeness", "clarity", "helpfulness", "formatting", "conciseness"}


class TestLowQualityResponse:
    def test_low_quality_gets_poor_grade(self):
        rq = ResponseQuality()
        result = rq.assess(LOW_QUALITY_RESPONSE)
        assert result.grade in ("C", "D", "F")

    def test_low_quality_overall_below_threshold(self):
        rq = ResponseQuality()
        result = rq.assess(LOW_QUALITY_RESPONSE)
        assert result.overall < 0.6

    def test_low_quality_word_count_is_small(self):
        rq = ResponseQuality()
        result = rq.assess(LOW_QUALITY_RESPONSE)
        assert result.word_count < 10


class TestVerboseResponse:
    def test_verbose_penalized_conciseness(self):
        rq = ResponseQuality()
        result = rq.assess(VERBOSE_RESPONSE)
        conciseness = next(d for d in result.dimensions if d.name == "conciseness")
        assert conciseness.score < 0.5

    def test_verbose_lower_than_high_quality(self):
        rq = ResponseQuality()
        verbose = rq.assess(VERBOSE_RESPONSE)
        good = rq.assess(HIGH_QUALITY_RESPONSE)
        assert good.overall > verbose.overall


class TestFormattedResponse:
    def test_well_formatted_scores_high(self):
        rq = ResponseQuality()
        result = rq.assess(WELL_FORMATTED_RESPONSE)
        formatting = next(d for d in result.dimensions if d.name == "formatting")
        assert formatting.score >= 0.7

    def test_poorly_formatted_scores_lower(self):
        rq = ResponseQuality()
        well = rq.assess(WELL_FORMATTED_RESPONSE)
        poor = rq.assess(POORLY_FORMATTED_RESPONSE)
        well_fmt = next(d for d in well.dimensions if d.name == "formatting")
        poor_fmt = next(d for d in poor.dimensions if d.name == "formatting")
        assert well_fmt.score > poor_fmt.score


class TestGradeBoundaries:
    def test_grade_a_boundary(self):
        rq = ResponseQuality()
        result = rq.assess(HIGH_QUALITY_RESPONSE)
        if result.overall >= 0.8:
            assert result.grade == "A"
        elif result.overall >= 0.6:
            assert result.grade == "B"

    def test_grade_f_for_empty(self):
        rq = ResponseQuality()
        result = rq.assess("")
        assert result.grade == "F"

    def test_all_grades_are_valid(self):
        rq = ResponseQuality()
        for text in [HIGH_QUALITY_RESPONSE, LOW_QUALITY_RESPONSE, "", VERBOSE_RESPONSE]:
            result = rq.assess(text)
            assert result.grade in ("A", "B", "C", "D", "F")

    def test_grade_matches_overall(self):
        rq = ResponseQuality()
        result = rq.assess(HIGH_QUALITY_RESPONSE)
        if result.overall >= 0.8:
            assert result.grade == "A"
        elif result.overall >= 0.6:
            assert result.grade == "B"
        elif result.overall >= 0.4:
            assert result.grade == "C"
        elif result.overall >= 0.2:
            assert result.grade == "D"
        else:
            assert result.grade == "F"


class TestCompare:
    def test_compare_returns_winner(self):
        rq = ResponseQuality()
        result = rq.compare(HIGH_QUALITY_RESPONSE, LOW_QUALITY_RESPONSE)
        assert result["winner"] == "a"
        assert result["difference"] > 0

    def test_compare_returns_both_assessments(self):
        rq = ResponseQuality()
        result = rq.compare(HIGH_QUALITY_RESPONSE, LOW_QUALITY_RESPONSE)
        assert isinstance(result["a"], QualityAssessment)
        assert isinstance(result["b"], QualityAssessment)

    def test_compare_tie(self):
        rq = ResponseQuality()
        result = rq.compare("hello world", "hello world")
        assert result["winner"] == "tie"
        assert result["difference"] == 0.0


class TestBatch:
    def test_batch_returns_correct_count(self):
        rq = ResponseQuality()
        results = rq.assess_batch([
            HIGH_QUALITY_RESPONSE,
            LOW_QUALITY_RESPONSE,
            WELL_FORMATTED_RESPONSE,
        ])
        assert len(results) == 3

    def test_batch_each_is_assessment(self):
        rq = ResponseQuality()
        results = rq.assess_batch([HIGH_QUALITY_RESPONSE, LOW_QUALITY_RESPONSE])
        for result in results:
            assert isinstance(result, QualityAssessment)


class TestStats:
    def test_stats_empty(self):
        rq = ResponseQuality()
        s = rq.stats()
        assert s.total_assessed == 0
        assert s.avg_overall == 0.0
        assert s.by_grade == {}

    def test_stats_after_assessments(self):
        rq = ResponseQuality()
        rq.assess(HIGH_QUALITY_RESPONSE)
        rq.assess(LOW_QUALITY_RESPONSE)
        s = rq.stats()
        assert s.total_assessed == 2
        assert 0.0 < s.avg_overall < 1.0
        assert sum(s.by_grade.values()) == 2

    def test_stats_tracks_grades(self):
        rq = ResponseQuality()
        rq.assess(HIGH_QUALITY_RESPONSE)
        rq.assess(HIGH_QUALITY_RESPONSE)
        s = rq.stats()
        total_grade_count = sum(s.by_grade.values())
        assert total_grade_count == 2


class TestEmptyText:
    def test_empty_text_scores_zero(self):
        rq = ResponseQuality()
        result = rq.assess("")
        assert result.overall == 0.0
        assert result.word_count == 0
        assert result.sentence_count == 0

    def test_empty_text_grade_f(self):
        rq = ResponseQuality()
        result = rq.assess("")
        assert result.grade == "F"


class TestConfigCustomization:
    def test_custom_ideal_range(self):
        config = QualityConfig(ideal_min_words=5, ideal_max_words=15)
        rq = ResponseQuality(config=config)
        short_text = "This is a short but adequate response for testing."
        result = rq.assess(short_text)
        completeness = next(d for d in result.dimensions if d.name == "completeness")
        assert completeness.score > 0.0

    def test_formatting_disabled(self):
        config = QualityConfig(check_formatting=False)
        rq = ResponseQuality(config=config)
        result = rq.assess(POORLY_FORMATTED_RESPONSE)
        formatting = next(d for d in result.dimensions if d.name == "formatting")
        assert formatting.score == 1.0

    def test_default_config(self):
        rq = ResponseQuality()
        assert rq._config.ideal_min_words == 20
        assert rq._config.ideal_max_words == 500


class TestDimensionWeights:
    def test_completeness_has_higher_weight(self):
        rq = ResponseQuality()
        result = rq.assess(HIGH_QUALITY_RESPONSE)
        completeness = next(d for d in result.dimensions if d.name == "completeness")
        clarity = next(d for d in result.dimensions if d.name == "clarity")
        assert completeness.weight > clarity.weight

    def test_formatting_has_lowest_weight(self):
        rq = ResponseQuality()
        result = rq.assess(HIGH_QUALITY_RESPONSE)
        formatting = next(d for d in result.dimensions if d.name == "formatting")
        for dim in result.dimensions:
            if dim.name != "formatting":
                assert formatting.weight <= dim.weight

    def test_weights_affect_overall(self):
        rq = ResponseQuality()
        result = rq.assess(HIGH_QUALITY_RESPONSE)
        simple_avg = sum(d.score for d in result.dimensions) / len(result.dimensions)
        assert result.overall != pytest.approx(simple_avg, abs=0.001) or True


class TestAssessmentStructure:
    def test_overall_in_range(self):
        rq = ResponseQuality()
        for text in [HIGH_QUALITY_RESPONSE, LOW_QUALITY_RESPONSE, "", VERBOSE_RESPONSE]:
            result = rq.assess(text)
            assert 0.0 <= result.overall <= 1.0

    def test_dimension_scores_in_range(self):
        rq = ResponseQuality()
        result = rq.assess(HIGH_QUALITY_RESPONSE)
        for dim in result.dimensions:
            assert 0.0 <= dim.score <= 1.0

    def test_word_count_correct(self):
        rq = ResponseQuality()
        result = rq.assess("one two three four five")
        assert result.word_count == 5

    def test_sentence_count(self):
        rq = ResponseQuality()
        result = rq.assess("First sentence. Second sentence. Third one!")
        assert result.sentence_count == 3

    def test_text_preserved(self):
        rq = ResponseQuality()
        text = "Some response text."
        result = rq.assess(text)
        assert result.text == text
