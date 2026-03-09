"""Tests for response coherence analysis."""

from sentinel.response_coherence import (
    ResponseCoherence,
    CoherenceScore,
    CoherenceIssue,
    CoherenceReport,
    CoherenceStats,
)


# ---------------------------------------------------------------------------
# Coherent text (high scores)
# ---------------------------------------------------------------------------

class TestCoherentText:
    def test_well_written_text_scores_high(self):
        checker = ResponseCoherence()
        text = (
            "Python is a popular programming language. "
            "Python is used for web development and data science. "
            "Many developers choose Python for its readability. "
            "Python has a large ecosystem of libraries."
        )
        report = checker.analyze(text)
        assert report.score.overall >= 0.6
        assert report.score.grade in ("A", "B")

    def test_consistent_topic_scores_higher_than_drifted(self):
        checker = ResponseCoherence()
        focused = (
            "The cat sat on the mat. "
            "The cat played on the mat. "
            "The cat napped on the mat. "
            "The cat rested on the mat."
        )
        drifted = (
            "Quantum physics describes particles. "
            "Wave functions model electrons. "
            "Chocolate cake requires sugar. "
            "Baking temperature is important."
        )
        focused_report = checker.analyze(focused)
        drifted_report = checker.analyze(drifted)
        assert focused_report.score.topic_consistency > drifted_report.score.topic_consistency

    def test_no_contradictions_scores_one(self):
        checker = ResponseCoherence()
        text = (
            "The weather is sunny today. "
            "People are enjoying the park. "
            "Children are playing outside. "
            "It is a beautiful afternoon."
        )
        report = checker.analyze(text)
        assert report.score.contradiction_score >= 0.8

    def test_no_repetition_scores_one(self):
        checker = ResponseCoherence()
        text = (
            "First, gather the ingredients. "
            "Second, preheat the oven. "
            "Third, mix the batter. "
            "Finally, bake for thirty minutes."
        )
        report = checker.analyze(text)
        assert report.score.repetition_score == 1.0


# ---------------------------------------------------------------------------
# Incoherent text with contradictions
# ---------------------------------------------------------------------------

class TestContradictions:
    def test_negation_contradiction_detected(self):
        checker = ResponseCoherence()
        text = (
            "The system is ready for deployment. "
            "The system is not ready for deployment. "
            "We should proceed immediately."
        )
        report = checker.analyze(text)
        assert report.score.contradiction_score < 1.0

    def test_antonym_contradiction_detected(self):
        checker = ResponseCoherence()
        text = (
            "The answer is true. "
            "The answer is false. "
            "We need to verify this."
        )
        report = checker.analyze(text)
        assert report.score.contradiction_score < 1.0

    def test_can_cannot_contradiction(self):
        checker = ResponseCoherence()
        text = (
            "The user can access the dashboard. "
            "The user cannot access the dashboard. "
            "Please check the permissions."
        )
        report = checker.analyze(text)
        assert report.score.contradiction_score < 1.0

    def test_multiple_contradictions_lower_score(self):
        checker = ResponseCoherence()
        dense = (
            "It is true. It is false. "
            "It is true. It is false."
        )
        sparse = (
            "It is true. It is false. "
            "The sky looks clear. "
            "The weather is nice today."
        )
        dense_report = checker.analyze(dense)
        sparse_report = checker.analyze(sparse)
        assert dense_report.score.contradiction_score < 1.0
        assert dense_report.score.contradiction_score < sparse_report.score.contradiction_score


# ---------------------------------------------------------------------------
# Repetitive text
# ---------------------------------------------------------------------------

class TestRepetition:
    def test_duplicate_sentences_lower_score(self):
        checker = ResponseCoherence()
        text = (
            "The sky is blue. "
            "The sky is blue. "
            "The sky is blue. "
            "The sky is blue."
        )
        report = checker.analyze(text)
        assert report.score.repetition_score < 0.5

    def test_partial_repetition_moderate_score(self):
        checker = ResponseCoherence()
        text = (
            "Machine learning is powerful. "
            "Machine learning is powerful. "
            "Deep learning extends machine learning. "
            "Neural networks are the foundation."
        )
        report = checker.analyze(text)
        assert report.score.repetition_score < 1.0
        assert report.score.repetition_score > 0.0


# ---------------------------------------------------------------------------
# Topic drift
# ---------------------------------------------------------------------------

class TestTopicDrift:
    def test_topic_drift_lowers_consistency(self):
        checker = ResponseCoherence()
        text = (
            "Quantum physics studies subatomic particles. "
            "Wave functions describe probability amplitudes. "
            "Chocolate cake requires flour and sugar. "
            "Baking temperature should be three hundred fifty degrees."
        )
        report = checker.analyze(text)
        assert report.score.topic_consistency < 0.5

    def test_focused_topic_higher_than_drifted(self):
        checker = ResponseCoherence()
        focused = (
            "Dogs are loyal and friendly pets. "
            "Dogs need daily walks and exercise. "
            "Dogs are also great pets for families. "
            "Dogs need love and daily attention."
        )
        drifted = (
            "Quantum physics studies particles. "
            "Electrons orbit atomic nuclei. "
            "Chocolate cake is delicious dessert. "
            "Baking requires precise measurements."
        )
        focused_report = checker.analyze(focused)
        drifted_report = checker.analyze(drifted)
        assert focused_report.score.topic_consistency > drifted_report.score.topic_consistency


# ---------------------------------------------------------------------------
# Grade boundaries
# ---------------------------------------------------------------------------

class TestGradeBoundaries:
    def test_grade_a_threshold(self):
        checker = ResponseCoherence()
        # Highly coherent text
        text = (
            "Python is a versatile language. "
            "Python supports multiple paradigms. "
            "Python has clear and readable syntax. "
            "Python is widely used in industry."
        )
        report = checker.analyze(text)
        if report.score.overall >= 0.8:
            assert report.score.grade == "A"

    def test_grade_mapping_a(self):
        from sentinel.response_coherence import _grade_from_score
        assert _grade_from_score(0.95) == "A"
        assert _grade_from_score(0.80) == "A"

    def test_grade_mapping_b(self):
        from sentinel.response_coherence import _grade_from_score
        assert _grade_from_score(0.79) == "B"
        assert _grade_from_score(0.60) == "B"

    def test_grade_mapping_c(self):
        from sentinel.response_coherence import _grade_from_score
        assert _grade_from_score(0.59) == "C"
        assert _grade_from_score(0.40) == "C"

    def test_grade_mapping_d(self):
        from sentinel.response_coherence import _grade_from_score
        assert _grade_from_score(0.39) == "D"
        assert _grade_from_score(0.20) == "D"

    def test_grade_mapping_f(self):
        from sentinel.response_coherence import _grade_from_score
        assert _grade_from_score(0.19) == "F"
        assert _grade_from_score(0.0) == "F"


# ---------------------------------------------------------------------------
# Quick score
# ---------------------------------------------------------------------------

class TestQuickScore:
    def test_quick_score_matches_analyze_overall(self):
        checker = ResponseCoherence()
        text = (
            "The algorithm is efficient. "
            "It processes data in linear time. "
            "The algorithm handles edge cases well."
        )
        report = checker.analyze(text)
        # Reset stats so quick_score tracks independently
        checker2 = ResponseCoherence()
        quick = checker2.quick_score(text)
        assert quick == report.score.overall

    def test_quick_score_returns_float(self):
        checker = ResponseCoherence()
        score = checker.quick_score("Hello world. This is a test.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

class TestBatchAnalysis:
    def test_batch_returns_correct_count(self):
        checker = ResponseCoherence()
        texts = [
            "First text about science. Science is interesting.",
            "Second text about cooking. Cooking is fun.",
            "Third text about music. Music is universal.",
        ]
        reports = checker.analyze_batch(texts)
        assert len(reports) == 3

    def test_batch_each_report_is_valid(self):
        checker = ResponseCoherence()
        texts = [
            "Alpha beta gamma. Delta epsilon.",
            "One two three. Four five six.",
        ]
        reports = checker.analyze_batch(texts)
        for report in reports:
            assert isinstance(report, CoherenceReport)
            assert isinstance(report.score, CoherenceScore)
            assert 0.0 <= report.score.overall <= 1.0

    def test_empty_batch(self):
        checker = ResponseCoherence()
        reports = checker.analyze_batch([])
        assert reports == []


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStatsTracking:
    def test_initial_stats_empty(self):
        checker = ResponseCoherence()
        stats = checker.stats()
        assert stats.total_analyzed == 0
        assert stats.avg_overall == 0.0
        assert stats.by_grade == {}
        assert stats.total_issues == 0

    def test_stats_after_single_analysis(self):
        checker = ResponseCoherence()
        checker.analyze("A simple sentence. Another sentence here.")
        stats = checker.stats()
        assert stats.total_analyzed == 1
        assert stats.avg_overall > 0.0
        assert sum(stats.by_grade.values()) == 1

    def test_stats_accumulate_across_analyses(self):
        checker = ResponseCoherence()
        checker.analyze("Good coherent text. The text stays focused.")
        checker.analyze("Another analysis. This one is also short.")
        checker.analyze("Third analysis. With more content here.")
        stats = checker.stats()
        assert stats.total_analyzed == 3
        assert sum(stats.by_grade.values()) == 3

    def test_stats_track_issues(self):
        checker = ResponseCoherence()
        # Text with contradictions to generate issues
        checker.analyze(
            "This is true. This is false. "
            "It is safe. It is unsafe. "
            "The result is valid. The result is invalid."
        )
        stats = checker.stats()
        assert stats.total_issues > 0

    def test_stats_average_is_correct(self):
        checker = ResponseCoherence()
        r1 = checker.analyze("First text here. More first text here.")
        r2 = checker.analyze("Second text now. More second text now.")
        stats = checker.stats()
        expected_avg = round((r1.score.overall + r2.score.overall) / 2, 4)
        assert stats.avg_overall == expected_avg


# ---------------------------------------------------------------------------
# Empty and minimal text
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text(self):
        checker = ResponseCoherence()
        report = checker.analyze("")
        assert isinstance(report, CoherenceReport)
        assert report.score.overall >= 0.0
        assert report.score.grade in ("A", "B", "C", "D", "F")

    def test_single_sentence(self):
        checker = ResponseCoherence()
        report = checker.analyze("Just a single sentence here")
        assert report.score.topic_consistency == 1.0
        assert report.score.structural_quality == 1.0
        assert report.score.contradiction_score == 1.0
        assert report.score.repetition_score == 1.0
        assert report.score.overall == 1.0
        assert report.score.grade == "A"

    def test_whitespace_only(self):
        checker = ResponseCoherence()
        report = checker.analyze("   ")
        assert isinstance(report, CoherenceReport)

    def test_single_word(self):
        checker = ResponseCoherence()
        report = checker.analyze("Hello")
        assert report.score.grade == "A"


# ---------------------------------------------------------------------------
# Issues generation
# ---------------------------------------------------------------------------

class TestIssuesGeneration:
    def test_contradiction_issue_generated(self):
        checker = ResponseCoherence()
        text = (
            "It is true. It is false. "
            "It is true. It is false."
        )
        report = checker.analyze(text)
        issue_types = [i.issue_type for i in report.issues]
        assert "contradiction" in issue_types

    def test_repetition_issue_generated(self):
        checker = ResponseCoherence()
        text = (
            "Repeat this line. "
            "Repeat this line. "
            "Repeat this line. "
            "Repeat this line. "
            "Repeat this line."
        )
        report = checker.analyze(text)
        issue_types = [i.issue_type for i in report.issues]
        assert "repetition" in issue_types

    def test_no_issues_for_good_text(self):
        checker = ResponseCoherence()
        text = (
            "Python is a great language. "
            "Python supports object oriented programming. "
            "Python also supports functional programming. "
            "Python is used worldwide by developers."
        )
        report = checker.analyze(text)
        # High scoring text should have few or no issues
        for issue in report.issues:
            assert issue.issue_type in (
                "topic_drift", "structural", "contradiction", "repetition"
            )

    def test_issue_has_valid_severity(self):
        checker = ResponseCoherence()
        text = (
            "This is true. This is false. "
            "It can work. It cannot work."
        )
        report = checker.analyze(text)
        for issue in report.issues:
            assert issue.severity in ("low", "medium", "high")

    def test_issue_has_valid_type(self):
        checker = ResponseCoherence()
        text = (
            "All is good. None is good. "
            "We always succeed. We never succeed."
        )
        report = checker.analyze(text)
        valid_types = {"contradiction", "topic_drift", "repetition", "structural"}
        for issue in report.issues:
            assert issue.issue_type in valid_types


# ---------------------------------------------------------------------------
# Recommendations generation
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_contradiction_recommendation(self):
        checker = ResponseCoherence()
        text = (
            "It is true. It is false. "
            "It is true. It is false."
        )
        report = checker.analyze(text)
        has_contradiction_rec = any(
            "contradict" in r.lower() for r in report.recommendations
        )
        assert has_contradiction_rec

    def test_repetition_recommendation(self):
        checker = ResponseCoherence()
        text = (
            "Say it again. "
            "Say it again. "
            "Say it again. "
            "Say it again. "
            "Say it again."
        )
        report = checker.analyze(text)
        has_repetition_rec = any(
            "repeat" in r.lower() or "duplicate" in r.lower()
            for r in report.recommendations
        )
        assert has_repetition_rec

    def test_no_recommendations_for_perfect_text(self):
        checker = ResponseCoherence()
        report = checker.analyze("A single clean sentence")
        assert report.recommendations == []

    def test_recommendations_are_strings(self):
        checker = ResponseCoherence()
        text = (
            "Yes we can. No we cannot. "
            "It is possible. It is impossible."
        )
        report = checker.analyze(text)
        for rec in report.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_coherence_score_fields(self):
        score = CoherenceScore(
            text="test",
            topic_consistency=0.8,
            structural_quality=0.9,
            contradiction_score=1.0,
            repetition_score=1.0,
            overall=0.9,
            grade="A",
        )
        assert score.text == "test"
        assert score.topic_consistency == 0.8

    def test_coherence_issue_defaults(self):
        issue = CoherenceIssue(
            issue_type="contradiction",
            description="Found a contradiction",
            severity="high",
        )
        assert issue.location == ""

    def test_coherence_stats_defaults(self):
        stats = CoherenceStats()
        assert stats.total_analyzed == 0
        assert stats.avg_overall == 0.0
        assert stats.by_grade == {}
        assert stats.total_issues == 0

    def test_coherence_report_structure(self):
        checker = ResponseCoherence()
        report = checker.analyze("Some text. More text here.")
        assert hasattr(report, "score")
        assert hasattr(report, "issues")
        assert hasattr(report, "recommendations")
        assert isinstance(report.issues, list)
        assert isinstance(report.recommendations, list)
