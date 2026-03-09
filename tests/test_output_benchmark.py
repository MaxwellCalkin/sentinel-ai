"""Tests for output benchmark."""

import pytest
from sentinel.output_benchmark import OutputBenchmark, BenchmarkScore, DimensionScore


class TestEvaluation:
    def test_perfect_match(self):
        b = OutputBenchmark()
        score = b.evaluate(
            output="Paris is the capital of France.",
            expected="Paris is the capital of France.",
            prompt="What is the capital of France?",
        )
        assert score.overall > 0.7
        assert score.passed

    def test_poor_match(self):
        b = OutputBenchmark()
        score = b.evaluate(
            output="Bananas are yellow.",
            expected="Paris is the capital of France.",
            prompt="What is the capital of France?",
        )
        assert score.overall < 0.7

    def test_no_expected(self):
        b = OutputBenchmark()
        score = b.evaluate(
            output="The answer is 42.",
            prompt="What is the meaning of life?",
        )
        assert 0.0 <= score.overall <= 1.0

    def test_empty_output(self):
        b = OutputBenchmark()
        score = b.evaluate(output="", expected="Something")
        assert score.overall < 0.5


class TestDimensions:
    def test_all_dimensions_present(self):
        b = OutputBenchmark()
        score = b.evaluate(output="Test output", expected="Expected")
        dim_names = [d.name for d in score.dimensions]
        assert "accuracy" in dim_names
        assert "relevance" in dim_names
        assert "completeness" in dim_names
        assert "safety" in dim_names
        assert "format" in dim_names

    def test_safety_dimension(self):
        b = OutputBenchmark()
        score = b.evaluate(output="Here is how to hack and exploit systems")
        safety = next(d for d in score.dimensions if d.name == "safety")
        assert safety.score < 1.0

    def test_safe_output(self):
        b = OutputBenchmark()
        score = b.evaluate(output="Python is a programming language")
        safety = next(d for d in score.dimensions if d.name == "safety")
        assert safety.score == 1.0


class TestGrading:
    def test_grade_a(self):
        b = OutputBenchmark()
        score = b.evaluate(
            output="Paris is the capital of France and a major European city.",
            expected="Paris is the capital of France.",
            prompt="What is the capital of France?",
        )
        assert score.grade in ["A", "B"]  # high quality

    def test_grade_f(self):
        b = OutputBenchmark()
        score = b.evaluate(output="", expected="Something important")
        assert score.grade == "F"

    def test_pass_threshold(self):
        b = OutputBenchmark(pass_threshold=0.9)
        score = b.evaluate(output="Partial answer", expected="Full detailed answer")
        assert isinstance(score.passed, bool)


class TestBatch:
    def test_batch_evaluate(self):
        b = OutputBenchmark()
        results = b.evaluate_batch([
            {"output": "Paris", "expected": "Paris", "prompt": "Capital?"},
            {"output": "Wrong", "expected": "Right", "prompt": "Answer?"},
        ])
        assert len(results) == 2


class TestStructure:
    def test_score_structure(self):
        b = OutputBenchmark()
        score = b.evaluate(output="Test")
        assert isinstance(score, BenchmarkScore)
        assert isinstance(score.dimensions, list)
        assert 0.0 <= score.overall <= 1.0
        assert score.grade in ["A", "B", "C", "D", "F"]

    def test_dimension_structure(self):
        b = OutputBenchmark()
        score = b.evaluate(output="Test")
        dim = score.dimensions[0]
        assert isinstance(dim, DimensionScore)
        assert isinstance(dim.weight, float)
        assert isinstance(dim.details, str)
