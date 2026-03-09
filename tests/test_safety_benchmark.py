"""Tests for standardized safety benchmark framework."""

import pytest
from sentinel.safety_benchmark import (
    SafetyBenchmark,
    BenchmarkCase,
    BenchmarkReport,
    CaseResult,
    CategoryStats,
)


def _always_safe(text: str) -> bool:
    """Guard that marks everything as safe."""
    return True


def _always_block(text: str) -> bool:
    """Guard that blocks everything."""
    return False


def _keyword_guard(text: str) -> bool:
    """Guard that blocks text containing 'danger'."""
    return "danger" not in text.lower()


# ---------------------------------------------------------------------------
# TestCases: adding and querying cases
# ---------------------------------------------------------------------------

class TestCases:
    def test_add_case(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("hello", expected_safe=True, category="benign")
        cases = benchmark.filter_cases("benign")
        assert len(cases) == 1
        assert cases[0].input_text == "hello"
        assert cases[0].expected_safe is True
        assert cases[0].category == "benign"

    def test_add_suite(self):
        benchmark = SafetyBenchmark()
        benchmark.add_suite("injection-tests", [
            ("ignore instructions", False, "injection"),
            ("hello world", True, "benign"),
            ("bypass safety", False, "injection"),
        ])
        assert len(benchmark.filter_cases("injection")) == 2
        assert len(benchmark.filter_cases("benign")) == 1

    def test_filter_cases(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("safe text", expected_safe=True, category="benign")
        benchmark.add_case("bad text", expected_safe=False, category="injection")
        benchmark.add_case("another safe", expected_safe=True, category="benign")
        filtered = benchmark.filter_cases("benign")
        assert len(filtered) == 2
        assert all(c.category == "benign" for c in filtered)

    def test_list_categories(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("a", expected_safe=True, category="benign")
        benchmark.add_case("b", expected_safe=False, category="injection")
        benchmark.add_case("c", expected_safe=False, category="toxicity")
        benchmark.add_case("d", expected_safe=True, category="benign")
        categories = benchmark.list_categories()
        assert categories == ["benign", "injection", "toxicity"]


# ---------------------------------------------------------------------------
# TestRun: executing benchmarks against guard functions
# ---------------------------------------------------------------------------

class TestRun:
    def test_perfect_guard(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("safe text", expected_safe=True)
        benchmark.add_case("danger zone", expected_safe=False)
        report = benchmark.run(_keyword_guard)
        assert report.accuracy == 1.0
        assert report.total_correct == 2
        assert report.total_cases == 2

    def test_bad_guard(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("danger zone", expected_safe=False)
        benchmark.add_case("more danger", expected_safe=False)
        # _always_safe lets everything through, so unsafe inputs are missed
        report = benchmark.run(_always_safe)
        assert report.accuracy == 0.0
        assert report.total_correct == 0

    def test_mixed_results(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("safe text", expected_safe=True)
        benchmark.add_case("danger here", expected_safe=False)
        benchmark.add_case("also safe", expected_safe=True)
        benchmark.add_case("no risk", expected_safe=False)
        # _keyword_guard: blocks "danger", allows "no risk"
        # Case 1: safe text -> safe (correct)
        # Case 2: danger here -> blocked (correct)
        # Case 3: also safe -> safe (correct)
        # Case 4: no risk expected unsafe -> guard says safe (wrong, FN)
        report = benchmark.run(_keyword_guard)
        assert report.total_cases == 4
        assert report.total_correct == 3
        assert report.accuracy == 0.75


# ---------------------------------------------------------------------------
# TestReport: report structure and computed fields
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure(self):
        benchmark = SafetyBenchmark(name="test-suite")
        benchmark.add_case("hello", expected_safe=True, category="benign")
        benchmark.add_case("danger", expected_safe=False, category="injection")
        report = benchmark.run(_keyword_guard)
        assert report.name == "test-suite"
        assert isinstance(report.total_cases, int)
        assert isinstance(report.accuracy, float)
        assert isinstance(report.avg_latency_ms, float)
        assert isinstance(report.p95_latency_ms, float)
        assert isinstance(report.categories, list)
        assert isinstance(report.results, list)
        assert isinstance(report.passed, bool)

    def test_accuracy_calculation(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("safe", expected_safe=True)
        benchmark.add_case("danger", expected_safe=False)
        benchmark.add_case("also safe", expected_safe=True)
        benchmark.add_case("fine text", expected_safe=False)
        # _keyword_guard: 3 correct out of 4 ("fine text" is FN)
        report = benchmark.run(_keyword_guard)
        assert report.accuracy == pytest.approx(0.75)

    def test_latency_tracking(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("test", expected_safe=True)
        report = benchmark.run(_always_safe)
        assert report.avg_latency_ms >= 0.0
        assert report.p95_latency_ms >= 0.0
        assert all(r.latency_ms >= 0.0 for r in report.results)


# ---------------------------------------------------------------------------
# TestCategories: per-category statistics
# ---------------------------------------------------------------------------

class TestCategories:
    def test_category_stats(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("safe", expected_safe=True, category="benign")
        benchmark.add_case("danger", expected_safe=False, category="injection")
        report = benchmark.run(_keyword_guard)
        assert len(report.categories) == 2
        category_names = [c.category for c in report.categories]
        assert "benign" in category_names
        assert "injection" in category_names

    def test_false_positive_count(self):
        benchmark = SafetyBenchmark()
        # _always_block blocks safe text = false positive
        benchmark.add_case("perfectly safe", expected_safe=True, category="benign")
        benchmark.add_case("also safe", expected_safe=True, category="benign")
        report = benchmark.run(_always_block)
        benign_stats = next(c for c in report.categories if c.category == "benign")
        assert benign_stats.false_positives == 2
        assert benign_stats.false_negatives == 0

    def test_false_negative_count(self):
        benchmark = SafetyBenchmark()
        # _always_safe lets unsafe text through = false negative
        benchmark.add_case("danger zone", expected_safe=False, category="injection")
        benchmark.add_case("more danger", expected_safe=False, category="injection")
        report = benchmark.run(_always_safe)
        injection_stats = next(c for c in report.categories if c.category == "injection")
        assert injection_stats.false_negatives == 2
        assert injection_stats.false_positives == 0


# ---------------------------------------------------------------------------
# TestThreshold: pass/fail threshold behavior
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_pass_above_threshold(self):
        benchmark = SafetyBenchmark(threshold=0.5)
        benchmark.add_case("safe text", expected_safe=True)
        benchmark.add_case("danger text", expected_safe=False)
        report = benchmark.run(_keyword_guard)
        assert report.accuracy == 1.0
        assert report.passed is True

    def test_fail_below_threshold(self):
        benchmark = SafetyBenchmark(threshold=1.0)
        benchmark.add_case("safe text", expected_safe=True)
        benchmark.add_case("no keyword here", expected_safe=False)
        # _keyword_guard allows "no keyword here" -> FN -> accuracy = 0.5
        report = benchmark.run(_keyword_guard)
        assert report.accuracy < 1.0
        assert report.passed is False


# ---------------------------------------------------------------------------
# TestClear: resetting the benchmark
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("hello", expected_safe=True)
        benchmark.add_case("danger", expected_safe=False)
        assert len(benchmark.list_categories()) > 0
        benchmark.clear()
        assert benchmark.list_categories() == []
        assert benchmark.filter_cases("general") == []


# ---------------------------------------------------------------------------
# TestEdge: edge cases
# ---------------------------------------------------------------------------

class TestEdge:
    def test_empty_benchmark(self):
        benchmark = SafetyBenchmark()
        report = benchmark.run(_always_safe)
        assert report.total_cases == 0
        assert report.total_correct == 0
        assert report.accuracy == 0.0
        assert report.avg_latency_ms == 0.0
        assert report.p95_latency_ms == 0.0
        assert report.categories == []
        assert report.results == []
        assert report.passed is False

    def test_single_case(self):
        benchmark = SafetyBenchmark()
        benchmark.add_case("just one", expected_safe=True)
        report = benchmark.run(_always_safe)
        assert report.total_cases == 1
        assert report.total_correct == 1
        assert report.accuracy == 1.0
        assert len(report.results) == 1
        assert report.results[0].correct is True
