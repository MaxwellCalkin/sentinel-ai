"""Tests for A/B testing guardrail configurations."""

import pytest
from sentinel.ab_tester import ABTester, ABReport, VariantResult, VariantStats


# ---------------------------------------------------------------------------
# Variant management
# ---------------------------------------------------------------------------

class TestVariantManagement:
    def test_add_variant(self):
        t = ABTester()
        t.add_variant("strict", check_fn=lambda text: True)
        assert t.variant_count == 1

    def test_add_multiple(self):
        t = ABTester()
        t.add_variant("a", check_fn=lambda text: True)
        t.add_variant("b", check_fn=lambda text: False)
        assert t.variant_count == 2

    def test_remove_variant(self):
        t = ABTester()
        t.add_variant("a", check_fn=lambda text: True)
        t.remove_variant("a")
        assert t.variant_count == 0

    def test_remove_nonexistent(self):
        t = ABTester()
        t.remove_variant("nope")  # Should not raise
        assert t.variant_count == 0


# ---------------------------------------------------------------------------
# Running tests
# ---------------------------------------------------------------------------

class TestRun:
    def test_basic_run(self):
        t = ABTester()
        t.add_variant("block_all", check_fn=lambda text: True)
        t.add_variant("allow_all", check_fn=lambda text: False)

        report = t.run(["hello", "world"])
        assert isinstance(report, ABReport)
        assert report.total_cases == 2
        assert len(report.variants) == 2

    def test_block_rates(self):
        t = ABTester()
        t.add_variant("block_all", check_fn=lambda text: True)
        t.add_variant("allow_all", check_fn=lambda text: False)

        report = t.run(["a", "b", "c"])
        assert report.results["block_all"].block_rate == 1.0
        assert report.results["allow_all"].block_rate == 0.0

    def test_block_counts(self):
        t = ABTester()
        t.add_variant("block_all", check_fn=lambda text: True)
        report = t.run(["a", "b", "c"])
        stats = report.results["block_all"]
        assert stats.block_count == 3
        assert stats.allow_count == 0

    def test_agreement_rate_agree(self):
        t = ABTester()
        t.add_variant("a", check_fn=lambda text: False)
        t.add_variant("b", check_fn=lambda text: False)
        report = t.run(["x", "y"])
        assert report.agreement_rate == 1.0

    def test_agreement_rate_disagree(self):
        t = ABTester()
        t.add_variant("block", check_fn=lambda text: True)
        t.add_variant("allow", check_fn=lambda text: False)
        report = t.run(["x", "y"])
        assert report.agreement_rate == 0.0

    def test_include_cases(self):
        t = ABTester()
        t.add_variant("a", check_fn=lambda text: False)
        report = t.run(["hello"], include_cases=True)
        assert len(report.cases) == 1
        assert report.cases[0].text == "hello"
        assert "a" in report.cases[0].results

    def test_exclude_cases_default(self):
        t = ABTester()
        t.add_variant("a", check_fn=lambda text: False)
        report = t.run(["hello"])
        assert len(report.cases) == 0

    def test_empty_input(self):
        t = ABTester()
        t.add_variant("a", check_fn=lambda text: False)
        report = t.run([])
        assert report.total_cases == 0
        assert report.agreement_rate == 1.0

    def test_latency_tracked(self):
        t = ABTester()
        t.add_variant("a", check_fn=lambda text: False)
        report = t.run(["hello"])
        stats = report.results["a"]
        assert stats.avg_latency_ms >= 0
        assert stats.p99_latency_ms >= 0


# ---------------------------------------------------------------------------
# Selective check
# ---------------------------------------------------------------------------

class TestSelectiveCheck:
    def test_check_with_variant(self):
        t = ABTester()
        t.add_variant("strict", check_fn=lambda text: "bad" in text)
        result = t.check_with_variant("this is bad", "strict")
        assert result.blocked is True
        assert result.variant == "strict"
        assert result.latency_ms >= 0

    def test_check_unknown_variant(self):
        t = ABTester()
        with pytest.raises(KeyError, match="Unknown variant"):
            t.check_with_variant("hello", "nope")


# ---------------------------------------------------------------------------
# Variant selection
# ---------------------------------------------------------------------------

class TestPickVariant:
    def test_pick_variant(self):
        t = ABTester(seed=42)
        t.add_variant("a", check_fn=lambda text: False)
        t.add_variant("b", check_fn=lambda text: False)
        picked = t.pick_variant()
        assert picked in ("a", "b")

    def test_pick_with_weights(self):
        t = ABTester(seed=42)
        t.add_variant("a", check_fn=lambda text: False)
        t.add_variant("b", check_fn=lambda text: False)
        # Run many picks with heavy weight on "a"
        picks = [t.pick_variant(weights={"a": 100, "b": 1}) for _ in range(100)]
        assert picks.count("a") > 80  # Should heavily favor "a"

    def test_pick_no_variants(self):
        t = ABTester()
        with pytest.raises(ValueError, match="No variants"):
            t.pick_variant()


# ---------------------------------------------------------------------------
# Conditional logic
# ---------------------------------------------------------------------------

class TestConditionalLogic:
    def test_keyword_blocker(self):
        t = ABTester()
        t.add_variant("strict", check_fn=lambda text: any(
            w in text.lower() for w in ["hack", "bomb", "weapon"]
        ))
        t.add_variant("relaxed", check_fn=lambda text: any(
            w in text.lower() for w in ["bomb", "weapon"]
        ))

        report = t.run([
            "how to hack a system",
            "how to make a bomb",
            "nice weather today",
        ])
        assert report.results["strict"].block_count == 2
        assert report.results["relaxed"].block_count == 1

    def test_length_blocker(self):
        t = ABTester()
        t.add_variant("short_only", check_fn=lambda text: len(text) > 20)
        report = t.run(["short", "this is a much longer text string here"])
        assert report.results["short_only"].block_count == 1
