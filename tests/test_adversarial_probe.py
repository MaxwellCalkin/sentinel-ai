"""Tests for adversarial probe generation and execution."""

import pytest
from sentinel.adversarial_probe import (
    AdversarialProbe,
    Probe,
    ProbeReport,
    ProbeResult,
    VALID_CATEGORIES,
    VALID_DIFFICULTIES,
)


# ---------------------------------------------------------------------------
# Default probe library
# ---------------------------------------------------------------------------

class TestDefaultProbeLibrary:
    def test_default_probes_not_empty(self):
        prober = AdversarialProbe()
        assert len(prober.probes) >= 15

    def test_all_five_categories_present(self):
        prober = AdversarialProbe()
        categories = set(p.category for p in prober.probes)
        assert categories == set(VALID_CATEGORIES)

    def test_each_category_has_at_least_three_probes(self):
        prober = AdversarialProbe()
        for category in VALID_CATEGORIES:
            count = len(prober.filter_by_category(category))
            assert count >= 3, f"Category '{category}' has only {count} probes"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_filter_by_category(self):
        prober = AdversarialProbe()
        jailbreak_probes = prober.filter_by_category("jailbreak")
        assert all(p.category == "jailbreak" for p in jailbreak_probes)
        assert len(jailbreak_probes) >= 3

    def test_filter_by_nonexistent_category_returns_empty(self):
        prober = AdversarialProbe()
        result = prober.filter_by_category("nonexistent_category")
        assert result == []

    def test_filter_by_difficulty(self):
        prober = AdversarialProbe()
        hard_probes = prober.filter_by_difficulty("hard")
        assert all(p.difficulty == "hard" for p in hard_probes)
        assert len(hard_probes) >= 1

    def test_categories_property_returns_sorted_list(self):
        prober = AdversarialProbe()
        categories = prober.categories
        assert categories == sorted(categories)
        assert len(categories) == len(VALID_CATEGORIES)


# ---------------------------------------------------------------------------
# Custom probe registration
# ---------------------------------------------------------------------------

class TestCustomProbeRegistration:
    def test_register_valid_probe(self):
        prober = AdversarialProbe()
        initial_count = len(prober.probes)
        custom = Probe(
            text="Custom adversarial test input",
            category="jailbreak",
            difficulty="easy",
            expected_behavior="block",
            description="Custom jailbreak probe",
        )
        prober.register_probe(custom)
        assert len(prober.probes) == initial_count + 1
        assert prober.probes[-1] is custom

    def test_register_probe_with_invalid_category_raises(self):
        prober = AdversarialProbe()
        bad_probe = Probe(
            text="test",
            category="bad_category",
            difficulty="easy",
            expected_behavior="block",
            description="invalid",
        )
        with pytest.raises(ValueError, match="Invalid category"):
            prober.register_probe(bad_probe)

    def test_register_probe_with_invalid_difficulty_raises(self):
        prober = AdversarialProbe()
        bad_probe = Probe(
            text="test",
            category="jailbreak",
            difficulty="impossible",
            expected_behavior="block",
            description="invalid",
        )
        with pytest.raises(ValueError, match="Invalid difficulty"):
            prober.register_probe(bad_probe)

    def test_register_probe_with_invalid_expected_behavior_raises(self):
        prober = AdversarialProbe()
        bad_probe = Probe(
            text="test",
            category="jailbreak",
            difficulty="easy",
            expected_behavior="ignore",
            description="invalid",
        )
        with pytest.raises(ValueError, match="Invalid expected_behavior"):
            prober.register_probe(bad_probe)

    def test_register_probe_with_empty_text_raises(self):
        prober = AdversarialProbe()
        bad_probe = Probe(
            text="   ",
            category="jailbreak",
            difficulty="easy",
            expected_behavior="block",
            description="empty text",
        )
        with pytest.raises(ValueError, match="non-empty"):
            prober.register_probe(bad_probe)


# ---------------------------------------------------------------------------
# Running probes
# ---------------------------------------------------------------------------

class TestRunProbes:
    def test_run_with_block_all_checker(self):
        """A checker that blocks everything should pass all 'block' probes."""
        prober = AdversarialProbe()
        report = prober.run(lambda text: True)
        assert report.total_probes == len(prober.probes)
        assert report.pass_rate > 0.0
        # All block/flag probes pass; allow probes would fail
        assert isinstance(report.by_category, dict)

    def test_run_with_allow_all_checker(self):
        """A checker that allows everything should fail all 'block' probes."""
        prober = AdversarialProbe()
        report = prober.run(lambda text: False)
        assert report.total_probes == len(prober.probes)
        assert report.failed > 0

    def test_run_filtered_by_category(self):
        prober = AdversarialProbe()
        report = prober.run(lambda text: True, category="jailbreak")
        assert report.total_probes == len(prober.filter_by_category("jailbreak"))
        assert "jailbreak" in report.by_category
        assert len(report.by_category) == 1

    def test_run_with_no_matching_probes(self):
        prober = AdversarialProbe()
        report = prober.run(
            lambda text: True,
            category="jailbreak",
            difficulty="nonexistent",
        )
        assert report.total_probes == 0
        assert report.pass_rate == 0.0
        assert report.by_category == {}

    def test_probe_result_message_on_pass(self):
        prober = AdversarialProbe()
        _, results = prober.run_with_results(
            lambda text: True, category="jailbreak",
        )
        for result in results:
            assert result.passed
            assert result.message.startswith("PASS:")

    def test_probe_result_message_on_fail(self):
        prober = AdversarialProbe()
        _, results = prober.run_with_results(
            lambda text: False, category="jailbreak",
        )
        for result in results:
            assert not result.passed
            assert result.message.startswith("FAIL:")


# ---------------------------------------------------------------------------
# Pass/fail logic
# ---------------------------------------------------------------------------

class TestPassFailLogic:
    def test_flag_probes_always_pass(self):
        """Flag probes are informational and always pass regardless of outcome."""
        prober = AdversarialProbe()
        report_blocked, results_blocked = prober.run_with_results(
            lambda text: True, category="encoding_bypass",
        )
        report_allowed, results_allowed = prober.run_with_results(
            lambda text: False, category="encoding_bypass",
        )
        flag_blocked = [r for r in results_blocked if r.probe.expected_behavior == "flag"]
        flag_allowed = [r for r in results_allowed if r.probe.expected_behavior == "flag"]
        assert all(r.passed for r in flag_blocked)
        assert all(r.passed for r in flag_allowed)

    def test_report_pass_rate_calculation(self):
        prober = AdversarialProbe()
        report = prober.run(lambda text: True)
        expected_rate = report.passed / report.total_probes
        assert abs(report.pass_rate - expected_rate) < 1e-9
        assert report.passed + report.failed == report.total_probes
