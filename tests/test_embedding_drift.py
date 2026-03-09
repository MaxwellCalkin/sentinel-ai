"""Tests for embedding drift monitor."""

import pytest
from sentinel.embedding_drift import (
    EmbeddingDrift,
    EmbeddingRecord,
    DriftCheck,
    DriftStats,
    DriftReport,
)


# ---------------------------------------------------------------------------
# Recording embeddings
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_returns_drift_check_and_increments_history(self):
        monitor = EmbeddingDrift()
        check = monitor.record([0.1, 0.2, 0.3])
        assert isinstance(check, DriftCheck)
        monitor.record([0.2, 0.3, 0.4], label="second")
        monitor.record([0.3, 0.4, 0.5], label="third")
        assert monitor.history_count == 3

    def test_window_trimming(self):
        monitor = EmbeddingDrift(window_size=3)
        for i in range(10):
            monitor.record([float(i), 1.0])
        assert monitor.history_count == 3


# ---------------------------------------------------------------------------
# Cosine similarity between consecutive embeddings
# ---------------------------------------------------------------------------

class TestConsecutiveSimilarity:
    def test_first_record_has_similarity_one(self):
        monitor = EmbeddingDrift()
        check = monitor.record([0.5, 0.5, 0.5])
        assert check.similarity == 1.0

    def test_identical_consecutive_embeddings(self):
        monitor = EmbeddingDrift()
        monitor.record([1.0, 0.0, 0.0])
        check = monitor.record([1.0, 0.0, 0.0])
        assert check.similarity == pytest.approx(1.0)

    def test_orthogonal_embeddings_have_zero_similarity(self):
        monitor = EmbeddingDrift()
        monitor.record([1.0, 0.0])
        check = monitor.record([0.0, 1.0])
        assert check.similarity == pytest.approx(0.0, abs=1e-9)

    def test_similar_embeddings_high_similarity(self):
        monitor = EmbeddingDrift()
        monitor.record([1.0, 0.0, 0.0])
        check = monitor.record([0.99, 0.01, 0.0])
        assert check.similarity > 0.99


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_no_drift_above_threshold(self):
        monitor = EmbeddingDrift(threshold=0.8)
        monitor.record([1.0, 0.0, 0.0])
        check = monitor.record([0.95, 0.05, 0.0])
        assert not check.drifted

    def test_drift_below_threshold_consecutive(self):
        monitor = EmbeddingDrift(threshold=0.95)
        monitor.record([1.0, 0.0])
        check = monitor.record([0.0, 1.0])
        assert check.drifted

    def test_drift_triggered_by_baseline_divergence(self):
        monitor = EmbeddingDrift(threshold=0.9)
        monitor.set_baseline([1.0, 0.0, 0.0])
        check = monitor.record([0.0, 1.0, 0.0])
        assert check.drifted
        assert check.baseline_similarity is not None
        assert check.baseline_similarity < 0.9


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_no_baseline_by_default(self):
        monitor = EmbeddingDrift()
        assert monitor.baseline is None
        check = monitor.record([1.0, 0.0])
        assert check.baseline_similarity is None

    def test_set_baseline_and_compare(self):
        monitor = EmbeddingDrift()
        monitor.set_baseline([1.0, 0.0])
        assert monitor.baseline == [1.0, 0.0]
        check = monitor.record([1.0, 0.0])
        assert check.baseline_similarity == pytest.approx(1.0)

    def test_baseline_is_copied_not_aliased(self):
        original = [1.0, 0.0, 0.0]
        monitor = EmbeddingDrift()
        monitor.set_baseline(original)
        original[0] = 999.0
        assert monitor.baseline == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Rolling window statistics
# ---------------------------------------------------------------------------

class TestRollingStats:
    def test_empty_stats(self):
        monitor = EmbeddingDrift()
        stats = monitor.rolling_stats()
        assert stats.total_checks == 0
        assert stats.mean_similarity == 0.0
        assert stats.drift_count == 0

    def test_stats_reflect_recorded_data(self):
        monitor = EmbeddingDrift(threshold=0.5)
        monitor.record([1.0, 0.0])
        monitor.record([0.9, 0.1])
        monitor.record([0.8, 0.2])
        stats = monitor.rolling_stats()
        assert stats.total_checks == 3
        assert stats.min_similarity <= stats.mean_similarity
        assert stats.max_similarity >= stats.mean_similarity

    def test_stats_custom_window_and_drift_count(self):
        monitor = EmbeddingDrift(threshold=0.99)
        monitor.record([1.0, 0.0])
        monitor.record([1.0, 0.0])  # identical -> no drift
        monitor.record([0.0, 1.0])  # orthogonal -> drift
        for i in range(7):
            monitor.record([float(i), 1.0])
        stats = monitor.rolling_stats(window=3)
        assert stats.total_checks == 3


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure_and_types(self):
        monitor = EmbeddingDrift()
        monitor.record([1.0, 0.0])
        monitor.record([0.9, 0.1])
        report = monitor.report()
        assert isinstance(report, DriftReport)
        assert isinstance(report.stats, DriftStats)
        assert all(isinstance(c, DriftCheck) for c in report.recent_checks)

    def test_report_baseline_flag(self):
        monitor = EmbeddingDrift()
        assert monitor.report().baseline_set is False
        monitor.set_baseline([1.0, 0.0])
        assert monitor.report().baseline_set is True

    def test_report_custom_window(self):
        monitor = EmbeddingDrift()
        for i in range(10):
            monitor.record([float(i), 1.0])
        report = monitor.report(window=5)
        assert len(report.recent_checks) == 5
        assert report.stats.total_checks == 5


# ---------------------------------------------------------------------------
# Clear and reset
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_everything(self):
        monitor = EmbeddingDrift()
        monitor.set_baseline([1.0, 0.0])
        monitor.record([0.5, 0.5])
        monitor.clear()
        assert monitor.history_count == 0
        assert monitor.check_count == 0
        assert monitor.baseline is None

    def test_clear_history_keeps_baseline(self):
        monitor = EmbeddingDrift()
        monitor.set_baseline([1.0, 0.0])
        monitor.record([0.5, 0.5])
        monitor.clear_history()
        assert monitor.history_count == 0
        assert monitor.check_count == 0
        assert monitor.baseline == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_vector_gives_zero_similarity(self):
        monitor = EmbeddingDrift()
        monitor.record([0.0, 0.0, 0.0])
        check = monitor.record([1.0, 0.0, 0.0])
        assert check.similarity == 0.0

    def test_custom_timestamp(self):
        monitor = EmbeddingDrift()
        monitor.record([1.0, 0.0], timestamp=1000.0)
        monitor.record([0.9, 0.1], timestamp=2000.0)
        assert monitor.history_count == 2
