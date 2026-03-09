"""Tests for token anomaly detection."""

import pytest
from sentinel.token_anomaly import (
    TokenAnomaly,
    TokenRecord,
    AnomalyCheck,
    TokenAnomalyStats,
)


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_all_dataclass_fields(self):
        rec = TokenRecord(input_tokens=10, output_tokens=200, timestamp=1.0, label="user1")
        assert rec.input_tokens == 10
        assert rec.output_tokens == 200
        assert rec.timestamp == 1.0
        assert rec.label == "user1"

        check = AnomalyCheck(
            is_anomalous=True,
            reason="spike",
            input_tokens=5,
            output_tokens=1000,
            ratio=200.0,
            avg_baseline=50.0,
        )
        assert check.is_anomalous is True
        assert check.ratio == 200.0

        stats = TokenAnomalyStats(
            total_records=10, spike_count=2, avg_input=50.0, avg_output=100.0, max_ratio=5.0,
        )
        assert stats.total_records == 10
        assert stats.spike_count == 2


# ---------------------------------------------------------------------------
# Recording and basic tracking
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_returns_anomaly_check(self):
        detector = TokenAnomaly()
        result = detector.record(input_tokens=100, output_tokens=100)
        assert isinstance(result, AnomalyCheck)

    def test_record_count_increments(self):
        detector = TokenAnomaly()
        detector.record(input_tokens=100, output_tokens=100)
        detector.record(input_tokens=200, output_tokens=200)
        assert detector.record_count == 2

    def test_record_with_label_and_custom_timestamp(self):
        detector = TokenAnomaly()
        result = detector.record(input_tokens=50, output_tokens=50, label="api-call-1", timestamp=1000.0)
        assert result.input_tokens == 50
        assert detector.record_count == 1


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

class TestSpikeDetection:
    def test_spike_detected_above_threshold(self):
        detector = TokenAnomaly(spike_threshold=2.0)
        for _ in range(10):
            detector.record(input_tokens=100, output_tokens=100)
        result = detector.record(input_tokens=100, output_tokens=1000)
        assert result.is_anomalous is True
        assert "spike" in result.reason.lower()

    def test_no_spike_below_threshold(self):
        detector = TokenAnomaly(spike_threshold=3.0)
        for _ in range(10):
            detector.record(input_tokens=100, output_tokens=100)
        result = detector.record(input_tokens=100, output_tokens=200)
        assert result.is_anomalous is False
        assert result.reason == ""

    def test_spike_threshold_boundary(self):
        detector = TokenAnomaly(spike_threshold=3.0)
        for _ in range(10):
            detector.record(input_tokens=100, output_tokens=100)
        # Exactly 3x should not trigger (must be strictly greater)
        result = detector.record(input_tokens=100, output_tokens=300)
        assert result.is_anomalous is False

    def test_first_record_no_spike_without_baseline(self):
        detector = TokenAnomaly()
        # Large output but proportional input; no baseline exists yet for spike
        result = detector.record(input_tokens=500, output_tokens=5000)
        assert "spike" not in result.reason.lower()


# ---------------------------------------------------------------------------
# Exfiltration pattern detection
# ---------------------------------------------------------------------------

class TestExfiltrationDetection:
    def test_exfiltration_small_input_huge_output(self):
        detector = TokenAnomaly()
        # Establish a baseline so the spike also fires, but we mainly check exfiltration
        result = detector.record(input_tokens=5, output_tokens=5000)
        # First record has no baseline, but exfiltration is ratio-based
        assert result.ratio >= 20.0

    def test_exfiltration_flagged_with_baseline(self):
        detector = TokenAnomaly(spike_threshold=2.0)
        for _ in range(5):
            detector.record(input_tokens=100, output_tokens=100)
        result = detector.record(input_tokens=10, output_tokens=5000)
        assert result.is_anomalous is True
        assert "exfiltration" in result.reason.lower()

    def test_no_exfiltration_when_output_below_minimum(self):
        detector = TokenAnomaly()
        # High ratio but low absolute output (below 500 threshold)
        result = detector.record(input_tokens=1, output_tokens=100)
        assert "exfiltration" not in result.reason.lower()


# ---------------------------------------------------------------------------
# Input/output ratio
# ---------------------------------------------------------------------------

class TestRatioCalculation:
    def test_ratio_normal(self):
        detector = TokenAnomaly()
        result = detector.record(input_tokens=100, output_tokens=200)
        assert result.ratio == pytest.approx(2.0)

    def test_ratio_zero_input(self):
        detector = TokenAnomaly()
        result = detector.record(input_tokens=0, output_tokens=500)
        assert result.ratio == 500.0

    def test_ratio_zero_both(self):
        detector = TokenAnomaly()
        result = detector.record(input_tokens=0, output_tokens=0)
        assert result.ratio == 0.0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_empty(self):
        detector = TokenAnomaly()
        stats = detector.stats()
        assert stats.total_records == 0
        assert stats.spike_count == 0
        assert stats.avg_input == 0.0
        assert stats.avg_output == 0.0
        assert stats.max_ratio == 0.0

    def test_stats_after_recording(self):
        detector = TokenAnomaly(spike_threshold=2.0)
        detector.record(input_tokens=100, output_tokens=200)
        detector.record(input_tokens=200, output_tokens=400)
        stats = detector.stats()
        assert stats.total_records == 2
        assert stats.avg_input == pytest.approx(150.0)
        assert stats.avg_output == pytest.approx(300.0)
        assert stats.max_ratio == pytest.approx(2.0)

    def test_stats_spike_count(self):
        detector = TokenAnomaly(spike_threshold=2.0)
        for _ in range(10):
            detector.record(input_tokens=100, output_tokens=100)
        detector.record(input_tokens=100, output_tokens=1000)
        stats = detector.stats()
        assert stats.spike_count >= 1


# ---------------------------------------------------------------------------
# Check last
# ---------------------------------------------------------------------------

class TestCheckLast:
    def test_check_last_raises_when_empty(self):
        detector = TokenAnomaly()
        with pytest.raises(ValueError, match="No records"):
            detector.check_last()

    def test_check_last_returns_anomaly_check(self):
        detector = TokenAnomaly()
        detector.record(input_tokens=100, output_tokens=100)
        result = detector.check_last()
        assert isinstance(result, AnomalyCheck)
        assert result.input_tokens == 100


# ---------------------------------------------------------------------------
# Clear and window
# ---------------------------------------------------------------------------

class TestClearAndWindow:
    def test_clear_resets_everything(self):
        detector = TokenAnomaly(spike_threshold=2.0)
        for _ in range(5):
            detector.record(input_tokens=100, output_tokens=100)
        detector.record(input_tokens=100, output_tokens=1000)
        detector.clear()
        assert detector.record_count == 0
        stats = detector.stats()
        assert stats.spike_count == 0

    def test_window_limits_baseline(self):
        detector = TokenAnomaly(spike_threshold=2.0, window_size=5)
        # Fill window with high values so the spike threshold is high
        for _ in range(10):
            detector.record(input_tokens=100, output_tokens=1000)
        # Now a value of 1000 should NOT be a spike (baseline is ~1000)
        result = detector.record(input_tokens=100, output_tokens=1000)
        assert result.is_anomalous is False
