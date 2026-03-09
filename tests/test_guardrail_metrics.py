"""Tests for guardrail metrics."""

import pytest
from sentinel.guardrail_metrics import GuardrailMetrics, CounterValue, GaugeValue, HistogramValue


class TestCounter:
    def test_increment(self):
        m = GuardrailMetrics()
        c = m.counter("scans_total")
        c.inc()
        c.inc()
        assert c.value == 2.0

    def test_increment_by(self):
        m = GuardrailMetrics()
        c = m.counter("tokens_total")
        c.inc(100)
        assert c.value == 100.0

    def test_same_counter(self):
        m = GuardrailMetrics()
        m.counter("x").inc()
        m.counter("x").inc()
        assert m.counter("x").value == 2.0

    def test_labeled_counter(self):
        m = GuardrailMetrics()
        m.counter("scans", labels={"scanner": "pii"}).inc()
        m.counter("scans", labels={"scanner": "injection"}).inc(3)
        assert m.counter("scans", labels={"scanner": "pii"}).value == 1
        assert m.counter("scans", labels={"scanner": "injection"}).value == 3


class TestGauge:
    def test_set(self):
        m = GuardrailMetrics()
        g = m.gauge("queue_size")
        g.set(42)
        assert g.value == 42

    def test_inc_dec(self):
        m = GuardrailMetrics()
        g = m.gauge("active")
        g.inc()
        g.inc()
        g.dec()
        assert g.value == 1.0


class TestHistogram:
    def test_observe(self):
        m = GuardrailMetrics()
        h = m.histogram("latency")
        h.observe(10)
        h.observe(20)
        h.observe(30)
        assert h.count == 3
        assert h.sum == 60
        assert h.mean == 20.0

    def test_percentile(self):
        m = GuardrailMetrics()
        h = m.histogram("latency")
        for i in range(100):
            h.observe(float(i))
        p50 = h.percentile(50)
        assert 45 <= p50 <= 55

    def test_empty_histogram(self):
        m = GuardrailMetrics()
        h = m.histogram("empty")
        assert h.count == 0
        assert h.mean == 0.0
        assert h.percentile(50) == 0.0


class TestExport:
    def test_text_export(self):
        m = GuardrailMetrics()
        m.counter("scans").inc(5)
        m.gauge("active").set(3)
        m.histogram("latency").observe(10)
        output = m.export()
        assert "scans" in output
        assert "active" in output
        assert "latency_count" in output

    def test_dict_export(self):
        m = GuardrailMetrics()
        m.counter("scans").inc()
        d = m.export_dict()
        assert "counters" in d
        assert "gauges" in d
        assert "histograms" in d

    def test_labeled_export(self):
        m = GuardrailMetrics()
        m.counter("scans", labels={"type": "pii"}).inc()
        output = m.export()
        assert 'type="pii"' in output


class TestReset:
    def test_reset(self):
        m = GuardrailMetrics()
        m.counter("a").inc()
        m.gauge("b").set(5)
        m.histogram("c").observe(10)
        m.reset()
        assert m.export() == ""


class TestStructure:
    def test_counter_type(self):
        m = GuardrailMetrics()
        c = m.counter("test")
        assert isinstance(c, CounterValue)

    def test_gauge_type(self):
        m = GuardrailMetrics()
        g = m.gauge("test")
        assert isinstance(g, GaugeValue)

    def test_histogram_type(self):
        m = GuardrailMetrics()
        h = m.histogram("test")
        assert isinstance(h, HistogramValue)
