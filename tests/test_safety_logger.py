"""Tests for SafetyLogger — structured safety event logging."""

from __future__ import annotations

import json
import time

import pytest

from sentinel.safety_logger import (
    SafetyLogEntry,
    SafetyLogger,
    LogFilter,
    LogSummary,
    LoggerStats,
    SEVERITY_LEVELS,
)


class TestLogEntryCreation:
    def test_log_returns_entry_with_correct_fields(self):
        logger = SafetyLogger()
        entry = logger.log("info", "scan", "Input scanned", metadata={"score": 0.9}, source="pii")

        assert entry.severity == "info"
        assert entry.event_type == "scan"
        assert entry.message == "Input scanned"
        assert entry.metadata == {"score": 0.9}
        assert entry.source == "pii"
        assert isinstance(entry.timestamp, float)

    def test_log_defaults_metadata_to_empty_dict(self):
        logger = SafetyLogger()
        entry = logger.log("debug", "test", "no metadata")

        assert entry.metadata == {}

    def test_log_defaults_source_to_empty_string(self):
        logger = SafetyLogger()
        entry = logger.log("debug", "test", "no source")

        assert entry.source == ""

    def test_entry_to_dict(self):
        entry = SafetyLogEntry(
            timestamp=1000.0,
            severity="warning",
            event_type="block",
            message="Blocked",
        )
        result = entry.to_dict()

        assert result["severity"] == "warning"
        assert result["event_type"] == "block"
        assert result["message"] == "Blocked"
        assert result["timestamp"] == 1000.0


class TestSeverityConvenienceMethods:
    def test_debug(self):
        logger = SafetyLogger()
        entry = logger.debug("test", "debug message")
        assert entry.severity == "debug"

    def test_info(self):
        logger = SafetyLogger()
        entry = logger.info("test", "info message")
        assert entry.severity == "info"

    def test_warning(self):
        logger = SafetyLogger()
        entry = logger.warning("test", "warning message")
        assert entry.severity == "warning"

    def test_error(self):
        logger = SafetyLogger()
        entry = logger.error("test", "error message")
        assert entry.severity == "error"

    def test_critical(self):
        logger = SafetyLogger()
        entry = logger.critical("test", "critical message")
        assert entry.severity == "critical"

    def test_convenience_methods_pass_kwargs(self):
        logger = SafetyLogger()
        entry = logger.warning("scan", "flagged", metadata={"risk": 0.8}, source="toxicity")

        assert entry.metadata == {"risk": 0.8}
        assert entry.source == "toxicity"


class TestInvalidSeverity:
    def test_invalid_severity_raises_value_error(self):
        logger = SafetyLogger()
        with pytest.raises(ValueError, match="Invalid severity"):
            logger.log("fatal", "test", "bad severity")

    def test_invalid_severity_message_lists_valid_options(self):
        logger = SafetyLogger()
        with pytest.raises(ValueError, match="debug"):
            logger.log("unknown", "test", "bad")


class TestQueryNoFilter:
    def test_query_returns_all_entries_when_no_filter(self):
        logger = SafetyLogger()
        logger.info("a", "first")
        logger.warning("b", "second")
        logger.error("c", "third")

        results = logger.query()
        assert len(results) == 3

    def test_query_returns_entries_in_insertion_order(self):
        logger = SafetyLogger()
        logger.info("a", "first")
        logger.warning("b", "second")

        results = logger.query()
        assert results[0].message == "first"
        assert results[1].message == "second"


class TestQueryBySeverity:
    def test_query_filters_by_minimum_severity(self):
        logger = SafetyLogger()
        logger.debug("d", "too low")
        logger.info("i", "too low")
        logger.warning("w", "included")
        logger.error("e", "included")
        logger.critical("c", "included")

        results = logger.query(LogFilter(min_severity="warning"))
        assert len(results) == 3
        severities = {e.severity for e in results}
        assert severities == {"warning", "error", "critical"}

    def test_query_severity_critical_returns_only_critical(self):
        logger = SafetyLogger()
        logger.error("e", "not included")
        logger.critical("c", "included")

        results = logger.query(LogFilter(min_severity="critical"))
        assert len(results) == 1
        assert results[0].severity == "critical"


class TestQueryByEventType:
    def test_query_filters_by_event_type(self):
        logger = SafetyLogger()
        logger.info("scan", "scanned input")
        logger.info("block", "blocked input")
        logger.warning("scan", "scanned again")

        results = logger.query(LogFilter(event_type="scan"))
        assert len(results) == 2
        assert all(e.event_type == "scan" for e in results)


class TestQueryBySource:
    def test_query_filters_by_source(self):
        logger = SafetyLogger()
        logger.info("scan", "from pii", source="pii_scanner")
        logger.info("scan", "from toxicity", source="toxicity_scanner")
        logger.info("scan", "from pii again", source="pii_scanner")

        results = logger.query(LogFilter(source="pii_scanner"))
        assert len(results) == 2
        assert all(e.source == "pii_scanner" for e in results)


class TestQueryByTimeRange:
    def test_query_filters_by_since_timestamp(self):
        logger = SafetyLogger()
        old_entry = logger.info("old", "old event")
        cutoff = time.time()
        new_entry = logger.info("new", "new event")

        results = logger.query(LogFilter(since=cutoff))
        assert len(results) == 1
        assert results[0].message == "new event"


class TestMultipleFiltersCombined:
    def test_query_combines_severity_and_event_type(self):
        logger = SafetyLogger()
        logger.debug("scan", "low severity scan")
        logger.error("scan", "high severity scan")
        logger.error("block", "high severity block")

        results = logger.query(LogFilter(min_severity="error", event_type="scan"))
        assert len(results) == 1
        assert results[0].message == "high severity scan"

    def test_query_combines_all_filters(self):
        logger = SafetyLogger()
        cutoff = time.time()
        logger.critical("breach", "match", source="firewall")
        logger.critical("breach", "wrong source", source="other")
        logger.info("breach", "too low", source="firewall")

        results = logger.query(LogFilter(
            min_severity="critical",
            event_type="breach",
            source="firewall",
            since=cutoff,
        ))
        assert len(results) == 1
        assert results[0].message == "match"


class TestEmptyLogQuery:
    def test_query_on_empty_log_returns_empty_list(self):
        logger = SafetyLogger()
        assert logger.query() == []

    def test_query_with_filter_on_empty_log_returns_empty_list(self):
        logger = SafetyLogger()
        results = logger.query(LogFilter(min_severity="error"))
        assert results == []


class TestSummary:
    def test_summary_counts_by_severity(self):
        logger = SafetyLogger()
        logger.info("a", "one")
        logger.info("b", "two")
        logger.error("c", "three")

        summary = logger.summary()
        assert summary.total_entries == 3
        assert summary.by_severity == {"info": 2, "error": 1}

    def test_summary_counts_by_event_type(self):
        logger = SafetyLogger()
        logger.info("scan", "one")
        logger.info("scan", "two")
        logger.warning("block", "three")

        summary = logger.summary()
        assert summary.by_event_type == {"scan": 2, "block": 1}

    def test_summary_tracks_latest_critical(self):
        logger = SafetyLogger()
        logger.critical("breach", "first critical")
        logger.info("scan", "not critical")
        logger.critical("breach", "second critical")

        summary = logger.summary()
        assert summary.latest_critical is not None
        assert summary.latest_critical.message == "second critical"

    def test_summary_latest_critical_is_none_when_no_criticals(self):
        logger = SafetyLogger()
        logger.info("scan", "just info")

        summary = logger.summary()
        assert summary.latest_critical is None

    def test_summary_on_empty_log(self):
        logger = SafetyLogger()
        summary = logger.summary()

        assert summary.total_entries == 0
        assert summary.by_severity == {}
        assert summary.by_event_type == {}
        assert summary.latest_critical is None


class TestExport:
    def test_export_as_dict_returns_list_of_dicts(self):
        logger = SafetyLogger()
        logger.info("scan", "test entry", source="scanner")

        exported = logger.export(format="dict")
        assert isinstance(exported, list)
        assert len(exported) == 1
        assert exported[0]["severity"] == "info"
        assert exported[0]["event_type"] == "scan"
        assert exported[0]["source"] == "scanner"

    def test_export_as_json_returns_valid_json_string(self):
        logger = SafetyLogger()
        logger.warning("block", "blocked something")

        exported = logger.export(format="json")
        assert isinstance(exported, str)
        parsed = json.loads(exported)
        assert len(parsed) == 1
        assert parsed[0]["severity"] == "warning"

    def test_export_default_format_is_dict(self):
        logger = SafetyLogger()
        logger.info("test", "default format")

        exported = logger.export()
        assert isinstance(exported, list)


class TestFIFOEviction:
    def test_evicts_oldest_when_max_entries_exceeded(self):
        logger = SafetyLogger(max_entries=3)
        logger.info("a", "first")
        logger.info("b", "second")
        logger.info("c", "third")
        logger.info("d", "fourth")

        entries = logger.query()
        assert len(entries) == 3
        messages = [e.message for e in entries]
        assert "first" not in messages
        assert messages == ["second", "third", "fourth"]

    def test_eviction_preserves_newest_entries(self):
        logger = SafetyLogger(max_entries=2)
        for i in range(10):
            logger.info("test", f"entry_{i}")

        entries = logger.query()
        assert len(entries) == 2
        assert entries[0].message == "entry_8"
        assert entries[1].message == "entry_9"


class TestClear:
    def test_clear_removes_all_entries(self):
        logger = SafetyLogger()
        logger.info("a", "one")
        logger.error("b", "two")

        logger.clear()
        assert logger.query() == []

    def test_clear_resets_summary(self):
        logger = SafetyLogger()
        logger.info("scan", "one")
        logger.clear()

        summary = logger.summary()
        assert summary.total_entries == 0


class TestStatsSurviveClear:
    def test_stats_track_total_logged(self):
        logger = SafetyLogger()
        logger.info("a", "one")
        logger.error("b", "two")

        stats = logger.stats()
        assert stats.total_logged == 2

    def test_stats_track_by_severity(self):
        logger = SafetyLogger()
        logger.info("a", "one")
        logger.info("b", "two")
        logger.error("c", "three")

        stats = logger.stats()
        assert stats.by_severity == {"info": 2, "error": 1}

    def test_stats_persist_after_clear(self):
        logger = SafetyLogger()
        logger.info("a", "one")
        logger.error("b", "two")
        logger.clear()
        logger.warning("c", "three")

        stats = logger.stats()
        assert stats.total_logged == 3
        assert stats.by_severity == {"info": 1, "error": 1, "warning": 1}

    def test_stats_persist_after_eviction(self):
        logger = SafetyLogger(max_entries=2)
        logger.info("a", "one")
        logger.info("b", "two")
        logger.info("c", "three")

        stats = logger.stats()
        assert stats.total_logged == 3
        assert stats.by_severity == {"info": 3}

    def test_stats_returns_copy_not_reference(self):
        logger = SafetyLogger()
        logger.info("a", "one")

        stats1 = logger.stats()
        logger.error("b", "two")
        stats2 = logger.stats()

        assert stats1.total_logged == 1
        assert stats2.total_logged == 2


class TestSeverityOrdering:
    def test_severity_levels_are_ordered(self):
        expected = ("debug", "info", "warning", "error", "critical")
        assert SEVERITY_LEVELS == expected
