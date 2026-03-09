"""Structured safety event logging with severity levels, filtering, and export.

Purpose-built logger for AI safety events with query capabilities,
FIFO eviction, cumulative stats, and multiple export formats.

Usage:
    from sentinel.safety_logger import SafetyLogger, LogFilter

    logger = SafetyLogger(max_entries=5000)
    logger.info("guardrail_triggered", "Blocked prompt injection attempt")
    logger.critical("security_breach", "Credential leak detected", source="pii_scanner")

    results = logger.query(LogFilter(min_severity="warning"))
    summary = logger.summary()
    exported = logger.export(format="json")
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any


SEVERITY_LEVELS = ("debug", "info", "warning", "error", "critical")
SEVERITY_ORDER = {level: index for index, level in enumerate(SEVERITY_LEVELS)}


@dataclass
class SafetyLogEntry:
    """A single safety log entry."""
    timestamp: float
    severity: str
    event_type: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LogFilter:
    """Filter criteria for querying log entries."""
    min_severity: str | None = None
    event_type: str | None = None
    source: str | None = None
    since: float | None = None


@dataclass
class LogSummary:
    """Summary statistics for current log entries."""
    total_entries: int
    by_severity: dict[str, int]
    by_event_type: dict[str, int]
    latest_critical: SafetyLogEntry | None = None


@dataclass
class LoggerStats:
    """Cumulative stats that persist across clears."""
    total_logged: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)


def _validate_severity(severity: str) -> None:
    if severity not in SEVERITY_ORDER:
        valid = ", ".join(SEVERITY_LEVELS)
        raise ValueError(f"Invalid severity '{severity}'. Must be one of: {valid}")


def _meets_severity_threshold(entry_severity: str, min_severity: str) -> bool:
    return SEVERITY_ORDER[entry_severity] >= SEVERITY_ORDER[min_severity]


def _matches_filter(entry: SafetyLogEntry, log_filter: LogFilter) -> bool:
    if log_filter.min_severity is not None:
        if not _meets_severity_threshold(entry.severity, log_filter.min_severity):
            return False
    if log_filter.event_type is not None:
        if entry.event_type != log_filter.event_type:
            return False
    if log_filter.source is not None:
        if entry.source != log_filter.source:
            return False
    if log_filter.since is not None:
        if entry.timestamp < log_filter.since:
            return False
    return True


class SafetyLogger:
    """Structured safety event logger with filtering, search, and export."""

    def __init__(self, max_entries: int = 10000) -> None:
        self._max_entries = max_entries
        self._entries: deque[SafetyLogEntry] = deque(maxlen=max_entries)
        self._stats = LoggerStats()

    def log(
        self,
        severity: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        source: str = "",
    ) -> SafetyLogEntry:
        _validate_severity(severity)
        entry = SafetyLogEntry(
            timestamp=time.time(),
            severity=severity,
            event_type=event_type,
            message=message,
            metadata=metadata or {},
            source=source,
        )
        self._entries.append(entry)
        self._update_stats(severity)
        return entry

    def debug(self, event_type: str, message: str, **kwargs: Any) -> SafetyLogEntry:
        return self.log("debug", event_type, message, **kwargs)

    def info(self, event_type: str, message: str, **kwargs: Any) -> SafetyLogEntry:
        return self.log("info", event_type, message, **kwargs)

    def warning(self, event_type: str, message: str, **kwargs: Any) -> SafetyLogEntry:
        return self.log("warning", event_type, message, **kwargs)

    def error(self, event_type: str, message: str, **kwargs: Any) -> SafetyLogEntry:
        return self.log("error", event_type, message, **kwargs)

    def critical(self, event_type: str, message: str, **kwargs: Any) -> SafetyLogEntry:
        return self.log("critical", event_type, message, **kwargs)

    def query(self, filter: LogFilter | None = None) -> list[SafetyLogEntry]:
        if filter is None:
            return list(self._entries)
        return [entry for entry in self._entries if _matches_filter(entry, filter)]

    def summary(self) -> LogSummary:
        by_severity: dict[str, int] = {}
        by_event_type: dict[str, int] = {}
        latest_critical: SafetyLogEntry | None = None

        for entry in self._entries:
            by_severity[entry.severity] = by_severity.get(entry.severity, 0) + 1
            by_event_type[entry.event_type] = by_event_type.get(entry.event_type, 0) + 1
            if entry.severity == "critical":
                latest_critical = entry

        return LogSummary(
            total_entries=len(self._entries),
            by_severity=by_severity,
            by_event_type=by_event_type,
            latest_critical=latest_critical,
        )

    def export(self, format: str = "dict") -> list[dict[str, Any]] | str:
        entries_as_dicts = [entry.to_dict() for entry in self._entries]
        if format == "json":
            return json.dumps(entries_as_dicts, indent=2)
        return entries_as_dicts

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> LoggerStats:
        return LoggerStats(
            total_logged=self._stats.total_logged,
            by_severity=dict(self._stats.by_severity),
        )

    def _update_stats(self, severity: str) -> None:
        self._stats.total_logged += 1
        self._stats.by_severity[severity] = self._stats.by_severity.get(severity, 0) + 1
