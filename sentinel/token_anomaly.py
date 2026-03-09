"""Token usage anomaly detection for LLM applications.

Detects anomalous token usage patterns that may indicate prompt injection,
data exfiltration, or resource abuse. Tracks input/output token counts,
detects spikes against rolling averages, and flags suspicious
input/output ratios matching known attack signatures.

Usage:
    from sentinel.token_anomaly import TokenAnomaly

    detector = TokenAnomaly(spike_threshold=3.0, window_size=50)
    detector.record(input_tokens=10, output_tokens=5000)
    result = detector.check_last()
    if result.is_anomalous:
        print(result.reason)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field


@dataclass
class TokenRecord:
    """A single token usage record."""
    input_tokens: int
    output_tokens: int
    timestamp: float
    label: str


@dataclass
class AnomalyCheck:
    """Result of an anomaly check on a single record."""
    is_anomalous: bool
    reason: str
    input_tokens: int
    output_tokens: int
    ratio: float
    avg_baseline: float


@dataclass
class TokenAnomalyStats:
    """Aggregate statistics across all tracked records."""
    total_records: int
    spike_count: int
    avg_input: float
    avg_output: float
    max_ratio: float


_EXFILTRATION_RATIO_THRESHOLD = 20.0
_EXFILTRATION_MIN_OUTPUT = 500


class TokenAnomaly:
    """Detect anomalous token usage patterns in LLM requests.

    Tracks per-request input/output token counts and flags:
    - Output token spikes (current vs rolling average)
    - Suspicious input/output ratios (e.g., data exfiltration)
    - Known attack signatures
    """

    def __init__(
        self,
        spike_threshold: float = 3.0,
        window_size: int = 50,
    ) -> None:
        """
        Args:
            spike_threshold: Multiplier over rolling average to flag a spike.
            window_size: Number of recent records used for baseline calculation.
        """
        self._spike_threshold = spike_threshold
        self._window_size = window_size
        self._records: list[TokenRecord] = []
        self._spike_count: int = 0
        self._lock = threading.Lock()

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        label: str = "",
        timestamp: float | None = None,
    ) -> AnomalyCheck:
        """Record a request's token usage and check for anomalies.

        Args:
            input_tokens: Number of input tokens in the request.
            output_tokens: Number of output tokens in the response.
            label: Optional label for this record (e.g., user ID).
            timestamp: Optional timestamp; defaults to current time.

        Returns:
            AnomalyCheck with detection results.
        """
        ts = timestamp if timestamp is not None else time.time()
        token_record = TokenRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=ts,
            label=label,
        )

        with self._lock:
            baseline_records = self._get_baseline_records()
            avg_baseline = self._compute_average_output(baseline_records)
            ratio = self._compute_ratio(input_tokens, output_tokens)

            self._records.append(token_record)

            is_spike = self._is_spike(output_tokens, avg_baseline)
            is_exfiltration = self._is_exfiltration_pattern(input_tokens, output_tokens, ratio)

            is_anomalous = is_spike or is_exfiltration
            reason = self._build_reason(is_spike, is_exfiltration, output_tokens, avg_baseline, ratio)

            if is_anomalous:
                self._spike_count += 1

        return AnomalyCheck(
            is_anomalous=is_anomalous,
            reason=reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ratio=ratio,
            avg_baseline=avg_baseline,
        )

    def check_last(self) -> AnomalyCheck:
        """Re-check the most recent record against the current baseline.

        Returns:
            AnomalyCheck for the last recorded request.

        Raises:
            ValueError: If no records have been recorded yet.
        """
        with self._lock:
            if not self._records:
                raise ValueError("No records to check")

            last = self._records[-1]
            baseline_records = self._get_baseline_records(exclude_last=True)
            avg_baseline = self._compute_average_output(baseline_records)
            ratio = self._compute_ratio(last.input_tokens, last.output_tokens)

            is_spike = self._is_spike(last.output_tokens, avg_baseline)
            is_exfiltration = self._is_exfiltration_pattern(
                last.input_tokens, last.output_tokens, ratio,
            )

            is_anomalous = is_spike or is_exfiltration
            reason = self._build_reason(
                is_spike, is_exfiltration, last.output_tokens, avg_baseline, ratio,
            )

        return AnomalyCheck(
            is_anomalous=is_anomalous,
            reason=reason,
            input_tokens=last.input_tokens,
            output_tokens=last.output_tokens,
            ratio=ratio,
            avg_baseline=avg_baseline,
        )

    def stats(self) -> TokenAnomalyStats:
        """Compute aggregate statistics across all tracked records.

        Returns:
            TokenAnomalyStats with totals and averages.
        """
        with self._lock:
            total = len(self._records)
            if total == 0:
                return TokenAnomalyStats(
                    total_records=0,
                    spike_count=0,
                    avg_input=0.0,
                    avg_output=0.0,
                    max_ratio=0.0,
                )

            avg_input = sum(r.input_tokens for r in self._records) / total
            avg_output = sum(r.output_tokens for r in self._records) / total
            max_ratio = max(
                self._compute_ratio(r.input_tokens, r.output_tokens)
                for r in self._records
            )

            return TokenAnomalyStats(
                total_records=total,
                spike_count=self._spike_count,
                avg_input=avg_input,
                avg_output=avg_output,
                max_ratio=max_ratio,
            )

    @property
    def record_count(self) -> int:
        """Number of records currently tracked."""
        return len(self._records)

    def clear(self) -> None:
        """Clear all tracked records and reset spike count."""
        with self._lock:
            self._records.clear()
            self._spike_count = 0

    def _get_baseline_records(self, exclude_last: bool = False) -> list[TokenRecord]:
        records = self._records
        if exclude_last and len(records) > 0:
            records = records[:-1]
        return records[-self._window_size:]

    @staticmethod
    def _compute_average_output(records: list[TokenRecord]) -> float:
        if not records:
            return 0.0
        return sum(r.output_tokens for r in records) / len(records)

    @staticmethod
    def _compute_ratio(input_tokens: int, output_tokens: int) -> float:
        if input_tokens == 0:
            return float(output_tokens) if output_tokens > 0 else 0.0
        return output_tokens / input_tokens

    def _is_spike(self, output_tokens: int, avg_baseline: float) -> bool:
        if avg_baseline <= 0:
            return False
        return output_tokens > avg_baseline * self._spike_threshold

    @staticmethod
    def _is_exfiltration_pattern(
        input_tokens: int, output_tokens: int, ratio: float,
    ) -> bool:
        return (
            ratio >= _EXFILTRATION_RATIO_THRESHOLD
            and output_tokens >= _EXFILTRATION_MIN_OUTPUT
        )

    @staticmethod
    def _build_reason(
        is_spike: bool,
        is_exfiltration: bool,
        output_tokens: int,
        avg_baseline: float,
        ratio: float,
    ) -> str:
        reasons: list[str] = []
        if is_spike:
            multiplier = output_tokens / avg_baseline if avg_baseline > 0 else 0
            reasons.append(
                f"Output spike: {output_tokens} tokens is {multiplier:.1f}x the rolling average of {avg_baseline:.0f}"
            )
        if is_exfiltration:
            reasons.append(
                f"Exfiltration pattern: input/output ratio {ratio:.1f} with {output_tokens} output tokens"
            )
        return "; ".join(reasons)
