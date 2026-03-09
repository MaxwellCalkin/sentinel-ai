"""Tests for prompt usage analytics and safety monitoring."""

import threading

from sentinel.prompt_analytics import PromptAnalytics, PromptRecord, AnalyticsSummary


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_single(self):
        analytics = PromptAnalytics()
        record = analytics.record("Hello world", tokens=10)

        assert isinstance(record, PromptRecord)
        assert record.tokens == 10
        assert record.flagged is False
        assert len(record.prompt_hash) == 12

    def test_record_multiple(self):
        analytics = PromptAnalytics()
        analytics.record("First prompt", tokens=5)
        analytics.record("Second prompt", tokens=15)
        analytics.record("Third prompt", tokens=10)

        summary = analytics.summary()
        assert summary.total_prompts == 3

    def test_record_with_labels(self):
        analytics = PromptAnalytics()
        record = analytics.record(
            "Translate this",
            tokens=20,
            labels={"category": "translation", "language": "en"},
        )

        assert record.labels["category"] == "translation"
        assert record.labels["language"] == "en"

    def test_record_stores_hash_not_raw_prompt(self):
        analytics = PromptAnalytics()
        record = analytics.record("Sensitive user data here")

        assert "Sensitive" not in record.prompt_hash
        assert len(record.prompt_hash) == 12

    def test_record_has_timestamp(self):
        analytics = PromptAnalytics()
        record = analytics.record("Test prompt")

        assert record.timestamp is not None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_structure(self):
        analytics = PromptAnalytics()
        analytics.record("Test", tokens=10)

        summary = analytics.summary()
        assert isinstance(summary, AnalyticsSummary)
        assert hasattr(summary, "total_prompts")
        assert hasattr(summary, "total_tokens")
        assert hasattr(summary, "avg_tokens")
        assert hasattr(summary, "flag_rate")
        assert hasattr(summary, "top_labels")
        assert hasattr(summary, "unique_prompts")

    def test_summary_counts(self):
        analytics = PromptAnalytics()
        analytics.record("Prompt A", tokens=10)
        analytics.record("Prompt B", tokens=20, flagged=True)

        summary = analytics.summary()
        assert summary.total_prompts == 2
        assert summary.total_tokens == 30
        assert summary.avg_tokens == 15.0
        assert summary.flag_rate == 0.5

    def test_unique_prompts(self):
        analytics = PromptAnalytics()
        analytics.record("Same prompt", tokens=5)
        analytics.record("Same prompt", tokens=10)
        analytics.record("Different prompt", tokens=15)

        summary = analytics.summary()
        assert summary.total_prompts == 3
        assert summary.unique_prompts == 2

    def test_summary_empty(self):
        analytics = PromptAnalytics()
        summary = analytics.summary()

        assert summary.total_prompts == 0
        assert summary.total_tokens == 0
        assert summary.avg_tokens == 0.0
        assert summary.flag_rate == 0.0
        assert summary.unique_prompts == 0


# ---------------------------------------------------------------------------
# Flag rate
# ---------------------------------------------------------------------------

class TestFlagRate:
    def test_no_flags(self):
        analytics = PromptAnalytics()
        analytics.record("Safe prompt 1", tokens=5)
        analytics.record("Safe prompt 2", tokens=10)

        assert analytics.flag_rate() == 0.0

    def test_all_flagged(self):
        analytics = PromptAnalytics()
        analytics.record("Bad prompt 1", tokens=5, flagged=True)
        analytics.record("Bad prompt 2", tokens=10, flagged=True)

        assert analytics.flag_rate() == 1.0

    def test_mixed_flags(self):
        analytics = PromptAnalytics()
        analytics.record("Safe", tokens=5, flagged=False)
        analytics.record("Unsafe", tokens=10, flagged=True)
        analytics.record("Safe again", tokens=5, flagged=False)
        analytics.record("Unsafe again", tokens=10, flagged=True)

        assert analytics.flag_rate() == 0.5

    def test_flag_rate_empty(self):
        analytics = PromptAnalytics()
        assert analytics.flag_rate() == 0.0


# ---------------------------------------------------------------------------
# Token averages
# ---------------------------------------------------------------------------

class TestTokens:
    def test_avg_tokens(self):
        analytics = PromptAnalytics()
        analytics.record("Short", tokens=10)
        analytics.record("Medium", tokens=30)
        analytics.record("Long", tokens=50)

        assert analytics.avg_tokens() == 30.0

    def test_zero_tokens(self):
        analytics = PromptAnalytics()
        analytics.record("No tokens counted")
        analytics.record("Also no tokens")

        assert analytics.avg_tokens() == 0.0

    def test_avg_tokens_empty(self):
        analytics = PromptAnalytics()
        assert analytics.avg_tokens() == 0.0


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

class TestTrend:
    def test_trend_increasing(self):
        analytics = PromptAnalytics()
        # First half: no flags
        for i in range(5):
            analytics.record(f"safe-{i}", tokens=10, flagged=False)
        # Second half: all flagged
        for i in range(5):
            analytics.record(f"unsafe-{i}", tokens=10, flagged=True)

        assert analytics.trend(window=10) == "increasing"

    def test_trend_decreasing(self):
        analytics = PromptAnalytics()
        # First half: all flagged
        for i in range(5):
            analytics.record(f"unsafe-{i}", tokens=10, flagged=True)
        # Second half: no flags
        for i in range(5):
            analytics.record(f"safe-{i}", tokens=10, flagged=False)

        assert analytics.trend(window=10) == "decreasing"

    def test_trend_stable(self):
        analytics = PromptAnalytics()
        # Equal flag rates in both halves (1 flagged per half of 5)
        flagged_pattern = [True, False, False, False, False,
                           True, False, False, False, False]
        for i, is_flagged in enumerate(flagged_pattern):
            analytics.record(f"prompt-{i}", tokens=10, flagged=is_flagged)

        assert analytics.trend(window=10) == "stable"

    def test_trend_with_few_records(self):
        analytics = PromptAnalytics()
        analytics.record("only one", tokens=5)

        assert analytics.trend(window=10) == "stable"

    def test_trend_empty(self):
        analytics = PromptAnalytics()
        assert analytics.trend() == "stable"


# ---------------------------------------------------------------------------
# Label tracking
# ---------------------------------------------------------------------------

class TestLabels:
    def test_top_labels(self):
        analytics = PromptAnalytics()
        analytics.record("P1", labels={"category": "chat", "model": "sonnet"})
        analytics.record("P2", labels={"category": "code"})
        analytics.record("P3", labels={"category": "chat"})

        top = analytics.top_labels(n=5)
        # "category" appears 3 times, "model" once
        assert top[0] == ("category", 3)
        assert ("model", 1) in top

    def test_top_labels_limit(self):
        analytics = PromptAnalytics()
        analytics.record("P1", labels={"a": "1", "b": "2", "c": "3"})
        analytics.record("P2", labels={"a": "1", "d": "4"})
        analytics.record("P3", labels={"a": "1", "b": "2", "e": "5"})

        top = analytics.top_labels(n=2)
        assert len(top) == 2
        # "a" (3 times) and "b" (2 times) should be top 2
        assert top[0] == ("a", 3)
        assert top[1] == ("b", 2)

    def test_top_labels_empty(self):
        analytics = PromptAnalytics()
        assert analytics.top_labels() == []


# ---------------------------------------------------------------------------
# Clear / reset
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets(self):
        analytics = PromptAnalytics()
        analytics.record("Prompt", tokens=50, flagged=True, labels={"k": "v"})
        analytics.clear()

        summary = analytics.summary()
        assert summary.total_prompts == 0
        assert summary.total_tokens == 0
        assert summary.unique_prompts == 0
        assert analytics.flag_rate() == 0.0
        assert analytics.avg_tokens() == 0.0
        assert analytics.top_labels() == []

    def test_clear_allows_fresh_recording(self):
        analytics = PromptAnalytics()
        analytics.record("Before clear", tokens=100, flagged=True)
        analytics.clear()
        analytics.record("After clear", tokens=25, flagged=False)

        assert analytics.summary().total_prompts == 1
        assert analytics.avg_tokens() == 25.0
        assert analytics.flag_rate() == 0.0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_recording(self):
        analytics = PromptAnalytics()
        num_threads = 10
        records_per_thread = 100

        def record_batch(thread_id: int) -> None:
            for i in range(records_per_thread):
                analytics.record(
                    f"thread-{thread_id}-prompt-{i}",
                    tokens=1,
                    flagged=(i % 2 == 0),
                )

        threads = [
            threading.Thread(target=record_batch, args=(t,))
            for t in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        expected_total = num_threads * records_per_thread
        assert analytics.summary().total_prompts == expected_total
