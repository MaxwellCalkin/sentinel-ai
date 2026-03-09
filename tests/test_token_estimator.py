"""Tests for TokenEstimator — heuristic token estimation across model families."""

import pytest
from sentinel.token_estimator import (
    TokenEstimator,
    TokenEstimate,
    CostEstimate,
    EstimatorStats,
)


# ---------------------------------------------------------------------------
# Simple English text estimation
# ---------------------------------------------------------------------------

class TestSimpleEnglishEstimation:
    def test_short_sentence_returns_positive_count(self):
        estimator = TokenEstimator()
        result = estimator.estimate("Hello, world!")
        assert result.token_count > 0

    def test_estimate_returns_token_estimate_dataclass(self):
        estimator = TokenEstimator()
        result = estimator.estimate("The quick brown fox")
        assert isinstance(result, TokenEstimate)

    def test_word_count_matches_whitespace_split(self):
        estimator = TokenEstimator()
        result = estimator.estimate("one two three four five")
        assert result.word_count == 5

    def test_char_count_matches_length(self):
        text = "Hello, world!"
        estimator = TokenEstimator()
        result = estimator.estimate(text)
        assert result.char_count == len(text)

    def test_text_is_preserved_in_result(self):
        text = "Some input text"
        estimator = TokenEstimator()
        result = estimator.estimate(text)
        assert result.text == text

    def test_ratio_matches_model_family(self):
        estimator = TokenEstimator()
        result = estimator.estimate("hello", model_family="claude")
        assert result.ratio == 1.35


# ---------------------------------------------------------------------------
# CJK text estimation (higher count)
# ---------------------------------------------------------------------------

class TestCjkEstimation:
    def test_cjk_text_has_higher_token_count_than_same_word_count_english(self):
        estimator = TokenEstimator()
        english = estimator.estimate("hello world")
        chinese = estimator.estimate("你好世界")
        # CJK chars add ~2 tokens each, so 4 CJK chars = 8 bonus tokens
        assert chinese.token_count > english.token_count

    def test_japanese_text_counts_cjk_bonus(self):
        estimator = TokenEstimator()
        result = estimator.estimate("東京タワー")
        assert result.token_count >= 4

    def test_mixed_english_and_cjk(self):
        estimator = TokenEstimator()
        result = estimator.estimate("Hello 你好 World 世界")
        assert result.token_count > 4


# ---------------------------------------------------------------------------
# Cost estimation for various models
# ---------------------------------------------------------------------------

class TestCostEstimation:
    def test_gpt4_cost_structure(self):
        estimator = TokenEstimator()
        cost = estimator.estimate_cost("Hello world", model="gpt-4")
        assert isinstance(cost, CostEstimate)
        assert cost.model == "gpt-4"
        assert cost.input_cost > 0
        assert cost.total_cost == cost.input_cost + cost.output_cost

    def test_claude_sonnet_cost(self):
        estimator = TokenEstimator()
        cost = estimator.estimate_cost(
            "Hello world",
            model="claude-sonnet",
            estimated_output_tokens=100,
        )
        assert cost.output_tokens == 100
        assert cost.output_cost > 0
        assert cost.total_cost > cost.input_cost

    def test_output_tokens_increase_total_cost(self):
        estimator = TokenEstimator()
        no_output = estimator.estimate_cost("test", model="gpt-4o", estimated_output_tokens=0)
        with_output = estimator.estimate_cost("test", model="gpt-4o", estimated_output_tokens=1000)
        assert with_output.total_cost > no_output.total_cost

    def test_unknown_pricing_model_returns_zero_cost(self):
        estimator = TokenEstimator()
        cost = estimator.estimate_cost("test", model="unknown-model")
        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.total_cost == 0.0

    def test_claude_haiku_cheaper_than_claude_opus(self):
        estimator = TokenEstimator()
        text = "This is a test sentence for cost comparison."
        haiku = estimator.estimate_cost(text, model="claude-haiku", estimated_output_tokens=500)
        opus = estimator.estimate_cost(text, model="claude-opus", estimated_output_tokens=500)
        assert haiku.total_cost < opus.total_cost


# ---------------------------------------------------------------------------
# Batch estimation
# ---------------------------------------------------------------------------

class TestBatchEstimation:
    def test_batch_returns_correct_count(self):
        estimator = TokenEstimator()
        texts = ["Hello", "World", "Foo bar baz"]
        results = estimator.estimate_batch(texts)
        assert len(results) == 3

    def test_batch_results_match_individual(self):
        estimator = TokenEstimator()
        texts = ["alpha", "beta gamma"]
        batch = estimator.estimate_batch(texts, model_family="claude")
        individual = [estimator.estimate(t, model_family="claude") for t in texts]
        for batch_item, indiv_item in zip(batch, individual):
            assert batch_item.token_count == indiv_item.token_count

    def test_empty_batch(self):
        estimator = TokenEstimator()
        results = estimator.estimate_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Fits context check
# ---------------------------------------------------------------------------

class TestFitsContext:
    def test_short_text_fits_large_context(self):
        estimator = TokenEstimator()
        assert estimator.fits_context("Hello", max_tokens=1000) is True

    def test_long_text_does_not_fit_small_context(self):
        estimator = TokenEstimator()
        long_text = "word " * 500
        assert estimator.fits_context(long_text, max_tokens=10) is False

    def test_boundary_fit(self):
        estimator = TokenEstimator()
        result = estimator.estimate("one two three")
        exact_budget = result.token_count
        assert estimator.fits_context("one two three", max_tokens=exact_budget) is True
        assert estimator.fits_context("one two three", max_tokens=exact_budget - 1) is False


# ---------------------------------------------------------------------------
# Truncate to fit
# ---------------------------------------------------------------------------

class TestTruncateToFit:
    def test_no_truncation_when_text_fits(self):
        estimator = TokenEstimator()
        text = "Short text"
        result = estimator.truncate_to_fit(text, max_tokens=100)
        assert result == text

    def test_truncation_reduces_token_count(self):
        estimator = TokenEstimator()
        long_text = " ".join(f"word{i}" for i in range(200))
        truncated = estimator.truncate_to_fit(long_text, max_tokens=10)
        estimate = estimator.estimate(truncated)
        assert estimate.token_count <= 10
        assert len(truncated) < len(long_text)

    def test_truncation_returns_empty_when_budget_is_zero(self):
        estimator = TokenEstimator()
        result = estimator.truncate_to_fit("some text here", max_tokens=0)
        assert result == ""


# ---------------------------------------------------------------------------
# Empty text
# ---------------------------------------------------------------------------

class TestEmptyText:
    def test_empty_string_returns_zero_tokens(self):
        estimator = TokenEstimator()
        result = estimator.estimate("")
        assert result.token_count == 0
        assert result.word_count == 0
        assert result.char_count == 0

    def test_empty_text_fits_any_context(self):
        estimator = TokenEstimator()
        assert estimator.fits_context("", max_tokens=1) is True


# ---------------------------------------------------------------------------
# Very long text
# ---------------------------------------------------------------------------

class TestVeryLongText:
    def test_long_text_scales_linearly(self):
        estimator = TokenEstimator()
        short = estimator.estimate("word " * 100)
        long = estimator.estimate("word " * 1000)
        # Should be roughly 10x, allow tolerance
        assert 8 <= long.token_count / short.token_count <= 12


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_start_at_zero(self):
        estimator = TokenEstimator()
        st = estimator.stats()
        assert st.total_estimated == 0
        assert st.total_tokens == 0
        assert st.by_model == {}

    def test_stats_accumulate_across_calls(self):
        estimator = TokenEstimator()
        estimator.estimate("hello world")
        estimator.estimate("foo bar baz", model_family="claude")
        st = estimator.stats()
        assert st.total_estimated == 2
        assert st.total_tokens > 0
        assert "default" in st.by_model
        assert "claude" in st.by_model

    def test_stats_returns_snapshot(self):
        estimator = TokenEstimator()
        estimator.estimate("test")
        snapshot = estimator.stats()
        estimator.estimate("more text")
        later = estimator.stats()
        assert later.total_estimated > snapshot.total_estimated


# ---------------------------------------------------------------------------
# Different model families produce different counts
# ---------------------------------------------------------------------------

class TestModelFamilyDifferences:
    def test_claude_and_llama_differ(self):
        estimator = TokenEstimator()
        text = "This is a moderately long sentence for testing model differences"
        claude_estimate = estimator.estimate(text, model_family="claude")
        llama_estimate = estimator.estimate(text, model_family="llama")
        assert claude_estimate.token_count != llama_estimate.token_count
        # Claude ratio (1.35) > Llama ratio (1.25), so Claude should be higher
        assert claude_estimate.token_count > llama_estimate.token_count

    def test_gpt4_and_default_are_equal(self):
        estimator = TokenEstimator()
        text = "Same ratio means same count"
        gpt4 = estimator.estimate(text, model_family="gpt-4")
        default = estimator.estimate(text, model_family="default")
        assert gpt4.token_count == default.token_count


# ---------------------------------------------------------------------------
# Unknown model falls back to default
# ---------------------------------------------------------------------------

class TestUnknownModelFallback:
    def test_unknown_family_uses_default_ratio(self):
        estimator = TokenEstimator()
        text = "Fallback test"
        unknown = estimator.estimate(text, model_family="nonexistent-model")
        default = estimator.estimate(text, model_family="default")
        assert unknown.token_count == default.token_count
        assert unknown.ratio == 1.3

    def test_unknown_family_recorded_in_stats(self):
        estimator = TokenEstimator()
        estimator.estimate("test", model_family="mystery")
        st = estimator.stats()
        assert "mystery" in st.by_model
