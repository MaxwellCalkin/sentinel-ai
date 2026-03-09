"""Heuristic-based token estimation for multiple LLM model families.

Estimates token counts using word-based ratios calibrated per model family
(GPT, Claude, Llama). Supports CJK text, cost estimation, batch processing,
context window checks, and cumulative statistics tracking.

Usage:
    from sentinel.token_estimator import TokenEstimator

    estimator = TokenEstimator()
    estimate = estimator.estimate("Hello, world!", model_family="claude")
    print(estimate.token_count)  # ~3

    cost = estimator.estimate_cost("Hello, world!", model="claude-sonnet")
    print(cost.total_cost)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_CJK_RANGES = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0xAC00, 0xD7AF),
]

_WORD_RE = re.compile(r'\S+')

_MODEL_FAMILY_RATIOS: dict[str, float] = {
    "gpt-4": 1.3,
    "gpt-3.5": 1.3,
    "claude": 1.35,
    "llama": 1.25,
    "default": 1.3,
}

_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4": (30.0, 60.0),
    "gpt-4o": (2.50, 10.0),
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (0.25, 1.25),
}


def _is_cjk(char: str) -> bool:
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in _CJK_RANGES)


def _count_cjk_characters(text: str) -> int:
    return sum(1 for ch in text if _is_cjk(ch))


@dataclass
class TokenEstimate:
    """Result of a token estimation."""
    text: str
    token_count: int
    model_family: str
    char_count: int
    word_count: int
    ratio: float


@dataclass
class CostEstimate:
    """Cost estimate for an LLM API call."""
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str


@dataclass
class EstimatorStats:
    """Cumulative estimation statistics."""
    total_estimated: int = 0
    total_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)


class TokenEstimator:
    """Estimate token counts for LLM text using model-family heuristics.

    Uses word-count-based ratios calibrated per model family. CJK characters
    receive additional token weight since they typically consume more tokens.
    """

    def __init__(self) -> None:
        self._stats = EstimatorStats()

    def estimate(self, text: str, model_family: str = "default") -> TokenEstimate:
        """Estimate token count for text under a given model family.

        Args:
            text: The input text to estimate.
            model_family: One of "gpt-4", "gpt-3.5", "claude", "llama", or "default".

        Returns:
            TokenEstimate with token count and metadata.
        """
        ratio = _MODEL_FAMILY_RATIOS.get(model_family, _MODEL_FAMILY_RATIOS["default"])
        words = _WORD_RE.findall(text)
        word_count = len(words)
        char_count = len(text)

        base_tokens = word_count * ratio
        cjk_bonus = _count_cjk_characters(text) * 2
        token_count = round(base_tokens + cjk_bonus) if text else 0

        self._record_estimation(model_family, token_count)

        return TokenEstimate(
            text=text,
            token_count=token_count,
            model_family=model_family,
            char_count=char_count,
            word_count=word_count,
            ratio=ratio,
        )

    def estimate_cost(
        self,
        text: str,
        model: str,
        estimated_output_tokens: int = 0,
    ) -> CostEstimate:
        """Estimate the dollar cost for processing text through a model.

        Args:
            text: Input text to estimate.
            model: Pricing model key (e.g. "gpt-4", "claude-sonnet").
            estimated_output_tokens: Expected output tokens for the response.

        Returns:
            CostEstimate with per-direction and total costs.
        """
        family = self._pricing_model_to_family(model)
        input_estimate = self.estimate(text, model_family=family)
        input_tokens = input_estimate.token_count

        input_price, output_price = _PRICING_PER_MILLION.get(model, (0.0, 0.0))
        input_cost = input_tokens * input_price / 1_000_000
        output_cost = estimated_output_tokens * output_price / 1_000_000

        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=estimated_output_tokens,
            input_cost=round(input_cost, 10),
            output_cost=round(output_cost, 10),
            total_cost=round(input_cost + output_cost, 10),
            model=model,
        )

    def estimate_batch(
        self,
        texts: list[str],
        model_family: str = "default",
    ) -> list[TokenEstimate]:
        """Estimate token counts for multiple texts.

        Args:
            texts: List of input texts.
            model_family: Model family to use for all estimates.

        Returns:
            List of TokenEstimate results in the same order as inputs.
        """
        return [self.estimate(text, model_family) for text in texts]

    def fits_context(
        self,
        text: str,
        max_tokens: int,
        model_family: str = "default",
    ) -> bool:
        """Check whether text fits within a token budget.

        Args:
            text: Input text to check.
            max_tokens: Maximum allowed tokens.
            model_family: Model family for estimation.

        Returns:
            True if the estimated token count is at or below max_tokens.
        """
        estimate = self.estimate(text, model_family)
        return estimate.token_count <= max_tokens

    def truncate_to_fit(
        self,
        text: str,
        max_tokens: int,
        model_family: str = "default",
    ) -> str:
        """Truncate text so it fits within a token budget.

        Uses binary search over word boundaries to find the longest prefix
        that fits within max_tokens.

        Args:
            text: Input text.
            max_tokens: Token budget.
            model_family: Model family for estimation.

        Returns:
            The (possibly truncated) text that fits within max_tokens.
        """
        if self.fits_context(text, max_tokens, model_family):
            return text

        words = text.split()
        low, high = 0, len(words)

        while low < high:
            mid = (low + high + 1) // 2
            candidate = " ".join(words[:mid])
            if self.fits_context(candidate, max_tokens, model_family):
                low = mid
            else:
                high = mid - 1

        return " ".join(words[:low]) if low > 0 else ""

    def stats(self) -> EstimatorStats:
        """Return cumulative estimation statistics.

        Returns:
            EstimatorStats with totals and per-model breakdowns.
        """
        return EstimatorStats(
            total_estimated=self._stats.total_estimated,
            total_tokens=self._stats.total_tokens,
            by_model=dict(self._stats.by_model),
        )

    def _record_estimation(self, model_family: str, token_count: int) -> None:
        self._stats.total_estimated += 1
        self._stats.total_tokens += token_count
        self._stats.by_model[model_family] = (
            self._stats.by_model.get(model_family, 0) + token_count
        )

    @staticmethod
    def _pricing_model_to_family(model: str) -> str:
        """Map a pricing model name to its estimation family."""
        if model.startswith("gpt"):
            return "gpt-4"
        if model.startswith("claude"):
            return "claude"
        if model.startswith("llama"):
            return "llama"
        return "default"
