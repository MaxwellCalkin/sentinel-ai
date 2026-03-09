"""Prompt augmentation with safety instructions and defensive framing.

Augment prompts with safety prefixes, boundary markers, and custom
augmentations to make them more resilient against injection and manipulation.

Usage:
    from sentinel.prompt_augmentor import PromptAugmentor

    augmentor = PromptAugmentor()
    result = augmentor.augment("Tell me about quantum physics")
    print(result.augmented)
"""

from __future__ import annotations

from dataclasses import dataclass, field


_SAFETY_PREFIX = (
    "You must refuse harmful, illegal, or unethical requests. "
    "Stay within your defined role."
)
_BOUNDARY_START = "=== USER INPUT START ==="
_BOUNDARY_END = "=== USER INPUT END ==="

_BUILTIN_SAFETY_PREFIX = "safety_prefix"
_BUILTIN_BOUNDARY_MARKERS = "boundary_markers"


@dataclass
class Augmentation:
    """A single augmentation to apply to a prompt."""

    name: str
    text: str
    position: str  # "prefix" | "suffix" | "wrap"
    priority: int = 0
    enabled: bool = True


@dataclass
class AugmentedPrompt:
    """Result of augmenting a prompt."""

    original: str
    augmented: str
    augmentations_applied: list[str]
    char_increase: int
    token_estimate: int


@dataclass
class AugmentorConfig:
    """Configuration for the prompt augmentor."""

    max_augmentation_ratio: float = 2.0
    include_safety_prefix: bool = True
    include_boundary_markers: bool = True


@dataclass
class AugmentorStats:
    """Cumulative statistics for augmentation operations."""

    total_augmented: int = 0
    avg_char_increase: float = 0.0
    augmentations_used: dict[str, int] = field(default_factory=dict)


class PromptAugmentor:
    """Augment prompts with safety instructions and defensive framing."""

    def __init__(self, config: AugmentorConfig | None = None) -> None:
        self._config = config or AugmentorConfig()
        self._augmentations: dict[str, Augmentation] = {}
        self._stats = AugmentorStats()
        self._total_char_increase = 0
        self._register_builtin_augmentations()

    def _register_builtin_augmentations(self) -> None:
        if self._config.include_safety_prefix:
            self._augmentations[_BUILTIN_SAFETY_PREFIX] = Augmentation(
                name=_BUILTIN_SAFETY_PREFIX,
                text=_SAFETY_PREFIX,
                position="prefix",
                priority=-10,
            )
        if self._config.include_boundary_markers:
            self._augmentations[_BUILTIN_BOUNDARY_MARKERS] = Augmentation(
                name=_BUILTIN_BOUNDARY_MARKERS,
                text="",  # handled specially in _apply_wrap
                position="wrap",
                priority=0,
            )

    def add_augmentation(self, aug: Augmentation) -> None:
        """Add a custom augmentation."""
        self._augmentations[aug.name] = aug

    def remove_augmentation(self, name: str) -> None:
        """Remove an augmentation by name. Raises KeyError if not found."""
        if name not in self._augmentations:
            raise KeyError(f"Augmentation '{name}' not found")
        del self._augmentations[name]

    def enable(self, name: str) -> None:
        """Enable an augmentation by name."""
        if name not in self._augmentations:
            raise KeyError(f"Augmentation '{name}' not found")
        self._augmentations[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable an augmentation by name."""
        if name not in self._augmentations:
            raise KeyError(f"Augmentation '{name}' not found")
        self._augmentations[name].enabled = False

    def list_augmentations(self) -> list[Augmentation]:
        """Return all augmentations sorted by priority (lowest first)."""
        return sorted(self._augmentations.values(), key=lambda a: a.priority)

    def augment(self, prompt: str) -> AugmentedPrompt:
        """Apply all enabled augmentations to a prompt."""
        enabled = self._get_enabled_sorted()
        max_length = self._compute_max_length(prompt)

        prefixes = [a for a in enabled if a.position == "prefix"]
        wraps = [a for a in enabled if a.position == "wrap"]
        suffixes = [a for a in enabled if a.position == "suffix"]

        result = prompt
        applied: list[str] = []

        result, applied = self._apply_prefixes(result, prefixes, applied, max_length)
        result, applied = self._apply_wraps(result, wraps, applied, max_length)
        result, applied = self._apply_suffixes(result, suffixes, applied, max_length)

        augmented_prompt = self._build_result(prompt, result, applied)
        self._record_stats(augmented_prompt)
        return augmented_prompt

    def augment_batch(self, prompts: list[str]) -> list[AugmentedPrompt]:
        """Augment a batch of prompts."""
        return [self.augment(p) for p in prompts]

    def stats(self) -> AugmentorStats:
        """Return cumulative augmentation statistics."""
        return AugmentorStats(
            total_augmented=self._stats.total_augmented,
            avg_char_increase=self._stats.avg_char_increase,
            augmentations_used=dict(self._stats.augmentations_used),
        )

    def _get_enabled_sorted(self) -> list[Augmentation]:
        enabled = [a for a in self._augmentations.values() if a.enabled]
        return sorted(enabled, key=lambda a: a.priority)

    def _compute_max_length(self, prompt: str) -> int:
        original_length = max(len(prompt), 1)
        ratio_limit = int(original_length * self._config.max_augmentation_ratio)
        # Floor prevents ratio from blocking augmentations on short prompts
        minimum_floor = 500
        return max(ratio_limit, minimum_floor)

    def _would_exceed_limit(self, candidate: str, max_length: int) -> bool:
        return len(candidate) > max_length

    def _apply_prefixes(
        self,
        result: str,
        prefixes: list[Augmentation],
        applied: list[str],
        max_length: int,
    ) -> tuple[str, list[str]]:
        for aug in prefixes:
            candidate = aug.text + "\n\n" + result
            if not self._would_exceed_limit(candidate, max_length):
                result = candidate
                applied.append(aug.name)
        return result, applied

    def _apply_wraps(
        self,
        result: str,
        wraps: list[Augmentation],
        applied: list[str],
        max_length: int,
    ) -> tuple[str, list[str]]:
        for aug in wraps:
            candidate = self._wrap_text(result, aug)
            if not self._would_exceed_limit(candidate, max_length):
                result = candidate
                applied.append(aug.name)
        return result, applied

    def _apply_suffixes(
        self,
        result: str,
        suffixes: list[Augmentation],
        applied: list[str],
        max_length: int,
    ) -> tuple[str, list[str]]:
        for aug in suffixes:
            candidate = result + "\n\n" + aug.text
            if not self._would_exceed_limit(candidate, max_length):
                result = candidate
                applied.append(aug.name)
        return result, applied

    def _wrap_text(self, text: str, aug: Augmentation) -> str:
        if aug.name == _BUILTIN_BOUNDARY_MARKERS:
            return f"{_BOUNDARY_START}\n{text}\n{_BOUNDARY_END}"
        return f"{aug.text}\n{text}\n{aug.text}"

    def _build_result(
        self, original: str, augmented: str, applied: list[str]
    ) -> AugmentedPrompt:
        char_increase = len(augmented) - len(original)
        token_estimate = int(len(augmented.split()) * 1.3)
        return AugmentedPrompt(
            original=original,
            augmented=augmented,
            augmentations_applied=applied,
            char_increase=char_increase,
            token_estimate=token_estimate,
        )

    def _record_stats(self, result: AugmentedPrompt) -> None:
        self._stats.total_augmented += 1
        self._total_char_increase += result.char_increase
        self._stats.avg_char_increase = (
            self._total_char_increase / self._stats.total_augmented
        )
        for name in result.augmentations_applied:
            self._stats.augmentations_used[name] = (
                self._stats.augmentations_used.get(name, 0) + 1
            )
