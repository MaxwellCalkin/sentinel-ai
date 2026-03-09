"""Tests for prompt augmentor."""

import pytest
from sentinel.prompt_augmentor import (
    Augmentation,
    AugmentedPrompt,
    AugmentorConfig,
    AugmentorStats,
    PromptAugmentor,
)


class TestDefaultAugmentations:
    def test_default_augmentations_applied(self):
        augmentor = PromptAugmentor()
        result = augmentor.augment("Hello world")
        assert len(result.augmentations_applied) >= 2

    def test_safety_prefix_present(self):
        augmentor = PromptAugmentor()
        result = augmentor.augment("Tell me about cats")
        assert "You must refuse harmful" in result.augmented
        assert "Stay within your defined role" in result.augmented

    def test_boundary_markers_wrap_content(self):
        augmentor = PromptAugmentor()
        result = augmentor.augment("Tell me about cats")
        assert "=== USER INPUT START ===" in result.augmented
        assert "=== USER INPUT END ===" in result.augmented
        start_pos = result.augmented.index("=== USER INPUT START ===")
        end_pos = result.augmented.index("=== USER INPUT END ===")
        assert start_pos < end_pos

    def test_original_content_preserved_inside_markers(self):
        prompt = "Tell me about cats"
        augmentor = PromptAugmentor()
        result = augmentor.augment(prompt)
        start_marker = "=== USER INPUT START ==="
        end_marker = "=== USER INPUT END ==="
        start_idx = result.augmented.index(start_marker) + len(start_marker)
        end_idx = result.augmented.index(end_marker)
        inner = result.augmented[start_idx:end_idx].strip()
        assert prompt in inner

    def test_original_content_inside_markers_without_prefix(self):
        config = AugmentorConfig(include_safety_prefix=False)
        augmentor = PromptAugmentor(config)
        prompt = "Tell me about cats"
        result = augmentor.augment(prompt)
        start_marker = "=== USER INPUT START ==="
        end_marker = "=== USER INPUT END ==="
        start_idx = result.augmented.index(start_marker) + len(start_marker)
        end_idx = result.augmented.index(end_marker)
        inner = result.augmented[start_idx:end_idx].strip()
        assert inner == prompt


class TestCustomAugmentations:
    def test_custom_prefix_augmentation(self):
        augmentor = PromptAugmentor(AugmentorConfig(
            include_safety_prefix=False, include_boundary_markers=False,
        ))
        augmentor.add_augmentation(Augmentation(
            name="custom_prefix", text="SYSTEM: Be concise.", position="prefix",
        ))
        result = augmentor.augment("Explain gravity")
        assert result.augmented.startswith("SYSTEM: Be concise.")
        assert "custom_prefix" in result.augmentations_applied

    def test_custom_suffix_augmentation(self):
        augmentor = PromptAugmentor(AugmentorConfig(
            include_safety_prefix=False, include_boundary_markers=False,
        ))
        augmentor.add_augmentation(Augmentation(
            name="reminder", text="Remember to cite sources.", position="suffix",
        ))
        result = augmentor.augment("Explain gravity")
        assert result.augmented.endswith("Remember to cite sources.")
        assert "reminder" in result.augmentations_applied

    def test_custom_wrap_augmentation(self):
        augmentor = PromptAugmentor(AugmentorConfig(
            include_safety_prefix=False, include_boundary_markers=False,
        ))
        augmentor.add_augmentation(Augmentation(
            name="fence", text="---FENCE---", position="wrap",
        ))
        result = augmentor.augment("My prompt")
        assert result.augmented.startswith("---FENCE---")
        assert result.augmented.endswith("---FENCE---")
        assert "My prompt" in result.augmented


class TestPriorityOrdering:
    def test_lower_priority_applied_first(self):
        augmentor = PromptAugmentor(AugmentorConfig(
            include_safety_prefix=False, include_boundary_markers=False,
        ))
        augmentor.add_augmentation(Augmentation(
            name="second", text="SECOND", position="prefix", priority=10,
        ))
        augmentor.add_augmentation(Augmentation(
            name="first", text="FIRST", position="prefix", priority=1,
        ))
        result = augmentor.augment("base")
        first_pos = result.augmented.index("FIRST")
        second_pos = result.augmented.index("SECOND")
        assert second_pos < first_pos

    def test_list_augmentations_sorted_by_priority(self):
        augmentor = PromptAugmentor(AugmentorConfig(
            include_safety_prefix=False, include_boundary_markers=False,
        ))
        augmentor.add_augmentation(Augmentation(
            name="high", text="H", position="prefix", priority=99,
        ))
        augmentor.add_augmentation(Augmentation(
            name="low", text="L", position="prefix", priority=-5,
        ))
        listed = augmentor.list_augmentations()
        assert listed[0].name == "low"
        assert listed[1].name == "high"


class TestMaxAugmentationRatio:
    def test_augmentation_skipped_when_exceeding_ratio(self):
        config = AugmentorConfig(
            max_augmentation_ratio=1.1,
            include_safety_prefix=False,
            include_boundary_markers=False,
        )
        augmentor = PromptAugmentor(config)
        augmentor.add_augmentation(Augmentation(
            name="huge", text="X" * 1000, position="prefix",
        ))
        result = augmentor.augment("Short prompt")
        assert "huge" not in result.augmentations_applied
        assert "X" * 1000 not in result.augmented

    def test_small_augmentation_fits_within_ratio(self):
        config = AugmentorConfig(
            max_augmentation_ratio=3.0,
            include_safety_prefix=False,
            include_boundary_markers=False,
        )
        augmentor = PromptAugmentor(config)
        augmentor.add_augmentation(Augmentation(
            name="small", text="Hi", position="prefix",
        ))
        result = augmentor.augment("A reasonably long prompt for testing")
        assert "small" in result.augmentations_applied


class TestEnableDisable:
    def test_disable_augmentation(self):
        augmentor = PromptAugmentor(AugmentorConfig(
            include_safety_prefix=True, include_boundary_markers=False,
        ))
        augmentor.disable("safety_prefix")
        result = augmentor.augment("Hello")
        assert "safety_prefix" not in result.augmentations_applied

    def test_enable_augmentation(self):
        augmentor = PromptAugmentor(AugmentorConfig(
            include_safety_prefix=True, include_boundary_markers=False,
        ))
        augmentor.disable("safety_prefix")
        augmentor.enable("safety_prefix")
        result = augmentor.augment("Hello")
        assert "safety_prefix" in result.augmentations_applied

    def test_disable_nonexistent_raises_key_error(self):
        augmentor = PromptAugmentor()
        with pytest.raises(KeyError):
            augmentor.disable("nonexistent")

    def test_enable_nonexistent_raises_key_error(self):
        augmentor = PromptAugmentor()
        with pytest.raises(KeyError):
            augmentor.enable("nonexistent")


class TestRemoveAugmentation:
    def test_remove_existing(self):
        augmentor = PromptAugmentor()
        augmentor.remove_augmentation("safety_prefix")
        names = [a.name for a in augmentor.list_augmentations()]
        assert "safety_prefix" not in names

    def test_remove_nonexistent_raises_key_error(self):
        augmentor = PromptAugmentor()
        with pytest.raises(KeyError):
            augmentor.remove_augmentation("does_not_exist")


class TestBatchAugmentation:
    def test_batch_returns_correct_count(self):
        augmentor = PromptAugmentor()
        results = augmentor.augment_batch(["Prompt A", "Prompt B", "Prompt C"])
        assert len(results) == 3

    def test_batch_each_result_is_augmented_prompt(self):
        augmentor = PromptAugmentor()
        results = augmentor.augment_batch(["Hello", "World"])
        for r in results:
            assert isinstance(r, AugmentedPrompt)
            assert len(r.augmentations_applied) > 0


class TestStats:
    def test_stats_initial_zeros(self):
        augmentor = PromptAugmentor()
        s = augmentor.stats()
        assert s.total_augmented == 0
        assert s.avg_char_increase == 0.0
        assert s.augmentations_used == {}

    def test_stats_after_augmentation(self):
        augmentor = PromptAugmentor()
        augmentor.augment("Test prompt")
        s = augmentor.stats()
        assert s.total_augmented == 1
        assert s.avg_char_increase > 0
        assert len(s.augmentations_used) > 0

    def test_stats_cumulative_across_calls(self):
        augmentor = PromptAugmentor()
        augmentor.augment("First")
        augmentor.augment("Second")
        s = augmentor.stats()
        assert s.total_augmented == 2

    def test_stats_returns_copy(self):
        augmentor = PromptAugmentor()
        augmentor.augment("Test")
        s1 = augmentor.stats()
        s1.total_augmented = 999
        s2 = augmentor.stats()
        assert s2.total_augmented == 1


class TestEmptyPrompt:
    def test_empty_prompt_augmented(self):
        augmentor = PromptAugmentor()
        result = augmentor.augment("")
        assert isinstance(result, AugmentedPrompt)
        assert result.original == ""

    def test_empty_prompt_char_increase_non_negative(self):
        augmentor = PromptAugmentor()
        result = augmentor.augment("")
        assert result.char_increase >= 0


class TestConfigToggles:
    def test_disable_safety_prefix_in_config(self):
        config = AugmentorConfig(include_safety_prefix=False)
        augmentor = PromptAugmentor(config)
        result = augmentor.augment("Hello")
        assert "You must refuse harmful" not in result.augmented
        assert "safety_prefix" not in result.augmentations_applied

    def test_disable_boundary_markers_in_config(self):
        config = AugmentorConfig(include_boundary_markers=False)
        augmentor = PromptAugmentor(config)
        result = augmentor.augment("Hello")
        assert "=== USER INPUT START ===" not in result.augmented
        assert "boundary_markers" not in result.augmentations_applied


class TestCharIncreaseAndTokenEstimate:
    def test_char_increase_calculated_correctly(self):
        config = AugmentorConfig(
            include_safety_prefix=False, include_boundary_markers=False,
        )
        augmentor = PromptAugmentor(config)
        augmentor.add_augmentation(Augmentation(
            name="tag", text="[SAFE]", position="prefix",
        ))
        result = augmentor.augment("My prompt")
        expected_increase = len(result.augmented) - len("My prompt")
        assert result.char_increase == expected_increase

    def test_token_estimate_present_and_positive(self):
        augmentor = PromptAugmentor()
        result = augmentor.augment("Tell me about quantum physics")
        assert result.token_estimate > 0

    def test_token_estimate_uses_word_count_formula(self):
        config = AugmentorConfig(
            include_safety_prefix=False, include_boundary_markers=False,
        )
        augmentor = PromptAugmentor(config)
        prompt = "one two three four five"
        result = augmentor.augment(prompt)
        expected = int(len(result.augmented.split()) * 1.3)
        assert result.token_estimate == expected


class TestDataclasses:
    def test_augmentation_defaults(self):
        aug = Augmentation(name="test", text="hello", position="prefix")
        assert aug.priority == 0
        assert aug.enabled is True

    def test_augmentor_config_defaults(self):
        config = AugmentorConfig()
        assert config.max_augmentation_ratio == 2.0
        assert config.include_safety_prefix is True
        assert config.include_boundary_markers is True

    def test_augmentor_stats_defaults(self):
        stats = AugmentorStats()
        assert stats.total_augmented == 0
        assert stats.avg_char_increase == 0.0
        assert stats.augmentations_used == {}
