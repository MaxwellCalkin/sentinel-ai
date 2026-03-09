"""Tests for centralized safety configuration management."""

import json

import pytest
from sentinel.safety_config import (
    SafetyConfig,
    ConfigProfile,
    ConfigStats,
    ConfigValidation,
    ConfigValue,
)


class TestDefaultProfile:
    def test_default_sensitivity(self):
        config = SafetyConfig()
        assert config.get("sensitivity") == "medium"

    def test_default_max_input_length(self):
        config = SafetyConfig()
        assert config.get("max_input_length") == 10000

    def test_default_block_on_detection(self):
        config = SafetyConfig()
        assert config.get("block_on_detection") is True

    def test_default_log_level(self):
        config = SafetyConfig()
        assert config.get("log_level") == "info"

    def test_default_pii_detection(self):
        config = SafetyConfig()
        assert config.get("pii_detection") is True

    def test_default_injection_detection(self):
        config = SafetyConfig()
        assert config.get("injection_detection") is True

    def test_default_content_filtering(self):
        config = SafetyConfig()
        assert config.get("content_filtering") is True


class TestGetAndSet:
    def test_get_missing_key_returns_none(self):
        config = SafetyConfig()
        assert config.get("nonexistent") is None

    def test_get_missing_key_returns_custom_default(self):
        config = SafetyConfig()
        assert config.get("nonexistent", 42) == 42

    def test_set_new_key(self):
        config = SafetyConfig()
        config.set("custom_key", "custom_value")
        assert config.get("custom_key") == "custom_value"

    def test_set_overrides_default(self):
        config = SafetyConfig()
        config.set("sensitivity", "high")
        assert config.get("sensitivity") == "high"


class TestProfileCreation:
    def test_create_profile_no_parent(self):
        config = SafetyConfig()
        config.create_profile("standalone", values={"foo": "bar"})
        config.switch_profile("standalone")
        assert config.get("foo") == "bar"

    def test_create_profile_with_parent(self):
        config = SafetyConfig()
        config.create_profile("strict", parent="default", values={"sensitivity": "high"})
        config.switch_profile("strict")
        assert config.get("sensitivity") == "high"

    def test_parent_fallback_for_missing_key(self):
        config = SafetyConfig()
        config.create_profile("strict", parent="default", values={"sensitivity": "high"})
        config.switch_profile("strict")
        assert config.get("log_level") == "info"

    def test_invalid_parent_raises_key_error(self):
        config = SafetyConfig()
        with pytest.raises(KeyError, match="does not exist"):
            config.create_profile("child", parent="nonexistent")

    def test_override_parent_value(self):
        config = SafetyConfig()
        config.create_profile("parent_custom", parent="default", values={"log_level": "debug"})
        config.create_profile("child_custom", parent="parent_custom", values={"log_level": "error"})
        config.switch_profile("child_custom")
        assert config.get("log_level") == "error"


class TestMultipleInheritanceLevels:
    def test_three_level_inheritance(self):
        config = SafetyConfig()
        config.create_profile("level1", parent="default", values={"custom_a": 1})
        config.create_profile("level2", parent="level1", values={"custom_b": 2})
        config.create_profile("level3", parent="level2", values={"custom_c": 3})
        config.switch_profile("level3")
        assert config.get("custom_a") == 1
        assert config.get("custom_b") == 2
        assert config.get("custom_c") == 3
        assert config.get("sensitivity") == "medium"

    def test_mid_level_override(self):
        config = SafetyConfig()
        config.create_profile("mid", parent="default", values={"sensitivity": "high"})
        config.create_profile("leaf", parent="mid", values={"pii_detection": False})
        config.switch_profile("leaf")
        assert config.get("sensitivity") == "high"
        assert config.get("pii_detection") is False
        assert config.get("max_input_length") == 10000


class TestSwitchProfile:
    def test_switch_to_valid_profile(self):
        config = SafetyConfig()
        config.create_profile("other", values={"x": 1})
        config.switch_profile("other")
        assert config.get("x") == 1

    def test_switch_to_invalid_profile_raises_key_error(self):
        config = SafetyConfig()
        with pytest.raises(KeyError, match="does not exist"):
            config.switch_profile("nonexistent")

    def test_switch_back_to_default(self):
        config = SafetyConfig()
        config.create_profile("temp", values={"x": 1})
        config.switch_profile("temp")
        config.switch_profile("default")
        assert config.get("sensitivity") == "medium"
        assert config.get("x") is None


class TestValidation:
    def test_valid_config(self):
        config = SafetyConfig()
        config.register_key("sensitivity", "str", required=True)
        result = config.validate()
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_required_key(self):
        config = SafetyConfig()
        config.register_key("mandatory_field", "str", required=True)
        result = config.validate()
        assert result.is_valid is False
        assert any("mandatory_field" in e for e in result.errors)

    def test_type_mismatch(self):
        config = SafetyConfig()
        config.register_key("sensitivity", "int", required=True)
        result = config.validate()
        assert result.is_valid is False
        assert any("type" in e for e in result.errors)

    def test_optional_missing_key_produces_warning(self):
        config = SafetyConfig()
        config.register_key("optional_field", "str", required=False)
        result = config.validate()
        assert result.is_valid is True
        assert any("optional_field" in w for w in result.warnings)


class TestExport:
    def test_export_dict_contains_defaults(self):
        config = SafetyConfig()
        exported = config.export()
        assert exported["sensitivity"] == "medium"
        assert exported["max_input_length"] == 10000

    def test_export_dict_includes_parent_values(self):
        config = SafetyConfig()
        config.create_profile("strict", parent="default", values={"sensitivity": "high"})
        config.switch_profile("strict")
        exported = config.export()
        assert exported["sensitivity"] == "high"
        assert exported["log_level"] == "info"

    def test_export_json_is_valid(self):
        config = SafetyConfig()
        json_str = config.export_json()
        parsed = json.loads(json_str)
        assert parsed["sensitivity"] == "medium"

    def test_export_json_round_trip(self):
        config = SafetyConfig()
        config.set("custom", [1, 2, 3])
        parsed = json.loads(config.export_json())
        assert parsed["custom"] == [1, 2, 3]


class TestListProfiles:
    def test_default_only(self):
        config = SafetyConfig()
        assert config.list_profiles() == ["default"]

    def test_multiple_profiles(self):
        config = SafetyConfig()
        config.create_profile("strict")
        config.create_profile("permissive")
        profiles = config.list_profiles()
        assert "default" in profiles
        assert "strict" in profiles
        assert "permissive" in profiles
        assert len(profiles) == 3


class TestDiff:
    def test_identical_profiles_empty_diff(self):
        config = SafetyConfig()
        config.create_profile("clone", values=dict(config.export()))
        diff = config.diff("default", "clone")
        assert diff == {}

    def test_diff_shows_changed_key(self):
        config = SafetyConfig()
        config.create_profile("strict", parent="default", values={"sensitivity": "high"})
        diff = config.diff("default", "strict")
        assert "sensitivity" in diff
        assert diff["sensitivity"]["a"] == "medium"
        assert diff["sensitivity"]["b"] == "high"

    def test_diff_shows_key_only_in_one_profile(self):
        config = SafetyConfig()
        config.create_profile("extra", parent="default", values={"new_key": True})
        diff = config.diff("default", "extra")
        assert "new_key" in diff
        assert diff["new_key"]["a"] is None
        assert diff["new_key"]["b"] is True


class TestStats:
    def test_initial_stats(self):
        config = SafetyConfig()
        s = config.stats()
        assert s.total_profiles == 1
        assert s.total_keys == 7
        assert s.active_profile == "default"

    def test_stats_after_adding_profile(self):
        config = SafetyConfig()
        config.create_profile("strict", values={"extra_key": True})
        s = config.stats()
        assert s.total_profiles == 2
        assert s.total_keys == 8
        assert s.active_profile == "default"

    def test_stats_reflects_active_profile(self):
        config = SafetyConfig()
        config.create_profile("other", values={})
        config.switch_profile("other")
        s = config.stats()
        assert s.active_profile == "other"


class TestDataclasses:
    def test_config_value_fields(self):
        cv = ConfigValue(key="k", value=1, value_type="int", description="desc", required=True)
        assert cv.key == "k"
        assert cv.value == 1
        assert cv.value_type == "int"
        assert cv.description == "desc"
        assert cv.required is True

    def test_config_profile_defaults(self):
        cp = ConfigProfile(name="p", values={"a": 1})
        assert cp.description == ""
        assert cp.parent is None

    def test_config_validation_structure(self):
        cv = ConfigValidation(is_valid=True, errors=[], warnings=["w1"])
        assert cv.is_valid is True
        assert cv.warnings == ["w1"]

    def test_config_stats_defaults(self):
        cs = ConfigStats()
        assert cs.total_profiles == 0
        assert cs.total_keys == 0
        assert cs.active_profile == ""
