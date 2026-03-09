"""Centralized safety configuration management.

Provides a single point of truth for all safety settings
across an application, with validation, profiles, and
inheritance support.

Usage:
    from sentinel.safety_config import SafetyConfig

    config = SafetyConfig()
    config.get("sensitivity")  # "medium"
    config.create_profile("strict", parent="default", values={"sensitivity": "high"})
    config.switch_profile("strict")
    config.get("sensitivity")  # "high"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConfigValue:
    """Schema definition for a configuration key."""
    key: str
    value: Any
    value_type: str
    description: str = ""
    required: bool = False


@dataclass
class ConfigProfile:
    """A named configuration profile with optional parent inheritance."""
    name: str
    values: dict[str, Any]
    description: str = ""
    parent: str | None = None


@dataclass
class ConfigValidation:
    """Result of validating the current configuration."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


@dataclass
class ConfigStats:
    """Summary statistics for the configuration state."""
    total_profiles: int = 0
    total_keys: int = 0
    active_profile: str = ""


_DEFAULT_VALUES: dict[str, Any] = {
    "sensitivity": "medium",
    "max_input_length": 10000,
    "block_on_detection": True,
    "log_level": "info",
    "pii_detection": True,
    "injection_detection": True,
    "content_filtering": True,
}


class SafetyConfig:
    """Centralized safety configuration with profiles and inheritance."""

    def __init__(self) -> None:
        self._profiles: dict[str, ConfigProfile] = {}
        self._registry: dict[str, ConfigValue] = {}
        self._active_profile_name: str = "default"
        self._create_default_profile()

    def _create_default_profile(self) -> None:
        default_profile = ConfigProfile(
            name="default",
            values=dict(_DEFAULT_VALUES),
            description="Built-in default safety configuration",
        )
        self._profiles["default"] = default_profile

    def _resolve_value(self, profile_name: str, key: str) -> tuple[bool, Any]:
        """Walk the inheritance chain to resolve a key's value.

        Returns (found, value) tuple. Stops at the first profile
        that contains the key.
        """
        visited: set[str] = set()
        current_name: str | None = profile_name

        while current_name is not None and current_name not in visited:
            visited.add(current_name)
            profile = self._profiles.get(current_name)
            if profile is None:
                break
            if key in profile.values:
                return True, profile.values[key]
            current_name = profile.parent

        return False, None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the active profile, walking parents if needed."""
        found, value = self._resolve_value(self._active_profile_name, key)
        if found:
            return value
        return default

    def set(self, key: str, value: Any) -> None:
        """Set a value in the active profile."""
        self._profiles[self._active_profile_name].values[key] = value

    def create_profile(
        self,
        name: str,
        parent: str | None = None,
        values: dict[str, Any] | None = None,
    ) -> None:
        """Create a new configuration profile.

        Raises KeyError if the specified parent does not exist.
        """
        if parent is not None and parent not in self._profiles:
            raise KeyError(f"Parent profile '{parent}' does not exist")

        profile = ConfigProfile(
            name=name,
            values=dict(values) if values else {},
            parent=parent,
        )
        self._profiles[name] = profile

    def switch_profile(self, name: str) -> None:
        """Switch the active profile.

        Raises KeyError if the profile does not exist.
        """
        if name not in self._profiles:
            raise KeyError(f"Profile '{name}' does not exist")
        self._active_profile_name = name

    def validate(self) -> ConfigValidation:
        """Validate the current configuration against the key registry."""
        errors: list[str] = []
        warnings: list[str] = []

        for key, schema in self._registry.items():
            found, value = self._resolve_value(self._active_profile_name, key)
            if not found:
                if schema.required:
                    errors.append(f"Required key '{key}' is missing")
                else:
                    warnings.append(f"Optional key '{key}' is not set")
                continue

            actual_type = type(value).__name__
            if actual_type != schema.value_type:
                errors.append(
                    f"Key '{key}' expected type '{schema.value_type}', "
                    f"got '{actual_type}'"
                )

        is_valid = len(errors) == 0
        return ConfigValidation(is_valid=is_valid, errors=errors, warnings=warnings)

    def register_key(
        self,
        key: str,
        value_type: str,
        required: bool = False,
        description: str = "",
        default: Any = None,
    ) -> None:
        """Register a configuration key schema for validation."""
        config_value = ConfigValue(
            key=key,
            value=default,
            value_type=value_type,
            required=required,
            description=description,
        )
        self._registry[key] = config_value

    def export(self) -> dict[str, Any]:
        """Export the resolved active profile as a flat dictionary.

        Walks the full inheritance chain so that parent values
        appear alongside the profile's own values.
        """
        resolved: dict[str, Any] = {}
        chain = self._build_inheritance_chain(self._active_profile_name)

        for profile in reversed(chain):
            resolved.update(profile.values)

        return resolved

    def export_json(self) -> str:
        """Export the resolved active profile as a JSON string."""
        return json.dumps(self.export(), indent=2)

    def list_profiles(self) -> list[str]:
        """Return all profile names."""
        return list(self._profiles.keys())

    def diff(self, profile_a: str, profile_b: str) -> dict[str, dict[str, Any]]:
        """Compare two profiles and return keys that differ.

        Each entry maps a key to {"a": value_in_a, "b": value_in_b}.
        Keys present in only one profile use None for the other.
        """
        resolved_a = self._resolve_all(profile_a)
        resolved_b = self._resolve_all(profile_b)

        all_keys = set(resolved_a.keys()) | set(resolved_b.keys())
        differences: dict[str, dict[str, Any]] = {}

        for key in sorted(all_keys):
            val_a = resolved_a.get(key)
            val_b = resolved_b.get(key)
            if val_a != val_b:
                differences[key] = {"a": val_a, "b": val_b}

        return differences

    def stats(self) -> ConfigStats:
        """Return summary statistics about the configuration."""
        all_keys: set[str] = set()
        for profile in self._profiles.values():
            all_keys.update(profile.values.keys())

        return ConfigStats(
            total_profiles=len(self._profiles),
            total_keys=len(all_keys),
            active_profile=self._active_profile_name,
        )

    def _build_inheritance_chain(self, profile_name: str) -> list[ConfigProfile]:
        """Build the list of profiles from child to root ancestor."""
        chain: list[ConfigProfile] = []
        visited: set[str] = set()
        current_name: str | None = profile_name

        while current_name is not None and current_name not in visited:
            visited.add(current_name)
            profile = self._profiles.get(current_name)
            if profile is None:
                break
            chain.append(profile)
            current_name = profile.parent

        return chain

    def _resolve_all(self, profile_name: str) -> dict[str, Any]:
        """Resolve all key-value pairs for a profile including parents."""
        chain = self._build_inheritance_chain(profile_name)
        resolved: dict[str, Any] = {}
        for profile in reversed(chain):
            resolved.update(profile.values)
        return resolved
