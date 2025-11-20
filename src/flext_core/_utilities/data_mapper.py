"""Utilities module - FlextUtilitiesDataMapper.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from flext_core.result import FlextResult

_logger = logging.getLogger(__name__)


class FlextUtilitiesDataMapper:
    """Data structure mapping and transformation utilities.

    Provides generic methods for mapping between data structures, building
    objects from flags/mappings, and transforming dict/list structures.

    **Usage Examples**:
    >>> # Map dict keys
    >>> mapping = {"old_key": "new_key", "foo": "bar"}
    >>> result = FlextUtilitiesDataMapper.map_dict_keys(
    ...     {"old_key": "value", "foo": "baz"}, mapping
    ... )
    >>> new_dict = result.unwrap()  # {"new_key": "value", "bar": "baz"}

    >>> # Build object from flags
    >>> flags = ["read", "write"]
    >>> mapping = {"read": "can_read", "write": "can_write"}
    >>> result = FlextUtilitiesDataMapper.build_flags_dict(flags, mapping)
    >>> perms = result.unwrap()  # {"can_read": True, "can_write": True, ...}
    """

    @staticmethod
    def convert_to_int_safe(value: object, default: int) -> int:
        """Convert value to int with safe fallback on error.

        **Generic replacement for**: Manual int conversion with try/except

        Args:
            value: Value to convert (int, str, or other)
            default: Default value to return on conversion failure

        Returns:
            Converted int or default value

        Example:
            >>> FlextUtilitiesDataMapper.convert_to_int_safe("123", 0)
            123
            >>> FlextUtilitiesDataMapper.convert_to_int_safe("invalid", 0)
            0
            >>> FlextUtilitiesDataMapper.convert_to_int_safe(42, 0)
            42

        """
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def map_dict_keys(
        source: dict[str, object],
        key_mapping: dict[str, str],
        *,
        keep_unmapped: bool = True,
    ) -> FlextResult[dict[str, object]]:
        """Map dictionary keys using mapping specification.

        **Generic replacement for**: Key renaming in dicts

        Args:
            source: Source dictionary
            key_mapping: Mapping of old_key → new_key
            keep_unmapped: Keep keys not in mapping (default: True)

        Returns:
            FlextResult with remapped dictionary or error

        Example:
            >>> mapping = {"oldName": "newName", "foo": "bar"}
            >>> result = FlextUtilitiesDataMapper.map_dict_keys(
            ...     {"oldName": "value1", "foo": "value2", "other": "value3"}, mapping
            ... )
            >>> new_dict = result.unwrap()
            >>> # {"newName": "value1", "bar": "value2", "other": "value3"}

        """
        try:
            result: dict[str, object] = {}

            for key, value in source.items():
                new_key = key_mapping.get(key)
                if new_key:
                    result[new_key] = value
                elif keep_unmapped:
                    result[key] = value

            return FlextResult[dict[str, object]].ok(result)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[dict[str, object]].fail(f"Failed to map dict keys: {e}")

    @staticmethod
    def build_flags_dict(
        active_flags: list[str],
        flag_mapping: dict[str, str],
        *,
        default_value: bool = False,
    ) -> FlextResult[dict[str, bool]]:
        """Build boolean flags dictionary from list of active flags.

        **Generic replacement for**: Permission building, feature flags

        Args:
            active_flags: List of active flag names
            flag_mapping: Mapping of flag_name → output_key
            default_value: Default value for inactive flags (default: False)

        Returns:
            FlextResult with flags dictionary or error

        Example:
            >>> flags = ["read", "write"]
            >>> mapping = {
            ...     "read": "can_read",
            ...     "write": "can_write",
            ...     "delete": "can_delete",
            ... }
            >>> result = FlextUtilitiesDataMapper.build_flags_dict(flags, mapping)
            >>> flags_dict = result.unwrap()
            >>> # {"can_read": True, "can_write": True, "can_delete": False}

        """
        try:
            result: dict[str, bool] = {}

            # Initialize all flags to default
            for output_key in flag_mapping.values():
                result[output_key] = default_value

            # Set active flags to True
            for flag in active_flags:
                mapped_key: str | None = flag_mapping.get(flag)
                if mapped_key:
                    result[mapped_key] = True

            return FlextResult[dict[str, bool]].ok(result)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[dict[str, bool]].fail(f"Failed to build flags dict: {e}")

    @staticmethod
    def collect_active_keys(
        source: dict[str, bool],
        key_mapping: dict[str, str],
    ) -> FlextResult[list[str]]:
        """Collect list of output keys where source value is True.

        **Generic replacement for**: Collecting active permissions/flags

        Args:
            source: Dictionary with boolean values
            key_mapping: Mapping of source_key → output_key

        Returns:
            FlextResult with list of active output keys or error

        Example:
            >>> source = {"read": True, "write": True, "delete": False}
            >>> mapping = {"read": "r", "write": "w", "delete": "d"}
            >>> result = FlextUtilitiesDataMapper.collect_active_keys(source, mapping)
            >>> active = result.unwrap()  # ["r", "w"]

        """
        try:
            active_keys: list[str] = []

            for source_key, output_key in key_mapping.items():
                if source.get(source_key):
                    active_keys.append(output_key)

            return FlextResult[list[str]].ok(active_keys)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[list[str]].fail(f"Failed to collect active keys: {e}")

    @staticmethod
    def transform_values(
        source: dict[str, object],
        transformer: Callable[[object], object],
    ) -> dict[str, object]:
        """Transform all values in dict using transformer function.

        **Generic replacement for**: Manual dict value transformations

        Args:
            source: Source dictionary
            transformer: Function to apply to each value

        Returns:
            Dictionary with transformed values

        Example:
            >>> source = {"a": "hello", "b": "world"}
            >>> result = FlextUtilitiesDataMapper.transform_values(
            ...     source, lambda v: str(v).upper()
            ... )
            >>> # {"a": "HELLO", "b": "WORLD"}

        """
        return {k: transformer(v) for k, v in source.items()}

    @staticmethod
    def filter_dict(
        source: dict[str, object],
        predicate: Callable[[str, object], bool],
    ) -> dict[str, object]:
        """Filter dict by predicate function on key-value pairs.

        **Generic replacement for**: Dict comprehensions with filters

        Args:
            source: Source dictionary
            predicate: Function(key, value) returning bool

        Returns:
            Filtered dictionary

        Example:
            >>> source = {"a": 1, "b": 2, "c": 3}
            >>> result = FlextUtilitiesDataMapper.filter_dict(
            ...     source, lambda k, v: v > 1
            ... )
            >>> # {"b": 2, "c": 3}

        """
        return {k: v for k, v in source.items() if predicate(k, v)}

    @staticmethod
    def invert_dict(
        source: dict[str, str],
        *,
        handle_collisions: str = "last",
    ) -> dict[str, str]:
        """Invert dict mapping (values become keys, keys become values).

        **Generic replacement for**: Manual dict inversion

        Args:
            source: Source dictionary
            handle_collisions: How to handle duplicate values:
                - "first": Keep first occurrence
                - "last": Keep last occurrence (default)

        Returns:
            Inverted dictionary

        Example:
            >>> source = {"a": "x", "b": "y", "c": "x"}
            >>> result = FlextUtilitiesDataMapper.invert_dict(
            ...     source, handle_collisions="first"
            ... )
            >>> # {"x": "a", "y": "b"}  (first "a" kept)

        """
        if handle_collisions == "first":
            result: dict[str, str] = {}
            for k, v in source.items():
                if v not in result:
                    result[v] = k
            return result
        # last
        return {v: k for k, v in source.items()}


__all__ = ["FlextUtilitiesDataMapper"]
