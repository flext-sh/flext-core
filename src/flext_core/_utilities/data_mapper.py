"""Utilities module - FlextDataMapper.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextDataMapper:
    """Data structure mapping and transformation utilities."""

    @property
    def logger(self) -> p.StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance (Logger protocol).
        Type annotation omitted to avoid importing structlog.typing here.
        """
        return FlextRuntime.get_logger(__name__)

    """Data structure mapping and transformation utilities.

    Provides generic methods for mapping between data structures, building
    objects from flags/mappings, and transforming dict/list structures.

    **Usage Examples**:
    >>> # Map dict keys
    >>> mapping = {"old_key": "new_key", "foo": "bar"}
    >>> result = uDataMapper.map_dict_keys(
    ...     {"old_key": "value", "foo": "baz"}, mapping
    ... )
    >>> new_dict = result.unwrap()  # {"new_key": "value", "bar": "baz"}

    >>> # Build object from flags
    >>> flags = ["read", "write"]
    >>> mapping = {"read": "can_read", "write": "can_write"}
    >>> result = uDataMapper.build_flags_dict(flags, mapping)
    >>> perms = result.unwrap()  # {"can_read": True, "can_write": True, ...}
    """

    @staticmethod
    def convert_to_int_safe(value: t.GeneralValueType, default: int) -> int:
        """Convert value to int with safe fallback on error.

        **Generic replacement for**: Manual int conversion with try/except

        Args:
            value: Value to convert (int, str, or other)
            default: Default value to return on conversion failure

        Returns:
            Converted int or default value

        Example:
            >>> uDataMapper.convert_to_int_safe("123", 0)
            123
            >>> uDataMapper.convert_to_int_safe("invalid", 0)
            0
            >>> uDataMapper.convert_to_int_safe(42, 0)
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
        source: dict[str, t.GeneralValueType],
        key_mapping: dict[str, str],
        *,
        keep_unmapped: bool = True,
    ) -> r[dict[str, t.GeneralValueType]]:
        """Map dictionary keys using mapping specification.

        **Generic replacement for**: Key renaming in dicts

        Args:
            source: Source dictionary
            key_mapping: Mapping of old_key → new_key
            keep_unmapped: Keep keys not in mapping (default: True)

        Returns:
            r with remapped dictionary or error

        Example:
            >>> mapping = {"oldName": "newName", "foo": "bar"}
            >>> result = uDataMapper.map_dict_keys(
            ...     {"oldName": "value1", "foo": "value2", "other": "value3"}, mapping
            ... )
            >>> new_dict = result.unwrap()
            >>> # {"newName": "value1", "bar": "value2", "other": "value3"}

        """
        try:
            result: dict[str, t.GeneralValueType] = {}

            for key, value in source.items():
                new_key = key_mapping.get(key)
                if new_key:
                    result[new_key] = value
                elif keep_unmapped:
                    result[key] = value

            return r[dict[str, t.GeneralValueType]].ok(result)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[dict[str, t.GeneralValueType]].fail(
                f"Failed to map dict keys: {e}",
            )

    @staticmethod
    def build_flags_dict(
        active_flags: list[str],
        flag_mapping: dict[str, str],
        *,
        default_value: bool = False,
    ) -> r[dict[str, bool]]:
        """Build boolean flags dictionary from list of active flags.

        **Generic replacement for**: Permission building, feature flags

        Args:
            active_flags: List of active flag names
            flag_mapping: Mapping of flag_name → output_key
            default_value: Default value for inactive flags (default: False)

        Returns:
            r with flags dictionary or error

        Example:
            >>> flags = ["read", "write"]
            >>> mapping = {
            ...     "read": "can_read",
            ...     "write": "can_write",
            ...     "delete": "can_delete",
            ... }
            >>> result = uDataMapper.build_flags_dict(flags, mapping)
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

            return r[dict[str, bool]].ok(result)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[dict[str, bool]].fail(f"Failed to build flags dict: {e}")

    @staticmethod
    def collect_active_keys(
        source: dict[str, bool],
        key_mapping: dict[str, str],
    ) -> r[list[str]]:
        """Collect list of output keys where source value is True.

        **Generic replacement for**: Collecting active permissions/flags

        Args:
            source: Dictionary with boolean values
            key_mapping: Mapping of source_key → output_key

        Returns:
            r with list of active output keys or error

        Example:
            >>> source = {"read": True, "write": True, "delete": False}
            >>> mapping = {"read": "r", "write": "w", "delete": "d"}
            >>> result = uDataMapper.collect_active_keys(source, mapping)
            >>> active = result.unwrap()  # ["r", "w"]

        """
        try:
            active_keys: list[str] = []

            for source_key, output_key in key_mapping.items():
                if source.get(source_key):
                    active_keys.append(output_key)

            return r[list[str]].ok(active_keys)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[list[str]].fail(f"Failed to collect active keys: {e}")

    @staticmethod
    def transform_values(
        source: dict[str, t.GeneralValueType],
        transformer: Callable[
            [t.GeneralValueType],
            t.GeneralValueType,
        ],
    ) -> dict[str, t.GeneralValueType]:
        """Transform all values in dict using transformer function.

        **Generic replacement for**: Manual dict value transformations

        Args:
            source: Source dictionary
            transformer: Function to apply to each value

        Returns:
            Dictionary with transformed values

        Example:
            >>> source = {"a": "hello", "b": "world"}
            >>> result = uDataMapper.transform_values(source, lambda v: str(v).upper())
            >>> # {"a": "HELLO", "b": "WORLD"}

        """
        # NOTE: Cannot use u.map() here due to circular import
        # u imports from _utilities, and _utilities cannot import from u
        # Keep implementation simple and direct
        return {k: transformer(v) for k, v in source.items()}

    @staticmethod
    def filter_dict(
        source: dict[str, t.GeneralValueType],
        predicate: Callable[[str, t.GeneralValueType], bool],
    ) -> dict[str, t.GeneralValueType]:
        """Filter dict by predicate function on key-value pairs.

        **NOTE**: Prefer using u.filter() for unified filtering.
        This method delegates to u.filter() for consistency.

        Args:
            source: Source dictionary
            predicate: Function(key, value) returning bool

        Returns:
            Filtered dictionary

        Example:
            >>> source = {"a": 1, "b": 2, "c": 3}
            >>> result = uDataMapper.filter_dict(source, lambda k, v: v > 1)
            >>> # {"b": 2, "c": 3}

        """
        # NOTE: Cannot use u.filter() here due to circular import
        # u imports from _utilities, and _utilities cannot import from u
        # This is a simple dict comprehension - keep it direct
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
            >>> result = uDataMapper.invert_dict(source, handle_collisions="first")
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

    @staticmethod
    def is_json_primitive(value: t.GeneralValueType) -> bool:
        """Check if value is a JSON primitive type (str, int, float, bool, None)."""
        return isinstance(value, (str, int, float, bool, type(None)))

    @classmethod
    def convert_to_json_value(
        cls,
        value: t.GeneralValueType,
    ) -> t.GeneralValueType:
        """Convert any value to JSON-compatible type.

        **Generic replacement for**: Manual type conversion to JSON values

        Conversion Strategy:
            1. Primitives (str, int, float, bool, None) → return as-is
            2. dict-like → recursively convert keys to str, values to JSON
            3. list-like → recursively convert items to JSON
            4. Other → convert to str()

        Args:
            value: t.GeneralValueType value to convert

        Returns:
            JSON-compatible value (str, int, float, bool, None, dict, list)

        Example:
            >>> uDataMapper.convert_to_json_value({"a": 1})
            {'a': 1}
            >>> uDataMapper.convert_to_json_value([1, 2, "three"])
            [1, 2, 'three']

        """
        if cls.is_json_primitive(value):
            return value
        if isinstance(value, dict):
            return {str(k): cls.convert_to_json_value(v) for k, v in value.items()}
        if isinstance(value, Sequence):
            # NOTE: Cannot use u.map() here due to circular import
            # (utilities.py -> data_mapper.py)
            return [cls.convert_to_json_value(item) for item in value]
        # Fallback: convert to string
        return str(value)

    @classmethod
    def convert_dict_to_json(
        cls,
        data: dict[str, t.GeneralValueType],
    ) -> dict[str, t.GeneralValueType]:
        """Convert dict with any values to JSON-compatible dict.

        **Generic replacement for**: Manual dict-to-JSON conversion loops

        Args:
            data: Source dictionary with any values

        Returns:
            Dictionary with all values converted to JSON-compatible types

        Example:
            >>> data = {"name": "test", "value": CustomObject()}
            >>> result = uDataMapper.convert_dict_to_json(data)
            >>> # {"name": "test", "value": "str(CustomObject())"}

        """
        return {
            key: cls.convert_to_json_value(value)
            for key, value in data.items()
            if isinstance(key, str)
        }

    @classmethod
    def convert_list_to_json(
        cls,
        data: Sequence[t.GeneralValueType],
    ) -> list[dict[str, t.GeneralValueType]]:
        """Convert list of dict-like items to JSON-compatible list.

        **Generic replacement for**: Manual list-to-JSON conversion loops

        Args:
            data: Source list of dict-like items

        Returns:
            List with all dict items converted to JSON-compatible format

        Example:
            >>> data = [{"a": 1}, {"b": 2}]
            >>> result = uDataMapper.convert_list_to_json(data)

        """
        return [
            cls.convert_dict_to_json(item) for item in data if isinstance(item, dict)
        ]

    @staticmethod
    def ensure_str(value: t.GeneralValueType, default: str = "") -> str:
        """Ensure value is a string, converting if needed.

        **Generic replacement for**: Manual str() conversions with isinstance checks

        Args:
            value: Value to convert to string
            default: Default value if None or conversion fails

        Returns:
            String value or default

        Example:
            >>> uDataMapper.ensure_str("hello")
            'hello'
            >>> uDataMapper.ensure_str(123)
            '123'
            >>> uDataMapper.ensure_str(None, "default")
            'default'

        """
        if value is None:
            return default
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def ensure_str_list(
        value: t.GeneralValueType,
        default: list[str] | None = None,
    ) -> list[str]:
        """Ensure value is a list of strings, converting if needed.

        **Generic replacement for**: [str(item) for item in list] patterns

        Args:
            value: Value to convert (list, tuple, set, or single value)
            default: Default value if None (empty list if not specified)

        Returns:
            List of strings

        Example:
            >>> uDataMapper.ensure_str_list(["a", "b"])
            ['a', 'b']
            >>> uDataMapper.ensure_str_list([1, 2, 3])
            ['1', '2', '3']
            >>> uDataMapper.ensure_str_list("single")
            ['single']
            >>> uDataMapper.ensure_str_list(None)
            []

        """
        if default is None:
            default = []
        if value is None:
            return default
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            # NOTE: Cannot use u.map() here due to circular import
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def ensure_str_or_none(value: t.GeneralValueType) -> str | None:
        """Ensure value is a string or None.

        **Generic replacement for**: value if isinstance(value, str) else None

        Args:
            value: Value to check/convert

        Returns:
            String value or None

        Example:
            >>> uDataMapper.ensure_str_or_none("hello")
            'hello'
            >>> uDataMapper.ensure_str_or_none(123)
            None
            >>> uDataMapper.ensure_str_or_none(None)
            None

        """
        return value if isinstance(value, str) else None


uDataMapper = FlextDataMapper  # noqa: N816

__all__ = [
    "FlextDataMapper",
    "uDataMapper",
]
