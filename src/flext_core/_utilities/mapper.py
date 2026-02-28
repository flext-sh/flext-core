"""Utilities module - FlextUtilitiesMapper.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeGuard, overload

if TYPE_CHECKING:
    import structlog.stdlib

from pydantic import BaseModel

from flext_core import FlextRuntime, m, p, r, t
from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.guards import FlextUtilitiesGuards


class _Predicate[T](Protocol):
    """Protocol for callable predicates in find_callable."""

    def __call__(self, value: T) -> bool:  # INTERFACE
        """Evaluate predicate against value."""
        ...


type _MapperCallable = Callable[[t.ConfigMapValue], t.ConfigMapValue]


class FlextUtilitiesMapper:
    """Data structure mapping and transformation utilities.

    Provides generic methods for mapping between data structures, building
    objects from flags/mappings, and transforming dict/list structures.

    **Usage** (flat namespace via runtime alias u; no subdivision):
        >>> from flext_core import u
        >>> result = u.get(data, "key", default="")
        >>> new_dict = u.map_dict_keys({"old_key": "value", "foo": "baz"}, mapping)
        >>> perms = u.build_flags_dict(flags, mapping)
    Subprojects use their project u. Aliases follow MRO registration only.
    """

    # =========================================================================
    # TYPE GUARDS AND HELPERS - Replace casts with proper type narrowing
    # =========================================================================

    @staticmethod
    def _narrow_to_configuration_dict(
        value: t.ConfigMapValue,
    ) -> dict[str, t.ConfigMapValue]:
        """Safely narrow object to ConfigurationDict with runtime validation."""
        # Use TypeGuard for proper type narrowing without cast
        if FlextUtilitiesGuards.is_configuration_dict(value):
            normalized_dict: dict[str, t.ConfigMapValue] = {}
            for key, item in value.items():
                normalized_dict[str(key)] = (
                    FlextUtilitiesMapper.narrow_to_general_value_type(item)
                )
            return normalized_dict
        error_msg = f"Cannot narrow {value.__class__.__name__} to ConfigurationDict"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_string_keyed_dict(
        value: t.ConfigMapValue,
    ) -> dict[str, t.ConfigMapValue]:
        """Narrow PayloadValue to ConfigurationDict (for conversion purposes).

        Validates that the value is a dict with string keys and PayloadValue values.
        Uses TypeGuard pattern for proper type narrowing.
        """
        if isinstance(value, Mapping):
            # Narrow dict values to PayloadValue with explicit type annotations
            result: dict[str, t.ConfigMapValue] = {}
            # Iterate over items with explicit key and value types
            key: str
            val: t.ConfigMapValue
            for key, val in value.items():
                # Convert key to string
                str_key = str(key)
                # Validate and convert value
                if FlextUtilitiesGuards.is_general_value_type(val):
                    result[str_key] = val
                else:
                    result[str_key] = str(val)
            return result
        error_msg = f"Cannot narrow {value.__class__.__name__} to ConfigurationDict"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_configuration_mapping(value: t.ConfigMapValue) -> m.ConfigMap:
        """Safely narrow object to ConfigurationMapping with runtime validation."""
        if isinstance(value, m.ConfigMap):
            return value
        if isinstance(value, Mapping):
            # Coerce to ConfigMap (Pydantic will validate keys are strings)
            coerced_result = r[m.ConfigMap].create_from_callable(
                lambda: m.ConfigMap(
                    root=FlextUtilitiesMapper._narrow_to_configuration_dict(value),
                ),
            )
            if coerced_result.is_success and coerced_result.value is not None:
                return coerced_result.value
            error_msg = f"Cannot coerce {value.__class__.__name__} to m.ConfigMap: {coerced_result.error}"
            raise TypeError(error_msg)
        error_msg = f"Cannot narrow {value.__class__.__name__} to m.ConfigMap"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_sequence(value: t.ConfigMapValue) -> Sequence[t.ConfigMapValue]:
        """Safely narrow object to Sequence[t.ConfigMapValue]."""
        if isinstance(value, (list, tuple)):
            # Explicit type annotation for narrowing items
            narrowed_items: list[t.ConfigMapValue] = []
            for item_raw in value:
                item = FlextUtilitiesMapper._to_general_value_from_object(item_raw)
                narrowed_item = FlextUtilitiesMapper.narrow_to_general_value_type(item)
                narrowed_items.append(narrowed_item)
            return narrowed_items
        error_msg = f"Cannot narrow {value.__class__.__name__} to Sequence"
        raise TypeError(error_msg)

    @staticmethod
    def narrow_to_general_value_type(value: t.ConfigMapValue) -> t.ConfigMapValue:
        """Safely narrow object to t.ConfigMapValue.

        Uses TypeGuard-based validation to ensure type safety.
        If value is not a valid t.ConfigMapValue, returns string representation.
        """
        # Use TypeGuard for proper type narrowing
        if FlextUtilitiesGuards.is_general_value_type(value):
            return value
        # Fallback: convert to string (str is a valid t.ConfigMapValue)
        return str(value)

    @staticmethod
    def _to_general_value_from_object(value: t.FlexibleValue) -> t.ConfigMapValue:
        if FlextUtilitiesGuards.is_general_value_type(value):
            return value
        return str(value)

    @staticmethod
    def _get_str_from_dict(
        ops: Mapping[str, t.ConfigMapValue],
        key: str,
        default: str = "",
    ) -> str:
        """Safely extract str value from ConfigurationDict."""
        value = ops.get(key, default)
        if isinstance(value, str):
            return str(value)
        return str(value) if value is not None else default

    @staticmethod
    def _get_callable_from_dict(
        ops: Mapping[str, t.ConfigMapValue],
        key: str,
    ) -> _MapperCallable | None:
        value: t.ConfigMapValue | _MapperCallable | None = ops.get(key)
        if callable(value):
            return value
        return None

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance (Logger protocol).
        Type annotation omitted to avoid importing structlog.typing here.
        """
        logger: structlog.stdlib.BoundLogger = FlextRuntime.get_logger(__name__)
        if FlextUtilitiesMapper._is_structlog_logger(logger):
            return logger
        msg = f"Unexpected logger type: {logger.__class__.__name__}"
        raise TypeError(msg)

    @staticmethod
    def _is_structlog_logger(value: object) -> TypeGuard[p.Log.StructlogLogger]:
        return bool(
            hasattr(value, "debug")
            and hasattr(value, "info")
            and hasattr(value, "warning")
            and hasattr(value, "error")
            and hasattr(value, "exception"),
        )

    @staticmethod
    def map_dict_keys(
        source: Mapping[str, t.ConfigMapValue],
        key_mapping: Mapping[str, str],
        *,
        keep_unmapped: bool = True,
    ) -> r[Mapping[str, t.ConfigMapValue]]:
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
            >>> result = FlextUtilitiesMapper.map_dict_keys(
            ...     {"oldName": "value1", "foo": "value2", "other": "value3"}, mapping
            ... )
            >>> new_dict = result.value
            >>> # {"newName": "value1", "bar": "value2", "other": "value3"}

        """

        def _map_keys() -> Mapping[str, t.ConfigMapValue]:
            result: dict[str, t.ConfigMapValue] = {}

            for key, value in source.items():
                new_key = key_mapping.get(key)
                if new_key:
                    result[new_key] = value
                elif keep_unmapped:
                    result[key] = value

            return result

        mapped_result = r[Mapping[str, t.ConfigMapValue]].create_from_callable(
            _map_keys,
        )
        if mapped_result.is_failure:
            return r[Mapping[str, t.ConfigMapValue]].fail(
                f"Failed to map dict keys: {mapped_result.error}",
            )
        return mapped_result

    @staticmethod
    def build_flags_dict(
        active_flags: list[str],
        flag_mapping: Mapping[str, str],
        *,
        default_value: bool = False,
    ) -> r[Mapping[str, bool]]:
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
            >>> result = FlextUtilitiesMapper.build_flags_dict(flags, mapping)
            >>> flags_dict = result.value
            >>> # {"can_read": True, "can_write": True, "can_delete": False}

        """

        def _build_flags() -> Mapping[str, bool]:
            result: dict[str, bool] = {}

            # Initialize all flags to default
            for output_key in flag_mapping.values():
                result[output_key] = default_value

            # Set active flags to True
            for flag in active_flags:
                mapped_key: str | None = flag_mapping.get(flag)
                if mapped_key:
                    result[mapped_key] = True

            return result

        flags_result = r[Mapping[str, bool]].create_from_callable(_build_flags)
        if flags_result.is_failure:
            return r[Mapping[str, bool]].fail(
                f"Failed to build flags dict: {flags_result.error}",
            )
        return flags_result

    @staticmethod
    def collect_active_keys(
        source: Mapping[str, bool],
        key_mapping: Mapping[str, str],
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
            >>> result = FlextUtilitiesMapper.collect_active_keys(source, mapping)
            >>> active = result.value  # ["r", "w"]

        """

        def _collect_keys() -> list[str]:
            active_keys: list[str] = []

            for source_key, output_key in key_mapping.items():
                if source.get(source_key):
                    active_keys.append(output_key)

            return active_keys

        active_keys_result = r[list[str]].create_from_callable(_collect_keys)
        if active_keys_result.is_failure:
            return r[list[str]].fail(
                f"Failed to collect active keys: {active_keys_result.error}",
            )
        return active_keys_result

    @staticmethod
    def transform_values(
        source: Mapping[str, t.ConfigMapValue],
        transformer: Callable[
            [t.ConfigMapValue],
            t.ConfigMapValue,
        ],
    ) -> Mapping[str, t.ConfigMapValue]:
        """Transform all values in dict using transformer function.

        **Generic replacement for**: Manual dict value transformations

        Args:
            source: Source dictionary
            transformer: Function to apply to each value

        Returns:
            Dictionary with transformed values

        Example:
            >>> source = {"a": "hello", "b": "world"}
            >>> result = FlextUtilitiesMapper.transform_values(
            ...     source, lambda v: str(v).upper()
            ... )
            >>> # {"a": "HELLO", "b": "WORLD"}

        """
        # NOTE: Cannot use u.map() here due to circular import
        # u imports from _utilities, and _utilities cannot import from u
        # Keep implementation simple and direct
        return {k: transformer(v) for k, v in source.items()}

    @staticmethod
    def filter_dict(
        source: Mapping[str, t.ConfigMapValue],
        predicate: Callable[[str, t.ConfigMapValue], bool],
    ) -> Mapping[str, t.ConfigMapValue]:
        """Filter dict by predicate function on key-value pairs.

        Args:
            source: Source dictionary
            predicate: Function(key, value) returning bool

        Returns:
            Filtered dictionary

        Example:
            >>> source = {"a": 1, "b": 2, "c": 3}
            >>> result = FlextUtilitiesMapper.filter_dict(
            ...     source, predicate=lambda k, v: v > 1
            ... )
            >>> # {"b": 2, "c": 3}

        """
        return {k: v for k, v in source.items() if predicate(k, v)}

    @staticmethod
    def invert_dict(
        source: Mapping[str, str],
        *,
        handle_collisions: str = "last",
    ) -> Mapping[str, str]:
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
            >>> result = FlextUtilitiesMapper.invert_dict(
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

    @staticmethod
    def is_json_primitive(value: t.ConfigMapValue) -> bool:
        """Check if value is a JSON primitive type (str, int, float, bool, None)."""
        return bool(
            FlextUtilitiesGuards.is_type(
                value,
                (str, int, float, bool, None.__class__),
            ),
        )

    @classmethod
    def convert_to_json_value(
        cls,
        value: t.ConfigMapValue,
    ) -> t.ConfigMapValue:
        """Convert any value to JSON-compatible type.

        **Generic replacement for**: Manual type conversion to JSON values

        Conversion Strategy:
            1. Primitives (str, int, float, bool, None) → return as-is
            2. dict-like → recursively convert keys to str, values to JSON
            3. list-like → recursively convert items to JSON
            4. Other → convert to str()

        Args:
            value: t.ConfigMapValue value to convert

        Returns:
            JSON-compatible value (str, int, float, bool, None, dict, list)

        Example:
            >>> FlextUtilitiesMapper.convert_to_json_value({"a": 1})
            {'a': 1}
            >>> FlextUtilitiesMapper.convert_to_json_value([1, 2, "three"])
            [1, 2, 'three']

        """
        # Type narrowing: ensure value is PayloadValue (not plain object)
        narrowed_value: t.ConfigMapValue
        if isinstance(value, str | int | float | bool):
            narrowed_value = value
        elif value is None:
            narrowed_value = None
        elif isinstance(value, (dict, list, BaseModel, Path)) or callable(value):
            narrowed_value = value
        else:
            # Non-PayloadValue object -> convert to string
            narrowed_value = str(value)

        if cls.is_json_primitive(narrowed_value):
            return narrowed_value
        # Use isinstance for type narrowing (is_type() doesn't return TypeGuard)
        if isinstance(narrowed_value, Mapping):
            # Convert any dict to JSON-compatible format
            # Purpose: CONVERT arbitrary dicts (even with non-PayloadValue values)
            # to JSON-safe format by recursively calling convert_to_json_value
            result_dict: dict[str, t.ConfigMapValue] = {}
            for key, val in narrowed_value.items():
                val_typed = FlextUtilitiesMapper.narrow_to_general_value_type(val)
                result_dict[str(key)] = FlextUtilitiesMapper.convert_to_json_value(
                    val_typed,
                )
            return result_dict
        # Use isinstance for sequence type narrowing
        if isinstance(narrowed_value, Sequence) and not isinstance(
            narrowed_value,
            str | bytes,
        ):
            # NOTE: Cannot use u.map() here due to circular import
            # (utilities.py -> mapper.py)
            # Type narrowing: narrowed_value is Sequence after isinstance check
            result_list: list[t.ConfigMapValue] = []
            for item in narrowed_value:
                converted_item = cls.convert_to_json_value(item)
                result_list.append(converted_item)
            return result_list
        # Fallback: already narrowed to PayloadValue
        return narrowed_value

    @classmethod
    def convert_dict_to_json(
        cls,
        data: Mapping[str, t.ConfigMapValue],
    ) -> Mapping[str, t.ConfigMapValue]:
        """Convert dict with any values to JSON-compatible dict.

        **Generic replacement for**: Manual dict-to-JSON conversion loops

        Args:
            data: Source dictionary with any values (must have string keys)

        Returns:
            Dictionary with all values converted to JSON-compatible types

        Example:
            >>> data = {"name": "test", "value": CustomObject()}
            >>> result = FlextUtilitiesMapper.convert_dict_to_json(data)
            >>> # {"name": "test", "value": "str(CustomObject())"}

        """
        return {key: cls.convert_to_json_value(value) for key, value in data.items()}

    @classmethod
    def convert_list_to_json(
        cls,
        data: Sequence[t.ConfigMapValue],
    ) -> list[Mapping[str, t.ConfigMapValue]]:
        """Convert list of dict-like items to JSON-compatible list.

        **Generic replacement for**: Manual list-to-JSON conversion loops

        Args:
            data: Source list of dict-like items

        Returns:
            List with all dict items converted to JSON-compatible format

        Example:
            >>> data = [{"a": 1}, {"b": 2}]
            >>> result = FlextUtilitiesMapper.convert_list_to_json(data)

        """
        return [
            FlextUtilitiesMapper.convert_dict_to_json(
                FlextUtilitiesMapper._narrow_to_string_keyed_dict(item),
            )
            for item in data
            if isinstance(item, Mapping)
        ]

    @staticmethod
    def ensure_str(value: t.ConfigMapValue, default: str = "") -> str:
        """Ensure value is a string, converting if needed.

        **Generic replacement for**: Manual str() conversions with isinstance checks

        Args:
            value: Value to convert to string
            default: Default value if None or conversion fails

        Returns:
            String value or default

        Example:
            >>> FlextUtilitiesMapper.ensure_str("hello")
            'hello'
            >>> FlextUtilitiesMapper.ensure_str(123)
            '123'
            >>> FlextUtilitiesMapper.ensure_str(None, "default")
            'default'

        """
        if value is None:
            return default
        if isinstance(value, str):
            return str(value)
        return str(value)

    @staticmethod
    def ensure(
        value: t.ConfigMapValue,
        default: list[str] | None = None,
    ) -> list[str]:
        """Ensure value is a list of strings, converting if needed.

        **Generic replacement for**: [str(item) for item in list] patterns
        **Renamed from**: ensure_str_list

        Args:
            value: Value to convert (list, tuple, set, or single value)
            default: Default value if None (empty list if not specified)

        Returns:
            List of strings

        Example:
            >>> FlextUtilitiesMapper.ensure(["a", "b"])
            ['a', 'b']
            >>> FlextUtilitiesMapper.ensure([1, 2, 3])
            ['1', '2', '3']
            >>> FlextUtilitiesMapper.ensure("single")
            ['single']
            >>> FlextUtilitiesMapper.ensure(None)
            []

        """
        if default is None:
            default = []
        if value is None:
            return default
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            # NOTE: Cannot use u.map() here due to circular import
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def ensure_str_or_none(value: t.ConfigMapValue) -> str | None:
        """Ensure value is a string or None.

        **Generic replacement for**: value if isinstance(value, str) else None

        Args:
            value: Value to check/convert

        Returns:
            String value or None

        Example:
            >>> FlextUtilitiesMapper.ensure_str_or_none("hello")
            'hello'
            >>> FlextUtilitiesMapper.ensure_str_or_none(123)
            None
            >>> FlextUtilitiesMapper.ensure_str_or_none(None)
            None

        """
        return value if isinstance(value, str) else None

    # =========================================================================
    # EXTRACT METHODS - Safe nested data extraction
    # =========================================================================

    @staticmethod
    def _extract_parse_array_index(part: str) -> tuple[str, str | None]:
        """Helper: Parse array index from path part (e.g., "items[0]")."""
        if "[" in part and part.endswith("]"):
            bracket_pos = part.index("[")
            array_match = part[bracket_pos + 1 : -1]
            key_part = part[:bracket_pos]
            return key_part, array_match
        return part, None

    @staticmethod
    def _extract_get_value(
        current: t.ConfigMapValue | BaseModel,
        key_part: str,
    ) -> tuple[t.ConfigMapValue | None, bool]:
        """Helper: Get raw value from dict/object/model."""
        if isinstance(current, Mapping):
            current_mapping: m.ConfigMap = (
                FlextUtilitiesMapper._narrow_to_configuration_mapping(current)
            )
            if key_part in current_mapping.root:
                return current_mapping.root[key_part], True
            return None, False

        # Handle object attributes (dataclasses, plain objects, etc.)
        if hasattr(current, key_part):
            attr_val = getattr(current, key_part)
            if FlextUtilitiesGuards.is_general_value_type(attr_val):
                return attr_val, True
            return str(attr_val), True

        # Handle Pydantic model_dump fallback
        if isinstance(current, BaseModel):
            model_dump_attr = current.model_dump
            if callable(model_dump_attr):
                model_dict = model_dump_attr()
                if FlextUtilitiesGuards.is_general_value_type(
                    model_dict,
                ) and isinstance(model_dict, Mapping):
                    model_dict_typed = (
                        FlextUtilitiesMapper._narrow_to_configuration_dict(
                            model_dict,
                        )
                    )
                    if key_part in model_dict_typed:
                        return model_dict_typed[key_part], True

        return None, False

    @staticmethod
    def _extract_handle_array_index(
        current: t.ConfigMapValue,
        array_match: str,
    ) -> tuple[t.ConfigMapValue | None, str | None]:
        """Helper: Handle array indexing with support for negative indices."""
        if current.__class__ not in {list, tuple}:
            return None, "Not a sequence"
        sequence: Sequence[t.ConfigMapValue] = FlextUtilitiesMapper._narrow_to_sequence(
            current,
        )
        try:
            idx = int(array_match)
            if idx < 0:
                idx = len(sequence) + idx
            if 0 <= idx < len(sequence):
                return sequence[idx], None
            return None, f"Index {int(array_match)} out of range"
        except (ValueError, IndexError):
            return None, f"Invalid index {array_match}"

    @staticmethod
    def extract(
        data: m.ConfigMap | BaseModel | t.ConfigMapValue,
        path: str,
        *,
        default: t.ConfigMapValue | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> r[t.ConfigMapValue | None]:
        """Safe nested data extraction with dot notation.

        Business Rule: Extracts nested values using dot notation paths.
        Supports dict access, object attributes, and Pydantic model fields.
        Array indexing supported via "key[0]" syntax. Required mode fails
        if path not found, otherwise returns default.

        Args:
            data: Source data (dict, object with attributes, or Pydantic model)
            path: Dot-separated path (e.g., "user.profile.name")
            default: Default value if path not found
            required: Fail if path not found
            separator: Path separator (default: ".")

        Returns:
            r containing extracted value or default

        Example:
            config = {"database": {"host": c.Platform.DEFAULT_HOST, "port": 5432}}
            result = FlextUtilitiesMapper.extract(config, "database.port")
            # → r.ok(5432)

        """
        try:
            parts = path.split(separator)
            # current starts as input data - preserve object reference for attr access
            # Only narrow Mappings to PayloadValue; keep objects as-is
            current: t.ConfigMapValue | BaseModel | None
            if isinstance(data, BaseModel):
                current = data
            elif isinstance(data, Mapping):
                current = FlextUtilitiesMapper.narrow_to_general_value_type(data)
            else:
                # Plain object (dataclass, etc.) - keep reference for hasattr/getattr
                current = data

            for i, part in enumerate(parts):
                if current is None:
                    if required:
                        return r[t.ConfigMapValue | None].fail(
                            f"Path '{separator.join(parts[:i])}' is None",
                        )
                    if default is None:
                        return r[t.ConfigMapValue | None].fail(
                            f"Path '{separator.join(parts[:i])}' is None and default is None",
                        )
                    return r[t.ConfigMapValue | None].ok(default)

                key_part, array_match = FlextUtilitiesMapper._extract_parse_array_index(
                    part,
                )

                value, found = FlextUtilitiesMapper._extract_get_value(
                    current,
                    key_part,
                )

                if not found:
                    path_context = separator.join(parts[:i])
                    if required:
                        return r[t.ConfigMapValue | None].fail(
                            f"Key '{key_part}' not found at '{path_context}'",
                        )
                    if default is None:
                        return r[t.ConfigMapValue | None].fail(
                            f"Key '{key_part}' not found at '{path_context}' and default is None",
                        )
                    return r[t.ConfigMapValue | None].ok(default)

                current = value

                # Handle array index
                if array_match is not None:
                    value, error = FlextUtilitiesMapper._extract_handle_array_index(
                        current,
                        array_match,
                    )
                    if error:
                        if required:
                            return r[t.ConfigMapValue | None].fail(
                                f"Array error at '{key_part}': {error}",
                            )
                        if default is None:
                            return r[t.ConfigMapValue | None].fail(
                                f"Array error at '{key_part}': {error} and default is None",
                            )
                        return r[t.ConfigMapValue | None].ok(default)
                    current = value

            if current is None:
                if required:
                    return r[t.ConfigMapValue | None].fail("Extracted value is None")
                if default is None:
                    return r[t.ConfigMapValue | None].fail(
                        "Extracted value is None and default is None",
                    )
                return r[t.ConfigMapValue | None].ok(default)

            # Type narrowing: use TypeGuard to ensure current is PayloadValue
            if FlextUtilitiesGuards.is_general_value_type(current):
                return r[t.ConfigMapValue | None].ok(current)
            # Fallback: convert to string representation for non-PayloadValue
            return r[t.ConfigMapValue | None].ok(str(current))

        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
            return r[t.ConfigMapValue | None].fail(f"Extract failed: {e}")

    # =========================================================================
    # GET METHODS - Unified get function for dict/object access
    # =========================================================================

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
    ) -> t.ConfigMapValue | None: ...

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: str,
    ) -> str: ...

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: bool,
    ) -> bool: ...

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: int,
    ) -> int: ...

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: float,
    ) -> float: ...

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: t.ConfigMapValue | None,
    ) -> t.ConfigMapValue | None: ...

    @staticmethod
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: t.ConfigMapValue | None = None,
    ) -> t.ConfigMapValue | None:
        """Unified get function for dict/object access with default.

        Generic replacement for: get(), get_str(), get_list()

        Automatically detects if data is dict or object and extracts value.
        Uses DSL conversion when default type indicates desired return type.

        Args:
            data: Source data (dict or object)
            key: Key/attribute name
            default: Default value if not found
                - str (e.g., "") -> returns str (generalized from get_str)
                - list[T] (e.g., []) -> returns list[T] (generalized from get_list)
                - Other -> returns T | None

        Returns:
            Extracted value or default (type inferred from default)

        Example:
            # String (generalized from get_str)
            name = FlextUtilitiesMapper.get(data, "name", default="")

            # List (generalized from get_list)
            models = FlextUtilitiesMapper.get(data, "models", default=[])

            # Generic
            port = FlextUtilitiesMapper.get(config, "port", default=c.Platform.DEFAULT_HTTP_PORT)

        """
        return FlextUtilitiesMapper._get_raw(data, key, default=default)

    @staticmethod
    def _get_raw(
        data: p.AccessibleData,
        key: str,
        *,
        default: t.ConfigMapValue | None = None,
    ) -> t.ConfigMapValue:
        """Internal helper for raw get without DSL conversion."""
        match data:
            case dict() | Mapping():
                raw_value = data.get(key)
                if raw_value is None:
                    return default if default is not None else ""
                return FlextUtilitiesMapper.narrow_to_general_value_type(raw_value)
            case _:
                if hasattr(data, key):
                    attr_val = getattr(data, key)
                    return FlextUtilitiesMapper.narrow_to_general_value_type(attr_val)
                return default if default is not None else ""

    @staticmethod
    def prop(
        key: str,
    ) -> Callable[[m.ConfigMap | BaseModel], t.ConfigMapValue]:
        """Create a property accessor function (functional pattern).

        Returns a function that extracts a property/attribute from an object.
        Useful for functional programming patterns and DSL composition.

        Args:
            key: Property/attribute name to access

        Returns:
            Function that takes an object and returns its property value

        Example:
            >>> get_name = FlextUtilitiesMapper.prop("name")
            >>> name = get_name(user)  # Equivalent to user.name

            >>> # Use in pipelines
            >>> names = [get_name(u) for u in users]

        """

        def accessor(
            obj: m.ConfigMap | BaseModel,
        ) -> t.ConfigMapValue:
            """Access property from object."""
            result = FlextUtilitiesMapper.get(obj, key)
            return result if result is not None else ""

        return accessor

    # =========================================================================
    # AT/TAKE/PICK/AS_ METHODS - Unified access functions
    # =========================================================================

    @staticmethod
    def at[T](
        items: list[T] | tuple[T, ...] | Mapping[str, T],
        index: int | str,
        *,
        default: T | None = None,
    ) -> T | None:
        """Get item at index/key (mnemonic: at = get at position).

        Generic replacement for: items[index] with safe access

        Args:
            items: Items to access
            index: Index (int) or key (str)
            default: Default if not found

        Returns:
            Item at index/key or default

        Example:
            user = FlextUtilitiesMapper.at(users, 0)
            value = FlextUtilitiesMapper.at(data_dict, "key")

        """
        try:
            if isinstance(items, Mapping):
                if isinstance(index, str) and index in items:
                    return items[index]
                return default
            if isinstance(index, int) and 0 <= index < len(items):
                return items[index]
            return default
        except (IndexError, KeyError, TypeError):
            return default

    @staticmethod
    @overload
    def take(
        data_or_items: Mapping[str, t.ConfigMapValue] | t.ConfigMapValue,
        key_or_n: str,
        *,
        as_type: type | None = None,
        default: t.ConfigMapValue | None = None,
        from_start: bool = True,
    ) -> t.ConfigMapValue | None: ...

    @staticmethod
    @overload
    def take(
        data_or_items: Mapping[str, t.ConfigMapValue],
        key_or_n: int,
        *,
        as_type: type | None = None,
        default: t.ConfigMapValue | None = None,
        from_start: bool = True,
    ) -> Mapping[str, t.ConfigMapValue]: ...

    @staticmethod
    @overload
    def take(
        data_or_items: list[t.ConfigMapValue] | tuple[t.ConfigMapValue, ...],
        key_or_n: int,
        *,
        as_type: type | None = None,
        default: t.ConfigMapValue | None = None,
        from_start: bool = True,
    ) -> list[t.ConfigMapValue]: ...

    @staticmethod
    def take(
        data_or_items: Mapping[str, t.ConfigMapValue]
        | t.ConfigMapValue
        | list[t.ConfigMapValue]
        | tuple[t.ConfigMapValue, ...],
        key_or_n: str | int,
        *,
        as_type: type | None = None,
        default: t.ConfigMapValue | None = None,
        from_start: bool = True,
    ) -> (
        Mapping[str, t.ConfigMapValue]
        | list[t.ConfigMapValue]
        | t.ConfigMapValue
        | None
    ):
        """Unified take function (generalized from take_n).

        Generic replacement for: list slicing, dict slicing

        Automatically detects operation based on second argument type:
        - If key_or_n is str: extracts value from dict/object with type guard
        - If key_or_n is int: takes first N items from list/dict

        Args:
            data_or_items: Source data (dict/object) or items (list/dict)
            key_or_n: Key name (str) or number of items (int)
            as_type: Optional type to guard against (for extraction mode)
            default: Default value if not found or type mismatch
            from_start: If True, take from start; if False, take from end

        Returns:
            Extracted value with type guard or sliced items

        Example:
            # Extract value (original take behavior)
            port = FlextUtilitiesMapper.take(
                config, "port", as_type=int, default=c.Platform.DEFAULT_HTTP_PORT
            )
            name = FlextUtilitiesMapper.take(
                obj, "name", as_type=str, default="unknown"
            )

            # Take N items (generalized from take_n)
            keys = FlextUtilitiesMapper.take(plugins_dict, 10)
            items = FlextUtilitiesMapper.take(items_list, 5)

        """
        # Detect operation mode based on key_or_n type
        if isinstance(key_or_n, str):
            # Extraction mode: extract value from dict/object
            # Type narrowing: data must be accessible data for get() call
            if FlextUtilitiesGuards.is_configuration_mapping(data_or_items):
                data: p.AccessibleData = data_or_items
            elif isinstance(data_or_items, BaseModel):
                data = data_or_items
            else:
                # Fallback: not accessible, return default
                return default
            key = key_or_n
            value = FlextUtilitiesMapper.get(data, key, default=default)
            if value is None:
                return default
            if as_type is not None and not isinstance(value, as_type):
                return default
            return value

        # Slice mode: take N items from list/dict
        n = key_or_n
        if isinstance(data_or_items, Mapping):
            keys = list(data_or_items.keys())
            selected_keys = keys[:n] if from_start else keys[-n:]
            return {k: data_or_items[k] for k in selected_keys}
        if isinstance(data_or_items, (list, tuple)):
            items_list: list[t.ConfigMapValue] = [
                FlextUtilitiesMapper.narrow_to_general_value_type(
                    FlextUtilitiesMapper._to_general_value_from_object(item),
                )
                for item in data_or_items
            ]
            return items_list[:n] if from_start else items_list[-n:]
        # Fallback for unsupported types: return None
        # Slice operations only make sense for collections
        return None

    @staticmethod
    def pick(
        data: p.AccessibleData,
        *keys: str,
        as_dict: bool = True,
    ) -> Mapping[str, t.ConfigMapValue] | list[t.ConfigMapValue | None]:
        """Pick multiple fields at once (mnemonic: pick = select fields).

        Generic replacement for: Multiple get() calls

        Args:
            data: Source data (dict or object)
            *keys: Field names to pick
            as_dict: If True, return dict; if False, return list

        Returns:
            Dict with picked fields or list of values

        Example:
            fields = FlextUtilitiesMapper.pick(data, "name", "email", "age")
            values = FlextUtilitiesMapper.pick(data, "x", "y", "z", as_dict=False)

        """
        if as_dict:
            return {k: FlextUtilitiesMapper.get(data, k) for k in keys}
        return [FlextUtilitiesMapper.get(data, k) for k in keys]

    @staticmethod
    @overload
    def as_(
        value: t.ConfigMapValue,
        target: type[int],
        *,
        default: int | None = None,
        strict: bool = False,
    ) -> int | None: ...

    @staticmethod
    @overload
    def as_(
        value: t.ConfigMapValue,
        target: type[float],
        *,
        default: float | None = None,
        strict: bool = False,
    ) -> float | None: ...

    @staticmethod
    @overload
    def as_(
        value: t.ConfigMapValue,
        target: type[str],
        *,
        default: str | None = None,
        strict: bool = False,
    ) -> str | None: ...

    @staticmethod
    def as_(
        value: t.ConfigMapValue,
        target: type,
        *,
        default: t.ConfigMapValue | None = None,
        strict: bool = False,
    ) -> t.ConfigMapValue | None:
        """Type conversion with guard (mnemonic: as_ = convert to type).

         Generic replacement for: type check +  patterns

        Args:
            value: Value to convert
            target: Target type
            default: Default if conversion fails
            strict: If True, only exact type; if False, allow coercion

        Returns:
            Converted value or default

        Example:
            port = FlextUtilitiesMapper.as_(config.get("port"), int, default=c.Platform.DEFAULT_HTTP_PORT)
            name = FlextUtilitiesMapper.as_(value, str, default="")

        """
        try:
            if isinstance(value, target):
                return FlextUtilitiesMapper._to_general_value_from_object(value)
        except TypeError:
            if strict:
                return default
        if strict:
            return default
        # Try coercion for basic types using proper type narrowing
        # When target is a specific type, we know T is that type, so conversions are type-safe
        try:
            if target is int and isinstance(value, str | float | bool):
                # Type narrowing: target is int, so T is int, and int(value) is int
                return int(value)
            if target is float and isinstance(value, str | int | bool):
                # Type narrowing: target is float, so T is float, and float(value) is float
                return float(value)
            if target is str:
                # Type narrowing: target is str, so T is str, and str(value) is str
                return str(value)
            if target is bool and isinstance(value, str):
                normalized = value.lower()
                if normalized in {"true", "1", "yes", "on"}:
                    # Type narrowing: target is bool, so T is bool, and True is bool
                    return True
                if normalized in {"false", "0", "no", "off"}:
                    # Type narrowing: target is bool, so T is bool, and False is bool
                    return False
            return default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def or_[T](
        *values: T | None,
        default: T | None = None,
    ) -> T | None:
        """Return first non-None value (mnemonic: or_ = fallback chain).

        Generic replacement for: value1 or value2 or default patterns

        Args:
            *values: Values to try in order
            default: Default if all are None

        Returns:
            First non-None value or default

        Example:
            port = FlextUtilitiesMapper.or_(config.get("port"), env.get("PORT"), default=c.Platform.DEFAULT_HTTP_PORT)

        """
        for value in values:
            if value is not None:
                return value
        return default

    @staticmethod
    def flat[T](
        items: list[list[T] | tuple[T, ...]]
        | list[list[T]]
        | list[tuple[T, ...]]
        | tuple[list[T], ...],
    ) -> list[T]:
        """Flatten nested lists (mnemonic: flat = flatten).

        Generic replacement for: [item for sublist in items for item in sublist]

        Args:
            items: Nested list/tuple structure

        Returns:
            Flattened list

        Example:
            flat_list = FlextUtilitiesMapper.flat([[1, 2], [3, 4]])
            # → [1, 2, 3, 4]

        """
        return [item for sublist in items for item in sublist]

    # =========================================================================
    # BUILD METHODS - DSL builder pattern
    # =========================================================================

    @staticmethod
    def _extract_field_value(
        item: t.ConfigMapValue,
        field_name: str,
    ) -> t.ConfigMapValue | None:
        """Extract field value from item (dict or object).

        Helper method to improve type inference for pyrefly.
        """
        if isinstance(item, Mapping):
            dict_item: dict[str, t.ConfigMapValue] = {}
            for key, value in item.items():
                coerced_value: t.ConfigMapValue = (
                    value
                    if FlextUtilitiesGuards.is_general_value_type(value)
                    else str(value)
                )
                dict_item[str(key)] = coerced_value
            return dict_item.get(field_name)
        if hasattr(item, field_name):
            # getattr returns object since we know the attribute exists
            attr_value = getattr(item, field_name)
            if FlextUtilitiesGuards.is_general_value_type(attr_value):
                return attr_value
            return str(attr_value)
        return None

    @staticmethod
    def agg[T](
        items: list[T] | tuple[T, ...],
        field: str | Callable[[T], int | float],
        *,
        fn: Callable[[list[int | float]], int | float] | None = None,
    ) -> int | float:
        """Aggregate field values from objects (mnemonic: agg = aggregate).

        Generic replacement for: sum(getattr(...)), max(getattr(...))

        Args:
            items: List/tuple of objects
            field: Field name (str) or extractor function (callable)
            fn: Aggregation function (default: sum)

        Returns:
            Aggregated value

        Example:
            # Sum field values
            total = FlextUtilitiesMapper.agg(items, "synced")
            # → 15

            # Max with custom extractor
            max_val = FlextUtilitiesMapper.agg(
                items, lambda r: r.total_entries, fn=max
            )
            # → 30

        """
        # items is list[T] | tuple[T, ...], so isinstance is redundant
        items_list: list[T] = list(items)
        numeric_values: list[int | float] = []
        if callable(field):
            for item in items_list:
                val = field(item)
                # field returns int | float per type signature
                # Type narrowing: val is int | float
                numeric_values.append(val)
        else:
            # After callable check, field is str, so isinstance is redundant
            field_name: str = field
            for item in items_list:
                # Extract value using helper method for better type inference
                val_raw = FlextUtilitiesMapper._extract_field_value(item, field_name)
                # Type narrowing: check if value is numeric
                if isinstance(val_raw, int | float):
                    numeric_values.append(val_raw)

        agg_fn: Callable[[list[int | float]], int | float] = (
            fn if fn is not None else sum
        )
        if numeric_values:
            return agg_fn(numeric_values)
        return 0

    # =========================================================================
    # BUILD METHODS - DSL builder pattern operations
    # =========================================================================

    # =========================================================================
    # BUILD METHODS - DSL builder pattern operations
    # =========================================================================

    @staticmethod
    def _build_apply_ensure(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
    ) -> t.ConfigMapValue:
        """Helper: Apply ensure operation."""
        if "ensure" not in ops:
            return current
        ensure_type = FlextUtilitiesMapper._get_str_from_dict(ops, "ensure", "")
        ensure_default_val = ops.get("ensure_default")
        # Default values by type
        default_map: Mapping[str, t.ConfigMapValue] = {
            "str_list": [],
            "dict": {},
            "list": [],
            "str": "",
        }
        default_val = (
            ensure_default_val
            if ensure_default_val is not None
            else default_map.get(ensure_type, "")
        )
        # Type coercion using match/case to reduce complexity
        match ensure_type:
            case "str":
                return str(current) if current is not None else default_val
            case "list":
                if isinstance(current, list):
                    # Type narrowing: current is list, convert items to t.ConfigMapValue
                    list_current: list[t.ConfigMapValue] = current
                    return [
                        FlextUtilitiesMapper.narrow_to_general_value_type(item)
                        for item in list_current
                    ]
                return (
                    default_val
                    if current is None
                    else [FlextUtilitiesMapper.narrow_to_general_value_type(current)]
                )
            case "str_list":
                if isinstance(current, list):
                    # Type narrowing: current is list[t.ConfigMapValue], convert each to str
                    list_current_str: list[t.ConfigMapValue] = current
                    return [
                        str(FlextUtilitiesMapper.narrow_to_general_value_type(x))
                        for x in list_current_str
                    ]
                return (
                    default_val
                    if current is None
                    else [
                        str(
                            FlextUtilitiesMapper.narrow_to_general_value_type(current),
                        ),
                    ]
                )
            case "dict":
                if isinstance(current, Mapping):
                    # Type narrowing: current is dict, return as ConfigurationDict
                    return FlextUtilitiesMapper._narrow_to_configuration_dict(current)
                return default_val
            case _:
                return current

    @staticmethod
    def _build_apply_filter(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
        default: t.ConfigMapValue,
    ) -> t.ConfigMapValue:
        """Helper: Apply filter operation."""
        if "filter" not in ops:
            return current
        filter_pred_raw = FlextUtilitiesMapper._get_callable_from_dict(ops, "filter")
        if filter_pred_raw is None:
            return current
        filter_pred_callable = filter_pred_raw

        def filter_pred(value: t.ConfigMapValue) -> bool:
            return bool(filter_pred_callable(value))

        # filter_pred returns PayloadValue, used as truthy check in filter context
        # Handle collections
        if isinstance(current, (list, tuple)):
            # Type narrowing: current is Sequence[object], x is t.ConfigMapValue
            seq_current: Sequence[t.ConfigMapValue] = current
            return [
                FlextUtilitiesMapper.narrow_to_general_value_type(x)
                for x in seq_current
                if filter_pred(FlextUtilitiesMapper.narrow_to_general_value_type(x))
            ]
        if isinstance(current, Mapping):
            # Type narrowing: current is dict, use ConfigurationDict
            current_dict: Mapping[str, t.ConfigMapValue] = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(current)
            )
            # Use filter_dict for consistency
            return FlextUtilitiesMapper.filter_dict(
                current_dict,
                lambda _k, v: bool(filter_pred(v)),
            )
        # Single value
        return default if not bool(filter_pred(current)) else current

    @staticmethod
    def _build_apply_map(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
    ) -> t.ConfigMapValue:
        """Helper: Apply map operation."""
        if "map" not in ops:
            return current
        map_func_raw = FlextUtilitiesMapper._get_callable_from_dict(ops, "map")
        if map_func_raw is None:
            return current
        map_callable = map_func_raw

        def map_func(value: t.ConfigMapValue) -> t.ConfigMapValue:
            return FlextUtilitiesMapper._to_general_value_from_object(
                map_callable(value),
            )

        if isinstance(current, (list, tuple)):
            # Type narrowing: current is Sequence, items are t.ConfigMapValue
            seq_current: Sequence[t.ConfigMapValue] = current
            return [
                map_func(FlextUtilitiesMapper._to_general_value_from_object(x))
                for x in seq_current
            ]
        if isinstance(current, Mapping):
            # Type narrowing: current is dict, use ConfigurationDict
            current_dict: Mapping[str, t.ConfigMapValue] = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(current)
            )
            # ConfigurationDict values are t.ConfigMapValue, so map_func works directly
            return {k: map_func(v) for k, v in current_dict.items()}
        # Single value case - narrow to t.ConfigMapValue before mapping
        current_general = FlextUtilitiesMapper.narrow_to_general_value_type(current)
        return map_func(current_general)

    @staticmethod
    def _build_apply_normalize(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
    ) -> t.ConfigMapValue:
        """Helper: Apply normalize operation."""
        if "normalize" not in ops:
            return current
        normalize_case = FlextUtilitiesMapper._get_str_from_dict(ops, "normalize", "")
        if isinstance(current, str):
            return current.lower() if normalize_case == "lower" else current.upper()
        if isinstance(current, (list, tuple)):
            # Type narrowing: current is Sequence, items are t.ConfigMapValue
            seq_current: Sequence[t.ConfigMapValue] = current
            result: list[t.ConfigMapValue] = []
            for x in seq_current:
                x_general = FlextUtilitiesMapper._to_general_value_from_object(x)
                if isinstance(x_general, str):
                    result.append(
                        x_general.lower()
                        if normalize_case == "lower"
                        else x_general.upper(),
                    )
                else:
                    result.append(x_general)
            return result
        return current

    @staticmethod
    def _build_apply_convert(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
    ) -> t.ConfigMapValue:
        """Helper: Apply convert operation."""
        if "convert" not in ops:
            return current
        convert_func = FlextUtilitiesMapper._get_callable_from_dict(ops, "convert")
        if convert_func is None:
            return current
        convert_callable_raw = convert_func

        convert_default = ops.get("convert_default")
        fallback = convert_default

        def convert_callable(value: t.ConfigMapValue) -> t.ConfigMapValue:
            return FlextUtilitiesMapper._to_general_value_from_object(
                convert_callable_raw(value),
            )

        converter_name = (
            convert_callable_raw.__name__
            if hasattr(convert_callable_raw, "__name__")
            else ""
        )
        if fallback is None:
            converter_defaults: Mapping[str, t.ConfigMapValue] = {
                "int": 0,
                "float": 0.0,
                "str": "",
                "bool": False,
                "list": [],
                "dict": {},
                "tuple": (),
                "set": [],
            }
            fallback = converter_defaults.get(converter_name, current)

        def _convert(value: t.ConfigMapValue) -> t.ConfigMapValue:
            converted_result = r[t.ConfigMapValue].create_from_callable(
                lambda: FlextUtilitiesMapper.narrow_to_general_value_type(
                    convert_callable(value),
                ),
            )
            if converted_result.is_success and converted_result.value is not None:
                return converted_result.value
            return FlextUtilitiesMapper.narrow_to_general_value_type(fallback)

        if isinstance(current, (list, tuple)):
            current_items: Sequence[t.ConfigMapValue] = current
            converted = [
                _convert(FlextUtilitiesMapper.narrow_to_general_value_type(item))
                for item in current_items
            ]
            return converted if isinstance(current, list) else tuple(converted)

        return _convert(current)

    @staticmethod
    def _extract_transform_options(
        transform_opts: Mapping[str, t.ConfigMapValue],
    ) -> tuple[
        bool,
        bool,
        bool,
        Mapping[str, str] | None,
        set[str] | None,
        set[str] | None,
        bool,
    ]:
        """Extract transform options from dict."""
        normalize_val = transform_opts.get("normalize")
        normalize_bool = normalize_val if isinstance(normalize_val, bool) else False
        strip_none_val = transform_opts.get("strip_none")
        strip_none_bool = strip_none_val if isinstance(strip_none_val, bool) else False
        strip_empty_val = transform_opts.get("strip_empty")
        strip_empty_bool = (
            strip_empty_val if isinstance(strip_empty_val, bool) else False
        )
        map_keys_val = transform_opts.get("map_keys")
        # Type narrowing: ensure dict values are strings for StringMapping
        map_keys_dict: Mapping[str, str] | None = None
        if isinstance(map_keys_val, Mapping) and all(
            isinstance(v, str) for v in map_keys_val.values()
        ):
            # Build StringMapping from validated dict - all keys/values confirmed as str
            map_keys_dict = {str(k): str(v) for k, v in map_keys_val.items()}
        filter_keys_val = transform_opts.get("filter_keys")
        filter_keys_set: set[str] | None = (
            {str(key) for key in filter_keys_val}
            if isinstance(filter_keys_val, set)
            else None
        )
        exclude_keys_val = transform_opts.get("exclude_keys")
        exclude_keys_set: set[str] | None = (
            {str(key) for key in exclude_keys_val}
            if isinstance(exclude_keys_val, set)
            else None
        )
        to_json_val = transform_opts.get("to_json")
        to_json_bool = to_json_val if isinstance(to_json_val, bool) else False
        return (
            normalize_bool,
            strip_none_bool,
            strip_empty_bool,
            map_keys_dict,
            filter_keys_set,
            exclude_keys_set,
            to_json_bool,
        )

    @staticmethod
    def _apply_normalize(
        result: Mapping[str, t.ConfigMapValue],
        *,
        normalize: bool,
    ) -> Mapping[str, t.ConfigMapValue]:
        """Apply normalize step."""
        if normalize:
            normalized = FlextUtilitiesCache.normalize_component(result)
            if isinstance(normalized, Mapping):
                normalized_result: dict[str, t.ConfigMapValue] = {}
                for key, value in normalized.items():
                    normalized_result[str(key)] = (
                        FlextUtilitiesMapper.narrow_to_general_value_type(value)
                    )
                return normalized_result
        return result

    @staticmethod
    def _apply_map_keys(
        result: Mapping[str, t.ConfigMapValue],
        *,
        map_keys: Mapping[str, str] | None,
    ) -> Mapping[str, t.ConfigMapValue]:
        """Apply map keys step."""
        if map_keys:
            mapped: r[Mapping[str, t.ConfigMapValue]] = (
                FlextUtilitiesMapper.map_dict_keys(
                    result,
                    map_keys,
                )
            )
            if mapped.is_success:
                mapped_value = mapped.value
                return {
                    str(key): FlextUtilitiesMapper.narrow_to_general_value_type(
                        value,
                    )
                    for key, value in mapped_value.items()
                }
        return result

    @staticmethod
    def _apply_filter_keys(
        result: Mapping[str, t.ConfigMapValue],
        *,
        filter_keys: set[str] | None,
    ) -> Mapping[str, t.ConfigMapValue]:
        """Apply filter keys step."""
        if filter_keys:
            filtered_dict: dict[str, t.ConfigMapValue] = {}
            for key in filter_keys:
                if key in result:
                    filtered_dict[key] = result[key]
            return filtered_dict
        return result

    @staticmethod
    def _apply_exclude_keys(
        result: Mapping[str, t.ConfigMapValue],
        *,
        exclude_keys: set[str] | None,
    ) -> Mapping[str, t.ConfigMapValue]:
        """Apply exclude keys step."""
        if exclude_keys:
            filtered_result: dict[str, t.ConfigMapValue] = dict(result)
            for key in exclude_keys:
                _ = filtered_result.pop(key, None)  # Result intentionally unused
            return filtered_result
        return result

    @staticmethod
    def _apply_strip_none(
        result: Mapping[str, t.ConfigMapValue],
        *,
        strip_none: bool,
    ) -> Mapping[str, t.ConfigMapValue]:
        """Apply strip none step."""
        if strip_none:
            # Use filter_dict for consistency
            return FlextUtilitiesMapper.filter_dict(
                result,
                lambda _k, v: v is not None,
            )
        return result

    @staticmethod
    def _apply_strip_empty(
        result: Mapping[str, t.ConfigMapValue],
        *,
        strip_empty: bool,
    ) -> Mapping[str, t.ConfigMapValue]:
        """Apply strip empty step."""
        if strip_empty:
            # Use filter_dict for consistency
            return FlextUtilitiesMapper.filter_dict(
                result,
                lambda _k, v: v not in ("", [], {}, None),
            )
        return result

    @staticmethod
    def _apply_to_json(
        result: Mapping[str, t.ConfigMapValue],
        *,
        to_json: bool,
    ) -> Mapping[str, t.ConfigMapValue]:
        """Apply to JSON step."""
        if to_json:
            return FlextUtilitiesMapper.convert_dict_to_json(result)
        return result

    @staticmethod
    def _apply_transform_steps(
        result: Mapping[str, t.ConfigMapValue],
        *,
        normalize: bool,
        map_keys: Mapping[str, str] | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
        to_json: bool,
    ) -> Mapping[str, t.ConfigMapValue]:
        """Apply transform steps to result dict."""
        result = FlextUtilitiesMapper._apply_normalize(result, normalize=normalize)
        result = FlextUtilitiesMapper._apply_map_keys(result, map_keys=map_keys)
        result = FlextUtilitiesMapper._apply_filter_keys(
            result,
            filter_keys=filter_keys,
        )
        result = FlextUtilitiesMapper._apply_exclude_keys(
            result,
            exclude_keys=exclude_keys,
        )
        result = FlextUtilitiesMapper._apply_strip_none(result, strip_none=strip_none)
        result = FlextUtilitiesMapper._apply_strip_empty(
            result,
            strip_empty=strip_empty,
        )
        return FlextUtilitiesMapper._apply_to_json(result, to_json=to_json)

    @staticmethod
    def apply_transform_steps(
        result: Mapping[str, t.ConfigMapValue],
        *,
        normalize: bool,
        map_keys: Mapping[str, str] | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
        to_json: bool,
    ) -> Mapping[str, t.ConfigMapValue]:
        return FlextUtilitiesMapper._apply_transform_steps(
            result,
            normalize=normalize,
            map_keys=map_keys,
            filter_keys=filter_keys,
            exclude_keys=exclude_keys,
            strip_none=strip_none,
            strip_empty=strip_empty,
            to_json=to_json,
        )

    @staticmethod
    def _build_apply_transform(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
        default: t.ConfigMapValue,
        on_error: str,
    ) -> t.ConfigMapValue:
        """Helper: Apply transform operation."""
        if "transform" not in ops or not FlextUtilitiesGuards.is_type(
            current,
            "mapping",
        ):
            return current
        transform_opts_raw: t.ConfigMapValue = ops["transform"]
        if not isinstance(transform_opts_raw, Mapping):
            return current
        transform_opts = FlextUtilitiesMapper._narrow_to_configuration_dict(
            FlextUtilitiesMapper.narrow_to_general_value_type(transform_opts_raw),
        )
        # Extract transform options
        (
            normalize_bool,
            strip_none_bool,
            strip_empty_bool,
            map_keys_dict,
            filter_keys_set,
            exclude_keys_set,
            to_json_bool,
        ) = FlextUtilitiesMapper._extract_transform_options(transform_opts)
        # Narrow current to ConfigurationMapping
        current_dict: m.ConfigMap = (
            FlextUtilitiesMapper._narrow_to_configuration_mapping(current)
        )
        # Implement transform logic directly using available utilities
        transform_result = r[Mapping[str, t.ConfigMapValue]].create_from_callable(
            lambda: FlextUtilitiesMapper.apply_transform_steps(
                dict(current_dict),
                normalize=normalize_bool,
                map_keys=map_keys_dict,
                filter_keys=filter_keys_set,
                exclude_keys=exclude_keys_set,
                strip_none=strip_none_bool,
                strip_empty=strip_empty_bool,
                to_json=to_json_bool,
            ),
        )
        if transform_result.is_failure:
            if on_error == "stop":
                return default
            return current
        return transform_result.value if transform_result.value is not None else current

    @staticmethod
    def _build_apply_process(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
        default: t.ConfigMapValue,
        on_error: str,
    ) -> t.ConfigMapValue:
        """Helper: Apply process operation."""
        if "process" not in ops:
            return current
        process_func_raw = FlextUtilitiesMapper._get_callable_from_dict(ops, "process")
        if process_func_raw is None:
            return current
        process_callable = process_func_raw

        def process_func(value: t.ConfigMapValue) -> t.ConfigMapValue:
            return FlextUtilitiesMapper._to_general_value_from_object(
                process_callable(value),
            )

        def _process_current() -> t.ConfigMapValue:
            if isinstance(current, (list, tuple)):
                # Type narrowing: current is Sequence, items are t.ConfigMapValue
                seq_current: Sequence[t.ConfigMapValue] = current
                return [
                    process_func(FlextUtilitiesMapper._to_general_value_from_object(x))
                    for x in seq_current
                ]
            if isinstance(current, Mapping):
                # Type narrowing: current is dict, use ConfigurationDict
                current_dict: Mapping[str, t.ConfigMapValue] = (
                    FlextUtilitiesMapper._narrow_to_configuration_dict(current)
                )
                # ConfigurationDict values are t.ConfigMapValue, so process_func works directly
                return {k: process_func(v) for k, v in current_dict.items()}
            # Single value case - narrow to t.ConfigMapValue before processing
            current_general = FlextUtilitiesMapper.narrow_to_general_value_type(
                current,
            )
            return process_func(current_general)

        process_result = r[t.ConfigMapValue].create_from_callable(_process_current)
        if process_result.is_failure:
            return default if on_error == "stop" else current
        return process_result.value if process_result.value is not None else current

    @staticmethod
    def _build_apply_group(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
    ) -> t.ConfigMapValue:
        """Helper: Apply group operation."""
        if "group" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        group_spec_raw: t.ConfigMapValue | _MapperCallable = ops["group"]
        current_items: Sequence[t.ConfigMapValue] = current
        current_list: list[t.ConfigMapValue] = [
            FlextUtilitiesMapper.narrow_to_general_value_type(item)
            for item in current_items
        ]
        # Group by field name (str) or key function (callable)
        # Returns grouped mapping compatible with ConfigMapValue containers
        if isinstance(group_spec_raw, str):
            group_spec = group_spec_raw
            grouped: dict[str, list[t.ConfigMapValue]] = {}
            for item in current_list:
                if isinstance(item, Mapping):
                    key_raw = item.get(group_spec)
                elif isinstance(item, BaseModel):
                    if not hasattr(item, group_spec):
                        continue
                    key_raw = getattr(item, group_spec)
                    if key_raw is None:
                        continue
                else:
                    continue
                key = "" if key_raw is None else str(key_raw)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(item)
            return grouped
        if callable(group_spec_raw):
            grouped_callable: dict[str, list[t.ConfigMapValue]] = {}
            for item in current_list:
                key = str(group_spec_raw(item))
                if key not in grouped_callable:
                    grouped_callable[key] = []
                grouped_callable[key].append(item)
            return grouped_callable
        return current

    @staticmethod
    def _build_apply_sort(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
    ) -> t.ConfigMapValue:
        """Helper: Apply sort operation."""
        if "sort" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        sort_spec_raw: t.ConfigMapValue | _MapperCallable = ops["sort"]
        current_items: Sequence[t.ConfigMapValue] = current
        current_list: list[t.ConfigMapValue] = [
            FlextUtilitiesMapper.narrow_to_general_value_type(item)
            for item in current_items
        ]
        if isinstance(sort_spec_raw, str):
            field_name: str = sort_spec_raw

            def key_func(item: t.ConfigMapValue) -> str:
                # Only call .get() on accessible data types (mappings/models)
                if isinstance(item, Mapping):
                    return str(item.get(field_name, ""))
                if isinstance(item, BaseModel):
                    if hasattr(item, field_name):
                        return str(getattr(item, field_name))
                    return ""
                # For scalar values, use str representation
                return ""

            sorted_list_key: list[t.ConfigMapValue] = sorted(
                current_list,
                key=key_func,
            )
            return (
                list(sorted_list_key)
                if isinstance(current, list)
                else tuple(sorted_list_key)
            )
        if callable(sort_spec_raw):
            sort_callable = sort_spec_raw

            def sort_key(item: t.ConfigMapValue) -> str:
                return str(sort_callable(item))

            sorted_result = r[list[t.ConfigMapValue]].create_from_callable(
                lambda: sorted(
                    current_list,
                    key=sort_key,
                ),
            )
            if sorted_result.is_failure:
                return current
            sorted_callable = sorted_result.value
            return (
                list(sorted_callable)
                if isinstance(current, list)
                else tuple(sorted_callable)
            )
        if sort_spec_raw is True:
            comparable_items: list[t.ConfigMapValue] = [
                FlextUtilitiesMapper.narrow_to_general_value_type(
                    item
                    if (item.__class__ in {str, int, float, bool} or item is None)
                    else str(item),
                )
                for item in current_list
            ]
            sorted_comparable: list[t.ConfigMapValue] = sorted(
                comparable_items,
                key=str,
            )
            if isinstance(current, list):
                return sorted_comparable
            return tuple(sorted_comparable)
        return current

    @staticmethod
    def _build_apply_unique(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
    ) -> t.ConfigMapValue:
        """Helper: Apply unique operation to remove duplicates."""
        if "unique" not in ops or not ops.get("unique"):
            return current
        if not isinstance(current, (list, tuple)):
            return current
        current_items: Sequence[t.ConfigMapValue] = current
        current_list_unique: list[t.ConfigMapValue] = [
            FlextUtilitiesMapper.narrow_to_general_value_type(item)
            for item in current_items
        ]
        seen: set[t.ConfigMapValue | str] = set()
        unique_list: list[t.ConfigMapValue] = []
        for item in current_list_unique:
            item_hashable: t.ConfigMapValue | str = (
                item
                if (item.__class__ in {str, int, float, bool} or item is None)
                else str(item)
            )
            if item_hashable not in seen:
                seen.add(item_hashable)
                unique_list.append(item)
        if isinstance(current, list):
            return unique_list
        return tuple(unique_list)

    @staticmethod
    def _build_apply_slice(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
    ) -> t.ConfigMapValue:
        """Helper: Apply slice operation."""
        if "slice" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        current_items: Sequence[t.ConfigMapValue] = current
        slice_spec = ops["slice"]
        min_slice_length = 2
        if (
            isinstance(slice_spec, (list, tuple))
            and len(slice_spec) >= min_slice_length
        ):
            start_raw = slice_spec[0]
            end_raw = slice_spec[1]
            start: int | None = start_raw if isinstance(start_raw, int) else None
            end: int | None = end_raw if isinstance(end_raw, int) else None
            if isinstance(current, list):
                sliced_list: list[t.ConfigMapValue] = [
                    FlextUtilitiesMapper.narrow_to_general_value_type(item)
                    for item in current_items[start:end]
                ]
                return sliced_list
            sliced_tuple: tuple[t.ConfigMapValue, ...] = tuple(
                FlextUtilitiesMapper.narrow_to_general_value_type(item)
                for item in current_items[start:end]
            )
            return sliced_tuple
        return current

    @staticmethod
    def _build_apply_chunk(
        current: t.ConfigMapValue,
        ops: Mapping[str, t.ConfigMapValue],
    ) -> t.ConfigMapValue:
        """Helper: Apply chunk operation to split into sublists."""
        if "chunk" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        chunk_size = ops["chunk"]
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            return current
        current_items: Sequence[t.ConfigMapValue] = current
        current_list: list[t.ConfigMapValue] = [
            FlextUtilitiesMapper.narrow_to_general_value_type(item)
            for item in current_items
        ]
        chunked: list[list[t.ConfigMapValue]] = []
        for i in range(0, len(current_list), chunk_size):
            chunk: list[t.ConfigMapValue] = current_list[i : i + chunk_size]
            chunked.append(chunk)
        return chunked

    @staticmethod
    def build(
        value: t.ConfigMapValue,
        *,
        ops: Mapping[str, t.ConfigMapValue] | None = None,
        default: t.ConfigMapValue | None = None,
        on_error: str = "stop",
    ) -> t.ConfigMapValue:
        """Builder pattern for fluent operation composition using DSL.

        Uses DSL dict to compose operations:
        {"ensure": "str", "map": lambda x: x*2, ...}

        Operations are applied in order:
        ensure → filter → map → normalize → convert → transform → process →
        group → sort → unique → slice → chunk

        Args:
            value: Initial value to process
            ops: Dict with operation keys:
                - "ensure": str target type ("str", "str_list", "list", "dict")
                - "ensure_default": default value for ensure
                - "filter": predicate function
                - "map": mapper function
                - "normalize": case ("lower" or "upper")
                - "convert": target type class
                - "convert_default": default for convert
                - "transform": dict with transform options
                - "process": processor function
                - "group": str field name or callable for grouping
                - "sort": str field name, callable, or True for natural sort
                - "unique": bool to remove duplicates
                - "slice": tuple[int, int] for slicing (start, end)
                - "chunk": int size for chunking into sublists
            default: Default value on error
            on_error: Error handling ("stop", "skip", "collect")

        Returns:
            Processed value (type depends on operations applied)

        Example:
            result = FlextUtilitiesMapper.build(
                value,
                ops={"ensure": "str", "normalize": "lower"},
            )

        """
        if ops is None:
            return value

        current: t.ConfigMapValue = value
        default_val: t.ConfigMapValue = default if default is not None else value

        # Apply operations in defined order
        current = FlextUtilitiesMapper._build_apply_ensure(current, ops)
        current = FlextUtilitiesMapper._build_apply_filter(current, ops, default_val)
        current = FlextUtilitiesMapper._build_apply_map(current, ops)
        current = FlextUtilitiesMapper._build_apply_normalize(current, ops)
        current = FlextUtilitiesMapper._build_apply_convert(current, ops)
        current = FlextUtilitiesMapper._build_apply_transform(
            current,
            ops,
            default_val,
            on_error,
        )
        current = FlextUtilitiesMapper._build_apply_process(
            current,
            ops,
            default_val,
            on_error,
        )
        current = FlextUtilitiesMapper._build_apply_group(current, ops)
        current = FlextUtilitiesMapper._build_apply_sort(current, ops)
        current = FlextUtilitiesMapper._build_apply_unique(current, ops)
        current = FlextUtilitiesMapper._build_apply_slice(current, ops)

        return FlextUtilitiesMapper._build_apply_chunk(current, ops)

    # =========================================================================
    # FIELDS METHODS - Multi-field extraction
    # =========================================================================

    @staticmethod
    def field(
        source: p.AccessibleData,
        name: str,
        *,
        default: t.ConfigMapValue | None = None,
        required: bool = False,
        ops: Mapping[str, t.ConfigMapValue] | None = None,
    ) -> t.ConfigMapValue | None:
        """Extract single field from source with optional DSL processing.

        FLEXT Pattern: Simplified single-field extraction (split from overloaded fields).

        Args:
            source: Source data (dict or object)
            name: Field name to extract
            default: Default value if field not found
            required: If True, returns None on missing
            ops: Optional DSL operations dict

        Returns:
            Extracted value or default/None

        Example:
            name = FlextUtilitiesMapper.field(payload, "name", default="")
            age = FlextUtilitiesMapper.field(user, "age", default=0)

        """
        # Get raw value first
        raw_value: t.ConfigMapValue | None = FlextUtilitiesMapper.get(source, name)
        # Use default if raw value is None
        value: t.ConfigMapValue | None = raw_value if raw_value is not None else default
        if value is None and required:
            return None
        if ops is not None:
            # Apply DSL operations to value
            value_for_build: t.ConfigMapValue = (
                FlextUtilitiesMapper.narrow_to_general_value_type(value)
                if value is not None
                else FlextUtilitiesMapper.narrow_to_general_value_type("")
            )
            return FlextUtilitiesMapper.build(
                value_for_build,
                ops=ops,
                on_error="stop",
            )
        return value

    @staticmethod
    def fields_multi(
        source: p.AccessibleData,
        spec: Mapping[str, t.ConfigMapValue],
    ) -> Mapping[str, t.ConfigMapValue]:
        """Extract multiple fields using specification dict.

        FLEXT Pattern: Simplified multi-field extraction (split from overloaded fields).

        Args:
            source: Source data (dict or object)
            spec: Field specification dict {field_name: default_value}

        Returns:
            dict with extracted values

        Example:
            data = FlextUtilitiesMapper.fields_multi(
                payload,
                {"name": "", "age": 0, "email": ""},
            )

        """
        result: dict[str, t.ConfigMapValue] = {}
        for field_name, field_default in spec.items():
            value: t.ConfigMapValue = FlextUtilitiesMapper.get(
                source,
                field_name,
                default=field_default,
            )
            result[field_name] = value if value is not None else field_default
        return result

    # =========================================================================
    # CONSTRUCT METHOD - Object construction from spec
    # =========================================================================

    @staticmethod
    def construct(
        spec: Mapping[str, t.ConfigMapValue],
        source: m.ConfigMap | BaseModel | None = None,
        *,
        on_error: str = "stop",
    ) -> Mapping[str, t.ConfigMapValue]:
        """Construct object using mnemonic specification pattern.

        Builds object from mnemonic spec that maps target keys to source
        fields or DSL operations. Supports field mapping, default values,
        and DSL operations.

        Args:
            spec: Construction specification:
                - Direct: {"target_key": "source_field"}
                - Default: {"target_key": {"field": "source_field", "default": value}}
                - DSL: {"target_key": {"field": "source_field", "ops": {...}}}
                - Literal: {"target_key": {"value": literal}}
            source: Optional source data
            on_error: Error handling ("stop", "skip", "collect")

        Returns:
            Constructed dict with target keys

        Example:
            plugin_info = FlextUtilitiesMapper.construct(
                {
                    "name": "plugin_name",
                    "type": "plugin_type",
                },
                source=plugin_data,
            )

        """
        constructed: dict[str, t.ConfigMapValue] = {}

        for target_key, target_spec in spec.items():
            try:
                target_spec_mapping: Mapping[str, t.ConfigMapValue] | None = None
                # Literal value
                if isinstance(target_spec, Mapping):
                    target_spec_mapping = target_spec
                    if "value" in target_spec_mapping:
                        constructed[target_key] = target_spec_mapping["value"]
                        continue

                # Field mapping
                if isinstance(target_spec, str):
                    source_field = target_spec
                    field_default = None
                    field_ops = None
                elif target_spec_mapping is not None:
                    source_field_raw = target_spec_mapping.get("field", target_key)
                    source_field = (
                        str(source_field_raw)
                        if source_field_raw is not None
                        else target_key
                    )
                    field_default = target_spec_mapping.get("default")
                    field_ops = target_spec_mapping.get("ops")
                else:
                    # After type checks, target_spec is t.ConfigMapValue
                    constructed[target_key] = target_spec
                    continue

                # Extract from source
                if source is None:
                    constructed[target_key] = field_default
                    continue

                # Extract field value
                extracted_result = FlextUtilitiesMapper.extract(
                    source,
                    source_field,
                    default=field_default,
                    required=False,
                )
                extracted_raw = (
                    extracted_result.value
                    if extracted_result.is_success
                    else field_default
                )

                # Apply ops if provided
                if field_ops is not None and extracted_raw is not None:
                    if isinstance(field_ops, dict):
                        field_ops_dict: Mapping[str, t.ConfigMapValue] = (
                            FlextUtilitiesMapper._narrow_to_configuration_dict(
                                field_ops,
                            )
                        )
                        extracted = FlextUtilitiesMapper.build(
                            extracted_raw,
                            ops=field_ops_dict,
                        )
                    else:
                        extracted = extracted_raw
                else:
                    extracted = extracted_raw

                # Narrow extracted to t.ConfigMapValue for type safety
                final_value: t.ConfigMapValue = (
                    FlextUtilitiesMapper.narrow_to_general_value_type(
                        extracted if extracted is not None else field_default,
                    )
                )
                constructed[target_key] = final_value

            except (TypeError, ValueError, KeyError, AttributeError) as e:
                error_msg = f"Failed to construct '{target_key}': {e}"
                if on_error == "stop":
                    raise ValueError(error_msg) from e
                if on_error == "skip":
                    continue

        return constructed

    @staticmethod
    def transform(
        source: Mapping[str, t.ConfigMapValue] | m.ConfigMap,
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: Mapping[str, str] | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
        to_json: bool = False,
    ) -> r[Mapping[str, t.ConfigMapValue]]:
        """Transform dictionary with multiple options.

        Args:
            source: Source dictionary to transform.
            normalize: Normalize values using cache normalization.
            strip_none: Remove keys with None values.
            strip_empty: Remove keys with empty values (empty strings, lists, dicts).
            map_keys: Dictionary mapping old keys to new keys.
            filter_keys: Set of keys to keep (all others removed).
            exclude_keys: Set of keys to remove.
            to_json: Convert to JSON-compatible values.

        Returns:
            FlextResult with transformed dictionary.

        Example:
            >>> result = FlextUtilitiesMapper.transform(
            ...     {"old": "value", "remove": None},
            ...     map_keys={"old": "new"},
            ...     strip_none=True,
            ... )
            >>> transformed = result.map_or({})  # {"new": "value"}

        """
        transform_result = r[Mapping[str, t.ConfigMapValue]].create_from_callable(
            lambda: FlextUtilitiesMapper._apply_transform_steps(
                dict(source),
                normalize=normalize,
                map_keys=map_keys,
                filter_keys=filter_keys,
                exclude_keys=exclude_keys,
                strip_none=strip_none,
                strip_empty=strip_empty,
                to_json=to_json,
            ),
        )
        if transform_result.is_failure:
            return r[Mapping[str, t.ConfigMapValue]].fail(
                f"Transform failed: {transform_result.error}",
            )
        return transform_result

    @staticmethod
    def to_dict[T](mapping: Mapping[str, T]) -> Mapping[str, T]:
        """Convert any Mapping to dict safely.

        Generic replacement for: dict(mapping) with type safety.

        Args:
            mapping: Mapping to convert to dict.

        Returns:
            Dictionary with same key-value pairs.

        Example:
            >>> mapping = {"a": 1, "b": 2}
            >>> result = FlextUtilitiesMapper.to_dict(mapping)
            >>> # {"a": 1, "b": 2}

        """
        return dict(mapping)

    @staticmethod
    def deep_eq(
        a: Mapping[str, t.ConfigMapValue],
        b: Mapping[str, t.ConfigMapValue],
    ) -> bool:
        """Deep equality comparison for nested structures.

        Generic replacement for: Manual deep dict comparison.

        Compares nested dictionaries recursively, handling:
        - Nested dicts and lists
        - Different ordering of keys (dict comparison)
        - None values
        - Primitive types (str, int, float, bool)

        Args:
            a: First dictionary to compare.
            b: Second dictionary to compare.

        Returns:
            True if dictionaries are deeply equal, False otherwise.

        Example:
            >>> a = {"nested": {"key": "value"}, "list": [1, 2, 3]}
            >>> b = {"nested": {"key": "value"}, "list": [1, 2, 3]}
            >>> FlextUtilitiesMapper.deep_eq(a, b)
            True

        """
        # Fast path: same object reference
        if a is b:
            return True

        # ConfigurationDict is always dict, so isinstance is redundant
        # Type check is implicit - both parameters are ConfigurationDict

        # Key count check
        if len(a) != len(b):
            return False

        # Recursive comparison using .items() for proper dict iteration
        for key, val_a in a.items():
            if key not in b:
                return False

            val_b = b[key]

            # None comparison
            if val_a is None and val_b is None:
                continue

            if val_a is None or val_b is None:
                return False

            # Nested dict comparison
            if isinstance(val_a, Mapping) and isinstance(val_b, Mapping):
                val_a_dict = FlextUtilitiesMapper._narrow_to_configuration_dict(val_a)
                val_b_dict = FlextUtilitiesMapper._narrow_to_configuration_dict(val_b)
                if not FlextUtilitiesMapper.deep_eq(val_a_dict, val_b_dict):
                    return False
                continue

            # List comparison (order matters)
            if isinstance(val_a, list) and isinstance(val_b, list):
                val_a_list = val_a
                val_b_list = val_b
                if len(val_a_list) != len(val_b_list):
                    return False
                for item_a, item_b in zip(val_a_list, val_b_list, strict=True):
                    if isinstance(item_a, Mapping) and isinstance(item_b, Mapping):
                        item_a_dict = (
                            FlextUtilitiesMapper._narrow_to_configuration_dict(
                                item_a,
                            )
                        )
                        item_b_dict = (
                            FlextUtilitiesMapper._narrow_to_configuration_dict(
                                item_b,
                            )
                        )
                        if not FlextUtilitiesMapper.deep_eq(item_a_dict, item_b_dict):
                            return False
                    elif item_a != item_b:
                        return False
                continue

            # Primitive comparison
            if val_a != val_b:
                return False

        return True

    @staticmethod
    def process_context_data(
        primary_data: m.ConfigMap | t.ConfigMapValue | None = None,
        secondary_data: m.ConfigMap | t.ConfigMapValue | None = None,
        *,
        transformer: Callable[[t.ConfigMapValue], t.ConfigMapValue] | None = None,
        field_overrides: Mapping[str, t.ConfigMapValue] | None = None,
        merge_strategy: str = "merge",
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> Mapping[str, t.ConfigMapValue]:
        """Process and merge contextual data with flexible transformation options.

        Generic utility for processing context data across the FLEXT ecosystem.
        Handles conversion, transformation, filtering, and merging of contextual information.

        **Usage Examples:**
        ```python
        # Exception context processing
        context = FlextUtilitiesMapper.process_context_data(
            primary_data=user_context,
            secondary_data=extra_kwargs,
            field_overrides={"error_type": "ValidationError"},
            transformer=FlextRuntime.normalize_to_metadata_value,
        )

        # Configuration merging
        config = FlextUtilitiesMapper.process_context_data(
            primary_data=base_config,
            secondary_data=user_overrides,
            merge_strategy="deep_merge",
        )

        # API request processing
        request_data = FlextUtilitiesMapper.process_context_data(
            primary_data=request_body,
            secondary_data=query_params,
            filter_keys={"password", "secret"},
            transformer=str,
        )
        ```

        Args:
            primary_data: Main data source (dict, object, or None)
            secondary_data: Additional data to merge (dict, object, or None)
            transformer: Function to transform all values (default: identity)
            field_overrides: Specific field values to override/add
            merge_strategy: How to merge data ("merge", "primary_only", "secondary_only")
            filter_keys: Only include these keys if specified
            exclude_keys: Exclude these keys from result

        Returns:
            Processed and merged configuration dictionary

        """
        # Default transformer (identity function)
        if transformer is None:

            def identity_transformer(x: t.ConfigMapValue) -> t.ConfigMapValue:
                return x

            transformer = identity_transformer

        result: dict[str, t.ConfigMapValue] = {}

        # Process primary data
        if primary_data is not None:
            primary_source: Mapping[str, t.ConfigMapValue] | None = None
            if isinstance(primary_data, m.ConfigMap):
                primary_source = primary_data.root
            else:
                primary_general = FlextUtilitiesMapper.narrow_to_general_value_type(
                    primary_data,
                )
                if FlextRuntime.is_dict_like(primary_general):
                    primary_source = dict(primary_general)

            if primary_source is not None:
                primary_dict = {
                    str(key): FlextUtilitiesMapper.narrow_to_general_value_type(value)
                    for key, value in primary_source.items()
                }
                transformed_primary = FlextUtilitiesMapper.transform_values(
                    primary_dict,
                    transformer,
                )
                result.update(transformed_primary)

        # Process secondary data based on merge strategy
        if secondary_data is not None and merge_strategy != "primary_only":
            secondary_source: Mapping[str, t.ConfigMapValue] | None = None
            if isinstance(secondary_data, m.ConfigMap):
                secondary_source = secondary_data.root
            else:
                secondary_general = FlextUtilitiesMapper.narrow_to_general_value_type(
                    secondary_data,
                )
                if FlextRuntime.is_dict_like(secondary_general):
                    secondary_source = dict(secondary_general)

            if secondary_source is not None:
                secondary_dict = {
                    str(key): FlextUtilitiesMapper.narrow_to_general_value_type(value)
                    for key, value in secondary_source.items()
                }
                transformed_secondary = FlextUtilitiesMapper.transform_values(
                    secondary_dict,
                    transformer,
                )

                if merge_strategy == "secondary_only":
                    result = dict(transformed_secondary)
                elif merge_strategy == "merge":
                    result.update(transformed_secondary)
                # For other strategies, secondary data is ignored

        # Apply field overrides
        if field_overrides:
            for key, value in field_overrides.items():
                # transformer is guaranteed to be callable after the None check above
                transformed_value: t.ConfigMapValue = transformer(value)
                result[key] = transformed_value

        # Apply filtering
        if filter_keys:
            result = dict(
                FlextUtilitiesMapper.filter_dict(
                    result,
                    lambda k, _v: k in filter_keys,
                ),
            )

        if exclude_keys:
            result = dict(
                FlextUtilitiesMapper.filter_dict(
                    result,
                    lambda k, _v: k not in exclude_keys,
                ),
            )

        return result

    @staticmethod
    def normalize_context_values(
        context: m.ConfigMap | None,
        extra_kwargs: m.ConfigMap,
        **specific_fields: t.MetadataAttributeValue,
    ) -> Mapping[str, t.MetadataAttributeValue]:
        """Normalize and merge context values for exception handling.

        Convenience method for exception context processing.
        Uses process_context_data with metadata normalization.

        Args:
            context: Optional context mapping to normalize
            extra_kwargs: Additional kwargs to normalize and merge
            **specific_fields: Specific fields to add (field, value, config_key, etc.)

        Returns:
            Normalized metadata attribute dict

        """
        # Convert specific_fields to ConfigurationDict for process_context_data
        field_overrides_config: Mapping[str, t.ConfigMapValue] = dict(specific_fields)
        raw_result: Mapping[str, t.ConfigMapValue] = (
            FlextUtilitiesMapper.process_context_data(
                primary_data=context,
                secondary_data=extra_kwargs,
                transformer=FlextRuntime.normalize_to_metadata_value,
                field_overrides=field_overrides_config,
                merge_strategy="merge",
            )
        )
        # Transformer ensures all values are MetadataAttributeValue
        # Build result with explicit type
        result: dict[str, t.MetadataAttributeValue] = {}
        for k, v in raw_result.items():
            result[k] = FlextRuntime.normalize_to_metadata_value(v)
        return result

    # ========================================================================
    # Additional Mapper Convenience Methods
    # ========================================================================

    @staticmethod
    def omit[T](data: Mapping[str, T], *keys: str) -> Mapping[str, T]:
        """Omit specific keys from mapping.

        Generic replacement for: {k: v for k, v in data.items() if k not in keys}

        Args:
            data: Source mapping
            *keys: Keys to omit

        Returns:
            Dict without the specified keys

        Example:
            clean = u.Mapper.omit(user_data, "password", "secret")
            # {"name": "John", "email": "john@test.com"}

        """
        keys_set = set(keys)
        return {k: v for k, v in data.items() if k not in keys_set}

    @staticmethod
    def pluck(
        items: Sequence[Mapping[str, object]],
        key: str,
        default: t.ConfigMapValue | None = None,
    ) -> list[t.ConfigMapValue | None]:
        """Extract single key from sequence of mappings.

        Generic replacement for: [item.get(key) for item in items]

        Args:
            items: Sequence of mappings
            key: Key to extract
            default: Default value if key not found

        Returns:
            List of values for the specified key

        Example:
            names = u.Mapper.pluck(users, "name")
            # ["Alice", "Bob", "Charlie"]

            ages = u.Mapper.pluck(users, "age", default=0)
            # [25, 30, 0]

        """
        values: list[t.ConfigMapValue | None] = []
        for item in items:
            raw_value = item.get(key, default)
            if raw_value is None:
                values.append(None)
            else:
                values.append(
                    FlextUtilitiesMapper._to_general_value_from_object(raw_value),
                )
        return values

    @staticmethod
    def key_by[T, K](
        items: Sequence[T],
        key_func: Callable[[T], K],
    ) -> Mapping[K, T]:
        """Create dict keyed by function result.

        Generic replacement for: {key_func(item): item for item in items}

        Args:
            items: Items to index
            key_func: Function to extract key from each item

        Returns:
            Dict mapping keys to items (last item wins if duplicate keys)

        Example:
            users_by_id = u.Mapper.key_by(users, lambda u: u.id)
            # {1: User(id=1, ...), 2: User(id=2, ...)}

            users_by_email = u.Mapper.key_by(users, lambda u: u.email.lower())

        """
        return {key_func(item): item for item in items}

    @staticmethod
    def fields(
        obj: m.ConfigMap | Mapping[str, t.ConfigMapValue] | t.ConfigMapValue,
        *field_names: str | Mapping[str, t.ConfigMapValue],
    ) -> Mapping[str, t.ConfigMapValue]:
        """Extract specified fields from object.

        Supports two patterns:
        1. Simple: u.fields(obj, "name", "email", "id")
        2. DSL spec: u.fields(obj, {"name": {"default": ""}, ...})

        Args:
            obj: Object or dict to extract from
            *field_names: Field names (str) or field specs (dict)

        Returns:
            Mapping with extracted fields (string-keyed payload values)

        Example:
            # Simple extraction
            data = u.fields(user, "name", "email", "id")

            # With field specs
            data = u.fields(payload, {
                "name": {"default": ""},
                "count": {"default": 0}
            })

        """
        result: dict[str, t.ConfigMapValue] = {}

        spec_item: str | Mapping[str, t.ConfigMapValue]
        for spec_item in field_names:
            # DSL pattern: dict with field specifications
            if isinstance(spec_item, Mapping):
                name: str
                field_config: t.ConfigMapValue
                for name, field_config in spec_item.items():
                    if isinstance(obj, Mapping):
                        if name in obj:
                            result[name] = obj[name]
                        elif isinstance(field_config, Mapping):
                            # Use default value from spec - narrow the type explicitly
                            default_value = field_config.get("default")
                            if default_value is not None:
                                result[name] = default_value
                        else:
                            result[name] = field_config
                    elif hasattr(obj, name):
                        result[name] = getattr(obj, name)
                    elif isinstance(field_config, Mapping):
                        # Use default value from spec - narrow the type explicitly
                        default_value = field_config.get("default")
                        if default_value is not None:
                            result[name] = default_value
            # Simple pattern: string field name (spec_item is str after type check)
            else:
                # After type check (spec_item, Mapping) is False, and field_names is
                # typed as Sequence[str | Mapping[...]], spec_item must be str
                field_name: str = spec_item
                if isinstance(obj, Mapping):
                    if field_name in obj:
                        result[field_name] = obj[field_name]
                elif hasattr(obj, field_name):
                    result[field_name] = getattr(obj, field_name)

        return result

    @staticmethod
    def cast_generic[T](
        value: t.ConfigMapValue,
        target_type: Callable[[object], T] | None = None,
        *,
        default: T | None = None,
    ) -> T | t.ConfigMapValue:
        """Safe cast with fallback.

        Args:
            value: Value to cast
            target_type: Callable/type that converts object to T (optional)
            default: Default value if cast fails

        Returns:
            Cast value or default

        Example:
            port = u.cast_generic(config.get("port"), int, default=8080)

        """
        if target_type is None:
            return value

        cast_result = r[T].create_from_callable(lambda: target_type(value))
        if cast_result.is_success and cast_result.value is not None:
            return cast_result.value
        if default is not None:
            return default
        return value

    @staticmethod
    def find_callable[T](
        callables: Mapping[str, _Predicate[T]],
        value: T,
    ) -> str | None:
        """Find first matching callable key from dict of predicates.

        Iterates through mapping of named predicates and returns the key of
        the first predicate that returns True for the given value.

        Args:
            callables: Mapping of name → predicate function
            value: Value to test against predicates

        Returns:
            Key of matching predicate, or None if no match found

        Example:
            >>> predicates = {
            ...     "is_empty": lambda v: len(v) == 0,
            ...     "is_single": lambda v: len(v) == 1,
            ...     "is_multiple": lambda v: len(v) > 1,
            ... }
            >>> result = u.Mapper.find_callable(predicates, [1, 2])
            >>> assert result == "is_multiple"

        """

        def build_predicate_call(predicate_fn: _Predicate[T]) -> Callable[[], bool]:
            def _call_predicate() -> bool:
                return predicate_fn(value)

            return _call_predicate

        for name, predicate in callables.items():
            predicate_result = r[bool].create_from_callable(
                build_predicate_call(predicate),
            )
            if predicate_result.is_success and predicate_result.value:
                return name
        return None


__all__ = [
    "FlextUtilitiesMapper",
]
