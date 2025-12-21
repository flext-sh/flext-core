"""Utilities module - FlextUtilitiesMapper.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeGuard, cast, overload

from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextUtilitiesMapper:
    """Data structure mapping and transformation utilities.

    Provides generic methods for mapping between data structures, building
    objects from flags/mappings, and transforming dict/list structures.

    **Preferred Usage**:
        Use `u.mapper()` to get an instance:
        >>> from flext_core.utilities import u
        >>> mapper = u.mapper()
        >>> result = mapper.get(data, "key", default="")

    **Alternative Usage**:
        Access via class attribute (still supported):
        >>> from flext_core.utilities import u
        >>> result = u.Mapper.get(data, "key", default="")

    **Usage Examples**:
    >>> # Map dict keys
    >>> mapping = {"old_key": "new_key", "foo": "bar"}
    >>> result = u.mapper().map_dict_keys({"old_key": "value", "foo": "baz"}, mapping)
    >>> new_dict = result.value  # {"new_key": "value", "bar": "baz"}

    >>> # Build object from flags
    >>> flags = ["read", "write"]
    >>> mapping = {"read": "can_read", "write": "can_write"}
    >>> result = u.mapper().build_flags_dict(flags, mapping)
    >>> perms = result.value  # {"can_read": True, "can_write": True, ...}
    """

    # =========================================================================
    # TYPE GUARDS AND HELPERS - Replace casts with proper type narrowing
    # =========================================================================

    @staticmethod
    def _is_configuration_dict(value: object) -> TypeGuard[t.ConfigurationDict]:
        """Type guard for ConfigurationDict (dict with str keys and t.GeneralValueType values)."""
        if not isinstance(value, dict):
            return False
        # Explicitly type the dict first to help pyright infer key types
        value_dict: dict[object, object] = value
        keys: list[object] = list(value_dict.keys())
        return all(isinstance(k, str) for k in keys)

    @staticmethod
    def _is_configuration_mapping(
        value: object,
    ) -> TypeGuard[t.ConfigurationMapping]:
        """Type guard for ConfigurationMapping."""
        if not isinstance(value, Mapping):
            return False
        # Explicitly type keys from Mapping.keys()
        keys: list[object] = list(value.keys())
        return all(isinstance(k, str) for k in keys)

    @staticmethod
    def _narrow_to_configuration_dict(value: object) -> t.ConfigurationDict:
        """Safely narrow object to ConfigurationDict with runtime validation."""
        if not isinstance(value, dict):
            error_msg = f"Cannot narrow {type(value)} to ConfigurationDict"
            raise TypeError(error_msg)
        # Explicit cast after isinstance - helps Pyright track type narrowing
        dict_typed: dict[str, object] = value
        # Verify keys are strings
        keys: list[object] = list(dict_typed.keys())
        if not all(isinstance(k, str) for k in keys):
            error_msg = f"Cannot narrow {type(value)} to ConfigurationDict - non-string keys found"
            raise TypeError(error_msg)
        # Type narrowing: dict is confirmed, keys are strings
        return value

    @staticmethod
    def _narrow_to_configuration_mapping(value: object) -> t.ConfigurationMapping:
        """Safely narrow object to ConfigurationMapping with runtime validation."""
        if not isinstance(value, Mapping):
            error_msg = f"Cannot narrow {type(value)} to ConfigurationMapping"
            raise TypeError(error_msg)
        # Explicit cast after isinstance - helps Pyright track type narrowing
        mapping_typed: Mapping[str, object] = value
        # Verify keys are strings
        keys: list[object] = list(mapping_typed.keys())
        if not all(isinstance(k, str) for k in keys):
            error_msg = f"Cannot narrow {type(value)} to ConfigurationMapping - non-string keys found"
            raise TypeError(error_msg)
        # Type narrowing: Mapping is confirmed, keys are strings
        return value

    @staticmethod
    def _narrow_to_sequence(value: object) -> Sequence[object]:
        """Safely narrow object to Sequence[object] with runtime validation."""
        if isinstance(value, (list, tuple)):
            return cast(
                "Sequence[object]", value
            )  # Type narrowing via isinstance + cast
        error_msg = f"Cannot narrow {type(value)} to Sequence"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_general_value_type(value: object) -> t.GeneralValueType:
        """Safely narrow object to t.GeneralValueType.

        Uses TypeGuard-based validation to ensure type safety.
        If value is not a valid t.GeneralValueType, returns string representation.
        """
        # Use TypeGuard for proper type narrowing
        if FlextUtilitiesGuards.is_general_value_type(value):
            return value
        # Fallback: convert to string (str is a valid t.GeneralValueType)
        return str(value)

    @staticmethod
    def _get_str_from_dict(
        ops: t.ConfigurationDict, key: str, default: str = ""
    ) -> str:
        """Safely extract str value from ConfigurationDict."""
        value = ops.get(key, default)
        if isinstance(value, str):
            return value
        return str(value) if value is not None else default

    @staticmethod
    def _get_callable_from_dict(
        ops: t.ConfigurationDict, key: str
    ) -> Callable[[t.GeneralValueType], t.GeneralValueType] | None:
        """Safely extract Callable from ConfigurationDict.

        Returns callable that accepts t.GeneralValueType (from lower layer) since
        ConfigurationDict values are t.GeneralValueType.
        """
        value = ops.get(key)
        if callable(value):
            return value
        return None

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance (Logger protocol).
        Type annotation omitted to avoid importing structlog.typing here.
        """
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def map_dict_keys(
        source: t.ConfigurationDict,
        key_mapping: t.StringMapping,
        *,
        keep_unmapped: bool = True,
    ) -> r[t.ConfigurationDict]:
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
        try:
            result: t.ConfigurationDict = {}

            for key, value in source.items():
                new_key = key_mapping.get(key)
                if new_key:
                    result[new_key] = value
                elif keep_unmapped:
                    result[key] = value

            return r[t.ConfigurationDict].ok(result)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[t.ConfigurationDict].fail(f"Failed to map dict keys: {e}")

    @staticmethod
    def build_flags_dict(
        active_flags: list[str],
        flag_mapping: t.StringMapping,
        *,
        default_value: bool = False,
    ) -> r[t.StringBoolDict]:
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
        try:
            result: t.StringBoolDict = {}

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
        source: t.StringBoolMapping,
        key_mapping: t.StringMapping,
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
        source: t.ConfigurationDict,
        transformer: Callable[
            [t.GeneralValueType],
            t.GeneralValueType,
        ],
    ) -> t.ConfigurationDict:
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
        source: t.ConfigurationDict,
        predicate: Callable[[str, t.GeneralValueType], bool],
    ) -> t.ConfigurationDict:
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
        source: t.StringMapping,
        *,
        handle_collisions: str = "last",
    ) -> t.StringMapping:
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
            result: t.StringDict = {}
            for k, v in source.items():
                if v not in result:
                    result[v] = k
            return result
        # last
        return {v: k for k, v in source.items()}

    @staticmethod
    def is_json_primitive(value: t.GeneralValueType) -> bool:
        """Check if value is a JSON primitive type (str, int, float, bool, None)."""
        return FlextUtilitiesGuards.is_type(value, (str, int, float, bool, type(None)))

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
            >>> FlextUtilitiesMapper.convert_to_json_value({"a": 1})
            {'a': 1}
            >>> FlextUtilitiesMapper.convert_to_json_value([1, 2, "three"])
            [1, 2, 'three']

        """
        if cls.is_json_primitive(value):
            return value
        # Use isinstance for type narrowing (is_type() doesn't return TypeGuard)
        if isinstance(value, dict):
            # Type narrowing: value is dict after isinstance check
            # Use helper for safe narrowing to ConfigurationDict
            value_dict: t.ConfigurationDict = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(value)
            )
            return {
                str(k): FlextUtilitiesMapper.convert_to_json_value(v)
                for k, v in value_dict.items()
            }
        # Use isinstance for sequence type narrowing
        if isinstance(value, Sequence) and not isinstance(value, str):
            # NOTE: Cannot use u.map() here due to circular import
            # (utilities.py -> mapper.py)
            # Type narrowing: value is Sequence[object] after _is_sequence check
            # Cast to Sequence[t.GeneralValueType] for iteration
            value_seq: Sequence[t.GeneralValueType] = value
            return [cls.convert_to_json_value(item) for item in value_seq]
        # Fallback: convert to string
        return str(value)

    @classmethod
    def convert_dict_to_json(
        cls,
        data: t.ConfigurationDict,
    ) -> t.ConfigurationDict:
        """Convert dict with any values to JSON-compatible dict.

        **Generic replacement for**: Manual dict-to-JSON conversion loops

        Args:
            data: Source dictionary with any values

        Returns:
            Dictionary with all values converted to JSON-compatible types

        Example:
            >>> data = {"name": "test", "value": CustomObject()}
            >>> result = FlextUtilitiesMapper.convert_dict_to_json(data)
            >>> # {"name": "test", "value": "str(CustomObject())"}

        """
        # ConfigurationDict keys are always str, so isinstance is redundant
        return {key: cls.convert_to_json_value(value) for key, value in data.items()}

    @classmethod
    def convert_list_to_json(
        cls,
        data: Sequence[t.GeneralValueType],
    ) -> list[t.ConfigurationDict]:
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
        # Use list comprehension for better performance
        return [
            FlextUtilitiesMapper.convert_dict_to_json(
                FlextUtilitiesMapper._narrow_to_configuration_dict(item)
            )
            for item in data
            if isinstance(item, dict)
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
            return value
        return str(value)

    @staticmethod
    def ensure(
        value: t.GeneralValueType,
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
        current: object,
        key_part: str,
    ) -> tuple[object, bool]:
        """Helper: Get raw value from dict/object/model."""
        # Handle Mapping (dict)
        # Use isinstance for type narrowing
        if isinstance(current, Mapping):
            # Type narrowing: current is Mapping after isinstance check
            # Use helper for safe narrowing to ConfigurationMapping
            current_mapping: t.ConfigurationMapping = (
                FlextUtilitiesMapper._narrow_to_configuration_mapping(current)
            )
            if key_part in current_mapping:
                return current_mapping[key_part], True
            return None, False

        # Handle object attribute
        if hasattr(current, key_part):
            return getattr(current, key_part), True

        # Handle Pydantic model
        if hasattr(current, "model_dump"):
            model_dump_attr = getattr(current, "model_dump", None)
            if callable(model_dump_attr):
                model_dict = model_dump_attr()
                # Use isinstance for type narrowing and combine conditions
                if isinstance(model_dict, dict) and key_part in model_dict:
                    # Type narrowing: model_dict is dict after isinstance check
                    # model_dump() returns dict[str, t.GeneralValueType] which is ConfigurationDict
                    model_dict_typed: t.ConfigurationDict = (
                        FlextUtilitiesMapper._narrow_to_configuration_dict(model_dict)
                    )
                    return model_dict_typed[key_part], True

        return None, False

    @staticmethod
    def _extract_handle_array_index(
        current: object,
        array_match: str,
    ) -> tuple[object, str | None]:
        """Helper: Handle array indexing with support for negative indices."""
        # Use isinstance for type narrowing
        if not isinstance(current, (list, tuple)):
            return None, "Not a sequence"
        # Type narrowing: current is Sequence[object] after isinstance check
        sequence: Sequence[object] = FlextUtilitiesMapper._narrow_to_sequence(current)
        try:
            idx = int(array_match)
            # Support negative indices (Python-style: -1 is last element)
            if idx < 0:
                idx = len(sequence) + idx
            if 0 <= idx < len(sequence):
                return sequence[idx], None
            return None, f"Index {int(array_match)} out of range"
        except (ValueError, IndexError):
            return None, f"Invalid index {array_match}"

    @staticmethod
    def extract[T](
        data: t.ConfigurationMapping | object,
        path: str,
        *,
        default: T | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> r[T | None]:
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
            config = {"database": {"host": "localhost", "port": 5432}}
            result = FlextUtilitiesMapper.extract(config, "database.port")
            # → r.ok(5432)

        """
        try:
            parts = path.split(separator)
            # Use t.GeneralValueType internally instead of object for better type safety
            # ConfigurationMapping values are t.GeneralValueType, and object attributes
            # extracted will be narrowed to t.GeneralValueType
            current: t.GeneralValueType | object = data

            for i, part in enumerate(parts):
                if current is None:
                    # Business rule: Cannot traverse None in path
                    # Return default if provided, otherwise fail
                    if required:
                        return r[T | None].fail(
                            f"Path '{separator.join(parts[:i])}' is None"
                        )
                    if default is None:
                        return r[T | None].fail(
                            f"Path '{separator.join(parts[:i])}' is None and default is None"
                        )
                    return r[T | None].ok(default)

                # Handle array indexing (e.g., "items[0]")
                key_part, array_match = FlextUtilitiesMapper._extract_parse_array_index(
                    part,
                )

                # Get value from dict, object, or Pydantic model
                value, found = FlextUtilitiesMapper._extract_get_value(
                    current,
                    key_part,
                )

                if not found:
                    path_context = separator.join(parts[:i])
                    if required:
                        return r[T | None].fail(
                            f"Key '{key_part}' not found at '{path_context}'"
                        )
                    if default is None:
                        return r[T | None].fail(
                            f"Key '{key_part}' not found at '{path_context}' and default is None"
                        )
                    return r[T | None].ok(default)

                current = value

                # Handle array index
                if array_match is not None:
                    value, error = FlextUtilitiesMapper._extract_handle_array_index(
                        current,
                        array_match,
                    )
                    if error:
                        if required:
                            return r[T | None].fail(
                                f"Array error at '{key_part}': {error}"
                            )
                        if default is None:
                            return r[T | None].fail(
                                f"Array error at '{key_part}': {error} and default is None"
                            )
                        return r[T | None].ok(default)
                    current = value

            if current is None:
                if required:
                    return r[T | None].fail("Extracted value is None")
                if default is None:
                    return r[T | None].fail(
                        "Extracted value is None and default is None"
                    )
                return r[T | None].ok(default)

            # Type narrowing: current is T after successful extraction
            # Use t.GeneralValueType from lower layer for proper type narrowing
            # Since we extract from ConfigurationMapping (dict[str, t.GeneralValueType]),
            # the extracted value is always t.GeneralValueType compatible
            # For generic T, we rely on type compatibility: T must be compatible with
            # t.GeneralValueType when extracting from ConfigurationMapping
            current_as_general: t.GeneralValueType = (
                FlextUtilitiesMapper._narrow_to_general_value_type(current)
            )
            # Cast to T | None for generic compatibility
            return r[T | None].ok(current_as_general)  # type: ignore[arg-type]

        except Exception as e:
            return r[T | None].fail(f"Extract failed: {e}")

    # =========================================================================
    # GET METHODS - Unified get function for dict/object access
    # =========================================================================

    @staticmethod
    @overload
    def get(
        data: t.ConfigurationMapping | object,
        key: str,
    ) -> t.GeneralValueType | None: ...

    @staticmethod
    @overload
    def get(
        data: t.ConfigurationMapping | object,
        key: str,
        *,
        default: str,
    ) -> str: ...

    @staticmethod
    @overload
    def get[T](
        data: t.ConfigurationMapping | object,
        key: str,
        *,
        default: list[T],
    ) -> list[T]: ...

    @staticmethod
    @overload
    def get[T](
        data: t.ConfigurationMapping | object,
        key: str,
        *,
        default: T,
    ) -> T: ...

    @staticmethod
    def get[T](
        data: t.ConfigurationMapping | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None:
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
    def _get_raw[T](
        data: t.ConfigurationMapping | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None:
        """Internal helper for raw get without DSL conversion."""
        match data:
            case dict() | Mapping():
                # Type narrowing: data is dict or Mapping
                # Use helper for safe narrowing to ConfigurationMapping
                data_mapping: t.ConfigurationMapping = (
                    FlextUtilitiesMapper._narrow_to_configuration_mapping(data)
                )
                # Get value from mapping, use default if key not found
                raw_value = data_mapping.get(key)
                if raw_value is None:
                    return default
                # Type narrowing: result is T if it matches the expected type
                # Since T is generic, return the raw value and let caller handle type
                if FlextUtilitiesGuards.is_general_value_type(raw_value):
                    # Safe to return as T | None since t.GeneralValueType is compatible
                    # TypeGuard ensures type narrowing to compatible type
                    return raw_value  # type: ignore[return-value]
                # Raw value not compatible with T, return default
                return default
            case _:
                return getattr(data, key, default)

    @staticmethod
    def prop(key: str) -> Callable[[object], object]:
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
            obj: Mapping[str, t.GeneralValueType] | object,
        ) -> object:
            """Access property from object."""
            result = FlextUtilitiesMapper.get(obj, key)
            # get() returns T | None, but for prop() we return t.GeneralValueType
            # This is safe because get() extracts from ConfigurationMapping or object attributes
            return result if result is not None else None

        return accessor

    # =========================================================================
    # AT/TAKE/PICK/AS_ METHODS - Unified access functions
    # =========================================================================

    @staticmethod
    def at[T](
        items: list[T] | tuple[T, ...] | dict[str, T],
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
            # Use isinstance for type narrowing
            if isinstance(items, dict):
                # Type narrowing: items is dict after isinstance check
                # Type inference: isinstance provides type narrowing to dict[str, T]
                items_dict: dict[str, T] = items
                if isinstance(index, str):
                    # Type narrowing: index is str after isinstance check
                    return items_dict.get(index, default)
                return default
            # Conditional access for list/tuple
            # Type narrowing: index is int after isinstance check
            # Type narrowing: items is Sequence[T] for list/tuple
            # list[T] and tuple[T, ...] are both Sequence[T], so direct access works
            if isinstance(index, int) and 0 <= index < len(items):
                return items[index]
            return default
        except (IndexError, KeyError, TypeError):
            return default

    @staticmethod
    @overload
    def take[T](
        data_or_items: Mapping[str, object] | object,
        key_or_n: str,
        *,
        as_type: type[T] | None = None,
        default: T | None = None,
        from_start: bool = True,
    ) -> T | None: ...

    @staticmethod
    @overload
    def take[T](
        data_or_items: dict[str, T],
        key_or_n: int,
        *,
        as_type: type[T] | None = None,
        default: T | None = None,
        from_start: bool = True,
    ) -> dict[str, T]: ...

    @staticmethod
    @overload
    def take[T](
        data_or_items: list[T] | tuple[T, ...],
        key_or_n: int,
        *,
        as_type: type[T] | None = None,
        default: T | None = None,
        from_start: bool = True,
    ) -> list[T]: ...

    @staticmethod
    def take[T](
        data_or_items: Mapping[str, object]
        | object
        | dict[str, T]
        | list[T]
        | tuple[T, ...],
        key_or_n: str | int,
        *,
        as_type: type[T] | None = None,
        default: T | None = None,
        from_start: bool = True,
    ) -> dict[str, T] | list[T] | T | None:
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
            data: Mapping[str, object] | object = data_or_items
            key = key_or_n
            value = FlextUtilitiesMapper.get(data, key, default=default)
            if value is None:
                return default
            if as_type is not None and not isinstance(value, as_type):
                return default
            return value

        # Slice mode: take N items from list/dict
        n = key_or_n
        # Use isinstance for type narrowing
        if isinstance(data_or_items, dict):
            # Type narrowing: data_or_items is dict after isinstance check
            # Extract keys from dict - keys are always str since it's a dict
            keys = list(data_or_items.keys())
            selected_keys = keys[:n] if from_start else keys[-n:]
            # Use dict comprehension directly with type-safe access
            return {k: data_or_items[k] for k in selected_keys}
        # Remaining cases: list or tuple
        if isinstance(data_or_items, (list, tuple)):
            # Type narrowing: data_or_items is Sequence after isinstance check
            # list[T] and tuple[T, ...] are both Sequence[T], so direct conversion works
            items_list: list[T] = list(data_or_items)
            return items_list[:n] if from_start else items_list[-n:]
        # Fallback for object case: wrap single item in list
        # Cast to T for generic compatibility
        items_list_obj: list[T] = [data_or_items]  # type: ignore[list-item,arg-type]
        return items_list_obj[:n] if from_start else items_list_obj[-n:]

    @staticmethod
    def pick(
        data: t.ConfigurationMapping | object,
        *keys: str,
        as_dict: bool = True,
    ) -> t.ConfigurationDict | list[object]:
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
        value: object,
        target: type[int],
        *,
        default: int | None = None,
        strict: bool = False,
    ) -> int | None: ...

    @staticmethod
    @overload
    def as_(
        value: object,
        target: type[float],
        *,
        default: float | None = None,
        strict: bool = False,
    ) -> float | None: ...

    @staticmethod
    @overload
    def as_[T](
        value: object,
        target: type[T],
        *,
        default: T | None = None,
        strict: bool = False,
    ) -> T | None: ...

    @staticmethod
    def as_[T](
        value: object,
        target: type[T],
        *,
        default: T | None = None,
        strict: bool = False,
    ) -> T | None:
        """Type conversion with guard (mnemonic: as_ = convert to type).

        Generic replacement for: isinstance() +  patterns

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
        if isinstance(value, target):
            return value  # isinstance narrows type to T
        if strict:
            return default
        # Try coercion for basic types using proper type narrowing
        # When target is a specific type, we know T is that type, so conversions are type-safe
        try:
            if target is int and isinstance(value, (str, float)):
                # Type narrowing: target is int, so T is int, and int(value) is int
                return int(value)  # type: ignore[return-value]
            if target is float and isinstance(value, (str, int)):
                # Type narrowing: target is float, so T is float, and float(value) is float
                return float(value)  # type: ignore[return-value]
            if target is str:
                # Type narrowing: target is str, so T is str, and str(value) is str
                return str(value)  # type: ignore[return-value]
            if target is bool and isinstance(value, str):
                normalized = value.lower()
                if normalized in {"true", "1", "yes", "on"}:
                    # Type narrowing: target is bool, so T is bool, and True is bool
                    return True  # type: ignore[return-value]
                if normalized in {"false", "0", "no", "off"}:
                    # Type narrowing: target is bool, so T is bool, and False is bool
                    return False  # type: ignore[return-value]
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
        item: object,
        field_name: str,
    ) -> object | None:
        """Extract field value from item (dict or object).

        Helper method to improve type inference for pyrefly.
        """
        if isinstance(item, dict):
            # Type narrowing: item is dict at this point
            # Use ConfigurationDict which is dict[str, t.GeneralValueType]
            dict_item: t.ConfigurationDict = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(item)
            )
            return dict_item.get(field_name)
        if hasattr(item, field_name):
            # getattr returns object since we know the attribute exists
            attr_value: object = getattr(item, field_name)
            return attr_value
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
                if isinstance(val_raw, (int, float)):
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
    def _build_apply_ensure(current: object, ops: t.ConfigurationDict) -> object:
        """Helper: Apply ensure operation."""
        if "ensure" not in ops:
            return current
        ensure_type = FlextUtilitiesMapper._get_str_from_dict(ops, "ensure", "")
        ensure_default_val = ops.get("ensure_default")
        # Default values by type
        default_map: t.ConfigurationDict = {
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
                    # Type narrowing: current is list, convert items to t.GeneralValueType
                    list_current: list[object] = current
                    return [
                        FlextUtilitiesMapper._narrow_to_general_value_type(item)
                        for item in list_current
                    ]
                return (
                    default_val
                    if current is None
                    else [FlextUtilitiesMapper._narrow_to_general_value_type(current)]
                )
            case "str_list":
                if isinstance(current, list):
                    # Type narrowing: current is list[object], convert each to str
                    list_current_str: list[object] = current
                    return [
                        str(FlextUtilitiesMapper._narrow_to_general_value_type(x))
                        for x in list_current_str
                    ]
                return (
                    default_val
                    if current is None
                    else [
                        str(FlextUtilitiesMapper._narrow_to_general_value_type(current))
                    ]
                )
            case "dict":
                if isinstance(current, dict):
                    # Type narrowing: current is dict, return as ConfigurationDict
                    return FlextUtilitiesMapper._narrow_to_configuration_dict(current)
                return default_val
            case _:
                return current

    @staticmethod
    def _build_apply_filter(
        current: object,
        ops: t.ConfigurationDict,
        default: object,
    ) -> object:
        """Helper: Apply filter operation."""
        if "filter" not in ops:
            return current
        filter_pred_raw = FlextUtilitiesMapper._get_callable_from_dict(ops, "filter")
        if filter_pred_raw is None:
            return current
        # Cast the callable to proper filter predicate type
        filter_pred: Callable[[object], bool] = cast(
            "Callable[[object], bool]", filter_pred_raw
        )
        # Handle collections
        if isinstance(current, (list, tuple)):
            # Type narrowing: current is Sequence[object], x is t.GeneralValueType
            seq_current: Sequence[object] = current
            return [
                FlextUtilitiesMapper._narrow_to_general_value_type(x)
                for x in seq_current
                if filter_pred(x)
            ]
        if isinstance(current, dict):
            # Type narrowing: current is dict, use ConfigurationDict
            current_dict: t.ConfigurationDict = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(current)
            )
            # Use filter_dict for consistency
            return FlextUtilitiesMapper.filter_dict(
                current_dict,
                lambda _k, v: filter_pred(v),
            )
        # Single value
        return default if not filter_pred(current) else current

    @staticmethod
    def _build_apply_map(current: object, ops: t.ConfigurationDict) -> object:
        """Helper: Apply map operation."""
        if "map" not in ops:
            return current
        map_func_raw = FlextUtilitiesMapper._get_callable_from_dict(ops, "map")
        if map_func_raw is None:
            return current
        # map_func accepts t.GeneralValueType since we work with ConfigurationDict values
        map_func: Callable[[t.GeneralValueType], t.GeneralValueType] = map_func_raw
        if isinstance(current, (list, tuple)):
            # Type narrowing: current is Sequence, items are t.GeneralValueType
            seq_current: Sequence[object] = current
            return [
                map_func(FlextUtilitiesMapper._narrow_to_general_value_type(x))
                for x in seq_current
            ]
        if isinstance(current, dict):
            # Type narrowing: current is dict, use ConfigurationDict
            current_dict: t.ConfigurationDict = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(current)
            )
            # ConfigurationDict values are t.GeneralValueType, so map_func works directly
            return {k: map_func(v) for k, v in current_dict.items()}
        # Single value case - narrow to t.GeneralValueType before mapping
        current_general = FlextUtilitiesMapper._narrow_to_general_value_type(current)
        return map_func(current_general)

    @staticmethod
    def _build_apply_normalize(
        current: object,
        ops: t.ConfigurationDict,
    ) -> object:
        """Helper: Apply normalize operation."""
        if "normalize" not in ops:
            return current
        normalize_case = FlextUtilitiesMapper._get_str_from_dict(ops, "normalize", "")
        if isinstance(current, str):
            return current.lower() if normalize_case == "lower" else current.upper()
        if isinstance(current, (list, tuple)):
            # Type narrowing: current is Sequence, items are t.GeneralValueType
            seq_current: Sequence[object] = current
            result: list[t.GeneralValueType] = []
            for x in seq_current:
                x_general = FlextUtilitiesMapper._narrow_to_general_value_type(x)
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
    def _build_apply_convert(current: object, ops: t.ConfigurationDict) -> object:
        """Helper: Apply convert operation."""
        if "convert" not in ops:
            return current
        convert_type = ops["convert"]
        if not isinstance(convert_type, type):
            return current
        convert_default = ops.get("convert_default", convert_type())
        try:
            return convert_type(current)
        except (ValueError, TypeError):
            return convert_default

    @staticmethod
    def _extract_transform_options(
        transform_opts: t.ConfigurationDict,
    ) -> tuple[
        bool,
        bool,
        bool,
        t.StringMapping | None,
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
        if isinstance(map_keys_val, dict) and all(
            isinstance(v, str) for v in map_keys_val.values()
        ):
            # Runtime check ensures all values are str, so cast to StringMapping is safe
            map_keys_dict: t.StringMapping | None = cast(
                "t.StringMapping", map_keys_val
            )
        else:
            map_keys_dict = None
        filter_keys_val = transform_opts.get("filter_keys")
        filter_keys_set: set[str] | None = (
            filter_keys_val if isinstance(filter_keys_val, set) else None
        )
        exclude_keys_val = transform_opts.get("exclude_keys")
        exclude_keys_set: set[str] | None = (
            exclude_keys_val if isinstance(exclude_keys_val, set) else None
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
        result: t.ConfigurationDict,
        *,
        normalize: bool,
    ) -> t.ConfigurationDict:
        """Apply normalize step."""
        if normalize:
            normalized = FlextUtilitiesCache.normalize_component(result)
            if isinstance(normalized, dict):
                return normalized
        return result

    @staticmethod
    def _apply_map_keys(
        result: t.ConfigurationDict,
        *,
        map_keys: t.StringMapping | None,
    ) -> t.ConfigurationDict:
        """Apply map keys step."""
        if map_keys:
            mapped = FlextUtilitiesMapper.map_dict_keys(result, map_keys)
            if mapped.is_success:
                return mapped.value
        return result

    @staticmethod
    def _apply_filter_keys(
        result: t.ConfigurationDict,
        *,
        filter_keys: set[str] | None,
    ) -> t.ConfigurationDict:
        """Apply filter keys step."""
        if filter_keys:
            filtered_dict: t.ConfigurationDict = {}
            for key in filter_keys:
                if key in result:
                    filtered_dict[key] = result[key]
            return filtered_dict
        return result

    @staticmethod
    def _apply_exclude_keys(
        result: t.ConfigurationDict,
        *,
        exclude_keys: set[str] | None,
    ) -> t.ConfigurationDict:
        """Apply exclude keys step."""
        if exclude_keys:
            for key in exclude_keys:
                _ = result.pop(key, None)  # Result intentionally unused
        return result

    @staticmethod
    def _apply_strip_none(
        result: t.ConfigurationDict,
        *,
        strip_none: bool,
    ) -> t.ConfigurationDict:
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
        result: t.ConfigurationDict,
        *,
        strip_empty: bool,
    ) -> t.ConfigurationDict:
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
        result: t.ConfigurationDict,
        *,
        to_json: bool,
    ) -> t.ConfigurationDict:
        """Apply to JSON step."""
        if to_json:
            return FlextUtilitiesMapper.convert_dict_to_json(result)
        return result

    @staticmethod
    def _apply_transform_steps(
        result: t.ConfigurationDict,
        *,
        normalize: bool,
        map_keys: t.StringMapping | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
        to_json: bool,
    ) -> t.ConfigurationDict:
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
    def _build_apply_transform(
        current: object,
        ops: t.ConfigurationDict,
        default: object,
        on_error: str,
    ) -> object:
        """Helper: Apply transform operation."""
        if "transform" not in ops or not FlextUtilitiesGuards.is_type(
            current, "mapping"
        ):
            return current
        transform_opts_raw: t.GeneralValueType = ops["transform"]
        if not isinstance(transform_opts_raw, dict):
            return current
        # transform_opts_raw is dict after isinstance check
        # Type narrowing: transform_opts_raw is dict[str, t.GeneralValueType]
        transform_opts: t.ConfigurationDict = transform_opts_raw
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
        current_dict: t.ConfigurationMapping = (
            FlextUtilitiesMapper._narrow_to_configuration_mapping(current)
        )
        # Implement transform logic directly using available utilities
        try:
            # Type narrowing: current_dict is ConfigurationMapping, dict() returns ConfigurationDict
            # dict() constructor returns dict[str, t.GeneralValueType] which is ConfigurationDict
            # Type narrowing: dict() returns ConfigurationDict
            result: t.ConfigurationDict = dict(current_dict)
            return FlextUtilitiesMapper._apply_transform_steps(
                result,
                normalize=normalize_bool,
                map_keys=map_keys_dict,
                filter_keys=filter_keys_set,
                exclude_keys=exclude_keys_set,
                strip_none=strip_none_bool,
                strip_empty=strip_empty_bool,
                to_json=to_json_bool,
            )
        except Exception:
            if on_error == "stop":
                return default
            return current

    @staticmethod
    def _build_apply_process(
        current: object,
        ops: t.ConfigurationDict,
        default: object,
        on_error: str,
    ) -> object:
        """Helper: Apply process operation."""
        if "process" not in ops:
            return current
        process_func_raw = FlextUtilitiesMapper._get_callable_from_dict(ops, "process")
        if process_func_raw is None:
            return current
        # process_func accepts t.GeneralValueType since we work with ConfigurationDict values
        process_func: Callable[[t.GeneralValueType], t.GeneralValueType] = (
            process_func_raw
        )
        try:
            if isinstance(current, (list, tuple)):
                # Type narrowing: current is Sequence, items are t.GeneralValueType
                seq_current: Sequence[object] = current
                return [
                    process_func(FlextUtilitiesMapper._narrow_to_general_value_type(x))
                    for x in seq_current
                ]
            if isinstance(current, dict):
                # Type narrowing: current is dict, use ConfigurationDict
                current_dict: t.ConfigurationDict = (
                    FlextUtilitiesMapper._narrow_to_configuration_dict(current)
                )
                # ConfigurationDict values are t.GeneralValueType, so process_func works directly
                return {k: process_func(v) for k, v in current_dict.items()}
            # Single value case - narrow to t.GeneralValueType before processing
            current_general = FlextUtilitiesMapper._narrow_to_general_value_type(
                current
            )
            return process_func(current_general)
        except Exception:
            # Type annotation: result is object
            return default if on_error == "stop" else current

    @staticmethod
    def _build_apply_group(current: object, ops: t.ConfigurationDict) -> object:
        """Helper: Apply group operation."""
        if "group" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        group_spec = ops["group"]
        # Type narrowing: current is Sequence[object], convert to list[object]
        current_list: list[object] = list(current)
        # Group by field name (str) or key function (callable)
        if isinstance(group_spec, str):
            grouped: dict[t.GeneralValueType, list[object]] = {}
            for item in current_list:
                # FlextUtilitiesMapper.get returns str, so key is always str
                key_raw = FlextUtilitiesMapper.get(item, group_spec)
                if key_raw is None:
                    continue
                key: str = str(key_raw)
                key_typed: t.GeneralValueType = (
                    FlextUtilitiesMapper._narrow_to_general_value_type(key)
                )
                if key_typed not in grouped:
                    grouped[key_typed] = []
                grouped[key_typed].append(item)
            return grouped
        if callable(group_spec):
            grouped_by_func: dict[t.GeneralValueType, list[object]] = {}
            for item in current_list:
                # group_spec is callable, return type is unknown, so narrow to t.GeneralValueType
                item_key_raw = group_spec(item)
                item_key: t.GeneralValueType = (
                    FlextUtilitiesMapper._narrow_to_general_value_type(item_key_raw)
                )
                item_key_typed: t.GeneralValueType = (
                    item_key
                    if item_key is not None
                    else FlextUtilitiesMapper._narrow_to_general_value_type("")
                )
                if item_key_typed not in grouped_by_func:
                    grouped_by_func[item_key_typed] = []
                grouped_by_func[item_key_typed].append(item)
            # Type annotation: return dict with list[object] values
            return grouped_by_func
        return current

    @staticmethod
    def _build_apply_sort(current: object, ops: t.ConfigurationDict) -> object:
        """Helper: Apply sort operation."""
        if "sort" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        sort_spec = ops["sort"]
        # Type narrowing: current is Sequence[object], convert to list[object]
        current_list: list[object] = list(current)
        if isinstance(sort_spec, str):
            # Sort by field name
            field_name: str = sort_spec

            def key_func(item: object) -> str:
                # get() with default="" returns str (never None), so is not None check is redundant
                result = FlextUtilitiesMapper.get(item, field_name, default="")
                return str(result)

            sorted_list_key: list[object] = sorted(current_list, key=key_func)
            return (
                list(sorted_list_key)
                if isinstance(current, list)
                else tuple(sorted_list_key)
            )
        if callable(sort_spec):
            # sort_spec accepts t.GeneralValueType since items are t.GeneralValueType
            sort_fn: Callable[[t.GeneralValueType], t.GeneralValueType] = sort_spec

            def key_func_wrapper(item: t.GeneralValueType) -> str | int | float:
                # Type narrowing: item is t.GeneralValueType from list[t.GeneralValueType]
                key_value: t.GeneralValueType = sort_fn(item)
                # ScalarValue (str | int | float | bool | datetime | None) is part of t.GeneralValueType
                if isinstance(key_value, (str, int, float)):
                    return key_value
                return str(key_value)

            sorted_list_wrapper: list[object] = sorted(
                current_list,
                key=key_func_wrapper,
            )
            return (
                list(sorted_list_wrapper)
                if isinstance(current, list)
                else tuple(sorted_list_wrapper)
            )
        if sort_spec is True:
            # Natural sort - use t.GeneralValueType from lower layer
            # ScalarValue (str | int | float | bool | datetime | None) is part of t.GeneralValueType
            comparable_items: list[t.GeneralValueType] = [
                FlextUtilitiesMapper._narrow_to_general_value_type(
                    item
                    if isinstance(item, (str, int, float, bool, type(None)))
                    else str(item)
                )
                for item in current_list
            ]
            sorted_comparable: list[t.GeneralValueType] = sorted(
                comparable_items, key=str
            )
            # Return using t.GeneralValueType from lower layer
            if isinstance(current, list):
                return sorted_comparable
            return tuple(sorted_comparable)
        return current

    @staticmethod
    def _build_apply_unique(current: object, ops: t.ConfigurationDict) -> object:
        """Helper: Apply unique operation to remove duplicates."""
        if "unique" not in ops or not ops.get("unique"):
            return current
        if not isinstance(current, (list, tuple)):
            return current
        # Remove duplicates while preserving order
        # Type narrowing: current is Sequence, convert to list using t.GeneralValueType
        current_list_unique: list[t.GeneralValueType] = [
            FlextUtilitiesMapper._narrow_to_general_value_type(item) for item in current
        ]
        seen: set[t.GeneralValueType | str] = set()
        unique_list: list[t.GeneralValueType] = []
        for item in current_list_unique:
            # Use t.GeneralValueType from lower layer - ScalarValue types are hashable
            item_hashable: t.GeneralValueType | str = (
                item
                if isinstance(item, (str, int, float, bool, type(None)))
                else str(item)
            )
            if item_hashable not in seen:
                seen.add(item_hashable)
                unique_list.append(item)
        if isinstance(current, list):
            return unique_list
        return tuple(unique_list)

    @staticmethod
    def _build_apply_slice(current: object, ops: t.ConfigurationDict) -> object:
        """Helper: Apply slice operation."""
        if "slice" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        slice_spec = ops["slice"]
        min_slice_length = 2
        if (
            isinstance(slice_spec, (tuple, list))
            and len(slice_spec) >= min_slice_length
        ):
            start_raw = slice_spec[0]
            end_raw = slice_spec[1]
            start: int | None = start_raw if isinstance(start_raw, int) else None
            end: int | None = end_raw if isinstance(end_raw, int) else None
            # Type narrowing: current is Sequence, slice returns same sequence type
            # Use t.GeneralValueType from lower layer for proper type safety
            if isinstance(current, list):
                sliced_list: list[t.GeneralValueType] = [
                    FlextUtilitiesMapper._narrow_to_general_value_type(item)
                    for item in current[start:end]
                ]
                return sliced_list
            # tuple case
            sliced_tuple: tuple[t.GeneralValueType, ...] = tuple(
                FlextUtilitiesMapper._narrow_to_general_value_type(item)
                for item in current[start:end]
            )
            return sliced_tuple
        return current

    @staticmethod
    def _build_apply_chunk(current: object, ops: t.ConfigurationDict) -> object:
        """Helper: Apply chunk operation to split into sublists."""
        if "chunk" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        chunk_size = ops["chunk"]
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            return current
        # Type narrowing: current is Sequence, convert to list using t.GeneralValueType
        current_list: list[t.GeneralValueType] = [
            FlextUtilitiesMapper._narrow_to_general_value_type(item) for item in current
        ]
        chunked: list[list[t.GeneralValueType]] = []
        for i in range(0, len(current_list), chunk_size):
            # Slice returns list[t.GeneralValueType]
            chunk: list[t.GeneralValueType] = current_list[i : i + chunk_size]
            chunked.append(chunk)
        # Return list of lists using t.GeneralValueType from lower layer
        return chunked

    @staticmethod
    def build[T](
        value: T,
        *,
        ops: t.ConfigurationDict | None = None,
        default: object = None,
        on_error: str = "stop",
    ) -> T | object:
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

        current: object = value
        default_val = default if default is not None else value

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
    def field[T](
        source: Mapping[str, object] | object,
        name: str,
        *,
        default: T | None = None,
        required: bool = False,
        ops: t.ConfigurationDict | None = None,
    ) -> T | None:
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
        value: T | None = FlextUtilitiesMapper.get(source, name, default=default)
        if value is None and required:
            return None
        if ops is not None:
            # Type narrowing: value is T | None, but ops requires value to be t.GeneralValueType
            value_for_build: t.GeneralValueType = (
                FlextUtilitiesMapper._narrow_to_general_value_type(value)
                if value is not None
                else FlextUtilitiesMapper._narrow_to_general_value_type("")
            )
            result = FlextUtilitiesMapper.build(
                value_for_build,
                ops=ops,
                on_error="stop",
            )
            # build returns r[T], use .value if success
            if isinstance(result, r) and result.is_success:
                # Use .value directly - FlextResult never returns None on success
                # Type annotation: result.value is T, but pyright needs explicit cast
                # Type narrowing: result is r[T] and is_success is True, so value is T
                # Use getattr to help pyright infer type
                result_value_raw = getattr(result, "value", None)
                if result_value_raw is None:
                    return None
                # Type narrowing: result.value is T when success, so result_value_raw is T
                result_value: T = result_value_raw
                return result_value
            return None
        # Type annotation: value is T | None
        # Type inference: value is already T | None, no cast needed
        return value

    @staticmethod
    def fields_multi(
        source: Mapping[str, object] | object,
        spec: t.ConfigurationDict,
    ) -> t.ConfigurationDict:
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
        result: t.ConfigurationDict = {}
        for field_name, field_default in spec.items():
            value: t.GeneralValueType = FlextUtilitiesMapper.get(
                source,
                field_name,
                default=field_default,
            )
            result[field_name] = value if value is not None else field_default
        return result

    @staticmethod
    def _fields_multi[T](
        source: Mapping[str, object] | object,
        spec: t.StringConfigurationDictDict | dict[str, T | None],
        *,
        on_error: str = "stop",
    ) -> dict[str, T | None] | r[dict[str, T]]:
        """Extract multiple fields using DSL mnemonic specification.

        Args:
            source: Source data (dict or object)
            spec: Field specification dict
            on_error: Error handling ("stop", "skip", "collect")

        Returns:
            dict[str, T | None] with extracted values

        """
        result: dict[str, T | None] = {}
        errors: list[str] = []

        for field_name, field_spec in spec.items():
            # Determine if field is required and get spec
            # Initialize variables to avoid mypy redefinition errors
            field_default: t.GeneralValueType | None
            field_ops: t.GeneralValueType | None
            field_required: bool

            if isinstance(field_spec, dict):
                # Type annotations for .get() results to help pyright inference
                # Type narrowing: isinstance provides type narrowing to ConfigurationDict
                field_spec_dict: t.ConfigurationDict = field_spec
                # Type inference: .get() returns t.GeneralValueType | None
                # Variables already declared above, just assign values
                # Use dict.get() directly - field_spec_dict is ConfigurationDict (dict subclass)
                # Access dict.get() via type to avoid confusion with FlextUtilitiesMapper.get()
                # Use Mapping.get() to avoid confusion with static method
                # field_spec_dict is already ConfigurationDict (dict subclass), use .get() directly
                # Access via dict.get() for cleaner code
                field_default_raw: t.GeneralValueType | None = field_spec_dict.get(
                    "default", None
                )
                field_ops_raw: t.GeneralValueType | None = field_spec_dict.get(
                    "ops", None
                )
                field_default = field_default_raw
                field_ops = field_ops_raw
                field_required = (
                    field_default is None and "default" not in field_spec_dict
                )
            else:
                field_default = (
                    FlextUtilitiesMapper._narrow_to_general_value_type(field_spec)
                    if field_spec is not None
                    else None
                )
                field_ops = None
                field_required = field_spec is None

            # Extract field
            # Type narrowing: field_default is t.GeneralValueType | None, but get() needs specific type
            # Use type narrowing - field_default is compatible with T | None
            field_default_typed: T | None = field_default  # type: ignore[assignment]
            # Use overload without type parameter - type inference will work from default
            value: T | None = FlextUtilitiesMapper.get(
                source, field_name, default=field_default_typed
            )
            if value is None and field_required:
                extracted: T | None = None
            elif field_ops is not None:
                if not isinstance(field_ops, dict):
                    extracted = None
                else:
                    field_ops_dict: t.ConfigurationDict = (
                        FlextUtilitiesMapper._narrow_to_configuration_dict(field_ops)
                    )
                    # Type narrowing: value is T | None, but build requires t.GeneralValueType
                    value_for_build: t.GeneralValueType = (
                        FlextUtilitiesMapper._narrow_to_general_value_type(value)
                        if value is not None
                        else FlextUtilitiesMapper._narrow_to_general_value_type("")
                    )
                    build_result = FlextUtilitiesMapper.build(
                        value_for_build,
                        ops=field_ops_dict,
                        on_error="stop",
                    )
                    # build_result is r[T], use .value if success
                    # Type annotation: build_result.value is T when success
                    # Type narrowing: build_result is r[T] and is_success is True, so value is T
                    build_result_value_raw = (
                        getattr(build_result, "value", None)
                        if isinstance(build_result, r) and build_result.is_success
                        else None
                    )
                    if build_result_value_raw is None:
                        extracted = None
                    else:
                        # Type narrowing: result.value is T when success
                        extracted = build_result_value_raw
            else:
                # Type annotation: value is T | None
                # Type inference: value is already T | None, no cast needed
                extracted = value

            if extracted is None and field_required:
                error_msg = f"Required field '{field_name}' is missing"
                if on_error == "stop":
                    return r[dict[str, T]].fail(error_msg)
                if on_error == "collect":
                    errors.append(error_msg)
                    continue
                continue

            result[field_name] = extracted

        if errors and on_error == "collect":
            return r[dict[str, T]].fail(
                f"Field extraction errors: {', '.join(errors)}",
            )

        return result

    # =========================================================================
    # CONSTRUCT METHOD - Object construction from spec
    # =========================================================================

    @staticmethod
    def construct(
        spec: t.ConfigurationDict,
        source: Mapping[str, object] | object | None = None,
        *,
        on_error: str = "stop",
    ) -> t.ConfigurationDict:
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
        constructed: t.ConfigurationDict = {}

        for target_key, target_spec in spec.items():
            try:
                # Literal value
                if isinstance(target_spec, dict) and "value" in target_spec:
                    constructed[target_key] = target_spec["value"]
                    continue

                # Field mapping
                if isinstance(target_spec, str):
                    source_field = target_spec
                    field_default = None
                    field_ops = None
                elif isinstance(target_spec, dict):
                    source_field_raw = target_spec.get("field", target_key)
                    source_field = (
                        str(source_field_raw)
                        if source_field_raw is not None
                        else target_key
                    )
                    field_default = target_spec.get("default")
                    field_ops = target_spec.get("ops")
                else:
                    # After isinstance checks, target_spec is t.GeneralValueType
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
                        field_ops_dict: t.ConfigurationDict = (
                            FlextUtilitiesMapper._narrow_to_configuration_dict(
                                field_ops
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

                # Narrow extracted to t.GeneralValueType for type safety
                final_value: t.GeneralValueType = (
                    FlextUtilitiesMapper._narrow_to_general_value_type(
                        extracted if extracted is not None else field_default
                    )
                )
                constructed[target_key] = final_value

            except Exception as e:
                error_msg = f"Failed to construct '{target_key}': {e}"
                if on_error == "stop":
                    raise ValueError(error_msg) from e
                if on_error == "skip":
                    continue

        return constructed

    @staticmethod
    def transform(
        source: t.ConfigurationDict | t.ConfigurationMapping,
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: t.StringMapping | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
        to_json: bool = False,
    ) -> r[t.ConfigurationDict]:
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
            >>> transformed = (
            ...     result.value if result.is_success else {}
            ... )  # {"new": "value"}

        """
        try:
            # ConfigurationMapping and ConfigurationDict are both Mapping, so isinstance is redundant
            source_dict: t.ConfigurationDict = dict(source)

            # Apply transform steps
            transformed = FlextUtilitiesMapper._apply_transform_steps(
                source_dict,
                normalize=normalize,
                map_keys=map_keys,
                filter_keys=filter_keys,
                exclude_keys=exclude_keys,
                strip_none=strip_none,
                strip_empty=strip_empty,
                to_json=to_json,
            )

            return r[t.ConfigurationDict].ok(transformed)
        except Exception as e:
            return r[t.ConfigurationDict].fail(f"Transform failed: {e}")

    @staticmethod
    def to_dict[T](mapping: Mapping[str, T]) -> dict[str, T]:
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
        a: t.ConfigurationDict,
        b: t.ConfigurationDict,
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

        # Recursive comparison
        for key in a:
            if key not in b:
                return False

            val_a = a[key]
            val_b = b[key]

            # None comparison
            if val_a is None and val_b is None:
                continue

            if val_a is None or val_b is None:
                return False

            # Nested dict comparison
            if isinstance(val_a, dict) and isinstance(val_b, dict):
                if not FlextUtilitiesMapper.deep_eq(val_a, val_b):
                    return False
                continue

            # List comparison (order matters)
            if isinstance(val_a, list) and isinstance(val_b, list):
                if len(val_a) != len(val_b):
                    return False
                for item_a, item_b in zip(val_a, val_b, strict=True):
                    if isinstance(item_a, dict) and isinstance(item_b, dict):
                        if not FlextUtilitiesMapper.deep_eq(item_a, item_b):
                            return False
                    elif item_a != item_b:
                        return False
                continue

            # Primitive comparison
            if val_a != val_b:
                return False

        return True


__all__ = [
    "FlextUtilitiesMapper",
]
