"""FLEXT Core Dict Helpers - Type-Safe Dictionary Operations.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Type-safe dictionary utilities that dramatically reduce boilerplate for common
dictionary operations while maintaining safety and providing clear error messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

T = TypeVar("T")
U = TypeVar("U")

# =============================================================================
# SAFE DICTIONARY ACCESS - Type-Safe Get Operations
# =============================================================================


class FlextDict:
    """Type-safe dictionary wrapper with fluent API.

    Provides safe access to dictionary values with automatic type conversion,
    validation, and error handling using FlextResult pattern.

    Example:
        data = {"name": "Alice", "age": "30", "active": True}
        fdict = FlextDict(data)

        name = fdict.get_str("name").unwrap()  # "Alice"
        age = fdict.get_int("age").unwrap()    # 30
        active = fdict.get_bool("active").unwrap()  # True
        missing = fdict.get_str("missing", "default")  # "default"
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize with optional data dictionary."""
        self._data = data or {}

    # -------------------------------------------------------------------------
    # Basic Access Methods
    # -------------------------------------------------------------------------

    def get_raw(self, key: str, default: Any = None) -> Any:
        """Get raw value without type conversion."""
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if key exists in dictionary."""
        return key in self._data

    def set(self, key: str, value: Any) -> FlextDict:
        """Set value and return self for chaining."""
        self._data[key] = value
        return self

    def remove(self, key: str) -> FlextResult[Any]:
        """Remove and return value."""
        if key not in self._data:
            return FlextResult.fail(f"Key '{key}' not found")
        value = self._data.pop(key)
        return FlextResult.ok(value)

    # -------------------------------------------------------------------------
    # Type-Safe Get Methods
    # -------------------------------------------------------------------------

    def get_str(self, key: str, default: str | None = None) -> FlextResult[str]:
        """Get string value with safe conversion."""
        if key not in self._data:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' not found")

        value = self._data[key]
        if value is None:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' is None")

        try:
            return FlextResult.ok(str(value))
        except Exception as e:  # noqa: BLE001
            return FlextResult.fail(f"Cannot convert '{key}' to string: {e}")

    def get_int(self, key: str, default: int | None = None) -> FlextResult[int]:
        """Get integer value with safe conversion."""
        if key not in self._data:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' not found")

        value = self._data[key]
        if value is None:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' is None")

        # Try direct conversion first
        if isinstance(value, int):
            return FlextResult.ok(value)

        # Try string to int conversion
        if isinstance(value, str):
            try:
                return FlextResult.ok(int(value))
            except ValueError:
                return FlextResult.fail(f"Cannot convert '{value}' to integer")

        # Try float to int
        if isinstance(value, float):
            if value.is_integer():
                return FlextResult.ok(int(value))
            return FlextResult.fail(f"Float '{value}' is not a whole number")

        return FlextResult.fail(f"Cannot convert {type(value)} to integer")

    def get_float(self, key: str, default: float | None = None) -> FlextResult[float]:
        """Get float value with safe conversion."""
        if key not in self._data:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' not found")

        value = self._data[key]
        if value is None:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' is None")

        try:
            return FlextResult.ok(float(value))
        except (ValueError, TypeError) as e:
            return FlextResult.fail(f"Cannot convert '{key}' to float: {e}")

    def get_bool(self, key: str, default: bool | None = None) -> FlextResult[bool]:
        """Get boolean value with safe conversion."""
        if key not in self._data:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' not found")

        value = self._data[key]
        if value is None:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' is None")

        # Direct boolean
        if isinstance(value, bool):
            return FlextResult.ok(value)

        # String conversion
        if isinstance(value, str):
            lower_val = value.lower().strip()
            if lower_val in {"true", "1", "yes", "on", "enabled"}:
                return FlextResult.ok(True)
            if lower_val in {"false", "0", "no", "off", "disabled"}:
                return FlextResult.ok(False)
            return FlextResult.fail(f"Cannot convert string '{value}' to boolean")

        # Numeric conversion
        if isinstance(value, (int, float)):
            return FlextResult.ok(bool(value))

        return FlextResult.fail(f"Cannot convert {type(value)} to boolean")

    def get_list(self, key: str, default: list[Any] | None = None) -> FlextResult[list[Any]]:
        """Get list value with safe conversion."""
        if key not in self._data:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' not found")

        value = self._data[key]
        if value is None:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' is None")

        if isinstance(value, list):
            return FlextResult.ok(value)

        # Try to convert from other iterables
        try:
            return FlextResult.ok(list(value))
        except TypeError:
            return FlextResult.fail(f"Cannot convert {type(value)} to list")

    def get_dict(self, key: str, default: dict[str, Any] | None = None) -> FlextResult[dict[str, Any]]:
        """Get nested dictionary value."""
        if key not in self._data:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' not found")

        value = self._data[key]
        if value is None:
            if default is not None:
                return FlextResult.ok(default)
            return FlextResult.fail(f"Key '{key}' is None")

        if isinstance(value, dict):
            return FlextResult.ok(value)

        return FlextResult.fail(f"Value at '{key}' is not a dictionary")

    # -------------------------------------------------------------------------
    # Advanced Operations
    # -------------------------------------------------------------------------

    def get_nested(self, path: str, separator: str = ".") -> FlextResult[Any]:
        """Get nested value using dot notation.

        Args:
            path: Path to nested value (e.g., "user.address.city")
            separator: Path separator character

        Returns:
            FlextResult with nested value

        Example:
            data = {"user": {"address": {"city": "New York"}}}
            fdict = FlextDict(data)
            city = fdict.get_nested("user.address.city").unwrap()  # "New York"
        """
        parts = path.split(separator)
        current = self._data

        for i, part in enumerate(parts):
            if not isinstance(current, dict):
                current_path = separator.join(parts[:i])
                return FlextResult.fail(f"Path '{current_path}' is not a dictionary")

            if part not in current:
                return FlextResult.fail(f"Path '{path}' not found at '{part}'")

            current = current[part]

        return FlextResult.ok(current)

    def set_nested(self, path: str, value: Any, separator: str = ".") -> FlextResult[None]:
        """Set nested value using dot notation.

        Creates intermediate dictionaries as needed.

        Args:
            path: Path to set (e.g., "user.address.city")
            value: Value to set
            separator: Path separator character

        Returns:
            FlextResult indicating success or failure
        """
        parts = path.split(separator)
        current = self._data

        # Navigate to parent of target
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                return FlextResult.fail(f"Cannot set nested path: '{part}' is not a dictionary")
            current = current[part]

        # Set final value
        current[parts[-1]] = value
        return FlextResult.ok(None)

    def transform_values(
        self,
        transformer: Callable[[Any], Any],
    ) -> FlextResult[FlextDict]:
        """Transform all values in dictionary.

        Args:
            transformer: Function to transform each value

        Returns:
            New FlextDict with transformed values
        """
        try:
            transformed_data = {
                key: transformer(value)
                for key, value in self._data.items()
            }
            return FlextResult.ok(FlextDict(transformed_data))
        except Exception as e:  # noqa: BLE001
            return FlextResult.fail(f"Value transformation failed: {e}")

    def filter_keys(self, predicate: Callable[[str], bool]) -> FlextDict:
        """Filter dictionary by key predicate.

        Args:
            predicate: Function that returns True for keys to keep

        Returns:
            New FlextDict with filtered keys
        """
        filtered_data = {
            key: value
            for key, value in self._data.items()
            if predicate(key)
        }
        return FlextDict(filtered_data)

    def pick(self, *keys: str) -> FlextDict:
        """Create new FlextDict with only specified keys.

        Args:
            keys: Keys to include in new dictionary

        Returns:
            New FlextDict with only specified keys
        """
        picked_data = {
            key: self._data[key]
            for key in keys
            if key in self._data
        }
        return FlextDict(picked_data)

    def omit(self, *keys: str) -> FlextDict:
        """Create new FlextDict excluding specified keys.

        Args:
            keys: Keys to exclude from new dictionary

        Returns:
            New FlextDict without specified keys
        """
        omitted_data = {
            key: value
            for key, value in self._data.items()
            if key not in keys
        }
        return FlextDict(omitted_data)

    # -------------------------------------------------------------------------
    # Validation and Checking
    # -------------------------------------------------------------------------

    def require_keys(self, *keys: str) -> FlextResult[None]:
        """Validate that all specified keys exist.

        Args:
            keys: Required keys

        Returns:
            FlextResult indicating if all keys exist
        """
        missing_keys = [key for key in keys if key not in self._data]
        if missing_keys:
            missing_str = ", ".join(missing_keys)
            return FlextResult.fail(f"Missing required keys: {missing_str}")
        return FlextResult.ok(None)

    def validate_types(self, type_map: dict[str, type]) -> FlextResult[None]:
        """Validate that values match expected types.

        Args:
            type_map: Dictionary mapping keys to expected types

        Returns:
            FlextResult indicating validation result
        """
        errors = []

        for key, expected_type in type_map.items():
            if key not in self._data:
                errors.append(f"Key '{key}' not found")
                continue

            value = self._data[key]
            if value is not None and not isinstance(value, expected_type):
                actual_type = type(value).__name__
                expected_name = expected_type.__name__
                errors.append(f"Key '{key}': expected {expected_name}, got {actual_type}")

        if errors:
            error_msg = "; ".join(errors)
            return FlextResult.fail(f"Type validation failed: {error_msg}")

        return FlextResult.ok(None)

    # -------------------------------------------------------------------------
    # Conversion and Export
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Get copy of underlying dictionary."""
        return dict(self._data)

    def merge(self, other: Mapping[str, Any]) -> FlextDict:
        """Merge with another dictionary, returning new FlextDict."""
        merged_data = {**self._data, **other}
        return FlextDict(merged_data)

    def deep_merge(self, other: Mapping[str, Any]) -> FlextDict:
        """Deep merge with another dictionary."""
        def _deep_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
            result = dict1.copy()
            for key, value in dict2.items():
                if (key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)):
                    result[key] = _deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged_data = _deep_merge(self._data, dict(other))
        return FlextDict(merged_data)

    def __len__(self) -> int:
        """Get number of items in dictionary."""
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        """Check if key exists (supports 'in' operator)."""
        return key in self._data

    def __repr__(self) -> str:
        """String representation."""
        return f"FlextDict({self._data!r})"


# =============================================================================
# UTILITY FUNCTIONS - Standalone Helper Functions
# =============================================================================


def safe_get[T](
    data: dict[str, Any],
    key: str,
    default: T | None = None,
) -> FlextResult[T]:
    """Safely get value from dictionary with type preservation.

    Args:
        data: Dictionary to query
        key: Key to retrieve
        default: Default value if key not found

    Returns:
        FlextResult with value or error

    Example:
        result = safe_get({"age": 30}, "age", 0)
        age = result.unwrap()  # 30
    """
    if key not in data:
        if default is not None:
            return FlextResult.ok(default)
        return FlextResult.fail(f"Key '{key}' not found")

    return FlextResult.ok(data[key])


def safe_set_nested(
    data: dict[str, Any],
    path: str,
    value: Any,
    separator: str = ".",
) -> FlextResult[None]:
    """Safely set nested value in dictionary.

    Args:
        data: Dictionary to modify
        path: Dot notation path
        value: Value to set
        separator: Path separator

    Returns:
        FlextResult indicating success or failure
    """
    return FlextDict(data).set_nested(path, value, separator)


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple dictionaries left to right.

    Args:
        dicts: Dictionaries to merge

    Returns:
        Merged dictionary

    Example:
        result = merge_dicts({"a": 1}, {"b": 2}, {"c": 3})
        # {"a": 1, "b": 2, "c": 3}
    """
    result: dict[str, Any] = {}
    for d in dicts:
        result.update(d)
    return result


def pick_keys(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Create new dict with only specified keys.

    Args:
        data: Source dictionary
        keys: Keys to include

    Returns:
        New dictionary with only specified keys
    """
    return {key: data[key] for key in keys if key in data}


def omit_keys(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Create new dict excluding specified keys.

    Args:
        data: Source dictionary
        keys: Keys to exclude

    Returns:
        New dictionary without specified keys
    """
    return {key: value for key, value in data.items() if key not in keys}


def flatten_dict(
    data: dict[str, Any],
    separator: str = ".",
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested dictionary using dot notation.

    Args:
        data: Dictionary to flatten
        separator: Key separator
        prefix: Key prefix

    Returns:
        Flattened dictionary

    Example:
        nested = {"user": {"name": "Alice", "address": {"city": "NYC"}}}
        flat = flatten_dict(nested)
        # {"user.name": "Alice", "user.address.city": "NYC"}
    """
    items: list[tuple[str, Any]] = []

    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key

        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))

    return dict(items)


# =============================================================================
# EXPORTS - Clean Public API
# =============================================================================

__all__ = [
    "FlextDict",
    "flatten_dict",
    "merge_dicts",
    "omit_keys",
    "pick_keys",
    "safe_get",
    "safe_set_nested",
]
