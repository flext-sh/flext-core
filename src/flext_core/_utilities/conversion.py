"""Utilities module - FlextUtilitiesConversion.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, overload

from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextUtilitiesConversion:
    """Utilities for value conversion operations.

    PHILOSOPHY:
    ──────────
    - Type-safe conversion with multiple modes
    - String conversion with defaults
    - List conversion with filtering
    - String normalization with case handling
    - String joining with separators
    - Reuses base types from flext_core.typings and constants from flext_core.constants
    """

    @staticmethod
    def to_str(value: t.GeneralValueType, *, default: str | None = None) -> str:
        """Convert value to string.

        Args:
            value: Value to convert
            default: Default value if None

        Returns:
            str: Converted string value

        """
        if value is None:
            return default or ""
        if isinstance(value, str):
            return value
        if isinstance(value, float):
            # Format float to 2 decimal places if it's a decimal number
            if value.is_integer():
                return str(int(value))
            return f"{value:.2f}"
        return str(value)

    @staticmethod
    def to_str_list(
        value: t.GeneralValueType,
        *,
        default: list[str] | None = None,
    ) -> list[str]:
        """Convert value to list of strings.

        Args:
            value: Value to convert
            default: Default value if None

        Returns:
            list[str]: Converted list of strings

        """
        if value is None:
            return default or []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set, frozenset)):
            return [str(item) for item in value if item is not None]
        # For other sequences, check if list-like and iterable
        if FlextRuntime.is_list_like(value):
            return [str(item) for item in value if item is not None]
        return [str(value)]

    @staticmethod
    def normalize(
        value: t.GeneralValueType,
        *,
        case: str | None = None,
    ) -> str:
        """Normalize string value with optional case conversion.

        Args:
            value: Value to normalize
            case: Case normalization ("lower", "upper", or None)

        Returns:
            str: Normalized string value

        """
        str_value = FlextUtilitiesConversion.to_str(value)
        if case == "lower":
            return str_value.lower()
        if case == "upper":
            return str_value.upper()
        return str_value

    @staticmethod
    def join(
        values: Sequence[str],
        *,
        separator: str = " ",
        case: str | None = None,
    ) -> str:
        """Join string values with separator and optional case conversion.

        Args:
            values: Sequence of strings to join
            separator: Separator string (default: " ")
            case: Case normalization ("lower", "upper", or None)

        Returns:
            str: Joined string value

        """
        if not values:
            return ""
        normalized = values
        if case == "lower":
            normalized = [v.lower() for v in values]
        elif case == "upper":
            normalized = [v.upper() for v in values]
        return separator.join(normalized)

    @overload
    @staticmethod
    def conversion(
        value: t.GeneralValueType,
        *,
        mode: Literal["to_str"] = "to_str",
        default: str | None = None,
        case: str | None = None,
    ) -> str: ...

    @overload
    @staticmethod
    def conversion(
        value: t.GeneralValueType,
        *,
        mode: Literal["to_str_list"],
        default: list[str] | None = None,
        case: str | None = None,
    ) -> list[str]: ...

    @overload
    @staticmethod
    def conversion(
        value: t.GeneralValueType,
        *,
        mode: Literal["normalize"],
        default: str | None = None,
        case: str | None = None,
    ) -> str: ...

    @overload
    @staticmethod
    def conversion(
        value: Sequence[str],
        *,
        mode: Literal["join"],
        default: str | None = None,
        case: str | None = None,
        separator: str = " ",
    ) -> str: ...

    @staticmethod
    def conversion(
        value: t.GeneralValueType,
        *,
        mode: str = "to_str",
        default: str | list[str] | None = None,
        case: str | None = None,
        separator: str = " ",
    ) -> str | list[str]:
        """Generalized conversion utility function.

        Args:
            value: Value to convert
            mode: Operation mode
                - "to_str": Convert to string (returns str)
                - "to_str_list": Convert to list of strings (returns list[str])
                - "normalize": Normalize string value (returns str)
                - "join": Join sequence of strings (returns str)
            default: Default value if None (str for to_str/normalize, list[str] for to_str_list)
            case: Case normalization ("lower", "upper", or None)
            separator: Separator for join mode (default: " ")

        Returns:
            Depends on mode - str or list[str]

        """
        if mode == "to_str":
            # Type narrowing: default should be str | None for to_str
            default_str: str | None = (
                default if isinstance(default, (str, type(None))) else None
            )
            return FlextUtilitiesConversion.to_str(value, default=default_str)
        if mode == "to_str_list":
            # Type narrowing: default should be list[str] | None for to_str_list
            default_list: list[str] | None = (
                default if isinstance(default, (list, type(None))) else None
            )
            return FlextUtilitiesConversion.to_str_list(value, default=default_list)
        if mode == "normalize":
            return FlextUtilitiesConversion.normalize(value, case=case)
        if mode == "join":
            if not isinstance(value, Sequence):
                error_msg = "join mode requires Sequence"
                raise TypeError(error_msg)
            # Convert sequence items to strings for type safety
            # Strings are valid sequences (of characters)
            str_values: list[str] = [str(v) for v in value]
            return FlextUtilitiesConversion.join(
                str_values,
                separator=separator,
                case=case,
            )
        error_msg = f"Unknown mode: {mode}"
        raise ValueError(error_msg)

    @staticmethod
    def to_general_value_type(value: object) -> t.GeneralValueType:
        """Convert object to GeneralValueType with runtime check.

        If value is already a GeneralValueType, return it.
        Otherwise convert to string representation.

        Args:
            value: Object to convert

        Returns:
            GeneralValueType: Converted value

        """
        # Check known types that are part of GeneralValueType
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value
        if FlextRuntime.is_base_model(value):
            return value
        if isinstance(value, Path):
            return value
        if callable(value):
            # Callable[..., GeneralValueType] - return as-is
            callable_typed: Callable[..., t.GeneralValueType] = value
            return callable_typed
        if isinstance(value, list):
            # list is Sequence[GeneralValueType] compatible
            seq_result: Sequence[t.GeneralValueType] = value
            return seq_result
        if isinstance(value, tuple):
            # tuple is Sequence[GeneralValueType] compatible
            seq_tuple: Sequence[t.GeneralValueType] = value
            return seq_tuple
        if isinstance(value, dict):
            # dict is Mapping[str, GeneralValueType] compatible
            map_result: Mapping[str, t.GeneralValueType] = value
            return map_result
        # Fallback: convert to string
        return str(value)

    @staticmethod
    def to_flexible_value(value: t.GeneralValueType) -> t.FlexibleValue | None:
        """Convert GeneralValueType to FlexibleValue if compatible.

        FlexibleValue is a subset of GeneralValueType that excludes
        BaseModel, Path, and Callable types.

        Args:
            value: GeneralValueType to convert

        Returns:
            FlexibleValue or None if not compatible

        """
        # FlexibleValue = str | int | float | bool | datetime | None
        #                | Sequence[scalar] | Mapping[str, scalar]
        # where scalar = str | int | float | bool | datetime | None
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value
        # Exclude BaseModel, Path, Callable - these are not FlexibleValue
        if FlextRuntime.is_base_model(value):
            return None
        if isinstance(value, Path):
            return None
        if callable(value):
            return None
        # Check for simple sequences (not nested GeneralValueType)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            # Can't easily validate element types at runtime, assume compatible
            return None  # Skip complex sequences for safety
        if isinstance(value, Mapping):
            # Can't easily validate value types at runtime, assume compatible
            return None  # Skip complex mappings for safety
        return None


conversion = FlextUtilitiesConversion.conversion

__all__ = ["FlextUtilitiesConversion", "conversion"]
