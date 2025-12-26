"""Utilities module - FlextUtilitiesConversion.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
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
        if FlextRuntime.is_list_like(value) and isinstance(value, Sequence):
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


conversion = FlextUtilitiesConversion.conversion

__all__ = ["FlextUtilitiesConversion", "conversion"]
