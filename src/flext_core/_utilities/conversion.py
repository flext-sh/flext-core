"""Utilities module - FlextUtilitiesConversion.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Sequence
from typing import Literal, overload

from flext_core.runtime import FlextRuntime
from flext_core.typings import t

# Approved modules that can import directly (for testing, internal use)
_APPROVED_MODULES = frozenset({
    "flext_core.utilities",
    "flext_core._utilities",
    "tests.",
})


def _check_direct_access() -> None:
    """Warn if accessed from non-approved module."""
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        caller_module = frame.f_back.f_back.f_globals.get("__name__", "")
        if not any(
            caller_module.startswith(approved) for approved in _APPROVED_MODULES
        ):
            warnings.warn(
                "Direct import from _utilities.conversion is deprecated. "
                "Use 'from flext_core import u; u.conversion(...)' instead.",
                DeprecationWarning,
                stacklevel=4,
            )


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
    def to_str(value: object, *, default: str | None = None) -> str:
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
        return str(value)

    @staticmethod
    def to_str_list(
        value: object,
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
        # Cast to GeneralValueType to help type checker
        value_typed: t.GeneralValueType = value
        if FlextRuntime.is_list_like(value_typed) and isinstance(value_typed, Sequence):
            return [str(item) for item in value_typed if item is not None]
        return [str(value)]

    @staticmethod
    def normalize(
        value: object,
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


# PUBLIC GENERALIZED METHOD - Single entry point with routing
@overload
def conversion(
    value: object,
    *,
    mode: Literal["to_str"] = "to_str",
    default: str | None = None,
    case: str | None = None,
) -> str: ...


@overload
def conversion(
    value: object,
    *,
    mode: Literal["to_str_list"],
    default: list[str] | None = None,
    case: str | None = None,
) -> list[str]: ...


@overload
def conversion(
    value: object,
    *,
    mode: Literal["normalize"],
    default: str | None = None,
    case: str | None = None,
) -> str: ...


@overload
def conversion(
    value: Sequence[str],
    *,
    mode: Literal["join"],
    default: str | None = None,
    case: str | None = None,
    separator: str = " ",
) -> str: ...


def conversion(
    value: object,
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
    _check_direct_access()

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
            error_msg = "join mode requires Sequence[str]"
            raise TypeError(error_msg)
        return FlextUtilitiesConversion.join(value, separator=separator, case=case)
    error_msg = f"Unknown mode: {mode}"
    raise ValueError(error_msg)


__all__ = [
    "FlextUtilitiesConversion",
    "conversion",
]
