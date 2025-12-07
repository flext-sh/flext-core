"""Utilities module - FlextUtilitiesCast.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Literal, TypeVar, overload

from flext_core.typings import t

T = TypeVar("T")

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
                "Direct import from _utilities.cast is deprecated. "
                "Use 'from flext_core import u; u.cast_safe(...)' instead.",
                DeprecationWarning,
                stacklevel=4,
            )


class FlextUtilitiesCast:
    """Utilities for type-safe casting operations.

    PHILOSOPHY:
    ──────────
    - Type-safe casting with multiple modes
    - Direct casting for known types
    - General value conversion for flexible types
    - Callable casting for function types
    - Reuses base types from flext_core.typings
    """

    @staticmethod
    def direct[T](value: object, target_type: type[T]) -> T:
        """Direct cast - assumes value is already of target type.

        Args:
            value: Value to cast
            target_type: Target type to cast to

        Returns:
            T: Cast value of target type

        Raises:
            TypeError: If value is not instance of target type

        """
        if not isinstance(value, target_type):
            error_msg = f"Value {value!r} is not instance of {target_type.__name__}"
            raise TypeError(error_msg)
        return value

    @staticmethod
    def general_value[T](value: object, target_type: type[T]) -> T:
        """Cast general value type to target type.

        Supports conversion for common types: str, int, float, bool.

        Args:
            value: Value to cast
            target_type: Target type to cast to

        Returns:
            T: Cast value of target type

        Raises:
            TypeError: If cast fails

        """
        value_typed: t.GeneralValueType = value
        if isinstance(value_typed, target_type):
            return value_typed
        # For GeneralValueType, try to convert
        if target_type is str:
            return str(value_typed)  # type: ignore[return-value]
        if target_type is int and isinstance(value_typed, (int, float, str)):
            return int(value_typed)  # type: ignore[return-value]
        if target_type is float and isinstance(value_typed, (int, float, str)):
            return float(value_typed)  # type: ignore[return-value]
        if target_type is bool:
            if isinstance(value_typed, bool):
                return value_typed  # type: ignore[return-value]
            if isinstance(value_typed, str):
                return value_typed.lower() in {"true", "1", "yes", "on"}  # type: ignore[return-value]
        error_msg = (
            f"Cannot cast {type(value_typed).__name__} to {target_type.__name__}"
        )
        raise TypeError(error_msg)

    @staticmethod
    def callable[T](value: object, target_type: type[T]) -> T:
        """Cast callable to target type.

        Args:
            value: Value to cast
            target_type: Target type to cast to

        Returns:
            T: Cast value of target type

        Raises:
            TypeError: If cast fails

        """
        if isinstance(value, target_type):
            return value
        if (
            callable(value)
            and isinstance(target_type, type)
            and isinstance(value, Callable)
        ):
            # For callable types, try to instantiate
            try:
                return target_type(value)  # type: ignore[return-value, call-overload]
            except (TypeError, ValueError) as err:
                error_msg = f"Cannot cast callable to {target_type.__name__}: {err}"
                raise TypeError(error_msg) from err
        error_msg = f"Value is not callable or instance of {target_type.__name__}"
        raise TypeError(error_msg)


# PUBLIC GENERALIZED METHOD - Single entry point with routing
@overload
def cast_safe[T](
    value: object,
    target_type: type[T],
    *,
    mode: Literal["direct"] = "direct",
) -> T: ...


@overload
def cast_safe[T](
    value: object,
    target_type: type[T],
    *,
    mode: Literal["general_value"],
) -> T: ...


@overload
def cast_safe[T](
    value: object,
    target_type: type[T],
    *,
    mode: Literal["callable"],
) -> T: ...


def cast_safe[T](
    value: object,
    target_type: type[T],
    *,
    mode: str = "direct",
) -> T:
    """Type-safe casting with routing.

    Args:
        value: Value to cast
        target_type: Target type to cast to
        mode: Operation mode
            - "direct": Direct cast (assumes value is already of target type)
            - "general_value": Cast general value type to target type
            - "callable": Cast callable to target type

    Returns:
        T: Cast value of target type

    Raises:
        TypeError: If cast fails

    """
    _check_direct_access()

    if mode == "direct":
        return FlextUtilitiesCast.direct(value, target_type)
    if mode == "general_value":
        return FlextUtilitiesCast.general_value(value, target_type)
    if mode == "callable":
        return FlextUtilitiesCast.callable(value, target_type)
    error_msg = f"Unknown mode: {mode}"
    raise ValueError(error_msg)


__all__ = [
    "FlextUtilitiesCast",
    "cast_safe",
]
