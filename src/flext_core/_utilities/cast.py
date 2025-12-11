"""Utilities module - FlextUtilitiesCast.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Literal, cast, overload

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
    - Callable casting for function types (using callable() builtin)
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
            type_name = getattr(target_type, "__name__", str(target_type))
            error_msg = f"Value {value!r} is not instance of {type_name}"
            raise TypeError(error_msg)
        return value

    @staticmethod
    @overload
    def general_value(value: object, target_type: type[str]) -> str: ...

    @staticmethod
    @overload
    def general_value(value: object, target_type: type[bool]) -> bool: ...

    @staticmethod
    @overload
    def general_value(value: object, target_type: type[int]) -> int: ...

    @staticmethod
    @overload
    def general_value(value: object, target_type: type[float]) -> float: ...

    @staticmethod
    @overload
    def general_value[T](value: object, target_type: type[T]) -> T: ...

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
        if isinstance(value, target_type):
            return value
        # For t.GeneralValueType, try to convert
        # Dispatch based on target_type identity
        # The overloads ensure correct return types for str, int, float, bool
        # Use  for type narrowing - type checker can't infer from identity checks
        if target_type is str:
            str_result: str = str(value)
            return cast("T", str_result)
        if target_type is int:
            if isinstance(value, (int, float, str)):
                int_result: int = int(value)
                return cast("T", int_result)
            source_name = getattr(type(value), "__name__", str(type(value)))
            error_msg = f"Cannot cast {source_name} to int"
            raise TypeError(error_msg)
        if target_type is float:
            if isinstance(value, (int, float, str)):
                float_result: float = float(value)
                return cast("T", float_result)
            source_name = getattr(type(value), "__name__", str(type(value)))
            error_msg = f"Cannot cast {source_name} to float"
            raise TypeError(error_msg)
        if target_type is bool:
            if isinstance(value, bool):
                bool_result: bool = value
                return cast("T", bool_result)
            if isinstance(value, str):
                bool_result_str: bool = value.lower() in {"true", "1", "yes", "on"}
                return cast("T", bool_result_str)
            source_name = getattr(type(value), "__name__", str(type(value)))
            error_msg = f"Cannot cast {source_name} to bool"
            raise TypeError(error_msg)
        source_name = getattr(type(value), "__name__", str(type(value)))
        target_name = getattr(target_type, "__name__", str(target_type))
        error_msg = f"Cannot cast {source_name} to {target_name}"
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
        # callable() check is sufficient - isinstance(value, Callable) is redundant
        if callable(value):
            # For callable types, try to instantiate
            # Use  for dynamic instantiation - type checker can't verify constructor signature
            # This is a casting utility, so dynamic calls are expected
            try:
                # Dynamic instantiation: target_type is type[T] but constructor signature unknown
                # Use callable() to check if target_type is callable, then call it
                # Type checker can't verify constructor signature, so use cast
                if isinstance(value, type):
                    # value is a class, try to instantiate by calling target_type
                    # This is dynamic instantiation - callable() checks if it's callable
                    if callable(target_type):
                        instantiated: T = target_type()
                    else:
                        msg = f"Cannot instantiate non-callable type: {target_type}"
                        raise TypeError(msg)
                # value is an instance, try to call target_type as constructor
                # Type checker can't verify constructor signature, use callable() check
                elif callable(target_type):
                    # target_type is callable, but mypy doesn't know the signature
                    # Use cast to tell mypy this is a constructor call
                    target_constructor: Callable[[object], T] = cast(
                        "Callable[[object], T]",
                        target_type,
                    )
                    instantiated = target_constructor(value)
                else:
                    msg = f"Cannot instantiate non-callable type: {target_type}"
                    raise TypeError(msg)
                return instantiated
            except (TypeError, ValueError) as err:
                target_name = getattr(target_type, "__name__", str(target_type))
                error_msg = f"Cannot cast callable to {target_name}: {err}"
                raise TypeError(error_msg) from err
        target_name = getattr(target_type, "__name__", str(target_type))
        error_msg = f"Value is not callable or instance of {target_name}"
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
