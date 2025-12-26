"""Utilities module - FlextUtilitiesCast.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import warnings


class FlextUtilitiesCast:
    """Utilities for type-safe casting operations."""

    # Approved modules that can import directly (for testing, internal use)
    _APPROVED_MODULES: frozenset[str] = frozenset({
        "flext_core.utilities",
        "flext_core._utilities",
        "tests.",
    })

    @staticmethod
    def _check_direct_access() -> None:
        """Warn if accessed from non-approved module."""
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_module = frame.f_back.f_back.f_globals.get("__name__", "")
            if not any(
                caller_module.startswith(approved)
                for approved in FlextUtilitiesCast._APPROVED_MODULES
            ):
                warnings.warn(
                    "Direct import from _utilities.cast is deprecated. "
                    "Use 'from flext_core import u; u.Cast.safe(...)' instead.",
                    DeprecationWarning,
                    stacklevel=4,
                )

    @staticmethod
    def direct(value: object, target_type: type) -> object:
        """Direct cast - assumes value is already of target type."""
        if isinstance(value, target_type):
            return value
        type_name = getattr(target_type, "__name__", str(target_type))
        error_msg = f"Value {value!r} is not instance of {type_name}"
        raise TypeError(error_msg)

    @staticmethod
    def general_value(value: object, target_type: type) -> object:
        """Cast general value type to target type."""
        if isinstance(value, target_type):
            return value
        if target_type is str:
            return str(value)
        if target_type is int:
            if isinstance(value, (int, float, str)):
                return int(value)
            source_name = getattr(type(value), "__name__", str(type(value)))
            error_msg = f"Cannot cast {source_name} to int"
            raise TypeError(error_msg)
        if target_type is float:
            if isinstance(value, (int, float, str)):
                return float(value)
            source_name = getattr(type(value), "__name__", str(type(value)))
            error_msg = f"Cannot cast {source_name} to float"
            raise TypeError(error_msg)
        if target_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"true", "1", "yes", "on"}
            source_name = getattr(type(value), "__name__", str(type(value)))
            error_msg = f"Cannot cast {source_name} to bool"
            raise TypeError(error_msg)
        source_name = getattr(type(value), "__name__", str(type(value)))
        target_name = getattr(target_type, "__name__", str(target_type))
        error_msg = f"Cannot cast {source_name} to {target_name}"
        raise TypeError(error_msg)

    @staticmethod
    def callable(value: object, target_type: type) -> object:
        """Cast callable to target type."""
        if isinstance(value, target_type):
            return value
        if callable(value):
            try:
                if target_type is object:
                    return value
                return target_type(value)
            except (TypeError, ValueError) as err:
                target_name = getattr(target_type, "__name__", str(target_type))
                error_msg = f"Cannot cast callable to {target_name}: {err}"
                raise TypeError(error_msg) from err
        source_name = getattr(type(value), "__name__", str(type(value)))
        target_name = getattr(target_type, "__name__", str(target_type))
        error_msg = f"Cannot cast {source_name} to {target_name}"
        raise TypeError(error_msg)

    @staticmethod
    def safe(value: object, target_type: type, *, mode: str = "direct") -> object:
        """Type-safe casting with routing.

        Args:
            value: Value to cast
            target_type: Target type to cast to
            mode: Casting mode - 'direct', 'general_value', or 'callable'

        Returns:
            Cast value

        Raises:
            TypeError: If cast is not possible
            ValueError: If unknown mode is provided

        """
        FlextUtilitiesCast._check_direct_access()

        if mode == "direct":
            return FlextUtilitiesCast.direct(value, target_type)
        if mode == "general_value":
            return FlextUtilitiesCast.general_value(value, target_type)
        if mode == "callable":
            return FlextUtilitiesCast.callable(value, target_type)
        error_msg = f"Unknown mode: {mode}"
        raise ValueError(error_msg)


# Backward compatibility alias
cast_safe = FlextUtilitiesCast.safe


__all__ = [
    "FlextUtilitiesCast",
    "cast_safe",
]
