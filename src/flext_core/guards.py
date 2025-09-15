"""FLEXT Guards - Validation guards and utilities.

This module contains the FlextGuards compatibility layer that was previously
defined using dynamic type creation in __init__.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.result import FlextResult
from flext_core.validations import FlextValidations


class FlextGuards:
    """Guards for validation and type checking - backward compatibility layer."""

    class _ValidationUtils:
        """Validation utilities for backward compatibility."""

        @staticmethod
        def require_not_none(value: object, name: str = "value") -> FlextResult[object]:
            """Require value is not None."""
            return FlextValidations.Guards.require_not_none(
                value, f"{name} cannot be None"
            )

        @staticmethod
        def require_positive(value: object, name: str = "value") -> FlextResult[object]:
            """Require value is positive."""
            return FlextValidations.Guards.require_positive(
                value, f"{name} must be positive"
            )

        @staticmethod
        def require_in_range(
            value: object, min_val: float, max_val: float, name: str = "value"
        ) -> FlextResult[object]:
            """Require value is in range."""
            return FlextValidations.Guards.require_in_range(
                value, min_val, max_val, f"{name} out of range"
            )

        @staticmethod
        def require_non_empty(
            value: object, name: str = "value"
        ) -> FlextResult[object]:
            """Require string is non-empty."""
            return FlextValidations.Guards.require_non_empty(
                value, f"{name} cannot be empty"
            )

    @staticmethod
    def is_dict_of(value: object, value_type: type) -> bool:
        """Check if value is a dictionary with values of specific type."""
        return FlextValidations.Guards.is_dict_of(value, value_type)

    @staticmethod
    def is_list_of(value: object, item_type: type) -> bool:
        """Check if value is a list of specific type."""
        return FlextValidations.Guards.is_list_of(value, item_type)

    @staticmethod
    def require_not_none(value: object, name: str = "value") -> FlextResult[object]:
        """Require value is not None."""
        return FlextValidations.Guards.require_not_none(value, f"{name} cannot be None")

    @staticmethod
    def require_positive(value: object, name: str = "value") -> FlextResult[object]:
        """Require value is positive."""
        return FlextValidations.Guards.require_positive(
            value, f"{name} must be positive"
        )

    @staticmethod
    def require_in_range(
        value: object, min_val: float, max_val: float, name: str = "value"
    ) -> FlextResult[object]:
        """Require value is in range."""
        return FlextValidations.Guards.require_in_range(
            value, min_val, max_val, f"{name} out of range"
        )

    @staticmethod
    def require_non_empty(value: object, name: str = "value") -> FlextResult[object]:
        """Require string is non-empty."""
        return FlextValidations.Guards.require_non_empty(
            value, f"{name} cannot be empty"
        )

    # Add ValidationUtils as class attribute for compatibility
    ValidationUtils = _ValidationUtils
