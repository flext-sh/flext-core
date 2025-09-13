"""FLEXT Fields - Field validation utilities.

This module contains the FlextFields compatibility layer that was previously
defined using dynamic type creation in __init__.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.result import FlextResult
from flext_core.validations import FlextValidations


class FlextFields:
    """Field validators - backward compatibility layer."""

    @staticmethod
    def validate_email(value: str) -> FlextResult[str]:
        """Validate email address format."""
        return FlextValidations.FieldValidators.validate_email(value)

    @staticmethod
    def validate_uuid(value: str) -> FlextResult[str]:
        """Validate UUID format."""
        return FlextValidations.FieldValidators.validate_uuid(value)

    @staticmethod
    def validate_url(value: str) -> FlextResult[str]:
        """Validate URL format."""
        return FlextValidations.FieldValidators.validate_url(value)

    @staticmethod
    def validate_phone(value: str) -> FlextResult[str]:
        """Validate phone number format."""
        return FlextValidations.FieldValidators.validate_phone(value)
