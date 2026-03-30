"""Utilities module - FlextUtilitiesConversion.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import ValidationError

from flext_core import m, t


class FlextUtilitiesConversion:
    """Utilities for value conversion operations."""

    _V: ClassVar[type[m.Validators]] = m.Validators

    @staticmethod
    def join(
        values: t.StrSequence,
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

    @staticmethod
    def normalize(value: t.StrictValue, *, case: str | None = None) -> str:
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
    def to_str(value: t.StrictValue, *, default: str | None = None) -> str:
        """Convert value to string.

        Args:
            value: Value to convert
            default: Default value if None

        Returns:
            str: Converted string value

        """
        if value is None:
            return default if default is not None else ""
        if isinstance(value, str):
            return str(value)
        try:
            float_value = FlextUtilitiesConversion._V.float_adapter().validate_python(
                value,
            )
            if float_value.is_integer():
                return str(int(float_value))
            return f"{float_value:.2f}"
        except ValidationError:
            pass
        return str(value)

    @staticmethod
    def to_str_list(
        value: t.StrictValue,
        *,
        default: t.StrSequence | None = None,
    ) -> t.StrSequence:
        """Convert value to list of strings.

        Args:
            value: Value to convert
            default: Default value if None

        Returns:
            t.StrSequence: Converted list of strings

        """
        if value is None:
            return default if default is not None else list[str]()
        value_class = value.__class__
        if value_class is str:
            return [str(value)]
        try:
            list_value = (
                FlextUtilitiesConversion._V.strict_json_list_adapter().validate_python(
                    value,
                )
            )
            return [str(item) for item in list_value if item is not None]
        except ValidationError:
            pass
        return [str(value)]


__all__ = ["FlextUtilitiesConversion"]
