"""Utilities module - FlextUtilitiesConversion.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from contextlib import suppress

from flext_core import FlextConstants as c, FlextTypes as t


class FlextUtilitiesConversion:
    """Utilities for value conversion operations."""

    @staticmethod
    def join(
        values: t.StrSequence,
        *,
        separator: str = " ",
        case: str | None = None,
    ) -> str:
        """Join string values with separator and optional case conversion."""
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
        """Normalize value to string with optional case conversion."""
        str_value = FlextUtilitiesConversion.to_str(value)
        if case == "lower":
            return str_value.lower()
        if case == "upper":
            return str_value.upper()
        return str_value

    @staticmethod
    def to_str(value: t.JsonPayload | None, *, default: str | None = None) -> str:
        """Convert value to string, formatting floats as integers when possible."""
        if value is None:
            return default if default is not None else ""
        if isinstance(value, str):
            return value
        try:
            float_value = t.float_adapter().validate_python(
                value,
            )
            if float_value.is_integer():
                return str(int(float_value))
            return f"{float_value:.2f}"
        except c.ValidationError:
            return str(value)

    @staticmethod
    def to_str_list(
        value: t.StrictValue | None,
        *,
        default: t.StrSequence | None = None,
    ) -> t.StrSequence:
        """Convert value to list of strings."""
        if value is None:
            return default if default is not None else list[str]()
        value_class = value.__class__
        if value_class is str:
            return [str(value)]
        try:
            list_value = t.strict_json_list_adapter().validate_python(value)
            return [str(item) for item in list_value]
        except c.ValidationError:
            return [str(value)]

    @staticmethod
    def to_int(value: t.JsonPayload | None, *, default: int = 0) -> int:
        """Convert value to int with safe fallback; bool returns default."""
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return (
                int(value)
                if isinstance(value, int) or math.isfinite(value)
                else default
            )
        if isinstance(value, str):
            with suppress(ValueError, OverflowError):
                parsed = float(value)
                if math.isfinite(parsed):
                    return int(parsed)
        return default

    @staticmethod
    def to_float(value: t.JsonPayload | None, *, default: float = 0.0) -> float:
        """Convert value to float with safe fallback; bool returns default."""
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, OverflowError):
                return default
        return default

    @staticmethod
    def to_bool(value: t.JsonPayload | None, *, default: bool = False) -> bool:
        """Convert value to bool with safe fallback."""
        if value is None:
            return default
        return bool(value)

    @staticmethod
    def to_positive_int(value: t.JsonPayload | None, *, default: int = 0) -> int:
        """Convert value to a strictly positive integer with safe fallback.

        Rejects ``None``, booleans, non-numeric strings, and non-positive values.
        """
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value if value > 0 else default
        if isinstance(value, float) and value.is_integer():
            return int(value) if value > 0 else default
        if isinstance(value, str) and value.strip().isdigit():
            return int(value)
        return default

    @staticmethod
    def to_optional_str(value: t.JsonPayload | None) -> str | None:
        """Return the value unchanged only when it is a non-empty string."""
        if value is None or not isinstance(value, str):
            return None
        return value or None


__all__: list[str] = ["FlextUtilitiesConversion"]
