"""Utilities module - FlextUtilitiesConversion.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from flext_core import c, m, t
from flext_core.runtime import FlextRuntime


class FlextUtilitiesConversion:
    """Utilities for value conversion operations."""

    _V: ClassVar[type[t]] = t

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
    def to_str(value: t.StrictValue, *, default: str | None = None) -> str:
        """Convert value to string, formatting floats as integers when possible."""
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
        except c.ValidationError:
            pass
        return str(value)

    @staticmethod
    def to_str_list(
        value: t.StrictValue,
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
            list_value = (
                FlextUtilitiesConversion._V.strict_json_list_adapter().validate_python(
                    value,
                )
            )
            return [str(item) for item in list_value if item is not None]
        except c.ValidationError:
            pass
        return [str(value)]

    @staticmethod
    def to_int(value: t.ValueOrModel, *, default: int = 0) -> int:
        """Convert value to int with safe fallback; bool returns default."""
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except (ValueError, OverflowError):
                return default
        return default

    @staticmethod
    def to_float(value: t.ValueOrModel, *, default: float = 0.0) -> float:
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
    def normalize_log_payload(
        payload: Mapping[str, t.ValueOrModel],
    ) -> t.FlatContainerMapping:
        """Normalize payload to flat container types, stringifying model instances."""
        normalized: t.MutableFlatContainerMapping = {}
        for key, value in payload.items():
            atomic = FlextRuntime.normalize_to_container(value)
            if isinstance(atomic, m.BaseModel):
                normalized[str(key)] = atomic.model_dump_json()
            else:
                normalized[str(key)] = atomic
        return normalized


__all__: list[str] = ["FlextUtilitiesConversion"]
