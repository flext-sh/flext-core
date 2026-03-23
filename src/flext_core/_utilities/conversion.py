"""Utilities module - FlextUtilitiesConversion.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal, overload

from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_core import m, r, t


class FlextUtilitiesConversion:
    """Utilities for value conversion operations.

    PHILOSOPHY:
    ──────────
    - Type-safe conversion with multiple modes
    - String conversion with defaults
    - List conversion with filtering
    - String normalization with case handling
    - String joining with separators
    - Reuses base types from flext_core and constants from flext_core.constants
    """

    _V = m.Validators

    @overload
    @staticmethod
    def conversion(
        value: t.StrictValue,
        *,
        mode: Literal["to_str"] = "to_str",
        default: str | None = None,
        case: str | None = None,
    ) -> str: ...

    @overload
    @staticmethod
    def conversion(
        value: t.StrictValue,
        *,
        mode: Literal["to_str_list"],
        default: Sequence[str] | None = None,
        case: str | None = None,
    ) -> Sequence[str]: ...

    @overload
    @staticmethod
    def conversion(
        value: t.StrictValue,
        *,
        mode: Literal["normalize"],
        default: str | None = None,
        case: str | None = None,
    ) -> str: ...

    @staticmethod
    def conversion(
        value: t.StrictValue,
        *,
        mode: str = "to_str",
        default: str | Sequence[str] | None = None,
        case: str | None = None,
        separator: str = " ",
    ) -> str | Sequence[str]:
        """Generalized conversion utility function.

        Args:
            value: Value to convert
            mode: Operation mode
                - "to_str": Convert to string (returns str)
                - "to_str_list": Convert to list of strings (returns Sequence[str])
                - "normalize": Normalize string value (returns str)
                - "join": Join sequence of strings (returns str)
            default: Default value if None (str for to_str/normalize, Sequence[str] for to_str_list)
            case: Case normalization ("lower", "upper", or None)
            separator: Separator for join mode (default: " ")

        Returns:
            Depends on mode - str or Sequence[str]

        """
        if mode == "to_str":
            default_str: str | None = None
            if default is not None:
                try:
                    default_str = (
                        FlextUtilitiesConversion._V.str_adapter().validate_python(
                            default,
                        )
                    )
                except ValidationError:
                    default_str = None
            return FlextUtilitiesConversion.to_str(value, default=default_str)
        if mode == "to_str_list":
            default_list: Sequence[str] | None = None
            if default is not None:
                try:
                    default_list = (
                        FlextUtilitiesConversion._V.tags_adapter().validate_python(
                            default,
                        )
                    )
                except ValidationError:
                    default_list = None
            return FlextUtilitiesConversion.to_str_list(value, default=default_list)
        if mode == "normalize":
            return FlextUtilitiesConversion.normalize(value, case=case)
        if mode == "join":
            raw_values: Sequence[t.StrictValue]
            try:
                raw_values = FlextUtilitiesConversion._V.strict_json_list_adapter().validate_python(
                    value,
                )
            except ValidationError as err:
                error_msg = "join mode requires Sequence"
                raise TypeError(error_msg) from err
            str_values: Sequence[str] = [str(v) for v in raw_values]
            return FlextUtilitiesConversion.join(
                str_values,
                separator=separator,
                case=case,
            )
        error_msg = f"Unknown mode: {mode}"
        raise ValueError(error_msg)

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
    def to_flexible_value(value: t.StrictValue) -> r[t.Scalar]:
        """Convert strict value to strict scalar if compatible.

        Strict scalar is a subset of strict value that excludes
        BaseModel, Path, and Callable types.

        Args:
            value: strict value to convert

        Returns:
            r containing strict scalar value or failure

        """
        if value is None:
            return r[t.Scalar].fail("None is not a scalar-compatible value")
        if isinstance(value, (BaseModel, Mapping, list, tuple, set, frozenset)):
            return r[t.Scalar].fail("Value is not a scalar-compatible type")
        if isinstance(value, datetime) and hasattr(value, "isoformat"):
            isoformat_method = value.isoformat
            if callable(isoformat_method):
                return r[t.Scalar].ok(str(value))
        try:
            strict_value = FlextUtilitiesConversion._V.strict_json_scalar_adapter().validate_python(
                value,
            )
            return r[t.Scalar].ok(strict_value)
        except ValidationError:
            return r[t.Scalar].ok(str(value))

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
        default: Sequence[str] | None = None,
    ) -> Sequence[str]:
        """Convert value to list of strings.

        Args:
            value: Value to convert
            default: Default value if None

        Returns:
            Sequence[str]: Converted list of strings

        """
        if value is None:
            return default if default is not None else []
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

    @staticmethod
    def to_str_list_safe(
        value: t.StrictValue,
        *,
        filter_list_like: bool = True,
    ) -> Sequence[str]:
        """Convert value to Sequence[str] with safe nested list handling.

        Safely handles nested list-like structures by filtering them out
        to prevent nested lists in the returned result.

        Args:
            value: Value to convert
            filter_list_like: If True, filter out list-like items from result

        Returns:
            Sequence[str]: List of string values

        Example:
            >>> u.to_str_list_safe("foo")
            ["foo"]
            >>> u.to_str_list_safe(["a", "b", ["nested"]])
            ["a", "b"]  # nested list filtered

        """
        if value is None:
            return []
        items: Sequence[t.StrictValue] = []
        value_class = value.__class__
        if value_class is str:
            items = [value]
        elif value_class is list:
            try:
                items = FlextUtilitiesConversion._V.strict_json_list_adapter().validate_python(
                    value,
                )
            except ValidationError:
                items = []
        else:
            items = [value]
        filtered_items: Sequence[t.StrictValue]
        if filter_list_like:
            filtered_items = [
                item
                for item in items
                if item is not None
                and (
                    not (
                        isinstance(item, (list, tuple, set, frozenset))
                        or (
                            isinstance(item, (list, tuple))
                            or (
                                hasattr(item, "__getitem__")
                                and not isinstance(item, (str, bytes))
                            )
                        )
                    )
                )
            ]
        else:
            filtered_items = [item for item in items if item is not None]
        return [str(item) for item in filtered_items]

    @staticmethod
    def to_str_list_truthy(value: t.StrictValue) -> Sequence[str]:
        """Convert value to Sequence[str] filtering out falsy values."""
        result = FlextUtilitiesConversion.to_str_list_safe(value, filter_list_like=True)
        return [item for item in result if item]

    @staticmethod
    def narrow[T](value: t.ValueOrModel, type_cls: type[T]) -> T:
        """Narrow *value* to *type_cls*, attempting coercion via Pydantic validation.

        Args:
            value: Value to narrow
            type_cls: Target type

        Returns:
            T: Value narrowed or coerced to type_cls

        """
        if isinstance(value, type_cls):
            return value
        adapter: TypeAdapter[T] = TypeAdapter(type_cls)
        return adapter.validate_python(value)


__all__ = ["FlextUtilitiesConversion"]
