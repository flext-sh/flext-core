"""Utilities module - FlextUtilitiesConversion.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, overload

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

type StrictJsonScalar = str | int | float | bool | None
type StrictJsonValue = (
    StrictJsonScalar | list[StrictJsonValue] | Mapping[str, StrictJsonValue]
)


class _StrictJsonScalarModel(BaseModel):
    """Strict scalar wrapper for narrow value validation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    value: StrictJsonScalar


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

    _strict_json_list_adapter: TypeAdapter[list[StrictJsonValue]] = TypeAdapter(
        list[StrictJsonValue],
    )
    _strict_json_scalar_adapter: TypeAdapter[StrictJsonScalar] = TypeAdapter(
        StrictJsonScalar,
    )
    _float_adapter = TypeAdapter(float)
    _str_adapter = TypeAdapter(str)
    _str_list_adapter = TypeAdapter(list[str])

    @staticmethod
    def to_str(value: StrictJsonValue, *, default: str | None = None) -> str:
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
            return str(value)
        try:
            float_value = FlextUtilitiesConversion._float_adapter.validate_python(value)
            # Format float to 2 decimal places if it's a decimal number
            if float_value.is_integer():
                return str(int(float_value))
            return f"{float_value:.2f}"
        except ValidationError:
            pass
        return str(value)

    @staticmethod
    def to_str_list(
        value: StrictJsonValue,
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
        value_class = value.__class__
        if value_class is str:
            return [str(value)]
        try:
            list_value = (
                FlextUtilitiesConversion._strict_json_list_adapter.validate_python(
                    value,
                )
            )
            return [str(item) for item in list_value if item is not None]
        except ValidationError:
            pass
        return [str(value)]

    @staticmethod
    def normalize(
        value: StrictJsonValue,
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

    @overload
    @staticmethod
    def conversion(
        value: StrictJsonValue,
        *,
        mode: Literal["to_str"] = "to_str",
        default: str | None = None,
        case: str | None = None,
    ) -> str: ...

    @overload
    @staticmethod
    def conversion(
        value: StrictJsonValue,
        *,
        mode: Literal["to_str_list"],
        default: list[str] | None = None,
        case: str | None = None,
    ) -> list[str]: ...

    @overload
    @staticmethod
    def conversion(
        value: StrictJsonValue,
        *,
        mode: Literal["normalize"],
        default: str | None = None,
        case: str | None = None,
    ) -> str: ...

    @staticmethod
    def conversion(
        value: StrictJsonValue,
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
        if mode == "to_str":
            # Type narrowing: default should be str | None for to_str
            default_str: str | None = None
            if default is not None:
                try:
                    default_str = FlextUtilitiesConversion._str_adapter.validate_python(
                        default,
                    )
                except ValidationError:
                    default_str = None
            return FlextUtilitiesConversion.to_str(value, default=default_str)
        if mode == "to_str_list":
            # Type narrowing: default should be list[str] | None for to_str_list
            default_list: list[str] | None = None
            if default is not None:
                try:
                    default_list = (
                        FlextUtilitiesConversion._str_list_adapter.validate_python(
                            default,
                        )
                    )
                except ValidationError:
                    default_list = None
            return FlextUtilitiesConversion.to_str_list(value, default=default_list)
        if mode == "normalize":
            return FlextUtilitiesConversion.normalize(value, case=case)
        if mode == "join":
            raw_values: list[StrictJsonValue]
            try:
                raw_values = (
                    FlextUtilitiesConversion._strict_json_list_adapter.validate_python(
                        value,
                    )
                )
            except ValidationError as err:
                error_msg = "join mode requires Sequence"
                raise TypeError(error_msg) from err
            # Convert sequence items to strings for type safety
            # Strings are valid sequences (of characters)
            str_values: list[str] = [str(v) for v in raw_values]
            return FlextUtilitiesConversion.join(
                str_values,
                separator=separator,
                case=case,
            )
        error_msg = f"Unknown mode: {mode}"
        raise ValueError(error_msg)

    @staticmethod
    def to_general_value_type(value: StrictJsonValue) -> StrictJsonValue:
        """Return strict value without compatibility coercion."""
        return value

    @staticmethod
    def to_flexible_value(value: StrictJsonValue) -> StrictJsonScalar | None:
        """Convert strict value to strict scalar if compatible.

        Strict scalar is a subset of strict value that excludes
        BaseModel, Path, and Callable types.

        Args:
            value: strict value to convert

        Returns:
            strict scalar or None if not compatible

        """
        # Strict scalar = str | int | float | bool | datetime | None
        #                | Sequence[scalar] | Mapping[str, scalar]
        # where scalar = str | int | float | bool | datetime | None
        if value is None:
            return None
        if isinstance(value, BaseModel | Mapping | list | tuple | set | frozenset):
            return None
        if isinstance(value, datetime) and hasattr(value, "isoformat"):
            isoformat_method = value.isoformat
            if callable(isoformat_method):
                return str(value)
        try:
            return _StrictJsonScalarModel(value=value).value
        except ValidationError:
            return str(value)

    @staticmethod
    def to_str_list_safe(
        value: StrictJsonValue,
        *,
        filter_list_like: bool = True,
    ) -> list[str]:
        """Convert value to list[str] with safe nested list handling.

        Safely handles nested list-like structures by filtering them out
        to prevent nested lists in the returned result.

        Args:
            value: Value to convert
            filter_list_like: If True, filter out list-like items from result

        Returns:
            list[str]: List of string values

        Example:
            >>> u.Conversion.to_str_list_safe("foo")
            ["foo"]
            >>> u.Conversion.to_str_list_safe(["a", "b", ["nested"]])
            ["a", "b"]  # nested list filtered

        """
        if value is None:
            return []

        items: list[StrictJsonValue] = []
        value_class = value.__class__
        if value_class is str:
            items = [value]
        elif value_class is list:
            try:
                items = (
                    FlextUtilitiesConversion._strict_json_list_adapter.validate_python(
                        value,
                    )
                )
            except ValidationError:
                items = []
        else:
            items = [value]

        filtered_items: list[StrictJsonValue]
        if filter_list_like:
            filtered_items = [
                item
                for item in items
                if item is not None
                and not (
                    item.__class__ in {list, tuple, set, frozenset}
                    or (
                        item.__class__ in {list, tuple}
                        or (
                            hasattr(item, "__getitem__")
                            and item.__class__ not in {str, bytes}
                        )
                    )
                )
            ]
        else:
            filtered_items = [item for item in items if item is not None]

        return [str(item) for item in filtered_items]

    @staticmethod
    def to_str_list_truthy(
        value: StrictJsonValue,
    ) -> list[str]:
        """Convert value to list[str] filtering out falsy values.

        Converts value to list of strings while filtering out falsy
        (empty strings, None, etc.) values for cleaner results.

        Args:
            value: Value to convert

        Returns:
            list[str]: List of truthy string values

        Example:
            >>> u.Conversion.to_str_list_truthy(["a", "", "b", None])
            ["a", "b"]

        """
        result = FlextUtilitiesConversion.to_str_list_safe(value, filter_list_like=True)
        return [item for item in result if item]


__all__ = ["FlextUtilitiesConversion"]
