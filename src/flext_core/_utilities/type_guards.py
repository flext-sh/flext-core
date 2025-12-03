"""Runtime type guard helpers for dispatcher-safe validations.

The utilities rely on structural typing (via ``p.TypeGuards``)
to keep handler and service checks lightweight while staying compatible with
duck-typed inputs used throughout the CQRS pipeline.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextTypeGuards:
    """Runtime type checking utilities for FLEXT ecosystem.

    Provides type guard functions for common validation patterns used throughout
    the FLEXT framework, implementing structural typing for duck-typed interfaces.

    Core Features:
    - String validation guards (non-empty, etc.)
    - Collection validation guards (dict, list)
    - Type-safe runtime checking
    - Consistent error handling patterns
    - Metadata value normalization
    """

    @staticmethod
    def is_string_non_empty(value: t.GeneralValueType) -> bool:
        """Check if value is a non-empty string using duck typing.

        Validates that the provided value is a string type and contains
        non-whitespace content after stripping.

        Args:
            value: Object to check for non-empty string type

        Returns:
            bool: True if value is non-empty string, False otherwise

        Example:
            >>> uTypeGuards.is_string_non_empty("hello")
            True
            >>> uTypeGuards.is_string_non_empty("   ")
            False
            >>> uTypeGuards.is_string_non_empty(123)
            False

        """
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def is_dict_non_empty(value: t.GeneralValueType) -> bool:
        """Check if value is a non-empty dictionary using duck typing.

        Validates that the provided value behaves like a dictionary
        (has dict-like interface) and contains at least one item.

        Args:
            value: Object to check for non-empty dict-like type

        Returns:
            bool: True if value is non-empty dict-like, False otherwise

        Example:
            >>> uTypeGuards.is_dict_non_empty({"key": "value"})
            True
            >>> uTypeGuards.is_dict_non_empty({})
            False
            >>> uTypeGuards.is_dict_non_empty("not_a_dict")
            False

        """
        return FlextRuntime.is_dict_like(value) and bool(value)

    @staticmethod
    def is_list_non_empty(value: t.GeneralValueType) -> bool:
        """Check if value is a non-empty list using duck typing.

        Validates that the provided value behaves like a list
        (has list-like interface) and contains at least one item.

        Args:
            value: Object to check for non-empty list-like type

        Returns:
            bool: True if value is non-empty list-like, False otherwise

        Example:
            >>> uTypeGuards.is_list_non_empty([1, 2, 3])
            True
            >>> uTypeGuards.is_list_non_empty([])
            False
            >>> uTypeGuards.is_list_non_empty("not_a_list")
            False

        """
        return FlextRuntime.is_list_like(value) and bool(value)

    @staticmethod
    def normalize_to_metadata_value(
        val: t.GeneralValueType,
    ) -> t.MetadataAttributeValue:
        """Normalize any value to MetadataAttributeValue.

        MetadataAttributeValue is more restrictive than t.GeneralValueType,
        so we need to normalize nested structures to flat types.

        Args:
            val: Value to normalize

        Returns:
            t.MetadataAttributeValue: Normalized value compatible with Metadata attributes

        Example:
            >>> uTypeGuards.normalize_to_metadata_value("test")
            'test'
            >>> uTypeGuards.normalize_to_metadata_value({"key": "value"})
            {'key': 'value'}
            >>> uTypeGuards.normalize_to_metadata_value([1, 2, 3])
            [1, 2, 3]

        """
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        if FlextRuntime.is_dict_like(val):
            # Convert to flat dict[str, t.MetadataAttributeValue]
            result: dict[str, str | int | float | bool | None] = {}
            dict_v = dict(val.items()) if hasattr(val, "items") else dict(val)
            for k, v in dict_v.items():
                if isinstance(k, str):
                    if isinstance(v, (str, int, float, bool, type(None))):
                        result[k] = v
                    else:
                        result[k] = str(v)
            return result
        if FlextRuntime.is_list_like(val):
            # Convert to list[t.MetadataAttributeValue]
            result_list: list[str | int | float | bool | None] = []
            for item in val:
                if isinstance(item, (str, int, float, bool, type(None))):
                    result_list.append(item)
                else:
                    result_list.append(str(item))
            return result_list
        return str(val)


uTypeGuards = FlextTypeGuards  # noqa: N816

__all__ = [
    "FlextTypeGuards",
    "uTypeGuards",
]
