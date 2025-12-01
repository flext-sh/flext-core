"""FlextUtilitiesTypeGuards - Type Guard Utilities Module.

This module provides runtime type checking utilities for the FLEXT ecosystem,
implementing structural typing via FlextProtocols.TypeGuards (duck typing - no inheritance required).

Scope: Runtime type validation, type guards for strings, dictionaries, lists,
and other common types with consistent error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models.metadata import MetadataAttributeValue
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes


class FlextUtilitiesTypeGuards:
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
    def is_string_non_empty(value: FlextTypes.GeneralValueType) -> bool:
        """Check if value is a non-empty string using duck typing.

        Validates that the provided value is a string type and contains
        non-whitespace content after stripping.

        Args:
            value: Object to check for non-empty string type

        Returns:
            bool: True if value is non-empty string, False otherwise

        Example:
            >>> FlextUtilitiesTypeGuards.is_string_non_empty("hello")
            True
            >>> FlextUtilitiesTypeGuards.is_string_non_empty("   ")
            False
            >>> FlextUtilitiesTypeGuards.is_string_non_empty(123)
            False

        """
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def is_dict_non_empty(value: FlextTypes.GeneralValueType) -> bool:
        """Check if value is a non-empty dictionary using duck typing.

        Validates that the provided value behaves like a dictionary
        (has dict-like interface) and contains at least one item.

        Args:
            value: Object to check for non-empty dict-like type

        Returns:
            bool: True if value is non-empty dict-like, False otherwise

        Example:
            >>> FlextUtilitiesTypeGuards.is_dict_non_empty({"key": "value"})
            True
            >>> FlextUtilitiesTypeGuards.is_dict_non_empty({})
            False
            >>> FlextUtilitiesTypeGuards.is_dict_non_empty("not_a_dict")
            False

        """
        return FlextRuntime.is_dict_like(value) and bool(value)

    @staticmethod
    def is_list_non_empty(value: FlextTypes.GeneralValueType) -> bool:
        """Check if value is a non-empty list using duck typing.

        Validates that the provided value behaves like a list
        (has list-like interface) and contains at least one item.

        Args:
            value: Object to check for non-empty list-like type

        Returns:
            bool: True if value is non-empty list-like, False otherwise

        Example:
            >>> FlextUtilitiesTypeGuards.is_list_non_empty([1, 2, 3])
            True
            >>> FlextUtilitiesTypeGuards.is_list_non_empty([])
            False
            >>> FlextUtilitiesTypeGuards.is_list_non_empty("not_a_list")
            False

        """
        return FlextRuntime.is_list_like(value) and bool(value)

    @staticmethod
    def normalize_to_metadata_value(val: object) -> MetadataAttributeValue:
        """Normalize any value to MetadataAttributeValue.

        MetadataAttributeValue is more restrictive than FlextTypes.GeneralValueType,
        so we need to normalize nested structures to flat types.

        Args:
            val: Value to normalize

        Returns:
            MetadataAttributeValue: Normalized value compatible with Metadata attributes

        Example:
            >>> FlextUtilitiesTypeGuards.normalize_to_metadata_value("test")
            'test'
            >>> FlextUtilitiesTypeGuards.normalize_to_metadata_value({"key": "value"})
            {'key': 'value'}
            >>> FlextUtilitiesTypeGuards.normalize_to_metadata_value([1, 2, 3])
            [1, 2, 3]

        """
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        if FlextRuntime.is_dict_like(val):
            # Convert to flat dict[str, MetadataAttributeValue]
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
            # Convert to list[MetadataAttributeValue]
            result_list: list[str | int | float | bool | None] = []
            for item in val:
                if isinstance(item, (str, int, float, bool, type(None))):
                    result_list.append(item)
                else:
                    result_list.append(str(item))
            return result_list
        return str(val)


__all__ = ["FlextUtilitiesTypeGuards"]
