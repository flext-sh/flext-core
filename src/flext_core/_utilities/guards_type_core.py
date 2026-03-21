"""Core scalar and container type guards.

Provides type narrowing functions for scalar values, containers, flexible
values, and common validation patterns. All methods use TypeIs for proper
type inference and return boolean for guard patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeIs

from flext_core import t


class FlextUtilitiesGuardsTypeCore:
    """Type guards for core scalar and container types.

    Provides utility methods for checking and narrowing types in the
    FLEXT type system, including scalars, containers, collections, and
    custom validators for domain-specific patterns.
    """

    @staticmethod
    def _is_object_sequence(value: object) -> TypeIs[Sequence[object]]:
        """Check if value is a sequence (list or tuple).

        Args:
            value: Object to check.

        Returns:
            True if value is a list or tuple, narrowed to Sequence[object].

        """
        return isinstance(value, (list, tuple))

    @staticmethod
    def _is_object_mapping(value: object) -> TypeIs[Mapping[str, object]]:
        """Check if value is a mapping type.

        Args:
            value: Object to check.

        Returns:
            True if value is a Mapping, narrowed to Mapping[str, object].

        """
        return isinstance(value, Mapping)

    @staticmethod
    def _all_container_sequence(value: Sequence[object]) -> bool:
        """Check if all items in sequence are valid containers.

        Args:
            value: Sequence of objects to validate.

        Returns:
            True if all items are containers, False otherwise.

        """
        for sequence_item in value:
            if not FlextUtilitiesGuardsTypeCore.is_container(sequence_item):
                return False
        return True

    @staticmethod
    def all_container_mapping_values(value: Mapping[str, object]) -> bool:
        """Check if all values in mapping are valid containers.

        Args:
            value: Mapping with object values to validate.

        Returns:
            True if all values are containers, False otherwise.

        """
        for mapped_value in value.values():
            if not FlextUtilitiesGuardsTypeCore.is_container(mapped_value):
                return False
        return True

    @staticmethod
    def is_dict_non_empty(value: t.NormalizedValue) -> bool:
        """Check if value is a non-empty mapping.

        Args:
            value: Value to check.

        Returns:
            True if value is a Mapping with at least one entry.

        """
        return isinstance(value, Mapping) and bool(value)

    @staticmethod
    def is_flexible_value(value: t.NormalizedValue) -> TypeIs[t.NormalizedValue]:
        """Check if value is a flexible value (scalar or containers of scalars).

        Flexible values are None, scalars, or collections containing only
        scalars and None values. Used for configuration values and simple data.

        Args:
            value: Value to check.

        Returns:
            True if value is a flexible value, narrowed to NormalizedValue.

        """
        if value is None or FlextUtilitiesGuardsTypeCore.is_scalar(value):
            return True
        if isinstance(value, (list, tuple)):
            return all(
                item is None or FlextUtilitiesGuardsTypeCore.is_scalar(item)
                for item in value
            )
        if isinstance(value, Mapping):
            return all(
                item is None or FlextUtilitiesGuardsTypeCore.is_scalar(item)
                for item in value.values()
            )
        return False

    @staticmethod
    def is_container(
        value: object,
    ) -> TypeIs[t.Container]:
        """Check if value is a valid container (recursive validation).

        Containers are None, scalars, Paths, or collections where all items
        recursively satisfy is_container. Used to validate nested data structures.

        Args:
            value: Value to check.

        Returns:
            True if value is a valid container, narrowed to Container.

        """
        if value is None or isinstance(value, t.SCALAR_TYPES):
            return True
        if FlextUtilitiesGuardsTypeCore._is_object_sequence(value):
            return FlextUtilitiesGuardsTypeCore._all_container_sequence(value)
        if FlextUtilitiesGuardsTypeCore._is_object_mapping(value):
            return FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value)
        return isinstance(value, Path)

    @staticmethod
    def is_general_value_type(value: t.NormalizedValue) -> bool:
        """Check if value is callable or container (deprecated).

        Deprecated in favor of is_container. Will be removed in v0.12.

        Args:
            value: Value to check.

        Returns:
            True if value is callable or valid container.

        Raises:
            DeprecationWarning: This method is deprecated.

        """
        warnings.warn(
            "is_general_value_type is deprecated; use is_container. Planned removal: v0.12.",
            DeprecationWarning,
            stacklevel=2,
        )
        return callable(value) or FlextUtilitiesGuardsTypeCore.is_container(value)

    @staticmethod
    def is_list(value: t.NormalizedValue) -> TypeIs[list[t.NormalizedValue]]:
        """Check if value is a list.

        Args:
            value: Value to check.

        Returns:
            True if value is a list, narrowed to list[NormalizedValue].

        """
        return isinstance(value, list)

    @staticmethod
    def is_list_non_empty(value: t.NormalizedValue) -> bool:
        """Check if value is a non-empty sequence (excluding strings/bytes).

        Args:
            value: Value to check.

        Returns:
            True if value is a non-string sequence with at least one item.

        """
        return (
            isinstance(value, Sequence)
            and (not isinstance(value, (str, bytes)))
            and len(value) > 0
        )

    @staticmethod
    def is_mapping(
        value: object,
    ) -> TypeIs[Mapping[str, t.NormalizedValue]]:
        """Check if value is a mapping type.

        Args:
            value: Value to check.

        Returns:
            True if value is a Mapping, narrowed to Mapping[str, NormalizedValue].

        """
        return isinstance(value, Mapping)

    @staticmethod
    def is_primitive(
        value: object,
    ) -> TypeIs[t.Primitives]:
        """Check if value is a primitive type (str, int, float, bool).

        Args:
            value: Value to check.

        Returns:
            True if value is primitive, narrowed to Primitives.

        """
        return isinstance(value, (str, int, float, bool))

    @staticmethod
    def is_scalar(
        value: object,
    ) -> TypeIs[t.Scalar]:
        """Check if value is a scalar type (str, int, float, bool, datetime).

        Args:
            value: Value to check.

        Returns:
            True if value is a scalar, narrowed to Scalar.

        """
        return isinstance(value, t.SCALAR_TYPES)

    @staticmethod
    def is_string_non_empty(value: t.NormalizedValue) -> TypeIs[str]:
        """Check if value is a non-empty string (after stripping whitespace).

        Args:
            value: Value to check.

        Returns:
            True if value is a string with non-whitespace content, narrowed to str.

        """
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def is_instance_of[T](value: object, type_cls: type[T]) -> TypeIs[T]:
        """Check if value is instance of type class (handles generics).

        Args:
            value: Value to check.
            type_cls: Type class to check against (may be generic).

        Returns:
            True if value is instance of type_cls, narrowed to T.

        """
        return isinstance(value, getattr(type_cls, "__origin__", None) or type_cls)

    @staticmethod
    def require_initialized[T](value: T | None, name: str) -> T:
        """Require that a value is initialized (not None).

        Args:
            value: Value that should be initialized.
            name: Name of the value (for error message).

        Returns:
            The non-None value.

        Raises:
            AttributeError: If value is None.

        """
        if value is None:
            msg = f"{name} is not initialized"
            raise AttributeError(msg)
        return value

    @staticmethod
    def has(obj: t.NormalizedValue, key: str) -> bool:
        """Check if object has key or attribute.

        Uses dict subscript for dicts, hasattr for other objects.

        Args:
            obj: Object to check (dict or object with attributes).
            key: Key or attribute name to check.

        Returns:
            True if object has the key or attribute.

        """
        return key in obj if isinstance(obj, dict) else hasattr(obj, key)

    @staticmethod
    def in_(value: t.NormalizedValue, container: t.NormalizedValue) -> bool:
        """Check if value is in container, handling TypeError gracefully.

        Args:
            value: Value to find.
            container: Container to search in (list, tuple, set, or dict).

        Returns:
            True if value is in container, False if not or container type unsupported.

        """
        if isinstance(container, (list, tuple, set, dict)):
            try:
                return value in container
            except TypeError:
                return False
        return False

    @staticmethod
    def none_(*values: t.NormalizedValue) -> bool:
        """Check if all provided values are None.

        Args:
            *values: Variable number of values to check.

        Returns:
            True if all values are None, False otherwise.

        """
        return all(v is None for v in values)


__all__ = ["FlextUtilitiesGuardsTypeCore"]
