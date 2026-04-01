"""Core scalar and container type guards.

Provides type narrowing functions for scalar values, containers, flexible
values, and common validation patterns. All methods use TypeIs for proper
type inference and return boolean for guard patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeIs

from flext_core.typings import t


class FlextUtilitiesGuardsTypeCore:
    """Type guards for core scalar and container types.

    Provides utility methods for checking and narrowing types in the
    FLEXT type system, including scalars, containers, collections, and
    custom validators for domain-specific patterns.
    """

    @staticmethod
    def _is_object_sequence(
        value: t.GuardInput,
    ) -> TypeIs[t.ContainerList]:
        """Check if value is a sequence (list or tuple)."""
        return isinstance(value, (list, tuple))

    @staticmethod
    def _is_object_mapping(
        value: t.GuardInput,
    ) -> TypeIs[t.ContainerMapping]:
        """Check if value is a mapping type."""
        return isinstance(value, Mapping)

    @staticmethod
    def _all_container_sequence(value: Sequence[t.GuardInput]) -> bool:
        """Check if all items in sequence are valid containers."""
        for sequence_item in value:
            if not FlextUtilitiesGuardsTypeCore.is_container(sequence_item):
                return False
        return True

    @staticmethod
    def all_container_mapping_values(value: Mapping[str, t.GuardInput]) -> bool:
        """Check if all values in mapping are valid containers."""
        for mapped_value in value.values():
            if not FlextUtilitiesGuardsTypeCore.is_container(mapped_value):
                return False
        return True

    @staticmethod
    def is_dict_non_empty(value: t.ValueOrModel) -> bool:
        """Check if value is a non-empty mapping."""
        return bool(isinstance(value, Mapping) and value)

    @staticmethod
    def is_container(
        value: t.GuardInput,
    ) -> TypeIs[t.Container]:
        """Check if value is a valid container (recursive validation).

        Containers are None, scalars, Paths, or collections where all items
        recursively satisfy is_container. Used to validate nested data structures.
        """
        if value is None or isinstance(value, t.SCALAR_TYPES):
            return True
        if FlextUtilitiesGuardsTypeCore._is_object_sequence(value):
            return FlextUtilitiesGuardsTypeCore._all_container_sequence(value)
        if FlextUtilitiesGuardsTypeCore._is_object_mapping(value):
            return FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value)
        return isinstance(value, Path)

    @staticmethod
    def is_list(value: t.GuardInput) -> TypeIs[t.ContainerList]:
        """Check if value is a list."""
        return isinstance(value, list)

    @staticmethod
    def is_mapping(
        value: t.GuardInput,
    ) -> TypeIs[t.ContainerMapping]:
        """Check if value is a mapping type."""
        return isinstance(value, Mapping)

    @staticmethod
    def is_primitive(
        value: t.GuardInput,
    ) -> TypeIs[t.Primitives]:
        """Check if value is a primitive type (str, int, float, bool)."""
        return isinstance(value, (str, int, float, bool))

    @staticmethod
    def is_scalar(
        value: t.GuardInput,
    ) -> TypeIs[t.Scalar]:
        """Check if value is a scalar type (str, int, float, bool, datetime)."""
        return isinstance(value, t.SCALAR_TYPES)

    @staticmethod
    def is_string_non_empty(value: t.GuardInput) -> TypeIs[str]:
        """Check if value is a non-empty string (after stripping whitespace)."""
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def is_instance_of[T](value: object, type_cls: type[T]) -> TypeIs[T]:
        """Check if value is instance of type class (handles generics)."""
        return isinstance(value, getattr(type_cls, "__origin__", None) or type_cls)

    @staticmethod
    def require_initialized[T](value: T | None, name: str) -> T:
        """Require that a value is initialized (not None).

        Raises:
            AttributeError: If value is None.

        """
        if value is None:
            msg = f"{name} is not initialized"
            raise AttributeError(msg)
        return value

    @staticmethod
    def in_(value: t.GuardInput, container: t.GuardInput) -> bool:
        """Check if value is in container, handling TypeError gracefully."""
        if isinstance(container, (list, tuple, set, dict)):
            try:
                return value in container
            except TypeError:
                return False
        return False


__all__ = ["FlextUtilitiesGuardsTypeCore"]
