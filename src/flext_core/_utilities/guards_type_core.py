"""Core scalar and container type guards.

Provides type narrowing functions for scalar values, containers, flexible
values, and common validation patterns. All methods use TypeIs for proper
type inference and return boolean for guard patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import TypeIs

from flext_core import t


class FlextUtilitiesGuardsTypeCore:
    """Type guards for core scalar and container types.

    Provides utility methods for checking and narrowing types in the
    FLEXT type system, including scalars, containers, collections, and
    custom validators for domain-specific patterns.
    """

    @staticmethod
    def _object_sequence(
        value: t.GuardInput | t.RuntimeData | t.MetadataData | t.MetadataValue,
    ) -> TypeIs[Sequence[t.RuntimeData]]:
        """Check if value is a sequence (list or tuple)."""
        return isinstance(value, (list, tuple))

    @staticmethod
    def _object_mapping(
        value: t.GuardInput | t.RuntimeData | t.MetadataData | t.MetadataValue,
    ) -> TypeIs[Mapping[str, t.RuntimeData]]:
        """Check if value is a mapping type."""
        return isinstance(value, Mapping)

    @staticmethod
    def _all_container_sequence(
        value: Sequence[t.MetadataValue | t.RuntimeData],
    ) -> bool:
        """Check if all items in sequence are valid containers."""
        for sequence_item in value:
            if not FlextUtilitiesGuardsTypeCore.container(sequence_item):
                return False
        return True

    @staticmethod
    def all_container_mapping_values(
        value: Mapping[str, t.MetadataValue | t.RuntimeData],
    ) -> bool:
        """Check if all values in mapping are valid containers."""
        for mapped_value in value.values():
            if not FlextUtilitiesGuardsTypeCore.container(mapped_value):
                return False
        return True

    @staticmethod
    def dict_non_empty(value: t.GuardInput | None) -> bool:
        """Check if value is a non-empty mapping."""
        return bool(isinstance(value, Mapping) and value)

    @staticmethod
    def container(
        value: t.GuardInput | t.RuntimeData | t.MetadataData | t.MetadataValue | None,
    ) -> TypeIs[t.MetadataValue]:
        """Check if value is a valid container (recursive validation).

        Containers are scalars, paths, or JSON-compatible collections whose
        members recursively satisfy the metadata contract.
        """
        if value is None:
            return False
        if isinstance(value, t.CONTAINER_TYPES):
            return True
        if FlextUtilitiesGuardsTypeCore._object_sequence(value):
            return FlextUtilitiesGuardsTypeCore._all_container_sequence(value)
        if FlextUtilitiesGuardsTypeCore._object_mapping(value):
            return FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value)
        return False

    @staticmethod
    def list_value(
        value: t.GuardInput | t.RuntimeData | t.MetadataData | t.MetadataValue,
    ) -> TypeIs[Sequence[t.Container]]:
        """Check if value is a list."""
        return isinstance(value, list)

    @staticmethod
    def mapping(
        value: t.GuardInput | t.RuntimeData | t.MetadataData | t.MetadataValue,
    ) -> TypeIs[Mapping[str, t.Container]]:
        """Check if value is a mapping type."""
        return isinstance(value, Mapping)

    @staticmethod
    def primitive(
        value: t.GuardInput | t.RuntimeData | t.MetadataData | t.MetadataValue,
    ) -> TypeIs[t.Primitives]:
        """Check if value is a primitive type (str, int, float, bool)."""
        return isinstance(value, (str, int, float, bool))

    @staticmethod
    def scalar(
        value: t.GuardInput | t.RuntimeData | t.MetadataData | t.MetadataValue,
    ) -> TypeIs[t.Scalar]:
        """Check if value is a scalar type (str, int, float, bool, datetime)."""
        return isinstance(value, t.SCALAR_TYPES)

    @staticmethod
    def _has_dict_protocol(
        obj: t.GuardInput | t.RuntimeData | t.MetadataData | t.MetadataValue,
    ) -> bool:
        if not isinstance(obj, Mapping):
            return False
        try:
            items_fn = getattr(obj, "items", None)
            if items_fn is not None and callable(items_fn):
                items_fn()
                return True
        except (AttributeError, TypeError):
            return False
        return False

    @staticmethod
    def dict_like(
        value: t.GuardInput | t.RuntimeData | t.MetadataData | t.MetadataValue,
    ) -> TypeIs[Mapping[str, t.RuntimeData]]:
        """Check if value behaves like a mapping accepted by FLEXT containers."""
        if isinstance(value, Mapping):
            return True
        return bool(FlextUtilitiesGuardsTypeCore._has_dict_protocol(value))

    @staticmethod
    def list_like(
        value: t.GuardInput | t.RuntimeData | t.MetadataData | t.MetadataValue,
    ) -> TypeIs[Sequence[t.RuntimeData]]:
        """Check if value behaves like a non-string object sequence."""
        return isinstance(value, (list, tuple)) and not isinstance(
            value,
            (str, bytes),
        )

    @staticmethod
    def string_non_empty(value: t.GuardInput) -> TypeIs[str]:
        """Check if value is a non-empty string (after stripping whitespace)."""
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def instance_of[T](value: t.GuardInput | T, type_cls: type[T]) -> bool:
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


__all__: list[str] = ["FlextUtilitiesGuardsTypeCore"]
