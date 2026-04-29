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
from typing import TypeIs, no_type_check

from pydantic import BaseModel as PydanticBaseModel

from flext_core import FlextTypes as t


@no_type_check
class FlextUtilitiesGuardsTypeCore:
    """Type guards for core scalar and container types.

    Provides utility methods for checking and narrowing types in the
    FLEXT type system, including scalars, containers, collections, and
    custom validators for domain-specific patterns.
    """

    @staticmethod
    def _object_sequence(
        value: t.GuardInput | t.JsonPayload | t.JsonValue | PydanticBaseModel,
    ) -> TypeIs[Sequence[t.JsonPayload]]:
        """Check if value is a sequence (list or tuple)."""
        return isinstance(value, (list, tuple))

    @staticmethod
    def _object_mapping(
        value: t.GuardInput | t.JsonPayload | t.JsonValue | PydanticBaseModel,
    ) -> TypeIs[Mapping[str, t.JsonPayload]]:
        """Check if value is a mapping type."""
        return isinstance(value, Mapping)

    @staticmethod
    def _all_container_sequence(
        value: Sequence[t.JsonValue | t.JsonPayload],
    ) -> bool:
        """Check if all items in sequence are valid containers."""
        for sequence_item in value:
            if not FlextUtilitiesGuardsTypeCore.container(sequence_item):
                return False
        return True

    @staticmethod
    def all_container_mapping_values(
        value: Mapping[str, t.JsonValue | t.JsonPayload],
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
    def empty_value(
        value: t.GuardInput | t.JsonPayload | t.JsonValue | None,
    ) -> bool:
        """Check whether a FLEXT value is absent or an empty text/container."""
        if value is None:
            return True
        if isinstance(value, (str, bytes, bytearray, Mapping)):
            return not value
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return not value
        return False

    @staticmethod
    def container(
        value: t.GuardInput | t.JsonPayload | t.JsonValue | PydanticBaseModel | None,
    ) -> TypeIs[t.JsonValue]:
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
        value: t.GuardInput | t.JsonPayload | t.JsonValue,
    ) -> TypeIs[t.JsonList]:
        """Check if value is a list."""
        return isinstance(value, list)

    @staticmethod
    def mapping(
        value: t.GuardInput | t.JsonPayload | t.JsonValue,
    ) -> TypeIs[t.JsonMapping]:
        """Check if value is a mapping type."""
        return isinstance(value, Mapping)

    @staticmethod
    def primitive(
        value: t.GuardInput | t.JsonPayload | t.JsonValue,
    ) -> TypeIs[t.Primitives]:
        """Check if value is a primitive type (str, int, float, bool)."""
        return isinstance(value, (str, int, float, bool))

    @staticmethod
    def scalar(
        value: t.GuardInput | t.Scalar | t.JsonPayload | t.JsonValue,
    ) -> TypeIs[t.Scalar]:
        """Check if value is a scalar type (str, int, float, bool, datetime)."""
        return isinstance(value, t.SCALAR_TYPES)

    @staticmethod
    def type_name(
        value: t.GuardInput | t.JsonPayload | t.JsonValue | None,
    ) -> str:
        """Return the concrete runtime type name for any FLEXT payload value."""
        return str(type(value).__name__)

    @staticmethod
    def _has_dict_protocol(
        obj: t.GuardInput | t.JsonPayload | t.JsonValue,
    ) -> bool:
        return isinstance(obj, Mapping)

    @staticmethod
    def dict_like(
        value: t.GuardInput | t.JsonPayload | t.JsonValue,
    ) -> TypeIs[Mapping[str, t.JsonPayload]]:
        """Check if value behaves like a mapping accepted by FLEXT containers."""
        if isinstance(value, Mapping):
            return True
        return FlextUtilitiesGuardsTypeCore._has_dict_protocol(value)

    @staticmethod
    def list_like(
        value: t.GuardInput | t.JsonPayload | t.JsonValue,
    ) -> TypeIs[Sequence[t.JsonPayload]]:
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
    def in_(value: t.GuardInput, container: t.GuardInput) -> bool:
        """Check if value is in container, handling TypeError gracefully."""
        if isinstance(container, (list, tuple, set, dict)):
            try:
                return value in container
            except TypeError:
                return False
        return False


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesGuardsTypeCore"]
