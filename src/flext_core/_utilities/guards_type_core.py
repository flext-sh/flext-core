"""Core scalar and container type guards.

Provides scalar narrowing, SDK shape inspection and native container predicates.
Shape classification does not establish JSON or domain-content validity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from typing import TYPE_CHECKING, TypeGuard, TypeIs

import flext_core

if TYPE_CHECKING:
    pass


class FlextUtilitiesGuardsTypeCore:
    """Type guards for core scalar and container types.

    Provides utility methods for checking and narrowing types in the
    FLEXT type system, including scalars, containers, collections, and
    custom validators for domain-specific patterns.
    """

    @staticmethod
    def _object_sequence(
        value: flext_core.p.AttributeProbe,
    ) -> TypeGuard[Sequence[flext_core.p.AttributeProbe]]:
        """Inspect a native list/tuple shape without validating its contents.

        Returns:
            Whether the opaque SDK value is a native list or tuple.

        """
        return isinstance(value, flext_core.c.SEQUENCE_PAIR_TYPES)

    @staticmethod
    def _all_container_sequence[ItemT](
        value: Sequence[ItemT],
    ) -> bool:
        """Check if all items in sequence are valid containers.

        Returns:
            The resulting ``bool``.

        """
        for sequence_item in value:
            if not FlextUtilitiesGuardsTypeCore.container_value(sequence_item):
                return False
        return True

    @staticmethod
    def all_container_mapping_values[KeyT, ValueT](
        value: Mapping[KeyT, ValueT],
    ) -> bool:
        """Check if all values in mapping are valid containers.

        Returns:
            The resulting ``bool``.

        """
        for mapped_value in value.values():
            if not FlextUtilitiesGuardsTypeCore.container_value(mapped_value):
                return False
        return True

    @staticmethod
    def dict_non_empty(value: flext_core.p.GuardInput | None) -> bool:
        """Check if value is a non-empty mapping.

        Returns:
            The resulting ``bool``.

        """
        return bool(FlextUtilitiesGuardsTypeCore.mapping_view(value) and value)

    @staticmethod
    def empty_value(
        value: flext_core.p.GuardInput
        | flext_core.t.JsonPayload
        | flext_core.t.JsonValue
        | None,
    ) -> bool:
        """Check whether a FLEXT value is absent or an empty text/container.

        Returns:
            The resulting ``bool``.

        """
        if value is None:
            return True
        if isinstance(value, (str, bytes, bytearray, Mapping)):
            return not value
        if isinstance(value, Sequence) and not isinstance(
            value, flext_core.c.STR_BINARY_TYPES
        ):
            return not value
        return False

    @staticmethod
    def container_value(value: flext_core.p.AttributeProbe) -> bool:
        """Check recursive native container capability without asserting JSON.

        Scalars (including datetime), paths, lists, tuples and mapping values
        retain their native behavior. None is rejected at every depth; mapping
        keys are not inspected. Accepted data is not converted or narrowed.

        Returns:
            Whether all values recursively satisfy the native container predicate.

        """
        if value is None:
            return False
        if isinstance(value, flext_core.c.CONTAINER_TYPES):
            return True
        if FlextUtilitiesGuardsTypeCore._object_sequence(value):
            return FlextUtilitiesGuardsTypeCore._all_container_sequence(value)
        if FlextUtilitiesGuardsTypeCore.mapping_view(value):
            return FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value)
        return False

    @staticmethod
    def list_value(
        value: flext_core.p.GuardInput
        | flext_core.t.JsonPayload
        | flext_core.t.JsonValue,
    ) -> bool:
        """Classify a native list without validating its items as JSON.

        Returns:
            Whether the value is a native list.

        """
        return isinstance(value, list)

    @staticmethod
    def mapping(
        value: flext_core.p.GuardInput
        | flext_core.t.JsonPayload
        | flext_core.t.JsonValue,
    ) -> bool:
        """Classify a native mapping without validating its keys or values.

        Returns:
            Whether the value is a native mapping.

        """
        return isinstance(value, Mapping)

    @staticmethod
    def mapping_view(
        value: flext_core.p.AttributeProbe,
    ) -> TypeIs[Mapping[flext_core.p.AttributeProbe, flext_core.p.AttributeProbe]]:
        """Refine an opaque SDK value to its native read-only mapping shape.

        Does not inspect keys or contents, require string keys, or validate JSON
        or domain data. Membership and item access retain their native behavior.

        Returns:
            Whether the opaque value is a native mapping.

        """
        return isinstance(value, Mapping)

    @staticmethod
    def subclass_view[BaseT](
        candidate: flext_core.p.AttributeProbe,
        base: type[BaseT],
    ) -> TypeGuard[type[BaseT]]:
        """Inspect an opaque SDK value as a native subclass without constructing it.

        Returns:
            Whether the value is a runtime class derived from the given base.

        """
        return isinstance(candidate, type) and issubclass(candidate, base)

    @staticmethod
    def primitive(
        value: flext_core.p.GuardInput
        | flext_core.t.JsonPayload
        | flext_core.t.JsonValue,
    ) -> TypeIs[flext_core.t.Primitives]:
        """Check if value is a primitive type (c.PRIMITIVES_TYPES).

        Returns:
            The resulting ``TypeIs[t.Primitives]``.

        """
        return isinstance(value, flext_core.c.PRIMITIVES_TYPES)

    @staticmethod
    def scalar(
        value: flext_core.p.GuardInput
        | flext_core.t.Scalar
        | flext_core.t.JsonPayload
        | flext_core.t.JsonValue,
    ) -> TypeIs[flext_core.t.Scalar]:
        """Check if value is a scalar type (str, int, float, bool, datetime).

        Returns:
            The resulting ``TypeIs[t.Scalar]``.

        """
        return isinstance(value, flext_core.c.SCALAR_TYPES)

    @staticmethod
    def type_name(
        value: flext_core.p.GuardInput
        | flext_core.t.JsonPayload
        | flext_core.t.JsonValue
        | None,
    ) -> str:
        """Return the concrete runtime type name for any FLEXT payload value.

        Returns:
            The concrete runtime type name for any FLEXT payload value.

        """
        return type(value).__qualname__

    @staticmethod
    def dict_like(
        value: flext_core.p.GuardInput
        | flext_core.t.JsonPayload
        | flext_core.t.JsonValue,
    ) -> bool:
        """Classify the native mapping shape without claiming payload validity.

        Returns:
            Whether the value is a native mapping.

        """
        return isinstance(value, Mapping)

    @staticmethod
    def list_like(
        value: flext_core.p.GuardInput
        | flext_core.t.JsonPayload
        | flext_core.t.JsonValue,
    ) -> bool:
        """Classify native lists/tuples without validating their items as JSON.

        Returns:
            Whether the value is a native non-string list or tuple.

        """
        return isinstance(value, flext_core.c.SEQUENCE_PAIR_TYPES) and not isinstance(
            value,
            flext_core.c.STR_BYTES_TYPES,
        )

    @staticmethod
    def string_non_empty(value: flext_core.p.GuardInput) -> TypeGuard[str]:
        """Check if value is a non-empty string (after stripping whitespace).

        Returns:
            The resulting ``TypeGuard[str]``.

        """
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def instance_of[T](value: flext_core.p.GuardInput | T, type_cls: type[T]) -> bool:
        """Check if value is instance of type class (handles generics).

        Returns:
            The resulting ``bool``.

        """
        return isinstance(value, getattr(type_cls, "__origin__", None) or type_cls)

    @staticmethod
    def in_(
        value: flext_core.p.GuardInput,
        container: flext_core.p.GuardInput,
    ) -> bool:
        """Check if value is in container, handling TypeError gracefully.

        Returns:
            The resulting ``bool``.

        """
        if isinstance(container, (list, tuple, set, dict)):
            contains: bool
            try:
                contains = value in container
            except TypeError:
                contains = False
            return contains
        return False


__all__: MutableSequence[str] = ["FlextUtilitiesGuardsTypeCore"]
