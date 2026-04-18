"""Core scalar and container type guards.

Provides type narrowing functions for scalar values, containers, flexible
values, and common validation patterns. All methods use TypeIs for proper
type inference and return boolean for guard patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, TypeIs

from flext_core import t


class FlextUtilitiesGuardsTypeCore:
    """Type guards for core scalar and container types.

    Provides utility methods for checking and narrowing types in the
    FLEXT type system, including scalars, containers, collections, and
    custom validators for domain-specific patterns.
    """

    @staticmethod
    def _object_sequence(
        value: t.GuardInput,
    ) -> TypeIs[t.RecursiveContainerList]:
        """Check if value is a sequence (list or tuple)."""
        return isinstance(value, (list, tuple))

    @staticmethod
    def _object_mapping(
        value: t.GuardInput,
    ) -> TypeIs[t.RecursiveContainerMapping]:
        """Check if value is a mapping type."""
        return isinstance(value, Mapping)

    @staticmethod
    def _all_container_sequence(value: Sequence[t.GuardInput]) -> bool:
        """Check if all items in sequence are valid containers."""
        for sequence_item in value:
            if not FlextUtilitiesGuardsTypeCore.container(sequence_item):
                return False
        return True

    @staticmethod
    def all_container_mapping_values(value: Mapping[str, t.GuardInput]) -> bool:
        """Check if all values in mapping are valid containers."""
        for mapped_value in value.values():
            if not FlextUtilitiesGuardsTypeCore.container(mapped_value):
                return False
        return True

    @staticmethod
    def dict_non_empty(value: t.ValueOrModel) -> bool:
        """Check if value is a non-empty mapping."""
        return bool(isinstance(value, Mapping) and value)

    @staticmethod
    def container(
        value: t.GuardInput,
    ) -> TypeIs[t.Container]:
        """Check if value is a valid container (recursive validation).

        Containers are None, scalars, Paths, or collections where all items
        recursively satisfy container. Used to validate nested data structures.
        """
        if value is None or isinstance(value, t.SCALAR_TYPES):
            return True
        if FlextUtilitiesGuardsTypeCore._object_sequence(value):
            return FlextUtilitiesGuardsTypeCore._all_container_sequence(value)
        if FlextUtilitiesGuardsTypeCore._object_mapping(value):
            return FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value)
        return isinstance(value, Path)

    @staticmethod
    def list_value(value: t.GuardInput) -> TypeIs[t.RecursiveContainerList]:
        """Check if value is a list."""
        return isinstance(value, list)

    @staticmethod
    def mapping(
        value: t.GuardInput,
    ) -> TypeIs[t.RecursiveContainerMapping]:
        """Check if value is a mapping type."""
        return isinstance(value, Mapping)

    @staticmethod
    def primitive(
        value: t.GuardInput,
    ) -> TypeIs[t.Primitives]:
        """Check if value is a primitive type (str, int, float, bool)."""
        return isinstance(value, (str, int, float, bool))

    @staticmethod
    def scalar(
        value: t.GuardInput,
    ) -> TypeIs[t.Scalar]:
        """Check if value is a scalar type (str, int, float, bool, datetime)."""
        return isinstance(value, t.SCALAR_TYPES)

    @staticmethod
    def _has_dict_protocol(obj: t.RuntimeData) -> bool:
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
        value: t.ConfigMap | t.RuntimeData,
    ) -> TypeIs[t.ConfigMap | t.RecursiveContainerMapping]:
        """Check if value behaves like a mapping accepted by FLEXT containers."""
        match value:
            case t.ConfigMap():
                return True
            case Mapping():
                return True
            case _:
                if value is None:
                    return False
                return bool(FlextUtilitiesGuardsTypeCore._has_dict_protocol(value))

    @staticmethod
    def list_like(
        value: t.RuntimeData,
    ) -> TypeIs[t.RecursiveContainerList]:
        """Check if value behaves like a non-string object sequence."""
        return isinstance(value, (list, tuple)) and not isinstance(
            value,
            (str, bytes),
        )

    @staticmethod
    def extract_generic_args(
        type_hint: t.TypeHintSpecifier,
    ) -> tuple[t.GenericTypeArgument | type, ...]:
        """Extract generic type arguments from a type hint."""
        try:
            resolved_hint = getattr(type_hint, "__value__", type_hint)
            args = typing.get_args(resolved_hint)
            if args:
                return args
            type_name = getattr(type_hint, "__name__", "")
            if not type_name:
                type_name = getattr(resolved_hint, "__name__", "")
            if not type_name:
                return ()
            return (
                FlextUtilitiesGuardsTypeCore._GENERIC_LIST_ALIASES.get(type_name)
                or FlextUtilitiesGuardsTypeCore._GENERIC_DICT_ALIASES.get(type_name)
                or ()
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return ()

    @staticmethod
    def _sequence_type_class(candidate: type) -> bool:
        candidate_name = getattr(candidate, "__name__", "")
        if candidate_name in {"list", "tuple", "range"}:
            return True
        if candidate_name in {"str", "bytes", "bytearray", "memoryview", "dict"}:
            return False
        candidate_mro = getattr(candidate, "__mro__", ())
        if any(getattr(base, "__name__", "") == "Sequence" for base in candidate_mro):
            return True
        required_members = ("__iter__", "__len__", "__getitem__", "count", "index")
        return all(hasattr(candidate, member) for member in required_members)

    @staticmethod
    def sequence_type(type_hint: t.TypeHintSpecifier) -> bool:
        """Check if a type hint represents a list-like sequence type."""
        try:
            origin = typing.get_origin(type_hint)
            if isinstance(origin, type):
                if origin in {list, tuple}:
                    return True
                return FlextUtilitiesGuardsTypeCore._sequence_type_class(origin)
            if type_hint in {list, tuple, str}:
                return True
            if isinstance(
                type_hint,
                type,
            ) and FlextUtilitiesGuardsTypeCore._sequence_type_class(type_hint):
                return True
            if isinstance(type_hint, type) and getattr(type_hint, "__name__", "") in {
                "StringList",
                "IntList",
                "FloatList",
                "BoolList",
                "List",
            }:
                return True
            if not isinstance(type_hint, str):
                return False
            return type_hint in {
                "StringList",
                "IntList",
                "FloatList",
                "BoolList",
                "List",
            }
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return False

    @staticmethod
    def valid_identifier(value: t.RuntimeData) -> TypeIs[str]:
        """Check if value is a valid Python identifier string."""
        return isinstance(value, str) and value.isidentifier()

    @staticmethod
    def string_non_empty(value: t.GuardInput) -> TypeIs[str]:
        """Check if value is a non-empty string (after stripping whitespace)."""
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def instance_of[T](value: object, type_cls: type[T]) -> bool:
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

    _GENERIC_LIST_ALIASES: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "StringList": (str,),
        "List": (str,),
        "IntList": (int,),
        "FloatList": (float,),
        "BoolList": (bool,),
    }

    _GENERIC_DICT_ALIASES: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "Dict": (str, str),
        "StringDict": (str, str),
        "NestedDict": (str, dict),
        "IntDict": (str, int),
        "FloatDict": (str, float),
        "BoolDict": (str, bool),
    }


__all__: list[str] = ["FlextUtilitiesGuardsTypeCore"]
