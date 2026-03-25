"""Protocol-based guards utilities for Flext type checking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TypeIs

from pydantic import BaseModel

from flext_core import c, p, t


class FlextUtilitiesGuardsTypeProtocol:
    """Protocol-based type guards for flext type checking.

    Provides static methods for checking whether values conform to flext framework
    protocols (Context, Handler, Service, etc.) and Python type specifications.
    Uses caching for performance-critical protocol lookups.
    """

    _protocol_specs_cache: Mapping[str, Callable[[t.GuardInput], bool]] | None = None
    _protocol_type_map_cache: MappingProxyType[type, str] | None = None

    @staticmethod
    def _get_protocol_specs() -> Mapping[str, Callable[[t.GuardInput], bool]]:
        """Get cached mapping of protocol names to type check predicates.

        Returns:
            Mapping of protocol name (str) to predicate function that validates if
            a value implements that protocol.

        """
        if FlextUtilitiesGuardsTypeProtocol._protocol_specs_cache is None:
            FlextUtilitiesGuardsTypeProtocol._protocol_specs_cache = MappingProxyType({
                c.FIELD_CONFIG: lambda v: isinstance(v, p.Settings),
                c.FIELD_CONTEXT: lambda v: isinstance(v, p.Context),
                "container": lambda v: isinstance(v, p.Container),
                "command_bus": lambda v: (
                    hasattr(v, "dispatch")
                    and hasattr(v, "publish")
                    and hasattr(v, "register_handler")
                ),
                "handler": lambda v: isinstance(v, p.Handler),
                "logger": lambda v: isinstance(v, p.Logger),
                "result": lambda v: FlextUtilitiesGuardsTypeProtocol.is_result_like(v),
                "service": lambda v: isinstance(v, p.Service),
                "middleware": lambda v: isinstance(v, p.Middleware),
            })
        return FlextUtilitiesGuardsTypeProtocol._protocol_specs_cache

    @staticmethod
    def _get_protocol_type_map() -> Mapping[type, str]:
        """Get cached mapping of protocol types to their string names.

        Returns:
            Mapping of protocol type (class) to its canonical string identifier.

        """
        if FlextUtilitiesGuardsTypeProtocol._protocol_type_map_cache is None:
            FlextUtilitiesGuardsTypeProtocol._protocol_type_map_cache = (
                MappingProxyType({
                    p.Settings: c.FIELD_CONFIG,
                    p.Context: c.FIELD_CONTEXT,
                    p.Container: "container",
                    p.Dispatcher: "command_bus",
                    p.Handler: "handler",
                    p.Logger: "logger",
                    p.Result: "result",
                    p.Service: "service",
                    p.Middleware: "middleware",
                })
            )
        return FlextUtilitiesGuardsTypeProtocol._protocol_type_map_cache

    @staticmethod
    def is_context(
        value: t.GuardInput,
    ) -> TypeIs[p.Context]:
        """Check if value is a Context protocol instance.

        Args:
            value: Value to check.

        Returns:
            True if value implements Context protocol, False otherwise.

        """
        return isinstance(value, p.Context)

    @staticmethod
    def is_handler_callable(
        value: object,
    ) -> TypeIs[t.HandlerCallable]:
        """Check if value is a callable handler function.

        Args:
            value: Value to check.

        Returns:
            True if value is callable, False otherwise.

        """
        return callable(value)

    @staticmethod
    def is_handler_type(
        value: t.NormalizedValue | t.HandlerCallable,
    ) -> TypeIs[t.HandlerLike]:
        """Check if value can be used as a handler.

        Handlers can be callables, mappings, BaseModel instances,
        or objects with handle/can_handle methods.

        Args:
            value: Value to check.

        Returns:
            True if value is a valid handler-like type, False otherwise.

        """
        return callable(value) or isinstance(
            value,
            (Mapping, p.HasModelDump, p.Handle, p.AutoDiscoverableHandler),
        )

    @staticmethod
    def is_registerable(
        value: t.GuardInput,
    ) -> TypeIs[t.RegisterableService]:
        """Check if value can be registered as a service.

        Args:
            value: Value to check.

        Returns:
            True if value is a registerable service type, False otherwise.

        """
        return FlextUtilitiesGuardsTypeProtocol.is_registerable_service(value)

    @staticmethod
    def is_factory(
        value: t.GuardInput,
    ) -> TypeIs[t.FactoryCallable]:
        """Check if value is a factory callable.

        Args:
            value: Value to check.

        Returns:
            True if value is callable, False otherwise.

        """
        return callable(value)

    @staticmethod
    def is_resource(
        value: t.GuardInput,
    ) -> TypeIs[t.ResourceCallable]:
        """Check if value is a resource factory callable.

        Args:
            value: Value to check.

        Returns:
            True if value is callable, False otherwise.

        """
        return callable(value)

    @staticmethod
    def is_result_like(
        value: t.GuardInput,
    ) -> TypeIs[p.Result[t.RuntimeAtomic]]:
        """Check if value is a Result protocol instance.

        Args:
            value: Value to check.

        Returns:
            True if value implements Result protocol, False otherwise.

        """
        return isinstance(value, p.Result)

    @staticmethod
    def is_registerable_service(
        value: t.GuardInput,
    ) -> TypeIs[t.RegisterableService]:
        """Check if value can be registered as a service in the DI container.

        Accepts None, primitives, BaseModel, Path, Mapping, callables, sequences
        (except str/bytes), Context, and objects with __dict__ or bind/info attrs.

        Args:
            value: Value to check.

        Returns:
            True if value is a registerable service, False otherwise.

        """
        return (
            value is None
            or isinstance(value, (str, int, float, bool, BaseModel, Path, Mapping))
            or callable(value)
            or (
                isinstance(value, Sequence)
                and (not isinstance(value, (str, bytes, bytearray)))
            )
            or FlextUtilitiesGuardsTypeProtocol.is_context(value)
            or hasattr(value, "__dict__")
            or bool(hasattr(value, "bind") and hasattr(value, "info"))
        )

    @staticmethod
    def _run_string_type_check(type_name: str, value: t.GuardInput) -> bool:
        """Check value against a string type specification.

        Args:
            type_name: String identifier for the type to check (e.g., 'str', 'list', 'dict').
            value: Value to check.

        Returns:
            True if value matches the type specification, False otherwise.

        """
        match type_name:
            case "str":
                return bool(isinstance(value, str))
            case "dict":
                return bool(isinstance(value, dict))
            case "list":
                return bool(isinstance(value, list))
            case "tuple":
                return bool(isinstance(value, tuple))
            case "sequence":
                return bool(isinstance(value, (list, tuple, range)))
            case "mapping":
                return bool(isinstance(value, Mapping))
            case "list_or_tuple":
                return bool(isinstance(value, (list, tuple)))
            case "sequence_not_str":
                return bool(
                    isinstance(value, (list, tuple, range))
                    and (not isinstance(value, str)),
                )
            case "sequence_not_str_bytes":
                return bool(
                    isinstance(value, (list, tuple, range))
                    and (not isinstance(value, (str, bytes))),
                )
            case "sized":
                return bool(hasattr(value, "__len__"))
            case "callable":
                return bool(callable(value))
            case "bytes":
                return bool(isinstance(value, bytes))
            case "int":
                return bool(isinstance(value, int))
            case "float":
                return bool(isinstance(value, float))
            case "bool":
                return bool(isinstance(value, bool))
            case "none":
                return bool(value is None)
            case "string_non_empty":
                return bool(isinstance(value, str) and bool(value.strip()))
            case "dict_non_empty":
                if isinstance(value, dict):
                    return value.__len__() > 0
                if isinstance(value, t.ConfigMap):
                    return value.root.__len__() > 0
                return False
            case "list_non_empty":
                if isinstance(value, (list, tuple)) and not isinstance(
                    value,
                    (str, bytes),
                ):
                    return value.__len__() > 0
                return False
            case _:
                return False

    @staticmethod
    def _check_protocol(value: t.GuardInput, name: str) -> bool:
        """Check if value implements a named protocol.

        Args:
            value: Value to check.
            name: Protocol name to verify against.

        Returns:
            True if value implements the protocol, False otherwise or on error.

        """
        if name == c.FIELD_CONTEXT:
            return FlextUtilitiesGuardsTypeProtocol.is_context(value)
        try:
            return FlextUtilitiesGuardsTypeProtocol._get_protocol_specs()[name](value)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def _is_type_tuple(
        value: t.GuardInput,
    ) -> TypeIs[tuple[type, ...]]:
        """Check if value is a tuple of types.

        Args:
            value: Value to check.

        Returns:
            True if value is a tuple, False otherwise.

        """
        return isinstance(value, tuple)

    @staticmethod
    def is_type(
        value: t.GuardInput,
        type_spec: str | type | tuple[type, ...],
    ) -> bool:
        """Check if value matches a type specification.

        Supports string type names (e.g., 'str', 'dict'), protocol types,
        actual type objects, and tuples of types for isinstance checks.

        Args:
            value: Value to check.
            type_spec: Type specification (string name, type, or tuple of types).

        Returns:
            True if value matches the type specification, False otherwise.

        """
        if isinstance(type_spec, str):
            type_name = type_spec.lower()
            if type_name in FlextUtilitiesGuardsTypeProtocol._get_protocol_specs():
                return FlextUtilitiesGuardsTypeProtocol._check_protocol(
                    value,
                    type_name,
                )
            if type_name in c.STRING_METHOD_MAP:
                if type_name in {
                    "string_non_empty",
                    "dict_non_empty",
                    "list_non_empty",
                }:
                    if isinstance(value, BaseModel):
                        return False
                    return FlextUtilitiesGuardsTypeProtocol._run_string_type_check(
                        type_name,
                        value,
                    )
                return FlextUtilitiesGuardsTypeProtocol._run_string_type_check(
                    type_name,
                    value,
                )
            return False
        if isinstance(type_spec, tuple):
            return isinstance(value, type_spec)
        if type_spec in FlextUtilitiesGuardsTypeProtocol._get_protocol_type_map():
            return FlextUtilitiesGuardsTypeProtocol._check_protocol(
                value,
                FlextUtilitiesGuardsTypeProtocol._get_protocol_type_map()[type_spec],
            )
        try:
            return isinstance(
                value,
                getattr(type_spec, "__origin__", None) or type_spec,
            )
        except TypeError:
            return False


__all__ = ["FlextUtilitiesGuardsTypeProtocol"]
