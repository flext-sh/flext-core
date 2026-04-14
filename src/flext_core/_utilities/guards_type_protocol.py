"""Protocol-based guards utilities for Flext type checking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard, TypeIs

from flext_core import FlextUtilitiesGuardsTypeModel, c, p, t


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
        """Get cached mapping of protocol names to type check predicates."""
        if FlextUtilitiesGuardsTypeProtocol._protocol_specs_cache is None:
            FlextUtilitiesGuardsTypeProtocol._protocol_specs_cache = MappingProxyType({
                c.Directory.CONFIG.value: lambda v: isinstance(v, p.Settings),
                c.FIELD_CONTEXT: lambda v: isinstance(v, p.Context),
                "container": lambda v: isinstance(v, p.Container),
                "command_bus": lambda v: (
                    hasattr(v, "dispatch")
                    and hasattr(v, "publish")
                    and hasattr(v, "register_handler")
                ),
                "handler": lambda v: isinstance(v, p.Handler),
                "logger": lambda v: isinstance(v, p.Logger),
                "result": lambda v: FlextUtilitiesGuardsTypeProtocol.result_like(v),
                "service": lambda v: isinstance(v, p.Service),
                "middleware": lambda v: isinstance(v, p.Middleware),
            })
        return FlextUtilitiesGuardsTypeProtocol._protocol_specs_cache

    @staticmethod
    def _get_protocol_type_map() -> Mapping[type, str]:
        """Get cached mapping of protocol types to their string names."""
        if FlextUtilitiesGuardsTypeProtocol._protocol_type_map_cache is None:
            FlextUtilitiesGuardsTypeProtocol._protocol_type_map_cache = (
                MappingProxyType({
                    p.Settings: c.Directory.CONFIG.value,
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
    def context(
        value: t.GuardInput,
    ) -> TypeIs[p.Context]:
        """Narrow value to Context protocol."""
        return isinstance(value, p.Context)

    @staticmethod
    def check_protocol_compliance(
        value: t.ProtocolSubject,
        protocol: type,
    ) -> bool:
        """Check runtime protocol compliance via stdlib isinstance()."""
        try:
            return isinstance(value, protocol)
        except TypeError:
            return False

    @staticmethod
    def handler_callable(
        value: t.GuardInput,
    ) -> bool:
        """Narrow value to callable handler function."""
        return callable(value)

    @staticmethod
    def factory(
        value: t.GuardInput,
    ) -> TypeIs[t.FactoryCallable]:
        """Narrow value to factory callable."""
        return callable(value)

    @staticmethod
    def resource(
        value: t.GuardInput,
    ) -> TypeIs[t.ResourceCallable]:
        """Narrow value to resource factory callable."""
        return callable(value)

    @staticmethod
    def result_like[TValue](
        value: TValue,
    ) -> TypeGuard[p.Result[t.RuntimeAtomic]]:
        """Narrow value to Result protocol."""
        return isinstance(value, p.Result)

    @staticmethod
    def registerable_service(
        value: t.GuardInput,
    ) -> TypeIs[t.RegisterableService]:
        """Narrow value to DI-registerable service (primitives, models, callables, etc.)."""
        if value is None:
            return True
        if isinstance(value, (str, int, float, bool, Path, Mapping)):
            return True
        if isinstance(value, Sequence):
            return not isinstance(value, (str, bytes, bytearray))
        if FlextUtilitiesGuardsTypeModel.pydantic_model(value):
            return True
        if callable(value):
            return True
        if FlextUtilitiesGuardsTypeProtocol.context(value):
            return True
        return hasattr(value, "__dict__") or bool(
            hasattr(value, "bind") and hasattr(value, "info"),
        )

    _STRING_TYPE_CHECKS: Mapping[str, Callable[[t.GuardInput], bool]] = {
        "str": lambda v: isinstance(v, str),
        "dict": lambda v: isinstance(v, dict),
        "list": lambda v: isinstance(v, list),
        "tuple": lambda v: isinstance(v, tuple),
        "sequence": lambda v: isinstance(v, (list, tuple, range)),
        "mapping": lambda v: isinstance(v, Mapping),
        "list_or_tuple": lambda v: isinstance(v, (list, tuple)),
        "sequence_not_str": lambda v: (
            isinstance(v, (list, tuple, range)) and not isinstance(v, str)
        ),
        "sequence_not_str_bytes": lambda v: (
            isinstance(v, (list, tuple, range)) and not isinstance(v, (str, bytes))
        ),
        "sized": lambda v: hasattr(v, "__len__"),
        "callable": lambda v: callable(v),
        "bytes": lambda v: isinstance(v, bytes),
        "int": lambda v: isinstance(v, int),
        "float": lambda v: isinstance(v, float),
        "bool": lambda v: isinstance(v, bool),
        "none": lambda v: v is None,
        "string_non_empty": lambda v: isinstance(v, str) and bool(v.strip()),
    }

    @staticmethod
    def _check_dict_non_empty(value: t.GuardInput) -> bool:
        """Check if value is a non-empty dict or ConfigMap."""
        if isinstance(value, dict):
            return value.__len__() > 0
        if isinstance(value, t.ConfigMap):
            return value.root.__len__() > 0
        return False

    @staticmethod
    def _check_list_non_empty(value: t.GuardInput) -> bool:
        """Check if value is a non-empty list/tuple (excluding str/bytes)."""
        if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
            return value.__len__() > 0
        return False

    @staticmethod
    def _run_string_type_check(type_name: str, value: t.GuardInput) -> bool:
        """Check value against a string type specification (e.g., 'str', 'list', 'dict')."""
        if type_name == "dict_non_empty":
            return FlextUtilitiesGuardsTypeProtocol._check_dict_non_empty(value)
        if type_name == "list_non_empty":
            return FlextUtilitiesGuardsTypeProtocol._check_list_non_empty(value)
        checker = FlextUtilitiesGuardsTypeProtocol._STRING_TYPE_CHECKS.get(type_name)
        if checker is None:
            return False
        return checker(value)

    @staticmethod
    def _check_protocol(value: t.GuardInput, name: str) -> bool:
        """Check if value implements a named protocol."""
        if name == c.FIELD_CONTEXT:
            return FlextUtilitiesGuardsTypeProtocol.context(value)
        try:
            return FlextUtilitiesGuardsTypeProtocol._get_protocol_specs()[name](value)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def _type_tuple(
        value: t.GuardInput,
    ) -> TypeIs[tuple[type, ...]]:
        """Narrow value to tuple of types."""
        return isinstance(value, tuple)

    @staticmethod
    def matches_type(
        value: t.GuardInput,
        type_spec: str | type | tuple[type, ...],
    ) -> bool:
        """Check if value matches a type spec (string name, type, or tuple of types)."""
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
                    if FlextUtilitiesGuardsTypeModel.pydantic_model(value):
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

    @staticmethod
    def settings_type(
        candidate: type | t.GuardInput,
    ) -> TypeIs[type[p.Settings]]:
        """Narrow candidate to Settings type with callable fetch_global."""
        return isinstance(candidate, type) and callable(
            getattr(candidate, "fetch_global", None),
        )

    @staticmethod
    def filter_registerable_services(
        services: Mapping[str, t.GuardInput] | None,
    ) -> Mapping[str, t.RegisterableService] | None:
        """Filter a service mapping to only registerable values."""
        if services is None:
            return None
        return {
            str(key): value
            for key, value in services.items()
            if FlextUtilitiesGuardsTypeProtocol.registerable_service(value)
        }

    @staticmethod
    def handler(obj: t.ProtocolSubject) -> bool:
        """Check if obj satisfies p.Handle protocol with validate capability."""
        return isinstance(obj, p.Handle) and callable(
            getattr(obj, "validate", None),
        )


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocol"]
