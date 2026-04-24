"""Protocol-based guards utilities for Flext type checking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from pathlib import Path
from types import MappingProxyType
from typing import TypeIs

from flext_core import FlextUtilitiesGuardsTypeModel as ugm, c, p, t


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
        value: t.GuardInput | None,
    ) -> TypeIs[p.Context]:
        """Narrow value to Context protocol."""
        return isinstance(value, p.Context)

    @staticmethod
    def factory(
        value: t.GuardInput,
    ) -> TypeIs[t.FactoryCallable]:
        """Narrow value to factory callable."""
        return callable(value)

    @staticmethod
    def result_like(
        value: t.GuardInput,
    ) -> TypeIs[p.Result[t.JsonPayload]]:
        """Narrow any value to Result protocol (runtime isinstance check)."""
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
        if ugm.pydantic_model(value):
            return True
        if callable(value):
            return True
        if FlextUtilitiesGuardsTypeProtocol.context(value):
            return True
        return hasattr(value, "__dict__") or bool(
            hasattr(value, "bind") and hasattr(value, "info"),
        )

    @staticmethod
    def _run_string_type_check(type_name: str, value: t.GuardInput) -> bool:
        """Check value against a string type specification (e.g., 'str', 'list', 'dict')."""
        if type_name == "dict_non_empty":
            return isinstance(value, Mapping) and len(value) > 0
        if type_name == "list_non_empty":
            return isinstance(value, (list, tuple)) and len(value) > 0
        checker = c.STRING_TYPE_PREDICATES.get(type_name)
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
    def matches_type(
        value: t.GuardInput,
        type_spec: str
        | type
        | tuple[type, ...]
        | t.Scalar,  # Scalar arm handles invalid spec at runtime
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
                    if ugm.pydantic_model(value):
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
        if not isinstance(type_spec, type):
            # type_spec is a non-type scalar (e.g. int value 123) — invalid spec
            return False
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


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocol"]
