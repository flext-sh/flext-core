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
from types import MappingProxyType
from typing import TypeIs

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._protocols.container import FlextProtocolsContainer as pc
from flext_core._protocols.context import FlextProtocolsContext as pcx
from flext_core._protocols.handler import FlextProtocolsHandler as ph
from flext_core._protocols.logging import FlextProtocolsLogging as pl
from flext_core._protocols.result import FlextProtocolsResult as pr
from flext_core._protocols.service import FlextProtocolsService as psrv
from flext_core._protocols.settings import FlextProtocolsSettings as ps
from flext_core.constants import c
from flext_core.typings import t

type ProtocolGuardInput = (
    t.JsonPayload
    | t.TypeHintSpecifier
    | Callable[..., t.JsonPayload]
    | pc.Container
    | pcx.Context
    | ph.Dispatcher
    | ph.Handle
    | ph.Middleware
    | pl.Logger
    | pr.ResultLike[t.JsonPayload]
    | ps.Settings
    | psrv.Service[t.JsonPayload]
    | None
)


class FlextUtilitiesGuardsTypeProtocol:
    """Protocol-based type guards for flext type checking.

    Provides static methods for checking whether values conform to flext framework
    protocols (Context, Handler, Service, etc.) and Python type specifications.
    Uses caching for performance-critical protocol lookups.
    """

    _protocol_specs_cache: t.MappingKV[str, Callable[[ProtocolGuardInput], bool]] | None = (
        None
    )
    _protocol_type_map_cache: MappingProxyType[type, str] | None = None

    @staticmethod
    def _get_protocol_specs() -> t.MappingKV[str, Callable[[ProtocolGuardInput], bool]]:
        """Get cached mapping of protocol names to type check predicates."""
        if FlextUtilitiesGuardsTypeProtocol._protocol_specs_cache is None:
            FlextUtilitiesGuardsTypeProtocol._protocol_specs_cache = MappingProxyType({
                c.Directory.CONFIG.value: lambda v: isinstance(v, ps.Settings),
                c.FIELD_CONTEXT: lambda v: isinstance(v, pcx.Context),
                "container": lambda v: isinstance(v, pc.Container),
                "command_bus": lambda v: (
                    hasattr(v, "dispatch")
                    and hasattr(v, "publish")
                    and hasattr(v, "register_handler")
                ),
                "handler": lambda v: isinstance(v, ph.Handler),
                "logger": lambda v: isinstance(v, pl.Logger),
                "result": lambda v: FlextUtilitiesGuardsTypeProtocol.result_like(v),
                "service": lambda v: isinstance(v, psrv.Service),
                "middleware": lambda v: isinstance(v, ph.Middleware),
            })
        return FlextUtilitiesGuardsTypeProtocol._protocol_specs_cache

    @staticmethod
    def _get_protocol_type_map() -> t.MappingKV[type, str]:
        """Get cached mapping of protocol types to their string names."""
        if FlextUtilitiesGuardsTypeProtocol._protocol_type_map_cache is None:
            FlextUtilitiesGuardsTypeProtocol._protocol_type_map_cache = (
                MappingProxyType({
                    ps.Settings: c.Directory.CONFIG.value,
                    pcx.Context: c.FIELD_CONTEXT,
                    pc.Container: "container",
                    ph.Dispatcher: "command_bus",
                    ph.Handler: "handler",
                    pl.Logger: "logger",
                    pr.Result: "result",
                    psrv.Service: "service",
                    ph.Middleware: "middleware",
                })
            )
        return FlextUtilitiesGuardsTypeProtocol._protocol_type_map_cache

    @staticmethod
    def context(
        value: ProtocolGuardInput,
    ) -> TypeIs[pcx.Context]:
        """Narrow value to Context protocol."""
        return isinstance(value, pcx.Context)

    @staticmethod
    def factory(
        value: ProtocolGuardInput,
    ) -> bool:
        """Return whether value is callable as a factory."""
        return callable(value)

    @staticmethod
    def result_like(
        value: ProtocolGuardInput,
    ) -> bool:
        """Return whether value satisfies the Result protocol at runtime."""
        return isinstance(value, pr.Result)

    @staticmethod
    def _run_string_type_check(type_name: str, value: ProtocolGuardInput) -> bool:
        """Check value against a string type specification (e.g., 'str', 'list', 'dict')."""
        match type_name:
            case "str":
                return isinstance(value, str)
            case "dict":
                return isinstance(value, dict)
            case "list":
                return isinstance(value, list)
            case "tuple":
                return isinstance(value, tuple)
            case "sequence":
                return isinstance(value, (list, tuple, range))
            case "mapping":
                return isinstance(value, Mapping)
            case "list_or_tuple":
                return isinstance(value, (list, tuple))
            case "sequence_not_str":
                return isinstance(value, (list, tuple, range)) and not isinstance(
                    value,
                    str,
                )
            case "sequence_not_str_bytes":
                return isinstance(value, (list, tuple, range)) and not isinstance(
                    value,
                    (str, bytes),
                )
            case "sized":
                return hasattr(value, "__len__")
            case "callable":
                return callable(value)
            case "bytes":
                return isinstance(value, bytes)
            case "int":
                return isinstance(value, int)
            case "float":
                return isinstance(value, float)
            case "bool":
                return isinstance(value, bool)
            case "none":
                return value is None
            case "string_non_empty":
                return isinstance(value, str) and bool(value.strip())
            case "dict_non_empty":
                return isinstance(value, Mapping) and len(value) > 0
            case "list_non_empty":
                return (
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray))
                    and len(value) > 0
                )
            case _:
                return False

    @staticmethod
    def _check_protocol(value: ProtocolGuardInput, name: str) -> bool:
        """Check if value implements a named protocol."""
        if name == c.FIELD_CONTEXT:
            return FlextUtilitiesGuardsTypeProtocol.context(value)
        try:
            return FlextUtilitiesGuardsTypeProtocol._get_protocol_specs()[name](value)
        except c.EXC_ATTR_RUNTIME_TYPE:
            return False

    @staticmethod
    def matches_type(
        value: ProtocolGuardInput,
        type_spec: str
        | type
        | tuple[type, ...]
        | t.Scalar,  # Scalar arm handles invalid spec at runtime
    ) -> bool:
        """Check if value matches a type spec (string name, type, or tuple of types)."""
        matched = False
        if isinstance(type_spec, str):
            type_name = type_spec.lower()
            protocol_specs = FlextUtilitiesGuardsTypeProtocol._get_protocol_specs()
            if type_name in protocol_specs:
                matched = FlextUtilitiesGuardsTypeProtocol._check_protocol(
                    value, type_name
                )
            elif type_name in c.STRING_METHOD_MAP:
                matched = not (
                    type_name
                    in {"string_non_empty", "dict_non_empty", "list_non_empty"}
                    and isinstance(value, (mp.BaseModel, mp.RootModel))
                ) and FlextUtilitiesGuardsTypeProtocol._run_string_type_check(
                    type_name,
                    value,
                )
        elif isinstance(type_spec, tuple):
            matched = isinstance(value, type_spec)
        elif isinstance(type_spec, type):
            protocol_name = (
                FlextUtilitiesGuardsTypeProtocol._get_protocol_type_map().get(
                    type_spec,
                )
            )
            if protocol_name is not None:
                matched = FlextUtilitiesGuardsTypeProtocol._check_protocol(
                    value,
                    protocol_name,
                )
            else:
                runtime_type = getattr(type_spec, "__origin__", None) or type_spec
                try:
                    matched = isinstance(value, runtime_type)
                except TypeError:
                    matched = False
        return matched


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocol"]
