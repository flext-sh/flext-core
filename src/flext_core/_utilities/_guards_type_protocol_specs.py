"""Guards type protocol specs module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, TypeIs

from flext_core import c, t
from flext_core._protocols import (
    FlextProtocolsContainer,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsResult,
    FlextProtocolsService,
    FlextProtocolsSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.typings import t.ProtocolGuardInput


class FlextUtilitiesGuardsTypeProtocolSpecsMixin:
    _protocol_specs_cache: (
        t.MappingKV[str, Callable[[t.ProtocolGuardInput], bool]] | None
    ) = None
    _protocol_type_map_cache: MappingProxyType[type, str] | None = None

    @classmethod
    def _get_protocol_specs(
        cls,
    ) -> t.MappingKV[str, Callable[[t.ProtocolGuardInput], bool]]:
        if cls._protocol_specs_cache is None:
            cls._protocol_specs_cache = MappingProxyType({
                c.Directory.CONFIG.value: lambda v: isinstance(
                    v,
                    FlextProtocolsSettings.Settings,
                ),
                c.FIELD_CONTEXT: lambda v: isinstance(v, FlextProtocolsContext.Context),
                "container": lambda v: isinstance(v, FlextProtocolsContainer.Container),
                "command_bus": lambda v: (
                    hasattr(v, "dispatch")
                    and hasattr(v, "publish")
                    and hasattr(v, "register_handler")
                ),
                "handler": lambda v: isinstance(v, FlextProtocolsHandler.Handler),
                "logger": lambda v: isinstance(v, FlextProtocolsLogging.Logger),
                "result": cls.result_like,
                "service": lambda v: isinstance(v, FlextProtocolsService.Service),
                "middleware": lambda v: isinstance(v, FlextProtocolsHandler.Middleware),
            })
        return cls._protocol_specs_cache

    @classmethod
    def _get_protocol_type_map(cls) -> t.MappingKV[type, str]:
        if cls._protocol_type_map_cache is None:
            cls._protocol_type_map_cache = MappingProxyType({
                FlextProtocolsSettings.Settings: c.Directory.CONFIG.value,
                FlextProtocolsContext.Context: c.FIELD_CONTEXT,
                FlextProtocolsContainer.Container: "container",
                FlextProtocolsHandler.Dispatcher: "command_bus",
                FlextProtocolsHandler.Handler: "handler",
                FlextProtocolsLogging.Logger: "logger",
                FlextProtocolsResult.Result: "result",
                FlextProtocolsService.Service: "service",
                FlextProtocolsHandler.Middleware: "middleware",
            })
        return cls._protocol_type_map_cache

    @staticmethod
    def context(value: t.ProtocolGuardInput) -> TypeIs[FlextProtocolsContext.Context]:
        return isinstance(value, FlextProtocolsContext.Context)

    @staticmethod
    def result_like(value: t.ProtocolGuardInput) -> bool:
        return isinstance(value, FlextProtocolsResult.Result)

    @classmethod
    def _check_protocol(cls, value: t.ProtocolGuardInput, name: str) -> bool:
        if name == c.FIELD_CONTEXT:
            return cls.context(value)
        matched = False
        try:
            matched = cls._get_protocol_specs()[name](value)
        except c.EXC_ATTR_RUNTIME_TYPE:
            matched = False
        return matched


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocolSpecsMixin"]
