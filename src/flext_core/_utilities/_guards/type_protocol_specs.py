"""Protocol guard tables memoized by Python's native cache.

Each declaring class owns its immutable protocol tables. Memoization belongs to
the table-producing methods, without annotated mutable holders in the utility
namespace or cached service instances.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeIs

from flext_core import c, t
from flext_core._protocols import (
    FlextProtocolsBase,
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


class FlextUtilitiesGuardsTypeProtocolSpecs:
    @classmethod
    @cache
    def _get_protocol_specs(
        cls,
    ) -> t.MappingKV[str, Callable[[FlextProtocolsContainer.ProtocolGuardInput], bool]]:
        return MappingProxyType({
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

    @classmethod
    @cache
    def _get_protocol_type_map(cls) -> t.MappingKV[type, str]:
        return MappingProxyType({
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

    @staticmethod
    def context(
        value: FlextProtocolsContainer.ProtocolGuardInput,
    ) -> TypeIs[FlextProtocolsContext.Context]:
        return isinstance(value, FlextProtocolsContext.Context)

    @staticmethod
    def result_like(value: FlextProtocolsContainer.ProtocolGuardInput) -> bool:
        return isinstance(value, FlextProtocolsResult.Result)

    @staticmethod
    def result_view(
        value: FlextProtocolsBase.AttributeProbe,
    ) -> TypeIs[FlextProtocolsResult.ResultView[FlextProtocolsBase.AttributeProbe]]:
        """Refine an opaque SDK value to the native read-only result shape.

        Uses static protocol inspection, without reading properties or checking
        payload/domain contents. It does not require the full railway protocol.

        Returns:
            Whether the opaque value satisfies the native read-only protocol.

        """
        return isinstance(value, FlextProtocolsResult.ResultView)

    @classmethod
    def _check_protocol(
        cls, value: FlextProtocolsContainer.ProtocolGuardInput, name: str
    ) -> bool:
        if name == c.FIELD_CONTEXT:
            return cls.context(value)
        matched = False
        try:
            matched = cls._get_protocol_specs()[name](value)
        except c.EXC_ATTR_RUNTIME_TYPE:
            matched = False
        return matched


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocolSpecs"]
