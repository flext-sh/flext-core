"""Protocol guard tables memoized by Python's native cache.

Each declaring class owns its immutable protocol tables. Memoization belongs to
the table-producing methods, without annotated mutable holders in the utility
namespace or cached service instances.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeIs

import flext_core

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextUtilitiesGuardsTypeProtocolSpecs:
    @classmethod
    @cache
    def _get_protocol_specs(
        cls,
    ) -> Mapping[str, Callable[[flext_core.p.ProtocolGuardInput], bool]]:
        return MappingProxyType({
            flext_core.c.Directory.CONFIG.value: lambda v: isinstance(
                v,
                flext_core.p.Settings,
            ),
            flext_core.c.FIELD_CONTEXT: lambda v: isinstance(v, flext_core.p.Context),
            "container": lambda v: isinstance(v, flext_core.p.Container),
            "command_bus": lambda v: (
                hasattr(v, "dispatch")
                and hasattr(v, "publish")
                and hasattr(v, "register_handler")
            ),
            "handler": lambda v: isinstance(v, flext_core.p.Handler),
            "logger": lambda v: isinstance(v, flext_core.p.Logger),
            "result": cls.result_like,
            "service": lambda v: isinstance(v, flext_core.p.Service),
            "middleware": lambda v: isinstance(v, flext_core.p.Middleware),
        })

    @classmethod
    @cache
    def _get_protocol_type_map(cls) -> Mapping[type, str]:
        return MappingProxyType({
            flext_core.p.Settings: flext_core.c.Directory.CONFIG.value,
            flext_core.p.Context: flext_core.c.FIELD_CONTEXT,
            flext_core.p.Container: "container",
            flext_core.p.Dispatcher: "command_bus",
            flext_core.p.Handler: "handler",
            flext_core.p.Logger: "logger",
            flext_core.p.Result: "result",
            flext_core.p.Service: "service",
            flext_core.p.Middleware: "middleware",
        })

    @staticmethod
    def context(
        value: flext_core.p.ProtocolGuardInput,
    ) -> TypeIs[flext_core.p.Context]:
        return isinstance(value, flext_core.p.Context)

    @staticmethod
    def result_like(value: flext_core.p.ProtocolGuardInput) -> bool:
        return isinstance(value, flext_core.p.Result)

    @staticmethod
    def result_view(
        value: flext_core.p.AttributeProbe,
    ) -> TypeIs[flext_core.p.ResultView[flext_core.p.AttributeProbe]]:
        """Refine an opaque SDK value to the native read-only result shape.

        Uses static protocol inspection, without reading properties or checking
        payload/domain contents. It does not require the full railway protocol.

        Returns:
            Whether the opaque value satisfies the native read-only protocol.

        """
        return isinstance(value, flext_core.p.ResultView)

    @classmethod
    def _check_protocol(cls, value: flext_core.p.ProtocolGuardInput, name: str) -> bool:
        if name == flext_core.c.FIELD_CONTEXT:
            return cls.context(value)
        matched = False
        try:
            matched = cls._get_protocol_specs()[name](value)
        except flext_core.c.EXC_ATTR_RUNTIME_TYPE:
            matched = False
        return matched


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocolSpecs"]
