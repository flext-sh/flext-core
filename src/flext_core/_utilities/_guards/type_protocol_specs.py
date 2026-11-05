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

from flext_core import c, p

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextUtilitiesGuardsTypeProtocolSpecs:
    @classmethod
    @cache
    def _get_protocol_specs(
        cls,
    ) -> Mapping[str, Callable[[p.ProtocolGuardInput], bool]]:
        return MappingProxyType({
            c.Directory.CONFIG.value: lambda v: isinstance(
                v,
                p.Settings,
            ),
            c.FIELD_CONTEXT: lambda v: isinstance(v, p.Context),
            "container": lambda v: isinstance(v, p.Container),
            "command_bus": lambda v: (
                hasattr(v, "dispatch")
                and hasattr(v, "publish")
                and hasattr(v, "register_handler")
            ),
            "handler": lambda v: isinstance(v, p.Handler),
            "logger": lambda v: isinstance(v, p.Logger),
            "result": cls.result_like,
            "service": lambda v: isinstance(v, p.Service),
            "middleware": lambda v: isinstance(v, p.Middleware),
        })

    @classmethod
    @cache
    def _get_protocol_type_map(cls) -> Mapping[type, str]:
        return MappingProxyType({
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

    @staticmethod
    def context(
        value: p.ProtocolGuardInput,
    ) -> TypeIs[p.Context]:
        return isinstance(value, p.Context)

    @staticmethod
    def result_like(value: p.ProtocolGuardInput) -> bool:
        return isinstance(value, p.Result)

    @staticmethod
    def result_view(
        value: p.AttributeProbe,
    ) -> TypeIs[p.ResultView[p.AttributeProbe]]:
        """Refine an opaque SDK value to the native read-only result shape.

        Uses static protocol inspection, without reading properties or checking
        payload/domain contents. It does not require the full railway protocol.

        Returns:
            Whether the opaque value satisfies the native read-only protocol.

        """
        return isinstance(value, p.ResultView)

    @classmethod
    def _check_protocol(cls, value: p.ProtocolGuardInput, name: str) -> bool:
        if name == c.FIELD_CONTEXT:
            return cls.context(value)
        matched = False
        try:
            matched = cls._get_protocol_specs()[name](value)
        except c.EXC_ATTR_RUNTIME_TYPE:
            matched = False
        return matched


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocolSpecs"]
