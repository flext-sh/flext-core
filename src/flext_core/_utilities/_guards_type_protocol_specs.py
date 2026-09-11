from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, TypeIs

from flext_core import c, t

from .._protocols.container import FlextProtocolsContainer as pc
from .._protocols.context import FlextProtocolsContext as pcx
from .._protocols.handler import FlextProtocolsHandler as ph
from .._protocols.logging import FlextProtocolsLogging as pl
from .._protocols.result import FlextProtocolsResult as pr
from .._protocols.service import FlextProtocolsService as psrv
from .._protocols.settings import FlextProtocolsSettings as ps

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._guards_type_protocol_types import ProtocolGuardInput


class FlextUtilitiesGuardsTypeProtocolSpecsMixin:
    _protocol_specs_cache: (
        t.MappingKV[str, Callable[[ProtocolGuardInput], bool]] | None
    ) = None
    _protocol_type_map_cache: MappingProxyType[type, str] | None = None

    @classmethod
    def _get_protocol_specs(
        cls,
    ) -> t.MappingKV[str, Callable[[ProtocolGuardInput], bool]]:
        if cls._protocol_specs_cache is None:
            cls._protocol_specs_cache = MappingProxyType({
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
                "result": cls.result_like,
                "service": lambda v: isinstance(v, psrv.Service),
                "middleware": lambda v: isinstance(v, ph.Middleware),
            })
        return cls._protocol_specs_cache

    @classmethod
    def _get_protocol_type_map(cls) -> t.MappingKV[type, str]:
        if cls._protocol_type_map_cache is None:
            cls._protocol_type_map_cache = MappingProxyType({
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
        return cls._protocol_type_map_cache

    @staticmethod
    def context(value: ProtocolGuardInput) -> TypeIs[pcx.Context]:
        return isinstance(value, pcx.Context)

    @staticmethod
    def result_like(value: ProtocolGuardInput) -> bool:
        return isinstance(value, pr.Result)

    @classmethod
    def _check_protocol(cls, value: ProtocolGuardInput, name: str) -> bool:
        if name == c.FIELD_CONTEXT:
            return cls.context(value)
        try:
            return cls._get_protocol_specs()[name](value)
        except c.EXC_ATTR_RUNTIME_TYPE:
            return False


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocolSpecsMixin"]
