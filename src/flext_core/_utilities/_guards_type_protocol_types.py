from __future__ import annotations

from collections.abc import Callable

from flext_core import t

from .._protocols.container import FlextProtocolsContainer as pc
from .._protocols.context import FlextProtocolsContext as pcx
from .._protocols.handler import FlextProtocolsHandler as ph
from .._protocols.logging import FlextProtocolsLogging as pl
from .._protocols.result import FlextProtocolsResult as pr
from .._protocols.service import FlextProtocolsService as psrv
from .._protocols.settings import FlextProtocolsSettings as ps

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
    | pr.Result[t.JsonPayload]
    | ps.Settings
    | psrv.Service[t.JsonPayload]
    | None
)


__all__: list[str] = ["ProtocolGuardInput"]
