from __future__ import annotations

from collections.abc import Callable

from flext_core import t
from flext_core._protocols.container import FlextProtocolsContainer as pc
from flext_core._protocols.context import FlextProtocolsContext as pcx
from flext_core._protocols.handler import FlextProtocolsHandler as ph
from flext_core._protocols.logging import FlextProtocolsLogging as pl
from flext_core._protocols.result import FlextProtocolsResult as pr
from flext_core._protocols.service import FlextProtocolsService as psrv
from flext_core._protocols.settings import FlextProtocolsSettings as ps

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


__all__: list[str] = ["ProtocolGuardInput"]
