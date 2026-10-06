"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

from flext_core._models._enforcement._sources import FlextModelsEnforcementSources
from flext_core._protocols.container import FlextProtocolsContainer as pc
from flext_core._protocols.context import FlextProtocolsContext as pcx
from flext_core._protocols.handler import FlextProtocolsHandler as ph
from flext_core._protocols.loggings import FlextProtocolsLogging as pl
from flext_core._protocols.result import (
    FlextProtocolsResult as pr,
    FlextProtocolsResult as prt,
)
from flext_core._protocols.service import FlextProtocolsService as psrv
from flext_core._protocols.settings import FlextProtocolsSettings as ps
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.config import FlextTypingConfig
from flext_core._typings.containers import FlextTypingContainers
from flext_core._typings.core import FlextTypesCore
from flext_core._typings.lazy import FlextTypesLazy
from flext_core._typings.project_metadata import FlextTypingProjectMetadata
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._typings.services import FlextTypesServices
from flext_core._typings.typeadapters import FlextTypesTypeAdapters


class FlextTypes(
    FlextTypingBase,
    FlextTypingConfig,
    FlextTypingContainers,
    FlextTypesCore,
    FlextTypesLazy,
    FlextTypesServices,
    FlextTypesTypeAdapters,
    FlextTypingProjectMetadata,
):
    """Type system foundation for FLEXT ecosystem.

    Strictly tiered layers - Primitives subset Scalar subset Container.
    ``object`` and ``Any`` are strictly forbidden in domain state.
    ``None`` is **never** baked into definitions.
    """


t = FlextTypes


type JsonMapping = Mapping[str, tp.JsonValue]

type JsonDict = dict[str, tp.JsonValue]

type ConfigModelInput = prt.HasModelDump | JsonMapping

T = TypeVar("T")


type ModuleGlobalValue = FlextTypesLazy.ModuleGlobalValue

type ModuleGlobals = FlextTypesLazy.ModuleGlobals

type EnforcementRuleSource = (
    FlextModelsEnforcementSources.EnforcementInfraRuleSource
    | FlextModelsEnforcementSources.EnforcementRuntimeWarningSource
    | FlextModelsEnforcementSources.EnforcementBeartypeSource
    | FlextModelsEnforcementSources.EnforcementCodeSmellSource
)

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

__all__: list[str] = ["FlextTypes", "t"]
