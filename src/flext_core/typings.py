"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

from flext_core._models import FlextModelsEnforcementSources
from flext_core._protocols import (
    FlextProtocolsContainer,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsResult,
    FlextProtocolsService,
    FlextProtocolsSettings,
)
from flext_core._typings import (
    FlextTypesCore,
    FlextTypesLazy,
    FlextTypesPydantic,
    FlextTypesServices,
    FlextTypesTypeAdapters,
    FlextTypingBase,
    FlextTypingConfig,
    FlextTypingContainers,
    FlextTypingProjectMetadata,
)


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


type JsonMapping = Mapping[str, FlextTypesPydantic.JsonValue]

type JsonDict = dict[str, FlextTypesPydantic.JsonValue]

type ConfigModelInput = FlextProtocolsResult.HasModelDump | JsonMapping

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
    | FlextProtocolsContainer.Container
    | FlextProtocolsContext.Context
    | FlextProtocolsHandler.Dispatcher
    | FlextProtocolsHandler.Handle
    | FlextProtocolsHandler.Middleware
    | FlextProtocolsLogging.Logger
    | FlextProtocolsResult.Result[t.JsonPayload]
    | FlextProtocolsSettings.Settings
    | FlextProtocolsService.Service[t.JsonPayload]
    | None
)

__all__: list[str] = ["FlextTypes", "t"]
