"""Context runtime state declarations, without scope operations.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from contextvars import ContextVar
from types import MappingProxyType

from flext_core import t
from flext_core._models import FlextModelsBase, FlextModelsPydantic
from flext_core._models._context.__scope_parts import flextmodelscontextscope_part_01
from flext_core._protocols import FlextProtocolsContainer


class FlextModelsContextScope(flextmodelscontextscope_part_01.FlextModelsContextScope):
    class ContextRuntimeState(FlextModelsBase.ArbitraryTypesModel):
        """Mutable context metadata, hooks, statistics and scope-variable registry."""

        metadata: FlextModelsBase.Metadata = FlextModelsPydantic.Field(
            default_factory=FlextModelsBase.Metadata,
            description="Normalized metadata bound to the active context",
        )
        hooks: t.ContextHookMap = FlextModelsPydantic.Field(
            default_factory=lambda: MappingProxyType[
                str, t.SequenceOf[t.ContextHookCallable]
            ]({}),
            validate_default=True,
            description="Lifecycle hooks keyed by event name",
        )
        statistics: flextmodelscontextscope_part_01.FlextModelsContextScope.ContextStatistics = FlextModelsPydantic.Field(
            default_factory=flextmodelscontextscope_part_01.FlextModelsContextScope.ContextStatistics,
            description="Operation counters for this context instance",
        )
        active: bool = FlextModelsPydantic.Field(
            default=True, description="Whether the context accepts operations"
        )
        suspended: bool = FlextModelsPydantic.Field(
            default=False, description="Whether the context is temporarily suspended"
        )
        scope_vars: t.MappingKV[
            str,
            ContextVar[FlextProtocolsContainer.MutableRootDict[t.JsonPayload] | None],
        ] = FlextModelsPydantic.Field(
            default_factory=lambda: MappingProxyType[
                str,
                ContextVar[
                    FlextProtocolsContainer.MutableRootDict[t.JsonPayload] | None
                ],
            ]({}),
            validate_default=True,
            description="ContextVar registry keyed by scope name",
        )


__all__: t.StrSequence = ("FlextModelsContextScope",)
