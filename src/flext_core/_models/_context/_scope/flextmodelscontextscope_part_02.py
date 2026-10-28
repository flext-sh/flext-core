"""Context runtime state declarations, without scope operations.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from types import MappingProxyType
from typing import Annotated

from flext_core import p, t
from flext_core._models import FlextModelsBase, FlextModelsPydantic
from flext_core._models._context._scope import FlextModelsContextScopePart01


class FlextModelsContextScopePart02(FlextModelsContextScopePart01):
    class ContextRuntimeState(FlextModelsBase.ArbitraryTypesModel):
        """Mutable context metadata, hooks, statistics and scope-variable registry."""

        metadata: FlextModelsBase.Metadata = FlextModelsPydantic.Field(
            default_factory=FlextModelsBase.Metadata,
            description="Normalized metadata bound to the active context",
        )
        hooks: t.ContextHookMap = FlextModelsPydantic.Field(
            default_factory=lambda: MappingProxyType[
                str, Sequence[t.ContextHookCallable]
            ]({}),
            validate_default=True,
            description="Lifecycle hooks keyed by event name",
        )
        statistics: Annotated[
            FlextModelsContextScopePart01.ContextStatistics,
            FlextModelsPydantic.Field(
                description="Operation counters for this context instance"
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=FlextModelsContextScopePart01.ContextStatistics,
        )
        active: bool = FlextModelsPydantic.Field(
            default=True, description="Whether the context accepts operations"
        )
        suspended: bool = FlextModelsPydantic.Field(
            default=False, description="Whether the context is temporarily suspended"
        )
        scope_vars: Mapping[
            str,
            t.Port[ContextVar[p.MutableRootDict[t.JsonPayload] | None]],
        ] = FlextModelsPydantic.Field(
            default_factory=lambda: MappingProxyType[
                str,
                ContextVar[p.MutableRootDict[t.JsonPayload] | None],
            ]({}),
            validate_default=True,
            description="ContextVar registry keyed by scope name",
        )


__all__: Sequence[str] = ("FlextModelsContextScopePart02",)
