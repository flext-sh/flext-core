"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from types import MappingProxyType
from typing import Annotated, ClassVar

from flext_core import c, p, t
from flext_core._models import FlextModelsBase, FlextModelsPydantic
from flext_core._models._context import FlextModelsContextData


class FlextModelsContextScopeBase:
    """Namespace for context scope statistics models."""

    class ContextStatistics(FlextModelsBase.ArbitraryTypesModel):
        """Statistics tracking for context operations."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True)
        )

        sets: Annotated[
            t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of set operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        gets: Annotated[
            t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of get operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        removes: Annotated[
            t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of remove operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        clears: Annotated[
            t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of clear operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        operations: Annotated[
            t.JsonMapping,
            FlextModelsPydantic.BeforeValidator(
                FlextModelsContextData.ContextData.validate_dict_serializable,
            ),
            FlextModelsPydantic.Field(
                description=(
                    "Additional metric counters and timing values grouped by metric "
                    "key."
                ),
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=lambda: MappingProxyType[str, t.JsonValue]({}),
            validate_default=True,
        )


class FlextModelsContextScopeRuntime(FlextModelsContextScopeBase):
    """Namespace for context runtime state models."""

    class ContextRuntimeState(FlextModelsBase.ArbitraryTypesModel):
        """Mutable context metadata, hooks, statistics and scope-variable registry."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True)
        )

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
            FlextModelsContextScopeBase.ContextStatistics,
            FlextModelsPydantic.Field(
                description="Operation counters for this context instance"
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=FlextModelsContextScopeBase.ContextStatistics,
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


class FlextModelsContextScope(FlextModelsContextScopeRuntime):
    """Public facade for FlextModelsContextScope."""

    class ContextContainerState(FlextModelsBase.ArbitraryTypesModel):
        """Centralized container binding state for `FlextContext`."""

        container: Annotated[
            t.Port[p.Container] | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Container configured for service namespace resolution",
            ),
        ] = None

        @FlextModelsPydantic.computed_field
        @property
        def configured(self) -> bool:
            """Whether a container is configured for service access."""

            return self.container is not None


__all__: list[str] = ["FlextModelsContextScope"]
