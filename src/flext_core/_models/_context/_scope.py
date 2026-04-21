"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
from collections.abc import (
    Mapping,
)
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar, Self

from flext_core import (
    FlextModelsBase,
    FlextModelsContainers,
    FlextModelsContextData,
    FlextModelsPydantic,
    FlextUtilitiesPydantic,
    c,
    p,
    t,
)


class FlextModelsContextScope:
    """Namespace for context scope and statistics models."""

    class ContextScopeData(FlextModelsBase.ArbitraryTypesModel):
        """Scope-specific data container for context management."""

        scope_name: Annotated[
            t.NonEmptyStr,
            FlextUtilitiesPydantic.Field(description="Name of the scope"),
        ] = c.ContextScope.GLOBAL
        scope_type: Annotated[
            str,
            FlextUtilitiesPydantic.Field(
                default="", description="Type/category of scope"
            ),
        ] = ""
        data: Annotated[
            Mapping[str, t.ValueOrModel],
            FlextModelsPydantic.BeforeValidator(
                lambda v: FlextModelsContextData.normalize_to_mapping(v)
            ),
            FlextUtilitiesPydantic.Field(description="Scope data"),
        ] = FlextUtilitiesPydantic.Field(default_factory=lambda: MappingProxyType({}))
        metadata: Annotated[
            Mapping[str, t.Container],
            FlextModelsPydantic.BeforeValidator(
                lambda v: FlextModelsContextData.normalize_to_mapping(v)
            ),
            FlextUtilitiesPydantic.Field(description="Scope metadata"),
        ] = FlextUtilitiesPydantic.Field(default_factory=lambda: MappingProxyType({}))

    class ContextStatistics(FlextModelsBase.ArbitraryTypesModel):
        """Statistics tracking for context operations.

        Enforcement exemption: counters and the ``operations`` map are
        incremented throughout the context lifecycle; fresh per-instance.
        """

        _flext_enforcement_exempt: ClassVar[bool] = True

        sets: Annotated[
            t.NonNegativeInt,
            FlextUtilitiesPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of set operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        gets: Annotated[
            t.NonNegativeInt,
            FlextUtilitiesPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of get operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        removes: Annotated[
            t.NonNegativeInt,
            FlextUtilitiesPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of remove operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        clears: Annotated[
            t.NonNegativeInt,
            FlextUtilitiesPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of clear operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        operations: Annotated[
            Mapping[str, t.MetadataValue],
            FlextModelsPydantic.BeforeValidator(
                lambda v: (
                    FlextModelsContextData.normalize_to_mapping(v)
                    if v is not None
                    else {}
                ),
            ),
            FlextUtilitiesPydantic.Field(
                description="Additional metric counters and timing values grouped by metric key.",
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=dict)

    class ContextRuntimeState(FlextModelsBase.ArbitraryTypesModel):
        """Centralized mutable runtime state for `FlextContext`."""

        metadata: Annotated[
            FlextModelsBase.Metadata,
            FlextUtilitiesPydantic.Field(
                default_factory=FlextModelsBase.Metadata,
                description="Normalized metadata bound to the active context",
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=FlextModelsBase.Metadata)
        hooks: Annotated[
            t.ContextHookMap,
            FlextUtilitiesPydantic.Field(
                default_factory=dict,
                description="Lifecycle hooks keyed by event name",
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=lambda: MappingProxyType({}))
        statistics: Annotated[
            FlextModelsContextScope.ContextStatistics,
            FlextUtilitiesPydantic.Field(
                default_factory=lambda: FlextModelsContextScope.ContextStatistics(),
                description="Operation counters for this context instance",
            ),
        ] = FlextUtilitiesPydantic.Field(
            default_factory=lambda: FlextModelsContextScope.ContextStatistics()
        )
        active: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                default=True, description="Whether the context accepts operations"
            ),
        ] = True
        suspended: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                default=False,
                description="Whether the context is temporarily suspended",
            ),
        ] = False
        scope_vars: Annotated[
            Mapping[
                str, contextvars.ContextVar[FlextModelsContainers.ConfigMap | None]
            ],
            FlextUtilitiesPydantic.Field(
                default_factory=dict,
                description="ContextVar registry keyed by scope name",
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=lambda: MappingProxyType({}))

        @classmethod
        def create_default(
            cls,
            metadata: FlextModelsBase.Metadata | None = None,
        ) -> Self:
            """Create default runtime state with canonical built-in scopes."""
            global_scope_var: contextvars.ContextVar[
                FlextModelsContainers.ConfigMap | None
            ] = contextvars.ContextVar(
                "flext_global_context",
                default=None,
            )
            user_scope_var: contextvars.ContextVar[
                FlextModelsContainers.ConfigMap | None
            ] = contextvars.ContextVar(
                "flext_user_context",
                default=None,
            )
            session_scope_var: contextvars.ContextVar[
                FlextModelsContainers.ConfigMap | None
            ] = contextvars.ContextVar(
                "flext_session_context",
                default=None,
            )
            metadata_model = (
                metadata.model_copy()
                if metadata is not None
                else FlextModelsBase.Metadata()
            )
            return cls(
                metadata=metadata_model,
                scope_vars={
                    c.ContextScope.GLOBAL: global_scope_var,
                    c.ContextScope.USER: user_scope_var,
                    c.ContextScope.SESSION: session_scope_var,
                },
            )

        def resolve_scope_var(
            self,
            scope: str,
        ) -> tuple[
            Self, contextvars.ContextVar[FlextModelsContainers.ConfigMap | None]
        ]:
            """Resolve an existing scope var or create one immutably."""
            existing = self.scope_vars.get(scope)
            if existing is not None:
                return self, existing
            scope_var: contextvars.ContextVar[
                FlextModelsContainers.ConfigMap | None
            ] = contextvars.ContextVar(
                f"flext_{scope}_context",
                default=None,
            )
            updated_scope_vars: dict[
                str, contextvars.ContextVar[FlextModelsContainers.ConfigMap | None]
            ] = dict(self.scope_vars)
            updated_scope_vars[scope] = scope_var
            return self.model_copy(update={"scope_vars": updated_scope_vars}), scope_var

        def with_operation_update(self, operation: str) -> Self:
            """Increment canonical statistics for the given operation."""
            counter_attr = f"{operation}s"
            statistics_updates: dict[str, t.RuntimeData] = {}
            current_statistics = self.statistics
            if counter_attr in type(current_statistics).model_fields:
                current_counter = getattr(current_statistics, counter_attr)
                if isinstance(current_counter, int):
                    statistics_updates[counter_attr] = current_counter + 1
            operations: dict[str, t.Container] = {
                str(key): value
                for key, value in current_statistics.operations.items()
                if isinstance(value, (str, int, float, bool, datetime, Path))
            }
            current_operation_value = operations.get(operation)
            if isinstance(current_operation_value, int):
                operations[operation] = current_operation_value + 1
                statistics_updates["operations"] = operations
            if not statistics_updates:
                return self
            updated_statistics = current_statistics.model_copy(
                update=statistics_updates,
            )
            return self.model_copy(update={"statistics": updated_statistics})

    class ContextContainerState(FlextModelsBase.ArbitraryTypesModel):
        """Centralized container binding state for `FlextContext`."""

        container: Annotated[
            p.Container | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                description="Container configured for service namespace resolution",
            ),
        ] = None

        @FlextUtilitiesPydantic.computed_field()
        def configured(self) -> bool:
            """Whether a container is configured for service access."""
            return self.container is not None

        def with_container(self, container: p.Container | None) -> Self:
            """Replace the configured container immutably."""
            return self.model_copy(update={"container": container})


__all__: list[str] = ["FlextModelsContextScope"]
