"""Handler state models - Pydantic v2, state-only surface.

Only fields, validators, and computed properties that are consumed by
``src/`` (handlers, registry, utilities). Orchestration mutations live in
``FlextUtilitiesHandler``. All helper methods, dead factories, dict-style
accessors, and redundant wrapper classes have been removed.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import MutableSequence, Sequence
from typing import TYPE_CHECKING, Annotated, ClassVar

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._models import (
    FlextModelsBase,
    FlextModelsContainers,
    FlextModelsPydantic,
)


class FlextModelsHandler:
    """Handler state namespace."""

    class HandlerRegistration(FlextModelsBase.ArbitraryTypesModel):
        """One immutable binding in an atomic dispatcher registration batch."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True)
        )
        handler: flext_core.t.Port[flext_core.p.DispatchableHandler] = (
            FlextModelsPydantic.Field(
                description="Original handler whose identity owns the binding",
                exclude=True,
            )
        )
        message_type: flext_core.t.RegistryBindingKey | None = (
            FlextModelsPydantic.Field(
                default=None,
                description="Explicit message type or route name",
            )
        )
        is_event: bool = FlextModelsPydantic.Field(
            default=False,
            description="Whether this binding subscribes to events",
        )

    class HandlerRegistrationReport(FlextModelsBase.ArbitraryTypesModel):
        """Exact new and idempotent bindings from one atomic registration."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True)
        )
        registered: tuple[flext_core.t.Port[flext_core.p.HandlerRegistration], ...] = (
            FlextModelsPydantic.Field(
                description="Bindings newly published by the batch",
                exclude=True,
            )
        )
        skipped: tuple[flext_core.t.Port[flext_core.p.HandlerRegistration], ...] = (
            FlextModelsPydantic.Field(
                description="Bindings already owned by the same handler",
                exclude=True,
            )
        )

    class RegistrationDetails(FlextModelsBase.ArbitraryTypesModel):
        """Registration details tracked by ``FlextRegistry``."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                frozen=True,
                json_schema_extra={
                    "title": "RegistrationDetails",
                    "description": "Handler registration tracking details",
                },
            )
        )
        registration_id: Annotated[
            flext_core.t.NonEmptyStr,
            FlextModelsPydantic.Field(
                description="Unique registration identifier",
                examples=["reg-abc123", "handler-create-user-001"],
            ),
        ]
        handler_mode: Annotated[
            flext_core.c.HandlerType,
            FlextModelsPydantic.Field(
                default=flext_core.c.HandlerType.COMMAND,
                description="Handler mode (command, query, or event)",
                examples=[
                    flext_core.c.HandlerMode.COMMAND,
                    flext_core.c.HandlerMode.QUERY,
                    flext_core.c.HandlerType.EVENT,
                ],
            ),
        ] = flext_core.c.HandlerType.COMMAND
        timestamp: Annotated[
            str,
            FlextModelsPydantic.Field(
                description=(
                    "ISO 8601 timestamp recording when the registration entry was "
                    "created."
                ),
                title="Registration Timestamp",
                examples=["2025-01-01T00:00:00Z", "2025-10-12T15:30:00+00:00"],
                pattern=flext_core.c.PATTERN_ISO8601_TIMESTAMP,
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=lambda: flext_core.c.DEFAULT_EMPTY_STRING
        )
        status: Annotated[
            flext_core.c.Status,
            FlextModelsPydantic.Field(
                default=flext_core.c.Status.RUNNING,
                description="Current registration status",
                examples=["running", "stopped", "failed"],
            ),
        ] = flext_core.c.Status.RUNNING

    class ExecutionContext(FlextModelsBase.ArbitraryTypesModel):
        """Handler execution state (identity + timing + metrics payload)."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                validate_assignment=True,
                json_schema_extra={
                    "title": "HandlerExecutionContext",
                    "description": (
                        "Handler execution context for tracking performance and state"
                    ),
                },
            )
        )
        handler_name: Annotated[
            flext_core.t.NonEmptyStr,
            FlextModelsPydantic.Field(
                description="Name of the handler being executed",
                examples=["ProcessOrderCommand", "GetUserQuery", "OrderCreatedEvent"],
            ),
        ]
        handler_mode: Annotated[
            flext_core.c.HandlerType,
            FlextModelsPydantic.Field(
                description="Mode of handler execution",
                examples=[
                    flext_core.c.HandlerMode.COMMAND,
                    flext_core.c.HandlerMode.QUERY,
                    flext_core.c.HandlerType.EVENT,
                ],
            ),
        ]
        started_at: Annotated[
            float | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Monotonic start timestamp used to compute execution time.",
            ),
        ] = None
        metrics_state_data: Annotated[
            FlextModelsContainers.Dict,
            FlextModelsPydantic.Field(
                default_factory=lambda: FlextModelsContainers.Dict(root={}),
                description="Mutable metrics payload for the active handler execution.",
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=lambda: FlextModelsContainers.Dict(root={}),
        )

        @FlextModelsPydantic.computed_field
        @property
        def execution_time_ms(self) -> float:
            """Elapsed execution time in milliseconds (0 until started)."""
            if self.started_at is None:
                return 0.0
            elapsed: float = time.time() - self.started_at
            return round(elapsed * flext_core.c.MS_PER_SECOND, 2)

    class HandlerRuntimeState(FlextModelsBase.ArbitraryTypesModel):
        """Aggregate runtime state for the active handler pipeline."""

        execution_context: Annotated[
            FlextModelsHandler.ExecutionContext,
            FlextModelsPydantic.Field(
                description="Execution context for the active handler",
            ),
        ]
        context_stack: Annotated[
            MutableSequence[FlextModelsHandler.ExecutionContext],
            FlextModelsPydantic.Field(
                description="Stack of nested execution contexts.",
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=list,
            validate_default=True,
        )

        @FlextModelsPydantic.computed_field
        @property
        def handler_name(self) -> str:
            """Active handler name taken from the execution context."""
            return self.execution_context.handler_name

        @FlextModelsPydantic.computed_field
        @property
        def handler_mode(self) -> flext_core.c.HandlerType:
            """Active handler mode taken from the execution context."""
            return self.execution_context.handler_mode

    class DecoratorConfig(FlextModelsBase.ArbitraryTypesModel):
        """Configuration extracted from @FlextHandlers.handler() decorator."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                frozen=True,
            )
        )
        command: Annotated[
            type,
            FlextModelsPydantic.Field(
                description="Command type this handler processes",
            ),
        ]
        priority: Annotated[
            flext_core.t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=flext_core.c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Handler priority (higher = processed first)",
            ),
        ] = flext_core.c.DEFAULT_MAX_COMMAND_RETRIES
        timeout: Annotated[
            float | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Handler execution timeout in seconds",
                gt=0.0,
            ),
        ] = None
        middleware: Annotated[
            Sequence[type[flext_core.p.Middleware]],
            FlextModelsPydantic.Field(
                description="Middleware types to apply to this handler",
            ),
            FlextModelsPydantic.PlainSerializer(
                lambda value: [
                    f"{middleware_type.__module__}.{middleware_type.__qualname__}"
                    for middleware_type in value
                ],
                return_type=list[str],
                when_used="always",
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=tuple,
        )

    class CombinedRailwayOptions(FlextModelsBase.ImmutableValueModel):
        """Railway configuration consumed by @d.combined()."""

        enabled: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=False,
                description="Whether combined() applies railway wrapping.",
            ),
        ] = False
        error_code: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description=(
                    "Error code passed to railway() when railway wrapping is enabled."
                ),
            ),
        ] = None


__all__: list[str] = ["FlextModelsHandler"]
