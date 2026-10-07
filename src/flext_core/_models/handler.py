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
from collections.abc import MutableSequence
from typing import Annotated, ClassVar

from flext_core import c, p, t
from flext_core._models.base import FlextModelsBase
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities import FlextUtilitiesPydantic


class FlextModelsHandler:
    """Handler state namespace."""

    class RegistrationDetails(FlextModelsBase.ArbitraryTypesModel):
        """Registration details tracked by ``FlextRegistry``."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                json_schema_extra={
                    "title": "RegistrationDetails",
                    "description": "Handler registration tracking details",
                },
            )
        )
        registration_id: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(
                description="Unique registration identifier",
                examples=["reg-abc123", "handler-create-user-001"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            FlextModelsPydantic.Field(
                default=c.HandlerType.COMMAND,
                description="Handler mode (command, query, or event)",
                examples=["command", "query", "event"],
            ),
        ] = c.HandlerType.COMMAND
        timestamp: Annotated[
            str,
            FlextModelsPydantic.Field(
                description=(
                    "ISO 8601 timestamp recording when the registration entry was "
                    "created."
                ),
                title="Registration Timestamp",
                examples=["2025-01-01T00:00:00Z", "2025-10-12T15:30:00+00:00"],
                pattern=c.PATTERN_ISO8601_TIMESTAMP,
            ),
        ] = FlextModelsPydantic.Field(default_factory=lambda: c.DEFAULT_EMPTY_STRING)
        status: Annotated[
            c.Status,
            FlextModelsPydantic.Field(
                default=c.Status.RUNNING,
                description="Current registration status",
                examples=["running", "stopped", "failed"],
            ),
        ] = c.Status.RUNNING

    class ExecutionContext(FlextModelsBase.ArbitraryTypesModel):
        """Handler execution state (identity + timing + metrics payload)."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                arbitrary_types_allowed=True,
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
            t.NonEmptyStr,
            FlextModelsPydantic.Field(
                description="Name of the handler being executed",
                examples=["ProcessOrderCommand", "GetUserQuery", "OrderCreatedEvent"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            FlextModelsPydantic.Field(
                description="Mode of handler execution",
                examples=["command", "query", "event"],
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

        @FlextUtilitiesPydantic.computed_field
        @property
        def execution_time_ms(self) -> float:
            """Elapsed execution time in milliseconds (0 until started)."""
            if self.started_at is None:
                return 0.0
            elapsed: float = time.time() - self.started_at
            return round(elapsed * c.MS_PER_SECOND, 2)

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
            default_factory=list[FlextModelsHandler.ExecutionContext],
        )

        @FlextModelsPydantic.computed_field
        @property
        def handler_name(self) -> str:
            """Active handler name taken from the execution context."""
            return self.execution_context.handler_name

        @FlextModelsPydantic.computed_field
        @property
        def handler_mode(self) -> c.HandlerType:
            """Active handler mode taken from the execution context."""
            return self.execution_context.handler_mode

    class DecoratorConfig(FlextModelsBase.ArbitraryTypesModel):
        """Configuration extracted from @FlextHandlers.handler() decorator."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                frozen=True,
                arbitrary_types_allowed=True,
            )
        )
        command: Annotated[
            type,
            FlextModelsPydantic.Field(
                description="Command type this handler processes",
            ),
        ]
        priority: Annotated[
            t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Handler priority (higher = processed first)",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        timeout: Annotated[
            float | None,
            FlextModelsPydantic.Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                description="Handler execution timeout in seconds",
                gt=0.0,
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        middleware: Annotated[
            t.SequenceOf[type[p.Middleware]],
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
        ] = FlextModelsPydantic.Field(default_factory=tuple)

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
