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
from collections.abc import (
    MutableSequence,
)
from typing import Annotated, ClassVar

from flext_core import (
    FlextConstants as c,
    FlextModelsPydantic as mp,
    FlextProtocols as p,
    FlextTypes as t,
    FlextUtilitiesPydantic as up,
)
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.containers import FlextModelsContainers as mc


class FlextModelsHandler:
    """Handler state namespace."""

    class RegistrationDetails(m.ArbitraryTypesModel):
        """Registration details tracked by ``FlextRegistry``."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            json_schema_extra={
                "title": "RegistrationDetails",
                "description": "Handler registration tracking details",
            },
        )
        registration_id: Annotated[
            t.NonEmptyStr,
            mp.Field(
                description="Unique registration identifier",
                examples=["reg-abc123", "handler-create-user-001"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            mp.Field(
                default=c.HandlerType.COMMAND,
                description="Handler mode (command, query, or event)",
                examples=["command", "query", "event"],
            ),
        ] = c.HandlerType.COMMAND
        timestamp: Annotated[
            str,
            mp.Field(
                description="ISO 8601 timestamp recording when the registration entry was created.",
                title="Registration Timestamp",
                examples=["2025-01-01T00:00:00Z", "2025-10-12T15:30:00+00:00"],
                pattern=c.PATTERN_ISO8601_TIMESTAMP,
            ),
        ] = mp.Field(default_factory=lambda: c.DEFAULT_EMPTY_STRING)
        status: Annotated[
            c.Status,
            mp.Field(
                default=c.Status.RUNNING,
                description="Current registration status",
                examples=["running", "stopped", "failed"],
            ),
        ] = c.Status.RUNNING

    class ExecutionContext(m.ArbitraryTypesModel):
        """Handler execution state (identity + timing + metrics payload)."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            json_schema_extra={
                "title": "HandlerExecutionContext",
                "description": "Handler execution context for tracking performance and state",
            },
        )
        handler_name: Annotated[
            t.NonEmptyStr,
            mp.Field(
                description="Name of the handler being executed",
                examples=["ProcessOrderCommand", "GetUserQuery", "OrderCreatedEvent"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            mp.Field(
                description="Mode of handler execution",
                examples=["command", "query", "event"],
            ),
        ]
        started_at: Annotated[
            float | None,
            mp.Field(
                default=None,
                description="Monotonic start timestamp used to compute execution time.",
            ),
        ] = None
        metrics_state_data: Annotated[
            mc.Dict,
            mp.Field(
                default_factory=lambda: mc.Dict(root={}),
                description="Mutable metrics payload for the active handler execution.",
            ),
        ] = mp.Field(default_factory=lambda: mc.Dict(root={}))

        @up.computed_field()
        def execution_time_ms(self) -> float:
            """Elapsed execution time in milliseconds (0 until started)."""
            if self.started_at is None:
                return 0.0
            elapsed: float = time.time() - self.started_at
            return round(elapsed * c.DEFAULT_SIZE, 2)

    class HandlerRuntimeState(m.ArbitraryTypesModel):
        """Aggregate runtime state for the active handler pipeline."""

        execution_context: Annotated[
            FlextModelsHandler.ExecutionContext,
            mp.Field(description="Execution context for the active handler"),
        ]
        context_stack: Annotated[
            MutableSequence[FlextModelsHandler.ExecutionContext],
            mp.Field(
                default_factory=list,
                description="Stack of nested execution contexts.",
            ),
        ]
        accepted_message_types: Annotated[
            tuple[t.TypeHintSpecifier, ...],
            mp.Field(
                default_factory=tuple,
                description="Accepted message types computed for dispatch routing",
            ),
        ]
        revalidate_pydantic_messages: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Whether Pydantic messages must be revalidated on dispatch",
            ),
        ] = False
        type_warning_emitted: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Whether the handler already emitted a type warning",
            ),
        ] = False

        @up.computed_field()
        @property
        def handler_name(self) -> str:
            """Active handler name taken from the execution context."""
            return self.execution_context.handler_name

        @up.computed_field()
        @property
        def handler_mode(self) -> c.HandlerType:
            """Active handler mode taken from the execution context."""
            return self.execution_context.handler_mode

    class DecoratorConfig(m.ArbitraryTypesModel):
        """Configuration extracted from @FlextHandlers.handler() decorator."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )
        command: Annotated[
            type,
            mp.Field(description="Command type this handler processes"),
        ]
        priority: Annotated[
            t.NonNegativeInt,
            mp.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Handler priority (higher = processed first)",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        timeout: Annotated[
            float | None,
            mp.Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                description="Handler execution timeout in seconds",
                gt=0.0,
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        middleware: Annotated[
            t.SequenceOf[type[p.Middleware]],
            mp.Field(description="Middleware types to apply to this handler"),
        ] = mp.Field(default_factory=tuple)

    class CombinedRailwayOptions(m.ImmutableValueModel):
        """Railway configuration consumed by @d.combined()."""

        enabled: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Whether combined() applies railway wrapping.",
            ),
        ] = False
        error_code: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Error code passed to railway() when railway wrapping is enabled.",
            ),
        ] = None


__all__: list[str] = ["FlextModelsHandler"]
