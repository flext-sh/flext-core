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
    Sequence,
)
from typing import Annotated, ClassVar

from flext_core import (
    FlextModelsBase as m,
    FlextModelsContainers as mc,
    FlextModelsPydantic as mp,
    FlextUtilitiesPydantic as up,
    c,
    p,
    t,
)


class FlextModelsHandler:
    """Handler state namespace."""

    class RegistrationResult(m.ArbitraryTypesModel):
        """Result of a handler registration operation."""

        handler_name: Annotated[
            t.NonEmptyStr,
            up.Field(description="Name of the handler"),
        ]
        status: Annotated[
            t.NonEmptyStr,
            up.Field(
                description="Registration status (registered, skipped, failed)",
            ),
        ]
        mode: Annotated[
            t.NonEmptyStr,
            up.Field(description="Registration mode (auto_discovery, explicit)"),
        ]
        handler_mode: Annotated[
            c.HandlerType | None,
            up.Field(None, description="Handler mode (command/query/event)"),
        ] = None
        message_type: Annotated[
            str | None,
            up.Field(None, description="Message type bound (for explicit mode)"),
        ] = None

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
            up.Field(
                description="Unique registration identifier",
                examples=["reg-abc123", "handler-create-user-001"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            up.Field(
                default=c.HandlerType.COMMAND,
                description="Handler mode (command, query, or event)",
                examples=["command", "query", "event"],
            ),
        ] = c.HandlerType.COMMAND
        timestamp: Annotated[
            str,
            up.Field(
                description="ISO 8601 timestamp recording when the registration entry was created.",
                title="Registration Timestamp",
                examples=["2025-01-01T00:00:00Z", "2025-10-12T15:30:00+00:00"],
                pattern=c.PATTERN_ISO8601_TIMESTAMP,
            ),
        ] = up.Field(default_factory=lambda: c.DEFAULT_EMPTY_STRING)
        status: Annotated[
            c.Status,
            up.Field(
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
            up.Field(
                description="Name of the handler being executed",
                examples=["ProcessOrderCommand", "GetUserQuery", "OrderCreatedEvent"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            up.Field(
                description="Mode of handler execution",
                examples=["command", "query", "event"],
            ),
        ]
        started_at: Annotated[
            float | None,
            up.Field(
                default=None,
                description="Monotonic start timestamp used to compute execution time.",
            ),
        ] = None
        metrics_state_data: Annotated[
            mc.Dict,
            up.Field(
                default_factory=lambda: mc.Dict(root={}),
                description="Mutable metrics payload for the active handler execution.",
            ),
        ] = up.Field(default_factory=lambda: mc.Dict(root={}))

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
            up.Field(description="Execution context for the active handler"),
        ]
        context_stack: Annotated[
            MutableSequence[FlextModelsHandler.ExecutionContext],
            up.Field(
                default_factory=list,
                description="Stack of nested execution contexts.",
            ),
        ]
        accepted_message_types: Annotated[
            tuple[t.TypeHintSpecifier, ...],
            up.Field(
                default_factory=tuple,
                description="Accepted message types computed for dispatch routing",
            ),
        ]
        revalidate_pydantic_messages: Annotated[
            bool,
            up.Field(
                default=False,
                description="Whether Pydantic messages must be revalidated on dispatch",
            ),
        ] = False
        type_warning_emitted: Annotated[
            bool,
            up.Field(
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
            up.Field(description="Command type this handler processes"),
        ]
        priority: Annotated[
            t.NonNegativeInt,
            up.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Handler priority (higher = processed first)",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        timeout: Annotated[
            float | None,
            up.Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                description="Handler execution timeout in seconds",
                gt=0.0,
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        middleware: Annotated[
            Sequence[type[p.Middleware]],
            up.Field(description="Middleware types to apply to this handler"),
        ] = up.Field(default_factory=tuple)


__all__: list[str] = ["FlextModelsHandler"]
