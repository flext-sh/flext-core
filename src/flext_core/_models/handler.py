"""Handler management patterns extracted from FlextModels.

This module contains the FlextModelsHandler class with all handler-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Handler instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Annotated, ClassVar, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)

from flext_core import FlextModelFoundation, c, p, t


class FlextModelsHandler:
    """Handler management pattern container class.

    This class acts as a namespace container for handler management patterns.
    All nested classes are accessed via FlextModels.Handler.* in the main models.py.
    """

    class Registration(FlextModelFoundation.ArbitraryTypesModel):
        """Handler registration with advanced validation."""

        name: Annotated[
            t.NonEmptyStr,
            Field(description="Handler name"),
        ]
        handler: Annotated[
            t.HandlerCallable,
            Field(description="Handler callable function or method"),
        ]
        event_types: Annotated[
            Sequence[str],
            Field(
                default_factory=list,
                description="Event types this handler processes",
            ),
        ]

        @field_validator("handler", mode="before")
        @classmethod
        def validate_handler(
            cls,
            v: t.HandlerCallable | p.Base | BaseModel,
        ) -> t.HandlerCallable | p.Base | BaseModel:
            if not callable(v):
                msg = f"Handler must be callable, got {v.__class__.__name__}"
                raise TypeError(msg)
            return v

        @model_validator(mode="after")
        def validate_handler_interface(self) -> Self:
            """Validate handler has handle() or execute() method or is callable."""
            if not callable(self.handler):
                msg = "Handler must be callable or have handle()/execute() method"
                raise TypeError(msg)
            return self

    class RegistrationResult(FlextModelFoundation.ArbitraryTypesModel):
        """Result of a handler registration operation.

        Provides structured feedback on the outcome of a handler registration,
        including status, mode, and identification.
        """

        handler_name: Annotated[t.NonEmptyStr, Field(description="Name of the handler")]
        status: Annotated[
            t.NonEmptyStr,
            Field(
                description="Registration status (registered, skipped, failed)",
            ),
        ]
        mode: Annotated[
            t.NonEmptyStr,
            Field(description="Registration mode (auto_discovery, explicit)"),
        ]
        handler_mode: Annotated[
            c.HandlerType | None,
            Field(default=None, description="Handler mode (command/query/event)"),
        ] = None
        message_type: Annotated[
            str | None,
            Field(
                default=None,
                description="Message type bound (for explicit mode)",
            ),
        ] = None
        _GETITEM_FIELDS: ClassVar[frozenset[str]] = frozenset({
            "handler_name",
            c.FIELD_STATUS,
            "mode",
            c.FIELD_HANDLER_MODE,
            "message_type",
        })

        def __getitem__(self, key: str) -> t.ValueOrModel:
            match key:
                case "handler_name":
                    return self.handler_name
                case c.FIELD_STATUS:
                    return self.status
                case "mode":
                    return self.mode
                case c.FIELD_HANDLER_MODE:
                    return self.handler_mode
                case "message_type":
                    return self.message_type
                case _:
                    raise KeyError(key)

    class RegistrationRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Request model for dynamic handler registration.

        Strictly typed model for handler registration parameters, replacing
        legacy dictionary-based configuration.
        """

        handler: Annotated[
            t.HandlerCallable | p.Handler[p.Model, t.ValueOrModel] | BaseModel,
            Field(
                description="Handler instance (callable, t.NormalizedValue, or FlextHandlers)",
            ),
        ]
        message_type: Annotated[
            t.MessageTypeSpecifier | None,
            Field(
                default=None,
                description="Message type to handle (required for explicit mode)",
            ),
        ] = None
        handler_mode: Annotated[
            c.HandlerType | None,
            Field(
                default=None,
                description="Handler operation mode (command, query, event)",
            ),
        ] = None
        handler_name: Annotated[
            str | None,
            Field(
                default=None,
                description="Explicit handler name override",
            ),
        ] = None

    class RegistrationDetails(FlextModelFoundation.ArbitraryTypesModel):
        """Registration details for handler registration tracking.

        Tracks metadata about handler registrations in the CQRS system,
        including unique identification, timing, and status information.

        This model is used by FlextRegistry to track which handlers have been
        registered with the dispatcher and monitor their lifecycle.

        Attributes:
            registration_id: Unique identifier for this registration
            handler_mode: Mode of handler (command, query, or event)
            timestamp: ISO 8601 timestamp when registration occurred
            status: Current status of the registration

        Examples:
            >>> details = FlextModelsHandler.RegistrationDetails(
            ...     registration_id="reg-123",
            ...     handler_mode="command",
            ...     timestamp="2025-01-01T00:00:00Z",
            ...     status="running",
            ... )
            >>> details.registration_id
            'reg-123'

        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "RegistrationDetails",
                "description": "Handler registration tracking details",
            },
        )
        registration_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Unique registration identifier",
                examples=["reg-abc123", "handler-create-user-001"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            Field(
                default=c.HandlerType.COMMAND,
                description="Handler mode (command, query, or event)",
                examples=["command", "query", "event"],
            ),
        ] = c.HandlerType.COMMAND
        timestamp: Annotated[
            str,
            Field(
                default_factory=lambda: c.DEFAULT_TIMESTAMP,
                description="ISO 8601 timestamp recording when the registration entry was created.",
                title="Registration Timestamp",
                examples=["2025-01-01T00:00:00Z", "2025-10-12T15:30:00+00:00"],
                pattern=c.PATTERN_ISO8601_TIMESTAMP,
            ),
        ] = Field(default_factory=lambda: c.DEFAULT_TIMESTAMP)
        status: Annotated[
            c.CommonStatus,
            Field(
                default=c.CommonStatus.RUNNING,
                description="Current registration status",
                examples=["running", "stopped", "failed"],
            ),
        ] = c.CommonStatus.RUNNING

    class ExecutionContext(FlextModelFoundation.ArbitraryTypesModel):
        """Handler execution context for tracking handler performance and state.

        Provides timing and metrics tracking for handler executions in the
        FlextContext system. Uses Pydantic 2 PrivateAttr for internal state.

        This mutable context t.NormalizedValue tracks handler execution performance,
        including timing, metrics, and execution state. It is designed to be
        created at the start of handler execution and updated throughout.

        Attributes:
            handler_name: Name of the handler being executed
            handler_mode: Mode of execution (command, query, or event)

        Examples:
            >>> context = FlextModelsHandler.ExecutionContext.create_for_handler(
            ...     handler_name="ProcessOrderCommand", handler_mode="command"
            ... )
            >>> context.start_execution()
            >>> # ... handler executes ...
            >>> elapsed_ms = context.execution_time_ms
            >>> context.set_metrics_state({"items_processed": 42})

        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True,
            json_schema_extra={
                "title": "HandlerExecutionContext",
                "description": "Handler execution context for tracking performance and state",
            },
        )
        handler_name: Annotated[
            t.NonEmptyStr,
            Field(
                description="Name of the handler being executed",
                examples=["ProcessOrderCommand", "GetUserQuery", "OrderCreatedEvent"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            Field(
                description="Mode of handler execution",
                examples=["command", "query", "event"],
            ),
        ]
        _start_time: float | None = PrivateAttr(default=None)
        _metrics_state: t.Dict | None = PrivateAttr(default=None)

        @computed_field
        def execution_time_ms(self) -> float:
            """Get execution time in milliseconds."""
            if self._start_time is None:
                return 0.0
            start_time: float = self._start_time
            elapsed: float = time.time() - start_time
            return round(elapsed * c.MILLISECONDS_MULTIPLIER, 2)

        @computed_field
        def has_metrics(self) -> bool:
            """Check if metrics have been recorded."""
            return self._metrics_state is not None and bool(self._metrics_state)

        @computed_field
        def is_running(self) -> bool:
            """Check if execution is currently running."""
            return self._start_time is not None

        @computed_field
        def metrics_state(self) -> t.Dict:
            """Get current metrics state.

            Returns:
                Dictionary containing metrics state (empty ConfigurationDict if not set)

            Examples:
                >>> context = FlextModelsHandler.ExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> metrics = context.metrics_state
                >>> FlextRuntime.is_dict_like(metrics)
                True

            """
            if self._metrics_state is None:
                self._metrics_state = t.Dict(root={})
            metrics_state_val: t.Dict = self._metrics_state
            return metrics_state_val

        @classmethod
        def create_for_handler(
            cls,
            handler_name: str,
            handler_mode: c.HandlerType,
        ) -> Self:
            """Create execution context for a handler.

            Factory method for creating handler execution contexts with
            validation of handler name and mode.

            Args:
                handler_name: Name of the handler
                handler_mode: Mode of the handler (command/query/event)

            Returns:
                New HandlerExecutionContext instance

            Examples:
                >>> context = FlextModelsHandler.ExecutionContext.create_for_handler(
                ...     handler_name="ProcessOrderCommand", handler_mode="command"
                ... )
                >>> context.handler_name
                'ProcessOrderCommand'
                >>> context.handler_mode
                'command'

            """
            return cls(handler_name=handler_name, handler_mode=handler_mode)

        def reset(self) -> None:
            """Reset execution context.

            Clears all timing and metrics state, preparing the context
            for reuse or cleanup.

            Examples:
                >>> context = FlextModelsHandler.ExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> context.start_execution()
                >>> context.reset()
                >>> context.execution_time_ms
                0.0

            """
            self._start_time = None
            self._metrics_state = None

        def set_metrics_state(self, state: t.Dict) -> None:
            """Set metrics state."""
            self._metrics_state = state

        def start_execution(self) -> None:
            """Start execution timing.

            Records the current time as the start time for execution metrics.
            Should be called at the beginning of handler execution.

            Examples:
                >>> context = FlextModelsHandler.ExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> context.start_execution()

            """
            self._start_time = time.time()

    class DecoratorConfig(FlextModelFoundation.ArbitraryTypesModel):
        """Configuration extracted from @FlextHandlers.handler() decorator.

        Used by handler discovery to auto-register handlers with FlextDispatcher.
        Stores metadata about command binding, priority, timeout, and middleware chain.

        Attributes:
            command: The command type this handler processes.
            priority: Handler priority (higher = processed first). Default: 0.
            timeout: Handler execution timeout in seconds (None = no timeout).
            middleware: List of middleware types to apply to this handler.

        Examples:
            >>> config = FlextModelsHandler.DecoratorConfig(
            ...     command=CreateUserCommand,
            ...     priority=10,
            ...     timeout=c.DEFAULT_TIMEOUT,
            ...     middleware=[LoggingMiddleware, ValidationMiddleware],
            ... )
            >>> config.command
            <class 'CreateUserCommand'>
            >>> config.priority
            10

        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True, arbitrary_types_allowed=True
        )
        command: Annotated[
            type,
            Field(description="Command type this handler processes"),
        ]
        priority: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.DEFAULT_PRIORITY,
                description="Handler priority (higher = processed first)",
            ),
        ] = c.DEFAULT_PRIORITY
        timeout: Annotated[
            float | None,
            Field(
                default=c.DEFAULT_TIMEOUT,
                description="Handler execution timeout in seconds",
                gt=0.0,
            ),
        ] = c.DEFAULT_TIMEOUT
        middleware: Annotated[
            Sequence[type[p.Middleware]],
            Field(
                default_factory=list,
                description="Middleware types to apply to this handler",
            ),
        ]


__all__ = ["FlextModelsHandler"]
