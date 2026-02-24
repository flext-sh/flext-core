"""Handler management patterns extracted from FlextModels.

This module contains the FlextModelsHandler class with all handler-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Handler instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
)

from flext_core._models.base import FlextModelFoundation
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.typings import t


class FlextModelsHandler:
    """Handler management pattern container class.

    This class acts as a namespace container for handler management patterns.
    All nested classes are accessed via FlextModels.Handler.* in the main models.py.
    """

    class Registration(FlextModelFoundation.ArbitraryTypesModel):
        """Handler registration with advanced validation."""

        name: str = Field(
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Handler name",
        )
        handler: t.HandlerCallable = Field(
            description="Handler callable function or method",
        )
        event_types: list[str] = Field(
            default_factory=list,
            description="Event types this handler processes",
        )

        @field_validator("handler", mode="before")
        @classmethod
        def validate_handler(
            cls,
            v: t.GuardInputValue,
        ) -> t.GuardInputValue:
            if not callable(v):
                msg = f"Handler must be callable, got {v.__class__.__name__}"
                raise TypeError(msg)
            return v

    class RegistrationResult(FlextModelFoundation.ArbitraryTypesModel):
        """Result of a handler registration operation.

        Provides structured feedback on the outcome of a handler registration,
        including status, mode, and identification.
        """

        handler_name: str = Field(
            description="Name of the handler",
        )
        status: str = Field(
            description="Registration status (registered, skipped, failed)",
        )
        mode: str = Field(
            description="Registration mode (auto_discovery, explicit)",
        )
        handler_mode: c.Cqrs.HandlerTypeLiteral | c.Cqrs.HandlerType | None = Field(
            default=None,
            description="Handler mode (command/query/event)",
        )
        message_type: str | None = Field(
            default=None,
            description="Message type bound (for explicit mode)",
        )

        def __getitem__(self, key: str) -> t.GuardInputValue:
            dump = self.model_dump()
            return dump[key]

    class RegistrationRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Request model for dynamic handler registration.

        Strictly typed model for handler registration parameters, replacing
        legacy dictionary-based configuration.
        """

        handler: Callable[..., object] | BaseModel = Field(
            description="Handler instance (callable, object, or FlextHandlers)",
        )
        message_type: str | type[object] | None = Field(
            default=None,
            description="Message type to handle (required for explicit mode)",
        )
        handler_mode: c.Cqrs.HandlerTypeLiteral | None = Field(
            default=None,
            description="Handler operation mode (command, query, event)",
        )
        handler_name: str | None = Field(
            default=None,
            description="Explicit handler name override",
        )

        @field_validator("handler_mode")
        @classmethod
        def validate_mode(cls, v: str | None) -> str | None:
            """Validate handler mode against allowed values."""
            if v is None:
                return None
            allowed = {
                c.Cqrs.HandlerType.COMMAND,
                c.Cqrs.HandlerType.QUERY,
                c.Cqrs.HandlerType.EVENT,
            }
            if v not in allowed:
                return v
            return v

    class RegistrationDetails(BaseModel):
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

        model_config = ConfigDict(
            json_schema_extra={
                "title": "RegistrationDetails",
                "description": "Handler registration tracking details",
            },
        )

        registration_id: Annotated[
            str,
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Unique registration identifier",
                examples=["reg-abc123", "handler-create-user-001"],
            ),
        ]
        handler_mode: Annotated[
            c.Cqrs.HandlerType,
            Field(
                default=c.Cqrs.HandlerType.COMMAND,
                description="Handler mode (command, query, or event)",
                examples=["command", "query", "event"],
            ),
        ] = c.Cqrs.HandlerType.COMMAND
        timestamp: Annotated[
            str,
            Field(
                default_factory=lambda: c.Cqrs.DEFAULT_TIMESTAMP,
                description="ISO 8601 registration timestamp",
                examples=["2025-01-01T00:00:00Z", "2025-10-12T15:30:00+00:00"],
                pattern=c.Platform.PATTERN_ISO8601_TIMESTAMP,
            ),
        ] = Field(default_factory=lambda: c.Cqrs.DEFAULT_TIMESTAMP)
        status: Annotated[
            c.Cqrs.CommonStatus,
            Field(
                default=c.Cqrs.CommonStatus.RUNNING,
                description="Current registration status",
                examples=["running", "stopped", "failed"],
            ),
        ] = c.Cqrs.CommonStatus.RUNNING

        @field_validator("timestamp", mode="after")
        @classmethod
        def validate_timestamp_format(cls, v: str) -> str:
            """Validate timestamp is in ISO 8601 format (using FlextRuntime)."""
            # Allow empty strings
            if not v:
                return v
            # Validate ISO 8601 format - removed due to missing method
            return v

    class ExecutionContext(BaseModel):
        """Handler execution context for tracking handler performance and state.

        Provides timing and metrics tracking for handler executions in the
        FlextContext system. Uses Pydantic 2 PrivateAttr for internal state.

        This mutable context object tracks handler execution performance,
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

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_schema_extra={
                "title": "HandlerExecutionContext",
                "description": "Handler execution context for tracking performance and state",
            },
        )

        handler_name: Annotated[
            str,
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Name of the handler being executed",
                examples=["ProcessOrderCommand", "GetUserQuery", "OrderCreatedEvent"],
            ),
        ]
        handler_mode: Annotated[
            c.Cqrs.HandlerTypeLiteral,
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Mode of handler execution",
                examples=["command", "query", "event"],
            ),
        ]
        # Use PrivateAttr for internal state (Pydantic v2 pattern)
        # PrivateAttr fields are not validated by Pydantic, so pyright needs explicit type hints
        _start_time: float | None = PrivateAttr(default=None)
        _metrics_state: t.Dict | None = PrivateAttr(default=None)

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
            # Use PrivateAttr for proper Pydantic v2 pattern
            self._start_time = time.time()

        @computed_field
        def execution_time_ms(self) -> float:
            """Get execution time in milliseconds.

            Returns:
                Execution time in milliseconds, or 0.0 if not started

            Examples:
                >>> context = FlextModelsHandler.ExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> context.start_execution()
                >>> # ... handler executes ...
                >>> elapsed = context.execution_time_ms
                >>> elapsed.__class__ is float
                True

            """
            if self._start_time is None:
                return 0.0

            # Type narrowing: _start_time is not None after check, so it's float
            # PrivateAttr type narrowing works after None check
            start_time: float = self._start_time
            elapsed: float = time.time() - start_time
            return round(elapsed * c.MILLISECONDS_MULTIPLIER, 2)

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
                # Use PrivateAttr for proper Pydantic v2 pattern
                self._metrics_state = t.Dict(root={})
            # Type narrowing: _metrics_state is not None after initialization above
            # PrivateAttr type narrowing works after None check and initialization
            metrics_state_val: t.Dict = self._metrics_state
            # ConfigurationDict (dict) is compatible with ConfigurationMapping (Mapping)
            # dict implements Mapping, so direct return works without cast
            return metrics_state_val

        def set_metrics_state(
            self,
            state: t.Dict,
        ) -> None:
            """Set metrics state.

            Direct assignment to _metrics_state. Use this to update metrics.

            Args:
                state: Metrics state to set

            Examples:
                >>> context = FlextModelsHandler.ExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> context.set_metrics_state({"items_processed": 42, "errors": 0})

            """
            # Use PrivateAttr for proper Pydantic v2 pattern
            self._metrics_state = state

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
            # Use PrivateAttr for proper Pydantic v2 pattern
            self._start_time = None
            self._metrics_state = None

        @classmethod
        def create_for_handler(
            cls,
            handler_name: str,
            handler_mode: c.Cqrs.HandlerTypeLiteral,
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

        @computed_field
        def is_running(self) -> bool:
            """Check if execution is currently running."""
            # Type narrowing: PrivateAttr type narrowing works directly
            return self._start_time is not None

        @computed_field
        def has_metrics(self) -> bool:
            """Check if metrics have been recorded."""
            # Type narrowing: PrivateAttr type narrowing works directly
            return self._metrics_state is not None and bool(self._metrics_state)

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
            ...     timeout=c.Network.DEFAULT_TIMEOUT,
            ...     middleware=[LoggingMiddleware, ValidationMiddleware],
            ... )
            >>> config.command
            <class 'CreateUserCommand'>
            >>> config.priority
            10

        """

        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

        command: Annotated[
            type,
            Field(
                description="Command type this handler processes",
            ),
        ]
        priority: Annotated[
            int,
            Field(
                default=c.Discovery.DEFAULT_PRIORITY,
                description="Handler priority (higher = processed first)",
                ge=0,
            ),
        ] = c.Discovery.DEFAULT_PRIORITY
        timeout: Annotated[
            float | None,
            Field(
                default=c.Discovery.DEFAULT_TIMEOUT,
                description="Handler execution timeout in seconds",
                gt=0.0,
            ),
        ] = c.Discovery.DEFAULT_TIMEOUT
        middleware: Annotated[
            list[type[p.Middleware]],
            Field(
                default_factory=list,
                description="Middleware types to apply to this handler",
            ),
        ]


__all__ = ["FlextModelsHandler"]
