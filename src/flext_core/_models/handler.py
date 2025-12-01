"""Handler management patterns extracted from FlextModels.

This module contains the FlextModelsHandler class with all handler-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Handler instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time as time_module
from collections.abc import Callable
from typing import Annotated, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
)

from flext_core._models.base import FlextModelsBase
from flext_core._utilities.validation import FlextUtilitiesValidation
from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes

# FlextUtilitiesValidation is safe to import at module level:
# - validation.py uses ResultProtocol (not concrete FlextResult) to break circular import
# - circular import issue was RESOLVED (validation.py doesn't import handler.py)


class FlextModelsHandler:
    """Handler management pattern container class.

    This class acts as a namespace container for handler management patterns.
    All nested classes are accessed via FlextModels.Handler.* in the main models.py.
    """

    class Registration(FlextModelsBase.ArbitraryTypesModel):
        """Handler registration with advanced validation."""

        name: str = Field(min_length=1, description="Handler name")
        handler: (
            Callable[[], FlextTypes.GeneralValueType]
            | Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType]
            | Callable[
                [FlextTypes.GeneralValueType, FlextTypes.GeneralValueType],
                FlextTypes.GeneralValueType,
            ]
        )
        event_types: list[str] = Field(
            default_factory=list,
            description="Event types this handler processes",
        )

        @field_validator("handler", mode="after")
        @classmethod
        def validate_handler(
            cls,
            v: FlextTypes.GeneralValueType | Callable[..., FlextTypes.GeneralValueType],
        ) -> (
            Callable[[], FlextTypes.GeneralValueType]
            | Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType]
            | Callable[
                [FlextTypes.GeneralValueType, FlextTypes.GeneralValueType],
                FlextTypes.GeneralValueType,
            ]
        ):
            """Validate handler is properly callable (direct validation, no circular imports)."""
            # Direct callable check - avoid circular import via FlextUtilitiesValidation
            if not callable(v):
                msg = f"Handler must be callable, got {type(v).__name__}"
                raise TypeError(msg)
            # Type-safe return: v is confirmed callable by validation above
            # Cast to handler callable since we've validated it's callable
            return cast(
                (
                    "Callable[[], FlextTypes.GeneralValueType] | Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType] | Callable[[FlextTypes.GeneralValueType, FlextTypes.GeneralValueType], FlextTypes.GeneralValueType]"
                ),
                v,
            )

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
                min_length=1,
                description="Unique registration identifier",
                examples=["reg-abc123", "handler-create-user-001"],
            ),
        ]
        handler_mode: Annotated[
            FlextConstants.Cqrs.HandlerType,
            Field(
                default=FlextConstants.Cqrs.HandlerType.COMMAND,
                description="Handler mode (command, query, or event)",
                examples=["command", "query", "event"],
            ),
        ] = FlextConstants.Cqrs.HandlerType.COMMAND
        timestamp: Annotated[
            str,
            Field(
                default_factory=lambda: FlextConstants.Cqrs.DEFAULT_TIMESTAMP,
                description="ISO 8601 registration timestamp",
                examples=["2025-01-01T00:00:00Z", "2025-10-12T15:30:00+00:00"],
                pattern=r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[Z+\-][0-9:]*)?$",
            ),
        ] = Field(default_factory=lambda: FlextConstants.Cqrs.DEFAULT_TIMESTAMP)
        status: Annotated[
            FlextConstants.Cqrs.Status,
            Field(
                default=FlextConstants.Cqrs.Status.RUNNING,
                description="Current registration status",
                examples=["running", "stopped", "failed"],
            ),
        ] = FlextConstants.Cqrs.Status.RUNNING

        @field_validator("timestamp", mode="after")
        @classmethod
        def validate_timestamp_format(cls, v: str) -> str:
            """Validate timestamp is in ISO 8601 format (using FlextUtilitiesValidation)."""
            result = FlextUtilitiesValidation.validate_iso8601_timestamp(
                v,
                allow_empty=True,
            )
            if result.is_failure:
                base_msg = "Timestamp validation failed"
                error_msg = (
                    f"{base_msg}: {result.error}"
                    if result.error
                    else f"{base_msg} (invalid timestamp value)"
                )
                raise ValueError(error_msg)
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
                min_length=1,
                description="Name of the handler being executed",
                examples=["ProcessOrderCommand", "GetUserQuery", "OrderCreatedEvent"],
            ),
        ]
        handler_mode: Annotated[
            FlextConstants.Cqrs.HandlerTypeLiteral,
            Field(
                min_length=1,
                description="Mode of handler execution",
                examples=["command", "query", "event"],
            ),
        ]
        _start_time: float | None = PrivateAttr(default=None)
        _metrics_state: dict[str, FlextTypes.GeneralValueType] | None = PrivateAttr(
            default=None,
        )

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
            self._start_time = time_module.time()

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
                >>> isinstance(elapsed, float)
                True

            """
            if self._start_time is None:
                return 0.0

            elapsed = time_module.time() - self._start_time
            return round(elapsed * 1000, 2)

        @computed_field
        def metrics_state(self) -> dict[str, FlextTypes.GeneralValueType]:
            """Get current metrics state.

            Returns:
                Dictionary containing metrics state (empty dict[str, FlextTypes.GeneralValueType] if not set)

            Examples:
                >>> context = FlextModelsHandler.ExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> metrics = context.metrics_state
                >>> FlextRuntime.is_dict_like(metrics)
                True

            """
            if self._metrics_state is None:
                self._metrics_state = {}
            return self._metrics_state

        def set_metrics_state(
            self,
            state: dict[str, FlextTypes.GeneralValueType],
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
            self._start_time = None
            self._metrics_state = None

        @classmethod
        def create_for_handler(
            cls,
            handler_name: str,
            handler_mode: FlextConstants.Cqrs.HandlerTypeLiteral,
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
            return self._start_time is not None

        @computed_field
        def has_metrics(self) -> bool:
            """Check if metrics have been recorded."""
            return self._metrics_state is not None and bool(self._metrics_state)


__all__ = ["FlextModelsHandler"]
