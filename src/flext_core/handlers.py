"""Layer 13: Unified CQRS handler base promoted for the FLEXT 1.0.0 rollout.

This module provides FlextHandlers base classes for implementing CQRS command
and query handlers throughout the FLEXT ecosystem. Use FlextHandlers for all
handler implementations in FLEXT applications.

Dependency Layer: 13 (Application Services)
Dependencies: FlextConstants, FlextTypes, FlextExceptions, FlextResult,
              FlextConfig, FlextUtilities, FlextLoggings, FlextMixins,
              FlextModels, FlextContainer, FlextProcessors, FlextCqrs
Used by: FlextBus, FlextDispatcher, FlextRegistry, and ecosystem handler implementations

Simplified and refactored to use extracted components for reduced complexity
while maintaining all functionality. Uses FlextConfig, FlextUtilities,
FlextContext for modular, reusable handler operations.

Usage:
    ```python
    from flext_core.result import FlextResult
    from flext_core.handlers import FlextHandlers


    class UserCommandHandler(FlextHandlers[CreateUserCommand, User]):
        def handle(self, command: CreateUserCommand) -> FlextResult[User]:
            # Implement command handling logic
            return FlextResult[User].ok(created_user)
    ```

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar, cast

from flext_core.context import FlextContext
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

if TYPE_CHECKING:
    from flext_core.loggings import FlextLogger

HandlerModeLiteral = Literal["command", "query"]
HandlerTypeLiteral = Literal["command", "query"]

MessageT_contra = TypeVar("MessageT_contra", bound=object, contravariant=True)
ResultT = TypeVar("ResultT", bound=object)


class FlextHandlers[MessageT_contra, ResultT](FlextMixins, ABC):
    """Simplified CQRS handler base with extracted complexity.

    Reduced from 700+ lines to ~150 lines by delegating to:
    - FlextConfig.HandlerConfiguration for configuration management
    - FlextUtilities.TypeChecker for type compatibility checking
    - FlextUtilities.MessageValidator for message validation
    - FlextContext.HandlerExecutionContext for execution state
    - FlextHandlers.Metrics for logging coordination

    Maintains all functionality while achieving single responsibility principle.
    """

    def __init__(self, *, config: FlextModels.CqrsConfig.Handler) -> None:
        """Initialize handler with simplified single-config approach.

        Args:
            config: Handler configuration object (required)

        """
        super().__init__()
        self._config_model: FlextModels.CqrsConfig.Handler = config
        self._execution_context = (
            FlextContext.HandlerExecutionContext.create_for_handler(
                handler_name=config.handler_name,
                handler_mode=config.handler_type,
            )
        )
        self._accepted_message_types = (
            FlextUtilities.TypeChecker.compute_accepted_message_types(type(self))
        )
        self._revalidate_pydantic_messages = self._extract_revalidation_setting()
        self._type_warning_emitted = False

    def _extract_revalidation_setting(self) -> bool:
        """Extract revalidation setting from configuration."""
        # Check metadata first
        if hasattr(self._config_model, "metadata") and self._config_model.metadata:
            metadata_value = self._config_model.metadata.get(
                "revalidate_pydantic_messages"
            )
            if metadata_value is not None:
                # Handle string values
                if isinstance(metadata_value, str):
                    return metadata_value.lower() in {"true", "1", "yes"}
                return bool(metadata_value)

        # Default to False if not specified
        return False

    def can_handle(self, message_type: type) -> bool:
        """Check if this handler can handle the given message type.

        Args:
            message_type: The type of message to check

        Returns:
            bool: True if this handler can handle the message type

        """
        return FlextUtilities.TypeChecker.can_handle_message_type(
            self._accepted_message_types, message_type
        )

    @property
    def logger(self) -> FlextLogger:
        """Get logger instance for this handler.

        Returns:
            FlextLogger: Logger instance

        """
        from flext_core.loggings import FlextLogger  # noqa: PLC0415

        return FlextLogger(self.__class__.__name__)

    @property
    def handler_id(self) -> str:
        """Get handler ID from config.

        Returns:
            str: Handler ID

        """
        return self._config_model.handler_id

    @property
    def handler_name(self) -> str:
        """Get handler name from config.

        Returns:
            str: Handler name

        """
        return self._config_model.handler_name

    @property
    def mode(self) -> str:
        """Get handler mode from config.

        Returns:
            str: Handler mode (command or query)

        """
        # handler_mode is always defined as Literal["command", "query"]
        return self._config_model.handler_mode

    @property
    def config(self) -> FlextModels.CqrsConfig.Handler:
        """Get handler configuration.

        Returns:
            FlextModels.CqrsConfig.Handler: Handler configuration

        """
        return self._config_model

    def validate_command(self, command: object) -> FlextResult[None]:
        """Validate a command message.

        Args:
            command: The command to validate

        Returns:
            FlextResult indicating validation success or failure

        """
        return FlextUtilities.MessageValidator.validate_message(
            command,
            operation="command",
            revalidate_pydantic_messages=self._revalidate_pydantic_messages,
        )

    def validate_query(self, query: object) -> FlextResult[None]:
        """Validate a query message.

        Args:
            query: The query to validate

        Returns:
            FlextResult indicating validation success or failure

        """
        return FlextUtilities.MessageValidator.validate_message(
            query,
            operation="query",
            revalidate_pydantic_messages=self._revalidate_pydantic_messages,
        )

    def execute(self, message: MessageT_contra) -> FlextResult[ResultT]:
        """Execute the handler with the message.

        Args:
            message: The message to execute

        Returns:
            FlextResult containing the execution result or error

        """
        return self._run_pipeline(message, operation=self.mode)

    def handle_command(self, command: MessageT_contra) -> FlextResult[ResultT]:
        """Handle a command message.

        Args:
            command: The command to handle

        Returns:
            FlextResult containing the command handling result or error

        """
        return self.execute(command)

    def handle_query(self, query: MessageT_contra) -> FlextResult[ResultT]:
        """Handle a query message.

        Args:
            query: The query to handle

        Returns:
            FlextResult containing the query handling result or error

        """
        return self.execute(query)

    def _run_pipeline(
        self, message: MessageT_contra, operation: str = "command"
    ) -> FlextResult[ResultT]:
        """Run the handler pipeline with message processing.

        Args:
            message: The message to process
            operation: The operation type (command or query)

        Returns:
            FlextResult containing the processing result or error

        """
        import time  # noqa: PLC0415

        # Extract message ID
        message_id: str = "unknown"
        if isinstance(message, dict):
            message_dict = cast("FlextTypes.Core.Dict", message)
            message_id = (
                str(message_dict.get(f"{operation}_id", "unknown"))
                or str(message_dict.get("message_id", "unknown"))
                or "unknown"
            )
        elif hasattr(message, f"{operation}_id"):
            message_id = str(getattr(message, f"{operation}_id", "unknown"))
        elif hasattr(message, "message_id"):
            message_id = str(getattr(message, "message_id", "unknown"))

        message_type = type(message).__name__

        # Log start
        FlextHandlers.Metrics.log_handler_start(
            self.logger,
            self.mode,
            message_type,
            message_id,
        )

        # Validate operation matches handler mode
        if operation != self.mode:
            FlextHandlers.Metrics.log_mode_validation_error(
                logger=self.logger,
                error_message=f"Handler with mode '{self.mode}' cannot execute {operation} pipelines",
                expected_mode=self.mode,
                actual_mode=operation,
            )
            return FlextResult[ResultT].fail(
                f"Handler with mode '{self.mode}' cannot execute {operation} pipelines"
            )

        # Validate message can be handled
        message_type_obj = type(message)
        if not self.can_handle(message_type_obj):
            FlextHandlers.Metrics.log_handler_cannot_handle(
                logger=self.logger,
                error_message=f"Handler cannot handle message type {message_type}",
                handler_name=self.handler_name,
                message_type=message_type,
            )
            return FlextResult[ResultT].fail(
                f"Handler cannot handle message type {message_type}"
            )

        # Validate message
        FlextHandlers.Metrics.log_handler_processing(
            self.logger,
            self.mode,
            message_type,
            message_id,
        )
        validation_result = (
            self.validate_command(cast("object", message))
            if operation == "command"
            else self.validate_query(cast("object", message))
        )
        if validation_result.is_failure:
            return FlextResult[ResultT].fail(
                f"Message validation failed: {validation_result.error}"
            )

        # Execute handler
        start_time = time.time()
        try:
            result = self.handle(message)
            execution_time_ms = (time.time() - start_time) * 1000
            FlextHandlers.Metrics.log_handler_completion(
                self.logger,
                self.mode,
                message_type,
                message_id,
                execution_time_ms,
                success=result.is_success,
            )
            return result
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            exception_type = type(e).__name__
            FlextHandlers.Metrics.log_handler_error(
                self.logger,
                self.mode,
                message_type,
                message_id,
                execution_time_ms=execution_time_ms,
                exception_type=exception_type,
            )
            FlextHandlers.Metrics.log_handler_completion(
                self.logger,
                self.mode,
                message_type,
                message_id,
                execution_time_ms,
                success=False,
            )
            return FlextResult[ResultT].fail(f"Critical handler failure: {e!s}")

    @abstractmethod
    def handle(self, message: MessageT_contra) -> FlextResult[ResultT]:
        """Handle a message and return a result.

        Args:
            message: The message to handle

        Returns:
            FlextResult containing the result or error

        """
        ...

    @classmethod
    def from_callable(
        cls,
        callable_func: Callable[[object], object],
        handler_name: str | None = None,
        handler_type: Literal["command", "query"] = "command",
        mode: str | None = None,
        handler_config: FlextModels.CqrsConfig.Handler
        | dict[str, object]
        | None = None,
    ) -> FlextHandlers[object, object]:
        """Create a handler from a callable function.

        Args:
            callable_func: The callable function to wrap
            handler_name: Name for the handler (defaults to function name)
            handler_type: Type of handler (command, query, etc.)
            mode: Handler mode (for compatibility)
            handler_config: Optional handler configuration

        Returns:
            A FlextHandlers instance wrapping the callable

        """
        # Ensure handler_name is always a string
        resolved_handler_name: str = (
            handler_name
            if handler_name is not None
            else getattr(callable_func, "__name__", "unknown_handler")
        )

        # Use mode if provided (compatibility), otherwise use handler_type
        effective_type: Literal["command", "query"] = (
            cast("Literal['command', 'query']", mode)
            if mode is not None
            else handler_type
        )

        # Validate mode/handler_type
        if effective_type not in {"command", "query"}:
            msg = (
                f"Invalid handler mode: {effective_type}. Must be 'command' or 'query'"
            )
            raise ValueError(msg)

        # Use provided config or create default
        if handler_config is not None:
            if isinstance(handler_config, dict):
                # Merge defaults with provided dict (dict values override defaults)
                config_data = {
                    "handler_id": f"{resolved_handler_name}_{id(callable_func)}",
                    "handler_name": resolved_handler_name,
                    "handler_type": effective_type,
                    "handler_mode": effective_type,
                    **handler_config,  # Override with provided values
                }
                try:
                    config = FlextModels.CqrsConfig.Handler.model_validate(config_data)
                except Exception as e:
                    msg = f"Invalid handler config: {e}"
                    raise ValueError(msg) from e
            else:
                config = handler_config
        else:
            try:
                config = FlextModels.CqrsConfig.Handler(
                    handler_id=f"{resolved_handler_name}_{id(callable_func)}",
                    handler_name=resolved_handler_name,
                    handler_type=effective_type,
                    handler_mode=effective_type,
                )
            except Exception as e:
                msg = f"Failed to create handler config: {e}"
                raise ValueError(msg) from e

        # Create a simple wrapper class
        class CallableHandler(FlextHandlers[object, object]):
            def handle(self, message: object) -> FlextResult[object]:
                try:
                    result = callable_func(message)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)
                except Exception as e:
                    return FlextResult[object].fail(str(e))

        return CallableHandler(config=config)

    class Metrics:
        """Metrics logging coordination for handlers using structured logging."""

        @staticmethod
        def log_handler_start(
            logger: FlextLogger | None,
            handler_mode: str,
            message_type: str,
            message_id: str,
        ) -> None:
            """Log handler start event with structured logging."""
            if logger is not None:
                logger.info(
                    "starting_handler_pipeline",
                    handler_mode=handler_mode,
                    message_type=message_type,
                    message_id=message_id,
                )

        @staticmethod
        def log_handler_processing(
            logger: FlextLogger | None,
            handler_mode: str,
            message_type: str,
            message_id: str,
        ) -> None:
            """Log handler processing event with structured logging."""
            if logger is not None:
                logger.debug(
                    "processing_message",
                    handler_mode=handler_mode,
                    message_type=message_type,
                    message_id=message_id,
                )

        @staticmethod
        def log_handler_completion(
            logger: FlextLogger | None,
            handler_mode: str,
            message_type: str,
            message_id: str,
            execution_time_ms: float,
            *,  # keyword-only arguments
            success: bool,
        ) -> None:
            """Log handler completion event with structured logging."""
            # Always use info() for both success and failure
            if logger is not None:
                logger.info(
                    "handler_pipeline_completed",
                    handler_mode=handler_mode,
                    message_type=message_type,
                    message_id=message_id,
                    execution_time_ms=execution_time_ms,
                    success=success,
                )

        @staticmethod
        def log_handler_error(
            logger: FlextLogger | None,
            handler_mode: str,
            message_type: str,
            message_id: str,
            execution_time_ms: float | None = None,
            exception_type: str | None = None,
            error_code: str | None = None,
            correlation_id: str | None = None,
        ) -> None:
            """Log handler error event with structured logging."""
            if logger is not None:
                kwargs: dict[str, object] = {
                    "handler_mode": handler_mode,
                    "message_type": message_type,
                    "message_id": message_id,
                }
                if execution_time_ms is not None:
                    kwargs["execution_time_ms"] = (
                        execution_time_ms  # Keep as float - tests expect this
                    )
                if exception_type is not None:
                    kwargs["exception_type"] = exception_type
                if error_code is not None:
                    kwargs["error_code"] = error_code
                if correlation_id is not None:
                    kwargs["correlation_id"] = correlation_id

                logger.error("handler_critical_failure", **kwargs)

        @staticmethod
        def log_mode_validation_error(
            logger: FlextLogger | None,
            error_message: str,
            expected_mode: str | None = None,
            actual_mode: str | None = None,
        ) -> None:
            """Log mode validation error with structured logging."""
            if logger is not None:
                kwargs = {"error_message": error_message}
                if expected_mode is not None:
                    kwargs["expected_mode"] = expected_mode
                if actual_mode is not None:
                    kwargs["actual_mode"] = actual_mode

                logger.error("invalid_handler_mode", **kwargs)

        @staticmethod
        def log_handler_cannot_handle(
            logger: FlextLogger | None,
            error_message: str,
            handler_name: str | None = None,
            message_type: str | None = None,
        ) -> None:
            """Log handler cannot handle message type with structured logging."""
            if logger is not None:
                kwargs = {"error_message": error_message}
                if handler_name is not None:
                    kwargs["handler_name"] = handler_name
                if message_type is not None:
                    kwargs["message_type"] = message_type

                logger.error("handler_cannot_handle", **kwargs)


__all__: FlextTypes.Core.StringList = [
    "FlextHandlers",
]
