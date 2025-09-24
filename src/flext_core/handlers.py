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
    from flext_core import FlextHandlers, FlextResult


    class UserCommandHandler(FlextHandlers[CreateUserCommand, User]):
        def handle(self, command: CreateUserCommand) -> FlextResult[User]:
            # Implement command handling logic
            return FlextResult[User].ok(created_user)
    ```

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, TypeVar, cast, override

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

HandlerModeLiteral = Literal["command", "query"]
HandlerTypeLiteral = Literal["command", "query"]

MessageT = TypeVar("MessageT")
ResultT = TypeVar("ResultT")


class FlextHandlers[MessageT, ResultT](FlextMixins, ABC):
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
        self._config_model = config
        self._execution_context = (
            FlextContext.HandlerExecutionContext.create_for_handler(
                handler_name=config.handler_name,
                handler_mode=config.handler_type,
            )
        )
        self._accepted_message_types = (
            FlextUtilities.TypeChecker.compute_accepted_message_types(self.__class__)
        )
        self._revalidate_pydantic_messages = self._extract_revalidation_setting()
        self._type_warning_emitted = False  # Track type warning state for compatibility  # Track type warning state for compatibility

    def _extract_revalidation_setting(self) -> bool:
        """Extract revalidation setting from config metadata."""
        metadata: FlextTypes.Core.Dict | None = getattr(
            self._config_model, "metadata", None
        )
        if isinstance(metadata, dict):
            # Use proper type annotation with Python 3.13+ syntax
            raw_flag: str | bool | None = cast(
                "str | bool | None", metadata.get("revalidate_pydantic_messages")
            )
            if isinstance(raw_flag, bool):
                return raw_flag
            if isinstance(raw_flag, str):
                normalized = raw_flag.strip().lower()
                return normalized in {"1", "true", "yes"}
        return False

    @property
    def mode(self) -> HandlerTypeLiteral:
        """Return configured handler mode from the config model."""
        return self._config_model.handler_type

    @property
    def handler_name(self) -> str:
        """Get handler name for identification."""
        return str(self._config_model.handler_name)

    @property
    def handler_id(self) -> str:
        """Get handler ID for identification."""
        return str(self._config_model.handler_id)

    @property
    def logger(self) -> FlextLogger:
        """Get logger instance for this handler."""
        return FlextLogger(self.__class__.__name__)

    @property
    def config(self) -> FlextModels.CqrsConfig.Handler:
        """Return validated handler configuration model."""
        return self._config_model

    def can_handle(self, message_type: object) -> bool:
        """Check if handler can process this message type."""
        return FlextUtilities.TypeChecker.can_handle_message_type(
            self._accepted_message_types, message_type
        )

    def validate_command(self, command: object) -> FlextResult[None]:
        """Validate command using extracted validation logic.

        🚨 AUDIT VIOLATION: This validation method violates FLEXT architectural principles!
        ❌ CRITICAL ISSUE: Command validation should be centralized in FlextModels.Validation
        ❌ INLINE VALIDATION: This delegates to utilities validation instead of centralized validation

        🔧 REQUIRED ACTION:
        - Remove this validation method from handlers
        - Use FlextModels.Command validation patterns directly
        - Centralize validation in FlextModels.Validation

        📍 SHOULD BE USED INSTEAD: FlextModels.Command validation patterns
        """
        # 🚨 AUDIT VIOLATION: Delegating to utilities validation - should use FlextModels.Validation
        return FlextUtilities.MessageValidator.validate_message(
            command,
            operation="command",
            revalidate_pydantic_messages=self._revalidate_pydantic_messages,
        )

    def validate_query(self, query: object) -> FlextResult[None]:
        """Validate query using extracted validation logic.

        🚨 AUDIT VIOLATION: This validation method violates FLEXT architectural principles!
        ❌ CRITICAL ISSUE: Query validation should be centralized in FlextModels.Validation
        ❌ INLINE VALIDATION: This delegates to utilities validation instead of centralized validation

        🔧 REQUIRED ACTION:
        - Remove this validation method from handlers
        - Use FlextModels.Query validation patterns directly
        - Centralize validation in FlextModels.Validation

        📍 SHOULD BE USED INSTEAD: FlextModels.Query validation patterns
        """
        # 🚨 AUDIT VIOLATION: Delegating to utilities validation - should use FlextModels.Validation
        return FlextUtilities.MessageValidator.validate_message(
            query,
            operation="query",
            revalidate_pydantic_messages=self._revalidate_pydantic_messages,
        )

    @abstractmethod
    def handle(self, message: MessageT) -> FlextResult[ResultT]:
        """Handle the message and return result.

        Subclasses must override this method.
        """

    def execute(self, message: MessageT) -> FlextResult[ResultT]:
        """Execute message with full validation and error handling."""
        return self._run_pipeline(message, operation=self.mode)

    def handle_query(self, query: MessageT) -> FlextResult[ResultT]:
        """Execute query with validation and error handling."""
        return self.execute(query)

    def handle_command(self, command: MessageT) -> FlextResult[ResultT]:
        """Execute command with validation and error handling."""
        return self.execute(command)

    def _run_pipeline(
        self,
        message: MessageT,
        *,
        operation: HandlerTypeLiteral,
    ) -> FlextResult[ResultT]:
        """Execute handler pipeline using railway-oriented programming."""
        message_type = type(message).__name__
        # Extract identifier from message, handling both object attributes and dict keys
        if isinstance(message, dict):
            identifier = message.get(
                f"{operation}_id",
                message.get("id", FlextConstants.Messages.UNKNOWN_ERROR),
            )
        else:
            identifier = getattr(
                message,
                f"{operation}_id",
                getattr(message, "id", FlextConstants.Messages.UNKNOWN_ERROR),
            )

        # Log start using extracted metrics
        FlextHandlers.Metrics.log_handler_start(
            self.logger, self.mode, message_type, str(identifier)
        )

        # Use railway pattern for validation chain
        validation_result = FlextResult.chain_validations(
            lambda: self._validate_mode(operation),
            lambda: self._validate_can_handle(message),
            lambda: self._validate_message(message, operation=operation),
        )

        return validation_result.flat_map(
            lambda _: self._execute_with_timing(message, message_type, str(identifier))
        )

    def _validate_mode(self, operation: HandlerTypeLiteral) -> FlextResult[None]:
        """Validate handler mode matches operation type."""
        if self.mode != operation:
            error_msg = (
                f"{self.handler_name} is configured for {self.mode} operations "
                f"and cannot execute {operation} pipelines"
            )
            FlextHandlers.Metrics.log_mode_validation_error(
                self.logger, error_msg, operation, self.mode
            )
            return FlextResult[None].fail(
                error_msg,
                error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
            )
        return FlextResult[None].ok(None)

    def _validate_can_handle(self, message: MessageT) -> FlextResult[None]:
        """Validate handler can process this message type."""
        if not self.can_handle(type(message)):
            message_type = type(message).__name__
            error_msg = f"{self.handler_name} cannot handle {message_type}"
            FlextHandlers.Metrics.log_handler_cannot_handle(
                self.logger, error_msg, self.handler_name, message_type
            )
            return FlextResult[None].fail(
                error_msg,
                error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
            )
        return FlextResult[None].ok(None)

    def _validate_message(
        self,
        message: object,
        *,
        operation: HandlerTypeLiteral,
    ) -> FlextResult[None]:
        """Validate message using extracted validation logic."""
        return FlextUtilities.MessageValidator.validate_message(
            message,
            operation=operation,
            revalidate_pydantic_messages=self._revalidate_pydantic_messages,
        )

    def _execute_with_timing(
        self, message: MessageT, message_type: str, identifier: str
    ) -> FlextResult[ResultT]:
        """Execute handler with timing using extracted execution context."""
        self._execution_context.start_execution()

        try:
            FlextHandlers.Metrics.log_handler_processing(
                self.logger, self.mode, message_type, identifier
            )

            result: FlextResult[ResultT] = self.handle(message)
            execution_time_ms = self._execution_context.get_execution_time_ms()

            FlextHandlers.Metrics.log_handler_completion(
                self.logger,
                self.mode,
                message_type,
                identifier,
                execution_time_ms,
                success=result.is_success,
            )
            return result

        except Exception as exc:
            execution_time_ms = self._execution_context.get_execution_time_ms()
            correlation_id = f"handler_{identifier}_{int(time.time() * 1000)}"

            critical_error = FlextExceptions.CriticalError(
                f"Critical handler failure during {self.mode} processing: {exc}",
                context={
                    "handler_mode": self.mode,
                    "message_type": message_type,
                    "message_id": identifier,
                    "execution_time_ms": execution_time_ms,
                    "handler_name": self.handler_name,
                    "exception_type": type(exc).__name__,
                    "original_exception": str(exc),
                },
                correlation_id=correlation_id,
            )

            FlextHandlers.Metrics.log_handler_error(
                self.logger,
                self.mode,
                message_type,
                identifier,
                execution_time_ms,
                type(exc).__name__,
                critical_error.error_code,
                critical_error.correlation_id,
            )

            return FlextResult[ResultT].fail(
                str(critical_error),
                error_code=critical_error.error_code,
                error_data={"exception_context": critical_error.context},
            )

    @staticmethod
    def from_callable(
        handler_func: Callable[[object], object | FlextResult[object]],
        *,
        mode: str,
        handler_config: FlextModels.CqrsConfig.Handler
        | dict[str, object]
        | None = None,
        handler_name: str | None = None,
    ) -> FlextHandlers[object, object]:
        """Create a FlextHandlers instance from a plain callable."""
        valid_modes = {
            FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
        }
        if mode not in valid_modes:
            msg = f"Invalid handler mode: {mode}"
            raise ValueError(msg)

        resolved_name = handler_name or getattr(
            handler_func,
            "__name__",
            "FunctionHandler",
        )

        # Create config dict using FlextConfig.HandlerConfiguration
        # Convert handler_config to dict format if it's a model
        config_dict = None
        if isinstance(handler_config, FlextModels.CqrsConfig.Handler):
            config_dict = handler_config.model_dump()
        elif isinstance(handler_config, dict):
            config_dict = handler_config

        # Get raw config dict from config.py (no models.py dependency)
        raw_config = FlextConfig.HandlerConfiguration.create_handler_config(
            handler_mode=mode,
            handler_name=resolved_name,
            handler_config=config_dict,
        )

        # Convert raw config dict to validated model
        typed_config = FlextModels.CqrsConfig.Handler.model_validate(raw_config)

        class CallableHandler(FlextHandlers[object, object]):
            def __init__(self) -> None:
                super().__init__(config=typed_config)
                self._handler_func = handler_func

            @override
            def handle(self, message: object) -> FlextResult[object]:
                try:
                    result = self._handler_func(message)
                except FlextExceptions.BaseError as exc:
                    return FlextResult[object].fail(
                        str(exc),
                        error_code=getattr(exc, "error_code", None),
                        error_data={
                            "exception_context": getattr(exc, "context", {}),
                        },
                    )
                except Exception as exc:
                    processing_error = FlextExceptions.ProcessingError(
                        f"Handler callable raised: {exc}",
                        business_rule=f"{mode}_handler_callable",
                        operation=f"handle_{mode}",
                        context={
                            "handler_name": resolved_name,
                            "message_type": type(message).__name__,
                        },
                    )
                    return FlextResult[object].fail(
                        str(processing_error),
                        error_code=processing_error.error_code,
                        error_data={
                            "exception_context": processing_error.context,
                        },
                    )

                if isinstance(result, FlextResult):
                    return cast("FlextResult[object]", result)
                return FlextResult[object].ok(result)

        return CallableHandler()

    class Metrics:
        """Handler metrics utilities for FlextHandlers complexity reduction.

        Extracts metrics collection and logging coordination from FlextHandlers
        to simplify handler execution and provide reusable metrics patterns.
        """

        @classmethod
        def log_handler_start(
            cls,
            logger: object,
            handler_mode: str,
            message_type: str,
            message_id: str,
        ) -> None:
            """Log start of handler pipeline.

            Args:
                logger: Logger instance to use
                handler_mode: Handler mode (command/query)
                message_type: Message type name
                message_id: Message identifier

            """
            if isinstance(logger, FlextLogger):
                logger.info(
                    "starting_handler_pipeline",
                    handler_mode=handler_mode,
                    message_type=message_type,
                    message_id=message_id,
                )

        @classmethod
        def log_handler_processing(
            cls,
            logger: object,
            handler_mode: str,
            message_type: str,
            message_id: str,
        ) -> None:
            """Log handler processing message.

            Args:
                logger: Logger instance to use
                handler_mode: Handler mode (command/query)
                message_type: Message type name
                message_id: Message identifier

            """
            if isinstance(logger, FlextLogger):
                logger.debug(
                    "processing_message",
                    handler_mode=handler_mode,
                    message_type=message_type,
                    message_id=message_id,
                )

        @classmethod
        def log_handler_completion(
            cls,
            logger: object,
            handler_mode: str,
            message_type: str,
            message_id: str,
            execution_time_ms: float,
            *,
            success: bool,
        ) -> None:
            """Log handler pipeline completion.

            Args:
                logger: Logger instance to use
                handler_mode: Handler mode (command/query)
                message_type: Message type name
                message_id: Message identifier
                execution_time_ms: Execution time in milliseconds
                success: Whether the operation was successful

            """
            if isinstance(logger, FlextLogger):
                logger.info(
                    "handler_pipeline_completed",
                    handler_mode=handler_mode,
                    message_type=message_type,
                    message_id=message_id,
                    execution_time_ms=execution_time_ms,
                    success=success,
                )

        @classmethod
        def log_handler_error(
            cls,
            logger: object,
            handler_mode: str,
            message_type: str,
            message_id: str,
            execution_time_ms: float,
            exception_type: str,
            error_code: str,
            correlation_id: str,
        ) -> None:
            """Log handler critical failure.

            Args:
                logger: Logger instance to use
                handler_mode: Handler mode (command/query)
                message_type: Message type name
                message_id: Message identifier
                execution_time_ms: Execution time in milliseconds
                exception_type: Type of exception that occurred
                error_code: Error code from the exception
                correlation_id: Correlation ID for tracking

            """
            if isinstance(logger, FlextLogger):
                logger.error(
                    "handler_critical_failure",
                    handler_mode=handler_mode,
                    message_type=message_type,
                    message_id=message_id,
                    execution_time_ms=execution_time_ms,
                    exception_type=exception_type,
                    error_code=error_code,
                    correlation_id=correlation_id,
                )

        @classmethod
        def log_mode_validation_error(
            cls,
            logger: object,
            error_message: str,
            expected_mode: str,
            actual_mode: str,
        ) -> None:
            """Log handler mode validation error.

            Args:
                logger: Logger instance to use
                error_message: Error message to log
                expected_mode: Expected handler mode
                actual_mode: Actual handler mode

            """
            if isinstance(logger, FlextLogger):
                logger.error(
                    "invalid_handler_mode",
                    error_message=error_message,
                    expected_mode=expected_mode,
                    actual_mode=actual_mode,
                )

        @classmethod
        def log_handler_cannot_handle(
            cls,
            logger: object,
            error_message: str,
            handler_name: str,
            message_type: str,
        ) -> None:
            """Log handler cannot handle message error.

            Args:
                logger: Logger instance to use
                error_message: Error message to log
                handler_name: Name of the handler
                message_type: Type of message that couldn't be handled

            """
            if isinstance(logger, FlextLogger):
                logger.error(
                    "handler_cannot_handle",
                    error_message=error_message,
                    handler_name=handler_name,
                    message_type=message_type,
                )


__all__: FlextTypes.Core.StringList = [
    "FlextHandlers",
]
