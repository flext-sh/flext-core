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
        @override
        def handle(self, command: CreateUserCommand) -> FlextResult[User]:
            # Implement command handling logic
            return FlextResult[User].ok(created_user)
    ```

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from typing import Literal, cast, override

from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

HandlerModeLiteral = Literal["command", "query", "event", "saga"]
HandlerTypeLiteral = Literal["command", "query", "event", "saga"]


class _MessageValidator:
    """Private message validation utilities for FlextHandlers.

    Moved from FlextUtilities to follow SOLID Single Responsibility Principle.
    This is handler-specific validation logic, not a general utility.
    """

    _SERIALIZABLE_MESSAGE_EXPECTATION = (
        "dict, str, int, float, bool, dataclass, attrs class, or object exposing "
        "model_dump/dict/as_dict/__slots__ representations"
    )

    @classmethod
    def validate_message(
        cls,
        message: object,
        *,
        operation: str,
        revalidate_pydantic_messages: bool = False,
    ) -> FlextResult[None]:
        """Validate a message for the given operation.

        Args:
            message: The message object to validate
            operation: The operation name for context
            revalidate_pydantic_messages: Whether to revalidate Pydantic models

        Returns:
            FlextResult[None]: Success if valid, failure with error details if invalid

        """
        # Check for custom validation methods first
        validation_method_name = f"validate_{operation}"
        if hasattr(message, validation_method_name):
            validation_method = getattr(message, validation_method_name)
            if callable(validation_method):
                try:
                    sig = inspect.signature(validation_method)
                    if len(sig.parameters) == 0:
                        validation_result_obj = validation_method()
                        if isinstance(validation_result_obj, FlextResult):
                            validation_result: FlextResult[object] = cast(
                                "FlextResult[object]", validation_result_obj
                            )
                            if validation_result.is_failure:
                                return FlextResult[None].fail(
                                    validation_result.error
                                    or f"{operation} validation failed",
                                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                                )
                except Exception as e:
                    # Skip if it's a Pydantic field validator - validation will proceed below
                    logger = FlextLogger(__name__)
                    logger.debug(
                        f"Skipping validation method {validation_method_name}: {type(e).__name__}"
                    )

        # If message is a Pydantic model, assume validated unless revalidation requested
        if isinstance(message, BaseModel):
            if not revalidate_pydantic_messages:
                return FlextResult[None].ok(None)

            try:
                message.__class__.model_validate(message.model_dump(mode="python"))
                return FlextResult[None].ok(None)
            except Exception as e:
                validation_error = FlextExceptions.ValidationError(
                    f"Pydantic revalidation failed: {e}",
                    field="pydantic_model",
                    value=str(message)[: FlextConstants.Defaults.MAX_MESSAGE_LENGTH]
                    if hasattr(message, "__str__")
                    else "unknown",
                    validation_details={
                        "pydantic_exception": str(e),
                        "model_class": message.__class__.__name__,
                        "revalidated": True,
                    },
                    context={
                        "operation": operation,
                        "message_type": type(message).__name__,
                        "validation_type": "pydantic_revalidation",
                        "revalidate_pydantic_messages": revalidate_pydantic_messages,
                    },
                    correlation_id=f"pydantic_validation_{int(time.time() * 1000)}",
                )
                return FlextResult[None].fail(
                    str(validation_error),
                    error_code=validation_error.error_code,
                    error_data={"exception_context": validation_error.context},
                )

        # For non-Pydantic objects, ensure serializable representation can be constructed
        try:
            cls._build_serializable_message_payload(message, operation=operation)
        except Exception as exc:
            if isinstance(exc, FlextExceptions.TypeError):
                return FlextResult[None].fail(
                    str(exc),
                    error_code=exc.error_code,
                    error_data={"exception_context": exc.context},
                )

            fallback_error = FlextExceptions.TypeError(
                f"Invalid message type for {operation}: {type(message).__name__}",
                expected_type=cls._SERIALIZABLE_MESSAGE_EXPECTATION,
                actual_type=type(message).__name__,
                context={
                    "operation": operation,
                    "message_type": type(message).__name__,
                    "validation_type": "serializable_check",
                    "original_exception": str(exc),
                },
                correlation_id=f"type_validation_{int(time.time() * 1000)}",
            )
            return FlextResult[None].fail(
                str(fallback_error),
                error_code=fallback_error.error_code,
                error_data={"exception_context": fallback_error.context},
            )

        return FlextResult[None].ok(None)

    @classmethod
    def _build_serializable_message_payload(
        cls,
        message: object,
        *,
        operation: str | None = None,
    ) -> object:
        """Build a serializable representation for message validation heuristics."""
        operation_name = operation or "message"
        context_operation = operation or "unknown"

        if isinstance(message, (dict, str, int, float, bool)):
            return cast("FlextTypes.Core.Dict | str | int | float | bool", message)

        if message is None:
            msg = f"Invalid message type for {operation_name}: NoneType"
            raise FlextExceptions.TypeError(
                msg,
                expected_type=cls._SERIALIZABLE_MESSAGE_EXPECTATION,
                actual_type="NoneType",
                context={
                    "operation": context_operation,
                    "message_type": type(None),
                    "validation_type": "serializable_check",
                },
                correlation_id=f"message_serialization_{int(time.time() * 1000)}",
            )

        if isinstance(message, BaseModel):
            return message.model_dump()

        if is_dataclass(message) and not isinstance(message, type):
            return asdict(message)

        # Handle attrs classes
        attrs_fields = getattr(message, "__attrs_attrs__", None)
        if (
            attrs_fields is not None
            and not isinstance(message, type)
            and hasattr(message, "__attrs_attrs__")
            and hasattr(message, "__class__")
        ):
            result: FlextTypes.Core.Dict = {}
            for attr_field in attrs_fields:
                field_name = attr_field.name
                if hasattr(message, field_name):
                    result[field_name] = getattr(message, field_name)
            return result

        # Try common serialization methods
        for method_name in ("model_dump", "dict", "as_dict"):
            method = getattr(message, method_name, None)
            if callable(method):
                try:
                    result_data = method()
                    if isinstance(result_data, dict):
                        return cast("FlextTypes.Core.Dict", result_data)
                except Exception as e:
                    logger = FlextLogger(__name__)
                    logger.debug(
                        f"Serialization method {method_name} failed: {type(e).__name__}"
                    )
                    continue

        # Handle __slots__
        slots = getattr(message, "__slots__", None)
        if slots:
            if isinstance(slots, str):
                slot_names: tuple[str, ...] = (slots,)
            elif isinstance(slots, (list, tuple)):
                slot_names = tuple(cast("list[str] | tuple[str, ...]", slots))
            else:
                msg = f"Invalid __slots__ type for {operation_name}: {type(slots).__name__}"
                raise FlextExceptions.TypeError(
                    msg,
                    expected_type="str, list, or tuple",
                    actual_type=type(slots).__name__,
                    context={
                        "operation": context_operation,
                        "message_type": type(message).__name__,
                        "validation_type": "serializable_check",
                        "__slots__": repr(slots),
                    },
                    correlation_id=f"message_serialization_{int(time.time() * 1000)}",
                )

            def get_slot_value(slot_name: str) -> object:
                return getattr(message, slot_name)

            return {
                slot_name: get_slot_value(slot_name)
                for slot_name in slot_names
                if hasattr(message, slot_name)
            }

        if hasattr(message, "__dict__"):
            return vars(message)

        msg = f"Invalid message type for {operation_name}: {type(message).__name__}"
        raise FlextExceptions.TypeError(
            msg,
            expected_type=cls._SERIALIZABLE_MESSAGE_EXPECTATION,
            actual_type=type(message).__name__,
            context={
                "operation": context_operation,
                "message_type": type(message).__name__,
                "validation_type": "serializable_check",
            },
            correlation_id=f"message_serialization_{int(time.time() * 1000)}",
        )


class FlextHandlers[MessageT_contra, ResultT](FlextMixins, ABC):
    """Handler base class for CQRS command and query implementations.

    FlextHandlers provides the foundation for implementing CQRS handlers
    with validation, execution context, metrics collection, and
    configuration management. Generic base supporting commands, queries,
    events, and sagas across all 32+ FLEXT projects.

    **Function**: Base class for CQRS handler implementations
        - Abstract base for command/query/event handlers
        - Handler execution with validation pipeline
        - Type checking for message compatibility
        - Metrics collection for handler performance
        - Configuration via FlextModels.CqrsConfig.Handler
        - Execution context tracking per handler
        - Message validation with FlextResult
        - Pydantic message revalidation support
        - Logger integration for handler operations
        - Support for handler_type (command/query/event/saga)
        - Generic types for message and result type safety
        - Integration with FlextContext for state

    **Uses**: CQRS infrastructure and validation components
        - FlextMixins for reusable behavior patterns
        - ABC for abstract base class enforcement
        - FlextResult[T] for all operation results
        - FlextModels.CqrsConfig.Handler for configuration
        - FlextContext.HandlerExecutionContext for state
        - FlextUtilities.TypeChecker for type validation
        - FlextUtilities.MessageValidator for messages
        - FlextLogger for handler operation logging
        - FlextConstants for handler defaults
        - FlextExceptions for structured errors
        - Generic types MessageT_contra and ResultT
        - inspect module for type introspection

    **How to use**: Implement handlers by subclassing
        ```python
        from flext_core import FlextHandlers, FlextResult, FlextModels


        # Example 1: Command handler implementation
        class CreateUserHandler(FlextHandlers[CreateUserCommand, User]):
            def __init__(self):
                config = FlextModels.CqrsConfig.Handler(
                    handler_name="CreateUserHandler", handler_type="command"
                )
                super().__init__(config=config)

            def handle(self, command: CreateUserCommand) -> FlextResult[User]:
                # Validate command
                validation = self._validate_message(command)
                if validation.is_failure:
                    return FlextResult[User].fail(validation.error)

                # Execute business logic
                user = User(name=command.name, email=command.email)
                return FlextResult[User].ok(user)


        # Example 2: Query handler with caching
        class GetUserHandler(FlextHandlers[GetUserQuery, User]):
            def __init__(self):
                config = FlextModels.CqrsConfig.Handler(
                    handler_name="GetUserHandler", handler_type="query"
                )
                super().__init__(config=config)

            def handle(self, query: GetUserQuery) -> FlextResult[User]:
                # Query execution logic
                user = database.get_user(query.user_id)
                if not user:
                    return FlextResult[User].fail("User not found")
                return FlextResult[User].ok(user)


        # Example 3: Check message compatibility
        handler = CreateUserHandler()
        can_handle = handler.can_handle(CreateUserCommand)
        print(f"Can handle: {can_handle}")

        # Example 4: Access handler metadata
        handler_id = handler.handler_id
        handler_mode = handler.handler_mode
        logger = handler.logger
        ```

    **TODO**: Enhanced handler features for 1.0.0+ releases
        - [ ] Add async/await handler support for concurrency
        - [ ] Implement handler chaining for workflows
        - [ ] Add enhanced validation with custom rules
        - [ ] Support handler composition patterns
        - [ ] Implement handler interceptors
        - [ ] Add transaction support for handlers
        - [ ] Support handler versioning
        - [ ] Implement handler timeout configuration
        - [ ] Add handler circuit breaker patterns
        - [ ] Support handler result caching

    Args:
        config: Handler configuration (CqrsConfig.Handler model).

    Attributes:
        _config_model (FlextModels.CqrsConfig.Handler): Configuration.
        _execution_context: Handler execution context state.
        _accepted_message_types (set): Compatible message types.
        _revalidate_pydantic_messages (bool): Revalidation flag.

    Returns:
        FlextHandlers: Abstract handler base for subclassing.

    Raises:
        NotImplementedError: When handle() not implemented.
        ValueError: When message validation fails.

    Note:
        Abstract base class - must implement handle() method.
        Generic types MessageT_contra and ResultT for type safety.
        Inherits from FlextMixins for reusable behaviors. All
        operations return FlextResult for railway pattern.

    Warning:
        Must call super().__init__(config=config) in subclasses.
        handle() method must be implemented by subclasses.
        Message type checking uses generic type parameters.
        Configuration required at instantiation.

    Example:
        Complete handler implementation:

        >>> class MyHandler(FlextHandlers[MyCommand, MyResult]):
        ...     def __init__(self):
        ...         config = FlextModels.CqrsConfig.Handler(
        ...             handler_name="MyHandler", handler_type="command"
        ...         )
        ...         super().__init__(config=config)
        ...
        ...     def handle(self, cmd: MyCommand) -> FlextResult[MyResult]:
        ...         return FlextResult[MyResult].ok(result)

    See Also:
        FlextBus: For handler registration and execution.
        FlextModels: For Command/Query base classes.
        FlextDispatcher: For higher-level dispatch patterns.

    """

    @override
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
        return _MessageValidator.validate_message(
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
        return _MessageValidator.validate_message(
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

    def _run_pipeline(
        self,
        message: MessageT_contra | FlextTypes.Core.Dict,
        operation: str = "command",
    ) -> FlextResult[ResultT]:
        """Run the handler pipeline with message processing.

        Args:
            message: The message to process
            operation: The operation type (command or query)

        Returns:
            FlextResult containing the processing result or error

        """
        # Extract message ID
        message_id: str = "unknown"
        if isinstance(message, dict):
            message_dict = message
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
            result = self.handle(cast("MessageT_contra", message))
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
        callable_func: FlextTypes.Core.Callable,
        handler_name: str | None = None,
        handler_type: Literal["command", "query"] = "command",
        mode: str | None = None,
        handler_config: FlextModels.CqrsConfig.Handler
        | FlextTypes.Core.Dict
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
            raise FlextExceptions.ValidationError(
                message=msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

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
                    raise FlextExceptions.ValidationError(
                        message=msg,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    ) from e
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
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                ) from e

        # Create a simple wrapper class
        class CallableHandler(FlextHandlers[object, object]):
            def __init__(self, config: FlextModels.CqrsConfig.Handler) -> None:
                super().__init__(config=config)
                self.original_callable = callable_func

            @override
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
                kwargs: FlextTypes.Core.Dict = {
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

    class HandlerPatterns:
        """Advanced handler patterns for complex domain operations."""

        @staticmethod
        def create_command_handler[TCommand, TResult](
            handler_func: FlextTypes.Core.Callable,
            command_type: str,
            validation_rules: list[Callable[[TCommand], FlextResult[None]]]
            | None = None,
        ) -> FlextHandlers[TCommand, TResult]:
            """Create a command handler with advanced validation patterns.

            Args:
                handler_func: Command handling function
                command_type: Type of command this handler processes
                validation_rules: Optional validation rules to apply

            Returns:
                FlextHandlers[TCommand, TResult]: Configured command handler

            Example:
                ```python
                handler = FlextHandlers.AdvancedPatterns.create_command_handler(
                    lambda cmd: process_create_order(cmd),
                    "CreateOrder",
                    [
                        lambda c: validate_order_data(c),
                        lambda c: validate_customer_permissions(c),
                    ],
                )
                ```

            """
            config = FlextModels.CqrsConfig.Handler(
                handler_id=f"{command_type.lower()}_handler",
                handler_name=f"{command_type}Handler",
                handler_type=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
                metadata={
                    "command_type": command_type,
                    "validation_rules": validation_rules,
                },
            )

            class CommandHandler(FlextHandlers[TCommand, TResult]):
                @override
                def handle(self, message: TCommand) -> FlextResult[TResult]:
                    # Apply validation rules if provided
                    if validation_rules:
                        for rule in validation_rules:
                            result = rule(message)
                            if result.is_failure:
                                return FlextResult[TResult].fail(
                                    f"Command validation failed: {result.error}",
                                    error_code="COMMAND_VALIDATION_FAILED",
                                    error_data={
                                        "command_type": command_type,
                                        "error": result.error,
                                    },
                                )

                    return handler_func(message)

            return CommandHandler(config=config)

        @staticmethod
        def create_query_handler[TQuery, TResult](
            handler_func: Callable[[TQuery], FlextResult[TResult]],
            query_type: str,
            *,
            caching_enabled: bool = False,
            cache_ttl: int = FlextConstants.Defaults.CACHE_TTL,
        ) -> FlextHandlers[TQuery, TResult]:
            """Create a query handler with caching patterns.

            Args:
                handler_func: Query handling function
                query_type: Type of query this handler processes
                caching_enabled: Whether to enable caching
                cache_ttl: Cache time-to-live in seconds

            Returns:
                FlextHandlers[TQuery, TResult]: Configured query handler

            Example:
                ```python
                handler = FlextHandlers.AdvancedPatterns.create_query_handler(
                    lambda q: get_order_by_id(q),
                    "GetOrderById",
                    caching_enabled=True,
                    cache_ttl=600,
                )
                ```

            """
            config = FlextModels.CqrsConfig.Handler(
                handler_id=f"{query_type.lower()}_handler",
                handler_name=f"{query_type}Handler",
                handler_type=FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
                metadata={
                    "query_type": query_type,
                    "caching_enabled": caching_enabled,
                    "cache_ttl": cache_ttl,
                },
            )

            class QueryHandler(FlextHandlers[TQuery, TResult]):
                @override
                def handle(self, message: TQuery) -> FlextResult[TResult]:
                    # Implement caching logic if enabled
                    if caching_enabled:
                        # cache_key = f"{query_type}:{hash(str(message))}"
                        # In real implementation, would check cache here
                        # cached_result = cache.get(cache_key)
                        # if cached_result:
                        #     return cached_result
                        pass

                    result = handler_func(message)

                    # Cache result if enabled and successful
                    if caching_enabled and result.is_success:
                        # In real implementation, would cache result here
                        # cache.set(cache_key, result, ttl=cache_ttl)
                        pass

                    return result

            return QueryHandler(config=config)

        @staticmethod
        def create_event_handler[TEvent](
            handler_func: Callable[[TEvent], FlextResult[None]],
            event_type: str,
            retry_policy: FlextTypes.Core.Dict | None = None,
        ) -> FlextHandlers[TEvent, None]:
            """Create an event handler with retry patterns.

            Args:
                handler_func: Event handling function
                event_type: Type of event this handler processes
                retry_policy: Optional retry policy configuration

            Returns:
                FlextHandlers[TEvent, None]: Configured event handler

            Example:
                ```python
                handler = FlextHandlers.AdvancedPatterns.create_event_handler(
                    lambda e: handle_order_created(e),
                    "OrderCreated",
                    retry_policy={
                        "max_retries": FlextConstants.Reliability.DEFAULT_MAX_RETRIES,
                        "retry_delay": FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS,
                    },
                )
                ```

            """
            config = FlextModels.CqrsConfig.Handler(
                handler_id=f"{event_type.lower()}_handler",
                handler_name=f"{event_type}Handler",
                handler_type=FlextConstants.Cqrs.EVENT_HANDLER_TYPE,
                metadata={"event_type": event_type, "retry_policy": retry_policy},
            )

            class EventHandler(FlextHandlers[TEvent, None]):
                @override
                def handle(self, message: TEvent) -> FlextResult[None]:
                    # Implement retry logic if policy provided
                    if retry_policy:
                        max_retries_val = retry_policy.get(
                            "max_retries", FlextConstants.Reliability.MAX_RETRY_ATTEMPTS
                        )
                        retry_delay_val = retry_policy.get(
                            "retry_delay",
                            FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS,
                        )
                        max_retries = (
                            int(max_retries_val)
                            if isinstance(max_retries_val, (int, str))
                            else FlextConstants.Reliability.MAX_RETRY_ATTEMPTS
                        )
                        retry_delay = (
                            float(retry_delay_val)
                            if isinstance(retry_delay_val, (int, float, str))
                            else float(
                                FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS
                            )
                        )

                        result = None
                        for attempt in range(max_retries + 1):
                            result = handler_func(message)
                            if result.is_success:
                                return result

                            if attempt < max_retries:
                                time.sleep(retry_delay)

                        return FlextResult[None].fail(
                            f"Event handling failed after {max_retries} retries: {result.error if result else 'No attempts made'}",
                            error_code="EVENT_HANDLING_FAILED",
                            error_data={
                                "event_type": event_type,
                                "max_retries": max_retries,
                            },
                        )

                    return handler_func(message)

            return EventHandler(config=config)

        @staticmethod
        def create_saga_handler[TState](
            saga_steps: list[Callable[[TState], FlextResult[TState]]],
            compensation_steps: list[Callable[[TState], FlextResult[TState]]],
            saga_type: str,
        ) -> FlextHandlers[TState, TState]:
            """Create a saga handler for distributed transactions.

            Args:
                saga_steps: List of saga step functions
                compensation_steps: List of compensation functions (in reverse order)
                saga_type: Type of saga this handler manages

            Returns:
                FlextHandlers[TState, TState]: Configured saga handler

            Example:
                ```python
                handler = FlextHandlers.AdvancedPatterns.create_saga_handler(
                    [
                        lambda state: create_order(state),
                        lambda state: reserve_inventory(state),
                        lambda state: process_payment(state),
                    ],
                    [
                        lambda state: refund_payment(state),
                        lambda state: release_inventory(state),
                        lambda state: cancel_order(state),
                    ],
                    "OrderProcessingSaga",
                )
                ```

            """
            config = FlextModels.CqrsConfig.Handler(
                handler_id=f"{saga_type.lower()}_handler",
                handler_name=f"{saga_type}Handler",
                handler_type=FlextConstants.Cqrs.SAGA_HANDLER_TYPE,
                metadata={
                    "saga_type": saga_type,
                    "saga_steps": saga_steps,
                    "compensation_steps": compensation_steps,
                },
            )

            class SagaHandler(FlextHandlers[TState, TState]):
                @override
                def handle(self, message: TState) -> FlextResult[TState]:
                    current_state = message
                    executed_steps: list[int] = []

                    # Execute saga steps
                    for i, step in enumerate(saga_steps):
                        result = step(current_state)
                        if result.is_failure:
                            # Execute compensation steps in reverse order
                            for j in range(len(executed_steps) - 1, -1, -1):
                                compensation_result = compensation_steps[j](
                                    current_state
                                )
                                if compensation_result.is_failure:
                                    return FlextResult[TState].fail(
                                        f"Saga compensation failed at step {j}: {compensation_result.error}",
                                        error_code="SAGA_COMPENSATION_FAILED",
                                        error_data={
                                            "saga_type": saga_type,
                                            "failed_step": i,
                                            "compensation_step": j,
                                        },
                                    )
                            return FlextResult[TState].fail(
                                f"Saga step {i} failed: {result.error}",
                                error_code="SAGA_STEP_FAILED",
                                error_data={"saga_type": saga_type, "failed_step": i},
                            )

                        current_state = result.unwrap()
                        executed_steps.append(i)

                    return FlextResult[TState].ok(current_state)

            return SagaHandler(config=config)


__all__: FlextTypes.Core.StringList = [
    "FlextHandlers",
]
