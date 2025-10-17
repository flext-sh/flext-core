"""Base classes for CQRS command and query handlers.

This module provides FlextHandlers, the base class for implementing
Command Query Responsibility Segregation (CQRS) handlers throughout
the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import (
    cast,
    override,
)

from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import (
    CallableInputT,
    CallableOutputT,
    FlextTypes,
    MessageT_contra,
    ResultT,
)
from flext_core.utilities import FlextUtilities

# Module-level constant for default handler type (avoids B008 lint error with cast in defaults)
_DEFAULT_HANDLER_TYPE: FlextConstants.Cqrs.HandlerType = cast(
    "FlextConstants.Cqrs.HandlerType",
    FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
)


class FlextHandlers[MessageT_contra, ResultT](FlextMixins, ABC):
    """Base class for CQRS command and query handlers.

    Implements FlextProtocols.Handler through structural typing. All handler
    subclasses automatically satisfy the Handler protocol by implementing the
    required methods: handle(), validate(), and __call__().

    Provides the foundation for implementing CQRS handlers with validation,
    execution context, metrics collection, and configuration management.
    Supports commands, queries, events, and sagas with type safety.

    Protocol Compliance:
        - execute(message: T) -> FlextResult[R] - Execute handler with message
        - handle(message: T) -> FlextResult[R] - Abstract method for implementation
        - __call__(input_data: T) -> FlextResult[R] - Callable interface
        - validate(data: FlextTypes.AcceptableMessageType) -> FlextResult[None] - Validate input data
        - validate_command(command: FlextTypes.AcceptableMessageType) -> FlextResult[None] - Validate command
        - validate_query(query: FlextTypes.AcceptableMessageType) -> FlextResult[None] - Validate query
        - can_handle(message_type: type[T]) -> bool - Check handler capability

    Features:
    - Abstract base for command/query/event handlers
    - Handler execution with validation pipeline
    - Type checking for message compatibility
    - Metrics collection for handler performance
    - Configuration via handler models
    - Execution context tracking per handler
    - Message validation with FlextResult
    - Logger integration for handler operations
    - Automatic protocol satisfaction via structural typing

    Usage:
        >>> from flext_core import FlextHandlers, FlextResult
        >>> from flext_core.protocols import FlextProtocols
        >>>
        >>> class CreateUserHandler(FlextHandlers[CreateUserCommand, User]):
        ...     def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        ...         return FlextResult[User].ok(User(name=command.name))
        >>>
        >>> handler = CreateUserHandler(config=...)
        >>> # FlextHandlers instances satisfy FlextProtocols.Handler protocol
        >>> assert isinstance(handler, FlextProtocols.Handler)
    """

    # Class-level logger for internal operations (not for subclass use)
    _internal_logger: FlextLogger = FlextLogger(__name__)

    class _MessageValidator:
        """Private message validation utilities for FlextHandlers."""

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
                        FlextHandlers._internal_logger.debug(
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
                        correlation_id=f"pydantic_validation_{int(time.time() * 1000)}",
                        metadata={
                            "validation_details": f"pydantic_exception: {e!s}, model_class: {message.__class__.__name__}, revalidated: True",
                            "context": f"operation: {operation}, message_type: {type(message).__name__}, validation_type: pydantic_revalidation",
                        },
                    )
                    return FlextResult[None].fail(
                        str(validation_error),
                        error_code=validation_error.error_code,
                        error_data={"exception_context": str(validation_error)},
                    )

            # For non-Pydantic objects, ensure serializable representation can be constructed
            try:
                cls._build_serializable_message_payload(message, operation=operation)
            except Exception as exc:
                if isinstance(exc, FlextExceptions.TypeError):
                    return FlextResult[None].fail(
                        str(exc),
                        error_code=exc.error_code,
                        error_data={
                            "exception_context": getattr(exc, "context", str(exc))
                        },
                    )

                fallback_error = FlextExceptions.TypeError(
                    f"Invalid message type for {operation}: {type(message).__name__}",
                    expected_type=cls._SERIALIZABLE_MESSAGE_EXPECTATION,
                    actual_type=type(message).__name__,
                    context=f"operation: {operation}, message_type: {type(message).__name__}, validation_type: serializable_check, original_exception: {exc!s}",
                    correlation_id=f"type_validation_{int(time.time() * 1000)}",
                )
                return FlextResult[None].fail(
                    str(fallback_error),
                    error_code=fallback_error.error_code,
                    error_data={"exception_context": str(fallback_error)},
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
                return cast("FlextTypes.Dict | str | int | float | bool", message)

            if message is None:
                msg = f"Invalid message type for {operation_name}: NoneType"
                raise FlextExceptions.TypeError(
                    msg,
                    expected_type=cls._SERIALIZABLE_MESSAGE_EXPECTATION,
                    actual_type="NoneType",
                    context=f"operation: {context_operation}, message_type: NoneType, validation_type: serializable_check",
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
                result: FlextTypes.Dict = {}
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
                            return cast("FlextTypes.Dict", result_data)
                    except Exception as e:
                        FlextHandlers._internal_logger.debug(
                            f"Serialization method {method_name} failed: {type(e).__name__}"
                        )
                        continue

            # Handle __slots__
            slots = getattr(message, "__slots__", None)
            if slots:
                if isinstance(slots, str):
                    slot_names: tuple[str, ...] = (slots,)
                elif isinstance(slots, (list, tuple)):
                    slot_names = tuple(
                        cast("FlextTypes.StringList | tuple[str, ...]", slots)
                    )
                else:
                    msg = f"Invalid __slots__ type for {operation_name}: {type(slots).__name__}"
                    raise FlextExceptions.TypeError(
                        msg,
                        expected_type="str, list, or tuple",
                        actual_type=type(slots).__name__,
                        context=f"operation: {context_operation}, message_type: {type(message).__name__}, validation_type: serializable_check, __slots__: {slots!r}",
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
                context=f"operation: {context_operation}, message_type: {type(message).__name__}, validation_type: serializable_check",
                correlation_id=f"message_serialization_{int(time.time() * 1000)}",
            )

    @override
    def __init__(self, *, config: FlextModels.Cqrs.Handler) -> None:
        """Initialize handler with simplified single-config approach.

        Args:
            config: Handler configuration object (required)

        """
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service(f"flext_handler_{config.handler_name}")

        # Enrich context with handler metadata (Phase 1 enhancement)
        # This automatically adds handler information to all logs
        self._enrich_context(
            handler_name=config.handler_name,
            handler_type=config.handler_type,
            handler_class=self.__class__.__name__,
        )

        self._config_model: FlextModels.Cqrs.Handler = config
        self._execution_context = FlextModels.HandlerExecutionContext(
            handler_name=config.handler_name,
            handler_mode=config.handler_mode,
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

    def can_handle(self, message_type: type[object]) -> bool:
        """Check if this handler can handle the given message type.

        Args:
            message_type: The type of message to check (type-safe parameter)

        Returns:
            bool: True if this handler can handle the message type

        """
        return FlextUtilities.TypeChecker.can_handle_message_type(
            self._accepted_message_types, message_type
        )

    # NOTE: logger property inherited from FlextMixins
    # Provides per-class logger instance via lazy initialization

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
        # handler_mode is always defined as FlextConstants.HandlerMode.TypeSimple
        return self._config_model.handler_mode

    @property
    def handler_config(self) -> FlextModels.Cqrs.Handler:
        """Get handler configuration.

        Returns:
            FlextModels.Cqrs.Handler: Handler configuration

        """
        return self._config_model

    def validate_command(
        self, command: FlextTypes.AcceptableMessageType
    ) -> FlextResult[None]:
        """Validate a command message with type-safe parameter.

        Args:
            command: The command to validate (typed parameter)

        Returns:
            FlextResult indicating validation success or failure

        """
        return FlextHandlers._MessageValidator.validate_message(
            command,
            operation=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            revalidate_pydantic_messages=self._revalidate_pydantic_messages,
        )

    def validate_query(
        self, query: FlextTypes.AcceptableMessageType
    ) -> FlextResult[None]:
        """Validate a query message with type-safe parameter.

        Args:
            query: The query to validate (typed parameter)

        Returns:
            FlextResult indicating validation success or failure

        """
        return FlextHandlers._MessageValidator.validate_message(
            query,
            operation=FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
            revalidate_pydantic_messages=self._revalidate_pydantic_messages,
        )

    def validate(self, _data: MessageT_contra) -> FlextResult[None]:
        """Validate input data based on handler mode with generic type support.

        Generic validation that delegates to mode-specific validation methods.
        Part of FlextProtocols.Handler protocol implementation.
        Type parameter MessageT_contra ensures type consistency.

        Args:
            _data: The data to validate (generic type MessageT_contra)

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        Examples:
            >>> handler = MyCommandHandler()
            >>> result = handler.validate(command_data)
            >>> if result.is_success:
            ...     # Validation passed
            ...     pass

        """
        if self.mode == FlextConstants.Cqrs.COMMAND_HANDLER_TYPE:
            return self.validate_command(_data)
        if self.mode == FlextConstants.Cqrs.QUERY_HANDLER_TYPE:
            return self.validate_query(_data)
        # For event and saga handlers, use generic validation
        return FlextHandlers._MessageValidator.validate_message(
            cast("object", _data),
            operation=self.mode,
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

    def __call__(self, input_data: MessageT_contra) -> FlextResult[ResultT]:
        """Callable interface for Application.Handler protocol.

        Makes the handler callable as a function, delegating to execute() method
        for consistent handler invocation. Part of FlextProtocols.Handler
        protocol implementation.

        Args:
            input_data: The input message to process

        Returns:
            FlextResult[ResultT]: Execution result

        Examples:
            >>> handler = MyCommandHandler()
            >>> result = handler(command)  # Callable interface
            >>> # Equivalent to: handler.execute(command)

        """
        return self.execute(input_data)

    def _run_pipeline(
        self,
        message: MessageT_contra | FlextTypes.Dict,
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
            message_id = (
                str(message.get(f"{operation}_id", "unknown"))
                or str(message.get("message_id", "unknown"))
                or "unknown"
            )
        elif hasattr(message, f"{operation}_id"):
            message_id = str(getattr(message, f"{operation}_id", "unknown"))
        elif hasattr(message, "message_id"):
            message_id = str(getattr(message, "message_id", "unknown"))

        message_type: str = str(type(message).__name__)

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
        message_type_obj: type[object] = type(message)
        if not self.can_handle(cast("type[MessageT_contra]", message_type_obj)):
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
        callable_func: FlextTypes.HandlerCallableType,
        handler_name: str | None = None,
        handler_type: FlextConstants.Cqrs.HandlerType = _DEFAULT_HANDLER_TYPE,
        mode: str | None = None,
        handler_config: FlextModels.Cqrs.Handler | None = None,
    ) -> FlextHandlers[object, object]:
        """Create a handler from a callable function.

        Args:
            callable_func: The callable function to wrap
            handler_name: Name for the handler (defaults to function name)
            handler_type: Type of handler (command, query, etc.)
            mode: Handler mode (for compatibility)
            handler_config: Optional handler configuration model (must be FlextModels.Cqrs.Handler)

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
        effective_type: FlextConstants.Cqrs.HandlerType = (
            cast("FlextConstants.Cqrs.HandlerType", mode)
            if mode is not None
            else handler_type
        )

        # Validate mode/handler_type
        if effective_type not in {
            FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
        }:
            msg = f"Invalid handler mode: {effective_type}. Must be '{FlextConstants.Cqrs.COMMAND_HANDLER_TYPE}' or '{FlextConstants.Cqrs.QUERY_HANDLER_TYPE}'"
            raise FlextExceptions.ValidationError(
                message=msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Use provided config or create default
        if handler_config is not None:
            config = handler_config
        else:
            try:
                config = FlextModels.Cqrs.Handler(
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

        # Create a wrapper class with proper generic types
        class CallableHandler(FlextHandlers[CallableInputT, CallableOutputT]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)
                # Store callable as object - type variables are method-local, not class-scoped
                self.original_callable: object = callable_func

            @override
            def handle(self, message: CallableInputT) -> FlextResult[CallableOutputT]:
                try:
                    result = callable_func(message)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[CallableOutputT]", result)
                    return FlextResult[CallableOutputT].ok(
                        cast("CallableOutputT", result)
                    )
                except Exception as e:
                    return FlextResult[CallableOutputT].fail(str(e))

            @override
            def can_handle(self, message_type: type[object]) -> bool:
                """Override can_handle for callable wrappers to enable auto-discovery.

                Callable wrappers created via from_callable may have method-local TypeVars
                that can't be introspected by compute_accepted_message_types(), resulting
                in empty _accepted_message_types. Instead, we rely on validation to catch
                type incompatibility, allowing auto-discovery to find this handler.
                """
                # Always return True for auto-discovery - validation will fail if incompatible
                return True

        return CallableHandler(config=config)

    class Metrics:
        """Metrics logging utilities for handlers."""

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
                kwargs: FlextTypes.Dict = {
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


__all__: FlextTypes.StringList = [
    "FlextHandlers",
]
