"""CQRS handler foundation used by the dispatcher pipeline.

FlextHandlers defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``FlextResult`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, cast

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextHandlers[MessageT_contra, ResultT](FlextMixins, ABC):
    """Abstract CQRS handler with validation and railway-style execution.

    Provides the base implementation for Command Query Responsibility Segregation
    (CQRS) handlers, implementing structural typing via FlextProtocols.Handler[MessageT_contra]
    through duck typing (no inheritance required). This class serves as the foundation
    for implementing command, query, and event handlers with comprehensive validation,
    execution pipelines, metrics collection, and configuration management.

    Core Features:
    - Abstract base class for command/query/event handlers using generics
    - Railway-oriented programming with FlextResult for error handling
    - Message validation pipeline with extensible validation methods
    - Type checking for message compatibility using duck typing
    - Execution context management with tracing and correlation IDs
    - Callable interface for seamless integration with dispatchers
    - Configuration-driven behavior through FlextConstants

    Architecture:
    - Single class with nested type definitions and validation logic
    - DRY principle applied through shared validation methods
    - SOLID principles: Open/Closed for extensibility, Single Responsibility for focused methods
    - Railway pattern for error handling without exceptions
    - Structural typing for protocol compliance without inheritance

    Type Parameters:
    - MessageT_contra: Contravariant message type (commands, queries, events)
    - ResultT: Covariant result type returned by handler execution

    Example Usage:
        >>> from flext_core.handlers import FlextHandlers
        >>> from flext_core.result import FlextResult
        >>>
        >>> class UserCommand:
        ...     user_id: str
        ...     action: str
        >>>
        >>> class UserHandler(FlextHandlers[UserCommand, bool]):
        ...     def handle(self, message: UserCommand) -> FlextResult[bool]:
        ...         # Implement command handling logic
        ...         return FlextResult[bool].ok(True)
        ...
        ...     def validate(
        ...         self, data: FlextTypes.Handler.AcceptableMessageType
        ...     ) -> FlextResult[bool]:
        ...         # Custom validation logic
        ...         if not isinstance(data, UserCommand):
        ...             return FlextResult[bool].fail("Invalid message type")
        ...         return FlextResult[bool].ok(True)
    """

    # Class variables for message type expectations (configurable via inheritance)
    _expected_message_type: ClassVar[type | None] = None
    _expected_result_type: ClassVar[type | None] = None

    def __init__(
        self,
        *,
        config: FlextModels.Cqrs.Handler | None = None,
    ) -> None:
        """Initialize handler with configuration and context.

        Sets up the handler with optional configuration parameters.
        The config parameter accepts a FlextModels.Cqrs.Handler instance.

        Args:
            config: Optional handler configuration model

        """
        # Do not pass kwargs to super() - FlextMixins and ABC don't accept them
        super().__init__()
        # Store config model if provided, otherwise create default
        if config is not None:
            self._config_model = config
        else:
            # Create default config if not provided
            self._config_model = FlextModels.Cqrs.Handler(
                handler_id=f"handler_{id(self)}",
                handler_name=self.__class__.__name__,
            )

        # Initialize execution context
        self._execution_context = FlextModels.HandlerExecutionContext(
            handler_name=self._config_model.handler_name,
            handler_mode=self._config_model.handler_mode,
        )

        # Initialize handler state
        self._accepted_message_types: list[type] = []
        self._revalidate_pydantic_messages: bool = False
        self._type_warning_emitted: bool = False
        self._context_stack: list[dict[str, FlextTypes.GeneralValueType]] = []
        self._metrics: dict[str, FlextTypes.GeneralValueType] = {}

    @property
    def handler_name(self) -> str:
        """Get handler name from configuration.

        Returns:
            str: The handler name

        """
        return self._config_model.handler_name

    @classmethod
    def create_from_callable(
        cls,
        handler_callable: Callable[
            [FlextTypes.GeneralValueType],
            FlextTypes.GeneralValueType,
        ],
        handler_name: str | None = None,
        handler_type: FlextConstants.Cqrs.HandlerType | None = None,
        mode: FlextConstants.Cqrs.HandlerType | str | None = None,
        handler_config: FlextModels.Cqrs.Handler | None = None,
    ) -> FlextHandlers[object, object]:
        """Create a handler instance from a callable function.

        Factory method that wraps a callable function in a FlextHandlers instance,
        enabling the use of simple functions as CQRS handlers.

        Args:
            handler_callable: Callable that takes a message and returns result
            handler_name: Optional handler name (defaults to function name)
            handler_type: Optional handler type (command, query, event)
            mode: Optional handler mode (compatibility alias for handler_type)
            handler_config: Optional FlextModels.Cqrs.Handler configuration

        Returns:
            FlextHandlers[object, object]: Handler instance wrapping the callable

        Raises:
            FlextExceptions.ValidationError: If invalid mode is provided

        Example:
            >>> def my_handler(msg: str) -> FlextResult[str]:
            ...     return FlextResult[str].ok(f"processed_{msg}")
            >>> handler = FlextHandlers.create_from_callable(my_handler)
            >>> result = handler.handle("test")

        """

        # Create a concrete handler class dynamically
        class CallableHandler(FlextHandlers[object, object]):
            """Dynamic handler created from callable."""

            _handler_fn: Callable[
                [FlextTypes.GeneralValueType],
                FlextTypes.GeneralValueType,
            ]

            def __init__(
                self,
                handler_fn: Callable[
                    [FlextTypes.GeneralValueType],
                    FlextTypes.GeneralValueType,
                ],
                config: FlextModels.Cqrs.Handler | None = None,
            ) -> None:
                # Call parent __init__ with config as keyword argument
                super().__init__(config=config)
                self._handler_fn = handler_fn

            def handle(
                self, message: FlextTypes.GeneralValueType
            ) -> FlextResult[FlextTypes.GeneralValueType]:
                """Execute the wrapped callable."""
                try:
                    # Cast message to FlextTypes.GeneralValueType for handler function
                    message_value: FlextTypes.GeneralValueType = cast(
                        "FlextTypes.GeneralValueType",
                        message,
                    )
                    result = self._handler_fn(message_value)
                    # If result is already FlextResult, return it
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    # Otherwise wrap it in FlextResult
                    return cast(
                        "FlextResult[object]",
                        FlextResult[FlextTypes.GeneralValueType].ok(result),
                    )
                except Exception as exc:
                    return cast(
                        "FlextResult[object]",
                        FlextResult[FlextTypes.GeneralValueType].fail(str(exc)),
                    )

        # Use handler_config if provided
        if handler_config is not None:
            return CallableHandler(handler_fn=handler_callable, config=handler_config)

        # Resolve handler type from mode or handler_type
        resolved_type = FlextConstants.Cqrs.HandlerType.COMMAND
        if mode is not None:
            if isinstance(mode, str):
                # Validate string mode using pre-defined frozenset
                if mode not in FlextConstants.Cqrs.VALID_HANDLER_MODES:
                    error_msg = f"Invalid handler mode: {mode}"
                    raise FlextExceptions.ValidationError(error_msg)
                resolved_type = FlextConstants.Cqrs.HandlerType(mode)
            else:
                resolved_type = mode
        elif handler_type is not None:
            resolved_type = handler_type

        # Get handler name from function if not provided
        resolved_name: str = handler_name or str(
            getattr(handler_callable, "__name__", "unknown_handler"),
        )

        # Create config
        config = FlextModels.Cqrs.Handler(
            handler_id=f"callable_{id(handler_callable)}",
            handler_name=resolved_name,
            handler_type=resolved_type,
            handler_mode=resolved_type,
        )

        return CallableHandler(handler_fn=handler_callable, config=config)

    @abstractmethod
    def handle(self, message: MessageT_contra) -> FlextResult[ResultT]:
        """Handle the message - abstract method to be implemented by subclasses.

        This is the core business logic method that must be implemented by all
        concrete handler subclasses. It contains the actual command/query/event
        processing logic specific to each handler implementation.

        Args:
            message: The message (command, query, or event) to handle

        Returns:
            FlextResult[ResultT]: Success with result or failure with error details

        Note:
            This method should focus on business logic only. Validation should
            be handled separately in the validate() method and executed via execute().

        """
        ...

    def execute(self, message: MessageT_contra) -> FlextResult[ResultT]:
        """Execute handler with complete validation and error handling pipeline.

        Implements the railway-oriented programming pattern by first validating
        the input message, then executing the business logic if validation passes.
        Uses FlextResult for consistent error handling without exceptions.

        Execution Pipeline:
        1. Validate input message using validate() method
        2. If validation fails, return failure result with error details
        3. If validation passes, execute handle() method with business logic
        4. Return result from handle() method (success or failure)

        Args:
            message: The message to execute handler for

        Returns:
            FlextResult[ResultT]: Success with handler result or failure with validation/business error

        Example:
            >>> handler = UserHandler()
            >>> result = handler.execute(UserCommand(user_id="123", action="create"))
            >>> if result.is_success:
            ...     print(f"Success: {result.value}")
            ... else:
            ...     print(f"Failed: {result.error}")

        """
        # Cast message to AcceptableMessageType for validation
        message_for_validation: FlextTypes.Handler.AcceptableMessageType = cast(
            "FlextTypes.Handler.AcceptableMessageType",
            message,
        )
        validation = self.validate(message_for_validation)
        if validation.is_failure:
            return FlextResult[ResultT].fail(validation.error or "Validation failed")
        return self.handle(message)

    def validate(  # noqa: PLR6301
        self,
        data: FlextTypes.Handler.AcceptableMessageType,
    ) -> FlextResult[bool]:
        """Validate input data using extensible validation pipeline.

        Base validation method that can be overridden by subclasses to implement
        custom validation logic. By default, performs basic type checking and
        returns success. Subclasses should extend this method for domain-specific
        validation rules.

        The validation follows railway-oriented programming principles, returning
        FlextResult[bool] to allow for detailed error reporting and chaining.

        Args:
            data: Input data to validate (message, command, query, or event)

        Returns:
            FlextResult[bool]: Success (True) if valid, failure with error details if invalid

        Example:
            >>> handler = UserHandler()
            >>> result = handler.validate(invalid_data)
            >>> if result.is_failure:
            ...     print(f"Validation error: {result.error}")

        Note: self is required for subclass override compatibility, even though
        this base implementation doesn't use instance state.

        """
        # Reject None values
        if data is None:
            return FlextResult[bool].fail("Message cannot be None")

        # Base validation - accept any AcceptableMessageType
        # Subclasses should override for specific validation rules
        return FlextResult[bool].ok(True)

    def validate_command(
        self,
        command: FlextTypes.Handler.AcceptableMessageType,
    ) -> FlextResult[bool]:
        """Validate command message with command-specific rules.

        Convenience method for command validation that delegates to the base
        validate() method. Commands typically have stricter validation requirements
        than queries or events. Subclasses can override this method for command-specific
        validation logic.

        Args:
            command: Command message to validate

        Returns:
            FlextResult[bool]: Success if command is valid, failure with error details

        Note:
            By default delegates to validate(). Override for command-specific validation.

        """
        return self.validate(command)

    def validate_query(
        self,
        query: FlextTypes.Handler.AcceptableMessageType,
    ) -> FlextResult[bool]:
        """Validate query message with query-specific rules.

        Convenience method for query validation that delegates to the base
        validate() method. Queries typically have different validation requirements
        than commands (e.g., read permissions vs write permissions).

        Args:
            query: Query message to validate

        Returns:
            FlextResult[bool]: Success if query is valid, failure with error details

        Note:
            By default delegates to validate(). Override for query-specific validation.

        """
        return self.validate(query)

    def validate_message(
        self,
        message: FlextTypes.Handler.AcceptableMessageType,
    ) -> FlextResult[bool]:
        """Validate message using type checking and validation rules.

        Validates the message against accepted message types and custom
        validation rules. Uses duck typing for flexible message validation.

        Args:
            message: Message to validate

        Returns:
            FlextResult[bool]: Success if message is valid, failure with error details

        """
        # Check accepted message types if specified
        if self._accepted_message_types:
            message_type = type(message)
            if not any(isinstance(message, t) for t in self._accepted_message_types):
                msg = f"Message type {message_type.__name__} not in accepted types"
                return FlextResult[bool].fail(msg)

        # Delegate to base validation
        return self.validate(message)

    def can_handle(self, message_type: type[object]) -> bool:
        """Check if handler can handle the specified message type.

        Determines message type compatibility using duck typing and class hierarchy.
        If _expected_message_type is set, checks if the message_type is a subclass
        of the expected type. If not set, accepts any message type (flexible handler).

        This method enables handler registration and routing in dispatcher systems,
        allowing handlers to declare their capabilities through configuration.

        Args:
            message_type: The message type to check compatibility for

        Returns:
            bool: True if handler can handle this message type, False otherwise

        Example:
            >>> class UserCommand:
            ...     pass
            >>> class AdminCommand:
            ...     pass
            >>> handler = UserHandler()
            >>> handler.can_handle(UserCommand)  # True
            >>> handler.can_handle(AdminCommand)  # Depends on _expected_message_type

        """
        if self._expected_message_type is None:
            # Flexible handler - accepts any message type
            return True

        # Strict handler - check type compatibility
        return isinstance(message_type, type) and issubclass(
            message_type,
            self._expected_message_type,
        )

    @property
    def mode(self) -> FlextConstants.Cqrs.HandlerType:
        """Get handler mode from configuration.

        Returns:
            FlextConstants.Cqrs.HandlerType: The handler mode (command, query, event, saga)

        """
        return self._config_model.handler_mode

    def push_context(
        self,
        context: dict[str, FlextTypes.GeneralValueType],
    ) -> FlextResult[bool]:
        """Push execution context onto the stack.

        Args:
            context: Context dictionary to push onto the stack

        Returns:
            FlextResult[bool]: Success if context was pushed

        """
        self._context_stack.append(context)
        return FlextResult[bool].ok(True)

    def pop_context(self) -> FlextResult[dict[str, FlextTypes.GeneralValueType]]:
        """Pop execution context from the stack.

        Returns:
            FlextResult[dict[str, FlextTypes.GeneralValueType]]: Success with popped context or empty dict

        """
        if self._context_stack:
            return FlextResult[dict[str, FlextTypes.GeneralValueType]].ok(
                self._context_stack.pop(),
            )
        return FlextResult[dict[str, FlextTypes.GeneralValueType]].ok({})

    def get_metrics(self) -> FlextResult[dict[str, FlextTypes.GeneralValueType]]:
        """Get current metrics dictionary.

        Returns:
            FlextResult[dict[str, FlextTypes.GeneralValueType]]: Success with metrics collection

        """
        return FlextResult[dict[str, FlextTypes.GeneralValueType]].ok(
            self._metrics.copy(),
        )

    def record_metric(
        self,
        name: str,
        value: FlextTypes.GeneralValueType,
    ) -> FlextResult[bool]:
        """Record a metric value.

        Args:
            name: Metric name
            value: Metric value to record

        Returns:
            FlextResult[bool]: Success if metric was recorded

        """
        self._metrics[name] = value
        return FlextResult[bool].ok(True)

    @staticmethod
    def _extract_message_id(message: FlextTypes.GeneralValueType) -> str | None:
        """Extract message ID from message object without type narrowing.

        Helper method to avoid type narrowing issues when checking message
        type before passing to handle().

        Args:
            message: Message object to extract ID from

        Returns:
            Message ID string or None if not available

        """
        if isinstance(message, dict):
            return (
                str(message.get("command_id") or message.get("message_id") or "")
                or None
            )
        if hasattr(message, "command_id"):
            return str(getattr(message, "command_id", "")) or None
        if hasattr(message, "message_id"):
            return str(getattr(message, "message_id", "")) or None
        return None

    def dispatch_message(
        self,
        message: MessageT_contra,
        operation: str = FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
    ) -> FlextResult[ResultT]:
        """Dispatch message through the handler execution pipeline.

        Public method that executes the full handler pipeline including
        mode validation, can_handle check, message validation, execution,
        context tracking, and metrics recording.

        This method is the primary entry point for external systems (like
        FlextDispatcher) to execute handlers with full CQRS support.

        Args:
            message: The message to process
            operation: Operation type (command, query, event)

        Returns:
            FlextResult[ResultT]: Handler execution result

        """
        return self._run_pipeline(message, operation)

    def _run_pipeline(
        self,
        message: MessageT_contra,
        operation: str = FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
    ) -> FlextResult[ResultT]:
        """Run the handler execution pipeline (internal).

        Internal implementation that executes the full handler pipeline including
        mode validation, can_handle check, message validation, execution,
        context tracking, and metrics recording.

        Args:
            message: The message to process
            operation: Operation type (command, query, event)

        Returns:
            FlextResult[ResultT]: Handler execution result

        """
        # Validate handler mode matches operation
        handler_mode = self._config_model.handler_mode.value
        valid_operations = {
            FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
            FlextConstants.Dispatcher.HANDLER_MODE_QUERY,
            FlextConstants.Cqrs.HandlerType.EVENT.value,
        }
        if operation != handler_mode and operation in valid_operations:
            error_msg = (
                f"Handler with mode '{handler_mode}' "
                f"cannot execute {operation} pipelines"
            )
            return FlextResult[ResultT].fail(error_msg)

        # Check if handler can handle message type
        message_type = type(message)
        if not self.can_handle(message_type):
            type_name = message_type.__name__
            error_msg = f"Handler cannot handle message type {type_name}"
            return FlextResult[ResultT].fail(error_msg)

        # Cast message to AcceptableMessageType for validation
        message_for_validation: FlextTypes.Handler.AcceptableMessageType = cast(
            "FlextTypes.Handler.AcceptableMessageType",
            message,
        )
        # Validate message based on operation type
        if operation == FlextConstants.Dispatcher.HANDLER_MODE_COMMAND:
            validation = self.validate_command(message_for_validation)
        elif operation == FlextConstants.Dispatcher.HANDLER_MODE_QUERY:
            validation = self.validate_query(message_for_validation)
        else:
            validation = self.validate(message_for_validation)

        if validation.is_failure:
            error_detail = validation.error or "Validation failed"
            error_msg = f"Message validation failed: {error_detail}"
            return FlextResult[ResultT].fail(error_msg)

        # Start execution timing
        self._execution_context.start_execution()

        # Extract message ID if available using helper to avoid type narrowing
        message_for_extraction: FlextTypes.GeneralValueType = cast(
            "FlextTypes.GeneralValueType",
            message,
        )
        message_id: str | None = FlextHandlers._extract_message_id(
            message_for_extraction
        )

        # Push execution context
        self.push_context({
            "operation": operation,
            "message_id": message_id,
            "handler_name": self._config_model.handler_name,
        })

        try:
            # Execute handler
            result = self.handle(message)

            # Record execution metrics (execution_time_ms is a property)
            # Access property value directly
            exec_time_value = self._execution_context.execution_time_ms
            exec_time: float = (
                exec_time_value if isinstance(exec_time_value, float) else 0.0
            )
            self.record_metric(
                "execution_time_ms",
                cast("FlextTypes.GeneralValueType", exec_time),
            )
            self.record_metric(
                "success",
                cast("FlextTypes.GeneralValueType", result.is_success),
            )

            return result
        except Exception as exc:
            # Record failure metrics
            # Access property value directly
            exec_time_value_exc = self._execution_context.execution_time_ms
            exec_time_exc: float = (
                exec_time_value_exc if isinstance(exec_time_value_exc, float) else 0.0
            )
            self.record_metric(
                "execution_time_ms",
                cast("FlextTypes.GeneralValueType", exec_time_exc),
            )
            self.record_metric("success", cast("FlextTypes.GeneralValueType", False))
            self.record_metric("error", cast("FlextTypes.GeneralValueType", str(exc)))
            error_msg = f"Critical handler failure: {exc}"
            return FlextResult[ResultT].fail(error_msg)
        finally:
            # Pop execution context
            self.pop_context()

    def __call__(self, input_data: MessageT_contra) -> FlextResult[ResultT]:
        """Callable interface for seamless integration with dispatchers.

        Enables handlers to be used as callable objects, providing a clean
        interface for dispatcher systems and middleware. Internally delegates
        to the execute() method for full validation and error handling pipeline.

        Args:
            input_data: Input message to handle

        Returns:
            FlextResult[ResultT]: Handler execution result

        Example:
            >>> handler = UserHandler()
            >>> result = handler(command)  # Equivalent to handler.execute(command)

        """
        return self.execute(input_data)


__all__ = ["FlextHandlers"]
