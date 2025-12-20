"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``FlextResult`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

TODO(docs/architecture/cqrs.md#modernization-roadmap): Phase 1 will introduce
``FlextMixins.CQRS`` utilities (MetricsTracker, ContextStack) to replace the
manual ``_metrics`` and ``_context_stack`` attributes. Deprecate
``record_metric()``, ``get_metrics()``, ``push_context()``, ``pop_context()``
once the mixin is available.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, cast

from flext_core.constants import c
from flext_core.exceptions import e
from flext_core.mixins import FlextMixins as x
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.typings import t
from flext_core.utilities import u


def _handler_type_to_literal(
    handler_type: c.Cqrs.HandlerType,
) -> c.Cqrs.HandlerTypeLiteral:
    """Convert HandlerType StrEnum to HandlerTypeLiteral.

    Business Rule: HandlerType StrEnum members are runtime-compatible with
    HandlerTypeLiteral (which is Literal[HandlerType.COMMAND, ...]). After
    validation that handler_type is one of the valid HandlerType values,
    the type checker understands the compatibility, so cast is not needed.

    Args:
        handler_type: HandlerType StrEnum member (validated to be one of
            HandlerType.COMMAND, HandlerType.QUERY, HandlerType.EVENT,
            HandlerType.OPERATION, HandlerType.SAGA).

    Returns:
        HandlerTypeLiteral compatible value.

    """
    # Runtime: HandlerType members are directly assignable to HandlerTypeLiteral
    # Use match to ensure type narrowing for both mypy and pyright
    match handler_type:
        case c.Cqrs.HandlerType.COMMAND:
            return "command"
        case c.Cqrs.HandlerType.QUERY:
            return "query"
        case c.Cqrs.HandlerType.EVENT:
            return "event"
        case c.Cqrs.HandlerType.OPERATION:
            return "operation"
        case c.Cqrs.HandlerType.SAGA:
            return "saga"
        case _:
            # Should never reach here as all HandlerType values are covered
            return "operation"


class FlextHandlers[MessageT_contra, ResultT](
    x.CQRS.MetricsTracker,
    x.CQRS.ContextStack,
    x,
    ABC,
):
    """Abstract CQRS handler with validation and railway-style execution.

    Provides the base implementation for Command Query Responsibility Segregation
    (CQRS) handlers, implementing structural typing via p.Handler[MessageT_contra]
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
        >>> from flext_core.handlers import h
        >>> from flext_core.result import r
        >>>
        >>> class UserCommand:
        ...     user_id: str
        ...     action: str
        >>>
        >>> class UserHandler(FlextHandlers[UserCommand, bool]):
        ...     def handle(self, message: UserCommand) -> r[bool]:
        ...         # Implement command handling logic
        ...         return r[bool].ok(True)
        ...
        ...     def validate(self, data: t.AcceptableMessageType) -> r[bool]:
        ...         # Custom validation logic
        ...         if not isinstance(data, UserCommand):
        ...             return r[bool].fail("Invalid message type")
        ...         return r[bool].ok(True)
    """

    # Class variables for message type expectations (configurable via inheritance)
    _expected_message_type: ClassVar[type | None] = None
    _expected_result_type: ClassVar[type | None] = None
    _config_model: m.Handler

    def __init__(
        self,
        *,
        config: m.Handler | None = None,
    ) -> None:
        """Initialize handler with configuration and context.

        Sets up the handler with optional configuration parameters.
        The config parameter accepts a m.Handler instance.

        Args:
            config: Optional handler configuration model

        """
        # Do not pass kwargs to super() - x and ABC don't accept them
        super().__init__()
        # Store config model if provided, otherwise create default
        if config is not None:
            self._config_model = config
        else:
            # Create default config if not provided
            self._config_model = m.Handler(
                handler_id=f"handler_{id(self)}",
                handler_name=self.__class__.__name__,
            )

        # Initialize execution context
        # HandlerType (StrEnum) is compatible with HandlerTypeLiteral - use directly
        handler_type = self._config_model.handler_mode
        # Validate handler_type is a valid HandlerType
        valid_handler_types = {
            c.Cqrs.HandlerType.COMMAND,
            c.Cqrs.HandlerType.QUERY,
            c.Cqrs.HandlerType.EVENT,
            c.Cqrs.HandlerType.OPERATION,
            c.Cqrs.HandlerType.SAGA,
        }
        if handler_type not in valid_handler_types:
            error_msg = f"Invalid handler mode: {handler_type}"
            raise e.ValidationError(error_msg)
        # handler_type is validated - HandlerType StrEnum values are compatible with HandlerTypeLiteral
        # After validation, we know handler_type is one of the valid HandlerType values
        # Business Rule: HandlerType StrEnum members are runtime-compatible with HandlerTypeLiteral
        # Use helper function for type-safe conversion
        handler_mode_literal = _handler_type_to_literal(handler_type)
        self._execution_context = m.Handler.ExecutionContext.create_for_handler(
            handler_name=self._config_model.handler_name,
            handler_mode=handler_mode_literal,
        )

        # Initialize handler state
        self._accepted_message_types: list[type] = []
        self._revalidate_pydantic_messages: bool = False
        self._type_warning_emitted: bool = False
        # Metrics and context stack are now provided by FlextMixins.CQRS mixins
        # MetricsTracker provides _metrics and record_metric()/get_metrics()
        # ContextStack provides _stack and push_context()/pop_context()/current_context()

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
            [t.GeneralValueType],
            t.GeneralValueType,
        ],
        handler_name: str | None = None,
        handler_type: c.Cqrs.HandlerType | None = None,
        mode: c.Cqrs.HandlerType | str | None = None,
        handler_config: m.Handler | None = None,
    ) -> FlextHandlers[t.GeneralValueType, t.GeneralValueType]:
        """Create a handler instance from a callable function.

        Factory method that wraps a callable function in a h instance,
        enabling the use of simple functions as CQRS handlers.

        Args:
            handler_callable: Callable that takes a message and returns result
            handler_name: Optional handler name (defaults to function name)
            handler_type: Optional handler type (command, query, event)
            mode: Optional handler mode (compatibility alias for handler_type)
            handler_config: Optional m.Handler configuration

        Returns:
            FlextHandlers[t.GeneralValueType, t.GeneralValueType]: Handler instance wrapping the callable

        Raises:
            e.ValidationError: If invalid mode is provided

        Example:
            >>> def my_handler(msg: str) -> r[str]:
            ...     return r[str].ok(f"processed_{msg}")
            >>> handler = FlextHandlers.create_from_callable(my_handler)
            >>> result = handler.handle("test")

        """

        # Create a concrete handler class dynamically
        class CallableHandler(
            FlextHandlers[t.GeneralValueType, t.GeneralValueType],
        ):
            """Dynamic handler created from callable."""

            _handler_fn: Callable[
                [t.GeneralValueType],
                t.GeneralValueType,
            ]

            def __init__(
                self,
                handler_fn: Callable[
                    [t.GeneralValueType],
                    t.GeneralValueType,
                ],
                config: m.Handler | None = None,
            ) -> None:
                # Call parent __init__ with config as keyword argument
                super().__init__(config=config)
                self._handler_fn = handler_fn

            def handle(self, message: object) -> r[t.GeneralValueType]:
                """Execute the wrapped callable."""
                try:
                    # Message is object, cast to t.GeneralValueType for handler function
                    message_value: t.GeneralValueType = cast(
                        "t.GeneralValueType",
                        message,
                    )
                    result = self._handler_fn(message_value)
                    # If result is already r, return it directly
                    if isinstance(result, r):
                        return result
                    # Otherwise wrap it in r
                    return r[t.GeneralValueType].ok(result)
                except Exception as exc:
                    # Wrap exception in r
                    return r[t.GeneralValueType].fail(str(exc))

        # Use handler_config if provided
        if handler_config is not None:
            return CallableHandler(handler_fn=handler_callable, config=handler_config)

        # Resolve handler type from mode or handler_type
        resolved_type: c.Cqrs.HandlerType = c.Cqrs.HandlerType.COMMAND
        if mode is not None:
            # Handle both HandlerType enum and string (HandlerType is StrEnum, so values are strings)
            if isinstance(mode, c.Cqrs.HandlerType):
                resolved_type = mode
            elif mode not in u.Enum.values(c.Cqrs.HandlerType):
                error_msg = f"Invalid handler mode: {mode}"
                raise e.ValidationError(error_msg)
            else:
                # Type narrowing: mode is valid string, HandlerType constructor accepts it
                resolved_type = c.Cqrs.HandlerType(mode)
        elif handler_type is not None:
            resolved_type = handler_type

        # Use get() for concise attribute extraction
        resolved_name: str = handler_name or str(
            u.mapper().get(handler_callable, "__name__", default="unknown_handler")
            or "unknown_handler",
        )

        # Create config
        config = m.Handler(
            handler_id=f"callable_{id(handler_callable)}",
            handler_name=resolved_name,
            handler_type=resolved_type,
            handler_mode=resolved_type,
        )

        return CallableHandler(handler_fn=handler_callable, config=config)

    @abstractmethod
    def handle(self, message: MessageT_contra) -> r[ResultT]:
        """Handle the message - abstract method to be implemented by subclasses.

        This is the core business logic method that must be implemented by all
        concrete handler subclasses. It contains the actual command/query/event
        processing logic specific to each handler implementation.

        Args:
            message: The message (command, query, or event) to handle

        Returns:
            r[ResultT]: Success with result or failure with error details

        Note:
            This method should focus on business logic only. Validation should
            be handled separately in the validate() method and executed via execute().

        """
        ...

    def execute(self, message: MessageT_contra) -> r[ResultT]:
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
            r[ResultT]: Success with handler result or failure with validation/business error

        Example:
            >>> handler = UserHandler()
            >>> result = handler.execute(UserCommand(user_id="123", action="create"))
            >>> if result.is_success:
            ...     print(f"Success: {result.value}")
            ... else:
            ...     print(f"Failed: {result.error}")

        """
        # Cast message to AcceptableMessageType for validation
        message_for_validation: t.AcceptableMessageType = cast(
            "t.AcceptableMessageType",
            message,
        )
        validation = self.validate(message_for_validation)
        if validation.is_failure:
            return r[ResultT].fail(validation.error or "Validation failed")
        return self.handle(message)

    def validate(
        self,
        data: t.AcceptableMessageType,
    ) -> r[bool]:
        """Validate input data using extensible validation pipeline.

        Base validation method that can be overridden by subclasses to implement
        custom validation logic. By default, performs basic type checking and
        returns success. Subclasses should extend this method for domain-specific
        validation rules.

        The validation follows railway-oriented programming principles, returning
        r[bool] to allow for detailed error reporting and chaining.

        Args:
            data: Input data to validate (message, command, query, or event)

        Returns:
            r[bool]: Success (True) if valid, failure with error details if invalid

        Example:
            >>> handler = UserHandler()
            >>> result = handler.validate(invalid_data)
            >>> if result.is_failure:
            ...     print(f"Validation error: {result.error}")

        Note: self is required for subclass override compatibility, even though
        this base implementation doesn't use instance state.

        """
        # Reject None values directly
        if data is None:
            return r[bool].fail("Message cannot be None")

        # Base validation - accept any AcceptableMessageType
        # Subclasses should override for specific validation rules
        return r[bool].ok(True)

    def validate_command(
        self,
        command: t.AcceptableMessageType,
    ) -> r[bool]:
        """Validate command message with command-specific rules.

        Convenience method for command validation that delegates to the base
        validate() method. Commands typically have stricter validation requirements
        than queries or events. Subclasses can override this method for command-specific
        validation logic.

        Args:
            command: Command message to validate

        Returns:
            r[bool]: Success if command is valid, failure with error details

        Note:
            By default delegates to validate(). Override for command-specific validation.

        """
        return self.validate(command)

    def validate_query(
        self,
        query: t.AcceptableMessageType,
    ) -> r[bool]:
        """Validate query message with query-specific rules.

        Convenience method for query validation that delegates to the base
        validate() method. Queries typically have different validation requirements
        than commands (e.g., read permissions vs write permissions).

        Args:
            query: Query message to validate

        Returns:
            r[bool]: Success if query is valid, failure with error details

        Note:
            By default delegates to validate(). Override for query-specific validation.

        """
        return self.validate(query)

    def validate_message(
        self,
        message: t.AcceptableMessageType,
    ) -> r[bool]:
        """Validate message using type checking and validation rules.

        Validates the message against accepted message types and custom
        validation rules. Uses duck typing for flexible message validation.

        Args:
            message: Message to validate

        Returns:
            r[bool]: Success if message is valid, failure with error details

        """
        # Check accepted message types if specified
        if self._accepted_message_types:
            message_type = type(message)
            if not any(isinstance(message, t) for t in self._accepted_message_types):
                msg = f"Message type {message_type.__name__} not in accepted types"
                return r[bool].fail(msg)

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
        return issubclass(
            message_type,
            self._expected_message_type,
        )

    @property
    def mode(self) -> c.Cqrs.HandlerType:
        """Get handler mode from configuration.

        Returns:
            c.Cqrs.HandlerType: The handler mode (command, query, event, saga)

        """
        return self._config_model.handler_mode

    # Metrics and context methods are now inherited from FlextMixins.CQRS mixins:
    # - MetricsTracker: record_metric(), get_metrics()
    # - ContextStack: push_context(), pop_context(), current_context()
    # These methods are available via multiple inheritance from the mixin classes

    @staticmethod
    def _extract_message_id(message: t.GeneralValueType) -> str | None:
        """Extract message ID from message object without type narrowing.

        Helper method to avoid type narrowing issues when checking message
        type before passing to handle().

        Args:
            message: Message object to extract ID from

        Returns:
            Message ID string or None if not available

        """
        if isinstance(message, dict):
            # Try command_id first, then message_id using extract
            cmd_id_result = u.Mapper.extract(
                message,
                "command_id",
                default=None,
                required=False,
            )
            if cmd_id_result.is_success and cmd_id_result.value:
                return str(cmd_id_result.value)
            msg_id_result = u.Mapper.extract(
                message,
                "message_id",
                default=None,
                required=False,
            )
            return (
                str(msg_id_result.value)
                if msg_id_result.is_success and msg_id_result.value
                else None
            )
        # Use get() for concise attribute extraction
        if hasattr(message, "command_id"):
            cmd_id = u.mapper().get(message, "command_id", default="") or ""
            return str(cmd_id) if cmd_id else None
        if hasattr(message, "message_id"):
            msg_id = u.mapper().get(message, "message_id", default="") or ""
            return str(msg_id) if msg_id else None
        return None

    def dispatch_message(
        self,
        message: MessageT_contra,
        operation: str = c.Dispatcher.HANDLER_MODE_COMMAND,
    ) -> r[ResultT]:
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
            r[ResultT]: Handler execution result

        """
        return self._run_pipeline(message, operation)

    def _run_pipeline(
        self,
        message: MessageT_contra,
        operation: str = c.Dispatcher.HANDLER_MODE_COMMAND,
    ) -> r[ResultT]:
        """Run the handler execution pipeline (internal).

        Internal implementation that executes the full handler pipeline including
        mode validation, can_handle check, message validation, execution,
        context tracking, and metrics recording.

        Args:
            message: The message to process
            operation: Operation type (command, query, event)

        Returns:
            r[ResultT]: Handler execution result

        """
        # Validate handler mode matches operation
        handler_mode = getattr(
            self._config_model.handler_mode, "value", self._config_model.handler_mode
        )
        valid_operations = {
            c.Dispatcher.HANDLER_MODE_COMMAND,
            c.Dispatcher.HANDLER_MODE_QUERY,
            c.Cqrs.HandlerType.EVENT.value,
        }
        if operation != handler_mode and operation in valid_operations:
            error_msg = (
                f"Handler with mode '{handler_mode}' "
                f"cannot execute {operation} pipelines"
            )
            return r[ResultT].fail(error_msg)

        # Check if handler can handle message type
        message_type = type(message)
        if not self.can_handle(message_type):
            type_name = message_type.__name__
            error_msg = f"Handler cannot handle message type {type_name}"
            return r[ResultT].fail(error_msg)

        # Cast message to AcceptableMessageType for validation
        message_for_validation: t.AcceptableMessageType = cast(
            "t.AcceptableMessageType",
            message,
        )
        # Validate message based on operation type
        if operation == c.Dispatcher.HANDLER_MODE_COMMAND:
            validation = self.validate_command(message_for_validation)
        elif operation == c.Dispatcher.HANDLER_MODE_QUERY:
            validation = self.validate_query(message_for_validation)
        else:
            validation = self.validate(message_for_validation)

        if validation.is_failure:
            error_detail = validation.error or "Validation failed"
            error_msg = f"Message validation failed: {error_detail}"
            return r[ResultT].fail(error_msg)

        # Start execution timing
        self._execution_context.start_execution()

        # Push execution context using ExecutionContext from mixin
        # Mixin push_context expects m.ExecutionContext, use existing _execution_context
        self.push_context(self._execution_context)

        try:
            # Execute handler
            result = self.handle(message)

            # Record execution metrics - extract helper method to reduce locals
            self._record_execution_metrics(success=result.is_success)

            return result
        except Exception as exc:
            # Record failure metrics
            self._record_execution_metrics(success=False, error=str(exc))
            error_msg = f"Critical handler failure: {exc}"
            return r[ResultT].fail(error_msg)
        finally:
            # Pop execution context
            _ = self.pop_context()

    def _record_execution_metrics(
        self,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record execution metrics (helper to reduce locals in _run_pipeline)."""
        exec_time_value = self._execution_context.execution_time_ms
        exec_time: float = (
            exec_time_value if isinstance(exec_time_value, float) else 0.0
        )
        # Mixin record_metric() returns None (not r[bool])
        self.record_metric(
            "execution_time_ms",
            exec_time,
        )
        self.record_metric(
            "success",
            success,
        )
        if error is not None:
            self.record_metric("error", error)

    def __call__(self, input_data: MessageT_contra) -> r[ResultT]:
        """Callable interface for seamless integration with dispatchers.

        Enables handlers to be used as callable objects, providing a clean
        interface for dispatcher systems and middleware. Internally delegates
        to the execute() method for full validation and error handling pipeline.

        Args:
            input_data: Input message to handle

        Returns:
            r[ResultT]: Handler execution result

        Example:
            >>> handler = UserHandler()
            >>> result = handler(command)  # Equivalent to handler.execute(command)

        """
        return self.execute(input_data)

    @staticmethod
    def handler(
        command: type[object],
        *,
        priority: int = c.Discovery.DEFAULT_PRIORITY,
        timeout: float | None = c.Discovery.DEFAULT_TIMEOUT,
        middleware: list[type[p.Middleware]] | None = None,
    ) -> t.DecoratorType:
        """Decorator to mark methods as handlers for commands.

        Stores handler configuration as metadata on the decorated method,
        enabling auto-discovery by FlextService and handler registries.

        Args:
            command: The command type this handler processes
            priority: Handler priority (higher = processed first). Default: 0
            timeout: Handler execution timeout in seconds. Default: None
            middleware: List of middleware types to apply to this handler

        Returns:
            Decorator function for marking handler methods

        Example:
            >>> @FlextHandlers.handler(command=CreateUserCommand, priority=10)
            ... def handle_create_user(self, cmd: CreateUserCommand) -> r[User]:
            ...     return r[User].ok(self._create(cmd))

        """

        def decorator(func: t.HandlerCallable) -> t.HandlerCallable:
            """Apply handler configuration metadata to function.

            Only sets the attribute if not already set - innermost decorator wins.
            When multiple @h.handler() decorators are stacked, the first (innermost)
            one to run takes precedence.
            """
            # Only set if not already set (innermost decorator wins)
            if not hasattr(func, c.Discovery.HANDLER_ATTR):
                config = m.Handler.DecoratorConfig(
                    command=command,
                    priority=priority,
                    timeout=timeout,
                    middleware=middleware or [],
                )
                setattr(func, c.Discovery.HANDLER_ATTR, config)
            return func

        return decorator

    class Discovery:
        """Auto-discovery mechanism for handler decorators.

        Scans classes for methods decorated with @handler() and provides
        utilities for finding and analyzing handler configurations.

        This class enables zero-config handler registration in FlextService
        by automatically discovering decorated methods at initialization time.
        """

        @staticmethod
        def scan_class(
            target_class: type,
        ) -> list[tuple[str, m.Handler.DecoratorConfig]]:
            """Scan class for methods decorated with @handler().

            Introspects the class to find all methods with handler configuration
            metadata, returning them sorted by priority (highest first).

            Args:
                target_class: Class to scan for handler decorators

            Returns:
                List of tuples (method_name, DecoratorConfig) sorted by priority

            Example:
                >>> handlers = FlextHandlers.Discovery.scan_class(MyService)
                >>> for method_name, config in handlers:
                ...     print(f"{method_name}: {config.command.__name__}")

            """
            handlers: list[tuple[str, m.Handler.DecoratorConfig]] = []
            for name in dir(target_class):
                method = getattr(target_class, name, None)
                if hasattr(method, c.Discovery.HANDLER_ATTR):
                    config: m.Handler.DecoratorConfig = getattr(
                        method,
                        c.Discovery.HANDLER_ATTR,
                    )
                    handlers.append((name, config))

            # Sort by priority (descending)
            return sorted(
                handlers,
                key=lambda x: x[1].priority,
                reverse=True,
            )

        @staticmethod
        def has_handlers(target_class: type) -> bool:
            """Check if class has any handler-decorated methods.

            Efficiently checks if a class contains any methods marked with
            the @handler() decorator without scanning all methods.

            Args:
                target_class: Class to check for handlers

            Returns:
                True if class has at least one handler, False otherwise

            Example:
                >>> if FlextHandlers.Discovery.has_handlers(MyService):
                ...     # Auto-setup dispatcher/registry
                ...     service._setup_dispatcher()

            """
            return any(
                hasattr(getattr(target_class, name, None), c.Discovery.HANDLER_ATTR)
                for name in dir(target_class)
            )

        @staticmethod
        def scan_module(
            module: object,
        ) -> list[tuple[str, t.HandlerCallable, m.Handler.DecoratorConfig]]:
            """Scan module for functions decorated with @handler().

            Introspects the module to find all functions with handler configuration
            metadata, returning them sorted by priority for consistent ordering.

            Args:
                module: Module object to scan for handler decorators

            Returns:
                List of tuples (function_name, function, DecoratorConfig) sorted by priority

            Example:
                >>> handlers = FlextHandlers.Discovery.scan_module(my_module)
                >>> for func_name, func, config in handlers:
                ...     print(f"{func_name}: {config.command.__name__}")

            """
            handlers: list[
                tuple[str, t.HandlerCallable, m.Handler.DecoratorConfig]
            ] = []
            for name in dir(module):
                if name.startswith("_"):
                    continue
                func = getattr(module, name, None)
                if callable(func) and hasattr(func, c.Discovery.HANDLER_ATTR):
                    config: m.Handler.DecoratorConfig = getattr(
                        func,
                        c.Discovery.HANDLER_ATTR,
                    )
                    # Type narrowing: func is callable, cast to HandlerCallable
                    func_typed = cast("t.HandlerCallable", func)
                    handlers.append((name, func_typed, config))

            # Sort by priority (descending), then by name for stability
            return sorted(
                handlers,
                key=lambda x: (-x[2].priority, x[0]),
            )

        @staticmethod
        def has_handlers_module(module: object) -> bool:
            """Check if module has any handler-decorated functions.

            Efficiently checks if a module contains any functions marked with
            the @handler() decorator without scanning all items.

            Args:
                module: Module object to check for handlers

            Returns:
                True if module has at least one handler, False otherwise

            Example:
                >>> if FlextHandlers.Discovery.has_handlers_module(my_module):
                ...     # Auto-register handlers from module
                ...     dispatcher.auto_register_handlers_from_module(my_module)

            """
            return any(
                hasattr(getattr(module, name, None), c.Discovery.HANDLER_ATTR)
                for name in dir(module)
                if not name.startswith("_") and callable(getattr(module, name, None))
            )


# Alias for simplified usage
h = FlextHandlers

__all__ = [
    "FlextHandlers",
    "h",
]
