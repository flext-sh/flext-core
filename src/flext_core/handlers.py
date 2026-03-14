"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

# CQRS utilities: FlextMixins.CQRS provides MetricsTracker and ContextStack for
# optional use in subclasses. FlextHandlers implements metrics and context tracking
# directly as the canonical pattern for handler base class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import ClassVar, Unpack, override

from pydantic import BaseModel, ConfigDict

from flext_core import c, e, m, p, r, t, u, x
from flext_core._models.containers import FlextModelsContainers


class FlextHandlers[MessageT_contra, ResultT](x):
    """Abstract CQRS handler with validation and railway-style execution.

    Provides the base implementation for Command Query Responsibility Segregation
    (CQRS) handlers, implementing structural typing via p.Handler[MessageT_contra]
    through duck typing (no inheritance required). This class serves as the foundation
    for implementing command, query, and event handlers with comprehensive validation,
    execution pipelines, metrics collection, and configuration management.

    Core Features:
    - Abstract base class for command/query/event handlers using generics
    - Railway-oriented programming with r for error handling
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
        >>> from flext_core import h
        >>> from flext_core import r
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
        ...     def validate(
        ...         self, data: t.Scalar | BaseModel | Sequence[t.Scalar]
        ...     ) -> r[bool]:
        ...         # Custom validation logic
        ...         if not isinstance(data, UserCommand):
        ...             return r[bool].fail("Invalid message type")
        ...         return r[bool].ok(True)
    """

    _expected_message_type: ClassVar[type | None] = None
    _expected_result_type: ClassVar[type | None] = None
    _HANDLER_TYPE_LITERALS: ClassVar[
        Mapping[c.Cqrs.HandlerType, c.Cqrs.HandlerTypeLiteral]
    ] = {
        c.Cqrs.HandlerType.COMMAND: "command",
        c.Cqrs.HandlerType.QUERY: "query",
        c.Cqrs.HandlerType.EVENT: "event",
        c.Cqrs.HandlerType.OPERATION: "operation",
        c.Cqrs.HandlerType.SAGA: "saga",
    }

    def __init__(self, *, config: m.Handler | None = None) -> None:
        """Initialize handler with configuration and context.

        Sets up the handler with optional configuration parameters.
        The config parameter accepts a m instance.

        Args:
            config: Optional handler configuration model

        """
        super().__init__()
        if config is not None:
            self._config_model = config
        else:
            self._config_model = m.Handler(
                handler_id=f"handler_{id(self)}", handler_name=self.__class__.__name__
            )
        handler_type = self._config_model.handler_mode
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
        handler_mode_literal = self._handler_type_to_literal(handler_type)
        self._execution_context = m.ExecutionContext.create_for_handler(
            handler_name=self._config_model.handler_name,
            handler_mode=handler_mode_literal,
        )
        self._accepted_message_types: list[type] = []
        self._revalidate_pydantic_messages: bool = False
        self._type_warning_emitted: bool = False
        self._metrics: dict[str, t.MetadataAttributeValue] = {}
        self._stack: list[m.ExecutionContext | FlextModelsContainers.ConfigMap] = []

    def __call__(self, message: MessageT_contra) -> r[ResultT]:
        """Callable interface for seamless dispatcher integration."""
        return self.handle(message)

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Validate non-abstract subclasses implement a handle() method.

        Chains with FlextMixins.__init_subclass__ via super() to preserve
        MRO-based container auto-initialization. Skips validation for
        abstract subclasses (intermediate bases).

        Raises:
            TypeError: If a concrete subclass does not override handle().

        """
        super().__init_subclass__(**kwargs)
        if "[" in cls.__qualname__:
            return
        abstract_methods_default: frozenset[str] = frozenset()
        abstract_methods = getattr(cls, "__abstractmethods__", abstract_methods_default)
        if abstract_methods:
            return
        for klass in cls.mro():
            if klass is FlextHandlers:
                msg = f"{cls.__qualname__} must implement a handle() method"
                raise TypeError(msg)
            if "handle" in klass.__dict__:
                break

    @property
    def handler_name(self) -> str:
        """Get handler name from configuration.

        Returns:
            str: The handler name

        """
        return self._config_model.handler_name

    @property
    def mode(self) -> c.Cqrs.HandlerType:
        """Get handler mode from configuration.

        Returns:
            c.Cqrs.HandlerType: The handler mode (command, query, event, saga)

        """
        return self._config_model.handler_mode

    @classmethod
    def create_from_callable(
        cls,
        handler_callable: Callable[[t.Scalar], t.Scalar],
        handler_name: str | None = None,
        handler_type: c.Cqrs.HandlerType | None = None,
        mode: c.Cqrs.HandlerType | str | None = None,
        handler_config: m.Handler | None = None,
    ) -> FlextHandlers[t.Scalar, t.Scalar]:
        """Create a handler instance from a callable function.

        Factory method that wraps a callable function in a h instance,
        enabling the use of simple functions as CQRS handlers.

        Args:
            handler_callable: Callable that takes a message and returns result
            handler_name: Optional handler name (defaults to function name)
            handler_type: Optional handler type (command, query, event)
            mode: Optional handler mode (compatibility alias for handler_type)
            handler_config: Optional m configuration

        Returns:
            FlextHandlers[t.Scalar, t.Scalar]: Handler instance wrapping the callable

        Raises:
            e.ValidationError: If invalid mode is provided

        Example:
            >>> def my_handler(msg: str) -> r[str]:
            ...     return r[str].ok(f"processed_{msg}")
            >>> handler = FlextHandlers.create_from_callable(my_handler)
            >>> result = handler.handle("test")

        """

        class CallableHandler(FlextHandlers[t.Scalar, t.Scalar]):
            """Dynamic handler created from callable."""

            _handler_fn: Callable[[t.Scalar], t.Scalar]

            def __init__(
                self,
                handler_fn: Callable[[t.Scalar], t.Scalar],
                config: m.Handler | None = None,
            ) -> None:
                super().__init__(config=config)
                self._handler_fn = handler_fn

            @override
            def handle(self, message: t.Scalar) -> r[t.Scalar]:
                """Execute the wrapped callable."""
                if isinstance(message, tuple):
                    return r[t.Scalar].fail("Unexpected message type")
                try:
                    result = self._handler_fn(message)
                    if isinstance(result, r):
                        return result
                    if isinstance(result, set):
                        return r[t.Scalar].fail("Result must be compatible with Scalar")
                    return r[t.Scalar].ok(result)
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                ) as exc:
                    self.logger.debug("Callable handler execution failed", exc_info=exc)
                    return r[t.Scalar].fail(str(exc))

        if handler_config is not None:
            return CallableHandler(handler_fn=handler_callable, config=handler_config)
        resolved_type: c.Cqrs.HandlerType = c.Cqrs.HandlerType.COMMAND
        if mode is not None:
            if isinstance(mode, c.Cqrs.HandlerType):
                resolved_type = mode
            elif mode not in u.values(c.Cqrs.HandlerType):
                error_msg = f"Invalid handler mode: {mode}"
                raise e.ValidationError(error_msg)
            else:
                resolved_type = c.Cqrs.HandlerType(str(mode))
        elif handler_type is not None:
            resolved_type = handler_type
        resolved_name: str = handler_name or str(
            getattr(handler_callable, "__name__", "unknown_handler")
            or "unknown_handler"
        )
        config = m.Handler(
            handler_id=f"callable_{id(handler_callable)}",
            handler_name=resolved_name,
            handler_type=resolved_type,
            handler_mode=resolved_type,
        )
        return CallableHandler(handler_fn=handler_callable, config=config)

    @staticmethod
    def _handler_type_to_literal(
        handler_type: c.Cqrs.HandlerType,
    ) -> c.Cqrs.HandlerTypeLiteral:
        """Convert HandlerType StrEnum to HandlerTypeLiteral."""
        if handler_type in FlextHandlers._HANDLER_TYPE_LITERALS:
            return FlextHandlers._HANDLER_TYPE_LITERALS[handler_type]
        msg = f"Unsupported handler type: {handler_type}"
        raise ValueError(msg)

    @staticmethod
    def handler(
        command: type,
        *,
        priority: int = c.Discovery.DEFAULT_PRIORITY,
        timeout: float | None = c.Discovery.DEFAULT_TIMEOUT,
        middleware: list[type[p.Middleware]] | None = None,
    ) -> Callable[[t.HandlerCallable], t.HandlerCallable]:
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
            if not hasattr(func, c.Discovery.HANDLER_ATTR):
                config = m.DecoratorConfig(
                    command=command, priority=priority, timeout=timeout, middleware=[]
                )
                if middleware is not None:
                    config = config.model_copy(update={"middleware": list(middleware)})
                setattr(func, c.Discovery.HANDLER_ATTR, config)
            return func

        return decorator

    def can_handle(self, message_type: type) -> bool:
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
            return True
        return issubclass(message_type, self._expected_message_type)

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

    def execute(self, message: MessageT_contra) -> r[ResultT]:
        """Execute handler with complete validation and error handling pipeline.

        Implements the railway-oriented programming pattern by first validating
        the input message, then executing the business logic if validation passes.
        Uses r for consistent error handling without exceptions.

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
        validation = self.validate(message)
        if validation.is_failure:
            return r[ResultT].fail(validation.error or "Validation failed")
        return self.handle(message)

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
        raise NotImplementedError

    def pop_context(self) -> r[FlextModelsContainers.ConfigMap]:
        """Pop execution context from the local handler stack."""
        if not self._stack:
            return r[FlextModelsContainers.ConfigMap].ok(
                FlextModelsContainers.ConfigMap(root={})
            )
        popped = self._stack.pop()
        if isinstance(popped, m.ExecutionContext):
            context_dict = FlextModelsContainers.ConfigMap(
                root={
                    "handler_name": popped.handler_name,
                    "handler_mode": popped.handler_mode,
                }
            )
            return r[FlextModelsContainers.ConfigMap].ok(context_dict)
        return r[FlextModelsContainers.ConfigMap].ok(popped)

    def push_context(
        self,
        ctx: m.ExecutionContext | Mapping[str, t.NormalizedValue | BaseModel],
    ) -> r[bool]:
        """Push execution context onto the local handler stack."""
        if isinstance(ctx, m.ExecutionContext):
            self._stack.append(ctx)
            return r[bool].ok(value=True)
        handler_name_raw = ctx.get("handler_name", "unknown")
        handler_name = (
            str(handler_name_raw) if handler_name_raw is not None else "unknown"
        )
        handler_mode_raw = ctx.get("handler_mode", "operation")
        handler_mode_str = (
            str(handler_mode_raw) if handler_mode_raw is not None else "operation"
        )
        handler_mode_literal: c.Cqrs.HandlerTypeLiteral = (
            "command"
            if handler_mode_str == "command"
            else "query"
            if handler_mode_str == "query"
            else "event"
            if handler_mode_str == "event"
            else "saga"
            if handler_mode_str == "saga"
            else "operation"
        )
        execution_ctx = m.ExecutionContext.create_for_handler(
            handler_name=handler_name, handler_mode=handler_mode_literal
        )
        self._stack.append(execution_ctx)
        return r[bool].ok(value=True)

    def record_metric(self, name: str, value: t.MetadataAttributeValue) -> r[bool]:
        """Record a metric value in the current handler state."""
        self._metrics[name] = value
        return r[bool].ok(value=True)

    def validate_message(self, data: MessageT_contra) -> r[bool]:
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
            >>> result = handler.validate_message(invalid_data)
            >>> if result.is_failure:
            ...     print(f"Validation error: {result.error}")

        Note: self is required for subclass override compatibility, even though
        this base implementation doesn't use instance state.

        """
        if data is None:
            return r[bool].fail("Message cannot be None")
        return r[bool].ok(value=True)

    def validate(self, value: MessageT_contra) -> r[bool]:
        """Validate input — override in subclasses for domain-specific logic."""
        return self.validate_message(value)

    def _record_execution_metrics(
        self, *, success: bool, error: str | None = None
    ) -> None:
        """Record execution metrics (helper to reduce locals in _run_pipeline)."""
        exec_time_value = self._execution_context.execution_time_ms
        try:
            if isinstance(exec_time_value, int | float | str):
                exec_time = float(exec_time_value)
            else:
                exec_time = 0.0
        except (TypeError, ValueError):
            exec_time = 0.0
        _ = self.record_metric("execution_time_ms", exec_time)
        _ = self.record_metric("success", success)
        if error is not None:
            _ = self.record_metric("error", error)

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
        handler_mode = getattr(
            self._config_model.handler_mode, "value", self._config_model.handler_mode
        )
        valid_operations = {
            c.Dispatcher.HANDLER_MODE_COMMAND,
            c.Dispatcher.HANDLER_MODE_QUERY,
            c.Cqrs.HandlerType.EVENT.value,
        }
        if operation != handler_mode and operation in valid_operations:
            error_msg = f"Handler with mode '{handler_mode}' cannot execute {operation} pipelines"
            return r[ResultT].fail(error_msg)
        message_type = message.__class__
        if not self.can_handle(message_type):
            type_name = message_type.__name__
            error_msg = f"Handler cannot handle message type {type_name}"
            return r[ResultT].fail(error_msg)
        validation = self.validate(message)
        if validation.is_failure:
            error_detail = validation.error or "Validation failed"
            error_msg = f"Message validation failed: {error_detail}"
            return r[ResultT].fail(error_msg)
        self._execution_context.start_execution()
        _ = self.push_context(self._execution_context)
        try:
            result = self.handle(message)
            self._record_execution_metrics(success=result.is_success)
            return result
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
            self.logger.warning("Critical handler pipeline failure", exc_info=exc)
            self._record_execution_metrics(success=False, error=str(exc))
            error_msg = f"Critical handler failure: {exc}"
            return r[ResultT].fail(error_msg)
        finally:
            _ = self.pop_context()

    class Discovery:
        """Auto-discovery mechanism for handler decorators.

        Scans classes for methods decorated with @handler() and provides
        utilities for finding and analyzing handler configurations.

        This class enables zero-config handler registration in FlextService
        by automatically discovering decorated methods at initialization time.
        """

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
        def has_handlers_module(module: ModuleType) -> bool:
            """Check if module has any handler-decorated functions.

            Efficiently checks if a module contains any functions marked with
            the @handler() decorator without scanning all items.

            Args:
                module: Module to check for handlers

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

        @staticmethod
        def scan_class(
            target_class: type,
        ) -> list[tuple[str, m.DecoratorConfig]]:
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
            handlers: list[tuple[str, m.DecoratorConfig]] = []
            for name in dir(target_class):
                method = getattr(target_class, name, None)
                if hasattr(method, c.Discovery.HANDLER_ATTR):
                    config: m.DecoratorConfig = getattr(
                        method, c.Discovery.HANDLER_ATTR
                    )
                    handlers.append((name, config))
            return sorted(handlers, key=lambda x: x[1].priority, reverse=True)

        @staticmethod
        def scan_module(
            module: ModuleType,
        ) -> list[
            tuple[
                str,
                Callable[..., t.Scalar | None],
                m.DecoratorConfig,
            ]
        ]:
            """Scan module for functions decorated with @handler().

            Introspects the module to find all functions with handler configuration
            metadata, returning them sorted by priority for consistent ordering.

            Args:
                module: Module to scan for handler decorators

            Returns:
                List of tuples (function_name, function, DecoratorConfig) sorted by priority

            Example:
                >>> handlers = FlextHandlers.Discovery.scan_module(my_module)
                >>> for func_name, func, config in handlers:
                ...     print(f"{func_name}: {config.command.__name__}")

            """
            handlers: list[
                tuple[
                    str,
                    Callable[..., t.Scalar | None],
                    m.DecoratorConfig,
                ]
            ] = []
            for name in dir(module):
                if name.startswith("_"):
                    continue
                func = getattr(module, name, None)
                if not u.is_handler_callable(func):
                    continue
                if not hasattr(func, c.Discovery.HANDLER_ATTR):
                    continue
                if not callable(func):
                    continue
                config: m.DecoratorConfig = getattr(func, c.Discovery.HANDLER_ATTR)
                callable_func: Callable[..., t.Container | BaseModel | None] = func

                def narrowed_func(
                    message: BaseModel | t.Container,
                    captured_callable: Callable[
                        ..., t.Container | BaseModel | None
                    ] = callable_func,
                    **kwargs: t.Scalar,
                ) -> t.Scalar | None:
                    fn_candidate = kwargs.get("fn", captured_callable)
                    if not callable(fn_candidate):
                        return ""
                    result = fn_candidate(message)
                    if result is None:
                        return None
                    if isinstance(result, str | int | float | bool):
                        return result
                    return str(result)

                setattr(narrowed_func, c.Discovery.HANDLER_ATTR, config)
                handlers.append((name, narrowed_func, config))
            return sorted(handlers, key=lambda x: (-x[2].priority, x[0]))


h = FlextHandlers
__all__ = ["FlextHandlers", "h"]
