"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from types import ModuleType
from typing import ClassVar, Unpack, override

from pydantic import ConfigDict

from flext_core import c, e, m, p, r, t, u, x


class FlextHandlers[MessageT_contra, ResultT](x):
    """Abstract CQRS handler with validation and railway-style execution.

    Provides the base implementation for Command Query Responsibility Segregation
    (CQRS) handlers, implementing structural typing via p.Handler[MessageT_contra]
    through duck typing (no inheritance required). This class serves as the foundation
    for implementing command, query, and event handlers with comprehensive validation,
    execution pipelines, metrics collection, and configuration management.
    """

    _expected_message_type: ClassVar[type | None] = None
    _expected_result_type: ClassVar[type | None] = None
    _HANDLER_TYPE_LITERALS: ClassVar[Mapping[c.HandlerType, c.HandlerType]] = {
        c.HandlerType.COMMAND: c.HandlerType.COMMAND,
        c.HandlerType.QUERY: c.HandlerType.QUERY,
        c.HandlerType.EVENT: c.HandlerType.EVENT,
        c.HandlerType.OPERATION: c.HandlerType.OPERATION,
        c.HandlerType.SAGA: c.HandlerType.SAGA,
    }

    def __init__(self, *, config: m.Handler | None = None) -> None:
        """Initialize handler with configuration and context.

        Sets up the handler with optional configuration parameters.
        The config parameter accepts a m instance.

        Args:
            config: Optional handler configuration model

        """
        super().__init__(config_type=None, config_overrides=None, initial_context=None)
        if config is not None:
            self._config_model = config
        else:
            self._config_model = m.Handler(
                handler_id=f"handler_{id(self)}",
                handler_name=self.__class__.__name__,
            )
        handler_type = self._config_model.handler_mode
        valid_handler_types = {
            c.HandlerType.COMMAND,
            c.HandlerType.QUERY,
            c.HandlerType.EVENT,
            c.HandlerType.OPERATION,
            c.HandlerType.SAGA,
        }
        if handler_type not in valid_handler_types:
            error_msg = c.ERR_HANDLER_INVALID_MODE.format(mode=handler_type)
            raise e.ValidationError(error_msg)
        handler_mode_literal = self._handler_type_to_literal(handler_type)
        self._execution_context = m.ExecutionContext.create_for_handler(
            handler_name=self._config_model.handler_name,
            handler_mode=handler_mode_literal,
        )
        self._accepted_message_types: Sequence[type] = []
        self._revalidate_pydantic_messages: bool = False
        self._type_warning_emitted: bool = False
        self._metrics: MutableMapping[str, t.MetadataAttributeValue] = {}
        self._stack: MutableSequence[m.ExecutionContext | t.ConfigMap] = []

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
                msg = c.ERR_HANDLER_MISSING_HANDLE_IMPLEMENTATION.format(
                    qualname=cls.__qualname__,
                )
                raise TypeError(msg)
            if c.MethodName.HANDLE in klass.__dict__:
                break

    @property
    def handler_name(self) -> str:
        """Get handler name from configuration.

        Returns:
            str: The handler name

        """
        return self._config_model.handler_name

    @property
    def mode(self) -> c.HandlerType:
        """Get handler mode from configuration.

        Returns:
            c.HandlerType: The handler mode (command, query, event, saga)

        """
        return self._config_model.handler_mode

    @classmethod
    def create_from_callable(
        cls,
        handler_callable: Callable[[t.Scalar], t.Scalar],
        handler_name: str | None = None,
        handler_type: c.HandlerType | None = None,
        mode: c.HandlerType | str | None = None,
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
                    return r[t.Scalar].fail(c.ERR_UNEXPECTED_MESSAGE_TYPE)
                try:
                    result = self._handler_fn(message)
                    if isinstance(result, r):
                        return result
                    if isinstance(result, set):
                        return r[t.Scalar].fail(c.ERR_RESULT_NOT_SCALAR_COMPATIBLE)
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
        resolved_type: c.HandlerType = c.HandlerType.COMMAND
        if mode is not None:
            if isinstance(mode, c.HandlerType):
                resolved_type = mode
            elif mode not in u.enum_values(c.HandlerType):
                error_msg = c.ERR_HANDLER_INVALID_MODE.format(mode=mode)
                raise e.ValidationError(error_msg)
            else:
                resolved_type = c.HandlerType(str(mode))
        elif handler_type is not None:
            resolved_type = handler_type
        resolved_name: str = handler_name or str(
            getattr(handler_callable, "__name__", "unknown_handler")
            or "unknown_handler",
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
        handler_type: c.HandlerType,
    ) -> c.HandlerType:
        """Convert handler type to canonical HandlerType."""
        if handler_type in FlextHandlers._HANDLER_TYPE_LITERALS:
            return FlextHandlers._HANDLER_TYPE_LITERALS[handler_type]
        raise ValueError(
            c.ERR_HANDLER_UNSUPPORTED_TYPE.format(handler_type=handler_type),
        )

    @staticmethod
    def handler(
        command: type,
        *,
        priority: int = c.DEFAULT_MAX_COMMAND_RETRIES,
        timeout: float | None = c.DEFAULT_TIMEOUT_SECONDS,
        middleware: Sequence[type[p.Middleware]] | None = None,
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
            if not hasattr(func, c.HANDLER_ATTR):
                config = m.DecoratorConfig(
                    command=command,
                    priority=priority,
                    timeout=timeout,
                    middleware=[],
                )
                if middleware is not None:
                    config = config.model_copy(update={"middleware": list(middleware)})
                setattr(func, c.HANDLER_ATTR, config)
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
        operation: str = c.DEFAULT_HANDLER_MODE,
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
            >>> if result.success:
            ...     print(f"Success: {result.value}")
            ... else:
            ...     print(f"Failed: {result.error}")

        """
        validation = self.validate_message(message)
        if validation.failure:
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
        _ = message
        raise NotImplementedError

    def pop_context(self) -> r[t.ConfigMap]:
        """Pop execution context from the local handler stack."""
        if not self._stack:
            return r[t.ConfigMap].ok(t.ConfigMap(root={}))
        popped = self._stack.pop()
        if isinstance(popped, m.ExecutionContext):
            context_dict = t.ConfigMap(
                root={
                    "handler_name": popped.handler_name,
                    c.FIELD_HANDLER_MODE: popped.handler_mode,
                },
            )
            return r[t.ConfigMap].ok(context_dict)
        return r[t.ConfigMap].ok(popped)

    def push_context(
        self,
        ctx: m.ExecutionContext | t.ContainerMapping,
    ) -> r[bool]:
        """Push execution context onto the local handler stack."""
        if isinstance(ctx, m.ExecutionContext):
            self._stack.append(ctx)
            return r[bool].ok(True)
        handler_name_raw = ctx.get("handler_name", c.IDENTIFIER_UNKNOWN)
        handler_name = (
            str(handler_name_raw)
            if handler_name_raw is not None
            else c.IDENTIFIER_UNKNOWN
        )
        handler_mode_raw = ctx.get(
            c.FIELD_HANDLER_MODE,
            c.HandlerType.OPERATION,
        )
        handler_mode_str = (
            str(handler_mode_raw)
            if handler_mode_raw is not None
            else c.HandlerType.OPERATION
        )
        handler_mode_literal: c.HandlerType = (
            c.HandlerType.COMMAND
            if handler_mode_str == c.HandlerType.COMMAND
            else c.HandlerType.QUERY
            if handler_mode_str == c.HandlerType.QUERY
            else c.HandlerType.EVENT
            if handler_mode_str == c.HandlerType.EVENT
            else c.HandlerType.SAGA
            if handler_mode_str == "saga"
            else c.HandlerType.OPERATION
        )
        execution_ctx = m.ExecutionContext.create_for_handler(
            handler_name=handler_name,
            handler_mode=handler_mode_literal,
        )
        self._stack.append(execution_ctx)
        return r[bool].ok(True)

    def record_metric(self, name: str, value: t.MetadataAttributeValue) -> r[bool]:
        """Record a metric value in the current handler state."""
        self._metrics[name] = value
        return r[bool].ok(True)

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
            >>> if result.failure:
            ...     print(f"Validation error: {result.error}")

        Note: self is required for subclass override compatibility, even though
        this base implementation doesn't use instance state.

        """
        if data is None:
            return r[bool].fail(c.ERR_MESSAGE_CANNOT_BE_NONE)
        return r[bool].ok(True)

    def _record_execution_metrics(
        self,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record execution metrics (helper to reduce locals in _run_pipeline)."""
        raw_time = self._execution_context.execution_time_ms
        exec_time = u.to_float(raw_time() if callable(raw_time) else raw_time)
        _ = self.record_metric("execution_time_ms", exec_time)
        _ = self.record_metric("success", success)
        if error is not None:
            _ = self.record_metric(c.WarningLevel.ERROR, error)

    def _run_pipeline(
        self,
        message: MessageT_contra,
        operation: str = c.DEFAULT_HANDLER_MODE,
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
            self._config_model.handler_mode,
            "value",
            self._config_model.handler_mode,
        )
        valid_operations = {
            c.DEFAULT_HANDLER_MODE,
            c.HandlerMode.QUERY,
            c.HandlerType.EVENT.value,
        }
        if operation != handler_mode and operation in valid_operations:
            error_msg = c.ERR_HANDLER_INCOMPATIBLE_PIPELINE_MODE.format(
                handler_mode=handler_mode,
                operation=operation,
            )
            return r[ResultT].fail(error_msg)
        message_type = message.__class__
        if not self.can_handle(message_type):
            type_name = message_type.__name__
            error_msg = c.ERR_HANDLER_CANNOT_HANDLE_MESSAGE_TYPE.format(
                type_name=type_name,
            )
            return r[ResultT].fail(error_msg)
        validation = self.validate_message(message)
        if validation.failure:
            error_detail = validation.error or c.ERR_VALIDATION_FAILED
            error_msg = c.ERR_HANDLER_MESSAGE_VALIDATION_FAILED.format(
                error=error_detail,
            )
            return r[ResultT].fail(error_msg)
        self._execution_context.start_execution()
        _ = self.push_context(self._execution_context)
        try:
            result = self.handle(message)
            self._record_execution_metrics(success=result.success)
            return result
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
            self.logger.warning(c.LOG_HANDLER_PIPELINE_FAILURE, exc_info=exc)
            self._record_execution_metrics(success=False, error=str(exc))
            error_msg = c.ERR_HANDLER_CRITICAL_FAILURE.format(error=str(exc))
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
                hasattr(getattr(target_class, name, None), c.HANDLER_ATTR)
                for name in dir(target_class)
            )

        @staticmethod
        def scan_class(
            target_class: type,
        ) -> Sequence[tuple[str, m.DecoratorConfig]]:
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
            handlers: Sequence[tuple[str, m.DecoratorConfig]] = [
                (name, getattr(method, c.HANDLER_ATTR))
                for name in dir(target_class)
                if hasattr(method := getattr(target_class, name, None), c.HANDLER_ATTR)
            ]
            return sorted(handlers, key=lambda x: x[1].priority, reverse=True)

        @staticmethod
        def scan_module(
            module: ModuleType,
        ) -> Sequence[tuple[str, Callable[..., t.Scalar | None], m.DecoratorConfig]]:
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
            handlers: MutableSequence[
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
                if not u.handler_callable(func):
                    continue
                if not hasattr(func, c.HANDLER_ATTR):
                    continue
                if not callable(func):
                    continue
                config: m.DecoratorConfig = getattr(func, c.HANDLER_ATTR)
                callable_func: Callable[..., t.RuntimeAtomic | None] = func

                def narrowed_func(
                    message: t.RuntimeAtomic,
                    captured_callable: Callable[
                        ...,
                        t.RuntimeAtomic | None,
                    ] = callable_func,
                    **kwargs: t.Scalar,
                ) -> t.Scalar | None:
                    fn_candidate = kwargs.get("fn", captured_callable)
                    if not callable(fn_candidate):
                        return ""
                    result = fn_candidate(message)
                    if result is None:
                        return None
                    if u.primitive(result):
                        return result
                    return str(result)

                setattr(narrowed_func, c.HANDLER_ATTR, config)
                handlers.append((name, narrowed_func, config))
            return sorted(handlers, key=lambda x: (-x[2].priority, x[0]))


h = FlextHandlers

__all__ = ["FlextHandlers", "h"]
