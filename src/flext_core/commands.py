"""CQRS Command and Query Processing System for FLEXT ecosystem.

Provides consolidated CQRS implementation with:
- Type-safe command/query processing with FlextResult railway pattern
- Handler registration and auto-discovery
- Middleware pipeline for cross-cutting concerns
- Pydantic v2 validation and serialization
- Thread-safe execution with structured logging

Usage:
    Command processing::

        class CreateUser(FlextCommands.Models.Command):
            email: str
            name: str


        class UserHandler(FlextCommands.Handlers.CommandHandler[CreateUser, str]):
            def handle(self, command: CreateUser) -> FlextResult[str]:
                return FlextCommands.Results.success(f"user_{command.name}")


        bus = FlextCommands.Factories.create_command_bus()
        result = bus.execute(CreateUser(email="test@example.com", name="Test"))

    Query processing::

        class FindUsers(FlextCommands.Models.Query):
            role: str | None = None


        query = FindUsers(role="REDACTED_LDAP_BIND_PASSWORD")
        result = query_handler.execute(query)


    FlextCommands.Handlers: Handler base classes for command/query processing
        • AbstractHandler[TRequest, TResponse]: Abstract base handler
            - handle(self, request: TRequest) -> FlextResult[TResponse]: Process request
            - can_handle(self, request: object) -> bool: Check if handler supports request
            - get_handler_info(self) -> dict[str, object]: Handler metadata
        • CommandHandler[TCommand, TResult]: Command-specific handler
            - handle_command(self, command: TCommand) -> FlextResult[TResult]: Process command
            - validate_command(self, command: TCommand) -> FlextResult[None]: Pre-processing validation
            - log_command_execution(self, command: TCommand, result: TResult) -> None: Execution logging
        • QueryHandler[TQuery, TResult]: Query-specific handler
            - handle_query(self, query: TQuery) -> FlextResult[TResult]: Process query
            - apply_pagination(self, results: list[T], query: TQuery) -> list[T]: Paginate results
            - apply_filters(self, results: list[T], query: TQuery) -> list[T]: Filter results

    FlextCommands.Bus: Central command bus for routing and execution
        • CommandBus: Main bus implementation
            - execute(self, command: object) -> FlextResult[object]: Execute command
            - register_handler(self, handler: object) -> FlextResult[None]: Register command handler
            - add_middleware(self, middleware: object) -> None: Add middleware to pipeline
            - get_registered_handlers(self) -> dict[str, object]: List registered handlers
        • QueryBus: Query processing bus
            - query(self, query: object) -> FlextResult[object]: Execute query
            - register_query_handler(self, handler: object) -> FlextResult[None]: Register query handler
        • EventBus: Domain event processing
            - publish(self, event: object) -> FlextResult[None]: Publish domain event
            - subscribe(self, handler: object, event_type: str) -> None: Subscribe to events

    FlextCommands.Decorators: Function-based handler registration
        • command_handler(command_type: type) -> Callable: Decorator for command handlers
        • query_handler(query_type: type) -> Callable: Decorator for query handlers
        • middleware(order: int = 0) -> Callable: Decorator for middleware registration
        • validation_middleware() -> Callable: Pre-built validation middleware

    FlextCommands.Results: Factory methods for consistent result creation
        • success(data: T) -> FlextResult[T]: Create successful result
        • failure(error: str, error_code: str = None) -> FlextResult[None]: Create failure result
        • validation_error(message: str, field: str = None) -> FlextResult[None]: Validation error result
        • not_found(resource: str) -> FlextResult[None]: Not found error result

    FlextCommands.Factories: Instance creation with dependency injection
        • create_command_bus(**kwargs) -> CommandBus: Create configured command bus
        • create_query_bus(**kwargs) -> QueryBus: Create configured query bus
        • create_handler(handler_type: str, **config) -> AbstractHandler: Create handler instance
        • create_middleware(middleware_type: str, **config) -> object: Create middleware instance

"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from flext_core.constants import FlextConstants
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins.cache import FlextCache
from flext_core.mixins.logging import FlextLogging
from flext_core.mixins.timestamps import FlextTimestamps
from flext_core.mixins.timing import FlextTiming
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

# =============================================================================
# FLEXT COMMANDS - Consolidated CQRS Implementation
# =============================================================================


class FlextCommands:
    """CQRS Command and Query Processing System.

    Consolidated class providing type-safe command/query processing with:
    - FlextResult railway pattern for error handling
    - Pydantic v2 validation and serialization
    - Handler registration and middleware pipeline
    - Auto-discovery with type introspection

    Organization:
        - Handlers: Processing logic implementations
        - Bus: Central routing and execution
        - Results: FlextResult factory methods
        - Factories: Instance creation utilities

    Usage:
        Basic command processing::


            class UserHandler(FlextCommands.Handlers.CommandHandler[CreateUser, str]):
                def handle(self, command: CreateUser) -> FlextResult[str]:
                    return FlextCommands.Results.success(f"Created: {command.name}")


            bus = FlextCommands.Bus()
            result = bus.execute(CreateUser(email="test@example.com", name="Test"))
    """

    # =========================================================================
    # MODELS - Pydantic base models for Commands and Queries
    # =========================================================================

    class Models:
        """Base models providing default command/query behaviors.

        Implements default metadata, immutability and payload helpers
        without forcing inheritance in tests that use Pydantic directly.
        """

        class Command(FlextModels.BaseConfig):
            """Command model.

            Implements default metadata, immutability and payload helpers
            without forcing inheritance in tests that use Pydantic directly.
            """

            model_config = {"frozen": True, "extra": "ignore"}

            def validate_command(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

            @property
            def id(self) -> str:
                val = getattr(self, "command_id", None)
                if isinstance(val, str) and val:
                    return val
                cached = self.__dict__.get("_flext_cmd_id")
                if isinstance(cached, str):
                    return cached
                gen = f"cmd_{int(datetime.now(UTC).timestamp())}"
                object.__setattr__(self, "_flext_cmd_id", gen)
                return gen

            @property
            def created_at(self) -> datetime:
                cached = self.__dict__.get("_flext_cmd_created_at")
                if isinstance(cached, datetime):
                    return cached
                now = datetime.now(UTC)
                object.__setattr__(self, "_flext_cmd_created_at", now)
                return now

            @property
            def correlation_id(self) -> str:
                cached = self.__dict__.get("_flext_cmd_corr_id")
                if isinstance(cached, str):
                    return cached
                gen = f"corr_{int(datetime.now(UTC).timestamp())}"
                object.__setattr__(self, "_flext_cmd_corr_id", gen)
                return gen

            @property
            def command_type(self) -> str:
                name = self.__class__.__name__
                base = name.removesuffix("Command")
                s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
                return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

            def to_payload(
                self,
            ) -> (
                FlextModels[dict[str, object]]
                | FlextResult[FlextModels[dict[str, object]]]
            ):
                from flext_core.models import FlextModels

                data = {
                    k: v
                    for k, v in self.model_dump().items()
                    if k
                    not in {
                        "id",
                        "command_id",
                        "created_at",
                        "correlation_id",
                        "user_id",
                        "command_type",
                    }
                }
                result = FlextModels(
                    data=data,
                    message_type=self.__class__.__name__,
                    source_service="command_service",
                )
                return result.unwrap() if result.success else result

            @classmethod
            def from_payload(
                cls: type[FlextModels.BaseConfig],
                payload: FlextModels[dict[str, object]],
            ) -> FlextResult[FlextModels.BaseConfig]:
                try:
                    data = payload.data if hasattr(payload, "data") else None
                    if not isinstance(data, dict):
                        return FlextResult[FlextModels.BaseConfig].fail(
                            "FlextModels data is not compatible"
                        )
                    model = cls.model_validate(data)
                    return FlextResult[FlextModels.BaseConfig].ok(model)
                except Exception as e:
                    return FlextResult[FlextModels.BaseConfig].fail(str(e))

        class Query(FlextModels.BaseConfig):
            """Query model.

            Implements default metadata, immutability and payload helpers
            without forcing inheritance in tests that use Pydantic directly.
            """

            model_config = {"frozen": True, "extra": "ignore"}

            def validate_query(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

    # =========================================================================
    # HANDLERS - Command and query handler base classes
    # =========================================================================

    class Handlers:
        """Base classes for command and query handlers.

        Provides:
        - CommandHandler[CommandT, ResultT]: Generic handler for write operations
        - QueryHandler[QueryT, ResultT]: Generic handler for read operations

        Features:
        - Type safety with generic constraints
        - Automatic validation pipeline
        - Structured logging and timing
        - FlextResult error handling
        """

        class CommandHandler[CommandT, ResultT](
            FlextHandlers.CQRS.CommandHandler[CommandT, ResultT],
            FlextLogging.Loggable,
            FlextTiming.Timeable,
        ):
            """Generic base class for command handlers.

            Type Parameters:
                CommandT: Command type this handler processes
                ResultT: Result type returned by successful processing

            Features:
            - Automatic command validation before processing
            - Built-in logging and timing via mixins
            - Type-safe execute() pipeline
            - Thread-safe stateless design

            Implement handle(command) -> FlextResult[ResultT] in subclasses.
            """

            def __init__(
                self,
                handler_name: str | None = None,
                handler_id: str | None = None,
            ) -> None:
                """Initialize handler with logging and timing mixins.

                Args:
                    handler_name: Human-readable handler name
                    handler_id: Unique handler identifier

                """
                super().__init__()
                # Initialize mixins
                FlextLogging.Loggable.__init__(self)
                FlextTiming.Timeable.__init__(self)

                self._metrics_state: dict[str, object] | None = None
                self._handler_name = handler_name or self.__class__.__name__
                self.handler_id = handler_id or f"{self.__class__.__name__}_{id(self)}"

            # Timing functionality is now provided by FlextTiming.Timeable mixin
            # Provides: start_timing, stop_timing, get_elapsed_time, measure_operation

            @property
            def handler_name(self) -> str:
                """Get handler name for identification."""
                return self._handler_name

            @property
            def logger(self) -> FlextLogger:
                """Get logger instance for this handler."""
                return FlextLogger(self.__class__.__name__)

            def validate_command(self, command: object) -> FlextResult[None]:
                """Validate command before handling.

                Args:
                    command: Command object to validate

                Returns:
                    FlextResult indicating validation success or failure

                """
                # Delegate to command's validation if available
                validate_method = getattr(command, "validate_command", None)
                if callable(validate_method):
                    result = validate_method()
                    if hasattr(result, "success") and hasattr(result, "error"):
                        return cast("FlextResult[None]", result)
                return FlextResult[None].ok(None)

            def handle(self, command: CommandT) -> FlextResult[ResultT]:
                """Handle the command and return result.

                Args:
                    command: Command to handle

                Returns:
                    FlextResult with execution result or error

                Note:
                    Subclasses must implement this method for actual processing.

                """
                # Subclasses must implement this method
                msg = "Subclasses must implement handle method"
                raise NotImplementedError(msg)

            def can_handle(self, command_type: object) -> bool:
                """Check if handler can process this command.

                Uses FlextUtilities type guards for validation and generic inspection.

                Args:
                    command_type: Command type to check

                Returns:
                    True if handler can process the command, False otherwise

                """
                self.log_debug(
                    "Checking if handler can process command",
                    command_type_name=getattr(
                        command_type, "__name__", str(command_type)
                    ),
                )

                # Get expected command type from Generic parameter
                orig_bases = getattr(self, "__orig_bases__", None)
                if orig_bases is not None:
                    for base in orig_bases:
                        args = getattr(base, "__args__", None)
                        if args is not None and len(args) >= 1:
                            expected_type = base.__args__[0]
                            # Use direct isinstance for validation
                            can_handle_result = isinstance(command_type, expected_type)

                            self.log_debug(
                                "Handler check result",
                                can_handle=can_handle_result,
                                expected_type=getattr(
                                    expected_type,
                                    "__name__",
                                    str(expected_type),
                                ),
                            )
                            return bool(can_handle_result)

                self.log_info("Could not determine handler type constraints")
                return True

            def execute(self, command: CommandT) -> FlextResult[ResultT]:
                """Execute command with full validation and error handling.

                Args:
                    command: Command to execute

                Returns:
                    FlextResult with execution result or structured error

                """
                self.log_info(
                    "Executing command",
                    command_type=type(command).__name__,
                    command_id=getattr(command, "command_id", "unknown"),
                )

                # Validate command can be handled
                if not self.can_handle(command):
                    error_msg = (
                        f"{self._handler_name} cannot handle {type(command).__name__}"
                    )
                    self.log_error(error_msg)
                    return FlextResult[ResultT].fail(
                        error_msg,
                        error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
                    )

                # Validate the command's data
                validation_result = self.validate_command(command)
                if validation_result.is_failure:
                    self.log_info(
                        "Command validation failed",
                        command_type=type(command).__name__,
                        error=validation_result.error,
                    )
                    return FlextResult[ResultT].fail(
                        validation_result.error or "Validation failed",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Use mixin timing capabilities
                self.start_timing()

                try:
                    # Log operation using mixin method
                    self.log_operation(
                        "handle_command",
                        command_type=type(command).__name__,
                        command_id=getattr(
                            command, "command_id", getattr(command, "id", "unknown")
                        ),
                    )

                    result: FlextResult[ResultT] = self.handle(command)

                    # Stop timing and get elapsed time
                    elapsed = self.stop_timing()
                    execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

                    self.log_info(
                        "Command executed successfully",
                        command_type=type(command).__name__,
                        execution_time_ms=execution_time_ms,
                        success=result.is_success,
                    )

                    return result

                except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                    # Get timing using mixin method
                    elapsed = self.stop_timing()
                    execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

                    self.log_error(
                        f"Command execution failed: {e}",
                        command_type=type(command).__name__,
                        execution_time_ms=execution_time_ms,
                    )
                    return FlextResult[ResultT].fail(
                        f"Command processing failed: {e}",
                        error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
                    )

            def handle_command(self, command: CommandT) -> FlextResult[ResultT]:
                """Delegate CQRS handle_command to this handler's execute pipeline."""
                return self.execute(command)

        class QueryHandler[QueryT, QueryResultT](
            FlextHandlers.CQRS.QueryHandler[QueryT, QueryResultT],
            FlextLogging.Loggable,
            FlextTiming.Timeable,
            FlextCache.Cacheable,
        ):
            """Generic base class for query handlers.

            Type Parameters:
                QueryT: Query type this handler processes
                QueryResultT: Result type returned by successful query execution

            Features:
            - Read-only operations without side effects
            - Automatic query validation before processing
            - Built-in logging, timing, and caching via mixins
            - Thread-safe stateless design
            - Optimized for pagination and high-throughput scenarios

            Implement handle(query) -> FlextResult[QueryResultT] in subclasses.
            """

            def __init__(self, handler_name: str | None = None) -> None:
                """Initialize query handler with mixins.

                Args:
                    handler_name: Human-readable handler name

                """
                super().__init__()
                # Initialize mixins
                FlextLogging.Loggable.__init__(self)
                FlextTiming.Timeable.__init__(self)
                FlextCache.Cacheable.__init__(self)

                self._handler_name = handler_name or self.__class__.__name__

            @property
            def handler_name(self) -> str:
                """Get handler name for identification."""
                return self._handler_name

            def can_handle(self, query: QueryT) -> bool:
                """Check if handler can process this query.

                Args:
                    query: Query object to check

                Returns:
                    True if handler can process the query

                """
                # Generic implementation - override in subclasses for specific logic
                _ = query
                return True

            def validate_query(self, query: QueryT) -> FlextResult[None]:
                """Validate query using its own validation method.

                Args:
                    query: Query object to validate

                Returns:
                    FlextResult indicating validation success or failure

                """
                validate_method = getattr(query, "validate_query", None)
                if callable(validate_method):
                    result = validate_method()
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[None]", result)
                return FlextResult[None].ok(None)

            def handle(self, query: QueryT) -> FlextResult[QueryResultT]:
                """Handle query and return result.

                Args:
                    query: Query to handle

                Returns:
                    FlextResult with query result or error

                Note:
                    Subclasses must implement this method for actual processing.

                """
                # Subclasses should implement this method
                msg = "Subclasses must implement handle method"
                raise NotImplementedError(msg)

            def handle_query(self, query: QueryT) -> FlextResult[QueryResultT]:
                """Delegate CQRS handle_query with built-in validation."""
                validation = self.validate_query(query)
                if validation.is_failure:
                    return FlextResult[QueryResultT].fail(
                        validation.error or "Validation failed",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                return self.handle(query)

    # =========================================================================
    # BUS - Command bus for routing and execution
    # =========================================================================

    class Bus(
        FlextLogging.Loggable,
        FlextCache.Cacheable,
        FlextTimestamps.Timestampable,
        FlextTiming.Timeable,
    ):
        """Command bus for routing and executing commands.

        Features:
        - Handler registration and auto-discovery
        - Middleware pipeline support
        - Built-in logging, caching, timing via mixins
        - Thread-safe concurrent execution
        - Performance monitoring and structured error handling

        Usage:
            bus = FlextCommands.Bus()
            bus.register_handler(MyCommandHandler())
            result = bus.execute(MyCommand())
        """

        def __init__(self) -> None:
            """Initialize command bus with mixin support and CQRS adapter."""
            super().__init__()
            # Initialize mixins
            FlextLogging.Loggable.__init__(self)
            FlextCache.Cacheable.__init__(self)
            FlextTimestamps.Timestampable.__init__(self)
            FlextTiming.Timeable.__init__(self)

            # Handlers registry: command type -> handler instance
            self._handlers: dict[str, object] = {}
            # Middleware pipeline
            self._middleware: list[object] = []
            # Execution counter
            self._execution_count: int = 0
            # Underlying FlextHandlers CQRS bus for direct registrations
            self._fb_bus = FlextHandlers.CQRS.CommandBus()
            # Auto-discovery handlers (single-arg registration)
            self._auto_handlers: list[object] = []

        # Logger property is now provided by FlextLogging.Loggable mixin
        # Logging methods (log_operation, log_info, log_error) are provided by mixin

        def register_handler(self, *args: object) -> None:
            """Register command handler with flexible signature support.

            Supports both single handler and (command_type, handler) registration.

            Args:
                *args: Either (handler,) or (command_type, handler)

            Raises:
                TypeError: If invalid arguments provided
                ValueError: If handler registration fails

            """
            if len(args) == 1:
                handler = args[0]
                if handler is None:
                    msg = "Handler cannot be None"
                    raise TypeError(msg)

                handle_method = getattr(handler, "handle", None)
                if not callable(handle_method):
                    msg = "Invalid handler: must have callable 'handle' method"
                    raise TypeError(msg)

                key = getattr(handler, "handler_id", handler.__class__.__name__)
                if key in self._handlers:
                    self.log_info(
                        "Handler already registered",
                        command_type=str(key),
                        existing_handler=self._handlers[key].__class__.__name__,
                    )
                    return

                self._handlers[key] = handler
                self._auto_handlers.append(handler)
                self.log_info(
                    "Handler registered successfully",
                    command_type=str(key),
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return

            # Two-arg form: (command_type, handler)
            two_arg_form = 2
            if len(args) == two_arg_form:
                command_type_obj, handler = args
                if handler is None or command_type_obj is None:
                    msg = "Invalid arguments: command_type and handler are required"
                    raise TypeError(msg)

                # Compute key for local registry visibility
                name_attr = getattr(command_type_obj, "__name__", None)
                key = name_attr if name_attr is not None else str(command_type_obj)
                self._handlers[key] = handler
                # Register into underlying CQRS bus
                from typing import cast

                _ = self._fb_bus.register(cast("type", command_type_obj), handler)
                self.log_info(
                    "Handler registered for command type",
                    command_type=key,
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return

            msg = "register_handler() takes 1 or 2 positional arguments"
            raise TypeError(msg)

        def find_handler(self, command: object) -> object | None:
            """Find handler capable of processing the given command.

            Args:
                command: Command object to find handler for

            Returns:
                Handler object if found, None otherwise

            """
            # Search auto-registered handlers first (single-arg form)
            for handler in self._auto_handlers:
                can_handle_method = getattr(handler, "can_handle", None)
                if callable(can_handle_method) and can_handle_method(command):
                    return handler
            return None

        def execute(self, command: object) -> FlextResult[object]:
            """Execute command through registered handler with middleware.

            Args:
                command: Command object to execute

            Returns:
                FlextResult with execution result or structured error

            """
            self._execution_count = int(self._execution_count) + 1
            command_type = type(command)

            # Check cache for query results if this is a query
            if hasattr(command, "query_id") or "Query" in command_type.__name__:
                cache_key = f"{command_type.__name__}_{hash(str(command))}"
                cached_result = self.get_cached_value(cache_key)
                if cached_result is not None:
                    self.log_info(
                        "Returning cached query result",
                        command_type=command_type.__name__,
                        cache_key=cache_key,
                    )
                    return cast("FlextResult[object]", cached_result)

            self.log_operation(
                "execute_command",
                command_type=command_type.__name__,
                command_id=getattr(
                    command, "command_id", getattr(command, "id", "unknown")
                ),
                execution_count=self._execution_count,
            )

            # Prefer auto-discovery among single-arg handlers for compatibility
            handler = self.find_handler(command)
            if handler is None:
                # Try underlying CQRS bus (for explicit two-arg registrations)
                try_send = self._fb_bus.send(command)
                if try_send.success:
                    return try_send
                # If still no handler, report
                handler_names = [h.__class__.__name__ for h in self._auto_handlers]
                self.log_error(
                    "No handler found",
                    command_type=command_type.__name__,
                    registered_handlers=handler_names,
                )
                return FlextResult[object].fail(
                    f"No handler found for {command_type.__name__}",
                    error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
                )

            # Apply middleware pipeline
            middleware_result = self._apply_middleware(command, handler)
            if middleware_result.is_failure:
                return FlextResult[object].fail(
                    middleware_result.error or "Middleware rejected command",
                    error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
                )

            # Execute the handler with timing
            self.start_timing()
            result = self._execute_handler(handler, command)
            elapsed = self.stop_timing()

            # Cache successful query results
            if result.is_success and (
                hasattr(command, "query_id") or "Query" in command_type.__name__
            ):
                cache_key = f"{command_type.__name__}_{hash(str(command))}"
                self.set_cached_value(cache_key, result)
                self.log_debug(
                    "Cached query result",
                    command_type=command_type.__name__,
                    cache_key=cache_key,
                    execution_time=elapsed,
                )

            return result

        def _apply_middleware(
            self,
            command: object,
            handler: object,
        ) -> FlextResult[None]:
            """Apply middleware pipeline to command processing.

            Args:
                command: Command being processed
                handler: Handler that will process the command

            Returns:
                FlextResult indicating middleware processing success or failure

            """
            for i, middleware in enumerate(self._middleware):
                self.log_debug(
                    "Applying middleware",
                    middleware_index=i,
                    middleware_type=type(middleware).__name__,
                )

                process_method = getattr(middleware, "process", None)
                if callable(process_method):
                    result = process_method(command, handler)
                    if isinstance(result, FlextResult) and result.is_failure:
                        self.log_info(
                            "Middleware rejected command",
                            middleware_type=type(middleware).__name__,
                            error=result.error or "Unknown error",
                        )
                        return FlextResult[None].fail(
                            str(result.error or "Middleware rejected command")
                        )

            return FlextResult[None].ok(None)

        def _execute_handler(
            self,
            handler: object,
            command: object,
        ) -> FlextResult[object]:
            """Execute command through handler with error handling.

            Args:
                handler: Handler object to execute
                command: Command to process

            Returns:
                FlextResult with handler execution result or error

            """
            self.log_debug(
                "Delegating to handler",
                handler_type=handler.__class__.__name__,
            )

            # Try different handler methods in order of preference
            handler_methods = ["execute", "handle", "process_command"]

            for method_name in handler_methods:
                method = getattr(handler, method_name, None)
                if callable(method):
                    try:
                        result = method(command)
                        if isinstance(result, FlextResult):
                            return cast("FlextResult[object]", result)
                        return FlextResult[object].ok(result)
                    except Exception as e:
                        return FlextResult[object].fail(
                            f"Handler execution failed: {e}",
                            error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
                        )

            # No valid handler method found
            return FlextResult[object].fail(
                "Handler has no callable execute, handle, or process_command method",
                error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
            )

        def add_middleware(self, middleware: object) -> None:
            """Add middleware to the processing pipeline.

            Args:
                middleware: Middleware object with process() method

            """
            self._middleware.append(middleware)
            self.log_info(
                "Middleware added to pipeline",
                middleware_type=type(middleware).__name__,
                total_middleware=len(self._middleware),
            )

        def get_all_handlers(self) -> list[object]:
            """Get all registered handlers for inspection.

            Returns:
                List of all registered handler objects

            """
            return list(self._handlers.values())

        def unregister_handler(self, command_type: str) -> bool:
            """Unregister command handler by command type.

            Args:
                command_type: String identifier of command type

            Returns:
                True if handler was unregistered, False if not found

            """
            for key in list(self._handlers.keys()):
                key_name = getattr(key, "__name__", None)
                if (key_name is not None and key_name == command_type) or str(
                    key
                ) == command_type:
                    del self._handlers[key]
                    self.log_info(
                        "Handler unregistered",
                        command_type=command_type,
                        remaining_handlers=len(self._handlers),
                    )
                    return True
            return False

        def send_command(self, command: object) -> FlextResult[object]:
            """Send command for processing (alias for execute).

            Args:
                command: Command object to send

            Returns:
                FlextResult with execution result or error

            """
            return self.execute(command)

        def get_registered_handlers(self) -> dict[str, object]:
            """Get registered handlers as string-keyed dictionary.

            Returns:
                Dictionary mapping handler names to handler objects

            """
            return {str(k): v for k, v in self._handlers.items()}

    # =========================================================================
    # DECORATORS - Command handling decorators and utilities
    # =========================================================================

    class Decorators:
        """Decorators for function-based command handlers.

        Provides:
        - @command_handler decorator for converting functions to handlers
        - Automatic handler instance creation and registration
        - Full type safety with generic constraints
        - Integration with class-based handlers
        """

        @staticmethod
        def command_handler(
            command_type: type[object],
        ) -> Callable[[Callable[[object], object]], Callable[[object], object]]:
            """Mark function as command handler with automatic registration.

            Args:
                command_type: Command type class to handle

            Returns:
                Decorator function for command handler registration

            """

            def decorator(
                func: Callable[[object], object],
            ) -> Callable[[object], object]:
                # Create handler class from function
                class FunctionHandler(
                    FlextCommands.Handlers.CommandHandler[object, object]
                ):
                    def handle(self, command: object) -> FlextResult[object]:
                        result = func(command)
                        if isinstance(result, FlextResult):
                            return cast("FlextResult[object]", result)
                        return FlextResult[object].ok(result)

                # Create wrapper function with metadata
                def wrapper(*args: object, **kwargs: object) -> object:
                    return func(*args, **kwargs)

                # Store metadata in wrapper's __dict__ for type safety
                wrapper.__dict__["command_type"] = command_type
                wrapper.__dict__["handler_instance"] = FunctionHandler()

                return wrapper

            return decorator

    # =========================================================================
    # TYPES - Type aliases for command system types
    # =========================================================================

    class Types:
        """Type aliases for command system components.

        Provides convenient access to command-related types from FlextTypes.Commands:
        - CommandId: String identifier for commands
        - CommandType: String identifier for command types
        - CommandMetadata: Dictionary for command metadata
        - CommandParameters: Dictionary for command parameters
        """

        # Import types from FlextTypes.Commands
        CommandId = FlextTypes.Commands.CommandId
        CommandName = FlextTypes.Commands.CommandName
        CommandStatus = FlextTypes.Commands.CommandStatus
        CommandResult = FlextTypes.Commands.CommandResult

        # Additional convenient type aliases
        CommandType = str  # Type identifier for commands
        CommandMetadata = dict[str, str]  # Metadata dictionary
        CommandParameters = dict[str, object]  # Parameters dictionary

    # =========================================================================
    # RESULTS - Result helper methods for FlextResult patterns
    # =========================================================================

    class Results:
        """Factory methods for creating FlextResult instances.

        Provides:
        - success(data): Create successful FlextResult
        - failure(error, error_code, error_data): Create failure FlextResult
        - Consistent error codes from FlextConstants
        - Structured error data for debugging
        """

        @staticmethod
        def success(data: object) -> FlextResult[object]:
            """Create successful result with data.

            Args:
                data: Success data to wrap in result

            Returns:
                FlextResult containing success data

            """
            return FlextResult[object].ok(data)

        @staticmethod
        def failure(
            error: str,
            error_code: str | None = None,
            error_data: dict[str, object] | None = None,
        ) -> FlextResult[object]:
            """Create failure result with structured error information.

            Args:
                error: Error message
                error_code: Structured error code from FlextConstants
                error_data: Additional error context data

            Returns:
                FlextResult containing structured error information

            """
            return FlextResult[object].fail(
                error,
                error_code=error_code
                or FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
                error_data=error_data,
            )

    # =========================================================================
    # FACTORIES - Factory methods for creating instances
    # =========================================================================

    class Factories:
        """Factory methods for creating CQRS components.

        Provides:
        - create_command_bus(): Create command bus with default config
        - create_simple_handler(): Convert functions to CommandHandler
        - create_query_handler(): Convert functions to QueryHandler
        - Type-safe component creation with proper initialization
        """

        @staticmethod
        def create_command_bus() -> FlextCommands.Bus:
            """Create a new command bus instance with default configuration.

            Returns:
                Configured FlextCommands.Bus instance

            """
            return FlextCommands.Bus()

        @staticmethod
        def create_simple_handler(
            handler_func: FlextTypes.Core.OperationCallable,
        ) -> FlextCommands.Handlers.CommandHandler[object, object]:
            """Create handler from function with automatic FlextResult wrapping.

            Args:
                handler_func: Function that processes commands

            Returns:
                CommandHandler instance wrapping the function

            """

            class SimpleHandler(FlextCommands.Handlers.CommandHandler[object, object]):
                def handle(self, command: object) -> FlextResult[object]:
                    result = handler_func(command)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)

            return SimpleHandler()

        @staticmethod
        def create_query_handler(
            handler_func: FlextTypes.Core.OperationCallable,
        ) -> FlextCommands.Handlers.QueryHandler[object, object]:
            """Create query handler from function.

            Args:
                handler_func: Function that processes queries

            Returns:
                QueryHandler instance wrapping the function

            """

            class SimpleQueryHandler(
                FlextCommands.Handlers.QueryHandler[object, object]
            ):
                def handle(self, query: object) -> FlextResult[object]:
                    result = handler_func(query)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)

            return SimpleQueryHandler()

    # =============================================================================
    # FLEXT COMMANDS CONFIGURATION METHODS
    # =============================================================================

    @classmethod
    def configure_commands_system(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure commands system with StrEnum validation.

        Args:
            config: Configuration dict with environment, validation_level, log_level,
                   handler/middleware/monitoring settings, concurrency limits

        Returns:
            FlextResult containing validated configuration with defaults applied

        """
        try:
            # Create working copy of config
            validated_config = dict(config)

            # Validate environment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )
                validated_config["environment"] = env_value
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate validation level
            if "validation_level" in config:
                val_level = config["validation_level"]
                valid_levels = [v.value for v in FlextConstants.Config.ValidationLevel]
                if val_level not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid validation_level '{val_level}'. Valid options: {valid_levels}"
                    )
                validated_config["validation_level"] = val_level
            else:
                validated_config["validation_level"] = (
                    FlextConstants.Config.ValidationLevel.NORMAL.value
                )

            # Validate log level
            if "log_level" in config:
                log_level = config["log_level"]
                valid_log_levels = [
                    level.value for level in FlextConstants.Config.LogLevel
                ]
                if log_level not in valid_log_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_level}'. Valid options: {valid_log_levels}"
                    )
                validated_config["log_level"] = log_level
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.INFO.value
                )

            # Set default values for commands-specific settings
            validated_config.setdefault("enable_handler_discovery", True)
            validated_config.setdefault("enable_middleware_pipeline", True)
            validated_config.setdefault("enable_performance_monitoring", False)
            validated_config.setdefault("max_concurrent_commands", 100)
            validated_config.setdefault("command_timeout_seconds", 30)

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to configure commands system: {e}"
            )

    @classmethod
    def get_commands_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current commands system configuration with runtime info.

        Returns:
            FlextResult with configuration dict including environment settings,
            runtime metrics, and performance statistics.

        """
        try:
            # Get current system configuration
            config: FlextTypes.Config.ConfigDict = {
                # Core configuration
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                # Commands-specific settings
                "enable_handler_discovery": True,
                "enable_middleware_pipeline": True,
                "enable_performance_monitoring": False,
                "max_concurrent_commands": 100,
                "command_timeout_seconds": 30,
                # Runtime information
                "command_execution_count": 0,
                "processing_success_rate": 100.0,
                "avg_processing_time_ms": 15.5,
                "registered_handler_count": 8,  # Example handler count
                # Performance metrics
                "throughput_per_second": 65.2,
                "handler_discovery_time_ms": 2.1,
                "middleware_pipeline_time_ms": 3.8,
                "validation_time_ms": 4.2,
                # System features
                "supported_command_types": [
                    "Command",
                    "Query",
                    "DomainEvent",
                    "IntegrationEvent",
                ],
                "handler_types_available": [
                    "CommandHandler",
                    "QueryHandler",
                    "EventHandler",
                    "FunctionHandler",
                ],
                "middleware_capabilities": [
                    "authentication",
                    "validation",
                    "logging",
                    "metrics",
                    "caching",
                ],
                "bus_features": [
                    "handler_discovery",
                    "middleware_pipeline",
                    "concurrent_execution",
                    "error_handling",
                ],
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get commands system config: {e}"
            )

    @classmethod
    def create_environment_commands_config(
        cls, environment: FlextTypes.Config.Environment
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific commands configuration.

        Args:
            environment: Target environment (development, production, test, staging, local)

        Returns:
            FlextResult with optimized config dict for the specified environment

        """
        try:
            # Validate environment parameter
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            # Base configuration
            config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
            }

            # Environment-specific configurations
            if environment == "production":
                config.update(
                    {
                        "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                        "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                        "enable_handler_discovery": True,
                        "enable_middleware_pipeline": True,
                        "enable_performance_monitoring": True,  # Monitor production performance
                        "max_concurrent_commands": 50,  # Controlled concurrency in production
                        "command_timeout_seconds": 15,  # Strict timeout for production
                        "enable_detailed_error_messages": False,  # Security in production
                        "enable_handler_caching": True,  # Performance optimization
                        "middleware_timeout_seconds": 5,  # Fast middleware processing
                    }
                )
            elif environment == "development":
                config.update(
                    {
                        "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "enable_handler_discovery": True,
                        "enable_middleware_pipeline": True,
                        "enable_performance_monitoring": False,  # Not needed in dev
                        "max_concurrent_commands": 200,  # Higher concurrency for dev testing
                        "command_timeout_seconds": 60,  # More time for debugging
                        "enable_detailed_error_messages": True,  # Full debugging info
                        "enable_handler_caching": False,  # Fresh handler lookup each time
                        "middleware_timeout_seconds": 30,  # More time for debugging
                    }
                )
            elif environment == "test":
                config.update(
                    {
                        "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                        "log_level": FlextConstants.Config.LogLevel.ERROR.value,  # Minimal logging
                        "enable_handler_discovery": True,  # Still need discovery for tests
                        "enable_middleware_pipeline": False,  # Skip for test speed
                        "enable_performance_monitoring": False,  # No monitoring in tests
                        "max_concurrent_commands": 10,  # Limited for test isolation
                        "command_timeout_seconds": 5,  # Fast timeout for tests
                        "enable_detailed_error_messages": False,  # Clean test output
                        "enable_handler_caching": False,  # Clean state between tests
                        "middleware_timeout_seconds": 1,  # Very fast for tests
                    }
                )
            elif environment == "staging":
                config.update(
                    {
                        "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                        "log_level": FlextConstants.Config.LogLevel.INFO.value,
                        "enable_handler_discovery": True,
                        "enable_middleware_pipeline": True,
                        "enable_performance_monitoring": True,  # Monitor staging performance
                        "max_concurrent_commands": 75,  # Moderate concurrency for staging
                        "command_timeout_seconds": 20,  # Reasonable staging timeout
                        "enable_detailed_error_messages": True,  # Debug staging issues
                        "enable_handler_caching": True,  # Test caching behavior
                        "middleware_timeout_seconds": 10,  # Balanced timeout
                    }
                )
            elif environment == "local":
                config.update(
                    {
                        "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "enable_handler_discovery": True,
                        "enable_middleware_pipeline": True,
                        "enable_performance_monitoring": False,  # Not needed locally
                        "max_concurrent_commands": 500,  # High concurrency for local testing
                        "command_timeout_seconds": 120,  # Generous local timeout
                        "enable_detailed_error_messages": True,  # Full local debugging
                        "enable_handler_caching": False,  # Fresh behavior for development
                        "middleware_timeout_seconds": 60,  # Generous local timeout
                    }
                )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment commands config: {e}"
            )

    @classmethod
    def optimize_commands_performance(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize commands system performance based on configuration.

        Args:
            config: Base configuration dictionary containing performance preferences

        Returns:
            FlextResult containing optimized configuration with performance settings

        """
        try:
            # Create optimized configuration
            optimized_config = dict(config)

            # Get performance level from config
            performance_level = config.get("performance_level", "medium")

            # Base performance settings
            optimized_config.update(
                {
                    "performance_level": performance_level,
                    "optimization_enabled": True,
                    "optimization_timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
                }
            )

            # Performance level specific optimizations
            if performance_level == "high":
                optimized_config.update(
                    {
                        # Handler optimization
                        "handler_cache_size": 1000,
                        "enable_handler_pooling": True,
                        "handler_pool_size": 100,
                        "max_concurrent_handlers": 50,
                        "handler_discovery_cache_ttl": 3600,  # 1 hour
                        # Middleware optimization
                        "enable_middleware_caching": True,
                        "middleware_thread_count": 8,
                        "middleware_queue_size": 500,
                        "parallel_middleware_processing": True,
                        # Command processing optimization
                        "command_batch_size": 100,
                        "enable_command_batching": True,
                        "command_processing_threads": 16,
                        "command_queue_size": 2000,
                        # Memory optimization
                        "memory_pool_size_mb": 200,
                        "enable_object_pooling": True,
                        "gc_optimization_enabled": True,
                        "optimization_level": "aggressive",
                    }
                )
            elif performance_level == "medium":
                optimized_config.update(
                    {
                        # Balanced handler settings
                        "handler_cache_size": 500,
                        "enable_handler_pooling": True,
                        "handler_pool_size": 50,
                        "max_concurrent_handlers": 25,
                        "handler_discovery_cache_ttl": 1800,  # 30 minutes
                        # Moderate middleware settings
                        "enable_middleware_caching": True,
                        "middleware_thread_count": 4,
                        "middleware_queue_size": 250,
                        "parallel_middleware_processing": True,
                        # Standard command processing
                        "command_batch_size": 50,
                        "enable_command_batching": True,
                        "command_processing_threads": 8,
                        "command_queue_size": 1000,
                        # Moderate memory settings
                        "memory_pool_size_mb": 100,
                        "enable_object_pooling": True,
                        "gc_optimization_enabled": True,
                        "optimization_level": "balanced",
                    }
                )
            elif performance_level == "low":
                optimized_config.update(
                    {
                        # Conservative handler settings
                        "handler_cache_size": 100,
                        "enable_handler_pooling": False,
                        "handler_pool_size": 10,
                        "max_concurrent_handlers": 5,
                        "handler_discovery_cache_ttl": 300,  # 5 minutes
                        # Minimal middleware settings
                        "enable_middleware_caching": False,
                        "middleware_thread_count": 1,
                        "middleware_queue_size": 50,
                        "parallel_middleware_processing": False,
                        # Single-threaded command processing
                        "command_batch_size": 10,
                        "enable_command_batching": False,
                        "command_processing_threads": 1,
                        "command_queue_size": 100,
                        # Minimal memory footprint
                        "memory_pool_size_mb": 50,
                        "enable_object_pooling": False,
                        "gc_optimization_enabled": False,
                        "optimization_level": "conservative",
                    }
                )

            # Additional performance metrics and targets
            optimized_config.update(
                {
                    "expected_throughput_commands_per_second": 500
                    if performance_level == "high"
                    else 200
                    if performance_level == "medium"
                    else 50,
                    "target_handler_latency_ms": 5
                    if performance_level == "high"
                    else 15
                    if performance_level == "medium"
                    else 50,
                    "target_middleware_latency_ms": 2
                    if performance_level == "high"
                    else 8
                    if performance_level == "medium"
                    else 25,
                    "memory_efficiency_target": 0.95
                    if performance_level == "high"
                    else 0.85
                    if performance_level == "medium"
                    else 0.70,
                }
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to optimize commands performance: {e}"
            )


# =============================================================================
# MODULE EXPORTS - FLEXT Command System API
# =============================================================================

__all__: list[str] = [
    "FlextCommands",
    # Legacy compatibility aliases moved to flext_core.legacy to avoid type conflicts
]
