"""CQRS command and query processing system.

Provides FlextCommands for implementing Command Query Responsibility
Segregation patterns with type-safe handlers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, Self, TypeVar, cast

from pydantic import model_validator

from flext_core.constants import FlextConstants
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

# Type variables for command system
T = TypeVar("T")
U = TypeVar("U")


class FlextCommands:
    """CQRS Command and Query Processing System."""

    # =========================================================================
    # MODELS - Pydantic base models for Commands and Queries
    # =========================================================================

    class Models:
        """Base models providing default command/query behaviors."""

        class Command(FlextModels.SystemConfigs.CommandModel):
            """Command model with metadata and immutability using Pydantic."""

            # Inherit all configuration from CommandModel
            # Commands are frozen (immutable) by default

            @model_validator(mode="before")
            @classmethod
            def _ensure_command_type(cls, data: object) -> object:
                """Populate command_type based on class name if missing.

                Returns:
                    Modified data with command_type populated.

                """
                if not isinstance(data, dict):
                    return data
                if "command_type" not in data or not data.get("command_type"):
                    name = cls.__name__
                    base = name.removesuffix("Command")
                    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
                    data["command_type"] = re.sub(
                        r"([a-z0-9])([A-Z])", r"\1_\2", s1
                    ).lower()
                return data

            def validate_command(self) -> FlextResult[None]:
                """Validate command data."""
                return FlextResult[None].ok(None)

            @property
            def id(self) -> str:
                """Get command ID (alias for command_id)."""
                return self.command_id

            def get_command_type(self) -> str:
                """Get command type derived from class name.

                Returns:
                    Command type string.

                """
                name = self.__class__.__name__
                base = name.removesuffix("Command")
                s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
                return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

            def to_payload(
                self,
            ) -> (
                FlextModels.Payload[FlextTypes.Core.Dict]
                | FlextResult[FlextModels.Payload[FlextTypes.Core.Dict]]
            ):
                """Convert command to payload."""
                try:
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

                    # Create payload directly using the class
                    return FlextModels.Payload[FlextTypes.Core.Dict](
                        data=data,
                        message_type=self.__class__.__name__,
                        source_service="command_service",
                        timestamp=datetime.now(UTC),
                        correlation_id=self.correlation_id,
                        message_id=FlextUtilities.Generators.generate_uuid(),
                    )
                except Exception as e:
                    return FlextResult[FlextModels.Payload[FlextTypes.Core.Dict]].fail(
                        f"Failed to create payload: {e}",
                    )

            @classmethod
            def from_payload(
                cls: type[Self],
                payload: FlextModels.Payload[FlextTypes.Core.Dict],
            ) -> FlextResult[Self]:
                """Create command from payload.

                Returns:
                    FlextResult containing the command instance.

                """
                try:
                    data = payload.data if hasattr(payload, "data") else None
                    if not isinstance(data, dict):
                        return FlextResult[Self].fail(
                            "FlextModels data is not compatible",
                        )
                    model = cls.model_validate(data)
                    return FlextResult[Self].ok(model)
                except Exception as e:
                    return FlextResult[Self].fail(str(e))

        class Query(FlextModels.SystemConfigs.QueryModel):
            """Query model with metadata and immutability using Pydantic."""

            # Inherit all configuration from QueryModel
            # Queries are frozen (immutable) by default

            @model_validator(mode="before")
            @classmethod
            def _ensure_query_type(cls, data: object) -> object:
                """Populate query_type based on class name if missing.

                Returns:
                    Modified data with query_type populated.

                """
                if not isinstance(data, dict):
                    return data
                if "query_type" not in data or not data.get("query_type"):
                    name = cls.__name__
                    base = name.removesuffix("Query")
                    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
                    data["query_type"] = re.sub(
                        r"([a-z0-9])([A-Z])", r"\1_\2", s1
                    ).lower()
                return data

            @property
            def id(self) -> str:
                """Get query ID (alias for query_id)."""
                return self.query_id

    # =========================================================================
    # HANDLERS - Command and query handler base classes
    # =========================================================================

    class Handlers:
        """Base classes for command and query handlers."""

        class CommandHandler[CommandT, ResultT](
            FlextHandlers.CQRS.CommandHandler[CommandT, ResultT],
            FlextMixins,
        ):
            """Generic base class for command handlers with Pydantic configuration."""

            def __init__(
                self,
                handler_name: str | None = None,
                handler_id: str | None = None,
                handler_config: FlextModels.SystemConfigs.HandlerConfig | None = None,
            ) -> None:
                """Initialize handler with optional Pydantic configuration."""
                super().__init__()
                # Initialize mixins manually since we don't inherit from them
                FlextMixins.initialize_validation(self)
                FlextMixins.start_timing(self)

                self._metrics_state: FlextTypes.Core.Dict | None = None

                # Use HandlerConfig if provided, otherwise create default
                if handler_config:
                    self._config = handler_config
                    self._handler_name = handler_config.handler_name
                    self.handler_id = handler_config.handler_id
                else:
                    # Create default config
                    default_name = handler_name or self.__class__.__name__
                    default_id = handler_id or f"{self.__class__.__name__}_{id(self)}"
                    self._config = FlextModels.SystemConfigs.HandlerConfig(
                        handler_id=default_id,
                        handler_name=default_name,
                        handler_type="command",
                    )
                    self._handler_name = default_name
                    self.handler_id = default_id

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
                """Validate command before handling."""
                # Delegate to command's validation if available
                validate_method = getattr(command, "validate_command", None)
                if callable(validate_method):
                    result = validate_method()
                    if hasattr(result, "success") and hasattr(result, "error"):
                        return cast("FlextResult[None]", result)
                return FlextResult[None].ok(None)

            def handle(self, command: CommandT) -> FlextResult[ResultT]:
                """Handle the command and return result."""
                # Subclasses must implement this method
                msg = "Subclasses must implement handle method"
                raise NotImplementedError(msg)

            def can_handle(self, command_type: object) -> bool:
                """Check if handler can process this command."""
                self.logger.debug(
                    "Checking if handler can process command",
                    command_type_name=getattr(
                        command_type,
                        "__name__",
                        str(command_type),
                    ),
                )

                # Get expected command type from Generic parameter
                orig_bases = getattr(self, "__orig_bases__", None)
                if orig_bases is not None:
                    for base in orig_bases:
                        args = getattr(base, "__args__", None)
                        if args is not None and len(args) >= 1:
                            expected_type = base.__args__[0]
                            # Support being called with instance or type
                            if isinstance(command_type, type):
                                can_handle_result = issubclass(
                                    command_type, expected_type
                                )
                            else:
                                can_handle_result = isinstance(
                                    command_type, expected_type
                                )

                            self.logger.debug(
                                "Handler check result",
                                can_handle=can_handle_result,
                                expected_type=getattr(
                                    expected_type,
                                    "__name__",
                                    str(expected_type),
                                ),
                            )
                            return bool(can_handle_result)

                self.log_info(self, "Could not determine handler type constraints")
                return True

            def execute(self, command: CommandT) -> FlextResult[ResultT]:
                """Execute command with full validation and error handling."""
                self.log_info(
                    self,
                    "Executing command",
                    command_type=type(command).__name__,
                    command_id=getattr(command, "command_id", "unknown"),
                )

                # Validate command can be handled (pass type, not instance)
                if not self.can_handle(type(command)):
                    error_msg = (
                        f"{self._handler_name} cannot handle {type(command).__name__}"
                    )
                    self.log_error(self, error_msg)
                    return FlextResult[ResultT].fail(
                        error_msg,
                        error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
                    )

                # Validate the command's data
                validation_result = self.validate_command(command)
                if validation_result.is_failure:
                    self.log_info(
                        self,
                        "Command validation failed",
                        command_type=type(command).__name__,
                        error=validation_result.error,
                    )
                    return FlextResult[ResultT].fail(
                        validation_result.error or "Validation failed",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Use mixin timing capabilities
                self.start_timing(self)

                try:
                    # Log operation using mixin method
                    self.log_operation(
                        self,
                        "handle_command",
                        command_type=type(command).__name__,
                        command_id=getattr(
                            command,
                            "command_id",
                            getattr(command, "id", "unknown"),
                        ),
                    )

                    result: FlextResult[ResultT] = self.handle(command)

                    # Stop timing and get elapsed time
                    elapsed = self.stop_timing(self)
                    execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

                    self.log_info(
                        self,
                        "Command executed successfully",
                        command_type=type(command).__name__,
                        execution_time_ms=execution_time_ms,
                        success=result.is_success,
                    )

                    return result

                except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                    # Get timing using mixin method
                    elapsed = self.stop_timing(self)
                    execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

                    self.log_error(
                        self,
                        f"Command execution failed: {e}",
                        command_type=type(command).__name__,
                        execution_time_ms=execution_time_ms,
                    )
                    return FlextResult[ResultT].fail(
                        f"Command processing failed: {e}",
                        error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
                    )

            def handle_command(self, command: CommandT) -> FlextResult[ResultT]:
                """Delegate CQRS handle_command."""
                return self.execute(command)

        class QueryHandler[QueryT, QueryResultT](
            FlextHandlers.CQRS.QueryHandler[QueryT, QueryResultT],
            FlextMixins,
        ):
            """Generic base class for query handlers with Pydantic configuration."""

            def __init__(
                self,
                handler_name: str | None = None,
                handler_id: str | None = None,
                handler_config: FlextModels.SystemConfigs.HandlerConfig | None = None,
            ) -> None:
                """Initialize query handler with optional Pydantic configuration."""
                super().__init__()
                # Initialize mixins manually since we don't inherit from them
                FlextMixins.initialize_validation(self)
                FlextMixins.start_timing(self)
                FlextMixins.clear_cache(self)

                # Use HandlerConfig if provided, otherwise create default
                if handler_config:
                    self._config = handler_config
                    self._handler_name = handler_config.handler_name
                    self.handler_id = handler_config.handler_id
                else:
                    # Create default config
                    default_name = handler_name or self.__class__.__name__
                    default_id = handler_id or f"{self.__class__.__name__}_{id(self)}"
                    self._config = FlextModels.SystemConfigs.HandlerConfig(
                        handler_id=default_id,
                        handler_name=default_name,
                        handler_type="query",
                    )
                    self._handler_name = default_name
                    self.handler_id = default_id

            @property
            def handler_name(self) -> str:
                """Get handler name for identification."""
                return self._handler_name

            def can_handle(self, query: QueryT) -> bool:
                """Check if handler can process this query."""
                # Generic implementation - override in subclasses for specific logic
                _ = query
                return True

            def validate_query(self, query: QueryT) -> FlextResult[None]:
                """Validate query."""
                validate_method = getattr(query, "validate_query", None)
                if callable(validate_method):
                    result = validate_method()
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[None]", result)
                return FlextResult[None].ok(None)

            def handle(self, query: QueryT) -> FlextResult[QueryResultT]:
                """Handle query and return result."""
                # Subclasses should implement this method
                msg = "Subclasses must implement handle method"
                raise NotImplementedError(msg)

            def handle_query(self, query: QueryT) -> FlextResult[QueryResultT]:
                """Delegate CQRS handle_query."""
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
        FlextMixins,
    ):
        """Command bus for routing and executing commands with Pydantic configuration."""

        def __init__(
            self,
            bus_config: FlextModels.SystemConfigs.BusConfig | None = None,
        ) -> None:
            """Initialize command bus with optional Pydantic configuration."""
            super().__init__()
            # Initialize mixins manually since we don't inherit from them
            FlextMixins.initialize_validation(self)
            FlextMixins.clear_cache(self)
            # Timestampable mixin initialization
            self._created_at = datetime.now(UTC)
            FlextMixins.start_timing(self)

            # Use BusConfig if provided, otherwise create default
            if bus_config:
                self._config = bus_config
            else:
                self._config = FlextModels.SystemConfigs.BusConfig()

            # Handlers registry: command type -> handler instance
            self._handlers: FlextTypes.Core.Dict = {}
            # Middleware pipeline (controlled by config)
            self._middleware: list[FlextModels.SystemConfigs.MiddlewareConfig] = []
            # Middleware instances cache
            self._middleware_instances: FlextTypes.Core.Dict = {}
            # Execution counter
            self._execution_count: int = 0
            # Underlying FlextCommands CQRS bus for direct registrations
            self._fb_bus = FlextHandlers.CQRS.CommandBus()
            # Auto-discovery handlers (single-arg registration)
            self._auto_handlers: FlextTypes.Core.List = []

        def register_handler(self, *args: object) -> None:
            """Register command handler."""
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
                        self,
                        "Handler already registered",
                        command_type=str(key),
                        existing_handler=self._handlers[key].__class__.__name__,
                    )
                    return

                self._handlers[key] = handler
                self._auto_handlers.append(handler)
                self.log_info(
                    self,
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
                _ = self._fb_bus.register(cast("type", command_type_obj), handler)
                self.log_info(
                    self,
                    "Handler registered for command type",
                    command_type=key,
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return

            msg = "register_handler() takes 1 or 2 positional arguments"
            raise TypeError(msg)

        def find_handler(self, command: object) -> object | None:
            """Find handler for command."""
            # Search auto-registered handlers first (single-arg form)
            for handler in self._auto_handlers:
                can_handle_method = getattr(handler, "can_handle", None)
                if callable(can_handle_method) and can_handle_method(type(command)):
                    return handler
            return None

        def execute(self, command: object) -> FlextResult[object]:
            """Execute command through registered handler."""
            # Check if bus is enabled
            if not self._config.enable_middleware and self._middleware:
                return FlextResult[object].fail(
                    "Middleware pipeline is disabled but middleware is configured",
                    error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
                )

            self._execution_count = int(self._execution_count) + 1
            command_type = type(command)

            # Check cache for query results if this is a query (and if metrics are enabled)
            if self._config.enable_metrics and (
                hasattr(command, "query_id") or "Query" in command_type.__name__
            ):
                cache_key = f"{command_type.__name__}_{hash(str(command))}"
                cached_result = self.get_cached_value(self, cache_key)
                if cached_result is not None:
                    self.log_info(
                        self,
                        "Returning cached query result",
                        command_type=command_type.__name__,
                        cache_key=cache_key,
                    )
                    return cast("FlextResult[object]", cached_result)

            self.log_operation(
                self,
                "execute_command",
                command_type=command_type.__name__,
                command_id=getattr(
                    command,
                    "command_id",
                    getattr(command, "id", "unknown"),
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
                    self,
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
            self.start_timing(self)
            result = self._execute_handler(handler, command)
            elapsed = self.stop_timing(self)

            # Cache successful query results
            if result.is_success and (
                hasattr(command, "query_id") or "Query" in command_type.__name__
            ):
                cache_key = f"{command_type.__name__}_{hash(str(command))}"
                self.set_cached_value(self, cache_key, result)
                self.log_debug(
                    self,
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
            """Apply middleware pipeline using Pydantic configs."""
            if not self._config.enable_middleware:
                return FlextResult[None].ok(None)

            # Sort middleware by order
            sorted_middleware = sorted(self._middleware, key=lambda m: m.order)

            for middleware_config in sorted_middleware:
                if not middleware_config.enabled:
                    continue

                # Get actual middleware instance

                middleware = self._middleware_instances.get(
                    middleware_config.middleware_id
                )
                if middleware is None:
                    # Skip middleware configs without instances
                    continue

                self.log_debug(
                    self,
                    "Applying middleware",
                    middleware_id=middleware_config.middleware_id,
                    middleware_type=middleware_config.middleware_type,
                    order=middleware_config.order,
                )

                process_method = getattr(middleware, "process", None)
                if callable(process_method):
                    result = process_method(command, handler)
                    if isinstance(result, FlextResult) and result.is_failure:
                        self.log_info(
                            self,
                            "Middleware rejected command",
                            middleware_type=middleware_config.middleware_type,
                            error=result.error or "Unknown error",
                        )
                        return FlextResult[None].fail(
                            str(result.error or "Middleware rejected command"),
                        )

            return FlextResult[None].ok(None)

        def _execute_handler(
            self,
            handler: object,
            command: object,
        ) -> FlextResult[object]:
            """Execute command through handler."""
            self.log_debug(
                self,
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

        def add_middleware(
            self,
            middleware: object,
            middleware_config: FlextModels.SystemConfigs.MiddlewareConfig | None = None,
        ) -> None:
            """Add middleware to pipeline with optional Pydantic configuration."""
            if not self._config.enable_middleware:
                # Middleware pipeline is disabled, skip adding
                return

            # Create config if not provided
            if middleware_config is None:
                middleware_config = FlextModels.SystemConfigs.MiddlewareConfig(
                    middleware_id=f"mw_{len(self._middleware)}",
                    middleware_type=type(middleware).__name__,
                    enabled=True,
                    order=len(self._middleware),
                )

            # Store both middleware and config
            self._middleware.append(middleware_config)
            # Also store the actual middleware instance
            self._middleware_instances[middleware_config.middleware_id] = middleware

            self.log_info(
                self,
                "Middleware added to pipeline",
                middleware_type=middleware_config.middleware_type,
                middleware_id=middleware_config.middleware_id,
                total_middleware=len(self._middleware),
            )

        def get_all_handlers(self) -> FlextTypes.Core.List:
            """Get all registered handlers."""
            return list(self._handlers.values())

        def unregister_handler(self, command_type: str) -> bool:
            """Unregister command handler."""
            for key in list(self._handlers.keys()):
                key_name = getattr(key, "__name__", None)
                if (key_name is not None and key_name == command_type) or str(
                    key,
                ) == command_type:
                    del self._handlers[key]
                    self.log_info(
                        self,
                        "Handler unregistered",
                        command_type=command_type,
                        remaining_handlers=len(self._handlers),
                    )
                    return True
            return False

        def send_command(self, command: object) -> FlextResult[object]:
            """Send command for processing."""
            return self.execute(command)

        def get_registered_handlers(self) -> FlextTypes.Core.Dict:
            """Get registered handlers as dictionary."""
            return {str(k): v for k, v in self._handlers.items()}

    # =========================================================================
    # DECORATORS - Command handling decorators and utilities
    # =========================================================================

    class Decorators:
        """Decorators for function-based command handlers."""

        @staticmethod
        def command_handler[TCmd, TResult](
            command_type: type[TCmd],
        ) -> Callable[[Callable[[TCmd], TResult]], Callable[[TCmd], TResult]]:
            """Mark function as command handler."""

            def decorator(
                func: Callable[[TCmd], TResult],
            ) -> Callable[[TCmd], TResult]:
                # Create handler class from function
                class FunctionHandler(
                    FlextCommands.Handlers.CommandHandler[TCmd, TResult],
                ):
                    def handle(self, command: TCmd) -> FlextResult[TResult]:
                        result = func(command)
                        if isinstance(result, FlextResult):
                            return cast("FlextResult[TResult]", result)
                        return FlextResult[TResult].ok(result)

                # Create wrapper function with metadata
                def wrapper(command: TCmd) -> TResult:
                    return func(command)

                # Preserve the original function's return type
                wrapper.__annotations__ = func.__annotations__

                # Store metadata in wrapper's __dict__ for type safety
                wrapper.__dict__["command_type"] = command_type
                wrapper.__dict__["handler_instance"] = FunctionHandler()

                return wrapper

            return decorator

    # =========================================================================
    # RESULTS - Result helper methods for FlextResult patterns
    # =========================================================================

    class Results:
        """Factory methods for creating FlextResult instances."""

        @staticmethod
        def success(data: object) -> FlextResult[object]:
            """Create successful result."""
            return FlextResult[object].ok(data)

        @staticmethod
        def failure(
            error: str,
            error_code: str | None = None,
            error_data: FlextTypes.Core.Dict | None = None,
        ) -> FlextResult[object]:
            """Create failure result."""
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
        """Factory methods for creating CQRS components."""

        @staticmethod
        def create_command_bus() -> FlextCommands.Bus:
            """Create a new command bus instance."""
            return FlextCommands.Bus()

        @staticmethod
        def create_simple_handler(
            handler_func: FlextTypes.Core.OperationCallable,
        ) -> FlextCommands.Handlers.CommandHandler[object, object]:
            """Create handler from function."""

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
            """Create query handler from function."""

            class SimpleQueryHandler(
                FlextCommands.Handlers.QueryHandler[object, object],
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
        cls,
        config: FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict,
    ) -> FlextResult[FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict]:
        """Configure commands system with Pydantic model or legacy dict."""
        try:
            # Handle dict input with deprecation warning
            if isinstance(config, dict):
                warnings.warn(
                    "Using dict for configure_commands_system is deprecated. Use FlextModels.SystemConfigs.CommandsConfig instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Convert dict to Pydantic model, validate, then convert back to dict
                model = FlextModels.SystemConfigs.CommandsConfig.model_validate(config)
                validated_dict = model.model_dump()
                # Return validated dict with proper type casting
                return FlextResult[
                    FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
                ].ok(validated_dict)

            # Pydantic model input - return as-is
            return FlextResult[
                FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
            ].ok(config)
        except Exception as e:
            return FlextResult[
                FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
            ].fail(
                f"Failed to configure commands system: {e}",
            )

    @classmethod
    def get_commands_system_config(
        cls,
        *,
        return_model: bool = False,
    ) -> FlextResult[FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict]:
        """Get current commands system configuration.

        Args:
            return_model: If True, return Pydantic model. If False, return dict for backward compatibility.

        """
        try:
            # Create default configuration model
            config = FlextModels.SystemConfigs.CommandsConfig()

            if return_model:
                return FlextResult[
                    FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
                ].ok(config)
            # Return dict for backward compatibility
            config_dict = config.model_dump()
            return FlextResult[
                FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
            ].ok(config_dict)
        except Exception as e:
            return FlextResult[
                FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
            ].fail(
                f"Failed to get commands system config: {e}",
            )

    class _EnvironmentConfigFactory:
        """Factory for environment-specific configurations."""

        @staticmethod
        def _get_base_config(
            environment: FlextTypes.Config.Environment,
        ) -> FlextTypes.Config.ConfigDict:
            """Get base configuration."""
            return {
                "environment": environment,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                "enable_handler_discovery": True,
            }

        @staticmethod
        def _get_environment_strategies() -> dict[str, FlextTypes.Config.ConfigDict]:
            """Get environment strategies."""
            return {
                "production": {
                    "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "enable_middleware_pipeline": True,
                    "enable_performance_monitoring": True,
                    "max_concurrent_commands": 50,
                    "command_timeout_seconds": 15,
                    "enable_detailed_error_messages": False,
                    "enable_handler_caching": True,
                    "middleware_timeout_seconds": 5,
                },
                "development": {
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_middleware_pipeline": True,
                    "enable_performance_monitoring": False,
                    "max_concurrent_commands": 200,
                    "command_timeout_seconds": 60,
                    "enable_detailed_error_messages": True,
                    "enable_handler_caching": False,
                    "middleware_timeout_seconds": 30,
                },
                "test": {
                    "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "log_level": FlextConstants.Config.LogLevel.ERROR.value,
                    "enable_middleware_pipeline": False,
                    "enable_performance_monitoring": False,
                    "max_concurrent_commands": 10,
                    "command_timeout_seconds": 5,
                    "enable_detailed_error_messages": False,
                    "enable_handler_caching": False,
                    "middleware_timeout_seconds": 1,
                },
                "staging": {
                    "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "enable_middleware_pipeline": True,
                    "enable_performance_monitoring": True,
                    "max_concurrent_commands": 75,
                    "command_timeout_seconds": 20,
                    "enable_detailed_error_messages": True,
                    "enable_handler_caching": True,
                    "middleware_timeout_seconds": 10,
                },
                "local": {
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_middleware_pipeline": True,
                    "enable_performance_monitoring": False,
                    "max_concurrent_commands": 500,
                    "command_timeout_seconds": 120,
                    "enable_detailed_error_messages": True,
                    "enable_handler_caching": False,
                    "middleware_timeout_seconds": 60,
                },
            }

        @classmethod
        def create_environment_config(
            cls,
            environment: FlextTypes.Config.Environment,
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Create environment configuration."""
            try:
                # Validate environment
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if environment not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{environment}'. Valid options: {valid_environments}",
                    )

                # Use Strategy Pattern to get environment-specific config
                strategies = cls._get_environment_strategies()
                base_config = cls._get_base_config(environment)
                environment_overrides = strategies.get(
                    environment,
                    strategies["production"],
                )

                # Compose final configuration
                config = {**base_config, **environment_overrides}
                return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

            except Exception as e:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Failed to create environment config: {e}",
                )

    @classmethod
    def create_environment_commands_config(
        cls,
        environment: str,
        *,
        return_model: bool = False,
    ) -> FlextResult[FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict]:
        """Create environment-specific commands configuration."""
        try:
            # Validate environment is valid before proceeding
            valid_environments = {
                "development",
                "production",
                "staging",
                "test",
                "local",
            }
            if environment not in valid_environments:
                error_msg = f"Invalid environment '{environment}'. Must be one of: {', '.join(sorted(valid_environments))}"
                raise ValueError(error_msg)

            # Use environment factory to get proper configuration for environment
            # Cast to proper literal type for mypy
            env_literal = cast(
                "Literal['development', 'production', 'staging', 'test', 'local']",
                environment,
            )
            config_result = cls._EnvironmentConfigFactory.create_environment_config(
                env_literal
            )
            if config_result.is_failure:
                return FlextResult[
                    FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
                ].fail(
                    config_result.error or "Failed to create environment config",
                )

            # Create CommandsConfig model from the factory result
            config_dict = config_result.value
            config = FlextModels.SystemConfigs.CommandsConfig.model_validate(
                config_dict
            )

            if return_model:
                return FlextResult[
                    FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
                ].ok(config)
            # Return the dict from factory for backward compatibility (default behavior)
            # Cast to expected type for mypy invariance
            typed_config_dict = cast("FlextTypes.Core.Dict", config_dict)
            return FlextResult[
                FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
            ].ok(typed_config_dict)
        except Exception as e:
            return FlextResult[
                FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
            ].fail(
                f"Failed to create environment commands config: {e}",
            )

    # =========================================================================
    # PERFORMANCE OPTIMIZATION - Progressive migration with dual signatures
    # =========================================================================

    @classmethod
    def optimize_commands_performance(
        cls,
        config: FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict,
        performance_level: Literal["low", "medium", "high", "extreme"] = "medium",
    ) -> FlextResult[FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict]:
        """Optimize commands system performance."""
        # Handle dict input with deprecation warning
        if isinstance(config, dict):
            warnings.warn(
                "Using dict for optimize_commands_performance is deprecated. Use FlextModels.SystemConfigs.CommandsConfig instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Call dict-specific method and return dict
            dict_result = cls._optimize_commands_performance_as_dict(
                cast(
                    "dict[str, str | int | float | bool | FlextTypes.Core.List | FlextTypes.Core.Dict]",
                    config,
                ),
                performance_level,
            )
            if dict_result.is_success:
                return FlextResult[
                    FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
                ].ok(cast("FlextTypes.Core.Dict", dict_result.value))
            return FlextResult[
                FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
            ].fail(dict_result.error or "Optimization failed")

        # Pydantic model input - use model method
        model_result = cls._optimize_commands_performance_as_model(
            config, performance_level
        )
        if model_result.is_success:
            return FlextResult[
                FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
            ].ok(model_result.value)
        return FlextResult[
            FlextModels.SystemConfigs.CommandsConfig | FlextTypes.Core.Dict
        ].fail(model_result.error or "Model optimization failed")

    @classmethod
    def _optimize_commands_performance_as_model(
        cls,
        config: FlextModels.SystemConfigs.CommandsConfig,
        performance_level: Literal["low", "medium", "high", "extreme"],
    ) -> FlextResult[FlextModels.SystemConfigs.CommandsConfig]:
        """Optimize commands performance using Pydantic model."""
        try:
            # Create performance-optimized configuration
            performance_config = FlextUtilities.Performance.create_performance_config(
                performance_level,
            )

            # Update model with performance optimizations
            optimized_data = config.model_dump()
            optimized_data.update(performance_config)

            # Create and validate optimized model
            optimized_model = FlextModels.SystemConfigs.CommandsConfig.model_validate(
                optimized_data
            )

            return FlextResult[FlextModels.SystemConfigs.CommandsConfig].ok(
                optimized_model
            )
        except Exception as e:
            return FlextResult[FlextModels.SystemConfigs.CommandsConfig].fail(
                f"Failed to optimize commands performance: {e}",
                error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
            )

    @classmethod
    def _optimize_commands_performance_as_dict(
        cls,
        config: FlextTypes.Config.ConfigDict,
        performance_level: Literal["low", "medium", "high", "extreme"],
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize commands performance using dict (legacy)."""
        try:
            # Create performance-optimized configuration
            performance_config = FlextUtilities.Performance.create_performance_config(
                performance_level,
            )

            # Merge with base config
            optimized_config = {**config, **performance_config}

            # Validate via model but return dict
            _ = FlextModels.SystemConfigs.CommandsConfig.model_validate(
                optimized_config
            )

            # Filter out None values to match expected type
            filtered_config = {
                k: v for k, v in optimized_config.items() if v is not None
            }
            return FlextResult[FlextTypes.Config.ConfigDict].ok(filtered_config)
        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to optimize commands performance: {e}",
                error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
            )


__all__: FlextTypes.Core.StringList = [
    "FlextCommands",
]
