"""CQRS Command and Query Processing System for FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from pydantic import ConfigDict

from flext_core.constants import FlextConstants
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextCommands:
    """CQRS Command and Query Processing System."""

    # =========================================================================
    # MODELS - Pydantic base models for Commands and Queries
    # =========================================================================

    class Models:
        """Base models providing default command/query behaviors.

        Implements default metadata, immutability and payload helpers
        without forcing inheritance in tests that use Pydantic directly.
        """

        class Command(FlextModels.Config):
            """Command model.

            Implements default metadata, immutability and payload helpers
            without forcing inheritance in tests that use Pydantic directly.
            """

            model_config = ConfigDict(
                # Validation settings (inherited from Config)
                validate_assignment=True,
                validate_default=True,
                use_enum_values=True,
                # JSON settings (inherited from Config)
                arbitrary_types_allowed=True,
                ser_json_bytes="base64",
                ser_json_timedelta="iso8601",
                # Command-specific overrides
                frozen=True,
                extra="ignore",
            )

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
                FlextModels.Payload[dict[str, object]]
                | FlextResult[FlextModels.Payload[dict[str, object]]]
            ):
                """Convert command to payload using direct class instantiation."""
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
                    return FlextModels.Payload[dict[str, object]](
                        data=data,
                        message_type=self.__class__.__name__,
                        source_service="command_service",
                        timestamp=datetime.now(UTC),
                        correlation_id=self.correlation_id,
                        message_id=FlextUtilities.Generators.generate_uuid(),
                    )
                except Exception as e:
                    return FlextResult[FlextModels.Payload[dict[str, object]]].fail(
                        f"Failed to create payload: {e}",
                    )

            @classmethod
            def from_payload(
                cls: type[FlextModels.Config],
                payload: FlextModels.Payload[dict[str, object]],
            ) -> FlextResult[FlextModels.Config]:
                try:
                    data = payload.data if hasattr(payload, "data") else None
                    if not isinstance(data, dict):
                        return FlextResult[FlextModels.Config].fail(
                            "FlextModels data is not compatible",
                        )
                    model = cls.model_validate(data)
                    return FlextResult[FlextModels.Config].ok(model)
                except Exception as e:
                    return FlextResult[FlextModels.Config].fail(str(e))

        class Query(FlextModels.Config):
            """Query model.

            Implements default metadata, immutability and payload helpers
            without forcing inheritance in tests that use Pydantic directly.
            """

            model_config = ConfigDict(
                # Validation settings (inherited from Config)
                validate_assignment=True,
                validate_default=True,
                use_enum_values=True,
                # JSON settings (inherited from Config)
                arbitrary_types_allowed=True,
                ser_json_bytes="base64",
                ser_json_timedelta="iso8601",
                # Query-specific overrides
                frozen=True,
                extra="ignore",
            )

            def validate_query(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

    # =========================================================================
    # HANDLERS - Command and query handler base classes
    # =========================================================================

    class Handlers:
        """Base classes for command and query handlers."""

        class CommandHandler[CommandT, ResultT](
            FlextHandlers.CQRS.CommandHandler[CommandT, ResultT],
            FlextMixins,
        ):
            """Generic base class for command handlers."""

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
                # Initialize mixins manually since we don't inherit from them
                FlextMixins.initialize_validation(self)
                FlextMixins.start_timing(self)

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
                """Validate command before handling."""
                # Delegate to command's validation if available
                validate_method = getattr(command, "validate_command", None)
                if callable(validate_method):
                    result = validate_method()
                    if hasattr(result, "success") and hasattr(result, "error"):
                        return cast("FlextResult[None]", result)
                return FlextResult[None].ok(None)

            def handle(self, command: CommandT) -> FlextResult[ResultT]:
                """Handle the command and return result.sing."""
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
                            # Use direct isinstance for validation
                            can_handle_result = isinstance(command_type, expected_type)

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

                # Validate command can be handled
                if not self.can_handle(command):
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
                """Delegate CQRS handle_command to this handler's execute pipeline."""
                return self.execute(command)

        class QueryHandler[QueryT, QueryResultT](
            FlextHandlers.CQRS.QueryHandler[QueryT, QueryResultT],
            FlextMixins,
        ):
            """Generic base class for query handlers."""

            def __init__(self, handler_name: str | None = None) -> None:
                """Initialize query handler with mixins.

                Args:
                    handler_name: Human-readable handler name

                """
                super().__init__()
                # Initialize mixins manually since we don't inherit from them
                FlextMixins.initialize_validation(self)
                FlextMixins.start_timing(self)
                FlextMixins.clear_cache(self)

                self._handler_name = handler_name or self.__class__.__name__

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
                """Validate query using its own validation method."""
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
        FlextMixins,
    ):
        """Command bus for routing and executing commands."""

        def __init__(self) -> None:
            """Initialize command bus with mixin support and CQRS adapter."""
            super().__init__()
            # Initialize mixins manually since we don't inherit from them
            FlextMixins.initialize_validation(self)
            FlextMixins.clear_cache(self)
            # Timestampable mixin initialization
            self._created_at = datetime.now(UTC)
            FlextMixins.start_timing(self)

            # Handlers registry: command type -> handler instance
            self._handlers: dict[str, object] = {}
            # Middleware pipeline
            self._middleware: list[object] = []
            # Execution counter
            self._execution_count: int = 0
            # Underlying FlextCommands CQRS bus for direct registrations
            self._fb_bus = FlextHandlers.CQRS.CommandBus()
            # Auto-discovery handlers (single-arg registration)
            self._auto_handlers: list[object] = []

        def register_handler(self, *args: object) -> None:
            """Register command handler with flexible signature support."""
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
            """Find handler capable of processing the given command."""
            # Search auto-registered handlers first (single-arg form)
            for handler in self._auto_handlers:
                can_handle_method = getattr(handler, "can_handle", None)
                if callable(can_handle_method) and can_handle_method(command):
                    return handler
            return None

        def execute(self, command: object) -> FlextResult[object]:
            """Execute command through registered handler with middleware."""
            self._execution_count = int(self._execution_count) + 1
            command_type = type(command)

            # Check cache for query results if this is a query
            if hasattr(command, "query_id") or "Query" in command_type.__name__:
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
            """Apply middleware pipeline to command processing."""
            for i, middleware in enumerate(self._middleware):
                self.log_debug(
                    self,
                    "Applying middleware",
                    middleware_index=i,
                    middleware_type=type(middleware).__name__,
                )

                process_method = getattr(middleware, "process", None)
                if callable(process_method):
                    result = process_method(command, handler)
                    if isinstance(result, FlextResult) and result.is_failure:
                        self.log_info(
                            self,
                            "Middleware rejected command",
                            middleware_type=type(middleware).__name__,
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
            """Execute command through handler with error handling."""
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

        def add_middleware(self, middleware: object) -> None:
            """Add middleware to the processing pipeline."""
            self._middleware.append(middleware)
            self.log_info(
                self,
                "Middleware added to pipeline",
                middleware_type=type(middleware).__name__,
                total_middleware=len(self._middleware),
            )

        def get_all_handlers(self) -> list[object]:
            """Get all registered handlers for inspection."""
            return list(self._handlers.values())

        def unregister_handler(self, command_type: str) -> bool:
            """Unregister command handler by command type."""
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
            """Send command for processing (alias for execute)."""
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
        """Decorators for function-based command handlers."""

        @staticmethod
        def command_handler(
            command_type: type[object],
        ) -> Callable[[Callable[[object], object]], Callable[[object], object]]:
            """Mark function as command handler with automatic registration."""

            def decorator(
                func: Callable[[object], object],
            ) -> Callable[[object], object]:
                # Create handler class from function
                class FunctionHandler(
                    FlextCommands.Handlers.CommandHandler[object, object],
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
        """Type aliases for command system components."""

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
        """Factory methods for creating FlextResult instances."""

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
        """Factory methods for creating CQRS components."""

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
            """Create handler from function with automatic FlextResult wrapping."""

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
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure commands system with StrEnum validation."""
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
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}",
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
                        f"Invalid validation_level '{val_level}'. Valid options: {valid_levels}",
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
                        f"Invalid log_level '{log_level}'. Valid options: {valid_log_levels}",
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
                f"Failed to configure commands system: {e}",
            )

    @classmethod
    def get_commands_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current commands system configuration with runtime info."""
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
                f"Failed to get commands system config: {e}",
            )

    class _EnvironmentConfigFactory:
        """Factory for creating environment-specific configurations using Strategy Pattern."""

        @staticmethod
        def _get_base_config(
            environment: FlextTypes.Config.Environment,
        ) -> FlextTypes.Config.ConfigDict:
            """Get base configuration shared across environments."""
            return {
                "environment": environment,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                "enable_handler_discovery": True,
            }

        @staticmethod
        def _get_environment_strategies() -> dict[str, FlextTypes.Config.ConfigDict]:
            """Get environment-specific configuration strategies."""
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
            """Create environment configuration using composition and strategy patterns."""
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
        environment: FlextTypes.Config.Environment,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific commands configuration using Factory Pattern."""
        return cls._EnvironmentConfigFactory.create_environment_config(environment)

    @classmethod
    def optimize_commands_performance(
        cls,
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize commands system performance based on configuration."""
        try:
            # Get performance level from config
            performance_level = config.get("performance_level", "medium")

            # Ensure performance_level is a string
            if not isinstance(performance_level, str):
                performance_level = "medium"

            # Create performance-optimized configuration using centralized utility
            performance_config = FlextUtilities.Performance.create_performance_config(
                performance_level,
            )

            # Merge with base config, performance config takes precedence
            optimized_config = {**config, **performance_config}

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to optimize commands performance: {e}",
            )


__all__: list[str] = [
    "FlextCommands",
]
