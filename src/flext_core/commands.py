"""CQRS command and query processing system.

Provides FlextCommands for implementing Command Query Responsibility
Segregation patterns with type-safe handlers.

For verified CQRS patterns and examples, see docs/ACTUAL_CAPABILITIES.md

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Self, cast, get_origin

from pydantic import model_validator

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextCommands:
    """CQRS Command and Query Processing System.

    # Request processing system with CQRS patterns
    # - handlers.py (with CQRS handlers)
    # - processors.py (with processing pipelines)
    # - services.py (with service processors)
    # Now commands.py adds ANOTHER layer of the same thing.
    """

    # =========================================================================
    # MODELS - Pydantic base models for Commands and Queries
    # =========================================================================

    class Models:
        """Base models providing default command/query behaviors."""

        class Command(FlextModels.Command):
            """Command model with metadata and immutability using Pydantic."""

            # Inherit all configuration from CommandModel
            # Commands are frozen (immutable) by default

            # Regex pattern to derive command_type from class name
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

            def validate_command(self) -> FlextResult[bool]:
                """Validate command data."""
                return FlextResult[bool].ok(data=True)

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
                payload: FlextModels.Payload[FlextTypes.Core.Dict]
                | FlextTypes.Core.Dict
                | None,
            ) -> FlextResult[FlextCommands.Models.Command]:
                """Create command from payload.

                Args:
                    payload: Either a FlextModels.Payload object or a dictionary.

                Returns:
                    FlextResult containing the command instance.

                """
                try:
                    # Handle different payload types
                    data: FlextTypes.Core.Dict | None = None

                    if payload is None:
                        return FlextResult[FlextCommands.Models.Command].fail(
                            "Payload cannot be None",
                        )

                    # If it's a dictionary, use it directly
                    if isinstance(payload, dict):
                        data = payload
                    elif hasattr(payload, "data"):
                        # payload is a Payload object
                        payload_data = getattr(payload, "data", None)
                        if isinstance(payload_data, dict):
                            data = payload_data

                    if data is None or not isinstance(data, dict):
                        return FlextResult[FlextCommands.Models.Command].fail(
                            "FlextModels data is not compatible",
                        )

                    model = cls.model_validate(data)
                    return FlextResult[FlextCommands.Models.Command].ok(model)
                except Exception as e:
                    return FlextResult[FlextCommands.Models.Command].fail(str(e))

        class Query(FlextModels.Query):
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

            @classmethod
            def from_payload(
                cls: type[Self],
                payload: FlextModels.Payload[FlextTypes.Core.Dict] | dict[str, object],
            ) -> FlextResult[FlextCommands.Models.Query]:
                """Create query from payload.

                Returns:
                    FlextResult containing the query instance.

                """
                try:
                    # Handle different payload types
                    data: dict[str, object] | None = None

                    if isinstance(payload, dict):
                        data = payload
                    elif hasattr(payload, "data"):
                        # payload is a Payload object
                        payload_data = getattr(payload, "data", None)
                        if isinstance(payload_data, dict):
                            data = payload_data

                    if not isinstance(data, dict):
                        return FlextResult[FlextCommands.Models.Query].fail(
                            "FlextModels data is not compatible",
                        )

                    model = cls.model_validate(data)
                    return FlextResult[FlextCommands.Models.Query].ok(model)
                except Exception as e:
                    return FlextResult[FlextCommands.Models.Query].fail(str(e))

            def to_payload(
                self,
            ) -> FlextModels.Payload[FlextTypes.Core.Dict]:
                """Convert query to payload."""
                try:
                    data = {
                        k: v
                        for k, v in self.model_dump().items()
                        if k
                        not in {
                            "id",
                            "query_id",
                            "created_at",
                            "correlation_id",
                            "user_id",
                        }
                    }

                    # Create payload directly using the class
                    return FlextModels.Payload[FlextTypes.Core.Dict](
                        data=data,
                        message_type=self.__class__.__name__,
                        source_service="query_service",
                        timestamp=datetime.now(UTC),
                        correlation_id=FlextUtilities.Generators.generate_correlation_id(),
                        message_id=FlextUtilities.Generators.generate_uuid(),
                    )
                except Exception as e:
                    # In case of error, return payload with error in metadata
                    return FlextModels.Payload[FlextTypes.Core.Dict](
                        data={},
                        message_type=self.__class__.__name__,
                        source_service="query_service",
                        timestamp=datetime.now(UTC),
                        correlation_id=FlextUtilities.Generators.generate_correlation_id(),
                        message_id=FlextUtilities.Generators.generate_uuid(),
                        metadata={"error": str(e)},
                    )

    # =========================================================================
    # HANDLERS - Command and query handler base classes
    # =========================================================================

    class Handlers:
        """Base classes for command and query handlers."""

        class CommandHandler[CommandT, ResultT](
            # FlextProcessing.CQRS.CommandHandler[CommandT, ResultT],  # Temporarily disabled
            FlextMixins,
        ):
            """Generic base class for command handlers with Pydantic configuration."""

            def __init__(
                self,
                handler_name: str | None = None,
                handler_id: str | None = None,
                handler_config: dict[str, object] | None = None,
            ) -> None:
                """Initialize handler with optional Pydantic configuration."""
                super().__init__()
                # Initialize timing manually
                self._start_time: float | None = None

                self._metrics_state: FlextTypes.Core.Dict | None = None

                # Use HandlerConfig if provided, otherwise create default
                if handler_config:
                    self._config = handler_config
                    self._handler_name = str(handler_config.get("handler_name", ""))
                    self.handler_id = handler_config.get("handler_id", "")
                else:
                    # Create default config
                    default_name = handler_name or self.__class__.__name__
                    default_id = handler_id or f"{self.__class__.__name__}_{id(self)}"
                    self._config = {
                        "handler_id": default_id,
                        "handler_name": default_name,
                        "handler_type": "command",
                    }
                    self._handler_name = default_name
                    self.handler_id = default_id

            # Timing functionality is now provided by FlextTiming.Timeable mixin
            # Provides: start_timing, stop_timing, get_elapsed_time, measure_operation

            @property
            def handler_name(self) -> str:
                """Get handler name for identification."""
                return (
                    str(self._handler_name)
                    if self._handler_name is not None
                    else self.__class__.__name__
                )

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

                            # Handle parameterized generics by getting origin type
                            origin_type = get_origin(expected_type) or expected_type

                            # Handle both parameterized generics and regular types
                            command_origin = get_origin(command_type) or command_type

                            # Support being called with instance or type
                            if isinstance(command_type, type) or hasattr(
                                command_type, "__origin__"
                            ):
                                try:
                                    # For parameterized generics, compare origin types
                                    if hasattr(command_type, "__origin__"):
                                        can_handle_result = (
                                            command_origin == origin_type
                                        )
                                    else:
                                        can_handle_result = issubclass(
                                            command_type, origin_type  # type: ignore[arg-type]
                                        )
                                except TypeError:
                                    # Handle cases where origin_type is not a valid class
                                    can_handle_result = command_type == expected_type
                            else:
                                try:
                                    can_handle_result = isinstance(
                                        command_type, origin_type
                                    )
                                except TypeError:
                                    # Handle cases where origin_type is not a valid class
                                    # Be more permissive for duck typing and flexibility
                                    can_handle_result = True

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

                self.logger.info("Could not determine handler type constraints")
                return True

            def execute(self, command: CommandT) -> FlextResult[ResultT]:
                """Execute command with full validation and error handling."""
                self.logger.info(
                    "Executing command",
                    command_type=type(command).__name__,
                    command_id=getattr(command, "command_id", "unknown"),
                )

                # Validate command can be handled (pass type, not instance)
                if not self.can_handle(type(command)):
                    error_msg = (
                        f"{self._handler_name} cannot handle {type(command).__name__}"
                    )
                    self.logger.error(error_msg)
                    return FlextResult[ResultT].fail(
                        error_msg,
                        error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
                    )

                # Validate the command's data
                validation_result = self.validate_command(command)
                if validation_result.is_failure:
                    self.logger.info(
                        "Command validation failed",
                        command_type=type(command).__name__,
                        error=validation_result.error,
                    )
                    return FlextResult[ResultT].fail(
                        validation_result.error or "Validation failed",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Use mixin timing capabilities
                self._start_time = time.time()

                try:
                    # Log operation using mixin method
                    self.logger.debug(
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
                    elapsed = time.time() - (getattr(self, "_start_time", 0))
                    execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

                    self.logger.info(
                        "Command executed successfully",
                        command_type=type(command).__name__,
                        execution_time_ms=execution_time_ms,
                        success=result.is_success,
                    )

                    return result

                except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                    # Get timing using mixin method
                    elapsed = time.time() - (getattr(self, "_start_time", 0))
                    execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

                    self.logger.exception(
                        "Command execution failed",
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
            # FlextProcessing.CQRS.QueryHandler[QueryT, QueryResultT],  # Temporarily disabled
            FlextMixins,
        ):
            """Generic base class for query handlers with Pydantic configuration."""

            def __init__(
                self,
                handler_name: str | None = None,
                handler_id: str | None = None,
                handler_config: dict[str, object] | None = None,
            ) -> None:
                """Initialize query handler with optional Pydantic configuration."""
                super().__init__()
                # Initialize mixins manually since we don't inherit from them
                FlextMixins.initialize_validation(self)
                self._start_time = time.time()
                FlextMixins.clear_cache(self)

                # Use HandlerConfig if provided, otherwise create default
                if handler_config:
                    self._config = handler_config
                    self._handler_name = str(handler_config.get("handler_name", ""))
                    self.handler_id = handler_config.get("handler_id", "")
                else:
                    # Create default config
                    default_name = handler_name or self.__class__.__name__
                    default_id = handler_id or f"{self.__class__.__name__}_{id(self)}"
                    self._config = {
                        "handler_id": default_id,
                        "handler_name": default_name,
                        "handler_type": "query",
                    }
                    self._handler_name = default_name
                    self.handler_id = default_id

            @property
            def handler_name(self) -> str:
                """Get handler name for identification."""
                return (
                    str(self._handler_name)
                    if self._handler_name is not None
                    else self.__class__.__name__
                )

            @property
            def logger(self) -> FlextLogger:
                """Get logger instance for this query handler."""
                return FlextLogger(self.__class__.__name__)

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

            def execute(self, query: QueryT) -> FlextResult[QueryResultT]:
                """Execute query - alias for handle_query for compatibility."""
                return self.handle_query(query)

    # =========================================================================
    # BUS - Command bus for routing and execution
    # =========================================================================

    class Bus(
        FlextMixins,
    ):
        """Command bus for routing and executing commands with Pydantic configuration."""

        def __init__(
            self,
            bus_config: dict[str, object] | None = None,
        ) -> None:
            """Initialize command bus with optional Pydantic configuration."""
            super().__init__()
            # Initialize mixins manually since we don't inherit from them
            FlextMixins.initialize_validation(self)
            FlextMixins.clear_cache(self)
            # Timestampable mixin initialization
            self._created_at = datetime.now(UTC)
            self._start_time = time.time()

            # Use BusConfig if provided, otherwise create default
            if bus_config:
                self._config = bus_config
            else:
                self._config = {}

            # Handlers registry: command type -> handler instance
            self._handlers: FlextTypes.Core.Dict = {}
            # Middleware pipeline (controlled by config)
            self._middleware: list[dict[str, object]] = []
            # Middleware instances cache
            self._middleware_instances: FlextTypes.Core.Dict = {}
            # Execution counter
            self._execution_count: int = 0
            # Underlying FlextCommands CQRS bus for direct registrations
            # # self._fb_bus  # Disabled = FlextProcessing.CQRS.CommandBus()  # Temporarily disabled
            # Auto-discovery handlers (single-arg registration)
            self._auto_handlers: FlextTypes.Core.List = []

            # Add logger
            self.logger = FlextLogger(self.__class__.__name__)

        def register_handler(self, *args: object) -> FlextResult[None]:
            """Register command handler."""
            if len(args) == 1:
                handler = args[0]
                if handler is None:
                    msg = "Handler cannot be None"
                    return FlextResult[None].fail(msg)

                handle_method = getattr(handler, "handle", None)
                if not callable(handle_method):
                    msg = "Invalid handler: must have callable 'handle' method"
                    return FlextResult[None].fail(msg)

                key = getattr(handler, "handler_id", handler.__class__.__name__)
                if key in self._handlers:
                    self.logger.info(
                        "Handler already registered",
                        command_type=str(key),
                        existing_handler=self._handlers[key].__class__.__name__,
                    )
                    return FlextResult[None].ok(None)

                self._handlers[key] = handler
                self._auto_handlers.append(handler)
                self.logger.info(
                    "Handler registered successfully",
                    command_type=str(key),
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return FlextResult[None].ok(None)

            # Two-arg form: (command_type, handler)
            two_arg_form = 2
            if len(args) == two_arg_form:
                command_type_obj, handler = args
                if handler is None or command_type_obj is None:
                    msg = "Invalid arguments: command_type and handler are required"
                    return FlextResult[None].fail(msg)

                # Compute key for local registry visibility
                # Handle parameterized generics first before checking __name__
                if hasattr(command_type_obj, "__origin__") and hasattr(
                    command_type_obj, "__args__"
                ):
                    # Reconstruct the string representation for parameterized generics
                    origin = getattr(command_type_obj, "__origin__")
                    origin_name = getattr(origin, "__name__", str(origin))
                    args = getattr(command_type_obj, "__args__")
                    if args:
                        args_str = ", ".join(
                            getattr(arg, "__name__", str(arg)) for arg in args
                        )
                        key = f"{origin_name}[{args_str}]"
                    else:
                        key = origin_name
                else:
                    name_attr = getattr(command_type_obj, "__name__", None)
                    key = name_attr if name_attr is not None else str(command_type_obj)
                self._handlers[key] = handler
                # Register into underlying CQRS bus
                # _ = self._fb_bus.register(cast("type", command_type_obj), handler)  # Disabled
                self.logger.info(
                    "Handler registered for command type",
                    command_type=key,
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return FlextResult[None].ok(None)

            msg = "register_handler() takes 1 or 2 positional arguments"
            return FlextResult[None].fail(msg)

        def find_handler(self, command: object) -> object | None:
            """Find handler for command."""
            command_type = type(command)
            command_name = command_type.__name__

            # First, try to find handler by command type name in _handlers (two-arg registration)
            if command_name in self._handlers:
                return self._handlers[command_name]

            # Search auto-registered handlers (single-arg form)
            for handler in self._auto_handlers:
                can_handle_method = getattr(handler, "can_handle", None)
                if callable(can_handle_method) and can_handle_method(command_type):
                    return handler
            return None

        def execute(self, command: object) -> FlextResult[object]:
            """Execute command through registered handler."""
            # Check if bus is enabled
            if not self._config.get("enable_middleware", True) and self._middleware:
                return FlextResult[object].fail(
                    "Middleware pipeline is disabled but middleware is configured",
                    error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
                )

            self._execution_count = int(self._execution_count) + 1
            command_type = type(command)

            # Check cache for query results if this is a query (and if metrics are enabled)
            if self._config.get("enable_metrics", True) and (
                hasattr(command, "query_id") or "Query" in command_type.__name__
            ):
                cache_key = f"{command_type.__name__}_{hash(str(command))}"
                cached_result = getattr(self, "_cache", {}).get(cache_key)
                if cached_result is not None:
                    self.logger.info(
                        "Returning cached query result",
                        command_type=command_type.__name__,
                        cache_key=cache_key,
                    )
                    return cast("FlextResult[object]", cached_result)

            self.logger.debug(
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
                # try_send = self._fb_bus  # Disabled.send(command)
                # if try_send.success:
                #     return try_send
                # If still no handler, report
                handler_names = [h.__class__.__name__ for h in self._auto_handlers]
                self.logger.error(
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
            self._start_time = time.time()
            result = self._execute_handler(handler, command)
            elapsed = time.time() - self._start_time

            # Cache successful query results
            if result.is_success and (
                hasattr(command, "query_id") or "Query" in command_type.__name__
            ):
                cache_key = f"{command_type.__name__}_{hash(str(command))}"
                getattr(self, "_cache", {}).setdefault(cache_key, result)
                self.logger.debug(
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
            if not self._config.get("enable_middleware", True):
                return FlextResult[None].ok(None)

            # Sort middleware by order
            def get_order(m: dict[str, object]) -> int:
                order = m.get("order", 0)
                if isinstance(order, int):
                    return order
                if isinstance(order, str):
                    try:
                        return int(order)
                    except ValueError:
                        return 0
                else:
                    return 0

            sorted_middleware = sorted(self._middleware, key=get_order)

            for middleware_config in sorted_middleware:
                if not getattr(middleware_config, "enabled", True):
                    continue

                # Get actual middleware instance

                middleware = self._middleware_instances.get(
                    str(getattr(middleware_config, "middleware_id", ""))
                )
                if middleware is None:
                    # Skip middleware configs without instances
                    continue

                self.logger.debug(
                    "Applying middleware",
                    middleware_id=getattr(middleware_config, "middleware_id", ""),
                    middleware_type=getattr(middleware_config, "middleware_type", ""),
                    order=getattr(middleware_config, "order", 0),
                )

                process_method = getattr(middleware, "process", None)
                if callable(process_method):
                    result = process_method(command, handler)
                    if isinstance(result, FlextResult) and result.is_failure:
                        self.logger.info(
                            "Middleware rejected command",
                            middleware_type=getattr(
                                middleware_config, "middleware_type", ""
                            ),
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
            self.logger.debug(
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
            middleware_config: dict[str, object] | None = None,
        ) -> FlextResult[None]:
            """Add middleware to pipeline with optional Pydantic configuration."""
            if not self._config.get("enable_middleware", True):
                # Middleware pipeline is disabled, skip adding
                return FlextResult[None].ok(None)

            # Create config if not provided
            if middleware_config is None:
                middleware_config = {
                    "middleware_id": f"mw_{len(self._middleware)}",
                    "middleware_type": type(middleware).__name__,
                    "enabled": True,
                    "order": len(self._middleware),
                }

            # Store both middleware and config
            self._middleware.append(middleware_config)
            # Also store the actual middleware instance
            self._middleware_instances[
                str(getattr(middleware_config, "middleware_id", ""))
            ] = middleware

            self.logger.info(
                "Middleware added to pipeline",
                middleware_type=getattr(middleware_config, "middleware_type", ""),
                middleware_id=getattr(middleware_config, "middleware_id", ""),
                total_middleware=len(self._middleware),
            )

            return FlextResult[None].ok(None)

        def get_all_handlers(self) -> FlextTypes.Core.List:
            """Get all registered handlers."""
            return list(self._handlers.values())

        def unregister_handler(self, command_type: type | str) -> bool:
            """Unregister command handler."""
            for key in list(self._handlers.keys()):
                # Handle both class objects and string comparisons
                if key == command_type:
                    # Direct match (class object)
                    del self._handlers[key]
                    self.logger.info(
                        "Handler unregistered",
                        command_type=getattr(
                            command_type, "__name__", str(command_type)
                        ),
                        remaining_handlers=len(self._handlers),
                    )
                    return True
                if isinstance(command_type, str):
                    # String comparison
                    key_name = getattr(key, "__name__", None)
                    if (key_name is not None and key_name == command_type) or str(
                        key
                    ) == command_type:
                        del self._handlers[key]
                        self.logger.info(
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

    #     # =========================================================================
    #     # RESULTS - Result helper methods for FlextResult patterns
    #     # =========================================================================
    #
    class Results:
        """Result helper methods for FlextResult patterns."""

        @staticmethod
        def success(data: object) -> FlextResult[object]:
            """Create a success result."""
            return FlextResult[object].ok(data)

        @staticmethod
        def failure(
            error: str,
            error_code: str | None = None,
            error_data: FlextTypes.Core.Dict | None = None,
        ) -> FlextResult[object]:
            """Create a failure result."""
            return FlextResult[object].fail(
                error,
                error_code=error_code
                or FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
                error_data=error_data,
            )

    #
    #     # =========================================================================
    #     # FACTORIES - Factory methods for creating instances
    #     # =========================================================================
    #
    class Factories:
        """Factory methods for creating CQRS components."""

        @staticmethod
        def create_command_bus() -> FlextCommands.Bus:
            """Create a new command bus instance."""
            return FlextCommands.Bus()

        @staticmethod
        def create_simple_handler(
            handler_func: Callable[[object], object],
        ) -> FlextCommands.Handlers.CommandHandler[object, object]:
            """Create a simple command handler from a function."""

            class SimpleHandler(FlextCommands.Handlers.CommandHandler[object, object]):
                def handle(self, command: object) -> FlextResult[object]:
                    result = handler_func(command)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)

                def __call__(self, command: object) -> FlextResult[object]:
                    """Make the handler callable."""
                    return self.handle(command)

            return SimpleHandler()

        @staticmethod
        def create_query_handler(
            handler_func: Callable[[object], object],
        ) -> FlextCommands.Handlers.QueryHandler[object, object]:
            """Create a simple query handler from a function."""

            class SimpleQueryHandler(
                FlextCommands.Handlers.QueryHandler[object, object],
            ):
                def handle(self, query: object) -> FlextResult[object]:
                    result = handler_func(query)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)

                def __call__(self, query: object) -> FlextResult[object]:
                    """Make the query handler callable."""
                    return self.handle(query)

            return SimpleQueryHandler()


__all__: FlextTypes.Core.StringList = [
    "FlextCommands",
]
