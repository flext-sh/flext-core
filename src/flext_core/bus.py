# ruff: disable=E402
"""Command bus implementation for CQRS message routing.

This module provides FlextBus, a command and query bus implementing
the Command Query Responsibility Segregation (CQRS) pattern with
middleware support, caching, and comprehensive error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import cast

from flext_core.constants import FlextConstants
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextBus(
    FlextProtocols.Commands.CommandBus,
    FlextProtocols.Commands.Middleware,
    FlextMixins,
):
    """Command and query bus for CQRS message routing.

    Provides message dispatching with middleware support, caching,
    and comprehensive error handling. Routes commands and queries
    to registered handlers with automatic validation and result wrapping.

    Features:
    - Handler registration and discovery
    - Middleware pipeline processing
    - Result caching for query optimization
    - Context-aware telemetry and logging
    - Thread-safe handler registry
    - Automatic FlextResult wrapping

    Usage:
        >>> from flext_core import FlextBus
        >>>
        >>> bus = FlextBus()
        >>> bus.register_handler("CreateUser", handler)
        >>> result = bus.execute("CreateUser", command)
    """

    class _Cache:
        """Private cache manager for command/query result caching."""

        def __init__(
            self, max_size: int = FlextConstants.Container.MAX_CACHE_SIZE
        ) -> None:
            """Initialize cache manager.

            Args:
                max_size: Maximum number of cached results

            """
            super().__init__()
            # Use OrderedDict runtime type with explicit annotation
            self._cache: OrderedDict[str, FlextResult[object]] = OrderedDict()
            self._max_size = max_size

        def get(self, key: str) -> FlextResult[object] | None:
            """Get cached result by key.

            Args:
                key: Cache key

            Returns:
                Cached result or None if not found

            """
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)
            return result

        def put(self, key: str, result: FlextResult[object]) -> None:
            """Store result in cache.

            Args:
                key: Cache key
                result: Result to cache

            """
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

        def clear(self) -> None:
            """Clear all cached results."""
            self._cache.clear()

        def size(self) -> int:
            """Get current cache size.

            Returns:
                Number of cached items

            """
            return len(self._cache)

    def __init__(
        self,
        bus_config: FlextModels.Cqrs.Bus | FlextTypes.Dict | None = None,
    ) -> None:
        """Initialize FlextBus with configuration and service infrastructure."""
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service("flext_bus")

        # Configuration model
        self._config_model = self._create_config_model(bus_config)

        # Handler registry
        self._handlers: FlextTypes.Dict = {}

        # Middleware pipeline - use parent's _middleware for callables only
        # Middleware configurations stored separately - using FlextTypes.Handlers
        self._middleware_configs: list[FlextTypes.Handlers.MiddlewareConfig] = []
        # Middleware instances cache
        self._middleware_instances: FlextTypes.Dict = {}
        # Execution counter
        self._execution_count: int = FlextConstants.ZERO
        # Auto-discovery handlers (single-arg registration)
        self._auto_handlers: FlextTypes.List = []

        # Cache configuration - use dedicated CqrsCache manager
        self._cache: FlextBus._Cache = FlextBus._Cache(
            max_size=self._config_model.max_cache_size
        )

        # Timing
        self._created_at: float = time.time()
        self._start_time: float = FlextConstants.INITIAL_TIME

        # Log initialization with context
        self._log_with_context(
            "info",
            "FlextBus initialized",
            max_cache_size=self._config_model.max_cache_size,
        )

    def _create_config_model(
        self,
        bus_config: FlextModels.Cqrs.Bus | FlextTypes.Dict | None,
    ) -> FlextModels.Cqrs.Bus:
        """Create configuration model from input."""
        if isinstance(bus_config, FlextModels.Cqrs.Bus):
            return bus_config
        if isinstance(bus_config, dict):
            # Cast dict to proper types for Bus model
            config_dict = dict(
                bus_config,
            )  # Create a new dict to avoid mutating the original
            return FlextModels.Cqrs.Bus.model_validate(config_dict)
        return FlextModels.Cqrs.Bus()

    @property
    def bus_config(self) -> FlextModels.Cqrs.Bus:
        """Access the bus configuration model."""
        return self._config_model

    @staticmethod
    def _normalize_command_key(command_type_obj: object) -> str:
        """Create a comparable key for command identifiers."""
        name_attr = getattr(command_type_obj, "__name__", None)
        if name_attr is not None:
            return str(name_attr)
        return str(command_type_obj)

    def _normalize_middleware_config(
        self,
        middleware_config: object | None,
    ) -> FlextTypes.Dict:
        """Convert middleware configuration into a dictionary."""
        if middleware_config is None:
            return {}

        if isinstance(middleware_config, dict):
            return cast("FlextTypes.Dict", middleware_config)

        if isinstance(middleware_config, Mapping):
            # Cast to proper type for type checker
            return dict(cast("Mapping[str, object]", middleware_config))

        # For Pydantic models or objects with model_dump
        for attr_name in ("model_dump", "dict"):
            method = getattr(middleware_config, attr_name, None)
            if callable(method):
                try:
                    result: object = method()
                except TypeError:
                    continue
                if isinstance(result, Mapping):
                    return dict(cast("Mapping[str, object]", result))
                if isinstance(result, dict):
                    return cast("FlextTypes.Dict", result)

        return {}

    def register_handler(self, *args: object) -> FlextResult[None]:
        """Register a handler instance.

        Args:
            *args: Handler instance or (command_type, handler) pair

        Returns:
            FlextResult: Success or failure result

        """
        if len(args) == 1:
            handler = args[0]
            if handler is None:
                return FlextResult[None].fail("Handler cannot be None")

            # Validate handler has required method
            method_name = FlextConstants.Mixins.METHOD_HANDLE
            handle_method = getattr(handler, method_name, None)
            if not callable(handle_method):
                return FlextResult[None].fail(
                    f"Invalid handler: must have callable '{method_name}' method"
                )

            # Add to auto-discovery list
            self._auto_handlers.append(handler)

            # Register by handler_id if available
            handler_id = getattr(handler, "handler_id", None)
            if handler_id is not None:
                self._handlers[str(handler_id)] = handler
                self.logger.info(
                    "Handler registered",
                    handler_type=handler.__class__.__name__,
                    handler_id=str(handler_id),
                    total_handlers=len(self._handlers),
                )
            else:
                self.logger.info(
                    "Handler registered for auto-discovery",
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._auto_handlers),
                )
            return FlextResult[None].ok(None)

        # Two-arg form: (command_type, handler)
        two_arg_count = 2
        if len(args) == two_arg_count:
            command_type_obj, handler = args
            if handler is None or command_type_obj is None:
                return FlextResult[None].fail(
                    "Invalid arguments: command_type and handler are required",
                )

            if isinstance(command_type_obj, str) and not command_type_obj.strip():
                return FlextResult[None].fail("Command type cannot be empty")

            key = self._normalize_command_key(command_type_obj)
            self._handlers[key] = handler
            self.logger.info(
                "Handler registered for command type",
                command_type=key,
                handler_type=handler.__class__.__name__,
                total_handlers=len(self._handlers),
            )
            return FlextResult[None].ok(None)

        return FlextResult[None].fail(
            f"register_handler takes 1 or 2 arguments but {len(args)} were given",
        )

    def find_handler(self, command: object) -> object | None:
        """Locate the handler that can process the provided message.

        Args:
            command: The command/query object to find handler for

        Returns:
            object | None: The handler instance or None if not found

        """
        command_type = type(command)
        command_name = command_type.__name__

        # First, try to find handler by command type name in _handlers
        # (two-arg registration)
        if command_name in self._handlers:
            return self._handlers[command_name]

        # Search auto-registered handlers (single-arg form)
        for handler in self._auto_handlers:
            can_handle_method = getattr(handler, "can_handle", None)
            if callable(can_handle_method) and can_handle_method(command_type):
                return handler
        return None

    def execute(self, command: object) -> FlextResult[object]:
        """Execute a command/query through middleware and the resolved handler.

        Args:
            command: The command or query object to execute

        Returns:
            FlextResult: Execution result

        """
        # Propagate context for distributed tracing
        command_type = type(command)
        self._propagate_context(f"execute_{command_type.__name__}")

        # Track operation metrics
        with self.track(f"bus_execute_{command_type.__name__}") as _:
            # Check if bus is enabled
            if not self._config_model.enable_middleware and (
                self._middleware_configs or self._middleware_instances
            ):
                return FlextResult[object].fail(
                    "Middleware pipeline is disabled but middleware is configured",
                    error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
                )

            self._execution_count = int(self._execution_count) + 1

            # Validate command if it has custom validation method
            # (not Pydantic field validator)
            if isinstance(command, FlextProtocols.Foundation.HasValidateCommand):
                validation_method = command.validate_command
                # Check if it's a custom validation method (callable without parameters)
                # and returns a FlextResult (not a Pydantic field validator)
                if callable(validation_method):
                    try:
                        # Try to call without parameters to see if custom method
                        sig = inspect.signature(validation_method)
                        # Allow 0 parameters (staticmethod) or 1 parameter
                        # (instance method with self)
                        if len(sig.parameters) <= 1:
                            validation_result: object = validation_method()
                            if (
                                hasattr(validation_result, "is_failure")
                                and hasattr(validation_result, "error")
                                and getattr(validation_result, "is_failure", False)
                            ):
                                self.logger.warning(
                                    "Command validation failed",
                                    command_type=command_type.__name__,
                                    validation_error=getattr(
                                        validation_result,
                                        "error",
                                        "Unknown validation error",
                                    ),
                                )
                                return FlextResult[object].fail(
                                    getattr(
                                        validation_result,
                                        "error",
                                        "Command validation failed",
                                    )
                                    or "Command validation failed",
                                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                                )
                    except Exception as e:
                        # If calling without parameters fails, it's likely a
                        # Pydantic field validator - skip custom validation
                        self.logger.debug("Skipping Pydantic field validator: %s", e)

            is_query = hasattr(command, "query_id") or "Query" in command_type.__name__

            should_consider_cache = self._config_model.enable_caching and is_query
            cache_key: str | None = None
            if should_consider_cache:
                # Generate a more deterministic cache key
                cache_key = self._generate_cache_key(command, command_type)
                cached_result: FlextResult[object] | None = self._cache.get(cache_key)
                if cached_result is not None:
                    self.logger.debug(
                        "Returning cached query result",
                        command_type=command_type.__name__,
                        cache_key=cache_key,
                    )
                    # cached_result is already FlextResult[object]
                    return cached_result

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
            middleware_result: FlextResult[None] = self._apply_middleware(
                command, handler
            )
            if middleware_result.is_failure:
                return FlextResult[object].fail(
                    middleware_result.error or "Middleware rejected command",
                    error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
                )

            # Execute the handler with timing
            self._start_time = time.time()
            result: FlextResult[object] = self._execute_handler(handler, command)
            elapsed = time.time() - self._start_time

            # Cache successful query results
            if result.is_success and should_consider_cache and cache_key is not None:
                self._cache.put(cache_key, result)
                self.logger.debug(
                    "Cached query result",
                    command_type=command_type.__name__,
                    cache_key=cache_key,
                    execution_time=elapsed,
                )

            return result

    def process(self, command: object, handler: object) -> object:
        """Process command through middleware pipeline.

        Args:
            command: The command being processed
            handler: The handler that will process the command

        Returns:
            The result of middleware processing

        """
        # Apply middleware pipeline
        middleware_result: FlextResult[None] = self._apply_middleware(command, handler)
        if middleware_result.is_failure:
            return middleware_result

        # Execute the handler
        return self._execute_handler(handler, command)

    def _apply_middleware(
        self,
        command: object,
        handler: object,
    ) -> FlextResult[None]:
        """Run the configured middleware pipeline for the current message.

        Args:
            command: The command/query to process
            handler: The handler that will execute the command

        Returns:
            FlextResult: Middleware processing result

        """
        if not self._config_model.enable_middleware or not self._middleware_configs:
            return FlextResult[None].ok(None)

        # Sort middleware by order
        def get_order(middleware_config: FlextTypes.Dict) -> int:
            order_value = middleware_config.get(
                "order",
                FlextConstants.Defaults.DEFAULT_MIDDLEWARE_ORDER,
            )
            if isinstance(order_value, str):
                try:
                    return int(order_value)
                except ValueError:
                    return FlextConstants.Defaults.DEFAULT_MIDDLEWARE_ORDER
            return (
                int(order_value)
                if isinstance(order_value, int)
                else FlextConstants.Defaults.DEFAULT_MIDDLEWARE_ORDER
            )

        sorted_middleware = sorted(self._middleware_configs, key=get_order)

        for middleware_config in sorted_middleware:
            # Extract configuration values
            config_dict = middleware_config
            middleware_id_value = config_dict.get("middleware_id")
            middleware_type_value = config_dict.get("middleware_type", "")
            enabled_value = config_dict.get("enabled", True)

            # Skip disabled middleware
            if not enabled_value:
                self.logger.debug(
                    "Skipping disabled middleware",
                    middleware_id=middleware_id_value or "",
                    middleware_type=str(middleware_type_value),
                )
                continue

            # Get actual middleware instance
            middleware_id_str = str(middleware_id_value) if middleware_id_value else ""
            middleware = self._middleware_instances.get(middleware_id_str)
            if middleware is None:
                continue

            self.logger.debug(
                "Applying middleware",
                middleware_id=middleware_id_value or "",
                middleware_type=str(middleware_type_value),
                order=config_dict.get(
                    "order",
                    FlextConstants.Defaults.DEFAULT_MIDDLEWARE_ORDER,
                ),
            )

            process_method = getattr(middleware, "process", None)
            if callable(process_method):
                result = process_method(command, handler)
                if isinstance(result, FlextResult):
                    result_typed = cast("FlextResult[object]", result)
                    if result_typed.is_failure:
                        self.logger.info(
                            "Middleware rejected command",
                            middleware_type=str(middleware_type_value),
                            error=result_typed.error or "Unknown error",
                        )
                        return FlextResult[None].fail(
                            str(result_typed.error or "Middleware rejected command"),
                        )

        return FlextResult[None].ok(None)

    def _generate_cache_key(self, command: object, command_type: type[object]) -> str:
        """Generate a deterministic cache key for the command.

        Args:
            command: The command/query object
            command_type: The type of the command

        Returns:
            str: Deterministic cache key

        """
        return FlextUtilities.Cache.generate_cache_key(command, command_type)

    def _execute_handler(
        self,
        handler: object,
        command: object,
    ) -> FlextResult[object]:
        """Execute the handler while normalizing return types to `FlextResult`.

        Args:
            handler: The handler instance to execute
            command: The command/query to process

        Returns:
            FlextResult: Handler execution result

        """
        self.logger.debug(
            "Delegating to handler",
            handler_type=handler.__class__.__name__,
        )

        # ISSUE: Fallback pattern - tries multiple method names instead
        # of requiring standard interface
        # Try different handler methods in order of preference
        handler_methods = [
            FlextConstants.Mixins.METHOD_EXECUTE,
            FlextConstants.Mixins.METHOD_HANDLE,
            FlextConstants.Mixins.METHOD_PROCESS_COMMAND,
        ]

        last_failure: FlextResult[object] | None = None

        for method_name in handler_methods:
            method = getattr(handler, method_name, None)
            if callable(method):
                try:
                    result = method(command)
                    if isinstance(result, FlextResult):
                        # Cast to FlextResult[object] to ensure type compatibility
                        typed_result: FlextResult[object] = cast(
                            "FlextResult[object]",
                            result,
                        )
                        if typed_result.is_success:
                            return typed_result
                        last_failure = typed_result
                    else:
                        return FlextResult[object].ok(result)
                except Exception as e:
                    return FlextResult[object].fail(
                        f"Handler execution failed: {e}",
                        error_code=str(FlextConstants.Errors.COMMAND_PROCESSING_FAILED),
                    )

        # No valid handler method found
        if not handler_methods:
            formatted_methods = "handler method"
        elif len(handler_methods) > 1:
            formatted_methods = (
                f"{', '.join(handler_methods[:-1])}, or {handler_methods[-1]}"
            )
        else:
            formatted_methods = handler_methods[0]
        if last_failure is not None:
            return last_failure

        return FlextResult[object].fail(
            f"Handler has no callable {formatted_methods} method",
            error_code=str(FlextConstants.Errors.COMMAND_BUS_ERROR),
        )

    def add_middleware(
        self,
        middleware: object,
        middleware_config: FlextTypes.Dict | None = None,
    ) -> FlextResult[None]:
        """Append middleware with validated configuration metadata.

        Args:
            middleware: The middleware instance to add
            middleware_config: Configuration for the middleware

        Returns:
            FlextResult: Success or failure result

        """
        if not self._config_model.enable_middleware:
            # Middleware pipeline is disabled, skip adding
            return FlextResult[None].ok(None)

        config_data: FlextTypes.Dict = self._normalize_middleware_config(
            middleware_config,
        )
        if not config_data:
            config_data = {}

        middleware_id = config_data.get("middleware_id")
        if middleware_id is None:
            middleware_id = getattr(
                middleware,
                "middleware_id",
                f"mw_{len(self._middleware_configs)}",
            )
            config_data["middleware_id"] = middleware_id

        # Ensure middleware_id is a string
        middleware_id_str = str(middleware_id)

        config_data.setdefault("middleware_type", type(middleware).__name__)
        config_data.setdefault("enabled", True)
        config_data.setdefault("order", len(self._middleware_configs))

        # Store middleware config separately from callables
        self._middleware_configs.append(config_data)
        # Also store the actual middleware instance using the resolved ID
        self._middleware_instances[middleware_id_str] = middleware

        self.logger.info(
            "Middleware added to pipeline",
            middleware_type=config_data.get("middleware_type", ""),
            middleware_id=config_data.get("middleware_id", ""),
            total_middleware=len(self._middleware_configs),
        )

        return FlextResult[None].ok(None)

    def get_all_handlers(self) -> FlextTypes.List:
        """Return all registered handler instances.

        Returns:
            FlextTypes.List: List of all registered handlers

        """
        return list(self._handlers.values())

    def unregister_handler(self, command_type: type | str) -> FlextResult[None]:
        """Remove a handler registration by type or name.

        Args:
            command_type: The command type or name to unregister.

        Returns:
            FlextResult[None]: Success if handler was removed, failure if not found

        """
        for key in list(self._handlers.keys()):
            candidate_names: set[str] = {str(key)}
            key_name = getattr(key, "__name__", None)
            if isinstance(key_name, str):
                candidate_names.add(key_name)

            # Direct match only if both are types (not str and type comparison)
            # Since _handlers has str keys, we can only match by name
            direct_match = False
            command_names: set[str] = {str(command_type)}
            command_name_attr = getattr(command_type, "__name__", None)
            if isinstance(command_name_attr, str):
                command_names.add(command_name_attr)
            normalized_command = self._normalize_command_key(command_type)
            # normalized_command is always a str from _normalize_command_key
            command_names.add(normalized_command)

            if direct_match or candidate_names.intersection(command_names):
                del self._handlers[key]
                len(self._handlers)

                self.logger.info(
                    "Handler unregistered",
                    command_type=normalized_command,
                    remaining_handlers=len(self._handlers),
                )
                return FlextResult[None].ok(None)

        return FlextResult[None].fail(
            f"Handler not found for command type: {command_type}"
        )

    def get_registered_handlers(self) -> FlextTypes.Dict:
        """Expose the handler registry keyed by command identifiers.

        Returns:
            FlextTypes.Dict: Dictionary of registered handlers

        """
        return {str(k): v for k, v in self._handlers.items()}

    def clear_handlers(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._auto_handlers.clear()

    # =========================================================================
    # EVENT PUBLISHER PROTOCOL IMPLEMENTATION (Phase 4.3)
    # =========================================================================

    def publish_event(self, event: object) -> FlextResult[None]:
        """Publish a domain event.

        Args:
            event: Domain event to publish

        Returns:
            FlextResult[None]: Success or failure result

        """
        try:
            # Use existing execute mechanism for event publishing
            result = self.execute(event)

            if result.is_failure:
                return FlextResult[None].fail(
                    f"Event publishing failed: {result.error}"
                )

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Event publishing error: {e}")

    def publish_events(self, events: FlextTypes.List) -> FlextResult[None]:
        """Publish multiple domain events.

        Uses FlextResult.from_callable() to eliminate try/except and
        flow_through() for declarative event processing pipeline.

        Args:
            events: List of domain events to publish

        Returns:
            FlextResult[None]: Success or failure result

        """

        def publish_all() -> None:
            # Convert events to FlextResult pipeline
            def make_publish_func(
                event_item: object,
            ) -> Callable[[None], FlextResult[None]]:
                def publish_func(_: None) -> FlextResult[None]:
                    return self.publish_event(event_item)

                return publish_func

            publish_funcs = [make_publish_func(event) for event in events]
            result = FlextResult[None].ok(None).flow_through(*publish_funcs)
            if result.is_failure:
                raise RuntimeError(result.error or "Event publishing failed")

        return FlextResult[None].from_callable(publish_all)

    def subscribe(self, event_type: str, handler: object) -> FlextResult[None]:
        """Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Handler callable for the event

        Returns:
            FlextResult[None]: Success or failure result

        """
        try:
            # Use existing register_handler mechanism
            return self.register_handler(event_type, handler)
        except Exception as e:
            return FlextResult[None].fail(f"Event subscription error: {e}")

    def unsubscribe(
        self,
        event_type: str,
        _handler: object,
    ) -> FlextResult[None]:
        """Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove (reserved for future use)

        Returns:
            FlextResult[None]: Success or error result

        """
        try:
            # Use existing unregister_handler mechanism which returns FlextResult
            return self.unregister_handler(event_type)
        except Exception as e:
            self.logger.exception("Event unsubscription error")
            return FlextResult[None].fail(f"Event unsubscription error: {e}")

    def publish(
        self,
        event_name: str,
        data: FlextTypes.Dict,
    ) -> FlextResult[None]:
        """Publish a named event with data.

        Convenience method for publishing events by name with associated data.

        Args:
            event_name: Name/identifier of the event
            data: Event data payload

        Returns:
            FlextResult[None]: Success or failure result

        """
        # Create a simple event dict with name and data
        event: FlextTypes.Dict = {
            "event_name": event_name,
            "data": data,
            "timestamp": getattr(self, "_get_timestamp", lambda: "now")(),
        }
        return self.publish_event(event)


# Direct class access - no legacy aliases

__all__: FlextTypes.StringList = [
    "FlextBus",
]
