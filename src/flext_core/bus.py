"""Command bus for FLEXT-Core 1.0.0 CQRS flows.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, cast

from flext_core.constants import FlextConstants
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class ModelDumpable(Protocol):
    """Protocol for objects that have a model_dump method."""

    def model_dump(self) -> dict[str, object]: ...


class FlextBus(FlextMixins):
    """Runtime bus that enforces the CQRS contract shared across FLEXT 1.x.

    It is the execution core described in the modernization plan: configuration
    is validated through ``FlextModels.CqrsConfig``, handlers are wrapped so they
    surface ``FlextResult`` outcomes, and every dispatch emits context-aware
    telemetry via ``FlextLogger``. Downstream packages reach it through
    ``FlextDispatcher`` to guarantee a uniform command/query experience.
    """

    def __init__(
        self,
        bus_config: FlextModels.CqrsConfig.Bus | dict[str, object] | None = None,
        *,
        enable_middleware: bool = True,
        enable_metrics: bool = True,
        enable_caching: bool = True,
        execution_timeout: int = FlextConstants.Defaults.TIMEOUT,
        max_cache_size: int = FlextConstants.Performance.DEFAULT_BATCH_SIZE,
        implementation_path: str = "flext_core.bus:FlextBus",
    ) -> None:
        """Initialise the bus using the CQRS configuration models."""
        super().__init__()
        # Initialize mixins manually since we don't inherit from them
        FlextMixins.initialize_validation(self, "bus_config")
        FlextMixins.clear_cache(self)
        # Timestampable mixin initialization
        self._created_at = datetime.now(UTC)
        self._start_time = time.time()

        # Convert bus_config to dict if it's a Bus object
        config_dict: dict[str, object] | None = None
        if bus_config is not None:
            # Check if it's a Pydantic model with model_dump method
            if hasattr(bus_config, "model_dump") and callable(
                getattr(bus_config, "model_dump")
            ):
                # Cast to ModelDumpable protocol to ensure type checker knows it has model_dump
                model_obj = cast("ModelDumpable", bus_config)
                config_dict = model_obj.model_dump()
            elif isinstance(bus_config, dict):
                config_dict = bus_config

        config_model = FlextModels.CqrsConfig.Bus.create_bus_config(
            config_dict,
            enable_middleware=enable_middleware,
            enable_metrics=enable_metrics,
            enable_caching=enable_caching,
            execution_timeout=execution_timeout,
            max_cache_size=max_cache_size,
            implementation_path=implementation_path,
        )

        self._config_model = config_model
        self._config = config_model.model_dump()

        # Handlers registry: command type -> handler instance
        self._handlers: FlextTypes.Core.Dict = {}
        # Middleware pipeline (controlled by config)
        self._middleware: list[dict[str, object]] = []
        # Middleware instances cache
        self._middleware_instances: FlextTypes.Core.Dict = {}
        # Execution counter
        self._execution_count: int = 0
        # Auto-discovery handlers (single-arg registration)
        self._auto_handlers: FlextTypes.Core.List = []

        # Add logger
        self.logger = FlextLogger(self.__class__.__name__)

    @property
    def config(self) -> FlextModels.CqrsConfig.Bus:
        """Expose the validated CQRS bus configuration model."""
        return self._config_model

    @classmethod
    def create_command_bus(
        cls,
        bus_config: FlextModels.CqrsConfig.Bus | dict[str, object] | None = None,
    ) -> FlextBus:
        """Create factory helper mirroring the documented ``create_command_bus`` API.

        Args:
            bus_config: Bus configuration or None for defaults

        Returns:
            FlextBus: Configured command bus instance

        """
        return cls(bus_config=bus_config)

    @staticmethod
    def create_simple_handler(
        handler_func: Callable[[object], object],
        handler_config: FlextModels.CqrsConfig.Handler
        | dict[str, object]
        | None = None,
    ) -> FlextHandlers[object, object]:
        """
        Wrap a bare callable into a CQRS command handler with validation.

        Args:
            handler_func: A callable that takes a single argument (the command payload) and returns a result.
                The function should implement the business logic for the command.
            handler_config: Optional handler configuration, either as a `FlextModels.CqrsConfig.Handler` instance
                or a dictionary. If None, default configuration is used.

        Returns:
            FlextHandlers[object, object]: A CQRS command handler that wraps the provided callable,
                with input/output validation and result wrapping. See `FlextHandlers.from_callable` for details.
        """
        handler_name = getattr(handler_func, "__name__", "SimpleHandler")

        return FlextHandlers.from_callable(
            handler_func,
            mode=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            handler_config=handler_config,
            handler_name=handler_name,
        )

    @staticmethod
    def create_query_handler(
        handler_func: Callable[[object], object],
        handler_config: FlextModels.CqrsConfig.Handler
        | dict[str, object]
        | None = None,
    ) -> FlextHandlers[object, object]:
        """Wrap a callable into a CQRS query handler that returns `FlextResult`."""

        handler_name = getattr(handler_func, "__name__", "SimpleQueryHandler")

        return FlextHandlers.from_callable(
            handler_func,
            mode=FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
            handler_config=handler_config,
            handler_name=handler_name,
        )

    def register_handler(self, *args: object) -> FlextResult[None]:
        """Register a handler instance (single or paired registration forms).

        Args:
            *args: Handler instance(s) to register

        Returns:
            FlextResult: Success or failure result

        """
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
                # Type guard: check if attributes exist and are not None
                origin_attr = getattr(command_type_obj, "__origin__", None)
                args_attr = getattr(command_type_obj, "__args__", None)
                if origin_attr is not None and args_attr is not None:
                    # Reconstruct the string representation for parameterized generics
                    origin_name = getattr(origin_attr, "__name__", str(origin_attr))
                    if args_attr:
                        args_str = ", ".join(
                            getattr(arg, "__name__", str(arg)) for arg in args_attr
                        )
                        key = f"{origin_name}[{args_str}]"
                    else:
                        key = origin_name
                else:
                    name_attr = getattr(command_type_obj, "__name__", None)
                    key = name_attr if name_attr is not None else str(command_type_obj)
            else:
                name_attr = getattr(command_type_obj, "__name__", None)
                key = name_attr if name_attr is not None else str(command_type_obj)
            self._handlers[key] = handler
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
        # Check if bus is enabled
        if not self._config_model.enable_middleware and self._middleware:
            return FlextResult[object].fail(
                "Middleware pipeline is disabled but middleware is configured",
                error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
            )

        self._execution_count = int(self._execution_count) + 1
        command_type = type(command)

        # Check cache for query results if this is a query (and if metrics are enabled)
        if self._config_model.enable_metrics and (
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
                # cached_result is already FlextResult[object]
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
        if (
            result.is_success
            and self._config_model.enable_caching
            and (hasattr(command, "query_id") or "Query" in command_type.__name__)
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
        """Run the configured middleware pipeline for the current message.

        Args:
            command: The command/query to process
            handler: The handler that will execute the command

        Returns:
            FlextResult: Middleware processing result

        """
        if not self._config_model.enable_middleware:
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
                str(getattr(middleware_config, "middleware_id", "")),
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
                            middleware_config,
                            "middleware_type",
                            "",
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

        # Try different handler methods in order of preference
        handler_methods = ["execute", "handle", "process_command"]

        for method_name in handler_methods:
            method = getattr(handler, method_name, None)
            if callable(method):
                try:
                    result = method(command)
                    if isinstance(result, FlextResult):
                        # Result is already a FlextResult, just return it
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
        """Return all registered handler instances.

        Returns:
            FlextTypes.Core.List: List of all registered handlers

        """
        return list(self._handlers.values())

    def unregister_handler(self, command_type: type | str) -> bool:
        """Remove a handler registration by type or name.

        Args:
            command_type: The command type or name to unregister

        Returns:
            bool: True if handler was removed, False otherwise

        """
        for key in list(self._handlers.keys()):
            # Handle both class objects and string comparisons
            if key == command_type:
                # Direct match (class object)
                del self._handlers[key]
                self.logger.info(
                    "Handler unregistered",
                    command_type=getattr(command_type, "__name__", str(command_type)),
                    remaining_handlers=len(self._handlers),
                )
                return True
            if isinstance(command_type, str):
                # String comparison
                key_name = getattr(key, "__name__", None)
                if (key_name is not None and key_name == command_type) or str(
                    key,
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
        """Compatibility shim that delegates to :meth:`execute`.

        Args:
            command: The command to send

        Returns:
            FlextResult: Execution result

        """
        return self.execute(command)

    def get_registered_handlers(self) -> FlextTypes.Core.Dict:
        """Expose the handler registry keyed by command identifiers.

        Returns:
            FlextTypes.Core.Dict: Dictionary of registered handlers

        """
        return {str(k): v for k, v in self._handlers.items()}


__all__: FlextTypes.Core.StringList = [
    "FlextBus",
]
