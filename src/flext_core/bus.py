"""Command bus for FLEXT-Core 1.0.0 CQRS flows.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import operator
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Protocol, cast

from flext_core.constants import FlextConstants
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextBus(FlextMixins):
    """Runtime bus that enforces the CQRS contract shared across FLEXT 1.x.

    It is the execution core described in the modernization plan: configuration
    is validated through ``FlextModels.CqrsConfig``, handlers are wrapped so they
    surface ``FlextResult`` outcomes, and every dispatch emits context-aware
    telemetry via ``FlextLogger``. Downstream packages reach it through
    ``FlextDispatcher`` to guarantee a uniform command/query experience.

    This is the single unified class containing all CQRS bus functionality
    with nested protocols and utility methods following the single class per
    module pattern established in the 1.0.0 refactoring.
    """

    # Nested Protocol for type safety
    class ModelDumpable(Protocol):
        """Protocol for objects that have a model_dump method."""

        def model_dump(self: object) -> FlextTypes.Core.Dict:
            """Convert model to dictionary representation."""
            ...

    class CacheUtilities:
        """Cache-related utility methods consolidated into nested class."""

        @staticmethod
        def sort_key(value: object) -> str:
            """Return a deterministic string for ordering normalized cache components."""
            return json.dumps(value, sort_keys=True, default=str)

        @staticmethod
        def normalize_component(value: object) -> object:
            """Normalize arbitrary objects into cache-friendly deterministic structures."""
            if value is None or isinstance(value, (bool, int, float, str)):
                return value

            if isinstance(value, bytes):
                return ("bytes", value.hex())

            if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
                # Cast to ModelDumpable protocol to ensure type checker knows it has model_dump
                model_obj = cast("FlextBus.ModelDumpable", value)
                try:
                    dumped: FlextTypes.Core.Dict = model_obj.model_dump()
                except TypeError:
                    dumped = {}
                # dumped is already a Dict from model_dump(), so it's always a Mapping
                return ("pydantic", FlextBus.CacheUtilities.normalize_component(dumped))

            if dataclasses.is_dataclass(value):
                # Ensure we have a dataclass instance, not a class
                if isinstance(value, type):
                    return ("dataclass_class", str(value))
                # Use the value directly since it's already a dataclass instance
                return (
                    "dataclass",
                    FlextBus.CacheUtilities.normalize_component(
                        dataclasses.asdict(value)
                    ),
                )

            if isinstance(value, Mapping):
                # Normalize keys and values, and sort by a deterministic key
                # Create a list of tuples first to avoid generator type issues
                mapping_items: list[tuple[object, object]] = []
                # Cast the items to avoid unknown type issues
                mapping_value = cast("Mapping[object, object]", value)
                for key, val in mapping_value.items():
                    normalized_key = FlextBus.CacheUtilities.normalize_component(key)
                    normalized_val = FlextBus.CacheUtilities.normalize_component(val)
                    mapping_items.append((normalized_key, normalized_val))

                # Sort by the first element (normalized key)
                mapping_items.sort(
                    key=lambda item: FlextBus.CacheUtilities.sort_key(item[0])
                )

                normalized_items = tuple(mapping_items)
                return ("mapping", normalized_items)

            if isinstance(value, (list, tuple)):
                # Cast the sequence to avoid unknown type issues
                sequence_value = cast("Sequence[object]", value)
                # Create a list first to avoid generator type issues
                sequence_items = [
                    FlextBus.CacheUtilities.normalize_component(item)
                    for item in sequence_value
                ]
                return ("sequence", tuple(sequence_items))

            if isinstance(value, set):
                # Cast the set to avoid unknown type issues
                set_value = cast("set[object]", value)
                # Create a list first to avoid generator type issues
                set_items = [
                    FlextBus.CacheUtilities.normalize_component(item)
                    for item in set_value
                ]

                # Sort by cache sort key
                set_items.sort(key=FlextBus.CacheUtilities.sort_key)

                normalized_set = tuple(set_items)
                return ("set", normalized_set)

            try:
                # Cast to proper type for type checker
                value_vars_dict: dict[str, object] = cast(
                    "dict[str, object]", vars(value)
                )
            except TypeError:
                return ("repr", repr(value))

            normalized_vars = tuple(
                (key, FlextBus.CacheUtilities.normalize_component(val))
                for key, val in sorted(
                    value_vars_dict.items(), key=operator.itemgetter(0)
                )
            )
            return ("vars", normalized_vars)

    def __init__(
        self,
        bus_config: FlextModels.CqrsConfig.Bus | FlextTypes.Core.Dict | None = None,
    ) -> None:
        """Initialize FlextBus with configuration."""
        super().__init__()

        # Configuration model
        self._config_model = self._create_config_model(bus_config)

        # Handler registry
        self._handlers: FlextTypes.Core.Dict = {}

        # Middleware pipeline - use parent's _middleware for callables only
        # Middleware configurations stored separately
        self._middleware_configs: list[dict[str, object]] = []
        # Middleware instances cache
        self._middleware_instances: FlextTypes.Core.Dict = {}
        # Execution counter
        self._execution_count: int = 0
        # Auto-discovery handlers (single-arg registration)
        self._auto_handlers: FlextTypes.Core.List = []

        # Cache configuration - use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, FlextResult[object]] = OrderedDict()
        self._max_cache_size: int = 100

        # Timing
        self._created_at: float = time.time()
        self._start_time: float = 0.0

        # Add logger
        self.logger = FlextLogger(self.__class__.__name__)

    def _create_config_model(
        self, bus_config: FlextModels.CqrsConfig.Bus | FlextTypes.Core.Dict | None
    ) -> FlextModels.CqrsConfig.Bus:
        """Create configuration model from input."""
        if isinstance(bus_config, FlextModels.CqrsConfig.Bus):
            return bus_config
        if isinstance(bus_config, dict):
            return FlextModels.CqrsConfig.Bus(**bus_config)
        return FlextModels.CqrsConfig.Bus()

    @property
    def config(self) -> FlextModels.CqrsConfig.Bus:
        """Access the bus configuration model."""
        return self._config_model

    @classmethod
    def create_command_bus(
        cls,
        bus_config: FlextModels.CqrsConfig.Bus | FlextTypes.Core.Dict | None = None,
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
        | FlextTypes.Core.Dict
        | None = None,
    ) -> FlextHandlers[object, object]:
        """Wrap a bare callable into a CQRS command handler with validation.

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
            callable_func=handler_func,
            handler_name=handler_name,
            handler_type=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            handler_config=handler_config,
        )

    @staticmethod
    def create_query_handler(
        handler_func: Callable[[object], object],
        handler_config: FlextModels.CqrsConfig.Handler
        | FlextTypes.Core.Dict
        | None = None,
    ) -> FlextHandlers[object, object]:
        """Wrap a callable into a CQRS query handler that returns `FlextResult`.

        Args:
            handler_func: The callable function to wrap
            handler_config: Handler configuration or None for defaults

        Returns:
            FlextHandlers: Configured query handler instance

        """
        handler_name = getattr(handler_func, "__name__", "SimpleQueryHandler")

        return FlextHandlers.from_callable(
            callable_func=handler_func,
            handler_name=handler_name,
            handler_type=FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
            handler_config=handler_config,
        )

    @staticmethod
    def _normalize_command_key(command_type_obj: object) -> str:
        """Create a comparable key for command identifiers."""
        if hasattr(command_type_obj, "__origin__") and hasattr(
            command_type_obj, "__args__"
        ):
            origin_attr = getattr(command_type_obj, "__origin__", None)
            args_attr = getattr(command_type_obj, "__args__", None)
            if origin_attr is not None and args_attr is not None:
                origin_name = getattr(origin_attr, "__name__", str(origin_attr))
                if args_attr:
                    args_str = ", ".join(
                        getattr(arg, "__name__", str(arg)) for arg in args_attr
                    )
                    return f"{origin_name}[{args_str}]"
                return origin_name

        name_attr = getattr(command_type_obj, "__name__", None)
        if name_attr is not None:
            return str(name_attr)
        return str(command_type_obj)

    def _normalize_middleware_config(
        self, middleware_config: object | None
    ) -> dict[str, object]:
        """Convert middleware configuration into a dictionary."""
        if middleware_config is None:
            return {}

        if isinstance(middleware_config, Mapping):
            # Cast to proper type for type checker
            return dict(cast("Mapping[str, object]", middleware_config))

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
                    return cast("dict[str, object]", result)

        normalized: dict[str, object] = {}
        sentinel = object()
        for key in ("middleware_id", "middleware_type", "enabled", "order"):
            value = getattr(middleware_config, key, sentinel)
            if value is not sentinel:
                normalized[key] = value

        return normalized

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

            handle_method_name = FlextConstants.Mixins.METHOD_HANDLE
            handle_method = getattr(handler, handle_method_name, None)
            if not callable(handle_method):
                msg = (
                    f"Invalid handler: must have callable '{handle_method_name}' method"
                )
                return FlextResult[None].fail(msg)

            # Always add to auto_handlers for discovery
            self._auto_handlers.append(handler)

            # Additionally, if handler has handler_id attribute, register in _handlers dict
            handler_id = getattr(handler, "handler_id", None)
            if handler_id is not None:
                self._handlers[str(handler_id)] = handler
                self.logger.info(
                    "Handler registered successfully",
                    handler_type=handler.__class__.__name__,
                    handler_id=str(handler_id),
                    total_handlers=len(self._handlers),
                )
            else:
                self.logger.info(
                    "Handler registered successfully",
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._auto_handlers),
                )
            return FlextResult[None].ok(None)

        # Two-arg form: (command_type, handler)
        two_arg_form = 2
        if len(args) == two_arg_form:
            command_type_obj, handler = args
            if handler is None or command_type_obj is None:
                msg = "Invalid arguments: command_type and handler are required"
                return FlextResult[None].fail(msg)

            # Validate command_type is not empty string
            if isinstance(command_type_obj, str) and not command_type_obj.strip():
                msg = "Command type cannot be empty"
                return FlextResult[None].fail(msg)

            # Compute key for local registry visibility
            key = self._normalize_command_key(command_type_obj)
            self._handlers[key] = handler
            self.logger.info(
                "Handler registered for command type",
                command_type=key,
                handler_type=handler.__class__.__name__,
                total_handlers=len(self._handlers),
            )
            return FlextResult[None].ok(None)

        # Error: Unsupported argument count
        msg = f"register_handler takes 1 or 2 positional arguments but {len(args)} were given"
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

        # Validate command if it has custom validation method (not Pydantic field validator)
        if hasattr(command, "validate_command"):
            validation_method = getattr(command, "validate_command")
            # Check if it's a custom validation method (callable without parameters)
            # and returns a FlextResult (not a Pydantic field validator)
            if callable(validation_method):
                try:
                    # Try to call without parameters to see if it's a custom method
                    sig = inspect.signature(validation_method)
                    if (
                        len(sig.parameters) == 0
                    ):  # No parameters = custom validation method
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
                    # If calling without parameters fails, it's likely a Pydantic field validator
                    # Skip custom validation in this case
                    self.logger.debug(f"Skipping Pydantic field validator: {e}")

        is_query = hasattr(command, "query_id") or "Query" in command_type.__name__

        should_consider_cache = (
            self._config_model.enable_caching and self._max_cache_size > 0 and is_query
        )
        cache_key: str | None = None
        if should_consider_cache:
            # Generate a more deterministic cache key
            cache_key = self._generate_cache_key(command, command_type)
            cached_result: FlextResult[object] | None = self._cache.get(cache_key)
            if cached_result is not None:
                self._cache.move_to_end(cache_key)
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
        middleware_result: FlextResult[None] = self._apply_middleware(command, handler)
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
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._max_cache_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self.logger.debug(
                    "Evicted cached query result",
                    command_type=command_type.__name__,
                    cache_key=evicted_key,
                )
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
        def get_order(middleware_item: object) -> int:
            order_value: object
            if isinstance(middleware_item, dict):
                # Cast to proper dict type for type checker
                middleware_dict: dict[str, object] = cast(
                    "dict[str, object]", middleware_item
                )
                order_value = middleware_dict.get("order", 0)
            else:
                order_value = getattr(middleware_item, "order", 0)

            if isinstance(order_value, str):
                try:
                    return int(order_value)
                except ValueError:
                    return 0
            elif isinstance(order_value, int):
                return order_value
            return 0

        sorted_middleware = sorted(self._middleware_configs, key=get_order)

        for middleware_config in sorted_middleware:
            # Extract configuration values - middleware_config is from self._middleware which stores dicts
            if isinstance(middleware_config, dict):  # It's a dict-like object
                config_dict = middleware_config
                middleware_id_value = config_dict.get("middleware_id")
                middleware_type_value = config_dict.get("middleware_type", "")
                order_value = config_dict.get("order", 0)
                enabled_value = config_dict.get(
                    "enabled", True
                )  # Default to True for backward compatibility
            else:
                middleware_id_value = getattr(middleware_config, "middleware_id", "")
                middleware_type_value = getattr(
                    middleware_config, "middleware_type", ""
                )
                order_value = getattr(middleware_config, "order", 0)
                enabled_value = getattr(
                    middleware_config, "enabled", True
                )  # Default to True

            # Skip disabled middleware
            if not enabled_value:
                self.logger.debug(
                    "Skipping disabled middleware",
                    middleware_id=middleware_id_value
                    if middleware_id_value is not None
                    else "",
                    middleware_type=str(middleware_type_value),
                )
                continue

            # Get actual middleware instance
            middleware_id_str = (
                "" if middleware_id_value is None else str(middleware_id_value)
            )
            middleware = self._middleware_instances.get(middleware_id_str)
            if middleware is None:
                # Skip middleware configs without instances
                continue

            self.logger.debug(
                "Applying middleware",
                middleware_id=middleware_id_value
                if middleware_id_value is not None
                else "",
                middleware_type=str(middleware_type_value),
                order=order_value,
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
        try:
            # For Pydantic models, use model_dump with sorted keys
            if hasattr(command, "model_dump"):
                model_dump_method = getattr(command, "model_dump")
                data = model_dump_method(mode="python")
                # Sort keys recursively for deterministic ordering
                sorted_data = self._sort_dict_keys(data)
                return f"{command_type.__name__}_{hash(str(sorted_data))}"

            # For dataclasses, use asdict with sorted keys
            if (
                hasattr(command, "__dataclass_fields__")
                and is_dataclass(command)
                and not isinstance(command, type)
            ):
                dataclass_data = asdict(command)
                dataclass_sorted_data = self._sort_dict_keys(dataclass_data)
                return f"{command_type.__name__}_{hash(str(dataclass_sorted_data))}"

            # For dictionaries, sort keys
            if isinstance(command, dict):
                dict_sorted_data = self._sort_dict_keys(
                    cast("FlextTypes.Core.Dict", command)
                )
                return f"{command_type.__name__}_{hash(str(dict_sorted_data))}"

            # For other objects, use string representation
            command_str = str(command) if command is not None else "None"
            command_hash = hash(command_str)
            return f"{command_type.__name__}_{command_hash}"

        except Exception:
            # Fallback to string representation if anything fails
            # Handle complex types by being more explicit about type handling
            command_str_fallback = str(command) if command is not None else "None"
            try:
                command_hash_fallback = hash(command_str_fallback)
                return f"{command_type.__name__}_{command_hash_fallback}"
            except TypeError:
                # If hash fails, use a deterministic fallback
                return f"{command_type.__name__}_{abs(hash(command_str_fallback.encode('utf-8')))}"

    def _sort_dict_keys(self, obj: object) -> object:
        """Recursively sort dictionary keys for deterministic ordering.

        Args:
            obj: Object to sort (dict, list, or other)

        Returns:
            Object with sorted keys

        """
        if isinstance(obj, dict):
            # Type-safe dict processing with explicit type annotations
            dict_obj: FlextTypes.Core.Dict = cast("FlextTypes.Core.Dict", obj)
            sorted_items: list[tuple[object, object]] = sorted(
                dict_obj.items(), key=lambda x: str(x[0])
            )
            return {str(k): self._sort_dict_keys(v) for k, v in sorted_items}
        if isinstance(obj, list):
            obj_list: list[object] = cast("list[object]", obj)
            return [self._sort_dict_keys(item) for item in obj_list]
        if isinstance(obj, tuple):
            obj_tuple: tuple[object, ...] = cast("tuple[object, ...]", obj)
            return tuple(self._sort_dict_keys(item) for item in obj_tuple)
        return obj

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
                            "FlextResult[object]", result
                        )
                        if typed_result.is_success:
                            return typed_result
                        last_failure = typed_result
                    else:
                        return FlextResult[object].ok(result)
                except Exception as e:
                    return FlextResult[object].fail(
                        f"Handler execution failed: {e}",
                        error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
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
            error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
        )

    def add_middleware(
        self,
        middleware: object,
        middleware_config: FlextTypes.Core.Dict | None = None,
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

        config_data: dict[str, object] = self._normalize_middleware_config(
            middleware_config
        )
        if not config_data:
            config_data = {}

        middleware_id = config_data.get("middleware_id")
        if middleware_id is None:
            middleware_id = getattr(
                middleware, "middleware_id", f"mw_{len(self._middleware_configs)}"
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

    def get_all_handlers(self) -> FlextTypes.Core.List:
        """Return all registered handler instances.

        Returns:
            FlextTypes.Core.List: List of all registered handlers

        """
        return list(self._handlers.values())

    def unregister_handler(self, command_type: type | str) -> FlextResult[None]:
        """Remove a handler registration by type or name.

        Args:
            command_type: The command type or name to unregister.

        Returns:
            FlextResult[None]: Success if handler was removed, failure otherwise

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

        return FlextResult[None].fail(f"Handler for {command_type} not found")

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


# Direct class access - no legacy aliases

__all__: FlextTypes.Core.StringList = [
    "FlextBus",
]
