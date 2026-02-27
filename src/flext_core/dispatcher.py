"""Message dispatch orchestration with reliability features.

Coordinates command and query execution with routing, reliability policies,
and observability features for handler registration and execution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import sys
import time
from collections.abc import Callable, Generator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager, suppress
from datetime import datetime as dt
from types import ModuleType
from typing import Annotated, Any, Literal, Self, TypeGuard, cast, override

from cachetools import LRUCache
from pydantic import BaseModel, Discriminator, TypeAdapter

from flext_core._dispatcher import (
    CircuitBreakerManager,
    RateLimiterManager,
    RetryPolicy,
    TimeoutEnforcer,
)
from flext_core._models.cqrs import FlextModelsCqrs
from flext_core.constants import c
from flext_core.context import FlextContext
from flext_core.handlers import h
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.service import s
from flext_core.typings import t
from flext_core.utilities import u

type DispatcherHandler = t.HandlerType | BaseModel
type HandlerRequestKey = str | type[object]
type HandlerRegistrationRequestInput = (
    m.HandlerRegistrationRequest
    | BaseModel
    | Mapping[str, object]
    | HandlerRequestKey
    | DispatcherHandler
)

# Concrete payload type for dispatcher (replaces _Payload per LAW)
type _Payload = t.ConfigMapValue


class _RegistrationModelInput(m.TaggedModel):
    input_type: Literal["model"] = "model"
    request: m.HandlerRegistrationRequest


class _RegistrationMappingInput(m.TaggedModel):
    input_type: Literal["mapping"] = "mapping"
    request: m.HandlerRegistrationRequest


type _RegistrationInputUnion = Annotated[
    _RegistrationModelInput | _RegistrationMappingInput,
    Discriminator("input_type"),
]


class _MetadataNoneInput(m.TaggedModel):
    input_type: Literal["none"] = "none"
    metadata: None = None


class _MetadataModelInput(m.TaggedModel):
    input_type: Literal["model"] = "model"
    metadata: m.Metadata


class _MetadataMappingInput(m.TaggedModel):
    input_type: Literal["mapping"] = "mapping"
    metadata: dict[str, object]


type _MetadataInputUnion = Annotated[
    _MetadataNoneInput | _MetadataModelInput | _MetadataMappingInput,
    Discriminator("input_type"),
]


_REGISTRATION_INPUT_ADAPTER: TypeAdapter[_RegistrationInputUnion] = TypeAdapter(
    _RegistrationInputUnion,
)
_METADATA_INPUT_ADAPTER: TypeAdapter[_MetadataInputUnion] = TypeAdapter(
    _MetadataInputUnion
)


class FlextDispatcher(s[bool]):
    """Application-level dispatcher that satisfies the command bus protocol.

    The dispatcher exposes CQRS routing, handler registration, and execution
    with layered reliability controls for message dispatching and handler execution.

    This is a specialized CQRS service that extends s for infrastructure
    integration while providing command/query routing capabilities.
    """

    @override
    def __init__(
        self,
        *,
        circuit_breaker: CircuitBreakerManager | None = None,
        rate_limiter: RateLimiterManager | None = None,
        timeout_enforcer: TimeoutEnforcer | None = None,
        retry_policy: RetryPolicy | None = None,
        **data: t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue],
    ) -> None:
        """Initialize dispatcher with reliability managers.

        s handles infrastructure (container, config, context) automatically.
        This init only configures dispatcher-specific reliability components.

        Args:
            circuit_breaker: Optional CircuitBreakerManager. If not provided, will be
                resolved from container or created with defaults.
            rate_limiter: Optional RateLimiterManager. If not provided, will be
                resolved from container or created with defaults.
            timeout_enforcer: Optional TimeoutEnforcer. If not provided, will be
                resolved from container or created with defaults.
            retry_policy: Optional RetryPolicy. If not provided, will be
                resolved from container or created with defaults.
            **data: Additional data passed to s.

        """
        # s handles container, config, context, runtime
        super().__init__(**data)

        # Access config from s (already initialized)
        config = self.config

        # Resolve or create circuit breaker manager
        self._circuit_breaker = (
            circuit_breaker
            or self._resolve_or_create_circuit_breaker(
                config,
            )
        )

        # Resolve or create rate limiter manager
        self._rate_limiter = rate_limiter or self._resolve_or_create_rate_limiter(
            config,
        )

        # Resolve or create timeout enforcer
        self._timeout_enforcer = (
            timeout_enforcer
            or self._resolve_or_create_timeout_enforcer(
                config,
            )
        )

        # Resolve or create retry policy
        self._retry_policy = retry_policy or self._resolve_or_create_retry_policy(
            config,
        )

        # =============== LAYER 2.5: TIMEOUT CONTEXT PROPAGATION ===============

        self._timeout_contexts: MutableMapping[
            str,
            _Payload,
        ] = {}  # operation_id → context
        self._timeout_deadlines: MutableMapping[
            str,
            float,
        ] = {}  # operation_id → deadline timestamp

        # ================ LAYER 1: CQRS ROUTING INITIALIZATION ================

        # Handler registry (from FlextDispatcher dual-mode registration)
        self._handlers: MutableMapping[
            str,
            DispatcherHandler,
        ] = {}  # Handler mappings by message type
        self._auto_handlers: list[DispatcherHandler] = []  # Auto-discovery handlers

        # Middleware pipeline (from FlextDispatcher)
        self._middleware_configs: list[
            m.Config.DispatcherMiddlewareConfig
        ] = []  # Config + ordering
        self._middleware_instances: MutableMapping[
            str,
            t.HandlerCallable,
        ] = {}  # Keyed by middleware_id

        # Query result caching (from FlextDispatcher - LRU cache)
        # Fast fail: use constant directly, no fallback
        max_cache_size = c.Container.MAX_CACHE_SIZE
        self._cache: MutableMapping[str, r[_Payload]] = LRUCache(
            maxsize=max_cache_size,
        )

        # Event subscribers (from FlextDispatcher event protocol)
        self._event_subscribers: MutableMapping[
            str,
            list[_Payload],
        ] = {}  # event_type → handlers

        self._execution_count: int = 0

        # ============= LAYER 3: ADVANCED PROCESSING INITIALIZATION =============

        # Group 1: Handler Registry (internal dispatcher handler registry)
        self._handler_registry: MutableMapping[
            str, t.HandlerType
        ] = {}  # name → handler function
        self._handler_configs: MutableMapping[
            str,
            m.Config.HandlerExecutionConfig,
        ] = {}  # name → handler config
        self._handler_validators: MutableMapping[
            str,
            Callable[[_Payload], bool],
        ] = {}  # validation functions

        # Group 4: Pipeline (dispatcher-managed processing pipeline)
        self._pipeline_steps: list[m.ConfigMap] = []  # Ordered pipeline steps
        self._pipeline_composition: MutableMapping[
            str,
            Callable[
                [_Payload],
                r[_Payload],
            ],
        ] = {}  # composed functions
        self._pipeline_memo: MutableMapping[
            str,
            _Payload,
        ] = {}  # Memoization cache for pipeline

        self._audit_log: list[m.ConfigMap] = []  # Operation audit trail
        self._performance_metrics: MutableMapping[
            str,
            _Payload,
        ] = {}  # Timing and throughput

    def _resolve_or_create_circuit_breaker(
        self,
        config: p.Config,
    ) -> CircuitBreakerManager:
        """Resolve circuit breaker from container or create with defaults.

        Args:
            config: Configuration instance for default values.

        Returns:
            CircuitBreakerManager instance from container or newly created.

        """
        # Try to resolve from container with typed resolution
        result = self.container.get_typed("circuit_breaker", CircuitBreakerManager)
        if result.is_success:
            return result.value

        # Create with defaults from config
        # Use getattr to safely access attributes that may not exist in protocol
        threshold_raw = (
            getattr(config, "circuit_breaker_threshold")
            if hasattr(config, "circuit_breaker_threshold")
            else 5
        )
        threshold: int = threshold_raw if threshold_raw is not None else 5
        return CircuitBreakerManager(
            threshold=threshold,
            recovery_timeout=c.Reliability.DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            success_threshold=c.Reliability.DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
        )

    def _resolve_or_create_rate_limiter(
        self,
        config: p.Config,
    ) -> RateLimiterManager:
        """Resolve rate limiter from container or create with defaults.

        Args:
            config: Configuration instance for default values.

        Returns:
            RateLimiterManager instance from container or newly created.

        """
        # Try to resolve from container with typed resolution
        result = self.container.get_typed("rate_limiter", RateLimiterManager)
        if result.is_success:
            return result.value

        # Create with defaults from config
        # Use getattr to safely access attributes that may not exist in protocol
        max_requests_raw = (
            getattr(config, "rate_limit_max_requests")
            if hasattr(config, "rate_limit_max_requests")
            else 100
        )
        max_requests: int = max_requests_raw if max_requests_raw is not None else 100
        window_seconds_raw = (
            getattr(config, "rate_limit_window_seconds")
            if hasattr(config, "rate_limit_window_seconds")
            else 60.0
        )
        window_seconds: float = (
            window_seconds_raw if window_seconds_raw is not None else 60.0
        )
        return RateLimiterManager(
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    def _resolve_or_create_timeout_enforcer(
        self,
        config: p.Config,
    ) -> TimeoutEnforcer:
        """Resolve timeout enforcer from container or create with defaults.

        Args:
            config: Configuration instance for default values.

        Returns:
            TimeoutEnforcer instance from container or newly created.

        """
        # Try to resolve from container with typed resolution
        result = self.container.get_typed("timeout_enforcer", TimeoutEnforcer)
        if result.is_success:
            return result.value

        # Create with defaults from config
        # Use getattr to safely access attributes that may not exist in protocol
        use_timeout_executor = (
            getattr(config, "enable_timeout_executor")
            if hasattr(config, "enable_timeout_executor")
            else False
        )
        executor_workers_raw = (
            getattr(config, "executor_workers")
            if hasattr(config, "executor_workers")
            else 4
        )
        executor_workers: int = (
            executor_workers_raw if executor_workers_raw is not None else 4
        )
        return TimeoutEnforcer(
            use_timeout_executor=use_timeout_executor,
            executor_workers=executor_workers,
        )

    def _resolve_or_create_retry_policy(
        self,
        config: p.Config,
    ) -> RetryPolicy:
        """Resolve retry policy from container or create with defaults.

        Args:
            config: Configuration instance for default values.

        Returns:
            RetryPolicy instance from container or newly created.

        """
        # Try to resolve from container with typed resolution
        result = self.container.get_typed("retry_policy", RetryPolicy)
        if result.is_success:
            return result.value

        # Create with defaults from config
        # Use getattr to safely access attributes that may not exist in protocol
        max_attempts_raw = (
            getattr(config, "max_retry_attempts")
            if hasattr(config, "max_retry_attempts")
            else 3
        )
        max_attempts: int = max_attempts_raw if max_attempts_raw is not None else 3
        retry_delay_raw = (
            getattr(config, "retry_delay") if hasattr(config, "retry_delay") else 1.0
        )
        retry_delay: float = retry_delay_raw if retry_delay_raw is not None else 1.0
        return RetryPolicy(
            max_attempts=max_attempts,
            retry_delay=retry_delay,
        )

    # ==================== LAYER 3: ADVANCED PROCESSING INTERNAL METHODS ====================

    def _validate_interface(
        self,
        obj: Any,  # Validates arbitrary runtime values
        method_names: list[str] | str,
        context: str,
        *,
        allow_callable: bool = False,
    ) -> r[bool]:
        """Generic interface validation (consolidates 3 validation methods).

        Args:
            obj: Object to validate
            method_names: Single method name or list of acceptable method names
            context: Context string for error messages
            allow_callable: If True, accept callable object without methods

        Returns:
            r[bool]: Success if valid, failure with descriptive error

        """
        if allow_callable and callable(obj):
            return r[bool].ok(value=True)

        # method_names is list[str] | str, convert to list[str]
        methods: list[str]
        match method_names:
            case str():
                methods = [method_names]
            case _:
                methods = [str(m) for m in method_names]
        for method_name in methods:
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                if callable(method):
                    return r[bool].ok(value=True)

        method_list = "' or '".join(methods)
        return r[bool].fail(f"Invalid {context}: must have '{method_list}' method")

    def _validate_handler_registry_interface(
        self,
        handler: DispatcherHandler,
        handler_context: str = "registry handler",
    ) -> r[bool]:
        """Validate handler registry protocol compliance (handle or execute method)."""
        return self._validate_interface(
            handler,
            [c.Mixins.METHOD_HANDLE, c.Mixins.METHOD_EXECUTE],
            handler_context,
        )

    # Note: Processor registry removed in dispatcher refactoring.
    # Processors should be registered as handlers using register_handler().

    # ==================== LAYER 1: CQRS ROUTING INTERNAL METHODS ====================

    @staticmethod
    def _normalize_command_key(
        command_type_obj: HandlerRequestKey,
    ) -> str:
        """Create comparable key for command identifiers (from FlextDispatcher).

        Args:
            command_type_obj: Object to create key from

        Returns:
            Normalized string key for command type

        """
        # Fast fail: __name__ should always exist on types, but handle gracefully
        # Use getattr for direct attribute access
        name_attr = (
            getattr(command_type_obj, "__name__")
            if hasattr(command_type_obj, "__name__")
            else None
        )
        if name_attr is not None:
            return str(name_attr)
        return str(command_type_obj)

    def _validate_handler_interface(
        self,
        handler: DispatcherHandler,
        handler_context: str = "handler",
    ) -> r[bool]:
        """Validate that handler has required handle() interface or is callable."""
        method_name = c.Mixins.METHOD_HANDLE
        return self._validate_interface(
            handler, method_name, handler_context, allow_callable=True
        )

    @staticmethod
    def _is_dispatcher_handler(value: object) -> TypeGuard[DispatcherHandler]:
        return callable(value) or isinstance(value, BaseModel)

    @staticmethod
    def _to_container_value(value: t.PayloadValue | str | int | float | bool | None) -> _Payload:
        """Convert handler return value to container-compatible payload.

        Accepts PayloadValue types and primitives, converts to _Payload (ConfigMapValue).
        """
        if u.is_general_value_type(value):
            return value
        return str(value)

    @staticmethod
    def _is_result_value(value: object) -> TypeGuard[r[_Payload]]:
        return bool(u.is_result_like(value))

    @staticmethod
    def _to_scalar_value(value: _Payload) -> t.ScalarValue:
        match value:
            case str() | int() | float() | bool() | dt() | None:
                return value
            case _:
                return str(value)

    @staticmethod
    def _is_scalar_primitive(
        value: Any,  # Validates arbitrary runtime values
    ) -> TypeGuard[str | int | float | bool | None]:
        match value:
            case str() | bool() | int() | float() | None:
                return True
            case _:
                return False

    @staticmethod
    def _is_metadata_scalar(
        value: Any,  # Validates arbitrary runtime values
    ) -> TypeGuard[str | int | float | bool | dt | None]:
        match value:
            case str() | bool() | int() | float() | dt() | None:
                return True
            case _:
                return False

    @staticmethod
    def _is_optional_str(value: object) -> TypeGuard[str | None]:
        match value:
            case str() | None:
                return True
            case _:
                return False

    @staticmethod
    def _is_optional_int(value: object) -> TypeGuard[int | None]:
        match value:
            case int() | None:
                return True
            case _:
                return False

    def _route_to_handler(
        self,
        command: _Payload,
    ) -> DispatcherHandler | None:
        """Locate the handler that can process the provided message.

        Args:
            command: The command/query object to find handler for

        Returns:
            The handler instance (HandlerType or object for h) or None if not found

        """
        route_names = FlextDispatcher._resolve_message_route_names(command)

        # First, try to find handler by command type name in _handlers
        # (two-arg registration)
        for route_name in route_names:
            if route_name in self._handlers:
                return self._handlers[route_name]

        # Search auto-registered handlers (single-arg form)
        for handler in self._auto_handlers:
            # Fast fail: check if can_handle method exists before calling
            if hasattr(handler, "can_handle"):
                can_handle_method = (
                    getattr(handler, "can_handle")
                    if hasattr(handler, "can_handle")
                    else None
                )
                # can_handle expects str (message_type), not type
                if callable(can_handle_method) and any(
                    can_handle_method(route_name) for route_name in route_names
                ):
                    return handler
        return None

    @staticmethod
    def _resolve_message_route_names(message: _Payload) -> list[str]:
        class_name = message.__class__.__name__
        parsed_route_name: str | None = None
        with suppress(Exception):
            parsed_message = FlextModelsCqrs.parse_message(message)
            parsed_route_name = parsed_message.message_type

        if parsed_route_name is None or parsed_route_name == class_name:
            return [class_name]
        return [parsed_route_name, class_name]

    @staticmethod
    def _is_query(
        command: _Payload,
        command_type: type,
    ) -> bool:
        """Determine if command is a query (cacheable).

        Args:
            command: The command object
            command_type: The type of the command

        Returns:
            bool: True if command is a query

        """
        return hasattr(command, "query_id") or "Query" in command_type.__name__

    @staticmethod
    def _generate_cache_key(
        command: t.GuardInputValue,
        command_type: type[object],
    ) -> str:
        """Generate a deterministic cache key for the command.

        Args:
            command: The command/query object
            command_type: The type of the command

        Returns:
            str: Deterministic cache key

        """
        # generate_cache_key accepts *args: t.GuardInputValue.
        command_type_name = command_type.__name__ if command_type else "unknown"
        return u.generate_cache_key(command, command_type_name)

    def _check_cache_for_result(
        self,
        command: _Payload,
        command_type: type,
        *,
        is_query: bool,
    ) -> r[_Payload]:
        """Check cache for query result and return if found.

        Args:
            command: The command object
            command_type: The type of the command
            is_query: Whether command is a query

        Returns:
            r[_Payload]: Cached result if found, failure if not cacheable or not cached

        """
        # Fast fail: use config value directly, no fallback
        cache_enabled = self.config.enable_caching
        should_consider_cache = cache_enabled and is_query
        if not should_consider_cache:
            return r[_Payload].fail(
                "Cache not enabled or not a query",
                error_code=c.Errors.CONFIGURATION_ERROR,
            )

        if not u.is_general_value_type(command):
            return r[_Payload].fail(
                "Command is not cache-key compatible",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        cache_key = FlextDispatcher._generate_cache_key(command, command_type)
        cached_value = self._cache.get(cache_key)
        if cached_value is not None:
            # Fast fail: cached value must be r[_Payload]
            # Type narrowing: cache stores r, so this is safe
            cached_result: r[_Payload] = cached_value
            self.logger.debug(
                "Returning cached query result",
                operation="check_cache",
                command_type=command_type.__name__,
                cache_key=cache_key,
                source="flext-core/src/flext_core/dispatcher.py",
            )
            return cached_result

        return r[_Payload].fail(
            "Cache miss",
            error_code=c.Errors.NOT_FOUND_ERROR,
        )

    def _execute_handler(
        self,
        handler: DispatcherHandler,
        command: _Payload,
        operation: str = c.Dispatcher.HANDLER_MODE_COMMAND,
    ) -> r[_Payload]:
        """Execute the handler using h pipeline when available.

        Delegates to h._run_pipeline() for full CQRS support including
        mode validation, can_handle check, message validation, context tracking,
        and metrics recording. Falls back to direct handle()/execute() for
        non-h instances.

        Args:
            handler: The handler instance to execute
            command: The command/query to process
            operation: Operation type (command, query, event)

        Returns:
            r: Handler execution result

        """
        handler_class = type(handler)
        # Use getattr for type attribute access (not mapper.get which is for dict/model access)
        handler_class_name = (
            getattr(handler_class, "__name__")
            if hasattr(handler_class, "__name__")
            else "Unknown"
        )
        self.logger.debug(
            "Delegating to handler",
            operation="execute_handler",
            handler_type=handler_class_name,
            command_type=command.__class__.__name__,
            source="flext-core/src/flext_core/dispatcher.py",
        )

        # Delegate to h.dispatch_message() for full CQRS support when present
        dispatch_method = (
            getattr(handler, "dispatch_message")
            if hasattr(handler, "dispatch_message")
            else None
        )
        if callable(dispatch_method):
            raw_dispatch_result = dispatch_method(command, operation=operation)
            if FlextDispatcher._is_result_value(raw_dispatch_result):
                if raw_dispatch_result.is_failure:
                    return r[_Payload].fail(
                        raw_dispatch_result.error or "Handler dispatch failed",
                        error_code=raw_dispatch_result.error_code,
                        error_data=raw_dispatch_result.error_data,
                    )
                return r[_Payload].ok(
                    FlextDispatcher._to_container_value(raw_dispatch_result.value),
                )
            return r[_Payload].ok(
                FlextDispatcher._to_container_value(raw_dispatch_result),
            )

        # Fallback for non-h: try handle() then execute()
        # Use hasattr + getattr for attribute access
        method_name = None
        if hasattr(handler, c.Mixins.METHOD_HANDLE):
            handle_method = (
                getattr(handler, c.Mixins.METHOD_HANDLE)
                if hasattr(handler, c.Mixins.METHOD_HANDLE)
                else None
            )
            if callable(handle_method):
                method_name = c.Mixins.METHOD_HANDLE
        elif hasattr(handler, c.Mixins.METHOD_EXECUTE):
            execute_method = (
                getattr(handler, c.Mixins.METHOD_EXECUTE)
                if hasattr(handler, c.Mixins.METHOD_EXECUTE)
                else None
            )
            if callable(execute_method):
                method_name = c.Mixins.METHOD_EXECUTE

        if not method_name:
            return r[_Payload].fail(
                f"Handler must have '{c.Mixins.METHOD_HANDLE}' or '{c.Mixins.METHOD_EXECUTE}' method",
                error_code=c.Errors.COMMAND_BUS_ERROR,
            )
        # Use getattr for attribute access
        handle_method = (
            getattr(handler, method_name) if hasattr(handler, method_name) else None
        )
        if not callable(handle_method):
            error_msg = f"Handler '{method_name}' must be callable"
            return r[_Payload].fail(
                error_msg,
                error_code=c.Errors.COMMAND_BUS_ERROR,
            )

        try:
            result_raw = handle_method(command)
            result = FlextDispatcher._to_container_value(result_raw)
            return r[_Payload].ok(result)
        except Exception as exc:
            error_msg = f"Handler execution failed: {exc}"
            return r[_Payload].fail(
                error_msg,
                error_code=c.Errors.COMMAND_PROCESSING_FAILED,
            )

    def _execute_middleware_chain(
        self,
        command: _Payload,
        handler: DispatcherHandler,
    ) -> r[bool]:
        """Run the configured middleware pipeline for the current message.

        Args:
            command: The command/query to process
            handler: The handler that will execute the command

        Returns:
            r: Middleware processing result

        """
        # Fast fail: middleware is always enabled if configs exist, no fallback
        # Middleware is enabled by default when configs are present
        if not self._middleware_configs:
            return r[bool].ok(value=True)

        # Sort middleware by order
        sorted_middleware = sorted(
            self._middleware_configs,
            key=self._get_middleware_order,
        )

        for middleware_config in sorted_middleware:
            result = self._process_middleware_instance(
                command,
                handler,
                middleware_config,
            )
            if result.is_failure:
                return result

        return r[bool].ok(value=True)

    @staticmethod
    def _get_middleware_order(
        middleware_config: m.Config.DispatcherMiddlewareConfig,
    ) -> int:
        """Extract middleware execution order from config."""
        return middleware_config.order

    def _process_middleware_instance(
        self,
        command: _Payload,
        handler: DispatcherHandler,
        middleware_config: m.Config.DispatcherMiddlewareConfig,
    ) -> r[bool]:
        """Process a single middleware instance."""
        middleware_id_str = middleware_config.middleware_id
        middleware_type_str = middleware_config.middleware_type
        enabled_value = middleware_config.enabled

        # Skip disabled middleware
        if not enabled_value:
            self.logger.debug(
                "Skipping disabled middleware",
                middleware_id=middleware_id_str,
                middleware_type=middleware_type_str,
            )
            return r[bool].ok(value=True)

        # Get actual middleware instance
        middleware = self._middleware_instances.get(middleware_id_str)
        if middleware is None:
            return r[bool].ok(value=True)

        self.logger.debug(
            "Applying middleware",
            middleware_id=middleware_id_str,
            middleware_type=middleware_type_str,
            order=middleware_config.order,
        )

        return self._invoke_middleware(
            middleware,
            command,
            handler,
            middleware_type_str,
        )

    def _invoke_middleware(
        self,
        middleware: t.HandlerCallable,
        command: _Payload,
        handler: DispatcherHandler,
        middleware_type: _Payload,
    ) -> r[bool]:
        """Invoke middleware and handle result.

        Fast fail: Middleware must have process() method. No fallback to callable.
        """
        # Use getattr for object attribute access (middleware may be callable/class instance)
        process_method = (
            getattr(middleware, "process") if hasattr(middleware, "process") else None
        )
        if not callable(process_method):
            return r[bool].fail(
                "Middleware must have callable 'process' method",
                error_code=c.Errors.CONFIGURATION_ERROR,
            )
        # Invoke middleware and get result
        result_raw = process_method(command, handler)

        # Ensure result is _Payload or FlextResult
        if FlextDispatcher._is_result_value(result_raw):
            result: _Payload | r[_Payload] = result_raw
        else:
            result = FlextDispatcher._to_container_value(result_raw)
        return self._handle_middleware_result(result, middleware_type)

    def _handle_middleware_result(
        self,
        result: _Payload | r[_Payload],
        middleware_type: _Payload,
    ) -> r[bool]:
        """Handle middleware execution result."""
        if FlextDispatcher._is_result_value(result) and result.is_failure:
            # Extract error directly from result.error property
            # result is r[_Payload] after type narrowing
            error_msg = result.error or "Unknown error"
            self.logger.warning(
                "Middleware rejected command - command processing stopped",
                operation="execute_middleware",
                middleware_type=str(middleware_type),
                error=error_msg,
                consequence="Command will not be processed by handler",
                source="flext-core/src/flext_core/dispatcher.py",
            )
            return r[bool].fail(error_msg)

        return r[bool].ok(value=True)

    # ==================== s CONTRACT ====================

    @override
    def execute(self) -> r[bool]:
        """Execute service - satisfies s abstract method.

        For FlextDispatcher, this indicates successful initialization.
        Use dispatch() for CQRS command/query routing.

        Returns:
            r[bool]: Success indicating dispatcher is ready.

        """
        return r[bool].ok(value=True)

    # ==================== LAYER 1 PUBLIC API: CQRS ROUTING & MIDDLEWARE ====================

    def _dispatch_command(
        self,
        command: _Payload,
    ) -> r[_Payload]:
        """Internal: Execute command/query through the CQRS dispatcher routing layer.

        This Layer 1 entry point performs routing with caching and middleware.
        For reliability patterns (circuit breaker, rate limit, retry, timeout),
        use ``dispatch`` which chains this execution with the Layer 2 controls.

        Args:
            command: The command or query object to execute.

        Returns:
            r: Execution result wrapped in r

        """
        # Propagate context for distributed tracing
        command_type = type(command)
        self._propagate_context(f"execute_{command_type.__name__}")

        # Track operation metrics
        with self.track(f"dispatch_execute_{command_type.__name__}") as _:
            self._execution_count += 1

            # Extract command ID for logging - handle various command types
            command_id_value: str = "unknown"
            if isinstance(command, t.ConfigMap):
                command_id_raw = command.root.get("command_id") or command.root.get(
                    "id"
                )
                command_id_value = str(command_id_raw) if command_id_raw else "unknown"
            elif isinstance(command, Mapping):
                command_id_raw = command.get("command_id") or command.get("id")
                command_id_value = str(command_id_raw) if command_id_raw else "unknown"
            elif hasattr(command, "command_id"):
                command_id_value = str(
                    getattr(command, "command_id")
                    if hasattr(command, "command_id")
                    else "unknown"
                )
            elif hasattr(command, "id"):
                command_id_value = str(
                    getattr(command, "id") if hasattr(command, "id") else "unknown"
                )

            self.logger.debug(
                "Executing command",
                operation=c.Mixins.METHOD_EXECUTE,
                command_type=command_type.__name__,
                command_id=command_id_value,
                execution_count=self._execution_count,
                source="flext-core/src/flext_core/dispatcher.py",
            )

            # Check cache for queries
            is_query = FlextDispatcher._is_query(command, command_type)
            cached_result = self._check_cache_for_result(
                command,
                command_type,
                is_query=is_query,
            )
            if cached_result.is_success:
                return cached_result

            # Resolve handler
            handler = self._route_to_handler(command)
            if handler is None:
                handler_names = list(
                    u.map(
                        self._auto_handlers,
                        lambda h: h.__class__.__name__,
                    ),
                )
                # Type note: handler_names is list[str] but logger accepts more general types
                # This is semantically correct - logger should accept list[str] for string list fields
                self.logger.error(
                    "FAILED to find handler for command - DISPATCH ABORTED",
                    operation=c.Mixins.METHOD_EXECUTE,
                    command_type=command_type.__name__,
                    registered_handlers=handler_names,
                    consequence="Command cannot be processed - handler not registered",
                    resolution_hint="Register handler using register_handler() before dispatch",
                    source="flext-core/src/flext_core/dispatcher.py",
                )
                return r[_Payload].fail(
                    f"No handler found for {command_type.__name__}",
                    error_code=c.Errors.COMMAND_HANDLER_NOT_FOUND,
                )

            # Apply middleware pipeline
            middleware_result: r[bool] = self._execute_middleware_chain(
                command,
                handler,
            )
            if middleware_result.is_failure:
                # Fast fail: use unwrap_error() for type-safe str
                return r[_Payload].fail(
                    middleware_result.error or "Unknown error",
                    error_code=c.Errors.COMMAND_BUS_ERROR,
                )

            # Execute handler with appropriate operation type
            operation = (
                c.Dispatcher.HANDLER_MODE_QUERY
                if is_query
                else c.Dispatcher.HANDLER_MODE_COMMAND
            )
            result: r[_Payload] = self._execute_handler(
                handler,
                command,
                operation=operation,
            )

            # Cache successful query results
            cache_key: str | None = None
            guard_command: t.GuardInputValue | None = (
                command if u.is_general_value_type(command) else None
            )
            if result.is_success and is_query and guard_command is not None:
                cache_key = FlextDispatcher._generate_cache_key(
                    guard_command,
                    command_type,
                )
                self._cache[cache_key] = result
                self.logger.debug(
                    "Cached query result",
                    operation="cache_result",
                    command_type=command_type.__name__,
                    cache_key=cache_key,
                    source="flext-core/src/flext_core/dispatcher.py",
                )

            return result

    def _layer1_register_handler(
        self,
        request: HandlerRequestKey | DispatcherHandler,
        handler: DispatcherHandler | None = None,
        *extra_handlers: DispatcherHandler,
    ) -> r[bool]:
        """Internal: Register handler with dual-mode support.

        Supports:
        - Single-arg: register_handler(handler) - Auto-discovery with can_handle()
        - Two-arg: register_handler(MessageType, handler) - Explicit mapping

        Args:
            request: Handler instance or message type (str or type).
            handler: Optional handler; if None, request is treated as single handler.

        Returns:
            r: Success or failure result

        """
        if extra_handlers:
            return r[bool].fail("Invalid arguments: expected 1 or 2 positional args")

        if handler is None:
            if self._is_dispatcher_handler(request):
                return self._register_single_handler(request)
            return r[bool].fail("Handler must be callable or BaseModel")

        request_name = (
            getattr(request, "__name__") if hasattr(request, "__name__") else None
        )
        command_request: str = (
            request_name if request_name is not None else str(request)
        )
        return self._register_two_arg_handler(command_request, handler)

    def _wire_handler_dependencies(
        self,
        handler: DispatcherHandler,
    ) -> None:
        """Wire handler modules/classes to the DI bridge for @inject usage."""
        modules: list[ModuleType] = []
        classes: list[type] = []

        handler_cls = handler if inspect.isclass(handler) else handler.__class__
        if handler_cls:
            classes.append(handler_cls)
            handler_module = inspect.getmodule(handler_cls)
            if handler_module:
                modules.append(handler_module)

        handler_module_from_instance = inspect.getmodule(handler)
        if handler_module_from_instance and handler_module_from_instance not in modules:
            modules.append(handler_module_from_instance)

        try:
            self.container.wire_modules(
                modules=modules or None,
                classes=classes or None,
            )
        except Exception:
            self.logger.debug(
                "DI wiring skipped for handler",
                handler_type=handler.__class__.__name__,
                exc_info=True,
            )

    def _register_single_handler(
        self,
        handler: DispatcherHandler,
    ) -> r[bool]:
        """Register single handler for auto-discovery.

        Args:
            handler: Handler instance

        Returns:
            r with success or error

        """
        # handler is already HandlerType from method signature
        validation_result = self._validate_handler_interface(handler)
        if validation_result.is_failure:
            return validation_result

        self._wire_handler_dependencies(handler)
        self._auto_handlers.append(handler)

        handler_id = (
            getattr(handler, "handler_id") if hasattr(handler, "handler_id") else None
        )
        handler_name = (
            getattr(handler.__class__, "__name__")
            if hasattr(handler.__class__, "__name__")
            else str(type(handler))
        )

        # Register handler as factory in container for DI access
        # Use handler class name or handler_id for factory name
        if handler_id is not None:
            factory_name = f"handler.{handler_id!s}"
            self._handlers[str(handler_id)] = handler
        else:
            # Use handler class name for factory name
            factory_name = f"handler.{handler_name.lower()}"
            # Also register with class name for easier access
            self._handlers[handler_name] = handler

        # Register handler class as factory in container
        # Factory will create new handler instances when resolved
        # Any class (not just BaseModel) can be used as a factory
        handler_cls: type | None = None
        if inspect.isclass(handler):
            handler_cls = handler
        elif isinstance(handler, BaseModel):
            handler_cls = handler.__class__

        if handler_cls is not None and self._container:
            try:
                # Create factory function that instantiates handler class
                # Capture in closure - class is always callable for instantiation
                cls_ref: type = handler_cls

                def _create_handler() -> _Payload:
                    """Factory function to create handler instance."""
                    return cls_ref()

                _ = self._container.register_factory(factory_name, _create_handler)
                self.logger.debug(
                    "Handler registered as factory in container",
                    factory_name=factory_name,
                    handler_type=handler_name,
                )
            except Exception as exc:
                # Log but don't fail registration if container registration fails
                self.logger.debug(
                    "Failed to register handler factory in container",
                    factory_name=factory_name,
                    handler_type=handler_name,
                    error=str(exc),
                )

        if handler_id is not None:
            self.logger.info(
                "Handler registered",
                handler_type=handler_name,
                handler_id=str(handler_id),
                total_handlers=len(self._handlers),
            )
        else:
            self.logger.info(
                "Handler registered for auto-discovery",
                handler_type=handler_name,
                total_handlers=len(self._auto_handlers),
            )

        return r[bool].ok(value=True)

    def _register_two_arg_handler(
        self,
        command_type_obj: HandlerRequestKey | None,
        handler: DispatcherHandler,
    ) -> r[bool]:
        """Register handler with explicit command type.

        Args:
            command_type_obj: Command type object or string
            handler: Handler instance

        Returns:
            r with success or error

        """
        match command_type_obj:
            case None:
                return r[bool].fail("Command type cannot be None")
            case str() if not command_type_obj.strip():
                return r[bool].fail("Command type cannot be empty")
            case _:
                pass

        # handler is already HandlerType from method signature
        validation_result = self._validate_handler_interface(
            handler,
            handler_context=f"handler for '{command_type_obj}'",
        )
        if validation_result.is_failure:
            return validation_result

        self._wire_handler_dependencies(handler)
        key = self._normalize_command_key(command_type_obj)
        self._handlers[key] = handler

        # Register handler as factory in container for DI access
        # Use normalized command key for factory name (e.g., "handler.user.get")
        factory_name = f"handler.{key}"
        handler_name = (
            getattr(handler.__class__, "__name__")
            if hasattr(handler.__class__, "__name__")
            else str(type(handler))
        )

        # Register handler class as factory in container
        # Factory will create new handler instances when resolved
        # Any class (not just BaseModel) can be used as a factory
        handler_cls: type | None = None
        if inspect.isclass(handler):
            handler_cls = handler
        elif isinstance(handler, BaseModel):
            handler_cls = handler.__class__

        if handler_cls is not None and self._container:
            try:
                # Create factory function that instantiates handler class
                # Capture in closure - class is always callable for instantiation
                cls_ref: type = handler_cls

                def _create_handler_for_type() -> _Payload:
                    """Factory function to create handler instance."""
                    return cls_ref()

                _ = self._container.register_factory(
                    factory_name,
                    _create_handler_for_type,
                )
                self.logger.debug(
                    "Handler registered as factory in container",
                    factory_name=factory_name,
                    handler_type=handler_name,
                )
            except Exception as exc:
                # Log but don't fail registration if container registration fails
                self.logger.debug(
                    "Failed to register handler factory in container",
                    factory_name=factory_name,
                    handler_type=handler_name,
                    error=str(exc),
                )

        self.logger.info(
            "Handler registered for command type",
            command_type=key,
            handler_type=handler_name,
            total_handlers=len(self._handlers),
        )

        return r[bool].ok(value=True)

    # ==================== LAYER 1 EVENT PUBLISHING PROTOCOL ====================

    def _publish_event(self, event: _Payload) -> r[bool]:
        """Internal: Publish single domain event to subscribers.

        Args:
            event: Domain event to publish

        Returns:
            r[bool]: Success with True, failure with error details

        """
        try:
            # Use dispatch_command mechanism for event publishing
            result = self._dispatch_command(event)

            if result.is_failure:
                return r[bool].fail(f"Event publishing failed: {result.error}")

            return r[bool].ok(value=True)
        except (TypeError, AttributeError, ValueError) as e:
            # TypeError: invalid event type
            # AttributeError: missing event attribute
            # ValueError: event validation failed
            return r[bool].fail(f"Event publishing error: {e}")

    def subscribe(
        self,
        event_type: str,
        handler: t.HandlerType,
    ) -> r[bool]:
        """Subscribe handler to event type (from FlextDispatcher).

        Args:
            event_type: Type of event to subscribe to
            handler: Handler callable for the event

        Returns:
            r[bool]: Success with True, failure with error details

        """
        try:
            # layer1_register_handler accepts t.HandlerType | _Payload
            # event_type is str, handler is t.HandlerType - both are valid args
            return self._layer1_register_handler(event_type, handler)
        except (TypeError, AttributeError, ValueError) as e:
            # TypeError: invalid handler type
            # AttributeError: handler missing required attributes
            # ValueError: handler validation failed
            return r[bool].fail(f"Event subscription error: {e}")

    def unsubscribe(
        self,
        event_type: str,
        _handler: _Payload | None = None,
    ) -> r[bool]:
        """Unsubscribe from an event type (from FlextDispatcher).

        Args:
            event_type: Type of event to unsubscribe from
            _handler: Handler to remove (reserved for future use)

        Returns:
            r[bool]: Success with True, failure with error details

        """
        try:
            # Remove handler from registry
            if event_type in self._handlers:
                del self._handlers[event_type]
                self.logger.info(
                    "Handler unregistered",
                    command_type=event_type,
                )
                return r[bool].ok(value=True)

            return r[bool].fail(f"Handler not found for event type: {event_type}")
        except (TypeError, KeyError, AttributeError) as e:
            # TypeError: invalid event_type
            # KeyError: event_type not registered
            # AttributeError: handler missing attributes
            self.logger.exception("Event unsubscription error")
            return r[bool].fail(f"Event unsubscription error: {e}")

    def publish(
        self,
        event: _Payload | list[_Payload],
        data: _Payload | None = None,
    ) -> r[bool]:
        """Publish event(s) to subscribers.

        Handles both single and batch event publishing automatically.

        Args:
            event: Single event (dict/model), list of events, or event name (with data)
            data: Optional data payload when event is a string name

        Returns:
            r[bool]: Success with True, failure with error details

        Examples:
            >>> dispatcher.publish(MyEvent(data="test"))  # Single event
            >>> dispatcher.publish([event1, event2, event3])  # Batch events
            >>> dispatcher.publish("user_created", {"user_id": 123})  # Named event

        """
        # Handle named event with data
        if data is not None:
            match event:
                case str():
                    event_dict: m.ConfigMap = m.ConfigMap(
                        root={
                            "event_name": event,
                            "data": data,
                            "timestamp": time.time(),
                        }
                    )
                    return self._publish_event(event_dict)
                case _:
                    pass

        # Handle batch events
        match event:
            case list():
                errors: list[str] = []
                for evt in event:
                    result = self._publish_event(evt)
                    if result.is_failure:
                        errors.append(result.error or "Unknown error")
                if errors:
                    return r[bool].fail(f"Some events failed: {'; '.join(errors)}")
                return r[bool].ok(value=True)
            case _:
                pass

        # Handle single event
        return self._publish_event(event)

    # ------------------------------------------------------------------
    # Registration methods using structured models
    # ------------------------------------------------------------------
    @staticmethod
    def _get_nested_attr(
        obj: Any,  # Validates arbitrary runtime values
        *path: str,
    ) -> object | None:
        """Get nested attribute safely (e.g., obj.attr1.attr2).

        Returns None if any attribute in path doesn't exist or is None.
        Uses u.extract() for dict-like objects, falls back to
        attribute access for objects.

        Args:
            obj: Object to traverse (can be handler or general value)
            *path: Sequence of attribute names to traverse

        """
        if not path:
            return None
        # Try mapping traversal first
        if isinstance(obj, t.ConfigMap):
            current_map = obj
            for attr in path:
                if attr not in current_map.root:
                    return None

                next_value = current_map.root[attr]
                if attr == path[-1]:
                    return next_value

                if isinstance(next_value, t.ConfigMap):
                    current_map = next_value
                elif isinstance(next_value, Mapping):
                    current_map = t.ConfigMap.model_validate(dict(next_value))
                else:
                    return None

            return current_map
        if isinstance(obj, Mapping):
            current_map = t.ConfigMap.model_validate(dict(obj))
            for attr in path:
                if attr not in current_map.root:
                    return None

                next_value = current_map.root[attr]
                if attr == path[-1]:
                    return next_value

                if isinstance(next_value, t.ConfigMap):
                    current_map = next_value
                elif isinstance(next_value, Mapping):
                    current_map = t.ConfigMap.model_validate(dict(next_value))
                else:
                    return None

            return current_map
        # Fall back to attribute access for objects
        current = obj
        for attr in path:
            if not hasattr(current, attr):
                return None
            current = getattr(current, attr) if hasattr(current, attr) else None
            if current is None:
                return None
        return current

    @staticmethod
    def _extract_name_from_handler(
        handler: DispatcherHandler,
    ) -> str:
        """Extract handler name from handler instance attributes.

        Args:
            handler: Handler instance

        Returns:
            Extracted name or empty string

        """
        # Try patterns in order of preference using consolidated helper
        patterns = [
            ("_config_model", "handler_name"),
            ("config", "handler_name"),
            ("handler_name",),
            ("__name__",),
            ("__class__", "__name__"),
        ]

        for pattern in patterns:
            value = FlextDispatcher._get_nested_attr(handler, *pattern)
            if value is not None:
                return str(value)

        return ""

    @staticmethod
    def _normalize_registration_input(
        request: m.HandlerRegistrationRequest
        | BaseModel
        | Mapping[str, t.ConfigMapValue],
    ) -> r[_RegistrationInputUnion]:
        try:
            if isinstance(request, m.HandlerRegistrationRequest):
                return r[_RegistrationInputUnion].ok(
                    _REGISTRATION_INPUT_ADAPTER.validate_python({
                        "input_type": "model",
                        "request": request.model_dump(),
                    })
                )

            if isinstance(request, BaseModel):
                return r[_RegistrationInputUnion].ok(
                    _REGISTRATION_INPUT_ADAPTER.validate_python({
                        "input_type": "model",
                        "request": request.model_dump(),
                    })
                )

            if isinstance(request, Mapping):
                return r[_RegistrationInputUnion].ok(
                    _REGISTRATION_INPUT_ADAPTER.validate_python({
                        "input_type": "mapping",
                        "request": dict(request.items()),
                    })
                )
        except Exception as e:
            return r[_RegistrationInputUnion].fail(f"Request validation failed: {e!s}")

        return r[_RegistrationInputUnion].fail(
            f"Invalid request type: {request.__class__.__name__}. Expected dict or HandlerRegistrationRequest.",
        )

    @staticmethod
    def _normalize_metadata_input(metadata: _Payload | None) -> r[_MetadataInputUnion]:
        if metadata is None:
            return r[_MetadataInputUnion].ok(
                _METADATA_INPUT_ADAPTER.validate_python({
                    "input_type": "none",
                    "metadata": None,
                })
            )

        if isinstance(metadata, m.Metadata):
            return r[_MetadataInputUnion].ok(
                _METADATA_INPUT_ADAPTER.validate_python({
                    "input_type": "model",
                    "metadata": metadata.model_dump(),
                })
            )

        if FlextRuntime.is_dict_like(metadata):
            return r[_MetadataInputUnion].ok(
                _METADATA_INPUT_ADAPTER.validate_python({
                    "input_type": "mapping",
                    "metadata": dict(metadata.items()),
                })
            )

        if isinstance(metadata, BaseModel):
            try:
                dumped = metadata.model_dump()
            except Exception as e:
                msg = f"Failed to dump BaseModel metadata: {e.__class__.__name__}: {e}"
                return r[_MetadataInputUnion].fail(msg)

            if not FlextRuntime.is_dict_like(dumped):
                msg = (
                    f"metadata.model_dump() returned {dumped.__class__.__name__}, "
                    "expected dict"
                )
                return r[_MetadataInputUnion].fail(msg)

            return r[_MetadataInputUnion].ok(
                _METADATA_INPUT_ADAPTER.validate_python({
                    "input_type": "mapping",
                    "metadata": dict(dumped.items()),
                })
            )

        extracted = FlextDispatcher._extract_from_object_attributes(metadata)
        if extracted is not None:
            return r[_MetadataInputUnion].ok(
                _METADATA_INPUT_ADAPTER.validate_python({
                    "input_type": "mapping",
                    "metadata": dict(extracted.items()),
                })
            )

        return r[_MetadataInputUnion].fail(
            f"Invalid metadata type: {metadata.__class__.__name__}",
        )

    @staticmethod
    def _normalize_request(
        request: m.HandlerRegistrationRequest
        | BaseModel
        | Mapping[str, t.ConfigMapValue],
    ) -> r[m.HandlerRegistrationRequest]:
        """Normalize and validate request to strict HandlerRegistrationRequest model.

        Enforces Pydantic validation for all handler registrations, rejecting loose
        dictionaries in favor of structured, validated input.

        Args:
            request: Raw request (dict, model, or compatible type)

        Returns:
            r with validated HandlerRegistrationRequest or error

        """
        normalized = FlextDispatcher._normalize_registration_input(request)
        if normalized.is_failure:
            return r[m.HandlerRegistrationRequest].fail(
                normalized.error or "Request validation failed"
            )

        normalized_value = normalized.value
        return r[m.HandlerRegistrationRequest].ok(normalized_value.request)

    def _validate_and_extract_handler(
        self,
        request_model: m.HandlerRegistrationRequest,
    ) -> r[tuple[DispatcherHandler, str]]:
        """Validate handler and extract handler name.

        Args:
            request_model: Validated registration request

        Returns:
            r with (handler, handler_name) tuple or error

        """
        handler_raw = request_model.handler
        if not handler_raw:
            return r[tuple[DispatcherHandler, str]].fail(
                "Handler is required",
            )

        if self._is_dispatcher_handler(handler_raw):
            handler_typed: DispatcherHandler = handler_raw
        else:
            return r[tuple[DispatcherHandler, str]].fail(
                "Handler must be callable or BaseModel",
            )

        # Validate handler has required interface
        validation_result = self._validate_handler_registry_interface(
            handler_typed,
            handler_context="registered handler",
        )
        if validation_result.is_failure:
            return r[tuple[DispatcherHandler, str]].fail(
                validation_result.error or "Handler validation failed",
            )

        # Extract handler name - prefer explicit request name, fall back to handler instance
        handler_name = request_model.handler_name or self._extract_name_from_handler(
            handler_typed
        )
        if not handler_name:
            handler_name = "unknown_handler"

        # Return handler - already type narrowed to t.HandlerType
        return r[tuple[DispatcherHandler, str]].ok((handler_typed, handler_name))

    def _register_auto_discovery_handler(
        self,
        handler: DispatcherHandler,
        handler_name: str,
    ) -> r[m.HandlerRegistrationResult]:
        """Register handler with auto-discovery mode.

        Args:
            handler: Handler instance with can_handle() method
            handler_name: Handler name for tracking

        Returns:
            r with registration details

        """
        # handler is validated to have can_handle() before calling this function
        # Type narrowing: treat as t.HandlerType directly
        if self._is_dispatcher_handler(handler) and handler not in self._auto_handlers:
            self._auto_handlers.append(handler)

        return r[m.HandlerRegistrationResult].ok(
            m.HandlerRegistrationResult(
                handler_name=handler_name,
                status="registered",
                mode="auto_discovery",
            )
        )

    def _register_explicit_handler(
        self,
        handler: DispatcherHandler,
        handler_name: str,
        request_dict: Mapping[str, t.ConfigMapValue],
    ) -> r[m.HandlerRegistrationResult]:
        """Register handler with explicit mode.

        Args:
            handler: Handler instance (callable, mapping, or handler object)
            handler_name: Handler name for tracking
            request_dict: Normalized request dictionary

        Returns:
            r with registration details or error

        """
        message_type = request_dict.get("message_type")
        if not message_type:
            return r[m.HandlerRegistrationResult].fail(
                "Handler without can_handle() requires message_type",
            )

        name_attr = (
            (
                getattr(message_type, "__name__")
                if hasattr(message_type, "__name__")
                else None
            )
            if hasattr(message_type, "__name__")
            else None
        )
        message_type_name = name_attr if name_attr is not None else str(message_type)

        # Store handler in handlers dict - handler has been validated as callable/mapping
        # _handlers stores PayloadValue values (handler was validated but stored as-is)
        self._handlers[message_type_name] = handler

        return r[m.HandlerRegistrationResult].ok(
            m.HandlerRegistrationResult(
                handler_name=handler_name,
                message_type=message_type_name,
                status="registered",
                mode="explicit",
            )
        )

    def _register_handler_with_request(
        self,
        request: m.HandlerRegistrationRequest
        | BaseModel
        | Mapping[str, t.ConfigMapValue],
    ) -> r[m.HandlerRegistrationResult]:
        """Internal: Register handler using structured request model.

        Args:
            request: Dict or Pydantic model containing registration details.

        Returns:
            r with registration details or error

        """
        # Normalize and validate request
        request_result = FlextDispatcher._normalize_request(request)
        if request_result.is_failure:
            return r[m.HandlerRegistrationResult].fail(
                request_result.error or "Failed to normalize request",
            )
        request_model: m.HandlerRegistrationRequest = request_result.value

        # Validate handler mode (already validated by Pydantic model, but check logic constraints if any)
        # Assuming Pydantic handled enum validation
        handler_mode = request_model.handler_mode
        # No need for extra validation if model covers it, but keeping logic consistent
        # mode_validation = self._validate_handler_mode(handler_mode) ... if needed

        # Validate and extract handler
        handler_result = self._validate_and_extract_handler(request_model)
        if handler_result.is_failure:
            return r[m.HandlerRegistrationResult].fail(
                handler_result.error or "Handler validation failed",
            )
        handler_value: tuple[DispatcherHandler, str] = handler_result.value
        handler, handler_name = handler_value

        # Override handler name if provided in request
        if request_model.handler_name:
            handler_name = request_model.handler_name

        # Determine registration mode and register
        can_handle_attr = (
            getattr(handler, "can_handle") if hasattr(handler, "can_handle") else None
        )
        if callable(can_handle_attr):
            # Auto-discovery mode
            if handler not in self._auto_handlers:
                self._auto_handlers.append(handler)
            return r[m.HandlerRegistrationResult].ok(
                m.HandlerRegistrationResult(
                    handler_name=handler_name,
                    status="registered",
                    mode="auto_discovery",
                    handler_mode=handler_mode or c.Cqrs.HandlerType.COMMAND,
                )
            )

        # Explicit registration requires message_type
        message_type = request_model.message_type
        if not message_type:
            return r[m.HandlerRegistrationResult].fail(
                "Handler without can_handle() requires message_type",
            )

        # Get message type name and store handler
        name_attr = (
            getattr(message_type, "__name__")
            if hasattr(message_type, "__name__")
            else None
        )
        message_type_name = name_attr if name_attr is not None else str(message_type)
        self._handlers[message_type_name] = handler

        return r[m.HandlerRegistrationResult].ok(
            m.HandlerRegistrationResult(
                handler_name=handler_name,
                message_type=message_type_name,
                status="registered",
                mode="explicit",
                handler_mode=handler_mode or c.Cqrs.HandlerType.COMMAND,
            )
        )

    def register_handler(
        self,
        request: HandlerRegistrationRequestInput,
        handler: DispatcherHandler | None = None,
    ) -> r[m.HandlerRegistrationResult]:
        """Register a handler dynamically.

        Args:
            request: Dict, Pydantic model, message_type string, or handler object
                (h or any object that can act as handler)
            handler: Handler instance (optional, for two-arg registration)

        Returns:
            r with registration details or error

        """
        if handler is not None:
            # Two-arg mode: register_handler(command_type, handler)
            # request is command type (string or class), handler is the handler
            # Validate request is a valid command type (string or type)
            if not isinstance(request, (str, type)):
                return r[m.HandlerRegistrationResult].fail(
                    f"Invalid command type: {type(request).__name__}. Expected string or type.",
                )
            # Extract command name for registration
            if isinstance(request, str):
                command_name = request
            else:
                # request is callable (type or function) after validation above
                command_name = (
                    getattr(request, "__name__")
                    if hasattr(request, "__name__")
                    else str(request)
                )
            # Register the handler with command name
            # handler is HandlerType | PayloadValue - layer1 accepts both
            result = self._layer1_register_handler(command_name, handler)
            if result.is_failure:
                return r[m.HandlerRegistrationResult].fail(
                    result.error or "Registration failed",
                )
            return r[m.HandlerRegistrationResult].ok(
                m.HandlerRegistrationResult(
                    handler_name=command_name,
                    status="registered",
                    mode="explicit",
                )
            )

        # Single-arg mode: register_handler(dict_or_model_or_handler)
        # First check if request is a BaseModel
        if self._is_dispatcher_handler(request):
            result = self._layer1_register_handler(request)
            if result.is_failure:
                return r[m.HandlerRegistrationResult].fail(
                    result.error or "Registration failed",
                )
            handler_name = (
                getattr(request, "__class__")
                if hasattr(request, "__class__")
                else type(request)
            ).__name__
            return r[m.HandlerRegistrationResult].ok(
                m.HandlerRegistrationResult(
                    handler_name=handler_name,
                    status="registered",
                    mode="auto_discovery",
                )
            )

        if isinstance(request, BaseModel | Mapping):
            normalized_request_result = FlextDispatcher._normalize_request(
                cast(
                    "m.HandlerRegistrationRequest | BaseModel | Mapping[str, t.ConfigMapValue]",
                    request,
                )
            )
            if normalized_request_result.is_success:
                return self._register_handler_with_request(
                    normalized_request_result.value
                )

        return r[m.HandlerRegistrationResult].fail(
            f"Invalid registration request type: {type(request).__name__}",
        )

    def register_handlers(
        self,
        handlers: Mapping[HandlerRequestKey, DispatcherHandler],
    ) -> r[FlextModelsCqrs.HandlerBatchRegistrationResult]:
        """Register multiple handlers in batch.

        Args:
            handlers: Mapping of command type (string or class) to handler instances.
                      Example: {CreateUserCommand: CreateUserHandler(),
                               "delete_user": delete_user_handler}

        Returns:
            r with summary of registration results

        Example:
            >>> dispatcher = FlextDispatcher()
            >>> handlers = {
            ...     CreateUserCommand: CreateUserHandler(),
            ...     UpdateUserCommand: UpdateUserHandler(),
            ...     "delete_user": delete_user_handler,
            ... }
            >>> result = dispatcher.register_handlers(handlers)
            >>> if result.is_success:
            ...     print(f"Registered {len(handlers)} handlers")

        """
        registered: list[str] = []
        errors: list[str] = []

        for command_type, handler in handlers.items():
            # Extract command type name for display
            if isinstance(command_type, str):
                type_name = command_type
            else:
                # command_type is callable (type or function)
                type_name = (
                    getattr(command_type, "__name__")
                    if hasattr(command_type, "__name__")
                    else str(command_type)
                )

            # Register using two-arg form
            reg_result = self.register_handler(command_type, handler)
            if reg_result.is_success:
                registered.append(type_name)
            else:
                errors.append(f"{type_name}: {reg_result.error}")

        # Return summary result
        if errors:
            return r[FlextModelsCqrs.HandlerBatchRegistrationResult].fail(
                f"Some handlers failed to register: {'; '.join(errors)}",
            )

        return r[FlextModelsCqrs.HandlerBatchRegistrationResult].ok(
            FlextModelsCqrs.HandlerBatchRegistrationResult(
                status="registered",
                count=len(registered),
                handlers=registered,
            )
        )

    # ------------------------------------------------------------------
    def _check_pre_dispatch_conditions(
        self,
        message_type: str,
    ) -> r[bool]:
        """Check all pre-dispatch conditions (circuit breaker, rate limiting).

        Orchestrates multiple validation checks in sequence. Returns first failure
        encountered, or success if all checks pass.

        Args:
            message_type: Message type string for reliability pattern checks

        Returns:
            r[bool]: Success with True if all checks pass, failure if any check fails

        """
        # Check circuit breaker state
        if not self._circuit_breaker.check_before_dispatch(message_type):
            failures = self._circuit_breaker.get_failure_count(message_type)
            return r[bool].fail(
                f"Circuit breaker is open for message type '{message_type}'",
                error_code=c.Errors.OPERATION_ERROR,
                error_data=t.ConfigMap(
                    root={
                        "message_type": message_type,
                        "failure_count": failures,
                        "threshold": self._circuit_breaker.get_threshold(),
                        "state": self._circuit_breaker.get_state(message_type),
                        "reason": "circuit_breaker_open",
                    }
                ),
            )

        # Check rate limiting
        rate_limit_result = self._rate_limiter.check_rate_limit(message_type)
        if rate_limit_result.is_failure:
            error_msg = rate_limit_result.error or "Rate limit exceeded"
            return r[bool].fail(error_msg)

        return r[bool].ok(value=True)

    def _execute_with_timeout(
        self,
        execute_func: Callable[[], r[_Payload]],
        timeout_seconds: float,
        timeout_override: int | None = None,
    ) -> r[_Payload]:
        """Execute a function with timeout enforcement using executor or direct execution.

        Handles timeout errors gracefully. If executor is shutdown, reinitializes it.
        This helper encapsulates the timeout orchestration logic.

        Args:
            execute_func: Callable that returns r[_Payload]
            timeout_seconds: Timeout in seconds
            timeout_override: Optional timeout override (forces executor usage)

        Returns:
            r[_Payload]: Execution result or timeout error

        """
        use_executor = (
            self._timeout_enforcer.should_use_executor() or timeout_override is not None
        )

        if use_executor:
            executor = self._timeout_enforcer.ensure_executor()
            future: concurrent.futures.Future[r[_Payload]] | None = None
            try:
                future = executor.submit(execute_func)
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                # Cancel the future and return timeout error
                if future is not None:
                    _ = future.cancel()
                return r[_Payload].fail(
                    f"Operation timeout after {timeout_seconds} seconds",
                )
            except RuntimeError as exc:
                error_text = str(exc).lower()
                if "shutdown" in error_text or "cannot schedule" in error_text:
                    # Executor was shut down; reinitialize and allow caller to retry
                    self._timeout_enforcer.reset_executor()
                    return r[_Payload].fail(
                        "Executor was shutdown, retry requested",
                    )
                raise
            except Exception:
                # Preserve original exception for upstream handling
                raise
        else:
            return execute_func()

    def _track_timeout_context(
        self,
        operation_id: str,
        timeout_seconds: float,
    ) -> float:
        """Track timeout context and calculate deadline for operation.

        Propagates timeout context for deadline tracking and upstream
        cancellation support. Stores deadline for each operation.

        Args:
            operation_id: Unique operation identifier
            timeout_seconds: Timeout duration in seconds

        Returns:
            Deadline timestamp (current_time + timeout_seconds)

        """
        deadline = time.time() + timeout_seconds

        # Store timeout context with metadata
        self._timeout_contexts[operation_id] = {
            "operation_id": operation_id,
            "timeout_seconds": timeout_seconds,
            "deadline": deadline,
            "start_time": time.time(),
        }

        # Store deadline for quick lookup
        self._timeout_deadlines[operation_id] = deadline

        return deadline

    def _cleanup_timeout_context(self, operation_id: str) -> None:
        """Clean up timeout context after operation completes.

        Removes timeout context and deadline tracking for operation.
        Called after operation succeeds or fails.

        Args:
            operation_id: Operation identifier to clean up

        """
        _ = self._timeout_contexts.pop(operation_id, None)
        _ = self._timeout_deadlines.pop(operation_id, None)

    def _should_retry_on_error(
        self,
        attempt: int,
        error_message: str | None = None,
    ) -> bool:
        """Check if an error should trigger a retry attempt.

        Encapsulates retry eligibility logic and handles retry delay.

        Args:
            attempt: Current attempt number (0-indexed)
            error_message: Error message (for retriable error checking)

        Returns:
            True if should retry (delay applied), False if should not retry

        """
        # Check retry policy eligibility
        if not self._retry_policy.should_retry(attempt):
            return False

        # For r errors, check if error is retriable
        if error_message is not None and not self._retry_policy.is_retriable_error(
            error_message,
        ):
            return False

        # Delay before retry
        time.sleep(self._retry_policy.get_retry_delay())
        return True

    def dispatch(
        self,
        message_or_type: _Payload,
        data: _Payload | None = None,
        *,
        config: m.DispatchConfig | _Payload | None = None,
        metadata: _Payload | None = None,
        correlation_id: str | None = None,
        timeout_override: int | None = None,
    ) -> r[_Payload]:
        """Dispatch message.

        Args:
            message_or_type: Message object or type string
            data: Message data
            config: DispatchConfig instance or legacy config dict
            metadata: Optional execution context metadata (used if config is None)
            correlation_id: Optional correlation ID for tracing (used if config is None)
            timeout_override: Optional timeout override (used if config is None)

        Returns:
            r with execution result or error

        """
        # Detect API pattern - (type, data) vs (object)
        message: _Payload
        message_type_name_override: str | None = None
        if data is not None or isinstance(message_or_type, str):
            # dispatch("type", data) pattern
            message_type_str = str(message_or_type)
            message_type_name_override = message_type_str

            class DispatchPayload(BaseModel):
                payload: _Payload | None = None

            message = DispatchPayload(payload=data)
        else:
            # dispatch(message_object) pattern
            message = message_or_type

        # Fast fail: message cannot be None
        if message is None:
            return r[_Payload].fail(
                "Message cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Simple dispatch for registered handlers
        message_type = type(message)
        route_names = FlextDispatcher._resolve_message_route_names(message)
        if message_type_name_override is not None:
            route_names = [message_type_name_override, *route_names]

        handler_route_name = next(
            (route_name for route_name in route_names if route_name in self._handlers),
            None,
        )
        if handler_route_name is not None:
            try:
                handler_raw = self._handlers[handler_route_name]
                if not callable(handler_raw):
                    return r[_Payload].fail(
                        f"Handler for {message_type} is not callable",
                    )
                # Type narrowing: after callable() check, handler_raw is callable
                handler_input = FlextDispatcher._to_scalar_value(message)
                result_raw = handler_raw(handler_input)
                result = FlextDispatcher._to_container_value(result_raw)
                return r[_Payload].ok(result)
            except Exception as e:
                return r[_Payload].fail(str(e))

        # Build DispatchConfig from arguments if not provided
        dispatch_config = FlextDispatcher._build_dispatch_config_from_args(
            config,
            metadata,
            correlation_id,
            timeout_override,
        )

        # Full dispatch pipeline
        # DispatchConfig (BaseModel) is compatible with _Payload (includes BaseModel via Mapping)
        return self._execute_dispatch_pipeline(
            message,
            dispatch_config,
            metadata,
            correlation_id,
            timeout_override,
        )

    @staticmethod
    def _convert_metadata_to_model(
        metadata: _Payload | None,
    ) -> m.Metadata | None:
        """Convert metadata from _Payload to m.Metadata model.

        Args:
            metadata: Optional metadata value of various types

        Returns:
            Metadata model instance or None

        """
        if metadata is None:
            return None
        if isinstance(metadata, m.Metadata):
            return metadata
        metadata_input_result = FlextDispatcher._normalize_metadata_input(metadata)
        if metadata_input_result.is_failure:
            return (
                m.Metadata(attributes={"value": str(metadata)})
                if metadata is not None
                else None
            )

        metadata_input = metadata_input_result.value
        if metadata_input.input_type == "none":
            return None

        if metadata_input.input_type == "model":
            return metadata_input.metadata

        if metadata_input.input_type == "mapping":
            metadata_map = metadata_input.metadata
            attributes_dict: dict[str, t.MetadataAttributeValue] = {}

            def convert_metadata_value(
                v: _Payload,
            ) -> t.MetadataAttributeValue:
                if FlextDispatcher._is_scalar_primitive(v):
                    return v
                if isinstance(v, list):
                    # Use list comprehension for concise list conversion
                    return [
                        item
                        if FlextDispatcher._is_scalar_primitive(item)
                        else str(item)
                        for item in v
                    ]
                if isinstance(v, Mapping):
                    # Serialize nested dicts to JSON for Metadata.attributes compatibility.
                    # Metadata.attributes only accepts flat scalar values, not nested dicts.
                    return json.dumps({str(k): str(v2) for k, v2 in v.items()})
                return str(v)

            process_result = u.process(
                list(metadata_map.items()),
                lambda kv: (
                    str(kv[0]),
                    convert_metadata_value(FlextDispatcher._to_container_value(kv[1])),
                ),
                on_error="collect",
            )
            if process_result.is_success:
                processed_items: list[tuple[str, t.MetadataAttributeValue]] = (
                    process_result.value
                )
                attributes_dict = dict(processed_items)

            # attributes_dict is mapping[str, str] which is assignable to mapping[str, _Payload]
            return m.Metadata(attributes=attributes_dict)
        # Convert other types to Metadata via dict with string value
        return m.Metadata(attributes={"value": str(metadata)})

    @staticmethod
    def _build_dispatch_config_from_args(
        config: m.DispatchConfig | _Payload | None,
        metadata: _Payload | None,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> m.DispatchConfig | _Payload | None:
        """Build DispatchConfig from arguments if not provided.

        Args:
            config: DispatchConfig instance or legacy config dict
            metadata: Optional execution context metadata
            correlation_id: Optional correlation ID for tracing
            timeout_override: Optional timeout override

        Returns:
            DispatchConfig instance or original config

        """
        if hasattr(config, "model_dump"):  # Check if it's a Pydantic model
            return config
        if config is not None:
            return config
        if metadata is None and correlation_id is None and timeout_override is None:
            return None

        # Build from individual arguments
        metadata_model = FlextDispatcher._convert_metadata_to_model(metadata)
        # Create config dynamically since we can't reference the type
        return m.DispatchConfig(
            metadata=metadata_model,
            correlation_id=correlation_id,
            timeout_override=timeout_override,
        )

    def _execute_dispatch_pipeline(
        self,
        message: _Payload,
        config: _Payload | None,
        metadata: _Payload | None,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> r[_Payload]:
        """Execute full dispatch pipeline with validation and retry.

        Returns:
            r with execution result or error

        """
        # Extract dispatch config
        config_result = FlextDispatcher._extract_dispatch_config(
            config,
            metadata,
            correlation_id,
            timeout_override,
        )
        if config_result.is_failure:
            return r[_Payload].fail(
                config_result.error or "Config extraction failed",
            )

        # Cast value to expected type since we checked is_failure/None
        # Remove quotes for direct type reference
        dispatch_config = config_result.value

        context_result = self._prepare_dispatch_context(
            message,
            None,
            dispatch_config,
        )
        if context_result.is_failure or context_result.value is None:
            return r[_Payload].fail(
                context_result.error or "Context preparation failed",
            )

        # Cast value to expected type
        context = context_result.value

        # Validate pre-dispatch conditions
        validated_result = self._validate_pre_dispatch_conditions(
            context,
        )
        if validated_result.is_failure or validated_result.value is None:
            return r[_Payload].fail(
                validated_result.error or "Pre-dispatch validation failed",
            )

        # Execute with retry policy
        return self._execute_with_retry_policy(validated_result.value)

    @staticmethod
    def _extract_dispatch_config(
        config: _Payload | None,
        metadata: _Payload | None,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> r[m.DispatchConfig]:
        """Extract and validate dispatch configuration using strict Pydantic model."""
        try:
            # Extract config values (config takes priority over individual params)
            if config is not None:
                metadata = (
                    getattr(config, "metadata")
                    if hasattr(config, "metadata")
                    else metadata
                )
                correlation_id = (
                    getattr(config, "correlation_id")
                    if hasattr(config, "correlation_id")
                    else correlation_id
                )
                timeout_override = (
                    getattr(config, "timeout_override")
                    if hasattr(config, "timeout_override")
                    else timeout_override
                )

            metadata_input_result = FlextDispatcher._normalize_metadata_input(metadata)
            if metadata_input_result.is_failure:
                msg = metadata_input_result.error or "Invalid metadata input"
                return r[m.DispatchConfig].fail(msg)

            metadata_input = metadata_input_result.value
            validated_metadata: m.Metadata | None
            if metadata_input.input_type == "none":
                validated_metadata = None
            elif metadata_input.input_type == "model":
                validated_metadata = metadata_input.metadata
            else:
                normalized_attrs: dict[str, t.MetadataAttributeValue] = {}
                for k, v in metadata_input.metadata.items():
                    if not FlextDispatcher._is_metadata_attribute_compatible(
                        cast("_Payload", v)
                    ):
                        return r[m.DispatchConfig].fail(
                            f"Invalid metadata attribute type for key '{k}': {type(v).__name__}"
                        )
                    value = FlextDispatcher._to_container_value(v)
                    if not FlextDispatcher._is_metadata_attribute_compatible(value):
                        return r[m.DispatchConfig].fail(
                            f"Invalid metadata attribute type for key '{k}'"
                        )
                    normalized_attrs[str(k)] = FlextRuntime.normalize_to_metadata_value(
                        value,
                    )
                validated_metadata = m.Metadata(attributes=normalized_attrs)

            if metadata is not None and validated_metadata is None:
                msg = (
                    f"Invalid metadata type: {metadata.__class__.__name__}. "
                    "Expected m.Metadata or compatible dict"
                )
                return r[m.DispatchConfig].fail(msg)

            # Create DispatchConfig object
            dispatch_config = m.DispatchConfig(
                metadata=validated_metadata,
                correlation_id=correlation_id,
                timeout_override=timeout_override,
            )

            return r[m.DispatchConfig].ok(dispatch_config)
        except Exception as e:
            return r[m.DispatchConfig].fail(f"Configuration extraction failed: {e}")

    def _prepare_dispatch_context(
        self,
        message: _Payload,
        _data: _Payload | None,
        dispatch_config: m.DispatchConfig,
    ) -> r[_Payload]:
        """Prepare dispatch context with message normalization and context propagation.

        Fast fail: Only accepts message objects. _data parameter is ignored (kept for signature compatibility).
        """
        try:
            # Propagate context for distributed tracing
            dispatch_type = message.__class__.__name__
            self._propagate_context(f"dispatch_{dispatch_type}")

            # Normalize message and get type
            message, message_type = FlextDispatcher._normalize_dispatch_message(
                message,
                _data,
            )

            # Use Pydantic model dump instead of Mapping check
            dispatch_config_dict = dispatch_config.model_dump()

            context: m.ConfigMap = m.ConfigMap.model_construct(
                root={
                    **dispatch_config_dict,
                    "message": message,
                    "message_type": message_type,
                    "dispatch_type": dispatch_type,
                }
            )

            return r[_Payload].ok(context)

        except Exception as e:
            return r[_Payload].fail(
                f"Context preparation failed: {e}",
            )

    def _validate_pre_dispatch_conditions(
        self,
        context: _Payload,
    ) -> r[_Payload]:
        """Validate pre-dispatch conditions (circuit breaker + rate limiting)."""
        # Fast fail: context must be dict-like to access message_type
        if not FlextRuntime.is_dict_like(context):
            msg = f"Context must be dict-like, got {context.__class__.__name__}"
            return r[_Payload].fail(msg)
        # Type narrowing: type check above narrows context to Mapping for mypy
        message_type_raw = context.get("message_type")
        if not isinstance(message_type_raw, str):
            msg = f"Invalid message_type in context: {message_type_raw.__class__.__name__}, expected str"
            return r[_Payload].fail(msg)
        message_type: str = message_type_raw

        # Check pre-dispatch conditions (circuit breaker + rate limiting)
        conditions_check = self._check_pre_dispatch_conditions(message_type)
        if conditions_check.is_failure:
            # Extract error directly from conditions_check.error property
            error_msg = conditions_check.error or "Pre-dispatch conditions check failed"
            return r[_Payload].fail(
                error_msg,
                error_code=conditions_check.error_code,
                error_data=conditions_check.error_data,
            )

        return r[_Payload].ok(context)

    def _execute_with_retry_policy(
        self,
        context: _Payload,
    ) -> r[_Payload]:
        """Execute dispatch with retry policy using u."""
        # Fast fail: context must be dict-like to access values
        if not FlextRuntime.is_dict_like(context):
            msg = f"Context must be dict-like, got {context.__class__.__name__}"
            return r[_Payload].fail(msg)
        # Type narrowing: type check above narrows context to Mapping for mypy
        message = context.get("message")
        message_type_raw = context.get("message_type")
        if not isinstance(message_type_raw, str):
            msg = f"Invalid message_type in context: {message_type_raw.__class__.__name__}, expected str"
            return r[_Payload].fail(msg)
        message_type: str = message_type_raw

        metadata_raw = context.get("metadata")
        metadata: _Payload | None = (
            metadata_raw
            if (metadata_raw is None or FlextRuntime.is_dict_like(metadata_raw))
            else None
        )

        correlation_id_raw = context.get("correlation_id")
        correlation_id: str | None = (
            correlation_id_raw if isinstance(correlation_id_raw, str | None) else None
        )

        timeout_override_raw = context.get("timeout_override")
        timeout_override: int | None = (
            timeout_override_raw
            if isinstance(timeout_override_raw, int | None)
            else None
        )

        # Generate operation ID using u
        operation_id = u.generate_operation_id(
            message_type,
            message,
        )

        # Use u for retry execution
        options = m.ExecuteDispatchAttemptOptions(
            message_type=message_type,
            metadata=metadata,
            correlation_id=correlation_id,
            timeout_override=timeout_override,
            operation_id=operation_id,
        )
        # with_retry returns RuntimeResult - convert to FlextResult
        runtime_result = u.with_retry(
            lambda: self._execute_dispatch_attempt(message, options),
            max_attempts=self._retry_policy.get_max_attempts(),
            should_retry_func=self._should_retry_on_error,
            cleanup_func=lambda: self._cleanup_timeout_context(operation_id),
        )
        # Convert RuntimeResult to FlextResult
        if runtime_result.is_success:
            return r[_Payload].ok(runtime_result.value)
        return r[_Payload].fail(runtime_result.error or "Dispatch failed")

    @staticmethod
    def _normalize_dispatch_message(
        message: _Payload,
        _data: _Payload | None,
    ) -> tuple[_Payload, str]:
        """Normalize message and extract message type.

        Fast fail: Only accepts message objects. No support for (message_type, data) API.
        """
        # Fast fail: message cannot be None
        if message is None:
            msg = "Message cannot be None. Use dispatch(message_object), not dispatch(None, data)."
            raise TypeError(msg)

        # Fast fail: message cannot be string
        if isinstance(message, str):
            msg = (
                "String message_type not supported. "
                "Use dispatch(message_object), not dispatch('message_type', data)."
            )
            raise TypeError(msg)

        # Extract message type from message object
        message_type = FlextDispatcher._resolve_message_route_names(message)[0]
        return message, message_type

    def _get_timeout_seconds(self, timeout_override: int | None) -> float:
        """Get timeout seconds from config or override.

        Args:
            timeout_override: Optional timeout override

        Returns:
            float: Timeout in seconds

        """
        # Fast fail: timeout_seconds must be numeric
        timeout_raw = self.config.timeout_seconds
        # Type narrowing: config.timeout_seconds is always int | float, so convert directly
        timeout_seconds: float = float(timeout_raw)
        if timeout_override:
            timeout_seconds = float(timeout_override)
        return timeout_seconds

    def _create_execute_with_context(
        self,
        message: _Payload,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> Callable[[], r[_Payload]]:
        """Create execution function with context.

        Args:
            message: Message to execute
            correlation_id: Optional correlation ID
            timeout_override: Optional timeout override

        Returns:
            Callable that executes message with context

        """

        def execute_with_context() -> r[_Payload]:
            if correlation_id is not None or timeout_override is not None:
                context_metadata: m.ConfigMap = m.ConfigMap(root={})
                if timeout_override is not None:
                    context_metadata.root["timeout_override"] = timeout_override
                with self._context_scope(
                    context_metadata,
                    correlation_id,
                ):
                    return self._dispatch_command(message)
            return self._dispatch_command(message)

        return execute_with_context

    def _handle_dispatch_result(
        self,
        dispatch_result: r[_Payload],
        message_type: str,
    ) -> r[_Payload]:
        """Handle dispatch result with circuit breaker tracking.

        Args:
            dispatch_result: Result from dispatcher execution
            message_type: Message type for circuit breaker

        Returns:
            r[_Payload]: Processed result

        Raises:
            e.OperationError: If result is failure but error is None

        """
        if dispatch_result.is_failure:
            # Use unwrap_error() for type-safe str
            error_msg = dispatch_result.error or "Unknown error"
            if "Executor was shutdown" in error_msg:
                return r[_Payload].fail(error_msg)
            self._circuit_breaker.record_success(message_type)
            return r[_Payload].fail(error_msg)

        self._circuit_breaker.record_success(message_type)
        return r[_Payload].ok(dispatch_result.value)

    def _execute_dispatch_attempt(
        self,
        message: _Payload,
        options: m.ExecuteDispatchAttemptOptions,
    ) -> r[_Payload]:
        """Execute a single dispatch attempt with timeout."""
        try:
            # Create structured request
            if options.metadata is not None and FlextRuntime.is_dict_like(
                options.metadata
            ):
                metadata_root = (
                    getattr(options.metadata, "root")
                    if hasattr(options.metadata, "root")
                    else None
                )
                metadata_source = (
                    dict(metadata_root.items())
                    if isinstance(metadata_root, Mapping)
                    else dict(options.metadata.items())
                )
                metadata_map: dict[str, t.ConfigMapValue] = {
                    str(key): FlextDispatcher._to_container_value(value)
                    for key, value in metadata_source.items()
                }
                metadata_attrs: dict[str, t.MetadataAttributeValue] = {}
                for key, value in metadata_map.items():
                    metadata_attrs[str(key)] = FlextRuntime.normalize_to_metadata_value(
                        value,
                    )
                _ = m.Metadata(attributes=metadata_attrs)

            timeout_seconds = self._get_timeout_seconds(options.timeout_override)
            _ = self._track_timeout_context(options.operation_id, timeout_seconds)

            execute_with_context = self._create_execute_with_context(
                message,
                options.correlation_id,
                options.timeout_override,
            )

            dispatch_result = self._execute_with_timeout(
                execute_with_context,
                timeout_seconds,
                options.timeout_override,
            )

            return self._handle_dispatch_result(dispatch_result, options.message_type)

        except Exception as e:
            self._circuit_breaker.record_failure(options.message_type)
            return r[_Payload].fail(f"Dispatch error: {e}")

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    @staticmethod
    def _is_metadata_attribute_compatible(value: _Payload) -> bool:
        if FlextDispatcher._is_metadata_scalar(value):
            return True

        if isinstance(value, list):
            return all(FlextDispatcher._is_metadata_scalar(item) for item in value)

        if FlextRuntime.is_dict_like(value):
            metadata_root = getattr(value, "root") if hasattr(value, "root") else None
            mapping_value = (
                dict(metadata_root.items())
                if isinstance(metadata_root, Mapping)
                else dict(value.items())
            )

            for nested in mapping_value.values():
                if isinstance(nested, list):
                    if not all(
                        FlextDispatcher._is_metadata_scalar(item) for item in nested
                    ):
                        return False
                elif not FlextDispatcher._is_metadata_scalar(nested):
                    return False
            return True

        return False

    @staticmethod
    def _extract_from_object_attributes(
        metadata: _Payload,
    ) -> Mapping[str, object] | None:
        """Extract metadata mapping from object's attributes (internal helper)."""
        attributes_value = (
            getattr(metadata, "attributes") if hasattr(metadata, "attributes") else None
        )
        if attributes_value is not None and isinstance(attributes_value, Mapping):
            return dict(attributes_value)

        model_dump = (
            getattr(metadata, "model_dump") if hasattr(metadata, "model_dump") else None
        )
        if callable(model_dump):
            with suppress(Exception):
                dumped = model_dump()
                if isinstance(dumped, Mapping):
                    return dict(dumped)

        return None

    @contextmanager
    def _context_scope(
        self,
        metadata: _Payload | None = None,
        correlation_id: str | None = None,
    ) -> Generator[None]:
        """Manage execution context with optional metadata and correlation ID.

        Args:
            metadata: Optional metadata to include in context
            correlation_id: Optional correlation ID for tracing

        """
        if not self.config.dispatcher_auto_context:
            yield
            return

        metadata_var = FlextContext.Variables.OperationMetadata
        correlation_var = FlextContext.Variables.CorrelationId
        parent_var = FlextContext.Variables.ParentCorrelationId

        # Store current context values for restoration
        current_parent_value = parent_var.get()
        current_parent: str | None = (
            current_parent_value if isinstance(current_parent_value, str) else None
        )

        # Set new correlation ID if provided
        if correlation_id is not None:
            _ = correlation_var.set(correlation_id)
            # Set parent correlation ID if there was a previous one
            if current_parent is not None and current_parent != correlation_id:
                _ = parent_var.set(current_parent)

        metadata_input_result = FlextDispatcher._normalize_metadata_input(metadata)
        if metadata_input_result.is_success:
            metadata_input = metadata_input_result.value
            if metadata_input.input_type == "mapping":
                metadata_dict: m.ConfigMap = m.ConfigMap.model_validate(
                    metadata_input.metadata
                )
                _ = metadata_var.set(metadata_dict)
            elif metadata_input.input_type == "model":
                metadata_dict = m.ConfigMap.model_validate(
                    metadata_input.metadata.attributes,
                )
                _ = metadata_var.set(metadata_dict)

        effective_correlation_id = correlation_id
        if effective_correlation_id is None:
            effective_correlation_id = u.generate("correlation")

        if self.config.dispatcher_enable_logging:
            self._log_with_context(
                "debug",
                "dispatch_context_entered",
                correlation_id=effective_correlation_id,
            )

        yield

        if self.config.dispatcher_enable_logging:
            self._log_with_context(
                "debug",
                "dispatch_context_exited",
                correlation_id=effective_correlation_id,
            )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, *, auto_discover_handlers: bool = False) -> Self:
        """Factory method to create a new FlextDispatcher instance.

        This is the preferred way to instantiate FlextDispatcher. It provides
        a clean factory pattern that each class owns, respecting Clean
        Architecture principles where higher layers create their own instances.

        Auto-discovery of handlers discovers all functions marked with
        @h.handler() decorator in the calling module and registers them
        automatically. This enables zero-config handler registration for services.

        Args:
            auto_discover_handlers: If True, scan calling module for @handler()
                decorated functions and auto-register them. Default: False.

        Returns:
            FlextDispatcher instance.

        """
        instance = cls()

        if auto_discover_handlers:
            # Get the caller's frame to discover handlers in calling module
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_globals = frame.f_back.f_globals
                # Get module name from globals
                module_name = caller_globals.get("__name__", "__main__")
                # Get module object from globals (usually available as __import__ or direct reference)
                caller_module = sys.modules.get(module_name)
                if caller_module:
                    # Scan module for handler-decorated functions
                    handlers: list[
                        tuple[str, t.HandlerCallable, m.Handler.DecoratorConfig]
                    ] = h.Discovery.scan_module(caller_module)
                    for handler_item in handlers:
                        # Unpack tuple with explicit type hints to help type inference
                        _handler_name: str = handler_item[0]
                        handler_func: t.HandlerCallable = handler_item[1]
                        handler_config: m.Handler.DecoratorConfig = handler_item[2]
                        # Get actual handler function from module
                        # Check if handler_func is not None before checking callable
                        # Use TypeGuard for proper handler type validation
                        if handler_func is not None and cls._is_dispatcher_handler(
                            handler_func,
                        ):
                            # Register handler with dispatcher
                            # Register under the handler command type name for routing
                            # handler_config.command is type, never None (from typing)
                            command_type_name = (
                                handler_config.command.__name__
                                if hasattr(handler_config.command, "__name__")
                                else str(handler_config.command)
                            )
                            # handler_func is narrowed to t.HandlerType by TypeGuard
                            # register_handler accepts t.HandlerType | _Payload
                            _ = instance.register_handler(
                                command_type_name,
                                handler_func,
                            )

        return instance

    def cleanup(self) -> None:
        """Clean up dispatcher resources using processors."""
        try:
            # Clear all handlers from dispatcher's internal structures
            self._handlers.clear()
            self._auto_handlers.clear()
            self._event_subscribers.clear()

            # Clear internal state
            self._circuit_breaker.cleanup()
            self._rate_limiter.cleanup()
            self._timeout_enforcer.cleanup()
            self._retry_policy.cleanup()

        except Exception as e:
            self._log_with_context("warning", "Cleanup failed", error=str(e))


__all__ = ["FlextDispatcher"]
