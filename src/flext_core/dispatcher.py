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
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from types import ModuleType
from typing import Self, cast, override

from cachetools import LRUCache
from pydantic import BaseModel

from flext_core._dispatcher import (
    CircuitBreakerManager,
    DispatcherConfig,
    RateLimiterManager,
    RetryPolicy,
    TimeoutEnforcer,
)

# Use public aliases from m.* instead of private _models.* imports
from flext_core.constants import c
from flext_core.context import FlextContext
from flext_core.handlers import FlextHandlers
from flext_core.mixins import FlextMixins as x
from flext_core.models import FlextModels as m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
from flext_core.typings import t
from flext_core.utilities import u


class DispatcherMiddlewareConfig(m.Config.MiddlewareConfig):
    """Internal configuration for dispatcher middleware."""

    middleware_id: str
    middleware_type: str


class HandlerBatchRegistrationResult(BaseModel):
    """Result of batch handler registration."""

    status: str
    count: int
    handlers: list[str]


class FlextDispatcher(FlextService[bool]):
    """Application-level dispatcher that satisfies the command bus protocol.

    The dispatcher exposes CQRS routing, handler registration, and execution
    with layered reliability controls for message dispatching and handler execution.

    This is a specialized CQRS service that extends FlextService for infrastructure
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
        **data: t.GeneralValueType,
    ) -> None:
        """Initialize dispatcher with reliability managers.

        FlextService handles infrastructure (container, config, context) automatically.
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
            **data: Additional data passed to FlextService.

        """
        # FlextService handles container, config, context, runtime
        super().__init__(**data)

        # Access config from FlextService (already initialized)
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

        self._timeout_contexts: dict[
            str,
            t.GeneralValueType,
        ] = {}  # operation_id → context
        self._timeout_deadlines: dict[
            str,
            float,
        ] = {}  # operation_id → deadline timestamp

        # ================ LAYER 1: CQRS ROUTING INITIALIZATION ================

        # Handler registry (from FlextDispatcher dual-mode registration)
        self._handlers: dict[
            str,
            t.HandlerType,
        ] = {}  # Handler mappings by message type
        self._auto_handlers: list[t.HandlerType] = []  # Auto-discovery handlers

        # Middleware pipeline (from FlextDispatcher)
        self._middleware_configs: list[
            DispatcherMiddlewareConfig
        ] = []  # Config + ordering
        self._middleware_instances: dict[
            str,
            t.HandlerCallable,
        ] = {}  # Keyed by middleware_id

        # Query result caching (from FlextDispatcher - LRU cache)
        # Fast fail: use constant directly, no fallback
        max_cache_size = c.Container.MAX_CACHE_SIZE
        self._cache: LRUCache[str, r[t.GeneralValueType]] = LRUCache(
            maxsize=max_cache_size,
        )

        # Event subscribers (from FlextDispatcher event protocol)
        self._event_subscribers: dict[
            str,
            list[t.GeneralValueType],
        ] = {}  # event_type → handlers

        self._execution_count: int = 0

        # ============= LAYER 3: ADVANCED PROCESSING INITIALIZATION =============

        # Group 1: Handler Registry (internal dispatcher handler registry)
        self._handler_registry: dict[str, t.HandlerType] = {}  # name → handler function
        self._handler_configs: dict[
            str,
            m.Config.HandlerExecutionConfig,
        ] = {}  # name → handler config
        self._handler_validators: dict[
            str,
            Callable[[t.GeneralValueType], bool],
        ] = {}  # validation functions

        # Group 4: Pipeline (dispatcher-managed processing pipeline)
        self._pipeline_steps: list[
            dict[str, t.GeneralValueType]
        ] = []  # Ordered pipeline steps
        self._pipeline_composition: dict[
            str,
            Callable[
                [t.GeneralValueType],
                r[t.GeneralValueType],
            ],
        ] = {}  # composed functions
        self._pipeline_memo: dict[
            str,
            t.GeneralValueType,
        ] = {}  # Memoization cache for pipeline

        self._audit_log: list[
            dict[str, t.GeneralValueType]
        ] = []  # Operation audit trail
        self._performance_metrics: dict[
            str,
            t.GeneralValueType,
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
        threshold_raw = getattr(config, "circuit_breaker_threshold", None)
        threshold: int = threshold_raw if isinstance(threshold_raw, int) else 5
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
        max_requests_raw = getattr(config, "rate_limit_max_requests", None)
        max_requests: int = (
            max_requests_raw if isinstance(max_requests_raw, int) else 100
        )
        window_seconds_raw = getattr(config, "rate_limit_window_seconds", None)
        window_seconds: float = (
            window_seconds_raw if isinstance(window_seconds_raw, (int, float)) else 60.0
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
        use_timeout_executor = getattr(config, "enable_timeout_executor", False)
        executor_workers_raw = getattr(config, "executor_workers", None)
        executor_workers: int = (
            executor_workers_raw if isinstance(executor_workers_raw, int) else 4
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
        max_attempts_raw = getattr(config, "max_retry_attempts", None)
        max_attempts: int = max_attempts_raw if isinstance(max_attempts_raw, int) else 3
        retry_delay_raw = getattr(config, "retry_delay", None)
        retry_delay: float = (
            retry_delay_raw if isinstance(retry_delay_raw, (int, float)) else 1.0
        )
        return RetryPolicy(
            max_attempts=max_attempts,
            retry_delay=retry_delay,
        )

    @property
    def dispatcher_config(self) -> DispatcherConfig:
        """Access the dispatcher configuration."""
        config_dict = self.config.model_dump()
        # model_dump() always returns dict, which implements Mapping
        # No need to check isinstance - dict always implements Mapping
        # Construct DispatcherConfig TypedDict from config values
        return DispatcherConfig(
            dispatcher_timeout_seconds=u.Mapper.get(
                config_dict,
                "dispatcher_timeout_seconds",
                default=float(c.Defaults.TIMEOUT),
            )
            or float(c.Defaults.TIMEOUT),
            executor_workers=u.Mapper.get(
                config_dict,
                "executor_workers",
                default=c.Container.DEFAULT_WORKERS,
            )
            or c.Container.DEFAULT_WORKERS,
            circuit_breaker_threshold=u.Mapper.get(
                config_dict,
                "circuit_breaker_threshold",
                default=c.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
            )
            or c.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
            rate_limit_max_requests=u.Mapper.get(
                config_dict,
                "rate_limit_max_requests",
                default=c.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
            )
            or c.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
            rate_limit_window_seconds=u.Mapper.get(
                config_dict,
                "rate_limit_window_seconds",
                default=float(c.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS),
            )
            or float(c.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS),
            max_retry_attempts=u.Mapper.get(
                config_dict,
                "max_retry_attempts",
                default=c.Reliability.MAX_RETRY_ATTEMPTS,
            )
            or c.Reliability.MAX_RETRY_ATTEMPTS,
            retry_delay=u.Mapper.get(
                config_dict,
                "retry_delay",
                default=float(c.Reliability.DEFAULT_RETRY_DELAY_SECONDS),
            )
            or float(c.Reliability.DEFAULT_RETRY_DELAY_SECONDS),
            enable_timeout_executor=u.Mapper.get(
                config_dict,
                "enable_timeout_executor",
                default=True,
            )
            or True,
            dispatcher_enable_logging=u.Mapper.get(
                config_dict,
                "dispatcher_enable_logging",
                default=True,
            )
            or True,
            dispatcher_auto_context=u.Mapper.get(
                config_dict,
                "dispatcher_auto_context",
                default=True,
            )
            or True,
            dispatcher_enable_metrics=u.Mapper.get(
                config_dict,
                "dispatcher_enable_metrics",
                default=True,
            )
            or True,
        )

    # ==================== LAYER 3: ADVANCED PROCESSING INTERNAL METHODS ====================

    def _validate_interface(
        self,
        obj: (
            t.GeneralValueType
            | t.HandlerType
            | BaseModel
            | p.VariadicCallable[t.GeneralValueType]
            | p.Processor
        ),
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
            return r[bool].ok(True)

        # method_names is list[str] | str, convert to list[str]
        methods: list[str]
        if isinstance(method_names, list):
            methods = [str(m) for m in method_names]
        else:
            # Type narrowing: if not list, must be str
            methods = [method_names]
        for method_name in methods:
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                if callable(method):
                    return r[bool].ok(True)

        method_list = "' or '".join(methods)
        return r[bool].fail(f"Invalid {context}: must have '{method_list}' method")

    def _validate_processor_interface(
        self,
        processor: t.HandlerCallable
        | p.VariadicCallable[t.GeneralValueType]
        | p.Processor,
        processor_context: str = "processor",
    ) -> r[bool]:
        """Validate that processor has required interface (callable or process method)."""
        return self._validate_interface(
            processor,
            "process",
            processor_context,
            allow_callable=True,
        )

    def _validate_handler_registry_interface(
        self,
        handler: (
            t.HandlerType
            | t.HandlerCallable
            | p.VariadicCallable[t.GeneralValueType]
            | BaseModel
        ),
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

    def get_audit_log(self) -> r[list[dict[str, t.GeneralValueType]]]:
        """Retrieve operation audit trail.

        Returns:
            r[list[dict]]: Audit log entries with operation details

        """
        return r[list[dict[str, t.GeneralValueType]]].ok(
            self._audit_log.copy(),
        )

    def get_performance_analytics(self) -> r[t.GeneralValueType]:
        """Get comprehensive performance analytics.

        Returns:
            r[dict]: Performance analytics including timings and audit log count

        """
        analytics: t.GeneralValueType = {
            "performance_timings": self._performance_metrics.copy(),
            "audit_log_entries": len(self._audit_log),
        }
        return r[t.GeneralValueType].ok(analytics)

    # ==================== LAYER 1: CQRS ROUTING INTERNAL METHODS ====================

    @staticmethod
    def _normalize_command_key(
        command_type_obj: t.GeneralValueType | str,
    ) -> str:
        """Create comparable key for command identifiers (from FlextDispatcher).

        Args:
            command_type_obj: Object to create key from

        Returns:
            Normalized string key for command type

        """
        # Fast fail: __name__ should always exist on types, but handle gracefully
        # Use getattr for direct attribute access
        name_attr = getattr(command_type_obj, "__name__", None)
        if name_attr is not None:
            return str(name_attr)
        return str(command_type_obj)

    def _validate_handler_interface(
        self,
        handler: t.HandlerType,
        handler_context: str = "handler",
    ) -> r[bool]:
        """Validate that handler has required handle() interface."""
        method_name = c.Mixins.METHOD_HANDLE
        return self._validate_interface(handler, method_name, handler_context)

    def _validate_handler_mode(self, handler_mode: str | None) -> r[bool]:
        """Validate handler mode against CQRS types (consolidates register_handler duplication)."""
        if handler_mode is None:
            return r[bool].ok(True)

        # Type hint: HandlerType is StrEnum class, so __members__ exists
        # Use getattr for type attribute access (not mapper.get which is for dict/model access)
        # __members__ returns mappingproxy[str, HandlerType], which is compatible with HandlerTypeDict
        handler_type_members_raw: (
            Mapping[str, t.HandlerType] | dict[str, t.HandlerType]
        ) = getattr(c.Cqrs.HandlerType, "__members__", {})
        # __members__ returns mappingproxy[str, HandlerType], cast to HandlerTypeDict
        # HandlerTypeDict is dict[str, HandlerType], which matches __members__ structure
        handler_type_members: dict[str, t.HandlerType] = dict(handler_type_members_raw)

        def extract_handler_mode(m: object) -> str:
            """Extract string value from handler mode enum."""
            if isinstance(m, c.Cqrs.HandlerType):
                return m.value
            return str(m)

        valid_modes = list(
            u.map(
                list(handler_type_members.values()),
                extract_handler_mode,
            ),
        )
        if str(handler_mode) not in valid_modes:
            return r[bool].fail(
                f"Invalid handler_mode: {handler_mode}. Must be one of {valid_modes}",
            )

        return r[bool].ok(True)

    def _route_to_handler(
        self,
        command: t.GeneralValueType,
    ) -> t.HandlerType | object | None:
        """Locate the handler that can process the provided message.

        Args:
            command: The command/query object to find handler for

        Returns:
            The handler instance (HandlerType or object for FlextHandlers) or None if not found

        """
        command_type = type(command)
        command_name = command_type.__name__

        # First, try to find handler by command type name in _handlers
        # (two-arg registration)
        if command_name in self._handlers:
            # handler_entry is always HandlerType (HandlerCallable | BaseModel)
            # based on _handlers type definition - return directly
            return self._handlers[command_name]

        # Search auto-registered handlers (single-arg form)
        for handler in self._auto_handlers:
            # Fast fail: check if can_handle method exists before calling
            if hasattr(handler, "can_handle"):
                can_handle_method = getattr(handler, "can_handle", None)
                # can_handle expects str (message_type), not type
                if callable(can_handle_method) and can_handle_method(command_name):
                    return handler
        return None

    @staticmethod
    def _is_query(
        command: t.GeneralValueType,
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
        command: t.GeneralValueType,
        command_type: type[t.GeneralValueType],
    ) -> str:
        """Generate a deterministic cache key for the command.

        Args:
            command: The command/query object
            command_type: The type of the command

        Returns:
            str: Deterministic cache key

        """
        # generate_cache_key accepts *args: t.GeneralValueType, so pass command and command_type name as string
        command_type_name = command_type.__name__ if command_type else "unknown"
        # Pass command and command_type_name (string) as t.GeneralValueType-compatible arguments
        return u.generate_cache_key(command, command_type_name)

    def _check_cache_for_result(
        self,
        command: t.GeneralValueType,
        command_type: type,
        *,
        is_query: bool,
    ) -> r[t.GeneralValueType]:
        """Check cache for query result and return if found.

        Args:
            command: The command object
            command_type: The type of the command
            is_query: Whether command is a query

        Returns:
            r[t.GeneralValueType]: Cached result if found, failure if not cacheable or not cached

        """
        # Fast fail: use config value directly, no fallback
        cache_enabled = self.config.enable_caching
        should_consider_cache = cache_enabled and is_query
        if not should_consider_cache:
            return r[t.GeneralValueType].fail(
                "Cache not enabled or not a query",
                error_code=c.Errors.CONFIGURATION_ERROR,
            )

        cache_key = FlextDispatcher._generate_cache_key(command, command_type)
        cached_value = self._cache.get(cache_key)
        if cached_value is not None:
            # Fast fail: cached value must be r[t.GeneralValueType]
            # Type narrowing: cache stores r, so this is safe
            cached_result: r[t.GeneralValueType] = cached_value
            self.logger.debug(
                "Returning cached query result",
                operation="check_cache",
                command_type=command_type.__name__,
                cache_key=cache_key,
                source="flext-core/src/flext_core/dispatcher.py",
            )
            return cached_result

        return r[t.GeneralValueType].fail(
            "Cache miss",
            error_code=c.Errors.NOT_FOUND_ERROR,
        )

    def _execute_handler(
        self,
        handler: t.HandlerType | object,
        command: t.GeneralValueType,
        operation: str = c.Dispatcher.HANDLER_MODE_COMMAND,
    ) -> r[t.GeneralValueType]:
        """Execute the handler using h pipeline when available.

        Delegates to FlextHandlers._run_pipeline() for full CQRS support including
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
        handler_class_name = getattr(handler_class, "__name__", "Unknown")
        self.logger.debug(
            "Delegating to handler",
            operation="execute_handler",
            handler_type=handler_class_name,
            command_type=type(command).__name__,
            source="flext-core/src/flext_core/dispatcher.py",
        )

        # Delegate to FlextHandlers.dispatch_message() for full CQRS support
        if isinstance(handler, FlextHandlers):
            # Type narrowing: handler is FlextHandlers[MessageT_contra, ResultT]
            # dispatch_message returns r[ResultT], but ResultT is unknown at runtime
            # Since handler is generic and ResultT cannot be inferred, we cast the result
            # to r[t.GeneralValueType] for compatibility with return type
            # This is safe because t.GeneralValueType is the base type for all values
            return handler.dispatch_message(command, operation=operation)
            # Type is already r[t.GeneralValueType] - no cast needed

        # Fallback for non-h: try handle() then execute()
        # Use hasattr + getattr for attribute access
        method_name = None
        if hasattr(handler, c.Mixins.METHOD_HANDLE):
            handle_method = getattr(
                handler,
                c.Mixins.METHOD_HANDLE,
                None,
            )
            if callable(handle_method):
                method_name = c.Mixins.METHOD_HANDLE
        elif hasattr(handler, c.Mixins.METHOD_EXECUTE):
            execute_method = getattr(
                handler,
                c.Mixins.METHOD_EXECUTE,
                None,
            )
            if callable(execute_method):
                method_name = c.Mixins.METHOD_EXECUTE

        if not method_name:
            return r[t.GeneralValueType].fail(
                f"Handler must have '{c.Mixins.METHOD_HANDLE}' or '{c.Mixins.METHOD_EXECUTE}' method",
                error_code=c.Errors.COMMAND_BUS_ERROR,
            )
        # Use getattr for attribute access
        handle_method = getattr(handler, method_name, None)
        if not callable(handle_method):
            error_msg = f"Handler '{method_name}' must be callable"
            return r[t.GeneralValueType].fail(
                error_msg,
                error_code=c.Errors.COMMAND_BUS_ERROR,
            )

        try:
            result_raw = handle_method(command)
            # Ensure result is t.GeneralValueType - handlers should return t.GeneralValueType or FlextResult
            # Type narrowing: result_raw is t.GeneralValueType after isinstance check

            result: t.GeneralValueType
            if isinstance(
                result_raw,
                (str, int, float, bool, type(None), list, dict, Mapping, Sequence),
            ):
                result = result_raw
            else:
                result = str(result_raw)
            return x.ResultHandling.ensure_result(result)
        except Exception as exc:
            error_msg = f"Handler execution failed: {exc}"
            return r[t.GeneralValueType].fail(
                error_msg,
                error_code=c.Errors.COMMAND_PROCESSING_FAILED,
            )

    def _execute_middleware_chain(
        self,
        command: t.GeneralValueType,
        handler: t.HandlerType | object,
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
            return r[bool].ok(True)

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

        return r[bool].ok(True)

    @staticmethod
    def _get_middleware_order(
        middleware_config: DispatcherMiddlewareConfig,
    ) -> int:
        """Extract middleware execution order from config."""
        return middleware_config.order

    def _process_middleware_instance(
        self,
        command: t.GeneralValueType,
        handler: t.HandlerType | object,
        middleware_config: DispatcherMiddlewareConfig,
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
            return r[bool].ok(True)

        # Get actual middleware instance
        middleware = self._middleware_instances.get(middleware_id_str)
        if middleware is None:
            return r[bool].ok(True)

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
        command: t.GeneralValueType,
        handler: t.HandlerType | object,
        middleware_type: t.GeneralValueType,
    ) -> r[bool]:
        """Invoke middleware and handle result.

        Fast fail: Middleware must have process() method. No fallback to callable.
        """
        # Use getattr for object attribute access (middleware may be callable/class instance)
        process_method = getattr(middleware, "process", None)
        if not callable(process_method):
            return r[bool].fail(
                "Middleware must have callable 'process' method",
                error_code=c.Errors.CONFIGURATION_ERROR,
            )
        # Invoke middleware and get result
        result_raw = process_method(command, handler)

        # Ensure result is t.GeneralValueType or FlextResult
        if isinstance(
            result_raw,
            (
                str,
                int,
                float,
                bool,
                type(None),
                list,
                dict,
                Mapping,
                Sequence,
                r,
            ),
        ):
            result: t.GeneralValueType | r[t.GeneralValueType] = result_raw
        else:
            result = str(result_raw)
        return self._handle_middleware_result(result, middleware_type)

    def _handle_middleware_result(
        self,
        result: t.GeneralValueType | r[t.GeneralValueType],
        middleware_type: t.GeneralValueType,
    ) -> r[bool]:
        """Handle middleware execution result."""
        if isinstance(result, r) and result.is_failure:
            # Extract error directly from result.error property
            # result is r[t.GeneralValueType] after isinstance narrowing
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

        return r[bool].ok(True)

    # ==================== FLEXTSERVICE CONTRACT ====================

    @override
    def execute(self) -> r[bool]:
        """Execute service - satisfies FlextService abstract method.

        For FlextDispatcher, this indicates successful initialization.
        Use dispatch() for CQRS command/query routing.

        Returns:
            r[bool]: Success indicating dispatcher is ready.

        """
        return r[bool].ok(True)

    # ==================== LAYER 1 PUBLIC API: CQRS ROUTING & MIDDLEWARE ====================

    def _dispatch_command(
        self,
        command: t.GeneralValueType,
    ) -> r[t.GeneralValueType]:
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
            if isinstance(command, (dict, Mapping)):
                command_id_raw = command.get("command_id") or command.get("id")
                command_id_value = str(command_id_raw) if command_id_raw else "unknown"
            elif hasattr(command, "command_id"):
                command_id_value = str(getattr(command, "command_id", "unknown"))
            elif hasattr(command, "id"):
                command_id_value = str(getattr(command, "id", "unknown"))

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
                return r[t.GeneralValueType].fail(
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
                return r[t.GeneralValueType].fail(
                    middleware_result.error or "Unknown error",
                    error_code=c.Errors.COMMAND_BUS_ERROR,
                )

            # Execute handler with appropriate operation type
            operation = (
                c.Dispatcher.HANDLER_MODE_QUERY
                if is_query
                else c.Dispatcher.HANDLER_MODE_COMMAND
            )
            result: r[t.GeneralValueType] = self._execute_handler(
                handler,
                command,
                operation=operation,
            )

            # Cache successful query results
            cache_key: str | None = None
            if result.is_success and is_query:
                cache_key = FlextDispatcher._generate_cache_key(command, command_type)
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
        *args: t.HandlerType | t.GeneralValueType,
    ) -> r[bool]:
        """Internal: Register handler with dual-mode support.

        Supports:
        - Single-arg: register_handler(handler) - Auto-discovery with can_handle()
        - Two-arg: register_handler(MessageType, handler) - Explicit mapping

        Args:
            *args: Handler instance or (message_type, handler) pair

        Returns:
            r: Success or failure result

        """
        # Use isinstance for length validation (more type-safe than guard with callable)
        if len(args) == c.Dispatcher.SINGLE_HANDLER_ARG_COUNT:
            # Single arg should be a handler (callable or mapping)
            handler_arg: object = args[0]
            # Type narrowing: isinstance/callable for mypy (HandlerType = HandlerCallable | BaseModel)
            if callable(handler_arg) or isinstance(handler_arg, BaseModel):
                return self._register_single_handler(handler_arg)
            return r[bool].fail("Handler must be callable or BaseModel")
        if len(args) == c.Dispatcher.TWO_HANDLER_ARG_COUNT:
            # First arg is command type (string, class, or GeneralValueType)
            # Second arg is handler (callable or mapping)
            command_type_arg: object = args[0]
            second_handler_arg: object = args[1]
            # Handle command type - extract string name from callable or use as-is
            command_type: t.GeneralValueType | str
            if isinstance(command_type_arg, str):
                command_type = command_type_arg
            elif callable(command_type_arg) and not isinstance(
                command_type_arg,
                Mapping,
            ):
                # It's a callable (likely a class/type) - extract its name
                command_type = getattr(
                    command_type_arg,
                    "__name__",
                    str(command_type_arg),
                )
            else:
                # Other GeneralValueType - use as-is
                command_type = str(command_type_arg)
            # Type narrowing: TypeGuard narrows to t.HandlerType (HandlerCallable | BaseModel)
            if u.is_handler_type(second_handler_arg):
                return self._register_two_arg_handler(command_type, second_handler_arg)
            return r[bool].fail("Handler must be callable or BaseModel")

        return r[bool].fail(
            f"register_handler takes 1 or 2 arguments but {len(args)} were given",
        )

    def _wire_handler_dependencies(
        self,
        handler: t.HandlerType,
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
                handler_type=type(handler).__name__,
                exc_info=True,
            )

    def _register_single_handler(
        self,
        handler: t.HandlerType | None,
    ) -> r[bool]:
        """Register single handler for auto-discovery.

        Args:
            handler: Handler instance

        Returns:
            r with success or error

        """
        if handler is None:
            return r[bool].fail("Handler cannot be None")

        # handler is already HandlerType from method signature
        validation_result = self._validate_handler_interface(handler)
        if validation_result.is_failure:
            return validation_result

        self._wire_handler_dependencies(handler)
        self._auto_handlers.append(handler)

        handler_id = getattr(handler, "handler_id", None)
        handler_name = getattr(
            handler.__class__,
            "__name__",
            str(type(handler)),
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
        # Only instantiable classes (not functions) can be used as factories
        handler_cls: type[object] | None = None
        if inspect.isclass(handler):
            handler_cls = handler
        elif hasattr(handler, "__class__") and inspect.isclass(handler.__class__):
            handler_cls = handler.__class__

        if handler_cls is not None and self._container:
            try:
                # Create factory function that instantiates handler class
                # Capture in closure - class is always callable for instantiation
                cls_ref: type = handler_cls

                def _create_handler() -> t.GeneralValueType:
                    """Factory function to create handler instance."""
                    instance = cls_ref()
                    # Handler instances are valid GeneralValueType (BaseModel subclass)
                    if isinstance(instance, BaseModel):
                        return instance
                    # For non-BaseModel handlers, convert to string representation
                    return str(instance)

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

        return r[bool].ok(True)

    def _register_two_arg_handler(
        self,
        command_type_obj: t.GeneralValueType | str,
        handler: t.HandlerType,
    ) -> r[bool]:
        """Register handler with explicit command type.

        Args:
            command_type_obj: Command type object or string
            handler: Handler instance

        Returns:
            r with success or error

        """
        if handler is None or command_type_obj is None:
            return r[bool].fail(
                "Invalid arguments: command_type and handler are required",
            )

        if u.is_type(command_type_obj, str) and not u.is_string_non_empty(
            command_type_obj,
        ):
            return r[bool].fail("Command type cannot be empty")

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
        handler_name = getattr(handler.__class__, "__name__", str(type(handler)))

        # Register handler class as factory in container
        # Factory will create new handler instances when resolved
        # Only instantiable classes (not functions) can be used as factories
        handler_cls: type[object] | None = None
        if inspect.isclass(handler):
            handler_cls = handler
        elif hasattr(handler, "__class__") and inspect.isclass(handler.__class__):
            handler_cls = handler.__class__

        if handler_cls is not None and self._container:
            try:
                # Create factory function that instantiates handler class
                # Capture in closure - class is always callable for instantiation
                cls_ref: type = handler_cls

                def _create_handler_for_type() -> t.GeneralValueType:
                    """Factory function to create handler instance."""
                    instance = cls_ref()
                    # Handler instances are valid GeneralValueType (BaseModel subclass)
                    if isinstance(instance, BaseModel):
                        return instance
                    # For non-BaseModel handlers, convert to string representation
                    return str(instance)

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

        return r[bool].ok(True)

    def layer1_add_middleware(
        self,
        middleware: t.HandlerCallable,
        middleware_config: m.Config.MiddlewareConfig | t.ConfigMap | None = None,
    ) -> r[bool]:
        """Add middleware to processing pipeline (from FlextDispatcher).

        Args:
            middleware: The middleware instance to add
            middleware_config: Configuration for the middleware (model or dict)

        Returns:
            r: Success or failure result

        """
        # Strict Model Enforcement: Convert dict to Model if needed
        config_model: m.Config.MiddlewareConfig | None = None
        if middleware_config is not None:
            if isinstance(middleware_config, (dict, Mapping)):
                # Convert dict to model
                try:
                    config_model = m.Config.MiddlewareConfig.model_validate(
                        middleware_config
                    )
                except Exception as e:
                    return r[bool].fail(
                        f"Invalid middleware config: {e}",
                        error_code=c.Errors.CONFIGURATION_ERROR,
                    )
            elif isinstance(middleware_config, m.Config.MiddlewareConfig):
                config_model = middleware_config
            elif isinstance(middleware_config, t.ConfigMap):
                # Convert ConfigMap to MiddlewareConfig via validation
                try:
                    config_model = m.Config.MiddlewareConfig.model_validate(
                        middleware_config.model_dump()
                    )
                except Exception as e:
                    return r[bool].fail(
                        f"Invalid middleware config from ConfigMap: {e}",
                        error_code=c.Errors.CONFIGURATION_ERROR,
                    )
            else:
                return r[bool].fail(
                    "Middleware config must be dict, ConfigMap or MiddlewareConfig model",
                    error_code=c.Errors.CONFIGURATION_ERROR,
                )

        # Resolve middleware_id safely
        # config_model might be MiddlewareConfig (no middleware_id) or DispatcherMiddlewareConfig (has middleware_id)
        middleware_id_val: str | None = None
        if config_model and hasattr(config_model, "middleware_id"):
            middleware_id_val = str(getattr(config_model, "middleware_id"))

        if middleware_id_val is None:
            middleware_id_val = getattr(middleware, "__name__", str(middleware))

        middleware_id: str = str(middleware_id_val)

        # Resolve middleware type
        middleware_type_str = type(middleware).__name__

        # Create internal config wrapper
        final_config = DispatcherMiddlewareConfig(
            middleware_id=middleware_id,
            middleware_type=getattr(
                config_model, "middleware_type", middleware_type_str
            )
            if config_model and hasattr(config_model, "middleware_type")
            else middleware_type_str,
            enabled=config_model.enabled if config_model else True,
            order=config_model.order if config_model else len(self._middleware_configs),
            name=config_model.name if config_model else None,
            config=config_model.config if config_model else t.ConfigMap(root={}),
        )

        self._middleware_configs.append(final_config)
        self._middleware_instances[middleware_id] = middleware

        self.logger.info(
            "Middleware added to pipeline",
            operation="add_middleware",
            middleware_type=middleware_type_str,
            middleware_id=middleware_id,
            total_middleware=len(self._middleware_configs),
            source="flext-core/src/flext_core/dispatcher.py",
        )

        return r[bool].ok(True)

    # ==================== LAYER 1 EVENT PUBLISHING PROTOCOL ====================

    def _publish_event(self, event: t.GeneralValueType) -> r[bool]:
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

            return r[bool].ok(True)
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
            # layer1_register_handler accepts t.HandlerType | t.GeneralValueType
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
        _handler: t.GeneralValueType | None = None,
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
                return r[bool].ok(True)

            return r[bool].fail(f"Handler not found for event type: {event_type}")
        except (TypeError, KeyError, AttributeError) as e:
            # TypeError: invalid event_type
            # KeyError: event_type not registered
            # AttributeError: handler missing attributes
            self.logger.exception("Event unsubscription error")
            return r[bool].fail(f"Event unsubscription error: {e}")

    def publish(
        self,
        event: t.GeneralValueType | list[t.GeneralValueType],
        data: t.GeneralValueType | None = None,
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
        if isinstance(event, str) and data is not None:
            event_dict: dict[str, t.GeneralValueType] = {
                "event_name": event,
                "data": data,
                "timestamp": time.time(),
            }
            return self._publish_event(event_dict)

        # Handle batch events
        if isinstance(event, list):
            errors: list[str] = []
            for evt in event:
                result = self._publish_event(evt)
                if result.is_failure:
                    errors.append(result.error or "Unknown error")
            if errors:
                return r[bool].fail(f"Some events failed: {'; '.join(errors)}")
            return r[bool].ok(True)

        # Handle single event
        return self._publish_event(event)

    # ------------------------------------------------------------------
    # Registration methods using structured models
    # ------------------------------------------------------------------
    @staticmethod
    def _get_nested_attr(
        obj: t.GeneralValueType | t.HandlerType,
        *path: str,
    ) -> t.GeneralValueType | t.HandlerType | None:
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
        # Try extract for dict-like objects first (isinstance for proper type narrowing)
        if isinstance(obj, Mapping):
            path_str = ".".join(path)
            result = u.extract(obj, path_str, default=None, required=False)
            # Use .value directly - FlextResult never returns None on success
            if result.is_success:
                return result.value
        # Fall back to attribute access for objects
        current = obj
        for attr in path:
            if not hasattr(current, attr):
                return None
            current = getattr(current, attr, None)
            if current is None:
                return None
        return current

    @staticmethod
    def _extract_name_from_handler(
        handler: t.GeneralValueType | t.HandlerType,
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
    def _normalize_request(
        request: t.GeneralValueType | BaseModel,
    ) -> r[m.HandlerRegistrationRequest]:
        """Normalize and validate request to strict HandlerRegistrationRequest model.

        Enforces Pydantic validation for all handler registrations, rejecting loose
        dictionaries in favor of structured, validated input.

        Args:
            request: Raw request (dict, model, or compatible type)

        Returns:
            r with validated HandlerRegistrationRequest or error

        """
        if isinstance(request, m.HandlerRegistrationRequest):
            return r[m.HandlerRegistrationRequest].ok(request)

        try:
            if isinstance(request, BaseModel):
                # Convert other models via dump-load to validate fields
                # This handles field aliasing and conversion automatically
                request_data = request.model_dump()
                return r[m.HandlerRegistrationRequest].ok(
                    m.HandlerRegistrationRequest(**request_data)
                )

            if FlextRuntime.is_dict_like(request):
                # Validate raw dictionary against model schema
                # Pydantic v2 handles validation and type coercion
                return r[m.HandlerRegistrationRequest].ok(
                    m.HandlerRegistrationRequest(**request)
                )

        except Exception as e:
            msg = f"Request validation failed: {e!s}"
            return r[m.HandlerRegistrationRequest].fail(msg)

        return r[m.HandlerRegistrationRequest].fail(
            f"Invalid request type: {type(request).__name__}. Expected dict or HandlerRegistrationRequest.",
        )

    def _validate_and_extract_handler(
        self,
        request_model: m.HandlerRegistrationRequest,
    ) -> r[tuple[t.HandlerType, str]]:
        """Validate handler and extract handler name.

        Args:
            request_model: Validated registration request

        Returns:
            r with (handler, handler_name) tuple or error

        """
        handler_raw: t.GeneralValueType = request_model.handler
        if not handler_raw:
            return r[tuple[t.HandlerType, str]].fail(
                "Handler is required",
            )

        # Type narrowing using TypeGuard for handler validation
        if not u.is_handler_type(handler_raw):
            return r[tuple[t.HandlerType, str]].fail(
                "Handler must be callable, mapping, or BaseModel",
            )

        # After TypeGuard, handler_raw is narrowed to t.HandlerType (HandlerCallable | BaseModel)
        handler_typed: t.HandlerType = handler_raw

        # Validate handler has required interface
        validation_result = self._validate_handler_registry_interface(
            handler_typed,
            handler_context="registered handler",
        )
        if validation_result.is_failure:
            return r[tuple[t.HandlerType, str]].fail(
                validation_result.error or "Handler validation failed",
            )

        # Extract handler name - prefer explicit request name, fall back to handler instance
        handler_name = request_model.handler_name or self._extract_name_from_handler(
            handler_typed
        )
        if not handler_name:
            handler_name = "unknown_handler"

        # Return handler - already type narrowed to t.HandlerType
        return r[tuple[t.HandlerType, str]].ok((handler_typed, handler_name))

    def _register_handler_by_mode(
        self,
        handler: t.HandlerType,
        handler_name: str,
        request_dict: dict[str, t.GeneralValueType],
    ) -> r[m.HandlerRegistrationResult]:
        """Register handler based on auto-discovery or explicit mode.

        Args:
            handler: Validated handler instance (callable, mapping, or handler object)
            handler_name: Extracted handler name
            request_dict: Normalized request dictionary

        Returns:
            r with registration details or error

        """
        can_handle_attr = (
            getattr(handler, "can_handle", None)
            if hasattr(handler, "can_handle")
            else None
        )
        has_can_handle = callable(can_handle_attr)

        if has_can_handle:
            return self._register_auto_discovery_handler(handler, handler_name)

        return self._register_explicit_handler(handler, handler_name, request_dict)

    def _register_auto_discovery_handler(
        self,
        handler: t.HandlerType,
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
        if u.is_handler_type(handler) and handler not in self._auto_handlers:
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
        handler: t.HandlerType,
        handler_name: str,
        request_dict: dict[str, t.GeneralValueType],
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
            getattr(message_type, "__name__", None)
            if hasattr(message_type, "__name__")
            else None
        )
        message_type_name = name_attr if name_attr is not None else str(message_type)

        # Store handler in handlers dict - handler has been validated as callable/mapping
        # _handlers stores GeneralValueType values (handler was validated but stored as-is)
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
        request: t.GeneralValueType | BaseModel,
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
        handler_value: tuple[t.HandlerType, str] = handler_result.value
        handler, handler_name = handler_value

        # Override handler name if provided in request
        if request_model.handler_name:
            handler_name = request_model.handler_name

        # Determine registration mode and register
        can_handle_attr = getattr(handler, "can_handle", None)
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
        name_attr = getattr(message_type, "__name__", None)
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
        request: t.GeneralValueType | BaseModel | object,
        handler: t.HandlerType | t.GeneralValueType | None = None,
    ) -> r[m.HandlerRegistrationResult]:
        """Register a handler dynamically.

        Args:
            request: Dict, Pydantic model, message_type string, or handler object
                (FlextHandlers or any object that can act as handler)
            handler: Handler instance (optional, for two-arg registration)

        Returns:
            r with registration details or error

        """
        if handler is not None:
            # Two-arg mode: register_handler(command_type, handler)
            # request is command type (string or class), handler is the handler
            # Validate handler is HandlerType
            if not u.is_handler_type(handler):
                return r[m.HandlerRegistrationResult].fail(
                    f"Invalid handler type: {type(handler).__name__}",
                )
            # Validate request is a valid command type (string, type, or callable)
            if not isinstance(request, (str, type)) and not callable(request):
                return r[m.HandlerRegistrationResult].fail(
                    f"Invalid command type: {type(request).__name__}. Expected string or type.",
                )
            # Extract command name for registration
            if isinstance(request, str):
                command_name = request
            else:
                # request is callable (type or function) after validation above
                command_name = getattr(request, "__name__", str(request))
            # Register the handler with command name
            # handler is HandlerType | GeneralValueType - layer1 accepts both
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
        if isinstance(request, BaseModel):
            return self._register_handler_with_request(request)
        # Check if request is a dict-like GeneralValueType
        if isinstance(request, dict):
            # dict is both GeneralValueType and dict-like
            return self._register_handler_with_request(request)
        if isinstance(request, Mapping):
            # Mapping is both GeneralValueType and dict-like
            return self._register_handler_with_request(request)

        # Single handler object - delegate to layer1_register_handler
        # At this point, dict/Mapping/BaseModel have been handled above
        # Only callable handlers remain
        if not callable(request):
            return r[m.HandlerRegistrationResult].fail(
                f"Invalid handler type: {type(request).__name__}",
            )
        # After callable() check, request is a callable handler
        result = self._layer1_register_handler(cast("t.HandlerType", request))
        if result.is_failure:
            return r[m.HandlerRegistrationResult].fail(
                result.error or "Registration failed",
            )
        # Convert to dict format for consistency
        handler_name = getattr(request, "__class__", type(request)).__name__
        return r[m.HandlerRegistrationResult].ok(
            m.HandlerRegistrationResult(
                handler_name=handler_name,
                status="registered",
                mode="auto_discovery",
            )
        )

    def register_handlers(
        self,
        handlers: Mapping[str | type, t.HandlerType],
    ) -> r[HandlerBatchRegistrationResult]:
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
                type_name = getattr(command_type, "__name__", str(command_type))

            # Register using two-arg form
            reg_result = self.register_handler(command_type, handler)
            if reg_result.is_success:
                registered.append(type_name)
            else:
                errors.append(f"{type_name}: {reg_result.error}")

        # Return summary result
        if errors:
            return r[HandlerBatchRegistrationResult].fail(
                f"Some handlers failed to register: {'; '.join(errors)}",
            )

        return r[HandlerBatchRegistrationResult].ok(
            HandlerBatchRegistrationResult(
                status="registered",
                count=len(registered),
                handlers=registered,
            )
        )

    @staticmethod
    def _ensure_handler(
        handler: t.GeneralValueType
        | FlextHandlers[t.GeneralValueType, t.GeneralValueType],
        mode: c.Cqrs.HandlerType = c.Cqrs.HandlerType.COMMAND,
    ) -> r[
        FlextHandlers[
            t.GeneralValueType,
            t.GeneralValueType,
        ]
    ]:
        """Ensure handler is a FlextHandlers instance, converting from callable if needed.

        Private helper to eliminate duplication in handler registration.

        Args:
            handler: Handler instance or callable to convert
            mode: Handler operation mode (command/query)

        Returns:
            r with h instance or error

        """
        del mode  # Was used for callable conversion, now handled upstream
        # If already FlextHandlers, return success
        if isinstance(handler, FlextHandlers):
            return r[
                FlextHandlers[
                    t.GeneralValueType,
                    t.GeneralValueType,
                ]
            ].ok(handler)

        # Invalid handler type
        return r[
            FlextHandlers[
                t.GeneralValueType,
                t.GeneralValueType,
            ]
        ].fail(
            (
                f"Handler must be FlextHandlers instance or callable, got {type(handler).__name__}"
            ),
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
                error_data={
                    "message_type": message_type,
                    "failure_count": failures,
                    "threshold": self._circuit_breaker.get_threshold(),
                    "state": self._circuit_breaker.get_state(message_type),
                    "reason": "circuit_breaker_open",
                },
            )

        # Check rate limiting
        rate_limit_result = self._rate_limiter.check_rate_limit(message_type)
        if rate_limit_result.is_failure:
            error_msg = rate_limit_result.error or "Rate limit exceeded"
            return r[bool].fail(error_msg)

        return r[bool].ok(True)

    def _execute_with_timeout(
        self,
        execute_func: Callable[[], r[t.GeneralValueType]],
        timeout_seconds: float,
        timeout_override: int | None = None,
    ) -> r[t.GeneralValueType]:
        """Execute a function with timeout enforcement using executor or direct execution.

        Handles timeout errors gracefully. If executor is shutdown, reinitializes it.
        This helper encapsulates the timeout orchestration logic.

        Args:
            execute_func: Callable that returns r[t.GeneralValueType]
            timeout_seconds: Timeout in seconds
            timeout_override: Optional timeout override (forces executor usage)

        Returns:
            r[t.GeneralValueType]: Execution result or timeout error

        """
        use_executor = (
            self._timeout_enforcer.should_use_executor() or timeout_override is not None
        )

        if use_executor:
            executor = self._timeout_enforcer.ensure_executor()
            future: concurrent.futures.Future[r[t.GeneralValueType]] | None = None
            try:
                future = executor.submit(execute_func)
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                # Cancel the future and return timeout error
                if future is not None:
                    _ = future.cancel()
                return r[t.GeneralValueType].fail(
                    f"Operation timeout after {timeout_seconds} seconds",
                )
            except RuntimeError as exc:
                error_text = str(exc).lower()
                if "shutdown" in error_text or "cannot schedule" in error_text:
                    # Executor was shut down; reinitialize and allow caller to retry
                    self._timeout_enforcer.reset_executor()
                    return r[t.GeneralValueType].fail(
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

    def _check_timeout_deadline(self, operation_id: str) -> bool:
        """Check if operation timeout deadline has been exceeded.

        Enables upstream timeout cancellation by checking current deadline.

        Args:
            operation_id: Operation identifier to check

        Returns:
            True if deadline exceeded, False if still within timeout window

        """
        deadline = self._timeout_deadlines.get(operation_id)
        if deadline is None:
            return False
        return time.time() > deadline

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
        message_or_type: t.GeneralValueType,
        data: t.GeneralValueType | None = None,
        *,
        config: m.DispatchConfig | t.GeneralValueType | None = None,
        metadata: t.GeneralValueType | None = None,
        correlation_id: str | None = None,
        timeout_override: int | None = None,
    ) -> r[t.GeneralValueType]:
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
        message: t.GeneralValueType
        if data is not None or u.is_type(message_or_type, str):
            # dispatch("type", data) pattern
            message_type_str = str(message_or_type)
            message_class = type(message_type_str, (), {"payload": data})
            message_raw = message_class()
            # Safe cast: dynamically created class instance is compatible with t.GeneralValueType
            message = message_raw
        else:
            # dispatch(message_object) pattern
            message = message_or_type

        # Fast fail: message cannot be None
        if message is None:
            return r[t.GeneralValueType].fail(
                "Message cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Simple dispatch for registered handlers
        message_type = type(message)
        message_type_name = message_type.__name__
        if message_type_name in self._handlers:
            try:
                handler_raw = self._handlers[message_type_name]
                if not callable(handler_raw):
                    return r[t.GeneralValueType].fail(
                        f"Handler for {message_type} is not callable",
                    )
                # Type narrowing: after callable() check, handler_raw is callable
                result_raw = handler_raw(message)
                # Ensure result is t.GeneralValueType
                result: t.GeneralValueType
                if isinstance(
                    result_raw,
                    (str, int, float, bool, type(None), list, dict, Mapping, Sequence),
                ):
                    result = result_raw
                else:
                    result = str(result_raw)
                return r[t.GeneralValueType].ok(result)
            except Exception as e:
                return r[t.GeneralValueType].fail(str(e))

        # Build DispatchConfig from arguments if not provided
        dispatch_config = FlextDispatcher._build_dispatch_config_from_args(
            config,
            metadata,
            correlation_id,
            timeout_override,
        )

        # Full dispatch pipeline
        # DispatchConfig (BaseModel) is compatible with t.GeneralValueType (includes BaseModel via Mapping)
        return self._execute_dispatch_pipeline(
            message,
            dispatch_config,
            metadata,
            correlation_id,
            timeout_override,
        )

    @staticmethod
    def _convert_metadata_to_model(
        metadata: t.GeneralValueType | None,
    ) -> m.Metadata | None:
        """Convert metadata from t.GeneralValueType to m.Metadata model.

        Args:
            metadata: Optional metadata value of various types

        Returns:
            Metadata model instance or None

        """
        if metadata is None:
            return None
        if isinstance(metadata, m.Metadata):
            return metadata
        # Use guard and process_dict for concise metadata conversion
        if u.is_type(metadata, "mapping"):

            def convert_metadata_value(
                v: t.GeneralValueType,
            ) -> t.MetadataAttributeValue:
                if isinstance(v, (str, int, float, bool, type(None))):
                    return v
                if isinstance(v, list):
                    # Use list comprehension for concise list conversion
                    return [
                        item
                        if isinstance(item, (str, int, float, bool, type(None)))
                        else str(item)
                        for item in v
                    ]
                if u.is_type(v, "mapping") and isinstance(v, dict):
                    # Serialize nested dicts to JSON for Metadata.attributes compatibility.
                    # Metadata.attributes only accepts flat scalar values, not nested dicts.
                    return json.dumps({str(k): str(v2) for k, v2 in v.items()})
                return str(v)

            # Convert metadata dict to items for processing
            if isinstance(metadata, dict):
                process_result = u.process(
                    list(metadata.items()),
                    lambda kv: (str(kv[0]), convert_metadata_value(kv[1])),
                    on_error="collect",
                )
                if process_result.is_success:
                    # process_result.value is list[tuple[str, converted_value]]
                    processed_items: list[tuple[str, t.MetadataAttributeValue]] = (
                        process_result.value
                    )
                    attributes_dict = dict(processed_items)
                else:
                    attributes_dict = {}
            else:
                attributes_dict = {}

            # attributes_dict is dict[str, str] which is assignable to dict[str, t.GeneralValueType]
            return m.Metadata(attributes=attributes_dict)
        # Convert other types to Metadata via dict with string value
        return m.Metadata(attributes={"value": str(metadata)})

    @staticmethod
    def _build_dispatch_config_from_args(
        config: m.DispatchConfig | t.GeneralValueType | None,
        metadata: t.GeneralValueType | None,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> m.DispatchConfig | t.GeneralValueType | None:
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

    def _try_simple_dispatch(
        self,
        message: t.GeneralValueType,
    ) -> r[t.GeneralValueType] | None:
        """Try simple dispatch for registered handlers.

        Returns:
            r if handler found and executed, None otherwise

        """
        message_type = type(message)
        message_type_key = (
            message_type.__name__
            if hasattr(message_type, "__name__")
            else str(message_type)
        )
        if message_type_key not in self._handlers:
            return None

        try:
            handler_raw = self._handlers[message_type_key]
            if not callable(handler_raw):
                return r[t.GeneralValueType].fail(
                    f"Handler for {message_type_key} is not callable",
                )
            # Type narrowing: after callable() check, handler_raw is callable
            # Handler may return GeneralValueType or FlextResult directly
            handler_result: t.GeneralValueType | r[t.GeneralValueType] = handler_raw(
                message,
            )
            # Handle case where handler returns a FlextResult directly
            if isinstance(handler_result, r):
                return handler_result
            return r[t.GeneralValueType].ok(handler_result)
        except Exception as e:
            return r[t.GeneralValueType].fail(str(e))

    def _execute_dispatch_pipeline(
        self,
        message: t.GeneralValueType,
        config: t.GeneralValueType | None,
        metadata: t.GeneralValueType | None,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> r[t.GeneralValueType]:
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
        if config_result.is_failure or config_result.value is None:
            return r[t.GeneralValueType].fail(
                config_result.error or "Config extraction failed",
            )

        # Cast value to expected type since we checked is_failure/None
        # Remove quotes for direct type reference
        dispatch_config = cast("m.DispatchConfig", config_result.value)

        context_result = self._prepare_dispatch_context(
            message,
            None,
            dispatch_config,
        )
        if context_result.is_failure or context_result.value is None:
            return r[t.GeneralValueType].fail(
                context_result.error or "Context preparation failed",
            )

        # Cast value to expected type
        context = context_result.value

        # Validate pre-dispatch conditions
        validated_result = self._validate_pre_dispatch_conditions(
            context,
        )
        if validated_result.is_failure or validated_result.value is None:
            return r[t.GeneralValueType].fail(
                validated_result.error or "Pre-dispatch validation failed",
            )

        # Execute with retry policy
        return self._execute_with_retry_policy(validated_result.value)

    @staticmethod
    def _extract_dispatch_config(
        config: t.GeneralValueType | None,
        metadata: t.GeneralValueType | None,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> r[m.DispatchConfig]:
        """Extract and validate dispatch configuration using strict Pydantic model."""
        try:
            # Extract config values (config takes priority over individual params)
            if config is not None:
                metadata = getattr(config, "metadata", metadata)
                correlation_id = getattr(config, "correlation_id", correlation_id)
                timeout_override = getattr(config, "timeout_override", timeout_override)

            # Validate metadata - NO fallback, explicit validation
            validated_metadata: m.Metadata | None = None

            if metadata is None:
                validated_metadata = None
            elif isinstance(metadata, m.Metadata):
                validated_metadata = metadata
            elif isinstance(metadata, (Mapping, t.ConfigMap)):
                # Convert to dict for validation
                meta_dict = (
                    metadata.model_dump()
                    if isinstance(metadata, t.ConfigMap)
                    else dict(metadata)
                )
                try:
                    # Strict validation using Pydantic
                    validated_metadata = m.Metadata.model_validate(meta_dict)
                except Exception:
                    # Fallback: treat as attributes if strict validation fails
                    # Creates metadata with provided dict as 'attributes' field
                    # This handles legacy cases where raw dict was passed as metadata
                    try:
                        validated_metadata = m.Metadata(attributes=meta_dict)
                    except Exception:
                        pass  # Validated metadata remains None, will cause validation error in DispatchConfig if required
            else:
                # Fast fail: invalid type
                msg = (
                    f"Invalid metadata type: {type(metadata).__name__}. "
                    "Expected object | m.Metadata | None"
                )
                return r[m.DispatchConfig].fail(msg)

            if metadata is not None and validated_metadata is None:
                msg = (
                    f"Invalid metadata type: {type(metadata).__name__}. "
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
        message: t.GeneralValueType,
        _data: t.GeneralValueType | None,
        dispatch_config: m.DispatchConfig,
    ) -> r[t.GeneralValueType]:
        """Prepare dispatch context with message normalization and context propagation.

        Fast fail: Only accepts message objects. _data parameter is ignored (kept for signature compatibility).
        """
        try:
            # Propagate context for distributed tracing
            dispatch_type = type(message).__name__
            self._propagate_context(f"dispatch_{dispatch_type}")

            # Normalize message and get type
            message, message_type = FlextDispatcher._normalize_dispatch_message(
                message,
                _data,
            )

            # Use Pydantic model dump instead of Mapping check
            dispatch_config_dict = dispatch_config.model_dump()

            context: dict[str, t.GeneralValueType] = {
                **dispatch_config_dict,
                "message": message,
                "message_type": message_type,
                "dispatch_type": dispatch_type,
            }

            return r[t.GeneralValueType].ok(context)

        except Exception as e:
            return r[t.GeneralValueType].fail(
                f"Context preparation failed: {e}",
            )

    def _validate_pre_dispatch_conditions(
        self,
        context: t.GeneralValueType,
    ) -> r[t.GeneralValueType]:
        """Validate pre-dispatch conditions (circuit breaker + rate limiting)."""
        # Fast fail: context must be dict-like to access message_type
        if not isinstance(context, Mapping):
            msg = f"Context must be dict-like, got {type(context).__name__}"
            return r[t.GeneralValueType].fail(msg)
        # Type narrowing: isinstance check above narrows context to Mapping for mypy
        message_type_raw = context.get("message_type")
        if not isinstance(message_type_raw, str):
            msg = f"Invalid message_type in context: {type(message_type_raw).__name__}, expected str"
            return r[t.GeneralValueType].fail(msg)
        message_type: str = message_type_raw

        # Check pre-dispatch conditions (circuit breaker + rate limiting)
        conditions_check = self._check_pre_dispatch_conditions(message_type)
        if conditions_check.is_failure:
            # Extract error directly from conditions_check.error property
            error_msg = conditions_check.error or "Pre-dispatch conditions check failed"
            return r[t.GeneralValueType].fail(
                error_msg,
                error_code=conditions_check.error_code,
                error_data=conditions_check.error_data,
            )

        return r[t.GeneralValueType].ok(context)

    def _execute_with_retry_policy(
        self,
        context: t.GeneralValueType,
    ) -> r[t.GeneralValueType]:
        """Execute dispatch with retry policy using u."""
        # Fast fail: context must be dict-like to access values
        if not isinstance(context, Mapping):
            msg = f"Context must be dict-like, got {type(context).__name__}"
            return r[t.GeneralValueType].fail(msg)
        # Type narrowing: isinstance check above narrows context to Mapping for mypy
        message = context.get("message")
        message_type_raw = context.get("message_type")
        if not isinstance(message_type_raw, str):
            msg = f"Invalid message_type in context: {type(message_type_raw).__name__}, expected str"
            return r[t.GeneralValueType].fail(msg)
        message_type: str = message_type_raw

        metadata_raw = context.get("metadata")
        metadata: t.GeneralValueType | None = (
            metadata_raw
            if (metadata_raw is None or FlextRuntime.is_dict_like(metadata_raw))
            else None
        )

        correlation_id_raw = context.get("correlation_id")
        correlation_id: str | None = (
            correlation_id_raw
            if isinstance(correlation_id_raw, (str, type(None)))
            else None
        )

        timeout_override_raw = context.get("timeout_override")
        timeout_override: int | None = (
            timeout_override_raw
            if isinstance(timeout_override_raw, (int, type(None)))
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
            return r[t.GeneralValueType].ok(runtime_result.value)
        return r[t.GeneralValueType].fail(runtime_result.error or "Dispatch failed")

    @staticmethod
    def _normalize_dispatch_message(
        message: t.GeneralValueType,
        _data: t.GeneralValueType | None,
    ) -> tuple[t.GeneralValueType, str]:
        """Normalize message and extract message type.

        Fast fail: Only accepts message objects. No support for (message_type, data) API.
        """
        # Fast fail: message cannot be None
        if message is None:
            msg = "Message cannot be None. Use dispatch(message_object), not dispatch(None, data)."
            raise TypeError(msg)

        # Fast fail: message cannot be string
        if u.is_type(message, str):
            msg = (
                "String message_type not supported. "
                "Use dispatch(message_object), not dispatch('message_type', data)."
            )
            raise TypeError(msg)

        # Extract message type from message object
        message_type = type(message).__name__
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
        message: t.GeneralValueType,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> Callable[[], r[t.GeneralValueType]]:
        """Create execution function with context.

        Args:
            message: Message to execute
            correlation_id: Optional correlation ID
            timeout_override: Optional timeout override

        Returns:
            Callable that executes message with context

        """

        def execute_with_context() -> r[t.GeneralValueType]:
            if correlation_id is not None or timeout_override is not None:
                context_metadata: dict[str, t.GeneralValueType] = {}
                if timeout_override is not None:
                    context_metadata["timeout_override"] = timeout_override
                with self._context_scope(
                    context_metadata,
                    correlation_id,
                ):
                    return self._dispatch_command(message)
            return self._dispatch_command(message)

        return execute_with_context

    def _handle_dispatch_result(
        self,
        dispatch_result: r[t.GeneralValueType],
        message_type: str,
    ) -> r[t.GeneralValueType]:
        """Handle dispatch result with circuit breaker tracking.

        Args:
            dispatch_result: Result from dispatcher execution
            message_type: Message type for circuit breaker

        Returns:
            r[t.GeneralValueType]: Processed result

        Raises:
            e.OperationError: If result is failure but error is None

        """
        if dispatch_result.is_failure:
            # Use unwrap_error() for type-safe str
            error_msg = dispatch_result.error or "Unknown error"
            if "Executor was shutdown" in error_msg:
                return r[t.GeneralValueType].fail(error_msg)
            self._circuit_breaker.record_success(message_type)
            return r[t.GeneralValueType].fail(error_msg)

        self._circuit_breaker.record_success(message_type)
        return r[t.GeneralValueType].ok(dispatch_result.value)

    def _execute_dispatch_attempt(
        self,
        message: t.GeneralValueType,
        options: m.ExecuteDispatchAttemptOptions,
    ) -> r[t.GeneralValueType]:
        """Execute a single dispatch attempt with timeout."""
        try:
            # Create structured request
            # Use isinstance for proper type narrowing of metadata mapping
            if options.metadata is not None and isinstance(options.metadata, Mapping):
                # options.metadata is now narrowed to Mapping via isinstance
                metadata_map: Mapping[str, t.GeneralValueType] = options.metadata
                transform_result = u.process(
                    list(metadata_map.items()),
                    lambda kv: (str(kv[0]), str(kv[1])),
                    on_error="collect",
                )
                # Convert keys to strings and values to MetadataAttributeValue
                metadata_attrs: dict[str, t.MetadataAttributeValue]
                if transform_result.is_success:
                    # transform_result.value is list[tuple[str, str]]
                    # str is assignable to MetadataAttributeValue
                    processed_items: list[tuple[str, str]] = transform_result.value
                    metadata_attrs = dict(processed_items)
                else:
                    metadata_attrs = {}
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
            return r[t.GeneralValueType].fail(f"Dispatch error: {e}")

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_context_metadata(
        metadata: t.GeneralValueType | None,
    ) -> dict[str, t.GeneralValueType] | None:
        """Normalize metadata payloads to plain dictionaries.

        Fast fail: Direct validation without helpers.
        Handles BaseModel.model_dump(), Mapping, and direct dicts.
        """
        if metadata is None:
            return None

        # Fast fail: Direct extraction without helper
        # This handles BaseModel.model_dump(), Mapping, and direct dicts
        raw_metadata = FlextDispatcher._extract_metadata_mapping(metadata)

        if raw_metadata is None:
            return None

        # raw_metadata is m.Metadata
        # Return its attributes as the normalized dict
        # Metadata attributes are already strictly typed as dict[str, MetadataAttributeValue]
        return dict(raw_metadata.attributes)

    @staticmethod
    def _extract_metadata_mapping(
        metadata: t.GeneralValueType,
    ) -> m.Metadata | None:
        """Extract metadata as m.Metadata from various types."""
        if isinstance(metadata, m.Metadata):
            return metadata

        extracted_map: Mapping[str, t.GeneralValueType] | None = None

        if isinstance(metadata, Mapping):
            extracted_map = metadata
        elif isinstance(metadata, BaseModel):
            try:
                dumped = metadata.model_dump()
                if not FlextRuntime.is_dict_like(dumped):
                    msg = (
                        f"metadata.model_dump() returned {type(dumped).__name__}, "
                        "expected dict"
                    )
                    raise TypeError(msg)
                extracted_map = dumped
            except Exception as e:
                msg = f"Failed to dump BaseModel metadata: {type(e).__name__}: {e}"
                raise TypeError(msg) from e
        else:
            extracted_map = FlextDispatcher._extract_from_object_attributes(metadata)

        if extracted_map is None:
            return None

        # Convert Mapping to m.Metadata
        # Filter keys that are valid for Metadata attributes (str keys)
        # Any non-str key will fail Pydantic validation for attributes (dict[str, ...])
        # We assume strict typing: create m.Metadata from the mapping
        # If mapping contains "attributes", use that. Otherwise, use mapping as attributes.

        attrs: dict[str, t.GeneralValueType]
        if "attributes" in extracted_map and isinstance(
            extracted_map["attributes"], dict
        ):
            attrs = extracted_map["attributes"]
        else:
            attrs = {str(k): v for k, v in extracted_map.items()}

        # Strict filtering for MetadataAttributeValue compliance
        # This satisfies strict definition of m.Metadata.attributes
        valid_attrs: dict[str, t.MetadataAttributeValue] = {
            k: v
            for k, v in attrs.items()
            if isinstance(v, (str, int, float, bool, type(None), datetime))
            or (
                isinstance(v, list)
                and all(
                    isinstance(x, (str, int, float, bool, type(None), datetime))
                    for x in v
                )
            )
        }

        return m.Metadata(attributes=valid_attrs)

    @staticmethod
    def _extract_from_flext_metadata(
        metadata: m.Metadata,
    ) -> m.Metadata | None:
        """Extract metadata (noop for m.Metadata)."""
        return metadata

    @staticmethod
    def _extract_from_object_attributes(
        metadata: t.GeneralValueType,
    ) -> Mapping[str, t.GeneralValueType] | None:
        """Extract metadata mapping from object's attributes (internal helper)."""
        attributes_value = getattr(metadata, "attributes", None)
        if (
            u.is_type(attributes_value, "mapping")
            and u.is_dict_non_empty(attributes_value)
            and isinstance(attributes_value, Mapping)
        ):
            return dict(attributes_value)

        model_dump = getattr(metadata, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
                if isinstance(dumped, Mapping):
                    return dumped
            except Exception:
                # Suppress errors from model_dump() as metadata extraction is best-effort
                pass

        return None

    @contextmanager
    def _context_scope(
        self,
        metadata: t.GeneralValueType | None = None,
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
            current_parent_value if u.is_type(current_parent_value, str) else None
        )

        # Set new correlation ID if provided
        if correlation_id is not None:
            _ = correlation_var.set(correlation_id)
            # Set parent correlation ID if there was a previous one
            if current_parent is not None and current_parent != correlation_id:
                _ = parent_var.set(current_parent)

        # Set metadata if provided
        if metadata is not None and isinstance(metadata, dict):
            # Type narrowing: metadata is dict via isinstance check
            # dict[str, GeneralValueType] is compatible with ConfigurationDict
            metadata_dict: dict[str, t.GeneralValueType] = metadata
            _ = metadata_var.set(metadata_dict)
        elif metadata is not None and isinstance(metadata, Mapping):
            # Convert Mapping to dict for context variable
            converted_dict: dict[str, t.GeneralValueType] = dict(metadata)
            _ = metadata_var.set(converted_dict)

            # Use provided correlation ID or generate one if needed
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
        @FlextHandlers.handler() decorator in the calling module and registers them
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
                    ] = FlextHandlers.Discovery.scan_module(caller_module)
                    for handler_item in handlers:
                        # Unpack tuple with explicit type hints to help type inference
                        _handler_name: str = handler_item[0]
                        handler_func: t.HandlerCallable = handler_item[1]
                        handler_config: m.Handler.DecoratorConfig = handler_item[2]
                        # Get actual handler function from module
                        # Check if handler_func is not None before checking callable
                        # Use TypeGuard for proper handler type validation
                        if handler_func is not None and u.is_handler_type(
                            handler_func,
                        ):
                            # Register handler with dispatcher
                            # Register under the handler command type name for routing
                            # handler_config.command is type[object], never None (from typing)
                            command_type_name = (
                                handler_config.command.__name__
                                if hasattr(handler_config.command, "__name__")
                                else str(handler_config.command)
                            )
                            # handler_func is narrowed to t.HandlerType by TypeGuard
                            # register_handler accepts t.HandlerType | t.GeneralValueType
                            _ = instance.register_handler(
                                command_type_name,
                                handler_func,
                            )

        return instance

    @classmethod
    def create_from_global_config(cls) -> r[FlextDispatcher]:
        """Create dispatcher using global FlextSettings instance.

        Returns:
            FlextResult with dispatcher instance or error

        """
        try:
            instance = cls()
            return r[FlextDispatcher].ok(instance)
        except Exception as error:
            return r[FlextDispatcher].fail(
                f"Dispatcher creation failed: {error}",
            )

    # =============================================================================
    # Missing Methods for Test Compatibility
    # =============================================================================

    def dispatch_batch(
        self,
        _message_type: str,
        messages: list[t.GeneralValueType],
    ) -> list[r[t.GeneralValueType]]:
        """Dispatch multiple messages in batch.

        Args:
            _message_type: Type of messages to dispatch (unused - extracted from message object)
            messages: List of message objects to dispatch

        Returns:
            List of FlextResult objects for each dispatched message

        """
        # Dispatch each message - message_type is extracted from message object
        # Use list comprehension for type safety
        return [self.dispatch(msg) for msg in messages]

    def get_performance_metrics(
        self,
    ) -> dict[str, t.GeneralValueType]:
        """Get performance metrics for the dispatcher.

        Returns:
            object: Dictionary containing performance metrics

        """
        # Get metrics from circuit breaker manager
        cb_metrics = self._circuit_breaker.get_metrics()
        executor_status = self._timeout_enforcer.get_executor_status()
        # Cast all values to t.GeneralValueType
        return {
            "total_dispatches": 0,
            "circuit_breaker_failures": cb_metrics["failures"],
            "circuit_breaker_states": cb_metrics["states"],
            "circuit_breaker_open_count": cb_metrics["open_count"],
            **executor_status,
        }

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
