"""Message dispatch orchestration with layered reliability.

FlextDispatcher coordinates command and query execution with CQRS routing,
reliability policies (circuit breaker, rate limiting, retry, timeout), and
context-aware observability. The dispatcher is the application entry point for
handler registration and execution in the current architecture.

The dispatcher now supports dependency injection for reliability managers.
Managers can be injected via constructor parameters or resolved from
FlextContainer. See the ``__init__`` signature and ``_resolve_reliability_manager``
for the implementation pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import concurrent.futures
import inspect
import sys
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
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
from flext_core.constants import c
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.handlers import h
from flext_core.mixins import x
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t
from flext_core.utilities import u


class FlextDispatcher(x):
    """Application-level dispatcher that satisfies the command bus protocol.

    The dispatcher exposes CQRS routing, handler registration, and execution
    with layered reliability controls. It leans on structural typing to satisfy
    ``p.Application.CommandBus`` without inheritance while providing context
    propagation, metrics, and audit logging aligned to the current architecture.

    Basic usage:
        >>> dispatcher = FlextDispatcher()
        >>> dispatcher.register_handler(CreateUserCommand, handler)
        >>> dispatcher.dispatch(CreateUserCommand(name="Alice"))
    """

    @override
    def __init__(
        self,
        *,
        container: FlextContainer | None = None,
        circuit_breaker: CircuitBreakerManager | None = None,
        rate_limiter: RateLimiterManager | None = None,
        timeout_enforcer: TimeoutEnforcer | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """Initialize dispatcher with configuration from FlextConfig singleton.

        Refactored to eliminate SOLID violations by delegating to specialized components.
        Configuration is accessed via x.config singleton.

        Args:
            container: Optional FlextContainer for dependency injection. If provided,
                managers will be resolved from container if registered.
            circuit_breaker: Optional CircuitBreakerManager. If not provided, will be
                resolved from container or created with defaults.
            rate_limiter: Optional RateLimiterManager. If not provided, will be
                resolved from container or created with defaults.
            timeout_enforcer: Optional TimeoutEnforcer. If not provided, will be
                resolved from container or created with defaults.
            retry_policy: Optional RetryPolicy. If not provided, will be
                resolved from container or created with defaults.

        """
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service("flext_dispatcher")

        # Store container for manager resolution
        self._container = container or FlextContainer.create()

        # Enrich context with dispatcher metadata for observability
        self._enrich_context(
            service_type="dispatcher",
            dispatcher_type="FlextDispatcher",
            circuit_breaker_enabled=True,
            timeout_enforcement=True,
            supports_async=True,
        )

        # Access FlextConfig directly (no cast needed - mixins returns concrete type)
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

        # ==================== LAYER 2.5: TIMEOUT CONTEXT PROPAGATION ====================

        # Timeout context tracking for deadline and cancellation propagation
        self._timeout_contexts: t.Types.ConfigurationDict = {}  # operation_id → context
        self._timeout_deadlines: t.Types.StringFloatDict = {}  # operation_id → deadline timestamp

        # ==================== LAYER 1: CQRS ROUTING INITIALIZATION ====================

        # Handler registry (from FlextDispatcher dual-mode registration)
        self._handlers: t.Types.HandlerTypeDict = {}  # Handler mappings
        self._auto_handlers: list[t.Handler.HandlerType] = []  # Auto-discovery handlers

        # Middleware pipeline (from FlextDispatcher)
        self._middleware_configs: list[
            t.Types.ConfigurationMapping
        ] = []  # Config + ordering
        self._middleware_instances: t.Types.HandlerCallableDict = {}  # Keyed by middleware_id

        # Query result caching (from FlextDispatcher - LRU cache)
        # Fast fail: use constant directly, no fallback
        max_cache_size = c.Container.MAX_CACHE_SIZE
        self._cache: LRUCache[str, r[t.GeneralValueType]] = LRUCache(
            maxsize=max_cache_size,
        )

    def _resolve_or_create_circuit_breaker(
        self,
        config: p.Configuration.Config,
    ) -> CircuitBreakerManager:
        """Resolve circuit breaker from container or create with defaults.

        Args:
            config: Configuration instance for default values.

        Returns:
            CircuitBreakerManager instance from container or newly created.

        """
        # Try to resolve from container
        result = self._container.get("circuit_breaker")
        if result.is_success and isinstance(result.value, CircuitBreakerManager):
            return result.value

        # Create with defaults from config
        return CircuitBreakerManager(
            threshold=config.circuit_breaker_threshold,
            recovery_timeout=c.Reliability.DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            success_threshold=c.Reliability.DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
        )

    def _resolve_or_create_rate_limiter(
        self,
        config: p.Configuration.Config,
    ) -> RateLimiterManager:
        """Resolve rate limiter from container or create with defaults.

        Args:
            config: Configuration instance for default values.

        Returns:
            RateLimiterManager instance from container or newly created.

        """
        # Try to resolve from container
        result = self._container.get("rate_limiter")
        if result.is_success and isinstance(result.value, RateLimiterManager):
            return result.value

        # Create with defaults from config
        return RateLimiterManager(
            max_requests=config.rate_limit_max_requests,
            window_seconds=config.rate_limit_window_seconds,
        )

    def _resolve_or_create_timeout_enforcer(
        self,
        config: p.Configuration.Config,
    ) -> TimeoutEnforcer:
        """Resolve timeout enforcer from container or create with defaults.

        Args:
            config: Configuration instance for default values.

        Returns:
            TimeoutEnforcer instance from container or newly created.

        """
        # Try to resolve from container
        result = self._container.get("timeout_enforcer")
        if result.is_success and isinstance(result.value, TimeoutEnforcer):
            return result.value

        # Create with defaults from config
        return TimeoutEnforcer(
            use_timeout_executor=config.enable_timeout_executor,
            executor_workers=config.executor_workers,
        )

    def _resolve_or_create_retry_policy(
        self,
        config: p.Configuration.Config,
    ) -> RetryPolicy:
        """Resolve retry policy from container or create with defaults.

        Args:
            config: Configuration instance for default values.

        Returns:
            RetryPolicy instance from container or newly created.

        """
        # Try to resolve from container
        result = self._container.get("retry_policy")
        if result.is_success and isinstance(result.value, RetryPolicy):
            return result.value

        # Create with defaults from config
        return RetryPolicy(
            max_attempts=config.max_retry_attempts,
            retry_delay=config.retry_delay,
        )

        # Event subscribers (from FlextDispatcher event protocol)
        self._event_subscribers: t.Types.StringListDict = {}  # event_type → handlers

        # Execution counter for metrics
        self._execution_count: int = 0

        # ==================== LAYER 3: ADVANCED PROCESSING INITIALIZATION ====================

        # Group 1: Processor Registry (internal processing hooks)
        self._processors: dict[
            str,
            t.Handler.HandlerCallable
            | p.Utility.Callable[t.GeneralValueType]
            | p.Application.Processor,
        ] = {}  # name → processor function
        self._processor_configs: dict[
            str,
            t.Types.ConfigurationMapping,
        ] = {}  # name → config
        self._processor_metrics_per_name: t.Types.NestedStringIntDict = {}  # per-processor metrics
        self._processor_locks: dict[
            str,
            threading.Lock,
        ] = {}  # per-processor thread safety

        # Group 2: Batch & Parallel Configuration
        # Fast fail: use config values directly, no fallback
        self._batch_size: int = config.max_batch_size
        self._parallel_workers: int = config.executor_workers

        # Group 3: Handler Registry (internal dispatcher handler registry)
        self._handler_registry: t.Types.HandlerTypeDict = {}  # name → handler function
        self._handler_configs: dict[
            str,
            t.Types.ConfigurationMapping,
        ] = {}  # name → handler config
        self._handler_validators: dict[
            str,
            Callable[[t.GeneralValueType], bool],
        ] = {}  # validation functions

        # Group 4: Pipeline (dispatcher-managed processing pipeline)
        self._pipeline_steps: list[
            t.Types.ConfigurationMapping
        ] = []  # Ordered pipeline steps
        self._pipeline_composition: dict[
            str,
            Callable[
                [t.GeneralValueType],
                r[t.GeneralValueType],
            ],
        ] = {}  # composed functions
        self._pipeline_memo: t.Types.ConfigurationDict = {}  # Memoization cache for pipeline

        # Group 5: Metrics & Auditing (dispatcher-level counters)
        self._process_metrics: t.Types.StringNumericDict = {
            "registrations": 0,
            "successful_processes": 0,
            "failed_processes": 0,
            "batch_operations": 0,
            "parallel_operations": 0,
            "pipeline_operations": 0,
            "timeout_executions": 0,
        }
        self._audit_log: list[
            t.Types.ConfigurationMapping
        ] = []  # Operation audit trail
        self._performance_metrics: t.Types.ConfigurationDict = {}  # Timing and throughput
        self._processor_execution_times: t.Types.FloatListDict = {}  # Per-processor times

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
            | t.Handler.HandlerType
            | BaseModel
            | p.Utility.Callable[t.GeneralValueType]
            | p.Application.Processor
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
            # Type narrowing: if not list, must be str (or empty list fallback)
            # pyright: ignore[reportUnnecessaryIsInstance] - isinstance check for type narrowing
            methods = [method_names] if isinstance(method_names, str) else []  # pyright: ignore[reportUnnecessaryIsInstance]
        for method_name in methods:
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                if callable(method):
                    return r[bool].ok(True)

        method_list = "' or '".join(methods)
        return r[bool].fail(f"Invalid {context}: must have '{method_list}' method")

    def _validate_processor_interface(
        self,
        processor: t.Handler.HandlerCallable
        | p.Utility.Callable[t.GeneralValueType]
        | p.Application.Processor,
        processor_context: str = "processor",
    ) -> r[bool]:
        """Validate that processor has required interface (callable or process method)."""
        return self._validate_interface(
            processor,
            "process",
            processor_context,
            allow_callable=True,
        )

    def _route_to_processor(
        self,
        processor_name: str,
    ) -> (
        t.Handler.HandlerCallable
        | p.Utility.Callable[t.GeneralValueType]
        | p.Application.Processor
        | None
    ):
        """Locate registered processor by name.

        Args:
            processor_name: Name of processor to find

        Returns:
            Processor object or None if not found

        """
        return self._processors.get(processor_name)

    def _apply_processor_circuit_breaker(
        self,
        _processor_name: str,
        processor: t.Handler.HandlerCallable
        | p.Utility.Callable[t.GeneralValueType]
        | p.Application.Processor,
    ) -> r[t.GeneralValueType]:
        """Apply per-processor circuit breaker pattern.

        Args:
            _processor_name: Name of processor
            processor: Processor object

        Returns:
            r[t.GeneralValueType]: Success if circuit breaker allows, failure if open

        """
        # Use global circuit breaker manager
        # Per-processor circuit breaking is handled at dispatch() level
        # For now, always allow (dispatch() will enforce global CB)
        # Type narrowing: processor is already GeneralValueType compatible
        # processor is HandlerCallable | Callable[GeneralValueType] | Processor
        # Convert to GeneralValueType for return
        processor_typed: t.GeneralValueType = cast(
            "t.GeneralValueType",
            processor
            if isinstance(processor, (str, int, float, bool, type(None), dict, list))
            else str(processor),
        )
        return r[t.GeneralValueType].ok(processor_typed)

    @staticmethod
    def _apply_processor_rate_limiter(_processor_name: str) -> r[bool]:
        """Apply per-processor rate limiting.

        Args:
            _processor_name: Processor name (unused, reserved for future per-processor limits)

        Returns:
            r[bool]: Success with True if within limit, failure if exceeded

        """
        # Use global rate limiter manager
        # Per-processor rate limiting is handled at dispatch() level
        # For now, always allow (dispatch() will enforce global RL)
        return r[bool].ok(True)

    def _execute_processor_with_metrics(
        self,
        processor_name: str,
        processor: t.Handler.HandlerCallable
        | p.Utility.Callable[t.GeneralValueType]
        | p.Application.Processor,
        data: t.GeneralValueType,
    ) -> r[t.GeneralValueType]:
        """Execute processor and collect metrics.

        Args:
            processor_name: Name of processor
            processor: Processor object
            data: Data to process

        Returns:
            r[t.GeneralValueType]: Processor result or error

        """
        start_time = time.time()
        try:
            # Execute processor
            processor_result: t.GeneralValueType
            if callable(processor):
                processor_result_raw = processor(data)
                # processor() returns object, but we need GeneralValueType
                # Type narrowing: processor_result_raw is object, convert to GeneralValueType
                processor_result = processor_result_raw
            else:
                # Fast fail: check if process method exists, no fallback
                if not hasattr(processor, "process"):
                    return r[t.GeneralValueType].fail(
                        (
                            f"Cannot execute processor '{processor_name}': "
                            "processor must be callable or have 'process' method"
                        ),
                    )
                process_method = getattr(processor, "process", None)
                if process_method is None or not callable(process_method):
                    return r[t.GeneralValueType].fail(
                        (
                            f"Cannot execute processor '{processor_name}': "
                            "'process' attribute must be callable"
                        ),
                    )
                # Type narrowing: process_method() returns object, convert to GeneralValueType
                process_result_raw = process_method(data)
                processor_result = cast(
                    "t.GeneralValueType",
                    (
                        process_result_raw
                        if isinstance(
                            process_result_raw,
                            (
                                str,
                                int,
                                float,
                                bool,
                                type(None),
                                dict,
                                list,
                                BaseModel,
                                r,
                            ),
                        )
                        else str(process_result_raw)
                    ),
                )

            # Convert to r if needed
            # Ensure result is wrapped in r using consolidated helper
            result_wrapped = x.ResultHandling.ensure_result(processor_result)

            # Update metrics
            execution_time = time.time() - start_time
            if processor_name not in self._processor_execution_times:
                self._processor_execution_times[processor_name] = []
            self._processor_execution_times[processor_name].append(execution_time)

            # Update processor-specific metrics
            if processor_name not in self._processor_metrics_per_name:
                self._processor_metrics_per_name[processor_name] = {
                    "successful_processes": 0,
                    "failed_processes": 0,
                    "executions": 0,
                }
            metrics = self._processor_metrics_per_name[processor_name]
            metrics["executions"] = metrics.get("executions", 0) + 1
            if result_wrapped.is_success:
                metrics["successful_processes"] = (
                    metrics.get("successful_processes", 0) + 1
                )
            else:
                metrics["failed_processes"] = metrics.get("failed_processes", 0) + 1

            return result_wrapped
        except Exception as e:
            execution_time = time.time() - start_time
            if processor_name not in self._processor_execution_times:
                self._processor_execution_times[processor_name] = []
            self._processor_execution_times[processor_name].append(execution_time)
            return r[t.GeneralValueType].fail(f"Processor execution failed: {e}")

    def _process_batch_internal(
        self,
        processor_name: str,
        data_list: list[t.GeneralValueType],
        batch_size: int | None = None,
    ) -> r[list[t.GeneralValueType]]:
        """Process items in batch (internal).

        Args:
            processor_name: Name of processor
            data_list: List of data items
            batch_size: Size of each batch (default from config)

        Returns:
            r[list[t.GeneralValueType]]: List of results

        """
        # Fast fail: batch_size must be provided explicitly
        if batch_size is None:
            batch_size = self._batch_size
        if batch_size <= 0:
            return r[list[t.GeneralValueType]].fail(
                f"Invalid batch_size: {batch_size}. Must be > 0.",
            )
        results: list[t.GeneralValueType] = []

        processor = self._route_to_processor(processor_name)
        if processor is None:
            return r[list[t.GeneralValueType]].fail(
                f"Processor not found: {processor_name}",
            )

        # Process in batches - use manual loop for fail-fast behavior
        # (u.batch with on_error="fail" still collects errors)
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i : i + batch_size]
            for data in batch:
                result = self._execute_processor_with_metrics(
                    processor_name,
                    processor,
                    data,
                )
                if result.is_success:
                    results.append(result.value)
                else:
                    # Use u.err() for unified error extraction (DSL pattern)
                    error_msg = u.err(result, default="Unknown error in processor")
                    return r[list[t.GeneralValueType]].fail(
                        error_msg,
                    )

        return r[list[t.GeneralValueType]].ok(results)

    def _process_parallel_internal(
        self,
        processor_name: str,
        data_list: list[t.GeneralValueType],
        max_workers: int | None = None,
    ) -> r[list[t.GeneralValueType]]:
        """Process items in parallel (internal).

        Args:
            processor_name: Name of processor
            data_list: List of data items
            max_workers: Number of parallel workers (default from config)

        Returns:
            r[list[t.GeneralValueType]]: List of results

        """
        # Fast fail: max_workers must be provided explicitly
        if max_workers is None:
            max_workers = self._parallel_workers
        if max_workers <= 0:
            return r[list[t.GeneralValueType]].fail(
                f"Invalid max_workers: {max_workers}. Must be > 0.",
            )
        results: list[t.GeneralValueType] = []

        processor = self._route_to_processor(processor_name)
        if processor is None:
            return r[list[t.GeneralValueType]].fail(
                f"Processor not found: {processor_name}",
            )

        # Process in parallel using ThreadPoolExecutor
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
            ) as executor:
                futures = {
                    executor.submit(
                        self._execute_processor_with_metrics,
                        processor_name,
                        processor,
                        data,
                    ): idx
                    for idx, data in enumerate(data_list)
                }

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result.is_success:
                        results.append(result.value)
                    else:
                        # Use u.err() for unified error extraction (DSL pattern)
                        error_msg = u.err(result, default="Unknown error in processor")
                        return r[list[t.GeneralValueType]].fail(
                            error_msg,
                        )

            return r[list[t.GeneralValueType]].ok(results)
        except Exception as e:
            return r[list[t.GeneralValueType]].fail(
                f"Parallel processing failed: {e}",
            )

    def _validate_handler_registry_interface(
        self,
        handler: (
            t.Handler.HandlerType
            | t.Handler.HandlerCallable
            | p.Utility.Callable[t.GeneralValueType]
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

    # ==================== LAYER 3: ADVANCED PROCESSING PUBLIC APIS ====================

    def register_processor(
        self,
        name: str,
        processor: t.Handler.HandlerCallable
        | p.Utility.Callable[t.GeneralValueType]
        | p.Application.Processor,
        config: t.Types.ConfigurationMapping | None = None,
    ) -> r[bool]:
        """Register processor for advanced processing.

        Args:
            name: Processor name identifier
            processor: Processor object (callable or has process() method)
            config: Optional processor-specific configuration

        Returns:
            r[bool]: Success with True if registered, failure if invalid processor

        """
        # Validate processor interface
        validation_result = self._validate_processor_interface(
            processor,
            f"processor '{name}'",
        )
        if validation_result.is_failure:
            return validation_result

        # Register processor and configuration
        self._processors[name] = processor
        if config is not None:
            # Convert Mapping to dict for mutability - use dict() constructor directly
            self._processor_configs[name] = dict(config.items())
        else:
            self._processor_configs[name] = {}

        # Initialize per-processor metrics and lock
        self._processor_metrics_per_name[name] = {
            "successful_processes": 0,
            "failed_processes": 0,
            "executions": 0,
        }
        self._processor_locks[name] = threading.Lock()

        # Update global metrics
        self._process_metrics["registrations"] += 1

        return r[bool].ok(True)

    def process(
        self,
        name: str,
        data: t.GeneralValueType,
    ) -> r[t.GeneralValueType]:
        """Process data through registered processor.

        This is the main entry point for Layer 3 processing. It routes to the
        registered processor and delegates through Layer 2 dispatch() for
        reliability patterns (circuit breaker, rate limiting, retry).

        Args:
            name: Processor name
            data: Data to process

        Returns:
            r[t.GeneralValueType]: Processed result or error

        """
        # Route to processor
        processor = self._route_to_processor(name)
        if processor is None:
            return r[t.GeneralValueType].fail(
                f"Processor '{name}' not registered. Register with register_processor().",
            )

        # Apply per-processor circuit breaker
        cb_result = self._apply_processor_circuit_breaker(name, processor)
        if cb_result.is_failure:
            return r[t.GeneralValueType].fail(
                f"Processor '{name}' circuit breaker is open",
            )

        # Apply per-processor rate limiter
        rate_limit_result = FlextDispatcher._apply_processor_rate_limiter(name)
        if rate_limit_result.is_failure:
            # Use u.err() for unified error extraction (DSL pattern)
            error_msg = u.err(rate_limit_result, default="Rate limit exceeded")
            return r[t.GeneralValueType].fail(error_msg)

        # Execute processor with metrics collection
        return self._execute_processor_with_metrics(name, processor, data)

    def _process_collection(
        self,
        name: str,
        data_list: list[t.GeneralValueType],
        resolved_param: int,
        operation_type: str,
        metric_key: str,
    ) -> r[list[t.GeneralValueType]]:
        """Process collection with specified operation type (DRY helper).

        Eliminates duplication between process_batch and process_parallel.
        Both methods follow identical pattern: empty check → call internal
        method → update metrics → return result.

        Args:
            name: Processor name
            data_list: List of items to process
            resolved_param: Resolved batch size or worker count
            operation_type: "batch" or "parallel" (for internal method routing)
            metric_key: Metric key to increment on success

        Returns:
            r[list[t.GeneralValueType]]: List of processed items or error

        """
        if not data_list:
            return r[list[t.GeneralValueType]].ok([])

        # Call appropriate internal method based on operation type
        if operation_type == c.Cqrs.ProcessingMode.BATCH:
            result = self._process_batch_internal(name, data_list, resolved_param)
        else:  # parallel
            result = self._process_parallel_internal(name, data_list, resolved_param)

        if result.is_success:
            self._process_metrics[metric_key] += 1

        return result

    def process_batch(
        self,
        name: str,
        data_list: list[t.GeneralValueType],
        batch_size: int | None = None,
    ) -> r[list[t.GeneralValueType]]:
        """Process multiple items in batch.

        Args:
            name: Processor name
            data_list: List of items to process
            batch_size: Optional batch size (uses config default if None)

        Returns:
            r[list[t.GeneralValueType]]: List of processed items or error

        """
        # Fast fail: explicit default value instead of 'or' fallback
        resolved_batch_size: int = (
            batch_size if batch_size is not None else self._batch_size
        )
        return self._process_collection(
            name,
            data_list,
            resolved_batch_size,
            c.Cqrs.ProcessingMode.BATCH,
            "batch_operations",
        )

    def process_parallel(
        self,
        name: str,
        data_list: list[t.GeneralValueType],
        max_workers: int | None = None,
    ) -> r[list[t.GeneralValueType]]:
        """Process multiple items in parallel.

        Args:
            name: Processor name
            data_list: List of items to process
            max_workers: Optional max worker threads (uses config default if None)

        Returns:
            r[list[t.GeneralValueType]]: List of processed items or error

        """
        # Fast fail: explicit default value instead of 'or' fallback
        resolved_workers: int = (
            max_workers if max_workers is not None else self._parallel_workers
        )
        return self._process_collection(
            name,
            data_list,
            resolved_workers,
            "parallel",
            "parallel_operations",
        )

    def execute_with_timeout(
        self,
        name: str,
        data: t.GeneralValueType,
        timeout: float,
    ) -> r[t.GeneralValueType]:
        """Process with timeout enforcement.

        Args:
            name: Processor name
            data: Data to process
            timeout: Timeout in seconds

        Returns:
            r[t.GeneralValueType]: Processed result or timeout error

        """
        # Use TimeoutEnforcer from Layer 2 (same as dispatch method)
        try:
            executor = self._timeout_enforcer.ensure_executor()
            future: concurrent.futures.Future[r[t.GeneralValueType]] = executor.submit(
                self.process,
                name,
                data,
            )
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self._process_metrics["failed_processes"] += 1
            self._process_metrics["timeout_executions"] += 1
            return r[t.GeneralValueType].fail(
                f"Processor '{name}' timeout after {timeout}s",
            )

    # ==================== LAYER 3: METRICS & AUDITING ====================

    @property
    def processor_metrics(self) -> t.Types.NestedStringIntDict:
        """Get processor execution metrics.

        Returns:
            dict: Per-processor metrics with execution counts and success/failure rates

        """
        return self._processor_metrics_per_name.copy()

    @property
    def batch_performance(self) -> t.Types.StringNumericDict:
        """Get batch operation performance metrics.

        Returns:
            dict: Batch operation statistics including operation count and metrics

        """
        batch_ops = self._process_metrics.get("batch_operations", 0)
        return {
            "batch_operations": batch_ops,
            "average_batch_size": self._batch_size,
        }

    @property
    def parallel_performance(self) -> t.Types.StringNumericDict:
        """Get parallel operation performance metrics.

        Returns:
            dict: Parallel operation statistics including operation count and worker count

        """
        parallel_ops = self._process_metrics.get("parallel_operations", 0)
        return {
            "parallel_operations": parallel_ops,
            "max_workers": self._parallel_workers,
        }

    def get_process_audit_log(
        self,
    ) -> r[list[t.Types.ConfigurationMapping]]:
        """Retrieve operation audit trail.

        Returns:
            r[list[dict]]: Audit log entries with operation details

        """
        return r[list[t.Types.ConfigurationMapping]].ok(
            self._audit_log.copy(),
        )

    def get_performance_analytics(
        self,
    ) -> r[t.GeneralValueType]:
        """Get comprehensive performance analytics.

        Returns:
            r[dict]: Complete performance analytics including all metrics

        """
        analytics: t.GeneralValueType = {
            "global_metrics": self._process_metrics.copy(),
            "processor_metrics": self._processor_metrics_per_name.copy(),
            "batch_performance": self.batch_performance,
            "parallel_performance": self.parallel_performance,
            "performance_timings": self._performance_metrics.copy(),
            "processor_execution_times": self._processor_execution_times.copy(),
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
        # Use u.has() + u.Mapper.get() for unified attribute access (DSL pattern)
        if u.has(command_type_obj, "__name__"):
            name_attr = u.Mapper.get(command_type_obj, "__name__", default=None)
            if name_attr is not None:
                return str(name_attr)
        return str(command_type_obj)

    def _validate_handler_interface(
        self,
        handler: t.Handler.HandlerType,
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
        # Use u.Mapper.get() for unified attribute access (DSL pattern)
        # __members__ returns mappingproxy[str, HandlerType], which is compatible with HandlerTypeDict
        handler_type_members_raw: (
            Mapping[str, t.Handler.HandlerType] | dict[str, t.Handler.HandlerType]
        ) = cast(
            "Mapping[str, t.Handler.HandlerType] | dict[str, t.Handler.HandlerType]",
            u.Mapper.get(c.Cqrs.HandlerType, "__members__", default={}) or {},
        )
        # __members__ returns mappingproxy[str, HandlerType], cast to HandlerTypeDict
        # HandlerTypeDict is dict[str, HandlerType], which matches __members__ structure
        # pyright: ignore[reportUnnecessaryIsInstance] - isinstance check for type narrowing
        handler_type_members: t.Types.HandlerTypeDict = cast(
            "t.Types.HandlerTypeDict",
            handler_type_members_raw
            if isinstance(handler_type_members_raw, Mapping)  # pyright: ignore[reportUnnecessaryIsInstance]
            else {},
        )
        valid_modes = list(
            u.Collection.map(
                list(handler_type_members.values()),
                lambda m: cast("c.Cqrs.HandlerType", m).value,
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
    ) -> t.Handler.HandlerType | None:
        """Locate the handler that can process the provided message.

        Args:
            command: The command/query object to find handler for

        Returns:
            The handler instance or None if not found

        """
        command_type = type(command)
        command_name = command_type.__name__

        # First, try to find handler by command type name in _handlers
        # (two-arg registration)
        if command_name in self._handlers:
            handler_entry = self._handlers[command_name]
            # handler_entry is always HandlerType based on _handlers type definition
            # Check if it's a dict-like structure with "handler" key (legacy support)
            if (
                isinstance(handler_entry, (dict, Mapping))
                and "handler" in handler_entry
            ):
                # Type narrowing: extract handler from dict structure
                # handler_entry is dict-like with "handler" key containing HandlerType
                handler_entry_dict: t.Types.ConfigurationMapping = cast(
                    "t.Types.ConfigurationMapping",
                    handler_entry,
                )
                extracted_handler: t.GeneralValueType = handler_entry_dict.get(
                    "handler",
                )
                # Validate it's callable or BaseModel (valid HandlerType)
                # HandlerType includes Callable and BaseModel instances
                # Type narrowing: extracted_handler is HandlerType after validation
                if callable(extracted_handler) or isinstance(
                    extracted_handler,
                    BaseModel,
                ):
                    return cast("t.Handler.HandlerType", extracted_handler)
            # Return handler directly (it's already HandlerType)
            # Type narrowing: handler_entry is HandlerType from dict definition
            # Type annotation: handler_entry is HandlerType
            handler_result: t.Handler.HandlerType = cast(
                "t.Handler.HandlerType", handler_entry
            )
            return handler_result

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
        # generate_cache_key accepts *args: GeneralValueType, so pass command and command_type name as string
        command_type_name = command_type.__name__ if command_type else "unknown"
        # Pass command and command_type_name (string) as GeneralValueType-compatible arguments
        return u.Cache.generate_cache_key(command, command_type_name)

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
        handler: t.Handler.HandlerType,
        command: t.GeneralValueType,
        operation: str = c.Dispatcher.HANDLER_MODE_COMMAND,
    ) -> r[t.GeneralValueType]:
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
        # Use u.Mapper.get() for unified attribute access (DSL pattern)
        handler_class_name = u.Mapper.get(handler_class, "__name__", default="Unknown")
        self.logger.debug(
            "Delegating to handler",
            operation="execute_handler",
            handler_type=handler_class_name,
            command_type=type(command).__name__,
            source="flext-core/src/flext_core/dispatcher.py",
        )

        # Delegate to h.dispatch_message() for full CQRS support
        if isinstance(handler, h):
            # Type narrowing: handler is h[MessageT_contra, ResultT]
            # dispatch_message returns r[ResultT], but ResultT is unknown at runtime
            # Since handler is generic and ResultT cannot be inferred, we cast the result
            # to r[t.GeneralValueType] for compatibility with return type
            # This is safe because GeneralValueType is the base type for all values
            dispatch_result_raw = handler.dispatch_message(command, operation=operation)
            # Cast to r[t.GeneralValueType] - safe because ResultT is always a GeneralValueType
            dispatch_result: r[t.GeneralValueType] = cast(
                "r[t.GeneralValueType]",
                dispatch_result_raw,
            )
            return dispatch_result

        # Fallback for non-h: try handle() then execute()
        # Use u.has() + u.Mapper.get() for unified attribute access (DSL pattern)
        method_name = None
        if u.has(handler, c.Mixins.METHOD_HANDLE):
            handle_method = u.Mapper.get(handler, c.Mixins.METHOD_HANDLE, default=None)
            if callable(handle_method):
                method_name = c.Mixins.METHOD_HANDLE
        elif u.has(handler, c.Mixins.METHOD_EXECUTE):
            execute_method = u.Mapper.get(
                handler,
                c.Mixins.METHOD_EXECUTE,
                default=None,
            )
            if callable(execute_method):
                method_name = c.Mixins.METHOD_EXECUTE

        if not method_name:
            return r[t.GeneralValueType].fail(
                f"Handler must have '{c.Mixins.METHOD_HANDLE}' or '{c.Mixins.METHOD_EXECUTE}' method",
                error_code=c.Errors.COMMAND_BUS_ERROR,
            )
        # Use u.Mapper.get() for unified attribute access (DSL pattern)
        handle_method = u.Mapper.get(handler, method_name, default=None)
        if not callable(handle_method):
            error_msg = f"Handler '{method_name}' must be callable"
            return r[t.GeneralValueType].fail(
                error_msg,
                error_code=c.Errors.COMMAND_BUS_ERROR,
            )

        try:
            result_raw = handle_method(command)
            # Ensure result is GeneralValueType - handlers should return GeneralValueType or FlextResult
            # Type narrowing: result_raw is GeneralValueType after isinstance check
            # pyright: ignore[reportUnnecessaryIsInstance] - isinstance check for type narrowing
            result: t.GeneralValueType
            if isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
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
        handler: t.Handler.HandlerType,
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
        middleware_config: t.Types.ConfigurationMapping,
    ) -> int:
        """Extract middleware execution order from config."""
        order_value = (
            u.Mapper.get(
                middleware_config,
                "order",
                default=c.Defaults.DEFAULT_MIDDLEWARE_ORDER,
            )
            or c.Defaults.DEFAULT_MIDDLEWARE_ORDER
        )
        if u.is_type(order_value, str):
            try:
                return int(order_value)
            except ValueError:
                return c.Defaults.DEFAULT_MIDDLEWARE_ORDER
        return (
            int(order_value)
            if u.is_type(order_value, int)
            else c.Defaults.DEFAULT_MIDDLEWARE_ORDER
        )

    def _process_middleware_instance(
        self,
        command: t.GeneralValueType,
        handler: t.Handler.HandlerType,
        middleware_config: t.Types.ConfigurationMapping,
    ) -> r[bool]:
        """Process a single middleware instance."""
        # Extract configuration values from dict using get()
        # u.Mapper.get() without default returns GeneralValueType | None
        middleware_id_value: t.GeneralValueType | None = u.Mapper.get(
            middleware_config, "middleware_id"
        )
        middleware_type_value = u.Mapper.get(middleware_config, "middleware_type")
        enabled_raw = u.Mapper.get(middleware_config, "enabled", default=True)
        enabled_value = bool(enabled_raw) if enabled_raw is not None else False

        # Convert middleware_id to string (handles None case)
        # Type narrowing: middleware_id_value can be None or GeneralValueType
        # pyright: ignore[reportUnnecessaryComparison] - middleware_id_value can be None
        middleware_id_str: str = (
            str(middleware_id_value) if middleware_id_value is not None else ""  # pyright: ignore[reportUnnecessaryComparison]
        )

        # Skip disabled middleware
        if not enabled_value:
            self.logger.debug(
                "Skipping disabled middleware",
                middleware_id=middleware_id_str,
                middleware_type=str(middleware_type_value),
            )
            return r[bool].ok(True)

        # Get actual middleware instance
        middleware = self._middleware_instances.get(middleware_id_str)
        if middleware is None:
            return r[bool].ok(True)

        self.logger.debug(
            "Applying middleware",
            middleware_id=middleware_id_str,
            middleware_type=str(middleware_type_value),
            order=u.Mapper.get(
                middleware_config,
                "order",
                default=c.Defaults.DEFAULT_MIDDLEWARE_ORDER,
            )
            or c.Defaults.DEFAULT_MIDDLEWARE_ORDER,
        )

        return self._invoke_middleware(
            middleware,
            command,
            handler,
            middleware_type_value,
        )

    def _invoke_middleware(
        self,
        middleware: t.Handler.HandlerCallable,
        command: t.GeneralValueType,
        handler: t.Handler.HandlerType,
        middleware_type: t.GeneralValueType,
    ) -> r[bool]:
        """Invoke middleware and handle result.

        Fast fail: Middleware must have process() method. No fallback to callable.
        """
        # Use u.Mapper.get() for unified attribute access (DSL pattern)
        process_method = u.Mapper.get(middleware, "process", default=None)
        if not callable(process_method):
            return r[bool].fail(
                "Middleware must have callable 'process' method",
                error_code=c.Errors.CONFIGURATION_ERROR,
            )
        result_raw = process_method(command, handler)
        # Ensure result is GeneralValueType or FlextResult
        # pyright: ignore[reportUnnecessaryIsInstance] - isinstance check for type narrowing
        result: t.GeneralValueType | r[t.GeneralValueType]
        if isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
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
            result = cast(
                "t.GeneralValueType | r[t.GeneralValueType]",
                result_raw,
            )
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
            # Use u.err() for unified error extraction (DSL pattern)
            error_msg = u.err(result, default="Unknown error")
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

    # ==================== LAYER 1 PUBLIC API: CQRS ROUTING & MIDDLEWARE ====================

    def execute(
        self,
        command: t.GeneralValueType,
    ) -> r[t.GeneralValueType]:
        """Execute command/query through the CQRS dispatcher routing layer.

        This Layer 1 entry point performs routing with caching and middleware.
        For reliability patterns (circuit breaker, rate limit, retry, timeout),
        use ``dispatch`` which chains this execution with the Layer 2 controls.

        Args:
            command: The command or query object to execute

        Returns:
            r: Execution result wrapped in r

        """
        # Propagate context for distributed tracing
        command_type = type(command)
        self._propagate_context(f"execute_{command_type.__name__}")

        # Track operation metrics
        with self.track(f"dispatch_execute_{command_type.__name__}") as _:
            self._execution_count += 1

            self.logger.debug(
                "Executing command",
                operation=c.Mixins.METHOD_EXECUTE,
                command_type=command_type.__name__,
                # Use u.Mapper.get() for unified attribute access (DSL pattern)
                command_id=u.Mapper.get(
                    command,
                    "command_id",
                    default=u.Mapper.get(command, "id", default="unknown"),
                ),
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
                    u.Collection.map(
                        self._auto_handlers,
                        lambda h: h.__class__.__name__,
                    ),
                )
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

    def layer1_register_handler(
        self,
        *args: t.Handler.HandlerType,
    ) -> r[bool]:
        """Register handler with dual-mode support (from FlextDispatcher).

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
            handler_single: t.Handler.HandlerType = args[0]
            return self._register_single_handler(handler_single)
        if len(args) == c.Dispatcher.TWO_HANDLER_ARG_COUNT:
            # Cast args[0] to GeneralValueType | str and args[1] to HandlerType
            # args[0] can be HandlerType (when it's a message type), so we need to handle both cases
            command_type_typed: t.GeneralValueType | str = cast(
                "t.GeneralValueType | str",
                args[0],
            )
            handler_two: t.Handler.HandlerType = args[1]
            return self._register_two_arg_handler(command_type_typed, handler_two)

        return r[bool].fail(
            f"register_handler takes 1 or 2 arguments but {len(args)} were given",
        )

    def _wire_handler_dependencies(
        self,
        handler: t.Handler.HandlerType,
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
        handler: t.Handler.HandlerType,
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
        if handler_id is not None:
            self._handlers[str(handler_id)] = handler
            self.logger.info(
                "Handler registered",
                handler_type=getattr(
                    handler.__class__,
                    "__name__",
                    str(type(handler)),
                ),
                handler_id=str(handler_id),
                total_handlers=len(self._handlers),
            )
        else:
            self.logger.info(
                "Handler registered for auto-discovery",
                handler_type=getattr(
                    handler.__class__,
                    "__name__",
                    str(type(handler)),
                ),
                total_handlers=len(self._auto_handlers),
            )

        return r[bool].ok(True)

    def _register_two_arg_handler(
        self,
        command_type_obj: t.GeneralValueType | str,
        handler: t.Handler.HandlerType,
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

        if isinstance(command_type_obj, str) and not command_type_obj.strip():
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
        self.logger.info(
            "Handler registered for command type",
            command_type=key,
            handler_type=getattr(handler.__class__, "__name__", str(type(handler))),
            total_handlers=len(self._handlers),
        )

        return r[bool].ok(True)

    def layer1_add_middleware(
        self,
        middleware: t.Handler.HandlerCallable,
        middleware_config: t.Types.ConfigurationMapping | None = None,
    ) -> r[bool]:
        """Add middleware to processing pipeline (from FlextDispatcher).

        Args:
            middleware: The middleware instance to add
            middleware_config: Configuration for the middleware (dict or None)

        Returns:
            r: Success or failure result

        """
        # Resolve middleware_id
        if middleware_config and middleware_config.get("middleware_id"):
            middleware_id_str = str(middleware_config.get("middleware_id"))
        else:
            middleware_id_str = getattr(
                middleware,
                "middleware_id",
                f"mw_{len(self._middleware_configs)}",
            )

        # Resolve middleware type
        middleware_type_str = (
            str(middleware_config.get("middleware_type"))
            if middleware_config and middleware_config.get("middleware_type")
            else type(middleware).__name__
        )

        # Create config - convert values to GeneralValueType compatible types
        # Extract enabled value safely
        enabled_value: bool = True
        if middleware_config:
            enabled_raw = middleware_config.get("enabled", True)
            enabled_value = bool(enabled_raw) if enabled_raw is not None else True

        # Extract order value safely
        order_value: int = len(self._middleware_configs)
        if middleware_config:
            order_raw = middleware_config.get("order", len(self._middleware_configs))
            if isinstance(order_raw, int):
                order_value = order_raw
            elif isinstance(order_raw, (str, float)):
                order_value = int(order_raw)

        final_config_raw: t.Types.ConfigurationMapping = {
            "middleware_id": middleware_id_str,
            "middleware_type": middleware_type_str,
            "enabled": enabled_value,
            "order": order_value,
        }
        # final_config_raw already matches ConfigurationMapping structure
        final_config: t.Types.ConfigurationMapping = final_config_raw

        self._middleware_configs.append(final_config)
        self._middleware_instances[middleware_id_str] = middleware

        self.logger.info(
            "Middleware added to pipeline",
            operation="add_middleware",
            middleware_type=final_config.get("middleware_type"),
            middleware_id=middleware_id_str,
            total_middleware=len(self._middleware_configs),
            source="flext-core/src/flext_core/dispatcher.py",
        )

        return r[bool].ok(True)

    # ==================== LAYER 1 EVENT PUBLISHING PROTOCOL ====================

    def publish_event(self, event: t.GeneralValueType) -> r[bool]:
        """Publish domain event to subscribers (from FlextDispatcher).

        Args:
            event: Domain event to publish

        Returns:
            r[bool]: Success with True, failure with error details

        """
        try:
            # Use execute mechanism for event publishing
            result = self.execute(event)

            if result.is_failure:
                return r[bool].fail(f"Event publishing failed: {result.error}")

            return r[bool].ok(True)
        except (TypeError, AttributeError, ValueError) as e:
            # TypeError: invalid event type
            # AttributeError: missing event attribute
            # ValueError: event validation failed
            return r[bool].fail(f"Event publishing error: {e}")

    def publish_events(
        self,
        events: list[t.GeneralValueType],
    ) -> r[bool]:
        """Publish multiple domain events (from FlextDispatcher).

        Uses r.from_callable() to eliminate try/except and
        flow_through() for declarative event processing pipeline.

        Args:
            events: List of domain events to publish

        Returns:
            r[bool]: Success with True, failure with error details

        """

        def publish_all() -> bool:
            # Convert events to r pipeline
            def make_publish_func(
                event_item: t.GeneralValueType,
            ) -> Callable[[t.GeneralValueType], r[bool]]:
                def publish_func(
                    _message: t.GeneralValueType,
                ) -> r[bool]:
                    # _message parameter required by signature but not used (event_item is used instead)
                    del _message  # Explicitly mark as intentionally unused
                    return self.publish_event(event_item)

                return publish_func

            # Convert functions to match flow_through signature: Callable[[t.GeneralValueType], r[t.GeneralValueType]]
            publish_funcs: list[
                Callable[
                    [t.GeneralValueType],
                    r[t.GeneralValueType],
                ]
            ] = []
            for event in events:
                publish_func = make_publish_func(event)

                def make_wrapper(
                    func: Callable[[t.GeneralValueType], r[bool]],
                ) -> Callable[
                    [t.GeneralValueType],
                    r[t.GeneralValueType],
                ]:
                    def wrapper(
                        message: t.GeneralValueType,
                    ) -> r[t.GeneralValueType]:
                        bool_result = func(message)
                        # Map boolean result to GeneralValueType (True is a valid GeneralValueType)
                        return bool_result.map(lambda _value: True)

                    return wrapper

                publish_funcs.append(make_wrapper(publish_func))

            result = r[bool].ok(True).flow_through(*publish_funcs)
            if result.is_failure:
                error_msg = result.error
                raise RuntimeError(error_msg or "Unknown error")
            # Fast fail: return bool True for success
            return True

        return r[bool].create_from_callable(publish_all)

    def subscribe(
        self,
        event_type: str,
        handler: t.Handler.HandlerType,
    ) -> r[bool]:
        """Subscribe handler to event type (from FlextDispatcher).

        Args:
            event_type: Type of event to subscribe to
            handler: Handler callable for the event

        Returns:
            r[bool]: Success with True, failure with error details

        """
        try:
            # Cast event_type and handler to HandlerType for layer1_register_handler
            event_type_typed: t.Handler.HandlerType = cast(
                "t.Handler.HandlerType",
                event_type,
            )
            handler_typed: t.Handler.HandlerType = handler
            return self.layer1_register_handler(event_type_typed, handler_typed)
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
        event_name: str,
        data: t.GeneralValueType,
    ) -> r[bool]:
        """Publish a named event with data (from FlextDispatcher).

        Convenience method for publishing events by name with associated data.

        Args:
            event_name: Name/identifier of the event
            data: Event data payload

        Returns:
            r[bool]: Success with True, failure with error details

        """
        # Create a simple event dict with name and data
        event: t.Types.ConfigurationMapping = {
            "event_name": event_name,
            "data": data,
            "timestamp": time.time(),
        }
        return self.publish_event(event)

    # ------------------------------------------------------------------
    # Registration methods using structured models
    # ------------------------------------------------------------------
    @staticmethod
    def _get_nested_attr(
        obj: t.GeneralValueType,
        *path: str,
    ) -> t.GeneralValueType | None:
        """Get nested attribute safely (e.g., obj.attr1.attr2).

        Returns None if any attribute in path doesn't exist or is None.
        Uses u.extract() for dict-like objects, falls back to
        attribute access for objects.
        """
        if not path:
            return None
        # Try extract for dict-like objects first
        if u.is_type(obj, "mapping"):
            path_str = ".".join(path)
            result = u.Mapper.extract(obj, path_str, default=None, required=False)
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
    def _extract_handler_name(
        handler: t.GeneralValueType,
        request_dict: t.Types.ConfigurationMapping,
    ) -> str:
        """Extract handler_name from request or handler config.

        Args:
            handler: Handler instance
            request_dict: Request dictionary

        Returns:
            Handler name string or empty string if not found

        """
        # Try extract for dict-like request first
        handler_name_result = u.Mapper.extract(
            request_dict,
            "handler_name",
            default="",
            required=False,
        )
        handler_name = (
            str(handler_name_result.value) if handler_name_result.is_success else ""
        )
        if handler_name:
            return handler_name

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
    def _normalize_request_to_dict(
        request: t.GeneralValueType,
    ) -> r[t.Types.ConfigurationDict]:
        """Normalize request to GeneralValueType dict.

        Args:
            request: Dict or Pydantic model containing registration details

        Returns:
            r with normalized dict or error

        """
        if not isinstance(request, BaseModel) and not FlextRuntime.is_dict_like(
            request,
        ):
            return r[t.Types.ConfigurationDict].fail(
                "Request must be dict or Pydantic model",
            )

        request_dict: t.Types.ConfigurationDict
        if isinstance(request, BaseModel):
            dumped = request.model_dump()
            normalized = FlextRuntime.normalize_to_general_value(dumped)
            request_dict = cast(
                "t.Types.ConfigurationDict",
                (normalized if u.is_type(normalized, dict) else {}),
            )
        elif u.is_type(request, dict):
            # Preserve handler objects directly (don't normalize them to strings)
            # Handler normalization would convert handler instances to string repr
            handler_keys = {"handler", "handlers", "processor", "processors"}
            # Use process() for concise request normalization with key conversion
            process_result = u.Collection.process(
                request,
                lambda k, v: (
                    v
                    if str(k) in handler_keys
                    else FlextRuntime.normalize_to_general_value(v)
                ),
                on_error="collect",
            )
            # Convert keys to strings - use process() to transform keys
            if process_result.is_success and isinstance(process_result.value, dict):
                # Create new dict with string keys
                request_dict = {str(k): v for k, v in process_result.value.items()}
            else:
                request_dict = {}
        else:
            normalized = FlextRuntime.normalize_to_general_value(request)
            request_dict = cast(
                "t.Types.ConfigurationDict",
                (normalized if u.is_type(normalized, dict) else {}),
            )

        return r[t.Types.ConfigurationDict].ok(request_dict)

    def _validate_and_extract_handler(
        self,
        request_dict: t.Types.ConfigurationMapping,
    ) -> r[tuple[t.GeneralValueType, str]]:
        """Validate handler and extract handler name.

        Args:
            request_dict: Normalized request dictionary

        Returns:
            r with (handler, handler_name) tuple or error

        """
        handler_raw = request_dict.get("handler")
        if not handler_raw:
            return r[tuple[t.GeneralValueType, str]].fail(
                "Handler is required",
            )

        # Cast handler to the expected type for validation
        handler_typed: (
            t.Handler.HandlerType
            | t.Handler.HandlerCallable
            | p.Utility.Callable[t.GeneralValueType]
            | BaseModel
        ) = cast(
            "t.Handler.HandlerType | t.Handler.HandlerCallable | p.Utility.Callable[t.GeneralValueType] | BaseModel",
            handler_raw,
        )

        validation_result = self._validate_handler_registry_interface(
            handler_typed,
            handler_context="registered handler",
        )
        if validation_result.is_failure:
            return r[tuple[t.GeneralValueType, str]].fail(
                validation_result.error or "Handler validation failed",
            )

        # Cast handler to GeneralValueType for _extract_handler_name
        handler_for_extraction: t.GeneralValueType = cast(
            "t.GeneralValueType",
            handler_typed,
        )
        handler_name = self._extract_handler_name(handler_for_extraction, request_dict)
        if not handler_name:
            return r[tuple[t.GeneralValueType, str]].fail(
                "handler_name is required",
            )

        # Cast handler_typed to GeneralValueType for return
        handler_for_return: t.GeneralValueType = cast(
            "t.GeneralValueType",
            handler_typed,
        )
        return r[tuple[t.GeneralValueType, str]].ok(
            (handler_for_return, handler_name),
        )

    def _register_handler_by_mode(
        self,
        handler: t.GeneralValueType,
        handler_name: str,
        request_dict: t.Types.ConfigurationMapping,
    ) -> r[t.Types.ConfigurationMapping]:
        """Register handler based on auto-discovery or explicit mode.

        Args:
            handler: Validated handler instance
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
        handler: t.GeneralValueType,
        handler_name: str,
    ) -> r[t.Types.ConfigurationMapping]:
        """Register handler with auto-discovery mode.

        Args:
            handler: Handler instance with can_handle() method
            handler_name: Handler name for tracking

        Returns:
            r with registration details

        """
        # Cast handler to HandlerType for append
        handler_typed: t.Handler.HandlerType = cast(
            "t.Handler.HandlerType",
            handler,
        )
        if handler_typed not in self._auto_handlers:
            self._auto_handlers.append(handler_typed)

        return r[t.Types.ConfigurationMapping].ok({
            "handler_name": handler_name,
            "status": "registered",
            "mode": "auto_discovery",
        })

    def _register_explicit_handler(
        self,
        handler: t.GeneralValueType,
        handler_name: str,
        request_dict: t.Types.ConfigurationMapping,
    ) -> r[t.Types.ConfigurationMapping]:
        """Register handler with explicit mode.

        Args:
            handler: Handler instance
            handler_name: Handler name for tracking
            request_dict: Normalized request dictionary

        Returns:
            r with registration details or error

        """
        message_type = request_dict.get("message_type")
        if not message_type:
            return r[t.Types.ConfigurationMapping].fail(
                "Handler without can_handle() requires message_type",
            )

        name_attr = (
            getattr(message_type, "__name__", None)
            if hasattr(message_type, "__name__")
            else None
        )
        message_type_name = name_attr if name_attr is not None else str(message_type)

        # Cast handler to HandlerType for assignment
        handler_typed: t.Handler.HandlerType = cast(
            "t.Handler.HandlerType",
            handler,
        )
        self._handlers[message_type_name] = handler_typed

        return r[t.Types.ConfigurationMapping].ok({
            "handler_name": handler_name,
            "message_type": message_type_name,
            "status": "registered",
            "mode": "explicit",
        })

    def register_handler_with_request(
        self,
        request: t.GeneralValueType,
    ) -> r[t.Types.ConfigurationMapping]:
        """Register handler using structured request model.

        Business Rule: Handler registration supports two modes:
        1. Auto-discovery: handlers with can_handle() method are queried at dispatch time
        2. Explicit: handlers registered for specific message_type

        This dual-mode architecture enables both dynamic routing (handlers decide
        what they can handle) and static routing (pre-registered type mappings).

        Args:
            request: Dict or Pydantic model containing registration details

        Returns:
            r with registration details or error

        """
        # Normalize and validate request
        request_dict_result = FlextDispatcher._normalize_request_to_dict(request)
        if request_dict_result.is_failure:
            return r[t.Types.ConfigurationMapping].fail(
                request_dict_result.error or "Failed to normalize request",
            )
        # Use .value directly - FlextResult never returns None on success
        request_dict = request_dict_result.value

        # Validate handler mode
        handler_mode_raw = request_dict.get("handler_mode")
        handler_mode = (
            handler_mode_raw
            if isinstance(handler_mode_raw, (str, type(None)))
            else None
        )
        mode_validation = self._validate_handler_mode(handler_mode)
        if mode_validation.is_failure:
            return r[t.Types.ConfigurationMapping].fail(
                mode_validation.error or "Invalid handler mode",
            )

        # Validate and extract handler
        handler_result = self._validate_and_extract_handler(request_dict)
        if handler_result.is_failure:
            return r[t.Types.ConfigurationMapping].fail(
                handler_result.error or "Handler validation failed",
            )
        # Use .value directly - FlextResult never returns None on success
        handler_general, handler_name = handler_result.value
        # Cast GeneralValueType to HandlerType for storage in typed collections
        handler = cast("t.Handler.HandlerType", handler_general)

        # Determine registration mode and register
        can_handle_attr = getattr(handler, "can_handle", None)
        if callable(can_handle_attr):
            # Auto-discovery mode
            if handler not in self._auto_handlers:
                self._auto_handlers.append(handler)
            return r[t.Types.ConfigurationMapping].ok({
                "handler_name": handler_name,
                "status": "registered",
                "mode": "auto_discovery",
            })

        # Explicit registration requires message_type
        message_type = request_dict.get("message_type")
        if not message_type:
            return r[t.Types.ConfigurationMapping].fail(
                "Handler without can_handle() requires message_type",
            )

        # Get message type name and store handler
        name_attr = getattr(message_type, "__name__", None)
        message_type_name = name_attr if name_attr is not None else str(message_type)
        self._handlers[message_type_name] = handler

        return r[t.Types.ConfigurationMapping].ok({
            "handler_name": handler_name,
            "message_type": message_type_name,
            "status": "registered",
            "mode": "explicit",
        })

    def register_handler(
        self,
        request: t.GeneralValueType | BaseModel,
        handler: t.GeneralValueType | None = None,
    ) -> r[t.Types.ConfigurationMapping]:
        """Register a handler dynamically.

        Args:
            request: Dict or Pydantic model containing registration details, or message_type string
            handler: Handler instance

        Returns:
            r with registration details or error

        Example:
            >>> from dataclasses import dataclass
            >>> from flext_core import FlextDispatcher, FlextResult
            >>>
            >>> @dataclass
            ... class CreateUser:
            ...     email: str
            >>>
            >>> def handle_create_user(message: CreateUser) -> FlextResult[str]:
            ...     if "@" not in message.email:
            ...         return FlextResult[str].fail("Invalid email")
            ...     return FlextResult[str].ok(f"Created: {message.email}")
            >>>
            >>> dispatcher = FlextDispatcher()
            >>> result = dispatcher.register_handler(CreateUser, handle_create_user)
            >>> if result.is_success:
            ...     print("Handler registered successfully")

        """
        if handler is not None:
            # Cast request and handler to HandlerType for layer1_register_handler
            request_typed_for_reg: t.Handler.HandlerType = cast(
                "t.Handler.HandlerType",
                request,
            )
            handler_typed: t.Handler.HandlerType = cast(
                "t.Handler.HandlerType",
                handler,
            )
            result = self.layer1_register_handler(request_typed_for_reg, handler_typed)
            if result.is_failure:
                return r[t.Types.ConfigurationMapping].fail(
                    result.error or "Registration failed",
                )
            # Convert to dict format for consistency
            handler_name: str = request if isinstance(request, str) else "unknown"
            return r[t.Types.ConfigurationMapping].ok({
                "handler_name": handler_name,
                "status": "registered",
                "mode": "explicit",
            })

        # Single-arg mode: register_handler(dict_or_model_or_handler)
        if isinstance(request, BaseModel) or FlextRuntime.is_dict_like(request):
            # Delegate to register_handler_with_request (eliminates ~100 lines of duplication)
            # Safe cast: request is BaseModel or dict-like, compatible with GeneralValueType
            request_for_registration: t.GeneralValueType = cast(
                "t.GeneralValueType",
                request,
            )
            return self.register_handler_with_request(request_for_registration)

        # Single handler object - delegate to layer1_register_handler
        # Cast request to HandlerType
        request_typed_for_single: t.Handler.HandlerType = cast(
            "t.Handler.HandlerType",
            request,
        )
        result = self.layer1_register_handler(request_typed_for_single)
        if result.is_failure:
            return r[t.Types.ConfigurationMapping].fail(
                result.error or "Registration failed",
            )
        # Convert to dict format for consistency
        handler_name = getattr(request, "__class__", type(request)).__name__
        return r[t.Types.ConfigurationMapping].ok({
            "handler_name": handler_name,
            "status": "registered",
            "mode": "auto_discovery",
        })

    def _register_handler[TMessage, TResult](
        self,
        message_type: type[TMessage],
        handler: h[TMessage, TResult],
        handler_mode: str,
        handler_config: t.Types.ConfigurationMapping | None = None,
    ) -> r[t.GeneralValueType]:
        """Register handler with specific mode (DRY helper).

        Eliminates duplication between register_command and register_query.
        Both methods follow identical pattern: create request dict → call
        register_handler_with_request().

        Args:
            message_type: Command or query message type
            handler: Handler instance
            handler_mode: Handler mode constant (COMMAND or QUERY)
            handler_config: Optional handler configuration

        Returns:
            r with registration details or error

        """
        # Cast handler and message_type to GeneralValueType for dict construction
        # Convert message_type (type[TMessage]) to string name to avoid type variable scope issue
        message_type_name = getattr(message_type, "__name__", str(message_type))
        request: t.Types.ConfigurationDict = {
            "handler": cast("t.GeneralValueType", handler),
            "message_type": message_type_name,
            "handler_mode": handler_mode,
            "handler_config": handler_config,
        }

        result = self.register_handler_with_request(
            cast("t.GeneralValueType", request),
        )
        # Type narrowing: ConfigurationMapping is a subtype of GeneralValueType
        # (ConfigurationMapping = Mapping[str, GeneralValueType], which is part of GeneralValueType union)
        if result.is_success:
            # result.value is ConfigurationMapping, which is already GeneralValueType
            return r[t.GeneralValueType].ok(result.value)
        return r[t.GeneralValueType].fail(
            result.error or "Registration failed",
        )

    def register_command[TCommand, TResult](
        self,
        command_type: type[TCommand],
        handler: h[TCommand, TResult],
        *,
        handler_config: t.Types.ConfigurationMapping | None = None,
    ) -> r[t.GeneralValueType]:
        """Register command handler using structured model internally.

        Args:
            command_type: Command message type
            handler: Handler instance
            handler_config: Optional handler configuration

        Returns:
            r with registration details or error

        """
        return self._register_handler(
            command_type,
            handler,
            c.Dispatcher.HANDLER_MODE_COMMAND,
            handler_config,
        )

    def register_query[TQuery, TResult](
        self,
        query_type: type[TQuery],
        handler: h[TQuery, TResult],
        *,
        handler_config: t.Types.ConfigurationMapping | None = None,
    ) -> r[t.GeneralValueType]:
        """Register query handler using structured model internally.

        Args:
            query_type: Query message type
            handler: Handler instance
            handler_config: Optional handler configuration

        Returns:
            r with registration details or error

        """
        return self._register_handler(
            query_type,
            handler,
            c.Dispatcher.HANDLER_MODE_QUERY,
            handler_config,
        )

    def register_function[TMessage, TResult](
        self,
        message_type: type[TMessage],
        handler_func: Callable[[TMessage], TResult],
        *,
        handler_config: t.Types.ConfigurationMapping | None = None,
        mode: c.Cqrs.HandlerType = c.Cqrs.HandlerType.COMMAND,
    ) -> r[t.GeneralValueType]:
        """Register function as handler using factory pattern.

        Args:
            message_type: Message type to handle
            handler_func: Function to wrap as handler
            handler_config: Optional handler configuration
            mode: Handler mode (command/query)

        Returns:
            r with registration details or error

        """
        # Validate mode
        if mode not in c.Dispatcher.VALID_HANDLER_MODES:
            return r[t.GeneralValueType].fail(
                c.Dispatcher.ERROR_INVALID_HANDLER_MODE,
            )

        # Simple registration for basic test compatibility
        if not handler_config:
            # Cast handler_func to HandlerType for assignment
            handler_func_typed: t.Handler.HandlerType = cast(
                "t.Handler.HandlerType",
                handler_func,
            )
            # Access __name__ attribute safely - type objects have this attribute
            handler_key = getattr(message_type, "__name__", str(message_type))
            self._handlers[handler_key] = handler_func_typed
            return r[t.GeneralValueType].ok({
                "status": "registered",
                "mode": mode,
            })

        # Create handler from function
        # Wrap generic handler_func to match HandlerCallableType signature
        def wrapped_handler(
            msg: t.GeneralValueType,
        ) -> t.GeneralValueType:
            # handler_func is callable, accept any arguments and convert result
            # Cast msg to TMessage for handler_func call
            msg_typed: TMessage = cast("TMessage", msg)
            result_raw = handler_func(msg_typed) if callable(handler_func) else msg
            # Convert result to GeneralValueType
            if isinstance(
                result_raw,
                (str, int, float, bool, type(None), list, dict, Mapping, Sequence),
            ):
                return cast("t.GeneralValueType", result_raw)
            return str(result_raw)

        handler_result = self.create_handler_from_function(
            wrapped_handler,
            handler_config,
            mode,
        )

        if handler_result.is_failure:
            return r[t.GeneralValueType].fail(
                f"Handler creation failed: {handler_result.error}",
            )

        # Register the created handler
        # Cast handler and message_type to GeneralValueType for dict construction
        # Convert message_type (type[TMessage]) to string name to avoid type variable scope issue
        message_type_name = getattr(message_type, "__name__", str(message_type))
        request: t.Types.ConfigurationDict = {
            "handler": cast("t.GeneralValueType", handler_result.value),
            "message_type": message_type_name,
            "handler_mode": mode,
            "handler_config": handler_config,
        }

        result = self.register_handler_with_request(
            cast("t.GeneralValueType", request),
        )
        # Type narrowing: ConfigurationMapping is a subtype of GeneralValueType
        # (ConfigurationMapping = Mapping[str, GeneralValueType], which is part of GeneralValueType union)
        if result.is_success:
            # result.value is ConfigurationMapping, which is already GeneralValueType
            return r[t.GeneralValueType].ok(result.value)
        return r[t.GeneralValueType].fail(
            result.error or "Registration failed",
        )

    @staticmethod
    def create_handler_from_function(
        handler_func: t.Handler.HandlerCallable,
        _handler_config: t.Types.ConfigurationMapping | None = None,
        mode: c.Cqrs.HandlerType = c.Cqrs.HandlerType.COMMAND,
    ) -> r[
        h[
            t.GeneralValueType,
            t.GeneralValueType,
        ]
    ]:
        """Create handler from function using h constructor.

        Args:
            handler_func: Function to wrap
            _handler_config: Optional configuration (reserved for future use)
            mode: Handler mode

        Returns:
            r with handler instance or error

        """
        try:
            # Create concrete handler class that implements handle method
            handler_name = getattr(handler_func, "__name__", "FunctionHandler")

            class FunctionHandler(
                h[
                    t.GeneralValueType,
                    t.GeneralValueType,
                ],
            ):
                """Concrete handler implementation for function-based handlers."""

                def handle(
                    self,
                    message: t.GeneralValueType,
                ) -> r[t.GeneralValueType]:
                    """Handle message by calling the wrapped function.

                    Note: self is required for protocol compliance (h),
                    even though handler_func is captured from closure.
                    """
                    result = handler_func(message)
                    # Ensure result is r
                    if isinstance(result, r):
                        return result
                    # Wrap non-r return values
                    return r[t.GeneralValueType].ok(result)

            # Create handler config with name and type
            handler_config = m.Cqrs.Handler(
                handler_id=f"function_{id(handler_func)}",
                handler_name=handler_name,
                handler_mode=mode,
            )
            handler = FunctionHandler(config=handler_config)
            return r[
                h[
                    t.GeneralValueType,
                    t.GeneralValueType,
                ]
            ].ok(handler)

        except Exception as error:
            return r[
                h[
                    t.GeneralValueType,
                    t.GeneralValueType,
                ]
            ].fail(
                f"Handler creation failed: {error}",
            )

    @staticmethod
    def _ensure_handler(
        handler: t.GeneralValueType,
        mode: c.Cqrs.HandlerType = c.Cqrs.HandlerType.COMMAND,
    ) -> r[
        h[
            t.GeneralValueType,
            t.GeneralValueType,
        ]
    ]:
        """Ensure handler is a h instance, converting from callable if needed.

        Private helper to eliminate duplication in handler registration.

        Args:
            handler: Handler instance or callable to convert
            mode: Handler operation mode (command/query)

        Returns:
            r with h instance or error

        """
        # If already h, return success
        if isinstance(handler, h):
            return r[
                h[
                    t.GeneralValueType,
                    t.GeneralValueType,
                ]
            ].ok(handler)

        # If callable, convert to h
        if callable(handler):
            return FlextDispatcher.create_handler_from_function(
                handler_func=handler,
                mode=mode,
            )

        # Invalid handler type
        return r[
            h[
                t.GeneralValueType,
                t.GeneralValueType,
            ]
        ].fail(
            (f"Handler must be h instance or callable, got {type(handler).__name__}"),
        )

    # ------------------------------------------------------------------
    # Dispatch execution using structured models
    # ------------------------------------------------------------------
    def dispatch_with_request(
        self,
        request: t.GeneralValueType | BaseModel,
    ) -> r[t.GeneralValueType]:
        """Enhanced dispatch accepting Pydantic models or dicts.

        Args:
            request: Dict or Pydantic model containing dispatch details

        Returns:
            r with structured dispatch result

        """
        # Convert Pydantic model to dict if needed
        # Validate request type and convert using consolidated helper
        if not isinstance(request, BaseModel) and not FlextRuntime.is_dict_like(
            request,
        ):
            return r[t.GeneralValueType].fail(
                "Request must be dict or Pydantic model",
            )
        # Normalize request to GeneralValueType dict using ModelConversion helper
        request_dict = x.ModelConversion.to_dict(request)

        # Propagate context for distributed tracing
        correlation_id = request_dict.get("correlation_id")
        if correlation_id and isinstance(correlation_id, str):
            FlextContext.Correlation.set_correlation_id(correlation_id)

        # Execute dispatch
        dispatch_result = self.dispatch(request_dict)

        # Wrap result in structured format
        if dispatch_result.is_success:
            structured_result = {
                "status": "success",
                "data": dispatch_result.value,
                "correlation_id": FlextContext.Correlation.get_correlation_id(),  # Runtime access needs concrete class
            }
            return r[t.GeneralValueType].ok(structured_result)
        error_msg = dispatch_result.error or "Dispatch failed"
        return r[t.GeneralValueType].fail(error_msg)

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
        config: m.Config.DispatchConfig | t.GeneralValueType | None = None,
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

        Example:
            >>> from dataclasses import dataclass
            >>> from flext_core import FlextDispatcher, FlextResult
            >>>
            >>> @dataclass
            ... class CreateUser:
            ...     email: str
            >>>
            >>> dispatcher = FlextDispatcher()
            >>>
            >>> def handle_create_user(message: CreateUser) -> FlextResult[str]:
            ...     return FlextResult[str].ok(f"Created user: {message.email}")
            >>>
            >>> dispatcher.register_handler(CreateUser, handle_create_user)
            >>> result = dispatcher.dispatch(CreateUser(email="user@example.com"))
            >>> if result.is_success:
            ...     print(result.value)

        """
        # Detect API pattern - (type, data) vs (object)
        message: t.GeneralValueType
        if data is not None or u.is_type(message_or_type, str):
            # dispatch("type", data) pattern
            message_type_str = str(message_or_type)
            message_class = type(message_type_str, (), {"payload": data})
            message_raw = message_class()
            # Safe cast: dynamically created class instance is compatible with GeneralValueType
            message = cast(
                "t.GeneralValueType",
                message_raw,
            )
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
                handler: Callable[
                    [t.GeneralValueType],
                    t.GeneralValueType,
                ] = handler_raw
                result_raw = handler(message)
                # Ensure result is GeneralValueType
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
        # DispatchConfig (BaseModel) is compatible with GeneralValueType (includes BaseModel via Mapping)
        return self._execute_dispatch_pipeline(
            message,
            cast("t.GeneralValueType | None", dispatch_config),
            metadata,
            correlation_id,
            timeout_override,
        )

    @staticmethod
    def _convert_metadata_to_model(
        metadata: t.GeneralValueType | None,
    ) -> m.Metadata | None:
        """Convert metadata from GeneralValueType to m.Metadata model.

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
        if isinstance(metadata, (dict, Mapping)):

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
                if isinstance(v, (dict, Mapping)):
                    # Use process() for concise dict conversion (transform values and convert keys)
                    transform_result = u.Collection.process(
                        v,
                        lambda _k, v2: v2
                        if isinstance(v2, (str, int, float, bool, type(None)))
                        else str(v2),
                        on_error="collect",
                    )
                    if transform_result.is_success and isinstance(
                        transform_result.value,
                        dict,
                    ):
                        # Convert keys to strings - u.map preserves keys, so use dict comprehension
                        # for key transformation (u.map only transforms values)
                        return {str(k): v for k, v in transform_result.value.items()}
                    return {}
                return str(v)

            # Use process() for metadata conversion
            process_result = u.Collection.process(
                metadata,
                lambda _k, v: convert_metadata_value(v),
                on_error="collect",
            )
            if process_result.is_success and isinstance(process_result.value, dict):
                # Convert keys to strings
                attributes_dict = {str(k): v for k, v in process_result.value.items()}
            else:
                attributes_dict = {}

            # Cast attributes_dict to t.Types.ConfigurationDict for invariance compatibility
            attributes_general: t.Types.ConfigurationDict = cast(
                "t.Types.ConfigurationDict",
                attributes_dict,
            )
            return m.Metadata(attributes=attributes_general)
        # Convert other types to Metadata via dict with string value
        return m.Metadata(attributes={"value": str(metadata)})

    @staticmethod
    def _build_dispatch_config_from_args(
        config: m.Config.DispatchConfig | t.GeneralValueType | None,
        metadata: t.GeneralValueType | None,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> m.Config.DispatchConfig | t.GeneralValueType | None:
        """Build DispatchConfig from arguments if not provided.

        Args:
            config: DispatchConfig instance or legacy config dict
            metadata: Optional execution context metadata
            correlation_id: Optional correlation ID for tracing
            timeout_override: Optional timeout override

        Returns:
            DispatchConfig instance or original config

        """
        if isinstance(config, m.Config.DispatchConfig):
            return config
        if config is not None:
            return config
        if metadata is None and correlation_id is None and timeout_override is None:
            return None

        # Build from individual arguments
        metadata_model = FlextDispatcher._convert_metadata_to_model(metadata)
        return m.Config.DispatchConfig(
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
            handler: Callable[
                [t.GeneralValueType],
                t.GeneralValueType,
            ] = handler_raw
            result = handler(message)
            return r[t.GeneralValueType].ok(result)
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
        if config_result.is_failure:
            return r[t.GeneralValueType].fail(
                config_result.error or "Config extraction failed",
            )

        # Prepare dispatch context
        context_result = self._prepare_dispatch_context(
            message,
            None,
            config_result.value,
        )
        if context_result.is_failure:
            return r[t.GeneralValueType].fail(
                context_result.error or "Context preparation failed",
            )

        # Validate pre-dispatch conditions
        validated_result = self._validate_pre_dispatch_conditions(
            context_result.value,
        )
        if validated_result.is_failure:
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
    ) -> r[t.GeneralValueType]:
        """Extract and validate dispatch configuration using u."""
        try:
            # Extract config values (config takes priority over individual params)
            if config is not None:
                metadata = getattr(config, "metadata", metadata)
                correlation_id = getattr(config, "correlation_id", correlation_id)
                timeout_override = getattr(config, "timeout_override", timeout_override)

            # Validate metadata - NO fallback, explicit validation
            validated_metadata: t.GeneralValueType
            if metadata is None:
                validated_metadata = {}
            elif FlextRuntime.is_dict_like(metadata):
                validated_metadata = dict(metadata)
            elif isinstance(metadata, m.Metadata):
                # m.Metadata - extract attributes dict
                validated_metadata = metadata.attributes
            else:
                # Fast fail: metadata must be dict, m.Metadata, or None
                # Type checker may think this is unreachable, but it's reachable at runtime
                msg = (
                    f"Invalid metadata type: {type(metadata).__name__}. "
                    "Expected object | m.Metadata | None"
                )
                return r[t.GeneralValueType].fail(msg)

            # Use u for configuration validation
            # Fast fail: explicit type annotation instead of cast
            config_dict: t.GeneralValueType = {
                "metadata": validated_metadata,
                "correlation_id": correlation_id,
                "timeout_override": timeout_override,
            }

            # Cast config_dict to Mapping[str, GeneralValueType] for validate_dispatch_config
            config_dict_mapping: t.Types.ConfigurationMapping = cast(
                "t.Types.ConfigurationMapping",
                config_dict,
            )
            validation_result = u.Validation.validate_dispatch_config(
                config_dict_mapping,
            )
            if validation_result.is_failure:
                return r[t.GeneralValueType].fail(
                    f"Invalid dispatch configuration: {validation_result.error}",
                )

            return r[t.GeneralValueType].ok(config_dict)

        except Exception as e:
            return r[t.GeneralValueType].fail(
                f"Configuration extraction failed: {e}",
            )

    def _prepare_dispatch_context(
        self,
        message: t.GeneralValueType,
        _data: t.GeneralValueType | None,
        dispatch_config: t.GeneralValueType,
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

            # Fast fail: dispatch_config must be dict-like for unpacking
            if not FlextRuntime.is_dict_like(dispatch_config):
                return r[t.GeneralValueType].fail(
                    f"dispatch_config must be dict-like, got {type(dispatch_config).__name__}",
                )
            # dispatch_config is dict-like at this point - convert to dict
            dispatch_config_dict: t.Types.ConfigurationMapping = dict(dispatch_config)

            context: t.Types.ConfigurationMapping = {
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
        if not FlextRuntime.is_dict_like(context):
            msg = f"Context must be dict-like, got {type(context).__name__}"
            return r[t.GeneralValueType].fail(msg)
        # Fast fail: message_type must be str (created by _normalize_dispatch_message)
        # Type narrowing: context is dict-like (checked above), which implements Mapping
        message_type_raw = context.get("message_type")
        if not isinstance(message_type_raw, str):
            msg = f"Invalid message_type in context: {type(message_type_raw).__name__}, expected str"
            return r[t.GeneralValueType].fail(msg)
        message_type: str = message_type_raw

        # Check pre-dispatch conditions (circuit breaker + rate limiting)
        conditions_check = self._check_pre_dispatch_conditions(message_type)
        if conditions_check.is_failure:
            # Use u.err() for unified error extraction (DSL pattern)
            error_msg = u.err(
                conditions_check,
                default="Pre-dispatch conditions check failed",
            )
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
        if not FlextRuntime.is_dict_like(context):
            msg = f"Context must be dict-like, got {type(context).__name__}"
            return r[t.GeneralValueType].fail(msg)
        # Fast fail: validate context values (created by _prepare_dispatch_context)
        # Type narrowing: context is dict-like (checked above), which implements Mapping
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
        operation_id = u.Generators.generate_operation_id(
            message_type,
            message,
        )

        # Use u for retry execution
        options = m.Config.ExecuteDispatchAttemptOptions(
            message_type=message_type,
            metadata=metadata,
            correlation_id=correlation_id,
            timeout_override=timeout_override,
            operation_id=operation_id,
        )
        return u.Reliability.with_retry(
            lambda: self._execute_dispatch_attempt(message, options),
            max_attempts=self._retry_policy.get_max_attempts(),
            should_retry_func=self._should_retry_on_error,
            cleanup_func=lambda: self._cleanup_timeout_context(operation_id),
        )

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

    @staticmethod
    def _create_message_wrapper(
        data: t.GeneralValueType,
        message_type: str,
    ) -> t.GeneralValueType:
        """Create message wrapper for string message types."""

        class MessageWrapper(m.Value):
            """Temporary message wrapper using FlextModels.Value."""

            data: t.GeneralValueType
            message_type: str

            def model_post_init(
                self,
                /,
                __context: t.GeneralValueType | None = None,
            ) -> None:
                """Post-initialization to set class name."""
                super().model_post_init(__context)
                self.__class__.__name__ = self.message_type

            def __str__(self) -> str:
                """String representation."""
                return str(self.data)

        # Cast MessageWrapper to GeneralValueType
        wrapper_instance = MessageWrapper(data=data, message_type=message_type)
        return cast("t.GeneralValueType", wrapper_instance)

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
                context_metadata: t.Types.ConfigurationDict = {}
                if timeout_override is not None:
                    context_metadata["timeout_override"] = timeout_override
                with self._context_scope(
                    cast("t.GeneralValueType", context_metadata),
                    correlation_id,
                ):
                    return self.execute(message)
            return self.execute(message)

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
        options: m.Config.ExecuteDispatchAttemptOptions,
    ) -> r[t.GeneralValueType]:
        """Execute a single dispatch attempt with timeout."""
        try:
            # Create structured request
            if options.metadata and u.is_type(options.metadata, "mapping"):
                # Use process() for concise conversion (transform values)
                transform_result = u.Collection.process(
                    options.metadata,
                    lambda _k, v: str(v),
                    on_error="collect",
                )
                # Convert keys to strings and values to MetadataAttributeValue
                # Convert keys to strings and values to MetadataAttributeValue
                metadata_attrs: t.Types.MetadataAttributeDict
                if transform_result.is_success and isinstance(
                    transform_result.value,
                    dict,
                ):
                    metadata_attrs = {
                        str(k): cast("t.MetadataAttributeValue", v)
                        for k, v in transform_result.value.items()
                    }
                else:
                    metadata_attrs = {}
                metadata_attrs_general = cast(
                    "t.Types.ConfigurationDict",
                    metadata_attrs,
                )
                _ = m.Metadata(attributes=metadata_attrs_general)

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
    ) -> t.Types.ConfigurationDict | None:
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

        # Use process() for concise key normalization - convert keys to strings
        normalized = {str(k): v for k, v in raw_metadata.items()}

        return cast("t.Types.ConfigurationDict | None", normalized)

    @staticmethod
    def _extract_metadata_mapping(
        metadata: t.GeneralValueType,
    ) -> t.Types.ConfigurationMapping | None:
        """Extract metadata as Mapping from various types.

        Fast fail: Direct validation without helpers.
        """
        if isinstance(metadata, m.Metadata):
            return FlextDispatcher._extract_from_flext_metadata(metadata)
        # Fast fail: type narrowing instead of cast
        if isinstance(metadata, (dict, Mapping)):
            return metadata

        # Handle Pydantic models directly - use model_dump() (Pydantic v2 pattern)
        if isinstance(metadata, BaseModel):
            # Fast fail: model_dump() must succeed for valid Pydantic models
            try:
                dumped = metadata.model_dump()
            except Exception as e:
                # Fast fail: model_dump() failure indicates invalid model
                msg = f"Failed to dump BaseModel metadata: {type(e).__name__}: {e}"
                raise TypeError(msg) from e

            # Fast fail: dumped must be dict (Pydantic guarantees this)
            if not FlextRuntime.is_dict_like(dumped):
                # Type checker may think this is unreachable, but it's reachable at runtime
                msg = (
                    f"metadata.model_dump() returned {type(dumped).__name__}, "
                    "expected dict"
                )
                raise TypeError(msg)
            # model_dump() returns dict, which implements Mapping[str, GeneralValueType]
            return dumped

        return FlextDispatcher._extract_from_object_attributes(metadata)

    @staticmethod
    def _extract_from_flext_metadata(
        metadata: m.Metadata,
    ) -> t.Types.ConfigurationMapping | None:
        """Extract metadata mapping from m.Metadata."""
        attributes = metadata.attributes
        # metadata.attributes is t.Types.ConfigurationDict
        # Use guard() with lambda for concise non-empty validation

        def non_empty_check(v: t.Types.ConfigurationDict) -> bool:
            return len(v) > 0

        # attributes is already t.Types.ConfigurationDict
        if attributes and not u.Validation.guard(
            attributes,
            non_empty_check,
            return_value=True,
        ):
            return attributes

        # Use model directly - Pydantic v2 supports direct attribute access
        # Fast fail: model_dump() must succeed for valid Pydantic models
        try:
            dumped = metadata.model_dump()
        except Exception as e:
            # Fast fail: model_dump() failure indicates invalid model
            msg = f"Failed to dump m.Metadata: {type(e).__name__}: {e}"
            raise TypeError(msg) from e

        # Fast fail: dumped must be dict (Pydantic guarantees this)
        if not FlextRuntime.is_dict_like(dumped):
            # Type checker may think this is unreachable, but it's reachable at runtime
            msg = (
                f"metadata.model_dump() returned {type(dumped).__name__}, expected dict"
            )
            raise TypeError(msg)

        # Extract attributes section if present - fast fail: must be dict or None
        attributes_section_raw = dumped.get("attributes")
        if attributes_section_raw is not None and FlextRuntime.is_dict_like(
            attributes_section_raw,
        ):
            return attributes_section_raw
        # Return full dump if no attributes section
        return dumped

    @staticmethod
    def _extract_from_object_attributes(
        metadata: t.GeneralValueType,
    ) -> t.Types.ConfigurationMapping | None:
        """Extract metadata mapping from object's attributes."""
        attributes_value = getattr(metadata, "attributes", None)
        if isinstance(attributes_value, (dict, Mapping)) and attributes_value:
            # Type narrowing: attributes_value is dict-like, convert to ConfigurationMapping
            # ConfigurationMapping is compatible with dict and Mapping types
            attributes_dict: t.Types.ConfigurationMapping = cast(
                "t.Types.ConfigurationMapping",
                attributes_value,
            )
            return attributes_dict

        # Use model_dump() directly if available - Pydantic v2 pattern
        model_dump = getattr(metadata, "model_dump", None)
        if callable(model_dump):
            # Fast fail: model_dump() must succeed for valid Pydantic models
            try:
                dumped = model_dump()
            except Exception as e:
                # Fast fail: model_dump() failure indicates invalid model
                msg = f"Failed to dump metadata object: {type(e).__name__}: {e}"
                raise TypeError(msg) from e

            # Fast fail: dumped must be dict (Pydantic guarantees this)
            # Cast to GeneralValueType for is_dict_like check
            dumped_as_general = cast("t.GeneralValueType", dumped)
            if not FlextRuntime.is_dict_like(dumped_as_general):
                # Type checker may think this is unreachable, but it's reachable at runtime
                msg = (
                    f"metadata.model_dump() returned {type(dumped).__name__}, "
                    "expected dict"
                )
                raise TypeError(msg)
            # Safe cast: model_dump() returns t.Types.ConfigurationDict, which is compatible with GeneralValueType
            return dumped_as_general

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

        metadata_var = FlextContext.Variables.Performance.OPERATION_METADATA
        correlation_var = FlextContext.Variables.Correlation.CORRELATION_ID
        parent_var = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID

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
        if metadata:
            # Cast metadata to t.Types.ConfigurationDict | None for set()
            metadata_dict: t.Types.ConfigurationDict | None = (
                cast("t.Types.ConfigurationDict | None", metadata)
                if FlextRuntime.is_dict_like(metadata)
                else None
            )
            if metadata_dict is not None:
                _ = metadata_var.set(metadata_dict)

            # Use provided correlation ID or generate one if needed
            effective_correlation_id = correlation_id
            if effective_correlation_id is None:
                effective_correlation_id = (
                    FlextContext.Correlation.generate_correlation_id()
                )

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

        Example:
            >>> dispatcher = FlextDispatcher.create(auto_discover_handlers=True)
            >>> result = dispatcher.dispatch(CreateUserCommand(name="Alice"))

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
                    handlers = h.Discovery.scan_module(caller_module)
                    for _handler_name, handler_func, handler_config in handlers:
                        # Get actual handler function from module
                        if handler_func and callable(handler_func):
                            # Register handler with dispatcher
                            # Register under the handler command type name for routing
                            command_type_name = (
                                handler_config.command.__name__
                                if hasattr(handler_config.command, "__name__")
                                else str(handler_config.command)
                            )
                            _ = instance.register_handler(
                                command_type_name,
                                handler_func,
                            )

        return instance

    @classmethod
    def create_from_global_config(cls) -> r[FlextDispatcher]:
        """Create dispatcher using global FlextConfig instance.

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
        # Use u.map for concise batch processing
        return list(u.Collection.map(messages, self.dispatch))

    def get_performance_metrics(
        self,
    ) -> t.Types.ConfigurationDict:
        """Get performance metrics for the dispatcher.

        Returns:
            object: Dictionary containing performance metrics

        """
        # Get metrics from circuit breaker manager
        cb_metrics = self._circuit_breaker.get_metrics()
        executor_status = self._timeout_enforcer.get_executor_status()
        # Cast all values to GeneralValueType
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
