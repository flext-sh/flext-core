"""Automation decorators for infrastructure concerns.

This module provides FlextDecorators, a collection of decorators that
automatically handle common infrastructure concerns to reduce boilerplate
code in services, handlers, and other components.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import suppress
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Literal, NoReturn, ParamSpec, TypeIs, TypeVar, overload

from flext_core import FlextContainer, FlextContext, c, e, m, p, r, t, u

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class FlextDecorators:
    """Automation decorators for infrastructure concerns.

    Provides decorators that automatically handle common infrastructure
    concerns to reduce boilerplate code in services, handlers, and other
    components. All decorators are designed to integrate seamlessly with
    `r`, `FlextContext`, the public logging DSL in `u`, and `FlextContainer`.
    """

    type _LoggerCarrier = p.HasLogger | p.Logger | t.RuntimeAtomic | m.BaseModel

    @staticmethod
    def _resolve_logger(
        first_arg: p.Logger | FlextDecorators._LoggerCarrier | None = None,
        *,
        func: t.DispatchableHandler | None = None,
        func_module: str | None = None,
    ) -> p.Logger:
        """Resolve the logger associated with the decorated call."""
        if isinstance(first_arg, p.Logger):
            return first_arg
        if first_arg is not None and FlextDecorators._has_flext_logger(first_arg):
            logger_value = first_arg.logger
            if isinstance(logger_value, p.Logger):
                return logger_value
        module_name = (
            func_module
            if isinstance(func_module, str)
            else (func.__module__ if callable(func) else __name__)
        )
        return u.fetch_logger(module_name)

    @staticmethod
    def deprecated(message: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to mark functions/variables as deprecated.

        Emits DeprecationWarning when decorated function is called.
        Used during v0.10 → v0.11 refactoring for constants migration.

        Args:
            message: Deprecation message explaining what to use instead

        Returns:
            Decorator function that wraps the target callable

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            """Apply deprecation warning to callable."""

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                """Wrapper that emits warning before execution."""
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def inject(**dependencies: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to automatically inject dependencies from FlextContainer.

        Automatically resolves and injects dependencies from the global
        container, eliminating manual container.resolve() calls in every
        method.

        Args:
            **dependencies: Mapping of parameter names to service types to
                inject

        Returns:
            Decorated function with automatic dependency injection

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                container = FlextContainer.create()
                for name, service_key in dependencies.items():
                    if name not in kwargs:
                        result = container.get(service_key)
                        if result.success:
                            kwargs[name] = result.value
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def log_operation(
        operation_name: str | None = None,
        *,
        track_perf: bool = False,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to automatically log operation execution with structured logging.

        Automatically logs operation start, completion, and failures with
        structured context, eliminating manual logging boilerplate.

        Args:
            operation_name: Name for the operation (defaults to function name)
            track_perf: If True, also tracks performance metrics (duration_ms,
                duration_seconds). Default False.

        Returns:
            Decorated function with automatic operation logging

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                op_name: str = (
                    operation_name if operation_name is not None else func.__name__
                )
                logger_carrier: FlextDecorators._LoggerCarrier | None = None
                if args:
                    first_arg_raw = args[0]
                    if isinstance(
                        first_arg_raw, (p.Logger, m.BaseModel, *t.CONTAINER_TYPES)
                    ) or FlextDecorators._has_flext_logger(first_arg_raw):
                        logger_carrier = first_arg_raw
                logger = FlextDecorators._resolve_logger(
                    logger_carrier,
                    func_module=func.__module__,
                )
                correlation_id = FlextDecorators._bind_operation_context(
                    operation=op_name,
                    logger=logger,
                    function_name=func.__name__,
                    ensure_correlation=True,
                )
                start_time = time.perf_counter() if track_perf else 0.0
                try:
                    FlextDecorators._log_start(
                        logger,
                        op_name,
                        func.__name__,
                        func.__module__,
                        correlation_id,
                    )
                    result = func(*args, **kwargs)
                    FlextDecorators._log_completion(
                        logger,
                        op_name,
                        func.__name__,
                        correlation_id,
                        start_time=start_time,
                        track_perf=track_perf,
                    )
                    return result
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as exc:
                    tracked_duration = (
                        time.perf_counter() - start_time if track_perf else 0.0
                    )
                    exc_kw = FlextDecorators._build_exception_kwargs(
                        exc,
                        func.__name__,
                        op_name,
                        correlation_id,
                        tracked_duration=tracked_duration,
                        track_perf=track_perf,
                    )
                    logger.exception(
                        op_name,
                        exception=exc,
                        **exc_kw,
                    )
                    raise
                finally:
                    with suppress(Exception):
                        FlextDecorators._clear_operation_scope(
                            logger=logger,
                            function_name=func.__name__,
                            operation=op_name,
                        )

            return wrapper

        return decorator

    @staticmethod
    def railway(
        error_code: str | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, r[T]]]:
        """Decorator to automatically wrap function in railway pattern.

        Automatically converts exceptions to r failures and
        successful returns to r successes, eliminating manual
        try/except boilerplate.

        Args:
            error_code: Optional error code for failures

        Returns:
            Decorated function that returns r[T]

        """

        def decorator(func: Callable[P, T]) -> Callable[P, r[T]]:

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> r[T]:
                try:
                    result = func(*args, **kwargs)
                    return r[T].ok(result)
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as exc:
                    effective_error_code: str = (
                        str(error_code) if error_code is not None else "OPERATION_ERROR"
                    )
                    return e.fail_operation(
                        func.__name__, exc, error_code=effective_error_code
                    )

            return wrapper

        return decorator

    @staticmethod
    def retry(
        max_attempts: int | None = None,
        delay_seconds: float | None = None,
        backoff_strategy: str | None = None,
        error_code: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to automatically retry failed operations with exponential backoff.

        Uses FlextConstants for default values and
        e for structured error handling, integrating foundation
        modules.

        Args:
            max_attempts: Maximum retry attempts (default:
                c.MAX_RETRY_ATTEMPTS)
            delay_seconds: Initial delay between retries (default:
                FlextConstants.DEFAULT_RETRY_DELAY_SECONDS)
            backoff_strategy: Backoff strategy ('exponential' or 'linear',
                default: FlextConstants.DEFAULT_BACKOFF_STRATEGY)
            error_code: Optional error code for failures

        Returns:
            Decorated function with automatic retry logic

        """
        attempts: int = (
            max_attempts if max_attempts is not None else c.MAX_RETRY_ATTEMPTS
        )
        delay: float = (
            delay_seconds
            if delay_seconds is not None
            else float(c.DEFAULT_RETRY_DELAY_SECONDS)
        )
        strategy: str = (
            backoff_strategy
            if backoff_strategy is not None
            else c.DEFAULT_BACKOFF_STRATEGY
        )

        def decorator(func: Callable[P, R]) -> Callable[P, R]:

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                logger_carrier: FlextDecorators._LoggerCarrier | None = None
                if args:
                    first_arg_raw = args[0]
                    if isinstance(
                        first_arg_raw, (p.Logger, m.BaseModel, *t.CONTAINER_TYPES)
                    ) or FlextDecorators._has_flext_logger(first_arg_raw):
                        logger_carrier = first_arg_raw
                logger = FlextDecorators._resolve_logger(
                    logger_carrier,
                    func_module=func.__module__,
                )
                retry_settings = m.RetryConfiguration.model_validate({
                    "max_retries": attempts,
                    "initial_delay_seconds": delay,
                    "exponential_backoff": strategy == c.DEFAULT_BACKOFF_STRATEGY,
                    "retry_on_exceptions": [],
                    "retry_on_status_codes": [],
                })
                try:
                    retry_args: tuple[t.ValueOrModel, ...] = tuple(
                        str(a.model_dump())
                        if isinstance(a, m.BaseModel)
                        else u.normalize_to_container(a)
                        if isinstance(a, (str, int, float, bool, datetime, Path))
                        else str(a)
                        if a is not None
                        else ""
                        for a in args
                    )
                    retry_kwargs: MutableMapping[str, t.ValueOrModel] = {}
                    for key, value in kwargs.items():
                        if isinstance(value, m.BaseModel):
                            retry_kwargs[str(key)] = str(value.model_dump())
                        elif isinstance(value, (str, int, float, bool, datetime, Path)):
                            retry_kwargs[str(key)] = u.normalize_to_container(value)
                        elif value is not None:
                            retry_kwargs[str(key)] = str(value)
                    retry_result = FlextDecorators._execute_retry_loop(
                        func,
                        retry_args,
                        retry_kwargs,
                        logger,
                        retry_settings=retry_settings,
                    )
                    if isinstance(retry_result, Exception):
                        FlextDecorators._handle_retry_exhaustion(
                            retry_result,
                            func,
                            attempts,
                            error_code,
                            logger,
                        )
                    return retry_result
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as exc:
                    FlextDecorators._handle_retry_exhaustion(
                        exc,
                        func,
                        attempts,
                        error_code,
                        logger,
                    )

            return wrapper

        return decorator

    @staticmethod
    def _bind_operation_context(
        *,
        operation: str,
        logger: p.Logger,
        function_name: str,
        ensure_correlation: bool,
    ) -> str | None:
        """Ensure correlation, bind operation context, and report failures."""
        correlation_id: str | None = None
        if ensure_correlation:
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
        else:
            current_id = FlextContext.Variables.CorrelationId.get()
            if isinstance(current_id, str):
                correlation_id = current_id
        FlextContext.Request.apply_operation_name(operation)
        binding_result = u.bind_context(
            c.ContextScope.OPERATION,
            operation=operation,
        )
        if binding_result.failure:
            logger.warning(
                "operation_context_binding_failed",
                function=function_name,
                operation=operation,
                error=binding_result.error or "",
                error_code=binding_result.error_code or "",
                correlation_id=correlation_id or "",
            )
        return correlation_id

    @staticmethod
    def _clear_operation_scope(
        *,
        logger: p.Logger,
        function_name: str,
        operation: str,
    ) -> None:
        """Clear operation scope and log if cleanup fails."""
        clear_result = u.clear_scope(c.ContextScope.OPERATION)
        if clear_result.failure:
            FlextDecorators._handle_log_result(
                result=clear_result,
                logger=logger,
                fallback_message="operation_context_clear_failed",
                kwargs={
                    "extra": {
                        "function": function_name,
                        c.HandlerType.OPERATION: operation,
                    },
                },
            )

    @staticmethod
    def _execute_retry_loop(
        func: Callable[..., R],
        args: tuple[t.ValueOrModel, ...],
        kwargs: Mapping[str, t.ValueOrModel],
        logger: p.Logger,
        *,
        retry_settings: m.RetryConfiguration | None = None,
    ) -> R | Exception:
        """Execute retry loop and return last exception.

        Uses RetryConfiguration model to reduce parameter count from 8 to 5.

        Args:
            func: Function to execute
            args: Function positional arguments
            kwargs: Function keyword arguments
            logger: Logger instance
            retry_settings: RetryConfiguration instance (Pydantic v2)

        Returns:
            Function result on success

        """
        if retry_settings is None:
            retry_settings = m.RetryConfiguration(
                retry_on_exceptions=[],
                retry_on_status_codes=[],
            )
        attempts = retry_settings.max_retries
        delay = retry_settings.initial_delay_seconds
        strategy = (
            c.DEFAULT_BACKOFF_STRATEGY
            if retry_settings.exponential_backoff
            else c.BackoffStrategy.LINEAR
        )
        last_exception: Exception | None = None
        current_delay = delay
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    logger.info(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=attempts,
                        delay_seconds=current_delay,
                    )
                    time.sleep(current_delay)
                return func(*args, **kwargs)
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as exc:
                last_exception = exc
                logger.warning(
                    "operation_failed_retrying",
                    function=func.__name__,
                    attempt=attempt,
                    max_attempts=attempts,
                    error=str(exc),
                    error_type=exc.__class__.__name__,
                )
                if strategy == c.DEFAULT_BACKOFF_STRATEGY:
                    current_delay *= 2
                elif strategy == c.BackoffStrategy.LINEAR:
                    current_delay += delay
                if attempt == attempts:
                    break
        if last_exception is None:
            msg = c.ERR_RUNTIME_RETRY_LOOP_ENDED_WITHOUT_RESULT
            return RuntimeError(msg)
        return last_exception

    @staticmethod
    def _handle_log_result(
        *,
        result: r[bool] | p.Result[bool],
        logger: p.Logger,
        fallback_message: str,
        kwargs: t.ConfigMap | p.ConfigObject | Mapping[str, t.ValueOrModel],
    ) -> None:
        """Ensure FlextLogger call results are handled for diagnostics."""
        if not result.failure:
            return
        fallback_logger = FlextDecorators._resolve_fallback_logger(logger)
        if fallback_logger is None:
            return
        warning_context = FlextDecorators._flatten_settings_kwargs(kwargs)
        warning_context["log_error"] = result.error or ""
        warning_context["log_error_code"] = result.error_code or ""
        fallback_logger.warning(fallback_message, **warning_context)

    @staticmethod
    def _handle_retry_exhaustion(
        last_exception: Exception,
        func: Callable[..., R],
        attempts: int,
        _error_code: str | None,
        logger: p.Logger,
    ) -> NoReturn:
        """Handle retry exhaustion and raise appropriate exception."""
        logger.error(
            "operation_failed_all_retries_exhausted",
            function=func.__name__,
            attempts=attempts,
            error=str(last_exception),
            error_type=last_exception.__class__.__name__,
        )
        effective_error_code: str = (
            _error_code if _error_code is not None else c.ErrorCode.TIMEOUT_ERROR.value
        )
        timeout_message = f"Operation {func.__name__} failed after {attempts} attempts"
        raise e.TimeoutError(
            timeout_message,
            error_code=effective_error_code,
            operation=func.__name__,
            attempts=attempts,
            original_error=str(last_exception),
        ) from last_exception

    @staticmethod
    def _flatten_settings_kwargs(
        kwargs: t.ConfigMap | p.ConfigObject | Mapping[str, t.ValueOrModel],
    ) -> MutableMapping[str, t.Container]:
        """Flatten one mapping into a flat warning-context dict."""
        context: MutableMapping[str, t.Container] = {}
        for key, value in kwargs.items():
            if key == "extra" and u.dict_like(value):
                extra_items: Mapping[str, t.ValueOrModel]
                if isinstance(value, t.ConfigMap):
                    extra_items = value.root
                else:
                    extra_items = {str(k): v for k, v in value.items()}
                for extra_key, extra_value in extra_items.items():
                    if isinstance(extra_value, (str, int, float, bool, datetime, Path)):
                        context[f"extra_{extra_key}"] = extra_value
                    else:
                        context[f"extra_{extra_key}"] = str(extra_value)
                continue
            if isinstance(value, t.CONTAINER_TYPES):
                context[str(key)] = value
                continue
            context[str(key)] = str(value)
        return context

    @staticmethod
    def _has_flext_logger(
        value: t.OpaqueValue,
    ) -> TypeIs[p.HasLogger]:
        if not hasattr(value, "logger"):
            return False
        logger_value = getattr(value, "logger", None)
        return (
            logger_value is not None
            and hasattr(logger_value, "bind")
            and hasattr(logger_value, "debug")
            and hasattr(logger_value, "info")
        )

    @staticmethod
    def _log_start(
        logger: p.Logger,
        op_name: str,
        func_name: str,
        func_module: str,
        correlation_id: str | None,
    ) -> None:
        """Log operation start with optional correlation ID."""
        if correlation_id is not None:
            logger.debug(
                "%s_started",
                op_name,
                function=func_name,
                func_module=func_module,
                correlation_id=correlation_id,
            )
        else:
            logger.debug(
                "%s_started",
                op_name,
                function=func_name,
                func_module=func_module,
            )

    @staticmethod
    def _log_completion(
        logger: p.Logger,
        op_name: str,
        func_name: str,
        correlation_id: str | None,
        *,
        start_time: float,
        track_perf: bool,
    ) -> None:
        """Log operation completion with optional perf metrics."""
        extra: t.MutableConfigurationMapping = {
            "function": func_name,
            "success": True,
        }
        if correlation_id is not None:
            extra[c.ContextKey.CORRELATION_ID] = correlation_id
        if track_perf:
            duration = time.perf_counter() - start_time
            extra["duration_ms"] = duration * c.DEFAULT_SIZE
            extra[c.MetadataKey.DURATION_SECONDS] = duration
        logger.debug("%s_completed", op_name, **extra)

    @staticmethod
    def _build_exception_kwargs(
        exc: Exception,
        func_name: str,
        op_name: str,
        correlation_id: str | None,
        *,
        tracked_duration: float,
        track_perf: bool,
    ) -> t.MutableConfigurationMapping:
        """Build kwargs dict for logger.exception call (must be called inside except)."""
        kw: t.MutableConfigurationMapping = {
            "function": func_name,
            "success": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
            "operation": op_name,
        }
        if correlation_id is not None:
            kw[c.ContextKey.CORRELATION_ID] = correlation_id
        if track_perf:
            kw["duration_ms"] = tracked_duration * c.DEFAULT_SIZE
            kw[c.MetadataKey.DURATION_SECONDS] = tracked_duration
        return kw

    @staticmethod
    def _raise_timeout(
        func_name: str,
        max_duration: float,
        duration: float,
        *,
        error_code: str,
        original_error: Exception | None = None,
    ) -> NoReturn:
        """Raise e.TimeoutError with consistent formatting."""
        suffix = (
            f" and raised {original_error.__class__.__name__}"
            if original_error is not None
            else ""
        )
        msg = (
            f"Operation {func_name} exceeded timeout of "
            f"{max_duration}s (took {duration:.2f}s){suffix}"
        )
        original_str = str(original_error) if original_error is not None else ""
        raise e.TimeoutError(
            msg,
            error_code=error_code,
            timeout_seconds=max_duration,
            operation=func_name,
            duration_seconds=duration,
            original_error=original_str,
        ) from original_error

    @staticmethod
    def _resolve_fallback_logger(logger: p.Logger) -> p.Logger | None:
        """Extract fallback logger from logger wrapper."""
        if FlextDecorators._has_flext_logger(logger):
            return logger.logger
        candidate: p.Logger | None = getattr(logger, "logger", None)
        if isinstance(candidate, p.Logger):
            return candidate
        return None

    @overload
    @staticmethod
    def combined(
        *,
        inject_deps: t.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        use_railway: Literal[False] = False,
        error_code: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    @overload
    @staticmethod
    def combined(
        *,
        inject_deps: t.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        use_railway: Literal[True],
        error_code: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, r[R]]]: ...

    @staticmethod
    def combined(
        *,
        inject_deps: t.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        use_railway: bool = False,
        error_code: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R] | Callable[P, r[R]]]:
        """Combined decorator applying multiple automation patterns at once.

        Combines @inject, @log_operation (with optional track_perf), and optionally
        @railway into a single decorator for maximum code reduction.

        Args:
            inject_deps: Dependencies to inject (name -> type mapping)
            operation_name: Name for logging (defaults to function name)
            track_perf: Whether to track performance (default: True)
            use_railway: Whether to apply railway pattern (default: False)
            error_code: Error code for railway pattern failures

        Returns:
            Decorated function with all requested automations.
            When use_railway=True, returns Callable[..., r[R]].
            When use_railway=False, returns Callable[..., R].

        """
        if use_railway:

            def railway_decorator(func: Callable[P, R]) -> Callable[P, r[R]]:
                result = FlextDecorators.railway(error_code=error_code)(func)
                if inject_deps:
                    result = FlextDecorators.inject(**inject_deps)(result)
                return FlextDecorators.log_operation(
                    operation_name=operation_name,
                    track_perf=track_perf,
                )(result)

            return railway_decorator

        def standard_decorator(func: Callable[P, R]) -> Callable[P, R]:
            result = func
            if inject_deps:
                result = FlextDecorators.inject(**inject_deps)(result)
            return FlextDecorators.log_operation(
                operation_name=operation_name,
                track_perf=track_perf,
            )(result)

        return standard_decorator

    @staticmethod
    def factory(
        name: str,
        *,
        singleton: bool = False,
        lazy: bool = True,
    ) -> Callable[[t.HandlerCallable], t.HandlerCallable]:
        """Decorator to mark functions as factories for DI container.

        Stores factory configuration as metadata on the decorated function,
        enabling auto-discovery by FlextContainer and factory registries.

        Args:
            name: Name to register the factory under in the container
            singleton: Whether factory creates singleton instances. Default: False
            lazy: Whether to defer factory invocation until first use. Default: True

        Returns:
            Decorator function for marking factory functions

        """

        def decorator(func: t.HandlerCallable) -> t.HandlerCallable:
            """Apply factory configuration metadata to function."""
            settings = m.FactoryDecoratorConfig(
                name=name,
                singleton=singleton,
                lazy=lazy,
            )
            setattr(func, c.FACTORY_ATTR, settings)
            return func

        return decorator

    @staticmethod
    def timeout(
        timeout_seconds: float | None = None,
        error_code: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to enforce operation timeout.

        Uses c.DEFAULT_TIMEOUT_SECONDS for default
        timeout and e.TimeoutError for structured error
        handling.

        Args:
            timeout_seconds: Timeout in seconds (default:
                c.DEFAULT_TIMEOUT_SECONDS)
            error_code: Optional error code for timeout

        Returns:
            Decorated function with timeout enforcement

        """
        max_duration = (
            timeout_seconds
            if timeout_seconds is not None
            else c.DEFAULT_TIMEOUT_SECONDS
        )

        def decorator(func: Callable[P, R]) -> Callable[P, R]:

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        FlextDecorators._raise_timeout(
                            func.__name__,
                            max_duration,
                            duration,
                            error_code=error_code or "OPERATION_TIMEOUT",
                        )
                    return result
                except e.TimeoutError:
                    raise
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as exc:
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        FlextDecorators._raise_timeout(
                            func.__name__,
                            max_duration,
                            duration,
                            error_code=error_code or c.ErrorCode.TIMEOUT_ERROR.value,
                            original_error=exc,
                        )
                    raise

            return wrapper

        return decorator

    @staticmethod
    def track_operation(
        operation_name: str | None = None,
        *,
        track_correlation: bool = True,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to track operation execution with u.Integration.

        Combines correlation ID management and structured logging using
        u.Integration pattern (Layer 0.5). Performance tracking
        happens automatically via u.Integration.
        No circular imports - uses structlog directly.

        Args:
            operation_name: Name for the operation (defaults to function name)
            track_correlation: Ensure correlation ID exists (default: True)

        Returns:
            Decorated function with comprehensive operation tracking

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                op_name: str = (
                    operation_name if operation_name is not None else func.__name__
                )
                logger_carrier: FlextDecorators._LoggerCarrier | None = None
                if args:
                    first_arg_raw = args[0]
                    if isinstance(
                        first_arg_raw, (p.Logger, m.BaseModel, *t.CONTAINER_TYPES)
                    ) or FlextDecorators._has_flext_logger(first_arg_raw):
                        logger_carrier = first_arg_raw
                logger = FlextDecorators._resolve_logger(
                    logger_carrier,
                    func_module=func.__module__,
                )
                correlation_id = FlextDecorators._bind_operation_context(
                    operation=op_name,
                    logger=logger,
                    function_name=func.__name__,
                    ensure_correlation=track_correlation,
                )
                if track_correlation and correlation_id is None:
                    logger.warning(
                        "correlation_id_missing",
                        function=func.__name__,
                        operation=op_name,
                    )
                try:
                    return func(*args, **kwargs)
                finally:
                    with suppress(Exception):
                        FlextDecorators._clear_operation_scope(
                            logger=logger,
                            function_name=func.__name__,
                            operation=op_name,
                        )

            return wrapper

        return decorator

    @staticmethod
    def with_context(
        **context_vars: t.Primitives | None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to manage context lifecycle for an operation.

        Automatically binds context variables for the operation duration and
        unbinds them after completion. Enables automatic context propagation
        without manual bind/unbind calls.

        Args:
            **context_vars: Context variables to bind for operation duration

        Returns:
            Decorated function with automatic context management

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                logger_carrier: FlextDecorators._LoggerCarrier | None = None
                if args:
                    first_arg_raw = args[0]
                    if isinstance(
                        first_arg_raw, (p.Logger, m.BaseModel, *t.CONTAINER_TYPES)
                    ) or FlextDecorators._has_flext_logger(first_arg_raw):
                        logger_carrier = first_arg_raw
                logger = FlextDecorators._resolve_logger(
                    logger_carrier,
                    func_module=func.__module__,
                )
                try:
                    if context_vars:
                        filtered_vars: t.FlatContainerMapping = {
                            k: v
                            for k, v in context_vars.items()
                            if v is not None and u.container(v)
                        }
                        bind_result = u.bind_global_context(**filtered_vars)
                        if bind_result.failure:
                            logger.warning(
                                "global_context_binding_failed",
                                function=func.__name__,
                                error=bind_result.error or "",
                                error_code=bind_result.error_code or "",
                                bound_keys=", ".join(context_vars.keys()),
                            )
                    return func(*args, **kwargs)
                finally:
                    if context_vars:
                        unbind_result = u.unbind_global_context(
                            *tuple(context_vars.keys()),
                        )
                        if unbind_result.failure:
                            logger.warning(
                                "global_context_unbind_failed",
                                function=func.__name__,
                                error=unbind_result.error or "",
                                error_code=unbind_result.error_code or "",
                                bound_keys=", ".join(context_vars.keys()),
                            )

            return wrapper

        return decorator

    @staticmethod
    def with_correlation() -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to ensure correlation ID exists for operation tracking.

        Automatically ensures a correlation ID is present in the context,
        generating one if needed. Essential for distributed tracing and
        request correlation across services.

        Returns:
            Decorated function with correlation ID management

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                _ = FlextContext.Utilities.ensure_correlation_id()
                return func(*args, **kwargs)

            return wrapper

        return decorator


d = FlextDecorators
__all__ = ["FlextDecorators", "d"]
