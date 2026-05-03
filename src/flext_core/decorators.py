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
from collections.abc import (
    Callable,
)
from contextlib import suppress
from functools import wraps
from typing import Literal, TypeIs, overload

from flext_core import (
    FlextConstants as c,
    FlextContainer,
    FlextContext,
    FlextExceptions as e,
    FlextModels as m,
    FlextProtocols as p,
    FlextResult as r,
    FlextTypes as t,
    FlextUtilities as u,
)


class FlextDecorators:
    """Automation decorators for infrastructure concerns.

    Provides decorators that automatically handle common infrastructure
    concerns to reduce boilerplate code in services, handlers, and other
    components. All decorators are designed to integrate seamlessly with
    `r`, `FlextContext`, the public logging DSL in `u`, and `FlextContainer`.
    """

    type _LoggerCarrier = p.HasLogger | p.Logger | t.JsonPayload | m.BaseModel
    _CAUGHT_EXCEPTIONS: tuple[type[Exception], ...] = (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
    )

    @staticmethod
    def _is_logger_carrier(
        value: p.AttributeProbe | None,
    ) -> TypeIs[FlextDecorators._LoggerCarrier]:
        return isinstance(value, (p.Logger, m.BaseModel, *t.CONTAINER_TYPES)) or (
            FlextDecorators._has_flext_logger(value)
        )

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
            return first_arg.logger
        module_name = (
            func_module
            if isinstance(func_module, str)
            else (func.__module__ if callable(func) else __name__)
        )
        return u.fetch_logger(module_name)

    @staticmethod
    def deprecated[**PCallback, TResult](
        reason: str,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Mark callable as deprecated and emit ``DeprecationWarning`` on use."""

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:
            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                warnings.warn(
                    f"{func.__name__} is deprecated: {reason}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def inject[**PCallback, TResult](
        **dependencies: str,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
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

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:

            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                container = FlextContainer.shared()
                for name, service_key in dependencies.items():
                    if name not in kwargs:
                        result = container.resolve(service_key)
                        if result.success:
                            kwargs[name] = result.value
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def log_operation[**PCallback, TResult](
        operation_name: str | None = None,
        *,
        track_perf: bool = False,
        ensure_correlation: bool = True,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Decorator to automatically log operation execution with structured logging."""

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:

            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                op_name: str = (
                    operation_name if operation_name is not None else func.__name__
                )
                logger_carrier: FlextDecorators._LoggerCarrier | None = None
                if args and FlextDecorators._is_logger_carrier(args[0]):
                    logger_carrier = args[0]
                logger = FlextDecorators._resolve_logger(
                    logger_carrier,
                    func_module=func.__module__,
                )
                correlation_id: str | None = None
                if ensure_correlation:
                    correlation_id = FlextContext.ensure_correlation_id()
                else:
                    current_id = u.CORRELATION_ID.get()
                    if isinstance(current_id, str):
                        correlation_id = current_id
                FlextContext.apply_operation_name(op_name)
                binding_result = u.bind_context(
                    c.ContextScope.OPERATION,
                    operation=op_name,
                )
                if binding_result.failure:
                    logger.warning(
                        "operation_context_binding_failed",
                        function=func.__name__,
                        operation=op_name,
                        error=binding_result.error or "",
                        error_code=binding_result.error_code or "",
                        correlation_id=correlation_id or "",
                    )
                start_time = time.perf_counter() if track_perf else 0.0
                try:
                    extra_start = {
                        "function": func.__name__,
                        "func_module": func.__module__,
                    }
                    if correlation_id:
                        extra_start["correlation_id"] = correlation_id
                    logger.debug("%s_started", op_name, **extra_start)
                    result = func(*args, **kwargs)
                    extra_done: t.MutableJsonMapping = {
                        "function": func.__name__,
                        "success": True,
                    }
                    if correlation_id is not None:
                        extra_done[c.ContextKey.CORRELATION_ID] = correlation_id
                    if track_perf:
                        duration = time.perf_counter() - start_time
                        extra_done["duration_ms"] = duration * c.DEFAULT_SIZE
                        extra_done[c.MetadataKey.DURATION_SECONDS] = duration
                    logger.debug("%s_completed", op_name, **extra_done)
                    return result
                except FlextDecorators._CAUGHT_EXCEPTIONS as exc:
                    tracked_duration = (
                        time.perf_counter() - start_time if track_perf else 0.0
                    )
                    exc_kw: t.MutableJsonMapping = {
                        "function": func.__name__,
                        "success": False,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "operation": op_name,
                    }
                    if correlation_id is not None:
                        exc_kw[c.ContextKey.CORRELATION_ID] = correlation_id
                    if track_perf:
                        exc_kw["duration_ms"] = tracked_duration * c.DEFAULT_SIZE
                        exc_kw[c.MetadataKey.DURATION_SECONDS] = tracked_duration
                    logger.exception(
                        op_name,
                        exception=exc,
                        **exc_kw,
                    )
                    raise
                finally:
                    with suppress(Exception):
                        clear_result = u.clear_scope(c.ContextScope.OPERATION)
                        if clear_result.failure:
                            logger.warning(
                                "operation_context_clear_failed",
                                function=func.__name__,
                                operation=op_name,
                                error=clear_result.error or "",
                                error_code=clear_result.error_code or "",
                            )

            return wrapper

        return decorator

    @staticmethod
    def railway[**PCallback, TValue](
        error_code: str | None = None,
    ) -> Callable[[Callable[PCallback, TValue]], Callable[PCallback, p.Result[TValue]]]:
        """Decorator to automatically wrap function in railway pattern.

        Automatically converts exceptions to r failures and
        successful returns to r successes, eliminating manual
        try/except boilerplate.

        Args:
            error_code: Optional error code for failures

        Returns:
            Decorated function that returns r[TValue]

        """

        def decorator(
            func: Callable[PCallback, TValue],
        ) -> Callable[PCallback, p.Result[TValue]]:

            @wraps(func)
            def wrapper(
                *args: PCallback.args,
                **kwargs: PCallback.kwargs,
            ) -> p.Result[TValue]:
                try:
                    result = func(*args, **kwargs)
                    return r[TValue].ok(result)
                except FlextDecorators._CAUGHT_EXCEPTIONS as exc:
                    effective_error_code: str = (
                        error_code if error_code is not None else "OPERATION_ERROR"
                    )
                    error_msg = f"{func.__name__} failed: {type(exc).__name__}: {exc}"
                    return r[TValue].fail(
                        error_msg,
                        error_code=effective_error_code,
                    )

            return wrapper

        return decorator

    @staticmethod
    def retry[**PCallback, TResult](
        max_attempts: int | None = None,
        delay_seconds: float | None = None,
        backoff_strategy: str | None = None,
        error_code: str | None = None,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
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

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:

            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                logger_carrier: FlextDecorators._LoggerCarrier | None = None
                if args:
                    first_arg_raw = args[0]
                    if FlextDecorators._is_logger_carrier(first_arg_raw):
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
                retry_result = FlextDecorators._execute_retry_loop(
                    lambda: func(*args, **kwargs),
                    func.__name__,
                    logger,
                    retry_settings=retry_settings,
                )
                if isinstance(retry_result, Exception):
                    logger.error(
                        "operation_failed_all_retries_exhausted",
                        function=func.__name__,
                        attempts=attempts,
                        error=str(retry_result),
                        error_type=retry_result.__class__.__name__,
                    )
                    effective_error_code: str = (
                        error_code
                        if error_code is not None
                        else c.ErrorCode.TIMEOUT_ERROR.value
                    )
                    timeout_message = (
                        f"Operation {func.__name__} failed after {attempts} attempts"
                    )
                    raise e.TimeoutError(
                        timeout_message,
                        error_code=effective_error_code,
                        operation=func.__name__,
                        attempts=attempts,
                        original_error=str(retry_result),
                    ) from retry_result
                return retry_result

            return wrapper

        return decorator

    @staticmethod
    def _execute_retry_loop[TResult](
        call: Callable[[], TResult],
        func_name: str,
        logger: p.Logger,
        *,
        retry_settings: m.RetryConfiguration,
    ) -> TResult | Exception:
        """Execute retry loop with closure; return last exception on exhaustion."""
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
                        function=func_name,
                        attempt=attempt,
                        max_attempts=attempts,
                        delay_seconds=current_delay,
                    )
                    time.sleep(current_delay)
                return call()
            except FlextDecorators._CAUGHT_EXCEPTIONS as exc:
                last_exception = exc
                logger.warning(
                    "operation_failed_retrying",
                    function=func_name,
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
    def _has_flext_logger(
        value: p.AttributeProbe | None,
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

    @overload
    @staticmethod
    def combined[**PCallback, TResult](
        *,
        inject_deps: t.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        railway_enabled: Literal[False] = False,
        railway_error_code: str | None = None,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]: ...

    @overload
    @staticmethod
    def combined[**PCallback, TResult](
        *,
        inject_deps: t.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        railway_enabled: Literal[True],
        railway_error_code: str | None = None,
    ) -> Callable[
        [Callable[PCallback, TResult]], Callable[PCallback, p.Result[TResult]]
    ]: ...

    @staticmethod
    def combined[**PCallback, TResult](
        *,
        inject_deps: t.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        railway_enabled: bool = False,
        railway_error_code: str | None = None,
    ) -> Callable[
        [Callable[PCallback, TResult]],
        Callable[PCallback, TResult] | Callable[PCallback, p.Result[TResult]],
    ]:
        """Combined decorator applying multiple automation patterns at once.

        Combines @inject, @log_operation (with optional track_perf), and optionally
        @railway into a single decorator for maximum code reduction.

        Args:
            inject_deps: Dependencies to inject (name -> type mapping)
            operation_name: Name for logging (defaults to function name)
            track_perf: Whether to track performance (default: True)
            railway_enabled: Whether combined() applies railway wrapping.
            railway_error_code: Error code passed to railway() when enabled.

        Returns:
            Decorated function with all requested automations.
            When railway is enabled, returns Callable[..., r[TResult]].
            Otherwise, returns Callable[..., TResult].

        """
        railway = m.CombinedRailwayOptions.model_validate(
            {
                "enabled": railway_enabled,
                "error_code": railway_error_code,
            },
        )
        if railway.enabled:

            def railway_decorator(
                func: Callable[PCallback, TResult],
            ) -> Callable[PCallback, p.Result[TResult]]:
                result = FlextDecorators.railway(error_code=railway.error_code)(func)
                if inject_deps:
                    result = FlextDecorators.inject(**inject_deps)(result)
                operation_logger: Callable[
                    [Callable[PCallback, p.Result[TResult]]],
                    Callable[PCallback, p.Result[TResult]],
                ] = FlextDecorators.log_operation(
                    operation_name=operation_name,
                    track_perf=track_perf,
                )
                return operation_logger(result)

            return railway_decorator

        def standard_decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:
            result = func
            if inject_deps:
                result = FlextDecorators.inject(**inject_deps)(result)
            operation_logger: Callable[
                [Callable[PCallback, TResult]],
                Callable[PCallback, TResult],
            ] = FlextDecorators.log_operation(
                operation_name=operation_name,
                track_perf=track_perf,
            )
            return operation_logger(result)

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
    def timeout[**PCallback, TResult](
        timeout_seconds: float | None = None,
        error_code: str | None = None,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
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

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:

            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        msg = f"Operation {func.__name__} exceeded timeout of {max_duration}s (took {duration:.2f}s)"
                        raise e.TimeoutError(
                            msg,
                            error_code=error_code or "OPERATION_TIMEOUT",
                            timeout_seconds=max_duration,
                            operation=func.__name__,
                            duration_seconds=duration,
                            original_error="",
                        )
                    return result
                except e.TimeoutError:
                    raise
                except FlextDecorators._CAUGHT_EXCEPTIONS as exc:
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        msg = f"Operation {func.__name__} exceeded timeout of {max_duration}s (took {duration:.2f}s) and raised {exc.__class__.__name__}"
                        raise e.TimeoutError(
                            msg,
                            error_code=error_code or c.ErrorCode.TIMEOUT_ERROR.value,
                            timeout_seconds=max_duration,
                            operation=func.__name__,
                            duration_seconds=duration,
                            original_error=str(exc),
                        ) from exc
                    raise

            return wrapper

        return decorator

    @staticmethod
    def with_correlation[**PCallback, TResult]() -> Callable[
        [Callable[PCallback, TResult]], Callable[PCallback, TResult]
    ]:
        """Decorator to ensure correlation ID exists for operation tracking.

        Automatically ensures a correlation ID is present in the context,
        generating one if needed. Essential for distributed tracing and
        request correlation across services.

        Returns:
            Decorated function with correlation ID management

        """

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:

            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                _ = FlextContext.ensure_correlation_id()
                return func(*args, **kwargs)

            return wrapper

        return decorator


d = FlextDecorators
__all__: list[str] = ["FlextDecorators", "d"]
