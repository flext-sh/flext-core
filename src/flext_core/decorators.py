"""Decorator utilities that automate dispatcher-friendly cross-cutting concerns.

The decorators in this module hide boilerplate for dependency injection,
logging, correlation tracking, and railway conversions so handlers and services
stay focused on domain logic while fitting naturally into the dispatcher
pipeline.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from functools import wraps
from typing import cast

from flext_core._models.config import FlextModelsConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes, P, R, T


class FlextDecorators:
    """Decorator suite for DI, logging, correlation, and result safety.

    Each decorator wraps callables with dispatcher-friendly behavior: dependency
    injection through ``FlextContainer``, structured logging that respects
    ``FlextContext`` correlation data, performance tracking, and conversion of
    raised errors into ``FlextResult`` when appropriate. Composition order
    mirrors the dispatcher middleware chain to keep telemetry consistent.
    """

    @staticmethod
    def inject(
        **dependencies: str,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Inject dependencies from FlextContainer into keyword-only params."""

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Get container from self if available, otherwise use global
                container = FlextContainer()

                # Inject dependencies that aren't already provided
                for name, service_key in dependencies.items():
                    if name not in kwargs:
                        # Get from container using the service key
                        result: FlextProtocols.ResultProtocol[
                            FlextTypes.GeneralValueType
                        ] = container.get(service_key)
                        if result.is_success:
                            kwargs[name] = result.unwrap()
                        else:
                            # If resolution fails, let the function handle it
                            # or fail with missing parameter
                            pass

                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def log_operation(
        operation_name: str | None = None,
        *,
        track_perf: bool = False,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Log operation start/completion/failure with structured context."""

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                op_name = operation_name or func.__name__
                # Convert P.args to tuple for logger resolution (type narrowing happens in _resolve_logger)
                args_tuple = tuple(args)
                logger = FlextDecorators._resolve_logger(args_tuple, func)
                # with_result() returns FlextLogger.ResultAdapter for result-aware logging
                # FlextLogger has with_result() method that returns adapter
                result_logger = logger.with_result()
                correlation_id = FlextDecorators._bind_operation_context(
                    operation=op_name,
                    logger=logger,
                    function_name=func.__name__,
                    ensure_correlation=True,
                )
                start_time = time.perf_counter() if track_perf else 0.0
                try:
                    # Log operation start
                    start_extra_config = FlextModelsConfig.OperationExtraConfig(
                        func_name=func.__name__,
                        func_module=func.__module__,
                        correlation_id=correlation_id,
                    )
                    start_extra = FlextDecorators._build_operation_extra(
                        start_extra_config
                    )
                    FlextDecorators._handle_log_result(
                        result=result_logger.debug(
                            "%s_started",
                            op_name,
                            extra=start_extra,
                        ),
                        logger=logger,
                        fallback_message="operation_log_emit_failed",
                        kwargs={"extra": {**start_extra, "log_state": "start"}},
                    )
                    result = func(*args, **kwargs)
                    # Log operation completion
                    complete_extra_config = FlextModelsConfig.OperationExtraConfig(
                        func_name=func.__name__,
                        func_module=func.__module__,
                        correlation_id=correlation_id,
                        success=True,
                        start_time=start_time,
                        track_perf=track_perf,
                    )
                    complete_extra = FlextDecorators._build_operation_extra(
                        complete_extra_config,
                    )
                    FlextDecorators._handle_log_result(
                        result=result_logger.debug(
                            "%s_completed",
                            op_name,
                            extra=complete_extra,
                        ),
                        logger=logger,
                        fallback_message="operation_log_emit_failed",
                        kwargs={"extra": {**complete_extra, "log_state": "completed"}},
                    )
                    return result
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    failure_config = FlextModelsConfig.LogOperationFailureConfig(
                        op_name=op_name,
                        func_name=func.__name__,
                        func_module=func.__module__,
                        correlation_id=correlation_id,
                        exc=e,
                        start_time=start_time,
                        track_perf=track_perf,
                    )
                    FlextDecorators._log_operation_failure(logger, failure_config)
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
    def track_performance(
        operation_name: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Track execution time and log performance metrics."""

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Fast fail: explicit default value instead of 'or' fallback
                op_name: str = (
                    operation_name if operation_name is not None else func.__name__
                )

                # Get logger from self if available, otherwise create one
                # Convert P.args to tuple for logger resolution (type narrowing happens in _resolve_logger)
                args_tuple = tuple(args)
                logger = FlextDecorators._resolve_logger(args_tuple, func)

                start_time = time.perf_counter()

                correlation_id = FlextDecorators._bind_operation_context(
                    operation=op_name,
                    logger=logger,
                    function_name=func.__name__,
                    ensure_correlation=True,
                )

                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start_time

                    success_extra: dict[str, FlextTypes.GeneralValueType] = {
                        "operation": op_name,
                        "duration_ms": duration * 1000,
                        "duration_seconds": duration,
                        "success": True,
                    }
                    if correlation_id is not None:
                        success_extra["correlation_id"] = correlation_id
                    logger.info(
                        "operation_completed",
                        extra=success_extra,
                    )
                    return result
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    duration = time.perf_counter() - start_time

                    failure_extra: dict[str, FlextTypes.GeneralValueType] = {
                        "operation": op_name,
                        "duration_ms": duration * 1000,
                        "duration_seconds": duration,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    if correlation_id is not None:
                        failure_extra["correlation_id"] = correlation_id
                    logger.exception(
                        "operation_failed",
                        extra=failure_extra,
                    )
                    raise
                finally:
                    # CRITICAL: Clear operation context (defensive cleanup)
                    # Use suppress to ensure cleanup never fails
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
    ) -> Callable[[Callable[P, T]], Callable[P, FlextResult[T]]]:
        """Convert exceptions to FlextResult.fail(), returns to FlextResult.ok()."""

        def decorator(
            func: Callable[P, T],
        ) -> Callable[P, FlextResult[T]]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> FlextResult[T]:
                try:
                    result = func(*args, **kwargs)

                    # If already a FlextResult, return as-is
                    if isinstance(result, FlextResult):
                        # Return as FlextResult - no need to recreate
                        return result

                    # Wrap successful result using FlextResult directly
                    return FlextResult.ok(result)

                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    # Convert exception to failure using FlextResult directly
                    effective_error_code: str = (
                        error_code if isinstance(error_code, str) else "OPERATION_ERROR"
                    )
                    return FlextResult.fail(
                        f"{e!s} ({effective_error_code})",
                        error_code=effective_error_code,
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
        """Retry failed operations with configurable backoff strategy."""
        # Fast fail: explicit default values instead of 'or' fallback
        attempts: int = (
            max_attempts
            if max_attempts is not None
            else FlextConstants.Reliability.DEFAULT_MAX_RETRIES
        )
        delay: float = (
            delay_seconds
            if delay_seconds is not None
            else float(FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS)
        )
        strategy: str = (
            backoff_strategy
            if backoff_strategy is not None
            else FlextConstants.Reliability.DEFAULT_BACKOFF_STRATEGY
        )

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                logger = FlextDecorators._resolve_logger(args, func)
                # Convert P.args and P.kwargs to types expected by _execute_retry_loop
                # Convert args and kwargs to GeneralValueType for RetryLoopConfig
                args_tuple = cast(
                    "tuple[FlextTypes.GeneralValueType, ...]",
                    tuple(args),
                )
                kwargs_dict = cast(
                    "Mapping[str, FlextTypes.GeneralValueType]",
                    kwargs or {},
                )
                retry_loop_config = FlextModelsConfig.RetryLoopConfig(
                    func=cast("Callable[..., R]", func),
                    args=args_tuple,
                    kwargs=kwargs_dict,
                    attempts=attempts,
                    delay=delay,
                    strategy=strategy,
                )
                result_or_exception = FlextDecorators._execute_retry_loop(
                    logger,
                    retry_loop_config,
                )
                # Check if we got a successful result or an exception
                # result_or_exception is R | Exception, so if it's Exception, handle it
                # Type narrowing: isinstance check narrows type to Exception
                # Note: This isinstance check is necessary for type narrowing even though
                # the type checker reports it as unnecessary - it distinguishes R from Exception
                # This is a false positive: the check is needed for runtime type narrowing
                if isinstance(result_or_exception, Exception):  # pyright: ignore[reportUnnecessaryIsInstance]
                    # All retries failed - handle exhaustion and raise
                    FlextDecorators._handle_retry_exhaustion(
                        result_or_exception,
                        func,
                        attempts,
                        error_code,
                        logger,
                    )
                    raise result_or_exception
                # Success - return the result (type narrowed to R)
                return result_or_exception

            return wrapper

        return decorator

    @staticmethod
    def _resolve_logger(
        args: tuple[object, ...],
        func: Callable[..., object],
    ) -> FlextLogger:
        """Resolve logger from first argument or create module logger."""
        first_arg = args[0] if args else None
        potential_logger: FlextLogger | None = None
        if first_arg is not None:
            attr = getattr(first_arg, "logger", None)
            if isinstance(attr, FlextLogger):
                potential_logger = attr
        if potential_logger is not None:
            return potential_logger
        return FlextLogger(func.__module__)

    @staticmethod
    def _build_operation_extra(
        config: FlextModelsConfig.OperationExtraConfig,
    ) -> dict[str, FlextTypes.GeneralValueType]:
        """Build extra dict for operation logging."""
        extra: dict[str, FlextTypes.GeneralValueType] = {
            "function": config.func_name,
            "func_module": config.func_module,
        }
        if config.correlation_id is not None:
            extra["correlation_id"] = config.correlation_id
        if config.success is not None:
            extra["success"] = config.success
        if config.error is not None:
            extra["error"] = config.error
            extra["error_type"] = config.error_type or "Unknown"
        if config.track_perf and config.start_time > 0:
            duration = time.perf_counter() - config.start_time
            extra["duration_ms"] = duration * 1000
            extra["duration_seconds"] = duration
        return extra

    @staticmethod
    def _log_operation_failure(
        logger: FlextLogger,
        config: FlextModelsConfig.LogOperationFailureConfig,
    ) -> None:
        """Log operation failure with exception details."""
        failure_extra_config = FlextModelsConfig.OperationExtraConfig(
            func_name=config.func_name,
            func_module=config.func_module,
            correlation_id=config.correlation_id,
            success=False,
            error=str(config.exc),
            error_type=type(config.exc).__name__,
            start_time=config.start_time,
            track_perf=config.track_perf,
        )
        failure_extra = FlextDecorators._build_operation_extra(failure_extra_config)
        failure_extra["operation"] = config.op_name
        # Build bind dict for logger
        bind_dict: dict[str, FlextTypes.GeneralValueType] = {
            "success": "False",
            "error": str(config.exc),
            "error_type": type(config.exc).__name__,
            "operation": config.op_name,
        }
        if config.correlation_id is not None:
            bind_dict["correlation_id"] = config.correlation_id
        if config.track_perf and config.start_time > 0:
            duration = time.perf_counter() - config.start_time
            bind_dict["duration_ms"] = duration * 1000
            bind_dict["duration_seconds"] = duration
        # Log failure with exception - using return_result=True to get FlextResult
        failure_result = logger.bind(**bind_dict).exception(
            f"{config.op_name}_failed",
            exception=config.exc,
            return_result=True,
        )
        FlextDecorators._handle_log_result(
            result=failure_result,
            logger=logger,
            fallback_message="operation_log_emit_failed",
            kwargs={
                "extra": {
                    **failure_extra,
                    "log_state": "failed",
                    "exception_repr": repr(config.exc),
                },
            },
        )

    @staticmethod
    def _execute_retry_loop[R](
        logger: FlextLogger,
        config: FlextModelsConfig.RetryLoopConfig,
    ) -> R | Exception:
        """Execute retry loop, returning result on success or Exception on failure."""
        # Extract config values (retry_config takes priority over individual params)
        if config.retry_config is not None:
            attempts = config.retry_config.max_attempts
            delay = config.retry_config.initial_delay_seconds
            # Map exponential_backoff bool to strategy string
            exponential = config.retry_config.exponential_backoff
            strategy = "exponential" if exponential else "linear"
        else:
            attempts = config.attempts
            delay = config.delay
            strategy = config.strategy

        last_exception: Exception | None = None
        current_delay = delay

        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    logger.info(
                        "retry_attempt",
                        extra={
                            "function": config.func.__name__,
                            "attempt": attempt,
                            "max_attempts": attempts,
                            "delay_seconds": current_delay,
                        },
                    )
                    time.sleep(current_delay)

                # Get args and kwargs from config
                args_tuple = config.args
                kwargs_mapping = config.kwargs

                # Convert kwargs to dict if it's a Mapping, otherwise use empty dict
                if FlextRuntime.is_dict_like(kwargs_mapping):
                    kwargs_dict = (
                        kwargs_mapping
                        if isinstance(kwargs_mapping, dict)
                        else dict(kwargs_mapping.items())
                    )
                    result = config.func(*args_tuple, **kwargs_dict)
                    # Cast to R | Exception to match return type
                    return cast("R | Exception", result)
                result = config.func(*args_tuple)
                # Cast to R | Exception to match return type
                return cast("R | Exception", result)

            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as e:
                last_exception = e

                logger.warning(
                    "operation_failed_retrying",
                    extra={
                        "function": config.func.__name__,
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

                # Calculate next delay based on strategy
                if strategy == "exponential":
                    current_delay *= 2
                elif strategy == "linear":
                    current_delay += delay
                # For unknown strategies, keep constant delay

                # If this was the last attempt, we'll raise after loop
                if attempt == attempts:
                    break

        # Should never be None (loop always catches exceptions), but handle defensively
        if last_exception is None:
            return RuntimeError("Retry loop completed without success or exception")
        return last_exception

    @staticmethod
    def _handle_retry_exhaustion(
        last_exception: Exception,
        func: Callable[P, R],
        attempts: int,
        error_code: str | None,
        logger: FlextLogger,
    ) -> None:
        """Handle retry exhaustion and raise appropriate exception."""
        # All retries exhausted
        if last_exception:
            logger.error(
                "operation_failed_all_retries_exhausted",
                extra={
                    "function": func.__name__,
                    "attempts": attempts,
                    "error": str(last_exception),
                    "error_type": type(last_exception).__name__,
                },
            )
        else:
            logger.error(
                "operation_failed_all_retries_exhausted",
                extra={
                    "function": func.__name__,
                    "attempts": attempts,
                    "error": "Unknown error",
                },
            )

        # Raise the last exception
        if last_exception:
            raise last_exception

        # Should never reach here, but type safety
        msg = f"Operation {func.__name__} failed after {attempts} attempts"
        # Fast fail: error_code must be str or None
        effective_error_code: str = (
            error_code if isinstance(error_code, str) else "RETRY_EXHAUSTED"
        )
        raise FlextExceptions.TimeoutError(
            msg,
            error_code=effective_error_code,
        )

    @staticmethod
    def _bind_operation_context(
        *,
        operation: str,
        logger: FlextLogger,
        function_name: str,
        ensure_correlation: bool,
    ) -> str | None:
        """Ensure correlation, bind operation context, and report failures."""
        correlation_id: str | None = None
        if ensure_correlation:
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
        else:
            current_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            if isinstance(current_id, str) and current_id:
                correlation_id = current_id

        FlextContext.Request.set_operation_name(operation)

        binding_result = FlextLogger.bind_context(
            FlextConstants.Context.SCOPE_OPERATION,
            operation=operation,
        )
        if binding_result.is_failure:
            logger.warning(
                "operation_context_binding_failed",
                extra={
                    "function": function_name,
                    "operation": operation,
                    "error": binding_result.error,
                    "error_code": binding_result.error_code,
                    "correlation_id": correlation_id,
                },
            )
        return correlation_id

    @staticmethod
    def _clear_operation_scope(
        *,
        logger: FlextLogger,
        function_name: str,
        operation: str,
    ) -> None:
        """Clear operation scope and log if cleanup fails."""
        clear_result = FlextLogger.clear_scope("operation")
        if clear_result.is_failure:
            FlextDecorators._handle_log_result(
                result=clear_result,
                logger=logger,
                fallback_message="operation_context_clear_failed",
                kwargs={
                    "extra": {
                        "function": function_name,
                        "operation": operation,
                    },
                },
            )

    @staticmethod
    def _handle_log_result(
        *,
        result: FlextProtocols.ResultProtocol[bool],
        logger: FlextLogger,
        fallback_message: str,
        kwargs: FlextTypes.GeneralValueType,
    ) -> None:
        """Ensure FlextLogger call results are handled for diagnostics."""
        if result.is_failure:
            fallback_logger = getattr(logger, "logger", None)
            if fallback_logger is None or not hasattr(fallback_logger, "warning"):
                return
            # Convert kwargs to dict if it's a Mapping, otherwise use empty dict
            if FlextRuntime.is_dict_like(kwargs):
                fallback_kwargs = (
                    kwargs if isinstance(kwargs, dict) else dict(kwargs.items())
                )
            else:
                fallback_kwargs = {}
            _ = fallback_kwargs.setdefault("extra", {})
            extra_payload = fallback_kwargs["extra"]
            if FlextRuntime.is_dict_like(extra_payload):
                extra_payload = dict(extra_payload)
                extra_payload["log_error"] = result.error
                extra_payload["log_error_code"] = result.error_code
                fallback_kwargs["extra"] = extra_payload
            else:
                fallback_kwargs["log_error"] = result.error
                fallback_kwargs["log_error_code"] = result.error_code
            fallback_logger.warning(fallback_message, **fallback_kwargs)

    @staticmethod
    def timeout(
        timeout_seconds: float | None = None,
        error_code: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Enforce operation timeout, raising TimeoutError if exceeded."""
        # Use FlextConstants.Defaults for default
        max_duration = (
            timeout_seconds
            if timeout_seconds is not None
            else FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS
        )

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start_time = time.perf_counter()

                try:
                    result = func(*args, **kwargs)

                    # Check if operation exceeded timeout
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        msg = f"Operation {func.__name__} exceeded timeout of {max_duration}s (took {duration:.2f}s)"
                        # Fast fail: error_code must be str or None
                        effective_error_code: str = (
                            error_code
                            if isinstance(error_code, str)
                            else "OPERATION_TIMEOUT"
                        )
                        # Use dict directly - no need to create Metadata object
                        raise FlextExceptions.TimeoutError(
                            msg,
                            error_code=effective_error_code,
                            timeout_seconds=max_duration,
                            operation=func.__name__,
                            context={"duration_seconds": duration},
                        )

                    return result

                except FlextExceptions.TimeoutError:
                    # Re-raise timeout errors
                    raise
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    # Check duration even on exception
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        msg = f"Operation {func.__name__} exceeded timeout of {max_duration}s (took {duration:.2f}s) and raised {type(e).__name__}"
                        # Use dict directly - no need to create Metadata object
                        raise FlextExceptions.TimeoutError(
                            msg,
                            error_code=error_code or "OPERATION_TIMEOUT",
                            timeout_seconds=max_duration,
                            operation=func.__name__,
                            context={
                                "duration_seconds": duration,
                                "original_error": str(e),
                            },
                        ) from e
                    # Re-raise original exception if not timeout
                    raise

            return wrapper

        return decorator

    @staticmethod
    def combined(
        *,
        inject_deps: dict[str, str] | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        use_railway: bool = False,
        error_code: str | None = None,
    ) -> Callable[
        [Callable[P, R]], Callable[P, R] | Callable[P, FlextProtocols.ResultProtocol[R]]
    ]:
        """Combine inject, log_operation, and optionally railway patterns."""

        def decorator(
            func: Callable[P, R],
        ) -> Callable[P, R] | Callable[P, FlextProtocols.ResultProtocol[R]]:
            # Apply railway pattern first if requested (outermost wrapper)
            # Note: When use_railway=True, the return type changes to Callable[P, FlextProtocols.ResultProtocol[R]]
            if use_railway:
                railway_decorated = FlextDecorators.railway(error_code=error_code)(func)
                # Apply other decorators to railway-wrapped function
                if inject_deps:
                    railway_decorated = FlextDecorators.inject(**inject_deps)(
                        railway_decorated,
                    )
                return FlextDecorators.log_operation(
                    operation_name=operation_name,
                    track_perf=track_perf,
                )(railway_decorated)

            # Start with the base function - non-railway path returns Callable[P, R]
            decorated: Callable[P, R] = func

            # Apply dependency injection
            if inject_deps:
                decorated = FlextDecorators.inject(**inject_deps)(decorated)

            # Apply unified log_operation with optional performance tracking
            # Return type is Callable[P, R] for non-railway path
            return FlextDecorators.log_operation(
                operation_name=operation_name,
                track_perf=track_perf,
            )(decorated)

        return decorator

    @staticmethod
    def with_context(
        **context_vars: str | int | bool | None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Bind context variables for operation duration with automatic cleanup."""

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Convert P.args to tuple for logger resolution (type narrowing happens in _resolve_logger)
                args_tuple = tuple(args)
                logger = FlextDecorators._resolve_logger(args_tuple, func)

                try:
                    # Bind context variables to global logging context
                    if context_vars:
                        bind_result = FlextLogger.bind_global_context(**context_vars)
                        if bind_result.is_failure:
                            logger.warning(
                                "global_context_binding_failed",
                                extra={
                                    "function": func.__name__,
                                    "error": bind_result.error,
                                    "error_code": bind_result.error_code,
                                    "bound_keys": tuple(context_vars.keys()),
                                },
                            )

                    return func(*args, **kwargs)

                finally:
                    # Unbind context variables
                    if context_vars:
                        unbind_result = FlextLogger.unbind_global_context(
                            *tuple(context_vars.keys()),
                        )
                        if unbind_result.is_failure:
                            logger.warning(
                                "global_context_unbind_failed",
                                extra={
                                    "function": func.__name__,
                                    "error": unbind_result.error,
                                    "error_code": unbind_result.error_code,
                                    "bound_keys": tuple(context_vars.keys()),
                                },
                            )

            return wrapper

        return decorator

    @staticmethod
    def track_operation(
        operation_name: str | None = None,
        *,
        track_correlation: bool = True,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Track operation with correlation ID and structured logging."""

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Fast fail: explicit default value instead of 'or' fallback
                op_name: str = (
                    operation_name if operation_name is not None else func.__name__
                )

                # Convert P.args to tuple for logger resolution (type narrowing happens in _resolve_logger)
                args_tuple = tuple(args)
                logger = FlextDecorators._resolve_logger(args_tuple, func)

                correlation_id = FlextDecorators._bind_operation_context(
                    operation=op_name,
                    logger=logger,
                    function_name=func.__name__,
                    ensure_correlation=track_correlation,
                )
                if track_correlation and correlation_id is None:
                    logger.warning(
                        "correlation_id_missing",
                        extra={
                            "function": func.__name__,
                            "operation": op_name,
                        },
                    )

                try:
                    # Call the actual function
                    # Performance tracking via FlextRuntime.Integration happens automatically
                    return func(*args, **kwargs)

                finally:
                    # Clear operation context
                    with suppress(Exception):
                        FlextDecorators._clear_operation_scope(
                            logger=logger,
                            function_name=func.__name__,
                            operation=op_name,
                        )

            return wrapper

        return decorator


__all__ = [
    "FlextDecorators",
]
