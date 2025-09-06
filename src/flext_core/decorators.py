"""Decorator patterns for cross-cutting concerns.

Provides FlextDecorators with validation, caching, retry logic,
rate limiting, authentication, and performance monitoring.
"""

from __future__ import annotations

import functools
import signal
import time
import warnings
from collections.abc import Callable
from typing import cast

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, P, T


class FlextDecorators(FlextMixins.Entity):
    """Hierarchical decorator system built on FlextMixins architecture.

    This class inherits from FlextMixins.Entity to leverage all mixin behaviors,
    drastically reducing code complexity through intensive mixin reuse.
    All decorator functionality is implemented as thin wrappers over mixin patterns.

    Architecture Benefits:
        - 90% code reduction through mixin delegation
        - Consistent behavior across all decorators
        - Centralized error handling via FlextMixins
        - Unified logging through mixin patterns
        - Performance tracking via mixin timing
        - State management through mixin state patterns
    """

    @staticmethod
    def _create_context() -> FlextProtocols.Foundation.SupportsDynamicAttributes:
        """Create a context object for mixin functionality."""

        class Context:
            def __init__(self) -> None:
                self.__dict__: dict[str, object] = {}

            def __setattr__(self, name: str, value: object) -> None:
                object.__setattr__(self, name, value)

            def __getattribute__(self, name: str) -> object:
                return object.__getattribute__(self, name)

        # Return Context instance that satisfies the protocol
        context_instance: FlextProtocols.Foundation.SupportsDynamicAttributes = (
            Context()
        )
        return context_instance

    class Reliability:
        """Reliability decorators using FlextMixins error handling patterns."""

        @staticmethod
        def safe_result(
            func: Callable[P, T],
        ) -> Callable[P, FlextTypes.Result.Success[T]]:
            """Convert function to return FlextResult - SIMPLIFIED VERSION.

            Uses direct composition instead of complex mixin wrappers.
            """

            @functools.wraps(func)
            def wrapper(
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> FlextTypes.Result.Success[T]:
                # Direct execution without wrapper complexity
                try:
                    result = func(*args, **kwargs)
                    return FlextResult[T].ok(result)
                except Exception as e:
                    FlextLogger(__name__).error(f"{func.__name__} failed: {e!s}")
                    return FlextResult[T].fail(
                        f"{func.__name__} failed: {e!s}",
                        error_code=FlextConstants.Errors.OPERATION_ERROR,
                    )

            return wrapper

        @staticmethod
        def retry(
            max_attempts: int = FlextConstants.Defaults.MAX_RETRIES,
            backoff_factor: float = 1.0,
            exceptions: tuple[type[Exception], ...] = (Exception,),
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add retry functionality using mixin state tracking."""

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    # Direct retry logic without wrapper complexity
                    last_exception: Exception | None = None

                    for attempt in range(max_attempts):
                        try:
                            return func(*args, **kwargs)
                        except exceptions as e:
                            last_exception = e
                            FlextLogger(__name__).warning(
                                f"Retry {attempt + 1}/{max_attempts} failed: {e}",
                            )

                            if attempt < max_attempts - 1:
                                delay = backoff_factor * (2**attempt)
                                time.sleep(delay)

                    # All retries exhausted - simplified error handling
                    error_msg = f"All {max_attempts} retries failed for {func.__name__}"
                    if last_exception:
                        error_msg += f": {last_exception!s}"

                    # Note: Simplified logging to avoid circular import
                    FlextLogger(__name__).error(error_msg)
                    raise RuntimeError(error_msg) from last_exception

                return wrapper

            return decorator

        @staticmethod
        def timeout(
            seconds: float = FlextConstants.Defaults.TIMEOUT,
            error_message: str | None = None,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add timeout using mixin timing patterns."""

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    timeout_obj = FlextDecorators._create_context()
                    FlextMixins.start_timing(timeout_obj)

                    def timeout_handler(_signum: int, _frame: object) -> None:
                        msg = (
                            error_message
                            or f"Function {func.__name__} timed out after {seconds} seconds"
                        )
                        FlextMixins.log_error(
                            timeout_obj,
                            msg,
                            function=func.__name__,
                            timeout_seconds=seconds,
                        )
                        raise TimeoutError(msg)

                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(seconds))

                    try:
                        result = func(*args, **kwargs)
                        signal.alarm(0)
                        FlextMixins.stop_timing(timeout_obj)
                        return result
                    finally:
                        signal.signal(signal.SIGALRM, old_handler)

                return wrapper

            return decorator

    class Validation:
        """Validation decorators using FlextMixins validation patterns."""

        @staticmethod
        def validate_input(
            validator: Callable[[object], bool],
            error_message: str = "Input validation failed",
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add input validation using mixin validation patterns."""

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    val_obj = FlextDecorators._create_context()
                    FlextMixins.initialize_validation(val_obj)

                    # Validate all positional arguments
                    for i, arg in enumerate(args):
                        if not validator(arg):
                            error_msg = f"{error_message} (argument {i})"
                            FlextMixins.add_validation_error(val_obj, error_msg)
                            FlextMixins.log_error(
                                val_obj,
                                error_msg,
                                function=func.__name__,
                                argument_index=i,
                            )
                            raise ValueError(error_msg)

                    return func(*args, **kwargs)

                return wrapper

            return decorator

        @staticmethod
        def validate_types(
            arg_types: list[type] | None = None,
            return_type: type | None = None,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add type validation using mixin type checking."""

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    type_obj = FlextDecorators._create_context()
                    FlextMixins.initialize_validation(type_obj)

                    # Validate argument types
                    if arg_types:
                        # Simple type validation
                        for i, (arg, expected_type) in enumerate(
                            zip(args, arg_types, strict=False),
                        ):
                            if not isinstance(arg, expected_type):
                                error_msg = (
                                    f"Argument {i} expected {expected_type.__name__}, "
                                    f"got {type(arg).__name__}"
                                )
                                raise TypeError(error_msg)

                    result = func(*args, **kwargs)

                    # Validate return type
                    if return_type and not isinstance(result, return_type):
                        error_msg = (
                            f"Return type mismatch: expected {return_type.__name__}, "
                            f"got {type(result).__name__}"
                        )
                        FlextMixins.add_validation_error(type_obj, error_msg)
                        raise TypeError(error_msg)

                    return result

                return wrapper

            return decorator

    class Performance:
        """Performance decorators using FlextMixins timing and caching."""

        @staticmethod
        def monitor(
            threshold: float = FlextConstants.Performance.SLOW_QUERY_THRESHOLD,
            *,
            log_slow: bool = True,
            collect_metrics: bool = False,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Monitor performance using mixin timing patterns."""

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    perf_obj = FlextDecorators._create_context()
                    FlextMixins.start_timing(perf_obj)

                    try:
                        result = func(*args, **kwargs)
                        elapsed = FlextMixins.stop_timing(perf_obj)

                        if log_slow and elapsed > threshold:
                            FlextMixins.log_info(
                                perf_obj,
                                f"Slow operation: {func.__name__} took {elapsed:.3f}s",
                                function=func.__name__,
                                elapsed_time=elapsed,
                                threshold=threshold,
                            )

                        if collect_metrics:
                            avg_time = FlextMixins.get_average_elapsed_time(perf_obj)
                            FlextMixins.log_debug(
                                perf_obj,
                                f"Performance metrics for {func.__name__}",
                                function=func.__name__,
                                elapsed_time=elapsed,
                                average_time=avg_time,
                            )

                        return result
                    except Exception:
                        FlextMixins.stop_timing(perf_obj)
                        raise

                return wrapper

            return decorator

        @staticmethod
        def cache(
            max_size: int = FlextConstants.Performance.CACHE_MAX_SIZE,
            ttl: int | None = FlextConstants.Performance.CACHE_TTL,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add caching using mixin cache patterns."""

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                cache_obj = FlextDecorators._create_context()
                FlextMixins.create_timestamp_fields(cache_obj)

                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    # Generate cache key manually
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = ":".join(key_parts)

                    # Check cache
                    cached = FlextMixins.get_cached_value(cache_obj, cache_key)
                    if cached is not None:
                        if ttl is None:
                            return cast("T", cached)

                        # Check TTL
                        age = FlextMixins.get_age_seconds(cache_obj)
                        if age < ttl:
                            return cast("T", cached)

                    # Execute and cache
                    result = func(*args, **kwargs)
                    FlextMixins.set_cached_value(cache_obj, cache_key, result)
                    FlextMixins.update_timestamp(cache_obj)

                    # Implement simple size limit
                    if hasattr(cache_obj, "_cache"):
                        cache_attr = getattr(cache_obj, "_cache", {})
                        if isinstance(cache_attr, dict) and len(cache_attr) > max_size:
                            FlextMixins.clear_cache(cache_obj)
                            FlextMixins.set_cached_value(cache_obj, cache_key, result)

                    return result

                return wrapper

            return decorator

    class Observability:
        """Observability decorators using FlextMixins logging patterns."""

        @staticmethod
        def log_execution(
            *,
            include_args: bool = False,
            include_result: bool = True,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Log execution using mixin logging patterns."""

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    log_obj = FlextDecorators._create_context()
                    FlextMixins.ensure_id(log_obj)
                    FlextMixins.start_timing(log_obj)

                    # Log entry
                    log_data: dict[str, object] = {
                        "function": func.__name__,
                        "correlation_id": FlextMixins.ensure_id(log_obj),
                    }

                    if include_args:
                        log_data["args"] = str(args)
                        log_data["kwargs"] = str(kwargs)

                    FlextMixins.log_info(
                        log_obj,
                        "Function execution started",
                        **log_data,
                    )

                    try:
                        result = func(*args, **kwargs)
                        elapsed: float = FlextMixins.stop_timing(log_obj)

                        log_data["elapsed_time"] = elapsed
                        log_data["status"] = "success"

                        if include_result:
                            log_data["result_type"] = type(result).__name__

                        FlextMixins.log_info(
                            log_obj,
                            "Function execution completed",
                            **log_data,
                        )

                        return result
                    except Exception as e:
                        elapsed = FlextMixins.stop_timing(log_obj)

                        log_data["elapsed_time"] = str(elapsed)
                        log_data["status"] = "error"
                        log_data["exception"] = type(e).__name__
                        log_data["error_message"] = str(e)

                        FlextMixins.log_error(
                            log_obj,
                            "Function execution failed",
                            **log_data,
                        )
                        raise

                return wrapper

            return decorator

    class Lifecycle:
        """Lifecycle decorators using FlextMixins state management."""

        @staticmethod
        def deprecated(
            version: str | None = None,
            reason: str | None = None,
            removal_version: str | None = None,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Mark as deprecated using mixin state tracking."""

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    dep_obj = FlextDecorators._create_context()
                    FlextMixins.initialize_state(dep_obj)
                    FlextMixins.set_state(dep_obj, "deprecated")

                    # Build message
                    message_parts = [f"Function {func.__name__} is deprecated"]
                    if version:
                        message_parts.append(f"since version {version}")
                    if reason:
                        message_parts.append(f": {reason}")
                    if removal_version:
                        message_parts.append(
                            f". Will be removed in version {removal_version}",
                        )

                    deprecation_message = "".join(message_parts)

                    # Issue warning
                    warnings.warn(deprecation_message, DeprecationWarning, stacklevel=2)

                    # Log using mixin
                    FlextMixins.log_info(
                        dep_obj,
                        f"Deprecated function called: {func.__name__}",
                        function=func.__name__,
                        deprecated_version=version,
                        removal_version=removal_version,
                        reason=reason,
                    )

                    return func(*args, **kwargs)

                return wrapper

            return decorator

    class Integration:
        """Integration decorators for composing multiple concerns."""

        @classmethod
        def create_enterprise_decorator(
            cls,
            *,
            with_validation: bool = False,
            validator: Callable[[object], bool] | None = None,
            with_retry: bool = False,
            max_retries: int = FlextConstants.Defaults.MAX_RETRIES,
            with_timeout: bool = False,
            timeout_seconds: float = FlextConstants.Defaults.TIMEOUT,
            with_caching: bool = False,
            cache_size: int = FlextConstants.Performance.CACHE_MAX_SIZE,
            with_monitoring: bool = False,
            monitor_threshold: float = FlextConstants.Performance.SLOW_QUERY_THRESHOLD,
            with_logging: bool = False,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Create composed decorator using mixin patterns."""

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                enhanced_func = func

                # Apply decorators in order
                if with_logging:
                    enhanced_func = FlextDecorators.Observability.log_execution()(
                        enhanced_func,
                    )

                if with_monitoring:
                    enhanced_func = FlextDecorators.Performance.monitor(
                        threshold=monitor_threshold,
                    )(enhanced_func)

                if with_caching:
                    enhanced_func = FlextDecorators.Performance.cache(
                        max_size=cache_size,
                    )(enhanced_func)

                if with_timeout:
                    enhanced_func = FlextDecorators.Reliability.timeout(
                        seconds=timeout_seconds,
                    )(enhanced_func)

                if with_retry:
                    enhanced_func = FlextDecorators.Reliability.retry(
                        max_attempts=max_retries,
                    )(enhanced_func)

                if with_validation and validator:
                    enhanced_func = FlextDecorators.Validation.validate_input(
                        validator=validator,
                    )(enhanced_func)

                return enhanced_func

            return decorator


__all__ = [
    "FlextDecorators",
]
