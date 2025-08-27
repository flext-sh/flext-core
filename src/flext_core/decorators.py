"""Enterprise decorator patterns for the FLEXT core library.

This module provides the hierarchical FlextDecorators class system following
FLEXT_REFACTORING_PROMPT.md requirements with proper SOLID principles,
zero circular imports, and full integration with FlextTypes, FlextConstants,
and FlextProtocols.

Built for Python 3.13+ with strict typing enforcement and enterprise-grade
reliability patterns.

Architecture:
    FlextDecorators
    ├── Reliability    # Safe execution, retries, timeouts
    ├── Validation     # Input/output validation, type checking
    ├── Performance    # Monitoring, caching, profiling
    ├── Observability  # Logging, tracing, metrics
    ├── Lifecycle      # Deprecation, versioning
    └── Integration    # Cross-cutting decorator composition

Patterns Applied:
    - Single Responsibility: Each nested class handles one decorator category
    - Open/Closed: Easy to extend with new decorator categories
    - Liskov Substitution: All decorators maintain consistent interfaces
    - Interface Segregation: Clients use only needed decorator categories
    - Dependency Inversion: Decorators depend on FlextTypes/FlextProtocols abstractions

Examples:
    Safe execution with FlextResult::

        @FlextDecorators.Reliability.safe_result
        def process_data(data: dict) -> int:
            return len(data)


        result = process_data({"key": "value"})
        # Returns: FlextResult.ok(1)

    Performance monitoring::

        @FlextDecorators.Performance.monitor(
            threshold=FlextConstants.Performance.SLOW_QUERY_THRESHOLD
        )
        def expensive_operation() -> dict:
            return {"status": "completed"}

    Input validation::

        @FlextDecorators.Validation.validate_input(
            lambda x: isinstance(x, str) and len(x) > 0
        )
        def process_text(text: str) -> str:
            return text.upper()

Note:
    All decorators integrate with FlextResult for consistent error handling
    and use FlextConstants for configuration defaults. Legacy compatibility
    is maintained through minimal facades in legacy.py without ABI changes.

"""

from __future__ import annotations

import functools
import signal
import time
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from flext_core.constants import FlextConstants
from flext_core.loggings import get_logger
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Type variables for decorator patterns
P = ParamSpec("P")
T = TypeVar("T")

# Logger instance using FLEXT logging patterns
logger = get_logger(__name__)


class FlextDecorators:
    """Hierarchical decorator system organized by concern and responsibility.

    This is the main entry point for all decorator patterns in the FLEXT ecosystem,
    following FLEXT_REFACTORING_PROMPT.md requirements with proper SOLID principles
    and Clean Architecture patterns.

    The decorator system is organized into the following categories:
        - Reliability: Safe execution, retries, timeouts, error handling
        - Validation: Input/output validation, type checking, constraints
        - Performance: Monitoring, caching, profiling, optimization
        - Observability: Logging, tracing, metrics, debugging
        - Lifecycle: Deprecation, versioning, compatibility
        - Integration: Cross-cutting decorator composition and factories

    Architecture Principles:
        - Single Responsibility: Each nested class handles one decorator category
        - Open/Closed: Easy to extend with new decorator categories
        - Liskov Substitution: All decorators maintain consistent FlextResult interfaces
        - Interface Segregation: Clients depend only on decorators they need
        - Dependency Inversion: All decorators depend on FlextTypes/FlextProtocols abstractions

    Examples:
        Using reliability decorators::

            # Safe execution with automatic FlextResult wrapping
            @FlextDecorators.Reliability.safe_result
            def risky_operation(data: dict) -> int:
                return data["count"]  # May raise KeyError


            result = risky_operation({})
            # Returns: FlextResult.fail("Key 'count' not found")

        Using performance decorators::

            # Performance monitoring with FlextConstants integration
            @FlextDecorators.Performance.monitor(
                threshold=FlextConstants.Performance.SLOW_QUERY_THRESHOLD,
                collect_metrics=True,
            )
            def database_query() -> list[dict]:
                return [{"id": 1, "name": "test"}]

        Using validation decorators::

            # Type-safe input validation
            @FlextDecorators.Validation.validate_input(
                lambda x: isinstance(x, str) and len(x) > 0,
                error_message="Input must be non-empty string",
            )
            def process_text(text: str) -> FlextResult[str]:
                return FlextResult.ok(text.upper())

    """

    class Reliability:
        """Reliability decorators for safe execution and error handling.

        This class provides decorators that enhance function reliability through
        safe execution patterns, retry logic, timeouts, and automatic error
        handling with FlextResult integration.

        All decorators in this class follow these principles:
            - Single Responsibility: Focus only on reliability concerns
            - Fail-Safe Defaults: Conservative settings using FlextConstants
            - Error Isolation: Prevent cascading failures through proper isolation
            - FlextResult Integration: Consistent error handling patterns

        Examples:
            Safe execution with FlextResult::

                @FlextDecorators.Reliability.safe_result
                def parse_json(text: str) -> dict:
                    import json

                    return json.loads(text)  # May raise JSONDecodeError


                result = parse_json("invalid json")
                # Returns: FlextResult.fail("JSON decode error: ...")

            Retry with exponential backoff::

                @FlextDecorators.Reliability.retry(
                    max_attempts=FlextConstants.Defaults.MAX_RETRIES,
                    exceptions=(ConnectionError, TimeoutError),
                )
                def connect_to_service() -> dict:
                    return {"status": "connected"}

        """

        @staticmethod
        def safe_result(
            func: Callable[P, T],
        ) -> Callable[P, FlextTypes.Result.Success[T]]:
            """Convert function to return FlextResult for safe execution.

            Wraps any function to automatically catch exceptions and convert them
            to FlextResult.fail() responses, enabling railway-oriented programming
            throughout the application.

            Args:
                func: Function to wrap with safe execution

            Returns:
                Function that returns FlextResult[T] instead of T

            Examples:
                Basic usage::

                    @FlextDecorators.Reliability.safe_result
                    def divide(a: int, b: int) -> float:
                        return a / b


                    result = divide(10, 0)
                    # Returns: FlextResult.fail("division by zero")

                With async functions::

                    @FlextDecorators.Reliability.safe_result
                    async def fetch_data() -> dict:
                        return {"data": "value"}


                    result = await fetch_data()
                    # Returns: FlextResult.ok({"data": "value"})

            """

            @functools.wraps(func)
            def wrapper(
                *args: P.args, **kwargs: P.kwargs
            ) -> FlextTypes.Result.Success[T]:
                try:
                    result = func(*args, **kwargs)
                    return FlextResult[T].ok(result)
                except Exception as e:
                    error_message = f"{func.__name__} failed: {e!s}"
                    logger.exception(
                        error_message,
                        extra={
                            "function": func.__name__,
                            "function_module": func.__module__,
                            "exception": type(e).__name__,
                            "error_code": FlextConstants.Errors.OPERATION_ERROR,
                        },
                    )
                    return FlextResult[T].fail(
                        error_message,
                        error_code=FlextConstants.Errors.OPERATION_ERROR,
                    )

            return wrapper

        @staticmethod
        def retry(
            max_attempts: int = FlextConstants.Defaults.MAX_RETRIES,
            backoff_factor: float = 1.0,
            exceptions: tuple[type[Exception], ...] = (Exception,),
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add retry functionality with exponential backoff.

            Retries the decorated function on specified exceptions with configurable
            backoff strategy using FlextConstants for defaults.

            Args:
                max_attempts: Maximum number of retry attempts
                backoff_factor: Multiplier for exponential backoff delay
                exceptions: Tuple of exception types to retry on

            Returns:
                Decorator that adds retry functionality

            Examples:
                Basic retry with defaults::

                    @FlextDecorators.Reliability.retry()
                    def unstable_operation() -> str:
                        if random.random() < 0.7:
                            raise ConnectionError("Network unavailable")
                        return "success"

                Custom retry configuration::

                    @FlextDecorators.Reliability.retry(
                        max_attempts=5,
                        backoff_factor=2.0,
                        exceptions=(ConnectionError, TimeoutError),
                    )
                    def critical_operation() -> dict:
                        return {"result": "processed"}

            """

            def decorator(
                func: Callable[P, T],
            ) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    last_exception: Exception | None = None

                    for attempt in range(max_attempts):
                        try:
                            return func(*args, **kwargs)
                        except exceptions as e:
                            last_exception = e
                            if attempt == max_attempts - 1:
                                break

                            delay = backoff_factor * (2**attempt)
                            logger.warning(
                                f"Retry attempt {attempt + 1}/{max_attempts} failed for {func.__name__}",
                                extra={
                                    "function": func.__name__,
                                    "attempt": attempt + 1,
                                    "max_attempts": max_attempts,
                                    "delay": delay,
                                    "exception": str(e),
                                },
                            )
                            time.sleep(delay)

                    # All retries exhausted
                    error_message = (
                        f"All {max_attempts} retry attempts failed for {func.__name__}"
                    )
                    if last_exception:
                        error_message += f": {last_exception!s}"

                    logger.error(
                        error_message,
                        extra={
                            "function": func.__name__,
                            "max_attempts": max_attempts,
                            "final_exception": str(last_exception)
                            if last_exception
                            else None,
                            "error_code": FlextConstants.Errors.RETRY_ERROR,
                        },
                    )
                    raise RuntimeError(error_message) from last_exception

                return wrapper

            return decorator

        @staticmethod
        def timeout(
            seconds: float = FlextConstants.Defaults.TIMEOUT,
            error_message: str | None = None,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add timeout functionality to function execution.

            Adds a timeout mechanism to function execution using FlextConstants
            for default timeout values.

            Args:
                seconds: Timeout duration in seconds
                error_message: Custom error message for timeout

            Returns:
                Decorator that adds timeout functionality

            Examples:
                Default timeout::

                    @FlextDecorators.Reliability.timeout()
                    def long_operation() -> str:
                        time.sleep(60)  # Will timeout after 30s
                        return "completed"

                Custom timeout::

                    @FlextDecorators.Reliability.timeout(
                        seconds=5.0, error_message="Database query timed out"
                    )
                    def database_query() -> list[dict]:
                        return [{"id": 1}]

            """

            def decorator(
                func: Callable[P, T],
            ) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    def timeout_handler(_signum: int, _frame: object) -> None:
                        msg = (
                            error_message
                            or f"Function {func.__name__} timed out after {seconds} seconds"
                        )
                        raise TimeoutError(msg)

                    # Set up timeout signal
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(seconds))

                    try:
                        result = func(*args, **kwargs)
                        signal.alarm(0)  # Cancel timeout
                        return result
                    except TimeoutError:
                        logger.exception(
                            f"Function {func.__name__} timed out",
                            extra={
                                "function": func.__name__,
                                "timeout_seconds": seconds,
                                "error_code": FlextConstants.Errors.TIMEOUT_ERROR,
                            },
                        )
                        raise
                    finally:
                        signal.signal(signal.SIGALRM, old_handler)

                return wrapper

            return decorator

    class Validation:
        """Validation decorators for input/output validation and type checking.

        This class provides decorators for validating function inputs, outputs,
        and type constraints with integration to FlextResult patterns and
        FlextConstants for error codes.

        All decorators in this class follow these principles:
            - Single Responsibility: Focus only on validation concerns
            - Type Safety: Leverage FlextTypes for proper type annotations
            - Error Consistency: Use FlextResult for validation error reporting
            - Configurable Validation: Use FlextProtocols.Foundation.Validator pattern

        Examples:
            Input validation with custom predicate::

                @FlextDecorators.Validation.validate_input(
                    lambda x: isinstance(x, int) and x > 0,
                    "Input must be positive integer",
                )
                def calculate_factorial(n: int) -> int:
                    result = 1
                    for i in range(1, n + 1):
                        result *= i
                    return result

            Type validation for multiple arguments::

                @FlextDecorators.Validation.validate_types(
                    arg_types=[str, int], return_type=str
                )
                def format_message(name: str, count: int) -> str:
                    return f"{name}: {count}"

        """

        @staticmethod
        def validate_input(
            validator: Callable[[object], bool],
            error_message: str = "Input validation failed",
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add input validation to function execution.

            Validates function inputs using provided validator before execution.
            Integrates with FlextResult patterns for consistent error handling.

            Args:
                validator: Validation function that returns bool
                error_message: Error message for validation failures

            Returns:
                Decorator that adds input validation

            Examples:
                String length validation::

                    @FlextDecorators.Validation.validate_input(
                        lambda text: isinstance(text, str) and len(text) > 0,
                        "Text must be non-empty string",
                    )
                    def process_text(text: str) -> str:
                        return text.upper()

                Numeric range validation::

                    @FlextDecorators.Validation.validate_input(
                        lambda x: isinstance(x, int) and 0 <= x <= 100,
                        "Value must be integer between 0 and 100",
                    )
                    def calculate_percentage(value: int) -> float:
                        return value / 100.0

            """

            def decorator(
                func: Callable[P, T],
            ) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    # Validate all positional arguments
                    for i, arg in enumerate(args):
                        if not validator(arg):
                            logger.warning(
                                f"Input validation failed for argument {i}",
                                extra={
                                    "function": func.__name__,
                                    "argument_index": i,
                                    "argument_type": type(arg).__name__,
                                    "error_code": FlextConstants.Errors.VALIDATION_ERROR,
                                },
                            )
                            msg = f"{error_message} (argument {i})"
                            raise ValueError(msg)

                    return func(*args, **kwargs)

                return wrapper

            return decorator

        @staticmethod
        def validate_types(
            arg_types: list[type] | None = None,
            return_type: type | None = None,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add type validation to function execution.

            Validates function argument and return types at runtime for additional
            type safety beyond static type checking.

            Args:
                arg_types: Expected types for positional arguments
                return_type: Expected return type

            Returns:
                Decorator that adds type validation

            Examples:
                Argument type validation::

                    @FlextDecorators.Validation.validate_types(arg_types=[str, int])
                    def repeat_string(text: str, count: int) -> str:
                        return text * count

                Return type validation::

                    @FlextDecorators.Validation.validate_types(return_type=dict)
                    def get_config() -> dict:
                        return {"debug": True}

            """

            def decorator(
                func: Callable[P, T],
            ) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    # Validate argument types
                    if arg_types:
                        for i, (arg, expected_type) in enumerate(
                            zip(args, arg_types, strict=False)
                        ):
                            if not isinstance(arg, expected_type):
                                error_msg = (
                                    f"Argument {i} type mismatch: expected {expected_type.__name__}, "
                                    f"got {type(arg).__name__}"
                                )
                                logger.error(
                                    error_msg,
                                    extra={
                                        "function": func.__name__,
                                        "argument_index": i,
                                        "expected_type": expected_type.__name__,
                                        "actual_type": type(arg).__name__,
                                        "error_code": FlextConstants.Errors.TYPE_ERROR,
                                    },
                                )
                                raise TypeError(error_msg)

                    result = func(*args, **kwargs)

                    # Validate return type
                    if return_type and not isinstance(result, return_type):
                        error_msg = (
                            f"Return type mismatch: expected {return_type.__name__}, "
                            f"got {type(result).__name__}"
                        )
                        logger.error(
                            error_msg,
                            extra={
                                "function": func.__name__,
                                "expected_type": return_type.__name__,
                                "actual_type": type(result).__name__,
                                "error_code": FlextConstants.Errors.TYPE_ERROR,
                            },
                        )
                        raise TypeError(error_msg)

                    return result

                return wrapper

            return decorator

    class Performance:
        """Performance decorators for monitoring, caching, and optimization.

        This class provides decorators that enhance function performance through
        monitoring, caching, profiling, and optimization with integration to
        FlextConstants for thresholds and configuration.

        All decorators in this class follow these principles:
            - Single Responsibility: Focus only on performance concerns
            - Metrics Integration: Use FlextConstants.Performance for thresholds
            - Observability: Integrate with structured logging for metrics
            - Memory Efficiency: Implement proper cache eviction policies

        Examples:
            Performance monitoring with thresholds::

                @FlextDecorators.Performance.monitor(
                    threshold=FlextConstants.Performance.SLOW_QUERY_THRESHOLD
                )
                def database_query() -> list[dict]:
                    return [{"id": 1, "name": "test"}]

            Caching with TTL::

                @FlextDecorators.Performance.cache(
                    max_size=FlextConstants.Performance.CACHE_MAX_SIZE,
                    ttl=FlextConstants.Performance.CACHE_TTL,
                )
                def expensive_calculation(x: int) -> int:
                    return x**2

        """

        @staticmethod
        def monitor(
            threshold: float = FlextConstants.Performance.SLOW_QUERY_THRESHOLD,
            *,
            log_slow: bool = True,
            collect_metrics: bool = False,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add performance monitoring to function execution.

            Monitors function execution time and logs warnings for slow operations
            based on FlextConstants.Performance thresholds.

            Args:
                threshold: Time threshold in seconds for slow operation warning
                log_slow: Whether to log warnings for slow operations
                collect_metrics: Whether to collect detailed performance metrics

            Returns:
                Decorator that adds performance monitoring

            Examples:
                Basic monitoring::

                    @FlextDecorators.Performance.monitor()
                    def database_operation() -> dict:
                        return {"result": "success"}

                Custom threshold monitoring::

                    @FlextDecorators.Performance.monitor(
                        threshold=0.5, log_slow=True, collect_metrics=True
                    )
                    def critical_operation() -> str:
                        return "processed"

            """

            def decorator(
                func: Callable[P, T],
            ) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    start_time = time.perf_counter()

                    try:
                        return func(*args, **kwargs)
                    finally:
                        elapsed_time = time.perf_counter() - start_time

                        # Log slow operations
                        if log_slow and elapsed_time > threshold:
                            logger.warning(
                                f"Slow operation detected: {func.__name__} took {elapsed_time:.3f}s",
                                extra={
                                    "function": func.__name__,
                                    "function_module": func.__module__,
                                    "elapsed_time": elapsed_time,
                                    "threshold": threshold,
                                    "performance_category": "slow_operation",
                                },
                            )

                        # Collect detailed metrics
                        if collect_metrics:
                            logger.info(
                                f"Performance metrics for {func.__name__}",
                                extra={
                                    "function": func.__name__,
                                    "elapsed_time": elapsed_time,
                                    "args_count": len(args),
                                    "kwargs_count": len(kwargs),
                                    "performance_category": "metrics",
                                },
                            )

                return wrapper

            return decorator

        @staticmethod
        def cache(
            max_size: int = FlextConstants.Performance.CACHE_MAX_SIZE,
            ttl: int | None = FlextConstants.Performance.CACHE_TTL,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add caching functionality to function execution.

            Implements LRU cache with optional TTL using FlextConstants.Performance
            for default cache configuration.

            Args:
                max_size: Maximum number of cached results
                ttl: Time-to-live for cached results in seconds

            Returns:
                Decorator that adds caching functionality

            Examples:
                Basic caching::

                    @FlextDecorators.Performance.cache()
                    def expensive_calculation(x: int, y: int) -> int:
                        return x**y

                Cache with TTL::

                    @FlextDecorators.Performance.cache(
                        max_size=100,
                        ttl=300,  # 5 minutes
                    )
                    def fetch_user_data(user_id: str) -> dict:
                        return {"id": user_id, "name": "User"}

            """

            def decorator(
                func: Callable[P, T],
            ) -> Callable[P, T]:
                # Create cache storage
                cache_storage: dict[str, tuple[T, float]] = {}
                cache_stats = {"hits": 0, "misses": 0}

                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    # Create cache key
                    key_parts = [str(arg) for arg in args]
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = f"{func.__name__}({','.join(key_parts)})"

                    current_time = time.time()

                    # Check cache hit
                    if cache_key in cache_storage:
                        cached_result, cached_time = cache_storage[cache_key]

                        # Check TTL expiration
                        if ttl is None or (current_time - cached_time) < ttl:
                            cache_stats["hits"] += 1
                            logger.debug(
                                f"Cache hit for {func.__name__}",
                                extra={
                                    "function": func.__name__,
                                    "cache_key": cache_key,
                                    "cache_age": current_time - cached_time,
                                },
                            )
                            return cached_result
                        # TTL expired, remove from cache
                        del cache_storage[cache_key]

                    # Cache miss - execute function
                    cache_stats["misses"] += 1
                    result = func(*args, **kwargs)

                    # Store in cache with size limit
                    if len(cache_storage) >= max_size:
                        # Remove oldest entry (simple FIFO eviction)
                        oldest_key = next(iter(cache_storage))
                        del cache_storage[oldest_key]

                    cache_storage[cache_key] = (result, current_time)

                    logger.debug(
                        f"Cache miss for {func.__name__}",
                        extra={
                            "function": func.__name__,
                            "cache_key": cache_key,
                            "cache_size": len(cache_storage),
                            "cache_hits": cache_stats["hits"],
                            "cache_misses": cache_stats["misses"],
                        },
                    )

                    return result

                return wrapper

            return decorator

    class Observability:
        """Observability decorators for logging, tracing, and debugging.

        This class provides decorators that enhance function observability through
        structured logging, execution tracing, and debugging support with integration
        to FlextConstants.Observability for configuration.

        All decorators in this class follow these principles:
            - Single Responsibility: Focus only on observability concerns
            - Structured Logging: Use structured logging with correlation IDs
            - Performance Awareness: Minimize observability overhead
            - Privacy Compliance: Configurable argument logging for sensitive data

        Examples:
            Execution logging::

                @FlextDecorators.Observability.log_execution()
                def process_order(order_id: str) -> dict:
                    return {"order_id": order_id, "status": "processed"}

            Custom logging with argument filtering::

                @FlextDecorators.Observability.log_execution(
                    include_args=False,  # Don't log sensitive arguments
                    log_level="INFO",
                )
                def authenticate_user(username: str, password: str) -> bool:
                    return True

        """

        @staticmethod
        def log_execution(
            *,
            include_args: bool = False,
            include_result: bool = True,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Add execution logging to function calls.

            Logs function entry, exit, and execution time with structured logging
            and correlation IDs for tracing.

            Args:
                include_args: Whether to include arguments in logs (security consideration)
                include_result: Whether to include return value in logs

            Returns:
                Decorator that adds execution logging

            Examples:
                Basic execution logging::

                    @FlextDecorators.Observability.log_execution()
                    def calculate_tax(amount: float) -> float:
                        return amount * 0.1

                Security-conscious logging::

                    @FlextDecorators.Observability.log_execution(
                        include_args=False,  # Don't log sensitive data
                        include_result=False,
                        log_level="DEBUG",
                    )
                    def process_payment(card_number: str, amount: float) -> bool:
                        return True

            """

            def decorator(
                func: Callable[P, T],
            ) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    correlation_id = f"exec_{int(time.time() * 1000)}"

                    log_data: FlextTypes.Core.Dict = {
                        "function": func.__name__,
                        "module": func.__module__,
                        "correlation_id": correlation_id,
                    }

                    if include_args:
                        log_data["args"] = args
                        log_data["kwargs"] = kwargs

                    logger.info("Function execution started", extra=log_data)

                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)

                        elapsed_time = time.perf_counter() - start_time
                        log_data.update(
                            {
                                "elapsed_time": elapsed_time,
                                "status": "success",
                            }
                        )

                        if include_result:
                            log_data["result_type"] = type(result).__name__

                        logger.info("Function execution completed", extra=log_data)
                        return result

                    except Exception as e:
                        elapsed_time = time.perf_counter() - start_time
                        log_data.update(
                            {
                                "elapsed_time": elapsed_time,
                                "status": "error",
                                "exception": type(e).__name__,
                                "error_message": str(e),
                            }
                        )

                        logger.exception("Function execution failed", extra=log_data)
                        raise

                return wrapper

            return decorator

    class Lifecycle:
        """Lifecycle decorators for deprecation, versioning, and compatibility.

        This class provides decorators that manage function lifecycle concerns
        including deprecation warnings, version compatibility, and API evolution
        support.

        All decorators in this class follow these principles:
            - Single Responsibility: Focus only on lifecycle concerns
            - Backward Compatibility: Maintain ABI compatibility during transitions
            - Clear Communication: Provide actionable deprecation messages
            - Version Awareness: Support semantic versioning patterns

        Examples:
            Deprecation with migration guidance::

                @FlextDecorators.Lifecycle.deprecated(
                    version="2.0.0",
                    reason="Use new_function() instead",
                    removal_version="3.0.0",
                )
                def old_function(x: int) -> int:
                    return x * 2

        """

        @staticmethod
        def deprecated(
            version: str | None = None,
            reason: str | None = None,
            removal_version: str | None = None,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Mark function as deprecated with migration guidance.

            Adds deprecation warnings to function calls with clear guidance
            for migration to newer APIs.

            Args:
                version: Version when function was deprecated
                reason: Reason for deprecation and migration guidance
                removal_version: Version when function will be removed

            Returns:
                Decorator that adds deprecation warnings

            Examples:
                Basic deprecation::

                    @FlextDecorators.Lifecycle.deprecated(
                        reason="Use process_data_v2() for better performance"
                    )
                    def process_data_v1(data: dict) -> dict:
                        return data

                Versioned deprecation::

                    @FlextDecorators.Lifecycle.deprecated(
                        version="1.5.0",
                        reason="Replaced by calculate_advanced()",
                        removal_version="2.0.0",
                    )
                    def calculate_simple(x: int) -> int:
                        return x + 1

            """

            def decorator(
                func: Callable[P, T],
            ) -> Callable[P, T]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    # Build deprecation message
                    message_parts = [f"Function {func.__name__} is deprecated"]

                    if version:
                        message_parts.append(f"since version {version}")

                    if reason:
                        message_parts.append(f": {reason}")

                    if removal_version:
                        message_parts.append(
                            f". Will be removed in version {removal_version}"
                        )

                    deprecation_message = "".join(message_parts)

                    # Issue warning
                    warnings.warn(deprecation_message, DeprecationWarning, stacklevel=2)

                    # Log deprecation usage
                    logger.warning(
                        f"Deprecated function called: {func.__name__}",
                        extra={
                            "function": func.__name__,
                            "function_module": func.__module__,
                            "deprecated_version": version,
                            "removal_version": removal_version,
                            "reason": reason,
                            "category": "deprecation",
                        },
                    )

                    return func(*args, **kwargs)

                return wrapper

            return decorator

    class Integration:
        """Integration decorators for cross-cutting decorator composition.

        This class provides factory methods and composition patterns for creating
        complex decorators that combine multiple concerns (validation + caching + logging)
        with proper integration of FlextResult patterns.

        All decorators in this class follow these principles:
            - Single Responsibility: Focus only on decorator composition
            - Composition Over Inheritance: Combine decorators functionally
            - Type Safety: Maintain type safety through composition
            - Configurability: Allow flexible decorator combinations

        Examples:
            Complete decorator composition::

                @FlextDecorators.Integration.create_enterprise_decorator(
                    with_validation=True,
                    with_caching=True,
                    with_monitoring=True,
                    with_logging=True,
                )
                def critical_business_operation(data: dict) -> dict:
                    return {"result": "processed", "data": data}

        """

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
            _log_level: str = FlextConstants.Observability.DEFAULT_LOG_LEVEL,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Create enterprise-grade decorator with multiple concerns.

            Composes multiple decorator concerns into a single decorator for
            enterprise applications requiring comprehensive function enhancement.

            Args:
                with_validation: Enable input validation
                validator: Validation function (required if with_validation=True)
                with_retry: Enable retry functionality
                max_retries: Maximum retry attempts
                with_timeout: Enable timeout protection
                timeout_seconds: Timeout duration
                with_caching: Enable result caching
                cache_size: Maximum cache size
                with_monitoring: Enable performance monitoring
                monitor_threshold: Slow operation threshold
                with_logging: Enable execution logging
                log_level: Logging level

            Returns:
                Composed decorator with all requested concerns

            Examples:
                Full enterprise decorator::

                    @FlextDecorators.Integration.create_enterprise_decorator(
                        with_validation=True,
                        validator=lambda x: isinstance(x, dict),
                        with_retry=True,
                        with_timeout=True,
                        with_caching=True,
                        with_monitoring=True,
                        with_logging=True,
                    )
                    def process_business_data(data: dict) -> dict:
                        return {"processed": True, "input": data}

                Minimal enterprise decorator::

                    @FlextDecorators.Integration.create_enterprise_decorator(
                        with_monitoring=True, with_logging=True
                    )
                    def simple_operation() -> str:
                        return "completed"

            """

            def decorator(
                func: Callable[P, T],
            ) -> Callable[P, T]:
                enhanced_func = func

                # Apply decorators in order (inside-out application)
                if with_logging:
                    enhanced_func = FlextDecorators.Observability.log_execution()(
                        enhanced_func
                    )

                if with_monitoring:
                    enhanced_func = FlextDecorators.Performance.monitor(
                        threshold=monitor_threshold
                    )(enhanced_func)

                if with_caching:
                    enhanced_func = FlextDecorators.Performance.cache(
                        max_size=cache_size
                    )(enhanced_func)

                if with_timeout:
                    enhanced_func = FlextDecorators.Reliability.timeout(
                        seconds=timeout_seconds
                    )(enhanced_func)

                if with_retry:
                    enhanced_func = FlextDecorators.Reliability.retry(
                        max_attempts=max_retries
                    )(enhanced_func)

                if with_validation and validator:
                    enhanced_func = FlextDecorators.Validation.validate_input(
                        validator=validator
                    )(enhanced_func)

                return enhanced_func

            return decorator


# =============================================================================
# EXPORTS - Hierarchical decorator system
# =============================================================================

__all__ = [
    "FlextDecorators",
]
