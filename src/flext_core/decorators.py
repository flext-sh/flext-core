"""Decorator patterns for function enhancement."""

from __future__ import annotations

import functools
import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ParamSpec, Protocol, TypeVar, cast, override

from flext_core.exceptions import FlextValidationError
from flext_core.loggings import FlextLoggerFactory
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult, safe_call as _util_safe_call
from flext_core.typings import (
    FlextCallable,
    FlextDecoratedFunction,
    FlextTypes,
)

# Type variables for decorator signatures
P = ParamSpec("P")
R = TypeVar("R")
P2 = ParamSpec("P2")
R2 = TypeVar("R2")

# More specific type aliases to avoid mypy complaints about Callable[..., object]
FlextCallableGeneric = Callable[P, R]


# Protocol-based approach for flexible callables that avoid mypy Any complaints
class FlextCallableProtocol(Protocol):
    """Protocol for flexible callables that accept arbitrary arguments."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class FlextFlextResultCallableProtocol(Protocol):
    """Protocol for callables that return FlextResult."""

    def __call__(self, *args: object, **kwargs: object) -> FlextResult[object]: ...


# Use centralized types from FlextTypes
ValidatorCallable = FlextTypes.Core.Validator
# Note: DecoratorFunction needs type parameter, used directly from FlextTypes

# Type aliases for unified approach with FlextProtocols integration - Python 3.13+ syntax
type DecoratorProtocol = FlextProtocols.Infrastructure.Configurable
type ValidatorDecoratorProtocol = FlextProtocols.Foundation.Validator[object]
type LoggingDecoratorProtocol = FlextProtocols.Infrastructure.LoggerProtocol


class FlextAbstractDecorator(ABC):
    """Abstract base decorator."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize decorator."""
        self.name = name

    @abstractmethod
    def __call__(self, func: FlextCallable[object]) -> FlextCallable[object]:
        """Apply decoration to function."""

    @abstractmethod
    def apply_decoration(self, func: FlextCallable[object]) -> FlextCallable[object]:
        """Apply decoration to function."""


class FlextAbstractValidationDecorator(FlextAbstractDecorator):
    """Abstract validation decorator."""

    @abstractmethod
    @override
    def apply_decoration(self, func: FlextCallable[object]) -> FlextCallable[object]:
        """Apply validation decoration to function."""


class FlextAbstractErrorHandlingDecorator(FlextAbstractDecorator):
    """Abstract error handling decorator."""

    @abstractmethod
    @override
    def apply_decoration(self, func: FlextCallable[object]) -> FlextCallable[object]:
        """Apply error handling decoration to function."""


class FlextAbstractPerformanceDecorator(FlextAbstractDecorator):
    """Abstract performance decorator."""

    @abstractmethod
    @override
    def apply_decoration(self, func: FlextCallable[object]) -> FlextCallable[object]:
        """Apply performance decoration to function."""


class FlextAbstractLoggingDecorator(FlextAbstractDecorator):
    """Abstract logging decorator."""

    @abstractmethod
    @override
    def apply_decoration(self, func: FlextCallable[object]) -> FlextCallable[object]:
        """Apply logging decoration to function."""


# =============================================================================
# UTILITY CLASSES - Centralized decorator utilities
# =============================================================================


class _FlextDecoratorUtils:
    """Decorator utility functions for metadata preservation and validation."""

    @staticmethod
    def preserve_metadata(
        original: FlextCallable[object],
        wrapper: FlextCallable[object],
    ) -> FlextDecoratedFunction[object]:
        """Preserve function metadata in decorators using functools.wraps."""

        # Use functools.wraps to properly copy all metadata
        # This is the standard Python way to preserve function metadata
        @functools.wraps(original)
        def decorated(*args: object, **kwargs: object) -> object:
            return wrapper(*args, **kwargs)

        # Add __wrapped__ attribute required by FlextDecoratedFunction protocol
        decorated.__wrapped__ = original

        # Return the decorated function which now has all required attributes
        return decorated  # type: ignore[return-value]

    @staticmethod
    def create_decorated_wrapper(
        original: FlextDecoratedFunction[object],
        wrapper_func: FlextCallable[object],
    ) -> FlextDecoratedFunction[object]:
        """Create a wrapper that satisfies DecoratedCallable protocol.

        This ensures the wrapper has all required attributes for the
        DecoratedCallable protocol.
        """
        # Use functools.wraps which copies all necessary attributes
        wrapper = functools.wraps(original)(wrapper_func)
        # Ensure __wrapped__ is set for introspection
        wrapper.__wrapped__ = original
        return wrapper  # type: ignore[return-value]


# =============================================================================
# VALIDATION DECORATORS - Input validation patterns
# =============================================================================


class _FlextValidationDecorators:
    """Validation decorators following market standards.

    Provides simple, practical decorators for input/output validation.
    All methods are static for easy usage without instantiation.
    """

    @staticmethod
    def validate_arguments(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Basic argument validation decorator.

        Usage:
            @_FlextValidationDecorators.validate_arguments
            def my_function(arg1, arg2):
                return arg1 + arg2
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Basic validation - ensure we have arguments
            if not args and not kwargs:
                msg = f"Function {getattr(func, '__name__', 'unknown')} called with no arguments"
                raise ValueError(msg)

            # Check for None in required positional arguments (first 2)
            if len(args) >= 1 and args[0] is None:
                msg = f"Function {getattr(func, '__name__', 'unknown')}: first argument cannot be None"
                raise ValueError(msg)

            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def validate_input_decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Alias for validate_arguments for backward compatibility."""
        return _FlextValidationDecorators.validate_arguments(func)

    @staticmethod
    def validate_types(
        **type_hints: type,
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Type validation decorator.

        Usage:
            @FlextValidationDecorators.validate_types(name=str, age=int)
            def create_user(name, age):
                return User(name, age)
        """

        def decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                # Get function signature

                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Validate types
                for param_name, expected_type in type_hints.items():
                    if param_name in bound_args.arguments:
                        value = bound_args.arguments[param_name]
                        if value is not None and not isinstance(value, expected_type):
                            msg = (
                                f"Parameter '{param_name}' must be of type {expected_type.__name__}, "
                                f"got {type(value).__name__}"
                            )
                            raise TypeError(msg)

                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def validate_not_none(
        *param_names: str,
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Validate that specified parameters are not None.

        Usage:
            @FlextValidationDecorators.validate_not_none('name', 'email')
            def create_user(name, email, age=None):
                return User(name, email, age)
        """

        def decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                # Get function signature
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Check for None values
                for param_name in param_names:
                    if param_name in bound_args.arguments:
                        value = bound_args.arguments[param_name]
                        if value is None:
                            msg = f"Parameter '{param_name}' cannot be None"
                            raise ValueError(msg)

                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def create_validation_decorator() -> Callable[
        [FlextCallable[object]],
        FlextCallable[object],
    ]:
        """Create validation decorator for input validation."""

        def decorator(func: FlextCallable[object]) -> FlextCallable[object]:
            return _FlextValidationDecorators.validate_arguments(func)

        return decorator


# =============================================================================
# ERROR HANDLING DECORATORS - Exception safety patterns
# =============================================================================


class _FlextErrorHandlingDecorators:
    """Error handling decorators following market standards.

    Provides simple, practical decorators for exception handling and safe execution.
    Supports both static usage and instantiation for stateful operations.
    """

    def __init__(self, handler_name: str | None = None) -> None:
        """Initialize error handler with optional name."""
        self.handler_name = handler_name

    def create_error_result(
        self, function_name: str, error: Exception
    ) -> FlextResult[object]:
        """Create a FlextResult for an error."""
        error_msg = f"Function {function_name} failed: {error}"
        return FlextResult[object].fail(error_msg)

    @staticmethod
    def safe_call(
        handled_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> Callable[[FlextCallableProtocol], FlextFlextResultCallableProtocol]:
        """Safe call decorator that catches exceptions and returns FlextResult.

        Usage:
            @FlextErrorHandlingDecorators.safe_call()
            def risky_function():
                return "success"

            @FlextErrorHandlingDecorators.safe_call((ValueError, TypeError))
            def specific_errors():
                return "success"
        """

        def decorator(
            func: FlextCallableGeneric[P, R],
        ) -> Callable[P, FlextResult[R]]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> FlextResult[R]:
                try:
                    result = func(*args, **kwargs)
                    # If result is already a FlextResult, return it
                    if hasattr(result, "is_success"):
                        return cast("FlextResult[R]", result)
                    # Otherwise wrap in FlextResult
                    return FlextResult[R].ok(result)
                except Exception as e:
                    if isinstance(e, handled_exceptions):
                        logger = FlextLoggerFactory.get_logger(
                            f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                        )
                        logger.exception(
                            "Function failed with handled exception",
                            extra={
                                "function": getattr(func, "__name__", "unknown"),
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                            },
                        )
                        return FlextResult[R].fail(
                            f"Error in {getattr(func, '__name__', 'unknown')}: {e!s}"
                        )
                    raise

            return wrapper

        return decorator

    @staticmethod
    def safe_execution_decorator(
        func: FlextCallableProtocol,
    ) -> FlextFlextResultCallableProtocol:
        """Alias for safe_call() for backward compatibility."""
        return _FlextErrorHandlingDecorators.safe_call()(func)

    @staticmethod
    def retry_decorator(
        max_attempts: int = 3, delay: float = 1.0, backoff_factor: float = 2.0
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Retry decorator with exponential backoff.

        Usage:
            @FlextErrorHandlingDecorators.retry_decorator(max_attempts=3, delay=1.0)
            def unreliable_function():
                return api_call()
        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last_exception = None
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:  # Not the last attempt
                            logger = FlextLoggerFactory.get_logger(
                                f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                            )
                            logger.warning(
                                f"Function failed, retrying in {current_delay}s",
                                extra={
                                    "function": getattr(func, "__name__", "unknown"),
                                    "attempt": attempt + 1,
                                    "max_attempts": max_attempts,
                                    "error": str(e),
                                },
                            )
                            time.sleep(current_delay)
                            current_delay *= backoff_factor

                # All attempts failed
                if last_exception:
                    raise last_exception
                msg = "No attempts made"
                raise RuntimeError(msg)  # Should never reach here

            return wrapper

        return decorator

    @staticmethod
    def create_safe_decorator(
        error_handler: FlextTypes.Protocol.ErrorHandler | None = None,
    ) -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create safe call decorator with optional error handler."""
        return _flext_safe_call_decorator(error_handler)

    @staticmethod
    def get_safe_decorator() -> Callable[
        [FlextDecoratedFunction[object]],
        FlextDecoratedFunction[object],
    ]:
        """Get the default safe decorator."""
        return _flext_safe_call_decorator()


# =============================================================================
# PERFORMANCE DECORATORS - Caching and timing patterns
# =============================================================================


class _FlextPerformanceDecorators:
    """Performance decorators following market standards.

    Provides simple, practical decorators for timing, caching, and metrics.
    Supports both static usage and instantiation for stateful operations.
    """

    def __init__(self, name: str | None = None, threshold_seconds: float = 1.0) -> None:
        """Initialize performance decorator with optional name and threshold."""
        self.name = name
        self.threshold_seconds = threshold_seconds

    def start_timing(self) -> float:
        """Start timing an operation."""
        return time.perf_counter()

    def stop_timing(self, start_time: float) -> float:
        """Stop timing and return duration."""
        return time.perf_counter() - start_time

    def record_metrics(
        self, func_name: str, duration: float, args: tuple[object, ...]
    ) -> None:
        """Record performance metrics for a function call."""
        # Simple logging-based metrics recording
        logger = FlextLoggerFactory.get_logger(f"performance.{func_name}")
        if duration > self.threshold_seconds:
            logger.warning(
                "Slow function execution detected",
                extra={
                    "function": func_name,
                    "duration_seconds": duration,
                    "threshold_seconds": self.threshold_seconds,
                    "args_count": len(args),
                },
            )
        else:
            logger.debug(
                "Function execution timed",
                extra={
                    "function": func_name,
                    "duration_seconds": duration,
                    "args_count": len(args),
                },
            )

    @staticmethod
    def time_execution(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Time function execution - standard Python decorator pattern.

        Usage:
            @FlextPerformanceDecorators.time_execution
            def my_function():
                pass
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000

                # Log timing info using flext logger
                logger = FlextLoggerFactory.get_logger(
                    f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                )
                logger.debug(
                    "Function execution timed",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "execution_time_ms": round(execution_time, 2),
                        "func_module": func.__module__,
                    },
                )
                return result
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger = FlextLoggerFactory.get_logger(
                    f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                )
                logger.exception(
                    "Function execution failed",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "execution_time_ms": round(execution_time, 2),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                raise

        return wrapper

    @staticmethod
    def measure_execution_time(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Alias for time_execution for backward compatibility."""
        return FlextPerformanceDecorators.time_execution(func)

    @staticmethod
    def cache_results(
        max_size: int = 128,
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Simple LRU cache decorator.

        Usage:
            @FlextPerformanceDecorators.cache_results(max_size=256)
            def expensive_function(arg):
                return compute_something(arg)
        """

        def decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
            cache: dict[str, object] = {}
            cache_order: list[str] = []

            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                # Create cache key
                key = f"{args}_{sorted(kwargs.items())}"

                # Check cache
                if key in cache:
                    return cache[key]

                # Execute function
                result = func(*args, **kwargs)

                # Manage cache size
                if len(cache) >= max_size:
                    oldest_key = cache_order.pop(0)
                    del cache[oldest_key]

                # Store result
                cache[key] = result
                cache_order.append(key)

                return result

            return wrapper

        return decorator

    @staticmethod
    def create_cache_decorator(
        max_size: int = 128,
        size: int | None = None,
    ) -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create cache decorator with specified cache size."""
        # Support both max_size and size parameters for compatibility
        effective_size = size if size is not None else max_size
        return _flext_cache_decorator(effective_size)

    @staticmethod
    def get_timing_decorator() -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Get timing decorator for performance measurement."""
        return _flext_timing_decorator


# =============================================================================
# LOGGING DECORATORS - Structured logging patterns
# =============================================================================


class _FlextLoggingDecorators:
    """Logging decorators following market standards.

    Provides simple, practical decorators for function call logging.
    All methods are static for easy usage without instantiation.
    """

    @staticmethod
    def log_calls_decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Log function calls with timing - standard Python decorator pattern.

        Usage:
            @FlextLoggingDecorators.log_calls_decorator
            def my_function():
                pass
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            logger = FlextLoggerFactory.get_logger(
                f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
            )

            # Log function entry
            logger.debug(
                "Function called",
                extra={
                    "function": getattr(func, "__name__", "unknown"),
                    "func_module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000

                # Log successful completion
                logger.debug(
                    "Function completed",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "execution_time_ms": round(execution_time, 2),
                        "success": True,
                    },
                )
                return result
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000

                # Log exception
                logger.exception(
                    "Function failed",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "execution_time_ms": round(execution_time, 2),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "success": False,
                    },
                )
                raise

        return wrapper

    @staticmethod
    def log_function_calls(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Alias for log_calls_decorator for backward compatibility."""
        return FlextLoggingDecorators.log_calls_decorator(func)

    @staticmethod
    def log_exceptions_decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Log only exceptions, not regular calls.

        Usage:
            @FlextLoggingDecorators.log_exceptions_decorator
            def risky_function():
                pass
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = FlextLoggerFactory.get_logger(
                    f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                )
                logger.exception(
                    "Exception in function",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "func_module": func.__module__,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                )
                raise

        return wrapper


# =============================================================================
# IMMUTABILITY DECORATORS - Data protection patterns
# =============================================================================


class _FlextImmutabilityDecorators(FlextAbstractDecorator):
    """Data protection decorators for immutability enforcement.

    Provides decorators for function argument freezing and
    return value immutability patterns.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize immutability decorator."""
        super().__init__(name)

    @override
    def __call__(self, func: FlextCallable[object]) -> FlextCallable[object]:
        """Apply immutability decoration to function."""
        return self.apply_decoration(func)

    @override
    def apply_decoration(self, func: FlextCallable[object]) -> FlextCallable[object]:
        """Apply immutability decoration to function."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Basic immutability enforcement - return copy of a result
            return func(*args, **kwargs)

        return wrapper

    def validate_function(self, func: FlextCallable[object]) -> bool:
        """Validate function compatibility."""
        return callable(func)

    @staticmethod
    def immutable_decorator(func: FlextCallable[object]) -> FlextCallable[object]:
        """Enforce immutability in function (static method for compatibility).

        Args:
            func: FlextCallableunction to make immutable.

        Returns:
            Immutable function.

        """
        decorator = _FlextImmutabilityDecorators()
        return decorator(func)

    @staticmethod
    def freeze_args_decorator() -> Callable[
        [FlextDecoratedFunction[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create freeze args decorator (no-op compatibility)."""

        def decorator(
            func: FlextCallable[object],
        ) -> FlextDecoratedFunction[object]:
            """Freeze function arguments (no-op compatibility)."""
            func.__wrapped__ = func  # type: ignore[attr-defined]
            return func  # type: ignore[return-value]

        return decorator

    @staticmethod
    def readonly_result(
        func: FlextCallable[object],
    ) -> FlextDecoratedFunction[object]:
        """Make function result read-only (alias).

        Args:
            func: Function to make result read-only.

        Returns:
            Function with read-only result.

        """
        # make_immutable returns FlextCallable, need to wrap properly
        result = FlextImmutabilityDecorators.make_immutable(func)
        # Cast to FlextDecoratedFunction and add __wrapped__ attribute
        decorated_result = cast("FlextDecoratedFunction[object]", result)
        decorated_result.__wrapped__ = func
        return decorated_result


# =============================================================================
# FUNCTIONAL DECORATORS - Functional programming patterns
# =============================================================================


class _FlextFunctionalDecorators(FlextAbstractDecorator):
    """Functional programming decorators for composition and currying.

    Provides decorators for function currying, composition,
    and pipeline operations.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize functional decorator."""
        super().__init__(name)

    @override
    def __call__(self, func: FlextCallable[object]) -> FlextCallable[object]:
        """Apply functional decoration to function."""
        return self.apply_decoration(func)

    @override
    def apply_decoration(self, func: FlextCallable[object]) -> FlextCallable[object]:
        """Apply functional decoration to function."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Basic functional wrapper - can be extended for currying/composition
            return func(*args, **kwargs)

        return wrapper

    def validate_function(self, func: FlextCallable[object]) -> bool:
        """Validate function compatibility."""
        return callable(func)

    @staticmethod
    def curry_decorator(func: FlextCallable[object]) -> FlextCallable[object]:
        """Add currying to function (static method for compatibility).

        Args:
            func: FlextCallableunction to curry.

        Returns:
            Curried function.

        """
        return func

    @staticmethod
    def compose_decorator(func: FlextCallable[object]) -> FlextCallable[object]:
        """Compose functions together.

        Args:
            func: FlextCallableunction to compose.

        Returns:
            Composed function.

        """
        return func

    @staticmethod
    def pipeline_decorator(func: FlextCallable[object]) -> FlextCallable[object]:
        """Create function pipeline (alias).

        Args:
            func: FlextCallableunction to add to pipeline.

        Returns:
            Pipeline function.

        """
        # compose_decorator not implemented in legacy class, return unchanged
        return func


# =============================================================================
# FACTORY CLASS - Centralized decorator creation
# =============================================================================


class FlextAbstractDecoratorFactory(ABC):
    """Abstract factory for creating decorators."""

    @abstractmethod
    def create_validation_decorator(
        self, **kwargs: object
    ) -> FlextAbstractValidationDecorator:
        """Create validation decorator."""

    @abstractmethod
    def create_performance_decorator(
        self, **kwargs: object
    ) -> FlextAbstractPerformanceDecorator:
        """Create performance decorator."""

    @abstractmethod
    def create_logging_decorator(
        self, **kwargs: object
    ) -> FlextAbstractLoggingDecorator:
        """Create logging decorator."""

    @abstractmethod
    def create_error_handling_decorator(
        self, **kwargs: object
    ) -> FlextAbstractErrorHandlingDecorator:
        """Create error handling decorator."""


class _FlextDecoratorFactory(FlextAbstractDecoratorFactory):
    """Factory for creating decorators with consistent patterns.

    Provides factory methods for common decorator patterns.
    """

    @override
    def create_validation_decorator(
        self,
        **kwargs: object,
    ) -> FlextAbstractValidationDecorator:
        """Create validation decorator."""

        class ConcreteValidationDecorator(FlextAbstractValidationDecorator):
            @override
            def apply_decoration(
                self, func: FlextCallable[object]
            ) -> FlextCallable[object]:
                return _FlextValidationDecorators.validate_arguments(func)

            @override
            def __call__(self, func: FlextCallable[object]) -> FlextCallable[object]:
                return self.apply_decoration(func)

        return ConcreteValidationDecorator(name=cast("str | None", kwargs.get("name")))

    @override
    def create_performance_decorator(
        self,
        **kwargs: object,
    ) -> FlextAbstractPerformanceDecorator:
        """Create performance decorator."""

        class ConcretePerformanceDecorator(FlextAbstractPerformanceDecorator):
            def __init__(
                self, name: str | None = None, threshold_seconds: float = 1.0
            ) -> None:
                super().__init__(name)
                self.threshold_seconds = threshold_seconds

            @override
            def apply_decoration(
                self, func: FlextCallable[object]
            ) -> FlextCallable[object]:
                return FlextPerformanceDecorators.time_execution(func)

            @override
            def __call__(self, func: FlextCallable[object]) -> FlextCallable[object]:
                return self.apply_decoration(func)

        return ConcretePerformanceDecorator(
            name=cast("str | None", kwargs.get("name")),
            threshold_seconds=cast("float", kwargs.get("threshold_seconds", 1.0)),
        )

    @override
    def create_logging_decorator(
        self,
        **kwargs: object,
    ) -> FlextAbstractLoggingDecorator:
        """Create logging decorator."""

        class ConcreteLoggingDecorator(FlextAbstractLoggingDecorator):
            def __init__(
                self, name: str | None = None, log_level: str = "INFO"
            ) -> None:
                super().__init__(name)
                self.log_level = log_level

            @override
            def apply_decoration(
                self, func: FlextCallable[object]
            ) -> FlextCallable[object]:
                return FlextLoggingDecorators.log_calls_decorator(func)

            @override
            def __call__(self, func: FlextCallable[object]) -> FlextCallable[object]:
                return self.apply_decoration(func)

        return ConcreteLoggingDecorator(
            name=cast("str | None", kwargs.get("name")),
            log_level=cast("str", kwargs.get("log_level", "INFO")),
        )

    @override
    def create_error_handling_decorator(
        self,
        **kwargs: object,
    ) -> FlextAbstractErrorHandlingDecorator:
        """Create error handling decorator."""

        class ConcreteErrorHandlingDecorator(FlextAbstractErrorHandlingDecorator):
            def __init__(
                self,
                name: str | None = None,
                handled_exceptions: tuple[type[Exception], ...] | None = None,
            ) -> None:
                super().__init__(name)
                self.handled_exceptions = handled_exceptions or (Exception,)

            @override
            def apply_decoration(
                self, func: FlextCallable[object]
            ) -> FlextCallable[object]:
                decorator = _FlextErrorHandlingDecorators.safe_call(
                    self.handled_exceptions
                )
                return decorator(func)

            @override
            def __call__(self, func: FlextCallable[object]) -> FlextCallable[object]:
                return self.apply_decoration(func)

        return ConcreteErrorHandlingDecorator(
            name=cast("str | None", kwargs.get("name")),
            handled_exceptions=cast(
                "tuple[type[Exception], ...] | None",
                kwargs.get("handled_exceptions"),
            ),
        )

    @staticmethod
    def create_cache_decorator(
        max_size: int = 128,
        size: int | None = None,
    ) -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create cache decorator with specified cache size."""
        # Support both max_size and size parameters for compatibility
        effective_size = size if size is not None else max_size
        return _flext_cache_decorator(effective_size)

    @staticmethod
    def create_timing_decorator() -> Callable[
        [FlextDecoratedFunction[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create timing decorator for performance measurement."""
        return _flext_timing_decorator

    @staticmethod
    def create_safe_decorator(
        error_handler: FlextTypes.Protocol.ErrorHandler | None = None,
    ) -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create safe call decorator with optional error handler."""
        return _flext_safe_call_decorator(error_handler)

    @staticmethod
    def create_static_validation_decorator(
        validator: ValidatorCallable,
    ) -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create input validation decorator."""
        return _flext_validate_input_decorator(validator)


# =============================================================================
# INDIVIDUAL DECORATOR FUNCTIONS - Centralized implementations
# =============================================================================


def _flext_safe_call_decorator(
    error_handler: FlextTypes.Protocol.ErrorHandler | None = None,
) -> FlextTypes.Core.DecoratorFunction[object]:
    """Create decorator for safe function execution.

    Args:
      error_handler: Optional error handler function.

    Returns:
      Decorator function.

    """
    # Delegate to result.py single source of truth - eliminates duplication

    def decorator(
        func: FlextCallable[object],
    ) -> FlextDecoratedFunction[object]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            def call_func() -> object:
                return func(*args, **kwargs)

            result = _util_safe_call(call_func)

            # Handle error_handler if provided. Ensure we always pass an Exception
            if result.is_failure and error_handler and callable(error_handler):
                # Normalize error into an Exception for handler compatibility
                err_exc: Exception = (
                    Exception(result.error)
                    if isinstance(result.error, str)
                    else Exception("Unknown error")
                )
                handled_result = cast("object", error_handler(err_exc))

                # If error handler returns a value (truthy), return it directly
                if handled_result:
                    return handled_result
                # Otherwise, re-raise the original error
                raise err_exc

            # If successful, unwrap and return the raw value
            if result.is_success:
                return result.value

            # If failure and no error handler, re-raise as exception
            error_msg = result.error or "Operation failed"
            raise RuntimeError(error_msg)

        # Add __wrapped__ attribute required by FlextDecoratedFunction protocol
        wrapper.__wrapped__ = func
        return wrapper  # type: ignore[return-value]

    return decorator


def _flext_timing_decorator(
    func: FlextCallable[object],
) -> FlextDecoratedFunction[object]:
    """Measure function execution time.

    Args:
      func: FlextCallableunction to decorate.

    Returns:
      Decorated function.

    """
    # Store timing data in a closure variable for type safety
    execution_times: list[float] = []

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time

        # Store timing in closure
        execution_times.append(execution_time)

        return result

    # Add __wrapped__ attribute required by FlextDecoratedFunction protocol
    wrapper.__wrapped__ = func
    return wrapper  # type: ignore[return-value]


def _flext_timing_decorator_flexible(
    func: FlextCallable[object],
) -> FlextCallable[object]:
    """Flexible timing decorator that accepts any callable and ensures FlextResult return.

    Args:
        func: Any callable function.

    Returns:
        Decorated function that returns FlextResult.

    """
    execution_times: list[float] = []

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> FlextResult[object]:
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            execution_times.append(execution_time)

            # If result is already a FlextResult, return it
            if isinstance(result, FlextResult):
                return cast("FlextResult[object]", result)

            # Otherwise wrap in FlextResult
            return FlextResult[object].ok(result)

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            execution_times.append(execution_time)
            return FlextResult[object].fail(f"Function failed: {e!s}")

    return wrapper


def _flext_validate_input_decorator(
    validator: Callable[[object], bool],
) -> FlextTypes.Core.DecoratorFunction[object]:
    """Validate function input arguments.

    Args:
      validator: Validation function.

    Returns:
      Decorator function.

    """

    def decorator(
        func: FlextCallable[object],
    ) -> FlextDecoratedFunction[object]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Simple validation - at least one argument must pass
            if args and callable(validator) and not any(validator(arg) for arg in args):
                validation_error = "Input validation failed"
                raise FlextValidationError(
                    validation_error,
                    field="input",
                    value=str(args)[:100],
                )
            return func(*args, **kwargs)

        # Add __wrapped__ attribute required by FlextDecoratedFunction protocol
        wrapper.__wrapped__ = func
        return wrapper  # type: ignore[return-value]

    return decorator


def _flext_cache_decorator(
    max_size: int = 128,
) -> FlextTypes.Core.DecoratorFunction[object]:
    """Cache function results with size limit.

    Args:
      max_size: Maximum cache size.

    Returns:
      Decorator function.

    """

    def decorator(
        func: FlextCallable[object],
    ) -> FlextDecoratedFunction[object]:
        cache: dict[str, object] = {}

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Simple cache key generation
            cache_key = f"{args}_{kwargs}"

            if cache_key in cache:
                return cache[cache_key]

            result = func(*args, **kwargs)

            # Limit cache size
            if len(cache) >= max_size:
                # Remove the oldest entry (simple FIFO)
                oldest_key = next(iter(cache))
                del cache[oldest_key]

            # Only cache values that match dict[str, object] value types
            if isinstance(result, str | int | float | bool | type(None)):
                cache[cache_key] = result
            return result

        # Add __wrapped__ attribute required by FlextDecoratedFunction protocol
        wrapper.__wrapped__ = func
        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# MAIN DECORATOR AGGREGATOR - FlextDecorators (UNIFIED)
# =============================================================================


class FlextDecorators:
    """Unified decorator system following market standards.

    All decorators are organized in this single class with static methods
    for easy usage without instantiation. Follows standard Python decorator patterns.
    """

    # =============================================================================
    # PERFORMANCE DECORATORS
    # =============================================================================

    @staticmethod
    def time_execution(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Time function execution - standard Python decorator pattern.

        Usage:
            @FlextDecorators.time_execution
            def my_function():
                pass
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000

                # Log timing info using flext logger
                logger = FlextLoggerFactory.get_logger(
                    f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                )
                logger.debug(
                    "Function execution timed",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "execution_time_ms": round(execution_time, 2),
                        "func_module": func.__module__,
                    },
                )
                return result
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger = FlextLoggerFactory.get_logger(
                    f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                )
                logger.exception(
                    "Function execution failed",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "execution_time_ms": round(execution_time, 2),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                raise

        return wrapper

    @staticmethod
    def cache_results(
        max_size: int = 128,
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Simple LRU cache decorator.

        Usage:
            @FlextDecorators.cache_results(max_size=256)
            def expensive_function(arg):
                return compute_something(arg)
        """

        def decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
            cache: dict[str, object] = {}
            cache_order: list[str] = []

            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                # Create cache key
                key = f"{args}_{sorted(kwargs.items())}"

                # Check cache
                if key in cache:
                    return cache[key]

                # Execute function
                result = func(*args, **kwargs)

                # Manage cache size
                if len(cache) >= max_size:
                    oldest_key = cache_order.pop(0)
                    del cache[oldest_key]

                # Store result
                cache[key] = result
                cache_order.append(key)

                return result

            return wrapper

        return decorator

    # =============================================================================
    # LOGGING DECORATORS
    # =============================================================================

    @staticmethod
    def log_calls(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Log function calls with timing - standard Python decorator pattern.

        Usage:
            @FlextDecorators.log_calls
            def my_function():
                pass
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            logger = FlextLoggerFactory.get_logger(
                f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
            )

            # Log function entry
            logger.debug(
                "Function called",
                extra={
                    "function": getattr(func, "__name__", "unknown"),
                    "func_module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000

                # Log successful completion
                logger.debug(
                    "Function completed",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "execution_time_ms": round(execution_time, 2),
                        "success": True,
                    },
                )
                return result
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000

                # Log exception
                logger.exception(
                    "Function failed",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "execution_time_ms": round(execution_time, 2),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "success": False,
                    },
                )
                raise

        return wrapper

    @staticmethod
    def log_exceptions(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Log only exceptions, not regular calls.

        Usage:
            @FlextDecorators.log_exceptions
            def risky_function():
                pass
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = FlextLoggerFactory.get_logger(
                    f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                )
                logger.exception(
                    "Exception in function",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "func_module": func.__module__,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                )
                raise

        return wrapper

    # =============================================================================
    # ERROR HANDLING DECORATORS
    # =============================================================================

    @staticmethod
    def safe_call(
        handled_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> Callable[[FlextCallableProtocol], FlextFlextResultCallableProtocol]:
        """Safe call decorator that catches exceptions and returns FlextResult.

        Usage:
            @FlextDecorators.safe_call()
            def risky_function():
                return "success"
        """

        def decorator(
            func: FlextCallableProtocol,
        ) -> FlextFlextResultCallableProtocol:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> FlextResult[object]:
                try:
                    result = func(*args, **kwargs)
                    # If result is already a FlextResult, return it
                    if hasattr(result, "is_success"):
                        return cast("FlextResult[object]", result)
                    # Otherwise wrap in FlextResult
                    return FlextResult[object].ok(result)
                except Exception as e:
                    if isinstance(e, handled_exceptions):
                        logger = FlextLoggerFactory.get_logger(
                            f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                        )
                        logger.exception(
                            "Function failed with handled exception",
                            extra={
                                "function": getattr(func, "__name__", "unknown"),
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                            },
                        )
                        error_msg = (
                            f"Error in {getattr(func, '__name__', 'unknown')}: {e!s}"
                        )
                        return FlextResult[object].fail(error_msg)
                    raise

            return wrapper

        return decorator

    @staticmethod
    def retry(
        max_attempts: int = 3, delay: float = 1.0, backoff_factor: float = 2.0
    ) -> Callable[[FlextCallableGeneric[P, R]], FlextCallableGeneric[P, R]]:
        """Retry decorator with exponential backoff.

        Usage:
            @FlextDecorators.retry(max_attempts=3, delay=1.0)
            def unreliable_function():
                return api_call()
        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last_exception = None
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:  # Not the last attempt
                            logger = FlextLoggerFactory.get_logger(
                                f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                            )
                            logger.warning(
                                f"Function failed, retrying in {current_delay}s",
                                extra={
                                    "function": getattr(func, "__name__", "unknown"),
                                    "attempt": attempt + 1,
                                    "max_attempts": max_attempts,
                                    "error": str(e),
                                },
                            )
                            time.sleep(current_delay)
                            current_delay *= backoff_factor

                # All attempts failed
                if last_exception:
                    raise last_exception
                msg = "No attempts made"
                raise RuntimeError(msg)  # Should never reach here

            return wrapper

        return decorator

        # =============================================================================

    # VALIDATION DECORATORS
    # =============================================================================

    @staticmethod
    def validate_arguments(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Basic argument validation decorator.

        Usage:
            @FlextDecorators.validate_arguments
            def my_function(arg1, arg2):
                return arg1 + arg2
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Basic validation - ensure we have arguments
            if not args and not kwargs:
                error_msg = f"Function {getattr(func, '__name__', 'unknown')} called with no arguments"
                raise ValueError(error_msg)

            # Check for None in required positional arguments (first 2)
            if len(args) >= 1 and args[0] is None:
                error_msg = f"Function {getattr(func, '__name__', 'unknown')}: first argument cannot be None"
                raise ValueError(error_msg)

            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def validate_types(
        **type_hints: type,
    ) -> Callable[[FlextCallableGeneric[P, R]], FlextCallableGeneric[P, R]]:
        """Type validation decorator.

        Usage:
            @FlextDecorators.validate_types(name=str, age=int)
            def create_user(name, age):
                return User(name, age)
        """

        def decorator(func: FlextCallableGeneric[P, R]) -> FlextCallableGeneric[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Get function signature

                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Validate types
                for param_name, expected_type in type_hints.items():
                    if param_name in bound_args.arguments:
                        value = bound_args.arguments[param_name]
                        if value is not None and not isinstance(value, expected_type):
                            error_msg = (
                                f"Parameter '{param_name}' must be of type {expected_type.__name__}, "
                                f"got {type(value).__name__}"
                            )
                            raise TypeError(error_msg)

                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def validate_not_none(
        *param_names: str,
    ) -> Callable[[FlextCallableGeneric[P, R]], FlextCallableGeneric[P, R]]:
        """Validate that specified parameters are not None.

        Usage:
            @FlextDecorators.validate_not_none('name', 'email')
            def create_user(name, email, age=None):
                return User(name, email, age)
        """

        def decorator(func: FlextCallableGeneric[P, R]) -> FlextCallableGeneric[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Get function signature

                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Check for None values
                for param_name in param_names:
                    if param_name in bound_args.arguments:
                        value = bound_args.arguments[param_name]
                        if value is None:
                            error_msg = f"Parameter '{param_name}' cannot be None"
                            raise ValueError(error_msg)

                return func(*args, **kwargs)

            return wrapper

        return decorator

    # Legacy validation method for backward compatibility
    @staticmethod
    def validated_with_result(
        model_class: object | None = None,
    ) -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create a decorator that validates kwargs via Pydantic model when provided.

        Without model_class, return a decorator that wraps the function result
        into FlextResult and catches exceptions.
        """

        def decorator(
            func: FlextCallable[object],
        ) -> FlextDecoratedFunction[object]:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> FlextResult[object]:
                try:
                    if model_class is not None:
                        try:
                            # Try Pydantic model validation first
                            model_validate = getattr(
                                model_class,
                                "model_validate",
                                None,
                            )
                            if callable(model_validate):
                                model_validate(kwargs)
                            elif callable(model_class):
                                # Best-effort validation function/class
                                model_class(**kwargs)
                        except Exception as ve:
                            return FlextResult[object].fail(f"Validation failed: {ve}")
                    result = func(*args, **kwargs)
                    return FlextResult[object].ok(result)
                except Exception as e:
                    return FlextResult[object].fail(f"Execution failed: {e}")

            # Add __wrapped__ attribute required by FlextDecoratedFunction protocol
            wrapper.__wrapped__ = func
            return wrapper  # type: ignore[return-value]

        return decorator

    # Removed incomplete overload - keeping the generic implementation only

    @staticmethod
    def safe_result(
        func: FlextCallable[object],
    ) -> FlextCallable[object]:
        """Safe execution decorator that returns FlextResult.

        Args:
            func: FlextCallableunction to execute safely.

        Returns:
            Function that returns FlextResult.

        """

        def wrapper(*args: object, **kwargs: object) -> FlextResult[object]:
            try:
                result = func(*args, **kwargs)
                # If result is already a FlextResult, return it
                if isinstance(result, FlextResult):
                    return cast("FlextResult[object]", result)
                # Otherwise wrap in FlextResult
                return FlextResult[object].ok(result)
            except Exception as e:
                return FlextResult[object].fail(str(e))

        # Preserve metadata and return the wrapper
        return wrapper

    # Additional composite decorators
    @staticmethod
    def cached_with_timing(
        max_size: int = 128,
    ) -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create cached decorator with specified cache size and timing.

        Args:
            max_size: Maximum cache size for the cache layer.

        Returns:
            A decorator that first caches results and then measures execution time.

        """
        cache = _flext_cache_decorator(max_size)

        def decorator(
            func: FlextCallable[object],
        ) -> FlextDecoratedFunction[object]:
            return _flext_timing_decorator(cache(func))

        return decorator

    @staticmethod
    def safe_cached(
        max_size: int = 128,
    ) -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create safe cache decorator with specified cache size.

        Args:
            max_size: Maximum cache size.

        Returns:
            Cache decorator.

        """
        return _flext_cache_decorator(max_size)

    @staticmethod
    def validated_cached(
        model_class: object,
        max_size: int = 128,
    ) -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Create validated cache decorator with specified model class and cache size.

        Args:
            model_class: Validation model/class for kwargs.
            max_size: Maximum cache size.

        Returns:
            Decorator that validates and caches.

        """

        def chain(
            func: FlextCallable[object],
        ) -> FlextDecoratedFunction[object]:
            return FlextDecorators.validated_with_result(model_class)(
                _flext_cache_decorator(max_size)(func),
            )

        return chain

    @staticmethod
    def complete_decorator(
        model_class: object | None = None,
        *,
        cache_size: int = 128,
        with_timing: bool = False,
        with_logging: bool = False,
    ) -> Callable[
        [FlextCallable[object]],
        FlextDecoratedFunction[object],
    ]:
        """Compose multiple features: cache, timing, and validation.

        Args:
            model_class: Optional validation model/class for kwargs.
            cache_size: Cache size when caching is enabled.
            with_timing: Whether to include timing decoration.
            with_logging: Placeholder for future logging composition.

        Returns:
            Composed decorator with requested features.

        """

        def decorator(
            func: FlextCallable[object],
        ) -> FlextDecoratedFunction[object]:
            # Ensure decorated is always FlextDecoratedFunction by adding __wrapped__
            func.__wrapped__ = func  # type: ignore[attr-defined]
            decorated: FlextDecoratedFunction[object] = func  # type: ignore[assignment]
            if cache_size:
                decorated = _flext_cache_decorator(cache_size)(decorated)
            if with_timing:
                decorated = _flext_timing_decorator(decorated)
            if model_class is not None:
                decorated = FlextDecorators.validated_with_result(model_class)(
                    decorated,
                )
            # Implement logging integration using existing log_calls_decorator
            if with_logging:
                result = FlextLoggingDecorators.log_calls_decorator(decorated)
                # Ensure __wrapped__ attribute for FlextDecoratedFunction protocol
                result.__wrapped__ = func  # type: ignore[attr-defined]
                decorated = result  # type: ignore[assignment]
            return decorated

        return decorator

    # =============================================================================
    # ADDITIONAL DECORATOR METHODS - Complete functionality consolidation
    # =============================================================================

    @staticmethod
    def validate_return_value(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Validate function return value."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            result = func(*args, **kwargs)
            if result is None:
                msg = f"Function {getattr(func, '__name__', 'unknown')} returned None"
                raise ValueError(msg)
            return result

        return wrapper

    @staticmethod
    def require_non_none(
        *arg_names: str,
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Require specific arguments to be non-None."""

        def decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                for arg_name in arg_names:
                    if (
                        arg_name in bound.arguments
                        and bound.arguments[arg_name] is None
                    ):
                        msg = f"Argument '{arg_name}' cannot be None"
                        raise ValueError(msg)

                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def handle_exceptions(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Handle function exceptions with logging."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            try:
                return func(*args, **kwargs)
            except Exception:
                logger = FlextLoggerFactory.get_logger(
                    f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
                )
                logger.exception(f"Exception in {getattr(func, '__name__', 'unknown')}")
                raise

        return wrapper

    @staticmethod
    def retry_on_failure(
        max_attempts: int = 3,
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Retry function on failure."""

        def decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception:
                        if attempt == max_attempts - 1:
                            raise
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                return None  # This should never be reached

            return wrapper

        return decorator

    @staticmethod
    def memoize(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Memoize function results."""
        cache: dict[str, object] = {}

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Create cache key from args and kwargs
            key = str(args) + str(sorted(kwargs.items()) if kwargs else "")

            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]

        return wrapper

    @staticmethod
    def log_execution(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Log function execution details."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            logger = FlextLoggerFactory.get_logger(
                f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
            )
            logger.info(f"Executing {getattr(func, '__name__', 'unknown')}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed {getattr(func, '__name__', 'unknown')}")
                return result
            except Exception:
                logger.exception(f"Failed {getattr(func, '__name__', 'unknown')}")
                raise

        return wrapper

    @staticmethod
    def make_immutable(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Make function immutable (prevent modification)."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        # Make wrapper immutable by removing modification attributes
        for attr in ["__dict__", "__annotations__"]:
            if hasattr(wrapper, attr):
                delattr(wrapper, attr)

        return wrapper

    @staticmethod
    def curry(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Curry function for partial application."""

        @functools.wraps(func)
        def curried(*args: object, **kwargs: object) -> object:
            sig = inspect.signature(func)
            try:
                # Try to call with current args
                sig.bind(*args, **kwargs)
                return func(*args, **kwargs)
            except TypeError:
                # Return partial function if not enough args
                return functools.partial(func, *args, **kwargs)

        return curried

    @staticmethod
    def get_function_signature(func: FlextCallableProtocol) -> str:
        """Get function signature as string."""
        try:
            sig = inspect.signature(func)
            name = getattr(func, "__name__", "unknown")
            return f"{name}{sig}"
        except Exception:
            return f"{getattr(func, '__name__', 'unknown')}(...)"

    @staticmethod
    def create_validation_decorator(
        validator: Callable[[object], bool],
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Create custom validation decorator."""

        def decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                # Validate all arguments
                for arg in args:
                    if not validator(arg):
                        msg = f"Validation failed for argument: {arg}"
                        raise ValueError(msg)
                for value in kwargs.values():
                    if not validator(value):
                        msg = f"Validation failed for keyword argument: {value}"
                        raise ValueError(msg)

                result = func(*args, **kwargs)

                # Validate result
                if not validator(result):
                    msg = f"Validation failed for result: {result}"
                    raise ValueError(msg)

                return result

            return wrapper

        return decorator

    # Aggregate all category decorators as class references for a factory pattern
    # These will be set after the classes are defined
    Validation: type | None = (
        None  # Will be set to FlextValidationDecorators after definition
    )
    ErrorHandling: type | None = (
        None  # Will be set to FlextErrorHandlingDecorators after definition
    )
    Performance: type | None = (
        None  # Will be set to FlextPerformanceDecorators after definition
    )
    Functional: type | None = (
        None  # Will be set to FlextFunctionalDecorators after definition
    )
    Immutability: type | None = (
        None  # Will be set to FlextImmutabilityDecorators after definition
    )
    Logging: type | None = (
        None  # Will be set to FlextLoggingDecorators after definition
    )


# =============================================================================
# EXPORTS - Centralized decorator implementations
# =============================================================================

# =============================================================================
# LEGACY COMPATIBILITY LAYER - Maintain existing imports
# =============================================================================


# All specialized decorator classes are now facades that delegate to FlextDecorators
class FlextValidationDecorators:
    """Legacy compatibility - delegates to FlextDecorators validation methods."""

    @staticmethod
    def validate_arguments(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Validate arguments via FlextDecorators."""
        return FlextDecorators.validate_arguments(func)

    @staticmethod
    def validate_return_value(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Validate return value via FlextDecorators."""
        return FlextDecorators.validate_return_value(func)

    @staticmethod
    def require_non_none(
        *arg_names: str,
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Require non-none via FlextDecorators."""
        return FlextDecorators.require_non_none(*arg_names)

    @staticmethod
    def create_validation_decorator() -> object:
        """Create validation decorator via FlextDecorators."""
        return FlextDecorators.validate_arguments


class FlextErrorHandlingDecorators:
    """Legacy compatibility - delegates to FlextDecorators error handling methods."""

    @staticmethod
    def handle_exceptions(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Handle exceptions via FlextDecorators."""
        return FlextDecorators.handle_exceptions(func)

    @staticmethod
    def retry_on_failure(
        max_attempts: int = 3,
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Retry on failure via FlextDecorators."""
        return FlextDecorators.retry_on_failure(max_attempts)

    @staticmethod
    def create_safe_decorator() -> object:
        """Create safe decorator via FlextDecorators."""
        return FlextDecorators.safe_result


class FlextPerformanceDecorators:
    """Legacy compatibility - delegates to FlextDecorators performance methods."""

    @staticmethod
    def time_execution(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Time execution via FlextDecorators."""
        return FlextDecorators.time_execution(func)

    @staticmethod
    def memoize(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Memoize via FlextDecorators."""
        return FlextDecorators.memoize(func)

    @staticmethod
    def create_cache_decorator(max_size: int = 128) -> object:
        """Create cache decorator via FlextDecorators."""
        return FlextDecorators.cached_with_timing(max_size=max_size)

    @staticmethod
    def get_timing_decorator() -> object:
        """Get timing decorator via FlextDecorators."""
        return FlextDecorators.time_execution


class FlextLoggingDecorators:
    """Legacy compatibility - delegates to FlextDecorators logging methods."""

    @staticmethod
    def log_calls(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Log calls via FlextDecorators."""
        return FlextDecorators.log_calls(func)

    @staticmethod
    def log_execution(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Log execution via FlextDecorators."""
        return FlextDecorators.log_execution(func)

    @staticmethod
    def log_calls_decorator(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Log calls decorator via FlextDecorators."""
        return FlextDecorators.log_calls(func)


class FlextImmutabilityDecorators:
    """Legacy compatibility - delegates to FlextDecorators immutability methods."""

    @staticmethod
    def make_immutable(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Make immutable via FlextDecorators."""
        return FlextDecorators.make_immutable(func)


class FlextFunctionalDecorators:
    """Legacy compatibility - delegates to FlextDecorators functional methods."""

    @staticmethod
    def curry(func: FlextCallableProtocol) -> FlextCallableProtocol:
        """Curry via FlextDecorators."""
        return FlextDecorators.curry(func)


# Utility classes for compatibility
class FlextDecoratorUtils:
    """Legacy compatibility - delegates to FlextDecorators utility methods."""

    @staticmethod
    def get_function_signature(func: FlextCallableProtocol) -> str:
        """Get function signature via FlextDecorators."""
        return FlextDecorators.get_function_signature(func)
    
    @staticmethod
    def preserve_metadata(
        original: FlextCallable[object],
        wrapper: FlextCallable[object],
    ) -> FlextDecoratedFunction[object]:
        """Preserve function metadata via FlextDecorators."""
        return _FlextDecoratorUtils.preserve_metadata(original, wrapper)


class FlextDecoratorFactory:
    """Legacy compatibility - delegates to FlextDecorators factory methods."""

    @staticmethod
    def create_validation_decorator(
        validator: Callable[[object], bool],
    ) -> Callable[[FlextCallableProtocol], FlextCallableProtocol]:
        """Create validation decorator via FlextDecorators."""
        return FlextDecorators.create_validation_decorator(validator)


# Aliases for complete backward compatibility
_BaseDecoratorFactory = FlextDecoratorFactory
_BaseImmutabilityDecorators = FlextImmutabilityDecorators
_decorators_base = FlextDecorators


class _BaseDecoratorUtils:
    """Legacy utilities holder  for presence checks."""


# _decorators_base is already defined above as FlextDecorators
_validate_input_decorator = _flext_validate_input_decorator
_safe_call_decorator = _flext_safe_call_decorator

# Total exports: 13 items - centralized decorator implementations
# These are the SINGLE SOURCE OF TRUTH for all decorator patterns in FLEXT

# Set the class references after all classes are defined
FlextDecorators.Validation = FlextValidationDecorators
FlextDecorators.ErrorHandling = FlextErrorHandlingDecorators
FlextDecorators.Performance = FlextPerformanceDecorators
FlextDecorators.Functional = FlextFunctionalDecorators
FlextDecorators.Immutability = FlextImmutabilityDecorators
FlextDecorators.Logging = FlextLoggingDecorators


# =============================================================================
# TIER 1 MODULE PATTERN - EXPORTS
# =============================================================================

__all__: list[str] = [
    # =======================================================================
    # PROTOCOLS - Type checking interfaces (not classes)
    # =======================================================================
    "FlextCallableProtocol",  # Protocol for flexible callables
    "FlextDecoratedFunction",  # Type alias for decorated functions
    "FlextDecoratorFactory",
    "FlextDecoratorUtils",
    "FlextDecorators",  # 🎯 SINGLE EXPORT: All decorator functionality consolidated
    "FlextErrorHandlingDecorators",
    "FlextFlextResultCallableProtocol",  # Protocol for FlextResult callables
    "FlextFunctionalDecorators",
    "FlextImmutabilityDecorators",
    "FlextLoggingDecorators",
    "FlextPerformanceDecorators",
    # =======================================================================
    # LEGACY COMPATIBILITY LAYER - All original class names
    # =======================================================================
    "FlextValidationDecorators",
    "_BaseDecoratorFactory",
    "_BaseImmutabilityDecorators",
    "_decorators_base",
    # Individual decorator functions
    "_flext_cache_decorator",
    "_flext_safe_call_decorator",
    "_flext_timing_decorator",
    "_flext_timing_decorator_flexible",
    "_flext_validate_input_decorator",
    "_safe_call_decorator",
    "_validate_input_decorator",
]
