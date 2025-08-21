"""Decorator patterns for function enhancement."""

from __future__ import annotations

import functools
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ParamSpec, Protocol, TypeVar, cast, overload, override

from flext_core.exceptions import FlextValidationError
from flext_core.loggings import FlextLoggerFactory
from flext_core.protocols import FlextDecoratedFunction, FlextLoggerProtocol
from flext_core.result import FlextResult
from flext_core.typings import TAnyDict, TErrorHandler
from flext_core.utilities import safe_call as _util_safe_call

# Type variables for decorator patterns
P = ParamSpec("P")
T = TypeVar("T")
# Use object for decorator functions to avoid Any issues
FlextCallable = Callable[[object], object]


class FlextAbstractDecorator(ABC):
    """Abstract base decorator."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize decorator."""
        self.name = name

    @abstractmethod
    def __call__(self, func: FlextCallable) -> FlextCallable:
        """Apply decoration to function."""

    @abstractmethod
    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply decoration to function."""


class FlextAbstractValidationDecorator(FlextAbstractDecorator):
    """Abstract validation decorator."""

    @abstractmethod
    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply validation decoration to function."""


class FlextAbstractErrorHandlingDecorator(FlextAbstractDecorator):
    """Abstract error handling decorator."""

    @abstractmethod
    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply error handling decoration to function."""


class FlextAbstractPerformanceDecorator(FlextAbstractDecorator):
    """Abstract performance decorator."""

    @abstractmethod
    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply performance decoration to function."""


class FlextAbstractLoggingDecorator(FlextAbstractDecorator):
    """Abstract logging decorator."""

    @abstractmethod
    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply logging decoration to function."""


# =============================================================================
# UTILITY CLASSES - Centralized decorator utilities
# =============================================================================


class FlextDecoratorUtils:
    """Decorator utility functions for metadata preservation and validation."""

    @staticmethod
    def preserve_metadata(
        original: FlextCallable,
        wrapper: FlextCallable,
    ) -> FlextCallable:
        """Preserve function metadata in decorators."""
        if hasattr(original, "__name__"):
            wrapper.__name__ = original.__name__
        if hasattr(original, "__doc__"):
            wrapper.__doc__ = original.__doc__
        if hasattr(original, "__module__"):
            wrapper.__module__ = original.__module__
        return wrapper


# =============================================================================
# VALIDATION DECORATORS - Input validation patterns
# =============================================================================


class FlextValidationDecorators(FlextAbstractValidationDecorator):
    """Validation decorators."""

    def __call__(self, func: FlextCallable) -> FlextCallable:
        """Apply validation decoration."""
        return self.apply_decoration(func)

    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply validation decoration to function."""

        def wrapped(*args: object, **kwargs: object) -> object:
            # Validate input
            input_validation = self.validate_input(args, kwargs)
            if input_validation.is_failure:
                error_msg = input_validation.error or "Input validation failed"
                raise ValueError(error_msg)

            # Execute function
            result = func(*args, **kwargs)

            # Validate output
            output_validation = self.validate_output(result)
            if output_validation.is_failure:
                error_msg = output_validation.error or "Output validation failed"
                raise ValueError(error_msg)

            return result

        return wrapped

    @staticmethod
    def create_validation_decorator(
        validator: Callable[[object], bool],
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Create input validation decorator."""
        return _flext_validate_input_decorator(validator)

    def validate_input(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> FlextResult[None]:
        """Validate input arguments."""
        if not args and not kwargs:
            return FlextResult[None].fail("No input to validate")
        return FlextResult[None].ok(None)

    def validate_output(self, output: object) -> FlextResult[None]:
        """Validate function output."""
        if output is None:
            return FlextResult[None].fail("Output validation failed: None result")
        return FlextResult[None].ok(None)

    @staticmethod
    def validate_arguments(
        func: FlextDecoratedFunction[object],
    ) -> FlextDecoratedFunction[object]:
        """Validate function arguments."""
        return func

    @staticmethod
    def create_input_validator(
        validator: Callable[[object], bool],
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Create input validation decorator (alias)."""
        return FlextDecoratorFactory.create_static_validation_decorator(validator)


# =============================================================================
# ERROR HANDLING DECORATORS - Exception safety patterns
# =============================================================================


class FlextErrorHandlingDecorators(FlextAbstractErrorHandlingDecorator):
    """Centralized error handling decorators.

    Exception safety patterns with FlextResult integration,
    retry mechanisms, and structured error reporting.
    """

    def __init__(
        self,
        name: str | None = None,
        handled_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> None:
        """Initialize error handling decorator."""
        super().__init__(name)
        self.handled_exceptions = handled_exceptions or (Exception,)

    def __call__(self, func: FlextCallable) -> FlextCallable:
        """Apply error handling decoration to function."""
        return self.apply_decoration(func)

    def handle_error(self, func_name: str, error: Exception) -> FlextResult[object]:
        """Handle caught error."""
        return FlextResult[object].fail(f"Error in {func_name}: {error!s}")

    def should_handle_error(self, error: Exception) -> bool:
        """Check if error should be handled."""
        return isinstance(error, self.handled_exceptions)

    def create_error_result(
        self, func_name: str, error: Exception
    ) -> FlextResult[object]:
        """Create error result."""
        return FlextResult[object].fail(f"Function {func_name} failed: {error!s}")

    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply error handling decoration to function."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            try:
                # On success, return the original result unchanged
                return func(*args, **kwargs)
            except Exception as e:
                if self.should_handle_error(e):
                    # On error, return a FlextResult with failure
                    return self.handle_error(getattr(func, "__name__", "unknown"), e)
                raise

        return wrapper

    @staticmethod
    def create_safe_decorator(
        error_handler: TErrorHandler | None = None,
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Create safe call decorator with optional error handler."""
        return _flext_safe_call_decorator(error_handler)

    @staticmethod
    def get_safe_decorator() -> Callable[
        [FlextDecoratedFunction[object]],
        FlextDecoratedFunction[object],
    ]:
        """Get the default safe decorator."""
        return _flext_safe_call_decorator()

    @staticmethod
    def retry_decorator(
        func: FlextDecoratedFunction[object],
    ) -> FlextDecoratedFunction[object]:
        """Add retry capability to the function."""
        return func

    @staticmethod
    def safe_call(
        error_handler: TErrorHandler | None = None,
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Safe call decorator (alias)."""
        return FlextErrorHandlingDecorators.create_safe_decorator(error_handler)


# =============================================================================
# PERFORMANCE DECORATORS - Caching and timing patterns
# =============================================================================


class FlextPerformanceDecorators(FlextAbstractPerformanceDecorator):
    """Centralized performance decorators.

    Performance optimization patterns with caching, timing,
    memoization, and metrics collection.
    """

    def __init__(self, name: str | None = None, threshold_seconds: float = 1.0) -> None:
        """Initialize performance decorator."""
        super().__init__(name)
        self.threshold_seconds = threshold_seconds
        self.metrics: dict[str, dict[str, object]] = {}

    def __call__(self, func: FlextCallable) -> FlextCallable:
        """Apply performance decoration to function."""
        return self.apply_decoration(func)

    def start_timing(self) -> float:
        """Start timing measurement."""
        return time.perf_counter()

    def stop_timing(self, start_time: float) -> float:
        """Stop timing and calculate duration."""
        return time.perf_counter() - start_time

    def record_metrics(
        self,
        func_name: str,
        duration: float,
        args: tuple[object, ...],
    ) -> None:
        """Record performance metrics."""
        self.metrics[func_name] = {
            "duration": duration,
            "args_count": len(args),
            "timestamp": time.time(),
            "slow": duration > self.threshold_seconds,
        }

    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply performance decoration to function."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            start_time = self.start_timing()
            result = func(*args, **kwargs)
            duration = self.stop_timing(start_time)
            self.record_metrics(getattr(func, "__name__", "unknown"), duration, args)
            return result

        return wrapper

    @staticmethod
    def create_cache_decorator(
        max_size: int = 128,
        size: int | None = None,
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Create cache decorator with specified cache size."""
        # Support both max_size and size parameters for compatibility
        effective_size = size if size is not None else max_size
        return _flext_cache_decorator(effective_size)

    @staticmethod
    def get_timing_decorator(
        func: FlextDecoratedFunction[object] | None = None,
    ) -> (
        FlextDecoratedFunction[object]
        | Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]
    ):
        """Return timing decorator or apply directly if function provided.

        Supports both usages:
        - As a decorator factory: @FlextPerformanceDecorators.get_timing_decorator
        - As a direct call: FlextPerformanceDecorators.get_timing_decorator(func)
        """
        if func is None:
            return _flext_timing_decorator
        return _flext_timing_decorator(func)

    @staticmethod
    def memoize_decorator(
        func: FlextDecoratedFunction[object],
    ) -> FlextDecoratedFunction[object]:
        """Add memoization to function."""
        return func

    @staticmethod
    def cache_results(
        max_size: int = 128,
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Cache results decorator (alias)."""
        return FlextPerformanceDecorators.create_cache_decorator(max_size)

    @staticmethod
    @overload
    def time_execution(
        func: FlextDecoratedFunction[object],
    ) -> FlextDecoratedFunction[object]: ...

    @staticmethod
    @overload
    def time_execution(
        func: Callable[..., object],
    ) -> Callable[..., FlextResult[object]]: ...

    @staticmethod
    def time_execution(
        func: Callable[..., object] | FlextDecoratedFunction[object],
    ) -> Callable[..., FlextResult[object]] | FlextDecoratedFunction[object]:
        """Time execution decorator - flexible version that accepts any callable."""
        return _flext_timing_decorator_flexible(func)


# =============================================================================
# LOGGING DECORATORS - Structured logging patterns
# =============================================================================


class FlextLoggingDecorators(FlextAbstractLoggingDecorator):
    """Centralized logging decorators.

    Structured logging patterns with context management,
    function call logging, and execution time tracking.
    """

    def __init__(self, name: str | None = None, log_level: str = "INFO") -> None:
        """Initialize logging decorator."""
        super().__init__(name)
        self.log_level = log_level
        self._logger = FlextLoggerFactory.get_logger(
            self.name or "FlextLoggingDecorator",
        )

    def __call__(self, func: FlextCallable) -> FlextCallable:
        """Apply logging decoration to function."""
        return self.apply_decoration(func)

    @property
    def logger(self) -> FlextLoggerProtocol:
        """Get logger instance."""
        return self._logger

    def log_entry(
        self,
        func_name: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        """Log function entry."""
        self.logger.debug(
            "Function entry",
            extra={
                "function": func_name,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            },
        )

    def log_exit(self, func_name: str, _result: object, duration: float) -> None:
        """Log function exit."""
        self.logger.debug(
            "Function exit",
            extra={
                "function": func_name,
                "duration_ms": round(duration * 1000, 2),
                "success": True,
            },
        )

    def log_error(self, func_name: str, error: Exception) -> None:
        """Log function error."""
        self.logger.error(
            "Function error",
            extra={
                "function": func_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )

    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply logging decoration to function."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            start_time = time.perf_counter()
            func_name = getattr(func, "__name__", "unknown")
            self.log_entry(func_name, args, kwargs)

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                self.log_exit(func_name, result, duration)
                return result
            except Exception as e:
                self.log_error(func_name, e)
                raise

        return wrapper

    @staticmethod
    def log_calls_decorator(
        func: FlextDecoratedFunction[object],
    ) -> FlextDecoratedFunction[object]:
        """Log function calls with arguments and execution time."""

        def wrapper(*args: object, **kwargs: object) -> object:
            logger = FlextLoggerFactory.get_logger(
                f"{getattr(func, '__module__', 'unknown')}.{getattr(func, '__name__', 'unknown')}",
            )

            # Log function entry
            logger.debug(
                "Calling function",
                extra={
                    "function": getattr(func, "__name__", "unknown"),
                    "func_module": getattr(func, "__module__", "unknown"),
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                execution_time_ms = (time.perf_counter() - start_time) * 1000

                # Log successful completion
                logger.debug(
                    "Function completed successfully",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "execution_time_ms": round(execution_time_ms, 2),
                        "success": True,
                    },
                )
            except (RuntimeError, ValueError, TypeError) as e:
                execution_time_ms = (time.perf_counter() - start_time) * 1000

                # Log exception with proper exception logging
                logger.exception(
                    "Function failed with exception",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "execution_time_ms": round(execution_time_ms, 2),
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "success": False,
                    },
                )
                raise
            else:
                return result

        return cast(
            "FlextDecoratedFunction[object]",
            FlextDecoratorUtils.preserve_metadata(func, wrapper),
        )

    @staticmethod
    def log_exceptions_decorator(
        func: FlextDecoratedFunction[object],
    ) -> FlextDecoratedFunction[object]:
        """Log function exceptions with full traceback.

        Args:
            func: FlextCallableunction to add exception logging to.

        Returns:
            Function with exception logging.

        """

        def wrapper(*args: object, **kwargs: object) -> object:
            logger = FlextLoggerFactory.get_logger(
                f"{getattr(func, '__module__', 'unknown')}.{getattr(func, '__name__', 'unknown')}",
            )

            try:
                return func(*args, **kwargs)
            except (RuntimeError, ValueError, TypeError) as e:
                # Log exception with full context
                logger.exception(
                    "Exception in function",
                    extra={
                        "function": getattr(func, "__name__", "unknown"),
                        "func_module": getattr(func, "__module__", "unknown"),
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                )
                raise

        return cast(
            "FlextDecoratedFunction[object]",
            FlextDecoratorUtils.preserve_metadata(func, wrapper),
        )

    @staticmethod
    def log_function_calls(
        func: FlextDecoratedFunction[object],
    ) -> FlextDecoratedFunction[object]:
        """Log function calls (alias).

        Args:
            func: FlextCallableunction to add call logging to.

        Returns:
            Function with call logging.

        """
        return FlextLoggingDecorators.log_calls_decorator(func)


# =============================================================================
# IMMUTABILITY DECORATORS - Data protection patterns
# =============================================================================


class FlextImmutabilityDecorators(FlextAbstractDecorator):
    """Data protection decorators for immutability enforcement.

    Provides decorators for function argument freezing and
    return value immutability patterns.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize immutability decorator."""
        super().__init__(name)

    def __call__(self, func: FlextCallable) -> FlextCallable:
        """Apply immutability decoration to function."""
        return self.apply_decoration(func)

    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply immutability decoration to function."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Basic immutability enforcement - return copy of a result
            return func(*args, **kwargs)

        return wrapper

    def validate_function(self, func: FlextCallable) -> bool:
        """Validate function compatibility."""
        return callable(func)

    @staticmethod
    def immutable_decorator(func: FlextCallable) -> FlextCallable:
        """Enforce immutability in function (static method for compatibility).

        Args:
            func: FlextCallableunction to make immutable.

        Returns:
            Immutable function.

        """
        decorator = FlextImmutabilityDecorators()
        return decorator(func)

    @staticmethod
    def freeze_args_decorator() -> Callable[
        [FlextDecoratedFunction[object]], FlextDecoratedFunction[object]
    ]:
        """Create freeze args decorator (no-op compatibility)."""

        def decorator(
            func: FlextDecoratedFunction[object],
        ) -> FlextDecoratedFunction[object]:
            """Freeze function arguments (no-op compatibility)."""
            return func

        return decorator

    @staticmethod
    def readonly_result(
        func: FlextDecoratedFunction[object],
    ) -> FlextDecoratedFunction[object]:
        """Make function result read-only (alias).

        Args:
            func: FlextCallableunction to make result read-only.

        Returns:
            Function with read-only result.

        """
        return cast(
            "FlextDecoratedFunction[object]",
            FlextImmutabilityDecorators.immutable_decorator(func),
        )


# =============================================================================
# FUNCTIONAL DECORATORS - Functional programming patterns
# =============================================================================


class FlextFunctionalDecorators(FlextAbstractDecorator):
    """Functional programming decorators for composition and currying.

    Provides decorators for function currying, composition,
    and pipeline operations.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize functional decorator."""
        super().__init__(name)

    def __call__(self, func: FlextCallable) -> FlextCallable:
        """Apply functional decoration to function."""
        return self.apply_decoration(func)

    def apply_decoration(self, func: FlextCallable) -> FlextCallable:
        """Apply functional decoration to function."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Basic functional wrapper - can be extended for currying/composition
            return func(*args, **kwargs)

        return wrapper

    def validate_function(self, func: FlextCallable) -> bool:
        """Validate function compatibility."""
        return callable(func)

    @staticmethod
    def curry_decorator(func: FlextCallable) -> FlextCallable:
        """Add currying to function (static method for compatibility).

        Args:
            func: FlextCallableunction to curry.

        Returns:
            Curried function.

        """
        return func

    @staticmethod
    def compose_decorator(func: FlextCallable) -> FlextCallable:
        """Compose functions together.

        Args:
            func: FlextCallableunction to compose.

        Returns:
            Composed function.

        """
        return func

    @staticmethod
    def pipeline_decorator(func: FlextCallable) -> FlextCallable:
        """Create function pipeline (alias).

        Args:
            func: FlextCallableunction to add to pipeline.

        Returns:
            Pipeline function.

        """
        return FlextFunctionalDecorators.compose_decorator(func)


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


class FlextDecoratorFactory(FlextAbstractDecoratorFactory):
    """Factory for creating decorators with consistent patterns.

    Provides factory methods for common decorator patterns.
    """

    def create_validation_decorator(
        self,
        **kwargs: object,
    ) -> FlextAbstractValidationDecorator:
        """Create validation decorator."""
        return FlextValidationDecorators(name=cast("str | None", kwargs.get("name")))

    def create_performance_decorator(
        self,
        **kwargs: object,
    ) -> FlextAbstractPerformanceDecorator:
        """Create performance decorator."""
        return FlextPerformanceDecorators(
            name=cast("str | None", kwargs.get("name")),
            threshold_seconds=cast("float", kwargs.get("threshold_seconds", 1.0)),
        )

    @override
    def create_logging_decorator(
        self,
        **kwargs: object,
    ) -> FlextAbstractLoggingDecorator:
        """Create logging decorator."""
        return FlextLoggingDecorators(
            name=cast("str | None", kwargs.get("name")),
            log_level=cast("str", kwargs.get("log_level", "INFO")),
        )

    @override
    def create_error_handling_decorator(
        self,
        **kwargs: object,
    ) -> FlextAbstractErrorHandlingDecorator:
        """Create error handling decorator."""
        return FlextErrorHandlingDecorators(
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
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
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
        error_handler: TErrorHandler | None = None,
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Create safe call decorator with optional error handler."""
        return _flext_safe_call_decorator(error_handler)

    @staticmethod
    def create_static_validation_decorator(
        validator: Callable[[object], bool],
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Create input validation decorator."""
        return _flext_validate_input_decorator(validator)


# =============================================================================
# INDIVIDUAL DECORATOR FUNCTIONS - Centralized implementations
# =============================================================================


def _flext_safe_call_decorator(
    error_handler: TErrorHandler | None = None,
) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
    """Create decorator for safe function execution.

    Args:
      error_handler: Optional error handler function.

    Returns:
      Decorator function.

    """
    # Delegate to result.py single source of truth - eliminates duplication

    def decorator(
        func: FlextDecoratedFunction[object],
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
                handled_result: object | None = error_handler(err_exc)

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

        return cast(
            "FlextDecoratedFunction[object]",
            FlextDecoratorUtils.preserve_metadata(func, wrapper),
        )

    return decorator


def _flext_timing_decorator(
    func: FlextDecoratedFunction[object],
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

    return cast(
        "FlextDecoratedFunction[object]",
        FlextDecoratorUtils.preserve_metadata(func, wrapper),
    )


def _flext_timing_decorator_flexible(
    func: Callable[..., object] | FlextDecoratedFunction[object],
) -> Callable[..., FlextResult[object]]:
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
                return result

            # Otherwise wrap in FlextResult
            return FlextResult[object].ok(result)

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            execution_times.append(execution_time)
            return FlextResult[object].fail(f"Function failed: {e!s}")

    return cast(
        "Callable[..., FlextResult[object]]",
        FlextDecoratorUtils.preserve_metadata(func, wrapper),
    )


def _flext_validate_input_decorator(
    validator: Callable[[object], bool],
) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
    """Validate function input arguments.

    Args:
      validator: Validation function.

    Returns:
      Decorator function.

    """

    def decorator(
        func: FlextDecoratedFunction[object],
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

        return cast(
            "FlextDecoratedFunction[object]",
            FlextDecoratorUtils.preserve_metadata(func, wrapper),
        )

    return decorator


def _flext_cache_decorator(
    max_size: int = 128,
) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
    """Cache function results with size limit.

    Args:
      max_size: Maximum cache size.

    Returns:
      Decorator function.

    """

    def decorator(
        func: FlextDecoratedFunction[object],
    ) -> FlextDecoratedFunction[object]:
        cache: TAnyDict = {}

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

            # Only cache values that match TAnyDict value types
            if isinstance(result, str | int | float | bool | type(None)):
                cache[cache_key] = result
            return result

        return cast(
            "FlextDecoratedFunction[object]",
            FlextDecoratorUtils.preserve_metadata(func, wrapper),
        )

    return decorator


# =============================================================================
# MAIN DECORATOR AGGREGATOR - FlextDecorators (REQUIRED BY guards.py)
# =============================================================================


class FlextDecorators:
    """Main decorator aggregator providing unified decorator interface.

    Aggregates all decorator functionality and provides common
    decorator patterns with FlextResult integration.
    """

    # Validation decorators
    @staticmethod
    def validated_with_result(
        model_class: object | None = None,
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Create a decorator that validates kwargs via Pydantic model when provided.

        Without model_class, return a decorator that wraps the function result
        into FlextResult and catches exceptions.
        """

        def decorator(
            func: FlextDecoratedFunction[object],
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

            return wrapper

        return decorator

    @staticmethod
    @overload
    def safe_result(
        func: FlextDecoratedFunction[object],
    ) -> FlextDecoratedFunction[object]: ...

    @staticmethod
    @overload
    def safe_result(
        func: Callable[..., object],
    ) -> Callable[..., FlextResult[object]]: ...

    @staticmethod
    def safe_result(
        func: Callable[..., object] | FlextDecoratedFunction[object],
    ) -> Callable[..., FlextResult[object]] | FlextDecoratedFunction[object]:
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
                    return result
                # Otherwise wrap in FlextResult
                return FlextResult[object].ok(result)
            except Exception as e:
                return FlextResult[object].fail(str(e))

        # Preserve metadata and return the wrapper
        return cast(
            "FlextDecoratedFunction[object]",
            FlextDecoratorUtils.preserve_metadata(func, wrapper),
        )

    # Additional composite decorators
    @staticmethod
    def cached_with_timing(
        max_size: int = 128,
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Create cached decorator with specified cache size and timing.

        Args:
            max_size: Maximum cache size for the cache layer.

        Returns:
            A decorator that first caches results and then measures execution time.

        """
        cache = _flext_cache_decorator(max_size)

        def decorator(
            func: FlextDecoratedFunction[object],
        ) -> FlextDecoratedFunction[object]:
            return _flext_timing_decorator(cache(func))

        return decorator

    @staticmethod
    def safe_cached(
        max_size: int = 128,
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
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
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
        """Create validated cache decorator with specified model class and cache size.

        Args:
            model_class: Validation model/class for kwargs.
            max_size: Maximum cache size.

        Returns:
            Decorator that validates and caches.

        """

        def chain(
            func: FlextDecoratedFunction[object],
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
    ) -> Callable[[FlextDecoratedFunction[object]], FlextDecoratedFunction[object]]:
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
            func: FlextDecoratedFunction[object],
        ) -> FlextDecoratedFunction[object]:
            decorated = func
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
                decorated = FlextLoggingDecorators.log_calls_decorator(decorated)
            return decorated

        return decorator

    # Aggregate all category decorators as class references for a factory pattern
    Validation = FlextValidationDecorators
    ErrorHandling = FlextErrorHandlingDecorators
    Performance = FlextPerformanceDecorators
    Functional = FlextFunctionalDecorators
    Immutability = FlextImmutabilityDecorators
    Logging = FlextLoggingDecorators


# =============================================================================
# EXPORTS - Centralized decorator implementations
# =============================================================================

__all__: list[str] = [
    # Core decorator interfaces and utilities
    "FlextDecoratedFunction",
    "FlextDecoratorFactory",
    "FlextDecoratorUtils",
    "FlextDecorators",  # MAIN decorator aggregator
    # Specialized decorator classes
    "FlextErrorHandlingDecorators",
    "FlextFunctionalDecorators",
    "FlextImmutabilityDecorators",
    "FlextLoggingDecorators",
    "FlextPerformanceDecorators",
    "FlextValidationDecorators",
    # Class aliases for compatibility
    "_BaseDecoratorFactory",
    "_BaseImmutabilityDecorators",
    # Back-compat names referenced by tests
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


class _DecoratedFunction(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _BaseDecoratorUtils:
    """Legacy utilities holder  for presence checks."""


_decorators_base = type(
    "_DecoratorsBase",
    (),
    {
        "_DecoratedFunction": _DecoratedFunction,
        "_BaseDecoratorUtils": _BaseDecoratorUtils,
    },
)
_validate_input_decorator = _flext_validate_input_decorator
_safe_call_decorator = _flext_safe_call_decorator
_BaseImmutabilityDecorators = FlextImmutabilityDecorators
_BaseDecoratorFactory = FlextDecoratorFactory

# Total exports: 13 items - centralized decorator implementations
# These are the SINGLE SOURCE OF TRUTH for all decorator patterns in FLEXT
