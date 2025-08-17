"""Railway-oriented programming result type."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TypeVar, cast

from flext_core.constants import ERROR_CODES
from flext_core.exceptions import FlextOperationError
from flext_core.loggings import FlextLoggerFactory

T = TypeVar("T")


# =============================================================================
# FLEXT RESULT - Simple implementation
# =============================================================================


class FlextResult[T]:
    """Result type for railway-oriented programming in FLEXT ecosystem.

    A generic container that represents either success (with data) or failure (with error).
    This class supports functional composition via map/flat_map operations, enabling
    clean error handling without exceptions. It's the foundation of FLEXT's error
    handling strategy across all ecosystem components.

    The class implements the Result pattern (also known as Either pattern) where
    operations can be chained without explicit error checking at each step.

    Attributes:
      is_success: True if the result represents success, False otherwise.
      is_failure: True if the result represents failure, False otherwise.
      data: The success value if successful, None otherwise.
      error: The error message if failed, None otherwise.
      error_code: Optional error code for structured error handling.
      error_data: Optional error metadata dictionary.

    Example:
      Basic usage with method chaining:

      >>> result = FlextResult.ok(10)
      >>> final = result.map(lambda x: x * 2).map(lambda x: str(x))
      >>> print(final.data)
      '20'

      Error handling without exceptions:

      >>> error_result = FlextResult.fail("Division by zero")
      >>> final = error_result.map(lambda x: x * 2)  # Skipped due to error
      >>> print(final.error)
      'Division by zero'

    """

    # Explicit instance attributes to aid static type checkers
    _data: T | None
    _error: str | None
    _error_code: str | None
    _error_data: dict[str, object]

    def __init__(
        self,
        data: T | None = None,
        error: str | None = None,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> None:
        """Initialize FlextResult with success data or error information.

        Creates a new FlextResult instance representing either a successful operation
        (with data) or a failed operation (with error details). The result is considered
        successful if no error is provided, otherwise it's considered a failure.

        Args:
            data: The success value to store. Can be None for successful void operations.
            error: Error message describing what went wrong. None for successful results.
            error_code: Optional structured error code for programmatic error handling.
            error_data: Optional dictionary containing additional error metadata and context.

        Note:
            Either data should be provided (for success) OR error should be provided (for failure).
            Providing both data and error results in a failure state (error takes precedence).

        """
        self._data = data
        self._error = error
        self._error_code = error_code
        self._error_data = error_data or {}

    @property
    def is_success(self) -> bool:
        """Check if the result represents a successful operation.

        Returns:
            bool: True if the result contains success data, False if it contains an error.

        """
        return self._error is None

    @property
    def success(self) -> bool:
        """Boolean success flag.

        Many call sites use `result.success` in boolean contexts.
        """
        return self._error is None

    @property
    def is_failure(self) -> bool:
        """Check if a result is failure."""
        return self._error is not None

    @property
    def is_fail(self) -> bool:
        """Alias for is_failure."""
        return self._error is not None

    @property
    def data(self) -> T:
        """Get success data.

        Returns the contained data. For failure results, this returns a sensible
        default of type T at runtime (typically None). Statically, the type is
        non-optional to enable ergonomic use in typed code after simple success
        checks in tests and application logic.
        """
        return cast("T", self._data)

    @property
    def error(self) -> str | None:
        """Get error message (None for successful results)."""
        return self._error

    @property
    def error_code(self) -> str | None:
        """Get error code."""
        return self._error_code

    @property
    def error_data(self) -> dict[str, object]:
        """Get error metadata."""
        return self._error_data

    @classmethod
    def ok(cls, data: T) -> FlextResult[T]:
        """Create a successful FlextResult containing the provided data.

        Factory method for creating successful results. This is the preferred way
        to create successful FlextResult instances throughout the FLEXT ecosystem.

        Args:
            data: The success value to wrap in the result.

        Returns:
            FlextResult[T]: A new successful result containing the provided data.

        Example:
            >>> result = FlextResult.ok("Hello World")
            >>> print(result.data)
            'Hello World'
            >>> print(result.is_success)
            True

        """
        return cls(data=data)

    # Note: Classmethod `success()` removed to avoid name collision with
    # the instance property `success`. Use `ok()` instead.

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> FlextResult[T]:
        """Alias for fail()."""
        return cls.fail(error, error_code=error_code, error_data=error_data)

    @classmethod
    def fail(
        cls,
        error: str,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> FlextResult[T]:
        """Create a failed FlextResult with error information.

        Factory method for creating failed results. This is the preferred way
        to create error results throughout the FLEXT ecosystem, providing
        structured error handling with optional error codes and metadata.

        Args:
            error: Human-readable error message describing what went wrong.
            error_code: Optional structured error code for programmatic handling.
            error_data: Optional dictionary containing additional error context and metadata.

        Returns:
            FlextResult[T]: A new failed result containing the error information.

        Example:
            >>> result = FlextResult.fail(
            ...     "Invalid input", error_code="VALIDATION_ERROR"
            ... )
            >>> print(result.error)
            'Invalid input'
            >>> print(result.error_code)
            'VALIDATION_ERROR'

        """
        # Provide default error message for empty strings
        actual_error = error.strip() if error else ""
        if not actual_error:
            actual_error = "Unknown error occurred"
        return cls(error=actual_error, error_code=error_code, error_data=error_data)

    # Operations expected by tests
    @staticmethod
    def chain_results(*results: FlextResult[object]) -> FlextResult[list[object]]:
        """Chain multiple results into a list, failing on first failure.

        If no results are provided, returns success with an empty list.
        """
        if not results:
            return FlextResult.ok([])
        aggregated: list[object] = []
        for res in results:
            if res.is_failure:
                return FlextResult.fail(res.error or "error")
            aggregated.append(res.data)
        return FlextResult.ok(aggregated)

    def unwrap(self) -> T:
        """Extract value or raise exception.

        Returns:
            The success value.

        Raises:
            FlextOperationError: If result is failure.

        """
        if self.is_failure:
            error_msg = self._error or "Unwrap failed"
            # When unwrapping, tests expect any provided error_code to pass through
            # and context to equal error_data directly.
            # Pass through error_code; if none, set UNWRAP_ERROR and do not
            # override with OPERATION_ERROR default inside exception.
            error_kwargs = dict(self._error_data or {})
            # Mark as unwrap-originated to allow default error code override
            error_kwargs["_unwrap_origin"] = True
            raise FlextOperationError(
                error_msg,
                code=self._error_code or ERROR_CODES["UNWRAP_ERROR"],
                operation=None,
                context=error_kwargs,
            )
        # For success cases, return data even if it's None
        #  is a valid value for successful results (e.g., void operations)
        # Type system guarantees that for success results, _data is of type T
        return cast("T", self._data)

    def map[U](self, func: Callable[[T], U]) -> FlextResult[U]:
        """Transform success value with function.

        Args:
            func: Function to apply to success value.

        Returns:
            New result with transformed value or original error.

        """
        if self.is_failure:
            return FlextResult.fail(
                self._error or "Unknown",
                self._error_code,
                self._error_data,
            )
        try:
            # Apply function to data, even if it's None
            # is a valid value for successful results
            # Type system guarantees that for success results, _data is of type T
            # Apply function to data - type system guarantees _data is T for success
            # Cast is safe here because for successful results, _data must be T
            result = func(cast("T", self._data))
            return FlextResult.ok(result)
        except (ValueError, TypeError, AttributeError) as e:
            # Handle specific transformation exceptions
            return FlextResult.fail(
                f"Transformation error: {e}",
                error_code=ERROR_CODES["EXCEPTION_ERROR"],
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )
        except Exception as e:
            # Use FLEXT Core structured error handling for all other exceptions
            return FlextResult.fail(
                f"Transformation failed: {e}",
                error_code=ERROR_CODES["MAP_ERROR"],
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )

    def flat_map[U](self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Chain operations that return results.

        Args:
            func: Function that returns a FlextResult.

        Returns:
            Result from the chained operation or original error.

        """
        if self.is_failure:
            return FlextResult.fail(
                self._error or "Unknown",
                self._error_code,
                self._error_data,
            )
        try:
            # Apply function to data, even if it's None
            #  is a valid value for successful result
            # Type system guarantees that for success results, _data is of type T
            return func(cast("T", self._data))
        except (TypeError, ValueError, AttributeError, IndexError, KeyError) as e:
            # Use FLEXT Core structured error handling
            return FlextResult.fail(
                f"Chained operation failed: {e}",
                error_code=ERROR_CODES["BIND_ERROR"],
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )
        except Exception as e:
            # Handle any other unexpected exceptions
            return FlextResult.fail(
                f"Unexpected chaining error: {e}",
                error_code=ERROR_CODES["CHAIN_ERROR"],
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )

    def __bool__(self) -> bool:
        """Boolean conversion - True for success, False for failure."""
        return self.is_success

    # Boolean methods as callables removed - use properties instead

    def unwrap_or[U](self, default: U) -> T | U:
        """Get data or return default if failure."""
        if self.is_failure:
            return default
        return self._data if self._data is not None else default

    def __eq__(self, other: object) -> bool:
        """Check equality with another result."""
        if not isinstance(other, FlextResult):
            return False
        return (
            self._data == other._data
            and self._error == other._error
            and self._error_code == other._error_code
            and self._error_data == other._error_data
        )

    def __hash__(self) -> int:
        """Return hash for a result to enable use in sets and dicts."""
        # Hash based on success state and primary content
        if self.is_success:
            # For success, hash the data (if hashable) or use a default
            try:
                return hash((True, self._data))
            except TypeError:
                # REAL SOLUTION: Proper handling of non-hashable data
                # Use type-safe approach based on data characteristics
                if hasattr(self._data, "__dict__"):
                    # For objects with __dict__, hash their attributes
                    try:
                        attrs = tuple(sorted(self._data.__dict__.items()))
                        return hash((True, attrs))
                    except (TypeError, AttributeError) as e:
                        logger = FlextLoggerFactory.get_logger(__name__)
                        logger.warning(
                            f"Failed to hash object attributes for "
                            f"{type(self._data).__name__}: {e}",
                        )

                # For complex objects, use a combination of type and memory ID
                return hash((True, type(self._data).__name__, id(self._data)))
        else:
            # For failure, hash the error message and code
            return hash((False, self._error, self._error_code))

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        if self.is_success:
            return f"FlextResult(data={self._data!r}, is_success=True, error=None)"
        return f"FlextResult(data=None, is_success=False, error={self._error!r})"

    # Enhanced methods for a railway pattern
    def then[U](self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Alias for flat_map."""
        return self.flat_map(func)

    def bind[U](self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Alias for flat_map (monadic bind)."""
        return self.flat_map(func)

    def or_else(self, alternative: FlextResult[T]) -> FlextResult[T]:
        """Return this result if successful, otherwise return an alternative result."""
        if self.is_success:
            return self
        return alternative

    def or_else_get(self, func: Callable[[], FlextResult[T]]) -> FlextResult[T]:
        """Return this result if successful, otherwise return result of func."""
        if self.is_success:
            return self
        try:
            return func()
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(str(e))

    def recover(self, func: Callable[[str], T]) -> FlextResult[T]:
        """Recover from failure by applying func to error."""
        if self.is_success:
            return self
        try:
            if self._error is not None:
                recovered_data = func(self._error)
                return FlextResult.ok(recovered_data)
            return FlextResult.fail("No error to recover from")
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(str(e))

    def recover_with(self, func: Callable[[str], FlextResult[T]]) -> FlextResult[T]:
        """Recover from failure by applying func to error, returning FlextResult."""
        if self.is_success:
            return self
        try:
            if self._error is not None:
                return func(self._error)
            return FlextResult.fail("No error to recover from")
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(str(e))

    def tap(self, func: Callable[[T], None]) -> FlextResult[T]:
        """Execute side effect function on success with non-None data, return self."""
        if self.is_success and self._data is not None:
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                func(self._data)
        return self

    def tap_error(self, func: Callable[[str], None]) -> FlextResult[T]:
        """Execute side effect function on error, return self."""
        if self.is_failure and self._error is not None:
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                func(self._error)
        return self

    def filter(
        self,
        predicate: Callable[[T], bool],
        error_msg: str = "Filter predicate failed",
    ) -> FlextResult[T]:
        """Filter success value with predicate."""
        if self.is_failure:
            return self
        try:
            if predicate(cast("T", self._data)):
                return self
            return FlextResult.fail(error_msg)
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(str(e))

    def zip_with[U](
        self,
        other: FlextResult[U],
        func: Callable[[T, U], object],
    ) -> FlextResult[object]:
        """Combine two results with a function."""
        if self.is_failure:
            return FlextResult.fail(self._error or "First result failed")
        if other.is_failure:
            return FlextResult.fail(other._error or "Second result failed")

        # Check for None data - treat as missing data
        if self._data is None or other._data is None:
            return FlextResult.fail("Missing data for zip operation")

        try:
            result = func(self._data, other._data)
            return FlextResult.ok(result)
        except (TypeError, ValueError, AttributeError, ZeroDivisionError) as e:
            return FlextResult.fail(str(e))

    def to_either(self) -> tuple[T | None, str | None]:
        """Convert a result to either tuple (data, error)."""
        if self.is_success:
            return self._data, None
        return None, self._error

    def to_exception(self) -> Exception | None:
        """Convert a result to exception or None."""
        if self.is_success:
            return None

        error_msg = self._error or "Result failed"
        return FlextOperationError(error_msg, code=ERROR_CODES["OPERATION_ERROR"])

    @staticmethod
    def from_exception(func: Callable[[], T]) -> FlextResult[T]:
        """Create a result from a function that might raise exception."""
        try:
            return FlextResult.ok(func())
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(str(e))

    @staticmethod
    def combine(*results: FlextResult[object]) -> FlextResult[list[object]]:
        """Combine multiple results into one."""
        data: list[object] = []
        for result in results:
            if result.is_failure:
                return FlextResult.fail(result.error or "Combine failed")
            if result.data is not None:
                data.append(result.data)
        return FlextResult.ok(data)

    @staticmethod
    def all_success(*results: FlextResult[object]) -> bool:
        """Check if all results are successful.

        Args:
            *results: Results to check.

        Returns:
            True if all results succeeded.

        """
        return all(result.is_success for result in results)

    @staticmethod
    def any_success(*results: FlextResult[object]) -> bool:
        """Check if any result is successful.

        Args:
            *results: Results to check.

        Returns:
            True if any result succeeded.

        """
        return any(result.is_success for result in results)

    @staticmethod
    def first_success(*results: FlextResult[T]) -> FlextResult[T]:
        """Return the first successful result.

        Args:
            *results: Results to search.

        Returns:
            First successful result or failure.

        """
        last_error = "No successful results found"
        for result in results:
            if result.is_success:
                return result
            last_error = result.error or "Unknown error"
        return FlextResult.fail(last_error)

    @staticmethod
    def try_all(*funcs: Callable[[], T]) -> FlextResult[T]:
        """Try functions until one succeeds.

        Args:
            *funcs: Functions to try.

        Returns:
            First successful result or failure.

        """
        if not funcs:
            return FlextResult.fail("No functions provided")
        last_error = "All functions failed"
        for func in funcs:
            try:
                return FlextResult.ok(func())
            except (
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
                ArithmeticError,
            ) as e:
                last_error = str(e)
                continue
        return FlextResult.fail(last_error)


# =============================================================================
# MODERN API - Use FlextResult class
# =============================================================================

# IMPORTANT: The following functions have been moved to legacy.py:
#   - chain() - Use FlextResult.chain() static method instead
#   - compose() - Use FlextResult.chain() static method instead
#   - safe_call() - Use FlextResult.from_callable() static method instead
#
# Import from legacy.py if needed for backward compatibility:
#   from flext_core.legacy import chain, compose, safe_call


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================


def safe_call[T](func: Callable[[], T]) -> FlextResult[T]:
    """Safely execute a function and wrap the result in a FlextResult.

    This utility function executes the provided function and automatically
    wraps the result in a FlextResult. If the function executes successfully,
    returns FlextResult.ok with the result. If an exception occurs, returns
    FlextResult.fail with the error message.

    Args:
      func: A callable that takes no arguments and returns a value of type T.

    Returns:
      FlextResult[T]: Success result containing the function's return value,
                     or failure result containing the exception message.

    Example:
      >>> def risky_operation() -> str:
      ...     return "success"
      >>> result = safe_call(risky_operation)
      >>> print(result.data)
      'success'

      >>> def failing_operation() -> str:
      ...     raise ValueError("Something went wrong")
      >>> result = safe_call(failing_operation)
      >>> print(result.error)
      'Something went wrong'

    """
    try:
        return FlextResult.ok(func())
    except Exception as e:
        return FlextResult.fail(str(e))


# Backward compatibility alias
FlextResultOperations = FlextResult

__all__: list[str] = [
    # Main result class
    "FlextResult",
    # Backward compatibility alias
    "FlextResultOperations",
    # Utility functions
    "safe_call",
    # Note: Other legacy functions (chain, compose) moved to legacy.py
]
