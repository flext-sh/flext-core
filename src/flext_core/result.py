"""Railway-oriented programming result type.

Provides FlextResult[T] for type-safe error handling without exceptions.
Enables function composition through map/flat_map chaining operations.

Classes:
    FlextResult: Generic result container with success/failure states.
    FlextResultOperations: Additional utility operations.

Functions:
    safe_call: Execute functions with automatic error handling.

"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, TypeVar, cast

from flext_core.constants import ERROR_CODES
from flext_core.exceptions import FlextOperationError
from flext_core.loggings import FlextLoggerFactory

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# FLEXT RESULT - Simple implementation
# =============================================================================


class FlextResult[T]:
    """Result type for railway-oriented programming.

    Container that represents either success (with data) or failure
    (with error). Supports functional composition via map/flat_map.

    Type Parameters:
        T: Type of the success value.
    """

    def __init__(
        self,
        data: T | None = None,
        error: str | None = None,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> None:
        """Initialize result with data or error.

        Args:
            data: Success value if no error.
            error: Error message if failed.
            error_code: Optional error code.
            error_data: Optional error metadata.

        """
        self._data = data
        self._error = error
        self._error_code = error_code
        self._error_data = error_data or {}

    @property
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self._error is None

    @property
    def success(self) -> bool:
        """Alias for is_success."""
        return self._error is None

    @property
    def is_failure(self) -> bool:
        """Check if result is failure."""
        return self._error is not None

    @property
    def is_fail(self) -> bool:
        """Alias for is_failure."""
        return self._error is not None

    @property
    def data(self) -> T | None:
        """Get success data."""
        return self._data

    @property
    def error(self) -> str | None:
        """Get error message."""
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
        """Create successful result.

        Args:
            data: The success value.

        Returns:
            Result containing the data.

        """
        return cls(data=data)

    # Backward-compat classmethod without shadowing property access
    @classmethod
    def success_factory(cls, data: T) -> FlextResult[T]:
        """Compatibility alias for ok() - renamed to avoid property conflict."""
        return cls.ok(data)

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> FlextResult[T]:
        """Backward-compat alias for fail()."""
        return cls.fail(error, error_code=error_code, error_data=error_data)

    @classmethod
    def fail(
        cls,
        error: str,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> FlextResult[T]:
        """Create failed result.

        Args:
            error: Error message.
            error_code: Optional error code.
            error_data: Optional error metadata.

        Returns:
            Result containing the error.

        """
        # Provide default error message for empty strings
        actual_error = error.strip() if error else ""
        if not actual_error:
            actual_error = "Unknown error occurred"
        return cls(error=actual_error, error_code=error_code, error_data=error_data)

    # Compatibility operations expected by tests
    @staticmethod
    def chain_results(*results: FlextResult[object]) -> FlextResult[list[object]]:
        """Chain multiple results into list, failing on first failure.

        If no results are provided, returns success with empty list.
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
            raise FlextOperationError(
                error_msg,
                error_code=self._error_code or ERROR_CODES["UNWRAP_ERROR"],
                context=self._error_data,
            )
        # For success cases, return data even if it's None
        # None is a valid value for successful results (e.g., void operations)
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
            # None is a valid value for successful results
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
            # None is a valid value for successful results
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

    # Compatibility boolean methods as callables removed - use properties instead

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
        """Return hash for result to enable use in sets and dicts."""
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

    # Enhanced methods for railway pattern
    def then[U](self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Alias for flat_map."""
        return self.flat_map(func)

    def bind[U](self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Alias for flat_map (monadic bind)."""
        return self.flat_map(func)

    def or_else(self, alternative: FlextResult[T]) -> FlextResult[T]:
        """Return this result if successful, otherwise return alternative result."""
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
        """Convert result to either tuple (data, error)."""
        if self.is_success:
            return (self._data, None)
        return (None, self._error)

    def to_exception(self) -> Exception | None:
        """Convert result to exception or None."""
        if self.is_success:
            return None

        error_msg = self._error or "Result failed"
        return FlextOperationError(error_msg, error_code=ERROR_CODES["OPERATION_ERROR"])

    @staticmethod
    def from_exception(func: Callable[[], T]) -> FlextResult[T]:
        """Create result from function that might raise exception."""
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
        """Return first successful result.

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
# MIGRATION NOTICE - Legacy functions moved to legacy.py
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
    """Safely call a function and wrap result in FlextResult (compat)."""
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
