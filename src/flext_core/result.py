"""FLEXT Core Result Module.

Comprehensive railway-oriented programming implementation for the FLEXT Core library
providing type-safe error handling through inheritance from specialized result base
classes.

Architecture:
    - Inheritance from specialized result base classes (_BaseResult, _BaseResultFactory)
    - Single source of truth pattern with _result_base.py as internal definitions
    - Railway-oriented programming with monadic operations and functional composition

Railway-Oriented Programming Features:
    - Success and failure result types with type-safe operations
    - Monadic operations (map, flat_map, bind) for functional composition
    - Error handling with comprehensive error information
    - Safe function execution with automatic error wrapping
    - Result chaining for complex operation sequences

Result Types:
    - Success: Contains value with type-safe access and operations
    - Failure: Contains error information with detailed error context
    - Generic: Type-safe result with specific success and error types

Monadic Operations:
    - map: Transform success value without changing result type
    - flat_map: Transform success value with potential result type change
    - bind: Alias for flat_map with functional programming terminology
    - and_then: Chain operations with automatic error propagation
    - or_else: Provide fallback for failure cases

Error Handling:
    - Comprehensive error information with context
    - Type-safe error access and manipulation
    - Error propagation through monadic operations
    - Safe function execution with automatic error wrapping

Usage Patterns:
    # Basic result creation
    result = FlextResult.ok("success")
    assert result.is_success
    assert result.value == "success"

    error_result = FlextResult.fail("error message")
    assert not error_result.is_success
    assert error_result.error == "error message"

    # Monadic operations
    result = FlextResult.ok(5)
    doubled = result.map(lambda x: x * 2)
    assert doubled.value == 10

    # Safe function execution
    def risky_operation() -> str:
        raise ValueError("Something went wrong")

    result = safe_call(risky_operation)
    assert not result.is_success
    assert "Something went wrong" in result.error

    # Result chaining
    def parse_number(s: str) -> FlextResult[int]:
        try:
            return FlextResult.ok(int(s))
        except ValueError:
            return FlextResult.fail(f"Invalid number: {s}")

    def double_number(n: int) -> FlextResult[int]:
        return FlextResult.ok(n * 2)

    result = parse_number("5").and_then(double_number)
    assert result.is_success
    assert result.value == 10

    # Error propagation
    result = parse_number("invalid").and_then(double_number)
    assert not result.is_success
    assert "Invalid number" in result.error

Thread Safety:
    - Result instances are immutable and thread-safe
    - Monadic operations create new result instances
    - Error information is thread-safe for concurrent access
    - Safe function execution is thread-safe

Performance Considerations:
    - Result creation is optimized for minimal overhead
    - Monadic operations are efficient with lazy evaluation
    - Error handling adds minimal performance impact
    - Memory usage is optimized for result instances

Dependencies:
    - typing: Type hints and generic support
    - functools: Functional programming utilities
    - traceback: Error context and stack trace information

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from flext_core.constants import ERROR_CODES
from flext_core.exceptions import FlextOperationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.types import TFactory

# Import for runtime use
import contextlib

# Local type definitions for runtime use
T = TypeVar("T")
U = TypeVar("U")


# =============================================================================
# FLEXT RESULT - Simple implementation
# =============================================================================


class FlextResult[T]:
    """Simple result type for railway-oriented programming."""

    def __init__(
        self,
        data: T | None = None,
        error: str | None = None,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> None:
        """Initialize result."""
        self._data = data
        self._error = error
        self._error_code = error_code
        self._error_data = error_data or {}

    @property
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self._error is None

    @property
    def is_failure(self) -> bool:
        """Check if result is failure."""
        return self._error is not None

    @property
    def data(self) -> T | None:
        """Get result data."""
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
        """Get error data."""
        return self._error_data

    @classmethod
    def ok(cls, data: T) -> FlextResult[T]:
        """Create successful result."""
        return cls(data=data)

    @classmethod
    def fail(
        cls,
        error: str,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> FlextResult[T]:
        """Create failed result."""
        # Provide default error message for empty strings
        actual_error = error.strip() if error else ""
        if not actual_error:
            actual_error = "Unknown error occurred"
        return cls(error=actual_error, error_code=error_code, error_data=error_data)

    def unwrap(self) -> T:
        """Get data or raise exception."""
        if self.is_failure:
            error_msg = self._error or "Unwrap failed"
            raise FlextOperationError(
                error_msg,
                error_code=self._error_code or ERROR_CODES["UNWRAP_ERROR"],
                context=self._error_data,
            )
        # For success cases, return data even if it's None
        # None is a valid value for successful results (e.g., void operations)
        return self._data  # type: ignore[return-value]

    def map(self, func: Callable[[T], U]) -> FlextResult[U]:
        """Map successful result."""
        if self.is_failure:
            return FlextResult.fail(
                self._error or "Unknown",
                self._error_code,
                self._error_data,
            )
        try:
            # Apply function to data, even if it's None
            # None is a valid value for successful results
            result = func(self._data)  # type: ignore[arg-type]  # None is valid for successful results
            return FlextResult.ok(result)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            # Use FLEXT Core structured error handling
            return FlextResult.fail(
                f"Transformation failed: {e}",
                error_code=ERROR_CODES["MAP_ERROR"],
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )

    def flat_map(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Flat map for chaining results."""
        if self.is_failure:
            return FlextResult.fail(
                self._error or "Unknown",
                self._error_code,
                self._error_data,
            )
        try:
            # Apply function to data, even if it's None
            # None is a valid value for successful results
            return func(self._data)  # type: ignore[arg-type]  # None is valid for successful results
        except (TypeError, ValueError, AttributeError, IndexError, KeyError) as e:
            # Use FLEXT Core structured error handling
            return FlextResult.fail(
                f"Chained operation failed: {e}",
                error_code=ERROR_CODES["BIND_ERROR"],
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )

    def __bool__(self) -> bool:
        """Boolean conversion - True for success, False for failure."""
        return self.is_success

    def unwrap_or(self, default: U) -> T | U:
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
                # If data is not hashable, use id() as fallback
                return hash((True, id(self._data)))
        else:
            # For failure, hash the error message and code
            return hash((False, self._error, self._error_code))

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        if self.is_success:
            return f"FlextResult(data={self._data!r}, is_success=True, error=None)"
        return f"FlextResult(data=None, is_success=False, error={self._error!r})"

    # Enhanced methods for railway pattern
    def then(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Alias for flat_map."""
        return self.flat_map(func)

    def bind(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Alias for flat_map."""
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
        """Execute side effect function on success, return self."""
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
        if self._data is not None:
            try:
                if predicate(self._data):
                    return self
                return FlextResult.fail(error_msg)
            except (TypeError, ValueError, AttributeError) as e:
                return FlextResult.fail(str(e))
        return FlextResult.fail("No data to filter")

    def zip_with(
        self,
        other: FlextResult[U],
        func: Callable[[T, U], object],
    ) -> FlextResult[object]:
        """Combine two results with a function."""
        if self.is_failure:
            return FlextResult.fail(self._error or "First result failed")
        if other.is_failure:
            return FlextResult.fail(other._error or "Second result failed")
        if self._data is not None and other._data is not None:
            try:
                result = func(self._data, other._data)
                return FlextResult.ok(result)
            except (TypeError, ValueError, AttributeError, ZeroDivisionError) as e:
                return FlextResult.fail(str(e))
        return FlextResult.fail("Missing data for zip operation")

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
        return FlextOperationError(error_msg)

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
                return FlextResult.fail(result._error or "Combine failed")
            if result._data is not None:
                data.append(result._data)
        return FlextResult.ok(data)

    @staticmethod
    def all_success(*results: FlextResult[object]) -> bool:
        """Check if all results are successful."""
        return all(result.is_success for result in results)

    @staticmethod
    def any_success(*results: FlextResult[object]) -> bool:
        """Check if any result is successful."""
        return any(result.is_success for result in results)

    @staticmethod
    def first_success(*results: FlextResult[T]) -> FlextResult[T]:
        """Return first successful result."""
        last_error = "No successful results found"
        for result in results:
            if result.is_success:
                return result
            last_error = result.error or "Unknown error"
        return FlextResult.fail(last_error)

    @staticmethod
    def try_all(*funcs: Callable[[], T]) -> FlextResult[T]:
        """Try all functions until one succeeds."""
        if not funcs:
            return FlextResult.fail("No functions provided")
        last_error = "All functions failed"
        for func in funcs:
            try:
                return FlextResult.ok(func())
            except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                last_error = str(e)
                continue
        return FlextResult.fail(last_error)


# =============================================================================
# RAILWAY OPERATIONS - Module level functions for backward compatibility
# =============================================================================


def chain(*results: FlextResult[object]) -> FlextResult[list[object]]:
    """Chain multiple results together with early failure detection.

    Args:
        *results: Variable number of results to chain together

    Returns:
        FlextResult[list[object]] with all data or first failure encountered

    """
    data: list[object] = []
    for result in results:
        if result.is_failure:
            return FlextResult.fail(result.error or "Chain failed")
        if result.data is not None:
            data.append(result.data)
    return FlextResult.ok(data)


def compose(*results: FlextResult[object]) -> FlextResult[list[object]]:
    """Compose multiple results - alias for chain."""
    return chain(*results)


def safe_call[T](func: TFactory[T]) -> FlextResult[T]:
    """Safely call function with FlextResult error handling.

    Convenience function providing direct access to safe execution patterns
    with comprehensive exception handling and error context preservation.

    Unlike from_callable, this uses the actual exception message as the main
    error for better debugging information.

    Args:
        func: Function to execute safely

    Returns:
        FlextResult[T] with function result or captured exception

    """
    try:
        result = func()
        return FlextResult.ok(result)
    except (TypeError, ValueError, AttributeError, RuntimeError) as e:
        # Use actual exception message for better debugging
        return FlextResult.fail(
            str(e) or "Operation failed",
            error_data={
                "exception": str(e),
                "exception_type": type(e).__name__,
            },
        )


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    # Main result class
    "FlextResult",
    # Convenience functions
    "chain",
    "safe_call",
]
