"""Railway-oriented programming result type."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import TypeGuard, TypeVar, cast, overload, override

from flext_core.constants import ERROR_CODES
from flext_core.exceptions import FlextOperationError
from flext_core.loggings import FlextLoggerFactory

# Define TypeVars locally to avoid import issues
U = TypeVar("U")

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

      >>> result = FlextResult[int].ok(10)
      >>> final = result.map(lambda x: x * 2).map(lambda x: str(x))
      >>> print(final.data)
      '20'

      Error handling without exceptions:

      >>> error_result = FlextResult[object].fail("Division by zero")
      >>> final = error_result.map(lambda x: x * 2)  # Skipped due to error
      >>> print(final.error)
      'Division by zero'

    """

    # Python 3.13+ discriminated union architecture
    # Type discriminant based on success/failure state
    __match_args__ = ("_data", "_error")

    # Overloaded constructor for proper type discrimination
    @overload
    def __init__(
        self,
        *,
        data: T,
        error: None = None,
        error_code: None = None,
        error_data: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        data: None = None,
        error: str,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        data: T | None = None,
        error: str | None = None,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> None:
        """Initialize FlextResult with discriminated union pattern.

        Python 3.13+ architecture enforces type safety through constructor overloads:
        - Success: data is T, error is None
        - Failure: data is None, error is str

        Args:
            data: The success value (T) for successful results, None for failures.
            error: Error message for failures, None for successes.
            error_code: Optional structured error code for programmatic handling.
            error_data: Optional dictionary containing additional error context.

        Note:
            The overloaded constructors ensure type safety at compile time.
            Runtime validation maintains architectural consistency.

        """
        # Architectural invariant: exactly one of data or error must be provided
        if error is not None:
            # Failure path: ensure data is None for type consistency
            self._data: T | None = None
            self._error: str | None = error
        else:
            # Success path: data can be T (including None if T allows it)
            self._data = data
            self._error = None

        self._error_code = error_code
        self._error_data = error_data or {}

    def _is_success_state(self, value: T | None) -> TypeGuard[T]:
        """Type guard that narrows _data to T for successful results."""
        return self._error is None and value is not None

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
        """LEGACY PROPERTY: Get success data using discriminated union type narrowing.

        This property maintains backward compatibility with existing code.
        For new code, consider using the standard 'value' property or
        direct access patterns after checking is_success.

        Returns the contained data for successful results.
        The architecture guarantees type safety through constructor overloads.
        """
        if self.is_failure:
            msg = f"Attempted to access data on failed result: {self._error}"
            raise TypeError(msg)
        # Python 3.13+ discriminated union: when is_failure is False,
        # the constructor overloads guarantee _data is T (including None if T allows it)
        # Allow None data for Optional/Union types (T | None)
        # Type cast required for static analyzers that don't support discriminated unions
        return cast(
            "T", self._data
        )  # Type narrowing: _data is T (including None if T allows it)

    @property
    def value(self) -> T:
        """MODERN PROPERTY: Get success data safely after checking is_success.

        This is the standard, professional way to access result data in new code.
        Use after checking is_success to get type-safe access without exceptions.

        Example:
            >>> result = FlextResult[str].ok("hello")
            >>> if result.is_success:
            ...     print(result.value.upper())  # Type-safe, no unwrap needed
            'HELLO'

        Returns:
            The success value with full type safety.

        Note:
            Should only be used after checking is_success for type safety.
            For legacy compatibility, use .data property or .unwrap() method.

        """
        # In standard usage, user should check is_success first
        # This provides the same type narrowing as .data but with clearer intent
        if self.is_failure:
            msg = f"Attempted to access value on failed result: {self._error}"
            raise TypeError(msg)
        # Allow None data for Optional/Union types (T | None)
        # The type system handles this correctly
        # Type cast required for static analyzers that don't support discriminated unions
        return cast(
            "T", self._data
        )  # Type narrowing: _data is T (including None if T allows it)

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

    @property
    def metadata(self) -> dict[str, object]:
        """Get metadata - alias for error_data for command result compatibility."""
        return self._error_data

    @classmethod
    def ok(cls: type[FlextResult[T]], data: T, /) -> FlextResult[T]:
        """Create a successful FlextResult containing the provided data.

        Factory method for creating successful results. This is the preferred way
        to create successful FlextResult instances throughout the FLEXT ecosystem.

        Uses discriminated union architecture to guarantee type safety.

        Args:
            data: The success value to wrap in the result.

        Returns:
            FlextResult[T]: A new successful result containing the provided data.

        Example:
            >>> result = FlextResult[object].ok("Hello World")
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
        cls: type[FlextResult[T]],
        error: str,
        /,
        *,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> FlextResult[T]:
        """Alias for fail()."""
        return cls.fail(error, error_code=error_code, error_data=error_data)

    @classmethod
    def fail(
        cls: type[FlextResult[T]],
        error: str,
        /,
        *,
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
            >>> result = FlextResult[object].fail(
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

        # Create a new instance with the correct type annotation
        return cls(error=actual_error, error_code=error_code, error_data=error_data)

    # Operations
    @staticmethod
    def chain_results(*results: FlextResult[object]) -> FlextResult[list[object]]:
        """Chain multiple results into a list, failing on first failure.

        If no results are provided, returns success with an empty list.
        """
        if not results:
            return FlextResult[list[object]].ok([])
        aggregated: list[object] = []
        for res in results:
            if res.is_failure:
                return FlextResult[list[object]].fail(res.error or "error")
            aggregated.append(res.value)
        return FlextResult[list[object]].ok(aggregated)

    def unwrap(self) -> T:
        """LEGACY METHOD: Extract value or raise exception.

        This method maintains backward compatibility with existing code.
        For new code, consider using the ergonomic access patterns:
        - result.value property for direct access after success check
        - result[0] for subscript access
        - for item in result: for iteration
        - result | default for default values

        Returns:
            The success value.

        Raises:
            FlextOperationError: If result is failure.

        """
        if self.is_failure:
            error_msg = self._error or "Unwrap failed"
            # When unwrapping, pass through any provided error_code
            # and context to equal error_data directly.
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
        # For success cases, return data using discriminated union type narrowing
        # Python 3.13+ discriminated union: constructor overloads guarantee _data is T
        # NOTE: FlextResult[None].ok(None) is VALID architecture (872 uses in ecosystem)
        # None is a legitimate value for FlextResult[None] representing "success without data"
        # Type cast required for static analyzers that don't support discriminated unions
        return cast(
            "T", self._data
        )  # Type narrowing: _data is guaranteed to be T here (including None)

    def map(self, func: Callable[[T], U]) -> FlextResult[U]:
        """Transform success value with function.

        Args:
            func: Function to apply to success value.

        Returns:
            New result with transformed value or original error.

        """
        if self.is_failure:
            # Type-safe error propagation - ensure error is not None
            error_msg = self._error or "Map operation failed"
            new_result: FlextResult[U] = FlextResult(
                error=error_msg,
                error_code=self._error_code,
                error_data=self._error_data,
            )
            return new_result
        try:
            # Apply function to data using discriminated union type narrowing
            # Python 3.13+ discriminated union: _data is guaranteed to be T for success
            if self._data is None:
                msg = "Success result has None data - this should not happen"
                raise RuntimeError(msg)  # noqa: TRY301
            result = func(self._data)  # Type narrowing: _data is T here
            return FlextResult[U](data=result)
        except (ValueError, TypeError, AttributeError) as e:
            # Handle specific transformation exceptions
            return FlextResult[U](
                error=f"Transformation error: {e}",
                error_code=ERROR_CODES["EXCEPTION_ERROR"],
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )
        except Exception as e:
            # Use FLEXT Core structured error handling for all other exceptions
            return FlextResult[U](
                error=f"Transformation failed: {e}",
                error_code=ERROR_CODES["MAP_ERROR"],
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )

    def flat_map(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Chain operations that return results.

        Args:
            func: Function that returns a FlextResult.

        Returns:
            Result from the chained operation or original error.

        """
        if self.is_failure:
            # Type-safe error propagation - ensure error is not None
            error_msg = self._error or "Flat map operation failed"
            new_result: FlextResult[U] = FlextResult(
                error=error_msg,
                error_code=self._error_code,
                error_data=self._error_data,
            )
            return new_result
        try:
            # Apply function to data using discriminated union type narrowing
            # Python 3.13+ discriminated union: _data is guaranteed to be T for success
            if self._data is None:
                msg = "Success result has None data - this should not happen"
                raise RuntimeError(msg)  # noqa: TRY301
            return func(self._data)  # Type narrowing: _data is T here
        except (TypeError, ValueError, AttributeError, IndexError, KeyError) as e:
            # Use FLEXT Core structured error handling
            return FlextResult[U](
                error=f"Chained operation failed: {e}",
                error_code=ERROR_CODES["BIND_ERROR"],
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )
        except Exception as e:
            # Handle any other unexpected exceptions
            return FlextResult[U](
                error=f"Unexpected chaining error: {e}",
                error_code=ERROR_CODES["CHAIN_ERROR"],
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )

    def __bool__(self) -> bool:
        """Boolean conversion - True for success, False for failure."""
        return self.is_success

    def __iter__(self) -> Iterator[T | None | str]:
        """Allow unpacking: value, error = result for ergonomic access."""
        if self.is_success:
            yield self._data
            yield None
        else:
            yield None
            yield self._error

    def __getitem__(self, key: int) -> T | str | None:
        """Allow subscript access: result[0] for data, result[1] for error."""
        if key == 0:
            return self._data if self.is_success else None
        if key == 1:
            return self._error
        msg = "FlextResult only supports indices 0 (data) and 1 (error)"
        raise IndexError(msg)

    def __or__(self, default: T) -> T:
        """Use | operator for default values: result | default_value."""
        if self.is_success:
            if self._data is None:
                return default  # Handle None data case
            return self._data
        return default

    def __enter__(self) -> T:
        """Context manager entry - returns value or raises on error."""
        if self.is_failure:
            error_msg = self._error or "Context manager failed"
            raise FlextOperationError(
                error_msg,
                code=self._error_code or ERROR_CODES["CONTEXT_ERROR"],
                operation="context_manager",
                context=self._error_data,
            )
        # Type narrowing: for success results, _data must be T
        if self._data is None:
            msg = "Success result has None data - this should not happen"
            raise RuntimeError(msg)
        return self._data

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit - no special handling needed."""
        return

    @property
    def value_or_none(self) -> T | None:
        """Explicit alias for value property - returns None on failure."""
        return self._data if self.is_success else None

    def expect(self, message: str) -> T:
        """Get value or raise with custom message - defensive alternative to unwrap().

        This method provides defensive validation and should be used when you want
        to ensure the data is not None. For normal operations where None is valid
        (like FlextResult[None]), use .value property instead.

        Args:
            message: Custom error message for failures

        Returns:
            The success value if not None

        Raises:
            FlextOperationError: If result is failure
            RuntimeError: If success result has None data (defensive validation)

        """
        if self.is_failure:
            msg = f"{message}: {self._error}"
            raise FlextOperationError(
                msg,
                code=self._error_code or ERROR_CODES["EXPECT_ERROR"],
                operation="expect",
                context=self._error_data,
            )
        # DEFENSIVE: .expect() validates None for safety (unlike .value/.unwrap)
        if self._data is None:
            msg = f"{message}: Success result has None data - use .value if None is expected"
            raise RuntimeError(msg)
        return self._data

    # Boolean methods as callables removed - use properties instead

    # unwrap_or method moved to a better location with improved implementation

    def __eq__(self, other: object) -> bool:
        """Check equality with another result using Python 3.13+ type narrowing."""
        if not isinstance(other, FlextResult):
            return False
        # Type narrowing: after isinstance check, other is FlextResult
        # Compare attributes directly without cast
        return (
            self._data == other._data
            and self._error == other._error
            and self._error_code == other._error_code
            and self._error_data == other._error_data
        )

    @override
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
                            f"Failed to hash object attributes for {type(self._data).__name__}: {e}",
                        )

                # For complex objects, use a combination of type and memory ID
                return hash((True, type(self._data).__name__, id(self._data)))
        else:
            # For failure, hash the error message and code
            return hash((False, self._error, self._error_code))

    @override
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        if self.is_success:
            return f"FlextResult(data={self._data!r}, is_success=True, error=None)"
        return f"FlextResult(data=None, is_success=False, error={self._error!r})"

    # Methods for a railway pattern
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
            return FlextResult[T].fail(str(e))

    def unwrap_or(self, default: T) -> T:
        """Return success value or default if failure.

        This method provides a clean, ergonomic way to handle FlextResult values
        without explicit success/failure checking, reducing code bloat.

        Args:
            default: Value to return if result is failure

        Returns:
            Success value if result is successful, default otherwise

        Examples:
            >>> # Instead of verbose checking
            >>> result = validate_password(password)
            >>> return result.value if result.success else False

            >>> # Use unwrap_or for cleaner code
            >>> return validate_password(password).unwrap_or(False)

            >>> # Chain operations cleanly
            >>> return process_data(input).map(lambda x: x.upper()).unwrap_or("default")

        """
        if self.is_success:
            # Type narrowing: success case guarantees _data is T (including None if T allows it)
            # Type cast required for static analyzers that don't support discriminated unions
            return cast("T", self._data)
        return default

    def recover(self, func: Callable[[str], T]) -> FlextResult[T]:
        """Recover from failure by applying func to error."""
        if self.is_success:
            return self
        try:
            if self._error is not None:
                recovered_data = func(self._error)
                return FlextResult[T].ok(recovered_data)
            return FlextResult[T].fail("No error to recover from")
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[T].fail(str(e))

    def recover_with(self, func: Callable[[str], FlextResult[T]]) -> FlextResult[T]:
        """Recover from failure by applying func to error, returning FlextResult."""
        if self.is_success:
            return self
        try:
            if self._error is not None:
                return func(self._error)
            return FlextResult[T].fail("No error to recover from")
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[T].fail(str(e))

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
            # Apply predicate using discriminated union type narrowing
            # Python 3.13+ discriminated union: _data is guaranteed to be T for success
            if self._data is None:
                msg = "Success result has None data - this should not happen"
                raise RuntimeError(msg)  # noqa: TRY301
            if predicate(self._data):  # Type narrowing: _data is T here
                return self
            return FlextResult[T].fail(error_msg)
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[T].fail(str(e))

    def zip_with[U](
        self,
        other: FlextResult[U],
        func: Callable[[T, U], object],
    ) -> FlextResult[object]:
        """Combine two results with a function."""
        if self.is_failure:
            return FlextResult[object].fail(self._error or "First result failed")
        if other.is_failure:
            return FlextResult[object].fail(other._error or "Second result failed")

        # Check for None data - treat as missing data
        if self._data is None or other._data is None:
            return FlextResult[object].fail("Missing data for zip operation")

        try:
            result = func(self._data, other._data)
            return FlextResult[object].ok(result)
        except (TypeError, ValueError, AttributeError, ZeroDivisionError) as e:
            return FlextResult[object].fail(str(e))

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

    @classmethod
    def from_exception(cls, func: Callable[[], T]) -> FlextResult[T]:
        """Create a result from a function that might raise exception."""
        try:
            return cls.ok(func())
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return cls.fail(str(e))

    @staticmethod
    def combine(*results: FlextResult[object]) -> FlextResult[list[object]]:
        """Combine multiple results into one."""
        data: list[object] = []
        for result in results:
            if result.is_failure:
                return FlextResult[list[object]].fail(result.error or "Combine failed")
            if result.value is not None:
                data.append(result.value)
        return FlextResult[list[object]].ok(data)

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

    @classmethod
    def first_success(cls, *results: FlextResult[T]) -> FlextResult[T]:
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
        return cls.fail(last_error)

    @classmethod
    def try_all(cls, *funcs: Callable[[], T]) -> FlextResult[T]:
        """Try functions until one succeeds.

        Args:
            *funcs: Functions to try.

        Returns:
            First successful result or failure.

        """
        if not funcs:
            return cls.fail("No functions provided")
        last_error = "All functions failed"
        for func in funcs:
            try:
                return cls.ok(func())
            except (
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
                ArithmeticError,
            ) as e:
                last_error = str(e)
                continue
        return cls.fail(last_error)


# =============================================================================
# MODERN API - Use FlextResult class
# =============================================================================

# IMPORTANT: The following functions have been moved to legacy.py:
#   - chain() - Use FlextResult.chain() static method instead
#   - compose() - Use FlextResult.chain() static method instead
#   - safe_call() - Use FlextResult.from_callable() static method instead
#
# Import from legacy.py if needed for compatibility:
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
        return FlextResult[T].ok(func())
    except Exception as e:
        return FlextResult[T].fail(str(e))


def ok_result[TResultLocal](data: TResultLocal) -> FlextResult[TResultLocal]:  # noqa: UP047
    """Typed helper that wraps FlextResult.ok.

    Use this when passing literals or values whose element types are not
    easily inferred by the analyzer from a classmethod call.
    """
    return FlextResult[TResultLocal].ok(data)


def fail_result(
    error: str,
    error_code: str | None = None,
    error_data: dict[str, object] | None = None,
) -> FlextResult[object]:
    """Typed helper that wraps FlextResult.fail.

    Returns a FlextResult with an unconstrained generic parameter. Static
    callers can annotate or cast the returned value when a concrete T is
    required.
    """
    return FlextResult[object](
        error=error, error_code=error_code, error_data=error_data
    )


# Backward compatibility alias
FlextResultOperations = FlextResult

# =============================================================================
# FLEXT RESULT UTILITY PATTERNS
# =============================================================================


class FlextResultUtils:
    """Utility functions for common FlextResult patterns.

    Modern utilities to simplify common FlextResult operations and reduce
    boilerplate code across the entire FLEXT ecosystem.
    """

    @staticmethod
    def safe_unwrap_or_none[T](result: FlextResult[T]) -> T | None:
        """Safely unwrap FlextResult or return None on failure.

        Common pattern: result.value if result.success else None
        Modern pattern: FlextResultUtils.safe_unwrap_or_none(result)

        Args:
            result: FlextResult to unwrap safely

        Returns:
            Result value if successful, None if failed

        """
        return result.unwrap_or(None)  # type: ignore[arg-type]

    @staticmethod
    def unwrap_or_raise[T](
        result: FlextResult[T], exception_type: type[Exception] = RuntimeError
    ) -> T:
        """Unwrap FlextResult or raise exception with error message.

        Common pattern: if result.success: return result.value else: raise Exception(result.error)
        Modern pattern: FlextResultUtils.unwrap_or_raise(result)

        Args:
            result: FlextResult to unwrap
            exception_type: Exception type to raise on failure

        Returns:
            Result value if successful

        Raises:
            exception_type: If result is failure

        """
        if result.success:
            return result.value
        raise exception_type(result.error or "Operation failed")

    @staticmethod
    def collect_successes[T](results: list[FlextResult[T]]) -> list[T]:
        """Collect all successful values from a list of FlextResults.

        Common pattern: [r.value for r in results if r.success]
        Modern pattern: FlextResultUtils.collect_successes(results)

        Args:
            results: List of FlextResults to filter

        Returns:
            List of successful values only

        """
        return [r.value for r in results if r.success]

    @staticmethod
    def collect_failures[T](results: list[FlextResult[T]]) -> list[str]:
        """Collect all error messages from failed FlextResults.

        Common pattern: [r.error for r in results if r.is_failure and r.error]
        Modern pattern: FlextResultUtils.collect_failures(results)

        Args:
            results: List of FlextResults to filter

        Returns:
            List of error messages from failed results

        """
        return [r.error for r in results if r.is_failure and r.error]

    @staticmethod
    def success_rate[T](results: list[FlextResult[T]]) -> float:
        """Calculate success rate from a list of FlextResults.

        Returns percentage (0.0 to 100.0) of successful results.

        Args:
            results: List of FlextResults to analyze

        Returns:
            Success rate as percentage (0.0 to 100.0)

        """
        if not results:
            return 0.0
        successes = sum(1 for r in results if r.success)
        return (successes / len(results)) * 100.0

    @staticmethod
    def batch_process[T, U](
        items: list[T], processor: Callable[[T], FlextResult[U]]
    ) -> tuple[list[U], list[str]]:
        """Process a batch of items and separate successes from failures.

        Args:
            items: List of items to process
            processor: Function that processes each item

        Returns:
            Tuple of (successful_results, error_messages)

        """
        results = [processor(item) for item in items]
        successes = FlextResultUtils.collect_successes(results)
        failures = FlextResultUtils.collect_failures(results)
        return successes, failures


__all__: list[str] = [
    # Main result class
    "FlextResult",
    # Backward compatibility alias
    "FlextResultOperations",
    # Utility class for FlextResult patterns
    "FlextResultUtils",
    # Utility functions
    "safe_call",
    # Note: Other legacy functions (chain, compose) moved to legacy.py
]
