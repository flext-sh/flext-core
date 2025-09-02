"""Railway-oriented programming monad for type-safe error handling.

Provides FlextResult[T] for functional error handling with monadic operations,
eliminating exceptions in business logic through railway-oriented programming.

Usage:
    result = (FlextResult.ok(10)
        .map(lambda x: x * 2)
        .flat_map(lambda x: process(x))
        .filter(lambda x: x > 0, "Invalid value"))

    if result.success:
        value = result.unwrap()
    else:
        print(f"Error: {result.error}")

Key Methods:
    ok(value) / fail(error) - Factory methods
    map() / flat_map() - Transform operations
    unwrap() / unwrap_or() - Value extraction
    filter() / tap() - Validation and side effects

    Transformation Methods:
        map(func: Callable[[T], U]) -> FlextResult[U]: Transform success value
        flat_map(func: Callable[[T], FlextResult[U]]) -> FlextResult[U]: Monadic bind
        filter(predicate: Callable[[T], bool], error: str) -> FlextResult[T]: Conditional filtering
        recover(func: Callable[[str], T]) -> FlextResult[T]: Error recovery to success
        recover_with(func: Callable[[str], FlextResult[T]]) -> FlextResult[T]: Monadic error recovery

    Access and Unwrapping:
        unwrap() -> T: Extract value or raise RuntimeError
        unwrap_or(default: T) -> T: Extract value or return default
        expect(message: str) -> T: Extract value or raise with custom message
        or_else(alternative: FlextResult[T]) -> FlextResult[T]: Alternative result
        or_else_get(func: Callable[[], FlextResult[T]]) -> FlextResult[T]: Lazy alternative

    Side Effects and Observation:
        tap(func: Callable[[T], None]) -> FlextResult[T]: Execute side effect on success
        tap_error(func: Callable[[str], None]) -> FlextResult[T]: Execute side effect on error
        to_either() -> tuple[T | None, str | None]: Convert to tuple representation
        to_exception() -> Exception | None: Convert error to exception

    Collection Operations:
        chain_results(*results) -> FlextResult[list]: Chain multiple results
        combine(*results) -> FlextResult[list]: Combine all successful results
        first_success(*results) -> FlextResult[T]: Return first successful result
        try_all(*funcs) -> FlextResult[T]: Try functions until one succeeds
        all_success(*results) -> bool: Check if all results are successful
        any_success(*results) -> bool: Check if any result is successful

    Special Methods:
        __bool__() -> bool: Boolean conversion (True for success)
        __iter__() -> Iterator[T | None | str]: Destructuring support (value, error)
        __getitem__(key: int) -> T | str | None: Subscript access [0]=value, [1]=error
        __or__(default: T) -> T: Default value operator (result | default)
        __enter__() -> T: Context manager entry (raises on failure)
        __exit__(): Context manager exit
        __eq__(other) -> bool: Equality comparison
        __hash__() -> int: Hash for use in collections
        __repr__() -> str: String representation for debugging

Examples:
    Basic railway-oriented programming:
    >>> result = (
    ...     FlextResult.ok(42)
    ...     .map(lambda x: x * 2)
    ...     .flat_map(lambda x: FlextResult.ok(x + 1))
    ...     .filter(lambda x: x > 50, "Value too small")
    ... )
    >>> if result.success:
    ...     print(f"Final value: {result.value}")  # 85

    Error handling and recovery:
    >>> failed_result = FlextResult.fail("Database error", error_code="DB_CONN")
    >>> recovered = failed_result.recover(lambda err: "default_value")
    >>> print(recovered.value)  # "default_value"

    Collection operations:
    >>> results = [FlextResult.ok(1), FlextResult.ok(2), FlextResult.fail("error")]
    >>> combined = FlextResult.first_success(*results[:2])
    >>> print(combined.value)  # 1

    Context manager usage:
    >>> try:
    ...     with FlextResult.ok("resource") as resource:
    ...         print(f"Using {resource}")
    ... except RuntimeError as e:
    ...     print(f"Failed: {e}")

    Destructuring assignment:
    >>> value, error = FlextResult.ok("success")
    >>> print(f"Value: {value}, Error: {error}")  # Value: success, Error: None

    Integration with FlextCore ecosystem:
    >>> from flext_core.core import FlextCore
    >>> core = FlextCore.get_instance()
    >>> result = (
    ...     FlextResult.ok({"name": "John", "email": "john@example.com"})
    ...     .flat_map(
    ...         lambda data: core.validate_required_fields(data, ["name", "email"])
    ...     )
    ...     .flat_map(lambda data: core.create_domain_entity("User", **data))
    ...     .tap(lambda user: core.logger.info(f"User created: {user.name}"))
    ...     .tap_error(lambda err: core.logger.error(f"Validation failed: {err}"))
    ... )
    >>> if result.success:
    ...     core.container.register("current_user", result.value)

    Advanced error handling with FlextCore:
    >>> from flext_core.constants import FlextConstants
    >>> result = FlextResult.fail(
    ...     "Database connection failed",
    ...     error_code=FlextConstants.Errors.CONNECTION_ERROR,
    ...     error_data={"retry_count": 3, "timeout": 5000},
    ... )
    >>> core.observability.record_metric("database_errors", 1, tags=result.error_data)

Notes:
    - All business operations should return FlextResult[T] for composability
    - Use map() for transformations that cannot fail
    - Use flat_map() for chaining operations that return FlextResult
    - Check success/failure before accessing value to maintain type safety
    - Leverage discriminated union features for pattern matching in Python 3.13+
    - Use error_code for structured error handling and programmatic responses

"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import TypeGuard, TypeVar, cast, overload, override

from flext_core.constants import FlextConstants

# Essential type variables redefined to avoid circular import with typings.py.
# Foundation layer modules cannot import from application/domain layers.
T = TypeVar("T")
U = TypeVar("U")


# =============================================================================
# FLEXT RESULT - Simple implementation
# =============================================================================


class FlextResult[T]:
    """Container for success value or error message.

    Holds either successful data of type T or error information. Operations
    can be chained - they execute only if previous operation succeeded.

    Attributes:
        is_success (bool): True if contains success data.
        is_failure (bool): True if contains error.
        value (T): Success data (use after checking is_success).
        error (str | None): Error message if failed.
        error_code (str | None): Optional error code.

    """

    # Python 3.13+ discriminated union architecture.
    # Type discriminant based on success/failure state.
    __match_args__ = ("_data", "_error")

    # Overloaded constructor for proper type discrimination.
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
        """Initialize result with either success data or error.

        Args:
            data: Success value for successful results.
            error: Error message for failures.
            error_code: Optional error code.
            error_data: Optional error metadata.

        """
        # Architectural invariant: exactly one of data or error must be provided.
        if error is not None:
            # Failure path: ensure data is None for type consistency.
            self._data: T | None = None
            self._error: str | None = error
        else:
            # Success path: data can be T (including None if T allows it).
            self._data = data
            self._error = None

        self._error_code = error_code
        self._error_data: dict[str, object] = error_data or {}

    def _is_success_state(self, value: T | None) -> TypeGuard[T]:
        """Type guard to check if value represents successful state.

        Args:
            value: Value to check for success state.

        Returns:
            True if value represents successful state (no error and value is not None),
            False otherwise. Acts as TypeGuard for type narrowing.

        """
        return self._error is None and value is not None

    def _ensure_success_data(self) -> T:
        """Ensure success data is available, raising RuntimeError if not.

        This is a defensive programming helper to detect logic errors.
        Should only be called when is_success is True.

        Returns:
            The success data of type T.

        Raises:
            RuntimeError: If success result has None data (logic error).

        """
        if self._data is None:
            msg = "Success result has None data - this should not happen"
            raise RuntimeError(msg)
        return self._data

    @property
    def is_success(self) -> bool:
        """Check if the result represents a successful operation.

        Returns:
            True if result contains success data, False if contains error.

        """
        return self._error is None

    @property
    def success(self) -> bool:
        """True if result is successful.

        Returns:
            True if successful, False if failed.

        """
        return self._error is None

    @property
    def is_failure(self) -> bool:
        """Check if result represents a failed operation.

        Returns:
            True if result contains error, False if contains success data.

        """
        return self._error is not None

    @property
    def failure(self) -> bool:
        """Alias for is_failure property for backward compatibility.

        Returns:
            True if result represents failure, False otherwise.

        """
        return self.is_failure

    @property
    def is_valid(self) -> bool:
        """Check if result is valid (alias for is_success for backward compatibility).

        Returns:
            True if result contains success data, False if contains error.

        """
        return self.is_success

    @property
    def error_message(self) -> str | None:
        """Get error message (alias for error property for backward compatibility).

        Returns:
            Error message string for failures, None for successful results.

        """
        return self.error

    @property
    def is_fail(self) -> bool:
        """Alias for is_failure property.

        Returns:
            True if result represents failure, False otherwise.

        """
        return self._error is not None

    @property
    def value(self) -> T:
        """Get contained value or raise TypeError on failure."""
        if self.is_failure:
            msg = "Attempted to access value on failed result"
            raise TypeError(msg)
        return cast("T", self._data)

    @property
    def data(self) -> T:
        """Alias for value property.

        Returns:
            Success data of type T.

        """
        return self.value

    @property
    def error(self) -> str | None:
        """Get error message from failed result.

        Returns:
            Error message string for failures, None for successful results.

        """
        return self._error

    @property
    def error_code(self) -> str | None:
        """Get structured error code for programmatic handling.

        Returns:
            Error code string for failures, None for successful results.

        """
        return self._error_code

    @property
    def error_data(self) -> dict[str, object]:
        """Get additional error metadata and context information.

        Returns:
            Dictionary containing error metadata. Empty dictionary for successful results.

        """
        return self._error_data

    @property
    def metadata(self) -> dict[str, object]:
        """Get metadata dictionary (alias for error_data).

        Provides compatibility with command result patterns that expect metadata access.

        Returns:
            Dictionary containing error metadata and context information.

        """
        return self._error_data

    @classmethod
    def ok(cls: type[FlextResult[T]], data: T) -> FlextResult[T]:
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
            >>> print(result.value)
            'Hello World'
            >>> print(result.is_success)
            True

        """
        return cls(data=data)

    # Note: Classmethod `success()` removed to avoid name collision with
    # the instance property `success`. Use `ok()` instead.

    @classmethod
    def create_failure(
        cls: type[FlextResult[T]],
        error: str,
        /,
        *,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> FlextResult[T]:
        """Create a failed result (alias for fail method).

        Args:
            error: Human-readable error message.
            error_code: Optional structured error code.
            error_data: Optional error metadata dictionary.

        Returns:
            FlextResult[T] containing error information.

        """
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
        # Normalize empty/whitespace errors to default message
        if not error or (isinstance(error, str) and error.isspace()):
            actual_error = "Unknown error occurred"
        else:
            actual_error = error

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

    def map(self, func: Callable[[T], U]) -> FlextResult[U]:
        """Transform success value using the provided function.

        Applies transformation function to the success value if result is successful.
        If result is failure, returns new failure result with same error information.

        Args:
            func: Transformation function that takes success value of type T
                  and returns transformed value of type U.

        Returns:
            FlextResult[U] containing transformed value on success, or
            FlextResult[U] with error information if original result failed
            or transformation raised exception.

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
            data = self._ensure_success_data()
            result = func(data)  # Type narrowing: data is T here
            return FlextResult[U](data=result)
        except (ValueError, TypeError, AttributeError) as e:
            # Handle specific transformation exceptions
            return FlextResult[U](
                error=f"Transformation error: {e}",
                error_code=FlextConstants.Errors.EXCEPTION_ERROR,
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )
        except Exception as e:
            # Use FLEXT Core structured error handling for all other exceptions
            return FlextResult[U](
                error=f"Transformation failed: {e}",
                error_code=FlextConstants.Errors.MAP_ERROR,
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )

    def flat_map(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Chain operations that return FlextResult (monadic bind).

        Enables chaining multiple operations that can fail without nested error checking.
        If current result is failure, returns new failure with same error information.
        If current result is success, applies function and returns its result directly.

        Args:
            func: Function that takes success value of type T and returns
                  FlextResult[U] representing another computation that can fail.

        Returns:
            FlextResult[U] from the chained operation if original result was successful,
            or FlextResult[U] with error information if original result failed
            or chained operation raised exception.

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
            # Safe cast after success check - _data is T here
            data = cast("T", self._data)
            return func(data)
        except (TypeError, ValueError, AttributeError, IndexError, KeyError) as e:
            # Use FLEXT Core structured error handling
            return FlextResult[U](
                error=f"Chained operation failed: {e}",
                error_code=FlextConstants.Errors.BIND_ERROR,
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )
        except Exception as e:
            # Handle any other unexpected exceptions
            return FlextResult[U](
                error=f"Unexpected chaining error: {e}",
                error_code=FlextConstants.Errors.CHAIN_ERROR,
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )

    def __bool__(self) -> bool:
        """Boolean conversion based on success/failure state.

        Returns:
            True if result is successful, False if result is failure.

        """
        return self.is_success

    def __iter__(self) -> Iterator[T | None | str]:
        """Iterator support for destructuring assignment.

        Enables unpacking pattern: value, error = result for ergonomic access
        to both success data and error information.

        Yields:
            For success: (data, None)
            For failure: (None, error_message)

        """
        if self.is_success:
            yield self._data
            yield None
        else:
            yield None
            yield self._error

    def __getitem__(self, key: int) -> T | str | None:
        """Subscript access to result components.

        Args:
            key: Index (0 for data/None, 1 for None/error).

        Returns:
            result[0] returns success data or None for failures.
            result[1] returns None for success or error message for failures.

        Raises:
            IndexError: If key is not 0 or 1.

        """
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
            raise RuntimeError(error_msg)
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
        """Context manager exit handler.

        Args:
            exc_type: Exception type if exception occurred, None otherwise.
            exc_val: Exception value if exception occurred, None otherwise.
            exc_tb: Exception traceback if exception occurred, None otherwise.

        Note:
            No special exception handling is performed. Exceptions propagate normally.

        """
        return

    @property
    def value_or_none(self) -> T | None:
        """Get success value or None for failures.

        Safe accessor that returns the success data if result is successful,
        or None if result represents a failure. Useful for optional chaining patterns.

        Returns:
            Success data of type T if successful, None if failed.

        """
        return self._data if self.is_success else None

    def expect(self, message: str) -> T:
        """Get success value or raise exception with custom message.

        Defensive alternative to .value property that provides explicit error messaging.
        Validates that success results don't contain None data (use .value if None is expected).

        Args:
            message: Custom error message to include in exception if result is failure.

        Returns:
            Success value of type T if result is successful and data is not None.

        Raises:
            RuntimeError: If result represents failure or contains None data.

        """
        if self.is_failure:
            msg = f"{message}: {self._error}"
            raise RuntimeError(msg)
        # DEFENSIVE: .expect() validates None for safety (unlike .value/.unwrap)
        if self._data is None:
            msg = f"{message}: Success result has None data - use .value if None is expected"
            raise RuntimeError(msg)
        return self._data

    # Boolean methods as callables removed - use properties instead

    # unwrap_or method moved to a better location with improved implementation

    @override
    def __eq__(self, other: object) -> bool:
        """Check equality with another result using Python 3.13+ type narrowing."""
        if not isinstance(other, FlextResult):
            return False
        # Type narrowing: after isinstance check, other is FlextResult
        # Cast other to FlextResult[object] to help type checker
        other_result = cast("FlextResult[object]", other)
        # Type-safe comparison - use try/except to handle comparison
        try:
            # Cast to help type checker with comparison
            self_data = cast("object", self._data)
            other_data = other_result._data  # No cast needed after explicit typing
            return bool(
                self_data == other_data
                and self._error == other_result._error
                and self._error_code == other_result._error_code
                and self._error_data == other_result._error_data
            )
        except Exception:
            return False

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
                    except (TypeError, AttributeError):
                        # Skip logging to avoid circular dependency
                        # Unable to hash object attributes, using fallback
                        pass

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

            >>> # Modern pattern: Use conditional .value access
            >>> return (
            ...     validate_password(password).value
            ...     if validate_password(password).success
            ...     else False
            ... )

            >>> # Chain operations cleanly (modern pattern)
            >>> result = process_data(input).map(lambda x: x.upper())
            >>> return result.value if result.success else "default"

        """
        if self.is_success:
            # Type narrowing: success case guarantees _data is T (including None if T allows it)
            # Type cast required for static analyzers that don't support discriminated unions
            return cast("T", self._data)
        return default

    def unwrap(self) -> T:
        """Return success value or raise exception if failure.

        This method provides backward compatibility for legacy code that expects
        an unwrap() method. For new code, prefer unwrap_or() or explicit success checking.

        Returns:
            Success value if result is successful

        Raises:
            RuntimeError: If result is failure

        Examples:
            >>> result = FlextResult.ok("success")
            >>> value = result.unwrap()  # Returns "success"

            >>> result = FlextResult.fail("error")
            >>> result.unwrap()  # Raises RuntimeError("error")

        """
        if self.is_success:
            return cast("T", self._data)
        raise RuntimeError(self._error or "Operation failed")

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
                raise RuntimeError(msg)
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
        return RuntimeError(error_msg)

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

    # =========================================================================
    # UTILITY METHODS - formerly FlextResultUtils
    # =========================================================================

    @classmethod
    def safe_unwrap_or_none[TUtil](cls, result: FlextResult[TUtil]) -> TUtil | None:
        """Safely unwrap FlextResult or return None on failure.

        Common pattern: result.value if result.success else None
        Modern pattern: FlextResult.safe_unwrap_or_none(result)

        Args:
            result: FlextResult to unwrap safely

        Returns:
            Result value if successful, None if failed

        """
        return result.value if result.success else None

    @classmethod
    def unwrap_or_raise[TUtil](
        cls, result: FlextResult[TUtil], exception_type: type[Exception] = RuntimeError
    ) -> TUtil:
        """Unwrap FlextResult or raise exception with error message.

        Common pattern: if result.success: return result.value else: raise Exception(result.error)
        Modern pattern: FlextResult.unwrap_or_raise(result)

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

    @classmethod
    def collect_successes[TUtil](cls, results: list[FlextResult[TUtil]]) -> list[TUtil]:
        """Collect all successful values from a list of FlextResults.

        Common pattern: [r.value for r in results if r.success]
        Modern pattern: FlextResult.collect_successes(results)

        Args:
            results: List of FlextResults to filter

        Returns:
            List of successful values only

        """
        return [r.value for r in results if r.success]

    @classmethod
    def collect_failures[TUtil](cls, results: list[FlextResult[TUtil]]) -> list[str]:
        """Collect all error messages from failed FlextResults.

        Common pattern: [r.error for r in results if r.is_failure and r.error]
        Modern pattern: FlextResult.collect_failures(results)

        Args:
            results: List of FlextResults to filter

        Returns:
            List of error messages from failed results

        """
        return [r.error for r in results if r.is_failure and r.error]

    @classmethod
    def success_rate[TUtil](cls, results: list[FlextResult[TUtil]]) -> float:
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

    @classmethod
    def batch_process[TItem, TUtil](
        cls, items: list[TItem], processor: Callable[[TItem], FlextResult[TUtil]]
    ) -> tuple[list[TUtil], list[str]]:
        """Process a batch of items and separate successes from failures.

        Args:
            items: List of items to process
            processor: Function that processes each item

        Returns:
            Tuple of (successful_results, error_messages)

        """
        results = [processor(item) for item in items]
        successes = cls.collect_successes(results)
        failures = cls.collect_failures(results)
        return successes, failures

    @classmethod
    def safe_call(cls: type[FlextResult[T]], func: Callable[[], T]) -> FlextResult[T]:
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
        >>> print(result.value)
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


def ok_result[TResultLocal](data: TResultLocal) -> FlextResult[TResultLocal]:
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


# =============================================================================
# FLEXT RESULT UTILITY PATTERNS
# =============================================================================


class FlextResultUtils:
    """COMPATIBILITY FACADE: Use FlextResult class methods instead.

    This class provides backward compatibility for existing code.
    All methods delegate to FlextResult class methods.

    DEPRECATED: Use FlextResult.[method_name] instead of FlextResultUtils.[method_name]
    """

    @staticmethod
    def safe_unwrap_or_none[T](result: FlextResult[T]) -> T | None:
        """DEPRECATED: Use FlextResult.safe_unwrap_or_none(result)."""
        return FlextResult.safe_unwrap_or_none(result)

    @staticmethod
    def unwrap_or_raise[T](
        result: FlextResult[T], exception_type: type[Exception] = RuntimeError
    ) -> T:
        """DEPRECATED: Use FlextResult.unwrap_or_raise(result, exception_type)."""
        return FlextResult.unwrap_or_raise(result, exception_type)

    @staticmethod
    def collect_successes[T](results: list[FlextResult[T]]) -> list[T]:
        """DEPRECATED: Use FlextResult.collect_successes(results)."""
        return FlextResult.collect_successes(results)

    @staticmethod
    def collect_failures[T](results: list[FlextResult[T]]) -> list[str]:
        """DEPRECATED: Use FlextResult.collect_failures(results)."""
        return FlextResult.collect_failures(results)

    @staticmethod
    def success_rate[T](results: list[FlextResult[T]]) -> float:
        """DEPRECATED: Use FlextResult.success_rate(results)."""
        return FlextResult.success_rate(results)

    @staticmethod
    def batch_process[T, U](
        items: list[T], processor: Callable[[T], FlextResult[U]]
    ) -> tuple[list[U], list[str]]:
        """DEPRECATED: Use FlextResult.batch_process(items, processor)."""
        return FlextResult.batch_process(items, processor)


__all__: list[str] = [
    "FlextResult",  # Main result class
    "fail_result",  # Convenience function for creating failure results
    "ok_result",  # Convenience function for creating success results
]
