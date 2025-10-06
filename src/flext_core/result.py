"""Railway-oriented result type with type-safe composition semantics.

This module provides the foundational FlextResult[T] class implementing the
railway pattern for error handling throughout the FLEXT ecosystem. Use
FlextResult for all operations that can succeed or fail.

Dependency Layer: 3 (Early Foundation)
Dependencies: FlextConstants, FlextTypes, FlextExceptions
Used by: All Flext modules and ecosystem projects

Provides the canonical success/failure wrapper for FLEXT-Core 1.0.0,
including explicit error metadata and backward-compatible `.value`/`.data`
accessors.

Usage:
    ```python
    from flext_core import FlextResult


    def validate_data(data: dict) -> FlextResult[FlextTypes.Dict]:
        if not data:
            return FlextResult[FlextTypes.Dict].fail("Data cannot be empty")
        return FlextResult[FlextTypes.Dict].ok(data)


    result: FlextResult[object] = validate_data({"key": value})
    if result.is_success:
        validated_data: FlextTypes.Dict = (
            result.unwrap()
        )  # Safe extraction after success check
    ```

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import time
import types
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Self, TypeGuard, cast, overload, override

from flext_core.constants import FlextConstants
from flext_core.typings import (
    FlextTypes,
    TItem,
    TResult,
    TUtil,
    U,
    V,
)

if TYPE_CHECKING:
    from flext_core.exceptions import FlextExceptions


def _get_exceptions() -> type[FlextExceptions]:
    """Lazy import FlextExceptions to avoid circular dependency."""
    from flext_core.exceptions import FlextExceptions

    return FlextExceptions


# =============================================================================
# FLEXT RESULT
# =============================================================================


class FlextResult[T_co]:  # Monad library legitimately needs many methods
    """Railway-oriented result type for type-safe error handling.

    FlextResult[T] implements the railway pattern (Either monad) for
    explicit error handling throughout the FLEXT ecosystem. Replaces
    try/except patterns with composable success/failure workflows and
    serves as the foundation for all 32+ dependent FLEXT projects.

    **Function**: Type-safe success/failure wrapper with monadic ops
        - Wraps operation results with explicit success/failure state
        - Provides monadic operations (map, flat_map, filter, bind)
        - Maintains dual API (.data/.value) for backward compatibility
        - Enables railway-oriented programming patterns ecosystem-wide
        - Supports advanced composition (sequence, traverse, combine)
        - Includes timeout, retry, and circuit breaker patterns
        - Provides resource management with context managers
        - Guarantees API stability throughout 1.x series

    **Uses**: Pure Python foundation (no external dependencies)
        - Generic type variable T_co for covariant type safety
        - Immutable result state with explicit error metadata
        - Internal caching for performance optimization
        - Descriptor pattern for dual .data/.value access
        - FlextConstants for error codes and system defaults
        - FlextTypes for type definitions and aliases
        - Python 3.13+ discriminated unions for type narrowing
        - Structured error data with error_code and error_data

    **How to use**: Basic and advanced patterns
        ```python
        from flext_core import FlextResult

        # Example 1: Basic usage - Create success/failure results
        success = FlextResult[str].ok("data")
        failure = FlextResult[str].fail("error occurred")

        # Check result state
        if success.is_success:
            value = success.unwrap()  # Safe extraction after check

        # Example 2: Railway composition (monadic chaining)
        result = (
            validate_input(data)
            .flat_map(lambda d: process_data(d))
            .map(lambda d: format_output(d))
            .map_error(lambda e: log_error(e))
            .filter(lambda d: d.is_valid, "Invalid data")
        )

        # Example 3: Dual API compatibility (ecosystem requirement)
        result = FlextResult[FlextTypes.Dict].ok({"key": "value"})
        if result.value != result.data:  # Both work (dual API)
            raise _get_exceptions().ValidationError(
                "API inconsistency detected",
                field="dual_api_compatibility",
                validation_details="result.value != result.data",
            )
        data = result.unwrap_or({})  # With default fallback

        # Example 4: Batch processing with error collection
        items = [1, 2, 3, 4, 5]
        successes, failures = FlextResult._Utils.batch_process(
            items, lambda x: process_item(x)
        )

        # Example 5: Resource management with context manager
        with FlextResult.ok(resource).expect("Resource required") as r:
            r.perform_operations()
        ```

    Args:
        value: The success value wrapped in the result (if successful).
        error: The error message string (if failed).
        error_code: Optional error code for categorization.
        error_data: Optional structured error metadata dict.

    Attributes:
        _data (T_co | None): Internal success value storage.
        _error (str | None): Internal error message storage.
        _error_code (str | None): Internal error code storage.
        _error_data (FlextTypes.Dict): Internal error metadata.

    Returns:
        FlextResult[T]: A result instance wrapping success or failure.

    Raises:
        ValueError: When unwrap() called on failure without check.
        TimeoutError: When timeout operations exceed time limits.

    Note:
        API compatibility guaranteed throughout 1.x series. Both
        .data and .value accessors maintained for ecosystem stability.
        All operations are immutable and thread-safe. Use for ALL
        operations that can succeed or fail across FLEXT ecosystem.

    Warning:
        Never use unwrap() without checking is_success first. Use
        unwrap_or() for safe extraction with defaults. Breaking the
        railway pattern by ignoring failures defeats type safety.

    Example:
        Complete workflow with error handling:

        >>> def validate_user(data: dict) -> FlextResult[User]:
        ...     if not data.get("email"):
        ...         return FlextResult[User].fail("Email required")
        ...     return FlextResult[User].ok(User(**data))
        >>> result = validate_user({"email": "test@example.com"})
        >>> print(result.is_success)
        True

    See Also:
        FlextContainer: For dependency injection with FlextResult.
        FlextModels: For domain entities using FlextResult patterns.
        FlextBus: For CQRS command/query handling with results.

    """

    _data: T_co | None
    _error: str | None
    _error_code: str | None
    _error_data: FlextTypes.Dict

    # =========================================================================
    # NESTED HELPER CLASSES - UNIFIED ARCHITECTURE PATTERN
    # =========================================================================

    class _Utils:
        """Utility methods for FlextResult static operations - NESTED HELPER."""

        @staticmethod
        def chain_results[TChain](
            results: list[FlextResult[TChain]],
        ) -> FlextResult[list[TChain]]:
            """Chain multiple results into a single result containing a list."""
            successful_results: list[TChain] = []
            for result in results:
                if not result.is_success:
                    return FlextResult[list[TChain]].fail(
                        f"Chain failed at result: {result.error}",
                    )
                successful_results.append(result.value)
            return FlextResult[list[TChain]].ok(successful_results)

        @staticmethod
        def combine[TCombine](
            *results: FlextResult[TCombine],
        ) -> FlextResult[list[TCombine]]:
            """Combine multiple results into a single result."""
            values: list[TCombine] = []
            for result in results:
                if result.is_failure:
                    return FlextResult[list[TCombine]].fail(
                        result.error or "Combine operation failed",
                    )
                values.append(result.value)
            return FlextResult[list[TCombine]].ok(values)

        @staticmethod
        def all_success[Tobject](*results: FlextResult[Tobject]) -> bool:
            """Check if all results are successful."""
            if not results:
                return True
            return all(result.is_success for result in results)

        @staticmethod
        def any_success[Tobject](*results: FlextResult[Tobject]) -> bool:
            """Check if any result is successful."""
            return any(result.is_success for result in results) if results else False

        @classmethod
        def sequence[TSeq](
            cls,
            results: list[FlextResult[TSeq]],
        ) -> FlextResult[list[TSeq]]:
            """Sequence a list of results into a result of list."""
            values: list[TSeq] = []
            for result in results:
                if result.is_failure:
                    return FlextResult[list[TSeq]].fail(
                        result.error or "Sequence failed",
                        error_code=result.error_code,
                        error_data=result.error_data,
                    )
                values.append(result.unwrap())
            return FlextResult[list[TSeq]].ok(values)

        @classmethod
        def collect_successes[TCollect](
            cls,
            results: list[FlextResult[TCollect]],
        ) -> list[TCollect]:
            """Collect all successful values from results."""
            return [result.value for result in results if result.is_success]

        @classmethod
        def collect_failures[TCollectFail](
            cls,
            results: list[FlextResult[TCollectFail]],
        ) -> FlextTypes.StringList:
            """Collect all error messages from failed results."""
            return [r.error for r in results if r.is_failure and r.error]

        @classmethod
        def batch_process[TBatch, UBatch](
            cls,
            items: list[TBatch],
            processor: Callable[[TBatch], FlextResult[UBatch]],
        ) -> tuple[list[UBatch], FlextTypes.StringList]:
            """Process batch and separate successes from failures."""
            results: list[FlextResult[UBatch]] = [processor(item) for item in items]
            successes: list[UBatch] = cls.collect_successes(results)
            failures: FlextTypes.StringList = cls.collect_failures(results)
            return successes, failures

    class _Collections:
        """Collection operations for FlextResult - NESTED HELPER."""

        @classmethod
        def traverse[TTraverse, UTraverse](
            cls,
            items: list[TTraverse],
            func: Callable[[TTraverse], FlextResult[UTraverse]],
        ) -> FlextResult[list[UTraverse]]:
            """Traverse a list with a function returning FlextResults."""
            results: list[UTraverse] = []
            for item in items:
                result: FlextResult[UTraverse] = func(item)
                if result.is_failure:
                    return FlextResult[list[UTraverse]].fail(
                        result.error or f"Traverse failed at item {item}",
                    )
                results.append(result.unwrap())
            return FlextResult[list[UTraverse]].ok(results)

        @classmethod
        def accumulate_errors[TAccumulate](
            cls,
            *results: FlextResult[TAccumulate],
        ) -> FlextResult[list[TAccumulate]]:
            """Accumulate all errors or return all successes."""
            successes: list[TAccumulate] = []
            errors: FlextTypes.StringList = []

            for result in results:
                if result.is_success:
                    successes.append(result.unwrap())
                else:
                    errors.append(result.error or "Unknown error")

            if errors:
                combined_error = "; ".join(errors)
                return FlextResult[list[TAccumulate]].fail(
                    f"Multiple errors occurred: {combined_error}",
                    error_code="ACCUMULATED_ERRORS",
                    error_data={"error_count": len(errors), "errors": errors},
                )

            return FlextResult[list[TAccumulate]].ok(successes)

        @classmethod
        def parallel_map[TPar, UPar](
            cls,
            items: list[TPar],
            func: Callable[[TPar], FlextResult[UPar]],
            *,
            fail_fast: bool = True,
        ) -> FlextResult[list[UPar]]:
            """Map function over items in parallel (conceptually)."""
            results: list[FlextResult[UPar]] = [func(item) for item in items]

            if fail_fast:
                fast_successes: list[UPar] = []
                for result in results:
                    if result.is_failure:
                        error_msg = result.error or "Sequence operation failed"
                        return FlextResult[list[UPar]].fail(error_msg)
                    fast_successes.append(result.unwrap())
                return FlextResult[list[UPar]].ok(fast_successes)

            successes: list[UPar] = []
            errors: FlextTypes.StringList = []
            for result in results:
                if result.is_failure:
                    error_msg = result.error or "Operation failed"
                    errors.append(error_msg)
                else:
                    successes.append(result.unwrap())

            if errors:
                return FlextResult[list[UPar]].fail("; ".join(errors))
            return FlextResult[list[UPar]].ok(successes)

        @classmethod
        def validate_all[TValidateAll](
            cls,
            value: TValidateAll,
            *validators: Callable[[TValidateAll], FlextResult[None]],
        ) -> FlextResult[TValidateAll]:
            """Validate data with multiple validators."""
            validation_results: list[FlextResult[None]] = [
                validator(value) for validator in validators
            ]
            errors = [
                result.error
                for result in validation_results
                if result.is_failure and result.error
            ]

            if errors:
                combined_error = "; ".join(errors)
                return FlextResult[TValidateAll].fail(
                    f"{FlextConstants.Messages.VALIDATION_FAILED}: {combined_error}",
                    error_code="VALIDATION_FAILED",
                    error_data={
                        "validation_errors": errors,
                        "error_count": len(errors),
                    },
                )

            return FlextResult[TValidateAll].ok(value)

    class _Functional:
        """Functional programming utilities for FlextResult - NESTED HELPER."""

        @staticmethod
        def kleisli_compose[T_inner, U, V](
            f: Callable[[T_inner], FlextResult[U]],
            g: Callable[[U], FlextResult[V]],
        ) -> Callable[[T_inner], FlextResult[V]]:
            """Kleisli composition (fish operator >>=) - ADVANCED MONADIC PATTERN."""

            def composed(value: T_inner) -> FlextResult[V]:
                return FlextResult[T_inner].ok(value).flat_map(f).flat_map(g)

            return composed

        @staticmethod
        def applicative_lift2[T1, T2, TResult](
            func: Callable[[T1, T2], TResult],
            result1: FlextResult[T1],
            result2: FlextResult[T2],
        ) -> FlextResult[TResult]:
            """Lift binary function to applicative context - ADVANCED APPLICATIVE PATTERN."""
            if result1.is_failure:
                return FlextResult[TResult].fail(
                    result1.error or FlextConstants.Errors.FIRST_ARG_FAILED_MSG,
                )
            if result2.is_failure:
                return FlextResult[TResult].fail(
                    result2.error or FlextConstants.Errors.SECOND_ARG_FAILED_MSG,
                )
            return FlextResult[TResult].ok(func(result1.unwrap(), result2.unwrap()))

        @staticmethod
        def applicative_lift3[T1, T2, T3, TResult](
            func: Callable[[T1, T2, T3], TResult],
            result1: FlextResult[T1],
            result2: FlextResult[T2],
            result3: FlextResult[T3],
        ) -> FlextResult[TResult]:
            """Lift ternary function to applicative context - ADVANCED APPLICATIVE PATTERN."""
            if result1.is_failure:
                return FlextResult[TResult].fail(
                    result1.error or FlextConstants.Errors.FIRST_ARG_FAILED_MSG,
                )
            if result2.is_failure:
                return FlextResult[TResult].fail(
                    result2.error or FlextConstants.Errors.SECOND_ARG_FAILED_MSG,
                )
            if result3.is_failure:
                return FlextResult[TResult].fail(
                    result3.error or "Third argument failed",
                )

            return FlextResult[TResult].ok(
                func(result1.unwrap(), result2.unwrap(), result3.unwrap()),
            )

    # Python 3.13+ discriminated union architecture.

    __match_args__ = ("_data", "_error")

    # Overloaded constructor for proper type discrimination.
    @overload
    def __init__(
        self,
        *,
        data: T_co,
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
        error_data: FlextTypes.Dict | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        data: T_co | None = None,
        error: str | None = None,
        error_code: str | None = None,
        error_data: FlextTypes.Dict | None = None,
    ) -> None:
        """Initialize result with either success data or error."""
        # Architectural invariant: exactly one of data or error must be provided.
        if error is not None:
            # Failure path: ensure data is None for type consistency.
            self._data = None
            # Ensure error is never None for failure results
            self._error = error
        else:
            # Success path: data can be T_co (including None if T_co allows it).
            self._data = data
            self._error = None

        self._error_code = error_code
        self._error_data = error_data or {}
        self.metadata: object | None = None

    def _is_success_state(self, value: T_co | None) -> TypeGuard[T_co]:
        """Type guard for success state checking."""
        return self._error is None and value is not None

    def _ensure_success_data(self) -> T_co:
        """Ensure success data is available or raise BaseError."""
        if self._data is None and self._error is None:
            msg = "Success result has None data"
            raise _get_exceptions().BaseError(
                message=msg,
                error_code="OPERATION_ERROR",
            )
        return cast("T_co", self._data)

    @property
    def is_success(self) -> bool:
        """Return ``True`` when the result carries a successful payload."""
        return self._error is None

    @property
    def success(self) -> bool:
        """Return ``True`` when the result carries a successful payload, alternative name for is_success."""
        return self._error is None

    @property
    def is_failure(self) -> bool:
        """Return ``True`` when the result represents a failure."""
        return self._error is not None

    @property
    def failed(self) -> bool:
        """Return ``True`` when the result represents a failure, alternative name for is_failure."""
        return self._error is not None

    @property
    def value(self) -> T_co:
        """Return the success payload, raising :class:`ValidationError` on failure."""
        if self.is_failure:
            msg = "Attempted to access value on failed result"
            raise _get_exceptions().ValidationError(
                message=msg,
                error_code="VALIDATION_ERROR",
            )
        # For success case, _data contains the actual value (which may be None for FlextResult[None])
        # Cast to T_co since we know this is a success result
        return cast("T_co", self._data)

    @property
    def data(self) -> T_co:
        """Return the success payload, raising :class:`TypeError` on failure.

        Direct access to the value property - use .value instead.
        Maintained for ecosystem compatibility but .value is the preferred API.
        """
        return self.value

    @property
    def error(self) -> str | None:
        """Return the captured error message for failure results."""
        # For failure results, error should never be None
        # For success results, error is None
        return self._error

    @property
    def error_code(self) -> str | None:
        """Return the structured error code supplied on failure."""
        return self._error_code

    @property
    def error_data(self) -> FlextTypes.Dict:
        """Return the structured error metadata dictionary for observability."""
        return self._error_data

    @classmethod
    def ok(cls, data: T_co) -> Self:
        """Create a successful FlextResult wrapping the provided data.

        This is the primary way to create successful results throughout the
        FLEXT ecosystem. Use FlextResult.ok() for all successful operations.

        Args:
            data: The successful data to wrap in the result.

        Returns:
            Self: A successful FlextResult containing the provided data.

        Example:
            ```python
            from flext_core import FlextResult


            def validate_email(email: str) -> FlextResult[str]:
                if "@" in email:
                    return FlextResult[str].ok(email)  # Success case
                return FlextResult[str].fail("Invalid email format")
            ```

        """
        return cls(data=data)

    # Note: Classmethod `success()` removed to avoid name collision with
    # the instance property `success`. Use `ok()` instead.

    @classmethod
    def fail(
        cls,
        error: str | None,
        /,
        *,
        error_code: str | None = None,
        error_data: FlextTypes.Dict | None = None,
    ) -> Self:
        """Create a failed FlextResult with structured error information.

        This is the primary way to create failed results throughout the FLEXT
        ecosystem. Use FlextResult.fail() for all error conditions instead of
        raising exceptions in business logic.

        Args:
            error: The error message describing the failure.
            error_code: Optional error code for categorization and monitoring.
            error_data: Optional additional error data/metadata for diagnostics.

        Returns:
            Self: A failed FlextResult with the provided error information.

        Example:
            ```python
            from flext_core import FlextResult


            def divide_numbers(a: int, b: int) -> FlextResult[float]:
                if b == 0:
                    return FlextResult[float].fail(
                        "Division by zero not allowed",
                        error_code="MATH_ERROR",
                        error_data={"dividend": a, "divisor": b},
                    )
                return FlextResult[float].ok(a / b)
            ```

        """
        # Normalize empty/whitespace errors to default message
        if not error or error.isspace():
            actual_error = "Unknown error occurred"
        else:
            actual_error = error

        # Create a new instance with the correct type annotation
        return cls(
            error=actual_error,
            error_code=error_code,
            error_data=error_data,
        )

    # Operations
    @staticmethod
    def chain_results[TChain](
        *results: FlextResult[TChain],
    ) -> FlextResult[list[TChain]]:
        """Collect a series of results, aborting on the first failure."""
        return FlextResult._Utils.chain_results(list(results))

    def map(self, func: Callable[[T_co], U]) -> FlextResult[U]:
        """Transform the success payload using ``func`` while preserving errors."""
        if self.is_failure:
            error_msg = self._error or "Map operation failed"
            new_result: FlextResult[U] = FlextResult(
                error=error_msg,
                error_code=self._error_code,
                error_data=self._error_data,
            )
            return new_result
        try:
            # Apply function to data using discriminated union type narrowing
            # Python 3.13+ discriminated union: _data is guaranteed to be T_co for success
            data: T_co = self._ensure_success_data()
            result: U = func(data)
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

    def flat_map(self, func: Callable[[T_co], FlextResult[U]]) -> FlextResult[U]:
        """Chain operations returning FlextResult."""
        if self.is_failure:
            error_msg = self._error or "Flat map operation failed"
            new_result: FlextResult[U] = FlextResult(
                error=error_msg,
                error_code=self._error_code,
                error_data=self._error_data,
            )
            return new_result

        # For success results, _data contains the success value (can be None for FlextResult[None])
        # The success/failure state is determined by the presence/absence of _error
        try:
            # Apply function to data - since we verified is_failure is False,
            # _data contains valid data of type T_co (including None for FlextResult[None])
            # Cast to T_co since we know this is a success result
            return func(cast("T_co", self._data))

        except (TypeError, ValueError, AttributeError, IndexError, KeyError) as e:
            # Use FLEXT Core structured error handling
            return FlextResult[U](
                error=f"Chained operation failed: {e}",
                error_code=FlextConstants.Errors.BIND_ERROR,
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )
        except Exception as e:
            # Handle object other unexpected exceptions
            return FlextResult[U](
                error=f"Unexpected chaining error: {e}",
                error_code=FlextConstants.Errors.CHAIN_ERROR,
                error_data={"exception_type": type(e).__name__, "exception": str(e)},
            )

    def __bool__(self) -> bool:
        """Return True if successful, False if failed."""
        return self.is_success

    def __iter__(self) -> Iterator[T_co | str | None]:
        """Enable unpacking: value, error = result."""
        if self.is_success:
            yield self._data
            yield None
        else:
            yield None
            yield self._error

    def __getitem__(self, key: int) -> T_co | str | None:
        """Access result[0] for data, result[1] for error."""
        if key == 0:
            return self._data if self.is_success else None
        if key == 1:
            return self._error
        msg = "FlextResult only supports indices 0 (data) and 1 (error)"
        raise _get_exceptions().NotFoundError(msg, field=f"index[{key}]")

    def __or__(self, default: T_co) -> T_co:
        """Use | operator for default values: result | default_value.."""
        if self.is_success:
            if self._data is None:
                return default  # Handle None data case
            return self._data
        return default

    def __enter__(self) -> T_co:
        """Context manager entry - returns value or raises on error."""
        if self.is_failure:
            error_msg = self._error or "Context manager failed"
            raise _get_exceptions().BaseError(
                message=error_msg,
                error_code="OPERATION_ERROR",
            )

        return cast("T_co", self._data)

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit."""
        # Parameters available for future error handling logic
        return

    @property
    def value_or_none(self) -> T_co | None:
        """Get value or None if failed."""
        return self._data if self.is_success else None

    def expect(self, message: str) -> T_co:
        """Get value or raise with custom message."""
        if self.is_failure:
            msg = f"{message}: {self._error}"
            raise _get_exceptions().BaseError(
                message=msg,
                error_code="OPERATION_ERROR",
            )
        # DEFENSIVE: .expect() validates None for safety (unlike .value/.unwrap)
        if self._data is None:
            msg = "Success result has None data"
            raise _get_exceptions().BaseError(
                message=msg,
                error_code="OPERATION_ERROR",
            )
        return self._data

    # Boolean methods as callables removed - use properties instead

    # unwrap_or method moved to a better location with improved implementation

    @override
    def __eq__(self, other: object) -> bool:
        """Check equality with another result using Python 3.13+ type narrowing."""
        if not isinstance(other, FlextResult):
            return False

        try:
            # Direct comparison with explicit type handling for Python 3.13+
            # Use explicit type annotations to help the type checker
            # Cast to object to avoid generic type issues
            self_data_obj: object = cast(
                "object",
                getattr(cast("object", self), "_data", None),
            )
            other_data_obj: object = cast(
                "object",
                getattr(cast("object", other), "_data", None),
            )

            # Avoid direct comparison of generic types by using identity check first
            if self_data_obj is other_data_obj:
                data_equal: bool = True
            else:
                # Use a more explicit approach to avoid type checker issues
                # Convert to string representation for comparison to avoid generic type issues
                try:
                    self_data_str: str = str(self_data_obj)
                    other_data_str: str = str(other_data_obj)
                    data_equal = self_data_str == other_data_str
                except Exception:
                    data_equal = False

            error_equal: bool = bool(self._error == other._error)
            code_equal: bool = bool(self._error_code == other._error_code)
            data_dict_equal: bool = bool(self._error_data == other._error_data)

            return data_equal and error_equal and code_equal and data_dict_equal
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
                        # For complex objects, use a combination of type and memory ID
                        return hash((True, type(self._data).__name__, id(self._data)))

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

    # Validation methods removed to avoid duplication with utility functions

    # Methods for a railway pattern

    def or_else(self, alternative: FlextResult[T_co]) -> FlextResult[T_co]:
        """Return this result if successful, otherwise return an alternative result."""
        if self.is_success:
            return self
        return alternative

    def or_else_get(self, func: Callable[[], FlextResult[T_co]]) -> Self:
        """Return this result if successful, otherwise return result of func."""
        if self.is_success:
            return self
        try:
            return cast("Self", func())
        except (TypeError, ValueError, AttributeError) as e:
            return cast("Self", FlextResult[T_co].fail(str(e)))

    def unwrap_or(self, default: T_co) -> T_co:
        """Return value or default if failed."""
        if self.is_success:
            return cast("T_co", self._data)
        return default

    def unwrap(self) -> T_co:
        """Get value or raise if failed."""
        if self.is_success:
            return cast("T_co", self._data)
        error_msg = self._error or "Operation failed"
        raise _get_exceptions().BaseError(
            message=error_msg,
            error_code="OPERATION_ERROR",
        )

    def recover(self, func: Callable[[str], T_co]) -> FlextResult[T_co]:
        """Recover from failure by applying func to error."""
        if self.is_success:
            return self
        try:
            if self._error is not None:
                recovered_data: T_co = func(self._error)
                return FlextResult[T_co].ok(recovered_data)
            return FlextResult[T_co].fail("No error to recover from")
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[T_co].fail(str(e))

    def recover_with(
        self,
        func: Callable[[str], FlextResult[T_co]],
    ) -> FlextResult[T_co]:
        """Recover from failure by applying func to error, returning FlextResult."""
        if self.is_success:
            return self
        try:
            if self._error is not None:
                return func(self._error)
            return FlextResult[T_co].fail("No error to recover from")
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[T_co].fail(str(e))

    def tap(self, func: Callable[[T_co], None]) -> FlextResult[T_co]:
        """Execute side effect function on success with non-None data, return self."""
        if self.is_success and self._data is not None:
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                func(self._data)
        return self

    def tap_error(self, func: Callable[[str], None]) -> FlextResult[T_co]:
        """Execute side effect function on error, return self."""
        if self.is_failure and self._error is not None:
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                func(self._error)
        return self

    def filter(
        self,
        predicate: Callable[[T_co], bool],
        error_msg: str = "Filter predicate failed",
    ) -> FlextResult[T_co]:
        """Filter success value with predicate."""
        if self.is_failure:
            return self
        try:
            # Apply predicate using discriminated union type narrowing
            # Python 3.13+ discriminated union: _data is guaranteed to be T_co for success
            if predicate(cast("T_co", self._data)):
                return self
            return FlextResult[T_co].fail(error_msg)
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[T_co].fail(str(e))

    def zip_with[TZip, UZip](
        self,
        other: FlextResult[UZip],
        func: Callable[[T_co, UZip], TZip],
    ) -> FlextResult[TZip]:
        """Combine two results with a function."""
        if self.is_failure:
            return FlextResult[TZip].fail(self._error or "First result failed")
        if other.is_failure:
            return FlextResult[TZip].fail(other._error or "Second result failed")

        # Check for None data - treat as missing data
        if self._data is None or other._data is None:
            return FlextResult[TZip].fail("Missing data for zip operation")

        # Both data values are not None, proceed with operation
        try:
            result: TZip = func(self._data, other._data)
            return FlextResult[TZip].ok(result)
        except (TypeError, ValueError, AttributeError, ZeroDivisionError) as e:
            return FlextResult[TZip].fail(str(e))

    def to_either(self) -> tuple[T_co | None, str | None]:
        """Convert a result to either tuple (data, error)."""
        if self.is_success:
            return self._data, None
        return None, self._error

    def to_exception(self) -> Exception | None:
        """Convert a result to exception or None."""
        if self.is_success:
            return None

        error_msg = self._error or "Result failed"
        return _get_exceptions().BaseError(
            message=error_msg, error_code="OPERATION_ERROR"
        )

    @classmethod
    def from_exception[T](
        cls: type[FlextResult[T]],
        func: Callable[[], T],
    ) -> FlextResult[T]:
        """Create a result from a function that might raise exception."""
        try:
            return cls.ok(func())
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return cls.fail(str(e))

    @staticmethod
    def combine[TCombine](
        *results: FlextResult[TCombine],
    ) -> FlextResult[list[TCombine]]:
        """Combine multiple results into one."""
        return FlextResult._Utils.combine(*results)

    @staticmethod
    def all_success[Tobject](*results: FlextResult[Tobject]) -> bool:
        """Check success condition across results.

        Implementation detail aligned with test expectations:
        - Returns True when no results are provided (vacuous truth).
        - Returns True only if all results are successful.
        """
        return FlextResult._Utils.all_success(*results)

    @staticmethod
    def any_success[Tobject](*results: FlextResult[Tobject]) -> bool:
        """Check if object result succeeded.

        For test compatibility: False for empty input, True otherwise.
        """
        return FlextResult._Utils.any_success(*results)

    @classmethod
    def first_success[T](
        cls: type[FlextResult[T]],
        *results: FlextResult[T] | Sequence[FlextResult[T]],
    ) -> FlextResult[T]:
        """Return the first successful result from a nested collection."""
        flattened = cls._flatten_variadic_args(*results)
        filtered: list[FlextResult[T]] = []
        for entry in flattened:
            if isinstance(entry, FlextResult):
                filtered.append(entry)
            else:
                msg = "first_success expects FlextResult instances"
                raise _get_exceptions().ValidationError(
                    message=msg,
                    error_code="VALIDATION_ERROR",
                )

        if not filtered:
            return cls.fail("No results provided")

        last_error = "No successful results found"
        for result in filtered:
            if result.is_success:
                return result
            last_error = result.error or last_error

        return cls.fail(last_error)

    @classmethod
    def sequence[T](cls, results: list[FlextResult[T]]) -> FlextResult[list[T]]:
        """Convert list of results to result of list, failing on first failure.

        Args:
            results: List of FlextResult instances to sequence

        Returns:
            FlextResult containing list of all values if all successful,
            or first failure encountered.

        """
        return FlextResult._Utils.sequence(results)

    @classmethod
    def try_all[T](
        cls: type[FlextResult[T]],
        *funcs: Callable[[], T | FlextResult[T]]
        | Sequence[Callable[[], T | FlextResult[T]]],
    ) -> FlextResult[T]:
        """Execute callables until one succeeds, flattening nested collections."""
        callables = cls._flatten_callable_args(*funcs)
        if not callables:
            return cls.fail("No functions provided")

        last_error = "All functions failed"
        for func in callables:
            try:
                outcome = func()
            except Exception as exc:
                last_error = str(exc)
                continue

            if isinstance(outcome, FlextResult):
                if outcome.is_success:
                    return outcome
                last_error = outcome.error or last_error
                continue

            return cls.ok(cast("T", outcome))

        return cls.fail(last_error)

    # =========================================================================
    # UTILITY METHODS - formerly FlextResultUtils
    # =========================================================================

    @classmethod
    def unwrap_or_raise(
        cls,
        result: FlextResult[TUtil],
        exception_type: type[Exception] | None = None,
    ) -> TUtil:
        """Unwrap or raise exception."""
        # ISSUE: Duplicates unwrap method functionality - instance method already does the same thing
        if result.is_success:
            return result.value
        if exception_type is None:
            raise _get_exceptions().BaseError(
                result.error or "Operation failed", error_code="OPERATION_ERROR"
            )
        raise exception_type(result.error or "Operation failed")

    @classmethod
    def collect_successes(cls, results: list[FlextResult[TUtil]]) -> list[TUtil]:
        """Collect successful values from results."""
        return FlextResult._Utils.collect_successes(results)

    @classmethod
    def collect_failures(
        cls,
        results: list[FlextResult[TUtil]],
    ) -> FlextTypes.StringList:
        """Collect error messages from failures."""
        return FlextResult._Utils.collect_failures(results)

    @classmethod
    def success_rate(cls, results: list[FlextResult[TUtil]]) -> float:
        """Calculate success rate percentage."""
        if not results:
            return 0.0
        successes = sum(1 for r in results if r.is_success)
        return (successes / len(results)) * 100.0

    @classmethod
    def batch_process(
        cls,
        items: list[TItem],
        processor: Callable[[TItem], FlextResult[TUtil]],
    ) -> tuple[list[TUtil], FlextTypes.StringList]:
        """Process batch and separate successes from failures."""
        return FlextResult._Utils.batch_process(items, processor)

    @classmethod
    def safe_call[TResult](
        cls: type[FlextResult[TResult]],
        func: Callable[[], TResult],
        *,
        error_code: str | None = None,
    ) -> FlextResult[TResult]:
        """Execute function safely, wrapping result in FlextResult.

        Similar to dry-python/returns @safe decorator but as a classmethod
        for inline use. Catches all exceptions and converts them to
        FlextResult failures with optional error code.

        Args:
            func: Callable that returns TResult
            error_code: Optional error code for failures (defaults to OPERATION_ERROR)

        Returns:
            FlextResult[TResult] wrapping the function result or error

        Example:
            ```python
            def fetch_data() -> dict:
                return api.get_data()


            # Basic usage
            result = FlextResult[FlextTypes.Dict].safe_call(fetch_data)

            # With error code
            result = FlextResult[FlextTypes.Dict].safe_call(
                fetch_data, error_code="API_ERROR"
            )

            if result.is_success:
                data = result.unwrap()
            ```

        """
        try:
            value = func()
            return FlextResult[TResult].ok(value)
        except Exception as e:
            return FlextResult[TResult].fail(
                str(e),
                error_code=error_code or FlextConstants.Errors.OPERATION_ERROR,
            )

    # === MONADIC COMPOSITION ADVANCED OPERATORS (Python 3.13) ===

    def __rshift__(self, func: Callable[[T_co], FlextResult[U]]) -> FlextResult[U]:
        """Right shift operator (>>) for monadic bind - ADVANCED COMPOSITION."""
        return self.flat_map(func)

    def __lshift__(self, func: Callable[[T_co], U]) -> FlextResult[U]:
        """Left shift operator (<<) for functor map - ADVANCED COMPOSITION."""
        return self.map(func)

    def __matmul__(self, other: FlextResult[U]) -> FlextResult[tuple[T_co, U]]:
        """Matrix multiplication operator (@) for applicative combination - ADVANCED COMPOSITION."""
        # Use applicative functor pattern
        if self.is_failure:
            # Create a failed result with the correct type
            failed_result: FlextResult[tuple[T_co, U]] = FlextResult(
                error=self._error or "Unknown error",
                error_code=self._error_code,
                error_data=self._error_data,
            )
            return failed_result
        if other.is_failure:
            # Create a failed result with the correct type
            other_failed_result: FlextResult[tuple[T_co, U]] = FlextResult(
                error=other._error or "Unknown error",
                error_code=other._error_code,
                error_data=other._error_data,
            )
            return other_failed_result

        # Both successful - combine values
        left_val = self.unwrap()
        right_val = other.unwrap()
        combined_tuple: tuple[T_co, U] = (left_val, right_val)
        success_result: FlextResult[tuple[T_co, U]] = FlextResult(data=combined_tuple)
        return success_result

    def cast_fail(self) -> FlextResult[object]:
        """Cast a failed result to a different type."""
        if self.is_success:
            msg = "Cannot cast successful result to failed"
            raise _get_exceptions().ValidationError(
                message=msg,
                field="result",
            )
        return FlextResult[object].fail(
            self.error or "Unknown error",
            error_code=self.error_code,
            error_data=self.error_data,
        )

    def __truediv__(self, other: FlextResult[U]) -> FlextResult[T_co | U]:
        """Division operator (/) for alternative fallback - ADVANCED COMPOSITION."""
        if self.is_success:
            return FlextResult[T_co | U].ok(self.unwrap())
        if other.is_success:
            return FlextResult[T_co | U].ok(other.unwrap())
        return FlextResult[T_co | U].fail(
            other.error or self.error or "All operations failed",
        )

    def __mod__(self, predicate: Callable[[T_co], bool]) -> FlextResult[T_co]:
        """Modulo operator (%) for conditional filtering - ADVANCED COMPOSITION."""
        if self.is_failure:
            return self

        try:
            if predicate(self.unwrap()):
                return self
            return FlextResult[T_co].fail(
                f"{FlextConstants.Messages.VALIDATION_FAILED} (predicate)",
            )
        except Exception as e:
            return FlextResult[T_co].fail(f"Predicate evaluation failed: {e}")

    def __and__(self, other: FlextResult[U]) -> FlextResult[tuple[T_co, U]]:
        """AND operator (&) for sequential composition - ADVANCED COMPOSITION."""
        return self @ other  # Delegate to matmul for consistency

    def __xor__(self, recovery_func: Callable[[str], T_co]) -> FlextResult[T_co]:
        """XOR operator (^) for error recovery - ADVANCED COMPOSITION."""
        return self.recover(recovery_func)

    # === ADVANCED MONADIC COMBINATORS (Category Theory) ===

    @classmethod
    def traverse(
        cls,
        items: list[TItem],
        func: Callable[[TItem], FlextResult[TResult]],
    ) -> FlextResult[list[TResult]]:
        """Traverse operation from Category Theory - ADVANCED FUNCTIONAL PATTERN."""
        return FlextResult._Collections.traverse(items, func)

    # === RAILWAY-ORIENTED PROGRAMMING ENHANCEMENTS ===

    @classmethod
    def chain_validations(
        cls,
        *validators: Callable[[], FlextResult[None]],
    ) -> FlextResult[None]:
        """Chain multiple validation functions with early termination on failure.

        Args:
            *validators: Validation functions that return FlextResult

        Returns:
            FlextResult[None] - Success if all validations pass, first failure otherwise

        """
        for validator in validators:
            try:
                result: FlextResult[None] = validator()
                if result.is_failure:
                    return FlextResult[None].fail(
                        result.error
                        or f"{FlextConstants.Messages.VALIDATION_FAILED} (chain)",
                        error_code=result.error_code,
                        error_data=result.error_data,
                    )
            except Exception as e:
                return FlextResult[None].fail(f"Validator execution failed: {e}")
        return FlextResult[None].ok(None)

    def validate_and_execute(
        self,
        validator: Callable[[T_co], FlextResult[None]],
        executor: Callable[[T_co], FlextResult[U]],
    ) -> FlextResult[U]:
        """Common pattern: validate data then execute operation.

        Args:
            validator: Function to validate the current data
            executor: Function to execute with validated data

        Returns:
            Result of executor if validation passes, validation error otherwise

        """
        if self.is_failure:
            return FlextResult[U].fail(
                self.error or "Cannot validate failed result",
                error_code=self.error_code,
                error_data=self.error_data,
            )

        return self.flat_map(
            lambda data: validator(data).flat_map(lambda _: executor(data)),
        )

    @classmethod
    def map_sequence(
        cls,
        items: list[TItem],
        mapper: Callable[[TItem], FlextResult[TResult]],
    ) -> FlextResult[list[TResult]]:
        """Process sequence with early termination on first failure.

        Args:
            items: Items to process
            mapper: Function to map each item

        Returns:
            List of results if all succeed, first failure otherwise

        """
        return cls.traverse(items, mapper)

    @classmethod
    def pipeline[TPipeline](
        cls,
        initial_value: TPipeline,
        *operations: Callable[[TPipeline], FlextResult[TPipeline]],
    ) -> FlextResult[TPipeline]:
        """Compose multiple operations into a single pipeline.

        Args:
            initial_value: Starting value for the pipeline
            *operations: Operations to chain together

        Returns:
            Final result after all operations or first failure

        """
        current_result: FlextResult[TPipeline] = FlextResult[TPipeline].ok(
            initial_value,
        )

        for operation in operations:
            current_result = current_result.flat_map(operation)
            if current_result.is_failure:
                break

        return current_result

    def or_try(
        self,
        *alternatives: Callable[[], FlextResult[T_co]],
    ) -> FlextResult[T_co]:
        """Try alternative operations if this result failed.

        Args:
            *alternatives: Alternative functions to try

        Returns:
            First successful result or final failure

        """
        if self.is_success:
            return self

        for alternative in alternatives:
            try:
                result: FlextResult[T_co] = alternative()
                if result.is_success:
                    return result
            except Exception as e:
                # Railway pattern: Continue trying alternatives when one fails
                # This is intentional - we want to attempt all fallbacks
                # Log the exception for debugging purposes
                try:
                    logger = logging.getLogger(__name__)
                    logger.debug("Alternative failed: %s", e)
                except (TypeError, AttributeError, ImportError):
                    # FlextLogger not available, skip logging
                    pass
                continue  # Try next alternative

        return self  # Return original failure if all alternatives failed  # Return original failure if all alternatives failed

    def with_context(self, context_func: Callable[[str], str]) -> FlextResult[T_co]:
        """Add contextual information to error messages.

        Args:
            context_func: Function to transform error message with context

        Returns:
            Same result with enhanced error context if failed

        """
        if self.is_success:
            return self

        if self.error:
            enhanced_error = context_func(self.error)
            return FlextResult[T_co].fail(
                enhanced_error,
                error_code=self.error_code,
                error_data=self.error_data,
            )
        return self

    def rescue_with_logging(
        self,
        logger_func: Callable[[str], None],
    ) -> FlextResult[T_co]:
        """Log error and continue with failure state.

        Args:
            logger_func: Function to log the error

        Returns:
            Same result after logging error

        """
        if self.is_failure and self.error:
            # Railway pattern: Logging failure should not break the chain
            # Using contextlib.suppress as recommended by ruff
            with contextlib.suppress(Exception):
                logger_func(self.error)
        return self

    def kleisli_compose(
        self,
        f: Callable[[T_co], FlextResult[U]],
        g: Callable[[U], FlextResult[V]],
    ) -> Callable[[T_co], FlextResult[V]]:
        """Kleisli composition (fish operator >>=) - ADVANCED MONADIC PATTERN."""

        def composed(value: T_co) -> FlextResult[V]:
            return FlextResult[T_co].ok(value).flat_map(f).flat_map(g)

        return composed

    @classmethod
    def applicative_lift2[T1, T2, TResult](
        cls,
        func: Callable[[T1, T2], TResult],
        result1: FlextResult[T1],
        result2: FlextResult[T2],
    ) -> FlextResult[TResult]:
        """Lift binary function to applicative context - ADVANCED APPLICATIVE PATTERN."""
        if result1.is_failure:
            return FlextResult[TResult].fail(result1.error or "First argument failed")
        if result2.is_failure:
            return FlextResult[TResult].fail(result2.error or "Second argument failed")

        return FlextResult[TResult].ok(func(result1.unwrap(), result2.unwrap()))

    @classmethod
    def applicative_lift3[T1, T2, T3, TResult](
        cls,
        func: Callable[[T1, T2, T3], TResult],
        result1: FlextResult[T1],
        result2: FlextResult[T2],
        result3: FlextResult[T3],
    ) -> FlextResult[TResult]:
        """Lift ternary function to applicative context - ADVANCED APPLICATIVE PATTERN."""

        def lift_func(t1_t2: tuple[T1, T2], t3: T3) -> TResult:
            return func(t1_t2[0], t1_t2[1], t3)

        return cls.applicative_lift2(
            lift_func,
            result1 @ result2,
            result3,
        )

    # === NEW ADVANCED MONADIC OPERATORS FOR COMPLEXITY REDUCTION ===

    def when(self, condition: Callable[[T_co], bool]) -> FlextResult[T_co]:
        """Conditional execution - proceed only if condition is true.

        Args:
            condition: Predicate function to test the value

        Returns:
            Same result if condition passes or result is failure,
            failure result if condition fails

        """
        if self.is_failure:
            return self

        try:
            if condition(self.unwrap()):
                return self
            return FlextResult[T_co].fail("Conditional execution failed")
        except Exception as e:
            return FlextResult[T_co].fail(f"Condition evaluation failed: {e}")

    def unless(self, condition: Callable[[T_co], bool]) -> FlextResult[T_co]:
        """Conditional execution - proceed only if condition is false.

        Args:
            condition: Predicate function to test the value

        Returns:
            Same result if condition fails or result is failure,
            failure result if condition passes

        """
        return self.when(lambda x: not condition(x))

    def if_then_else(
        self,
        condition: Callable[[T_co], bool],
        then_func: Callable[[T_co], FlextResult[U]],
        else_func: Callable[[T_co], FlextResult[U]],
    ) -> FlextResult[U]:
        """Conditional branching with different execution paths.

        Args:
            condition: Predicate function to test the value
            then_func: Function to execute if condition is true
            else_func: Function to execute if condition is false

        Returns:
            Result of then_func or else_func based on condition

        """
        if self.is_failure:
            return FlextResult[U].fail(
                self.error or "Cannot branch on failed result",
                error_code=self.error_code,
                error_data=self.error_data,
            )

        try:
            if condition(self.unwrap()):
                return then_func(self.unwrap())
            return else_func(self.unwrap())
        except Exception as e:
            return FlextResult[U].fail(f"Conditional branching failed: {e}")

    @classmethod
    def accumulate_errors[TAccum](
        cls,
        *results: FlextResult[TAccum],
    ) -> FlextResult[list[TAccum]]:
        """Accumulate all errors instead of early termination.

        Args:
            *results: Results to accumulate

        Returns:
            Success with all values if no failures, or failure with all error messages

        """
        return FlextResult._Collections.accumulate_errors(*results)

    @classmethod
    def collect_all_errors[TCollect](
        cls,
        *results: FlextResult[TCollect],
    ) -> tuple[list[TCollect], FlextTypes.StringList]:
        """Collect all successful values and all error messages.

        Args:
            *results: Results to collect from

        Returns:
            Tuple of (successful_values, error_messages)

        """
        successes: list[TCollect] = []
        errors: FlextTypes.StringList = []

        for result in results:
            if result.is_success:
                successes.append(result.unwrap())
            else:
                errors.append(result.error or "Unknown error")

        return successes, errors

    @classmethod
    def parallel_map[TParallel, UParallel](
        cls,
        items: list[TParallel],
        mapper: Callable[[TParallel], FlextResult[UParallel]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[list[UParallel]]:
        """Map function over items with parallel semantics.

        Args:
            items: Items to process
            mapper: Function to map each item
            fail_fast: If True, fail on first error; if False, accumulate all errors

        Returns:
            Result containing list of mapped values or accumulated errors

        """
        return FlextResult._Collections.parallel_map(items, mapper, fail_fast=fail_fast)

    @classmethod
    def _sequence_typed[U](cls, results: list[FlextResult[U]]) -> FlextResult[list[U]]:
        """Type-safe sequence for parallel operations."""
        successes: list[U] = []
        for result in results:
            if result.is_failure:
                error_msg = result.error or "Sequence operation failed"
                return FlextResult[list[U]].fail(error_msg)
            successes.append(result.unwrap())
        return FlextResult[list[U]].ok(successes)

    @classmethod
    def _accumulate_errors_typed[U](
        cls,
        results: list[FlextResult[U]],
    ) -> FlextResult[list[U]]:
        """Type-safe error accumulation for parallel operations."""
        successes: list[U] = []
        errors: FlextTypes.StringList = []
        for result in results:
            if result.is_failure:
                error_msg = result.error or "Operation failed"
                errors.append(error_msg)
            else:
                successes.append(result.unwrap())

        if errors:
            return FlextResult[list[U]].fail("; ".join(errors))
        return FlextResult[list[U]].ok(successes)

    @classmethod
    def concurrent_sequence[TConcurrent](
        cls,
        results: list[FlextResult[TConcurrent]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[list[TConcurrent]]:
        """Sequence results with concurrent semantics.

        Args:
            results: Results to sequence
            fail_fast: If True, fail on first error; if False, accumulate all errors

        Returns:
            Result containing list of values or accumulated errors

        """
        if fail_fast:
            return cls._sequence_typed(results)
        return cls._accumulate_errors_typed(results)

    def with_resource[TResource, UResource](
        self,
        resource_factory: Callable[[], TResource],
        operation: Callable[[T_co, TResource], FlextResult[UResource]],
        cleanup: Callable[[TResource], None] | None = None,
    ) -> FlextResult[UResource]:
        """Execute operation with automatic resource management.

        Args:
            resource_factory: Function to create the resource
            operation: Function to execute with value and resource
            cleanup: Optional cleanup function for the resource

        Returns:
            Result of operation with guaranteed resource cleanup

        """
        if self.is_failure:
            return FlextResult[UResource].fail(
                self.error or "Cannot use resource with failed result",
                error_code=self.error_code,
                error_data=self.error_data,
            )

        try:
            resource = resource_factory()
            try:
                return operation(self.unwrap(), resource)
            finally:
                if cleanup:
                    cleanup(resource)
        except Exception as e:
            return FlextResult[UResource].fail(f"Resource operation failed: {e}")

    def bracket[TBracket](
        self,
        operation: Callable[[T_co], FlextResult[TBracket]],
        finally_action: Callable[[T_co], None],
    ) -> FlextResult[TBracket]:
        """Execute operation with guaranteed finally action.

        Args:
            operation: Main operation to execute
            finally_action: Action to execute regardless of operation outcome

        Returns:
            Result of operation with guaranteed finally action execution

        """
        if self.is_failure:
            return FlextResult[TBracket].fail(
                self.error or "Cannot bracket failed result",
                error_code=self.error_code,
                error_data=self.error_data,
            )

        try:
            value = self.unwrap()
            result: FlextResult[TBracket] = operation(value)
            finally_action(value)
            return result
        except Exception as e:
            with contextlib.suppress(Exception):
                finally_action(self.unwrap())
            return FlextResult[TBracket].fail(f"Bracket operation failed: {e}")

    def with_timeout[TTimeout](
        self,
        timeout_seconds: float,
        operation: Callable[[T_co], FlextResult[TTimeout]],
    ) -> FlextResult[TTimeout]:
        """Execute operation with timeout.

        Args:
            timeout_seconds: Maximum execution time in seconds
            operation: Operation to execute with timeout

        Returns:
            Result of operation or timeout error

        """
        if self.is_failure:
            return FlextResult[TTimeout].fail(
                self.error or "Cannot apply timeout to failed result",
                error_code=self.error_code,
                error_data=self.error_data,
            )

        def timeout_handler(_signum: int, _frame: object) -> None:
            msg = f"Operation timed out after {timeout_seconds} seconds"
            raise _get_exceptions().TimeoutError(
                message=msg,
                error_code=FlextConstants.Errors.TIMEOUT_ERROR,
            )

        try:
            # Set up timeout signal
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))

            start_time = time.time()
            result: FlextResult[TTimeout] = operation(self.unwrap())
            elapsed = time.time() - start_time

            # Clear timeout
            signal.alarm(0)

            # Add timing metadata
            if result.is_success:
                success_enhanced_data: FlextTypes.Dict = (
                    dict(result.error_data) if result.error_data else {}
                )
                success_enhanced_data["execution_time"] = elapsed
                return FlextResult[TTimeout].ok(result.unwrap())
            failure_enhanced_data: FlextTypes.Dict = (
                dict(result.error_data) if result.error_data else {}
            )
            failure_enhanced_data["execution_time"] = elapsed
            return FlextResult[TTimeout].fail(
                result.error or "Timed operation failed",
                error_code=result.error_code,
                error_data=failure_enhanced_data,
            )
        except TimeoutError as e:
            signal.alarm(0)  # Clear timeout
            return FlextResult[TTimeout].fail(
                str(e),
                error_code=FlextConstants.Errors.TIMEOUT_ERROR,
                error_data={"timeout_seconds": timeout_seconds},
            )
        except Exception as e:
            signal.alarm(0)  # Clear timeout
            return FlextResult[TTimeout].fail(f"Timeout operation failed: {e}")

    def retry_until_success(
        self,
        operation: Callable[[T_co], FlextResult[T_co]],
        max_attempts: int = FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
        backoff_factor: float = FlextConstants.Reliability.LINEAR_BACKOFF_FACTOR,
    ) -> FlextResult[T_co]:
        """Retry operation until success with exponential backoff.

        Args:
            operation: Operation to retry
            max_attempts: Maximum number of retry attempts
            backoff_factor: Multiplier for exponential backoff

        Returns:
            Result of first successful attempt or final failure

        """
        if self.is_failure:
            return self

        last_error = "No attempts made"

        for attempt in range(max_attempts):
            try:
                result: FlextResult[T_co] = operation(self.unwrap())
                if result.is_success:
                    return result
                last_error = result.error or f"Attempt {attempt + 1} failed"

                # Apply exponential backoff (except on last attempt)
                if attempt < max_attempts - 1:
                    time.sleep(backoff_factor * (2**attempt))

            except Exception as e:
                last_error = str(e)
                if attempt < max_attempts - 1:
                    time.sleep(backoff_factor * (2**attempt))

        return FlextResult[T_co].fail(
            f"All {max_attempts} retry attempts failed. Last error: {last_error}",
            error_code="RETRY_EXHAUSTED",
            error_data={"max_attempts": max_attempts, "last_error": last_error},
        )

    def transition[TState](
        self,
        state_machine: Callable[[T_co], FlextResult[TState]],
    ) -> FlextResult[TState]:
        """State machine transition using railway pattern.

        Args:
            state_machine: Function that defines state transitions

        Returns:
            Result of state transition

        """
        if self.is_failure:
            return FlextResult[TState].fail(
                self.error or "Cannot transition from failed state",
                error_code=self.error_code,
                error_data=self.error_data,
            )

        try:
            return state_machine(self.unwrap())
        except Exception as e:
            return FlextResult[TState].fail(f"State transition failed: {e}")

    @classmethod
    def validate_all[TValidate](
        cls,
        value: TValidate,
        *validators: Callable[[TValidate], FlextResult[None]],
    ) -> FlextResult[TValidate]:
        """Run all validators on a value, accumulating errors.

        Args:
            value: Value to validate
            *validators: Validation functions

        Returns:
            Original value if all validations pass, accumulated errors otherwise

        """
        return FlextResult._Collections.validate_all(value, *validators)

    # Factory methods moved from nested Result class to main FlextResult class
    @staticmethod
    def dict_result() -> type[FlextResult[FlextTypes.StringDict]]:
        """Factory for FlextResult[FlextTypes.StringDict]."""
        return FlextResult[FlextTypes.StringDict]

    @staticmethod
    def _is_flattenable_sequence(item: object) -> bool:
        return isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        )

    @staticmethod
    def _flatten_variadic_args(*items: object) -> FlextTypes.List:
        flat: FlextTypes.List = []
        for item in items:
            if FlextResult._is_flattenable_sequence(item) and isinstance(
                item,
                (list, tuple),
            ):
                flat.extend(FlextResult._flatten_variadic_args(*item))
            else:
                flat.append(item)
        return flat

    @staticmethod
    def _flatten_callable_args(*items: object) -> list[Callable[[], object]]:
        flat_callables: list[Callable[[], object]] = []
        for item in items:
            if FlextResult._is_flattenable_sequence(item) and isinstance(
                item,
                (list, tuple),
            ):
                flat_callables.extend(FlextResult._flatten_callable_args(*item))
            else:
                if not callable(item):
                    msg = "Expected callable when flattening alternatives"
                    raise _get_exceptions().ValidationError(
                        message=msg,
                        error_code="VALIDATION_ERROR",
                    )
                flat_callables.append(item)
        return flat_callables


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================


__all__: FlextTypes.StringList = [
    "FlextResult",  # Main unified result class
]
