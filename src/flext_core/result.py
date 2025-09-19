"""Railway-oriented result type with type-safe composition semantics.

Provides the canonical success/failure wrapper for FLEXT-Core 1.0.0,
including explicit error metadata and backward-compatible `.value`/`.data`
accessors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import TypeGuard, cast, overload, override

from flext_core.constants import FlextConstants
from flext_core.typings import T1, T2, T3, FlextTypes, T_co, TItem, TResult, TUtil, U, V

# =============================================================================
# FLEXT RESULT
# =============================================================================


class FlextResult[T_co]:
    """Foundation result type that powers FLEXT's railway pattern.

    The implementation mirrors the behaviour promised for the 1.0.0 release:
    explicit success/failure states, functional composition helpers, and
    ergonomic metadata for telemetry. It backs every service contract inside
    the FLEXT ecosystem and is guaranteed stable throughout the 1.x line.
    """

    # Removed performance optimization duplications - use FlextUtilities instead

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
        error_data: FlextTypes.Core.Dict | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        data: T_co | None = None,
        error: str | None = None,
        error_code: str | None = None,
        error_data: FlextTypes.Core.Dict | None = None,
    ) -> None:
        """Initialize result with either success data or error."""
        # Architectural invariant: exactly one of data or error must be provided.
        if error is not None:
            # Failure path: ensure data is None for type consistency.
            self._data: T_co | None = None
            self._error: str | None = error
        else:
            # Success path: data can be T_co (including None if T_co allows it).
            self._data = data
            self._error = None

        self._error_code = error_code
        self._error_data: FlextTypes.Core.Dict = error_data or {}

    def _is_success_state(self, value: T_co | None) -> TypeGuard[T_co]:
        """Type guard for success state checking."""
        return self._error is None and value is not None

    def _ensure_success_data(self) -> T_co:
        """Ensure success data is available or raise RuntimeError."""
        if self._data is None:
            msg = "Success result has None data - this should not happen"
            raise RuntimeError(msg)
        return self._data

    @property
    def is_success(self) -> bool:
        """Return ``True`` when the result carries a successful payload."""
        return self._error is None

    @property
    def success(self) -> bool:
        """Legacy name that mirrors :attr:`is_success` for 1.x support."""
        return self._error is None

    @property
    def is_failure(self) -> bool:
        """Return ``True`` when the result represents a failure."""
        return self._error is not None

    @property
    def value(self) -> T_co:
        """Return the success payload, raising :class:`TypeError` on failure."""
        if self.is_failure:
            msg = "Attempted to access value on failed result"
            raise TypeError(msg)
        return cast("T_co", self._data)

    @property
    def data(self) -> T_co | None:
        """Return the success payload, preserving ``None`` for legacy callers."""
        if self.is_success:
            return self.value
        return None

    @property
    def error(self) -> str | None:
        """Return the captured error message for failure results."""
        return self._error

    @property
    def error_code(self) -> str | None:
        """Return the structured error code supplied on failure."""
        return self._error_code

    @property
    def error_data(self) -> FlextTypes.Core.Dict:
        """Return the structured error metadata dictionary for observability."""
        return self._error_data

    @classmethod
    def ok(cls: type[FlextResult[T_co]], data: T_co) -> FlextResult[T_co]:
        """Create a successful result matching the 1.0.0 API contract.

        Args:
            data: The successful data to wrap in the result.

        Returns:
            A successful FlextResult containing the provided data.

        """
        return cls(data=data)

    # Note: Classmethod `success()` removed to avoid name collision with
    # the instance property `success`. Use `ok()` instead.

    @classmethod
    def fail(
        cls: type[FlextResult[T_co]],
        error: str,
        /,
        *,
        error_code: str | None = None,
        error_data: FlextTypes.Core.Dict | None = None,
    ) -> FlextResult[T_co]:
        """Create a failure result with optional error metadata.

        Args:
            error: The error message describing the failure.
            error_code: Optional error code for categorization.
            error_data: Optional additional error data/metadata.

        Returns:
            A failed FlextResult with the provided error information.

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
    def chain_results(
        *results: FlextResult[object],
    ) -> FlextResult[FlextTypes.Core.List]:
        """Collect a series of results, aborting on the first failure."""
        if not results:
            return FlextResult[FlextTypes.Core.List].ok([])
        aggregated: FlextTypes.Core.List = []
        for res in results:
            if res.is_failure:
                return FlextResult[FlextTypes.Core.List].fail(res.error or "error")
            aggregated.append(res.value)
        return FlextResult[FlextTypes.Core.List].ok(aggregated)

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
            data = self._ensure_success_data()
            result = func(data)
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

        # Safety check: ensure data is not None when result is success
        if self._data is None:
            return FlextResult[U](
                error="Unexpected chaining error: Internal error: data is None when result is success",
                error_code=FlextConstants.Errors.CHAIN_ERROR,
                error_data={"internal_inconsistency": True},
            )

        try:
            # Apply function to data using discriminated union type narrowing
            # Python 3.13+ discriminated union: _data is guaranteed to be T_co for success
            data = self._data
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
        raise IndexError(msg)

    def __or__(self, default: T_co) -> T_co:
        """Use | operator for default values: result | default_value."""
        if self.is_success:
            if self._data is None:
                return default  # Handle None data case
            return self._data
        return default

    def __enter__(self) -> T_co:
        """Context manager entry - returns value or raises on error."""
        if self.is_failure:
            error_msg = self._error or "Context manager failed"
            raise RuntimeError(error_msg)

        if self._data is None:
            msg = "Success result has None data - this should not happen"
            raise RuntimeError(msg)
        return self._data

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
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

        # Cast other to FlextResult[object] to help type checker
        other_result = cast("FlextResult[object]", other)

        try:
            # Cast to help type checker with comparison
            self_data = cast("object", self._data)
            other_data = other_result._data  # No cast needed after explicit typing
            return bool(
                self_data == other_data
                and self._error == other_result._error
                and self._error_code == other_result._error_code
                and self._error_data == other_result._error_data,
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

    def or_else_get(self, func: Callable[[], FlextResult[T_co]]) -> FlextResult[T_co]:
        """Return this result if successful, otherwise return result of func."""
        if self.is_success:
            return self
        try:
            return func()
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[T_co].fail(str(e))

    def unwrap_or(self, default: T_co) -> T_co:
        """Return value or default if failed."""
        if self.is_success:
            return cast("T_co", self._data)
        return default

    def unwrap(self) -> T_co:
        """Get value or raise if failed."""
        if self.is_success:
            return cast("T_co", self._data)
        raise RuntimeError(self._error or "Operation failed")

    def recover(self, func: Callable[[str], T_co]) -> FlextResult[T_co]:
        """Recover from failure by applying func to error."""
        if self.is_success:
            return self
        try:
            if self._error is not None:
                recovered_data = func(self._error)
                return FlextResult[T_co].ok(recovered_data)
            return FlextResult[T_co].fail("No error to recover from")
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[T_co].fail(str(e))

    def recover_with(
        self, func: Callable[[str], FlextResult[T_co]],
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
            if self._data is None:
                msg = "Success result has None data - this should not happen"
                raise RuntimeError(msg)
            if predicate(self._data):
                return self
            return FlextResult[T_co].fail(error_msg)
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[T_co].fail(str(e))

    def zip_with(
        self,
        other: FlextResult[U],
        func: Callable[[T_co, U], object],
    ) -> FlextResult[object]:
        """Combine two results with a function."""
        if self.is_failure:
            return FlextResult[object].fail(self._error or "First result failed")
        if other.is_failure:
            return FlextResult[object].fail(other._error or "Second result failed")

        # Check for None data - treat as missing data
        if self._data is None or other._data is None:
            return FlextResult[object].fail("Missing data for zip operation")

        # Both data values are not None, proceed with operation
        try:
            result = func(self._data, other._data)
            return FlextResult[object].ok(result)
        except (TypeError, ValueError, AttributeError, ZeroDivisionError) as e:
            return FlextResult[object].fail(str(e))

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
        return RuntimeError(error_msg)

    @classmethod
    def from_exception(cls, func: Callable[[], T_co]) -> FlextResult[T_co]:
        """Create a result from a function that might raise exception."""
        try:
            return cls.ok(func())
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return cls.fail(str(e))

    @staticmethod
    def combine(*results: FlextResult[object]) -> FlextResult[FlextTypes.Core.List]:
        """Combine multiple results into one."""
        data: FlextTypes.Core.List = []
        for result in results:
            if result.is_failure:
                return FlextResult[FlextTypes.Core.List].fail(
                    result.error or "Combine failed",
                )
            if result.value is not None:
                data.append(result.value)
        return FlextResult[FlextTypes.Core.List].ok(data)

    @staticmethod
    def all_success(*results: FlextResult[object]) -> bool:
        """Check success condition across results.

        Implementation detail aligned with test expectations:
        - Returns True when no results are provided (vacuous truth).
        - Returns True only if all results are successful.
        """
        if not results:
            return True
        return all(result.success for result in results)

    @staticmethod
    def any_success(*results: FlextResult[object]) -> bool:
        """Check if any result succeeded.

        For test compatibility: False for empty input, True otherwise.
        """
        return any(result.success for result in results) if results else False

    @classmethod
    def first_success(cls, *results: FlextResult[T_co]) -> FlextResult[T_co]:
        """Return first successful result."""
        last_error = "No successful results found"
        for result in results:
            if result.is_success:
                return result
            last_error = result.error or "Unknown error"
        return cls.fail(last_error)

    @classmethod
    def sequence(cls, results: list[FlextResult[T_co]]) -> FlextResult[list[T_co]]:
        """Convert list of results to result of list, failing on first failure.

        Args:
            results: List of FlextResult instances to sequence

        Returns:
            FlextResult containing list of all values if all successful,
            or first failure encountered.

        """
        values: list[T_co] = []
        for result in results:
            if result.is_failure:
                return FlextResult[list[T_co]].fail(
                    result.error or "Sequence failed",
                    error_code=result.error_code,
                    error_data=result.error_data,
                )
            values.append(result.unwrap())
        return FlextResult[list[T_co]].ok(values)

    @classmethod
    def try_all(cls, *funcs: Callable[[], T_co]) -> FlextResult[T_co]:
        """Try functions until one succeeds."""
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
    def safe_unwrap_or_none(cls, result: FlextResult[T_co]) -> T_co | None:
        """Unwrap value or None if failed."""
        return result.value if result.success else None

    @classmethod
    def unwrap_or_raise(
        cls,
        result: FlextResult[TUtil],
        exception_type: type[Exception] = RuntimeError,
    ) -> TUtil:
        """Unwrap or raise exception."""
        if result.success:
            return result.value
        raise exception_type(result.error or "Operation failed")

    @classmethod
    def collect_successes(cls, results: list[FlextResult[TUtil]]) -> list[TUtil]:
        """Collect successful values from results."""
        return [r.value for r in results if r.success]

    @classmethod
    def collect_failures(cls, results: list[FlextResult[TUtil]]) -> list[str]:
        """Collect error messages from failures."""
        return [r.error for r in results if r.is_failure and r.error]

    @classmethod
    def success_rate(cls, results: list[FlextResult[TUtil]]) -> float:
        """Calculate success rate percentage."""
        if not results:
            return 0.0
        successes = sum(1 for r in results if r.success)
        return (successes / len(results)) * 100.0

    @classmethod
    def batch_process(
        cls,
        items: list[TItem],
        processor: Callable[[TItem], FlextResult[TUtil]],
    ) -> tuple[list[TUtil], list[str]]:
        """Process batch and separate successes from failures."""
        results = [processor(item) for item in items]
        successes = cls.collect_successes(results)
        failures = cls.collect_failures(results)
        return successes, failures

    @classmethod
    def safe_call(
        cls: type[FlextResult[T_co]], func: Callable[[], T_co],
    ) -> FlextResult[T_co]:
        """Execute function safely, wrapping result."""
        try:
            return FlextResult[T_co].ok(func())
        except Exception as e:
            return FlextResult[T_co].fail(str(e))

    # === MONADIC COMPOSITION ADVANCED OPERATORS (Python 3.13) ===

    def __rshift__(self, func: Callable[[T_co], FlextResult[U]]) -> FlextResult[U]:
        """Right shift operator (>>) for monadic bind - ADVANCED COMPOSITION."""
        return self.flat_map(func)

    def __lshift__(self, func: Callable[[T_co], U]) -> FlextResult[U]:
        """Left shift operator (<<) for functor map - ADVANCED COMPOSITION."""
        return self.map(func)

    def __matmul__(self, other: FlextResult[U]) -> FlextResult[tuple[T_co, U]]:
        """Matrix multiplication operator (@) for applicative combination - ADVANCED COMPOSITION."""
        if self.is_failure:
            return FlextResult[tuple[T_co, U]].fail(self.error or "Left operand failed")
        if other.is_failure:
            return FlextResult[tuple[T_co, U]].fail(
                other.error or "Right operand failed",
            )

        # Both successful - combine values
        return FlextResult[tuple[T_co, U]].ok((self.unwrap(), other.unwrap()))

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
            return FlextResult[T_co].fail("Predicate validation failed")
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
        results: list[TResult] = []

        for item in items:
            result = func(item)
            if result.is_failure:
                return FlextResult[list[TResult]].fail(
                    result.error or f"Traverse failed at item {item}",
                )
            results.append(result.unwrap())

        return FlextResult[list[TResult]].ok(results)

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
    def applicative_lift2(
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
    def applicative_lift3(
        cls,
        func: Callable[[T1, T2, T3], TResult],
        result1: FlextResult[T1],
        result2: FlextResult[T2],
        result3: FlextResult[T3],
    ) -> FlextResult[TResult]:
        """Lift ternary function to applicative context - ADVANCED APPLICATIVE PATTERN."""

        def lift_func(t1_t2: tuple[T1, T2] | None, t3: T3) -> TResult:
            if t1_t2 is None:
                # This should not happen in practice due to applicative lifting
                msg = "Unexpected None value in applicative lift"
                raise ValueError(msg)
            return func(t1_t2[0], t1_t2[1], t3)

        return cls.applicative_lift2(
            lift_func,
            result1 @ result2,
            result3,
        )

    class Result:
        """Result factory methods and types."""

        @staticmethod
        def dict_result() -> type:
            """Factory for FlextResult[dict[str, object]]."""
            return FlextResult[dict[str, object]]

        type Success = object  # Generic success type without FlextResult dependency


__all__: list[str] = [
    "FlextResult",  # Main unified result class
]
