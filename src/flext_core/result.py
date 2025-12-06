"""Railway-oriented result type for dispatcher-driven applications.

FlextResult wraps outcomes with explicit success/failure states so dispatcher
handlers, services, and middleware can compose operations without exceptions.
It underpins CQRS flows with monadic helpers for predictable, typed pipelines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import types
from collections.abc import Callable, Sequence
from typing import Self, cast

from returns.io import IO, IOFailure, IOResult, IOSuccess
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success

from flext_core.exceptions import FlextExceptions as e
from flext_core.protocols import p
from flext_core.typings import U, t


class FlextResult[T_co]:
    """Type-safe railway result with monadic helpers for CQRS pipelines.

    Use FlextResult to compose dispatcher handlers and domain services without
    raising exceptions, while preserving optional error codes and metadata for
    structured logging.

    TODO(docs/FLEXT_SERVICE_ARCHITECTURE.md#smart-resolution): expose an
    explicit ``and_then`` helper so the implementation matches the "Smart
    Resolution" flow described in the documentation. At the moment
    ``flat_map`` covers the scenario with slightly different syntax.
    """

    def __init__(
        self,
        _result: Result[T_co, str],
        error_code: str | None = None,
        error_data: t.Types.ConfigurationMapping | None = None,
    ) -> None:
        """Initialize FlextResult with internal Result and optional metadata."""
        super().__init__()
        self._result = _result
        self._error_code = error_code
        self._error_data = error_data

    # ═══════════════════════════════════════════════════════════════════
    # NESTED OPERATION GROUPS (Organization via Composition)
    # ═══════════════════════════════════════════════════════════════════

    class Monad[T]:
        """Monadic operations: map, flat_map, filter, alt, lash, flow_through."""

        @staticmethod
        def alt(result: FlextResult[T], func: Callable[[str], str]) -> FlextResult[T]:
            """Apply alternative function on failure."""
            # Type annotation: alt returns Result[T, str]
            alt_result: Result[T, str] = result.result.alt(func)
            return FlextResult[T](alt_result)

        @staticmethod
        def lash(
            result: FlextResult[T],
            func: Callable[[str], FlextResult[T]],
        ) -> FlextResult[T]:
            """Apply recovery function on failure."""

            def inner(error: str) -> Result[T, str]:
                recovery = func(error)
                return recovery.result

            # Type annotation: lash returns Result[T, str]
            # pyright: ignore[reportUnknownMemberType] - returns library method type inference
            lash_result: Result[T, str] = result.result.lash(inner)  # type: ignore[assignment]
            return FlextResult[T](lash_result)

    class Convert[T]:
        """Conversion operations: to/from Maybe, IO, IOResult."""

        @staticmethod
        def to_io(result: FlextResult[T]) -> IO[T]:
            """Convert to returns.io.IO."""
            if result.is_success:
                return IO(result.value)
            msg = "Cannot convert failure to IO"
            raise e.ValidationError(msg)

        @staticmethod
        def from_io_result(
            io_result: IOResult[t.GeneralValueType, str],
        ) -> FlextResult[t.GeneralValueType]:
            """Create from returns.io.IOResult."""
            try:
                if isinstance(io_result, IOSuccess):
                    # IOSuccess is a successful IOResult
                    # Access value using getattr with fallback (returns library internal structure)
                    try:
                        # Try to access value via internal attribute (returns library pattern)
                        # Use getattr to safely access _inner_value which stores the IO value
                        unwrapped_value_raw = getattr(io_result, "_inner_value", None)
                        if unwrapped_value_raw is None:
                            # Fallback: try value attribute if _inner_value doesn't exist
                            unwrapped_value_raw = getattr(io_result, "value", None)
                        if unwrapped_value_raw is None:
                            return FlextResult[t.GeneralValueType].fail(
                                "Cannot extract value from IOSuccess",
                            )
                        # Type narrowing: unwrapped_value_raw is the inner value type
                        # Convert to GeneralValueType with proper type narrowing
                        unwrapped_value: t.GeneralValueType
                        if isinstance(
                            unwrapped_value_raw,
                            (
                                str,
                                int,
                                float,
                                bool,
                                type(None),
                                dict,
                                list,
                                Success,
                                Failure,
                                IOSuccess,
                                IOFailure,
                            ),
                        ):
                            unwrapped_value = cast(
                                "t.GeneralValueType",
                                unwrapped_value_raw,
                            )
                        else:
                            unwrapped_value = cast(
                                "t.GeneralValueType",
                                str(unwrapped_value_raw),
                            )
                        return FlextResult[t.GeneralValueType].ok(unwrapped_value)
                    except Exception as unwrap_error:  # pragma: no cover
                        # Defensive exception handling - hard to test without complex mocking
                        # since IOSuccess is immutable
                        return FlextResult[t.GeneralValueType].fail(
                            f"Error processing IO result: {unwrap_error}",
                        )
                if isinstance(io_result, IOFailure):
                    # IOFailure stores error in _inner_value attribute
                    error = getattr(io_result, "_inner_value", "Unknown error")
                    return FlextResult[t.GeneralValueType].fail(str(error))
                return FlextResult[t.GeneralValueType].fail(
                    f"Invalid IO result type: {type(io_result)}",
                )
            except Exception as e:  # pragma: no cover
                # Defensive exception handling - hard to test without complex mocking
                return FlextResult[t.GeneralValueType].fail(
                    f"Error processing IO result: {e}",
                )

    class Operations[T]:
        """Utility operations: safe, traverse, accumulate_errors, parallel_map."""

        @staticmethod
        def safe(
            func: p.Utility.Callable[T],
        ) -> p.Utility.Callable[FlextResult[T]]:
            """Decorator to wrap function in FlextResult."""

            def wrapper(
                *args: t.GeneralValueType,
                **kwargs: t.GeneralValueType,
            ) -> FlextResult[T]:
                try:
                    result = func(*args, **kwargs)
                    return FlextResult[T].ok(result)
                except Exception as e:
                    return FlextResult[T].fail(str(e))

            return wrapper

    @classmethod
    def ok(cls, value: T_co) -> FlextResult[T_co]:
        """Create successful result wrapping data.

        Business Rule: Creates successful FlextResult wrapping value. Raises ValueError
        if value is None (None values are not allowed in success results). Uses returns
        library Success wrapper for internal representation. This is the primary factory
        method for success results in railway-oriented programming pattern.

        Audit Implication: Success result creation ensures audit trail completeness by
        tracking successful operations. All success results are created through this
        factory method, ensuring consistent result representation across FLEXT.

        Core implementation - runtime.py cannot import result.py to avoid circular dependencies.

        Args:
            value: Value to wrap in success result (must not be None)

        Returns:
            Successful FlextResult instance

        Raises:
            ValueError: If value is None

        """
        if value is None:
            msg = "Cannot create success result with None value"
            raise ValueError(msg)
        return cls(Success(value))

    @classmethod
    def fail(
        cls,
        error: str | None,
        error_code: str | None = None,
        error_data: t.Types.ConfigurationMapping | None = None,
    ) -> FlextResult[T_co]:
        """Create failed result with error message.

        Business Rule: Creates failed FlextResult with error message, optional error
        code, and optional error metadata. Converts None error to empty string for
        consistency. Uses returns library Failure wrapper for internal representation.
        This is the primary factory method for failure results in railway-oriented
        programming pattern.

        Audit Implication: Failure result creation ensures audit trail completeness by
        tracking failed operations with error codes and metadata. All failure results
        are created through this factory method, ensuring consistent error representation
        across FLEXT.

        Core implementation - runtime.py cannot import result.py to avoid circular dependencies.

        Args:
            error: Error message (None will be converted to empty string)
            error_code: Optional error code for categorization
            error_data: Optional error metadata

        Returns:
            Failed FlextResult instance

        """
        error_msg = error if error is not None else ""
        return cls(Failure(error_msg), error_code=error_code, error_data=error_data)

    @staticmethod
    def safe[T](
        func: p.Utility.Callable[T],
    ) -> p.Utility.Callable[FlextResult[T]]:
        """Decorator to wrap function in FlextResult.

        Catches exceptions and returns FlextResult.fail() on error.

        Example:
            @FlextResult.safe
            def risky_operation() -> int:
                return 42

        """
        # Type annotation: Operations.safe returns Callable[FlextResult[T]]
        # pyright: ignore[reportUnknownMemberType] - returns library method type inference limitation
        safe_result = FlextResult.Operations.safe(func)  # type: ignore[assignment]
        safe_wrapper: p.Utility.Callable[FlextResult[T]] = cast(
            "p.Utility.Callable[FlextResult[T]]", safe_result
        )
        return safe_wrapper

    @property
    def is_success(self) -> bool:
        """Check if result represents success."""
        return isinstance(self._result, Success)

    @property
    def is_failure(self) -> bool:
        """Check if result represents failure."""
        return isinstance(self._result, Failure)

    @property
    def result(self) -> Result[T_co, str]:
        """Access the internal Result[T_co, str] for advanced operations."""
        return self._result

    @property
    def value(self) -> T_co:
        """Get the success value.

        ARCHITECTURAL NOTE: FlextCore never returns None on success, so this
        property always returns a valid value when is_success is True. Use this
        property directly instead of the deprecated unwrap() method.

        Returns:
            The success value (never None)

        Raises:
            RuntimeError: If result is failure

        """
        if self.is_failure:
            msg = f"Cannot access value of failed result: {self.error}"
            raise RuntimeError(msg)
        # Direct access to internal Result's value - this is the canonical way
        # to get the value from a Success result
        return self._result.unwrap()

    @property
    def data(self) -> T_co:
        """Alias for value - backward compatibility with older API."""
        return self.value

    @property
    def error(self) -> str | None:
        """Get the error message if failure."""
        if self.is_success:
            return None
        return self._result.failure()

    @property
    def error_code(self) -> str | None:
        """Get error code."""
        return self._error_code

    @property
    def error_data(self) -> t.Types.ConfigurationMapping | None:
        """Get error metadata."""
        return self._error_data

    def unwrap(self) -> T_co:
        """Unwrap the result value or raise RuntimeError.

        .. deprecated:: 2025-01-XX
            Use :attr:`value` property directly instead. This method is deprecated
            as FlextCore never returns None on success, so direct property access
            is safer and more explicit.

        Returns:
            The success value

        Raises:
            RuntimeError: If result is failure

        """
        if self.is_failure:
            msg = f"Cannot unwrap failed result: {self.error}"
            raise RuntimeError(msg)
        # Use .value property directly instead of internal _result.unwrap()
        return self.value

    def unwrap_or(self, default: T_co) -> T_co:
        """Unwrap the result value or return default if failure.

        Args:
            default: Default value to return on failure

        Returns:
            The result value if success, otherwise the default

        """
        if self.is_failure:
            return default
        # Use .value property directly instead of internal _result.unwrap()
        return self.value

    def map[U](self, func: Callable[[T_co], U]) -> FlextResult[U]:
        """Transform success value using function."""
        return FlextResult[U](self._result.map(func))

    def flat_map[U](self, func: Callable[[T_co], FlextResult[U]]) -> FlextResult[U]:
        """Chain operations returning FlextResult."""

        def inner(value: T_co) -> Result[U, str]:
            result = func(value)
            if result.is_success:
                return Success(result.value)
            return Failure(result.error or "")

        # Type annotation: bind returns Result[U, str]
        # pyright: ignore[reportUnknownMemberType] - returns library method type inference limitation
        bind_result: Result[U, str] = self._result.bind(inner)  # type: ignore[assignment]
        return FlextResult[U](bind_result)

    def map_error(self, func: Callable[[str], str]) -> Self:
        """Transform error message on failure.

        Args:
            func: Function to transform error message

        Returns:
            New FlextResult with transformed error if failure, unchanged if success

        """
        if self.is_success:
            return self
        error_msg = self.error or ""
        transformed_error = func(error_msg)
        # Type narrowing: type(self).fail() returns Self
        return cast(
            "Self",
            type(self).fail(
                transformed_error,
                error_code=self.error_code,
                error_data=self.error_data,
            ),
        )

    def filter(self, predicate: Callable[[T_co], bool]) -> Self:
        """Filter success value using predicate."""
        if self.is_success and not predicate(self.value):
            # Type narrowing: type(self).fail() returns Self
            return cast("Self", type(self).fail("Filter predicate failed"))
        return self

    def flow_through[U](
        self,
        *funcs: Callable[[T_co | U], FlextResult[U]],
    ) -> FlextResult[U]:
        """Chain multiple operations in a pipeline."""
        # Start with self, which is FlextResult[T_co]
        result: FlextResult[T_co] | FlextResult[U] = self
        for func in funcs:
            if result.is_failure:
                break
            # Type narrowing: func returns FlextResult[U]
            func_result = func(result.value)
            # func_result is already FlextResult[U] from func signature
            result = func_result
        # Final type narrowing: result is now FlextResult[U] after all transformations
        return cast("FlextResult[U]", result)

    @classmethod
    def create_from_callable(
        cls,
        func: Callable[[], T_co],
        error_code: str | None = None,
    ) -> FlextResult[T_co]:
        """Create result from callable, catching exceptions."""
        try:
            value = func()
            if value is None:
                return cls.fail("Callable returned None", error_code=error_code)
            return cls.ok(value)
        except Exception as e:
            return cls.fail(str(e), error_code=error_code)

    def __or__(self, default: T_co) -> T_co:
        """Operator overload for default values."""
        return self.unwrap_or(default)

    def __bool__(self) -> bool:
        """Boolean conversion based on success state."""
        return self.is_success

    def to_maybe(self) -> Maybe[T_co]:
        """Convert to returns.maybe.Maybe."""
        if self.is_success:
            return Some(self.value)
        return Nothing

    @classmethod
    def from_maybe(
        cls,
        maybe: Maybe[T_co],
        error: str = "Value is Nothing",
    ) -> FlextResult[T_co]:
        """Create from returns.maybe.Maybe."""
        if isinstance(maybe, Some):
            return cls.ok(maybe.unwrap())
        return cls.fail(error)

    def alt(self, func: Callable[[str], str]) -> Self:
        """Apply alternative function on failure.

        Transforms the error message using the provided function.
        On success, returns self unchanged.

        Args:
            func: Function to transform error message

        Returns:
            return FlextResult[T_co] with transformed error on failure, unchanged on success

        """
        # Type narrowing: Monad.alt returns Self
        # pyright: ignore[reportUnknownMemberType] - returns library method type inference limitation
        alt_result_raw = FlextResult.Monad.alt(self, func)  # type: ignore[assignment]
        alt_result: Self = cast("Self", alt_result_raw)
        return alt_result

    def lash(self, func: Callable[[str], FlextResult[T_co]]) -> Self:
        """Apply recovery function on failure.

        On failure, calls the provided function with the error message
        to produce a new FlextResult (recovery attempt).
        On success, returns self unchanged.

        Args:
            func: Recovery function that returns a new FlextResult

        Returns:
            Result of recovery function on failure, unchanged on success

        """
        # Type narrowing: Monad.lash returns Self
        # pyright: ignore[reportUnknownMemberType] - returns library method type inference limitation
        lash_result_raw = FlextResult.Monad.lash(self, func)  # type: ignore[assignment]
        lash_result: Self = cast("Self", lash_result_raw)
        return lash_result

    def to_io(self) -> IO[T_co]:
        """Convert to returns.io.IO.

        Returns an IO wrapper around the success value.
        Raises ValidationError if the result is a failure.

        Returns:
            IO[T_co]: IO wrapper around the success value

        Raises:
            e.ValidationError: If result is failure

        """
        # Type annotation: Convert.to_io returns IO[T_co]
        # pyright: ignore[reportUnknownMemberType] - returns library method type inference limitation
        io_result_raw = FlextResult.Convert.to_io(self)  # type: ignore[assignment]
        io_result: IO[T_co] = cast("IO[T_co]", io_result_raw)
        return io_result

    def to_io_result(self) -> IOResult[T_co, str]:
        """Convert to returns.io.IOResult."""
        if self.is_success:
            return IOSuccess(self.value)
        return IOFailure(self.error or "")

    @classmethod
    def from_io_result(
        cls,
        io_result: IOResult[t.GeneralValueType, str],
    ) -> FlextResult[t.GeneralValueType]:
        """Create FlextResult from returns.io.IOResult.

        Args:
            io_result: IOResult to convert

        Returns:
            FlextResult representing the same success/failure state

        """
        return FlextResult.Convert.from_io_result(io_result)

    @classmethod
    def traverse[T, U](
        cls,
        items: Sequence[T],
        func: Callable[[T], FlextResult[U]],
    ) -> FlextResult[list[U]]:
        """Map over sequence with failure propagation."""
        results: list[U] = []
        for item in items:
            result = func(item)
            if result.is_failure:
                return FlextResult[list[U]].fail(result.error or "Unknown error")
            results.append(result.value)
        return FlextResult[list[U]](Success(results))

    @classmethod
    def accumulate_errors(cls, *results: FlextResult[U]) -> FlextResult[list[U]]:
        """Collect all successes, fail if any failure."""
        successes: list[U] = []
        errors: list[str] = []
        for result in results:
            if result.is_success:
                # Type annotation: result.value is U when is_success
                # pyright: ignore[reportUnknownMemberType] - result.value type inference limitation
                value: U = cast("U", result.value)  # type: ignore[arg-type]
                successes.append(value)  # type: ignore[arg-type]
            else:
                error_msg: str = result.error or "Unknown error"
                errors.append(error_msg)
        if errors:
            return FlextResult[list[U]].fail("; ".join(errors))
        # Type annotation: Success constructor returns Result[list[U], str]
        success_result: Result[list[U], str] = cast(
            "Result[list[U], str]", Success(successes)
        )
        return FlextResult[list[U]](success_result)

    @classmethod
    def parallel_map[T, U](
        cls,
        items: Sequence[T],
        func: Callable[[T], FlextResult[U]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[list[U]]:
        """Map with parallel processing and configurable failure handling."""
        # NOTE: Cannot use u.map() here due to circular import
        # (utilities.py -> _utilities/args.py -> result.py)
        results: list[FlextResult[U]] = [func(item) for item in items]
        if fail_fast:
            for result in results:
                if result.is_failure:
                    return FlextResult[list[U]].fail(result.error or "Unknown error")
            # Use .value property directly - FlextCore never returns None on success
            # Type narrowing: all results are success at this point
            values: list[U] = [result.value for result in results]
            return FlextResult[list[U]](Success(values))
        return cls.accumulate_errors(*results)

    @classmethod
    def with_resource[R](
        cls,
        factory: Callable[[], R],
        op: Callable[[R], FlextResult[T_co]],
        cleanup: Callable[[R], None] | None = None,
    ) -> FlextResult[T_co]:
        """Resource management with automatic cleanup."""
        resource = factory()
        try:
            return op(resource)
        finally:
            if cleanup:
                cleanup(resource)

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit."""

    def __repr__(self) -> str:
        """String representation using short alias 'r' for brevity."""
        if self.is_success:
            return f"r.ok({self.value!r})"
        return f"r.fail({self.error!r})"


# Short alias for FlextResult - assignment for runtime compatibility
# mypy handles generic class aliases correctly with this pattern
r = FlextResult

__all__ = ["FlextResult", "r"]
