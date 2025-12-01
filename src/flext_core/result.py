"""Railway-oriented result type for dispatcher-driven applications.

FlextResult wraps outcomes with explicit success/failure states so dispatcher
handlers, services, and middleware can compose operations without exceptions.
It underpins CQRS flows with monadic helpers for predictable, typed pipelines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import types
from collections.abc import Callable, Mapping, Sequence
from typing import Self, cast

from returns.io import IO, IOFailure, IOResult, IOSuccess
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success

from flext_core.exceptions import FlextExceptions
from flext_core.protocols import FlextProtocols
from flext_core.typings import FlextTypes, T_co, U


class FlextResult[T_co]:  # noqa: PLR0904
    """Type-safe railway result with monadic helpers for CQRS pipelines.

    Use FlextResult to compose dispatcher handlers and domain services without
    raising exceptions, while preserving optional error codes and metadata for
    structured logging.
    """

    def __init__(
        self,
        _result: Result[T_co, str],
        error_code: str | None = None,
        error_data: Mapping[str, FlextTypes.GeneralValueType] | None = None,
    ) -> None:
        """Initialize FlextResult with internal Result and optional metadata."""
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
            return FlextResult(result._result.alt(func))

        @staticmethod
        def lash(
            result: FlextResult[T], func: Callable[[str], FlextResult[T]]
        ) -> FlextResult[T]:
            """Apply recovery function on failure."""

            def inner(error: str) -> Result[T, str]:
                recovery = func(error)
                return recovery.result

            return FlextResult(result._result.lash(inner))

    class Convert[T]:
        """Conversion operations: to/from Maybe, IO, IOResult."""

        @staticmethod
        def to_io(result: FlextResult[T]) -> IO[T]:
            """Convert to returns.io.IO."""
            if result.is_success:
                return IO(result.value)
            msg = "Cannot convert failure to IO"
            raise FlextExceptions.ValidationError(msg)

        @staticmethod
        def from_io_result(
            io_result: IOResult[FlextTypes.GeneralValueType, str],
        ) -> FlextResult[FlextTypes.GeneralValueType]:
            """Create from returns.io.IOResult."""
            try:
                if isinstance(io_result, IOSuccess):
                    value = io_result.unwrap()
                    return FlextResult.ok(value)
                if isinstance(io_result, IOFailure):
                    error = io_result.failure()
                    return FlextResult.fail(str(error))
                return FlextResult.fail(f"Invalid IO result type: {type(io_result)}")
            except Exception as e:
                return FlextResult.fail(f"Error processing IO result: {e}")

    class Operations[T]:
        """Utility operations: safe, traverse, accumulate_errors, parallel_map, with_resource."""

        @staticmethod
        def safe(
            func: FlextProtocols.VariadicCallable[T],
        ) -> FlextProtocols.VariadicCallable[FlextResult[T]]:
            """Decorator to wrap function in FlextResult."""

            def wrapper(
                *args: FlextTypes.FlexibleValue,
                **kwargs: FlextTypes.FlexibleValue,
            ) -> FlextResult[T]:
                try:
                    result = func(*args, **kwargs)
                    return FlextResult.ok(result)
                except Exception as e:
                    return FlextResult.fail(str(e))

            return wrapper

    @classmethod
    def ok(cls, value: T_co) -> FlextResult[T_co]:
        """Create successful result wrapping data."""
        if value is None:
            msg = "Cannot create success result with None value"
            raise ValueError(msg)
        return cls(Success(value))

    @classmethod
    def fail(
        cls,
        error: str,
        error_code: str | None = None,
        error_data: Mapping[str, FlextTypes.GeneralValueType] | None = None,
    ) -> FlextResult[T_co]:
        """Create failed result with error message."""
        return cls(Failure(error), error_code=error_code, error_data=error_data)

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
        """Get the success value."""
        if self.is_failure:
            msg = f"Cannot access value of failed result: {self.error}"
            raise RuntimeError(msg)
        return self._result.unwrap()

    @property
    def error(self) -> str | None:
        """Get the error message if failure."""
        if self.is_success:
            return None
        return self._result.failure()

    @property
    def error_code(self) -> str | None:
        """Get error code."""
        return getattr(self, "_error_code", None)

    @property
    def error_data(self) -> Mapping[str, FlextTypes.GeneralValueType] | None:
        """Get error metadata."""
        return getattr(self, "_error_data", None)

    def unwrap(self) -> T_co:
        """Unwrap the result value or raise RuntimeError."""
        if self.is_failure:
            msg = f"Cannot unwrap failed result: {self.error}"
            raise RuntimeError(msg)
        return self._result.unwrap()

    def unwrap_or(self, default: T_co) -> T_co:
        """Unwrap the result value or return default if failure.

        Args:
            default: Default value to return on failure

        Returns:
            The result value if success, otherwise the default

        """
        if self.is_failure:
            return default
        return self._result.unwrap()

    def map[U](self, func: Callable[[T_co], U]) -> FlextResult[U]:
        """Transform success value using function."""
        return FlextResult(self._result.map(func))

    def flat_map[U](self, func: Callable[[T_co], FlextResult[U]]) -> FlextResult[U]:
        """Chain operations returning FlextResult."""

        def inner(value: T_co) -> Result[U, str]:
            result = func(value)
            if result.is_success:
                return Success(result.value)
            return Failure(result.error or "")

        return FlextResult(self._result.bind(inner))

    def filter(self, predicate: Callable[[T_co], bool]) -> FlextResult[T_co]:
        """Filter success value using predicate."""
        if self.is_success and not predicate(self.value):
            return FlextResult.fail("Filter predicate failed")
        return self

    def flow_through[U](
        self,
        *funcs: Callable[[T_co | U], FlextResult[U]],
    ) -> FlextResult[U]:
        """Chain multiple operations in a pipeline."""
        # Start with current result, then apply each function in sequence
        current: FlextResult[T_co | U] = cast("FlextResult[T_co | U]", self)
        for func in funcs:
            if current.is_failure:
                return cast("FlextResult[U]", current)
            current = cast("FlextResult[T_co | U]", func(current.value))
        return cast("FlextResult[U]", current)

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

    def to_io_result(self) -> IOResult[T_co, str]:
        """Convert to returns.io.IOResult."""
        if self.is_success:
            return IOSuccess(self.value)
        return IOFailure(self.error or "")

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
                return FlextResult.fail(result.error or "Unknown error")
            results.append(result.value)
        return FlextResult(Success(results))

    @classmethod
    def accumulate_errors(cls, *results: FlextResult[U]) -> FlextResult[list[U]]:
        """Collect all successes, fail if any failure."""
        successes: list[U] = []
        errors = []
        for result in results:
            if result.is_success:
                successes.append(result.value)
            else:
                errors.append(result.error or "Unknown error")
        if errors:
            return FlextResult.fail("; ".join(errors))
        return FlextResult(Success(successes))

    @classmethod
    def parallel_map[T, U](
        cls,
        items: Sequence[T],
        func: Callable[[T], FlextResult[U]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[list[U]]:
        """Map with parallel processing and configurable failure handling."""
        results = [func(item) for item in items]
        if fail_fast:
            for result in results:
                if result.is_failure:
                    return FlextResult.fail(result.error or "Unknown error")
            return FlextResult(Success([r.value for r in results]))
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
        """String representation."""
        if self.is_success:
            return f"FlextResult.ok({self.value!r})"
        return f"FlextResult.fail({self.error!r})"


__all__ = ["FlextResult"]
