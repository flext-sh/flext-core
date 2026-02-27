"""Type-safe result type for operations.

Provides success/failure handling with monadic helpers for composing
operations without exceptions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Protocol, Self, TypeIs, overload

from pydantic import BaseModel
from returns.io import IO, IOFailure, IOResult, IOSuccess
from returns.maybe import Maybe, Nothing, Some
from returns.primitives.exceptions import UnwrapFailedError
from returns.result import Failure, Result, Success

from flext_core.runtime import FlextRuntime
from flext_core.typings import T_Model, U, t

_module_logger = logging.getLogger(__name__)


class _RuntimeResultLike(Protocol):
    """Protocol for objects that look like RuntimeResult (duck typing)."""

    is_success: bool
    is_failure: bool


class FlextResult[T_co](FlextRuntime.RuntimeResult[T_co]):
    """Type-safe result with monadic helpers for operation composition.

    Provides success/failure handling with various conversion and operation
    methods for composing operations without exceptions.
    """

    _result: Result[T_co, str] | None

    def __init__(
        self,
        source: Result[T_co, str] | None = None,
        error_code: str | None = None,
        error_data: t.ConfigMap | None = None,
        *,
        value: T_co | None = None,
        error: str | None = None,
        is_success: bool = True,
    ) -> None:
        """Initialize FlextResult from value/error/is_success only (direct typing, no Result unwrap)."""
        if source is not None and value is None and error is None:
            self._result = source
            try:
                failure_value = source.failure()
            except UnwrapFailedError as exc:
                _module_logger.debug(
                    "Result source is success path during initialization",
                    exc_info=exc,
                )
                super().__init__(
                    value=source.unwrap(),
                    error_code=error_code,
                    error_data=error_data,
                    is_success=True,
                )
                return
            super().__init__(
                error_code=error_code,
                error_data=error_data,
                error=str(failure_value),
                is_success=False,
            )
            return

        self._result = source
        super().__init__(
            value=value,
            error=error,
            error_code=error_code,
            error_data=error_data,
            is_success=is_success,
        )

    @classmethod
    def ok[T](cls, value: T) -> FlextResult[T]:
        """Create successful result wrapping data using Python 3.13 advanced patterns.

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
        result = FlextResult[T](value=value, is_success=True)
        result._result = Success(value)
        return result

    @classmethod
    def fail[U](
        cls,
        error: str | None,
        error_code: str | None = None,
        error_data: t.ConfigMap | None = None,
        expected_type: type[U] | None = None,
    ) -> FlextResult[U]:
        """Create failed result with error message using Python 3.13 advanced patterns.

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
        _ = expected_type
        error_msg = error if error is not None else ""
        result = FlextResult[U](
            error_code=error_code,
            error_data=error_data,
            error=error_msg,
            is_success=False,
        )
        result._result = Failure(error_msg)
        return result

    @staticmethod
    def safe[T, **PFunc](
        func: Callable[PFunc, T],
    ) -> Callable[PFunc, FlextResult[T]]:
        """Decorator to wrap function in FlextResult.

        Catches exceptions and returns FlextResult.fail() on error.

        Example:
            @FlextResult.safe
            def risky_operation() -> int:
                return 42

        """

        def wrapper(*args: PFunc.args, **kwargs: PFunc.kwargs) -> FlextResult[T]:
            try:
                result = func(*args, **kwargs)
                return FlextResult[T].ok(result)
            except Exception as e:
                _module_logger.debug("FlextResult.safe callable failed", exc_info=e)
                return FlextResult[T].fail(str(e))

        return wrapper

    @property
    def result(self) -> Self:
        """Protocol compatibility: return self (same as RuntimeResult)."""
        return self

    @property
    def data(self) -> T_co:
        """Return success data alias for protocol compatibility."""
        return self.value

    @property
    def returns_result(self) -> Result[T_co, str]:
        """Access the internal returns library Result[T_co, str] for advanced operations."""
        if self._result is None:
            if self.is_success:
                self._result = Success(self.value)
            else:
                self._result = Failure(self.error or "")
        return self._result

    def to_maybe(self) -> Maybe[T_co]:
        """Convert result into Maybe, dropping failure details."""
        if self.is_success:
            return Some(self.value)
        return Nothing

    @classmethod
    def from_maybe[U](
        cls,
        maybe: Maybe[U],
        error_message: str = "No value",
    ) -> FlextResult[U]:
        """Build result from Maybe by mapping Nothing to failure."""
        if maybe is Nothing:
            return cls.fail(error_message)
        return cls.ok(maybe.unwrap())

    def to_io(self) -> IO[T_co]:
        """Convert successful value to IO; fail by raising validation error."""
        if self.is_failure:
            exception_module = __import__("flext_core.exceptions", fromlist=["e"])
            raise exception_module.e.ValidationError(
                self.error or "Cannot convert failure to IO"
            )
        return IO(self.value)

    def to_io_result(self) -> IOResult[T_co, str]:
        """Convert result to IOResult while preserving success/failure state."""
        if self.is_success:
            return IOSuccess(self.value)
        return IOFailure(self.error or "")

    @classmethod
    def from_io_result[U](
        cls,
        io_result: IOResult[U, str],
    ) -> FlextResult[U | IO[U]]:
        """Build result from IOResult by unwrapping value or failure message."""
        if not hasattr(io_result, "unwrap") or not hasattr(io_result, "failure"):
            return cls.fail("Invalid IOResult structure")
        try:
            io_value = io_result.unwrap()
            return cls.ok(io_value)
        except UnwrapFailedError as exc:
            _module_logger.debug("Failed to unwrap IOResult", exc_info=exc)
            io_error = io_result.failure()
            return cls.fail(str(io_error))

    # error_code and error_data properties are inherited from RuntimeResult

    # unwrap, unwrap_or, unwrap_or_else are inherited from RuntimeResult

    def map[U](self, func: Callable[[T_co], U]) -> FlextResult[U]:
        """Transform success value using function.

        Overrides RuntimeResult.map to use returns library for compatibility.
        """
        if self.is_success:
            try:
                mapped_value = func(self.value)
                return FlextResult[U](value=mapped_value, is_success=True)
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                RuntimeError,
            ) as e:
                _module_logger.debug("FlextResult.map callable failed", exc_info=e)
                return FlextResult[U](error=str(e), is_success=False)
        return FlextResult[U](error=self.error or "", is_success=False)

    def flat_map[U](
        self,
        func: Callable[[T_co], FlextRuntime.RuntimeResult[U]],
    ) -> FlextResult[U]:
        """Chain operations returning FlextResult.

        Applies func to the success value and returns the result directly.
        If this result is a failure, returns a new failure with the same error.

        Args:
            func: Function that takes the success value and returns a FlextResult.

        Returns:
            FlextResult[U]: The result from func on success, or failure propagated.

        Example:
            result = r[int].ok(42).flat_map(lambda x: r[str].ok(str(x)))

        """
        if self.is_success:
            inner_result = func(self.value)
            if inner_result.is_success:
                return FlextResult[U].ok(inner_result.value)
            return FlextResult[U].fail(inner_result.error or "")
        return FlextResult[U].fail(self.error or "")

    def and_then[U](
        self,
        func: Callable[[T_co], FlextRuntime.RuntimeResult[U]],
    ) -> FlextResult[U]:
        """Alias for ``flat_map`` to support railway naming conventions."""
        return self.flat_map(func)

    def recover(self, func: Callable[[str], T_co]) -> FlextResult[T_co]:
        """Recover from failure with fallback value.

        Overrides RuntimeResult.recover to return FlextResult for type consistency.
        """
        if self.is_success:
            return self
        fallback_value = func(self.error or "")
        return FlextResult[T_co].ok(fallback_value)

    def tap(self, func: Callable[[T_co], None]) -> FlextResult[T_co]:
        """Apply side effect to success value, return unchanged.

        Overrides RuntimeResult.tap to return FlextResult for type consistency.
        """
        if self.is_success and self.value is not None:
            func(self.value)
        return self

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
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            _module_logger.debug("Callable execution failed", exc_info=e)
            return cls.fail(str(e), error_code=error_code)

    # __or__, __bool__, __repr__, __enter__, __exit__ are inherited from RuntimeResult

    @classmethod
    def from_validation(
        cls: type[FlextResult[T_Model]],
        data: t.ConfigMapValue,
        model: type[T_Model],
    ) -> FlextResult[T_Model]:
        """Create result from Pydantic validation.

        Validates data against a Pydantic model and returns a successful result
        with the validated model, or a failure result with validation errors.

        Args:
            data: Data to validate.
            model: Pydantic model class to validate against.

        Returns:
            FlextResult[T]: Success with validated model, or failure with
                validation errors.

        """
        # Use model directly - validated result is guaranteed to be T_Model
        # since model_validate returns an instance of the model class
        # Note: T_Model is bound to BaseModel, so no runtime check needed
        try:
            validated = model.model_validate(data)
            return cls.ok(validated)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            _module_logger.debug("Model validation failed", exc_info=e)
            errors_fn = getattr(e, "errors", None)
            if callable(errors_fn):
                raw = errors_fn()
                error_msg = str(raw)
            else:
                error_msg = str(e)
            return FlextResult[T_Model].fail(f"Validation failed: {error_msg}")

    def to_model[U: BaseModel](self, model: type[U]) -> FlextResult[U]:
        """Convert successful value to Pydantic model.

        If the result is successful, attempts to convert the value to the
        specified Pydantic model. If the result is already a failure, returns
        it unchanged.

        Args:
            model: Pydantic model class to convert to.

        Returns:
            FlextResult[U]: Success with converted model, or failure with
                conversion errors.

        """
        if self.is_failure:
            return FlextResult.fail(self.error or "")
        try:
            converted = model.model_validate(self.value)
            return FlextResult.ok(converted)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            _module_logger.debug("Model conversion failed", exc_info=e)
            return FlextResult.fail(f"Model conversion failed: {e!s}")

    # alt and lash are inherited from RuntimeResult
    # But we override to return FlextResult for type consistency
    def alt(self, func: Callable[[str], str]) -> FlextResult[T_co]:
        """Apply alternative function on failure.

        Overrides RuntimeResult.alt to return FlextResult for type consistency.
        """
        if self.is_failure:
            transformed_error = func(self.error or "")
            return FlextResult[T_co].fail(
                transformed_error,
                error_code=self.error_code,
                error_data=self.error_data,
            )
        return self

    # Alias for alt - more intuitive name for error transformation
    map_error = alt

    def lash(
        self,
        func: Callable[[str], FlextRuntime.RuntimeResult[T_co]],
    ) -> FlextResult[T_co]:
        """Apply recovery function on failure.

        Applies func to the error message on failure and returns the result directly.
        If this result is a success, returns self unchanged.

        Args:
            func: Function that takes the error message and returns a RuntimeResult.

        Returns:
            FlextResult[T_co]: Self if success, or result from func on failure.

        Example:
            result = r[int].fail("error").lash(lambda e: r[int].ok(0))

        """
        if self.is_failure:
            inner_result = func(self.error or "")
            if inner_result.is_success:
                return FlextResult[T_co].ok(inner_result.value)
            return FlextResult[T_co].fail(inner_result.error or "")
        return self

    # Alias for lash - RFC-standard name for recovery
    or_else = lash

    def tap_error(self, func: Callable[[str], None]) -> Self:
        """Execute side effect on failure, return unchanged.

        Useful for logging or metrics on failure without affecting the result.
        """
        if self.is_failure:
            func(self.error or "")
        return self

    def unwrap_or(self, default: T_co) -> T_co:
        """Return value if success, otherwise return default.

        Safe way to extract value without raising exceptions.
        Replaces pattern: result.value if result.is_success else default

        Args:
            default: Value to return if result is a failure.

        Returns:
            The success value or the default.

        Example:
            # Extract value or use fallback
            name = result.unwrap_or("Unknown")

            # With None as default
            entry = result.unwrap_or(None)

        """
        if self.is_success and self.value is not None:
            return self.value
        return default

    def get_or_else(self, default: T_co) -> T_co:
        """Return value on success or default on failure.

        Alias for unwrap_or - provides Haskell/Scala-style naming.

        Args:
            default: Default value to return on failure.

        Returns:
            Value on success, or default on failure.

        Example:
            value = result.get_or_else("default")

        """
        if self.is_success and self.value is not None:
            return self.value
        return default

    @overload
    def map_or(
        self,
        default: None,
        func: None = None,
    ) -> T_co | None: ...

    @overload
    def map_or[U](
        self,
        default: U,
        func: None = None,
    ) -> T_co | U: ...

    @overload
    def map_or[U](
        self,
        default: U,
        func: Callable[[T_co], U],
    ) -> U: ...

    def map_or[U](
        self,
        default: U,
        func: Callable[[T_co], U] | None = None,
    ) -> U | T_co:
        """Map success value with function or return default.

        Applies func to value on success, returns default on failure.
        If func is None, returns value directly on success (or default on failure).
        This is the standard map_or pattern from functional programming.

        Args:
            default: Default value to return on failure.
            func: Optional function to apply to success value.

        Returns:
            Mapped value on success (or value itself if func is None), or default on failure.

        Example:
            # Transform value or get default
            length = result.map_or(0, len)  # Returns len(value) or 0

            # Get attribute or default
            name = result.map_or("Unknown", lambda u: u.name)

            # Get value or default (no func)
            value = result.map_or("default")  # Returns value or "default"

        """
        if self.is_success and self.value is not None:
            if func is not None:
                return func(self.value)
            # When func is None, return value directly
            # Type is preserved via overload: when func=None, return type is T_co
            return self.value
        return default

    def filter(
        self,
        predicate: Callable[[T_co], bool],
    ) -> FlextResult[T_co]:
        """Filter success value based on predicate.

        If successful and predicate returns True, returns self unchanged.
        If successful and predicate returns False, returns failure.
        If already a failure, returns self unchanged.

        Args:
            predicate: Function to test the success value.

        Returns:
            Self if predicate passes, failure otherwise.

        Example:
            # Filter entries with specific attribute
            result.filter(lambda entry: entry.has_attribute("cn"))

            # Chain with map for type narrowing
            result.filter(lambda v: User in v.__class__.__mro__).map(process_user)

        """
        if self.is_success and self.value is not None:
            if predicate(self.value):
                return self
            return FlextResult[T_co].fail("Value did not pass filter predicate")
        return self

    def fold[U](
        self,
        on_failure: Callable[[str], U],
        on_success: Callable[[T_co], U],
    ) -> U:
        """Catamorphism - reduce result to a single value.

        Applies on_success function if successful, on_failure function if failed.
        Always produces a value of type U, eliminating the Result wrapper.

        Note: Parameter order matches RuntimeResult.fold (on_failure, on_success)
        to satisfy Liskov Substitution Principle.

        Args:
            on_failure: Function to apply to error message.
            on_success: Function to apply to success value.

        Returns:
            Result of applying the appropriate function.

        Example:
            # Convert result to HTTP response
            response = result.fold(
                on_failure=lambda err: {"status": 400, "error": err},
                on_success=lambda user: {"status": 200, "data": user.model_dump()},
            )

            # Convert to message
            message = result.fold(
                on_failure=lambda e: f"Operation failed: {e}",
                on_success=lambda _: "Operation succeeded",
            )

        """
        if self.is_success and self.value is not None:
            return on_success(self.value)
        return on_failure(self.error or "")

    @classmethod
    def traverse[T, U](
        cls,
        items: Sequence[T],
        func: Callable[[T], FlextResult[U]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[list[U]]:
        """Map over sequence with configurable failure handling.

        Args:
            items: Sequence of items to process.
            func: Function that takes an item and returns FlextResult.
            fail_fast: If True (default), stop on first failure.
                      If False, collect all errors (like accumulate_errors).

        Returns:
            FlextResult containing list of values on success,
            or error(s) on failure.

        """
        if fail_fast:
            # Stop on first failure
            results: list[U] = []
            for item in items:
                result = func(item)
                if result.is_failure:
                    return FlextResult[list[U]].fail(result.error or "Unknown error")
                results.append(result.value)
            return FlextResult[list[U]](value=results, is_success=True)
        # Collect all errors
        all_results = [func(item) for item in items]
        return cls.accumulate_errors(*all_results)

    @classmethod
    def accumulate_errors(cls, *results: FlextResult[U]) -> FlextResult[list[U]]:
        """Collect all successes, fail if any failure with all errors combined."""
        successes: list[U] = []
        errors: list[str] = []
        for result in results:
            if result.is_success:
                successes.append(result.value)
            else:
                errors.append(result.error or "Unknown error")
        if errors:
            return FlextResult[list[U]].fail("; ".join(errors))
        return FlextResult[list[U]](value=successes, is_success=True)

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

    @classmethod
    def parallel_map[T, U2](
        cls,
        items: Sequence[T],
        func: Callable[[T], FlextResult[U2]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[list[U2]]:
        """Map function over items, collecting results.

        Args:
            items: Items to process
            func: Function returning FlextResult for each item
            fail_fast: If True, stop on first failure

        Returns:
            FlextResult with list of successes or combined errors

        """
        if fail_fast:
            results: list[U2] = []
            for item in items:
                result = func(item)
                if result.is_failure:
                    return FlextResult[list[U2]].fail(result.error or "")
                results.append(result.value)
            return FlextResult[list[U2]](value=results, is_success=True)
        # Collect all results and accumulate errors
        all_results = [func(item) for item in items]
        return cls.accumulate_errors(*all_results)

    def _protocol_name(self) -> str:
        """Return the protocol name for introspection.

        Satisfies BaseProtocol requirement for ResultLike protocol.
        """
        return "FlextResult"


r = FlextResult


def is_success_result(value: _RuntimeResultLike) -> TypeIs[FlextResult]:
    """Return ``True`` when value is a successful runtime result."""
    return isinstance(value, FlextRuntime.RuntimeResult) and value.is_success


def is_failure_result(value: _RuntimeResultLike) -> TypeIs[FlextResult]:
    """Return ``True`` when value is a failed runtime result."""
    return isinstance(value, FlextRuntime.RuntimeResult) and value.is_failure


__all__ = ["FlextResult", "is_failure_result", "is_success_result", "r"]
