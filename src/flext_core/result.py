"""Type-safe result type for operations.

Provides success/failure handling with monadic helpers for composing
operations without exceptions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Self, TypeIs, TypeVar, cast

from pydantic import BaseModel
from returns.io import IO, IOFailure, IOResult, IOSuccess
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success

from flext_core.exceptions import FlextExceptions as e
from flext_core.protocols import FlextProtocols
from flext_core.runtime import FlextRuntime
from flext_core.typings import U, t

T = TypeVar("T")
T_BaseModel = TypeVar("T_BaseModel", bound=BaseModel)
E = TypeVar("E", default=str)


def is_success_result[T](result: FlextResult[T]) -> TypeIs[FlextResult[T]]:
    """Type guard that narrows to successful FlextResult."""
    return result.is_success and result.value is not None


def is_failure_result[T](result: FlextResult[T]) -> TypeIs[FlextResult[T]]:
    """Type guard that narrows to failed FlextResult."""
    return result.is_failure


class FlextResult[T_co](FlextRuntime.RuntimeResult[T_co]):
    """Type-safe result with monadic helpers for operation composition.

    Provides success/failure handling with various conversion and operation
    methods for composing operations without exceptions.
    """

    _result: Result[T_co, str] | None

    def __init__(
        self,
        _result: Result[T_co, str] | None = None,
        error_code: str | None = None,
        error_data: t.ConfigurationMapping | None = None,
        *,
        # RuntimeResult initialization parameters
        value: T_co | None = None,
        error: str | None = None,
        is_success: bool = True,
    ) -> None:
        """Initialize FlextResult with internal Result or RuntimeResult parameters.

        Supports two initialization modes:
        1. From returns library Result (legacy mode): _result is provided
        2. From RuntimeResult parameters (new mode): value/error/is_success provided

        Args:
            _result: Internal Result[T_co, str] from returns library (legacy mode)
            error_code: Optional error code for categorization
            error_data: Optional error metadata
            value: Success value (new mode, for RuntimeResult compatibility)
            error: Error message (new mode, for RuntimeResult compatibility)
            is_success: Success state (new mode, for RuntimeResult compatibility)

        """
        # If _result is provided, use legacy initialization
        if _result is not None:
            self._result = _result
            # Initialize RuntimeResult with values from _result
            if isinstance(_result, Success):
                super().__init__(
                    value=_result.unwrap(),
                    error_code=error_code,
                    error_data=error_data,
                    is_success=True,
                )
            elif isinstance(_result, Failure):
                super().__init__(
                    error=_result.failure() or "",
                    error_code=error_code,
                    error_data=error_data,
                    is_success=False,
                )
            else:
                # Fallback - should not happen
                super().__init__(
                    value=value,
                    error=error,
                    error_code=error_code,
                    error_data=error_data,
                    is_success=is_success,
                )
        else:
            # New mode: initialize from RuntimeResult parameters
            self._result = None  # Will be created lazily if needed
            # Initialize RuntimeResult parent
            super().__init__(
                value=value,
                error=error,
                error_code=error_code,
                error_data=error_data,
                is_success=is_success,
            )
            # Store error_code and error_data for FlextResult-specific access
            # (RuntimeResult already stores them, but we keep them for compatibility)

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

        # Use returns library Success wrapper for railway-oriented programming
        return FlextResult[T](Success(value))

    @classmethod
    def fail[T](
        cls,
        error: str | None,
        error_code: str | None = None,
        error_data: t.ConfigurationMapping | None = None,
    ) -> FlextResult[T]:
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
        error_msg = error if error is not None else ""

        # Use returns library Failure wrapper for railway-oriented programming
        result = Failure(error_msg)
        return FlextResult[T](result, error_code=error_code, error_data=error_data)

    @staticmethod
    def safe[T](
        func: FlextProtocols.VariadicCallable[T],
    ) -> FlextProtocols.VariadicCallable[FlextResult[T]]:
        """Decorator to wrap function in FlextResult.

        Catches exceptions and returns FlextResult.fail() on error.

        Example:
            @FlextResult.safe
            def risky_operation() -> int:
                return 42

        """

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

    @property
    def result(self) -> Result[T_co, str]:
        """Access the internal Result[T_co, str] for advanced operations.

        Creates Result from RuntimeResult state if _result is None (lazy creation).
        """
        if self._result is None:
            # Create Result from RuntimeResult state using public properties
            if self.is_success:
                self._result = Success(self.value)
            else:
                self._result = Failure(self.error or "")
        return self._result

    # error_code and error_data properties are inherited from RuntimeResult

    # unwrap, unwrap_or, unwrap_or_else are inherited from RuntimeResult

    def map[U](self, func: Callable[[T_co], U]) -> FlextResult[U]:
        """Transform success value using function.

        Overrides RuntimeResult.map to use returns library for compatibility.
        """
        if self.is_success:
            try:
                mapped_value = func(self.value)
                return FlextResult[U](Success(mapped_value))
            except Exception as e:
                return FlextResult[U](Failure(str(e)))
        return FlextResult[U](Failure(self.error or ""))

    def flat_map[U](
        self,
        func: Callable[[T_co], FlextRuntime.RuntimeResult[U]],
    ) -> FlextResult[U]:
        """Chain operations returning FlextResult.

        Overrides RuntimeResult.flat_map to properly handle RuntimeResult returns.
        The function may return RuntimeResult, FlextResult, or returns.Result.
        """
        if self.is_success:
            result = func(self.value)
            # Handle RuntimeResult/FlextResult (has is_success property)
            if hasattr(result, "is_success"):
                if result.is_success:
                    return FlextResult[U].ok(result.value)
                return FlextResult[U].fail(result.error or "")
            # Handle returns.Result (Success/Failure) with proper type narrowing
            if isinstance(result, Success):
                return FlextResult[U].ok(result.unwrap())
            if isinstance(result, Failure):
                return FlextResult[U].fail(result.failure())
            # Fallback for unknown types - should not happen in practice
            return FlextResult[U].fail("Unknown result type from flat_map function")
        return FlextResult[U](Failure(self.error or ""))

    def and_then[U](
        self,
        func: Callable[[T_co], FlextRuntime.RuntimeResult[U]],
    ) -> FlextResult[U]:
        """RFC-compliant alias for flat_map.

        This method provides an RFC-compliant name for flat_map, making the
        composition pattern more explicit and aligned with functional programming
        conventions.

        Args:
            func: Function that takes the success value and returns a new FlextResult.

        Returns:
            FlextResult[U]: New result from the function application.

        """
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
        except Exception as e:
            return cls.fail(str(e), error_code=error_code)

    # __or__, __bool__, __repr__, __enter__, __exit__ are inherited from RuntimeResult

    @classmethod
    def from_validation[T_BaseModel: BaseModel](
        cls, data: object, model: type[T_BaseModel]
    ) -> FlextResult[T_BaseModel]:
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
        # Check if model is a BaseModel subclass before calling model_validate
        if not issubclass(model, BaseModel):
            return FlextResult[T_BaseModel].fail(
                f"Type {model} is not a BaseModel subclass",
            )
        # Use model directly - validated result is guaranteed to be T_BaseModel
        # since model_validate returns an instance of the model class
        try:
            validated = model.model_validate(data)
            # validated is instance of model which is T_BaseModel
            return cls.ok(validated)
        except Exception as e:
            # Extract error message from Pydantic ValidationError if available
            if hasattr(e, "errors") and callable(getattr(e, "errors", None)):
                error_msg = "; ".join(
                    f"{err.get('loc', [])}: {err.get('msg', '')}" for err in e.errors()
                )
            else:
                error_msg = str(e)
            return FlextResult[T_BaseModel].fail(f"Validation failed: {error_msg}")

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
        except Exception as e:
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

        Overrides RuntimeResult.lash to return FlextResult for type consistency.
        The function may return RuntimeResult, FlextResult, or returns.Result.
        """
        if self.is_failure:
            result = func(self.error or "")
            # Handle RuntimeResult/FlextResult (has is_success property)
            if hasattr(result, "is_success"):
                if result.is_success:
                    return FlextResult[T_co].ok(result.value)
                return FlextResult[T_co].fail(result.error or "")
            # Handle returns.Result (Success/Failure) with proper type narrowing
            if isinstance(result, Success):
                return FlextResult[T_co].ok(result.unwrap())
            if isinstance(result, Failure):
                return FlextResult[T_co].fail(result.failure())
            # Fallback for unknown types - should not happen in practice
            return FlextResult[T_co].fail("Unknown result type from lash function")
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

    def map_or[U](
        self,
        default: U,
        func: Callable[[T_co], U] | None = None,
    ) -> U:
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
            # INTENTIONAL CAST: When func is None, this is a type-erasing pattern
            # where the caller is responsible for ensuring T_co is assignable to U.
            # This pattern is widely used (75+ usages) for cases like:
            #   result.map_or(None)  - returns value | None
            #   result.map_or({})    - returns value | dict
            # The type system cannot express "T_co == U when func is None", so
            # cast is necessary. Alternative would be split into separate methods.
            return cast("U", self.value)
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
            result.filter(lambda v: isinstance(v, User)).map(process_user)

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
                on_success=lambda user: {"status": 200, "data": user.dict()},
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
            return FlextResult[list[U]](Success(results))
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
        return FlextResult[list[U]](Success(successes))

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
            return FlextResult[list[U2]](Success(results))
        # Collect all results and accumulate errors
        all_results = [func(item) for item in items]
        return cls.accumulate_errors(*all_results)

    def to_io(self) -> IO[T_co]:
        """Convert to returns.io.IO.

        Returns IO wrapping the success value.
        Raises ValidationError on failure.
        """
        if self.is_failure:
            raise e.ValidationError(self.error or "Result is failure")
        return IO(self.value)

    def to_io_result(self) -> IOResult[T_co, str]:
        """Convert to returns.io.IOResult.

        Returns IOResult[T, str] - success wraps value, failure wraps error.
        """
        if self.is_success:
            return IOSuccess(self.value)
        return IOFailure(self.error or "Unknown error")

    def to_maybe(self) -> Maybe[T_co]:
        """Convert to returns.maybe.Maybe.

        Returns Some(value) on success, Nothing on failure.
        """
        if self.is_success:
            return Some(self.value)
        return Nothing

    @classmethod
    def from_maybe[T](
        cls,
        maybe: Maybe[T],
        error: str = "Value was Nothing",
    ) -> FlextResult[T]:
        """Create FlextResult from returns.maybe.Maybe.

        Args:
            maybe: Maybe value to convert
            error: Error message if maybe is Nothing

        Returns:
            FlextResult[T]: Ok(value) if Some, Fail(error) if Nothing

        """
        if isinstance(maybe, Some):
            return FlextResult[T].ok(maybe.unwrap())
        return FlextResult[T].fail(error)

    @classmethod
    def from_io_result[T](
        cls,
        io_result: IOResult[T, str],
    ) -> FlextResult[T]:
        """Create FlextResult from returns.io.IOResult.

        The returns library's IOResult has a nested structure:
        - IOSuccess wraps Success(Success(value))
        - IOFailure wraps Failure(error)

        We need to unwrap the nested Success wrappers to get the actual value.

        Args:
            io_result: IOResult to convert

        Returns:
            FlextResult[T]: Ok(value) if IOSuccess, Fail(error) if IOFailure

        """
        if isinstance(io_result, IOFailure):
            # IOFailure.failure() returns IO[E], we need to extract the actual error
            io_wrapped = io_result.failure()
            # Access _inner_value to get the actual error (returns lib has no public API)
            error_val = getattr(io_wrapped, "_inner_value", None)
            error_msg = str(error_val) if error_val is not None else "Unknown error"
            return FlextResult[T].fail(error_msg)
        # IOSuccess case - extract value from nested Success wrappers
        # Structure: IOSuccess._inner_value = Success(Success(value))
        # Note: _inner_value access is required - returns library has no public API
        inner_result = getattr(io_result, "_inner_value", None)
        if inner_result is None:
            return FlextResult[T].fail("Invalid IOResult structure")
        # Unwrap first Success layer
        if isinstance(inner_result, Success):
            inner_value = inner_result.unwrap()
            # Unwrap second Success layer if present
            if isinstance(inner_value, Success):
                return FlextResult[T].ok(inner_value.unwrap())
            return FlextResult[T].ok(inner_value)
        return FlextResult[T].fail("Unexpected IOResult inner type")

    # __enter__, __exit__, __repr__ are inherited from RuntimeResult


# Short alias for FlextResult - assignment for runtime compatibility
# mypy handles generic class aliases correctly with this pattern
r = FlextResult

__all__ = ["FlextResult", "r"]
