"""Railway-oriented result type for dispatcher-driven applications.

FlextResult wraps outcomes with explicit success/failure states so dispatcher
handlers, services, and middleware can compose operations without exceptions.
It underpins CQRS flows with monadic helpers for predictable, typed pipelines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Self, cast, override

from pydantic import BaseModel
from returns.io import IO, IOFailure, IOResult, IOSuccess
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success

from flext_core.exceptions import FlextExceptions as e
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import U, t


class FlextResult[T_co](FlextRuntime.RuntimeResult[T_co]):
    """Type-safe railway result with monadic helpers for CQRS pipelines.

    Extends RuntimeResult with advanced functionality:
    - Integration with returns library (Result[T, str], Success, Failure)
    - Pydantic validation (from_validation, to_model)
    - Conversions (to_maybe, from_maybe, to_io, to_io_result, from_io_result)
    - Sequence operations (traverse, accumulate_errors, parallel_map)
    - Resource management (with_resource)
    - Nested operation groups (Monad, Convert, Operations)

    Use FlextResult to compose dispatcher handlers and domain services without
    raising exceptions, while preserving optional error codes and metadata for
    structured logging.

    TODO(docs/FLEXT_SERVICE_ARCHITECTURE.md#smart-resolution): expose an
    explicit ``and_then`` helper so the implementation matches the "Smart
    Resolution" flow described in the documentation. At the moment
    ``flat_map`` covers the scenario with slightly different syntax.
    """

    # Instance attribute type annotation for lazy Result creation
    _result: Result[T_co, str] | None

    def __init__(
        self,
        _result: Result[T_co, str] | None = None,
        error_code: str | None = None,
        error_data: t.Types.ConfigurationMapping | None = None,
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
            lash_result: Result[T, str] = result.result.lash(inner)
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
                        return cast(
                            "FlextResult[t.GeneralValueType]",
                            FlextResult.fail(
                                f"Error processing IO result: {unwrap_error}",
                            ),
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
            func: p.VariadicCallable[T],
        ) -> p.VariadicCallable[FlextResult[T]]:
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
    def ok[T](cls, value: T) -> FlextResult[T]:
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
        return FlextResult[T](Success(value))

    @classmethod
    @override
    def fail[T](
        cls,
        error: str | None,
        error_code: str | None = None,
        error_data: t.Types.ConfigurationMapping | None = None,
    ) -> FlextResult[T]:
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
        result = Failure(error_msg)
        return FlextResult[T](result, error_code=error_code, error_data=error_data)

    @staticmethod
    def safe[T](
        func: p.VariadicCallable[T],
    ) -> p.VariadicCallable[FlextResult[T]]:
        """Decorator to wrap function in FlextResult.

        Catches exceptions and returns FlextResult.fail() on error.

        Example:
            @FlextResult.safe
            def risky_operation() -> int:
                return 42

        """
        # Operations.safe returns p.VariadicCallable[FlextResult[T]]
        return FlextResult.Operations.safe(func)

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
        self, func: Callable[[T_co], FlextRuntime.RuntimeResult[U]]
    ) -> FlextResult[U]:
        """Chain operations returning FlextResult.

        Overrides RuntimeResult.flat_map to use returns library for compatibility.
        """
        if self.is_success:
            result = func(self.value)
            return cast("FlextResult[U]", result)
        return FlextResult[U](Failure(self.error or ""))

    def and_then[U](
        self, func: Callable[[T_co], FlextRuntime.RuntimeResult[U]]
    ) -> FlextResult[U]:
        """RFC-compliant alias for flat_map.

        This method provides an RFC-compliant name for flat_map, making the
        composition pattern more explicit and aligned with functional programming
        conventions.

        Args:
            func: Function that takes the success value and returns a new FlextResult.

        Returns:
            FlextResult[U]: New result from the function application.

        Example:
            >>> result = FlextResult.ok(5)
            >>> result.and_then(lambda x: FlextResult.ok(x * 2))
            FlextResult.ok(10)

        """
        return self.flat_map(func)

    # fold, tap_error, map_error, filter are inherited from RuntimeResult
    # But we override recover, tap, and flow_through to return FlextResult instead of RuntimeResult
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

    @override
    def flow_through[U](  # pyrefly: ignore[bad-override]
        self,
        *funcs: Callable[[T_co | U], FlextRuntime.RuntimeResult[U]],
    ) -> FlextResult[U]:
        """Chain multiple operations in sequence.

        Overrides RuntimeResult.flow_through to return FlextResult for type consistency.
        """
        # Call parent method and convert result to FlextResult
        parent_result: FlextRuntime.RuntimeResult[U] = super().flow_through(*funcs)
        if parent_result.is_success:
            value: U = parent_result.value  # parent_result.value is U when is_success
            return FlextResult[U].ok(value)
        return cast(
            "FlextResult[U]",
            FlextResult.fail(
                parent_result.error or "",
                error_code=getattr(parent_result, "error_code", None),
                error_data=getattr(parent_result, "error_data", None),
            ),
        )

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
    def from_validation[T](cls, data: object, model: type[T]) -> FlextResult[T]:
        """Create result from Pydantic validation.

        Validates data against a Pydantic model and returns a successful result
        with the validated model, or a failure result with validation errors.

        Args:
            data: Data to validate.
            model: Pydantic model class to validate against.

        Returns:
            FlextResult[T]: Success with validated model, or failure with
                validation errors.

        Example:
            >>> result = FlextResult.from_validation({"name": "John"}, UserModel)
            >>> if result.is_success:
            ...     user = result.value

        """
        # Check if model is a BaseModel subclass before calling model_validate
        if not issubclass(model, BaseModel):
            fail_result: FlextResult[str] = cls.fail(
                f"Type {model} is not a BaseModel subclass"
            )
            return cast("FlextResult[T]", fail_result)
        # After issubclass check, model is guaranteed to be BaseModel subclass
        # Cast to type[BaseModel] to help type checker understand model_validate is available
        model_typed: type[BaseModel] = cast("type[BaseModel]", model)
        try:
            validated: T = cast("T", model_typed.model_validate(data))
            # Cast validated to T_co to match FlextResult[T_co] signature
            validated_co: T_co = cast("T_co", validated)
            result = cls.ok(validated_co)
            return cast("FlextResult[T]", result)
        except Exception as e:
            # Extract error message from Pydantic ValidationError if available
            if hasattr(e, "errors") and callable(getattr(e, "errors", None)):
                error_msg = "; ".join(
                    f"{err.get('loc', [])}: {err.get('msg', '')}"
                    for err in getattr(e, "errors")()
                )
            else:
                error_msg = str(e)
            # Cast to help type checker understand that T_co == T in this context
            fail_result = cls.fail(f"Validation failed: {error_msg}")
            return cast("FlextResult[T]", fail_result)

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

        Example:
            >>> result = FlextResult.ok({"name": "John", "age": 30})
            >>> user_result = result.to_model(UserModel)

        """
        if self.is_failure:
            return FlextResult.fail(self.error or "")
        try:
            converted = model.model_validate(self.value)
            return FlextResult.ok(converted)
        except Exception as e:
            return FlextResult.fail(f"Model conversion failed: {e!s}")

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
            value = maybe.value
            if value is not None:
                return cls.ok(value)
        return cls.fail(error)

    # alt and lash are inherited from RuntimeResult
    # But we override to return FlextResult for type consistency
    def alt(self, func: Callable[[str], str]) -> Self:
        """Apply alternative function on failure.

        Overrides RuntimeResult.alt to return FlextResult for type consistency.
        """
        if self.is_failure:
            transformed_error = func(self.error or "")
            return cast(
                "Self",
                type(self).fail(
                    transformed_error,
                    error_code=self.error_code,
                    error_data=self.error_data,
                ),
            )
        return self

    def lash(
        self, func: Callable[[str], FlextRuntime.RuntimeResult[T_co]]
    ) -> FlextResult[T_co]:
        """Apply recovery function on failure.

        Overrides RuntimeResult.lash to return FlextResult for type consistency.
        """
        if self.is_failure:
            result = func(self.error or "")
            return cast("FlextResult[T_co]", result)
        return self

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
        return FlextResult.Convert.to_io(self)

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
                # Type narrowing: when is_success is True, result.value is U
                # FlextCore never returns None on success, so value is guaranteed to be U
                value = result.value
                successes.append(value)
            else:
                error_msg: str = result.error or "Unknown error"
                errors.append(error_msg)
        if errors:
            return FlextResult[list[U]].fail("; ".join(errors))
        # Success constructor returns Result[list[U], str]
        # Type narrowing: Success(successes) creates Result[list[U], str]
        return FlextResult[list[U]](Success(successes))

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

    # __enter__, __exit__, __repr__ are inherited from RuntimeResult


# Short alias for FlextResult - assignment for runtime compatibility
# mypy handles generic class aliases correctly with this pattern
r = FlextResult

__all__ = ["FlextResult", "r"]
