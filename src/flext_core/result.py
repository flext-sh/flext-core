"""Type-safe result type for operations.

Provides success/failure handling with monadic helpers for composing
operations without exceptions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from collections.abc import Callable, MutableSequence, Sequence
from typing import Self, TypeIs, overload, override

from pydantic import BaseModel, PrivateAttr, ValidationError
from returns.primitives.exceptions import UnwrapFailedError
from returns.result import Failure, Result, Success

from flext_core import FlextRuntime, T_Model, U, t


class FlextResult[T](FlextRuntime.RuntimeResult[T]):
    """Type-safe result with monadic helpers for operation composition.

    Provides success/failure handling with various conversion and operation
    methods for composing operations without exceptions.
    """

    _result: Result[T, str] | None = PrivateAttr(default=None)

    @staticmethod
    def _validate_error_data(
        error_data: t.ResultErrorData | t.ConfigModelInput | None,
    ) -> t.ConfigMap | None:
        """Convert error_data to ConfigMap, matching RuntimeResult.fail() logic."""
        if error_data is None:
            return None
        if isinstance(error_data, t.ConfigMap):
            return error_data
        if isinstance(error_data, BaseModel):
            dump = error_data.model_dump()
            return t.ConfigMap(dump)
        return t.ConfigMap(dict(error_data))

    def __init__(
        self,
        source: Result[T, str] | None = None,
        error_code: str | None = None,
        error_data: t.ResultErrorData | t.ConfigModelInput | None = None,
        *,
        value: T | None = None,
        error: str | None = None,
        is_success: bool = True,
    ) -> None:
        """Initialize FlextResult from value/error/is_success only (direct typing, no Result unwrap)."""
        # Convert error_data to ConfigMap using the same logic as RuntimeResult.fail()
        validated_error_data = FlextResult._validate_error_data(error_data)

        if source is not None and value is None and (error is None):
            try:
                failure_value = source.failure()
            except UnwrapFailedError as exc:
                super().__init__(
                    error_code=error_code,
                    error=None,
                    is_success=True,
                    error_data=validated_error_data,
                )
                self._result = source
                self._payload = source.unwrap()
                self.result_logger.debug(
                    "Result source is success path during initialization",
                    exc_info=exc,
                )
                return
            super().__init__(
                error_code=error_code,
                error=str(failure_value),
                is_success=False,
                error_data=validated_error_data,
            )
            self._result = source
            return
        super().__init__(
            error=error,
            error_code=error_code,
            is_success=is_success,
            error_data=validated_error_data,
        )
        self._result = source
        if value is not None and is_success:
            self._payload = value

    @property
    def _returns_result(self) -> Result[T, str]:
        """Access the internal returns library Result[T, str] for advanced operations."""
        if self._result is None:
            if self.is_success:
                self._result = Success(self.value)
            else:
                self._result = Failure(self.error or "")
        return self._result

    @classmethod
    def _fail_like[S](
        cls,
        source: FlextRuntime.RuntimeResult[S],
        *,
        default_error: str = "",
    ) -> FlextResult[S]:
        if source.is_success:
            msg = "Cannot mirror failure from successful result"
            raise ValueError(msg)
        return FlextResult[S].fail(
            source.error or default_error,
            error_code=source.error_code,
            error_data=source.error_data,
            exception=source.exception,
        )

    @classmethod
    def _from_runtime_result[U](
        cls,
        source: FlextRuntime.RuntimeResult[U],
    ) -> FlextResult[U]:
        if source.is_success:
            return FlextResult[U].ok(source.value)
        return FlextResult[U].fail(
            source.error or "",
            error_code=source.error_code,
            error_data=source.error_data,
            exception=source.exception,
        )

    @classmethod
    def _validate_model[UModel: BaseModel](
        cls,
        data: t.ScalarMapping | BaseModel,
        model: type[UModel],
        *,
        failure_prefix: str,
    ) -> FlextResult[UModel]:
        try:
            return FlextResult[UModel].ok(model.model_validate(data))
        except (
            ValidationError,
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            Exception,
        ) as e:
            logging.getLogger(__name__).debug(
                "%s during model validation",
                failure_prefix,
                exc_info=e,
            )
            return FlextResult[UModel].fail(
                f"{failure_prefix}: {cls._model_error_message(e)}",
                exception=e,
            )

    @classmethod
    def accumulate_errors(cls, *results: FlextResult[U]) -> FlextResult[Sequence[U]]:
        """Collect all successes, fail if any failure with all errors combined."""
        successes: MutableSequence[U] = []
        errors: MutableSequence[str] = []
        for result in results:
            if result.is_success:
                successes.append(result.value)
            else:
                errors.append(result.error or "Unknown error")
        if errors:
            return FlextResult[Sequence[U]].fail("; ".join(errors))
        return FlextResult[Sequence[U]](value=successes, is_success=True)

    @classmethod
    def create_from_callable[V](
        cls,
        func: Callable[[], V | None],
        error_code: str | None = None,
    ) -> FlextResult[V]:
        """Create result from callable, catching exceptions."""
        try:
            value = func()
            if value is None:
                return FlextResult[V].fail(
                    "Callable returned None",
                    error_code=error_code,
                )
            return FlextResult[V].ok(value)
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            logging.getLogger(__name__).debug("Callable execution failed", exc_info=e)
            return FlextResult[V].fail(str(e), error_code=error_code, exception=e)

    @classmethod
    @override
    def fail(
        cls,
        error: str | None,
        error_code: str | None = None,
        error_data: t.ResultErrorData | t.ConfigModelInput | None = None,
        *,
        exception: BaseException | None = None,
    ) -> Self:
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
        result: Self = cls(
            error_code=error_code,
            error_data=error_data,
            error=error_msg,
            is_success=False,
        )
        result._result = Failure(error_msg)
        result._exception = exception
        return result

    @classmethod
    def from_validation(
        cls: type[FlextResult[T_Model]],
        data: t.ScalarMapping | BaseModel,
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
        return cls._validate_model(data, model, failure_prefix="Validation failed")

    @override
    @classmethod
    def ok[U](cls, value: U) -> FlextResult[U]:
        """Create successful result wrapping value.

        None IS a valid value when T includes None (e.g. r[str | None].ok(None)).
        DO NOT add None rejection here — use create_from_callable for None-means-failure.

        Args:
            value: Value to wrap (any T, including None when T allows it)

        """
        result = FlextResult[U](value=value, is_success=True)
        result._result = Success(value)
        return result

    @classmethod
    def traverse[V, U](
        cls,
        items: Sequence[V],
        func: Callable[[V], FlextResult[U]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[Sequence[U]]:
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
            results: MutableSequence[U] = []
            for item in items:
                result = func(item)
                if result.is_failure:
                    return FlextResult[Sequence[U]].fail(
                        result.error or "Unknown error",
                        error_code=result.error_code,
                        error_data=result.error_data,
                        exception=result.exception,
                    )
                results.append(result.value)
            return FlextResult[Sequence[U]](value=results, is_success=True)
        all_results = [func(item) for item in items]
        return cls.accumulate_errors(*all_results)

    @classmethod
    def with_resource[R, U](
        cls,
        factory: Callable[[], R],
        op: Callable[[R], FlextResult[U]],
        cleanup: Callable[[R], None] | None = None,
    ) -> FlextResult[U]:
        """Resource management with automatic cleanup."""
        resource = factory()
        try:
            return op(resource)
        finally:
            if cleanup:
                cleanup(resource)

    @staticmethod
    def _model_error_message(error: BaseException) -> str:
        if isinstance(error, ValidationError):
            return str(error.errors())
        errors_fn = getattr(error, "errors", None)
        if callable(errors_fn):
            return str(errors_fn())
        return str(error)

    @staticmethod
    def is_failure_result(
        value: FlextRuntime.RuntimeResult[t.Container] | t.Container,
    ) -> TypeIs[FlextResult[t.Container]]:
        """Return ``True`` when *value* is a failed runtime result."""
        return isinstance(value, FlextRuntime.RuntimeResult) and value.is_failure

    @staticmethod
    def is_success_result(
        value: FlextRuntime.RuntimeResult[t.Container] | t.Container,
    ) -> TypeIs[FlextResult[t.Container]]:
        """Return ``True`` when *value* is a successful runtime result."""
        return isinstance(value, FlextRuntime.RuntimeResult) and value.is_success

    @staticmethod
    def safe[U, **PFunc](func: Callable[PFunc, U]) -> Callable[PFunc, FlextResult[U]]:
        """Decorator to wrap function in FlextResult.

        Catches exceptions and returns FlextResult.fail() on error.

        Example:
            @FlextResult.safe
            def risky_operation() -> int:
                return 42

        """

        def wrapper(*args: PFunc.args, **kwargs: PFunc.kwargs) -> FlextResult[U]:
            try:
                result = func(*args, **kwargs)
                return FlextResult[U].ok(result)
            except (
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
                AttributeError,
                OSError,
                LookupError,
                ArithmeticError,
            ) as e:
                logging.getLogger(__name__).debug(
                    "FlextResult.safe callable failed",
                    exc_info=e,
                )
                return FlextResult[U].fail(str(e), exception=e)

        return wrapper

    @override
    def filter(self, predicate: Callable[[T], bool]) -> FlextResult[T]:
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
            return FlextResult[T].fail("Value did not pass filter predicate")
        return self

    @override
    def flat_map[U](
        self,
        func: Callable[[T], FlextRuntime.RuntimeResult[U]],
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
            return FlextResult[U]._from_runtime_result(inner_result)
        return FlextResult[U].fail(
            self.error or "",
            error_code=self.error_code,
            error_data=self.error_data,
            exception=self.exception,
        )

    @override
    def flow_through[U](
        self,
        *funcs: Callable[[T | U], FlextRuntime.RuntimeResult[U]],
    ) -> FlextResult[T] | FlextResult[U]:
        """Chain multiple operations in a pipeline.

        Overrides RuntimeResult.flow_through to return FlextResult for type
        consistency. Accepts callables returning any RuntimeResult subclass
        (including FlextResult) and wraps results as FlextResult.

        Args:
            funcs: Functions to apply in sequence, each taking the previous
                result's value and returning a RuntimeResult.

        Returns:
            FlextResult[T] if no funcs, value is None, or chain
            short-circuits on failure. FlextResult[U] if all funcs applied.

        Example:
            result = r[ConfigMap].ok(data).flow_through(validate, enrich)

        """
        if self.is_failure or not funcs:
            return self
        current: FlextResult[T] | FlextResult[U] = self
        for func in funcs:
            if current.is_success:
                result_value = current.value
                if result_value is not None:
                    inner: FlextRuntime.RuntimeResult[U] = func(result_value)
                    converted: FlextResult[U] = FlextResult[U]._from_runtime_result(
                        inner,
                    )
                    current = converted
                else:
                    break
            else:
                break
        return current

    @override
    def fold[U](
        self,
        on_failure: Callable[[str], U],
        on_success: Callable[[T], U],
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

    @override
    def lash(
        self,
        func: Callable[[str], FlextRuntime.RuntimeResult[T]],
    ) -> FlextResult[T]:
        """Apply recovery function on failure.

        Applies func to the error message on failure and returns the result directly.
        If this result is a success, returns self unchanged.

        Args:
            func: Function that takes the error message and returns a RuntimeResult.

        Returns:
            FlextResult[T]: Self if success, or result from func on failure.

        Example:
            result = r[int].fail("error").lash(lambda e: r[int].ok(0))

        """
        if self.is_failure:
            inner_result = func(self.error or "")
            return FlextResult[T]._from_runtime_result(inner_result)
        return self

    @override
    def map[U](self, func: Callable[[T], U]) -> FlextResult[U]:
        """Transform success value using function.

        Overrides RuntimeResult.map to use returns library for compatibility.
        """
        if self.is_success:
            try:
                mapped_value = func(self.value)
                return FlextResult[U](value=mapped_value, is_success=True)
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                self.result_logger.debug("FlextResult.map callable failed", exc_info=e)
                result = FlextResult[U](error=str(e), is_success=False)
                result._exception = e
                return result
        result = FlextResult[U](error=self.error or "", is_success=False)
        result._exception = self._exception
        return result

    @override
    def map_error(self, func: Callable[[str], str]) -> FlextResult[T]:
        """Apply transformation function to error message on failure.

        Overrides RuntimeResult.map_error to return FlextResult for type consistency.
        """
        if self.is_failure:
            transformed_error = func(self.error or "")
            return FlextResult[T].fail(
                transformed_error,
                error_code=self.error_code,
                error_data=self.error_data,
                exception=self.exception,
            )
        return self

    @overload
    def map_or(self, default: None, func: None = None) -> T | None: ...

    @overload
    def map_or[U](self, default: U, func: None = None) -> T | U: ...

    @overload
    def map_or[U](self, default: U, func: Callable[[T], U]) -> U: ...

    def map_or[U](self, default: U, func: Callable[[T], U] | None = None) -> U | T:
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
            return self.value
        return default

    @override
    def recover(self, func: Callable[[str], T]) -> FlextResult[T]:
        """Recover from failure with fallback value.

        Overrides RuntimeResult.recover to return FlextResult for type consistency.
        """
        if self.is_success:
            return self
        fallback_value = func(self.error or "")
        return FlextResult[T].ok(fallback_value)

    @override
    def tap(self, func: Callable[[T], None]) -> FlextResult[T]:
        """Apply side effect to success value, return unchanged.

        Overrides RuntimeResult.tap to return FlextResult for type consistency.
        """
        if self.is_success and self.value is not None:
            func(self.value)
        return self

    @override
    def tap_error(self, func: Callable[[str], None]) -> Self:
        """Execute side effect on failure, return unchanged.

        Useful for logging or metrics on failure without affecting the result.
        """
        if self.is_failure:
            func(self.error or "")
        return self

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
            return FlextResult[U].fail(
                self.error or "",
                error_code=self.error_code,
                error_data=self.error_data,
                exception=self.exception,
            )
        try:
            return FlextResult[U].ok(model.model_validate(self.value))
        except (
            ValidationError,
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            Exception,
        ) as e:
            logging.getLogger(__name__).debug(
                "Model conversion failed during model validation",
                exc_info=e,
            )
            return FlextResult[U].fail(
                f"Model conversion failed: {self._model_error_message(e)}",
                exception=e,
            )

    @override
    def unwrap_or[D](self, default: D) -> T | D:
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


# Runtime assignment to pass duck typing and isinstance
r = FlextResult

# Ensure we export all types needed for module clients
__all__ = ["FlextResult", "r"]
