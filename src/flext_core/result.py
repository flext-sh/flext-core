"""Type-safe result type for operations.

Provides success/failure handling with monadic helpers for composing
operations without exceptions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from types import TracebackType
from typing import (
    Annotated,
    ClassVar,
    Self,
    TypeIs,
    cast,
    overload,
    override,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    computed_field,
)
from returns.primitives.exceptions import UnwrapFailedError
from returns.result import Failure, Result, Success

from flext_core._protocols.logging import FlextProtocolsLogging
from flext_core._protocols.result import FlextProtocolsResult
from flext_core.constants import c
from flext_core.typings import t


class FlextResult[T](BaseModel, FlextProtocolsResult.Result[T]):
    """Type-safe result with monadic helpers for operation composition.

    Provides success/failure handling with various conversion and operation
    methods for composing operations without exceptions.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    result_success: Annotated[bool, Field(default=True, alias="success")]
    result_error: Annotated[str | None, Field(default=None, alias="error")]
    result_error_code: Annotated[
        str | None,
        Field(default=None, alias="error_code"),
    ]
    result_error_data: Annotated[
        t.ConfigMap | None,
        Field(default=None, alias="error_data"),
    ]

    _payload: T | None = PrivateAttr(default=None)
    _exception: BaseException | None = PrivateAttr(default=None)
    _result: Result[T, str] | None = PrivateAttr(default=None)
    _result_logger: FlextProtocolsLogging.Logger | None = PrivateAttr(default=None)

    @property
    @override
    def success(self) -> bool:
        """Public success flag backed by an aliased Pydantic field."""
        return self.result_success

    @property
    @override
    def error(self) -> str | None:
        """Public error message backed by an aliased Pydantic field."""
        return self.result_error

    @property
    @override
    def error_code(self) -> str | None:
        """Public error code backed by an aliased Pydantic field."""
        return self.result_error_code

    @property
    @override
    def error_data(self) -> t.ConfigMap | None:
        """Public error metadata backed by an aliased Pydantic field."""
        return self.result_error_data

    @override
    def __repr__(self) -> str:
        """String representation using short alias 'r' for brevity."""
        if self.success:
            return f"r[T].ok({self.value!r})"
        return f"r[T].fail({self.error!r})"

    @override
    def __bool__(self) -> bool:
        """Boolean conversion based on success state."""
        return self.success

    @override
    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    @override
    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""

    @overload
    def __or__(self, default: T) -> T: ...
    @overload
    def __or__[DefaultT](self, default: DefaultT) -> T | DefaultT: ...

    @override
    def __or__[DefaultT](self, default: DefaultT) -> T | DefaultT:
        """Operator overload for default values."""
        return self.unwrap_or(default)

    @property
    @override
    def exception(self) -> BaseException | None:
        """Get the exception if one was captured."""
        return self._exception

    @computed_field
    @property
    @override
    def failure(self) -> bool:
        """Check if result is a failure."""
        return not self.success

    @property
    @override
    def value(self) -> T:
        """Result value — returns _payload directly on success."""
        if not self.success:
            msg = c.ERR_RESULT_CANNOT_ACCESS_VALUE.format(error=self.error)
            raise RuntimeError(msg)
        return cast("T", self._payload)

    @staticmethod
    def _validate_error_data(
        error_data: t.ResultErrorData | t.ConfigModelInput | None,
    ) -> t.ConfigMap | None:
        """Convert error_data to ConfigMap, matching RuntimeResult.fail() logic."""
        if error_data is None:
            return None
        if isinstance(error_data, t.ConfigMap):
            return error_data
        if isinstance(error_data, FlextProtocolsResult.HasModelDump):
            dump = error_data.model_dump()
            return t.ConfigMap.model_validate(dump)
        return t.ConfigMap.model_validate(dict(error_data))

    @staticmethod
    def _exception_message(exception: BaseException | None) -> str | None:
        """Extract a stable message from a structured or plain exception."""
        if exception is None:
            return None
        message = getattr(exception, "message", None)
        if isinstance(message, str) and message:
            return message
        text = str(exception)
        return text or None

    @staticmethod
    def _exception_error_code(exception: BaseException | None) -> str | None:
        """Extract structured error code when present on an exception."""
        if exception is None:
            return None
        error_code = getattr(exception, "error_code", None)
        if isinstance(error_code, str) and error_code:
            return error_code
        return None

    @staticmethod
    def _exception_error_data(exception: BaseException | None) -> t.ConfigMap | None:
        """Extract structured metadata attributes from an exception."""
        if exception is None:
            return None
        metadata = getattr(exception, "metadata", None)
        raw_attributes = getattr(metadata, c.FIELD_ATTRIBUTES, None)
        if raw_attributes is None:
            return None
        try:
            payload = t.ConfigMap.model_validate(raw_attributes)
        except ValidationError:
            return None
        correlation_id = getattr(exception, "correlation_id", None)
        if isinstance(correlation_id, str) and correlation_id:
            payload[c.ContextKey.CORRELATION_ID] = correlation_id
        return payload

    def __init__(
        self,
        source: Result[T, str] | None = None,
        error_code: str | None = None,
        error_data: t.ResultErrorData | t.ConfigModelInput | None = None,
        *,
        value: T | None = None,
        error: str | None = None,
        success: bool = True,
    ) -> None:
        """Initialize FlextResult from value/error/success only."""
        # Convert error_data to SettingsMap using the same logic as RuntimeResult.fail()
        validated_error_data = FlextResult._validate_error_data(error_data)

        if source is not None and value is None and (error is None):
            try:
                failure_value = source.failure()
            except UnwrapFailedError:
                super().__init__(
                    error_code=error_code,
                    error=None,
                    success=True,
                    error_data=validated_error_data,
                )
                self._result = source
                self._payload = source.unwrap()
                return
            super().__init__(
                error_code=error_code,
                error=str(failure_value),
                success=False,
                error_data=validated_error_data,
            )
            self._result = source
            return
        super().__init__(
            error=error,
            error_code=error_code,
            success=success,
            error_data=validated_error_data,
        )
        self._result = source
        if value is not None and success:
            self._payload = value

    @property
    def _returns_result(self) -> Result[T, str]:
        """Access the internal returns library Result[T, str] for advanced operations."""
        if self._result is None:
            if self.success:
                self._result = Success(self.value)
            else:
                self._result = Failure(self.error or "")
        return self._result

    @classmethod
    def _from_result[V](
        cls,
        source: FlextProtocolsResult.Result[V],
    ) -> FlextResult[V]:
        if source.success:
            return FlextResult[V].ok(source.value)
        return FlextResult[V].fail(
            source.error or "",
            error_code=source.error_code,
            error_data=source.error_data,
            exception=source.exception,
        )

    @staticmethod
    def _validate_model[UModel: t.ModelCarrier](
        data: t.ModelInput,
        model: t.ModelClass[UModel],
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
            return FlextResult[UModel].fail(
                f"{failure_prefix}: {FlextResult._model_error_message(e)}",
                exception=e,
            )

    @classmethod
    def accumulate_errors[ValueT](
        cls,
        *results: FlextProtocolsResult.Result[ValueT],
    ) -> FlextResult[Sequence[ValueT]]:
        """Collect all successes, fail if any failure with all errors combined."""
        successes: MutableSequence[ValueT] = []
        errors: MutableSequence[str] = []
        for result in results:
            if result.success:
                successes.append(result.value)
            else:
                errors.append(result.error or "Unknown error")
        if errors:
            return FlextResult[Sequence[ValueT]].fail("; ".join(errors))
        return FlextResult[Sequence[ValueT]](value=successes, success=True)

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
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
            return FlextResult[V].fail(str(exc), error_code=error_code, exception=exc)

    @classmethod
    def fail[V](
        cls: type[FlextResult[V]],
        error: str | None,
        error_code: str | None = None,
        error_data: t.ResultErrorData | t.ConfigModelInput | None = None,
        *,
        exception: BaseException | None = None,
    ) -> FlextResult[V]:
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
        resolved_error_code = error_code or cls._exception_error_code(exception)
        resolved_error_data = (
            error_data
            if error_data is not None
            else cls._exception_error_data(exception)
        )
        result = cls(
            error_code=resolved_error_code,
            error_data=resolved_error_data,
            error=error_msg,
            success=False,
        )
        result._result = Failure(error_msg)
        result._exception = exception
        return result

    @classmethod
    def from_exception[V](
        cls: type[FlextResult[V]],
        exception: BaseException,
        *,
        error: str | None = None,
        error_code: str | None = None,
        error_data: t.ResultErrorData | t.ConfigModelInput | None = None,
    ) -> FlextResult[V]:
        """Create a failed result directly from an exception public surface."""
        return cls.fail(
            error if error is not None else cls._exception_message(exception),
            error_code=error_code,
            error_data=error_data,
            exception=exception,
        )

    @classmethod
    def fail_exc[V](
        cls: type[FlextResult[V]],
        exc: BaseException,
    ) -> FlextResult[V]:
        """Create failed result from a BaseException (e.BaseError or stdlib).

        Usage::

            return r[bool].fail_exc(exc)

        """
        return cls.fail(
            str(exc),
            error_code=cls._exception_error_code(exc),
            error_data=cls._exception_error_data(exc),
            exception=exc,
        )

    @classmethod
    def fail_op[V](
        cls: type[FlextResult[V]],
        operation: str,
        exc: Exception | str | None = None,
        *,
        error_code: str | None = None,
    ) -> FlextResult[V]:
        """Create failed result for an operation that failed.

        Usage::

            return r[bool].fail_op("load config", exc)

        """
        reason = str(exc) if exc is not None else None
        msg = (
            f"Failed to {operation}: {reason}"
            if reason is not None
            else f"Failed to {operation}"
        )
        return cls.fail(
            msg,
            error_code=error_code,
            exception=exc if isinstance(exc, BaseException) else None,
        )

    @staticmethod
    def from_validation[ModelT: t.ModelCarrier](
        data: t.ModelInput,
        model: t.ModelClass[ModelT],
    ) -> FlextResult[ModelT]:
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
        return FlextResult._validate_model(
            data,
            model,
            failure_prefix="Validation failed",
        )

    @classmethod
    def ok[V](cls: type[FlextResult[V]], value: V) -> FlextResult[V]:
        """Create successful result wrapping value.

        None IS a valid value when T includes None (e.g. r[str | None].ok(None)).
        DO NOT add None rejection here — use create_from_callable for None-means-failure.

        Args:
            value: Value to wrap (any T, including None when T allows it)

        """
        result = cls(value=value, success=True)
        result._result = Success(value)
        return result

    @classmethod
    def from_result[V](
        cls,
        source: FlextProtocolsResult.Result[V],
    ) -> FlextResult[V]:
        """Normalize any structural FLEXT result into FlextResult."""
        return FlextResult[V]._from_result(source)

    @classmethod
    def traverse[V, U](
        cls,
        items: Sequence[V],
        func: Callable[[V], FlextProtocolsResult.Result[U]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[Sequence[U]]:
        """Map over sequence with settingsurable failure handling.

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
                if result.failure:
                    return FlextResult[Sequence[U]].fail(
                        result.error or "Unknown error",
                        error_code=result.error_code,
                        error_data=result.error_data,
                        exception=result.exception,
                    )
                results.append(result.value)
            return FlextResult[Sequence[U]](value=results, success=True)
        all_results = [FlextResult[U].from_result(func(item)) for item in items]
        return cls.accumulate_errors(*all_results)

    @classmethod
    def with_resource[R, U](
        cls,
        factory: Callable[[], R],
        op: Callable[[R], FlextProtocolsResult.Result[U]],
        cleanup: Callable[[R], None] | None = None,
    ) -> FlextResult[U]:
        """Resource management with automatic cleanup."""
        resource = factory()
        try:
            return FlextResult[U].from_result(op(resource))
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
    def failed_result(
        value: t.ProtocolSubject,
    ) -> TypeIs[FlextResult[t.RecursiveContainer]]:
        """Return ``True`` when *value* is a failed runtime result."""
        return isinstance(value, FlextResult) and value.failure

    @staticmethod
    def successful_result(
        value: t.ProtocolSubject,
    ) -> TypeIs[FlextResult[t.RecursiveContainer]]:
        """Return ``True`` when *value* is a successful runtime result."""
        return isinstance(value, FlextResult) and value.success

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
            ) as exc:
                return FlextResult[U].fail(str(exc), exception=exc)

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
        if self.success and self.value is not None:
            if predicate(self.value):
                return self
            return self.__class__.fail(c.ERR_RESULT_FILTER_PREDICATE_FAILED)
        return self

    @override
    def flat_map[U](
        self,
        func: Callable[[T], FlextProtocolsResult.Result[U]],
    ) -> FlextResult[U]:
        """Chain operations returning a Result (monadic bind).

        Applies func to the success value and returns its result directly.
        Unlike map, func already produces a Result — no re-wrapping occurs.
        On failure, propagates the current error as a new Result[U].

        Args:
            func: Function that takes the success value and returns a Result.

        Returns:
            The result from func on success, or failure propagated.

        Example:
            result = r[int].ok(42).flat_map(lambda x: r[str].ok(str(x)))

        """
        if self.failure:
            return FlextResult[U].fail(
                self.error or "",
                error_code=self.error_code,
                error_data=self.error_data,
                exception=self.exception,
            )
        return FlextResult[U].from_result(func(self.value))

    @override
    def flow_through(
        self,
        *funcs: Callable[[T], FlextProtocolsResult.Result[T]],
    ) -> FlextResult[T]:
        """Chain multiple operations in a homogeneous pipeline.

        Each func receives the previous success value and returns
        a FlextResult[T]. The chain short-circuits on first failure.

        Args:
            funcs: Functions to apply in sequence, each taking T and
                returning FlextResult[T].

        Returns:
            FlextResult[T]: Final result after all funcs, or first failure.

        Example:
            result = r[SettingsMap].ok(data).flow_through(validate, enrich)

        """
        current: FlextResult[T] = self
        for func in funcs:
            if current.success:
                result_value = current.value
                if result_value is not None:
                    current = FlextResult[T].from_result(func(result_value))
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
        if self.success and self.value is not None:
            return on_success(self.value)
        return on_failure(self.error or "")

    @override
    def lash(
        self,
        func: Callable[[str], FlextProtocolsResult.Result[T]],
    ) -> FlextResult[T]:
        """Apply recovery function on failure.

        Applies func to the error message on failure and returns the result directly.
        If this result is a success, returns self unchanged.

        Args:
            func: Function that takes the error message and returns a Result.

        Returns:
            FlextResult[T]: Self if success, or result from func on failure.

        Example:
            result = r[int].fail("error").lash(lambda e: r[int].ok(0))

        """
        if self.failure:
            inner_result = func(self.error or "")
            return FlextResult[T]._from_result(inner_result)
        return self

    @override
    def map[U](self, func: Callable[[T], U]) -> FlextResult[U]:
        """Transform success value using function.

        Applies func to the success value and wraps the result.
        On failure, propagates the current error.
        """
        if self.success:
            try:
                mapped_value = func(self.value)
                return FlextResult[U](value=mapped_value, success=True)
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                RuntimeError,
            ) as exc:
                result = FlextResult[U](error=str(exc), success=False)
                result._exception = exc
                return result
        result = FlextResult[U](error=self.error or "", success=False)
        result._exception = self._exception
        return result

    @override
    def map_error(
        self,
        func: Callable[[str], str],
    ) -> FlextResult[T]:
        """Apply transformation function to error message on failure.

        Overrides RuntimeResult.map_error to return FlextResult for type consistency.
        """
        if self.failure:
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

    @override
    def map_or[U](
        self,
        default: U,
        func: Callable[[T], U] | None = None,
    ) -> U | T:
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
        if self.success and self.value is not None:
            if func is not None:
                return func(self.value)
            return self.value
        return default

    @override
    def recover[U](self, func: Callable[[str], U]) -> FlextResult[T | U]:
        """Recover from failure with fallback value.

        Overrides RuntimeResult.recover to return FlextResult for type consistency.
        """
        if self.success:
            return cast("FlextResult[T | U]", self)
        fallback_value = func(self.error or "")
        return FlextResult[T | U].ok(fallback_value)

    @override
    def tap(
        self,
        func: Callable[[T], None],
    ) -> FlextResult[T]:
        """Apply side effect to success value, return unchanged.

        Overrides RuntimeResult.tap to return FlextResult for type consistency.
        """
        if self.success and self.value is not None:
            func(self.value)
        return self

    @override
    def tap_error(self, func: Callable[[str], None]) -> Self:
        """Execute side effect on failure, return unchanged.

        Useful for logging or metrics on failure without affecting the result.
        """
        if self.failure:
            func(self.error or "")
        return self

    @override
    def to_model[U: t.ModelCarrier](
        self,
        model: t.ModelClass[U],
    ) -> FlextResult[U]:
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
        if self.failure:
            return FlextResult[U].fail(
                self.error or "",
                error_code=self.error_code,
                error_data=self.error_data,
                exception=self.exception,
            )
        try:
            validation_input = cast("t.ModelInput", self.value)
            return FlextResult[U].ok(model.model_validate(validation_input))
        except (
            ValidationError,
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            Exception,
        ) as e:
            return FlextResult[U].fail(
                f"Model conversion failed: {self._model_error_message(e)}",
                exception=e,
            )

    @override
    def to_type[U](self, adapter: TypeAdapter[U]) -> FlextResult[U]:
        """Convert successful value using a cached Pydantic v2 TypeAdapter."""
        if self.failure:
            return FlextResult[U].fail(
                self.error or "",
                error_code=self.error_code,
                error_data=self.error_data,
                exception=self.exception,
            )
        try:
            return FlextResult[U].ok(adapter.validate_python(self.value))
        except (
            ValidationError,
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            Exception,
        ) as e:
            return FlextResult[U].fail(
                f"Type conversion failed: {self._model_error_message(e)}",
                exception=e,
            )

    @override
    def unwrap(self) -> T:
        """Unwrap the success value or raise RuntimeError."""
        if self.failure:
            msg = c.ERR_RESULT_CANNOT_UNWRAP.format(error=self.error)
            raise RuntimeError(msg)
        return self.value

    @override
    def unwrap_model[U: t.ModelCarrier](self, model: t.ModelClass[U]) -> U:
        """Unwrap successful value after Pydantic model conversion."""
        return self.to_model(model).unwrap()

    @override
    def unwrap_type[U](self, adapter: TypeAdapter[U]) -> U:
        """Unwrap successful value after TypeAdapter conversion."""
        return self.to_type(adapter).unwrap()

    @overload
    def unwrap_or(self, default: T) -> T: ...
    @overload
    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT: ...

    @override
    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT:
        """Return value if success, otherwise return default.

        Safe way to extract value without raising exceptions.
        Replaces pattern: result.value if result.success else default

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
        if self.success and self.value is not None:
            return self.value
        return default

    @overload
    def unwrap_or_else(self, func: Callable[[], T]) -> T: ...
    @overload
    def unwrap_or_else[DefaultT](
        self,
        func: Callable[[], DefaultT],
    ) -> T | DefaultT: ...

    @override
    def unwrap_or_else[DefaultT](
        self,
        func: Callable[[], DefaultT],
    ) -> T | DefaultT:
        """Return the success value or call func if failed."""
        if self.success and self.value is not None:
            return self.value
        return func()


r = FlextResult


__all__: list[str] = ["FlextResult", "r"]
