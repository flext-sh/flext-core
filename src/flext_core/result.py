"""Type-safe result type for operations.

Provides success/failure handling with monadic helpers for composing
operations without exceptions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    MutableSequence,
    Sequence,
)
from types import TracebackType
from typing import (
    Annotated,
    ClassVar,
    Self,
    TypeIs,
    no_type_check,
    overload,
    override,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    computed_field,
)
from returns.result import Failure, Result, Success

from flext_core import (
    FlextModelsContainers as mc,
    FlextModelsPydantic as mp,
    FlextProtocolsLogging as pl,
    FlextProtocolsResult as p,
    FlextRuntime,
    c,
    t,
)


@no_type_check
class FlextResult[T](BaseModel, p.Result[T]):
    """Type-safe result with monadic railway-oriented operations."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    result_success: Annotated[bool, Field(alias="success")] = True
    result_error: Annotated[str | None, Field(alias="error")] = None
    result_error_code: Annotated[str | None, Field(alias="error_code")] = None
    result_error_data: Annotated[mc.ConfigMap | None, Field(alias="error_data")] = None

    _payload: T | None = PrivateAttr(default=None)
    _exception: BaseException | None = PrivateAttr(default=None)
    _result: Result[T, str] | None = PrivateAttr(default=None)
    _result_logger: pl.Logger | None = PrivateAttr(default=None)

    @property
    @override
    def success(self) -> bool:
        """Success flag."""
        return self.result_success

    @property
    @override
    def error(self) -> str | None:
        """Error message."""
        return self.result_error

    @property
    @override
    def error_code(self) -> str | None:
        """Error code."""
        return self.result_error_code

    @property
    @override
    def error_data(self) -> t.JsonMapping | None:
        """Error metadata."""
        data = self.result_error_data
        if data is None:
            return None
        normalized_raw: dict[str, t.JsonValue] = {}
        for key, value in data.root.items():
            normalized_raw[key] = FlextRuntime.normalize_to_metadata(value)
        return normalized_raw

    @override
    def __repr__(self) -> str:
        if self.success:
            return f"r[T].ok({self.value!r})"
        return f"r[T].fail({self.error!r})"

    @override
    def __bool__(self) -> bool:
        return self.success

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        pass

    @overload
    def __or__(self, default: T) -> T: ...
    @overload
    def __or__[DefaultT](self, default: DefaultT) -> T | DefaultT: ...

    @override
    def __or__[DefaultT](self, default: DefaultT) -> T | DefaultT:
        return self.unwrap_or(default)

    @property
    @override
    def exception(self) -> BaseException | None:
        return self._exception

    @computed_field
    @property
    @override
    def failure(self) -> bool:
        return not self.success

    @property
    @override
    def value(self) -> T:
        if not self.success:
            msg = c.ERR_RESULT_CANNOT_ACCESS_VALUE.format(error=self.error)
            raise RuntimeError(msg)
        if self._payload is None:
            msg_0 = "Successful result must have a non-None payload"
            raise ValueError(msg_0)
        return self._payload

    @staticmethod
    def _validate_error_data(
        error_data: t.JsonMapping | t.ConfigModelInput | None,
    ) -> mc.ConfigMap | None:
        normalized_error_data = FlextRuntime.normalize_model_input_mapping(error_data)
        return (
            None
            if normalized_error_data is None
            else mc.ConfigMap.model_validate(normalized_error_data)
        )

    @staticmethod
    def _extract_exception_error_code(exception: BaseException | None) -> str | None:
        if exception is None:
            return None
        error_code = getattr(exception, "error_code", None)
        return error_code if isinstance(error_code, str) and error_code else None

    @staticmethod
    def _extract_exception_error_data(
        exception: BaseException | None,
    ) -> mc.ConfigMap | None:
        if exception is None:
            return None
        metadata = getattr(exception, "metadata", None)
        raw_attributes = getattr(metadata, c.FIELD_ATTRIBUTES, None)
        if raw_attributes is None:
            return None
        try:
            payload = FlextResult._validate_error_data(raw_attributes)
        except ValidationError:
            return None
        if payload is None:
            return None
        correlation_id = getattr(exception, "correlation_id", None)
        if isinstance(correlation_id, str) and correlation_id:
            payload[c.ContextKey.CORRELATION_ID] = correlation_id
        return payload

    def __init__(
        self,
        error_code: str | None = None,
        error_data: t.JsonMapping | t.ConfigModelInput | None = None,
        *,
        value: T | None = None,
        error: str | None = None,
        success: bool = True,
    ) -> None:
        """Initialize a FlextResult with optional value, error, and metadata."""
        super().__init__(
            error=error,
            error_code=error_code,
            success=success,
            error_data=FlextResult._validate_error_data(error_data),
        )
        if success:
            self._payload = value

    @property
    def _returns_result(self) -> Result[T, str]:
        if self._result is None:
            if self.success:
                self._result = Success(self.value)
            else:
                self._result = Failure(self.error or "")
        return self._result

    @classmethod
    def _from_result[V](
        cls,
        source: p.Result[V],
    ) -> FlextResult[V]:
        if source.success:
            return FlextResult[V].ok(source.value)
        return FlextResult[V].fail(
            source.error or "",
            error_code=source.error_code,
            error_data=source.error_data,
            exception=source.exception,
        )

    @classmethod
    def accumulate_errors[ValueT](
        cls,
        *results: FlextResult[ValueT],
    ) -> FlextResult[Sequence[ValueT]]:
        """Collect successes or all errors combined."""
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
        """Execute callable; catch exceptions and None returns."""
        try:
            value = func()
            if value is None:
                return FlextResult[V].fail(
                    "Callable returned None",
                    error_code=error_code,
                )
            return FlextResult[V].ok(value)
        except c.EXC_BROAD_RUNTIME as exc:
            return FlextResult[V].fail(str(exc), error_code=error_code, exception=exc)

    @classmethod
    def fail[V](
        cls: type[FlextResult[V]],
        error: str | None,
        *,
        error_code: str | None = None,
        error_data: t.JsonMapping | t.ConfigModelInput | None = None,
        exception: BaseException | None = None,
    ) -> FlextResult[V]:
        """Create failed result with error message, optional code and metadata."""
        error_msg = error if error is not None else ""
        resolved_error_code = error_code or cls._extract_exception_error_code(
            exception,
        )
        resolved_error_data = (
            error_data
            if error_data is not None
            else cls._extract_exception_error_data(exception)
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
    def fail_op[V](
        cls: type[FlextResult[V]],
        operation: str,
        exc: Exception | str | None = None,
    ) -> FlextResult[V]:
        """Create a failure result for a named operation with optional exception."""
        if isinstance(exc, Exception):
            return cls.fail(
                f"{operation} failed: {exc}",
                exception=exc,
            )
        error_msg = (
            f"{operation} failed" if exc is None else f"{operation} failed: {exc}"
        )
        return cls.fail(error_msg)

    @staticmethod
    def from_validation[ModelT: mp.BaseModel](
        data: t.ModelInput,
        model: t.ModelClass[ModelT],
    ) -> FlextResult[ModelT]:
        """Create result from Pydantic validation."""
        try:
            return FlextResult[ModelT].ok(model.model_validate(data))
        except c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            return FlextResult[ModelT].fail(str(exc), exception=exc)

    @classmethod
    def ok[V](cls: type[FlextResult[V]], value: V) -> FlextResult[V]:
        """Create successful result wrapping value (None is valid)."""
        result = cls(value=value, success=True)
        result._result = Success(value)
        return result

    @classmethod
    def from_result[V](
        cls,
        source: p.Result[V],
    ) -> FlextResult[V]:
        """Normalize structural result to FlextResult."""
        return FlextResult[V]._from_result(source)

    @classmethod
    def traverse[V, U](
        cls,
        items: t.SequenceOf[V],
        func: Callable[[V], p.Result[U]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[Sequence[U]]:
        """Map sequence via func; fail_fast stops on first error."""
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
        op: Callable[[R], p.Result[U]],
        cleanup: Callable[[R], None] | None = None,
    ) -> FlextResult[U]:
        """Manage resource lifecycle with automatic cleanup."""
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
    def safe[U, **PFunc](func: Callable[PFunc, U]) -> Callable[PFunc, FlextResult[U]]:
        """Decorator: wrap function in FlextResult, catch exceptions."""

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
        """Filter success value; returns self or failure if predicate fails."""
        if self.success and self.value is not None:
            if predicate(self.value):
                return self
            return self.__class__.fail(c.ERR_RESULT_FILTER_PREDICATE_FAILED)
        return self

    @override
    def flat_map[U](
        self,
        func: Callable[[T], p.Result[U]],
    ) -> FlextResult[U]:
        """Chain operations returning a Result; func produces Result directly."""
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
        *funcs: Callable[[T], p.Result[T]],
    ) -> FlextResult[T]:
        """Chain multiple homogeneous Result-returning operations in sequence."""
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
        """Catamorphism: reduce result to a single value via callbacks."""
        if self.success and self.value is not None:
            return on_success(self.value)
        return on_failure(self.error or "")

    @override
    def lash(
        self,
        func: Callable[[str], p.Result[T]],
    ) -> FlextResult[T]:
        """Apply recovery function on failure; returns self if success."""
        if self.failure:
            inner_result = func(self.error or "")
            return FlextResult[T]._from_result(inner_result)
        return self

    @override
    def map[U](self, func: Callable[[T], U]) -> FlextResult[U]:
        """Transform success value; propagates failure."""
        if self.success:
            try:
                mapped_value = func(self.value)
                return FlextResult[U](value=mapped_value, success=True)
            except c.EXC_BROAD_RUNTIME as exc:
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
        """Transform error message; returns self if success."""
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
        """Apply func to success value or return default; func optional."""
        if self.success and self.value is not None:
            if func is not None:
                return func(self.value)
            return self.value
        return default

    @override
    def recover[U](self, func: Callable[[str], U]) -> FlextResult[T | U]:
        """Recover from failure with fallback value via callback."""
        if self.success:
            # When success, return self as FlextResult[T | U] by wrapping the existing result
            return FlextResult[T | U](
                value=self.value,
                error_code=self.error_code,
                error_data=self.error_data,
                success=True,
            )
        fallback_value = func(self.error or "")
        return FlextResult[T | U].ok(fallback_value)

    @override
    def tap(
        self,
        func: Callable[[T], None],
    ) -> FlextResult[T]:
        """Apply side effect to success value; return unchanged."""
        if self.success and self.value is not None:
            func(self.value)
        return self

    @override
    def tap_error(self, func: Callable[[str], None]) -> Self:
        """Side effect on failure; return unchanged."""
        if self.failure:
            func(self.error or "")
        return self

    @override
    def to_model[U: mp.BaseModel](
        self,
        model: t.ModelClass[U],
    ) -> FlextResult[U]:
        """Convert success value to Pydantic model; propagates failure."""
        if self.failure:
            return FlextResult[U].fail(
                self.error or "",
                error_code=self.error_code,
                error_data=self.error_data,
                exception=self.exception,
            )
        try:
            return FlextResult[U].ok(model.model_validate(self.value))
        except c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            return FlextResult[U].fail(str(exc), exception=exc)

    @override
    def unwrap(self) -> T:
        """Unwrap the success value or raise RuntimeError."""
        if self.failure:
            msg = c.ERR_RESULT_CANNOT_UNWRAP.format(error=self.error)
            raise RuntimeError(msg)
        return self.value

    @overload
    def unwrap_or(self, default: T) -> T: ...
    @overload
    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT: ...

    @override
    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT:
        """Return success value or default; safe extraction."""
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

    @staticmethod
    def successful_result[V](obj: FlextResult[V] | V) -> TypeIs[FlextResult[V]]:
        """Type guard for successful result."""
        return isinstance(obj, FlextResult) and obj.success

    @staticmethod
    def failed_result[V](obj: FlextResult[V] | V) -> TypeIs[FlextResult[V]]:
        """Type guard for failed result."""
        return isinstance(obj, FlextResult) and obj.failure


r = FlextResult

__all__: list[str] = ["FlextResult", "r"]
