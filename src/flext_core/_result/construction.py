"""Construction operations for FlextResult."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

from pydantic import BaseModel, ValidationError

from flext_core import c

from .base import FlextResultBase
from .behavior import FlextResultBehavior

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core import p, t


class FlextResultConstruction[T](FlextResultBehavior[T]):
    """Factory methods for the concrete result facade.

    Contracts are typed as ``p.Result`` (abstract). Instances are built only via
    ``cls(...)`` — never by importing the public ``FlextResult`` facade.
    """

    @staticmethod
    def require_error(source: p.FailureLike) -> str:
        """Extract error message from any failed Result."""
        error = source.error
        if not error:
            msg = c.ERR_RESULT_FAILURE_MESSAGE_REQUIRED
            raise ValueError(msg)
        return error

    @classmethod
    def from_failure(cls: type[Self], source: p.FailureLike) -> p.Result[T]:
        if source.success:
            msg = c.ERR_RESULT_FAILURE_REQUIRED
            raise ValueError(msg)
        return cls.fail(
            cls.require_error(source),
            error_code=source.error_code,
            error_data=source.error_data,
            exception=source.exception,
        )

    @classmethod
    def _extract_exception_error_code(
        cls, exception: BaseException | None
    ) -> str | None:
        if exception is None:
            return None
        error_code = getattr(exception, "error_code", None)
        return error_code if isinstance(error_code, str) and error_code else None

    @classmethod
    def _extract_exception_error_data(
        cls, exception: BaseException | None
    ) -> t.JsonDict | None:
        if exception is None:
            return None
        metadata = getattr(exception, "metadata", None)
        raw_attributes = getattr(metadata, c.FIELD_ATTRIBUTES, None)
        if raw_attributes is None:
            return None
        try:
            payload = cls.validate_error_data(raw_attributes)
        except ValidationError:
            return None
        if payload is None:
            return None
        correlation_id = getattr(exception, "correlation_id", None)
        if isinstance(correlation_id, str) and correlation_id:
            payload[c.ContextKey.CORRELATION_ID] = correlation_id
        return payload

    @classmethod
    def copy_from_result(cls: type[Self], source: p.Result[T]) -> p.Result[T]:
        if source.success:
            try:
                return cls.ok(source.value)
            except ValueError as exc:
                return cls.fail(str(exc))
        return cls.fail(
            cls.require_error(source),
            error_code=source.error_code,
            error_data=source.error_data,
            exception=source.exception,
        )

    @classmethod
    def create_from_callable[V](
        cls: type[Self], func: Callable[[], V | None], error_code: str | None = None
    ) -> p.Result[V]:
        try:
            value = func()
            if value is None:
                return cast(
                    "p.Result[V]",
                    cls.fail("Callable returned None", error_code=error_code),
                )
            return cls.ok(value)
        except c.EXC_BROAD_RUNTIME as exc:
            return cast(
                "p.Result[V]",
                cls.fail(str(exc), error_code=error_code, exception=exc),
            )

    @classmethod
    def fail(
        cls: type[Self],
        error: str | None,
        *,
        error_code: str | None = None,
        error_data: t.JsonMapping | t.ConfigModelInput | None = None,
        exception: BaseException | None = None,
    ) -> p.Result[T]:
        error_msg = error if error is not None else ""
        resolved_error_code = error_code or cls._extract_exception_error_code(exception)
        resolved_error_data = (
            error_data
            if error_data is not None
            else cls._extract_exception_error_data(exception)
        )
        return cast(
            "p.Result[T]",
            cls(
                error_code=resolved_error_code,
                error_data=cls.validate_error_data(resolved_error_data),
                error=error_msg,
                success=False,
                exception=exception,
            ),
        )

    @classmethod
    def fail_op(
        cls: type[Self], operation: str, exc: Exception | str | None = None
    ) -> p.Result[T]:
        if isinstance(exc, Exception):
            return cls.fail(f"{operation} failed: {exc}", exception=exc)
        error_msg = (
            f"{operation} failed" if exc is None else f"{operation} failed: {exc}"
        )
        return cls.fail(error_msg)

    @classmethod
    def from_validation[ModelT: BaseModel](
        cls: type[Self], data: t.ModelInput, model: t.ModelClass[ModelT]
    ) -> p.Result[ModelT]:
        try:
            validated: ModelT = model.model_validate(data)
            return cls.ok(validated)
        except c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            return cast("p.Result[ModelT]", cls.fail(str(exc), exception=exc))

    @classmethod
    def ok[V](cls: type[Self], value: V) -> p.Result[V]:
        return cast("p.Result[V]", cls(value=value, success=True))

    @staticmethod
    def successful_result(obj: object) -> bool:
        """Check whether an object is a successful result instance."""
        return isinstance(obj, FlextResultBase) and obj.success

    @staticmethod
    def failed_result(obj: object) -> bool:
        """Check whether an object is a failed result instance."""
        return isinstance(obj, FlextResultBase) and not obj.success

    @classmethod
    def from_result[V](cls: type[Self], source: p.Result[V]) -> p.Result[V]:
        if source.success:
            try:
                return cls.ok(source.value)
            except ValueError as exc:
                return cast("p.Result[V]", cls.fail(str(exc)))
        return cast(
            "p.Result[V]",
            cls.fail(
                cls.require_error(source),
                error_code=source.error_code,
                error_data=source.error_data,
                exception=source.exception,
            ),
        )


__all__: list[str] = ["FlextResultConstruction"]
