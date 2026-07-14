"""Construction and factory operations for FlextResult."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, cast

from pydantic import ValidationError
from returns.result import Failure, Success

from flext_core._constants.errors import FlextConstantsErrors as c
from flext_core._constants.infrastructure import FlextConstantsInfrastructure
from flext_core._constants.mixins import FlextConstantsMixins
from flext_core._models.pydantic import FlextModelsPydantic as mp

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core import FlextTypes as t
    from flext_core._models.containers import FlextModelsContainers as mc
    from flext_core._protocols.result import FlextProtocolsResult as p

from .behavior import FlextResultBehaviorMixin


class FlextResultConstructionMixin[T](FlextResultBehaviorMixin[T], ABC):
    """Factory methods for the concrete result facade."""

    @staticmethod
    def require_error[V](source: p.ResultLike[V]) -> str:
        """Return a failure message or raise when the Result invariant is broken."""
        error = source.error
        if not error:
            raise ValueError(c.ERR_RESULT_FAILURE_MESSAGE_REQUIRED)
        return error

    @classmethod
    def from_failure[V](
        cls: type[FlextResultConstructionMixin[T]], source: p.ResultLike[V]
    ) -> FlextResultConstructionMixin[T]:
        """Rebind one failure payload type while preserving its full error state."""
        if source.success:
            raise ValueError(c.ERR_RESULT_FAILURE_REQUIRED)
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
    ) -> mc.ConfigMap | None:
        if exception is None:
            return None
        metadata = getattr(exception, "metadata", None)
        raw_attributes = getattr(metadata, FlextConstantsMixins.FIELD_ATTRIBUTES, None)
        if raw_attributes is None:
            return None
        try:
            payload = cls._validate_error_data(raw_attributes)
        except ValidationError:
            return None
        if payload is None:
            return None
        correlation_id = getattr(exception, "correlation_id", None)
        if isinstance(correlation_id, str) and correlation_id:
            payload[FlextConstantsInfrastructure.ContextKey.CORRELATION_ID] = (
                correlation_id
            )
        return payload

    @classmethod
    def _from_result[V](cls, source: p.Result[V]) -> FlextResultConstructionMixin[V]:
        if source.success:
            # Cannot use source.value: it raises ValueError when _payload is None.
            # r[None].ok(None) is a valid success with a None payload — access the
            # private attribute directly to bypass the non-None guard on the property.
            raw: V | None = getattr(source, "_payload", None)
            return cls.ok(cast("V", raw))
        # Type bridge: normalized failures carry the source result payload type.
        result_class = cast("type[FlextResultConstructionMixin[V]]", cls)
        return result_class.fail(
            cls.require_error(source),
            error_code=source.error_code,
            error_data=source.error_data,
            exception=source.exception,
        )

    @classmethod
    def create_from_callable[V](
        cls, func: Callable[[], V | None], error_code: str | None = None
    ) -> FlextResultConstructionMixin[V]:
        """Execute callable; catch exceptions and None returns."""
        try:
            value = func()
            if value is None:
                # Type bridge: callable failures carry the callable payload type.
                result_class = cast("type[FlextResultConstructionMixin[V]]", cls)
                return result_class.fail(
                    "Callable returned None", error_code=error_code
                )
            return cls.ok(value)
        except c.EXC_BROAD_RUNTIME as exc:
            # Type bridge: callable exceptions carry the callable payload type.
            result_class = cast("type[FlextResultConstructionMixin[V]]", cls)
            return result_class.fail(str(exc), error_code=error_code, exception=exc)

    @classmethod
    def fail[V](
        cls: type[FlextResultConstructionMixin[V]],
        error: str | None,
        *,
        error_code: str | None = None,
        error_data: t.JsonMapping | t.ConfigModelInput | None = None,
        exception: BaseException | None = None,
    ) -> FlextResultConstructionMixin[V]:
        """Create failed result with error message, optional code and metadata."""
        error_msg = error if error is not None else ""
        resolved_error_code = error_code or cls._extract_exception_error_code(exception)
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
        cls: type[FlextResultConstructionMixin[V]],
        operation: str,
        exc: Exception | str | None = None,
    ) -> FlextResultConstructionMixin[V]:
        """Create a failure result for a named operation with optional exception."""
        if isinstance(exc, Exception):
            return cls.fail(f"{operation} failed: {exc}", exception=exc)
        error_msg = (
            f"{operation} failed" if exc is None else f"{operation} failed: {exc}"
        )
        return cls.fail(error_msg)

    @classmethod
    def from_validation[ModelT: mp.BaseModel](
        cls, data: t.ModelInput, model: t.ModelClass[ModelT]
    ) -> FlextResultConstructionMixin[ModelT]:
        """Create result from Pydantic validation."""
        try:
            return cls.ok(model.model_validate(data))
        except c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            # Type bridge: validation failures carry the model payload type.
            result_class = cast("type[FlextResultConstructionMixin[ModelT]]", cls)
            return result_class.fail(str(exc), exception=exc)

    @classmethod
    def ok[V](cls, value: V) -> FlextResultConstructionMixin[V]:
        """Create successful result wrapping value."""
        # Type bridge: class factories intentionally rebind the generic payload.
        result_class = cast("type[FlextResultConstructionMixin[V]]", cls)
        result = result_class(value=value, success=True)
        result._result = Success(value)
        return result

    @classmethod
    def from_result[V](cls, source: p.Result[V]) -> FlextResultConstructionMixin[V]:
        """Normalize structural result to FlextResult."""
        return cls._from_result(source)


__all__: list[str] = ["FlextResultConstructionMixin"]
